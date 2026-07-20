//! The immutable lazy operation graph.
//!
//! Chaining does no pixel work: each call wraps the current node in a new one,
//! producing a DAG of [`Arc<Node>`] (ARCHITECTURE §Layer 3). Sharing is free —
//! cloning an [`Image`] clones an `Arc`, and two pipelines branching from a
//! common prefix share those nodes, so the evaluator computes them once.
//!
//! Descriptors are resolved **as the graph is built**. By the time a node
//! exists, its output shape is already known, so [`Image::metadata`] is a field
//! read rather than a traversal.

use crate::{
    Format, ImageDescriptor, Metadata, Op, PixelsError, Producer, Region, Result,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A unique identifier for a graph node.
///
/// Identity is per-node, not per-op: two `crop` nodes with identical parameters
/// are distinct. The M1 evaluator memoizes on this so a shared subgraph
/// evaluates once, and M2's tile cache keys on `(NodeId, Region)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u64);

impl NodeId {
    /// The identifier's raw value, for diagnostics and cache keys.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Allocate the next process-unique identifier.
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// What a node does: originate pixels, or transform its inputs.
#[derive(Debug, Clone)]
enum NodeKind {
    /// A root node that produces pixels from a decoder or buffer.
    Source(Arc<dyn Producer>),
    /// An interior node that transforms its inputs.
    Op(Arc<dyn Op>),
}

/// A node in the immutable operation graph.
///
/// Nodes are never mutated after construction. Build them through [`Image`]
/// rather than directly.
#[derive(Debug)]
pub struct Node {
    id: NodeId,
    kind: NodeKind,
    inputs: Vec<Arc<Node>>,
    descriptor: ImageDescriptor,
}

impl Node {
    /// This node's unique identifier.
    #[must_use]
    pub const fn id(&self) -> NodeId {
        self.id
    }

    /// The shape of this node's output, resolved at build time.
    #[must_use]
    pub const fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    /// This node's inputs, in the order the op consumes them.
    #[must_use]
    pub fn inputs(&self) -> &[Arc<Node>] {
        &self.inputs
    }

    /// The op this node applies, or [`None`] if it is a source.
    #[must_use]
    pub fn op(&self) -> Option<&Arc<dyn Op>> {
        match &self.kind {
            NodeKind::Op(op) => Some(op),
            NodeKind::Source(_) => None,
        }
    }

    /// The producer at this node, or [`None`] if it is an interior op.
    #[must_use]
    pub fn producer(&self) -> Option<&Arc<dyn Producer>> {
        match &self.kind {
            NodeKind::Source(producer) => Some(producer),
            NodeKind::Op(_) => None,
        }
    }

    /// A short name for this node, for diagnostics.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match &self.kind {
            NodeKind::Source(producer) => producer.name(),
            NodeKind::Op(op) => op.name(),
        }
    }

    /// The number of nodes reachable from here, counting shared nodes once.
    ///
    /// Useful for asserting in tests that chaining built the graph it should
    /// have, and that a shared prefix really is shared.
    #[must_use]
    pub fn node_count(self: &Arc<Self>) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![Arc::clone(self)];
        while let Some(node) = stack.pop() {
            if !seen.insert(node.id) {
                continue;
            }
            stack.extend(node.inputs.iter().map(Arc::clone));
        }
        seen.len()
    }
}

impl Drop for Node {
    /// Tear the graph down iteratively.
    ///
    /// The derived drop glue would recurse once per node: a node drops its
    /// `inputs`, each of which drops *its* inputs, and so on. A long chain —
    /// which callers build simply by chaining many ops — would then overflow
    /// the stack at teardown, aborting the process. Since an abort is exactly
    /// what the failure model forbids, teardown uses an explicit worklist.
    ///
    /// Only uniquely-owned inputs are unwrapped: a node still shared by
    /// another branch of the DAG is left for its last owner to drop.
    fn drop(&mut self) {
        let mut stack: Vec<Arc<Self>> = std::mem::take(&mut self.inputs);
        while let Some(node) = stack.pop() {
            if let Some(mut node) = Arc::into_inner(node) {
                // Move the grandchildren onto our worklist before `node` drops,
                // so its own `drop` finds nothing left to recurse into.
                stack.append(&mut node.inputs);
            }
        }
    }
}

/// A lazily evaluated image: a handle onto one node of an op graph.
///
/// Constructing an `Image` and chaining ops onto it performs **no** pixel work
/// and reads no source bytes beyond the header (SPEC §Guarantees 3). Pixels
/// move only when a terminal pulls them.
///
/// `Image` is cheap to clone — it shares graph nodes rather than copying them —
/// and is `Send + Sync`.
#[derive(Debug, Clone)]
pub struct Image {
    node: Arc<Node>,
    format: Format,
}

impl Image {
    /// Build an image rooted at `producer`.
    ///
    /// `format` is the container the pixels came from, reported by
    /// [`Image::metadata`]. Use [`Format::Raw`] for pixels the caller supplied
    /// directly.
    #[must_use]
    pub fn from_producer(producer: Arc<dyn Producer>, format: Format) -> Self {
        let descriptor = producer.descriptor();
        Self {
            node: Arc::new(Node {
                id: NodeId::next(),
                kind: NodeKind::Source(producer),
                inputs: Vec::new(),
                descriptor,
            }),
            format,
        }
    }

    /// The graph node this handle points at.
    #[must_use]
    pub fn node(&self) -> &Arc<Node> {
        &self.node
    }

    /// The shape of this image, resolved when the node was built.
    #[must_use]
    pub fn descriptor(&self) -> ImageDescriptor {
        self.node.descriptor
    }

    /// Header-only facts about this image.
    ///
    /// Free: descriptors flowed forward at graph-build time, so this reads a
    /// field and decodes nothing.
    ///
    /// # Errors
    ///
    /// Infallible for graphs built through this API. It returns [`Result`] so
    /// that formats whose headers are parsed lazily can report a malformed
    /// header here without a breaking signature change.
    pub fn metadata(&self) -> Result<Metadata> {
        Ok(Metadata::new(&self.node.descriptor, self.format))
    }

    /// Chain a single-input op onto this image.
    ///
    /// The op's output descriptor is computed now, so an op that cannot apply
    /// to this input fails here rather than at evaluation time.
    ///
    /// # Errors
    ///
    /// Propagates [`Op::output_descriptor`], and returns
    /// [`PixelsError::Graph`] if `op` does not take exactly one input.
    pub fn apply(&self, op: Arc<dyn Op>) -> Result<Self> {
        Self::combine(std::slice::from_ref(self), op)
    }

    /// Chain a multi-input op over `inputs`.
    ///
    /// This is how ops like `composite` join two branches of a graph. The
    /// container format reported by [`Image::metadata`] is taken from the first
    /// input.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Graph`] if `inputs` is empty or its length does
    /// not match [`Op::arity`], and propagates [`Op::output_descriptor`].
    pub fn combine(inputs: &[Self], op: Arc<dyn Op>) -> Result<Self> {
        let Some(first) = inputs.first() else {
            return Err(PixelsError::graph(format!("op `{}` needs at least one input", op.name())));
        };
        if inputs.len() != op.arity() {
            return Err(PixelsError::graph(format!(
                "op `{}` takes {} input(s), got {}",
                op.name(),
                op.arity(),
                inputs.len()
            )));
        }
        let descriptors: Vec<ImageDescriptor> = inputs.iter().map(Self::descriptor).collect();
        let descriptor = op.output_descriptor(&descriptors)?;
        let format = first.format;
        Ok(Self {
            node: Arc::new(Node {
                id: NodeId::next(),
                kind: NodeKind::Op(op),
                inputs: inputs.iter().map(|image| Arc::clone(&image.node)).collect(),
                descriptor,
            }),
            format,
        })
    }

    /// The region covering this whole image.
    #[must_use]
    pub fn region(&self) -> Region {
        self.node.descriptor.region()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests operate on known-good values and assert shapes directly")]
mod tests {
    use super::*;
    use crate::testing::{ConstantOp, CountingProducer};
    use crate::{AccessPattern, Op, PixelFormat, Tile, TileMut};

    fn source(width: u32, height: u32) -> Image {
        let producer = CountingProducer::new(
            ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap(),
        );
        Image::from_producer(Arc::new(producer), Format::Raw)
    }

    #[test]
    fn chaining_builds_a_dag_without_touching_pixels() {
        let producer =
            Arc::new(CountingProducer::new(ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap()));
        let image = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Raw);
        let chained = image.apply(Arc::new(ConstantOp::new(7))).unwrap();
        let _ = chained.apply(Arc::new(ConstantOp::new(9))).unwrap();
        assert_eq!(producer.produce_calls(), 0, "graph construction must not pull pixels");
    }

    #[test]
    fn metadata_is_available_without_evaluation() {
        let producer =
            Arc::new(CountingProducer::new(ImageDescriptor::new(6, 3, PixelFormat::Gray8).unwrap()));
        let image = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Raw);
        let meta = image.metadata().unwrap();
        assert_eq!((meta.width, meta.height), (6, 3));
        assert_eq!(meta.format, Format::Raw);
        assert_eq!(meta.pixel, PixelFormat::Gray8);
        assert_eq!(producer.produce_calls(), 0);
    }

    #[test]
    fn descriptors_flow_forward_through_the_chain() {
        /// An op that halves its input's width, to prove shapes propagate.
        #[derive(Debug)]
        struct Halve;
        impl Op for Halve {
            fn name(&self) -> &'static str {
                "halve"
            }
            fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
                let input = inputs.first().ok_or_else(|| PixelsError::graph("no input"))?;
                input.resized(input.width / 2, input.height)
            }
            fn input_regions(&self, out: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
                Ok(vec![out])
            }
            fn access_pattern(&self) -> AccessPattern {
                AccessPattern::Sequential
            }
            fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
                Ok(())
            }
        }
        let image = source(16, 4).apply(Arc::new(Halve)).unwrap().apply(Arc::new(Halve)).unwrap();
        assert_eq!(image.descriptor().width, 4);
        assert_eq!(image.descriptor().height, 4);
    }

    #[test]
    fn an_op_rejecting_its_input_fails_at_build_time() {
        /// An op that refuses every input, to prove build-time validation.
        #[derive(Debug)]
        struct Refuses;
        impl Op for Refuses {
            fn name(&self) -> &'static str {
                "refuses"
            }
            fn output_descriptor(&self, _: &[ImageDescriptor]) -> Result<ImageDescriptor> {
                Err(PixelsError::unsupported("never applicable"))
            }
            fn input_regions(&self, out: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
                Ok(vec![out])
            }
            fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
                Ok(())
            }
        }
        let err = source(4, 4).apply(Arc::new(Refuses)).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Unsupported);
    }

    #[test]
    fn branches_share_their_common_prefix() {
        let base = source(4, 4).apply(Arc::new(ConstantOp::new(1))).unwrap();
        let left = base.apply(Arc::new(ConstantOp::new(2))).unwrap();
        let right = base.apply(Arc::new(ConstantOp::new(3))).unwrap();
        // Both branches point at the same prefix node, not a copy of it.
        assert_eq!(left.node().inputs()[0].id(), right.node().inputs()[0].id());
        // source + base + left = 3 distinct nodes on the left branch.
        assert_eq!(left.node().node_count(), 3);
    }

    #[test]
    fn cloning_an_image_shares_the_node() {
        let image = source(4, 4);
        let clone = image.clone();
        assert_eq!(image.node().id(), clone.node().id());
        assert!(Arc::ptr_eq(image.node(), clone.node()));
    }

    #[test]
    fn node_ids_are_unique() {
        let a = source(2, 2);
        let b = source(2, 2);
        assert_ne!(a.node().id(), b.node().id());
        assert_ne!(a.node().id().get(), b.node().id().get());
    }

    #[test]
    fn arity_mismatch_is_a_graph_error() {
        let image = source(4, 4);
        let err = Image::combine(&[image.clone(), image], Arc::new(ConstantOp::new(1)))
            .unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Graph);
        let err = Image::combine(&[], Arc::new(ConstantOp::new(1))).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Graph);
    }

    #[test]
    fn images_are_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Image>();
        assert_send_sync::<Arc<Node>>();
    }
}
