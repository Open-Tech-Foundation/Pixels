//! Shrink-on-load: deciding, from the whole graph, that the source can be
//! produced smaller.
//!
//! # Why this is a graph pass and not a decoder option
//!
//! Some formats can produce a reduced image far more cheaply than a full one —
//! JPEG most of all, where the low-frequency corner of a DCT block is a
//! smaller version of that block. But the useful size is the *pipeline's*
//! target, which is not known when the source is opened: `Image::from_stream`
//! runs before `.resize(200, 150)` is ever called. Only once the graph is
//! complete does the answer exist, and this pass is where it is computed.
//!
//! # When it fires
//!
//! Three conditions, all necessary:
//!
//! 1. The graph has exactly **one source**. With two, only one would shrink,
//!    and a `composite` would align pictures of different sizes.
//! 2. Every op is [`Op::scale_covariant`] — it means the same thing against a
//!    reduced input. `crop` and `composite` carry coordinates in source
//!    pixels; `convolve` carries a kernel in pixels. None of them do.
//! 3. Re-deriving every descriptor from the reduced source leaves the **root
//!    descriptor unchanged**. This is what distinguishes `resize(200, 150)`,
//!    which pins its output size and therefore absorbs the reduction, from a
//!    bare `flip`, which would simply emit a smaller image.
//!
//! Condition 3 is checked by simulation before anything is committed, because
//! reducing a source is irreversible — a stream cannot be rewound.
//!
//! # When it does not fire
//!
//! Nothing fails. A pipeline that cannot shrink decodes at full resolution,
//! which is what it did before this pass existed. That is deliberate: `crop`
//! is a legal thing to do to a JPEG, and refusing it to protect an
//! optimization would be the wrong trade. Whether it fired is reported in
//! [`RunStats`], so a pipeline that expected the fast path and did not get it
//! is diagnosable rather than silently slow.
//!
//! [`Op::scale_covariant`]: crate::Op::scale_covariant
//! [`RunStats`]: crate::RunStats

use crate::{Image, ImageDescriptor, Node, NodeId, Result};
use std::collections::HashMap;
use std::sync::Arc;

/// A source resolution the planner lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduction {
    /// The size the source would have decoded at.
    pub from: (u32, u32),
    /// The size it will decode at instead.
    pub to: (u32, u32),
}

impl Reduction {
    /// How many times fewer pixels the source now produces.
    #[must_use]
    pub fn factor(&self) -> f64 {
        let before = f64::from(self.from.0) * f64::from(self.from.1);
        let after = (f64::from(self.to.0) * f64::from(self.to.1)).max(1.0);
        before / after
    }
}

/// Rebuild `image` over a reduced source, if the graph permits it.
///
/// Returns the graph to evaluate and what was decided. The returned graph is
/// `image` itself when no reduction applies, so a caller can use the result
/// unconditionally.
///
/// # Errors
///
/// Returns [`PixelsError::Graph`] if the graph cannot be walked, or propagates
/// a producer's failure to commit to a reduction it had offered.
///
/// [`PixelsError::Graph`]: crate::PixelsError::Graph
pub fn shrink_on_load(image: &Image) -> Result<(Image, Option<Reduction>)> {
    let root = Arc::clone(image.node());
    let order = topological_order(&root);

    // Condition 2: every op must survive a change of input resolution, and
    // hand back an instance carrying no state bound to the old one. Each node
    // gets its own copy, which is then used for both the simulation and the
    // rebuild — so the tables an op memoizes while being simulated are the
    // tables the evaluated graph uses.
    let mut rescaled: HashMap<NodeId, Arc<dyn crate::Op>> = HashMap::new();
    for node in &order {
        let Some(op) = node.op() else { continue };
        let Some(copy) = op.rescaled() else {
            return Ok((image.clone(), None));
        };
        rescaled.insert(node.id(), copy);
    }

    // Condition 1: exactly one source.
    let sources: Vec<&Arc<Node>> = order
        .iter()
        .filter(|node| node.producer().is_some())
        .collect();
    let [source] = sources.as_slice() else {
        return Ok((image.clone(), None));
    };
    let Some(producer) = source.producer() else {
        return Ok((image.clone(), None));
    };

    let full = source.descriptor();
    let target = (root.descriptor().width, root.descriptor().height);
    let Some(reduced) = producer.reduced_descriptor(target) else {
        return Ok((image.clone(), None));
    };
    if (reduced.width, reduced.height) == (full.width, full.height) {
        return Ok((image.clone(), None));
    }

    // Condition 3: simulate the whole graph over the reduced source and check
    // the root is unmoved. Nothing is committed until this holds.
    let Some(simulated) = simulate(&order, source.id(), reduced, &rescaled) else {
        return Ok((image.clone(), None));
    };
    let Some(&new_root) = simulated.get(&root.id()) else {
        return Ok((image.clone(), None));
    };
    if (new_root.width, new_root.height) != (root.descriptor().width, root.descriptor().height) {
        return Ok((image.clone(), None));
    }

    // Committed from here: the producer will not decode at full size again.
    producer.reduce_to(reduced)?;
    let rebuilt = rebuild(
        &root,
        source.id(),
        producer,
        image,
        &rescaled,
        &mut HashMap::new(),
    )?;

    Ok((
        rebuilt,
        Some(Reduction {
            from: (full.width, full.height),
            to: (reduced.width, reduced.height),
        }),
    ))
}

/// Re-derive every node's descriptor with the source replaced.
///
/// Returns `None` if any op rejects the reduced shapes — an op that cannot
/// apply is a reason not to reduce, not an error: the un-reduced pipeline is
/// still perfectly valid.
fn simulate(
    order: &[Arc<Node>],
    source: NodeId,
    reduced: ImageDescriptor,
    rescaled: &HashMap<NodeId, Arc<dyn crate::Op>>,
) -> Option<HashMap<NodeId, ImageDescriptor>> {
    let mut descriptors: HashMap<NodeId, ImageDescriptor> = HashMap::with_capacity(order.len());
    for node in order {
        if node.id() == source {
            descriptors.insert(node.id(), reduced);
            continue;
        }
        // Every op node was rescaled above or the whole pass bailed, so a
        // miss here is a logic error. Refusing to reduce is the safe answer:
        // carrying the node's old descriptor forward instead would compare
        // the unreduced root against itself and wave the reduction through,
        // which is precisely the bug this line replaced.
        let op = rescaled.get(&node.id())?;
        let inputs: Vec<ImageDescriptor> = node
            .inputs()
            .iter()
            .filter_map(|input| descriptors.get(&input.id()).copied())
            .collect();
        if inputs.len() != node.inputs().len() {
            return None;
        }
        descriptors.insert(node.id(), op.output_descriptor(&inputs).ok()?);
    }
    Some(descriptors)
}

/// Build a fresh graph of the same ops over the now-reduced producer.
///
/// Nodes are immutable and shared, so the reduction cannot be applied in
/// place: a rebuilt graph is how the new descriptors reach every node. Shared
/// sub-graphs stay shared, through the memo.
fn rebuild(
    node: &Arc<Node>,
    source: NodeId,
    producer: &Arc<dyn crate::Producer>,
    original: &Image,
    rescaled: &HashMap<NodeId, Arc<dyn crate::Op>>,
    memo: &mut HashMap<NodeId, Image>,
) -> Result<Image> {
    if let Some(built) = memo.get(&node.id()) {
        return Ok(built.clone());
    }
    let built = if node.id() == source {
        // The producer now reports the reduced descriptor, so this picks it up
        // with no further arrangement.
        Image::from_producer(Arc::clone(producer), original.metadata()?.format)
    } else {
        // The rescaled copy, not the original: the original's tables are
        // bound to the shape it was built against.
        let op = rescaled
            .get(&node.id())
            .ok_or_else(|| crate::PixelsError::graph("an op was rebuilt without being rescaled"))?;
        let inputs = node
            .inputs()
            .iter()
            .map(|input| rebuild(input, source, producer, original, rescaled, memo))
            .collect::<Result<Vec<_>>>()?;
        Image::combine(&inputs, Arc::clone(op))?
    };
    memo.insert(node.id(), built.clone());
    Ok(built)
}

/// Every node reachable from `root`, inputs before the nodes that consume them.
fn topological_order(root: &Arc<Node>) -> Vec<Arc<Node>> {
    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    visit(root, &mut seen, &mut order);
    order
}

fn visit(
    node: &Arc<Node>,
    seen: &mut std::collections::HashSet<NodeId>,
    order: &mut Vec<Arc<Node>>,
) {
    if !seen.insert(node.id()) {
        return;
    }
    for input in node.inputs() {
        visit(input, seen, order);
    }
    order.push(Arc::clone(node));
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::{
        AccessPattern, DecodeCapability, Format, Op, PixelFormat, PixelsError, Producer, Region,
        TileMut,
    };
    use std::sync::Mutex;

    /// A producer that can halve itself, once.
    #[derive(Debug)]
    struct Shrinkable {
        descriptor: Mutex<ImageDescriptor>,
        reduced: Mutex<bool>,
    }

    impl Shrinkable {
        fn new(width: u32, height: u32) -> Arc<Self> {
            Arc::new(Self {
                descriptor: Mutex::new(
                    ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap(),
                ),
                reduced: Mutex::new(false),
            })
        }

        fn shape(&self) -> ImageDescriptor {
            *self.descriptor.lock().unwrap()
        }
    }

    impl Producer for Shrinkable {
        fn name(&self) -> &'static str {
            "shrinkable"
        }
        fn descriptor(&self) -> ImageDescriptor {
            self.shape()
        }
        fn capability(&self) -> DecodeCapability {
            DecodeCapability::Regions
        }
        fn produce(&self, _: Region, _: &mut TileMut<'_>) -> Result<()> {
            Ok(())
        }
        fn reduced_descriptor(&self, target: (u32, u32)) -> Option<ImageDescriptor> {
            let full = self.shape();
            let (half_width, half_height) = (full.width / 2, full.height / 2);
            if half_width >= target.0 && half_height >= target.1 {
                ImageDescriptor::new(half_width, half_height, full.pixel).ok()
            } else {
                None
            }
        }
        fn reduce_to(&self, descriptor: ImageDescriptor) -> Result<()> {
            *self.descriptor.lock().unwrap() = descriptor;
            *self.reduced.lock().unwrap() = true;
            Ok(())
        }
    }

    /// An op that resizes to a fixed size, like the real `resize`.
    #[derive(Debug)]
    struct FixedResize {
        width: u32,
        height: u32,
        covariant: bool,
    }

    impl Op for FixedResize {
        fn name(&self) -> &'static str {
            "fixed-resize"
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Spatial
        }
        fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
            let input = inputs
                .first()
                .ok_or_else(|| PixelsError::graph("no input"))?;
            ImageDescriptor::new(self.width, self.height, input.pixel)
        }
        fn input_regions(&self, _: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
            Ok(vec![inputs[0].region()])
        }
        fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
            Ok(())
        }
        fn rescaled(&self) -> Option<Arc<dyn Op>> {
            self.covariant.then(|| {
                Arc::new(Self {
                    width: self.width,
                    height: self.height,
                    covariant: true,
                }) as Arc<dyn Op>
            })
        }
    }

    use crate::Tile;

    /// An op that keeps its input's shape, like `flip`.
    #[derive(Debug)]
    struct SameShape {
        covariant: bool,
    }

    impl Op for SameShape {
        fn name(&self) -> &'static str {
            "same-shape"
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Sequential
        }
        fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
            inputs
                .first()
                .copied()
                .ok_or_else(|| PixelsError::graph("no input"))
        }
        fn input_regions(&self, output: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
            Ok(vec![output])
        }
        fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
            Ok(())
        }
        fn rescaled(&self) -> Option<Arc<dyn Op>> {
            self.covariant
                .then(|| Arc::new(Self { covariant: true }) as Arc<dyn Op>)
        }
    }

    fn source(width: u32, height: u32) -> (Image, Arc<Shrinkable>) {
        let producer = Shrinkable::new(width, height);
        let image = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Jpeg);
        (image, producer)
    }

    #[test]
    fn a_resize_pipeline_shrinks_its_source() {
        let (image, producer) = source(800, 600);
        let pipeline = image
            .apply(Arc::new(FixedResize {
                width: 100,
                height: 75,
                covariant: true,
            }))
            .unwrap();

        let (rebuilt, reduction) = shrink_on_load(&pipeline).unwrap();
        let reduction = reduction.expect("a resize to 1/8 should shrink the source");
        assert_eq!(reduction.from, (800, 600));
        assert_eq!(reduction.to, (400, 300));
        assert!((reduction.factor() - 4.0).abs() < 0.001);

        // The producer now reports the reduced size, and the rebuilt graph
        // still produces exactly what was asked for.
        assert_eq!(producer.shape().width, 400);
        assert_eq!(
            (rebuilt.descriptor().width, rebuilt.descriptor().height),
            (100, 75)
        );
    }

    #[test]
    fn an_op_that_is_not_scale_covariant_blocks_the_reduction() {
        let (image, producer) = source(800, 600);
        // Stands for `crop` or `convolve`: correctly shaped, wrong picture.
        let pipeline = image
            .apply(Arc::new(FixedResize {
                width: 100,
                height: 75,
                covariant: false,
            }))
            .unwrap();

        let (rebuilt, reduction) = shrink_on_load(&pipeline).unwrap();
        assert!(reduction.is_none(), "a non-covariant op must block it");
        assert_eq!(producer.shape().width, 800, "the source was reduced anyway");
        assert_eq!(rebuilt.descriptor().width, 100);
    }

    #[test]
    fn a_pipeline_with_no_resize_keeps_its_size() {
        // Every op is covariant, but nothing pins the output size, so
        // reducing the source would just emit a smaller image.
        let (image, producer) = source(800, 600);
        let pipeline = image
            .apply(Arc::new(SameShape { covariant: true }))
            .unwrap();

        let (rebuilt, reduction) = shrink_on_load(&pipeline).unwrap();
        assert!(
            reduction.is_none(),
            "reducing here would change the output size"
        );
        assert_eq!(producer.shape().width, 800);
        assert_eq!(rebuilt.descriptor().width, 800);
    }

    #[test]
    fn a_source_that_cannot_reduce_is_left_alone() {
        let descriptor = ImageDescriptor::new(64, 64, PixelFormat::Gray8).unwrap();
        let buffer = Arc::new(crate::TileBuf::for_image(&descriptor).unwrap());
        let image = Image::from_producer(
            Arc::new(crate::BufferSource::new(descriptor, buffer).unwrap()),
            Format::Raw,
        );
        let pipeline = image
            .apply(Arc::new(FixedResize {
                width: 8,
                height: 8,
                covariant: true,
            }))
            .unwrap();

        let (_, reduction) = shrink_on_load(&pipeline).unwrap();
        assert!(reduction.is_none(), "pixels in memory have one resolution");
    }

    #[test]
    fn the_reduction_never_goes_below_the_target() {
        // The stub halves only while the result still covers the target, so a
        // target just over half the source must leave it alone.
        let (image, producer) = source(800, 600);
        let pipeline = image
            .apply(Arc::new(FixedResize {
                width: 401,
                height: 301,
                covariant: true,
            }))
            .unwrap();

        let (_, reduction) = shrink_on_load(&pipeline).unwrap();
        assert!(reduction.is_none());
        assert_eq!(producer.shape().width, 800);
    }

    #[test]
    fn a_shared_subgraph_stays_shared_through_the_rebuild() {
        let (image, _) = source(800, 600);
        let resized = image
            .apply(Arc::new(FixedResize {
                width: 100,
                height: 75,
                covariant: true,
            }))
            .unwrap();
        // Two ops over one resize: the rebuilt graph must not duplicate the
        // shared prefix into two independent decodes of the same stream.
        let branch = resized
            .apply(Arc::new(SameShape { covariant: true }))
            .unwrap();

        let before = branch.node().node_count();
        let (rebuilt, reduction) = shrink_on_load(&branch).unwrap();
        assert!(reduction.is_some());
        assert_eq!(
            rebuilt.node().node_count(),
            before,
            "the rebuild changed the graph's shape"
        );
    }
}
