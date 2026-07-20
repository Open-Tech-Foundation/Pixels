//! The naive whole-image evaluator.
//!
//! This is M1's deliberate non-optimization: single-threaded, no tiling, one
//! full-image buffer per node. It exists to be **obviously correct**, so that
//! M2's tile scheduler can be diffed against it byte for byte (ROADMAP M2
//! exit criterion). Nothing here should get clever; when this evaluator and the
//! scheduler disagree, this one is presumed right.
//!
//! What it does share with M2 is the contracts: it drives [`Op::input_regions`]
//! and [`Op::compute`] exactly as the scheduler will, and it memoizes on
//! [`NodeId`] so a shared subgraph evaluates once — the same property M2's tile
//! cache provides at tile granularity.
//!
//! What it does **not** do is bound memory. Peak usage is proportional to the
//! live intermediates of the graph, so this path is not the constant-memory
//! guarantee in SPEC §Guarantees 1; M2 is.

use crate::{Image, Node, NodeId, PixelsError, Region, Result, Tile, TileBuf};
use std::collections::HashMap;
use std::sync::Arc;

/// Evaluate `image` to a whole-image buffer.
///
/// # Errors
///
/// Propagates any error from a producer or op in the graph. A failure anywhere
/// fails the whole evaluation — partial output is never returned
/// (ARCHITECTURE §Failure model).
pub fn evaluate(image: &Image) -> Result<TileBuf> {
    let mut cache: HashMap<NodeId, Arc<TileBuf>> = HashMap::new();
    let buffer = evaluate_node(image.node(), &mut cache)?;
    // The root is usually uniquely owned; unwrap the Arc when it is, and fall
    // back to a clone when the root is also an interior node of a shared graph.
    Ok(Arc::try_unwrap(buffer).unwrap_or_else(|shared| (*shared).clone()))
}

/// Evaluate one node, memoizing on [`NodeId`].
///
/// Iterative rather than recursive: graph depth is caller-controlled, and a
/// deeply chained pipeline must not overflow the stack. Nodes are pushed twice
/// — once to schedule their inputs, once to compute them after those inputs
/// are ready — which is a post-order walk without recursion.
fn evaluate_node(
    root: &Arc<Node>,
    cache: &mut HashMap<NodeId, Arc<TileBuf>>,
) -> Result<Arc<TileBuf>> {
    /// One step of the explicit work stack.
    enum Step {
        /// Ensure this node's inputs are scheduled.
        Visit(Arc<Node>),
        /// Inputs are ready; compute this node.
        Compute(Arc<Node>),
    }

    let mut stack = vec![Step::Visit(Arc::clone(root))];
    while let Some(step) = stack.pop() {
        match step {
            Step::Visit(node) => {
                if cache.contains_key(&node.id()) {
                    continue;
                }
                // Compute runs after every input has been popped and computed.
                stack.push(Step::Compute(Arc::clone(&node)));
                for input in node.inputs() {
                    stack.push(Step::Visit(Arc::clone(input)));
                }
            }
            Step::Compute(node) => {
                if cache.contains_key(&node.id()) {
                    continue;
                }
                let buffer = compute_node(&node, cache)?;
                cache.insert(node.id(), Arc::new(buffer));
            }
        }
    }

    cache
        .get(&root.id())
        .map(Arc::clone)
        .ok_or_else(|| PixelsError::graph("graph evaluation produced no result for the root node"))
}

/// Produce one node's whole-image buffer, with its inputs already cached.
fn compute_node(node: &Arc<Node>, cache: &HashMap<NodeId, Arc<TileBuf>>) -> Result<TileBuf> {
    let descriptor = node.descriptor();
    let output_region = descriptor.region();
    let mut output = TileBuf::for_image(&descriptor)?;

    if let Some(producer) = node.producer() {
        let mut tile = output.as_tile_mut()?;
        producer.produce(output_region, &mut tile)?;
        return Ok(output);
    }

    let op = node.op().ok_or_else(|| {
        PixelsError::graph(format!("node `{}` is neither op nor source", node.name()))
    })?;

    // Demand propagation, whole-image: ask the op what it needs for the entire
    // output. M2 asks the same question per tile.
    let input_descriptors: Vec<_> = node.inputs().iter().map(|n| n.descriptor()).collect();
    let requested = op.input_regions(output_region, &input_descriptors)?;
    if requested.len() != node.inputs().len() {
        return Err(PixelsError::graph(format!(
            "op `{}` requested {} input region(s) for {} input(s)",
            op.name(),
            requested.len(),
            node.inputs().len()
        )));
    }

    // Hold the input buffers alive for the duration of the borrow below.
    let input_buffers: Vec<Arc<TileBuf>> = node
        .inputs()
        .iter()
        .map(|input| {
            cache.get(&input.id()).map(Arc::clone).ok_or_else(|| {
                PixelsError::graph(format!("input `{}` was not evaluated first", input.name()))
            })
        })
        .collect::<Result<_>>()?;

    let mut input_tiles: Vec<Tile<'_>> = Vec::with_capacity(input_buffers.len());
    for (buffer, region) in input_buffers.iter().zip(&requested) {
        let tile = buffer.as_tile()?;
        if !tile.region().contains(*region) {
            return Err(PixelsError::graph(format!(
                "op `{}` asked for {region}, outside its input {}",
                op.name(),
                tile.region()
            )));
        }
        input_tiles.push(tile);
    }

    {
        let mut tile = output.as_tile_mut()?;
        op.compute(&input_tiles, &mut tile)?;
    }
    Ok(output)
}

/// Evaluate `image` and hand each output row, top to bottom, to `consume`.
///
/// This is the shape a sink pulls in: rows in order, so an encoder can write
/// incrementally. In M1 the rows come from an already-materialized buffer; in
/// M2 they arrive as the scheduler completes strips, and the consumer does not
/// change.
///
/// # Errors
///
/// Propagates evaluation errors, and any error `consume` returns.
pub fn evaluate_rows(
    image: &Image,
    mut consume: impl FnMut(u32, &[u8]) -> Result<()>,
) -> Result<()> {
    let buffer = evaluate(image)?;
    let tile = buffer.as_tile()?;
    let region = tile.region();
    for y in region.y..region.y.saturating_add(region.height) {
        let row = tile
            .row(y)
            .ok_or_else(|| PixelsError::graph(format!("evaluated buffer is missing row {y}")))?;
        consume(y, row)?;
    }
    Ok(())
}

/// The regions each node of `image`'s graph would be asked for, for diagnostics.
///
/// This walks the same inverse mapping the evaluator uses, without computing
/// any pixels — useful for asserting in tests that an op's demand propagation
/// is what it claims, and a stepping stone to M2's scheduling.
///
/// # Errors
///
/// Propagates [`Op::input_regions`].
///
/// [`Op::input_regions`]: crate::Op::input_regions
pub fn demand(image: &Image, output: Region) -> Result<Vec<(NodeId, Region)>> {
    let mut out = Vec::new();
    let mut stack = vec![(Arc::clone(image.node()), output)];
    while let Some((node, region)) = stack.pop() {
        out.push((node.id(), region));
        let Some(op) = node.op() else { continue };
        let descriptors: Vec<_> = node.inputs().iter().map(|n| n.descriptor()).collect();
        let requested = op.input_regions(region, &descriptors)?;
        for (input, region) in node.inputs().iter().zip(requested) {
            stack.push((Arc::clone(input), region));
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::testing::{ConstantOp, CountingProducer, RampProducer};
    use crate::{Format, ImageDescriptor, PixelFormat, Producer};

    fn ramp(width: u32, height: u32) -> Image {
        let desc = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
        Image::from_producer(Arc::new(RampProducer::new(desc)), Format::Raw)
    }

    #[test]
    fn a_bare_source_evaluates_to_its_pixels() {
        let buffer = evaluate(&ramp(3, 2)).unwrap();
        assert_eq!(buffer.bytes(), &[0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn ops_run_in_dependency_order() {
        let image = ramp(2, 2)
            .apply(Arc::new(ConstantOp::new(5)))
            .unwrap()
            .apply(Arc::new(ConstantOp::new(9)))
            .unwrap();
        assert_eq!(evaluate(&image).unwrap().bytes(), &[9, 9, 9, 9]);
    }

    #[test]
    fn a_shared_subgraph_is_evaluated_once() {
        let desc = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let producer = Arc::new(CountingProducer::new(desc));
        let base = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Raw);
        // Two branches over one source, joined so both are pulled.
        let left = base.apply(Arc::new(ConstantOp::new(1))).unwrap();
        let right = base.apply(Arc::new(ConstantOp::new(2))).unwrap();
        let joined = Image::combine(&[left, right], Arc::new(crate::testing::SumOp)).unwrap();
        assert_eq!(evaluate(&joined).unwrap().bytes(), &[3, 3, 3, 3]);
        assert_eq!(
            producer.produce_calls(),
            1,
            "shared source pulled exactly once"
        );
    }

    #[test]
    fn nothing_is_pulled_until_a_terminal_runs() {
        let desc = ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap();
        let producer = Arc::new(CountingProducer::new(desc));
        let image = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Raw)
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap();
        assert_eq!(producer.produce_calls(), 0);
        let _ = image.metadata().unwrap();
        assert_eq!(producer.produce_calls(), 0, "metadata must not pull pixels");
        evaluate(&image).unwrap();
        assert_eq!(producer.produce_calls(), 1);
    }

    #[test]
    fn deep_chains_do_not_overflow_the_stack() {
        let mut image = ramp(2, 2);
        for _ in 0..10_000 {
            image = image.apply(Arc::new(ConstantOp::new(3))).unwrap();
        }
        assert_eq!(evaluate(&image).unwrap().bytes(), &[3, 3, 3, 3]);
    }

    #[test]
    fn a_failing_op_fails_the_whole_evaluation() {
        let image = ramp(2, 2)
            .apply(Arc::new(crate::testing::FailingOp))
            .unwrap();
        let err = evaluate(&image).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Malformed);
    }

    #[test]
    fn evaluate_rows_yields_rows_in_order() {
        let mut seen = Vec::new();
        evaluate_rows(&ramp(2, 3), |y, row| {
            seen.push((y, row.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(
            seen,
            vec![(0, vec![0, 1]), (1, vec![2, 3]), (2, vec![4, 5])]
        );
    }

    #[test]
    fn evaluate_rows_propagates_consumer_errors() {
        let err = evaluate_rows(&ramp(2, 3), |_, _| {
            Err(PixelsError::unsupported("sink refused"))
        })
        .unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Unsupported);
    }

    #[test]
    fn demand_walks_the_inverse_mapping_without_computing() {
        let desc = ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap();
        let producer = Arc::new(CountingProducer::new(desc));
        let image = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Raw)
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap();
        let pairs = demand(&image, Region::from_size(4, 4)).unwrap();
        assert_eq!(pairs.len(), 2, "one op node and one source node");
        assert_eq!(
            producer.produce_calls(),
            0,
            "demand propagation touches no pixels"
        );
    }
}
