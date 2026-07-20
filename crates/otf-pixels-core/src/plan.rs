//! Static analysis of a graph, done once before any pixel moves.
//!
//! A [`Plan`] answers two questions the scheduler needs and neither the ops nor
//! the codecs can answer alone:
//!
//! 1. **What shape are tiles?** Runs of [`AccessPattern::Sequential`] ops move
//!    full-width strips; segments containing a [`AccessPattern::Spatial`] op
//!    move square tiles (ADR-0003).
//! 2. **Where must the pipeline materialize?** Where the region sequence a node
//!    is asked for is not forward-monotonic, and the pixels below it come from
//!    a forward-only source (ADR-0009).
//!
//! Both are derived from what ops already declare — [`Op::access_pattern`] and
//! [`Op::input_regions`] — so an op cannot get the analysis wrong separately
//! from getting its own contract wrong.

use crate::{AccessPattern, DecodeCapability, Image, Node, NodeId, PixelsError, Region, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// The shape tiles take through a segment of the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileShape {
    /// Full-width strips of at most `rows` rows.
    ///
    /// Matches how codecs produce and consume pixels, so a sequential segment
    /// moves data with no repacking.
    Strip {
        /// Maximum rows per strip; the final strip may be shorter.
        rows: u32,
    },
    /// Square tiles of at most `size` × `size` pixels.
    ///
    /// Used where a spatial op needs a neighbourhood, so that border pixels are
    /// recomputed for four edges rather than for the full image width.
    Square {
        /// Maximum edge length; edge tiles may be smaller.
        size: u32,
    },
}

impl TileShape {
    /// The default strip height, in rows.
    pub const DEFAULT_STRIP_ROWS: u32 = 64;
    /// The default square tile edge, in pixels (ADR-0003).
    pub const DEFAULT_SQUARE_SIZE: u32 = 128;

    /// Split `region` into tiles of this shape, in production order.
    ///
    /// Order is top-to-bottom, then left-to-right, so a sink consuming the
    /// result sees rows in order wherever the shape allows.
    #[must_use]
    pub fn tiles(self, region: Region) -> Vec<Region> {
        if region.is_empty() {
            return Vec::new();
        }
        let (tile_width, tile_height) = match self {
            Self::Strip { rows } => (region.width, rows.max(1)),
            Self::Square { size } => (size.max(1), size.max(1)),
        };
        let mut tiles = Vec::new();
        let mut y = region.y;
        while u64::from(y) < region.bottom() {
            let height = tile_height.min((region.bottom() - u64::from(y)) as u32);
            let mut x = region.x;
            while u64::from(x) < region.right() {
                let width = tile_width.min((region.right() - u64::from(x)) as u32);
                tiles.push(Region::new(x, y, width, height));
                x = x.saturating_add(width);
            }
            y = y.saturating_add(height);
        }
        tiles
    }
}

/// How the scheduler must treat one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NodePlan {
    /// The shape tiles of this node's output take.
    pub shape: TileShape,
    /// Whether this node's whole output must be realized before its consumers
    /// pull from it.
    ///
    /// Set where demand on this node is not forward-monotonic and the pixels
    /// below come from a forward-only source (ADR-0009). It is a property of
    /// this *position in this pipeline*, not of the node's op.
    pub materialize: bool,
    /// Whether this node's tiles are worth retaining in the tile cache.
    ///
    /// Only nodes whose output is genuinely demanded more than once qualify:
    /// a shared graph prefix, or a node feeding a spatial op whose tile
    /// requests overlap. In a linear pipeline every tile is produced once and
    /// consumed once, so caching it would be pure waste — and, worse, would
    /// fill the byte budget with garbage and make a streaming pipeline's
    /// memory look like the cache budget rather than a few tiles.
    pub cacheable: bool,
}

/// The analysis result for one pipeline.
#[derive(Debug, Clone)]
pub struct Plan {
    nodes: HashMap<NodeId, NodePlan>,
    output_tiles: Vec<Region>,
    root: NodeId,
}

/// Knobs for [`Plan::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlanOptions {
    /// Rows per strip in sequential segments.
    pub strip_rows: u32,
    /// Edge length of square tiles in spatial segments.
    pub square_size: u32,
}

impl PlanOptions {
    /// Set the rows per strip in sequential segments.
    ///
    /// `PlanOptions` is `#[non_exhaustive]`, so downstream crates cannot use a
    /// struct literal; these setters are the only way to configure it.
    #[must_use]
    pub const fn with_strip_rows(mut self, rows: u32) -> Self {
        self.strip_rows = rows;
        self
    }

    /// Set the edge length of square tiles in spatial segments.
    #[must_use]
    pub const fn with_square_size(mut self, size: u32) -> Self {
        self.square_size = size;
        self
    }
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            strip_rows: TileShape::DEFAULT_STRIP_ROWS,
            square_size: TileShape::DEFAULT_SQUARE_SIZE,
        }
    }
}

impl Plan {
    /// Analyse `image`'s graph.
    ///
    /// # Errors
    ///
    /// Propagates [`Op::input_regions`] for any op whose demand mapping
    /// rejects a region this plan would ask for.
    ///
    /// [`Op::input_regions`]: crate::Op::input_regions
    pub fn build(image: &Image, options: PlanOptions) -> Result<Self> {
        let root = Arc::clone(image.node());
        let order = topological_order(&root);

        // 1. Tile shapes. A node's output is square if it or any consumer is
        //    spatial: a spatial op needs its input as squares, and producing
        //    squares from a strip producer is what the seam line-cache is for.
        let spatial_consumers = spatial_consumers(&order);
        let mut nodes = HashMap::with_capacity(order.len());
        for node in &order {
            let is_spatial = node
                .op()
                .is_some_and(|op| op.access_pattern() == AccessPattern::Spatial);
            let feeds_spatial = spatial_consumers.contains(&node.id());
            let shape = if is_spatial || feeds_spatial {
                TileShape::Square {
                    size: options.square_size,
                }
            } else {
                TileShape::Strip {
                    rows: options.strip_rows,
                }
            };
            nodes.insert(
                node.id(),
                NodePlan {
                    shape,
                    materialize: false,
                    // Filled in below, once fan-out is known.
                    cacheable: false,
                },
            );
        }

        // 2. The output tile sequence, in the order the sink will pull it.
        let root_shape = nodes
            .get(&root.id())
            .ok_or_else(|| PixelsError::graph("root node missing from plan"))?
            .shape;
        let output_tiles = root_shape.tiles(root.descriptor().region());

        // 3. Demand order. Replay the whole output sequence through
        //    `input_regions` and record what each node is asked for, in order.
        let demand = demand_sequences(&root, &output_tiles)?;

        // 4. Retention. A tile is only worth caching if something will ask
        //    for it twice: a shared prefix (fan-out above one) or a spatial
        //    consumer whose tile requests overlap at the borders.
        let fan_out = fan_out(&order);
        for node in &order {
            let shared = fan_out.get(&node.id()).copied().unwrap_or(0) > 1;
            let overlapping = spatial_consumers.contains(&node.id());
            if let Some(plan) = nodes.get_mut(&node.id()) {
                plan.cacheable = shared || overlapping;
            }
        }

        // 5. Materialize where non-monotonic demand meets a forward-only
        //    source (ADR-0009).
        let forward_only = forward_only_nodes(&order);
        for node in &order {
            let Some(regions) = demand.get(&node.id()) else {
                continue;
            };
            if is_forward_monotonic(regions) || !forward_only.contains(&node.id()) {
                continue;
            }
            if let Some(plan) = nodes.get_mut(&node.id()) {
                plan.materialize = true;
            }
        }

        Ok(Self {
            nodes,
            output_tiles,
            root: root.id(),
        })
    }

    /// The plan for one node, if it is part of this pipeline.
    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<NodePlan> {
        self.nodes.get(&id).copied()
    }

    /// The output regions the sink will pull, in order.
    #[must_use]
    pub fn output_tiles(&self) -> &[Region] {
        &self.output_tiles
    }

    /// The root node this plan was built for.
    #[must_use]
    pub const fn root(&self) -> NodeId {
        self.root
    }

    /// How many nodes the plan covers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the plan covers no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Whether any node in the pipeline must be materialized.
    ///
    /// A pipeline with no materialization points runs in memory bounded by
    /// tiles in flight; one with them buffers a full intermediate per point
    /// (SPEC §Guarantees 1).
    #[must_use]
    pub fn materializes(&self) -> bool {
        self.nodes.values().any(|plan| plan.materialize)
    }
}

/// Nodes in dependency order: every node appears after its inputs.
fn topological_order(root: &Arc<Node>) -> Vec<Arc<Node>> {
    /// One step of the explicit walk, mirroring the evaluator's.
    enum Step {
        Visit(Arc<Node>),
        Emit(Arc<Node>),
    }
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![Step::Visit(Arc::clone(root))];
    while let Some(step) = stack.pop() {
        match step {
            Step::Visit(node) => {
                if seen.contains(&node.id()) {
                    continue;
                }
                stack.push(Step::Emit(Arc::clone(&node)));
                for input in node.inputs() {
                    stack.push(Step::Visit(Arc::clone(input)));
                }
            }
            Step::Emit(node) => {
                if seen.insert(node.id()) {
                    order.push(node);
                }
            }
        }
    }
    order
}

/// How many consumers each node has within this graph.
fn fan_out(order: &[Arc<Node>]) -> HashMap<NodeId, usize> {
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for node in order {
        for input in node.inputs() {
            *counts.entry(input.id()).or_default() += 1;
        }
    }
    counts
}

/// Nodes that feed at least one spatial op.
fn spatial_consumers(order: &[Arc<Node>]) -> HashSet<NodeId> {
    let mut feeding = HashSet::new();
    for node in order {
        let spatial = node
            .op()
            .is_some_and(|op| op.access_pattern() == AccessPattern::Spatial);
        if spatial {
            for input in node.inputs() {
                feeding.insert(input.id());
            }
        }
    }
    feeding
}

/// Nodes whose pixels ultimately come from a forward-only producer.
///
/// A node inherits the constraint from its inputs: if anything below it can
/// only go forward, it can only be produced going forward.
fn forward_only_nodes(order: &[Arc<Node>]) -> HashSet<NodeId> {
    let mut forward_only = HashSet::new();
    // `order` is topological, so inputs are classified before their consumers.
    for node in order {
        let constrained = match node.producer() {
            Some(producer) => producer.capability() == DecodeCapability::Sequential,
            None => node
                .inputs()
                .iter()
                .any(|input| forward_only.contains(&input.id())),
        };
        if constrained {
            forward_only.insert(node.id());
        }
    }
    forward_only
}

/// Replay the output tile sequence, recording what each node is asked for.
fn demand_sequences(
    root: &Arc<Node>,
    output_tiles: &[Region],
) -> Result<HashMap<NodeId, Vec<Region>>> {
    let mut sequences: HashMap<NodeId, Vec<Region>> = HashMap::new();
    for tile in output_tiles {
        let mut stack = vec![(Arc::clone(root), *tile)];
        while let Some((node, region)) = stack.pop() {
            sequences.entry(node.id()).or_default().push(region);
            let Some(op) = node.op() else { continue };
            let descriptors: Vec<_> = node.inputs().iter().map(|n| n.descriptor()).collect();
            let requested = op.input_regions(region, &descriptors)?;
            if requested.len() != node.inputs().len() {
                return Err(PixelsError::graph(format!(
                    "op `{}` requested {} input region(s) for {} input(s)",
                    op.name(),
                    requested.len(),
                    node.inputs().len()
                )));
            }
            for (input, region) in node.inputs().iter().zip(requested) {
                stack.push((Arc::clone(input), region));
            }
        }
    }
    Ok(sequences)
}

/// Whether a demand sequence only ever moves forward through the image.
///
/// "Forward" means each request starts no earlier than the previous one
/// started — the exact condition a forward-only source can satisfy. Repeats and
/// overlaps are fine (a rolling window covers them); going backwards is not.
fn is_forward_monotonic(regions: &[Region]) -> bool {
    regions.windows(2).all(|pair| {
        let (Some(previous), Some(next)) = (pair.first(), pair.get(1)) else {
            return true;
        };
        next.y >= previous.y
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::testing::{ConstantOp, CountingProducer};
    use crate::{
        BufferSource, Format, ImageDescriptor, Op, PixelFormat, Producer, Tile, TileBuf, TileMut,
    };

    /// A source with a chosen capability, to drive ADR-0009's analysis.
    #[derive(Debug)]
    struct Source {
        descriptor: ImageDescriptor,
        capability: DecodeCapability,
    }

    impl Producer for Source {
        fn name(&self) -> &'static str {
            "test-source"
        }
        fn descriptor(&self) -> ImageDescriptor {
            self.descriptor
        }
        fn capability(&self) -> DecodeCapability {
            self.capability
        }
        fn produce(&self, _: Region, _: &mut TileMut<'_>) -> Result<()> {
            Ok(())
        }
    }

    fn image(width: u32, height: u32, capability: DecodeCapability) -> Image {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
        Image::from_producer(
            Arc::new(Source {
                descriptor,
                capability,
            }),
            Format::Raw,
        )
    }

    /// An op that mirrors vertically, like `Flip`: demand runs backwards.
    #[derive(Debug)]
    struct Reverse;
    impl Op for Reverse {
        fn name(&self) -> &'static str {
            "reverse"
        }
        fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
            inputs
                .first()
                .copied()
                .ok_or_else(|| PixelsError::graph("no input"))
        }
        fn input_regions(&self, out: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
            let input = inputs
                .first()
                .ok_or_else(|| PixelsError::graph("no input"))?;
            let y = (u64::from(input.height) - out.bottom()) as u32;
            Ok(vec![Region::new(out.x, y, out.width, out.height)])
        }
        fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
            Ok(())
        }
    }

    /// A neighbourhood op, to exercise square-tile negotiation.
    #[derive(Debug)]
    struct Blur;
    impl Op for Blur {
        fn name(&self) -> &'static str {
            "blur"
        }
        fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
            inputs
                .first()
                .copied()
                .ok_or_else(|| PixelsError::graph("no input"))
        }
        fn input_regions(&self, out: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
            let input = inputs
                .first()
                .ok_or_else(|| PixelsError::graph("no input"))?;
            // Grown by one pixel, clamped to the image.
            Ok(vec![Region::new(
                out.x.saturating_sub(1),
                out.y.saturating_sub(1),
                (out.width + 2).min(input.width),
                (out.height + 2).min(input.height),
            )])
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Spatial
        }
        fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
            Ok(())
        }
    }

    // --- Tile shape splitting ------------------------------------------------

    #[test]
    fn strips_cover_the_region_exactly_once() {
        let region = Region::from_size(10, 25);
        let tiles = TileShape::Strip { rows: 10 }.tiles(region);
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0], Region::new(0, 0, 10, 10));
        assert_eq!(tiles[1], Region::new(0, 10, 10, 10));
        assert_eq!(tiles[2], Region::new(0, 20, 10, 5), "final strip is short");
        assert_eq!(
            tiles.iter().map(|t| t.pixel_count()).sum::<u64>(),
            region.pixel_count()
        );
    }

    #[test]
    fn squares_cover_the_region_exactly_once() {
        let region = Region::from_size(10, 10);
        let tiles = TileShape::Square { size: 4 }.tiles(region);
        assert_eq!(tiles.len(), 9, "3x3 grid of tiles over a 10x10 image");
        assert_eq!(tiles[0], Region::new(0, 0, 4, 4));
        assert_eq!(tiles[2], Region::new(8, 0, 2, 4), "right edge is narrow");
        assert_eq!(tiles[8], Region::new(8, 8, 2, 2), "corner is small");
        assert_eq!(
            tiles.iter().map(|t| t.pixel_count()).sum::<u64>(),
            region.pixel_count()
        );
    }

    #[test]
    fn tiles_are_produced_top_to_bottom() {
        let tiles = TileShape::Square { size: 4 }.tiles(Region::from_size(8, 8));
        let ys: Vec<u32> = tiles.iter().map(|t| t.y).collect();
        assert!(
            ys.windows(2).all(|w| w[0] <= w[1]),
            "rows must not go backwards"
        );
    }

    #[test]
    fn splitting_handles_degenerate_inputs() {
        assert!(TileShape::Strip { rows: 4 }.tiles(Region::EMPTY).is_empty());
        // A zero tile size is clamped rather than looping forever.
        assert_eq!(
            TileShape::Strip { rows: 0 }
                .tiles(Region::from_size(2, 2))
                .len(),
            2
        );
        assert_eq!(
            TileShape::Square { size: 0 }
                .tiles(Region::from_size(2, 2))
                .len(),
            4
        );
        // A tile larger than the image yields exactly one tile.
        assert_eq!(
            TileShape::Strip { rows: 999 }
                .tiles(Region::from_size(4, 4))
                .len(),
            1
        );
    }

    // --- Tile shape negotiation (ADR-0003) -----------------------------------

    #[test]
    fn a_sequential_pipeline_moves_strips() {
        let image = image(64, 64, DecodeCapability::Regions)
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap();
        let plan = Plan::build(&image, PlanOptions::default()).unwrap();
        for node in [image.node().id(), image.node().inputs()[0].id()] {
            assert!(
                matches!(plan.node(node).unwrap().shape, TileShape::Strip { .. }),
                "sequential nodes should move strips"
            );
        }
    }

    #[test]
    fn a_spatial_op_switches_its_segment_to_squares() {
        // ADR-0003: the spatial op and the node feeding it both go square.
        let source = image(64, 64, DecodeCapability::Regions);
        let blurred = source.apply(Arc::new(Blur)).unwrap();
        let plan = Plan::build(&blurred, PlanOptions::default()).unwrap();
        assert!(matches!(
            plan.node(blurred.node().id()).unwrap().shape,
            TileShape::Square { .. }
        ));
        assert!(
            matches!(
                plan.node(source.node().id()).unwrap().shape,
                TileShape::Square { .. }
            ),
            "the node feeding a spatial op must deliver squares"
        );
    }

    #[test]
    fn sequential_nodes_above_a_spatial_op_stay_strips() {
        // Only the segment around the spatial op changes shape.
        let source = image(64, 64, DecodeCapability::Regions);
        let blurred = source.apply(Arc::new(Blur)).unwrap();
        let after = blurred.apply(Arc::new(ConstantOp::new(1))).unwrap();
        let plan = Plan::build(&after, PlanOptions::default()).unwrap();
        assert!(
            matches!(
                plan.node(after.node().id()).unwrap().shape,
                TileShape::Strip { .. }
            ),
            "a sequential consumer keeps strips"
        );
    }

    #[test]
    fn plan_options_choose_the_tile_dimensions() {
        let image = image(100, 100, DecodeCapability::Regions);
        let options = PlanOptions {
            strip_rows: 25,
            square_size: 10,
        };
        let plan = Plan::build(&image, options).unwrap();
        assert_eq!(
            plan.node(image.node().id()).unwrap().shape,
            TileShape::Strip { rows: 25 }
        );
        assert_eq!(plan.output_tiles().len(), 4);
    }

    // --- Demand order and materialization (ADR-0009) -------------------------

    #[test]
    fn a_forward_pipeline_never_materializes() {
        let image = image(256, 256, DecodeCapability::Sequential)
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap();
        let plan = Plan::build(&image, PlanOptions::default()).unwrap();
        assert!(!plan.materializes(), "forward demand needs no buffer");
    }

    #[test]
    fn reversal_over_a_sequential_source_materializes() {
        // The ADR-0009 case: non-forward demand meeting a forward-only source.
        let source = image(256, 256, DecodeCapability::Sequential);
        let reversed = source.apply(Arc::new(Reverse)).unwrap();
        let plan = Plan::build(&reversed, PlanOptions::default()).unwrap();
        assert!(
            plan.materializes(),
            "reversal over a forward-only source must buffer"
        );
        assert!(
            plan.node(source.node().id()).unwrap().materialize,
            "the buffer belongs at the source being read backwards"
        );
        assert!(
            !plan.node(reversed.node().id()).unwrap().materialize,
            "the reversing node itself is pulled in output order"
        );
    }

    #[test]
    fn reversal_over_a_random_access_source_streams() {
        // The point of ADR-0009: the good case is not sacrificed to the bad
        // one. Same pipeline, different upstream, no buffer.
        let source = image(256, 256, DecodeCapability::Regions);
        let reversed = source.apply(Arc::new(Reverse)).unwrap();
        let plan = Plan::build(&reversed, PlanOptions::default()).unwrap();
        assert!(
            !plan.materializes(),
            "a random-access source serves bands in any order"
        );
    }

    #[test]
    fn a_memory_buffer_source_streams_under_reversal() {
        // The same property, through the real BufferSource rather than a stub.
        let descriptor = ImageDescriptor::new(32, 32, PixelFormat::Gray8).unwrap();
        let buffer = Arc::new(TileBuf::for_image(&descriptor).unwrap());
        let source = BufferSource::new(descriptor, buffer).unwrap();
        let image = Image::from_producer(Arc::new(source), Format::Raw)
            .apply(Arc::new(Reverse))
            .unwrap();
        let plan = Plan::build(&image, PlanOptions::default()).unwrap();
        assert!(!plan.materializes());
    }

    #[test]
    fn the_constraint_propagates_down_a_chain() {
        // A sequential source seen through several ops still forces a buffer.
        let source = image(128, 128, DecodeCapability::Sequential);
        let image = source
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap()
            .apply(Arc::new(ConstantOp::new(2)))
            .unwrap()
            .apply(Arc::new(Reverse))
            .unwrap();
        let plan = Plan::build(&image, PlanOptions::default()).unwrap();
        assert!(plan.materializes());
    }

    #[test]
    fn a_single_tile_pipeline_is_trivially_forward() {
        // One output tile means one demand per node: nothing can go backwards.
        let source = image(8, 8, DecodeCapability::Sequential);
        let reversed = source.apply(Arc::new(Reverse)).unwrap();
        let options = PlanOptions {
            strip_rows: 64,
            square_size: 128,
        };
        let plan = Plan::build(&reversed, options).unwrap();
        assert_eq!(plan.output_tiles().len(), 1);
        assert!(
            !plan.materializes(),
            "a single whole-image tile never reverses"
        );
    }

    #[test]
    fn monotonicity_allows_repeats_and_overlap() {
        // A rolling window covers overlap; only going backwards is fatal.
        assert!(is_forward_monotonic(&[]));
        assert!(is_forward_monotonic(&[Region::new(0, 0, 4, 4)]));
        assert!(is_forward_monotonic(&[
            Region::new(0, 0, 4, 4),
            Region::new(0, 4, 4, 4)
        ]));
        assert!(is_forward_monotonic(&[
            Region::new(0, 0, 4, 4),
            Region::new(0, 0, 4, 4)
        ]));
        assert!(is_forward_monotonic(&[
            Region::new(0, 0, 4, 8),
            Region::new(0, 4, 4, 8)
        ]));
        assert!(!is_forward_monotonic(&[
            Region::new(0, 4, 4, 4),
            Region::new(0, 0, 4, 4)
        ]));
    }

    // --- Plan shape ----------------------------------------------------------

    #[test]
    fn a_linear_pipeline_caches_nothing() {
        // Every tile in a chain is produced once and consumed once, so
        // retaining any of them is pure waste — and would make a streaming
        // pipeline's peak memory equal the cache budget.
        let source = image(64, 64, DecodeCapability::Regions);
        let a = source.apply(Arc::new(ConstantOp::new(1))).unwrap();
        let b = a.apply(Arc::new(ConstantOp::new(2))).unwrap();
        let plan = Plan::build(&b, PlanOptions::default()).unwrap();
        for node in [source.node().id(), a.node().id(), b.node().id()] {
            assert!(
                !plan.node(node).unwrap().cacheable,
                "a linear chain cached a tile"
            );
        }
    }

    #[test]
    fn a_shared_prefix_is_cacheable() {
        // Fan-out above one is the case the cache exists for: without it the
        // shared prefix is recomputed once per branch.
        let source = image(32, 32, DecodeCapability::Regions);
        let base = source.apply(Arc::new(ConstantOp::new(1))).unwrap();
        let left = base.apply(Arc::new(ConstantOp::new(2))).unwrap();
        let right = base.apply(Arc::new(ConstantOp::new(3))).unwrap();
        let joined =
            Image::combine(&[left.clone(), right], Arc::new(crate::testing::SumOp)).unwrap();
        let plan = Plan::build(&joined, PlanOptions::default()).unwrap();
        assert!(
            plan.node(base.node().id()).unwrap().cacheable,
            "shared prefix not cached"
        );
        assert!(
            !plan.node(left.node().id()).unwrap().cacheable,
            "single consumer cached"
        );
        assert!(
            !plan.node(joined.node().id()).unwrap().cacheable,
            "the root has no consumer"
        );
    }

    #[test]
    fn a_node_feeding_a_spatial_op_is_cacheable() {
        // Neighbourhood demand overlaps at tile borders, so those tiles are
        // genuinely asked for more than once.
        let source = image(64, 64, DecodeCapability::Regions);
        let blurred = source.apply(Arc::new(Blur)).unwrap();
        let plan = Plan::build(&blurred, PlanOptions::default()).unwrap();
        assert!(plan.node(source.node().id()).unwrap().cacheable);
    }

    #[test]
    fn the_plan_covers_every_node_once() {
        let source = image(32, 32, DecodeCapability::Regions);
        let a = source.apply(Arc::new(ConstantOp::new(1))).unwrap();
        let b = a.apply(Arc::new(ConstantOp::new(2))).unwrap();
        let plan = Plan::build(&b, PlanOptions::default()).unwrap();
        assert_eq!(plan.len(), 3);
        assert!(!plan.is_empty());
        assert_eq!(plan.root(), b.node().id());
        assert!(plan.node(source.node().id()).is_some());
    }

    #[test]
    fn a_shared_prefix_is_planned_once() {
        let source = image(32, 32, DecodeCapability::Regions);
        let base = source.apply(Arc::new(ConstantOp::new(1))).unwrap();
        let left = base.apply(Arc::new(ConstantOp::new(2))).unwrap();
        let right = base.apply(Arc::new(ConstantOp::new(3))).unwrap();
        let joined = Image::combine(&[left, right], Arc::new(crate::testing::SumOp)).unwrap();
        let plan = Plan::build(&joined, PlanOptions::default()).unwrap();
        // source + base + left + right + join = 5 distinct nodes.
        assert_eq!(plan.len(), 5);
    }

    #[test]
    fn output_tiles_tile_the_whole_image() {
        let image = image(100, 70, DecodeCapability::Regions);
        let plan = Plan::build(
            &image,
            PlanOptions {
                strip_rows: 16,
                square_size: 128,
            },
        )
        .unwrap();
        let covered: u64 = plan.output_tiles().iter().map(|t| t.pixel_count()).sum();
        assert_eq!(covered, 100 * 70);
        assert_eq!(plan.output_tiles().len(), 5, "70 rows in strips of 16");
    }

    #[test]
    fn planning_touches_no_pixels() {
        let descriptor = ImageDescriptor::new(64, 64, PixelFormat::Gray8).unwrap();
        let producer = Arc::new(CountingProducer::new(descriptor));
        let image = Image::from_producer(Arc::clone(&producer) as Arc<dyn Producer>, Format::Raw)
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap();
        let _plan = Plan::build(&image, PlanOptions::default()).unwrap();
        assert_eq!(producer.produce_calls(), 0, "planning is pure analysis");
    }

    #[test]
    fn an_op_rejecting_a_planned_region_fails_the_plan() {
        /// An op whose demand mapping refuses everything.
        #[derive(Debug)]
        struct Refuses;
        impl Op for Refuses {
            fn name(&self) -> &'static str {
                "refuses"
            }
            fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
                inputs
                    .first()
                    .copied()
                    .ok_or_else(|| PixelsError::graph("no input"))
            }
            fn input_regions(&self, _: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
                Err(PixelsError::invalid_argument("output", "never satisfiable"))
            }
            fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
                Ok(())
            }
        }
        let image = image(32, 32, DecodeCapability::Regions)
            .apply(Arc::new(Refuses))
            .unwrap();
        let err = Plan::build(&image, PlanOptions::default()).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidArgument);
    }
}
