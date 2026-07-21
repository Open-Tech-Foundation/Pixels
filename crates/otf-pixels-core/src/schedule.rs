//! The pull-based tile scheduler.
//!
//! This is the demand-driven evaluator ADR-0001 commits to, and the thing M1's
//! whole-image evaluator exists to be checked against: for any pipeline the two
//! must agree byte for byte (ROADMAP M2).
//!
//! # How a run proceeds
//!
//! 1. [`Plan`] analyses the graph once: tile shapes (ADR-0003) and
//!    materialization points (ADR-0009). No pixels move.
//! 2. Nodes the plan marked are materialized, in forward order.
//! 3. Output tiles are processed in **batches**. Within a batch, tiles are
//!    evaluated in parallel on the work-stealing pool; batches themselves run
//!    in order, and results are handed to the sink in order so an encoder still
//!    sees rows top to bottom.
//!
//! # Where the parallelism comes from
//!
//! A linear pipeline has no *intra*-tile parallelism — every node depends on
//! the one below it. Parallelism therefore comes from having several output
//! tiles in flight, which is also what bounds memory: peak usage is a batch's
//! working set, not the image.
//!
//! # Why forward-only sources stay correct
//!
//! A [`DecodeCapability::Sequential`] producer cannot serve concurrent
//! out-of-order requests. Source bands for a batch are therefore pulled
//! **serially and in order** before the batch's parallel phase begins, and held
//! for its duration. The plan has already guaranteed that order is forward.
//!
//! # Why the cache is only ever an optimisation
//!
//! Each output tile evaluates into its own working set. The shared
//! [`TileCache`] is consulted and populated, but never depended on: an eviction
//! between producing a tile and consuming it can cost recomputation, never
//! correctness. That is what lets the cache be byte-budgeted without the
//! scheduler having to pin anything.
//!
//! [`DecodeCapability::Sequential`]: crate::DecodeCapability::Sequential

use crate::{
    Image, Node, NodeId, PixelsError, Plan, PlanOptions, Region, Result, ThreadPool, Tile, TileBuf,
    TileCache, TileKey,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Tuning for a [`Scheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SchedulerOptions {
    /// Worker threads. Zero means [`ThreadPool::default_threads`].
    pub threads: usize,
    /// Byte budget for the shared tile cache.
    pub cache_budget: usize,
    /// Output tiles evaluated concurrently.
    ///
    /// This is the memory/parallelism dial: peak usage scales with it, and so
    /// does available concurrency. Zero means "one per worker thread".
    pub batch_tiles: usize,
    /// Tile shape negotiation options.
    pub plan: PlanOptions,
}

impl SchedulerOptions {
    /// Set the worker thread count; zero means one per core.
    ///
    /// `SchedulerOptions` is `#[non_exhaustive]`, so downstream crates cannot
    /// use a struct literal; these setters are the only way to configure it.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Set the tile cache byte budget.
    #[must_use]
    pub const fn with_cache_budget(mut self, bytes: usize) -> Self {
        self.cache_budget = bytes;
        self
    }

    /// Set how many output tiles are evaluated concurrently.
    #[must_use]
    pub const fn with_batch_tiles(mut self, tiles: usize) -> Self {
        self.batch_tiles = tiles;
        self
    }

    /// Set the tile shape negotiation options.
    #[must_use]
    pub const fn with_plan(mut self, plan: PlanOptions) -> Self {
        self.plan = plan;
        self
    }
}

impl Default for SchedulerOptions {
    fn default() -> Self {
        Self {
            threads: 0,
            cache_budget: TileCache::DEFAULT_BUDGET,
            batch_tiles: 0,
            plan: PlanOptions::default(),
        }
    }
}

/// Counters describing one run, for tests and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct RunStats {
    /// Output tiles produced.
    pub output_tiles: u64,
    /// Nodes materialized because of ADR-0009.
    pub materialized_nodes: u64,
    /// Peak bytes held in working sets and materialized buffers.
    ///
    /// This is the number the constant-memory guarantee is about: it must not
    /// grow with image height for a streaming pipeline.
    pub peak_bytes: u64,
    /// The source resolution shrink-on-load lowered, if it did.
    ///
    /// `None` means the source decoded at full size — either because it has
    /// one resolution, or because the pipeline was not eligible (see
    /// [`shrink_on_load`]). Reported rather than left silent so a pipeline
    /// that expected the fast path and did not get it can be diagnosed
    /// instead of merely being slow.
    ///
    /// Filled in by whoever applied the rewrite. [`Scheduler::run`] does not
    /// apply it: the rewrite has to happen *above* the choice of evaluator, or
    /// the scheduler and the reference evaluator would be handed different
    /// graphs and stop agreeing — and their agreement is the oracle the whole
    /// engine is checked against (ROADMAP M2).
    ///
    /// [`shrink_on_load`]: crate::shrink_on_load
    pub reduction: Option<crate::Reduction>,
}

/// A demand-driven, parallel tile evaluator.
///
/// One scheduler can run many pipelines; the pool and cache are reused.
#[derive(Debug)]
pub struct Scheduler {
    pool: ThreadPool,
    cache: Arc<TileCache>,
    options: SchedulerOptions,
}

impl Scheduler {
    /// Build a scheduler with `options`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if worker threads cannot be spawned.
    pub fn new(options: SchedulerOptions) -> Result<Self> {
        let threads = if options.threads == 0 {
            ThreadPool::default_threads()
        } else {
            options.threads
        };
        Ok(Self {
            pool: ThreadPool::new(threads)?,
            cache: Arc::new(TileCache::new(options.cache_budget)),
            options,
        })
    }

    /// A scheduler with default tuning.
    ///
    /// # Errors
    ///
    /// As [`Scheduler::new`].
    pub fn with_defaults() -> Result<Self> {
        Self::new(SchedulerOptions::default())
    }

    /// The worker thread count.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// The shared tile cache.
    #[must_use]
    pub fn cache(&self) -> &Arc<TileCache> {
        &self.cache
    }

    /// Evaluate `image`, handing each output tile to `consume` in order.
    ///
    /// `consume` is called on the calling thread, in tile order, so a sink may
    /// hold non-`Send` state and an encoder still sees rows top to bottom.
    ///
    /// # Errors
    ///
    /// Propagates any producer, op or consumer error. A failure anywhere fails
    /// the whole run; the sink is simply not given the remaining tiles
    /// (ARCHITECTURE §Failure model).
    pub fn run(
        &self,
        image: &Image,
        mut consume: impl FnMut(Region, &Tile<'_>) -> Result<()>,
    ) -> Result<RunStats> {
        let plan = Plan::build(image, self.options.plan)?;
        let root = Arc::clone(image.node());
        let mut stats = RunStats::default();

        // Which nodes are worth caching; see `NodePlan::cacheable`.
        let cacheable: Arc<std::collections::HashSet<NodeId>> = Arc::new(
            dependency_order(&root)
                .iter()
                .filter(|node| plan.node(node.id()).is_some_and(|p| p.cacheable))
                .map(|node| node.id())
                .collect(),
        );

        // Phase 1: realize the nodes ADR-0009 marked, in dependency order.
        let materialized = self.materialize(&root, &plan, &cacheable, &mut stats)?;

        // Phase 2: stream output tiles in batches.
        let batch_size = if self.options.batch_tiles == 0 {
            self.threads()
        } else {
            self.options.batch_tiles
        }
        .max(1);
        let tiles = plan.output_tiles().to_vec();

        for batch in tiles.chunks(batch_size) {
            // Source bands first, serially and in order: a forward-only
            // producer cannot serve concurrent out-of-order requests.
            let sources = self.pull_sources(&root, batch, &materialized)?;

            let context = Arc::new(Context {
                cache: Arc::clone(&self.cache),
                materialized: Arc::clone(&materialized),
                sources: Arc::new(sources),
                cacheable: Arc::clone(&cacheable),
            });

            // Evaluate the batch's tiles in parallel.
            let outputs: Arc<Vec<std::sync::Mutex<Option<Arc<TileBuf>>>>> =
                Arc::new(batch.iter().map(|_| std::sync::Mutex::new(None)).collect());
            let tasks: Vec<_> = batch
                .iter()
                .enumerate()
                .map(|(index, region)| {
                    let (root, context, outputs, region) = (
                        Arc::clone(&root),
                        Arc::clone(&context),
                        Arc::clone(&outputs),
                        *region,
                    );
                    move || {
                        let tile = produce(&root, region, &context)?;
                        if let Some(slot) = outputs.get(index) {
                            *slot
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(tile);
                        }
                        Ok(())
                    }
                })
                .collect();
            self.pool.run_all(tasks)?;

            // Hand results to the sink in order.
            let mut batch_bytes = 0_u64;
            for (index, region) in batch.iter().enumerate() {
                let slot = outputs
                    .get(index)
                    .ok_or_else(|| PixelsError::graph("output slot vanished"))?;
                let buffer = slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .ok_or_else(|| {
                        PixelsError::graph(format!("output tile {region} was never produced"))
                    })?;
                batch_bytes += buffer.bytes().len() as u64;
                consume(*region, &buffer.as_tile()?)?;
                stats.output_tiles += 1;
            }
            stats.peak_bytes = stats
                .peak_bytes
                .max(batch_bytes + materialized_bytes(&materialized));
        }
        Ok(stats)
    }

    /// Realize every node the plan marked, in dependency order.
    fn materialize(
        &self,
        root: &Arc<Node>,
        plan: &Plan,
        cacheable: &Arc<std::collections::HashSet<NodeId>>,
        stats: &mut RunStats,
    ) -> Result<Arc<HashMap<NodeId, Arc<TileBuf>>>> {
        let mut materialized: HashMap<NodeId, Arc<TileBuf>> = HashMap::new();
        for node in dependency_order(root) {
            let should = plan.node(node.id()).is_some_and(|p| p.materialize);
            if !should {
                continue;
            }
            // Realized in one pass, forward, so a sequential source is read in
            // the only order it can be read.
            let context = Context {
                cache: Arc::clone(&self.cache),
                materialized: Arc::new(materialized.clone()),
                sources: Arc::new(HashMap::new()),
                cacheable: Arc::clone(cacheable),
            };
            let whole = node.descriptor().region();
            let buffer = produce_uncached(&node, whole, &context)?;
            stats.materialized_nodes += 1;
            materialized.insert(node.id(), buffer);
        }
        Ok(Arc::new(materialized))
    }

    /// Pull every source band this batch needs, serially and in order.
    fn pull_sources(
        &self,
        root: &Arc<Node>,
        batch: &[Region],
        materialized: &HashMap<NodeId, Arc<TileBuf>>,
    ) -> Result<HashMap<TileKey, Arc<TileBuf>>> {
        let mut sources = HashMap::new();
        for region in batch {
            for (node, needed) in source_demand(root, *region)? {
                if materialized.contains_key(&node.id()) {
                    continue;
                }
                let key = TileKey::new(node.id(), needed);
                if sources.contains_key(&key) {
                    continue;
                }
                let Some(producer) = node.producer() else {
                    continue;
                };
                let mut buffer = TileBuf::zeroed(needed, node.descriptor().pixel)?;
                producer.produce(needed, &mut buffer.as_tile_mut()?)?;
                sources.insert(key, Arc::new(buffer));
            }
        }
        Ok(sources)
    }
}

/// Total bytes held by materialized intermediates.
///
/// Accounted separately from the cache budget, per ADR-0009: a full
/// intermediate is by definition not bounded by a cache budget.
fn materialized_bytes(materialized: &HashMap<NodeId, Arc<TileBuf>>) -> u64 {
    materialized.values().map(|b| b.bytes().len() as u64).sum()
}

/// Everything a task needs to evaluate one output tile.
#[derive(Debug)]
struct Context {
    cache: Arc<TileCache>,
    materialized: Arc<HashMap<NodeId, Arc<TileBuf>>>,
    sources: Arc<HashMap<TileKey, Arc<TileBuf>>>,
    /// Nodes whose tiles are worth retaining; see [`NodePlan::cacheable`].
    ///
    /// [`NodePlan::cacheable`]: crate::NodePlan::cacheable
    cacheable: Arc<std::collections::HashSet<NodeId>>,
}

/// Produce `region` of `node`, consulting the cache where that can pay off.
///
/// Nodes with a single consumer bypass the cache entirely: their tiles are
/// consumed exactly once, so retaining them would evict tiles that *are*
/// reused and would make a streaming pipeline's peak memory equal to the cache
/// budget rather than a few tiles in flight.
fn produce(node: &Arc<Node>, region: Region, context: &Context) -> Result<Arc<TileBuf>> {
    if !context.cacheable.contains(&node.id()) {
        return produce_uncached(node, region, context);
    }
    let key = TileKey::new(node.id(), region);
    if let Some(tile) = context.cache.get(&key) {
        return Ok(tile);
    }
    let buffer = produce_uncached(node, region, context)?;
    Ok(context.cache.insert(key, buffer))
}

/// Produce `region` of `node` without consulting the cache for `node` itself.
fn produce_uncached(node: &Arc<Node>, region: Region, context: &Context) -> Result<Arc<TileBuf>> {
    // A materialized ancestor short-circuits everything below it.
    if let Some(whole) = context.materialized.get(&node.id()) {
        let mut buffer = TileBuf::zeroed(region, node.descriptor().pixel)?;
        crate::copy_region(&whole.as_tile()?, &mut buffer.as_tile_mut()?, region)?;
        return Ok(Arc::new(buffer));
    }

    if let Some(producer) = node.producer() {
        // Pre-pulled by `pull_sources` when running a batch; pulled directly
        // when materializing, where the caller is already serial.
        let key = TileKey::new(node.id(), region);
        if let Some(band) = context.sources.get(&key) {
            return Ok(Arc::clone(band));
        }
        let mut buffer = TileBuf::zeroed(region, node.descriptor().pixel)?;
        producer.produce(region, &mut buffer.as_tile_mut()?)?;
        return Ok(Arc::new(buffer));
    }

    let op = node.op().ok_or_else(|| {
        PixelsError::graph(format!("node `{}` is neither op nor source", node.name()))
    })?;
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

    // Depth-first: each input tile is produced, then consumed immediately, so
    // the working set is the pipeline depth rather than the whole graph.
    let mut inputs = Vec::with_capacity(requested.len());
    for (input, needed) in node.inputs().iter().zip(&requested) {
        inputs.push(produce(input, *needed, context)?);
    }
    let mut tiles = Vec::with_capacity(inputs.len());
    for buffer in &inputs {
        tiles.push(buffer.as_tile()?);
    }

    let mut output = TileBuf::zeroed(region, node.descriptor().pixel)?;
    op.compute(&tiles, &mut output.as_tile_mut()?)?;
    Ok(Arc::new(output))
}

/// The source nodes and regions needed to produce `region` of `root`.
fn source_demand(root: &Arc<Node>, region: Region) -> Result<Vec<(Arc<Node>, Region)>> {
    let mut found = Vec::new();
    let mut stack = vec![(Arc::clone(root), region)];
    while let Some((node, region)) = stack.pop() {
        let Some(op) = node.op() else {
            found.push((node, region));
            continue;
        };
        let descriptors: Vec<_> = node.inputs().iter().map(|n| n.descriptor()).collect();
        let requested = op.input_regions(region, &descriptors)?;
        for (input, needed) in node.inputs().iter().zip(requested) {
            stack.push((Arc::clone(input), needed));
        }
    }
    // Sorting by row keeps a batch's pulls forward-ordered even when the walk
    // visits branches in an arbitrary order.
    found.sort_by_key(|(_, region)| (region.y, region.x));
    Ok(found)
}

/// Nodes in dependency order: inputs before the nodes that consume them.
fn dependency_order(root: &Arc<Node>) -> Vec<Arc<Node>> {
    enum Step {
        Visit(Arc<Node>),
        Emit(Arc<Node>),
    }
    let mut seen = std::collections::HashSet::new();
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

/// Evaluate `image` into one whole-image buffer using the tile scheduler.
///
/// Convenience for callers that want the pixels rather than a stream; the
/// streaming path is [`Scheduler::run`].
///
/// # Errors
///
/// As [`Scheduler::run`].
pub fn evaluate_tiled(image: &Image, options: SchedulerOptions) -> Result<TileBuf> {
    let scheduler = Scheduler::new(options)?;
    let descriptor = image.descriptor();
    let mut out = TileBuf::for_image(&descriptor)?;
    {
        let mut view = out.as_tile_mut()?;
        scheduler.run(image, |region, tile| {
            crate::copy_region(tile, &mut view, region)
        })?;
    }
    Ok(out)
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
    use crate::testing::{ConstantOp, RampProducer, SumOp};
    use crate::{
        AccessPattern, BufferSource, DecodedSource, Decoder, Format, ImageDescriptor, Op,
        PixelFormat, PixelsError, TileMut, evaluate,
    };

    fn ramp_image(width: u32, height: u32) -> Image {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
        Image::from_producer(Arc::new(RampProducer::new(descriptor)), Format::Raw)
    }

    /// A decoder emitting a row-numbered ramp, to drive a sequential source.
    #[derive(Debug)]
    struct RampDecoder {
        descriptor: ImageDescriptor,
        row: u32,
    }

    impl Decoder for RampDecoder {
        fn descriptor(&self) -> ImageDescriptor {
            self.descriptor
        }
        fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
            if self.row >= self.descriptor.height {
                return Err(PixelsError::invalid_argument("out", "past the end"));
            }
            for (x, cell) in out.iter_mut().enumerate() {
                *cell = (self.row as usize).wrapping_mul(7).wrapping_add(x) as u8;
            }
            self.row += 1;
            Ok(())
        }
    }

    /// A pipeline rooted in a forward-only streaming decoder.
    fn streamed_image(width: u32, height: u32) -> Image {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
        let decoder = RampDecoder { descriptor, row: 0 };
        Image::from_producer(Arc::new(DecodedSource::new(Box::new(decoder))), Format::Raw)
    }

    /// A pipeline rooted in a random-access memory buffer.
    fn buffered_image(width: u32, height: u32) -> Image {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
        let bytes: Vec<u8> = (0..descriptor.byte_len().unwrap())
            .map(|i| (i % 251) as u8)
            .collect();
        let buffer = TileBuf::from_vec(descriptor.region(), descriptor.pixel, bytes).unwrap();
        let source = BufferSource::new(descriptor, Arc::new(buffer)).unwrap();
        Image::from_producer(Arc::new(source), Format::Raw)
    }

    /// A vertical mirror, like `Flip`: correct demand mapping, reversed order.
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
        fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
            let input = inputs
                .first()
                .ok_or_else(|| PixelsError::graph("no input tile"))?;
            let (out_region, in_region) = (output.region(), input.region());
            for offset in 0..out_region.height {
                let from = input
                    .row(in_region.y + (in_region.height - 1 - offset))
                    .ok_or_else(|| PixelsError::graph("missing input row"))?;
                let into = output
                    .row_mut(out_region.y + offset)
                    .ok_or_else(|| PixelsError::graph("missing output row"))?;
                into.copy_from_slice(from);
            }
            Ok(())
        }
    }

    /// A 3-row neighbourhood op, to exercise square tiles and overlap.
    #[derive(Debug)]
    struct RowBlur;
    impl Op for RowBlur {
        fn name(&self) -> &'static str {
            "row-blur"
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
            let top = out.y.saturating_sub(1);
            let bottom = (out.bottom() + 1).min(u64::from(input.height));
            Ok(vec![Region::new(
                out.x,
                top,
                out.width,
                (bottom - u64::from(top)) as u32,
            )])
        }
        fn access_pattern(&self) -> AccessPattern {
            AccessPattern::Spatial
        }
        fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
            let input = inputs
                .first()
                .ok_or_else(|| PixelsError::graph("no input tile"))?;
            let out_region = output.region();
            let in_region = input.region();
            for y in out_region.y..out_region.y.saturating_add(out_region.height) {
                // Average this row with its available neighbours.
                let mut sums = vec![0_u32; out_region.width as usize];
                let mut count = 0_u32;
                for neighbour in [y.saturating_sub(1), y, y + 1] {
                    if neighbour < in_region.y || u64::from(neighbour) >= in_region.bottom() {
                        continue;
                    }
                    let Some(row) = input.row(neighbour) else {
                        continue;
                    };
                    let start = (out_region.x - in_region.x) as usize;
                    for (i, sum) in sums.iter_mut().enumerate() {
                        *sum += u32::from(row[start + i]);
                    }
                    count += 1;
                }
                let into = output
                    .row_mut(y)
                    .ok_or_else(|| PixelsError::graph("missing output row"))?;
                for (cell, sum) in into.iter_mut().zip(&sums) {
                    *cell = (sum / count.max(1)) as u8;
                }
            }
            Ok(())
        }
    }

    fn options(threads: usize, strip_rows: u32) -> SchedulerOptions {
        SchedulerOptions {
            threads,
            plan: PlanOptions {
                strip_rows,
                square_size: 8,
            },
            ..SchedulerOptions::default()
        }
    }

    // --- Agreement with the M1 oracle (the M2 exit criterion) ----------------

    #[test]
    fn a_bare_source_matches_the_m1_evaluator() {
        let image = ramp_image(16, 40);
        let expected = evaluate(&image).unwrap();
        let actual = evaluate_tiled(&image, options(4, 8)).unwrap();
        assert_eq!(actual.bytes(), expected.bytes());
    }

    #[test]
    fn a_chain_matches_the_m1_evaluator() {
        let image = buffered_image(23, 37)
            .apply(Arc::new(ConstantOp::new(9)))
            .unwrap()
            .apply(Arc::new(Reverse))
            .unwrap();
        let expected = evaluate(&image).unwrap();
        let actual = evaluate_tiled(&image, options(4, 8)).unwrap();
        assert_eq!(actual.bytes(), expected.bytes());
    }

    #[test]
    fn agreement_holds_across_tile_sizes_and_thread_counts() {
        // The core M2 exit criterion, swept: whatever the tiling and however
        // many workers, the answer is the M1 answer.
        let image = buffered_image(31, 53).apply(Arc::new(Reverse)).unwrap();
        let expected = evaluate(&image).unwrap();
        for threads in [1, 2, 4, 8] {
            for strip_rows in [1, 3, 8, 64, 1000] {
                let actual = evaluate_tiled(&image, options(threads, strip_rows)).unwrap();
                assert_eq!(
                    actual.bytes(),
                    expected.bytes(),
                    "threads={threads} strip_rows={strip_rows}"
                );
            }
        }
    }

    #[test]
    fn a_spatial_op_matches_the_m1_evaluator() {
        // Square tiles plus overlapping demand: the case where getting
        // input_regions or the seam wrong shows up as wrong borders.
        let image = buffered_image(20, 20).apply(Arc::new(RowBlur)).unwrap();
        let expected = evaluate(&image).unwrap();
        for threads in [1, 4] {
            let actual = evaluate_tiled(&image, options(threads, 4)).unwrap();
            assert_eq!(actual.bytes(), expected.bytes(), "threads={threads}");
        }
    }

    #[test]
    fn a_branching_graph_matches_the_m1_evaluator() {
        let base = buffered_image(16, 24);
        let left = base.apply(Arc::new(ConstantOp::new(3))).unwrap();
        let right = base.apply(Arc::new(ConstantOp::new(4))).unwrap();
        let joined = Image::combine(&[left, right], Arc::new(SumOp)).unwrap();
        let expected = evaluate(&joined).unwrap();
        let actual = evaluate_tiled(&joined, options(4, 8)).unwrap();
        assert_eq!(actual.bytes(), expected.bytes());
        assert_eq!(actual.bytes()[0], 7);
    }

    #[test]
    fn a_streaming_source_matches_the_m1_evaluator() {
        let build = || {
            streamed_image(12, 40)
                .apply(Arc::new(ConstantOp::new(2)))
                .unwrap()
        };
        let expected = evaluate(&build()).unwrap();
        let actual = evaluate_tiled(&build(), options(4, 8)).unwrap();
        assert_eq!(actual.bytes(), expected.bytes());
    }

    #[test]
    fn reversal_over_a_streaming_source_matches_the_m1_evaluator() {
        // ADR-0009's hard case end to end: the plan materializes, and the
        // result is still exactly what the oracle produces.
        let build = || streamed_image(12, 40).apply(Arc::new(Reverse)).unwrap();
        let expected = evaluate(&build()).unwrap();
        let actual = evaluate_tiled(&build(), options(4, 8)).unwrap();
        assert_eq!(actual.bytes(), expected.bytes());
    }

    // --- Ordering and streaming ---------------------------------------------

    #[test]
    fn output_tiles_reach_the_sink_in_order() {
        // Encoders write rows top to bottom, so this is not negotiable.
        let image = buffered_image(16, 50);
        let scheduler = Scheduler::new(options(8, 4)).unwrap();
        let mut seen = Vec::new();
        scheduler
            .run(&image, |region, _| {
                seen.push(region.y);
                Ok(())
            })
            .unwrap();
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        assert_eq!(seen, sorted, "tiles arrived out of order");
        assert_eq!(seen.first(), Some(&0));
    }

    #[test]
    fn a_forward_pipeline_does_not_materialize() {
        let image = streamed_image(8, 64)
            .apply(Arc::new(ConstantOp::new(1)))
            .unwrap();
        let scheduler = Scheduler::new(options(4, 8)).unwrap();
        let stats = scheduler.run(&image, |_, _| Ok(())).unwrap();
        assert_eq!(stats.materialized_nodes, 0);
        assert_eq!(stats.output_tiles, 8);
    }

    #[test]
    fn reversal_over_a_streaming_source_materializes_exactly_once() {
        let image = streamed_image(8, 64).apply(Arc::new(Reverse)).unwrap();
        let scheduler = Scheduler::new(options(4, 8)).unwrap();
        let stats = scheduler.run(&image, |_, _| Ok(())).unwrap();
        assert_eq!(stats.materialized_nodes, 1, "ADR-0009 buffer");
    }

    #[test]
    fn reversal_over_a_buffer_source_materializes_nothing() {
        // The ADR-0009 payoff, observed at runtime rather than in the plan.
        let image = buffered_image(8, 64).apply(Arc::new(Reverse)).unwrap();
        let scheduler = Scheduler::new(options(4, 8)).unwrap();
        let stats = scheduler.run(&image, |_, _| Ok(())).unwrap();
        assert_eq!(
            stats.materialized_nodes, 0,
            "a memory source needs no buffer"
        );
    }

    // --- Memory --------------------------------------------------------------

    #[test]
    fn peak_memory_does_not_grow_with_image_height() {
        // SPEC §Guarantees 1. The same pipeline over a 10x taller image must
        // not cost 10x the memory.
        let measure = |height: u32| {
            let image = streamed_image(64, height)
                .apply(Arc::new(ConstantOp::new(1)))
                .unwrap();
            let scheduler = Scheduler::new(options(2, 8)).unwrap();
            scheduler.run(&image, |_, _| Ok(())).unwrap().peak_bytes
        };
        let small = measure(64);
        let large = measure(4096);
        assert_eq!(
            small, large,
            "peak memory scaled with height: {small} vs {large}"
        );
    }

    #[test]
    fn materializing_pipelines_report_their_buffer() {
        // The honest accounting ADR-0009 requires: the buffer is visible, not
        // hidden inside a cache budget it does not respect.
        let image = streamed_image(64, 256).apply(Arc::new(Reverse)).unwrap();
        let scheduler = Scheduler::new(options(2, 8)).unwrap();
        let stats = scheduler.run(&image, |_, _| Ok(())).unwrap();
        assert!(
            stats.peak_bytes >= 64 * 256,
            "a materialized full image should be accounted: {}",
            stats.peak_bytes
        );
    }

    #[test]
    fn a_tiny_cache_budget_changes_cost_not_correctness() {
        // The cache is an optimisation; evicting everything must not alter the
        // pixels, only how often they are recomputed.
        let image = buffered_image(16, 32)
            .apply(Arc::new(ConstantOp::new(5)))
            .unwrap();
        let expected = evaluate(&image).unwrap();
        for budget in [0, 1, 64, 1024, 1 << 20] {
            let options = SchedulerOptions {
                threads: 4,
                cache_budget: budget,
                plan: PlanOptions {
                    strip_rows: 4,
                    square_size: 8,
                },
                ..SchedulerOptions::default()
            };
            let actual = evaluate_tiled(&image, options).unwrap();
            assert_eq!(actual.bytes(), expected.bytes(), "cache_budget={budget}");
        }
    }

    // --- Failure -------------------------------------------------------------

    #[test]
    fn a_failing_op_fails_the_run() {
        let image = buffered_image(16, 32)
            .apply(Arc::new(crate::testing::FailingOp))
            .unwrap();
        let err = evaluate_tiled(&image, options(4, 8)).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Malformed);
    }

    #[test]
    fn a_failing_sink_stops_the_run() {
        let image = buffered_image(16, 32);
        let scheduler = Scheduler::new(options(4, 8)).unwrap();
        let mut seen = 0;
        let err = scheduler
            .run(&image, |_, _| {
                seen += 1;
                Err(PixelsError::unsupported("sink refused"))
            })
            .unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Unsupported);
        assert_eq!(seen, 1, "the sink should not be called again after failing");
    }

    #[test]
    fn failures_are_deterministic_across_runs() {
        // SPEC §Guarantees 2 extends to errors, not just pixels.
        let image = buffered_image(16, 64)
            .apply(Arc::new(crate::testing::FailingOp))
            .unwrap();
        let first = evaluate_tiled(&image, options(8, 4))
            .unwrap_err()
            .to_string();
        for _ in 0..10 {
            assert_eq!(
                evaluate_tiled(&image, options(8, 4))
                    .unwrap_err()
                    .to_string(),
                first
            );
        }
    }

    // --- Configuration -------------------------------------------------------

    #[test]
    fn defaults_produce_a_usable_scheduler() {
        let scheduler = Scheduler::with_defaults().unwrap();
        assert!(scheduler.threads() >= 1);
        assert_eq!(scheduler.cache().budget(), TileCache::DEFAULT_BUDGET);
        let image = buffered_image(8, 8);
        let stats = scheduler.run(&image, |_, _| Ok(())).unwrap();
        assert_eq!(stats.output_tiles, 1);
    }

    #[test]
    fn one_scheduler_runs_many_pipelines() {
        let scheduler = Scheduler::new(options(4, 8)).unwrap();
        for height in [8_u32, 16, 32] {
            let image = buffered_image(8, height);
            let expected = evaluate(&image).unwrap();
            let mut out = TileBuf::for_image(&image.descriptor()).unwrap();
            {
                let mut view = out.as_tile_mut().unwrap();
                scheduler
                    .run(&image, |region, tile| {
                        crate::copy_region(tile, &mut view, region)
                    })
                    .unwrap();
            }
            assert_eq!(out.bytes(), expected.bytes(), "height={height}");
        }
    }
}
