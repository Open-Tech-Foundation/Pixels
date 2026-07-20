# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- `otf-pixels` now evaluates pipelines with the tile scheduler instead of the
  M1 whole-image evaluator. Rows still reach an encoder top to bottom.
- `DecodedSource` now streams instead of materializing the whole image on
  first pull. It advances a forward-only decoder to the requested band and
  retains just that band as a rolling window, which is what makes constant
  memory real rather than promised. A request behind the window is a reported
  error, never a silent rewind.
- `Flip` now declares `AccessPattern::Sequential`, not `Spatial`.
  `AccessPattern` describes tile *shape*; a vertical mirror reads one input
  row per output row and wants full-width strips. Its row reversal is tile
  *order*, which `input_regions` already expresses and the scheduler resolves
  at the seam (ADR-0009).
- SPEC §Guarantees 1 now states the constant-memory condition as "where the
  format *and pipeline order* allow", naming the reverse-order-over-sequential
  -source exception rather than leaving the guarantee quietly overstated.

### Added
- ADR-0008 (crossbeam-deque for work stealing) and ADR-0009
  (scheduler-inserted materialization at order-incompatible seams).
- `otf-pixels-core`: `TileCache`, a byte-budgeted LRU of graph intermediates
  keyed by `(NodeId, Region)`. Eviction bounds what the cache *retains*, never
  what a caller holds alive, so tiles need no pinning.
- `otf-pixels-core`: `ThreadPool`, a work-stealing pool over `crossbeam-deque`
  (ADR-0008). Panicking tasks are contained and reported as errors; a batch
  reports its lowest-indexed failure so errors stay deterministic.
- `otf-pixels-core`: `Plan`, the pre-execution graph analysis. Negotiates tile
  shapes per segment (ADR-0003) and marks materialization points where
  non-forward demand meets a forward-only source (ADR-0009). Pure analysis —
  it reads no pixels.
- `Producer::capability`, the upstream half of ADR-0009's seam analysis.
  `BufferSource` reports `Regions`; `DecodedSource` delegates to its decoder.
- `otf-pixels-core`: `Scheduler`, the demand-driven parallel tile evaluator,
  plus `evaluate_tiled` and `RunStats`. Output tiles are evaluated in parallel
  batches and delivered to the sink in order.
- `NodePlan::cacheable`: only nodes demanded more than once (a shared prefix,
  or one feeding a spatial op) are retained in the tile cache.
- `Output::bytes_via_reference`, running a pipeline through the M1 evaluator so
  the scheduler can be differentially tested against it.
- `Output::threads` / `Output::scheduler_options` for tuning a run.
- Builder setters on `Limits`, `PlanOptions` and `SchedulerOptions`, which are
  `#[non_exhaustive]` and were otherwise unconfigurable downstream.
- M2 exit-criterion tests and `benches/scaling.rs`, a std-only benchmark
  reporting speedup and parallel efficiency across thread counts.
- Project documentation: README, ARCHITECTURE, SPEC, ROADMAP, ADR-0001..0007.
- Cargo workspace: `otf-pixels`, `otf-pixels-core`, `otf-pixels-ops`,
  `otf-pixels-codec-raw` (ADR-0006). No external dependencies.
- `otf-pixels-core`: `PixelsError` with stable `ErrorCode`s, `PixelFormat` and
  the `Sample` dispatch seam (ADR-0002), `Region`/`ImageDescriptor`/`Limits`
  with `max_pixels` enforced before allocation, strided `Tile`/`TileMut`/
  `TileBuf` views, streaming `Source`/`Sink` traits (ADR-0005), `Decoder`/
  `Encoder`/`Codec` traits, the `Op` trait and lazy `Image` op graph, and the
  naive whole-image evaluator that M2 will be diffed against.
- `otf-pixels-codec-raw`: `RawDecoder`/`RawEncoder` with caller-supplied
  dimensions, pixel format and stride, streaming a row at a time in both
  directions. Truncated streams are malformed-input errors, never panics.
- `otf-pixels-ops`: `Crop`, `Flip` and `Flop` geometry ops, each declaring its
  demand mapping and access pattern for M2's scheduler.
- `otf-pixels`: the chainable facade — `Image::from_raw`, `from_raw_stream`,
  `crop`/`flip`/`flop`, and the `output(format, options)` terminal with
  `write(sink)` and `bytes()`. Errors raised mid-chain are captured and
  surfaced at the terminal, so pipelines read as one expression.
- M1 exit-criterion test suite: raw → crop/flip → raw round-trips, graph
  laziness (zero source bytes read before a terminal), malformed-input and
  limit handling, determinism, and concurrent evaluation.
