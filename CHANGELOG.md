# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Changed
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
