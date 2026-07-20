# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
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
