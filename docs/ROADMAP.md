# Roadmap

Milestones are sequential; each ends in something runnable and tested.
Estimates deliberately omitted — scope, not dates, is the commitment.

## v1

### M1 — Core skeleton
- `otf-pixels-core`: `Image`, op graph, descriptors, `Op` trait,
  `Source`/`Sink`, error type.
- Raw pixel codec (decode + encode).
- Naive single-threaded evaluator (whole-image, no tiles) as the
  correctness baseline everything else is tested against.
- **Exit**: raw → crop/flip → raw round-trips; graph laziness proven by test.

### M2 — Tile scheduler
- Demand propagation (`input_region`), strip/square negotiation, line-cache
  seam, work-stealing pool, byte-budgeted LRU tile cache.
- **Exit**: pipelines produce byte-identical output vs M1 evaluator;
  constant-memory test on a synthetic huge raw source; scaling benchmark
  across cores.

### M3 — PNG (first real codec, from scratch)
- Inflate, all filter types, interlace (buffered), palette, 8/16-bit;
  encoder with configurable zlib level.
- `probe()`/metadata path; `max_pixels` enforcement; fuzzing in CI.
- **Exit**: decodes PNG test suite (PngSuite) correctly; fuzz-clean.

### M4 — Core op set + SIMD
- resize (all filters), rotate/flip, modulate, convolve, composite,
  channel ops — scalar first, then SIMD for resize + pointwise.
- Scalar/SIMD exact-equality CI gate.
- **Exit**: benchmark vs `image` + `fast_image_resize`; publish numbers.

### M5 — GIF + TIFF (from scratch)
- GIF: LZW, palettes, frame disposal; single-frame encode + quantization.
- TIFF: IFD, baseline tags, none/LZW/deflate, strip + tiled; tiled TIFF
  wired to region random-access decode (the streaming showcase).
- **Exit**: giant tiled TIFF → thumbnail in constant memory, benchmarked
  against libvips.

### M6 — JPEG baseline (from scratch) + wrapped codecs
- Baseline JPEG decode/encode: Huffman, DCT, chroma subsampling; EXIF
  orientation; M/8 scaled decode fast path.
- Progressive JPEG, WebP, AVIF wrapped behind the codec traits.
- **Exit**: full format table in SPEC.md is green; safety guards complete.

### M7 — Release hardening
- Embedding guide for host bindings; API docs; benchmark suite vs
  Bun.Image/sharp/libvips published in README.
- **Exit**: `otf-pixels` 1.0 on crates.io.

## v2

Ordered by expected value, each gated on its own ADR before work starts:

1. **GPU backend** — wgpu compute, opt-in per pipeline (ADR-0007).
2. **Op fusion** — merge adjacent pointwise ops into one kernel pass.
3. **ICC color management** — real color pipeline beyond sRGB-assumed.
4. **Progressive JPEG encode; WebP from scratch** (if still desired once
   v1 ships; the trait boundary makes it a drop-in).
5. **Animation pipelines** — multi-frame GIF/WebP/AVIF encode.
6. **ThumbHash placeholders** — cheap LQIP terminal.

## Explicit non-goals

- Computer vision (feature detection, tracking, DNN inference) — OpenCV's
  territory, permanently out of scope.
- Display/rendering — this is a processing engine, not a raster canvas.
- AV1 codec implementation from scratch — wrapped forever (ADR-0004).
