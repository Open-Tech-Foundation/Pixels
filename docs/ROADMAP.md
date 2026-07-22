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

### M3 — PNG (first real codec, from scratch) — **complete**
- Inflate, all filter types, interlace (buffered), palette, 8/16-bit;
  encoder with configurable zlib level.
- `probe()`/metadata path; `max_pixels` enforcement; fuzzing in CI.
- **Exit**: decodes PNG test suite (PngSuite) correctly; fuzz-clean. ✅
  All 86 decodable PngSuite files match libpng, all 14 corrupt files are
  rejected, and libpng reads back every PNG we write. Non-interlaced decode
  streams, so SPEC §Formats' claim holds rather than being aspirational.

### M4 — Core op set + SIMD — **complete**
- resize (all filters), rotate/flip, modulate, convolve, composite,
  channel ops — scalar first, then SIMD for resize + pointwise.
- Scalar/SIMD exact-equality CI gate. ✅ Restated by ADR-0011: there is no
  separate scalar path to compare against, because kernels are written in one
  vectorizable form and 8-bit arithmetic is fixed point. What CI gates instead
  is the property that gate protected — output identical run to run, across
  thread counts, across tile shapes, and against the M1 oracle.
- **Exit**: benchmark vs `image` + `fast_image_resize`; publish numbers. ✅
  Published in the README. 4.85x faster than `image`, 1.49x slower than
  `fast_image_resize` end to end; roughly 2x slower on the kernel alone once
  the shared input copy is netted out. That is worse than ADR-0011 predicted,
  and is recorded as such.

### M5 — GIF + TIFF (from scratch) — **complete**
- GIF: LZW, palettes, frame disposal; single-frame encode + quantization. ✅
- TIFF: IFD, baseline tags, none/LZW/deflate/PackBits, strip + tiled; tiled
  TIFF wired to region random-access decode (the streaming showcase). ✅
- **Exit**: giant tiled TIFF → thumbnail in constant memory, benchmarked
  against libvips. ✅ The constant-memory half is asserted in
  `tests/m5_exit_criteria.rs`; the comparison is `benches/thumbnail.rs`,
  which needs `libvips-tools` and skips cleanly without it. CI installs it.

### M6 — JPEG baseline (from scratch) + wrapped codecs
- Baseline JPEG decode/encode: Huffman, DCT, chroma subsampling; EXIF
  orientation; M/8 scaled decode fast path.
- Progressive JPEG wrapped behind the codec traits.
- **Exit**: full format table in SPEC.md is green; safety guards complete.

### M6.5 — AVIF from scratch (ADR-0013)
- Its own milestone because it is the largest single codec in the workspace by a
  wide margin: the ISOBMFF/HEIF container plus the AV1 still-picture (key-frame)
  bitstream, both owned. Reverses ADR-0004's "wrap AVIF" clause.
- Decode first (container → AV1 front end → intra reconstruction → post-filters →
  exotic tools → grid region decode), then a conformant encoder.
- **Exit**: pixel-exact against libaom on the Argon conformance streams; grid
  AVIFs decode region-by-region; libavif/dav1d read our encoder output.

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
- AV1 *inter* coding (motion, reference frames, compound prediction) — only the
  still-picture key-frame subset is owned (ADR-0013); full video AV1 stays out
  of scope.
