# Specification

Behavioral contracts for `otf-pixels` v1. Anything not specified here is an
implementation detail and may change without notice.

## Formats

| Format | Decode | Encode | Ownership (v1) | Streaming decode |
|---|---|---|---|---|
| Raw pixels | ✅ | ✅ | own | yes |
| PNG | ✅ | ✅ | own | yes (interlaced: internal buffer) |
| GIF | ✅ | ✅ | own | yes (per frame) |
| TIFF | ✅ | ✅ | own | yes; tiled TIFF = region random access |
| JPEG (baseline) | ✅ | ✅ | own | yes |
| JPEG (progressive) | ✅ | ❌ (v2) | wrapped | internal buffer |
| WebP | ✅ | ✅ lossy and lossless | own (ADR-0014) | internal buffer |
| AVIF | ✅ | ✅ lossy, 8-bit 4:2:0 | own | internal buffer |

- Format detection is by magic bytes only; extensions and MIME are ignored.
- Raw pixel contract: caller supplies width, height, pixel format, stride.
- TIFF: baseline tag set + none/LZW/deflate compression; exotic tags are
  skipped, not errors.
- GIF v1 scope: full decode (all frames, disposal handled); encode is
  single-frame + palette quantization. Animation pipelines are v2.
- WebP v1 scope: decode covers lossy, lossless and alpha, owned and matching
  libwebp exactly (colour conversion included). Encode is **lossy by default**
  at `EncodeOptions::quality`, mapped to the quantizer as libwebp maps it,
  and lossless with `EncodeOptions::lossless`; alpha is always coded
  losslessly, even in a lossy file. Greyscale has no native WebP mode and
  returns as RGB. Animation decodes to the first frame, placed on its canvas
  as libwebp's animation decoder does.
- AVIF v1 scope: owned outright, container and AV1 bitstream both (ADR-0013).
  Still images only — an AVIF still is an AV1 **key frame**, so inter-coded
  content is out of scope. Image *sequences* (the `avis` brand) are animation
  and report `Unsupported` in v1. 8/10/12-bit and 4:2:0/4:2:2/4:4:4/monochrome
  decode into `Rgb8`/`Rgb16` (plus alpha via an auxiliary item, un-premultiplied
  when a `prem` reference marks it premultiplied), through the BT.601/709/2020
  matrices or YCgCo at either range; grid-derived
  images give region random access, the payoff owning the container buys.
  This row is the v1 target; M6.5 is in progress (ROADMAP), and until it
  lands a file using a part not yet implemented is refused as `Unsupported`,
  never decoded wrong. The codec crate's `tests/unsupported.rs` lists what is
  still refused.
  Encode writes one 8-bit 4:2:0 key frame (monochrome for grey input) at
  `EncodeOptions::quality`, BT.601 full range with sRGB `colr`, and
  transparency as an auxiliary alpha item; opaque alpha is dropped. Lossless
  AVIF and deeper or 4:4:4 encoding are `Unsupported`. libaom, dav1d and
  libgav1 must decode every emitted file to our decoder's pixels
  (`scripts/check-avif-interop.sh`).

## Pixel formats

v1 supports: `Gray8`, `Gray16`, `GrayA8`, `Rgb8`, `Rgba8`, `Rgb16`,
`Rgba16`, `RgbF32`, `RgbaF32` (f32 used internally for filter math).
Color space: pixels without a profile are sRGB. An embedded ICC profile is
read from PNG (`iCCP`), JPEG (`APP2`), WebP (`ICCP`), TIFF (tag 34675) and
AVIF (`colr`), carried through the pipeline (`Image::icc_profile`), and
written into any output format with a place for one; GIF and raw drop it.
Alpha is
unassociated (straight) at API boundaries; ops that require premultiplied
alpha convert internally.

## Core ops (v1)

| Op | Notes |
|---|---|
| `resize(w, h, opts)` | `fit: fill \| inside`; `without_enlargement`; filters: lanczos3 (default), lanczos2, mitchell, catmull-rom, box, bilinear, nearest |
| `crop(x, y, w, h)` | zero-cost region remap |
| `rotate(deg)` | multiples of 90 |
| `flip()` / `flop()` | vertical / horizontal mirror |
| `modulate(opts)` | brightness, saturation, hue |
| `convolve(kernel)` | arbitrary small kernels; blur/sharpen presets |
| `composite(other, x, y, blend)` | source-over minimum; more blend modes as capacity allows |
| `extract_channel` / `flatten(bg)` | channel ops |

JPEG fast path: when the source is JPEG and the target size is ≤ ½ source,
decode at the nearest M/8 IDCT scale — a thumbnail from a 24 MP photo never
materializes the full-resolution image.

**Status**: implemented. `Image` pipelines select the scale automatically —
`from_stream(jpeg).resize(200, 150)` decodes at 1/8 — and
`Output::write_with_stats` reports which reduction was applied, so a pipeline
that expected the fast path and did not get it is diagnosable rather than
merely slow.

The choice is made by a graph rewrite (`shrink_on_load`) once the pipeline is
complete, because the useful size is the pipeline's target and that does not
exist when the source is opened. It applies only when all three hold:

- The graph has **one source**. With two, only one would shrink.
- Every op is willing to be **rescaled** (`Op::rescaled`). `crop` and
  `composite` carry coordinates in source pixels and `convolve` carries a
  kernel in pixels — a 3x3 blur over a 1/8 decode is eight times the blur
  relative to image content — so all three decline.
- Re-deriving every descriptor from the reduced source leaves the **root
  unchanged**, which is what distinguishes `resize(200, 150)`, that pins its
  output size, from a bare `flip`, that would simply emit a smaller image.

A pipeline that does not qualify decodes at full resolution rather than
failing: cropping a JPEG is a legal thing to do, and refusing it to protect an
optimization would be the wrong trade.

## API surface (Rust)

```rust
// Construction — no decode work yet
let img = Image::open(source)?;              // Source: path, bytes, mmap, reader
let img = Image::from_raw(desc, bytes)?;     // raw pixel input

// Metadata — header parse only, no pixel decode
let meta = img.metadata()?;                  // { width, height, format, pixel }

// Chaining — builds the graph, executes nothing
let g = img.resize(800, 600, Fit::Inside).modulate(m);

// Terminals — pull the graph
g.output(Format::WebP, opts).write(sink)?;   // streaming write
g.output(Format::Png, opts).bytes()?;        // Vec<u8>
```

`Image` is `Clone` (cheap; shares graph nodes) and `Send + Sync`.

## Embedding notes

- `output(format, options)` is the single encode terminal with format as
  data — host bindings can expose it directly, and unsupported formats are
  a stable, catchable error detectable at runtime.
- Path-string sources are filesystem paths; hosts exposing them to
  untrusted input must validate (arbitrary-file-read caution) — documented
  for binding authors.

## Safety and limits

- `max_pixels` (default 268 MP, Sharp-compatible): checked at header parse,
  before pixel allocation. Exceeding it is an error, not a truncation.
- `auto_orient` (default on, `OpenOptions`): the orientation the file
  declares is applied before any op — EXIF `Orientation` in JPEG, TIFF, PNG
  (an `eXIf` chunk before `IDAT`) and WebP, `irot`/`imir` in AVIF. Decoders
  report it (`Decoder::orientation`) and never apply it themselves, so turning
  it off yields the pixels as stored; `Image::orient` applies one by hand. A
  quarter turn over a sequential source is the reverse-order exception of
  §Guarantees 1, at the shrink-on-load size when that applies.
- Malformed input never panics and never aborts the process; all codec
  parsers are fuzzed in CI.
- Decompression bombs beyond dimensions (e.g. malicious deflate streams)
  are bounded by tile-granular allocation: the engine never allocates
  output proportional to claimed-but-undelivered input.

## Guarantees

1. **Constant memory** where the format *and pipeline order* allow. For
   pipelines whose formats stream (see table), memory is bounded by tiles in
   flight; otherwise it is bounded by the buffering codec's need, never by
   downstream ops. One documented exception: reverse-order ops (`flip`,
   `rotate`) over a sequential source buffer one full intermediate, because
   demand order and a forward-only source are incompatible (ADR-0009). The
   same pipeline over a random-access source streams in constant memory.
2. **Determinism**: identical input + pipeline + version ⇒ byte-identical
   output on every platform and backend (scalar and SIMD paths must agree
   exactly; this is CI-enforced).
3. **Laziness**: no source bytes are read before `metadata()` (header only)
   or a terminal (pixels).
4. **Error stability**: error codes are part of the public API and follow
   semver.

## Versioning

Semver. The `Op` trait, backend trait, and codec traits are `pub` but
`#[non_exhaustive]` where forward compatibility demands; kernel internals
are not public API.
