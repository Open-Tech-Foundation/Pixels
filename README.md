<div align="center">

# Pixels

***Streaming image processing for Rust: lazy, tiled, pure Rust***

[crates.io](https://crates.io/crates/otf-pixels) | [Docs](https://docs.rs/otf-pixels) | [Embedding guide](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/EMBEDDING.md)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> An image is a lazy graph of operations. Pixels are pulled through it in tiles
> only when the output asks for them, so memory stays bounded by the tiles in
> flight, not by the size of the image.

```rust
use otf_pixels::{EncodeOptions, Fit, Format, Image, Modulate, ResizeOptions};

let webp = Image::open("photo.jpg")?
    .resize_with(800, 600, ResizeOptions::default().with_fit(Fit::Cover))
    .modulate(Modulate::identity().with_saturation(0.0)?)
    .output(Format::WebP, EncodeOptions::with_quality(80)?)
    .bytes()?;
```

Nothing runs until `bytes()`. Rows then stream through resize, modulate and
the WebP encoder in one pass, and a JPEG much larger than its output decodes
at reduced scale.

> [!NOTE]
> **Pre-1.0.** The API can change between minor versions; see the
> [changelog](https://github.com/Open-Tech-Foundation/Pixels/blob/main/CHANGELOG.md).

## Why Pixels

| Need | How Pixels meets it |
| --- | --- |
| 🧵 **Streaming, not eager** | Rust crates such as `image` decode the whole image, then apply one op at a time. Pixels evaluates on demand, tile by tile. |
| 🦀 **Pure Rust** | No libvips, OpenCV or system libraries; builds the same on every platform. |
| 🔒 **Safe on hostile input** | `unsafe` is forbidden in every crate; malformed files are errors, never panics; size limits are enforced before allocating. |
| 🧪 **Checked against references** | Every codec is cross-checked against libpng, libjpeg, libgif, libtiff, libwebp and libaom. |
| 🚦 **Concurrency with no setup** | Every call shares one process-wide worker pool; no per-request threads to manage. |

## Compared with other libraries

| | **Pixels** | libvips / sharp | ImageMagick | Pillow | `image` (Rust) |
| --- | --- | --- | --- | --- | --- |
| **Written in** | Rust | C | C | Python + C | Rust |
| **Native dependencies** | None | libjpeg-turbo, libwebp, libspng… | Many delegate libraries | libjpeg, zlib, libwebp… | None |
| **Evaluation** | Lazy, demand-driven tiles | Lazy, demand-driven regions | Eager | Eager | Eager |
| **Peak memory** | Tiles in flight | Regions in flight | Whole image | Whole image | Whole image |
| **Memory-safe** | Yes, no `unsafe` | No | No | No (C core) | Yes |
| **Formats** | 7 | Dozens | 200+ | 30+ | About 15 |
| **Maturity** | New (0.x) | Decades | Decades | Decades | Mature |

Pixels follows libvips' design. What it adds is that design in pure, safe Rust,
with no native libraries to build, ship or patch.

### Speed today

Time for one job (best of 10), quality 80, same input files, i7-8700K (12 threads):

| Workload | **Pixels** | sharp 0.35 (libvips 8.18) | Pillow 11 | `image` 0.25 |
| --- | --- | --- | --- | --- |
| 12 MP JPEG → 400 px WebP | 56 ms | 23 ms | 24 ms | — (no lossy WebP) |
| 12 MP JPEG → 400 px JPEG | 40 ms | 16 ms | 16 ms | 201 ms |
| 1280×800 PNG → 320 px JPEG | 20 ms | 15 ms | 27 ms | 24 ms |

The C libraries are still faster on JPEG and WebP, where they use
libjpeg-turbo's and libwebp's hand-written SIMD. For 40 concurrent jobs against
Bun, Deno and Node, see [ES-Runtime's benchmarks](https://esrun.opentechf.org/docs/benchmarks/#images).

## Used by

[**ES-Runtime**](https://esrun.opentechf.org) ([repo](https://github.com/Open-Tech-Foundation/ES-Runtime)),
the Open Tech Foundation's server JavaScript runtime, builds its image module on Pixels.

| Aspect | Detail |
| --- | --- |
| **What** | [`runtime:images`](https://esrun.opentechf.org/api/images): decode, transform and encode JPEG, PNG, WebP, AVIF, GIF and TIFF, with a `Bun.Image`-style chain |
| **Why Pixels** | Its codecs are Pixels' own, written in Rust, where Bun and sharp call C libraries |
| **Safe for uploads** | Images over 268 megapixels are refused at the header, before any memory is allocated |
| **Learn more** | [Images guide](https://esrun.opentechf.org/docs/guides/images) · [API](https://esrun.opentechf.org/api/images) · [Benchmarks](https://esrun.opentechf.org/docs/benchmarks/#images) |

## Install

```sh
cargo add otf-pixels
```

Requires Rust 1.85+. Every codec is on by default; for a smaller build, use
`default-features = false` and enable only the formats you need.

| Feature | Format |
| --- | --- |
| `png`, `gif`, `tiff`, `webp`, `avif`, `raw` | Each codec on its own |
| `jpeg` | Baseline JPEG |
| `jpeg-progressive` | Progressive JPEG decode, the build's only third-party codec (`jpeg-decoder`) |

## Formats

| Format | Decode | Encode |
| --- | --- | --- |
| **PNG** | All bit depths, palettes, transparency, interlaced | 8/16-bit, every filter |
| **JPEG** | Baseline, scaled (1/2, 1/4, 1/8); progressive via feature | Baseline, chroma subsampling |
| **WebP** | Lossy, lossless, alpha; first frame of animations | Lossy (default) or lossless |
| **AVIF** | Still images, 8/10/12-bit | Lossy, 8-bit |
| **GIF** | Every frame, interlace, disposal; first frame composited | Single frame, quantized palette |
| **TIFF** | Baseline 6.0, strips and tiles, LZW/Deflate/PackBits; any page of a multi-page file | Strips or tiles, Deflate |
| **Raw** | Caller-described pixels | Packed pixels |

The format is identified from the file's first bytes, never from its name.

## Operations

| Operation | Method |
| --- | --- |
| Resize (seven filters, fit modes) | `resize`, `resize_with`, `thumbnail` |
| Geometry | `crop`, `rotate` (multiples of 90°), `flip`, `flop`, `orient` |
| Colour | `modulate`, `to_srgb`, `to_pixel_format`, `flatten`, `extract_channel` |
| Filters | `blur`, `sharpen`, `convolve` |
| Compositing | `composite`, `composite_with` |
| Output | `output(format, options)` then `bytes()` or `write(sink)` |

Chaining never fails: an error is carried to `bytes()` or `write()` and
returned there.

## Concurrency

Call `output(...).bytes()` from any number of threads; every call shares one
pool of workers, one per core. Don't create a scheduler per request. The
[embedding guide](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/EMBEDDING.md#threading)
covers limits, errors and fixed-size pools.

## Crates

| Crate | Role |
| --- | --- |
| [`otf-pixels`](https://crates.io/crates/otf-pixels) | **Start here.** The chainable `Image` API over everything below |
| [`otf-pixels-core`](https://crates.io/crates/otf-pixels-core) | Op graph, tiles, scheduler, and the codec and op traits |
| [`otf-pixels-ops`](https://crates.io/crates/otf-pixels-ops) | Resize, rotate, colour, filters, compositing |
| [`otf-pixels-compress`](https://crates.io/crates/otf-pixels-compress) | DEFLATE, zlib, LZW and checksums |
| `otf-pixels-codec-*` | One crate per format: [`png`](https://crates.io/crates/otf-pixels-codec-png), [`jpeg`](https://crates.io/crates/otf-pixels-codec-jpeg), [`webp`](https://crates.io/crates/otf-pixels-codec-webp), [`avif`](https://crates.io/crates/otf-pixels-codec-avif), [`gif`](https://crates.io/crates/otf-pixels-codec-gif), [`tiff`](https://crates.io/crates/otf-pixels-codec-tiff), [`raw`](https://crates.io/crates/otf-pixels-codec-raw) |

## Design

| Pillar | In one line |
| --- | --- |
| **Lazy op graph** | Chaining builds an immutable graph; nothing runs until an output pulls. |
| **Demand-driven tiles** | The output asks for regions, and only the tiles they need are computed, in parallel. |
| **Streaming I/O** | Sources are readers and sinks are writers; codecs that can't stream buffer internally. |
| **Hybrid typing** | One dynamic `Image` at the API; typed kernels inside, chosen once per tile. |
| **Own the codecs** | Every format written from scratch except progressive JPEG decode. |

GPU compute is planned for v2 ([ADR-0007](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/adr/0007-gpu-deferred-to-v2.md)).

## Documentation

| Doc | Purpose |
| --- | --- |
| [EMBEDDING.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/EMBEDDING.md) | Putting Pixels behind a runtime or server: threading, limits, errors |
| [ARCHITECTURE.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/ARCHITECTURE.md) | Layers, graph, scheduler, backends |
| [SPEC.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/SPEC.md) | API contracts, formats, guarantees, safety limits |
| [ROADMAP.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/ROADMAP.md) | v1 and v2 scope |
| [docs/adr/](https://github.com/Open-Tech-Foundation/Pixels/tree/main/docs/adr) | One record per architecture decision |
| [CHANGELOG.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/CHANGELOG.md) | What changed, per release |

## Develop

```sh
tsr ci        # fmt, clippy, tests, docs, feature combinations
tsr interop   # cross-check output against the reference libraries
tsr fuzz      # a short run of every fuzz target (nightly + cargo-fuzz)
tsr bench     # benchmarks, compared with libvips when it is installed
```

Tasks are defined in `tasks.toml` and run with [tsr](https://tsr.opentechf.org).

## License

Apache-2.0 — see [LICENSE](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE) and [NOTICE](https://github.com/Open-Tech-Foundation/Pixels/blob/main/NOTICE).
