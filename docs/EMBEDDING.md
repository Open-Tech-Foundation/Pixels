# Embedding otf-pixels

How to put `otf-pixels` behind a host API: a JavaScript runtime's image
module, a server's image endpoint, a CLI. Everything here is the public API
of the `otf-pixels` crate. The `otf-pixels-*` crates under it are its
implementation, and the facade re-exports what an embedder needs.

## Threading

The API is synchronous (ADR-0005): `Output::bytes` and `Output::write` block
until the image is encoded. Call them off the host's event loop, on whatever
blocking pool the host already has. `Image`, `Output`, `OpenOptions`,
`Scheduler` and `PixelsError` are `Send + Sync`, so a pipeline built on one
thread runs on another.

**Concurrency needs no setup.** Every output runs on `Scheduler::global()`, a
process-wide pool with one worker per core and one tile cache, built on first
use and shared by every pipeline in the process. Forty requests at once queue
their tiles on the same workers; they do not bring forty pools. So the whole
integration is:

```rust
// per request, from any thread of the host's blocking pool:
let bytes = image.resize(400, 300).output(format, options).bytes()?;
```

Do **not**, in a host that runs outputs concurrently:

- build a `Scheduler` per request: each one spawns a worker per core;
- call `Output::threads(n)` or `Output::scheduler_options(...)`: both give
  that one run a private pool, spawned and joined each time. They exist for
  tests, benchmarks and one-off tools.

Either way, concurrent requests end up with a pool each, competing for the
same cores.

To give image work a fixed share of the machine instead of every core, build
**one** scheduler at startup and pass that same one to every output:

```rust
use std::sync::Arc;
use otf_pixels::{Scheduler, SchedulerOptions};

// once, at startup: image work uses at most 4 threads
let scheduler = Arc::new(Scheduler::new(SchedulerOptions::default().with_threads(4))?);
// per request:
let bytes = pipeline.output(format, options).with_scheduler(Arc::clone(&scheduler)).bytes()?;
```

The global pool's workers sleep while there is no work and live as long as
the process. Its tile cache holds at most 64 MB, shared across all
requests.

## Input

```rust
use otf_pixels::{Image, Limits, OpenOptions};

let options = OpenOptions::default()
    .with_limits(Limits::default().with_max_pixels(40_000_000)); // per request
let image = Image::from_stream_with(std::io::Cursor::new(buffer), options)?; // a JS Uint8Array's bytes
let image = Image::open_with("photo.jpg", options)?;                         // or a path
```

The format comes from the bytes, never from a name or MIME type. Opening
reads headers only: `image.metadata()` (width, height, format, pixel format)
decodes nothing. What opening does by default, each switchable on
`OpenOptions`:

| Option | Default | Effect |
|---|---|---|
| `auto_orient` | on | EXIF / HEIF orientation applied, so `metadata()` reports upright dimensions |
| `to_srgb` | on | an ICC profile is converted to sRGB (matrix/TRC profiles) and dropped |
| `animated` | off | an animation's first frame is the image; on, every frame is processed; see below |
| `first_frame`, `frame_count` | 0, all | which frames of an animation (sharp's `page`, `pages`); `with_frames(first, count)` |
| `page` | 0 | which page of a multi-page TIFF to open; `Image::pages()` counts them |
| `limits` | 268 MP | images over `max_pixels` fail at the header with `LimitExceeded` |

Input limits matter for untrusted uploads. A 50000×50000 PNG is 100 bytes
of header that would ask for 10 GB of pixels, and the limit stops it before
any allocation.

## What to expose, mapped from sharp

| sharp | otf-pixels |
|---|---|
| `metadata()` | `Image::source_metadata()`, read from the headers alone: `format`, `size` (opened from a path), `width`/`height` as stored (`oriented_width()`/`oriented_height()` for sharp's `autoOrient`), `pixel`, `channels`, `has_alpha`, `bits_per_sample`, `orientation`, `pages`, `animation` (sharp's `loop` and `delay`), `density` (`x_dpi()`; `unit` is `resolutionUnit`), `progressive` (JPEG, or interlaced PNG/GIF), `palette`, `chroma_subsampling`, `background`, `compression`, `comments` (PNG text chunks, JPEG and GIF comments), `has_profile()`, and the `icc`, `exif` and `xmp` bytes. `Image::metadata()` is the pipeline's current shape |
| `stats()` | `Image::stats()`, a terminal: per channel `min`, `max`, `sum`, `squares_sum`, `mean`, `stdev` (sample), `min_x`/`min_y`/`max_x`/`max_y`; `is_opaque`, `entropy`, `sharpness` and `dominant`, computed as libvips computes them |
| `sharp(input, { page })` | `OpenOptions::with_page(page)`; a page past the end is `InvalidArgument` |
| `resize(w, h, { fit, position, background, withoutEnlargement, withoutReduction, kernel })` | `resize_with(w, h, ResizeOptions::default().with_fit(..).with_position(..).with_background(..).without_enlargement(..).without_reduction(..).with_filter(..))`, fits `Fill`/`Inside`/`Outside`/`Cover`/`Contain`; `position` is one of sharp's nine gravities (`Position::North` is sharp's `top`/`north`); the `entropy`/`attention` strategies are not implemented |
| `extract` | `crop` |
| `extend({ top, right, bottom, left, extendWith, background })` | `extend(Extend::new(top, right, bottom, left).extend_with(ExtendWith::..).with_background(rgba))`; `extend(n)` is `Extend::uniform(n)`; `extendWith` is `Background`, `Copy`, `Repeat` or `Mirror` |
| `trim({ background, threshold, lineArt })` | `trim(Trim::default().with_background(rgba).with_threshold(t)?.line_art(..))`; `info.trimOffsetLeft`/`Top` are `Image::trim_offset()`, negated. `trim` runs the pipeline before it when called and holds the result in memory, so put it after a reducing resize |
| `rotate()` (auto) / `rotate(angle, { background })` | `auto_orient` / `rotate(degrees)` for an integer angle, `rotate_with(degrees, RotateOptions::default().with_background(rgba).with_interpolator(..))` for any angle. Multiples of 90 are exact; other angles resample bilinearly by default onto the rotated bounding box, as libvips does, but in premultiplied alpha, so edges over a transparent background do not darken. A translucent background adds alpha, as for `extend` |
| `flip` / `flop` | `flip` / `flop` |
| `affine(matrix, { background, idx, idy, odx, ody, interpolator })` | `affine([a, b, c, d], AffineOptions::default().with_background(rgba).with_input_offset(idx, idy).with_output_offset(odx, ody).with_interpolator(..))`; a `[[a, b], [c, d]]` matrix is flattened row by row. Interpolators are `Nearest`, `Bilinear` and `Bicubic` (the default, as in sharp); sharp's `nohalo`, `lbb` and `vsqbs` are not implemented. Alpha is resampled premultiplied |
| `modulate`, `blur`, `sharpen`, `convolve` | same names; `modulate({ lightness })` is `Modulate::with_lightness`, added to luma rather than Lab L* |
| `negate({ alpha })`, `greyscale()`, `tint(rgb)`, `threshold(t, { greyscale })` | `negate(alpha)`, `greyscale()`, `tint(r, g, b)`, `threshold(t, greyscale)`. `greyscale` keeps three channels, as sharp's does; `threshold` keeps the input's channel count (sharp returns one); `tint` works in YCbCr (sharp: Lab), so colours are close, not identical |
| `gamma(g, gOut)` | `gamma(1.0 / g)` before the resize, `gamma(gOut)` after it; `gamma(e)` is libvips' `value^(1/e)` |
| `linear(a, b)`, `recomb(matrix)` | `linear(a, b)` or `linear_per_channel([..], [..])` (`b` in 8-bit units); `recomb` (3x3) or `recomb_rgba` (4x4) |
| `ensureAlpha(level)`, `removeAlpha()` | `ensure_alpha(level)`, `remove_alpha()` |
| `composite([{ input, blend, gravity, left, top, tile }, ..])` | `composite_overlays([Overlay::new(image).with_blend(Blend::..).with_gravity(Position::..).at(left, top).tiled(..), ..])`; every sharp blend mode is a `Blend` (`colour-dodge` is `ColourDodge`, `dest-over` is `DestOver`); an overlay larger than the base is clipped where sharp refuses it; `premultiplied` is not supported |
| `flatten`, `extractChannel`, `toColourspace('b-w')` | `flatten`, `extract_channel(index)` or `extract_channel_named(Channel::Red)`, `to_pixel_format(PixelFormat::Gray8)` |
| `jxl({ quality, lossless })` | `output(Format::Jxl, EncodeOptions::with_quality(q)?.with_lossless(..))`: lossy VarDCT at the distance `cjxl` gives `quality` (90 is distance 1), or lossless Modular; images with alpha are lossless for now. `distance` is `quality` by `cjxl`'s mapping; `effort` has no effect yet |
| `jpeg/png/webp/avif/gif/tiff({ quality, lossless })` | `output(Format::.., EncodeOptions::with_quality(q)?.with_lossless(..))` |
| `png({ palette, colours, dither, compressionLevel })` | `EncodeOptions::default().with_palette(true).with_colours(n)?.with_dither(d)?.with_compression_level(l)?`; `colours` or `dither` imply `palette`, as in sharp. PNG `quality` and `effort`, which steer libimagequant, have no counterpart: the palette is a median cut |
| `gif({ colours, dither })` | `with_colours(n)?`, `with_dither(d)?` |
| `webp({ effort, alphaQuality, nearLossless })` | `with_effort(e)?`, `with_alpha_quality(q)?`, `with_near_lossless(true)` (at `quality`, as in sharp); efforts above 2 give the same lossy file, since the VP8 search has nothing further to try; `smartSubsample` and `preset` have no counterpart |
| `jpeg({ progressive })` | `with_progressive(true)`: libjpeg's default scan script with per-scan optimized tables. mozjpeg's `optimiseScans` search and `trellisQuantisation` have no counterpart |
| `jpeg({ chromaSubsampling })`, `avif({ chromaSubsampling, effort })` | `with_chroma_subsampling(ChromaSubsampling::Yuv444)` for `'4:4:4'`, `Yuv420` for `'4:2:0'`; `with_effort(e)?`. sharp's AVIF default is 4:4:4 and ours 4:2:0, so pass `Yuv444` to match it. AVIF effort above 4 searches no further |
| `keepExif()`, `withExif(exif)` | `EncodeOptions::with_keep_exif(true)`; `Image::with_exif(Some(block))` |
| `keepXmp()`, `withXmp(xmp)` | `EncodeOptions::with_keep_xmp(true)`; `Image::with_xmp(Some(packet))` |
| `keepMetadata()` | `EncodeOptions::with_keep_metadata(true)`: EXIF, XMP and the source's density (the ICC profile is always kept) |
| `withMetadata({ density })` | `Image::with_density(Some(Density::dpi(d)))` |
| `keepIccProfile` | open with `to_srgb` off; the profile is written to the output |

`output()` narrows pixels to what the format can hold: 16-bit or float input
is written as 8-bit JPEG, WebP, AVIF or GIF, and as 16-bit PNG or TIFF. JPEG
and GIF composite partial transparency over black. EXIF is not written unless an output asks for it with
`EncodeOptions::with_keep_exif(true)` (sharp's `keepExif`), so by default
location data is stripped from uploads. Kept, it is copied from a JPEG, PNG,
WebP or AVIF source into a JPEG, PNG, WebP or AVIF output, with its
Orientation tag reset to 1 when `auto_orient` turned the pixels upright.
`Image::exif()` reads the block and `Image::with_exif` replaces it (sharp's
`withExif`, given as a whole TIFF-structured block). JPEG XL carries it in
an `Exif` box. TIFF has no separate block: its descriptive tags (make,
model, date, resolution, orientation...) and Exif and GPS directories are
read as one, and written back into the output's own directory.

XMP works the same way, with `EncodeOptions::with_keep_xmp(true)`,
`Image::xmp()` and `Image::with_xmp`: JPEG's `APP1`, PNG's
`XML:com.adobe.xmp` `iTXt`, WebP's `XMP ` chunk, TIFF's tag 700, AVIF's
`application/rdf+xml` item and JPEG XL's `xml ` box. GIF's is neither read
nor written, as in libvips, and a JPEG packet over 64 KB (extended XMP) is
not written. Density is written to JPEG's JFIF header, PNG's `pHYs` and
TIFF's resolution tags when set with `Image::with_density`, or when the
output keeps all metadata.

## Animation

Opened plainly, an animation's first frame is the image, as sharp does by
default, and `Image::animation()` reports what the file holds (frame
count, loop count, per-frame durations). With
`OpenOptions::with_animated(true)` (sharp's `animated: true`), GIF, WebP
and JPEG XL are processed frame by frame: every op chained on is applied to
each frame (ADR-0020), `Image::is_animated()` and `frame_count()` say so,
and `with_frames(first, count)` picks frames (sharp's `page` and `pages`).
`trim` cuts the box holding every frame's content; `stats` covers every
frame.

Writing an animated pipeline to a format that holds one image (PNG, JPEG,
TIFF, AVIF, raw) is `Unsupported`, unlike sharp, which stacks the frames
silently. The host states which still it wants: `first_frame()` or
`frame(i)` for one frame, `into_strip()` for all of them stacked top to
bottom (sharp's result). Built-in encoders do not yet write animated GIF,
WebP or JPEG XL; `Output::write_with_encoder` takes an encoder of the
host's own. An animated file is held in memory, compressed, so each pass
over its frames decodes it afresh; `Limits::with_max_frames` and
`with_max_animation_pixels` bound the work.

## Errors

Every error carries an `ErrorCode`, which is the semver-stable part.
`code().as_str()` gives a string for the host's error objects:

| Code | String | Meaning | Typical host mapping |
|---|---|---|---|
| `Io` | `io` | reading the source or writing the sink failed | system error |
| `Malformed` | `malformed` | the bytes are not a valid image of their format | 4xx: bad upload |
| `Unsupported` | `unsupported` | a valid file using something not implemented, or an unknown format | 415 |
| `LimitExceeded` | `limit_exceeded` | over `OpenOptions::limits` | 413 |
| `InvalidArgument` | `invalid_argument` | a bad parameter (zero size, crop outside the image) | the caller's bug |
| `Graph` | `graph` | the pipeline cannot be evaluated as built | the caller's bug |

Chaining methods never fail: an error mid-chain is carried to the terminal
and returned from `bytes()`/`write()`. Malformed input never panics.

## Memory

Pixels stream through the pipeline in tiles, so a resize of a huge PNG, JPEG
or TIFF holds a band of the image rather than all of it. Formats with no
streamable prefix are buffered inside their codec: WebP, AVIF, GIF (the
compressed stream plus one canvas), progressive JPEG and interlaced PNG.

## Build

All codecs are default features, and nothing outside this workspace is a
dependency. Build with `default-features = false` plus the codecs you want
for a smaller binary. `jpeg-progressive` is a deprecated alias of `jpeg`. The
crates are 0.x: pin the minor version.
