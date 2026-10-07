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
| `animated` | off | an animation's first frame is the image; see below |
| `limits` | 268 MP | images over `max_pixels` fail at the header with `LimitExceeded` |

Input limits matter for untrusted uploads. A 50000×50000 PNG is 100 bytes
of header that would ask for 10 GB of pixels, and the limit stops it before
any allocation.

## What to expose, mapped from sharp

| sharp | otf-pixels |
|---|---|
| `metadata()` | `Image::metadata()`, `Image::animation()`, `Image::icc_profile()` |
| `resize(w, h, { fit, background, withoutEnlargement, kernel })` | `resize_with(w, h, ResizeOptions::default().with_fit(..).with_background(..).without_enlargement(..).with_filter(..))`, fits `Fill`/`Inside`/`Outside`/`Cover`/`Contain` |
| `extract` | `crop` |
| `rotate()` (auto) / `rotate(90)` | `auto_orient` / `rotate(degrees)`, multiples of 90 (other angles are `InvalidArgument` in v1) |
| `flip` / `flop` | `flip` / `flop` |
| `modulate`, `blur`, `sharpen`, `convolve` | same names |
| `composite` | `composite`, `composite_with` |
| `flatten`, `extractChannel`, `toColourspace('b-w')` | `flatten`, `extract_channel`, `to_pixel_format(PixelFormat::Gray8)` |
| `jpeg/png/webp/avif/gif/tiff({ quality, lossless })` | `output(Format::.., EncodeOptions::with_quality(q)?.with_lossless(..))` |
| `keepIccProfile` | open with `to_srgb` off; the profile is written to the output |

`output()` narrows pixels to what the format can hold: 16-bit or float input
is written as 8-bit JPEG, WebP, AVIF or GIF, and as 16-bit PNG or TIFF. JPEG
and GIF composite partial transparency over black. EXIF and XMP are not
written to outputs, which also strips location data from uploads.

## Animation

v1 processes an animation's first frame, as sharp does by default.
`Image::animation()` reports what the file holds (frame count, loop count,
per-frame durations), so the host can decide: serve the original untouched,
refuse it, or accept the still. Expose an `animated` option in the host API
from the start and pass it to `OpenOptions::with_animated`. Until the
multi-frame pipeline exists, it fails as `Unsupported` on animated input
instead of returning a still, so code written against the option keeps its
meaning when frames arrive.

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

All codecs are default features. The dependencies outside this workspace
are `crossbeam-deque` (the scheduler's work-stealing queue, ADR-0008) and,
through `jpeg-progressive`, the wrapped progressive JPEG decoder (ADR-0004).
Build with `default-features = false` plus the codecs you want to leave the
latter out; progressive JPEGs then report `Unsupported`. The crates are 0.x:
pin the minor version.
