# ADR-0021: Cooperative cancellation through one `Cancel`

Status: Accepted · 2026-11-25

## Context
A run cannot be stopped once started. `Limits` bound what a file may claim
(`max_pixels`, `max_frames`, `max_animation_pixels`), but a legal file near
those limits can still take seconds, and on a public upload endpoint one
such file holds a worker for as long as it takes. A host cannot stop the
run from outside: a run blocks its caller or works on the shared pool's
threads, and Rust threads cannot be killed. The host's only hooks are the
`Source` and `Sink`, which the formats that buffer the whole file (GIF,
WebP, JPEG XL, AVIF) stop calling before the expensive part begins.

sharp has `.timeout({ seconds })`: it sets libvips' kill flag, which the
evaluation loop checks between regions. It cannot land inside a format
library's single whole-image call. Every codec here is our own code, so a
check can go inside those loops too.

## Decision
1. **One type, `Cancel`**, in core: a cheap, cloneable, thread-safe flag.
   `Cancel::new()` is cancelled by `cancel.cancel()`, from any thread at any
   time; `Cancel::after(duration)` is the same flag with a deadline. There
   is no second mechanism for timeouts.
2. **One place to attach it, the `Image`**: `Image::with_cancel(&cancel)`.
   Everything built from that image carries it — ops, `composite`
   overlays, `trim`, `stats`, `into_strip` and the output — so one call
   covers a whole request, including the work done while the pipeline is
   built.
3. **Checks are cooperative**: the scheduler before each tile, the
   reference evaluator before each node, the decoded source per row, and
   each codec in its long loops (per row, macroblock row, group,
   superblock row, scan or frame). `Producer::set_cancel`,
   `Decoder::set_cancel` and `Encoder::set_cancel` carry it in, with no-op
   defaults, so an outside codec that ignores it still runs and is still
   stopped between tiles.
4. **A cancelled run fails with `ErrorCode::Cancelled`** (`"cancelled"`),
   saying whether the deadline passed or `cancel()` was called. Buffers are
   dropped as the error unwinds; a worker returns to the pool.

## Consequences
+ A host cancels the way its platform does: a JS binding maps an
  `AbortSignal` (or a `timeout` option) onto one `Cancel` per call.
+ Cancellation lands within milliseconds even inside a whole-image decode
  or encode, which libvips' cannot.
- A check is an atomic load and, with a deadline, a clock read; they are
  placed per row or coarser, never per pixel.
- Codecs check through a thread-local set by `Cancel::scope` around their
  heavy entry points (no codec here spawns threads), so their inner loops
  call `Cancel::checkpoint()` without the flag in every signature.
- A few steps run whole and are checked only before and after: lossless
  WebP (VP8L) encoding, palette quantization and LZW (in
  `otf-pixels-compress`, which has no dependencies), the XYB transform and
  quantization of a lossy JPEG XL frame. Measured in release: a 14 s still
  pipeline stopped about 80 ms after its deadline, a 3 s JPEG XL decode
  5 ms after.
- `ErrorCode::Cancelled` is a new code: hosts matching codes exhaustively
  must handle it (the enum is `#[non_exhaustive]`, so they already have a
  fallback arm).
- Bytes already written to a `Sink` before the cancel stay written, as for
  any failed run (ARCHITECTURE §Failure model).
