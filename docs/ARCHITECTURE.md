# Architecture

`otf-pixels` is a demand-driven, streaming image processing engine. This
document describes the system's layers and the contracts between them.
Decisions referenced as ADR-NNNN are recorded in [adr/](adr/).

## Overview

```
Sources (file / mmap / memory / stream)
   │
Codecs (Decoder trait — region decode where format allows)
   │
Op graph (immutable lazy DAG of Arc<dyn Op>)
   │
Tile scheduler (pull-based, negotiated tile shapes, work stealing)
   │            ├── CPU backend (SIMD, multiversioned)   [v1]
   │            └── GPU backend (wgpu compute, opt-in)   [v2]
   │
Encoders / sinks (streaming write)
```

Data flows top to bottom; *demand* flows bottom to top. The sink asks for
output tiles; each op translates that request into the input regions it
needs; requests propagate to the decoder, which produces only those pixels.

## Layer 1 — Sources

A `Source` is the byte-input abstraction: file path, memory-mapped file,
in-memory buffer, or a streaming reader. The contract is **read-forward
only** — no seek is required of the caller (ADR-0005). Sources perform no
decoding and no allocation proportional to image size.

The engine core is synchronous. Async integration (host promises,
tokio) happens strictly at the source/sink boundary: an async host hands the
engine a reader/writer and runs the pipeline on a worker thread (ADR-0005).

## Layer 2 — Codecs

`Decoder` and `Encoder` are traits; each format is a plugin behind a cargo
feature. Two capability levels a decoder may declare:

- **Sequential**: emits pixel rows in order as bytes arrive (PNG
  non-interlaced, baseline JPEG, GIF frames, strip TIFF, raw).
- **Random-access regions**: can produce an arbitrary region without full
  decode (tiled TIFF; scaled-decode JPEG).

Formats that cannot stream (progressive JPEG, interlaced PNG, AVIF) buffer
internally; the external contract stays streaming (ADR-0005). Ownership per
format — from scratch vs wrapped crate — is recorded in ADR-0004 and is
swappable without API change because everything sits behind the same traits.

Codecs also implement `probe()`: format sniffing from magic bytes and header
parsing for `metadata()` without pixel decode. Dimension limits
(`max_pixels`) are enforced at header parse, before any pixel allocation
(SPEC §Safety).

## Layer 3 — Op graph

Calling `resize`, `crop`, `modulate`, etc. does no work: it wraps the current
node in a new one, producing an immutable DAG of `Arc<dyn Op>`. Sharing is
free — two pipelines can branch from a common prefix and the shared nodes
evaluate once (tile cache, Layer 4).

Each `Op` declares:

- `output_descriptor()` — dimensions, pixel format, and color model of its
  output, computed from its inputs (metadata flows forward at graph-build
  time; this is what makes `metadata()` free).
- `input_region(out_region) -> Vec<Region>` — the inverse mapping used by
  demand propagation. A pointwise op returns the same region; a 5×5
  convolution returns the region grown by 2px; a resize returns the scaled
  region plus filter support.
- `access_pattern()` — `Sequential` or `Spatial`, used by tile negotiation
  (ADR-0003).
- `compute(inputs: &[Tile], out: &mut TileMut)` — the kernel entry point.

**Typing model (ADR-0002).** The graph is dynamic: one `Image` handle, pixel
format known at runtime, matching the Bun/sharp/libvips API shape. Inside
`compute`, the op dispatches **once per tile** on the pixel format into a
monomorphized generic kernel. One match per tile is noise; the inner loop is
fully specialized and SIMD-friendly.

## Layer 4 — Tile scheduler

The engine's heart. Evaluation is pull-based:

1. The sink requests the next output region (typically a strip of rows, in
   order, so encoders can stream).
2. The scheduler walks the DAG backwards via `input_region`, building the
   set of (node, region) pairs required.
3. Pairs with no unmet dependencies become tasks on a work-stealing thread
   pool. Completed tiles satisfy their dependents until the sink's region is
   ready.

**Tile shapes are negotiated per pipeline segment (ADR-0003).** Runs of
`Sequential` ops (decode → pointwise → encode) move full-width strips —
zero-copy friendly, matches how codecs naturally produce and consume rows.
Where a `Spatial` op sits in the chain, the scheduler switches that segment
to square tiles (default 128×128) to minimize redundant border computation,
and inserts a small rolling line-cache at the seam so a spatial op can sit
on a streaming source without full-image buffering. This negotiation is why
the engine streams where a fixed-square design would be forced to buffer.

**Tile cache.** A byte-budgeted LRU keyed by (node id, region) holds
intermediates shared by multiple consumers (graph branches, overlapping
spatial requests). Eviction policy details are internal and may change
(ADR index: deferred).

## Layer 5 — Compute backends

Every kernel has a portable scalar implementation — the reference for
correctness tests. The CPU backend adds SIMD paths (via `std::simd` or
runtime feature multiversioning); selection happens once at startup.

The v2 GPU backend implements the same backend trait over `wgpu` compute
shaders. It is opt-in per pipeline, not a default: GPU wins on large batched
spatial work and loses to SIMD on small thumbnails due to transfer overhead
(ADR-0007). The graph and scheduler contracts do not change when it lands.

## Layer 6 — Encoders and sinks

Encoders consume output strips in row order as the scheduler completes them
and write bytes to the `Sink` (writer). End-to-end memory is bounded by
tiles in flight + codec working state, not image size.

## Embedding contract

The engine is a plain synchronous Rust library with no runtime opinions.
Host environments (JS runtimes, async servers, CLIs) embed it by running
terminals on their own worker threads and integrating at the source/sink
boundary (ADR-0005). Because chaining is graph construction, any host-side
chainable/builder API maps onto the engine one-to-one; host bindings live
in the host's repository, not here.

## Crate layout

```
otf-pixels           facade: Image API, graph builders, re-exports
otf-pixels-core      graph, scheduler, tile cache, backend traits
otf-pixels-ops       op kernels (scalar + SIMD)
otf-pixels-codec-*   one crate per format (png, jpeg, gif, tiff, raw, webp, avif)
```

Host bindings (e.g. a JS runtime's image API) are separate crates in their
own repositories, built on the `otf-pixels` facade.

## Failure model

Errors are values end to end (`PixelsError`): source I/O, malformed input,
unsupported format/feature, limits exceeded. Malformed input must never
panic — codec fuzzing is part of CI from the first decoder onward. A failed
tile fails the pipeline deterministically; partial output is never silently
written.
