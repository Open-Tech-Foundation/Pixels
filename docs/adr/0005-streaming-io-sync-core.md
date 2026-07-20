# ADR-0005: Streaming-only I/O, sync core

Status: Accepted · 2026-07-20

## Context
The engine targets embedding in async hosts (JS runtimes, servers) that
run pipelines off their event loops (Bun.Image model). Formats split into streamable (baseline
JPEG, non-interlaced PNG, GIF, strip/tiled TIFF, raw) and
non-streamable-in-practice (progressive JPEG, interlaced PNG, AVIF, WebP).
An async-aware scheduler would push async color through the entire engine
for no throughput gain, since pixel compute is CPU-bound.

## Decision
Sources are forward-only readers, sinks are writers — streaming is the only
external I/O contract. Codecs that cannot decode incrementally buffer
internally; the memory guarantee is "constant where the format allows,
bounded by codec need otherwise." The engine core is synchronous; async
hosts integrate at the boundary by running pipelines on worker threads.

## Consequences
+ Simple core; trivial embedding in a host worker pool, tokio, or
  plain threads.
+ Honest, precise memory guarantee (documented per format in SPEC).
- No seek-based tricks even where a source could seek (acceptable: region
  decode covers the important case, tiled TIFF).
- Buffering formats are only as memory-bounded as their codecs.
