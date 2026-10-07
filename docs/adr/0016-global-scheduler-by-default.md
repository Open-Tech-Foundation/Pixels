# ADR-0016: One process-wide scheduler by default

Status: Accepted · 2026-10-07

## Context
An output that was not handed a scheduler built one for itself: a pool of one
worker per core, spawned for the run and joined when it ended. A host was
expected to build one `Scheduler` at startup and pass it to every output, and
EMBEDDING.md said so. The first host to embed Pixels (a JavaScript runtime)
did not, and ran 40 concurrent jobs as 40 pools on 12 cores. The easy path
was the wrong one, so documentation alone was not enough.

## Decision
`Scheduler::global()` is a process-wide scheduler, built on first use with
default options and never torn down. An output runs on it unless the caller
passes its own (`with_scheduler`) or asks for a private pool (`threads`,
`scheduler_options`). A run started on a pool worker thread gets a private
pool, because a run blocks its caller and would otherwise hold a worker of the
pool it waits on. Idle workers park until signalled (with a one-second
backstop) instead of polling every millisecond, which only mattered once a
pool could outlive its run.

## Consequences
+ Concurrent hosts are correct with no setup: forty requests share one set of
  workers and one tile cache.
+ No per-run thread spawn and join.
- The process keeps one idle thread per core, and up to the cache budget
  (64 MB) of tiles, after its first image, as a rayon-style global pool does.
- `threads` and `scheduler_options` still give a run a private pool, which
  now means opting *out* of sharing; a host that sets them to "tune" a busy
  server gets the old behaviour back. Their docs say so, and EMBEDDING.md
  says not to use them there.
