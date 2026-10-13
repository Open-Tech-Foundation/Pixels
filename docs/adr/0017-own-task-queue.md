# ADR-0017: Our own task queue, replacing crossbeam-deque

Status: Accepted · 2026-10-13 · Supersedes ADR-0008

## Context
ADR-0008 took `crossbeam-deque` (and with it `crossbeam-epoch` and
`crossbeam-utils`) because a lock-free work-stealing deque cannot be written
under `unsafe_code = "forbid"`. Those three crates were the only dependencies
outside the workspace in the engine itself, and the project now aims to ship
none at all.

Looking at how the pool is used settled the design. Workers never submit
tasks: every task comes from `run_all` on the caller's thread, one scheduler
wave of about one tile per worker at a time. With crossbeam, workers pulled
batches from the global queue into their own deques and then stole from each
other to even out a batch that one of them had grabbed.

## Decision
One FIFO queue shared by every worker: a `VecDeque` behind a `Mutex`. A wave
is queued under one lock, then as many workers are woken as it has tasks.
Idle workers park on a condition variable and re-check the queue under the
parking lock before sleeping, so a wakeup cannot be lost and there is no
polling.

Measured on the same scheduler against crossbeam (12 MP image, i7-8700K,
median of six alternating runs): a tiled blur is 17% faster, resize level
(+2%), a strip pipeline +7%, and 40 concurrent PNG thumbnails +5%. The blur
gain comes from not letting one worker hoard a batch its peers must steal.

## Consequences
+ `otf-pixels-core` has no dependencies; the pool is 100% ours and safe.
+ No 1 ms polling for stealable work: idle workers sleep until signalled.
- A single lock is contended when many jobs run at once (the +5%). If that
  grows with core counts, shard the queue; per-worker deques are the known
  alternative.
- ADR-0001's "work stealing" is now "a shared queue": the scheduler still
  evaluates tiles in parallel and on demand; only the balancing changed.
