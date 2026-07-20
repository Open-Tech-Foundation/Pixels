# ADR-0008: crossbeam-deque for work stealing

Status: Accepted · 2026-07-20

## Context
ADR-0001 commits to a work-stealing scheduler as part of what distinguishes
this engine. The workspace sets `unsafe_code = "forbid"` so that a hostile
image cannot reach memory-unsafe code — a lock-free Chase-Lev deque cannot be
written under that constraint, since its whole point is unsynchronised racy
reads resolved by atomics.

That leaves three options. `rayon` is designed for nested fork-join
parallelism; a tile graph is a DAG with arbitrary dependencies, so we would
write the dependency tracking anyway while fighting rayon-core's model. A
std-only `Mutex<VecDeque>` + `Condvar` pool is dependency-free and safe, but a
single global queue contends on every pop and is not work stealing — it would
fail ADR-0001's promise and likely flatten M2's scaling benchmark. Taking the
deque primitive alone leaves every interesting decision — task graph, tile
negotiation, worker parking, cache policy — in our hands.

## Decision
Depend on `crossbeam-deque` for the `Worker`/`Stealer` primitive only. The
scheduler, dependency tracking, tile negotiation, worker loop and tile cache
are ours. `unsafe_code = "forbid"` stays in force for every crate we write;
the unsafe needed for lock-free stealing lives in a widely-audited dependency
rather than in a hand-rolled copy of it.

The from-scratch ethos in ADR-0004 is explicitly about **codecs** — the domain
where owning the implementation buys region decode, safety and depth. A
work-stealing deque is a general concurrency primitive, not image processing,
and reimplementing it buys nothing this project is trying to prove.

## Consequences
+ Real work stealing, so ADR-0001's promise holds and the M2 scaling
  benchmark measures the scheduler rather than queue contention.
+ `unsafe_code = "forbid"` is preserved across the whole workspace.
+ Pure Rust, no C build dependencies; the crate is ubiquitous and audited.
- A dependency in `otf-pixels-core`, which was dependency-free at M1. Its
  tree (`crossbeam-epoch`, `crossbeam-utils`) comes along.
- If crossbeam's deque ever becomes a bottleneck we own the pool but not the
  primitive underneath it.
