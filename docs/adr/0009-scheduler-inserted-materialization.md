# ADR-0009: Scheduler-inserted materialization at order-incompatible seams

Status: Accepted · 2026-07-20

## Context
ADR-0001's consequences predicted that "some ops fit the pull model awkwardly
and will need explicit materialization points". M1's `flip` made it concrete:
a vertical mirror emits output row 0 from input row `H-1`, so over a
forward-only source it cannot be served in output order.

The tempting fix — mark such ops "always materialize" — is over-broad and
wrong. `flip` is a pure region remap, and its `input_regions` already states
so exactly: output rows `y0..y1` come from input rows `H-y1..H-y0`. Over a
random-access source (tiled TIFF, a memory buffer) that streams perfectly well
in output order, and a static flag would force a full-image buffer for no
reason. The same argument applies to `rotate` in M4.

The incompatibility is therefore not a property of the op. It is a property of
the **seam** between an op's demand order and its upstream's capability:
demand that is not forward-monotonic, meeting a source that can only go
forward. Either side alone is fine.

## Decision
Materialization is **scheduler-inserted**, never op-declared.

The scheduler derives demand order from `input_regions` across successive
output tiles. Where the region sequence required of a
`DecodeCapability::Sequential` upstream is not forward-monotonic, it inserts a
materialization buffer at that seam. Against a `DecodeCapability::Regions`
upstream it inserts nothing and the pipeline streams.

This is the same family as ADR-0003's rolling line-cache at the
sequential→spatial seam: the line cache is the bounded case, materialization
the unbounded one. ADR-0003 is **not** superseded — `AccessPattern` stays
two-valued and keeps its existing meaning, which is tile *shape* (full-width
strips vs squares), not tile *order*. `flip` is `Sequential` under that
reading, because square tiles buy it nothing.

Ops that genuinely consume their whole input — full-image statistics, the
other case ADR-0001 named — express that by returning the full input region
from `input_regions`, and receive the buffer through this same mechanism. No
new op-facing API is added for either case.

Materialized buffers are accounted **separately** from the tile-cache byte
budget and bounded by `max_pixels`, because a full intermediate is by
definition not bounded by a cache budget. SPEC §Guarantees 1 is amended to
state the resulting condition honestly rather than quietly weakening:

> Constant memory where the format **and pipeline order** allow; reverse-order
> ops over sequential sources buffer one full intermediate.

## Consequences
+ `flip ∘ tiled-TIFF` and `flip ∘ memory-buffer` stay constant-memory with no
  special-casing — the good case is free rather than sacrificed to the bad one.
+ One mechanism covers `flip`, M4's `rotate`, and full-image statistics.
+ Correctness rests on `input_regions`, which every op already implements, so
  no op gains a declaration it could get wrong.
+ The memory guarantee stays true as written instead of acquiring a silent
  exception.
- The scheduler must derive and reason about demand order across tiles, which
  is more analysis than reading a static flag.
- A materialized intermediate can exceed the tile-cache budget, so peak memory
  is now a property of the pipeline, not only of the formats in it. It must be
  documented and tested per pipeline shape.
- Conservative order analysis may materialize where a cleverer scheduler would
  not. Tightening it later is an optimisation, not a contract change.
