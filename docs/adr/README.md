# Architecture Decision Records

One record per significant decision. Records are **append-only**: a decision
is never edited after acceptance — it is superseded by a new record that
links back. Status values: `Proposed`, `Accepted`, `Superseded by ADR-NNNN`.

Format per record: Context → Decision → Consequences (including what we gave
up). Deviations discovered during implementation get their own ADR rather
than silent divergence — this replaces the previous DECISIONS.md practice.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-libvips-style-demand-driven-engine.md) | libvips-style demand-driven engine | Accepted |
| [0002](0002-hybrid-pixel-typing.md) | Hybrid pixel typing | Accepted |
| [0003](0003-negotiated-tile-shapes.md) | Negotiated tile shapes | Accepted |
| [0004](0004-codec-ownership-split.md) | Codec ownership split | Accepted |
| [0005](0005-streaming-io-sync-core.md) | Streaming-only I/O, sync core | Accepted |
| [0006](0006-naming-and-crate-layout.md) | Naming and crate layout | Accepted |
| [0007](0007-gpu-deferred-to-v2.md) | GPU compute deferred to v2, opt-in | Accepted |

Deferred (no ADR yet, decide when reached): tile cache eviction policy
details, fusion pass design, ICC pipeline, error taxonomy granularity.
