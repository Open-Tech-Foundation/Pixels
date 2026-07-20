# ADR-0007: GPU compute deferred to v2, opt-in

Status: Accepted · 2026-07-20

## Context
GPU (wgpu compute) pays off for large batched spatial ops but loses to SIMD
CPU on small images due to transfer overhead — the thumbnail path, our most
common workload, would regress if GPU were default. v1's value is the
engine + codecs; the backend trait already isolates compute.

## Decision
v1 ships CPU-only (scalar reference + SIMD). GPU lands in v2 as a second
implementation of the backend trait, opt-in per pipeline, never a silent
default. Scope is GPU *compute* for ops — not rendering/display.

## Consequences
+ v1 scope stays shippable; no wgpu dependency at 1.0.
+ Backend trait designed against two targets from day one, preventing
  CPU-only assumptions from leaking into op contracts.
- Kernels get written twice eventually (WGSL + Rust); fusion design (v2)
  must account for both.
