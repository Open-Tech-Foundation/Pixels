# ADR-0001: libvips-style demand-driven engine

Status: Accepted · 2026-07-20

## Context
The Rust ecosystem has eager image libraries (`image`, `zune-image`) but no
streaming pipeline engine. The two proven high-performance models are OpenCV
(algorithm breadth, eager, CV-focused) and libvips (lazy demand-driven
pipeline, constant memory, powers sharp/imgproxy). Our primary workloads — server-style load→transform→encode pipelines — are
squarely libvips's domain. Bun.Image is lazy but eager-per-op once
triggered; a true tile engine is a differentiator, not catch-up.

## Decision
Build a libvips-style engine: immutable lazy op DAG, pull-based tile
evaluation, streaming sources/sinks. Modernized: typed monomorphized
kernels, work-stealing scheduler, optional GPU backend later. OpenCV-style
CV breadth is a permanent non-goal.

## Consequences
+ Constant memory, automatic parallelism, huge-image capability.
+ Clear differentiation in the Rust ecosystem.
- Scheduler complexity is the hardest part of the project and must land
  early (M2) since everything depends on its contracts.
- Some ops (e.g. full-image statistics) fit the pull model awkwardly and
  will need explicit materialization points.
