# ADR-0003: Negotiated tile shapes

Status: Accepted · 2026-07-20

## Context
Fixed square tiles are simple and cache-friendly for spatial ops but force
buffering at codec boundaries (codecs produce rows, not squares). Full-width
strips match codecs and pointwise ops but waste work for neighborhood ops.
libvips solves this with per-op access-pattern declaration and per-segment
tile negotiation, and it is the proven reason libvips streams.

## Decision
Each op declares `Sequential` or `Spatial` access. The scheduler moves
full-width strips through sequential segments and square tiles (default
128×128) through spatial segments, inserting a rolling line-cache at
sequential→spatial seams.

## Consequences
+ True streaming end to end; codec output is consumed zero-copy-friendly.
+ Spatial ops keep square-tile cache behavior.
- Scheduler carries negotiation logic and the seam line-cache; tile size
  tuning becomes a benchmark-driven task in M2/M4.
