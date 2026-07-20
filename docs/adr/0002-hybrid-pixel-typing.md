# ADR-0002: Hybrid pixel typing

Status: Accepted · 2026-07-20

## Context
Fully generic (`Image<Rgba<u8>>`) gives maximal codegen but infects the
public API with type parameters, bloats binaries, and cannot express "load
whatever file arrives" ergonomically. Fully dynamic (libvips) is flexible
but risks per-pixel dispatch cost. Survey of consumer pipeline APIs
(Bun.Image, sharp, libvips) shows all converge on a single dynamic image
handle with runtime format sniffing.

## Decision
Dynamic at the graph/API boundary: one `Image` type, pixel format known at
runtime. Inside each op, dispatch once per tile into a monomorphized
generic kernel.

## Consequences
+ Bun/sharp-shaped ergonomic API; host bindings stay thin.
+ Hot loops fully specialized and SIMD-friendly; one match per 128×128
  tile is negligible.
- Per-op dispatch boilerplate (macro-generated); kernel monomorphization
  still costs compile time, controlled by limiting supported pixel formats
  per op to what's meaningful.
