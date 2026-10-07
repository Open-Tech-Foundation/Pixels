# otf-pixels-core

Core engine for `otf-pixels`: the op graph, tiles, codec traits and the evaluator.

An `Image` is a handle onto a node of an immutable, lazy operation graph.
Chaining ops adds nodes; nothing runs until a terminal pulls output regions,
and the evaluator then walks the graph backwards and computes only the tiles
that demand needs.

This crate defines that model and the contracts the rest of Pixels builds on:

- image descriptors, pixel formats and tiles
- the `Op` trait for operations
- the `Decoder` and `Encoder` traits for codecs
- the evaluator and its error type, `PixelsError`

Depend on it directly when you implement a codec or an op.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; depend on this crate directly only to write a codec or an op.

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
