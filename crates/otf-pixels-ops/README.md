# otf-pixels-ops

Operation kernels for `otf-pixels`.

Each op implements `otf_pixels_core::Op` and is chained onto an `Image` to
build graph structure. Ops do no work when chained: they declare their output
shape and the input region they need, and run only when a terminal pulls
pixels through them, tile by tile.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it re-exports these ops through its `Image` methods.

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
