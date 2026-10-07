# otf-pixels-codec-jpeg

JPEG codec for `otf-pixels`, implemented from scratch.

Baseline JPEG (8-bit, sequential DCT, Huffman coding), decode and encode,
written from scratch. Decode streams one MCU row at a time and supports
scaled (M/8) decoding.

Progressive JPEG decode is behind the `progressive` feature, on by default,
which wraps the `jpeg-decoder` crate. Turn it off for a build with no
dependencies; progressive files then report `Unsupported`.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `jpeg` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
