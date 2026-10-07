# otf-pixels-codec-webp

WebP codec for `otf-pixels`, lossy and lossless, implemented from scratch.

The RIFF container, VP8L lossless decode and encode, VP8 lossy decode and
encode, and `ALPH` alpha. Decoding matches libwebp exactly, and libwebp
checks every file this crate writes.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `webp` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
