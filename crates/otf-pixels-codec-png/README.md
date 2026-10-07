# otf-pixels-codec-png

PNG codec for `otf-pixels`, implemented from scratch, including DEFLATE.

Decodes and encodes PNG. The DEFLATE implementation is owned too (in
`otf-pixels-compress`), so the whole format is implemented here, not just
the container.

Every parser reads untrusted bytes and returns errors rather than panicking,
and the crate forbids `unsafe` code.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `png` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
