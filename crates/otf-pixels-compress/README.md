# otf-pixels-compress

Compression and checksum primitives for `otf-pixels` codecs.

zlib/DEFLATE, LZW and the checksums they need, written from scratch and
shared by the codecs that use them: PNG needs zlib, TIFF needs zlib and LZW,
GIF needs LZW.

The crate knows about bit streams and byte buffers, not images, and does not
depend on `otf-pixels-core`. That keeps it testable directly against
reference implementations.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; codecs depend on this crate; applications rarely need it directly.

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
