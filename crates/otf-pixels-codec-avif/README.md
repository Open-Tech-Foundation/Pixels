# otf-pixels-codec-avif

AVIF codec for `otf-pixels`, implemented from scratch.

Both layers of AVIF are owned: the ISOBMFF/HEIF container and the AV1
bitstream. Still images are supported, decode and lossy encode; AVIF image
sequences report `Unsupported`. Output is checked against libaom and
libavif.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `avif` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
