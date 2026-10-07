# otf-pixels-codec-raw

Raw (uncompressed) pixel codec for `otf-pixels`.

Raw has no container, header or magic bytes, so the caller supplies the
width, height, pixel format and stride. Input shorter than those dimensions
require is reported as malformed, never a panic.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `raw` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
