# otf-pixels-codec-tiff

TIFF codec for `otf-pixels`, implemented from scratch.

Baseline TIFF 6.0: both byte orders, strip and tile layouts,
none/LZW/Deflate/PackBits compression, and greyscale, RGB and palette images
at 1, 8 and 16 bits. Unknown tags are skipped rather than rejected.

Tiled TIFF is the format where streaming pays off most: each tile is
compressed independently, so any region can be decoded without reading the
rest of the file.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `tiff` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
