# otf-pixels-codec-gif

GIF codec for `otf-pixels`, implemented from scratch.

Decode covers the whole format: all frames, both interlace layouts,
transparency and every disposal method. The engine sees the first frame,
composited onto the canvas. Encode is single-frame, with palette
quantization.

## Part of Pixels

This crate is one piece of [Pixels](https://github.com/Open-Tech-Foundation/Pixels), a streaming, demand-driven image
processing engine in Rust. Most users want the [`otf-pixels`](https://crates.io/crates/otf-pixels)
facade, which chains operations, picks codecs by format and streams the
result; it enables this codec with its `gif` feature (on by default).

Licensed under [Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
