<div align="center">

# otf-pixels-codec-jxl

***JPEG XL for Pixels, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-jxl) | [Docs](https://docs.rs/otf-pixels-codec-jxl) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Decodes JPEG XL, every coding tool, bare or in its container; encodes
> lossy VarDCT in XYB, or lossless Modular, from grey or RGB at 8 or 16
> bits (images with alpha are lossless for now).

Files libjxl's `cjxl` writes decode to `djxl`'s pixels, lossless ones
bit-exactly, and every file this crate writes decodes in `djxl`. Use it
through [`otf-pixels`](https://crates.io/crates/otf-pixels), which
recognises `.jxl` input and writes it with `output(Format::Jxl, ..)`, or
directly as a `Decoder` and `Encoder`.
