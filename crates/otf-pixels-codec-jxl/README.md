<div align="center">

# otf-pixels-codec-jxl

***JPEG XL for Pixels, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-jxl) | [Docs](https://docs.rs/otf-pixels-codec-jxl) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Encodes JPEG XL: lossy VarDCT in XYB, or lossless Modular, from grey or
> RGB at 8 or 16 bits; images with alpha are lossless for now. Decoding is
> in progress.

Every file decodes in libjxl's `djxl`, lossless ones bit-exactly. Use it through
[`otf-pixels`](https://crates.io/crates/otf-pixels) with
`output(Format::Jxl, ..)`, or directly as an `Encoder`.
