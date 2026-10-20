<div align="center">

# otf-pixels-codec-jxl

***JPEG XL for Pixels, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-jxl) | [Docs](https://docs.rs/otf-pixels-codec-jxl) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Encodes lossless JPEG XL (Modular): grey or RGB, with or without alpha, at
> 8 or 16 bits. Lossy VarDCT and decoding are in progress.

Files decode bit-exactly in libjxl's `djxl`. Use it through
[`otf-pixels`](https://crates.io/crates/otf-pixels) with
`output(Format::Jxl, ..)`, or directly as an `Encoder`.
