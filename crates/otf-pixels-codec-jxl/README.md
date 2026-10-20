<div align="center">

# otf-pixels-codec-jxl

***JPEG XL for Pixels, lossy and lossless, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-jxl) | [Docs](https://docs.rs/otf-pixels-codec-jxl) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Decodes JPEG XL with every coding tool, bare or in its container, and
> encodes lossy VarDCT in XYB or lossless Modular. Opening reads only the
> headers; pixels decode on the first row.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `jxl` feature.

## Use it

Transcode row by row:

```rust
use otf_pixels_codec_jxl::{JxlDecoder, JxlEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};
use otf_pixels_core::EncodeOptions;

let mut decoder = JxlDecoder::new(std::fs::File::open("in.jxl")?, Limits::default())?;
let descriptor = decoder.descriptor();
let mut encoder = JxlEncoder::from_options(&EncodeOptions::with_quality(90)?);
let mut out = Vec::new();
encoder.write_header(&descriptor, &mut out)?;
let mut row = vec![0; descriptor.row_bytes()];
for _ in 0..descriptor.height {
    decoder.read_row(&mut row)?;
    encoder.write_row(&row, &mut out)?;
}
encoder.finish(&mut out)?;
```

## Capabilities

| Area | Support |
| --- | --- |
| Decode | Every coding tool: Modular (all predictors, MA trees, RCT, palette, squeeze), VarDCT (all 27 block types, chroma from luma, progressive passes, LF frames), ANS and prefix codes with LZ77, patches, splines, noise, Gaborish, the edge-preserving filter, upsampling, extra channels, blending; JPEG recompression with chroma subsampling; the container (`jxlc`, `jxlp`, `Exif`) and ICC profiles |
| Decode output | Gray, GrayA, RGB or RGBA at 8 or 16 bits (float samples as F32); an animation's first displayed frame, with its frames and timing reported; orientation, EXIF and ICC reported |
| Encode | Lossy VarDCT in XYB (the default; `quality` maps to distance as `cjxl` maps it, 90 is distance 1) or lossless Modular; grey or RGB at 8 or 16 bits; a bare codestream |
| Not yet | Lossy alpha (images with alpha are written lossless); encoder ANS, learned trees, larger transforms, adaptive quantization and filters; animation encode; region decode |
| Differs from `djxl` | XYB images come out as sRGB and report no ICC profile, where `djxl` converts to the embedded profile; premultiplied alpha is divided out, where `djxl` writes the premultiplied values; orientation is reported, not applied |
| Checked against | libjxl: 24 files `cjxl` writes across its tools decode to `djxl`'s pixels, lossless exactly and lossy within one level (`tests/reference.rs`); every file written decodes in `djxl`, lossless bit-exactly (`scripts/check-jxl-interop.sh`) |
| Size | A 960x600 photo, lossless: 266 KB (PNG 327 KB, `cjxl -e3` 245 KB, `-e7` 218 KB); at distance 1: 40 KB at 42.4 dB PSNR (`cjxl -e7` 38.6 KB at 42.7 dB) |
| Speed | Decode, one core, 1920x1200: lossless 0.69 s (`djxl` 0.49 s), distance 1 0.32 s (`djxl` 0.06 s) |
| Memory | Both ways buffered: the encoder fits its codes to the whole image; the decoder holds the file and decodes it whole on the first row |

## API

| Item | What it does |
| --- | --- |
| `JxlDecoder` | Decoder |
| `JxlEncoder` | Encoder; `new` (lossy, distance 1), `lossless`, `with_distance`, `from_options` (quality, lossless) |
| `JxlCodec`, `probe` | Signature detection, bare codestream or container |

## Part of Pixels

| Crate | Role |
| --- | --- |
| [`otf-pixels`](https://crates.io/crates/otf-pixels) | The chainable `Image` API most users want |
| [`otf-pixels-core`](https://crates.io/crates/otf-pixels-core) | Graph, tiles, scheduler, codec and op traits |
| [`otf-pixels-ops`](https://crates.io/crates/otf-pixels-ops) | Image operations |
| [`otf-pixels-compress`](https://crates.io/crates/otf-pixels-compress) | DEFLATE, zlib, LZW, checksums |
| `otf-pixels-codec-*` | One crate per format |

## License

Apache-2.0 — see [LICENSE](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE). The decoder follows jxl-oxide's procedures (MIT); see [NOTICE](https://github.com/Open-Tech-Foundation/Pixels/blob/main/NOTICE).
