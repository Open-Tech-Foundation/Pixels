<div align="center">

# otf-pixels-codec-jpeg

***JPEG for Pixels, baseline written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-jpeg) | [Docs](https://docs.rs/otf-pixels-codec-jpeg) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Baseline JPEG decode and encode, streaming one MCU row at a time. Decoding at
> 1/2, 1/4 or 1/8 scale makes thumbnails of large photos cheap.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `jpeg` feature.

## Use it

Transcode row by row, without holding the image:

```rust
use otf_pixels_codec_jpeg::{JpegDecoder, JpegEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};
use otf_pixels_codec_jpeg::{Scale, Subsampling};

let file = std::fs::File::open("photo.jpg")?;
let mut decoder = JpegDecoder::with_scale(file, Limits::default(), Scale::Quarter)?;
let descriptor = decoder.descriptor();
let mut encoder = JpegEncoder::with_quality(85)?.with_subsampling(Subsampling::Both);
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
| Decode | Baseline 8-bit, standard chroma subsampling; scaled 1/2, 1/4, 1/8 |
| Progressive | Decode behind the `progressive` feature (on by default), via `jpeg-decoder` |
| Encode | Baseline, quality 1–100, chroma subsampling none / horizontal / both |
| Checked against | libjpeg, both directions |

## API

| Item | What it does |
| --- | --- |
| `JpegDecoder` | Decoder; `new`, `with_scale` |
| `JpegEncoder`, `Subsampling` | Encoder and its chroma option |
| `Scale` | Reduced-size decode |
| `JpegCodec`, `probe` | Signature detection |

## Part of Pixels

| Crate | Role |
| --- | --- |
| [`otf-pixels`](https://crates.io/crates/otf-pixels) | The chainable `Image` API most users want |
| [`otf-pixels-core`](https://crates.io/crates/otf-pixels-core) | Graph, tiles, scheduler, codec and op traits |
| [`otf-pixels-ops`](https://crates.io/crates/otf-pixels-ops) | Image operations |
| [`otf-pixels-compress`](https://crates.io/crates/otf-pixels-compress) | DEFLATE, zlib, LZW, checksums |
| `otf-pixels-codec-*` | One crate per format |

## License

Apache-2.0 — see [LICENSE](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE).
