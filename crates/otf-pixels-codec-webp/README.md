<div align="center">

# otf-pixels-codec-webp

***WebP for Pixels, lossy and lossless, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-webp) | [Docs](https://docs.rs/otf-pixels-codec-webp) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Decodes and encodes both WebP bitstreams, VP8 lossy and VP8L lossless, with
> alpha. Opening reads only the header; pixels decode on the first row.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `webp` feature.

## Use it

Transcode row by row, without holding the image:

```rust
use otf_pixels_codec_webp::{WebPDecoder, WebPEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};
use otf_pixels_core::EncodeOptions;

let mut decoder = WebPDecoder::new(std::fs::File::open("in.webp")?, Limits::default())?;
let descriptor = decoder.descriptor();
let mut encoder = WebPEncoder::from_options(&EncodeOptions::with_quality(80)?);
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
| Decode | Lossy, lossless, alpha; an animation's first frame |
| Encode | Lossy at a quality (default), or lossless |
| Checked against | libwebp: decodes match exactly, and libwebp reads back every file written |

## API

| Item | What it does |
| --- | --- |
| `WebPDecoder` | Decoder |
| `WebPEncoder` | Encoder; `new`, `from_options` (quality, lossless) |
| `WebPCodec`, `probe` | Signature detection |
| `vp8l` | The lossless bitstream, for direct use |

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
