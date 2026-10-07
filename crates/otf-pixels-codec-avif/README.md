<div align="center">

# otf-pixels-codec-avif

***AVIF for Pixels, container and AV1 written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-avif) | [Docs](https://docs.rs/otf-pixels-codec-avif) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Decodes AVIF stills at 8, 10 and 12 bits and encodes lossy 8-bit, owning both
> the HEIF container and the AV1 bitstream.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `avif` feature.

## Use it

Transcode row by row, without holding the image:

```rust
use otf_pixels_codec_avif::{AvifDecoder, AvifEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};
use otf_pixels_core::EncodeOptions;

let mut decoder = AvifDecoder::new(std::fs::File::open("in.avif")?, Limits::default())?;
let descriptor = decoder.descriptor();
let mut encoder = AvifEncoder::from_options(&EncodeOptions::with_quality(60)?);
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
| Decode | Still images, 8/10/12-bit, alpha |
| Encode | Lossy, 8-bit |
| Not yet | Image sequences (`avis`) report `Unsupported` |
| Checked against | libaom and libavif |

## API

| Item | What it does |
| --- | --- |
| `AvifDecoder`, `AvifInfo` | Decoder and the parsed file summary |
| `AvifEncoder` | Encoder; `new`, `from_options` |
| `AvifCodec`, `probe` | Signature detection |
| `Meta`, `Item`, boxes | The HEIF container, for direct use |

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
