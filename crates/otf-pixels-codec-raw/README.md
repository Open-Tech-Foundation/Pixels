<div align="center">

# otf-pixels-codec-raw

***Raw pixels for Pixels***

[crates.io](https://crates.io/crates/otf-pixels-codec-raw) | [Docs](https://docs.rs/otf-pixels-codec-raw) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Uncompressed pixels with no header: the caller states width, height, pixel
> format and stride. A short input is an error, never a panic.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `raw` feature.

## Use it

Transcode row by row, without holding the frame:

```rust
use otf_pixels_codec_raw::{RawDecoder, RawEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};
use otf_pixels_codec_raw::RawFormat;
use otf_pixels_core::{ImageDescriptor, PixelFormat};

let layout = RawFormat::packed(ImageDescriptor::new(640, 480, PixelFormat::Rgb8)?);
let mut decoder = RawDecoder::new(layout, std::fs::File::open("frame.rgb")?)?;
let descriptor = decoder.descriptor();
let mut encoder = RawEncoder::new();
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
| Decode | Any `PixelFormat`, packed or with a row stride |
| Encode | Packed rows |

## API

| Item | What it does |
| --- | --- |
| `RawFormat` | The layout: `packed`, `with_stride`, `from_dimensions` |
| `RawDecoder`, `RawEncoder` | Decoder and encoder |
| `RawCodec` | Registry entry (raw has no magic bytes) |

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
