<div align="center">

# otf-pixels-codec-png

***PNG for Pixels, written from scratch, DEFLATE included***

[crates.io](https://crates.io/crates/otf-pixels-codec-png) | [Docs](https://docs.rs/otf-pixels-codec-png) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Decodes every PNG colour type and bit depth and encodes 8 and 16-bit, with
> its own DEFLATE. Non-interlaced images decode one row at a time.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `png` feature.

## Use it

Transcode row by row, without holding the image:

```rust
use otf_pixels_codec_png::{PngDecoder, PngEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};

let mut decoder = PngDecoder::new(std::fs::File::open("in.png")?, Limits::default())?;
let descriptor = decoder.descriptor();
let mut encoder = PngEncoder::new();
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
| Decode | All colour types, 1–16-bit, palettes, `tRNS`, Adam7 interlace |
| Encode | 8/16-bit grey, RGB and RGBA; adaptive filters; levels via `with_level` |
| Checked against | libpng: PngSuite decodes match, and libpng reads back every file written |

## API

| Item | What it does |
| --- | --- |
| `PngDecoder` | Streaming decoder |
| `PngEncoder` | Encoder; `new`, `with_level`, `from_options` |
| `PngCodec`, `probe` | Signature detection |
| `Header`, `ColorType`, `Filter` | The format's own types |

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
