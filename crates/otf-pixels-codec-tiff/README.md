<div align="center">

# otf-pixels-codec-tiff

***TIFF for Pixels, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-tiff) | [Docs](https://docs.rs/otf-pixels-codec-tiff) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Baseline TIFF 6.0 in strips and tiles. A tiled TIFF decodes any region on its
> own, so a thumbnail of a huge scan reads only the tiles it needs.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `tiff` feature.

## Use it

Transcode row by row, without holding the image:

```rust
use otf_pixels_codec_tiff::{TiffDecoder, TiffEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};
use otf_pixels_codec_tiff::TiffLayout;

let mut decoder = TiffDecoder::new(std::fs::File::open("scan.tif")?, Limits::default())?;
let descriptor = decoder.descriptor();
let mut encoder = TiffEncoder::new().with_layout(TiffLayout::Tiles { width: 256, height: 256 })?;
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
| Decode | Both byte orders, strips and tiles, none/LZW/Deflate/PackBits, 1/8/16-bit |
| Pages | Counts every page at open; decodes the one you choose |
| Encode | Strips or tiles, uncompressed or Deflate |
| Checked against | libtiff, both directions |

## API

| Item | What it does |
| --- | --- |
| `TiffDecoder` | Decoder, with region access for tiled files; `with_page(source, limits, n)` picks a page, `pages()` counts them |
| `TiffEncoder`, `TiffLayout` | Encoder and its strip/tile layout |
| `TiffImage`, `Directory`, `tag` | The format's own structures |
| `TiffCodec`, `probe` | Signature detection |

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
