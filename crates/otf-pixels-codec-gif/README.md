<div align="center">

# otf-pixels-codec-gif

***GIF for Pixels, written from scratch***

[crates.io](https://crates.io/crates/otf-pixels-codec-gif) | [Docs](https://docs.rs/otf-pixels-codec-gif) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Decodes every GIF frame, interlace layout and disposal method, and encodes a
> single frame with a quantized palette.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it picks this codec from the file's bytes with its `gif` feature.

## Use it

Transcode row by row, without holding the image:

```rust
use otf_pixels_codec_gif::{GifDecoder, GifEncoder};
use otf_pixels_core::{Decoder, Encoder, Limits};

let mut decoder = GifDecoder::new(std::fs::File::open("in.gif")?, Limits::default())?;
let descriptor = decoder.descriptor();
let mut encoder = GifEncoder::new().with_colours(64)?;
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
| Decode | All frames, interlace, transparency, disposal; the engine sees the first frame composited |
| Encode | One frame, palette quantization to up to 256 colours |
| Checked against | libgif, both directions |

## API

| Item | What it does |
| --- | --- |
| `GifDecoder`, `Frame` | Decoder and its frames |
| `GifEncoder` | Encoder; `new`, `with_colours`, `from_options` |
| `quantize`, `build_palette`, `Palette` | Colour quantization, for direct use |
| `GifCodec`, `probe` | Signature detection |

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
