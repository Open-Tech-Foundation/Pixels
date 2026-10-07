<div align="center">

# otf-pixels-compress

***DEFLATE, zlib, LZW and checksums, in safe Rust***

[crates.io](https://crates.io/crates/otf-pixels-compress) | [Docs](https://docs.rs/otf-pixels-compress) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> The compression the Pixels codecs share: PNG and TIFF use zlib, GIF and TIFF
> use LZW. Every decoder takes an output limit, so a decompression bomb is an error.

## Use it

```rust
use otf_pixels_compress::{Crc32, Level, zlib_compress, zlib_decompress};

let data = b"hello hello hello hello";
let packed = zlib_compress(data, Level::new(6)?)?;
let unpacked = zlib_decompress(&packed, 1 << 20)?; // refuses more than 1 MiB
assert_eq!(unpacked, data);
let checksum = Crc32::of(data);
```

## API

| Item | What it does |
| --- | --- |
| `zlib_compress`, `deflate`, `Level` | Compress, zlib-wrapped or raw, levels 0–9 |
| `zlib_decompress`, `inflate_to` | Decompress a whole buffer, up to a byte limit |
| `ZlibStream`, `Inflater` | Decompress incrementally, as input arrives |
| `LzwEncoder`, `LzwDecoder`, `BitOrder` | LZW in GIF (LSB-first) or TIFF (MSB-first) order |
| `Crc32`, `Adler32` | The checksums PNG and zlib use |

Output is checked against the reference zlib, both ways.

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
