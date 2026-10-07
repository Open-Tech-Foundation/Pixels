<div align="center">

# otf-pixels-core

***The engine under Pixels: graph, tiles, scheduler***

[crates.io](https://crates.io/crates/otf-pixels-core) | [Docs](https://docs.rs/otf-pixels-core) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> An image is a node in an immutable, lazy graph. The scheduler pulls output
> tiles through it on demand, on one shared pool of worker threads.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> it wraps this engine in a chainable API with every codec and op wired in.

## Use it

Depend on this crate to write a codec or an op. A codec implements `Decoder`
(rows out) and `Encoder` (rows in):

```rust
use otf_pixels_core::{Decoder, ImageDescriptor, PixelFormat, Result};

/// A decoder for a flat grey image.
#[derive(Debug)]
struct Grey(ImageDescriptor);

impl Decoder for Grey {
    fn descriptor(&self) -> ImageDescriptor {
        self.0
    }
    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        out.fill(128);
        Ok(())
    }
}

let grey = Grey(ImageDescriptor::new(640, 480, PixelFormat::Rgb8)?);
let png = otf_pixels::Image::from_decoder(Box::new(grey), otf_pixels::Format::Raw)
    .output(otf_pixels::Format::Png, Default::default())
    .bytes()?;
```

## API

| Item | Purpose |
| --- | --- |
| `Image`, `Node` | The lazy graph: `apply` an op, `combine` several inputs |
| `Decoder`, `Encoder`, `Codec` | The codec contracts: rows out, rows in, magic-byte probe |
| `Op`, `Producer`, `AccessPattern` | The op contract: output shape, input demand, per-tile compute |
| `Scheduler`, `SchedulerOptions` | Tile evaluation; `Scheduler::global()` is the shared default |
| `ImageDescriptor`, `PixelFormat`, `Region` | Shape, pixel layout and rectangles |
| `Tile`, `TileBuf`, `TileMut` | The pixel buffers ops read and write |
| `Source`, `Sink` | Streaming input and output; any `Read` / `Write` qualifies |
| `Limits`, `PixelsError`, `ErrorCode` | Input limits and the error taxonomy |
| `evaluate` | The whole-image reference evaluator the scheduler is tested against |

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
