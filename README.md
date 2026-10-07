# Pixels

`otf-pixels` — a streaming, demand-driven image processing engine in Rust.

Pixels is a libvips-class pipeline engine built from scratch: images are lazy
operation graphs, pixels are pulled through the graph in tiles on demand, and
memory stays constant regardless of image size. It is a standalone Rust
library designed to be embedded — in runtimes, servers, and CLIs — behind a
small, synchronous, streaming API.

```rust
use otf_pixels::{EncodeOptions, Fit, Format, Image, Modulate, ResizeOptions};

let webp = Image::open("photo.jpg")?
    .resize_with(800, 600, ResizeOptions::default().with_fit(Fit::Cover))
    .modulate(Modulate::identity().with_saturation(0.0)?)
    .output(Format::WebP, EncodeOptions::with_quality(80)?)
    .bytes()?;
```

Embedding it in a runtime or server: see [docs/EMBEDDING.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/EMBEDDING.md).

## Why another image library

The fast image libraries in most ecosystems wrap a C/C++ engine — usually
libvips (sharp, pyvips) or OpenCV. Rust can call both through bindings, but
has no native engine of that kind: nothing that streams pixels through a lazy
pipeline on demand. The pure-Rust crates (`image`, `zune-image`) are eager —
they decode the whole image into memory and apply one operation at a time.

Pixels brings the libvips execution model to Rust and modernizes it: typed
kernels, memory safety and work-stealing tile scheduling. It is pure Rust, so
it builds the same way on every platform, with no C toolchain or system
libraries to install. Optional GPU compute is planned for v2
([ADR-0007](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/adr/0007-gpu-deferred-to-v2.md)).

## Design pillars

1. **Lazy op graph** — chaining builds an immutable DAG; nothing executes
   until a sink pulls.
2. **Demand-driven tiles** — the sink requests output regions; the scheduler
   walks the graph backwards and evaluates only what is needed, in parallel.
3. **Streaming I/O** — sources are readers, sinks are writers. Constant
   memory wherever the format allows; codecs buffer internally where it
   doesn't.
4. **Hybrid typing** — one dynamic `Image` type at the API; monomorphized
   SIMD kernels inside, dispatched once per tile.
5. **Own the codecs** — PNG, GIF, TIFF, baseline JPEG, WebP, AVIF and raw
   implemented from scratch, every one checked against its reference
   implementation; only progressive JPEG decode is wrapped behind the same
   trait.

## Documents

| Doc | Purpose |
|---|---|
| [ARCHITECTURE.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/ARCHITECTURE.md) | System design: layers, graph, scheduler, backends |
| [SPEC.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/SPEC.md) | API contracts, formats, guarantees, safety limits |
| [EMBEDDING.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/EMBEDDING.md) | Putting it behind a runtime or server API: threading, limits, errors, sharp mapping |
| [ROADMAP.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/docs/ROADMAP.md) | v1/v2 scope and milestone plan |
| [docs/adr/](https://github.com/Open-Tech-Foundation/Pixels/tree/main/docs/adr) | Architecture Decision Records — one per decision, append-only |
| [CHANGELOG.md](https://github.com/Open-Tech-Foundation/Pixels/blob/main/CHANGELOG.md) | Keep a Changelog format |

## License

[Apache-2.0](https://github.com/Open-Tech-Foundation/Pixels/blob/main/LICENSE) — see [NOTICE](https://github.com/Open-Tech-Foundation/Pixels/blob/main/NOTICE) for details.
