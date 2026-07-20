# Pixels

`otf-pixels` — a streaming, demand-driven image processing engine in Rust.

Pixels is a libvips-class pipeline engine built from scratch: images are lazy
operation graphs, pixels are pulled through the graph in tiles on demand, and
memory stays constant regardless of image size. It is a standalone Rust
library designed to be embedded — in runtimes, servers, and CLIs — behind a
small, synchronous, streaming API.

```rust
let out = Image::open(source)?
    .resize(800, 600, Fit::Inside)
    .modulate(Modulate { saturation: 0.0, ..Default::default() })
    .output(Format::WebP, EncodeOptions { quality: 80, ..Default::default() })
    .write(sink)?;
```

## Why another image library

Every high-performance image library in every ecosystem is OpenCV or libvips
underneath. Rust has neither: no streaming pipeline engine, no
demand-driven evaluation. Existing crates (`image`, `zune-image`) are eager —
whole image in memory, op by op. Pixels brings the libvips execution model to
Rust, modernized: typed kernels, memory safety, work-stealing tile
scheduling, and (v2) optional GPU compute — pure Rust on every platform, no
OS-backend gaps.

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
5. **Own the codecs** — PNG, GIF, TIFF, baseline JPEG, and raw implemented
   from scratch; WebP and AVIF wrapped behind the same trait, swappable
   later.

## Documents

| Doc | Purpose |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design: layers, graph, scheduler, backends |
| [SPEC.md](docs/SPEC.md) | API contracts, formats, guarantees, safety limits |
| [ROADMAP.md](docs/ROADMAP.md) | v1/v2 scope and milestone plan |
| [docs/adr/](docs/adr/) | Architecture Decision Records — one per decision, append-only |
| [CHANGELOG.md](CHANGELOG.md) | Keep a Changelog format |

## Status

**M1 — core skeleton — complete.** The workspace, op graph, codec traits,
raw codec, geometry ops and the naive whole-image evaluator are in place and
tested; see [ROADMAP.md](docs/ROADMAP.md) for what each milestone adds.

The evaluator is deliberately naive — it exists as the correctness oracle M2's
tile scheduler is diffed against. Until M2 lands, pipelines are evaluated
whole-image, so the constant-memory guarantee is **not yet met**: what exists
today is the API and the codec contracts that will meet it.

All v1 architecture decisions are recorded in [docs/adr/](docs/adr/).

## License

[Apache-2.0](LICENSE) — see [NOTICE](NOTICE) for details.
