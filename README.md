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

**M4 — core op set + SIMD — complete.** The full v1 op set from
[SPEC §Core ops](docs/SPEC.md) is implemented: resize with seven filters,
rotate, flip/flop, crop, modulate, convolve, composite, and the channel ops.

Performance, measured on the same machine and the same pixels
(`cargo bench -p otf-pixels --bench ops`), 4000x3000 RGB8 → 400x300, Lanczos3:

| | time | throughput | vs otf-pixels |
|---|---|---|---|
| otf-pixels | 43.3 ms | 277 MP/s | 1.00x |
| `image` 0.25 | 210.2 ms | 57 MP/s | 4.85x slower |
| `fast_image_resize` 6.0 | 29.2 ms | 412 MP/s | **1.49x faster** |

Read those honestly. About 15 ms of every row is the 36 MB input copy that all
three pay. Netting it out, our resampling kernel is roughly **2x slower than
`fast_image_resize`**, which is hand-written SIMD with runtime dispatch, and
roughly **5x faster than `image`**, which is scalar.

[ADR-0011](docs/adr/0011-autovectorization-and-fixed-point.md) chose
autovectorized safe Rust over intrinsics, keeping `unsafe_code = "forbid"`
intact in every crate, and predicted a 10–30% penalty against hand-written
SIMD. The measured penalty is larger than that. The ADR is recorded as
predicting it, the benchmark is recorded as refuting it, and neither has been
quietly adjusted to agree with the other.

**M3 — PNG — complete.** The workspace, op graph, codec traits, raw codec,
geometry ops, the demand-driven parallel tile scheduler and a from-scratch PNG
codec are in place and tested; see [ROADMAP.md](docs/ROADMAP.md) for what each
milestone adds.

PNG is ours down to the DEFLATE, per [ADR-0010](docs/adr/0010-own-deflate.md):
inflate, deflate, CRC-32, Adler-32, all five filters, Adam7, palettes, `tRNS`
and 1/2/4/8/16-bit depths. Correctness is measured against the reference
implementations rather than against ourselves — a shared misreading of a
specification round-trips perfectly and is still wrong. All 86 decodable
PngSuite files match libpng's decoding, all 14 corrupt files are rejected, and
libpng reads back every PNG we write. `Image::open` identifies formats by
magic bytes only; the file extension is ignored.

Pipelines now stream: peak memory is bounded by tiles in flight, not by image
size, and is verified against a ~100 MP synthetic source. The M1 whole-image
evaluator is retained as the correctness oracle — the scheduler is diffed
against it byte for byte across pipeline shapes, thread counts and tile sizes.

Scaling is honest rather than flattering: forward-only sources are capped by
their serial decode stage (ADR-0005), and today's ops are byte movement, so
they saturate memory bandwidth before they saturate cores. `cargo bench
--bench scaling` prints the numbers. M4's arithmetic kernels are where that
range should widen.

All v1 architecture decisions are recorded in [docs/adr/](docs/adr/).

## License

[Apache-2.0](LICENSE) — see [NOTICE](NOTICE) for details.
