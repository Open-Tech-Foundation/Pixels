<div align="center">

# otf-pixels-ops

***The image operations of Pixels***

[crates.io](https://crates.io/crates/otf-pixels-ops) | [Docs](https://docs.rs/otf-pixels-ops) | [Pixels](https://github.com/Open-Tech-Foundation/Pixels)

</div>

<div align="right">

*An [Open Tech Foundation](https://opentechf.org/) project*

</div>

> Each op declares its output shape and the input region it needs, and computes
> one tile at a time. Nothing runs when an op is chained, only when output is pulled.

> [!TIP]
> **Building an app or a server?** Use [`otf-pixels`](https://crates.io/crates/otf-pixels) instead:
> `image.resize(400, 300)` there is `Resize` here, already wired in.

## Use it

Ops apply to an [`otf-pixels-core`](https://crates.io/crates/otf-pixels-core) `Image`:

```rust
use std::sync::Arc;
use otf_pixels_core::{Image, Result};
use otf_pixels_ops::{Fit, Modulate, Resize, ResizeOptions};

fn thumbnail(image: &Image) -> Result<Image> {
    let options = ResizeOptions::default().with_fit(Fit::Cover);
    image
        .apply(Arc::new(Resize::new(400, 300, options)?))?
        .apply(Arc::new(Modulate::identity().with_saturation(0.0)?))
}
```

## API

| Op | What it does |
| --- | --- |
| `Resize`, `ResizeOptions`, `Fit`, `Filter` | Resample with seven filters and fit modes |
| `Crop`, `Flip`, `Flop`, `Rotate`, `Quarter` | Geometry: windows, mirrors, quarter turns |
| `Modulate` | Brightness, saturation and hue |
| `Convolve`, `Kernel` | Blur, sharpen and custom kernels |
| `Composite`, `Blend` | Overlay one image on another |
| `ExtractChannel`, `Flatten` | Pick a channel; flatten alpha onto a colour |
| `ConvertFormat` | Change pixel format (bit depth, channels) |
| `ToSrgb`, `Conversion` | ICC profile to sRGB |

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
