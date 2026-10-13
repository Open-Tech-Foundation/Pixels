//! Per-pixel adjustments, after sharp: [`Negate`], [`Gamma`], [`Linear`],
//! [`Threshold`], [`Tint`], [`Greyscale`], [`Recomb`] and [`EnsureAlpha`].
//!
//! Each reads one pixel as RGBA in 0..=1 — whatever its depth and layout —
//! and writes one back, so a single definition serves 8-bit, 16-bit and float
//! images alike, and the output is independent of tiling by construction.
//!
//! # Alpha
//!
//! Alpha is unassociated (SPEC §Formats) and passes through untouched unless
//! an op says otherwise: [`Negate`] inverts it when asked, [`Recomb`] mixes it
//! with a 4x4 matrix, and [`EnsureAlpha`] adds it.

use std::sync::Arc;

use otf_pixels_core::{
    ChannelLayout, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, Tile, TileMut,
};

use crate::convert::{from_rgba, to_rgba};

/// BT.601 luma, the weighting [`crate::ConvertFormat`] and sharp's
/// `greyscale` use.
fn luma([r, g, b, _]: [f32; 4]) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

/// The colour format a greyscale one widens to, for ops whose result has
/// colour even when the input has none.
const fn with_colour(format: PixelFormat) -> PixelFormat {
    match format {
        PixelFormat::Gray8 => PixelFormat::Rgb8,
        PixelFormat::GrayA8 => PixelFormat::Rgba8,
        PixelFormat::Gray16 => PixelFormat::Rgb16,
        other => other,
    }
}

/// What every op in this module is: a function from one RGBA pixel to
/// another, and the format it writes.
trait PixelMap: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    /// The output format for `input`; the same by default.
    fn output(&self, input: PixelFormat) -> Result<PixelFormat> {
        Ok(input)
    }

    /// Map one pixel, RGBA in 0..=1. For a greyscale input, red, green and
    /// blue are equal and the result is narrowed back by luma.
    fn map(&self, rgba: [f32; 4]) -> [f32; 4];
}

/// Apply `map` to every pixel of `input`, writing `output`.
fn compute<M: PixelMap + ?Sized>(
    map: &M,
    inputs: &[Tile<'_>],
    output: &mut TileMut<'_>,
) -> Result<()> {
    let input = inputs
        .first()
        .ok_or_else(|| PixelsError::graph(format!("`{}` takes one input tile", map.name())))?;
    let (from, to) = (input.pixel(), output.pixel());
    let (in_bytes, out_bytes) = (from.bytes_per_pixel(), to.bytes_per_pixel());
    let region = output.region();
    for y in region.y..region.y + region.height {
        let (Some(source), Some(target)) = (input.row(y), output.row_mut(y)) else {
            continue;
        };
        for (pixel, slot) in source
            .chunks_exact(in_bytes)
            .zip(target.chunks_exact_mut(out_bytes))
        {
            from_rgba(map.map(to_rgba(pixel, from)), slot, to);
        }
    }
    Ok(())
}

/// Implement [`Op`] for a [`PixelMap`]: one input, the same region, and a
/// pointwise op is the same at any resolution.
macro_rules! pointwise_op {
    ($type:ty) => {
        impl Op for $type {
            fn name(&self) -> &'static str {
                PixelMap::name(self)
            }

            fn rescaled(&self) -> Option<Arc<dyn Op>> {
                Some(Arc::new(self.clone()))
            }

            fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
                let [input] = inputs else {
                    return Err(PixelsError::graph(format!(
                        "`{}` takes exactly one input",
                        PixelMap::name(self)
                    )));
                };
                let mut out = *input;
                out.pixel = self.output(input.pixel)?;
                Ok(out)
            }

            fn input_regions(
                &self,
                output: Region,
                _inputs: &[ImageDescriptor],
            ) -> Result<Vec<Region>> {
                Ok(vec![output])
            }

            fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
                compute(self, inputs, output)
            }
        }
    };
}

/// Check a parameter is a finite number.
fn finite(name: &'static str, value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PixelsError::invalid_argument(
            name,
            format!("must be finite, got {value}"),
        ))
    }
}

/// Invert colours: each channel becomes `1 - value` (sharp's `negate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Negate {
    alpha: bool,
}

impl Negate {
    /// Invert colours, and alpha too if `alpha` is set. sharp inverts alpha
    /// by default.
    #[must_use]
    pub const fn new(alpha: bool) -> Self {
        Self { alpha }
    }
}

impl PixelMap for Negate {
    fn name(&self) -> &'static str {
        "negate"
    }

    fn map(&self, [r, g, b, a]: [f32; 4]) -> [f32; 4] {
        [
            1.0 - r,
            1.0 - g,
            1.0 - b,
            if self.alpha { 1.0 - a } else { a },
        ]
    }
}
pointwise_op!(Negate);

/// Gamma: each colour channel becomes `value^(1 / exponent)`, as libvips'
/// `gamma` defines it. An exponent above 1 brightens, below 1 darkens.
///
/// sharp's `gamma(g, gOut)` is two of these around its resize: `1 / g`
/// before (darken) and `gOut` after (brighten), which resamples in a more
/// nearly linear space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gamma {
    exponent: f32,
}

impl Gamma {
    /// Gamma with `exponent`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] unless `exponent` is finite
    /// and positive.
    pub fn new(exponent: f32) -> Result<Self> {
        if finite("exponent", exponent)? <= 0.0 {
            return Err(PixelsError::invalid_argument(
                "exponent",
                format!("must be positive, got {exponent}"),
            ));
        }
        Ok(Self { exponent })
    }
}

impl PixelMap for Gamma {
    fn name(&self) -> &'static str {
        "gamma"
    }

    fn map(&self, [r, g, b, a]: [f32; 4]) -> [f32; 4] {
        let power = 1.0 / self.exponent;
        let apply = |v: f32| v.max(0.0).powf(power);
        [apply(r), apply(g), apply(b), a]
    }
}
pointwise_op!(Gamma);

/// A per-channel linear map, `value * a + b` (sharp's `linear`). `b` is in
/// 8-bit units, as in sharp: an offset of 255 is the full range, at any
/// depth. Alpha is untouched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linear {
    multiply: [f32; 3],
    add: [f32; 3],
}

impl Linear {
    /// The same `a` and `b` for every colour channel.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] unless both are finite.
    pub fn new(a: f32, b: f32) -> Result<Self> {
        Self::per_channel([a; 3], [b; 3])
    }

    /// A separate `a` and `b` for red, green and blue. Greyscale images use
    /// the first of each.
    ///
    /// # Errors
    ///
    /// As [`Linear::new`].
    pub fn per_channel(a: [f32; 3], b: [f32; 3]) -> Result<Self> {
        for value in a {
            finite("a", value)?;
        }
        for value in b {
            finite("b", value)?;
        }
        Ok(Self {
            multiply: a,
            add: b.map(|v| v / 255.0),
        })
    }
}

impl PixelMap for Linear {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn map(&self, [r, g, b, a]: [f32; 4]) -> [f32; 4] {
        let [mr, mg, mb] = self.multiply;
        let [ar, ag, ab] = self.add;
        [r * mr + ar, g * mg + ag, b * mb + ab, a]
    }
}
pointwise_op!(Linear);

/// Two-level threshold (sharp's `threshold`): a value at or above the
/// threshold becomes full, anything below becomes zero.
///
/// With `greyscale`, the default in sharp, the pixel's luma is thresholded
/// and written to every colour channel; without it each channel is
/// thresholded on its own. The channel count stays the input's, so an RGB
/// image stays RGB (sharp returns one channel); the values are the same.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold {
    level: f32,
    greyscale: bool,
}

impl Threshold {
    /// Threshold at `level`, in 8-bit units: 128 is sharp's default.
    #[must_use]
    pub fn new(level: u8, greyscale: bool) -> Self {
        Self {
            level: f32::from(level) / 255.0,
            greyscale,
        }
    }
}

impl PixelMap for Threshold {
    fn name(&self) -> &'static str {
        "threshold"
    }

    fn map(&self, rgba: [f32; 4]) -> [f32; 4] {
        // Compared in 8-bit steps, so a 16-bit 0x8000 and an 8-bit 128 agree,
        // and float rounding cannot move a value across the line.
        let step = |v: f32| {
            if (v * 255.0).round() >= (self.level * 255.0).round() {
                1.0
            } else {
                0.0
            }
        };
        let [r, g, b, a] = rgba;
        if self.greyscale {
            let value = step(luma(rgba));
            [value, value, value, a]
        } else {
            [step(r), step(g), step(b), a]
        }
    }
}
pointwise_op!(Threshold);

/// Tint with a colour (sharp's `tint`): each pixel keeps its luma and takes
/// its chroma from the tint.
///
/// sharp does this in Lab; this does it in YCbCr (BT.601), so tinted colours
/// are close to sharp's but not identical. A saturated tint can push a pixel
/// outside the RGB gamut, where it is clipped, as in sharp. A greyscale input
/// becomes colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tint {
    /// The tint's blue-difference and red-difference chroma.
    chroma: [f32; 2],
}

impl Tint {
    /// Tint towards an 8-bit RGB colour.
    #[must_use]
    pub fn new(red: u8, green: u8, blue: u8) -> Self {
        let rgba = [red, green, blue].map(|v| f32::from(v) / 255.0);
        let [r, g, b] = rgba;
        let y = luma([r, g, b, 1.0]);
        Self {
            chroma: [(b - y) * 0.564, (r - y) * 0.713],
        }
    }
}

impl PixelMap for Tint {
    fn name(&self) -> &'static str {
        "tint"
    }

    fn output(&self, input: PixelFormat) -> Result<PixelFormat> {
        Ok(with_colour(input))
    }

    fn map(&self, rgba: [f32; 4]) -> [f32; 4] {
        let y = luma(rgba);
        let [cb, cr] = self.chroma;
        [
            y + 1.403 * cr,
            y - 0.344 * cb - 0.714 * cr,
            y + 1.773 * cb,
            rgba[3],
        ]
    }
}
pointwise_op!(Tint);

/// Greyscale (sharp's `greyscale`): every colour channel becomes the pixel's
/// BT.601 luma. The channel count is kept, as in sharp, whose greyscale
/// output is three identical channels; alpha is untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Greyscale;

impl PixelMap for Greyscale {
    fn name(&self) -> &'static str {
        "greyscale"
    }

    fn map(&self, rgba: [f32; 4]) -> [f32; 4] {
        let y = luma(rgba);
        [y, y, y, rgba[3]]
    }
}
pointwise_op!(Greyscale);

/// Recombine channels through a matrix (sharp's `recomb`): sepia, channel
/// swaps and the like.
///
/// A 3x3 matrix mixes red, green and blue and leaves alpha alone; a 4x4 one
/// mixes alpha in too, and adds an opaque alpha to an image without one so
/// there is somewhere to put it. A greyscale input becomes colour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recomb {
    matrix: [[f32; 4]; 4],
    alpha: bool,
}

impl Recomb {
    /// Recombine red, green and blue: `out[i] = sum(matrix[i][j] * in[j])`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if any entry is not finite.
    pub fn rgb(matrix: [[f32; 3]; 3]) -> Result<Self> {
        let mut full = [[0.0; 4]; 4];
        for (row, from) in full.iter_mut().zip(matrix) {
            for (slot, value) in row.iter_mut().zip(from) {
                *slot = finite("matrix", value)?;
            }
        }
        full[3][3] = 1.0;
        Ok(Self {
            matrix: full,
            alpha: false,
        })
    }

    /// Recombine red, green, blue and alpha.
    ///
    /// # Errors
    ///
    /// As [`Recomb::rgb`].
    pub fn rgba(matrix: [[f32; 4]; 4]) -> Result<Self> {
        for value in matrix.iter().flatten() {
            finite("matrix", *value)?;
        }
        Ok(Self {
            matrix,
            alpha: true,
        })
    }
}

impl PixelMap for Recomb {
    fn name(&self) -> &'static str {
        "recomb"
    }

    fn output(&self, input: PixelFormat) -> Result<PixelFormat> {
        let colour = with_colour(input);
        Ok(if self.alpha {
            with_alpha(colour)
        } else {
            colour
        })
    }

    fn map(&self, rgba: [f32; 4]) -> [f32; 4] {
        self.matrix.map(|row| {
            row.iter()
                .zip(rgba)
                .map(|(weight, value)| weight * value)
                .sum()
        })
    }
}
pointwise_op!(Recomb);

/// The format with an alpha channel added, if it has none.
const fn with_alpha(format: PixelFormat) -> PixelFormat {
    match format {
        PixelFormat::Gray8 => PixelFormat::GrayA8,
        PixelFormat::Rgb8 => PixelFormat::Rgba8,
        PixelFormat::Rgb16 => PixelFormat::Rgba16,
        PixelFormat::RgbF32 => PixelFormat::RgbaF32,
        // There is no 16-bit grey-with-alpha format; 16-bit RGBA holds it.
        PixelFormat::Gray16 => PixelFormat::Rgba16,
        other => other,
    }
}

/// Add an alpha channel at a fixed level if there is none (sharp's
/// `ensureAlpha`); an image that has one is unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsureAlpha {
    level: f32,
}

impl EnsureAlpha {
    /// Add alpha at `level`, 0 transparent to 1 opaque; sharp's default is 1.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] unless `level` is in 0..=1.
    pub fn new(level: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&finite("level", level)?) {
            return Err(PixelsError::invalid_argument(
                "level",
                format!("must be between 0 and 1, got {level}"),
            ));
        }
        Ok(Self { level })
    }
}

impl PixelMap for EnsureAlpha {
    fn name(&self) -> &'static str {
        "ensure_alpha"
    }

    fn output(&self, input: PixelFormat) -> Result<PixelFormat> {
        Ok(with_alpha(input))
    }

    fn map(&self, rgba: [f32; 4]) -> [f32; 4] {
        // Only reached with a new alpha channel to fill when the input had
        // none, in which case `to_rgba` reported it as opaque.
        rgba
    }
}

impl Op for EnsureAlpha {
    fn name(&self) -> &'static str {
        PixelMap::name(self)
    }

    fn rescaled(&self) -> Option<Arc<dyn Op>> {
        Some(Arc::new(*self))
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let [input] = inputs else {
            return Err(PixelsError::graph("`ensure_alpha` takes exactly one input"));
        };
        let mut out = *input;
        out.pixel = self.output(input.pixel)?;
        Ok(out)
    }

    fn input_regions(&self, output: Region, _inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let had_alpha = inputs.first().is_some_and(|tile| {
            matches!(
                tile.pixel().layout(),
                ChannelLayout::GrayAlpha | ChannelLayout::Rgba
            )
        });
        if had_alpha {
            return compute(self, inputs, output);
        }
        let level = self.level;
        compute(&WithLevel(level), inputs, output)
    }
}

/// [`EnsureAlpha`]'s map for an image gaining alpha: set it to the level.
#[derive(Debug)]
struct WithLevel(f32);

impl PixelMap for WithLevel {
    fn name(&self) -> &'static str {
        "ensure_alpha"
    }

    fn map(&self, [r, g, b, _]: [f32; 4]) -> [f32; 4] {
        [r, g, b, self.0]
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::TileBuf;

    /// Run `op` over one whole image given as raw bytes.
    fn run(op: &dyn Op, format: PixelFormat, width: u32, bytes: &[u8]) -> (PixelFormat, Vec<u8>) {
        let height = (bytes.len() / format.bytes_per_pixel()) as u32 / width;
        let input = ImageDescriptor::new(width, height, format).unwrap();
        let out = op.output_descriptor(&[input]).unwrap();
        let source = TileBuf::from_vec(input.region(), format, bytes.to_vec()).unwrap();
        let mut target = TileBuf::for_image(&out).unwrap();
        op.compute(
            &[source.as_tile().unwrap()],
            &mut target.as_tile_mut().unwrap(),
        )
        .unwrap();
        (out.pixel, target.into_bytes())
    }

    #[test]
    fn negate_inverts_colour_and_optionally_alpha() {
        let (_, out) = run(
            &Negate::new(true),
            PixelFormat::Rgba8,
            1,
            &[0, 100, 255, 40],
        );
        assert_eq!(out, [255, 155, 0, 215]);
        let (_, out) = run(
            &Negate::new(false),
            PixelFormat::Rgba8,
            1,
            &[0, 100, 255, 40],
        );
        assert_eq!(out, [255, 155, 0, 40]);
        let (_, out) = run(
            &Negate::new(true),
            PixelFormat::Gray16,
            1,
            &1000_u16.to_ne_bytes(),
        );
        assert_eq!(u16::from_ne_bytes([out[0], out[1]]), 64535);
    }

    #[test]
    fn gamma_brightens_above_one_and_leaves_the_ends_fixed() {
        let op = Gamma::new(2.0).unwrap();
        let (_, out) = run(&op, PixelFormat::Rgb8, 1, &[0, 64, 255]);
        // sqrt(64 / 255) * 255 = 127.75
        assert_eq!(out, [0, 128, 255]);
        assert!(Gamma::new(0.0).is_err());
        assert!(Gamma::new(f32::NAN).is_err());
    }

    #[test]
    fn linear_multiplies_and_offsets_in_8_bit_units_and_clamps() {
        let op = Linear::per_channel([2.0, 1.0, 0.5], [10.0, -20.0, 0.0]).unwrap();
        let (_, out) = run(&op, PixelFormat::Rgba8, 1, &[100, 10, 200, 77]);
        assert_eq!(out, [210, 0, 100, 77]);
        let (_, out) = run(
            &Linear::new(1.0, 255.0).unwrap(),
            PixelFormat::Gray8,
            1,
            &[3],
        );
        assert_eq!(out, [255]);
    }

    #[test]
    fn threshold_uses_luma_or_each_channel() {
        let pixel = [200, 50, 50];
        // Luma is 0.299*200 + 0.587*50 + 0.114*50 = 94.85, below 128.
        let (_, out) = run(&Threshold::new(128, true), PixelFormat::Rgb8, 1, &pixel);
        assert_eq!(out, [0, 0, 0]);
        let (_, out) = run(&Threshold::new(128, false), PixelFormat::Rgb8, 1, &pixel);
        assert_eq!(out, [255, 0, 0]);
        // "At or above" is inclusive, at 8-bit and 16-bit depth alike.
        let (_, out) = run(
            &Threshold::new(128, false),
            PixelFormat::Gray8,
            2,
            &[127, 128],
        );
        assert_eq!(out, [0, 255]);
        let half = 0x8080_u16.to_ne_bytes();
        let (_, out) = run(&Threshold::new(128, false), PixelFormat::Gray16, 1, &half);
        assert_eq!(out, [255, 255]);
    }

    #[test]
    fn greyscale_keeps_three_equal_channels_and_alpha() {
        let (format, out) = run(&Greyscale, PixelFormat::Rgba8, 1, &[200, 50, 50, 9]);
        assert_eq!(format, PixelFormat::Rgba8);
        assert_eq!(out, [95, 95, 95, 9]);
    }

    #[test]
    fn tint_keeps_luma_and_takes_the_tints_hue() {
        // A tint whose chroma stays in gamut at this luma; a fully saturated
        // one would clip, and clipping is what moves luma.
        let (format, out) = run(&Tint::new(200, 100, 100), PixelFormat::Gray8, 1, &[128]);
        assert_eq!(format, PixelFormat::Rgb8);
        let [r, g, b] = [out[0], out[1], out[2]];
        assert!(r > g && r > b, "a red tint should be red: {out:?}");
        let y = 0.299 * f32::from(r) + 0.587 * f32::from(g) + 0.114 * f32::from(b);
        assert!((y - 128.0).abs() <= 2.0, "luma moved to {y}");
        // A grey tint changes nothing on a grey pixel.
        let (_, out) = run(&Tint::new(90, 90, 90), PixelFormat::Rgb8, 1, &[70, 70, 70]);
        assert_eq!(out, [70, 70, 70]);
    }

    #[test]
    fn recomb_applies_the_matrix() {
        let swap = Recomb::rgb([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).unwrap();
        let (_, out) = run(&swap, PixelFormat::Rgba8, 1, &[10, 20, 30, 40]);
        assert_eq!(out, [30, 20, 10, 40]);
        let alpha_from_red = Recomb::rgba([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        .unwrap();
        let (format, out) = run(&alpha_from_red, PixelFormat::Rgb8, 1, &[10, 20, 30]);
        assert_eq!(format, PixelFormat::Rgba8);
        assert_eq!(out, [10, 20, 30, 10]);
        assert!(Recomb::rgb([[f32::INFINITY; 3]; 3]).is_err());
    }

    #[test]
    fn ensure_alpha_adds_a_level_only_when_missing() {
        let op = EnsureAlpha::new(0.5).unwrap();
        let (format, out) = run(&op, PixelFormat::Rgb8, 1, &[1, 2, 3]);
        assert_eq!(format, PixelFormat::Rgba8);
        assert_eq!(out, [1, 2, 3, 128]);
        let (format, out) = run(&op, PixelFormat::Rgba8, 1, &[1, 2, 3, 4]);
        assert_eq!(format, PixelFormat::Rgba8);
        assert_eq!(out, [1, 2, 3, 4]);
        let (format, _) = run(&op, PixelFormat::Gray16, 1, &[0, 0]);
        assert_eq!(format, PixelFormat::Rgba16);
        assert!(EnsureAlpha::new(1.5).is_err());
    }
}
