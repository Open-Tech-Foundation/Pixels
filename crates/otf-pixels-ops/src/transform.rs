//! Resampling through an inverse coordinate map: the engine shared by
//! [`crate::RotateArbitrary`] and [`crate::Affine`].
//!
//! Each op reduces itself to a [`Mapping`] — a 2x2 matrix and two shifts
//! taking an output pixel to the input point it shows — and an output size.
//! Everything else is here: the [`Interpolator`], the demand an output tile
//! makes, and the kernel.
//!
//! # Conventions
//!
//! Pixel `i` covers `[i, i + 1)`, so its centre is at `i + 0.5`; that is what
//! makes the identity map, a mirror and a quarter turn exact. Taps outside
//! the input read the background colour, so edges are anti-aliased into it.
//! Interpolation runs on premultiplied samples and is divided back out, for
//! the reason [`crate::Composite`] gives: interpolating straight alpha bleeds
//! the colour of transparent pixels into their neighbours, a dark fringe
//! along an edge over a transparent background.
//! One consequence: a fully transparent pixel comes out transparent black,
//! whatever colour it held.
//!
//! There is no prefilter, as in libvips: a map that shrinks by more than
//! about two aliases, and a reducing resize first is the cure.

use otf_pixels_core::{
    ImageDescriptor, PixelFormat, PixelsError, Region, Result, Sample, Tile, TileMut,
    dispatch_sample,
};

use crate::resize::background_pixel;

/// The most channels a pixel format has.
const MAX_CHANNELS: usize = 4;

/// How [`crate::RotateArbitrary`] and [`crate::Affine`] read the input between pixel centres: the subset
/// of sharp's `interpolator`s that this crate implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Interpolator {
    /// The nearest input pixel; hard edges, no new colours.
    Nearest,
    /// Linear between the four nearest pixels — libvips' default for
    /// `rotate`, and so sharp's.
    #[default]
    Bilinear,
    /// Catmull-Rom over the sixteen nearest pixels: sharper than bilinear,
    /// with a slight overshoot at hard edges that is clamped for integer
    /// formats.
    Bicubic,
}

impl Interpolator {
    /// The taps along one axis for source coordinate `at`, as
    /// `(first index, weights)`; unused weights are zero.
    pub(crate) fn taps(self, at: f64) -> (i64, [f32; 4]) {
        match self {
            Self::Nearest => ((at + 0.5).floor() as i64, [1.0, 0.0, 0.0, 0.0]),
            Self::Bilinear => {
                let base = at.floor();
                let t = (at - base) as f32;
                (base as i64, [1.0 - t, t, 0.0, 0.0])
            }
            Self::Bicubic => {
                let base = at.floor();
                let t = (at - base) as f32;
                let (t2, t3) = (t * t, t * t * t);
                // Catmull-Rom, a = -0.5; the four weights sum to one.
                (
                    base as i64 - 1,
                    [
                        0.5 * (-t3 + 2.0 * t2 - t),
                        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
                        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
                        0.5 * (t3 - t2),
                    ],
                )
            }
        }
    }

    /// How many taps [`Self::taps`] uses.
    pub(crate) const fn width(self) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Bilinear => 2,
            Self::Bicubic => 4,
        }
    }
}

/// Where each output pixel of one transform comes from, for one input size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Mapping {
    /// The input image's size.
    pub input: (u32, u32),
    /// The output image's size.
    pub output: (u32, u32),
    /// The inverse matrix, row-major.
    pub inverse: [f64; 4],
    /// Added to an output point before the inverse matrix.
    pub before: (f64, f64),
    /// Added to the result after it.
    pub after: (f64, f64),
}

impl Mapping {
    /// The input coordinate, in pixel-index space, that output pixel
    /// `(x, y)`'s centre comes from.
    pub fn source(&self, x: f64, y: f64) -> (f64, f64) {
        let px = x + 0.5 + self.before.0;
        let py = y + 0.5 + self.before.1;
        let [a, b, c, d] = self.inverse;
        (
            a * px + b * py + self.after.0 - 0.5,
            c * px + d * py + self.after.1 - 0.5,
        )
    }

    /// The input region output tile `output` reads.
    pub fn demand(&self, output: Region) -> Region {
        // A linear map takes the tile to a parallelogram, whose bounding box
        // is spanned by its corners; so the four corner pixel centres bound
        // every centre in the tile.
        let (x0, y0) = (f64::from(output.x), f64::from(output.y));
        let x1 = x0 + f64::from(output.width.saturating_sub(1));
        let y1 = y0 + f64::from(output.height.saturating_sub(1));
        let (mut low_u, mut high_u, mut low_v, mut high_v) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
            let (u, v) = self.source(x, y);
            low_u = low_u.min(u);
            high_u = high_u.max(u);
            low_v = low_v.min(v);
            high_v = high_v.max(v);
        }

        // Widened by the filter's reach, and a pixel more for rounding.
        let reach = 3.0;
        let span = |low: f64, high: f64, size: u32| -> Option<(u32, u32)> {
            let last = f64::from(size) - 1.0;
            let start = (low.floor() - reach).max(0.0);
            let end = (high.floor() + reach).min(last);
            (start <= end).then(|| (start as u32, end as u32 + 1))
        };
        match (
            span(low_u, high_u, self.input.0),
            span(low_v, high_v, self.input.1),
        ) {
            (Some((left, right)), Some((top, bottom))) => {
                Region::new(left, top, right - left, bottom - top)
            }
            // A tile of pure background still has to name a valid region,
            // so it asks for one pixel it will not read.
            _ => Region::new(0, 0, 1, 1),
        }
    }
}

/// One side of an output, `extent` long, as a pixel count: rounded up, so
/// that nothing is cut off, but from a hair under, so that rounding error
/// in a trigonometric extent cannot add a pixel.
pub(crate) fn side(extent: f64, what: &'static str) -> Result<u32> {
    let v = (extent - 1e-9).ceil().max(1.0);
    if !v.is_finite() || v > f64::from(u32::MAX) {
        return Err(PixelsError::invalid_argument(
            what,
            "the transformed image is too large",
        ));
    }
    Ok(v as u32)
}

/// Fetch the sole input descriptor of op `name`.
pub(crate) fn sole<'a>(inputs: &'a [ImageDescriptor], name: &str) -> Result<&'a ImageDescriptor> {
    inputs
        .first()
        .ok_or_else(|| PixelsError::graph(format!("`{name}` takes one input, got none")))
}

/// Resample one output tile of op `name` through `mapping`.
pub(crate) fn resample(
    name: &str,
    mapping: &Mapping,
    background: [u8; 4],
    interpolator: Interpolator,
    inputs: &[Tile<'_>],
    output: &mut TileMut<'_>,
) -> Result<()> {
    let input = inputs
        .first()
        .ok_or_else(|| PixelsError::graph(format!("`{name}` takes one input tile, got none")))?;
    if input.pixel() != output.pixel() {
        return Err(PixelsError::graph(format!(
            "`{name}` input is {} but output is {}",
            input.pixel(),
            output.pixel()
        )));
    }
    let format = output.pixel();
    if format.channels() > MAX_CHANNELS {
        return Err(PixelsError::unsupported(format!(
            "{name} handles up to {MAX_CHANNELS} channels, {format} has {}",
            format.channels()
        )));
    }
    dispatch_sample!(format.sample_kind(), S => {
        resample_tile::<S>(mapping, background, interpolator, input, output, format)
    })
}

/// Reading and writing one sample type as `f32` in its own scale.
trait Value: Sample {
    fn load(bytes: &[u8], index: usize) -> f32;
    fn store(value: f32, into: &mut Vec<u8>);
    fn full_scale() -> f32;
}

impl Value for u8 {
    fn load(bytes: &[u8], index: usize) -> f32 {
        bytes.get(index).map_or(0.0, |&v| f32::from(v))
    }
    fn store(value: f32, into: &mut Vec<u8>) {
        into.push((value.clamp(0.0, 255.0) + 0.5) as u8);
    }
    fn full_scale() -> f32 {
        255.0
    }
}

impl Value for u16 {
    fn load(bytes: &[u8], index: usize) -> f32 {
        let at = index * 2;
        bytes
            .get(at..at + 2)
            .and_then(|b| b.try_into().ok())
            .map_or(0.0, |b| f32::from(u16::from_ne_bytes(b)))
    }
    fn store(value: f32, into: &mut Vec<u8>) {
        into.extend_from_slice(&((value.clamp(0.0, 65535.0) + 0.5) as u16).to_ne_bytes());
    }
    fn full_scale() -> f32 {
        65535.0
    }
}

impl Value for f32 {
    fn load(bytes: &[u8], index: usize) -> f32 {
        let at = index * 4;
        bytes
            .get(at..at + 4)
            .and_then(|b| b.try_into().ok())
            .map_or(0.0, f32::from_ne_bytes)
    }
    fn store(value: f32, into: &mut Vec<u8>) {
        into.extend_from_slice(&value.to_ne_bytes());
    }
    fn full_scale() -> f32 {
        1.0
    }
}

/// Premultiply `pixel` in place when the format has alpha.
fn premultiply(pixel: &mut [f32; MAX_CHANNELS], channels: usize, alpha: bool, full: f32) {
    if alpha {
        let a = pixel.get(channels - 1).map_or(1.0, |&a| a / full);
        for value in pixel.iter_mut().take(channels - 1) {
            *value *= a;
        }
    }
}

/// Resample one output tile.
fn resample_tile<S: Value>(
    mapping: &Mapping,
    background: [u8; 4],
    interpolator: Interpolator,
    input: &Tile<'_>,
    output: &mut TileMut<'_>,
    format: PixelFormat,
) -> Result<()> {
    let channels = format.channels();
    let alpha = format.has_alpha();
    let full = S::full_scale();
    let taps = interpolator.width();
    let source = input.region();
    let region = output.region();
    let (width, height) = (i64::from(mapping.input.0), i64::from(mapping.input.1));

    let background = {
        let bytes = background_pixel(format, background);
        let mut pixel = [0.0_f32; MAX_CHANNELS];
        for (c, slot) in pixel.iter_mut().enumerate().take(channels) {
            *slot = S::load(&bytes, c);
        }
        premultiply(&mut pixel, channels, alpha, full);
        pixel
    };

    // One input pixel, premultiplied, or the background if it lies outside
    // the image (or, defensively, outside the tile that was delivered).
    let fetch = |x: i64, y: i64| -> [f32; MAX_CHANNELS] {
        if x < 0 || y < 0 || x >= width || y >= height {
            return background;
        }
        let (x, y) = (x as u32, y as u32);
        if x < source.x || x >= source.x + source.width {
            return background;
        }
        let Some(row) = input.row(y) else {
            return background;
        };
        let first = (x - source.x) as usize * channels;
        let mut pixel = [0.0_f32; MAX_CHANNELS];
        for (c, slot) in pixel.iter_mut().enumerate().take(channels) {
            *slot = S::load(row, first + c);
        }
        premultiply(&mut pixel, channels, alpha, full);
        pixel
    };

    let mut row_out = Vec::with_capacity(region.width as usize * format.bytes_per_pixel());
    for y in region.y..region.y + region.height {
        row_out.clear();
        for x in region.x..region.x + region.width {
            let (u, v) = mapping.source(f64::from(x), f64::from(y));
            let (first_x, weights_x) = interpolator.taps(u);
            let (first_y, weights_y) = interpolator.taps(v);

            let mut sum = [0.0_f32; MAX_CHANNELS];
            for (j, &wy) in weights_y.iter().enumerate().take(taps) {
                let mut line = [0.0_f32; MAX_CHANNELS];
                for (i, &wx) in weights_x.iter().enumerate().take(taps) {
                    let pixel = fetch(first_x + i as i64, first_y + j as i64);
                    // All four lanes: the unused ones stay zero.
                    for (acc, value) in line.iter_mut().zip(pixel) {
                        *acc += wx * value;
                    }
                }
                for (acc, value) in sum.iter_mut().zip(line) {
                    *acc += wy * value;
                }
            }

            if alpha {
                let a = sum.get(channels - 1).map_or(full, |a| a.clamp(0.0, full));
                let scale = if a > 0.0 { full / a } else { 0.0 };
                for &value in sum.iter().take(channels - 1) {
                    S::store(value * scale, &mut row_out);
                }
                S::store(a, &mut row_out);
            } else {
                for &value in sum.iter().take(channels) {
                    S::store(value, &mut row_out);
                }
            }
        }
        if let Some(target) = output.row_mut(y) {
            let len = target.len().min(row_out.len());
            if let (Some(to), Some(from)) = (target.get_mut(..len), row_out.get(..len)) {
                to.copy_from_slice(from);
            }
        }
    }
    Ok(())
}
