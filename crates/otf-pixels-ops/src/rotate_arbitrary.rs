//! [`RotateArbitrary`] — rotation by any angle (sharp's `rotate(angle)` for an
//! angle that is not a multiple of 90).
//!
//! A quarter turn is a pure permutation and lives in [`crate::Rotate`]. Any
//! other angle lands output pixel centres between input pixels, so it needs
//! the three decisions [`crate::Rotate`] deliberately refuses to make:
//!
//! - **A filter.** [`Interpolator`]: nearest, bilinear (the default, as for
//!   libvips' `rotate`, which sharp calls) or Catmull-Rom bicubic.
//! - **An edge policy.** Everything outside the input is the background
//!   colour, and the filter reads it like any other pixel, so the rotated
//!   image's edges are anti-aliased into the background rather than jagged.
//! - **An output size.** The bounding box of the rotated input, as libvips
//!   and sharp give it: `w·|cos θ| + h·|sin θ|` by `w·|sin θ| + h·|cos θ|`,
//!   rounded up so that no corner is cut off. The rotation is about the
//!   centre of both images.
//!
//! Positive angles turn clockwise, as for [`crate::Rotate`] and sharp.
//!
//! # Alpha
//!
//! Interpolation runs on premultiplied samples and is divided back out, for
//! the reason [`crate::Composite`] gives: interpolating straight alpha bleeds
//! the colour of transparent pixels into their neighbours, which shows as a
//! dark fringe along the edge of an image rotated over a transparent
//! background.

use std::sync::OnceLock;

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, Sample, Tile,
    TileMut, dispatch_sample,
};

use crate::resize::background_pixel;

/// The most channels a pixel format has.
const MAX_CHANNELS: usize = 4;

/// How [`RotateArbitrary`] reads the input between pixel centres: the subset
/// of sharp's `interpolator`s that apply to a rotation.
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
    fn taps(self, at: f64) -> (i64, [f32; 4]) {
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
    const fn width(self) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Bilinear => 2,
            Self::Bicubic => 4,
        }
    }
}

/// The geometry of one rotation against one input size.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Geometry {
    input: (u32, u32),
    output: (u32, u32),
    cos: f64,
    sin: f64,
}

impl Geometry {
    fn new(radians: f64, width: u32, height: u32) -> Result<Self> {
        let (sin, cos) = radians.sin_cos();
        let (w, h) = (f64::from(width), f64::from(height));
        // Up to a hair under: 30 degrees of a 100-pixel side is 86.6025...,
        // and 45 degrees of an even square can land within an ulp of an
        // integer that rounding error must not push one pixel further.
        let side = |v: f64| -> Result<u32> {
            let v = (v - 1e-9).ceil().max(1.0);
            if v > f64::from(u32::MAX) {
                return Err(PixelsError::invalid_argument(
                    "degrees",
                    "the rotated image is too large",
                ));
            }
            Ok(v as u32)
        };
        let output = (
            side(w * cos.abs() + h * sin.abs())?,
            side(w * sin.abs() + h * cos.abs())?,
        );
        Ok(Self {
            input: (width, height),
            output,
            cos,
            sin,
        })
    }

    /// The input coordinate, in pixel-index space, that output pixel
    /// `(x, y)`'s centre comes from.
    fn source(&self, x: f64, y: f64) -> (f64, f64) {
        let dx = x + 0.5 - f64::from(self.output.0) / 2.0;
        let dy = y + 0.5 - f64::from(self.output.1) / 2.0;
        // The inverse of a clockwise turn in y-down coordinates.
        let u = dx * self.cos + dy * self.sin;
        let v = -dx * self.sin + dy * self.cos;
        (
            u + f64::from(self.input.0) / 2.0 - 0.5,
            v + f64::from(self.input.1) / 2.0 - 0.5,
        )
    }
}

/// Options for [`RotateArbitrary`]: sharp's `rotate(angle, { background })`,
/// with the interpolator its `affine` takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RotateOptions {
    background: [u8; 4],
    interpolator: Interpolator,
}

impl Default for RotateOptions {
    /// Opaque black and bilinear, sharp's defaults.
    fn default() -> Self {
        Self {
            background: [0, 0, 0, 255],
            interpolator: Interpolator::Bilinear,
        }
    }
}

impl RotateOptions {
    /// The RGBA colour of the corners the rotation uncovers. Grey formats
    /// take its luma, and formats without alpha ignore its alpha.
    #[must_use]
    pub const fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// The filter used between pixel centres.
    #[must_use]
    pub const fn with_interpolator(mut self, interpolator: Interpolator) -> Self {
        self.interpolator = interpolator;
        self
    }

    /// The background colour.
    #[must_use]
    pub const fn background(&self) -> [u8; 4] {
        self.background
    }

    /// The interpolator.
    #[must_use]
    pub const fn interpolator(&self) -> Interpolator {
        self.interpolator
    }
}

/// Rotate an image clockwise by any angle, onto a canvas that holds all of it.
///
/// Like [`crate::Extend`], an instance belongs to one graph edge: the output
/// size and the coordinate mapping depend on the input's size, which is bound
/// when the op is chained.
#[derive(Debug)]
pub struct RotateArbitrary {
    degrees: f64,
    options: RotateOptions,
    bound: OnceLock<Geometry>,
}

impl Clone for RotateArbitrary {
    fn clone(&self) -> Self {
        // Not the binding: a clone may be chained onto another image.
        Self {
            degrees: self.degrees,
            options: self.options,
            bound: OnceLock::new(),
        }
    }
}

impl RotateArbitrary {
    /// Rotate by `degrees` clockwise; negative turns anticlockwise.
    ///
    /// A multiple of 90 is accepted and resampled like any other angle, which
    /// is exact for nearest and bilinear but slower than [`crate::Rotate`];
    /// `Image::rotate_with` picks [`crate::Rotate`] for those itself.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `degrees` is not finite.
    pub fn new(degrees: f64, options: RotateOptions) -> Result<Self> {
        if !degrees.is_finite() {
            return Err(PixelsError::invalid_argument(
                "degrees",
                format!("rotation must be a finite angle, got {degrees}"),
            ));
        }
        Ok(Self {
            degrees,
            options,
            bound: OnceLock::new(),
        })
    }

    /// The angle, in degrees clockwise, as given.
    #[must_use]
    pub const fn degrees(&self) -> f64 {
        self.degrees
    }

    /// The options.
    #[must_use]
    pub const fn options(&self) -> RotateOptions {
        self.options
    }

    /// Bind to `input`'s size, or check it matches the size already bound.
    fn bind(&self, input: &ImageDescriptor) -> Result<Geometry> {
        if let Some(geometry) = self.bound.get() {
            if geometry.input != (input.width, input.height) {
                return Err(PixelsError::graph(format!(
                    "`rotate` is bound to a {}x{} input but was given {}x{}; \
                     build a new RotateArbitrary per image",
                    geometry.input.0, geometry.input.1, input.width, input.height
                )));
            }
            return Ok(*geometry);
        }
        // Reduced to one turn first, so a large angle loses no precision in
        // the conversion to radians.
        let radians = self.degrees.rem_euclid(360.0).to_radians();
        let geometry = Geometry::new(radians, input.width, input.height)?;
        Ok(*self.bound.get_or_init(|| geometry))
    }
}

/// Fetch the sole input descriptor.
fn sole(inputs: &[ImageDescriptor]) -> Result<&ImageDescriptor> {
    inputs
        .first()
        .ok_or_else(|| PixelsError::graph("`rotate` takes one input, got none"))
}

impl Op for RotateArbitrary {
    /// An angle means the same thing at any resolution, and the output size
    /// is recomputed from whatever input arrives, so a rotation before a
    /// resize keeps shrink-on-load. The copy is unbound.
    fn rescaled(&self) -> Option<std::sync::Arc<dyn Op>> {
        Some(std::sync::Arc::new(self.clone()))
    }

    fn name(&self) -> &'static str {
        "rotate"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole(inputs)?;
        let geometry = self.bind(input)?;
        input.resized(geometry.output.0, geometry.output.1)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let input = sole(inputs)?;
        let geometry = self.bind(input)?;

        // A rotated rectangle's bounding box is spanned by its corners, so
        // the four corner pixel centres bound every centre in the tile.
        let (x0, y0) = (f64::from(output.x), f64::from(output.y));
        let x1 = x0 + f64::from(output.width.saturating_sub(1));
        let y1 = y0 + f64::from(output.height.saturating_sub(1));
        let corners = [
            geometry.source(x0, y0),
            geometry.source(x1, y0),
            geometry.source(x0, y1),
            geometry.source(x1, y1),
        ];
        let (mut low_u, mut high_u, mut low_v, mut high_v) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for (u, v) in corners {
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
        Ok(vec![match (
            span(low_u, high_u, input.width),
            span(low_v, high_v, input.height),
        ) {
            (Some((left, right)), Some((top, bottom))) => {
                Region::new(left, top, right - left, bottom - top)
            }
            // A tile of pure background still has to name a valid
            // region, so it asks for one pixel it will not read.
            _ => Region::new(0, 0, 1, 1),
        }])
    }

    fn access_pattern(&self) -> AccessPattern {
        // An output row crosses the input diagonally, so square tiles keep
        // the demand to the tile's own neighbourhood.
        AccessPattern::Spatial
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`rotate` takes one input tile, got none"))?;
        if input.pixel() != output.pixel() {
            return Err(PixelsError::graph(format!(
                "`rotate` input is {} but output is {}",
                input.pixel(),
                output.pixel()
            )));
        }
        let geometry = *self.bound.get().ok_or_else(|| {
            PixelsError::graph("`rotate` ran before its output shape was computed")
        })?;
        let format = output.pixel();
        if format.channels() > MAX_CHANNELS {
            return Err(PixelsError::unsupported(format!(
                "rotate handles up to {MAX_CHANNELS} channels, {format} has {}",
                format.channels()
            )));
        }
        dispatch_sample!(format.sample_kind(), S => {
            rotate_tile::<S>(&geometry, self.options, input, output, format)
        })
    }
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
fn rotate_tile<S: Value>(
    geometry: &Geometry,
    options: RotateOptions,
    input: &Tile<'_>,
    output: &mut TileMut<'_>,
    format: PixelFormat,
) -> Result<()> {
    let channels = format.channels();
    let alpha = format.has_alpha();
    let full = S::full_scale();
    let interpolator = options.interpolator;
    let taps = interpolator.width();
    let source = input.region();
    let region = output.region();
    let (width, height) = (i64::from(geometry.input.0), i64::from(geometry.input.1));

    let background = {
        let bytes = background_pixel(format, options.background);
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
            let (u, v) = geometry.source(f64::from(x), f64::from(y));
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::{Quarter, Rotate};
    use otf_pixels_core::TileBuf;

    fn apply(
        op: &dyn Op,
        input: &ImageDescriptor,
        bytes: &[u8],
    ) -> Result<(ImageDescriptor, Vec<u8>)> {
        let out_desc = op.output_descriptor(std::slice::from_ref(input))?;
        let source = TileBuf::from_vec(input.region(), input.pixel, bytes.to_vec())?;
        let mut target = TileBuf::for_image(&out_desc)?;
        op.compute(&[source.as_tile()?], &mut target.as_tile_mut()?)?;
        Ok((out_desc, target.into_bytes()))
    }

    fn rotate(degrees: f64) -> RotateArbitrary {
        RotateArbitrary::new(degrees, RotateOptions::default()).unwrap()
    }

    fn rotate_with(degrees: f64, options: RotateOptions) -> RotateArbitrary {
        RotateArbitrary::new(degrees, options).unwrap()
    }

    /// A smooth, non-symmetric pattern.
    fn pattern(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let len = descriptor.byte_len().unwrap();
        let bytes = (0..len).map(|i| (i * 7 % 251) as u8).collect();
        (descriptor, bytes)
    }

    fn size_after(degrees: f64, width: u32, height: u32) -> (u32, u32) {
        let input = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();
        let out = rotate(degrees)
            .output_descriptor(std::slice::from_ref(&input))
            .unwrap();
        (out.width, out.height)
    }

    #[test]
    fn the_canvas_is_the_rotated_bounding_box() {
        // 100·cos45 + 100·sin45 = 141.42..., rounded up as libvips does.
        assert_eq!(size_after(45.0, 100, 100), (142, 142));
        // 100·0.866 + 50·0.5 = 111.6; 100·0.5 + 50·0.866 = 93.3.
        assert_eq!(size_after(30.0, 100, 50), (112, 94));
        assert_eq!(size_after(-30.0, 100, 50), (112, 94));
        assert_eq!(size_after(390.0, 100, 50), (112, 94));
        // Exact quarter turns through this op still give exact sizes.
        assert_eq!(size_after(90.0, 100, 50), (50, 100));
        assert_eq!(size_after(180.0, 100, 50), (100, 50));
        assert_eq!(size_after(0.0, 1, 1), (1, 1));
    }

    #[test]
    fn a_non_finite_angle_is_an_error() {
        for degrees in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(RotateArbitrary::new(degrees, RotateOptions::default()).is_err());
        }
    }

    #[test]
    fn a_quarter_turn_by_resampling_matches_the_exact_rotation() {
        // At 90 degrees every output centre lands on an input centre, so
        // nearest and bilinear must reproduce `Rotate` exactly. This pins the
        // direction, the centre and the half-pixel convention at once.
        let (desc, bytes) = pattern(9, 6, PixelFormat::Rgb8);
        for (degrees, quarter) in [
            (90.0, Quarter::Clockwise90),
            (180.0, Quarter::Half),
            (270.0, Quarter::Clockwise270),
            (-90.0, Quarter::Clockwise270),
        ] {
            let (_, exact) = apply(&Rotate::new(quarter), &desc, &bytes).unwrap();
            for interpolator in [Interpolator::Nearest, Interpolator::Bilinear] {
                let op = rotate_with(
                    degrees,
                    RotateOptions::default().with_interpolator(interpolator),
                );
                let (_, resampled) = apply(&op, &desc, &bytes).unwrap();
                assert_eq!(resampled, exact, "{degrees} with {interpolator:?}");
            }
        }
    }

    #[test]
    fn positive_angles_turn_clockwise() {
        // A white dot right of centre on black moves down when turned
        // clockwise (y grows downwards).
        let desc = ImageDescriptor::new(21, 21, PixelFormat::Gray8).unwrap();
        let mut bytes = vec![0_u8; 21 * 21];
        bytes[10 * 21 + 18] = 255;
        let (out, rotated) = apply(&rotate(30.0), &desc, &bytes).unwrap();
        let (mut best, mut at) = (0, (0, 0));
        for y in 0..out.height {
            for x in 0..out.width {
                let v = rotated[(y * out.width + x) as usize];
                if v > best {
                    best = v;
                    at = (x, y);
                }
            }
        }
        let centre = (out.width as f64 / 2.0, out.height as f64 / 2.0);
        let (dx, dy) = (at.0 as f64 + 0.5 - centre.0, at.1 as f64 + 0.5 - centre.1);
        assert!(dx > 0.0 && dy > 0.0, "dot moved to ({dx}, {dy})");
        let angle = dy.atan2(dx).to_degrees();
        assert!((angle - 30.0).abs() < 4.0, "dot at {angle} degrees");
    }

    #[test]
    fn uncovered_corners_are_the_background_and_a_flat_interior_stays_flat() {
        let desc = ImageDescriptor::new(40, 40, PixelFormat::Rgb8).unwrap();
        let bytes = [200_u8, 100, 50].repeat(40 * 40);
        for interpolator in [
            Interpolator::Nearest,
            Interpolator::Bilinear,
            Interpolator::Bicubic,
        ] {
            let op = rotate_with(
                45.0,
                RotateOptions::default()
                    .with_background([10, 20, 30, 255])
                    .with_interpolator(interpolator),
            );
            let (out, rotated) = apply(&op, &desc, &bytes).unwrap();
            let px = |x: u32, y: u32| {
                let at = ((y * out.width + x) * 3) as usize;
                [rotated[at], rotated[at + 1], rotated[at + 2]]
            };
            assert_eq!(px(0, 0), [10, 20, 30], "{interpolator:?}");
            assert_eq!(px(out.width - 1, out.height - 1), [10, 20, 30]);
            // The weights sum to one, so the interior is exactly the colour.
            assert_eq!(px(out.width / 2, out.height / 2), [200, 100, 50]);
        }
    }

    #[test]
    fn edges_over_a_transparent_background_do_not_darken() {
        // Premultiplied interpolation: a partly covered edge pixel is the
        // image's colour at partial alpha, not a mix with transparent black.
        let desc = ImageDescriptor::new(30, 30, PixelFormat::Rgba8).unwrap();
        let bytes = [255_u8, 128, 0, 255].repeat(30 * 30);
        let op = rotate_with(20.0, RotateOptions::default().with_background([0, 0, 0, 0]));
        let (_, rotated) = apply(&op, &desc, &bytes).unwrap();
        let mut partial = 0;
        for pixel in rotated.chunks_exact(4) {
            match pixel[3] {
                0 => assert_eq!(pixel, [0, 0, 0, 0]),
                255 => assert_eq!(pixel, [255, 128, 0, 255]),
                _ => {
                    partial += 1;
                    assert_eq!(pixel[0], 255, "red fringed in {pixel:?}");
                    assert!((i32::from(pixel[1]) - 128).abs() <= 1, "{pixel:?}");
                    assert_eq!(pixel[2], 0, "{pixel:?}");
                }
            }
        }
        assert!(partial > 0, "the edge should be anti-aliased");
    }

    #[test]
    fn turning_there_and_back_recovers_the_middle() {
        let desc = ImageDescriptor::new(32, 32, PixelFormat::Gray8).unwrap();
        // Smooth, so bilinear's blur stays small.
        let bytes: Vec<u8> = (0..32 * 32)
            .map(|i| ((i % 32) * 4 + (i / 32) * 3) as u8)
            .collect();
        // 28 degrees grows 32 to 44 and then 60, an even margin, so the
        // original's pixel centres land on the final canvas's.
        let (mid_desc, mid) = apply(&rotate(28.0), &desc, &bytes).unwrap();
        let (back_desc, back) = apply(&rotate(-28.0), &mid_desc, &mid).unwrap();
        assert_eq!((back_desc.width, back_desc.height), (60, 60));
        // Crop the centre 32x32 of the grown canvas back out.
        let ox = (back_desc.width - 32) / 2;
        let oy = (back_desc.height - 32) / 2;
        for y in 8..24 {
            for x in 8..24 {
                let a = i32::from(bytes[(y * 32 + x) as usize]);
                let b = i32::from(back[((y + oy) * back_desc.width + x + ox) as usize]);
                assert!((a - b).abs() <= 3, "({x}, {y}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        for &format in &[PixelFormat::Rgb8, PixelFormat::Rgba16, PixelFormat::RgbaF32] {
            for interpolator in [
                Interpolator::Nearest,
                Interpolator::Bilinear,
                Interpolator::Bicubic,
            ] {
                for degrees in [17.0, 45.0, -123.4, 200.0] {
                    let (desc, bytes) = pattern(23, 15, format);
                    let op = rotate_with(
                        degrees,
                        RotateOptions::default()
                            .with_interpolator(interpolator)
                            .with_background([1, 2, 3, 100]),
                    );
                    let (out_desc, whole) = apply(&op, &desc, &bytes).unwrap();
                    let source =
                        TileBuf::from_vec(desc.region(), desc.pixel, bytes.clone()).unwrap();
                    for (tw, th) in [(4_u32, 4_u32), (1, 7), (out_desc.width, 1), (5, 3)] {
                        let mut target = TileBuf::for_image(&out_desc).unwrap();
                        let mut y = 0;
                        while y < out_desc.height {
                            let h = th.min(out_desc.height - y);
                            let mut x = 0;
                            while x < out_desc.width {
                                let w = tw.min(out_desc.width - x);
                                let region = Region::new(x, y, w, h);
                                let demand = op
                                    .input_regions(region, std::slice::from_ref(&desc))
                                    .unwrap()[0];
                                assert!(
                                    demand.x + demand.width <= desc.width
                                        && demand.y + demand.height <= desc.height,
                                    "demand {demand} leaves the input"
                                );
                                let mut cut = TileBuf::zeroed(demand, desc.pixel).unwrap();
                                otf_pixels_core::copy_region(
                                    &source.as_tile().unwrap(),
                                    &mut cut.as_tile_mut().unwrap(),
                                    demand,
                                )
                                .unwrap();
                                let mut sub = TileBuf::zeroed(region, out_desc.pixel).unwrap();
                                op.compute(
                                    &[cut.as_tile().unwrap()],
                                    &mut sub.as_tile_mut().unwrap(),
                                )
                                .unwrap();
                                otf_pixels_core::copy_region(
                                    &sub.as_tile().unwrap(),
                                    &mut target.as_tile_mut().unwrap(),
                                    region,
                                )
                                .unwrap();
                                x += w;
                            }
                            y += h;
                        }
                        assert_eq!(
                            target.bytes(),
                            whole.as_slice(),
                            "{format} {interpolator:?} {degrees} at {tw}x{th} tiles"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_pixel_format_rotates() {
        for &format in PixelFormat::ALL {
            let (desc, bytes) = pattern(7, 5, format);
            let (out, rotated) = apply(&rotate(10.0), &desc, &bytes).unwrap();
            assert_eq!(out.pixel, format);
            assert_eq!(rotated.len(), out.byte_len().unwrap(), "{format}");
        }
    }

    #[test]
    fn an_instance_is_bound_to_one_input_size() {
        let op = rotate(10.0);
        let a = ImageDescriptor::new(10, 10, PixelFormat::Rgb8).unwrap();
        let b = ImageDescriptor::new(20, 10, PixelFormat::Rgb8).unwrap();
        op.output_descriptor(std::slice::from_ref(&a)).unwrap();
        assert!(op.output_descriptor(std::slice::from_ref(&b)).is_err());
        // A clone, and the rescaled copy, are free to bind afresh.
        op.clone()
            .output_descriptor(std::slice::from_ref(&b))
            .unwrap();
        op.rescaled()
            .unwrap()
            .output_descriptor(std::slice::from_ref(&b))
            .unwrap();
    }
}
