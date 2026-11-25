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
//!   centre of both images, so the picture sits centred on the canvas.
//!
//! Positive angles turn clockwise, as for [`crate::Rotate`] and sharp. The
//! resampling, alpha handling included, is [`crate::Affine`]'s; a rotation
//! differs from the equivalent affine only in centring its result exactly.

use std::sync::OnceLock;

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Region, Result, Tile, TileMut,
};

use crate::transform::{Interpolator, Mapping, resample, side, sole};

/// The mapping of a clockwise turn by `radians` of a `width` x `height`
/// image about its centre.
fn mapping(radians: f64, width: u32, height: u32) -> Result<Mapping> {
    let (sin, cos) = radians.sin_cos();
    let (w, h) = (f64::from(width), f64::from(height));
    let output = (
        side(w * cos.abs() + h * sin.abs(), "degrees")?,
        side(w * sin.abs() + h * cos.abs(), "degrees")?,
    );
    Ok(Mapping {
        input: (width, height),
        output,
        // The inverse of a clockwise turn in y-down coordinates.
        inverse: [cos, sin, -sin, cos],
        before: (-f64::from(output.0) / 2.0, -f64::from(output.1) / 2.0),
        after: (w / 2.0, h / 2.0),
    })
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
    bound: OnceLock<Mapping>,
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
    fn bind(&self, input: &ImageDescriptor) -> Result<Mapping> {
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
        let geometry = mapping(radians, input.width, input.height)?;
        Ok(*self.bound.get_or_init(|| geometry))
    }
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
        let input = sole(inputs, "rotate")?;
        let mapping = self.bind(input)?;
        input.resized(mapping.output.0, mapping.output.1)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let mapping = self.bind(sole(inputs, "rotate")?)?;
        Ok(vec![mapping.demand(output)])
    }

    fn access_pattern(&self) -> AccessPattern {
        // An output row crosses the input diagonally, so square tiles keep
        // the demand to the tile's own neighbourhood.
        AccessPattern::Spatial
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let mapping = self.bound.get().ok_or_else(|| {
            PixelsError::graph("`rotate` ran before its output shape was computed")
        })?;
        resample(
            "rotate",
            mapping,
            self.options.background,
            self.options.interpolator,
            inputs,
            output,
        )
    }
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
    use otf_pixels_core::{PixelFormat, TileBuf};

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
