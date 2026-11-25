//! [`Affine`] — any 2x2 linear transform (sharp's `affine`, which is
//! libvips').
//!
//! An input point `p` goes to `M·(p + input offset) + output offset`, with
//! `M = [[a, b], [c, d]]`, so `X = a·x + b·y` and `Y = c·x + d·y` in y-down
//! pixel coordinates. That covers scaling (different along each axis if
//! wanted), shear, mirroring and rotation, and any product of them resampled
//! once rather than once per step.
//!
//! # Output
//!
//! As in libvips, the canvas is the bounding box of the transformed input,
//! with its edges rounded outwards to whole pixels. Moving the whole image
//! moves its bounding box with it, so the offsets only change where the
//! picture falls *within* a pixel: `0.5` is a half-pixel shift, and a whole
//! number changes nothing.
//!
//! Resampling is shared with [`crate::RotateArbitrary`]: pixel centres at
//! `i + 0.5`, so the identity, mirrors and quarter turns are exact; taps
//! outside the input read the background, anti-aliasing the edges; and
//! interpolation runs on premultiplied alpha, so a fully transparent pixel
//! comes out transparent black. There is no prefilter, as in libvips: a
//! matrix that shrinks by more than about two aliases, so reduce with a
//! resize first. sharp's default interpolator here is bicubic, and so is
//! this one's.

use std::sync::OnceLock;

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Region, Result, Tile, TileMut,
};

use crate::transform::{Interpolator, Mapping, resample, side, sole};

/// Options for [`Affine`]: sharp's `affine(matrix, { background, idx, idy,
/// odx, ody, interpolator })`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineOptions {
    background: [u8; 4],
    interpolator: Interpolator,
    input_offset: (f64, f64),
    output_offset: (f64, f64),
}

impl Default for AffineOptions {
    /// Opaque black, bicubic and no offsets: sharp's defaults.
    fn default() -> Self {
        Self {
            background: [0, 0, 0, 255],
            interpolator: Interpolator::Bicubic,
            input_offset: (0.0, 0.0),
            output_offset: (0.0, 0.0),
        }
    }
}

impl AffineOptions {
    /// The RGBA colour of the canvas the image does not cover. Grey formats
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

    /// Added to input coordinates before the matrix (sharp's `idx`, `idy`).
    #[must_use]
    pub const fn with_input_offset(mut self, x: f64, y: f64) -> Self {
        self.input_offset = (x, y);
        self
    }

    /// Added to output coordinates after it (sharp's `odx`, `ody`).
    #[must_use]
    pub const fn with_output_offset(mut self, x: f64, y: f64) -> Self {
        self.output_offset = (x, y);
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

    /// The input offset, `(idx, idy)`.
    #[must_use]
    pub const fn input_offset(&self) -> (f64, f64) {
        self.input_offset
    }

    /// The output offset, `(odx, ody)`.
    #[must_use]
    pub const fn output_offset(&self) -> (f64, f64) {
        self.output_offset
    }

    /// Whether both offsets are zero.
    fn unshifted(&self) -> bool {
        self.input_offset == (0.0, 0.0) && self.output_offset == (0.0, 0.0)
    }
}

/// Transform an image by a 2x2 matrix onto a canvas that holds all of it.
///
/// Like [`crate::Extend`], an instance belongs to one graph edge: the output
/// size depends on the input's, which is bound when the op is chained.
#[derive(Debug)]
pub struct Affine {
    matrix: [f64; 4],
    options: AffineOptions,
    bound: OnceLock<Mapping>,
}

impl Clone for Affine {
    fn clone(&self) -> Self {
        // Not the binding: a clone may be chained onto another image.
        Self {
            matrix: self.matrix,
            options: self.options,
            bound: OnceLock::new(),
        }
    }
}

impl Affine {
    /// Transform by `[a, b, c, d]`, the matrix `[[a, b], [c, d]]` row by row
    /// (sharp takes either form).
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if an entry or an offset is
    /// not finite, or if the matrix is singular: it would flatten the image
    /// to a line, and has no inverse to resample through.
    pub fn new(matrix: [f64; 4], options: AffineOptions) -> Result<Self> {
        let (ix, iy) = options.input_offset;
        let (ox, oy) = options.output_offset;
        if !matrix
            .iter()
            .chain(&[ix, iy, ox, oy])
            .all(|v| v.is_finite())
        {
            return Err(PixelsError::invalid_argument(
                "matrix",
                "every matrix entry and offset must be finite",
            ));
        }
        let [a, b, c, d] = matrix;
        let scale = matrix.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let determinant = a * d - b * c;
        if determinant.abs() <= scale * scale * 1e-12 {
            return Err(PixelsError::invalid_argument(
                "matrix",
                format!("[[{a}, {b}], [{c}, {d}]] is singular, so it cannot be inverted"),
            ));
        }
        Ok(Self {
            matrix,
            options,
            bound: OnceLock::new(),
        })
    }

    /// The matrix, `[a, b, c, d]`.
    #[must_use]
    pub const fn matrix(&self) -> [f64; 4] {
        self.matrix
    }

    /// The options.
    #[must_use]
    pub const fn options(&self) -> AffineOptions {
        self.options
    }

    /// The mapping for a `width` x `height` input.
    fn mapping(&self, width: u32, height: u32) -> Result<Mapping> {
        let [a, b, c, d] = self.matrix;
        let (ix, iy) = self.options.input_offset;
        let (ox, oy) = self.options.output_offset;
        let forward = |x: f64, y: f64| {
            let (x, y) = (x + ix, y + iy);
            (a * x + b * y + ox, c * x + d * y + oy)
        };
        let (w, h) = (f64::from(width), f64::from(height));
        let (mut left, mut right, mut top, mut bottom) = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for (x, y) in [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)] {
            let (x, y) = forward(x, y);
            left = left.min(x);
            right = right.max(x);
            top = top.min(y);
            bottom = bottom.max(y);
        }
        // The canvas starts on the whole pixel at or before the image, from a
        // hair over so rounding error cannot add a column.
        let left = (left + 1e-9).floor();
        let top = (top + 1e-9).floor();
        let output = (side(right - left, "matrix")?, side(bottom - top, "matrix")?);

        let determinant = a * d - b * c;
        Ok(Mapping {
            input: (width, height),
            output,
            inverse: [
                d / determinant,
                -b / determinant,
                -c / determinant,
                a / determinant,
            ],
            before: (left - ox, top - oy),
            after: (-ix, -iy),
        })
    }

    /// Bind to `input`'s size, or check it matches the size already bound.
    fn bind(&self, input: &ImageDescriptor) -> Result<Mapping> {
        if let Some(mapping) = self.bound.get() {
            if mapping.input != (input.width, input.height) {
                return Err(PixelsError::graph(format!(
                    "`affine` is bound to a {}x{} input but was given {}x{}; \
                     build a new Affine per image",
                    mapping.input.0, mapping.input.1, input.width, input.height
                )));
            }
            return Ok(*mapping);
        }
        let mapping = self.mapping(input.width, input.height)?;
        Ok(*self.bound.get_or_init(|| mapping))
    }
}

impl Op for Affine {
    /// A matrix scales with the image, so without offsets the transform means
    /// the same at any resolution and keeps shrink-on-load. The offsets are
    /// in input and output pixels, which a reduced decode would change the
    /// meaning of, so with them it is not rescaled.
    fn rescaled(&self) -> Option<std::sync::Arc<dyn Op>> {
        self.options
            .unshifted()
            .then(|| std::sync::Arc::new(self.clone()) as std::sync::Arc<dyn Op>)
    }

    fn name(&self) -> &'static str {
        "affine"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole(inputs, "affine")?;
        let mapping = self.bind(input)?;
        input.resized(mapping.output.0, mapping.output.1)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let mapping = self.bind(sole(inputs, "affine")?)?;
        Ok(vec![mapping.demand(output)])
    }

    fn access_pattern(&self) -> AccessPattern {
        // Shear and rotation read the input across rows.
        AccessPattern::Spatial
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let mapping = self.bound.get().ok_or_else(|| {
            PixelsError::graph("`affine` ran before its output shape was computed")
        })?;
        resample(
            "affine",
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
    use crate::{Flip, Flop, Quarter, Rotate};
    use otf_pixels_core::{PixelFormat, TileBuf};

    const INTERPOLATORS: [Interpolator; 3] = [
        Interpolator::Nearest,
        Interpolator::Bilinear,
        Interpolator::Bicubic,
    ];

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

    fn affine(matrix: [f64; 4], options: AffineOptions) -> Affine {
        Affine::new(matrix, options).unwrap()
    }

    fn with(interpolator: Interpolator) -> AffineOptions {
        AffineOptions::default().with_interpolator(interpolator)
    }

    fn pattern(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let len = descriptor.byte_len().unwrap();
        let bytes = (0..len).map(|i| (i * 7 % 251) as u8).collect();
        (descriptor, bytes)
    }

    fn size_after(matrix: [f64; 4], options: AffineOptions, w: u32, h: u32) -> (u32, u32) {
        let input = ImageDescriptor::new(w, h, PixelFormat::Rgb8).unwrap();
        let out = affine(matrix, options)
            .output_descriptor(std::slice::from_ref(&input))
            .unwrap();
        (out.width, out.height)
    }

    #[test]
    fn the_canvas_is_the_transformed_bounding_box() {
        let none = AffineOptions::default();
        assert_eq!(size_after([1.0, 0.0, 0.0, 1.0], none, 20, 10), (20, 10));
        assert_eq!(size_after([2.0, 0.0, 0.0, 0.5], none, 20, 10), (40, 5));
        // Horizontal shear: X = x + y/2, so the far corner is at 20 + 5.
        assert_eq!(size_after([1.0, 0.5, 0.0, 1.0], none, 20, 10), (25, 10));
        assert_eq!(size_after([1.0, 0.0, -0.3, 1.0], none, 20, 10), (20, 16));
        // A shrink never reaches zero.
        assert_eq!(size_after([0.01, 0.0, 0.0, 0.01], none, 20, 10), (1, 1));
        // An offset moves the box with the image: a whole pixel changes
        // nothing, a fraction straddles one more.
        let whole = none
            .with_output_offset(3.0, -2.0)
            .with_input_offset(1.0, 0.0);
        assert_eq!(size_after([1.0, 0.0, 0.0, 1.0], whole, 20, 10), (20, 10));
        let half = none.with_output_offset(0.5, 0.0);
        assert_eq!(size_after([1.0, 0.0, 0.0, 1.0], half, 20, 10), (21, 10));
    }

    #[test]
    fn singular_and_non_finite_matrices_are_errors() {
        let options = AffineOptions::default();
        for matrix in [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0, 4.0],
            [f64::NAN, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, f64::INFINITY],
        ] {
            assert!(Affine::new(matrix, options).is_err(), "{matrix:?}");
        }
        let shifted = options.with_input_offset(f64::NAN, 0.0);
        assert!(Affine::new([1.0, 0.0, 0.0, 1.0], shifted).is_err());
        // Tiny but well-conditioned is fine.
        assert!(Affine::new([1e-3, 0.0, 0.0, 1e-3], options).is_ok());
    }

    #[test]
    fn the_identity_and_whole_offsets_are_exact() {
        // Without alpha: a fully transparent pixel resamples to transparent
        // black, whatever colour it had (see `transform`).
        let (desc, bytes) = pattern(9, 7, PixelFormat::Rgb16);
        for interpolator in INTERPOLATORS {
            for options in [
                with(interpolator),
                with(interpolator).with_output_offset(5.0, 2.0),
                with(interpolator).with_input_offset(-1.0, 3.0),
            ] {
                let (out, result) =
                    apply(&affine([1.0, 0.0, 0.0, 1.0], options), &desc, &bytes).unwrap();
                assert_eq!((out.width, out.height), (9, 7));
                assert_eq!(result, bytes, "{options:?}");
            }
        }
    }

    #[test]
    fn mirrors_and_quarter_turns_match_the_layout_ops() {
        // Every output centre lands on an input centre, so every filter is
        // exact: this pins the direction and the centre convention.
        let (desc, bytes) = pattern(8, 5, PixelFormat::Rgb8);
        let cases: [([f64; 4], &dyn Op); 4] = [
            ([-1.0, 0.0, 0.0, 1.0], &Flop),
            ([1.0, 0.0, 0.0, -1.0], &Flip),
            // X = -y, Y = x takes right to down: clockwise.
            ([0.0, -1.0, 1.0, 0.0], &Rotate::new(Quarter::Clockwise90)),
            ([0.0, 1.0, -1.0, 0.0], &Rotate::new(Quarter::Clockwise270)),
        ];
        for (matrix, layout) in cases {
            let (_, expected) = apply(layout, &desc, &bytes).unwrap();
            for interpolator in INTERPOLATORS {
                let (_, got) = apply(&affine(matrix, with(interpolator)), &desc, &bytes).unwrap();
                assert_eq!(got, expected, "{matrix:?} {interpolator:?}");
            }
        }
    }

    #[test]
    fn doubling_with_nearest_repeats_each_pixel() {
        let (desc, bytes) = pattern(4, 3, PixelFormat::Gray8);
        let op = affine([2.0, 0.0, 0.0, 2.0], with(Interpolator::Nearest));
        let (out, doubled) = apply(&op, &desc, &bytes).unwrap();
        assert_eq!((out.width, out.height), (8, 6));
        for y in 0..6 {
            for x in 0..8 {
                assert_eq!(doubled[y * 8 + x], bytes[(y / 2) * 4 + x / 2], "({x}, {y})");
            }
        }
    }

    #[test]
    fn a_half_pixel_offset_averages_neighbours() {
        let desc = ImageDescriptor::new(4, 1, PixelFormat::Gray8).unwrap();
        let bytes = vec![0, 100, 200, 50];
        let options = with(Interpolator::Bilinear)
            .with_output_offset(0.5, 0.0)
            .with_background([10, 10, 10, 255]);
        let (out, shifted) = apply(&affine([1.0, 0.0, 0.0, 1.0], options), &desc, &bytes).unwrap();
        assert_eq!((out.width, out.height), (5, 1));
        assert_eq!(shifted, [5, 50, 150, 125, 30]);
    }

    #[test]
    fn a_shear_slants_a_vertical_line() {
        // A one-pixel column at x = 2 in a 10x10; X = x + y/2 slants it right
        // by half a pixel per row.
        let desc = ImageDescriptor::new(10, 10, PixelFormat::Gray8).unwrap();
        let mut bytes = vec![0_u8; 100];
        for y in 0..10 {
            bytes[y * 10 + 2] = 255;
        }
        let op = affine([1.0, 0.5, 0.0, 1.0], with(Interpolator::Nearest));
        let (out, sheared) = apply(&op, &desc, &bytes).unwrap();
        assert_eq!((out.width, out.height), (15, 10));
        for y in 0..10 {
            let row = &sheared[y * 15..(y + 1) * 15];
            let lit: Vec<usize> = (0..15).filter(|&x| row[x] == 255).collect();
            assert_eq!(lit.len(), 1, "row {y}: {row:?}");
            // Output centre (x + 0.5, y + 0.5) reads input u = x + 0.5 -
            // (y + 0.5) / 2, and nearest lights it when floor(u - 0.5 + 0.5)
            // is the column: x is the first integer at or past 1.75 + y/2.
            let expected = (1.75 + y as f64 / 2.0).ceil() as usize;
            assert_eq!(lit, [expected], "row {y}");
        }
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        let matrices = [
            [1.0, 0.4, 0.0, 1.0],
            [1.7, 0.0, 0.0, 0.6],
            [0.8, -0.5, 0.5, 0.8],
            [-0.6, 0.3, 0.2, 1.3],
        ];
        for &format in &[PixelFormat::Rgb8, PixelFormat::Rgba16, PixelFormat::RgbaF32] {
            for interpolator in INTERPOLATORS {
                for matrix in matrices {
                    let (desc, bytes) = pattern(19, 13, format);
                    let op = affine(
                        matrix,
                        with(interpolator)
                            .with_background([1, 2, 3, 100])
                            .with_input_offset(0.25, -0.5)
                            .with_output_offset(0.3, 0.0),
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
                            "{format} {interpolator:?} {matrix:?} at {tw}x{th} tiles"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_pixel_format_transforms() {
        for &format in PixelFormat::ALL {
            let (desc, bytes) = pattern(7, 5, format);
            let op = affine([1.0, 0.3, 0.1, 0.9], AffineOptions::default());
            let (out, result) = apply(&op, &desc, &bytes).unwrap();
            assert_eq!(out.pixel, format);
            assert_eq!(result.len(), out.byte_len().unwrap(), "{format}");
        }
    }

    #[test]
    fn offsets_stop_it_being_rescaled() {
        let plain = affine([1.0, 0.2, 0.0, 1.0], AffineOptions::default());
        assert!(plain.rescaled().is_some());
        let shifted = affine(
            [1.0, 0.2, 0.0, 1.0],
            AffineOptions::default().with_output_offset(0.5, 0.0),
        );
        assert!(shifted.rescaled().is_none());
    }

    #[test]
    fn an_instance_is_bound_to_one_input_size() {
        let op = affine([1.0, 0.2, 0.0, 1.0], AffineOptions::default());
        let a = ImageDescriptor::new(10, 10, PixelFormat::Rgb8).unwrap();
        let b = ImageDescriptor::new(20, 10, PixelFormat::Rgb8).unwrap();
        op.output_descriptor(std::slice::from_ref(&a)).unwrap();
        assert!(op.output_descriptor(std::slice::from_ref(&b)).is_err());
        op.clone()
            .output_descriptor(std::slice::from_ref(&b))
            .unwrap();
    }
}
