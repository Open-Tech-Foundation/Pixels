//! [`Rotate`] — quarter-turn rotation.
//!
//! Like [`Crop`], [`Flip`] and [`Flop`], this is a pure layout op: every output
//! pixel is some input pixel, moved. It copies whole pixels as opaque byte runs
//! and never inspects sample values, so one implementation covers every pixel
//! format including ones added later.
//!
//! [`Crop`]: crate::Crop
//! [`Flip`]: crate::Flip
//! [`Flop`]: crate::Flop
//!
//! # Only quarter turns
//!
//! SPEC §Core ops says multiples of 90 degrees, and that restriction is what
//! keeps this exact. An arbitrary angle needs resampling, which means a filter,
//! an edge policy and an output-size convention — three decisions that belong
//! to a `rotate_arbitrary` op rather than being smuggled in here.

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Region, Result, Tile, TileMut,
};

/// A quarter-turn rotation, clockwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Quarter {
    /// No rotation.
    #[default]
    None,
    /// 90 degrees clockwise.
    Clockwise90,
    /// 180 degrees.
    Half,
    /// 270 degrees clockwise, i.e. 90 anticlockwise.
    Clockwise270,
}

impl Quarter {
    /// The rotation for `degrees`, which must be a multiple of 90.
    ///
    /// Negative and out-of-range angles are normalized, so -90 and 270 are the
    /// same rotation.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `degrees` is not a multiple
    /// of 90. Rounding to the nearest quarter turn would silently rotate an
    /// image differently from what was asked.
    pub fn from_degrees(degrees: i32) -> Result<Self> {
        if degrees % 90 != 0 {
            return Err(PixelsError::invalid_argument(
                "degrees",
                format!("rotation must be a multiple of 90, got {degrees}"),
            ));
        }
        Ok(match degrees.rem_euclid(360) / 90 {
            1 => Self::Clockwise90,
            2 => Self::Half,
            3 => Self::Clockwise270,
            _ => Self::None,
        })
    }

    /// Whether this rotation exchanges width and height.
    #[must_use]
    pub const fn transposes(self) -> bool {
        matches!(self, Self::Clockwise90 | Self::Clockwise270)
    }

    /// The rotation that undoes this one.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Clockwise90 => Self::Clockwise270,
            Self::Half => Self::Half,
            Self::Clockwise270 => Self::Clockwise90,
        }
    }
}

/// Rotate an image by a multiple of 90 degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Rotate {
    quarter: Quarter,
}

impl Rotate {
    /// Rotate by `quarter`.
    #[must_use]
    pub const fn new(quarter: Quarter) -> Self {
        Self { quarter }
    }

    /// Rotate by `degrees`, which must be a multiple of 90.
    ///
    /// # Errors
    ///
    /// As [`Quarter::from_degrees`].
    pub fn degrees(degrees: i32) -> Result<Self> {
        Ok(Self::new(Quarter::from_degrees(degrees)?))
    }

    /// The rotation this op applies.
    #[must_use]
    pub const fn quarter(&self) -> Quarter {
        self.quarter
    }

    /// Map an output coordinate back to the input coordinate it came from.
    ///
    /// `input` is the size of the *input* image, which the mapping needs
    /// because a rotation reflects across an axis whose position depends on it.
    const fn source_of(&self, x: u32, y: u32, input_width: u32, input_height: u32) -> (u32, u32) {
        match self.quarter {
            Quarter::None => (x, y),
            // Clockwise 90: output (x, y) comes from input (x', y') where the
            // top-left of the output is the bottom-left of the input.
            Quarter::Clockwise90 => (y, input_height.saturating_sub(1).saturating_sub(x)),
            Quarter::Half => (
                input_width.saturating_sub(1).saturating_sub(x),
                input_height.saturating_sub(1).saturating_sub(y),
            ),
            Quarter::Clockwise270 => (input_width.saturating_sub(1).saturating_sub(y), x),
        }
    }
}

impl Op for Rotate {
    fn name(&self) -> &'static str {
        "rotate"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`rotate` takes one input, got none"))?;
        if self.quarter.transposes() {
            input.resized(input.height, input.width)
        } else {
            Ok(*input)
        }
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`rotate` takes one input, got none"))?;

        // The rotated image of a rectangle is a rectangle, so the demand is
        // exact rather than a bounding box: map the two opposite corners and
        // normalize. Nothing outside the output tile is ever requested.
        let (x0, y0) = self.source_of(output.x, output.y, input.width, input.height);
        let last_x = output.x + output.width.saturating_sub(1);
        let last_y = output.y + output.height.saturating_sub(1);
        let (x1, y1) = self.source_of(last_x, last_y, input.width, input.height);

        let (left, right) = (x0.min(x1), x0.max(x1));
        let (top, bottom) = (y0.min(y1), y0.max(y1));
        Ok(vec![Region::new(
            left,
            top,
            right - left + 1,
            bottom - top + 1,
        )])
    }

    fn access_pattern(&self) -> AccessPattern {
        // A quarter turn reads a column per output row, so a full-width strip
        // would demand the whole input height. Square tiles keep the working
        // set to the tile itself — the same reasoning as ADR-0003's spatial
        // case, and the reason `Flip` is *not* spatial: a mirror still reads
        // one input row per output row.
        if self.quarter.transposes() {
            AccessPattern::Spatial
        } else {
            AccessPattern::Sequential
        }
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

        let bytes = output.pixel().bytes_per_pixel();
        let region = output.region();
        let source = input.region();

        // The input tile covers `source`, which is the demand computed above.
        // Reconstructing the whole-image size from it is what lets the same
        // mapping work per tile as for the whole image.
        let (input_width, input_height) = whole_input_size(self.quarter, region, source);

        for y in region.y..region.y + region.height {
            // Collected first, then written: `row_mut` borrows the output
            // mutably, so the input row lookups have to finish before it.
            let mut row_bytes = Vec::with_capacity(region.width as usize * bytes);
            for x in region.x..region.x + region.width {
                let (sx, sy) = self.source_of(x, y, input_width, input_height);
                let Some(from) = input.row(sy) else {
                    row_bytes.resize(row_bytes.len() + bytes, 0);
                    continue;
                };
                let at = (sx.saturating_sub(source.x) as usize) * bytes;
                match from.get(at..at + bytes) {
                    Some(pixel) => row_bytes.extend_from_slice(pixel),
                    None => row_bytes.resize(row_bytes.len() + bytes, 0),
                }
            }
            if let Some(target) = output.row_mut(y) {
                let len = target.len().min(row_bytes.len());
                if let (Some(to), Some(from)) = (target.get_mut(..len), row_bytes.get(..len)) {
                    to.copy_from_slice(from);
                }
            }
        }
        Ok(())
    }
}

/// Recover the input image's size from an output tile and the region it maps to.
///
/// The rotation mapping needs the *image's* dimensions, not the tile's, because
/// it reflects across an axis at the image edge. The scheduler hands over only
/// regions, so the size is reconstructed from the two together — which is exact
/// because `input_regions` computed the region from that same size.
const fn whole_input_size(quarter: Quarter, output: Region, source: Region) -> (u32, u32) {
    match quarter {
        Quarter::None => (source.x + source.width, source.y + source.height),
        Quarter::Clockwise90 => (
            source.x + source.width,
            // x was reflected: source.y = height - 1 - (output.x + width - 1).
            source.y + source.height + output.x,
        ),
        Quarter::Half => (
            source.x + source.width + output.x,
            source.y + source.height + output.y,
        ),
        Quarter::Clockwise270 => (source.x + source.width + output.y, source.y + source.height),
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
    use otf_pixels_core::{PixelFormat, TileBuf};

    /// Run an op over a whole image.
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

    /// An image whose every pixel encodes its own coordinates.
    fn coded(width: u32, height: u32) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();
        let mut bytes = Vec::new();
        for y in 0..height {
            for x in 0..width {
                bytes.extend_from_slice(&[x as u8, y as u8, 0]);
            }
        }
        (descriptor, bytes)
    }

    #[test]
    fn degrees_normalize_to_quarter_turns() {
        assert_eq!(Quarter::from_degrees(0).unwrap(), Quarter::None);
        assert_eq!(Quarter::from_degrees(90).unwrap(), Quarter::Clockwise90);
        assert_eq!(Quarter::from_degrees(180).unwrap(), Quarter::Half);
        assert_eq!(Quarter::from_degrees(270).unwrap(), Quarter::Clockwise270);
        assert_eq!(Quarter::from_degrees(360).unwrap(), Quarter::None);
        assert_eq!(Quarter::from_degrees(-90).unwrap(), Quarter::Clockwise270);
        assert_eq!(Quarter::from_degrees(450).unwrap(), Quarter::Clockwise90);
    }

    #[test]
    fn an_angle_that_is_not_a_quarter_turn_is_an_error() {
        // Rounding to the nearest quarter would rotate the image differently
        // from what was asked, silently.
        for degrees in [1, 45, 89, -30, 100] {
            assert!(
                Quarter::from_degrees(degrees).is_err(),
                "{degrees} should be rejected"
            );
        }
    }

    #[test]
    fn rotating_transposes_the_shape_only_for_quarter_turns() {
        let input = ImageDescriptor::new(30, 20, PixelFormat::Rgb8).unwrap();
        for (quarter, expected) in [
            (Quarter::None, (30, 20)),
            (Quarter::Clockwise90, (20, 30)),
            (Quarter::Half, (30, 20)),
            (Quarter::Clockwise270, (20, 30)),
        ] {
            let out = Rotate::new(quarter)
                .output_descriptor(std::slice::from_ref(&input))
                .unwrap();
            assert_eq!((out.width, out.height), expected, "{quarter:?}");
        }
    }

    #[test]
    fn four_quarter_turns_return_the_original() {
        // The cheapest complete check on the coordinate mapping: any error in
        // any single turn fails to cancel over four.
        let (desc, bytes) = coded(7, 5);
        let mut current = (desc, bytes.clone());
        for _ in 0..4 {
            current = apply(&Rotate::new(Quarter::Clockwise90), &current.0, &current.1).unwrap();
        }
        assert_eq!(current.0.width, 7);
        assert_eq!(current.0.height, 5);
        assert_eq!(current.1, bytes, "four turns did not return the original");
    }

    #[test]
    fn a_rotation_and_its_inverse_cancel() {
        let (desc, bytes) = coded(9, 4);
        for quarter in [
            Quarter::None,
            Quarter::Clockwise90,
            Quarter::Half,
            Quarter::Clockwise270,
        ] {
            let (mid_desc, mid) = apply(&Rotate::new(quarter), &desc, &bytes).unwrap();
            let (back_desc, back) =
                apply(&Rotate::new(quarter.inverse()), &mid_desc, &mid).unwrap();
            assert_eq!((back_desc.width, back_desc.height), (9, 4), "{quarter:?}");
            assert_eq!(back, bytes, "{quarter:?} did not cancel with its inverse");
        }
    }

    #[test]
    fn clockwise_ninety_moves_the_corners_where_it_should() {
        // The sign of a rotation is exactly the thing that is easy to get
        // backwards and hard to notice, so pin one corner explicitly.
        let (desc, bytes) = coded(4, 3);
        let (out_desc, out) = apply(&Rotate::new(Quarter::Clockwise90), &desc, &bytes).unwrap();
        assert_eq!((out_desc.width, out_desc.height), (3, 4));

        // Clockwise: the input's bottom-left corner becomes the output's
        // top-left.
        let top_left = &out[0..3];
        assert_eq!(top_left, [0, 2, 0], "expected input (0,2) at output (0,0)");
    }

    #[test]
    fn half_turn_is_flip_composed_with_flop() {
        let (desc, bytes) = coded(6, 5);
        let (_, rotated) = apply(&Rotate::new(Quarter::Half), &desc, &bytes).unwrap();
        let (mid_desc, flipped) = apply(&crate::Flip, &desc, &bytes).unwrap();
        let (_, both) = apply(&crate::Flop, &mid_desc, &flipped).unwrap();
        assert_eq!(rotated, both, "180 degrees is not flip then flop");
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        // The same guarantee resize carries: a rotation must give the same
        // pixels whether it runs in one tile or in many, or SPEC §Guarantees 2
        // is false. This is also what exercises `whole_input_size`.
        for quarter in [
            Quarter::None,
            Quarter::Clockwise90,
            Quarter::Half,
            Quarter::Clockwise270,
        ] {
            let (desc, bytes) = coded(13, 11);
            let op = Rotate::new(quarter);
            let (out_desc, whole) = apply(&op, &desc, &bytes).unwrap();
            let source = TileBuf::from_vec(desc.region(), desc.pixel, bytes.clone()).unwrap();
            let mut target = TileBuf::for_image(&out_desc).unwrap();

            for (tw, th) in [(4_u32, 4_u32), (1, 11), (13, 1), (5, 3)] {
                let mut y = 0;
                while y < out_desc.height {
                    let h = th.min(out_desc.height - y);
                    let mut x = 0;
                    while x < out_desc.width {
                        let w = tw.min(out_desc.width - x);
                        let region = Region::new(x, y, w, h);
                        let demand = op
                            .input_regions(region, std::slice::from_ref(&desc))
                            .unwrap();
                        let mut cut = TileBuf::zeroed(demand[0], desc.pixel).unwrap();
                        otf_pixels_core::copy_region(
                            &source.as_tile().unwrap(),
                            &mut cut.as_tile_mut().unwrap(),
                            demand[0],
                        )
                        .unwrap();
                        let mut sub = TileBuf::zeroed(region, out_desc.pixel).unwrap();
                        op.compute(&[cut.as_tile().unwrap()], &mut sub.as_tile_mut().unwrap())
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
                    "{quarter:?} at {tw}x{th} tiles differed from the whole-image result"
                );
            }
        }
    }

    #[test]
    fn demand_never_reaches_outside_the_input() {
        let input = ImageDescriptor::new(17, 9, PixelFormat::Rgb8).unwrap();
        for quarter in [
            Quarter::None,
            Quarter::Clockwise90,
            Quarter::Half,
            Quarter::Clockwise270,
        ] {
            let op = Rotate::new(quarter);
            let out = op.output_descriptor(std::slice::from_ref(&input)).unwrap();
            for y in 0..out.height {
                for x in 0..out.width {
                    let demand = op
                        .input_regions(Region::new(x, y, 1, 1), std::slice::from_ref(&input))
                        .unwrap();
                    let r = demand[0];
                    assert!(
                        r.x + r.width <= input.width && r.y + r.height <= input.height,
                        "{quarter:?} demand {r} leaves a {}x{} input",
                        input.width,
                        input.height
                    );
                }
            }
        }
    }

    #[test]
    fn rotation_works_for_every_pixel_format() {
        // A layout op copies opaque byte runs, so this should hold for every
        // format including ones added later — that is the claim being tested.
        for &format in PixelFormat::ALL {
            let descriptor = ImageDescriptor::new(5, 3, format).unwrap();
            let len = descriptor.byte_len().unwrap();
            let bytes: Vec<u8> = (0..len).map(|i| (i % 253) as u8).collect();
            let op = Rotate::new(Quarter::Clockwise90);
            let (mid_desc, mid) = apply(&op, &descriptor, &bytes).unwrap();
            let (_, back) = apply(&Rotate::new(Quarter::Clockwise270), &mid_desc, &mid).unwrap();
            assert_eq!(
                back, bytes,
                "{format} did not round-trip through a rotation"
            );
        }
    }
}
