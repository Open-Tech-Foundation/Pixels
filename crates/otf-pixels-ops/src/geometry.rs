//! Geometry ops: [`Crop`], [`Flip`] and [`Flop`].
//!
//! These three are pure **layout** ops: every output pixel is some input pixel,
//! moved. They copy whole pixels as opaque byte runs and never inspect sample
//! values, so one implementation is correct for every pixel format in SPEC
//! §Pixel formats — including formats added later.
//!
//! That is why they do not use [`dispatch_sample!`]: the per-tile
//! monomorphized dispatch of ADR-0002 exists so that *arithmetic* kernels get
//! specialized inner loops, and there is no arithmetic here. Introducing a
//! dispatch would multiply codegen by three to reach identical machine code.
//! Ops that do sample math — `modulate`, `convolve`, `resize` in M4 — are where
//! that dispatch belongs.
//!
//! [`dispatch_sample!`]: otf_pixels_core::dispatch_sample

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Region, Result, Tile, TileMut,
};

/// Fetch the sole input descriptor an op was given.
fn sole_input<'a>(op: &str, inputs: &'a [ImageDescriptor]) -> Result<&'a ImageDescriptor> {
    inputs
        .first()
        .ok_or_else(|| PixelsError::graph(format!("`{op}` takes one input, got none")))
}

/// Fetch the sole input tile an op was given.
fn sole_tile<'a, 'b>(op: &str, inputs: &'a [Tile<'b>]) -> Result<&'a Tile<'b>> {
    inputs
        .first()
        .ok_or_else(|| PixelsError::graph(format!("`{op}` takes one input tile, got none")))
}

/// Extract a rectangular window of an image.
///
/// Crop is a coordinate remap, not a copy: it shifts the origin of the region
/// its input is asked for, so under the M2 scheduler no pixel outside the
/// window is ever decoded or computed. Cropping the corner off a huge image
/// costs the corner, not the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crop {
    window: Region,
}

impl Crop {
    /// Crop to `window`, in input image coordinates.
    ///
    /// The window is validated against the actual input when the op is chained
    /// onto an image, not here, since the input shape is not yet known.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if the window is empty.
    pub fn new(window: Region) -> Result<Self> {
        if window.is_empty() {
            return Err(PixelsError::invalid_argument(
                "window",
                format!("crop window {window} has no pixels"),
            ));
        }
        Ok(Self { window })
    }

    /// Crop to a window given as origin and size.
    ///
    /// # Errors
    ///
    /// As [`Crop::new`].
    pub fn at(x: u32, y: u32, width: u32, height: u32) -> Result<Self> {
        Self::new(Region::new(x, y, width, height))
    }

    /// The window this op extracts, in input coordinates.
    #[must_use]
    pub const fn window(&self) -> Region {
        self.window
    }
}

impl Op for Crop {
    fn name(&self) -> &'static str {
        "crop"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole_input("crop", inputs)?;
        if !input.region().contains(self.window) {
            return Err(PixelsError::invalid_argument(
                "window",
                format!("crop window {} lies outside the {input} input", self.window),
            ));
        }
        input.resized(self.window.width, self.window.height)
    }

    fn input_regions(&self, output: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
        // The output's origin is the window's origin in input space.
        Ok(vec![output.translate(i64::from(self.window.x), i64::from(self.window.y))])
    }

    fn access_pattern(&self) -> AccessPattern {
        // Row order is preserved, so a crop can sit in a streaming segment.
        AccessPattern::Sequential
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = sole_tile("crop", inputs)?;
        let out_region = output.region();
        let bpp = output.pixel().bytes_per_pixel();
        let span = out_region.width as usize * bpp;
        // Output coordinates are window-relative; input coordinates are not.
        let (dx, dy) = (self.window.x, self.window.y);
        for y in out_region.y..out_region.y.saturating_add(out_region.height) {
            let source_y = y.checked_add(dy).ok_or_else(|| {
                PixelsError::invalid_argument("window", "crop origin overflows the image")
            })?;
            let Some(src_row) = input.row(source_y) else {
                return Err(PixelsError::graph(format!(
                    "crop input {} is missing row {source_y}",
                    input.region()
                )));
            };
            let start = (out_region.x.saturating_add(dx) - input.region().x) as usize * bpp;
            let Some(from) = src_row.get(start..start + span) else {
                return Err(PixelsError::graph("crop input row is too short for the window"));
            };
            let Some(into) = output.row_mut(y) else {
                return Err(PixelsError::graph(format!("crop output is missing row {y}")));
            };
            into.copy_from_slice(from);
        }
        Ok(())
    }
}

/// Mirror an image vertically: the top row becomes the bottom row.
///
/// # Access pattern
///
/// `Flip` declares [`AccessPattern::Spatial`], which is the conservative
/// reading of ADR-0003's two-valued declaration rather than a comfortable fit.
/// A vertical mirror is not a neighbourhood op — square tiles buy it nothing —
/// but it is emphatically not `Sequential` either: emitting output row 0
/// requires input row `height - 1`, so it cannot stream in row order over a
/// forward-only source.
///
/// ADR-0001 anticipates exactly this class ("some ops fit the pull model
/// awkwardly and will need explicit materialization points"). Over a
/// [`DecodeCapability::Regions`] source M2 can serve it by pulling bands from
/// the bottom up; over a sequential source it needs a materialization point.
/// Which of those the scheduler picks is an M2 decision, and if it turns out
/// the two-valued enum cannot express it, that is a superseding ADR to
/// ADR-0003 rather than a silent third variant.
///
/// [`DecodeCapability::Regions`]: otf_pixels_core::DecodeCapability::Regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flip;

impl Op for Flip {
    fn name(&self) -> &'static str {
        "flip"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        sole_input("flip", inputs).copied()
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let input = sole_input("flip", inputs)?;
        // Output row y comes from input row (height - 1 - y), so the requested
        // band mirrors about the image's horizontal axis.
        let bottom = u64::from(input.height);
        if output.bottom() > bottom {
            return Err(PixelsError::invalid_argument(
                "output",
                format!("{output} extends past the {input} input"),
            ));
        }
        // Cast is safe: bottom <= input.height, which is a u32.
        let y = (bottom - output.bottom()) as u32;
        Ok(vec![Region::new(output.x, y, output.width, output.height)])
    }

    fn access_pattern(&self) -> AccessPattern {
        // A vertical mirror needs the *last* input row to emit the first output
        // row, so it cannot stream in row order like a pointwise op can.
        AccessPattern::Spatial
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = sole_tile("flip", inputs)?;
        let (out_region, in_region) = (output.region(), input.region());
        if out_region.height != in_region.height || out_region.width != in_region.width {
            return Err(PixelsError::graph(format!(
                "flip input {in_region} does not match output {out_region}"
            )));
        }
        for offset in 0..out_region.height {
            let out_y = out_region.y + offset;
            // Mirror within the band: last input row feeds the first output row.
            let in_y = in_region.y + (in_region.height - 1 - offset);
            let Some(from) = input.row(in_y) else {
                return Err(PixelsError::graph(format!("flip input is missing row {in_y}")));
            };
            let Some(into) = output.row_mut(out_y) else {
                return Err(PixelsError::graph(format!("flip output is missing row {out_y}")));
            };
            into.copy_from_slice(from);
        }
        Ok(())
    }
}

/// Mirror an image horizontally: the left column becomes the right column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Flop;

impl Op for Flop {
    fn name(&self) -> &'static str {
        "flop"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        sole_input("flop", inputs).copied()
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let input = sole_input("flop", inputs)?;
        let right = u64::from(input.width);
        if output.right() > right {
            return Err(PixelsError::invalid_argument(
                "output",
                format!("{output} extends past the {input} input"),
            ));
        }
        // Cast is safe: right <= input.width, which is a u32.
        let x = (right - output.right()) as u32;
        Ok(vec![Region::new(x, output.y, output.width, output.height)])
    }

    fn access_pattern(&self) -> AccessPattern {
        // Each output row depends only on the corresponding input row, so a
        // horizontal mirror still streams in row order.
        AccessPattern::Sequential
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = sole_tile("flop", inputs)?;
        let (out_region, in_region) = (output.region(), input.region());
        if out_region.width != in_region.width || out_region.height != in_region.height {
            return Err(PixelsError::graph(format!(
                "flop input {in_region} does not match output {out_region}"
            )));
        }
        let bpp = output.pixel().bytes_per_pixel();
        let width = out_region.width as usize;
        for offset in 0..out_region.height {
            let Some(from) = input.row(in_region.y + offset) else {
                return Err(PixelsError::graph("flop input is missing a row"));
            };
            let Some(into) = output.row_mut(out_region.y + offset) else {
                return Err(PixelsError::graph("flop output is missing a row"));
            };
            // Reverse whole pixels, not bytes: the samples within a pixel keep
            // their channel order.
            for x in 0..width {
                let src = x * bpp;
                let dst = (width - 1 - x) * bpp;
                let (Some(pixel), Some(slot)) =
                    (from.get(src..src + bpp), into.get_mut(dst..dst + bpp))
                else {
                    return Err(PixelsError::graph("flop row is too short"));
                };
                slot.copy_from_slice(pixel);
            }
        }
        Ok(())
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
    use otf_pixels_core::{
        BufferSource, ErrorCode, Format, Image, PixelFormat, Producer, TileBuf, evaluate,
    };
    use std::sync::Arc;

    /// A `width` x `height` Gray8 image whose pixel values are `y * width + x`.
    fn ramp(width: u32, height: u32) -> Image {
        ramp_with(width, height, PixelFormat::Gray8)
    }

    fn ramp_with(width: u32, height: u32, pixel: PixelFormat) -> Image {
        let desc = ImageDescriptor::new(width, height, pixel).unwrap();
        let len = desc.byte_len().unwrap();
        let bytes: Vec<u8> = (0..len).map(|i| i as u8).collect();
        let buffer = TileBuf::from_vec(desc.region(), pixel, bytes).unwrap();
        let source = BufferSource::new(desc, Arc::new(buffer)).unwrap();
        Image::from_producer(Arc::new(source) as Arc<dyn Producer>, Format::Raw)
    }

    #[test]
    fn crop_extracts_the_window() {
        // 4x4 ramp: rows are 0..3, 4..7, 8..11, 12..15.
        let image = ramp(4, 4).apply(Arc::new(Crop::at(1, 1, 2, 2).unwrap())).unwrap();
        assert_eq!(image.descriptor().width, 2);
        assert_eq!(image.descriptor().height, 2);
        assert_eq!(evaluate(&image).unwrap().bytes(), &[5, 6, 9, 10]);
    }

    #[test]
    fn crop_at_the_origin_and_the_far_corner() {
        let top_left = ramp(4, 4).apply(Arc::new(Crop::at(0, 0, 2, 2).unwrap())).unwrap();
        assert_eq!(evaluate(&top_left).unwrap().bytes(), &[0, 1, 4, 5]);
        let bottom_right = ramp(4, 4).apply(Arc::new(Crop::at(2, 2, 2, 2).unwrap())).unwrap();
        assert_eq!(evaluate(&bottom_right).unwrap().bytes(), &[10, 11, 14, 15]);
        // The whole image is a legal window.
        let whole = ramp(2, 2).apply(Arc::new(Crop::at(0, 0, 2, 2).unwrap())).unwrap();
        assert_eq!(evaluate(&whole).unwrap().bytes(), &[0, 1, 2, 3]);
    }

    #[test]
    fn a_window_outside_the_image_is_rejected_at_build_time() {
        let err = ramp(4, 4).apply(Arc::new(Crop::at(3, 3, 2, 2).unwrap())).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
        let err = ramp(4, 4).apply(Arc::new(Crop::at(5, 0, 1, 1).unwrap())).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn an_empty_window_is_rejected_at_construction() {
        assert_eq!(Crop::at(0, 0, 0, 4).unwrap_err().code(), ErrorCode::InvalidArgument);
        assert_eq!(Crop::at(0, 0, 4, 0).unwrap_err().code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn crop_demands_only_its_window() {
        // The point of crop: demand propagation asks for the window, not the
        // whole image, so upstream work is proportional to the output.
        let crop = Crop::at(10, 20, 4, 4).unwrap();
        let input = ImageDescriptor::new(100, 100, PixelFormat::Gray8).unwrap();
        let regions = crop.input_regions(Region::from_size(4, 4), &[input]).unwrap();
        assert_eq!(regions, vec![Region::new(10, 20, 4, 4)]);
    }

    #[test]
    fn flip_mirrors_rows() {
        // 2x3 ramp: rows are [0,1], [2,3], [4,5].
        let image = ramp(2, 3).apply(Arc::new(Flip)).unwrap();
        assert_eq!(evaluate(&image).unwrap().bytes(), &[4, 5, 2, 3, 0, 1]);
    }

    #[test]
    fn flop_mirrors_columns() {
        // 3x2 ramp: rows are [0,1,2], [3,4,5].
        let image = ramp(3, 2).apply(Arc::new(Flop)).unwrap();
        assert_eq!(evaluate(&image).unwrap().bytes(), &[2, 1, 0, 5, 4, 3]);
    }

    #[test]
    fn flop_reverses_pixels_not_bytes() {
        // 2x1 RGB8: pixels are (0,1,2) and (3,4,5). Reversing bytes would give
        // 5,4,3,2,1,0 — reversing pixels gives 3,4,5,0,1,2.
        let image = ramp_with(2, 1, PixelFormat::Rgb8).apply(Arc::new(Flop)).unwrap();
        assert_eq!(evaluate(&image).unwrap().bytes(), &[3, 4, 5, 0, 1, 2]);
    }

    #[test]
    fn flip_preserves_pixels_within_rows() {
        // 2x2 RGBA8: flip swaps rows wholesale, leaving each row's bytes intact.
        let image = ramp_with(2, 2, PixelFormat::Rgba8).apply(Arc::new(Flip)).unwrap();
        assert_eq!(
            evaluate(&image).unwrap().bytes(),
            &[8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn flip_and_flop_are_their_own_inverses() {
        for pixel in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba16] {
            let original = evaluate(&ramp_with(3, 4, pixel)).unwrap();
            let flipped = ramp_with(3, 4, pixel)
                .apply(Arc::new(Flip))
                .unwrap()
                .apply(Arc::new(Flip))
                .unwrap();
            assert_eq!(evaluate(&flipped).unwrap().bytes(), original.bytes(), "flip² for {pixel}");
            let flopped = ramp_with(3, 4, pixel)
                .apply(Arc::new(Flop))
                .unwrap()
                .apply(Arc::new(Flop))
                .unwrap();
            assert_eq!(evaluate(&flopped).unwrap().bytes(), original.bytes(), "flop² for {pixel}");
        }
    }

    #[test]
    fn flip_and_flop_preserve_dimensions_and_format() {
        for pixel in PixelFormat::ALL {
            let image = ramp_with(3, 5, *pixel).apply(Arc::new(Flip)).unwrap();
            assert_eq!(image.descriptor().width, 3, "{pixel}");
            assert_eq!(image.descriptor().height, 5, "{pixel}");
            assert_eq!(image.descriptor().pixel, *pixel);
        }
    }

    #[test]
    fn flip_demand_mirrors_the_requested_band() {
        let input = ImageDescriptor::new(4, 10, PixelFormat::Gray8).unwrap();
        // The top two output rows come from the bottom two input rows.
        let regions = Flip.input_regions(Region::new(0, 0, 4, 2), &[input]).unwrap();
        assert_eq!(regions, vec![Region::new(0, 8, 4, 2)]);
        // A middle band mirrors about the centre.
        let regions = Flip.input_regions(Region::new(0, 3, 4, 2), &[input]).unwrap();
        assert_eq!(regions, vec![Region::new(0, 5, 4, 2)]);
    }

    #[test]
    fn flop_demand_mirrors_the_requested_band() {
        let input = ImageDescriptor::new(10, 4, PixelFormat::Gray8).unwrap();
        let regions = Flop.input_regions(Region::new(0, 0, 2, 4), &[input]).unwrap();
        assert_eq!(regions, vec![Region::new(8, 0, 2, 4)]);
    }

    #[test]
    fn demand_beyond_the_input_is_an_error_not_a_wrap() {
        let input = ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap();
        let err = Flip.input_regions(Region::new(0, 0, 4, 8), &[input]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
        let err = Flop.input_regions(Region::new(0, 0, 8, 4), &[input]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn ops_declare_their_access_patterns() {
        // ADR-0003: these declarations drive tile negotiation in M2.
        assert_eq!(Crop::at(0, 0, 1, 1).unwrap().access_pattern(), AccessPattern::Sequential);
        assert_eq!(Flop.access_pattern(), AccessPattern::Sequential);
        // A vertical mirror needs the last input row first, so it is not
        // row-streamable.
        assert_eq!(Flip.access_pattern(), AccessPattern::Spatial);
    }

    #[test]
    fn ops_reject_being_called_with_no_input() {
        assert!(Flip.output_descriptor(&[]).is_err());
        assert!(Flop.output_descriptor(&[]).is_err());
        assert!(Crop::at(0, 0, 1, 1).unwrap().output_descriptor(&[]).is_err());
        assert!(Flip.compute(&[], &mut TileBuf::zeroed(
            Region::from_size(1, 1),
            PixelFormat::Gray8
        )
        .unwrap()
        .as_tile_mut()
        .unwrap())
        .is_err());
    }

    #[test]
    fn geometry_ops_compose() {
        // crop then flip then flop, on a 4x4 ramp.
        let image = ramp(4, 4)
            .apply(Arc::new(Crop::at(1, 1, 2, 2).unwrap()))
            .unwrap()
            .apply(Arc::new(Flip))
            .unwrap()
            .apply(Arc::new(Flop))
            .unwrap();
        // Window is [[5,6],[9,10]]; flip -> [[9,10],[5,6]]; flop -> [[10,9],[6,5]].
        assert_eq!(evaluate(&image).unwrap().bytes(), &[10, 9, 6, 5]);
    }
}
