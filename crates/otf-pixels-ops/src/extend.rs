//! [`Extend`] — grow the canvas by a margin on each side (sharp's `extend`).
//!
//! A layout op like [`crate::Crop`]: every output pixel is either an input
//! pixel, moved, or the background colour. Whole pixels are copied as opaque
//! byte runs, so one implementation serves every pixel format.
//!
//! # The four ways to fill a margin
//!
//! sharp's `extendWith`, which is libvips' `extend`:
//!
//! - [`ExtendWith::Background`] paints the margin one colour.
//! - [`ExtendWith::Copy`] repeats the edge pixel outwards.
//! - [`ExtendWith::Repeat`] tiles the image, so the left margin shows the
//!   image's right edge.
//! - [`ExtendWith::Mirror`] tiles it reflected, the edge pixel appearing
//!   twice: one pixel left of the image is its first column, not its second.

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Region, Result, Tile, TileMut,
};

use std::sync::OnceLock;

use crate::resize::background_pixel;

/// What fills the margin [`Extend`] adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ExtendWith {
    /// One colour, [`Extend::with_background`]'s.
    #[default]
    Background,
    /// The nearest edge pixel, repeated outwards.
    Copy,
    /// The image tiled.
    Repeat,
    /// The image tiled and reflected, so each copy meets its neighbour at a
    /// mirrored edge.
    Mirror,
}

impl ExtendWith {
    /// The input coordinate an output coordinate `at` takes its pixel from,
    /// on an axis where the image starts at `before` and is `size` long;
    /// `None` for the background.
    fn source(self, at: u32, before: u32, size: u32) -> Option<u32> {
        let i = i64::from(at) - i64::from(before);
        let n = i64::from(size);
        let mapped = match self {
            Self::Background => (0..n).contains(&i).then_some(i)?,
            Self::Copy => i.clamp(0, n - 1),
            Self::Repeat => i.rem_euclid(n),
            Self::Mirror => {
                let m = i.rem_euclid(2 * n);
                if m < n { m } else { 2 * n - 1 - m }
            }
        };
        u32::try_from(mapped).ok()
    }

    /// The input span `start..end` that output span `from..to` reads, or
    /// `None` if it is all background.
    fn span(self, from: u32, to: u32, before: u32, size: u32) -> Option<(u32, u32)> {
        // A span inside the image maps to itself. Only one touching a margin
        // needs the walk below, which is a tile's width long.
        if from >= before && to <= before.saturating_add(size) {
            return Some((from - before, to - before));
        }
        let mut low = u32::MAX;
        let mut high = 0;
        for at in from..to {
            if let Some(source) = self.source(at, before, size) {
                low = low.min(source);
                high = high.max(source + 1);
            }
        }
        (low < high).then_some((low, high))
    }
}

/// Add `top`, `right`, `bottom` and `left` pixels around an image.
///
/// Like [`crate::Resize`], an instance belongs to one graph edge: copy,
/// repeat and mirror wrap around the input's size, which is bound when the
/// op is chained, and chaining the same instance onto a differently sized
/// image is a [`PixelsError::Graph`].
#[derive(Debug)]
pub struct Extend {
    top: u32,
    right: u32,
    bottom: u32,
    left: u32,
    with: ExtendWith,
    background: [u8; 4],
    /// The input's width and height, filled by `output_descriptor`.
    bound: OnceLock<(u32, u32)>,
}

impl Clone for Extend {
    fn clone(&self) -> Self {
        // Not the binding: a clone may be chained onto another image.
        Self {
            bound: OnceLock::new(),
            ..*self
        }
    }
}

impl Extend {
    /// Margins of `top`, `right`, `bottom` and `left` pixels, filled with
    /// opaque black, sharp's default.
    #[must_use]
    pub const fn new(top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
            with: ExtendWith::Background,
            background: [0, 0, 0, 255],
            bound: OnceLock::new(),
        }
    }

    /// The same margin on every side (sharp's `extend(n)`).
    #[must_use]
    pub const fn uniform(margin: u32) -> Self {
        Self::new(margin, margin, margin, margin)
    }

    /// Fill the margin by `with` instead of the background colour.
    #[must_use]
    pub const fn extend_with(mut self, with: ExtendWith) -> Self {
        self.with = with;
        self
    }

    /// The RGBA colour [`ExtendWith::Background`] paints. Grey formats take
    /// its luma, and formats without alpha ignore its alpha.
    #[must_use]
    pub const fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// The margins, as `(top, right, bottom, left)`.
    #[must_use]
    pub const fn margins(&self) -> (u32, u32, u32, u32) {
        (self.top, self.right, self.bottom, self.left)
    }

    /// How the margin is filled.
    #[must_use]
    pub const fn method(&self) -> ExtendWith {
        self.with
    }

    /// The background colour.
    #[must_use]
    pub const fn background(&self) -> [u8; 4] {
        self.background
    }

    /// Bind to `input`'s size, or check it matches the size already bound.
    fn bind(&self, input: &ImageDescriptor) -> Result<(u32, u32)> {
        let size = *self.bound.get_or_init(|| (input.width, input.height));
        if size != (input.width, input.height) {
            return Err(PixelsError::graph(format!(
                "`extend` is bound to a {}x{} input but was given {}x{}; \
                 build a new Extend per image",
                size.0, size.1, input.width, input.height
            )));
        }
        Ok(size)
    }
}

/// Fetch the sole input descriptor.
fn sole(inputs: &[ImageDescriptor]) -> Result<&ImageDescriptor> {
    inputs
        .first()
        .ok_or_else(|| PixelsError::graph("`extend` takes one input, got none"))
}

impl Op for Extend {
    /// Never rescaled: the margins are in output pixels, and a margin of 10
    /// around a source decoded eight times smaller would be 80 at full size.
    fn rescaled(&self) -> Option<std::sync::Arc<dyn Op>> {
        None
    }

    fn name(&self) -> &'static str {
        "extend"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole(inputs)?;
        self.bind(input)?;
        let grow = |size: u32, a: u32, b: u32| {
            size.checked_add(a)
                .and_then(|s| s.checked_add(b))
                .ok_or_else(|| {
                    PixelsError::invalid_argument("margin", "the extended image is too large")
                })
        };
        input.resized(
            grow(input.width, self.left, self.right)?,
            grow(input.height, self.top, self.bottom)?,
        )
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let (width, height) = self.bind(sole(inputs)?)?;
        let columns = self
            .with
            .span(output.x, output.x + output.width, self.left, width);
        let rows = self
            .with
            .span(output.y, output.y + output.height, self.top, height);
        Ok(vec![match (columns, rows) {
            (Some((x0, x1)), Some((y0, y1))) => Region::new(x0, y0, x1 - x0, y1 - y0),
            // A tile of pure margin still has to name a valid region, so it
            // asks for one pixel it will not read.
            _ => Region::new(0, 0, 1, 1),
        }])
    }

    fn access_pattern(&self) -> AccessPattern {
        // One input row per output row, as for crop.
        AccessPattern::Sequential
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`extend` takes one input tile, got none"))?;
        let (width, height) = self
            .bound
            .get()
            .copied()
            .ok_or_else(|| PixelsError::graph("`extend` computed before it was chained"))?;
        let bpp = output.pixel().bytes_per_pixel();
        let fill = background_pixel(output.pixel(), self.background);
        let region = output.region();
        let area = input.region();

        // Where each output column reads from, as a byte offset into an input
        // row, or `None` for background. The same for every row of the tile.
        let columns: Vec<Option<usize>> = (region.x..region.x + region.width)
            .map(|x| {
                self.with
                    .source(x, self.left, width)
                    .map(|sx| (sx - area.x) as usize * bpp)
            })
            .collect();

        // A tile whose columns are all margin was given a placeholder input,
        // and reads no row of it.
        let reads = columns.iter().any(Option::is_some);

        for y in region.y..region.y + region.height {
            let source = match self.with.source(y, self.top, height).filter(|_| reads) {
                Some(sy) => Some(input.row(sy).ok_or_else(|| {
                    PixelsError::graph(format!("extend input {area} is missing row {sy}"))
                })?),
                None => None,
            };
            let target = output
                .row_mut(y)
                .ok_or_else(|| PixelsError::graph(format!("extend output is missing row {y}")))?;
            // Columns that read consecutive input pixels are copied as one
            // run, which is the whole image part of every row.
            let mut i = 0;
            while i < columns.len() {
                let start = columns.get(i).copied().flatten();
                let (Some(row), Some(start)) = (source, start) else {
                    if let Some(slot) = target.get_mut(i * bpp..(i + 1) * bpp) {
                        slot.copy_from_slice(&fill);
                    }
                    i += 1;
                    continue;
                };
                let mut end = i + 1;
                while columns.get(end).copied().flatten() == Some(start + (end - i) * bpp) {
                    end += 1;
                }
                let span = (end - i) * bpp;
                let (Some(from), Some(into)) = (
                    row.get(start..start + span),
                    target.get_mut(i * bpp..i * bpp + span),
                ) else {
                    return Err(PixelsError::graph(
                        "extend input row is too short for its tile",
                    ));
                };
                into.copy_from_slice(from);
                i = end;
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
    use otf_pixels_core::{PixelFormat, TileBuf, copy_region};

    /// Extend a whole image.
    fn apply(op: &Extend, descriptor: ImageDescriptor, bytes: &[u8]) -> (ImageDescriptor, Vec<u8>) {
        let out = op.output_descriptor(&[descriptor]).unwrap();
        let source =
            TileBuf::from_vec(descriptor.region(), descriptor.pixel, bytes.to_vec()).unwrap();
        let demand = op.input_regions(out.region(), &[descriptor]).unwrap();
        let mut cut = TileBuf::zeroed(demand[0], descriptor.pixel).unwrap();
        copy_region(
            &source.as_tile().unwrap(),
            &mut cut.as_tile_mut().unwrap(),
            demand[0],
        )
        .unwrap();
        let mut target = TileBuf::for_image(&out).unwrap();
        op.compute(
            &[cut.as_tile().unwrap()],
            &mut target.as_tile_mut().unwrap(),
        )
        .unwrap();
        (out, target.into_bytes())
    }

    /// A one-row grey image `[10, 20, 30]` extended by four on the left and
    /// right.
    fn row(with: ExtendWith) -> Vec<u8> {
        let descriptor = ImageDescriptor::new(3, 1, PixelFormat::Gray8).unwrap();
        let op = Extend::new(0, 4, 0, 4)
            .extend_with(with)
            .with_background([1, 1, 1, 255]);
        apply(&op, descriptor, &[10, 20, 30]).1
    }

    #[test]
    fn each_method_fills_the_margin_as_libvips_does() {
        assert_eq!(
            row(ExtendWith::Background),
            [1, 1, 1, 1, 10, 20, 30, 1, 1, 1, 1]
        );
        assert_eq!(
            row(ExtendWith::Copy),
            [10, 10, 10, 10, 10, 20, 30, 30, 30, 30, 30]
        );
        assert_eq!(
            row(ExtendWith::Repeat),
            [30, 10, 20, 30, 10, 20, 30, 10, 20, 30, 10]
        );
        // The edge pixel appears twice where a copy meets its reflection.
        assert_eq!(
            row(ExtendWith::Mirror),
            [30, 30, 20, 10, 10, 20, 30, 30, 20, 10, 10]
        );
    }

    #[test]
    fn the_background_is_placed_around_the_image() {
        let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Rgba8).unwrap();
        let op = Extend::new(1, 2, 3, 4).with_background([9, 8, 7, 6]);
        let (out, bytes) = apply(&op, descriptor, &[255; 16]);
        assert_eq!((out.width, out.height), (8, 6));
        let at = |x: usize, y: usize| &bytes[(y * 8 + x) * 4..(y * 8 + x) * 4 + 4];
        assert_eq!(at(0, 0), [9, 8, 7, 6]);
        assert_eq!(at(4, 1), [255; 4], "the image's top-left");
        assert_eq!(at(5, 2), [255; 4], "the image's bottom-right");
        assert_eq!(at(6, 2), [9, 8, 7, 6]);
        assert_eq!(at(4, 3), [9, 8, 7, 6]);
    }

    #[test]
    fn wide_formats_take_the_background_at_their_depth() {
        let descriptor = ImageDescriptor::new(1, 1, PixelFormat::Rgb16).unwrap();
        let op = Extend::new(0, 1, 0, 0).with_background([255, 0, 0, 255]);
        let (_, bytes) = apply(&op, descriptor, &[0; 6]);
        let red = u16::from_ne_bytes([bytes[6], bytes[7]]);
        assert_eq!(red, 65535);
    }

    #[test]
    fn a_margin_wider_than_the_image_wraps_more_than_once() {
        let descriptor = ImageDescriptor::new(2, 1, PixelFormat::Gray8).unwrap();
        let op = Extend::new(0, 5, 0, 5).extend_with(ExtendWith::Mirror);
        let (_, bytes) = apply(&op, descriptor, &[1, 2]);
        assert_eq!(bytes, [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]);
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        let descriptor = ImageDescriptor::new(7, 5, PixelFormat::Rgb8).unwrap();
        let bytes: Vec<u8> = (0..7 * 5 * 3).map(|i| (i * 37 % 251) as u8).collect();
        let source =
            TileBuf::from_vec(descriptor.region(), descriptor.pixel, bytes.clone()).unwrap();
        for with in [
            ExtendWith::Background,
            ExtendWith::Copy,
            ExtendWith::Repeat,
            ExtendWith::Mirror,
        ] {
            let op = Extend::new(3, 9, 6, 2).extend_with(with);
            let (out, whole) = apply(&op, descriptor, &bytes);
            for (tw, th) in [(4_u32, 4_u32), (1, 1), (18, 1), (5, 3)] {
                let mut target = TileBuf::for_image(&out).unwrap();
                for y in (0..out.height).step_by(th as usize) {
                    for x in (0..out.width).step_by(tw as usize) {
                        let region =
                            Region::new(x, y, tw.min(out.width - x), th.min(out.height - y));
                        let demand = op.input_regions(region, &[descriptor]).unwrap();
                        let mut cut = TileBuf::zeroed(demand[0], descriptor.pixel).unwrap();
                        copy_region(
                            &source.as_tile().unwrap(),
                            &mut cut.as_tile_mut().unwrap(),
                            demand[0],
                        )
                        .unwrap();
                        let mut sub = TileBuf::zeroed(region, out.pixel).unwrap();
                        op.compute(&[cut.as_tile().unwrap()], &mut sub.as_tile_mut().unwrap())
                            .unwrap();
                        copy_region(
                            &sub.as_tile().unwrap(),
                            &mut target.as_tile_mut().unwrap(),
                            region,
                        )
                        .unwrap();
                    }
                }
                assert_eq!(target.bytes(), whole.as_slice(), "{with:?} at {tw}x{th}");
            }
        }
    }

    #[test]
    fn a_tile_of_pure_margin_asks_for_almost_nothing() {
        let descriptor = ImageDescriptor::new(100, 100, PixelFormat::Rgb8).unwrap();
        let op = Extend::uniform(50);
        op.output_descriptor(&[descriptor]).unwrap();
        let demand = op
            .input_regions(Region::new(0, 0, 40, 40), &[descriptor])
            .unwrap();
        assert_eq!(demand[0], Region::new(0, 0, 1, 1));
        let demand = op
            .input_regions(Region::new(60, 60, 10, 10), &[descriptor])
            .unwrap();
        assert_eq!(demand[0], Region::new(10, 10, 10, 10));
    }

    #[test]
    fn one_instance_on_two_shapes_is_a_graph_error() {
        let op = Extend::uniform(1);
        let small = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let large = ImageDescriptor::new(3, 3, PixelFormat::Gray8).unwrap();
        op.output_descriptor(&[small]).unwrap();
        assert!(op.output_descriptor(&[large]).is_err());
        // A clone starts unbound.
        assert!(op.clone().output_descriptor(&[large]).is_ok());
    }

    #[test]
    fn an_overflowing_size_is_an_invalid_argument() {
        let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        assert!(
            Extend::new(0, u32::MAX, 0, 0)
                .output_descriptor(&[descriptor])
                .is_err()
        );
    }
}
