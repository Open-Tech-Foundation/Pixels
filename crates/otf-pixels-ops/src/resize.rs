//! [`Resize`] — separable resampling to a target size.
//!
//! # Two passes, one intermediate
//!
//! A separable filter applied in two one-dimensional passes costs
//! `O(w · h · (support_x + support_y))` instead of the product, which for
//! Lanczos3 is the difference between 6 taps per pixel and 36. The cost is one
//! intermediate buffer, `output_width × input_height`, held only for the
//! duration of a tile.
//!
//! Horizontal runs first, deliberately: when downscaling it narrows the rows
//! before the vertical pass ever touches them, so the intermediate is the
//! smaller of the two possible orderings.
//!
//! # Demand
//!
//! Resize is the first op whose `input_regions` is not a translation. An
//! output tile needs the union of the filter footprints of its rows and
//! columns, which [`Weights::footprint`] computes from the same tables the
//! kernel uses — so what is requested and what is read cannot drift apart.

use std::sync::OnceLock;

use otf_pixels_core::{
    AccessPattern, ChannelLayout, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result,
    SampleKind, Tile, TileBuf, TileMut,
};

use crate::filter::{Filter, Weights};
use crate::resample::{column_f32, column_u8, column_u16, row_f32, row_u8, row_u16};

/// How a resize reconciles the requested box with the source aspect ratio,
/// with sharp's meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Fit {
    /// Stretch to exactly the box, ignoring aspect ratio.
    #[default]
    Fill,
    /// Preserve aspect, scaled so the image fits inside the box; the output
    /// is the scaled image, at most the box.
    Inside,
    /// Preserve aspect, scaled so the image covers the box; the output is
    /// the scaled image, at least the box.
    Outside,
    /// Preserve aspect and fill the box exactly: scale to cover it, then
    /// crop the overflow, centred.
    Cover,
    /// Preserve aspect and fill the box exactly: scale to fit inside it,
    /// then pad the rest with [`ResizeOptions::background`], centred.
    Contain,
}

/// Options for [`Resize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ResizeOptions {
    /// The resampling filter. Defaults to [`Filter::Lanczos3`].
    pub filter: Filter,
    /// How the target box is interpreted.
    pub fit: Fit,
    /// Never scale *up*: an image already smaller than the box is left alone.
    pub without_enlargement: bool,
    /// The RGBA colour [`Fit::Contain`] pads with: opaque black by default,
    /// as in sharp. Grey formats take its luma; formats without alpha
    /// ignore its alpha.
    pub background: [u8; 4],
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            filter: Filter::default(),
            fit: Fit::default(),
            without_enlargement: false,
            background: [0, 0, 0, 255],
        }
    }
}

impl ResizeOptions {
    /// Options with an explicit filter.
    #[must_use]
    pub const fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = filter;
        self
    }

    /// Options with an explicit fit mode.
    #[must_use]
    pub const fn with_fit(mut self, fit: Fit) -> Self {
        self.fit = fit;
        self
    }

    /// Options with an explicit padding colour for [`Fit::Contain`].
    #[must_use]
    pub const fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = rgba;
        self
    }

    /// Options that refuse to enlarge.
    #[must_use]
    pub const fn without_enlargement(mut self, refuse: bool) -> Self {
        self.without_enlargement = refuse;
        self
    }
}

/// The weight tables this op was bound to when it was chained onto an image.
///
/// Tables are built **per image, not per tile**. Building them per tile would
/// make an output pixel depend on which tile it landed in, because a tile's
/// local scale is not the image's scale — and SPEC §Guarantees 2 requires the
/// output to be independent of tiling. So they are resolved once, at
/// graph-build time, and indexed by the tile's position within the image.
#[derive(Debug)]
struct Binding {
    input: ImageDescriptor,
    /// Tables for the visible part of the scaled image only.
    horizontal: Weights,
    vertical: Weights,
    layout: Layout,
}

/// Where the scaled image lands in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    /// The whole output.
    output: (u32, u32),
    /// The image's part of it, and where that part sits: everything outside
    /// is padding ([`Fit::Contain`]).
    placed: Region,
}

/// Resample an image to a new size.
#[derive(Debug)]
pub struct Resize {
    width: u32,
    height: u32,
    options: ResizeOptions,
    /// Filled by `output_descriptor` at graph-build time.
    bound: OnceLock<Binding>,
}

impl Clone for Resize {
    fn clone(&self) -> Self {
        // The binding is deliberately not cloned: a clone may be chained onto
        // a differently shaped image, and inheriting the original's tables
        // would silently resample against the wrong scale.
        Self {
            width: self.width,
            height: self.height,
            options: self.options,
            bound: OnceLock::new(),
        }
    }
}

impl Resize {
    /// Resize to `width` by `height` with `options`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if either dimension is zero.
    pub fn new(width: u32, height: u32, options: ResizeOptions) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PixelsError::invalid_argument(
                "size",
                format!("resize target {width}x{height} has no pixels"),
            ));
        }
        Ok(Self {
            width,
            height,
            options,
            bound: OnceLock::new(),
        })
    }

    /// Resize to `width` by `height` with the default filter and fit.
    ///
    /// # Errors
    ///
    /// As [`Resize::new`].
    pub fn to(width: u32, height: u32) -> Result<Self> {
        Self::new(width, height, ResizeOptions::default())
    }

    /// The options this op was built with.
    #[must_use]
    pub const fn options(&self) -> ResizeOptions {
        self.options
    }

    /// The output size for a given input, after fit and enlargement rules.
    ///
    /// Computed rather than stored, because the requested box is a *constraint*
    /// and the answer depends on the input the op is eventually chained onto.
    #[must_use]
    pub fn target(&self, input: &ImageDescriptor) -> (u32, u32) {
        self.layout(input).1.output
    }

    /// The scaled size, the crop window within it, and the output layout.
    fn layout(&self, input: &ImageDescriptor) -> ((u32, u32), Layout, (u32, u32)) {
        // Scale by whichever axis binds, in f64 so a 30000-pixel input does
        // not lose precision on the ratio.
        let by_width = f64::from(self.width) / f64::from(input.width);
        let by_height = f64::from(self.height) / f64::from(input.height);
        let preserving = |scale: f64| {
            (
                ((f64::from(input.width) * scale).round() as u32).max(1),
                ((f64::from(input.height) * scale).round() as u32).max(1),
            )
        };
        let (mut width, mut height) = match self.options.fit {
            Fit::Fill => (self.width, self.height),
            Fit::Inside | Fit::Contain => preserving(by_width.min(by_height)),
            Fit::Outside | Fit::Cover => preserving(by_width.max(by_height)),
        };
        if self.options.without_enlargement {
            width = width.min(input.width);
            height = height.min(input.height);
        }
        let scaled = (width.max(1), height.max(1));
        let (sw, sh) = scaled;
        match self.options.fit {
            Fit::Cover => {
                let (ow, oh) = (sw.min(self.width), sh.min(self.height));
                let window = ((sw - ow) / 2, (sh - oh) / 2);
                (
                    scaled,
                    Layout {
                        output: (ow, oh),
                        placed: Region::new(0, 0, ow, oh),
                    },
                    window,
                )
            }
            Fit::Contain => {
                let (ow, oh) = (self.width.max(sw), self.height.max(sh));
                let placed = Region::new((ow - sw) / 2, (oh - sh) / 2, sw, sh);
                (
                    scaled,
                    Layout {
                        output: (ow, oh),
                        placed,
                    },
                    (0, 0),
                )
            }
            _ => (
                scaled,
                Layout {
                    output: scaled,
                    placed: Region::new(0, 0, sw, sh),
                },
                (0, 0),
            ),
        }
    }

    /// The weight tables for resampling `input`, resolved once and reused.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::graph`] if this op was already bound to a
    /// differently shaped input. An op instance belongs to one graph edge;
    /// reusing it across two shapes would resample the second against the
    /// first one's scale, which is a silent wrong answer rather than a loud
    /// one.
    fn binding(&self, input: &ImageDescriptor) -> Result<&Binding> {
        if let Some(bound) = self.bound.get() {
            return check_binding(bound, input);
        }
        let ((width, height), layout, (wx, wy)) = self.layout(input);
        let (visible_w, visible_h) = (layout.placed.width, layout.placed.height);
        let candidate = Binding {
            input: *input,
            horizontal: Weights::build(self.options.filter, input.width, width)?
                .window(wx, visible_w),
            vertical: Weights::build(self.options.filter, input.height, height)?
                .window(wy, visible_h),
            layout,
        };
        let bound = self.bound.get_or_init(|| candidate);
        check_binding(bound, input)
    }

    /// The tables resolved at graph-build time, for use during `compute`.
    fn bound(&self) -> Result<&Binding> {
        self.bound.get().ok_or_else(|| {
            PixelsError::graph("`resize` computed before its output shape was resolved")
        })
    }
}

/// Confirm a cached binding matches the input it is about to be used for.
fn check_binding<'a>(bound: &'a Binding, input: &ImageDescriptor) -> Result<&'a Binding> {
    if bound.input.width != input.width || bound.input.height != input.height {
        return Err(PixelsError::graph(format!(
            "`resize` is bound to a {}x{} input but was given {}x{}; \
             build a new Resize per image",
            bound.input.width, bound.input.height, input.width, input.height
        )));
    }
    Ok(bound)
}

impl Op for Resize {
    /// Resize pins its own output size, so a smaller input reaches the same
    /// answer with less work. This is the op shrink-on-load exists for.
    ///
    /// The clone drops the filter tables, which are bound to the input shape
    /// they were first built for — reusing them against a reduced source is
    /// exactly the resample-at-the-wrong-scale this guards against.
    fn rescaled(&self) -> Option<std::sync::Arc<dyn Op>> {
        Some(std::sync::Arc::new(self.clone()))
    }
    fn name(&self) -> &'static str {
        "resize"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`resize` takes one input, got none"))?;
        // Resolving the binding here is what makes the tables per-image:
        // `output_descriptor` runs exactly once per graph edge, at build time.
        self.binding(input)?;
        let (width, height) = self.target(input);
        input.resized(width, height)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`resize` takes one input, got none"))?;
        let bound = self.binding(input)?;
        // Only the part of the tile the image covers reads anything; a tile
        // of pure padding still names a valid (1x1) input region, as
        // `composite` does for an overlay it misses.
        let Some(image) = bound.image_part(output) else {
            return Ok(vec![Region::new(0, 0, 1, 1)]);
        };
        // The footprint of exactly the output rows and columns requested,
        // taken from the same tables the kernel will use — so what is asked
        // for and what is read cannot drift apart.
        let (x, width) = bound.horizontal.footprint(image.x, image.width);
        let (y, height) = bound.vertical.footprint(image.y, image.height);
        Ok(vec![Region::new(x, y, width, height)])
    }

    fn access_pattern(&self) -> AccessPattern {
        // Resize reads a two-dimensional neighbourhood: every output row draws
        // on a band of input rows, so square tiles keep that band small.
        AccessPattern::Spatial
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`resize` takes one input tile, got none"))?;

        let format = output.pixel();
        if input.pixel() != format {
            return Err(PixelsError::graph(format!(
                "`resize` input is {} but output is {format}",
                input.pixel()
            )));
        }
        let channels = format.channels();
        let bound = self.bound()?;
        let region = output.region();
        if bound.layout.placed.contains(region)
            && bound.layout.placed.x == 0
            && bound.layout.placed.y == 0
        {
            return resample_tile(bound, input, output, channels);
        }
        // Padding: background everywhere, then the image's part resampled in
        // the scaled image's own coordinates and copied into place.
        let fill = background_pixel(format, self.options.background);
        for row in output.rows_mut() {
            for (slot, &byte) in row.iter_mut().zip(fill.iter().cycle()) {
                *slot = byte;
            }
        }
        let Some(image) = bound.image_part(region) else {
            return Ok(());
        };
        let mut part = TileBuf::zeroed(image, format)?;
        resample_tile(bound, input, &mut part.as_tile_mut()?, channels)?;
        let (px, py) = (bound.layout.placed.x, bound.layout.placed.y);
        let bytes = format.bytes_per_pixel();
        let row_len = image.width as usize * bytes;
        for (i, from) in part.bytes().chunks_exact(row_len.max(1)).enumerate() {
            let Some(row) = output.row_mut(image.y + py + i as u32) else {
                continue;
            };
            let at = (image.x + px - region.x) as usize * bytes;
            if let Some(to) = row.get_mut(at..at + row_len) {
                to.copy_from_slice(from);
            }
        }
        Ok(())
    }
}

impl Binding {
    /// The part of output `region` the image covers, in the scaled image's
    /// coordinates; `None` for a tile of pure padding.
    fn image_part(&self, region: Region) -> Option<Region> {
        let placed = self.layout.placed;
        let overlap = region.intersect(placed);
        (overlap.width > 0 && overlap.height > 0).then(|| {
            Region::new(
                overlap.x - placed.x,
                overlap.y - placed.y,
                overlap.width,
                overlap.height,
            )
        })
    }
}

/// One pixel of `rgba` in `format`'s bytes.
fn background_pixel(format: PixelFormat, rgba: [u8; 4]) -> Vec<u8> {
    let [r, g, b, a] = rgba;
    let luma = ((u32::from(r) * 299 + u32::from(g) * 587 + u32::from(b) * 114 + 500) / 1000) as u8;
    let samples: Vec<u8> = match format.layout() {
        ChannelLayout::Gray => vec![luma],
        ChannelLayout::GrayAlpha => vec![luma, a],
        ChannelLayout::Rgb => vec![r, g, b],
        ChannelLayout::Rgba => vec![r, g, b, a],
    };
    match format.sample_kind() {
        SampleKind::U8 => samples,
        SampleKind::U16 => samples
            .iter()
            .flat_map(|&v| (u16::from(v) * 257).to_ne_bytes())
            .collect(),
        SampleKind::F32 => samples
            .iter()
            .flat_map(|&v| (f32::from(v) / 255.0).to_ne_bytes())
            .collect(),
    }
}

/// Run both passes for one output tile.
///
/// The tables are the image's, not the tile's. Each output row and column of
/// the tile looks up its own whole-image run, and reads it out of the input
/// tile at an offset — which is why the answer is the same however the image
/// is cut up.
fn resample_tile(
    bound: &Binding,
    input: &Tile<'_>,
    output: &mut TileMut<'_>,
    channels: usize,
) -> Result<()> {
    let region = output.region();
    let source = input.region();
    let horizontal = bound
        .horizontal
        .for_tile(region.x, region.width, source.x)?;
    let vertical = bound.vertical.for_tile(region.y, region.height, source.y)?;

    let format = output.pixel();
    let intermediate_row = region.width as usize * channels;

    match format.sample_kind() {
        SampleKind::U8 => {
            // Horizontal pass into an intermediate of output width by input
            // height, then vertical down each column of that.
            let mut intermediate = vec![0_u8; intermediate_row * source.height as usize];
            for y in 0..source.height {
                let Some(row) = input.row(source.y + y) else {
                    continue;
                };
                let at = y as usize * intermediate_row;
                let Some(target) = intermediate.get_mut(at..at + intermediate_row) else {
                    continue;
                };
                row_u8(row, target, channels, &horizontal);
            }

            // The vertical pass accumulates a whole output row at a time,
            // reading input rows sequentially. See the note in `resample`.
            let mut accumulator = vec![0_i32; intermediate_row];
            let mut resampled = vec![0_u8; intermediate_row];
            for (index, run) in vertical.runs().iter().enumerate() {
                column_u8(
                    &intermediate,
                    intermediate_row,
                    run.start as usize,
                    vertical.quantized(run),
                    &mut accumulator,
                    &mut resampled,
                );
                write_row(output, region.y + index as u32, &resampled);
            }
        }
        SampleKind::U16 => {
            let mut intermediate = vec![0_u16; intermediate_row * source.height as usize];
            for y in 0..source.height {
                let Some(row) = input.row(source.y + y) else {
                    continue;
                };
                let wide = to_u16(row);
                let at = y as usize * intermediate_row;
                let Some(target) = intermediate.get_mut(at..at + intermediate_row) else {
                    continue;
                };
                row_u16(&wide, target, channels, &horizontal);
            }
            let mut accumulator = vec![0.0_f32; intermediate_row];
            let mut resampled = vec![0_u16; intermediate_row];
            for (index, run) in vertical.runs().iter().enumerate() {
                column_u16(
                    &intermediate,
                    intermediate_row,
                    run.start as usize,
                    vertical.exact(run),
                    &mut accumulator,
                    &mut resampled,
                );
                let bytes: Vec<u8> = resampled.iter().flat_map(|v| v.to_ne_bytes()).collect();
                write_row(output, region.y + index as u32, &bytes);
            }
        }
        SampleKind::F32 => {
            let mut intermediate = vec![0.0_f32; intermediate_row * source.height as usize];
            for y in 0..source.height {
                let Some(row) = input.row(source.y + y) else {
                    continue;
                };
                let floats = to_f32(row);
                let at = y as usize * intermediate_row;
                let Some(target) = intermediate.get_mut(at..at + intermediate_row) else {
                    continue;
                };
                row_f32(&floats, target, channels, &horizontal);
            }
            let mut accumulator = vec![0.0_f32; intermediate_row];
            let mut resampled = vec![0.0_f32; intermediate_row];
            for (index, run) in vertical.runs().iter().enumerate() {
                column_f32(
                    &intermediate,
                    intermediate_row,
                    run.start as usize,
                    vertical.exact(run),
                    &mut accumulator,
                    &mut resampled,
                );
                let bytes: Vec<u8> = resampled.iter().flat_map(|v| v.to_ne_bytes()).collect();
                write_row(output, region.y + index as u32, &bytes);
            }
        }
    }
    Ok(())
}

/// Reinterpret a native-endian byte row as 16-bit samples.
fn to_u16(row: &[u8]) -> Vec<u16> {
    row.chunks_exact(2)
        .map(|pair| {
            u16::from_ne_bytes([
                pair.first().copied().unwrap_or(0),
                pair.get(1).copied().unwrap_or(0),
            ])
        })
        .collect()
}

/// Reinterpret a native-endian byte row as float samples.
fn to_f32(row: &[u8]) -> Vec<f32> {
    row.chunks_exact(4)
        .map(|quad| {
            let mut bytes = [0_u8; 4];
            for (slot, &byte) in bytes.iter_mut().zip(quad) {
                *slot = byte;
            }
            f32::from_ne_bytes(bytes)
        })
        .collect()
}

/// Write a resampled row into the output tile.
fn write_row(output: &mut TileMut<'_>, y: u32, bytes: &[u8]) {
    let Some(row) = output.row_mut(y) else { return };
    let len = row.len().min(bytes.len());
    if let (Some(to), Some(from)) = (row.get_mut(..len), bytes.get(..len)) {
        to.copy_from_slice(from);
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

    /// Resize a whole image in one tile, the simple case.
    fn resize_whole(
        op: &Resize,
        input: &ImageDescriptor,
        bytes: &[u8],
    ) -> Result<(ImageDescriptor, Vec<u8>)> {
        let out_desc = op.output_descriptor(std::slice::from_ref(input))?;
        let source = TileBuf::from_vec(input.region(), input.pixel, bytes.to_vec())?;
        let mut target = TileBuf::for_image(&out_desc)?;
        let regions = op.input_regions(out_desc.region(), std::slice::from_ref(input))?;
        assert_eq!(
            regions[0],
            input.region(),
            "whole-image demand should be the whole input"
        );
        op.compute(&[source.as_tile()?], &mut target.as_tile_mut()?)?;
        Ok((out_desc, target.into_bytes()))
    }

    fn ramp(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let len = descriptor.byte_len().unwrap();
        let bytes = (0..len).map(|i| ((i * 37) % 251) as u8).collect();
        (descriptor, bytes)
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        // The property that makes per-image weight tables necessary rather
        // than merely tidy. Resizing in one tile and in many must give the
        // same bytes, or SPEC §Guarantees 2 is false.
        let (input, bytes) = ramp(97, 71, PixelFormat::Rgb8);
        let op = Resize::new(41, 33, ResizeOptions::default()).unwrap();
        let (out_desc, whole) = resize_whole(&op, &input, &bytes).unwrap();

        for (tile_w, tile_h) in [(8_u32, 8_u32), (16, 4), (41, 1), (1, 33), (7, 13)] {
            // A fresh op per run: a binding belongs to one graph edge.
            let op = Resize::new(41, 33, ResizeOptions::default()).unwrap();
            op.output_descriptor(std::slice::from_ref(&input)).unwrap();
            let source = TileBuf::from_vec(input.region(), input.pixel, bytes.clone()).unwrap();
            let mut target = TileBuf::for_image(&out_desc).unwrap();

            let mut y = 0;
            while y < out_desc.height {
                let h = tile_h.min(out_desc.height - y);
                let mut x = 0;
                while x < out_desc.width {
                    let w = tile_w.min(out_desc.width - x);
                    let region = Region::new(x, y, w, h);
                    let demand = op
                        .input_regions(region, std::slice::from_ref(&input))
                        .unwrap();
                    let window = source.as_tile().unwrap();
                    let mut sub = TileBuf::zeroed(region, out_desc.pixel).unwrap();

                    // Hand the op exactly the footprint it asked for.
                    let mut cut = TileBuf::zeroed(demand[0], input.pixel).unwrap();
                    otf_pixels_core::copy_region(
                        &window,
                        &mut cut.as_tile_mut().unwrap(),
                        demand[0],
                    )
                    .unwrap();

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
                target.into_bytes(),
                whole,
                "tiling at {tile_w}x{tile_h} changed the pixels"
            );
        }
    }

    /// Run `op` over the whole output in `tile`-sized pieces, each handed
    /// exactly the input it demands.
    fn resize_tiled(
        op: &Resize,
        input: &ImageDescriptor,
        bytes: &[u8],
        tile: (u32, u32),
    ) -> (ImageDescriptor, Vec<u8>) {
        let out_desc = op.output_descriptor(std::slice::from_ref(input)).unwrap();
        let source = TileBuf::from_vec(input.region(), input.pixel, bytes.to_vec()).unwrap();
        let mut target = TileBuf::for_image(&out_desc).unwrap();
        let mut y = 0;
        while y < out_desc.height {
            let h = tile.1.min(out_desc.height - y);
            let mut x = 0;
            while x < out_desc.width {
                let w = tile.0.min(out_desc.width - x);
                let region = Region::new(x, y, w, h);
                let demand = op
                    .input_regions(region, std::slice::from_ref(input))
                    .unwrap();
                let mut cut = TileBuf::zeroed(demand[0], input.pixel).unwrap();
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
        (out_desc, target.into_bytes())
    }

    fn with_fit(width: u32, height: u32, fit: Fit) -> Resize {
        Resize::new(
            width,
            height,
            ResizeOptions::default()
                .with_fit(fit)
                .with_background([10, 20, 30, 255]),
        )
        .unwrap()
    }

    #[test]
    fn outside_and_inside_scale_by_the_other_axis() {
        let (input, _) = ramp(200, 100, PixelFormat::Rgb8);
        assert_eq!(with_fit(50, 50, Fit::Inside).target(&input), (50, 25));
        assert_eq!(with_fit(50, 50, Fit::Outside).target(&input), (100, 50));
        assert_eq!(with_fit(50, 50, Fit::Cover).target(&input), (50, 50));
        assert_eq!(with_fit(50, 50, Fit::Contain).target(&input), (50, 50));
        assert_eq!(with_fit(50, 50, Fit::Fill).target(&input), (50, 50));
    }

    #[test]
    fn cover_is_outside_then_a_centred_crop() {
        for (w, h, format) in [
            (200, 100, PixelFormat::Rgb8),
            (61, 97, PixelFormat::Rgba16),
            (90, 90, PixelFormat::Gray8),
        ] {
            let (input, bytes) = ramp(w, h, format);
            let (outside_desc, outside) = resize_tiled(
                &with_fit(40, 30, Fit::Outside),
                &input,
                &bytes,
                (1000, 1000),
            );
            let bpp = format.bytes_per_pixel();
            for tile in [(1000, 1000), (7, 5), (40, 1)] {
                let (desc, cover) =
                    resize_tiled(&with_fit(40, 30, Fit::Cover), &input, &bytes, tile);
                assert_eq!(
                    (desc.width, desc.height),
                    (40.min(outside_desc.width), 30.min(outside_desc.height))
                );
                let (dx, dy) = (
                    (outside_desc.width - desc.width) / 2,
                    (outside_desc.height - desc.height) / 2,
                );
                for y in 0..desc.height as usize {
                    let from =
                        ((y + dy as usize) * outside_desc.width as usize + dx as usize) * bpp;
                    let row = desc.width as usize * bpp;
                    assert_eq!(
                        &cover[y * row..(y + 1) * row],
                        &outside[from..from + row],
                        "{w}x{h} {format} row {y} tile {tile:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn contain_is_inside_centred_on_the_background() {
        for (w, h, format) in [
            (200, 100, PixelFormat::Rgb8),
            (61, 97, PixelFormat::Rgba8),
            (30, 90, PixelFormat::Gray16),
        ] {
            let (input, bytes) = ramp(w, h, format);
            let (inside_desc, inside) =
                resize_tiled(&with_fit(40, 30, Fit::Inside), &input, &bytes, (1000, 1000));
            let bpp = format.bytes_per_pixel();
            let fill = background_pixel(format, [10, 20, 30, 255]);
            for tile in [(1000, 1000), (7, 5), (3, 30), (40, 2)] {
                let (desc, contain) =
                    resize_tiled(&with_fit(40, 30, Fit::Contain), &input, &bytes, tile);
                assert_eq!((desc.width, desc.height), (40, 30));
                let (px, py) = ((40 - inside_desc.width) / 2, (30 - inside_desc.height) / 2);
                for y in 0..30_u32 {
                    for x in 0..40_u32 {
                        let at = ((y * 40 + x) as usize) * bpp;
                        let got = &contain[at..at + bpp];
                        let inside_x = x.checked_sub(px).filter(|&v| v < inside_desc.width);
                        let inside_y = y.checked_sub(py).filter(|&v| v < inside_desc.height);
                        let want = match (inside_x, inside_y) {
                            (Some(ix), Some(iy)) => {
                                let from = ((iy * inside_desc.width + ix) as usize) * bpp;
                                &inside[from..from + bpp]
                            }
                            _ => &fill[..],
                        };
                        assert_eq!(got, want, "{w}x{h} {format} at ({x}, {y}) tile {tile:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_background_follows_the_pixel_format() {
        assert_eq!(
            background_pixel(PixelFormat::Rgba8, [1, 2, 3, 4]),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            background_pixel(PixelFormat::Rgb8, [1, 2, 3, 4]),
            vec![1, 2, 3]
        );
        assert_eq!(
            background_pixel(PixelFormat::Gray8, [255, 255, 255, 0]),
            vec![255]
        );
        assert_eq!(
            background_pixel(PixelFormat::GrayA8, [0, 0, 0, 9]),
            vec![0, 9]
        );
        assert_eq!(
            background_pixel(PixelFormat::Gray16, [255, 0, 0, 255]),
            (76_u16 * 257).to_ne_bytes().to_vec()
        );
    }

    #[test]
    fn a_one_to_one_resize_returns_the_image_unchanged() {
        for format in [
            PixelFormat::Gray8,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Gray16,
            PixelFormat::Rgb16,
        ] {
            let (input, bytes) = ramp(31, 23, format);
            let op = Resize::new(31, 23, ResizeOptions::default()).unwrap();
            let (out_desc, out) = resize_whole(&op, &input, &bytes).unwrap();
            assert_eq!(out_desc.width, 31);
            assert_eq!(out, bytes, "{format} changed at 1:1");
        }
    }

    #[test]
    fn a_flat_image_stays_flat_at_every_scale_and_filter() {
        for filter in [
            Filter::Nearest,
            Filter::Box,
            Filter::Bilinear,
            Filter::CatmullRom,
            Filter::Mitchell,
            Filter::Lanczos2,
            Filter::Lanczos3,
        ] {
            for (w, h) in [(10_u32, 10_u32), (200, 150), (37, 91)] {
                let input = ImageDescriptor::new(64, 64, PixelFormat::Rgb8).unwrap();
                let bytes = vec![137_u8; input.byte_len().unwrap()];
                let op = Resize::new(w, h, ResizeOptions::default().with_filter(filter)).unwrap();
                let (_, out) = resize_whole(&op, &input, &bytes).unwrap();
                assert!(
                    out.iter().all(|&v| v == 137),
                    "{} to {w}x{h} did not stay flat",
                    filter.as_str()
                );
            }
        }
    }

    #[test]
    fn fit_inside_preserves_the_aspect_ratio() {
        let input = ImageDescriptor::new(1000, 500, PixelFormat::Rgb8).unwrap();
        let op = Resize::new(100, 100, ResizeOptions::default().with_fit(Fit::Inside)).unwrap();
        assert_eq!(
            op.target(&input),
            (100, 50),
            "wide image should bind on width"
        );

        let tall = ImageDescriptor::new(500, 1000, PixelFormat::Rgb8).unwrap();
        assert_eq!(
            op.target(&tall),
            (50, 100),
            "tall image should bind on height"
        );
    }

    #[test]
    fn fit_fill_ignores_the_aspect_ratio() {
        let input = ImageDescriptor::new(1000, 500, PixelFormat::Rgb8).unwrap();
        let op = Resize::new(100, 100, ResizeOptions::default()).unwrap();
        assert_eq!(op.target(&input), (100, 100));
    }

    #[test]
    fn without_enlargement_leaves_a_small_image_alone() {
        let input = ImageDescriptor::new(40, 30, PixelFormat::Rgb8).unwrap();
        let options = ResizeOptions::default()
            .with_fit(Fit::Inside)
            .without_enlargement(true);
        let op = Resize::new(1000, 1000, options).unwrap();
        assert_eq!(op.target(&input), (40, 30));

        // The same options must still shrink something larger than the box.
        let big = ImageDescriptor::new(4000, 3000, PixelFormat::Rgb8).unwrap();
        assert_eq!(op.target(&big), (1000, 750));
    }

    #[test]
    fn a_zero_target_is_an_error() {
        assert!(Resize::new(0, 10, ResizeOptions::default()).is_err());
        assert!(Resize::new(10, 0, ResizeOptions::default()).is_err());
    }

    #[test]
    fn a_target_never_collapses_to_zero() {
        // Fit::Inside on an extreme aspect ratio would round a dimension to
        // zero, which is not a representable image.
        let input = ImageDescriptor::new(10_000, 3, PixelFormat::Gray8).unwrap();
        let op = Resize::new(50, 50, ResizeOptions::default().with_fit(Fit::Inside)).unwrap();
        let (w, h) = op.target(&input);
        assert!(w >= 1 && h >= 1, "target collapsed to {w}x{h}");
    }

    #[test]
    fn reusing_an_op_across_two_shapes_is_an_error_not_a_wrong_answer() {
        // A binding belongs to one graph edge. Silently resampling the second
        // image against the first one's scale would be a wrong answer with no
        // symptom, so it is refused instead.
        let first = ImageDescriptor::new(100, 100, PixelFormat::Gray8).unwrap();
        let second = ImageDescriptor::new(200, 200, PixelFormat::Gray8).unwrap();
        let op = Resize::to(50, 50).unwrap();
        op.output_descriptor(std::slice::from_ref(&first)).unwrap();
        let error = op
            .output_descriptor(std::slice::from_ref(&second))
            .unwrap_err();
        assert!(error.to_string().contains("bound to"), "{error}");
    }

    #[test]
    fn a_clone_can_be_bound_to_a_different_shape() {
        // The counterpart: cloning is how a caller reuses a configured op, so
        // a clone must start unbound rather than inherit the original tables.
        let first = ImageDescriptor::new(100, 100, PixelFormat::Gray8).unwrap();
        let second = ImageDescriptor::new(200, 200, PixelFormat::Gray8).unwrap();
        let op = Resize::to(50, 50).unwrap();
        op.output_descriptor(std::slice::from_ref(&first)).unwrap();
        let fresh = op.clone();
        assert!(
            fresh
                .output_descriptor(std::slice::from_ref(&second))
                .is_ok()
        );
    }

    #[test]
    fn demand_never_reaches_outside_the_input() {
        // Op::input_regions requires clamped regions; an op reading past its
        // input is a defect, and Lanczos3 near an edge is where it would.
        let input = ImageDescriptor::new(50, 40, PixelFormat::Rgb8).unwrap();
        let op = Resize::new(200, 160, ResizeOptions::default()).unwrap();
        let out = op.output_descriptor(std::slice::from_ref(&input)).unwrap();
        for y in 0..out.height {
            for x in 0..out.width {
                let region = Region::new(x, y, 1, 1);
                let demand = op
                    .input_regions(region, std::slice::from_ref(&input))
                    .unwrap();
                let r = demand[0];
                assert!(
                    r.x + r.width <= input.width && r.y + r.height <= input.height,
                    "demand {r} for output {region} leaves a {}x{} input",
                    input.width,
                    input.height
                );
            }
        }
    }

    #[test]
    fn resize_is_deterministic() {
        let (input, bytes) = ramp(123, 87, PixelFormat::Rgba8);
        let op = Resize::to(61, 43).unwrap();
        let (_, first) = resize_whole(&op, &input, &bytes).unwrap();
        for _ in 0..5 {
            let op = Resize::to(61, 43).unwrap();
            let (_, again) = resize_whole(&op, &input, &bytes).unwrap();
            assert_eq!(again, first, "resize is not deterministic");
        }
    }
}
