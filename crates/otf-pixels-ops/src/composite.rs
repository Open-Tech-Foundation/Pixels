//! [`Composite`] — draw one image over another.
//!
//! This is the first op with two inputs, which makes it the first place
//! `Op::arity` and the two-element `input_regions` vector do real work.
//!
//! # Alpha
//!
//! SPEC §Formats: alpha is unassociated at API boundaries. Source-over is
//! therefore computed in premultiplied form internally and converted back on
//! the way out, because straight-alpha compositing has no correct one-line
//! form — the naive `f*a + b*(1-a)` is only right when the backdrop is opaque.
//!
//! Getting this wrong shows up as a dark halo around soft edges, which is the
//! single most common compositing bug and looks like a bad matte rather than
//! like arithmetic.

use otf_pixels_core::{
    ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, SampleKind, Tile, TileMut,
};

/// How the source is combined with the backdrop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Blend {
    /// Porter-Duff source-over: the source is drawn on top of the backdrop.
    #[default]
    Over,
    /// The source replaces the backdrop entirely within its rectangle.
    Source,
}

/// Draw `overlay` over `base` at an offset.
///
/// Input order is `[base, overlay]`, matching the reading order of "composite
/// this over that".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Composite {
    x: i64,
    y: i64,
    blend: Blend,
}

impl Composite {
    /// Draw the overlay with its top-left corner at `(x, y)` in base
    /// coordinates.
    ///
    /// Negative offsets are allowed: the overlay is clipped to the base rather
    /// than rejected, which is what makes "centre a watermark" expressible
    /// without the caller doing the clipping arithmetic.
    #[must_use]
    pub const fn at(x: i64, y: i64, blend: Blend) -> Self {
        Self { x, y, blend }
    }

    /// Draw the overlay over the base at the origin with source-over.
    #[must_use]
    pub const fn over() -> Self {
        Self::at(0, 0, Blend::Over)
    }

    /// The offset the overlay is drawn at.
    #[must_use]
    pub const fn offset(&self) -> (i64, i64) {
        (self.x, self.y)
    }

    /// The blend mode.
    #[must_use]
    pub const fn blend(&self) -> Blend {
        self.blend
    }

    /// The region of the overlay that lands inside `output`, in overlay
    /// coordinates, or `None` if the overlay misses this region entirely.
    fn overlay_region(&self, output: Region, overlay: &ImageDescriptor) -> Option<Region> {
        // Output coordinates are base coordinates; subtract the offset to get
        // overlay coordinates, then intersect with the overlay's own bounds.
        let left = (i64::from(output.x) - self.x).max(0);
        let top = (i64::from(output.y) - self.y).max(0);
        let right = (i64::from(output.x + output.width) - self.x).min(i64::from(overlay.width));
        let bottom = (i64::from(output.y + output.height) - self.y).min(i64::from(overlay.height));

        if right <= left || bottom <= top {
            return None;
        }
        Some(Region::new(
            left as u32,
            top as u32,
            (right - left) as u32,
            (bottom - top) as u32,
        ))
    }
}

impl Op for Composite {
    fn name(&self) -> &'static str {
        "composite"
    }

    fn arity(&self) -> usize {
        2
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let (base, overlay) = pair(inputs)?;
        if base.pixel != overlay.pixel {
            return Err(PixelsError::unsupported(format!(
                "composite needs matching formats: base is {}, overlay is {}",
                base.pixel, overlay.pixel
            )));
        }
        if base.pixel.sample_kind() != SampleKind::U8 {
            return Err(PixelsError::unsupported(format!(
                "composite is implemented for 8-bit formats; got {}",
                base.pixel
            )));
        }
        // The output is the base's shape: compositing draws *onto* something,
        // so the backdrop defines the canvas.
        Ok(*base)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let (_, overlay) = pair(inputs)?;
        // The base is needed wherever the output is. The overlay is needed only
        // where it actually lands — a watermark in the corner must not make the
        // scheduler pull the whole overlay for every tile.
        let wanted = self
            .overlay_region(output, overlay)
            // An empty region still has to be a valid one, so a miss is
            // reported as a 1x1 at the origin rather than a zero-sized region
            // the tile machinery would reject.
            .unwrap_or_else(|| Region::new(0, 0, 1, 1));
        Ok(vec![output, wanted])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let base = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`composite` needs a base tile"))?;
        let overlay = inputs
            .get(1)
            .ok_or_else(|| PixelsError::graph("`composite` needs an overlay tile"))?;

        let format = output.pixel();
        let channels = format.channels();
        let region = output.region();

        // The backdrop always shows through, so start from it.
        for y in region.y..region.y + region.height {
            let (Some(source), Some(target)) = (base.row(y), output.row_mut(y)) else {
                continue;
            };
            let len = target.len().min(source.len());
            if let (Some(to), Some(from)) = (target.get_mut(..len), source.get(..len)) {
                to.copy_from_slice(from);
            }
        }

        let overlay_area = overlay.region();
        let has_alpha = matches!(format, PixelFormat::GrayA8 | PixelFormat::Rgba8);

        for y in region.y..region.y + region.height {
            // Where in the overlay this output row comes from.
            let source_y = i64::from(y) - self.y;
            if source_y < 0 {
                continue;
            }
            let source_y = source_y as u32;
            if source_y < overlay_area.y || source_y >= overlay_area.y + overlay_area.height {
                continue;
            }
            let Some(over_row) = overlay.row(source_y) else {
                continue;
            };
            let Some(target) = output.row_mut(y) else {
                continue;
            };

            for x in region.x..region.x + region.width {
                let source_x = i64::from(x) - self.x;
                if source_x < 0 {
                    continue;
                }
                let source_x = source_x as u32;
                if source_x < overlay_area.x || source_x >= overlay_area.x + overlay_area.width {
                    continue;
                }

                let from = (source_x - overlay_area.x) as usize * channels;
                let to = (x - region.x) as usize * channels;
                let Some(over) = over_row.get(from..from + channels) else {
                    continue;
                };

                match self.blend {
                    Blend::Source => {
                        if let Some(slot) = target.get_mut(to..to + channels) {
                            slot.copy_from_slice(over);
                        }
                    }
                    Blend::Over => {
                        if !has_alpha {
                            // No alpha means fully opaque, so source-over is
                            // just replacement.
                            if let Some(slot) = target.get_mut(to..to + channels) {
                                slot.copy_from_slice(over);
                            }
                            continue;
                        }
                        let alpha = channels - 1;
                        let sa = u32::from(over.get(alpha).copied().unwrap_or(255));
                        let ba = u32::from(target.get(to + alpha).copied().unwrap_or(255));
                        // Porter-Duff over, on unassociated alpha:
                        //   out_a = sa + ba*(1-sa)
                        //   out_c = (sc*sa + bc*ba*(1-sa)) / out_a
                        // Computing it premultiplied and dividing back out is
                        // what avoids the dark halo the naive form produces
                        // over a translucent backdrop.
                        let inverse = 255 - sa;
                        let out_a = sa * 255 + ba * inverse;
                        if out_a == 0 {
                            // Fully transparent result: colour is undefined, so
                            // leave it zeroed rather than dividing by zero.
                            for channel in 0..channels {
                                if let Some(slot) = target.get_mut(to + channel) {
                                    *slot = 0;
                                }
                            }
                            continue;
                        }
                        for channel in 0..alpha {
                            let sc = u32::from(over.get(channel).copied().unwrap_or(0));
                            let bc = u32::from(target.get(to + channel).copied().unwrap_or(0));
                            let numerator = sc * sa * 255 + bc * ba * inverse;
                            let value = (numerator + out_a / 2) / out_a;
                            if let Some(slot) = target.get_mut(to + channel) {
                                *slot = value.min(255) as u8;
                            }
                        }
                        if let Some(slot) = target.get_mut(to + alpha) {
                            *slot = ((out_a + 127) / 255).min(255) as u8;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Fetch the two input descriptors a composite was given.
fn pair(inputs: &[ImageDescriptor]) -> Result<(&ImageDescriptor, &ImageDescriptor)> {
    match (inputs.first(), inputs.get(1)) {
        (Some(base), Some(overlay)) => Ok((base, overlay)),
        _ => Err(PixelsError::graph(format!(
            "`composite` takes two inputs, got {}",
            inputs.len()
        ))),
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
    use otf_pixels_core::TileBuf;

    fn solid(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixel: &[u8],
    ) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let mut bytes = Vec::with_capacity(descriptor.byte_len().unwrap());
        for _ in 0..(width * height) {
            bytes.extend_from_slice(pixel);
        }
        (descriptor, bytes)
    }

    /// Composite two whole images.
    fn apply(
        op: &Composite,
        base: (&ImageDescriptor, &[u8]),
        overlay: (&ImageDescriptor, &[u8]),
    ) -> Result<(ImageDescriptor, Vec<u8>)> {
        let inputs = [*base.0, *overlay.0];
        let out_desc = op.output_descriptor(&inputs)?;
        let base_buf = TileBuf::from_vec(base.0.region(), base.0.pixel, base.1.to_vec())?;
        let over_buf = TileBuf::from_vec(overlay.0.region(), overlay.0.pixel, overlay.1.to_vec())?;
        let mut target = TileBuf::for_image(&out_desc)?;
        op.compute(
            &[base_buf.as_tile()?, over_buf.as_tile()?],
            &mut target.as_tile_mut()?,
        )?;
        Ok((out_desc, target.into_bytes()))
    }

    #[test]
    fn an_opaque_overlay_replaces_the_backdrop() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[10, 20, 30, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[200, 100, 50, 255]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel, [200, 100, 50, 255], "opaque overlay did not replace");
        }
    }

    #[test]
    fn a_fully_transparent_overlay_leaves_the_backdrop_alone() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[10, 20, 30, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[200, 100, 50, 0]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(
                pixel,
                [10, 20, 30, 255],
                "transparent overlay changed the base"
            );
        }
    }

    #[test]
    fn half_alpha_over_an_opaque_backdrop_is_the_midpoint() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[0, 0, 0, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel[3], 255, "an opaque backdrop must stay opaque");
            assert_eq!(pixel[0], 128, "expected the midpoint, got {pixel:?}");
        }
    }

    #[test]
    fn compositing_onto_a_translucent_backdrop_does_not_darken() {
        // The dark-halo bug: the naive `f*a + b*(1-a)` is only correct over an
        // opaque backdrop. Over a translucent one it under-weights the
        // backdrop's colour and the result creeps toward black.
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            // White over white is white at any alpha. The naive formula gives
            // 191 here instead.
            assert_eq!(
                &pixel[..3],
                [255, 255, 255],
                "white over white darkened to {pixel:?}"
            );
            assert!(pixel[3] > 128, "alpha should accumulate, got {}", pixel[3]);
        }
    }

    #[test]
    fn two_transparent_pixels_do_not_divide_by_zero() {
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgba8, &[9, 9, 9, 0]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[7, 7, 7, 0]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel[3], 0, "result should stay transparent");
        }
    }

    #[test]
    fn the_overlay_is_placed_at_its_offset() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[0, 0, 0, 255]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 0, 0, 255]);
        let op = Composite::at(1, 1, Blend::Over);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();

        let pixel_at = |x: usize, y: usize| -> &[u8] { &out[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4] };
        assert_eq!(pixel_at(0, 0), [0, 0, 0, 255], "outside the overlay");
        assert_eq!(pixel_at(1, 1), [255, 0, 0, 255], "inside the overlay");
        assert_eq!(pixel_at(2, 2), [255, 0, 0, 255], "inside the overlay");
        assert_eq!(pixel_at(3, 3), [0, 0, 0, 255], "outside the overlay");
    }

    #[test]
    fn a_negative_offset_clips_rather_than_failing() {
        // "Centre a watermark larger than the base" must work without the
        // caller doing the clipping arithmetic.
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[0, 0, 0, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[255, 0, 0, 255]);
        let op = Composite::at(-2, -2, Blend::Over);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();

        let pixel_at = |x: usize, y: usize| -> &[u8] { &out[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4] };
        assert_eq!(
            pixel_at(0, 0),
            [255, 0, 0, 255],
            "clipped overlay should cover here"
        );
        assert_eq!(pixel_at(3, 3), [0, 0, 0, 255], "beyond the clipped overlay");
    }

    #[test]
    fn an_overlay_entirely_outside_the_base_changes_nothing() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[1, 2, 3, 255]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 0, 0, 255]);
        let op = Composite::at(100, 100, Blend::Over);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();
        assert_eq!(out, base, "a missed overlay changed the base");
    }

    #[test]
    fn demand_asks_only_for_the_overlay_that_lands() {
        // A watermark in one corner must not make every tile pull the whole
        // overlay — that is the difference between a demand-driven engine and
        // a loop.
        let base = ImageDescriptor::new(1000, 1000, PixelFormat::Rgba8).unwrap();
        let overlay = ImageDescriptor::new(50, 50, PixelFormat::Rgba8).unwrap();
        let op = Composite::at(900, 900, Blend::Over);

        let far = op
            .input_regions(Region::new(0, 0, 100, 100), &[base, overlay])
            .unwrap();
        assert_eq!(
            far[0],
            Region::new(0, 0, 100, 100),
            "base demand is the output"
        );
        assert!(
            far[1].width <= 1 && far[1].height <= 1,
            "a tile the overlay misses should not demand it: {}",
            far[1]
        );

        let near = op
            .input_regions(Region::new(900, 900, 50, 50), &[base, overlay])
            .unwrap();
        assert_eq!(near[1], Region::new(0, 0, 50, 50), "overlapping tile");
    }

    #[test]
    fn the_source_blend_ignores_alpha() {
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgba8, &[9, 9, 9, 255]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[1, 2, 3, 0]);
        let op = Composite::at(0, 0, Blend::Source);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel, [1, 2, 3, 0], "Source should copy verbatim");
        }
    }

    #[test]
    fn an_opaque_format_composites_as_replacement() {
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgb8, &[0, 0, 0]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgb8, &[5, 6, 7]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(3) {
            assert_eq!(pixel, [5, 6, 7]);
        }
    }

    #[test]
    fn mismatched_formats_are_an_error() {
        let base = ImageDescriptor::new(4, 4, PixelFormat::Rgba8).unwrap();
        let overlay = ImageDescriptor::new(4, 4, PixelFormat::Rgb8).unwrap();
        assert!(
            Composite::over()
                .output_descriptor(&[base, overlay])
                .is_err()
        );
    }

    #[test]
    fn wide_formats_are_unsupported_rather_than_wrong() {
        let base = ImageDescriptor::new(4, 4, PixelFormat::Rgba16).unwrap();
        assert!(Composite::over().output_descriptor(&[base, base]).is_err());
    }

    #[test]
    fn one_input_is_a_graph_error() {
        let base = ImageDescriptor::new(4, 4, PixelFormat::Rgba8).unwrap();
        assert!(Composite::over().output_descriptor(&[base]).is_err());
        assert_eq!(Composite::over().arity(), 2);
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        let (base_desc, base) = solid(16, 12, PixelFormat::Rgba8, &[40, 80, 120, 200]);
        let (over_desc, over) = solid(7, 5, PixelFormat::Rgba8, &[255, 0, 0, 128]);
        let op = Composite::at(3, 2, Blend::Over);
        let (out_desc, whole) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();

        let base_buf = TileBuf::from_vec(base_desc.region(), base_desc.pixel, base).unwrap();
        let over_buf = TileBuf::from_vec(over_desc.region(), over_desc.pixel, over).unwrap();

        for (tw, th) in [(4_u32, 4_u32), (1, 12), (16, 1), (5, 3)] {
            let mut target = TileBuf::for_image(&out_desc).unwrap();
            let mut y = 0;
            while y < out_desc.height {
                let h = th.min(out_desc.height - y);
                let mut x = 0;
                while x < out_desc.width {
                    let w = tw.min(out_desc.width - x);
                    let region = Region::new(x, y, w, h);
                    let demand = op.input_regions(region, &[base_desc, over_desc]).unwrap();

                    let mut base_cut = TileBuf::zeroed(demand[0], base_desc.pixel).unwrap();
                    otf_pixels_core::copy_region(
                        &base_buf.as_tile().unwrap(),
                        &mut base_cut.as_tile_mut().unwrap(),
                        demand[0],
                    )
                    .unwrap();
                    let mut over_cut = TileBuf::zeroed(demand[1], over_desc.pixel).unwrap();
                    otf_pixels_core::copy_region(
                        &over_buf.as_tile().unwrap(),
                        &mut over_cut.as_tile_mut().unwrap(),
                        demand[1],
                    )
                    .unwrap();

                    let mut sub = TileBuf::zeroed(region, out_desc.pixel).unwrap();
                    op.compute(
                        &[base_cut.as_tile().unwrap(), over_cut.as_tile().unwrap()],
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
                "tiling at {tw}x{th} changed the pixels"
            );
        }
    }
}
