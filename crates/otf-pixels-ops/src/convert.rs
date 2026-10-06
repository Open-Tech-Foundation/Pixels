//! [`ConvertFormat`] — change an image's pixel format.
//!
//! Depth (8-bit, 16-bit, float) and layout (grey, grey with alpha, RGB,
//! RGBA) both convert. Depth rescales to the full range of the target, with
//! rounding to nearest; grey widens to RGB by repetition and RGB narrows to
//! grey by BT.601 luma, sharp's `greyscale`; alpha is added opaque, or
//! dropped without compositing (use `flatten` to composite against a colour
//! first).

use otf_pixels_core::{
    ChannelLayout, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, SampleKind, Tile,
    TileMut,
};
use std::sync::Arc;

/// Convert to a different [`PixelFormat`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertFormat {
    to: PixelFormat,
}

impl ConvertFormat {
    /// Convert whatever comes in to `to`.
    #[must_use]
    pub const fn to(to: PixelFormat) -> Self {
        Self { to }
    }
}

/// One sample at `index` of a pixel, as 0..=1.
fn read(pixel: &[u8], kind: SampleKind, index: usize) -> f32 {
    match kind {
        SampleKind::U8 => f32::from(pixel.get(index).copied().unwrap_or(0)) / 255.0,
        SampleKind::U16 => {
            let at = index * 2;
            let v = u16::from_ne_bytes([
                pixel.get(at).copied().unwrap_or(0),
                pixel.get(at + 1).copied().unwrap_or(0),
            ]);
            f32::from(v) / 65535.0
        }
        SampleKind::F32 => {
            let at = index * 4;
            let mut b = [0_u8; 4];
            if let Some(s) = pixel.get(at..at + 4) {
                b.copy_from_slice(s);
            }
            f32::from_ne_bytes(b)
        }
    }
}

fn write(pixel: &mut [u8], kind: SampleKind, index: usize, value: f32) {
    match kind {
        SampleKind::U8 => {
            if let Some(slot) = pixel.get_mut(index) {
                *slot = (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
        SampleKind::U16 => {
            let v = (value.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            if let Some(slot) = pixel.get_mut(index * 2..index * 2 + 2) {
                slot.copy_from_slice(&v.to_ne_bytes());
            }
        }
        SampleKind::F32 => {
            if let Some(slot) = pixel.get_mut(index * 4..index * 4 + 4) {
                slot.copy_from_slice(&value.to_ne_bytes());
            }
        }
    }
}

/// A pixel as RGBA in 0..=1.
fn to_rgba(pixel: &[u8], format: PixelFormat) -> [f32; 4] {
    let kind = format.sample_kind();
    let at = |i| read(pixel, kind, i);
    match format.layout() {
        ChannelLayout::Gray => [at(0), at(0), at(0), 1.0],
        ChannelLayout::GrayAlpha => [at(0), at(0), at(0), at(1)],
        ChannelLayout::Rgb => [at(0), at(1), at(2), 1.0],
        ChannelLayout::Rgba => [at(0), at(1), at(2), at(3)],
    }
}

fn from_rgba(rgba: [f32; 4], pixel: &mut [u8], format: PixelFormat) {
    let kind = format.sample_kind();
    let [r, g, b, a] = rgba;
    let luma = || 0.299 * r + 0.587 * g + 0.114 * b;
    match format.layout() {
        ChannelLayout::Gray => write(pixel, kind, 0, luma()),
        ChannelLayout::GrayAlpha => {
            write(pixel, kind, 0, luma());
            write(pixel, kind, 1, a);
        }
        ChannelLayout::Rgb => {
            for (i, v) in [r, g, b].into_iter().enumerate() {
                write(pixel, kind, i, v);
            }
        }
        ChannelLayout::Rgba => {
            for (i, v) in [r, g, b, a].into_iter().enumerate() {
                write(pixel, kind, i, v);
            }
        }
    }
}

/// 16-bit to 8-bit exactly as the rounding in `write` would, without floats:
/// the conversion every narrowing for output goes through.
fn narrow_16_to_8(v: u16) -> u8 {
    ((u32::from(v) * 255 + 32_767) / 65_535) as u8
}

impl Op for ConvertFormat {
    fn name(&self) -> &'static str {
        "convert_format"
    }

    /// Pointwise: resolution means nothing to it.
    fn rescaled(&self) -> Option<Arc<dyn Op>> {
        Some(Arc::new(*self))
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let [input] = inputs else {
            return Err(PixelsError::graph("convert_format takes exactly one input"));
        };
        ImageDescriptor::new(input.width, input.height, self.to)
    }

    fn input_regions(&self, output: Region, _inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let [input] = inputs else {
            return Err(PixelsError::graph("convert_format takes exactly one input"));
        };
        let (from, to) = (input.pixel(), self.to);
        let (in_bytes, out_bytes) = (from.bytes_per_pixel(), to.bytes_per_pixel());
        let region = output.region();
        let same_layout = from.layout() == to.layout();
        for y in region.y..region.y + region.height {
            let (Some(source), Some(target)) = (input.row(y), output.row_mut(y)) else {
                continue;
            };
            if from == to {
                let len = source.len().min(target.len());
                if let (Some(t), Some(s)) = (target.get_mut(..len), source.get(..len)) {
                    t.copy_from_slice(s);
                }
                continue;
            }
            if same_layout
                && from.sample_kind() == SampleKind::U16
                && to.sample_kind() == SampleKind::U8
            {
                // The hot narrowing for output, integer only.
                for (pair, slot) in source.chunks_exact(2).zip(target.iter_mut()) {
                    if let &[lo, hi] = pair {
                        *slot = narrow_16_to_8(u16::from_ne_bytes([lo, hi]));
                    }
                }
                continue;
            }
            for (from_px, to_px) in source
                .chunks_exact(in_bytes)
                .zip(target.chunks_exact_mut(out_bytes))
            {
                from_rgba(to_rgba(from_px, from), to_px, to);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;
    use otf_pixels_core::TileBuf;

    fn convert(from: PixelFormat, bytes: Vec<u8>, to: PixelFormat) -> Vec<u8> {
        let pixels = bytes.len() / from.bytes_per_pixel();
        let input = ImageDescriptor::new(pixels as u32, 1, from).unwrap();
        let op = ConvertFormat::to(to);
        let out = op.output_descriptor(&[input]).unwrap();
        let source = TileBuf::from_vec(input.region(), from, bytes).unwrap();
        let mut target = TileBuf::for_image(&out).unwrap();
        op.compute(
            &[source.as_tile().unwrap()],
            &mut target.as_tile_mut().unwrap(),
        )
        .unwrap();
        target.into_bytes()
    }

    fn wide(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    #[test]
    fn depth_rescales_with_rounding() {
        assert_eq!(
            convert(
                PixelFormat::Gray16,
                wide(&[0, 128, 32_896, 65_535]),
                PixelFormat::Gray8
            ),
            vec![0, 0, 128, 255]
        );
        // Every 8-bit value survives a round trip through 16 bits.
        let all: Vec<u8> = (0..=255).collect();
        let up = convert(PixelFormat::Gray8, all.clone(), PixelFormat::Gray16);
        assert_eq!(up[2..4], 257_u16.to_ne_bytes());
        assert_eq!(convert(PixelFormat::Gray16, up, PixelFormat::Gray8), all);
        // Floats clamp to the range.
        let floats: Vec<u8> = [-0.5_f32, 0.5, 2.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(
            convert(PixelFormat::RgbF32, floats, PixelFormat::Rgb8),
            vec![0, 128, 255]
        );
    }

    #[test]
    fn layouts_widen_narrow_and_add_or_drop_alpha() {
        assert_eq!(
            convert(PixelFormat::Gray8, vec![7], PixelFormat::Rgba8),
            vec![7, 7, 7, 255]
        );
        assert_eq!(
            convert(PixelFormat::Rgb8, vec![255, 0, 0], PixelFormat::Gray8),
            vec![76]
        );
        assert_eq!(
            convert(PixelFormat::Rgba8, vec![1, 2, 3, 4], PixelFormat::Rgb8),
            vec![1, 2, 3]
        );
        assert_eq!(
            convert(PixelFormat::GrayA8, vec![9, 100], PixelFormat::Rgba16),
            wide(&[9 * 257, 9 * 257, 9 * 257, 100 * 257])
        );
        assert_eq!(
            convert(
                PixelFormat::Rgba16,
                wide(&[65_535, 0, 0, 32_768]),
                PixelFormat::GrayA8
            ),
            vec![76, 128]
        );
    }
}
