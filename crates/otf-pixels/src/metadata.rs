//! What a source file says about itself, and what its pixels measure: the
//! facade's answers to sharp's `metadata()` and `stats()`.

#![allow(
    clippy::indexing_slicing,
    reason = "sample indices are bounded by the descriptor's own dimensions"
)]

use std::sync::Arc;

use otf_pixels_core::{
    Animation, Background, ChannelLayout, Comment, Density, Format, ImageDescriptor, Orientation,
    PixelFormat, SampleKind, SourceInfo,
};

/// What a source file says about itself, read from its headers (sharp's
/// `metadata()`). Nothing here costs a pixel decode.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SourceMetadata {
    /// The file's format.
    pub format: Format,
    /// The file's size in bytes, when it was opened from a path.
    pub size: Option<u64>,
    /// Width as stored, before any orientation is applied.
    pub width: u32,
    /// Height as stored.
    pub height: u32,
    /// The pixel format the decoder produces.
    pub pixel: PixelFormat,
    /// Channels in that pixel format, alpha included.
    pub channels: usize,
    /// Whether there is an alpha channel.
    pub has_alpha: bool,
    /// Bits per stored sample, which may be fewer than the decoder's output
    /// (a 4-bit PNG, a 10-bit AVIF).
    pub bits_per_sample: Option<u8>,
    /// How the image must be turned to display upright; EXIF's 1–8 as
    /// [`Orientation::exif`] gives it.
    pub orientation: Orientation,
    /// Pages in the document, at least 1 (multi-page TIFF).
    pub pages: u32,
    /// The animation's frames, loop count and delays, for more than one
    /// frame.
    pub animation: Option<Animation>,
    /// The stated resolution, if any.
    pub density: Option<Density>,
    /// Progressive JPEG, or interlaced PNG or GIF.
    pub progressive: bool,
    /// Samples index a palette.
    pub palette: bool,
    /// Chroma subsampling as sharp spells it (`"4:2:0"`, `"4:4:4"`, ...).
    pub chroma_subsampling: Option<&'static str>,
    /// The background colour the file names (PNG `bKGD`, GIF).
    pub background: Option<Background>,
    /// The compression scheme where a format has several.
    pub compression: Option<&'static str>,
    /// Text entries (PNG text chunks, JPEG and GIF comments).
    pub comments: Vec<Comment>,
    /// The embedded ICC profile.
    pub icc: Option<Arc<[u8]>>,
    /// The EXIF block, TIFF-structured, as the file stores it.
    pub exif: Option<Arc<[u8]>>,
    /// The XMP packet.
    pub xmp: Option<Arc<[u8]>>,
}

impl SourceMetadata {
    #[allow(
        clippy::too_many_arguments,
        reason = "everything a decoder reports, once"
    )]
    pub(crate) fn new(
        format: Format,
        stored: ImageDescriptor,
        orientation: Orientation,
        pages: u32,
        animation: Option<Animation>,
        icc: Option<&[u8]>,
        exif: Option<&[u8]>,
        xmp: Option<Arc<[u8]>>,
        info: SourceInfo,
    ) -> Self {
        Self {
            format,
            size: None,
            width: stored.width,
            height: stored.height,
            pixel: stored.pixel,
            channels: stored.pixel.channels(),
            has_alpha: stored.pixel.has_alpha(),
            bits_per_sample: info.bits_per_sample,
            orientation,
            pages,
            animation,
            density: info.density,
            progressive: info.progressive,
            palette: info.palette,
            chroma_subsampling: info.chroma_subsampling,
            background: info.background,
            compression: info.compression,
            comments: info.comments,
            icc: icc.map(Arc::from),
            exif: exif.map(Arc::from),
            xmp,
        }
    }

    /// Whether the file carries an ICC profile (sharp's `hasProfile`).
    #[must_use]
    pub const fn has_profile(&self) -> bool {
        self.icc.is_some()
    }

    /// Width once turned upright, as sharp's `autoOrient` reports it.
    #[must_use]
    pub const fn oriented_width(&self) -> u32 {
        if self.orientation.transposes() {
            self.height
        } else {
            self.width
        }
    }

    /// Height once turned upright.
    #[must_use]
    pub const fn oriented_height(&self) -> u32 {
        if self.orientation.transposes() {
            self.width
        } else {
            self.height
        }
    }
}

/// One channel's statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ChannelStats {
    /// The smallest sample.
    pub min: f64,
    /// The largest sample.
    pub max: f64,
    /// The sum of every sample.
    pub sum: f64,
    /// The sum of every sample squared.
    pub squares_sum: f64,
    /// The mean.
    pub mean: f64,
    /// The sample standard deviation.
    pub stdev: f64,
    /// Where the minimum first occurs, in raster order.
    pub min_x: u32,
    /// Its row.
    pub min_y: u32,
    /// Where the maximum first occurs.
    pub max_x: u32,
    /// Its row.
    pub max_y: u32,
}

/// Pixel statistics of an image (sharp's `stats()`), in its own sample
/// scale: 0–255 for 8-bit, 0–65535 for 16-bit, 0–1 for float.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Stats {
    /// Each channel, alpha included, in order.
    pub channels: Vec<ChannelStats>,
    /// No alpha channel, or every alpha sample opaque.
    pub is_opaque: bool,
    /// Shannon entropy, in bits, of the greyscale histogram.
    pub entropy: f64,
    /// The standard deviation of the greyscale Laplacian: higher is sharper.
    pub sharpness: f64,
    /// The most common sRGB colour, by a 16-level-per-channel histogram,
    /// each channel the centre of its bin.
    pub dominant: [u8; 3],
}

/// Linear light from an sRGB-encoded sample in 0..=1.
fn to_linear(v: f64) -> f64 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB encoding of linear light in 0..=1.
fn from_linear(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

impl Stats {
    /// Measure `pixels`, packed as `descriptor` lays them out.
    #[must_use]
    pub fn of(descriptor: &ImageDescriptor, pixels: &[u8]) -> Self {
        let pixel = descriptor.pixel;
        let channels = pixel.channels();
        let (width, height) = (descriptor.width as usize, descriptor.height as usize);
        let n = width * height;
        let kind = pixel.sample_kind();
        let full = match kind {
            SampleKind::U8 => 255.0,
            SampleKind::U16 => 65_535.0,
            _ => 1.0,
        };
        let sample = |i: usize| -> f64 {
            match kind {
                SampleKind::U8 => f64::from(pixels[i]),
                SampleKind::U16 => {
                    f64::from(u16::from_ne_bytes([pixels[2 * i], pixels[2 * i + 1]]))
                }
                _ => f64::from(f32::from_ne_bytes([
                    pixels[4 * i],
                    pixels[4 * i + 1],
                    pixels[4 * i + 2],
                    pixels[4 * i + 3],
                ])),
            }
        };
        let mut stats = Vec::with_capacity(channels);
        for c in 0..channels {
            let (mut min, mut max, mut sum, mut squares) =
                (f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0);
            let (mut min_at, mut max_at) = (0, 0);
            for p in 0..n {
                let v = sample(p * channels + c);
                if v < min {
                    min = v;
                    min_at = p;
                }
                if v > max {
                    max = v;
                    max_at = p;
                }
                sum += v;
                squares += v * v;
            }
            let count = n as f64;
            let mean = if n > 0 { sum / count } else { 0.0 };
            let variance = if n > 1 {
                ((squares - sum * sum / count) / (count - 1.0)).max(0.0)
            } else {
                0.0
            };
            let at = |p: usize| ((p % width.max(1)) as u32, (p / width.max(1)) as u32);
            let ((min_x, min_y), (max_x, max_y)) = (at(min_at), at(max_at));
            stats.push(ChannelStats {
                min: if n > 0 { min } else { 0.0 },
                max: if n > 0 { max } else { 0.0 },
                sum,
                squares_sum: squares,
                mean,
                stdev: variance.sqrt(),
                min_x,
                min_y,
                max_x,
                max_y,
            });
        }
        let layout = pixel.layout();
        let alpha = layout.has_alpha().then(|| channels - 1);
        let is_opaque = alpha.is_none_or(|a| stats[a].min >= full);
        let colour = layout != ChannelLayout::Gray && layout != ChannelLayout::GrayAlpha;

        // Greyscale as sharp takes it: luminance of linear light, sRGB
        // encoded again, at the image's own depth (float as 16-bit).
        let levels: usize = if kind == SampleKind::U8 { 256 } else { 65_536 };
        let top = (levels - 1) as f64;
        let grey: Vec<u32> = (0..n)
            .map(|p| {
                let v = if colour {
                    let ch =
                        |c: usize| to_linear((sample(p * channels + c) / full).clamp(0.0, 1.0));
                    from_linear(0.2126 * ch(0) + 0.7152 * ch(1) + 0.0722 * ch(2))
                } else {
                    (sample(p * channels) / full).clamp(0.0, 1.0)
                };
                (v * top).round() as u32
            })
            .collect();
        let mut histogram = vec![0_u64; levels];
        for &g in &grey {
            histogram[g as usize] += 1;
        }
        let entropy = histogram
            .iter()
            .filter(|&&h| h > 0)
            .map(|&h| {
                let p = h as f64 / n as f64;
                -p * p.log2()
            })
            .sum();

        // The Laplacian, scaled by 9 and clipped to the depth as libvips's
        // integer convolution does, edges extended.
        let sharpness = if n > 1 {
            let at = |x: isize, y: isize| -> f64 {
                let x = x.clamp(0, width as isize - 1) as usize;
                let y = y.clamp(0, height as isize - 1) as usize;
                f64::from(grey[y * width + x])
            };
            let (mut s, mut s2) = (0.0, 0.0);
            for y in 0..height as isize {
                for x in 0..width as isize {
                    let l =
                        at(x, y - 1) + at(x - 1, y) + at(x + 1, y) + at(x, y + 1) - 4.0 * at(x, y);
                    let v = (l / 9.0).round().clamp(0.0, top);
                    s += v;
                    s2 += v * v;
                }
            }
            let count = n as f64;
            ((s2 - s * s / count) / (count - 1.0)).max(0.0).sqrt()
        } else {
            0.0
        };

        // Dominant colour over a 16x16x16 histogram of 8-bit sRGB.
        let mut bins = vec![0_u32; 4096];
        for p in 0..n {
            let byte = |c: usize| {
                ((sample(p * channels + c) / full).clamp(0.0, 1.0) * 255.0).round() as usize
            };
            let (r, g, b) = if colour {
                (byte(0), byte(1), byte(2))
            } else {
                (byte(0), byte(0), byte(0))
            };
            bins[(r >> 4) * 256 + (g >> 4) * 16 + (b >> 4)] += 1;
        }
        let best = bins
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0)))
            .map_or(0, |(i, _)| i);
        let centre = |bin: usize| (bin * 16 + 8) as u8;
        let dominant = [
            centre(best / 256),
            centre(best / 16 % 16),
            centre(best % 16),
        ];

        Self {
            channels: stats,
            is_opaque,
            entropy,
            sharpness,
            dominant,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;

    #[test]
    fn channel_statistics_follow_their_definitions() {
        let d = ImageDescriptor::new(2, 2, PixelFormat::Rgba8).unwrap();
        let px = [10, 0, 0, 255, 20, 0, 0, 255, 30, 0, 0, 255, 40, 200, 0, 128];
        let s = Stats::of(&d, &px);
        let r = s.channels[0];
        assert_eq!((r.min, r.max, r.sum, r.mean), (10.0, 40.0, 100.0, 25.0));
        assert_eq!(r.squares_sum, 100.0 + 400.0 + 900.0 + 1600.0);
        assert!((r.stdev - (500.0_f64 / 3.0).sqrt()).abs() < 1e-9);
        assert_eq!((r.min_x, r.min_y, r.max_x, r.max_y), (0, 0, 1, 1));
        assert!(!s.is_opaque);
        assert_eq!(s.channels.len(), 4);
    }

    #[test]
    fn a_flat_image_has_no_entropy_and_no_sharpness() {
        let d = ImageDescriptor::new(8, 8, PixelFormat::Rgb8).unwrap();
        let px = [200_u8, 30, 90].repeat(64);
        let s = Stats::of(&d, &px);
        assert_eq!(s.entropy, 0.0);
        assert_eq!(s.sharpness, 0.0);
        assert!(s.is_opaque);
        assert_eq!(s.dominant, [200, 24, 88]);
    }

    #[test]
    fn a_two_level_image_has_one_bit_of_entropy_and_edges_are_sharp() {
        let d = ImageDescriptor::new(8, 8, PixelFormat::Gray8).unwrap();
        let px: Vec<u8> = (0..64).map(|i| if (i % 8) < 4 { 0 } else { 255 }).collect();
        let s = Stats::of(&d, &px);
        assert!((s.entropy - 1.0).abs() < 1e-9);
        assert!(s.sharpness > 0.0);
    }
}
