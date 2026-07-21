//! Pointwise ops: [`Modulate`], [`ExtractChannel`] and [`Flatten`].
//!
//! Every op here reads one input pixel and writes one output pixel. That makes
//! them the easiest ops to get right and the most important ones to make fast:
//! they are memory-bound, so the whole game is keeping the inner loop free of
//! branches and letting the compiler widen it (ADR-0011).
//!
//! # Alpha
//!
//! SPEC §Formats: alpha is **unassociated** at API boundaries. `modulate`
//! therefore leaves alpha alone rather than scaling it, and `flatten`
//! composites against a background using straight alpha. An op that quietly
//! premultiplied would change what a subsequent `composite` means.

use otf_pixels_core::{
    ChannelLayout, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, SampleKind, Tile,
    TileMut,
};

/// Brightness, saturation and hue adjustment (SPEC §Core ops).
///
/// Saturation and hue are defined in HSV, which is what `modulate` means
/// everywhere else in this corner of the ecosystem. That is a deliberate
/// compatibility choice rather than a claim that HSV is the right colour
/// model: it is not perceptually uniform, and v2's ICC work is where a
/// principled version belongs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Modulate {
    /// Multiplier on value/lightness. 1.0 leaves it unchanged.
    pub brightness: f32,
    /// Multiplier on saturation. 1.0 leaves it unchanged, 0.0 is greyscale.
    pub saturation: f32,
    /// Rotation of hue in degrees.
    pub hue: f32,
}

impl Default for Modulate {
    fn default() -> Self {
        Self {
            brightness: 1.0,
            saturation: 1.0,
            hue: 0.0,
        }
    }
}

impl Modulate {
    /// A modulation that changes nothing.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Scale brightness by `factor`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `factor` is negative or not
    /// finite — a NaN multiplier would propagate silently into every pixel.
    pub fn with_brightness(mut self, factor: f32) -> Result<Self> {
        self.brightness = check_factor("brightness", factor)?;
        Ok(self)
    }

    /// Scale saturation by `factor`.
    ///
    /// # Errors
    ///
    /// As [`Modulate::with_brightness`].
    pub fn with_saturation(mut self, factor: f32) -> Result<Self> {
        self.saturation = check_factor("saturation", factor)?;
        Ok(self)
    }

    /// Rotate hue by `degrees`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `degrees` is not finite.
    pub fn with_hue(mut self, degrees: f32) -> Result<Self> {
        if !degrees.is_finite() {
            return Err(PixelsError::invalid_argument(
                "hue",
                format!("rotation must be finite, got {degrees}"),
            ));
        }
        self.hue = degrees;
        Ok(self)
    }

    /// Whether this modulation is the identity, and can be skipped entirely.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.brightness == 1.0 && self.saturation == 1.0 && self.hue % 360.0 == 0.0
    }
}

/// Reject a multiplier that would poison every pixel it touched.
fn check_factor(name: &'static str, factor: f32) -> Result<f32> {
    if !factor.is_finite() || factor < 0.0 {
        return Err(PixelsError::invalid_argument(
            name,
            format!("must be finite and non-negative, got {factor}"),
        ));
    }
    Ok(factor)
}

/// Convert RGB in 0..=1 to HSV, with hue in degrees.
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let hue = if delta <= f32::EPSILON {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let saturation = if max <= f32::EPSILON {
        0.0
    } else {
        delta / max
    };
    (hue, saturation, max)
}

/// The inverse of [`rgb_to_hsv`].
fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> (f32, f32, f32) {
    let hue = hue.rem_euclid(360.0);
    let c = value * saturation;
    let x = c * (1.0 - (((hue / 60.0) % 2.0) - 1.0).abs());
    let m = value - c;
    let (r, g, b) = match (hue / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

/// Apply the modulation to one colour, in 0..=1.
fn modulate_rgb(m: &Modulate, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (hue, saturation, value) = rgb_to_hsv(r, g, b);
    hsv_to_rgb(
        hue + m.hue,
        (saturation * m.saturation).clamp(0.0, 1.0),
        (value * m.brightness).clamp(0.0, 1.0),
    )
}

impl Op for Modulate {
    fn name(&self) -> &'static str {
        "modulate"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole(self.name(), inputs)?;
        Ok(*input)
    }

    fn input_regions(&self, output: Region, _inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = sole_tile(self.name(), inputs)?;
        let format = output.pixel();
        let region = output.region();
        let channels = format.channels();
        let colour = format.layout();

        for y in region.y..region.y + region.height {
            let Some(source) = input.row(y) else { continue };
            let Some(target) = output.row_mut(y) else {
                continue;
            };
            match format.sample_kind() {
                SampleKind::U8 => modulate_row_u8(self, source, target, channels, colour),
                SampleKind::U16 => modulate_row_u16(self, source, target, channels, colour),
                SampleKind::F32 => modulate_row_f32(self, source, target, channels, colour),
            }
        }
        Ok(())
    }
}

/// Whether a layout carries an alpha channel that must pass through untouched.
const fn has_alpha(layout: ChannelLayout) -> bool {
    matches!(layout, ChannelLayout::GrayAlpha | ChannelLayout::Rgba)
}

/// Whether a layout is greyscale, for which hue and saturation are meaningless.
const fn is_gray(layout: ChannelLayout) -> bool {
    matches!(layout, ChannelLayout::Gray | ChannelLayout::GrayAlpha)
}

fn modulate_row_u8(
    m: &Modulate,
    source: &[u8],
    target: &mut [u8],
    channels: usize,
    layout: ChannelLayout,
) {
    let colour_channels = if has_alpha(layout) {
        channels - 1
    } else {
        channels
    };
    for (from, to) in source
        .chunks_exact(channels)
        .zip(target.chunks_exact_mut(channels))
    {
        if is_gray(layout) {
            // Hue and saturation have no meaning on one channel; brightness
            // still does, and silently ignoring it would be surprising.
            let value = f32::from(from.first().copied().unwrap_or(0)) / 255.0;
            let scaled = (value * m.brightness).clamp(0.0, 1.0);
            if let Some(slot) = to.first_mut() {
                *slot = (scaled * 255.0 + 0.5) as u8;
            }
        } else {
            let r = f32::from(from.first().copied().unwrap_or(0)) / 255.0;
            let g = f32::from(from.get(1).copied().unwrap_or(0)) / 255.0;
            let b = f32::from(from.get(2).copied().unwrap_or(0)) / 255.0;
            let (r, g, b) = modulate_rgb(m, r, g, b);
            for (index, value) in [r, g, b].into_iter().enumerate().take(colour_channels) {
                if let Some(slot) = to.get_mut(index) {
                    *slot = (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
        // Alpha is unassociated (SPEC §Formats), so it passes through.
        if has_alpha(layout) {
            if let (Some(alpha), Some(slot)) = (from.get(channels - 1), to.get_mut(channels - 1)) {
                *slot = *alpha;
            }
        }
    }
}

fn modulate_row_u16(
    m: &Modulate,
    source: &[u8],
    target: &mut [u8],
    channels: usize,
    layout: ChannelLayout,
) {
    let pixel_bytes = channels * 2;
    for (from, to) in source
        .chunks_exact(pixel_bytes)
        .zip(target.chunks_exact_mut(pixel_bytes))
    {
        let read = |index: usize| -> f32 {
            let at = index * 2;
            let value = u16::from_ne_bytes([
                from.get(at).copied().unwrap_or(0),
                from.get(at + 1).copied().unwrap_or(0),
            ]);
            f32::from(value) / 65535.0
        };
        let mut write = |index: usize, value: f32| {
            let scaled = (value.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            let at = index * 2;
            for (offset, byte) in scaled.to_ne_bytes().iter().enumerate() {
                if let Some(slot) = to.get_mut(at + offset) {
                    *slot = *byte;
                }
            }
        };

        if is_gray(layout) {
            write(0, read(0) * m.brightness);
        } else {
            let (r, g, b) = modulate_rgb(m, read(0), read(1), read(2));
            write(0, r);
            write(1, g);
            write(2, b);
        }
        if has_alpha(layout) {
            let alpha = channels - 1;
            let at = alpha * 2;
            for offset in 0..2 {
                if let (Some(byte), Some(slot)) = (from.get(at + offset), to.get_mut(at + offset)) {
                    *slot = *byte;
                }
            }
        }
    }
}

fn modulate_row_f32(
    m: &Modulate,
    source: &[u8],
    target: &mut [u8],
    channels: usize,
    layout: ChannelLayout,
) {
    let pixel_bytes = channels * 4;
    for (from, to) in source
        .chunks_exact(pixel_bytes)
        .zip(target.chunks_exact_mut(pixel_bytes))
    {
        let read = |index: usize| -> f32 {
            let at = index * 4;
            let mut bytes = [0_u8; 4];
            for (slot, offset) in bytes.iter_mut().zip(0..4) {
                *slot = from.get(at + offset).copied().unwrap_or(0);
            }
            f32::from_ne_bytes(bytes)
        };
        let mut write = |index: usize, value: f32| {
            let at = index * 4;
            for (offset, byte) in value.to_ne_bytes().iter().enumerate() {
                if let Some(slot) = to.get_mut(at + offset) {
                    *slot = *byte;
                }
            }
        };

        if is_gray(layout) {
            write(0, read(0) * m.brightness);
        } else {
            let (r, g, b) = modulate_rgb(m, read(0), read(1), read(2));
            write(0, r);
            write(1, g);
            write(2, b);
        }
        if has_alpha(layout) {
            let at = (channels - 1) * 4;
            for offset in 0..4 {
                if let (Some(byte), Some(slot)) = (from.get(at + offset), to.get_mut(at + offset)) {
                    *slot = *byte;
                }
            }
        }
    }
}

/// Extract one channel as a greyscale image (SPEC §Core ops).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtractChannel {
    index: usize,
}

impl ExtractChannel {
    /// Extract channel `index`, counted from zero in memory order.
    ///
    /// The index is validated against the actual input when chained, since the
    /// channel count is not known here.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    /// The channel this op extracts.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }
}

impl Op for ExtractChannel {
    fn name(&self) -> &'static str {
        "extract_channel"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole(self.name(), inputs)?;
        if self.index >= input.pixel.channels() {
            return Err(PixelsError::invalid_argument(
                "index",
                format!(
                    "channel {} does not exist in {} ({} channels)",
                    self.index,
                    input.pixel,
                    input.pixel.channels()
                ),
            ));
        }
        let gray = match input.pixel.sample_kind() {
            SampleKind::U8 => PixelFormat::Gray8,
            SampleKind::U16 => PixelFormat::Gray16,
            // There is no single-channel float format in v1, so extraction
            // from a float image is refused rather than silently widened.
            SampleKind::F32 => {
                return Err(PixelsError::unsupported(format!(
                    "cannot extract a channel from {}: v1 has no float greyscale format",
                    input.pixel
                )));
            }
        };
        let mut out = *input;
        out.pixel = gray;
        Ok(out)
    }

    fn input_regions(&self, output: Region, _inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = sole_tile(self.name(), inputs)?;
        let region = output.region();
        let in_channels = input.pixel().channels();
        let sample = input.pixel().sample_kind().size();

        for y in region.y..region.y + region.height {
            let Some(source) = input.row(y) else { continue };
            let Some(target) = output.row_mut(y) else {
                continue;
            };
            for (pixel, slot) in source
                .chunks_exact(in_channels * sample)
                .zip(target.chunks_exact_mut(sample))
            {
                let at = self.index * sample;
                let Some(from) = pixel.get(at..at + sample) else {
                    continue;
                };
                slot.copy_from_slice(from);
            }
        }
        Ok(())
    }
}

/// Composite an image onto an opaque background, discarding alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Flatten {
    background: [u8; 3],
}

impl Flatten {
    /// Flatten onto an 8-bit RGB background.
    #[must_use]
    pub const fn onto(red: u8, green: u8, blue: u8) -> Self {
        Self {
            background: [red, green, blue],
        }
    }

    /// Flatten onto black, the usual default.
    #[must_use]
    pub const fn black() -> Self {
        Self::onto(0, 0, 0)
    }

    /// The background colour.
    #[must_use]
    pub const fn background(&self) -> [u8; 3] {
        self.background
    }
}

impl Op for Flatten {
    fn name(&self) -> &'static str {
        "flatten"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = sole(self.name(), inputs)?;
        let opaque = match input.pixel {
            PixelFormat::GrayA8 => PixelFormat::Gray8,
            PixelFormat::Rgba8 => PixelFormat::Rgb8,
            PixelFormat::Rgba16 => PixelFormat::Rgb16,
            PixelFormat::RgbaF32 => PixelFormat::RgbF32,
            // Already opaque: flatten is a no-op rather than an error, so it
            // can sit unconditionally in a pipeline.
            other => other,
        };
        let mut out = *input;
        out.pixel = opaque;
        Ok(out)
    }

    fn input_regions(&self, output: Region, _inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = sole_tile(self.name(), inputs)?;
        let region = output.region();
        let in_format = input.pixel();
        let out_format = output.pixel();

        if in_format == out_format {
            // Nothing to composite; copy through.
            for y in region.y..region.y + region.height {
                let (Some(source), Some(target)) = (input.row(y), output.row_mut(y)) else {
                    continue;
                };
                let len = target.len().min(source.len());
                if let (Some(from), Some(to)) = (source.get(..len), target.get_mut(..len)) {
                    to.copy_from_slice(from);
                }
            }
            return Ok(());
        }

        if in_format.sample_kind() != SampleKind::U8 {
            return Err(PixelsError::unsupported(format!(
                "flatten is implemented for 8-bit input; got {in_format}"
            )));
        }

        let in_channels = in_format.channels();
        let out_channels = out_format.channels();
        for y in region.y..region.y + region.height {
            let (Some(source), Some(target)) = (input.row(y), output.row_mut(y)) else {
                continue;
            };
            for (pixel, slot) in source
                .chunks_exact(in_channels)
                .zip(target.chunks_exact_mut(out_channels))
            {
                let alpha = u32::from(pixel.get(in_channels - 1).copied().unwrap_or(255));
                for channel in 0..out_channels {
                    let foreground = u32::from(pixel.get(channel).copied().unwrap_or(0));
                    let background = u32::from(self.background.get(channel).copied().unwrap_or(0));
                    // Straight-alpha source-over, rounded rather than
                    // truncated: `(f*a + b*(255-a) + 127) / 255`.
                    let blended = (foreground * alpha + background * (255 - alpha) + 127) / 255;
                    if let Some(out) = slot.get_mut(channel) {
                        *out = blended as u8;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Fetch the sole input descriptor an op was given.
fn sole<'a>(op: &str, inputs: &'a [ImageDescriptor]) -> Result<&'a ImageDescriptor> {
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

    fn image(
        width: u32,
        height: u32,
        format: PixelFormat,
        fill: &[u8],
    ) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let len = descriptor.byte_len().unwrap();
        let bytes = (0..len).map(|i| fill[i % fill.len()]).collect();
        (descriptor, bytes)
    }

    // -----------------------------------------------------------------
    // Modulate
    // -----------------------------------------------------------------

    #[test]
    fn the_identity_modulation_changes_nothing() {
        // The most important property of a colour op: doing nothing must
        // really do nothing, including no rounding drift.
        for format in [
            PixelFormat::Gray8,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
        ] {
            let (desc, bytes) = image(8, 8, format, &[13, 200, 77, 255, 4, 91]);
            let (_, out) = apply(&Modulate::identity(), &desc, &bytes).unwrap();
            assert_eq!(out, bytes, "{format} changed under the identity");
        }
    }

    #[test]
    fn hsv_round_trips_for_every_colour() {
        // The conversion is the substance of modulate; if it does not round
        // trip, every adjustment is wrong in a way that looks like a filter.
        for r in 0..8 {
            for g in 0..8 {
                for b in 0..8 {
                    let (rf, gf, bf) = (r as f32 / 7.0, g as f32 / 7.0, b as f32 / 7.0);
                    let (h, s, v) = rgb_to_hsv(rf, gf, bf);
                    let (r2, g2, b2) = hsv_to_rgb(h, s, v);
                    assert!(
                        (rf - r2).abs() < 1e-4 && (gf - g2).abs() < 1e-4 && (bf - b2).abs() < 1e-4,
                        "({rf},{gf},{bf}) -> ({h},{s},{v}) -> ({r2},{g2},{b2})"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_saturation_produces_grey() {
        let (desc, bytes) = image(4, 4, PixelFormat::Rgb8, &[200, 50, 30]);
        let m = Modulate::identity().with_saturation(0.0).unwrap();
        let (_, out) = apply(&m, &desc, &bytes).unwrap();
        for pixel in out.chunks_exact(3) {
            assert_eq!(pixel[0], pixel[1], "not grey: {pixel:?}");
            assert_eq!(pixel[1], pixel[2], "not grey: {pixel:?}");
        }
    }

    #[test]
    fn brightness_scales_and_saturates_rather_than_wrapping() {
        // A doubled bright pixel must clamp to white, not wrap to black.
        let (desc, bytes) = image(4, 4, PixelFormat::Rgb8, &[200, 200, 200]);
        let m = Modulate::identity().with_brightness(2.0).unwrap();
        let (_, out) = apply(&m, &desc, &bytes).unwrap();
        assert!(
            out.iter().all(|&v| v == 255),
            "expected white, got {:?}",
            &out[..3]
        );

        let m = Modulate::identity().with_brightness(0.5).unwrap();
        let (_, dim) = apply(&m, &desc, &bytes).unwrap();
        assert!(
            dim.iter().all(|&v| v == 100),
            "expected 100, got {:?}",
            &dim[..3]
        );
    }

    #[test]
    fn a_full_hue_rotation_is_the_identity() {
        let (desc, bytes) = image(4, 4, PixelFormat::Rgb8, &[200, 50, 30]);
        let m = Modulate::identity().with_hue(360.0).unwrap();
        let (_, out) = apply(&m, &desc, &bytes).unwrap();
        for (a, b) in out.iter().zip(&bytes) {
            assert!(a.abs_diff(*b) <= 1, "360 degrees changed {b} to {a}");
        }
    }

    #[test]
    fn alpha_passes_through_untouched() {
        // SPEC §Formats: alpha is unassociated, so a colour op must not
        // scale it. Scaling it here would silently premultiply.
        let (desc, bytes) = image(4, 4, PixelFormat::Rgba8, &[200, 50, 30, 137]);
        let m = Modulate::identity()
            .with_brightness(0.25)
            .unwrap()
            .with_saturation(2.0)
            .unwrap();
        let (_, out) = apply(&m, &desc, &bytes).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel[3], 137, "alpha was modulated");
        }
    }

    #[test]
    fn greyscale_takes_brightness_but_not_hue() {
        let (desc, bytes) = image(4, 4, PixelFormat::Gray8, &[100]);
        let m = Modulate::identity()
            .with_brightness(0.5)
            .unwrap()
            .with_hue(180.0)
            .unwrap();
        let (_, out) = apply(&m, &desc, &bytes).unwrap();
        assert!(out.iter().all(|&v| v == 50), "got {:?}", &out[..4]);
    }

    #[test]
    fn a_non_finite_factor_is_an_error_not_a_poisoned_image() {
        assert!(Modulate::identity().with_brightness(f32::NAN).is_err());
        assert!(Modulate::identity().with_brightness(f32::INFINITY).is_err());
        assert!(Modulate::identity().with_brightness(-1.0).is_err());
        assert!(Modulate::identity().with_saturation(f32::NAN).is_err());
        assert!(Modulate::identity().with_hue(f32::NAN).is_err());
    }

    // -----------------------------------------------------------------
    // ExtractChannel
    // -----------------------------------------------------------------

    #[test]
    fn extracting_a_channel_gives_that_channel() {
        let (desc, bytes) = image(4, 4, PixelFormat::Rgba8, &[10, 20, 30, 40]);
        for index in 0..4 {
            let (out_desc, out) = apply(&ExtractChannel::new(index), &desc, &bytes).unwrap();
            assert_eq!(out_desc.pixel, PixelFormat::Gray8);
            let expected = (index as u8 + 1) * 10;
            assert!(
                out.iter().all(|&v| v == expected),
                "channel {index} gave {:?}",
                &out[..4]
            );
        }
    }

    #[test]
    fn extracting_sixteen_bit_channels_preserves_precision() {
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::Rgb16).unwrap();
        let mut bytes = Vec::new();
        for _ in 0..16 {
            for value in [1000_u16, 30000, 65535] {
                bytes.extend_from_slice(&value.to_ne_bytes());
            }
        }
        let (out_desc, out) = apply(&ExtractChannel::new(1), &descriptor, &bytes).unwrap();
        assert_eq!(out_desc.pixel, PixelFormat::Gray16);
        for pair in out.chunks_exact(2) {
            assert_eq!(u16::from_ne_bytes([pair[0], pair[1]]), 30000);
        }
    }

    #[test]
    fn extracting_a_channel_that_does_not_exist_is_an_error() {
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::Rgb8).unwrap();
        let error = ExtractChannel::new(3)
            .output_descriptor(std::slice::from_ref(&descriptor))
            .unwrap_err();
        assert!(error.to_string().contains("does not exist"), "{error}");
    }

    #[test]
    fn extracting_from_a_float_image_is_unsupported_not_wrong() {
        // v1 has no float greyscale format, so this is refused rather than
        // quietly widened to RGB.
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::RgbF32).unwrap();
        assert!(
            ExtractChannel::new(0)
                .output_descriptor(std::slice::from_ref(&descriptor))
                .is_err()
        );
    }

    // -----------------------------------------------------------------
    // Flatten
    // -----------------------------------------------------------------

    #[test]
    fn flattening_an_opaque_pixel_keeps_its_colour() {
        let (desc, bytes) = image(4, 4, PixelFormat::Rgba8, &[200, 100, 50, 255]);
        let (out_desc, out) = apply(&Flatten::black(), &desc, &bytes).unwrap();
        assert_eq!(out_desc.pixel, PixelFormat::Rgb8);
        for pixel in out.chunks_exact(3) {
            assert_eq!(pixel, [200, 100, 50], "opaque pixel changed");
        }
    }

    #[test]
    fn flattening_a_transparent_pixel_gives_the_background() {
        let (desc, bytes) = image(4, 4, PixelFormat::Rgba8, &[200, 100, 50, 0]);
        let (_, out) = apply(&Flatten::onto(9, 8, 7), &desc, &bytes).unwrap();
        for pixel in out.chunks_exact(3) {
            assert_eq!(
                pixel,
                [9, 8, 7],
                "transparent pixel did not take the background"
            );
        }
    }

    #[test]
    fn flattening_at_half_alpha_is_the_midpoint() {
        // 128/255 is just over half, so a 0-and-255 blend rounds to 128.
        let (desc, bytes) = image(4, 4, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (_, out) = apply(&Flatten::black(), &desc, &bytes).unwrap();
        for pixel in out.chunks_exact(3) {
            assert_eq!(pixel, [128, 128, 128], "blend is not the midpoint");
        }
    }

    #[test]
    fn flattening_an_image_without_alpha_is_a_pass_through() {
        // Flatten must be safe to put in a pipeline unconditionally.
        let (desc, bytes) = image(4, 4, PixelFormat::Rgb8, &[1, 2, 3]);
        let (out_desc, out) = apply(&Flatten::onto(200, 200, 200), &desc, &bytes).unwrap();
        assert_eq!(out_desc.pixel, PixelFormat::Rgb8);
        assert_eq!(out, bytes);
    }

    #[test]
    fn flattening_greyscale_alpha_gives_greyscale() {
        let (desc, bytes) = image(4, 4, PixelFormat::GrayA8, &[255, 0]);
        let (out_desc, out) = apply(&Flatten::onto(30, 0, 0), &desc, &bytes).unwrap();
        assert_eq!(out_desc.pixel, PixelFormat::Gray8);
        assert!(out.iter().all(|&v| v == 30), "got {:?}", &out[..4]);
    }
}
