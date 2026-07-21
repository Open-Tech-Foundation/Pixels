//! The GIF encoder: single frame, with palette quantization.
//!
//! SPEC §Formats fixes the scope: "encode is single-frame + palette
//! quantization. Animation pipelines are v2." An encoder that accepted frames
//! would need a frame model in the engine, which is the thing v2 is for.
//!
//! # Memory
//!
//! Encoding is **internally buffered**. Quantization needs a histogram of the
//! whole image before it can choose a palette, and dithering carries error
//! forward between rows, so there is no prefix of the image that can be
//! encoded before the rest has been seen. That is ADR-0005's allowance for
//! formats that leave no choice, and it is a property of palette quantization
//! rather than of this implementation.

use otf_pixels_compress::LzwEncoder;
use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

use crate::format::{SIGNATURE_89A, label, write_sub_blocks};
use crate::quantize::{Dither, build_palette, quantize};

/// Encodes a single-frame GIF.
#[derive(Debug)]
pub struct GifEncoder {
    dither: Dither,
    /// Palette entries to aim for, at most 256.
    colours: usize,
    state: Option<State>,
}

/// Everything fixed once the descriptor is known.
#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    /// The image as RGB8, accumulated until `finish`.
    rgb: Vec<u8>,
    rows_written: u32,
}

impl Default for GifEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl GifEncoder {
    /// An encoder with a full 256-colour palette and dithering.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dither: Dither::FloydSteinberg,
            colours: 256,
            state: None,
        }
    }

    /// An encoder aiming for at most `colours` palette entries.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] outside 2..=256.
    pub fn with_colours(mut self, colours: usize) -> Result<Self> {
        if !(2..=256).contains(&colours) {
            return Err(PixelsError::invalid_argument(
                "colours",
                format!("palette size must be in 2..=256, got {colours}"),
            ));
        }
        self.colours = colours;
        Ok(self)
    }

    /// An encoder with dithering disabled.
    #[must_use]
    pub const fn without_dithering(mut self) -> Self {
        self.dither = Dither::None;
        self
    }

    /// An encoder configured from generic encode options.
    ///
    /// GIF has no quality dial in the JPEG sense — it is lossless *given a
    /// palette*, and the palette is where all the loss lives. So quality is
    /// read as palette size, mapping 1..=100 onto 2..=256 entries, which is
    /// the axis that actually trades size for fidelity.
    #[must_use]
    pub fn from_options(options: &EncodeOptions) -> Self {
        let quality = u32::from(options.quality.clamp(1, 100));
        let colours = 2 + (quality - 1) * 254 / 99;
        Self {
            dither: Dither::FloydSteinberg,
            colours: colours as usize,
            state: None,
        }
    }

    /// The palette size this encoder aims for.
    #[must_use]
    pub const fn colours(&self) -> usize {
        self.colours
    }
}

/// Convert one row of `format` into RGB8, appending to `out`.
///
/// Alpha is composited against black rather than dropped. GIF's only
/// transparency is a palette index, so a translucent pixel has to become
/// *something*; blending against black is what viewers show for a flattened
/// image and is the least surprising of the available wrong answers.
fn append_as_rgb(row: &[u8], format: PixelFormat, out: &mut Vec<u8>) -> Result<()> {
    let channels = format.channels();
    match format {
        PixelFormat::Rgb8 => out.extend_from_slice(row),
        PixelFormat::Rgba8 => {
            for pixel in row.chunks_exact(channels) {
                let alpha = u32::from(pixel.get(3).copied().unwrap_or(255));
                for channel in 0..3 {
                    let value = u32::from(pixel.get(channel).copied().unwrap_or(0));
                    out.push(((value * alpha + 127) / 255) as u8);
                }
            }
        }
        PixelFormat::Gray8 => {
            for &value in row {
                out.extend_from_slice(&[value, value, value]);
            }
        }
        PixelFormat::GrayA8 => {
            for pixel in row.chunks_exact(2) {
                let alpha = u32::from(pixel.get(1).copied().unwrap_or(255));
                let value = u32::from(pixel.first().copied().unwrap_or(0));
                let blended = ((value * alpha + 127) / 255) as u8;
                out.extend_from_slice(&[blended, blended, blended]);
            }
        }
        other => {
            return Err(PixelsError::unsupported(format!(
                "GIF encoding needs an 8-bit format; got {other}. Convert first."
            )));
        }
    }
    Ok(())
}

impl Encoder for GifEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        // GIF dimensions are 16-bit; a larger image cannot be represented at
        // all, so this is a format limit rather than a policy.
        if desc.width > u32::from(u16::MAX) || desc.height > u32::from(u16::MAX) {
            return Err(PixelsError::unsupported(format!(
                "GIF dimensions are 16-bit; {}x{} does not fit",
                desc.width, desc.height
            )));
        }
        // Reject the format here rather than on the first row, so a caller
        // learns before streaming an image.
        let mut probe = Vec::new();
        append_as_rgb(&[], desc.pixel, &mut probe)?;

        // Nothing is written yet: the palette is not known until every pixel
        // has been seen, and the palette is part of the header.
        self.state = Some(State {
            descriptor: *desc,
            rgb: Vec::with_capacity(desc.width as usize * desc.height as usize * 3),
            rows_written: 0,
        });
        Ok(())
    }

    fn write_row(&mut self, row: &[u8], _sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "row",
                "write_row called before write_header",
            ));
        };
        let expected = state.descriptor.row_bytes();
        if row.len() != expected {
            return Err(PixelsError::invalid_argument(
                "row",
                format!("row is {} bytes, expected {expected}", row.len()),
            ));
        }
        if state.rows_written >= state.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "row",
                format!("more than {} rows written", state.descriptor.height),
            ));
        }
        append_as_rgb(row, state.descriptor.pixel, &mut state.rgb)?;
        state.rows_written += 1;
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::malformed(
                "gif",
                "finish called before write_header",
            ));
        };
        if state.rows_written != state.descriptor.height {
            return Err(PixelsError::malformed(
                "gif",
                format!(
                    "{} of {} rows written; a partial image is never emitted",
                    state.rows_written, state.descriptor.height
                ),
            ));
        }

        let width = state.descriptor.width;
        let height = state.descriptor.height;
        let palette = build_palette(&state.rgb, self.colours);
        let indices = quantize(&state.rgb, width as usize, &palette, self.dither);
        let table = palette.padded();
        let bits = palette.code_bits();

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&SIGNATURE_89A);
        out.extend_from_slice(&(width as u16).to_le_bytes());
        out.extend_from_slice(&(height as u16).to_le_bytes());
        // Global table present, colour resolution 8 bits, table size 2^(n+1).
        out.push(0x80 | 0x70 | ((bits - 1) as u8));
        out.push(0); // background index
        out.push(0); // pixel aspect ratio: unspecified
        for entry in &table {
            out.extend_from_slice(entry);
        }

        // Image descriptor: one frame covering the whole canvas, no local
        // table, not interlaced.
        out.push(label::IMAGE);
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(&(width as u16).to_le_bytes());
        out.extend_from_slice(&(height as u16).to_le_bytes());
        out.push(0);

        // LZW's minimum code width must be at least 2, even for a 2-colour
        // image: a width of 1 leaves no room for both the clear and end codes.
        let minimum_width = bits.max(2);
        out.push(minimum_width as u8);
        let compressed = LzwEncoder::gif(minimum_width)
            .map_err(crate::compress_error)?
            .encode(&indices);
        write_sub_blocks(&mut out, &compressed);
        out.push(label::TRAILER);

        sink.write_all(&out)?;
        sink.flush()?;

        // Release the raster now rather than at drop.
        state.rgb = Vec::new();
        Ok(())
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
    use crate::decoder::GifDecoder;
    use otf_pixels_core::{Decoder, Limits};

    fn encode(desc: &ImageDescriptor, raster: &[u8], encoder: GifEncoder) -> Result<Vec<u8>> {
        let mut encoder = encoder;
        let mut out: Vec<u8> = Vec::new();
        encoder.write_header(desc, &mut out)?;
        for row in raster.chunks_exact(desc.row_bytes()) {
            encoder.write_row(row, &mut out)?;
        }
        encoder.finish(&mut out)?;
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<(ImageDescriptor, Vec<u8>)> {
        let mut decoder = GifDecoder::new(bytes, Limits::default())?;
        let descriptor = decoder.descriptor();
        let mut raster = Vec::new();
        let mut row = vec![0_u8; descriptor.row_bytes()];
        for _ in 0..descriptor.height {
            decoder.read_row(&mut row)?;
            raster.extend_from_slice(&row);
        }
        Ok((descriptor, raster))
    }

    /// An image using at most `colours` distinct values, so it can round-trip
    /// through a palette exactly.
    fn flat_art(width: u32, height: u32, colours: usize) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();
        let mut bytes = Vec::new();
        for i in 0..(width * height) as usize {
            // Spread `colours` distinct values across the full range without
            // leaving u8, so the fixture stays legal for any palette size.
            let n = (i % colours) as u32;
            let spread = (n * 255 / colours.max(1) as u32) as u8;
            bytes.extend_from_slice(&[spread, 255 - spread, spread / 2]);
        }
        (descriptor, bytes)
    }

    #[test]
    fn flat_colour_art_round_trips_exactly() {
        // The case GIF is still genuinely used for. Fewer distinct colours
        // than the palette holds means the encoding must be lossless.
        for colours in [2_usize, 5, 16, 200] {
            let (descriptor, raster) = flat_art(23, 17, colours);
            let bytes = encode(&descriptor, &raster, GifEncoder::new())
                .unwrap_or_else(|e| panic!("{colours} colours: {e}"));
            let (out_desc, decoded) = decode(&bytes).unwrap();
            assert_eq!((out_desc.width, out_desc.height), (23, 17));
            assert_eq!(out_desc.pixel, PixelFormat::Rgba8);

            for (index, (want, got)) in raster
                .chunks_exact(3)
                .zip(decoded.chunks_exact(4))
                .enumerate()
            {
                assert_eq!(&got[..3], want, "{colours} colours: pixel {index} changed");
                assert_eq!(got[3], 255, "pixel {index} lost opacity");
            }
        }
    }

    #[test]
    fn the_output_starts_with_a_signature_and_ends_with_a_trailer() {
        let (descriptor, raster) = flat_art(4, 4, 4);
        let bytes = encode(&descriptor, &raster, GifEncoder::new()).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
        assert_eq!(bytes.last().copied(), Some(label::TRAILER));
    }

    #[test]
    fn a_photographic_image_round_trips_approximately() {
        // 256 colours cannot represent a gradient exactly, so the check is
        // that it is close rather than equal.
        let descriptor = ImageDescriptor::new(64, 64, PixelFormat::Rgb8).unwrap();
        let mut raster = Vec::new();
        for y in 0..64_u32 {
            for x in 0..64_u32 {
                raster.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]);
            }
        }
        let bytes = encode(&descriptor, &raster, GifEncoder::new()).unwrap();
        let (_, decoded) = decode(&bytes).unwrap();

        let mut total = 0_u64;
        for (want, got) in raster.chunks_exact(3).zip(decoded.chunks_exact(4)) {
            for channel in 0..3 {
                total += u64::from(want[channel].abs_diff(got[channel]));
            }
        }
        let mean = total as f64 / (raster.len() as f64);
        assert!(mean < 8.0, "mean per-channel error was {mean}");
    }

    #[test]
    fn every_input_format_encodes() {
        for format in [
            PixelFormat::Gray8,
            PixelFormat::GrayA8,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
        ] {
            let descriptor = ImageDescriptor::new(8, 8, format).unwrap();
            let raster = vec![128_u8; descriptor.byte_len().unwrap()];
            let bytes = encode(&descriptor, &raster, GifEncoder::new())
                .unwrap_or_else(|e| panic!("{format}: {e}"));
            let (out_desc, _) = decode(&bytes).unwrap();
            assert_eq!((out_desc.width, out_desc.height), (8, 8), "{format}");
        }
    }

    #[test]
    fn a_wide_format_is_unsupported_not_silently_narrowed() {
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::Rgb16).unwrap();
        let mut encoder = GifEncoder::new();
        let mut out = Vec::new();
        let error = encoder.write_header(&descriptor, &mut out).unwrap_err();
        assert!(error.to_string().contains("8-bit"), "{error}");
        assert!(
            out.is_empty(),
            "nothing should be written for a rejected format"
        );
    }

    #[test]
    fn a_short_image_is_an_error_not_a_truncated_gif() {
        let (descriptor, _) = flat_art(4, 4, 4);
        let mut encoder = GifEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&descriptor, &mut out).unwrap();
        encoder.write_row(&[0; 12], &mut out).unwrap();
        assert!(encoder.finish(&mut out).is_err());
    }

    #[test]
    fn extra_rows_and_wrong_lengths_are_errors() {
        let descriptor = ImageDescriptor::new(4, 1, PixelFormat::Rgb8).unwrap();
        let mut encoder = GifEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&descriptor, &mut out).unwrap();
        assert!(encoder.write_row(&[0; 11], &mut out).is_err(), "short row");
        assert!(encoder.write_row(&[0; 13], &mut out).is_err(), "long row");
        encoder.write_row(&[0; 12], &mut out).unwrap();
        assert!(encoder.write_row(&[0; 12], &mut out).is_err(), "extra row");
    }

    #[test]
    fn a_palette_size_outside_the_legal_range_is_rejected() {
        assert!(GifEncoder::new().with_colours(1).is_err());
        assert!(GifEncoder::new().with_colours(257).is_err());
        assert!(GifEncoder::new().with_colours(2).is_ok());
        assert!(GifEncoder::new().with_colours(256).is_ok());
    }

    #[test]
    fn options_map_quality_onto_palette_size() {
        assert_eq!(
            GifEncoder::from_options(&EncodeOptions::default()).colours(),
            204
        );
        let lowest = EncodeOptions::with_quality(1).unwrap();
        assert_eq!(GifEncoder::from_options(&lowest).colours(), 2);
        let highest = EncodeOptions::with_quality(100).unwrap();
        assert_eq!(GifEncoder::from_options(&highest).colours(), 256);
    }

    #[test]
    fn a_two_colour_image_uses_a_legal_code_width() {
        // LZW needs a minimum code width of 2 even for a 1-bit palette,
        // because width 1 has no room for both the clear and end codes. This
        // is the smallest image that exercises that clamp.
        let (descriptor, raster) = flat_art(3, 3, 2);
        let bytes = encode(
            &descriptor,
            &raster,
            GifEncoder::new().with_colours(2).unwrap(),
        )
        .unwrap();
        let (_, decoded) = decode(&bytes).unwrap();
        assert_eq!(decoded.len(), 3 * 3 * 4);
    }

    #[test]
    fn encoding_is_deterministic() {
        // SPEC §Guarantees 2, and the reason quantization sorts its histogram.
        let descriptor = ImageDescriptor::new(40, 30, PixelFormat::Rgb8).unwrap();
        let raster: Vec<u8> = (0..descriptor.byte_len().unwrap())
            .map(|i| ((i * 31) % 251) as u8)
            .collect();
        let first = encode(&descriptor, &raster, GifEncoder::new()).unwrap();
        for _ in 0..4 {
            let again = encode(&descriptor, &raster, GifEncoder::new()).unwrap();
            assert_eq!(again, first, "GIF encoding is not deterministic");
        }
    }

    #[test]
    fn a_single_pixel_image_round_trips() {
        let descriptor = ImageDescriptor::new(1, 1, PixelFormat::Rgb8).unwrap();
        let raster = vec![9_u8, 8, 7];
        let bytes = encode(&descriptor, &raster, GifEncoder::new()).unwrap();
        let (_, decoded) = decode(&bytes).unwrap();
        assert_eq!(&decoded[..3], [9, 8, 7]);
    }
}
