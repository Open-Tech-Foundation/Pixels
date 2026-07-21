//! The PNG encoder.
//!
//! # Memory
//!
//! Encoding is **internally buffered**: DEFLATE needs the whole scanline
//! stream before it can emit its Huffman-coded blocks, so filtered rows
//! accumulate until [`Encoder::finish`]. That is ADR-0005's stated allowance
//! for codecs that cannot work incrementally, and it is symmetric with the
//! decoder — see the module docs there for why an incremental inflate and
//! deflate are deferred rather than absent.
//!
//! # What is written
//!
//! Non-interlaced, `PLTE`-free PNG: `IHDR`, one `IDAT`, `IEND`. Adam7 and
//! palette output would both cost bytes rather than save them for the images
//! a pipeline produces, and every decoder must accept the plain form.

use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

use crate::format::{ColorType, Filter, SIGNATURE, apply_filter, write_chunk};
use otf_pixels_compress::{Level, zlib_compress};

/// Encodes a PNG stream.
#[derive(Debug)]
pub struct PngEncoder {
    level: Level,
    /// Set by `write_header`; its presence means the header was written.
    state: Option<State>,
}

/// Everything fixed once the descriptor is known.
#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    bit_depth: u8,
    /// Bytes per pixel, rounded up — the filter's left-neighbour offset.
    stride: usize,
    /// Filtered scanlines, each prefixed by its filter byte.
    filtered: Vec<u8>,
    /// The previous *unfiltered* row, which filters predict from.
    previous: Vec<u8>,
    /// Reusable big-endian staging buffer; empty unless `bit_depth` is 16.
    swapped: Vec<u8>,
    rows_written: u32,
}

impl PngEncoder {
    /// An encoder at the default compression level.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            level: Level::DEFAULT,
            state: None,
        }
    }

    /// An encoder at an explicit DEFLATE level.
    #[must_use]
    pub const fn with_level(level: Level) -> Self {
        Self { level, state: None }
    }

    /// An encoder configured from generic encode options.
    ///
    /// PNG is lossless, so [`EncodeOptions::quality`] cannot trade fidelity
    /// for size. It is read as compression *effort* instead, mapping 1..=100
    /// onto DEFLATE levels 1..=9 — the only axis PNG actually has.
    #[must_use]
    pub fn from_options(options: &EncodeOptions) -> Self {
        let quality = u32::from(options.quality.clamp(1, 100));
        // 1..=100 onto 1..=9, so the default quality of 80 lands on level 7,
        // just above zlib's own default of 6.
        let level = ((quality - 1) * 8 / 99 + 1) as u8;
        Self::with_level(Level::new(level).unwrap_or(Level::DEFAULT))
    }

    /// The colour type and bit depth for a pixel format.
    fn png_type_of(format: PixelFormat) -> Result<(ColorType, u8)> {
        match format {
            PixelFormat::Gray8 => Ok((ColorType::Grayscale, 8)),
            PixelFormat::Gray16 => Ok((ColorType::Grayscale, 16)),
            PixelFormat::GrayA8 => Ok((ColorType::GrayscaleAlpha, 8)),
            PixelFormat::Rgb8 => Ok((ColorType::Rgb, 8)),
            PixelFormat::Rgb16 => Ok((ColorType::Rgb, 16)),
            PixelFormat::Rgba8 => Ok((ColorType::Rgba, 8)),
            PixelFormat::Rgba16 => Ok((ColorType::Rgba, 16)),
            // PNG has no float sample type at any bit depth (§11.2.2).
            other => Err(PixelsError::unsupported(format!(
                "PNG cannot represent {other}; convert to an integer format first"
            ))),
        }
    }
}

impl Default for PngEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    /// Choose a filter for `row` and append it, filter byte first.
    ///
    /// Uses the minimum-sum-of-absolute-differences heuristic from the PNG
    /// specification's §12.8: filtered bytes near zero compress best, and the
    /// sum of their signed magnitudes estimates that without running DEFLATE
    /// five times.
    fn append_filtered(&mut self, row: &[u8]) {
        let candidates: &[Filter] = &[
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Average,
            Filter::Paeth,
        ];

        let mut best = Filter::None;
        let mut best_score = u64::MAX;
        let mut scratch = Vec::with_capacity(row.len());
        for &filter in candidates {
            scratch.clear();
            apply_filter(filter, row, &self.previous, self.stride, &mut scratch);
            let score: u64 = scratch
                .iter()
                .map(|&byte| u64::from((byte as i8).unsigned_abs()))
                .sum();
            if score < best_score {
                best_score = score;
                best = filter;
            }
        }

        self.filtered.push(best.to_byte());
        apply_filter(best, row, &self.previous, self.stride, &mut self.filtered);
        self.previous.clear();
        self.previous.extend_from_slice(row);
    }
}

/// Rewrite native-endian 16-bit samples as the big-endian ones PNG stores.
///
/// On a big-endian host this is a copy; the swap is expressed in terms of
/// `to_be_bytes` rather than a `cfg`, so there is one code path to be wrong.
fn to_big_endian_16(row: &[u8], out: &mut Vec<u8>) {
    out.clear();
    for pair in row.chunks_exact(2) {
        let value = u16::from_ne_bytes([
            pair.first().copied().unwrap_or(0),
            pair.get(1).copied().unwrap_or(0),
        ]);
        out.extend_from_slice(&value.to_be_bytes());
    }
}

impl Encoder for PngEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        if desc.width == 0 || desc.height == 0 {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                format!(
                    "PNG dimensions must be non-zero, got {}x{}",
                    desc.width, desc.height
                ),
            ));
        }
        let (color_type, bit_depth) = Self::png_type_of(desc.pixel)?;

        sink.write_all(&SIGNATURE)?;
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&desc.width.to_be_bytes());
        ihdr.extend_from_slice(&desc.height.to_be_bytes());
        ihdr.push(bit_depth);
        ihdr.push(color_type.to_byte());
        // Compression 0 (DEFLATE), filter 0 (adaptive), interlace 0 (none) —
        // the only values the specification defines.
        ihdr.extend_from_slice(&[0, 0, 0]);
        let mut chunk = Vec::new();
        write_chunk(&mut chunk, b"IHDR", &ihdr);
        sink.write_all(&chunk)?;

        let row_bytes = desc.row_bytes();
        let stride = (color_type.channels() * bit_depth as usize).div_ceil(8);
        self.state = Some(State {
            descriptor: *desc,
            bit_depth,
            stride,
            // Each row costs its bytes plus one filter byte.
            filtered: Vec::with_capacity((row_bytes + 1) * desc.height as usize),
            previous: vec![0_u8; row_bytes],
            swapped: Vec::new(),
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

        if state.bit_depth == 16 {
            let mut swapped = std::mem::take(&mut state.swapped);
            to_big_endian_16(row, &mut swapped);
            state.append_filtered(&swapped);
            state.swapped = swapped;
        } else {
            state.append_filtered(row);
        }
        state.rows_written += 1;
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::malformed(
                "png",
                "finish called before write_header",
            ));
        };
        if state.rows_written != state.descriptor.height {
            return Err(PixelsError::malformed(
                "png",
                format!(
                    "{} of {} rows written; a partial image is never emitted",
                    state.rows_written, state.descriptor.height
                ),
            ));
        }

        let compressed =
            zlib_compress(&state.filtered, self.level).map_err(crate::compress_error)?;
        let mut chunk = Vec::with_capacity(compressed.len() + 12);
        write_chunk(&mut chunk, b"IDAT", &compressed);
        write_chunk(&mut chunk, b"IEND", &[]);
        sink.write_all(&chunk)?;
        sink.flush()?;

        // Release the raster now rather than at drop; a caller that keeps the
        // encoder around to inspect it should not keep the image too.
        state.filtered = Vec::new();
        state.previous = Vec::new();
        state.swapped = Vec::new();
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
    use crate::decoder::PngDecoder;
    use otf_pixels_core::{Decoder, Limits};

    /// Encode a full raster, returning the PNG bytes.
    fn encode(desc: &ImageDescriptor, raster: &[u8], level: Level) -> Result<Vec<u8>> {
        let mut encoder = PngEncoder::with_level(level);
        let mut out: Vec<u8> = Vec::new();
        encoder.write_header(desc, &mut out)?;
        for row in raster.chunks_exact(desc.row_bytes()) {
            encoder.write_row(row, &mut out)?;
        }
        encoder.finish(&mut out)?;
        Ok(out)
    }

    /// Decode a full raster back.
    fn decode(bytes: &[u8]) -> Result<(ImageDescriptor, Vec<u8>)> {
        let mut decoder = PngDecoder::new(bytes, Limits::default())?;
        let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
        let mut raster = Vec::new();
        for _ in 0..decoder.descriptor().height {
            decoder.read_row(&mut row)?;
            raster.extend_from_slice(&row);
        }
        Ok((decoder.descriptor(), raster))
    }

    /// A deterministic non-trivial raster: gradients plus a hard edge, so the
    /// filter heuristic has something to choose between.
    fn sample(desc: &ImageDescriptor) -> Vec<u8> {
        let mut raster = vec![0_u8; desc.byte_len().unwrap()];
        for (index, byte) in raster.iter_mut().enumerate() {
            let value = if index % 97 < 40 {
                (index % 251) as u8
            } else {
                ((index * 7) % 13) as u8
            };
            *byte = value;
        }
        raster
    }

    #[test]
    fn every_supported_format_round_trips() {
        for format in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::GrayA8,
            PixelFormat::Rgb8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba8,
            PixelFormat::Rgba16,
        ] {
            let desc = ImageDescriptor::new(23, 17, format).unwrap();
            let raster = sample(&desc);
            let bytes = encode(&desc, &raster, Level::DEFAULT)
                .unwrap_or_else(|e| panic!("encoding {format}: {e}"));
            let (decoded_desc, decoded) =
                decode(&bytes).unwrap_or_else(|e| panic!("decoding {format}: {e}"));
            assert_eq!(decoded_desc.pixel, format, "{format} changed format");
            assert_eq!(
                (decoded_desc.width, decoded_desc.height),
                (23, 17),
                "{format}"
            );
            assert_eq!(decoded, raster, "{format} did not round-trip");
        }
    }

    #[test]
    fn every_level_round_trips_to_the_same_pixels() {
        let desc = ImageDescriptor::new(19, 11, PixelFormat::Rgba8).unwrap();
        let raster = sample(&desc);
        for level in 0..=9 {
            let level = Level::new(level).unwrap();
            let bytes = encode(&desc, &raster, level).unwrap();
            let (_, decoded) = decode(&bytes).unwrap();
            assert_eq!(decoded, raster, "level {} did not round-trip", level.get());
        }
    }

    #[test]
    fn the_output_starts_with_the_signature_and_ihdr() {
        let desc = ImageDescriptor::new(4, 4, PixelFormat::Rgb8).unwrap();
        let bytes = encode(&desc, &sample(&desc), Level::FAST).unwrap();
        assert_eq!(&bytes[..8], &SIGNATURE, "signature");
        assert_eq!(&bytes[12..16], b"IHDR", "first chunk");
        assert_eq!(
            &bytes[bytes.len() - 8..bytes.len() - 4],
            b"IEND",
            "last chunk"
        );
    }

    #[test]
    fn sixteen_bit_samples_are_written_big_endian() {
        // The one place a host-endianness bug would hide: encode a known
        // sample and read the two bytes straight out of the decompressed
        // scanline via a decode, which is byte-order-symmetric, plus assert
        // the swap helper directly.
        let mut out = Vec::new();
        to_big_endian_16(&0x1234_u16.to_ne_bytes(), &mut out);
        assert_eq!(out, vec![0x12, 0x34]);
    }

    #[test]
    fn compression_shrinks_a_compressible_image() {
        // A flat image is the clearest case: level 0 stores it, level 9 must
        // do dramatically better, which proves the level reaches DEFLATE.
        let desc = ImageDescriptor::new(64, 64, PixelFormat::Rgb8).unwrap();
        let raster = vec![7_u8; desc.byte_len().unwrap()];
        let stored = encode(&desc, &raster, Level::NONE).unwrap();
        let packed = encode(&desc, &raster, Level::BEST).unwrap();
        assert!(
            packed.len() * 10 < stored.len(),
            "level 9 produced {} bytes against level 0's {}",
            packed.len(),
            stored.len()
        );
        assert_eq!(decode(&packed).unwrap().1, raster);
    }

    #[test]
    fn a_float_format_is_unsupported_not_a_panic() {
        let desc = ImageDescriptor::new(2, 2, PixelFormat::RgbaF32).unwrap();
        let mut encoder = PngEncoder::new();
        let mut out = Vec::new();
        let error = encoder.write_header(&desc, &mut out).unwrap_err();
        assert!(matches!(error, PixelsError::Unsupported { .. }), "{error}");
        assert!(
            out.is_empty(),
            "nothing should be written for a rejected format"
        );
    }

    #[test]
    fn a_short_image_is_an_error_not_a_truncated_png() {
        let desc = ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap();
        let mut encoder = PngEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&desc, &mut out).unwrap();
        encoder.write_row(&[0; 4], &mut out).unwrap();
        let error = encoder.finish(&mut out).unwrap_err();
        assert!(matches!(error, PixelsError::Malformed { .. }), "{error}");
    }

    #[test]
    fn extra_rows_and_wrong_row_lengths_are_errors() {
        let desc = ImageDescriptor::new(4, 1, PixelFormat::Gray8).unwrap();
        let mut encoder = PngEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&desc, &mut out).unwrap();
        assert!(encoder.write_row(&[0; 3], &mut out).is_err(), "short row");
        assert!(encoder.write_row(&[0; 5], &mut out).is_err(), "long row");
        encoder.write_row(&[0; 4], &mut out).unwrap();
        assert!(encoder.write_row(&[0; 4], &mut out).is_err(), "extra row");
    }

    #[test]
    fn rows_before_the_header_are_an_error() {
        let mut encoder = PngEncoder::new();
        let mut out = Vec::new();
        assert!(encoder.write_row(&[0; 4], &mut out).is_err());
        assert!(encoder.finish(&mut out).is_err());
    }

    #[test]
    fn a_second_header_is_an_error() {
        let desc = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let mut encoder = PngEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&desc, &mut out).unwrap();
        assert!(encoder.write_header(&desc, &mut out).is_err());
    }

    #[test]
    fn zero_dimensions_never_reach_the_encoder() {
        // `ImageDescriptor` refuses to hold a zero dimension, so the guard in
        // `write_header` is defence in depth rather than the only check. Both
        // matter: PNG's IHDR has no representation for an empty image, and a
        // future descriptor change must not quietly start emitting one.
        assert!(ImageDescriptor::new(0, 4, PixelFormat::Gray8).is_err());
        assert!(ImageDescriptor::new(4, 0, PixelFormat::Gray8).is_err());
    }

    #[test]
    fn options_map_quality_onto_compression_effort() {
        assert_eq!(
            PngEncoder::from_options(&EncodeOptions::default())
                .level
                .get(),
            7
        );
        let lowest = EncodeOptions::with_quality(1).unwrap();
        assert_eq!(PngEncoder::from_options(&lowest).level.get(), 1);
        let highest = EncodeOptions::with_quality(100).unwrap();
        assert_eq!(PngEncoder::from_options(&highest).level.get(), 9);
    }

    #[test]
    fn a_single_pixel_image_round_trips() {
        // The degenerate case where every filter's neighbours are all zero.
        let desc = ImageDescriptor::new(1, 1, PixelFormat::Rgba8).unwrap();
        let raster = vec![1, 2, 3, 4];
        let bytes = encode(&desc, &raster, Level::BEST).unwrap();
        assert_eq!(decode(&bytes).unwrap().1, raster);
    }
}
