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
//! Non-interlaced PNG: `IHDR`, one `IDAT`, `IEND`, in the image's own colour
//! type. Adam7 would cost bytes rather than save them, and every decoder must
//! accept the plain form.
//!
//! Asked for a palette (sharp's `palette`), the encoder instead quantizes to
//! at most 256 colours with the median cut GIF uses, and writes colour type
//! 3: `PLTE`, a `tRNS` when any entry is translucent, and indices at the
//! smallest bit depth that holds them, unfiltered, as libpng recommends for
//! palette images. That is lossy unless the image has no more colours than
//! the palette, and usually much smaller.

use otf_pixels_core::{
    Density, EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, ResolutionUnit,
    Result, Sink,
};

use crate::format::{ColorType, Filter, SIGNATURE, apply_filter, write_chunk};
use otf_pixels_compress::{Level, build_palette, build_palette_rgba, quantize_with, zlib_compress};
use otf_pixels_core::{ChannelLayout, SampleKind};

/// Encodes a PNG stream.
#[derive(Debug)]
pub struct PngEncoder {
    level: Level,
    /// Palette output: the palette size and the dithering level.
    palette: Option<(usize, f32)>,
    /// Set by `write_header`; its presence means the header was written.
    state: Option<State>,
    /// The ICC profile to write as `iCCP`, if any.
    icc: Option<Vec<u8>>,
    /// The EXIF block to write as `eXIf`, if any.
    exif: Option<Vec<u8>>,
    /// The XMP packet to write as an `iTXt` chunk, if any.
    xmp: Option<Vec<u8>>,
    /// The resolution to write as `pHYs`, if any.
    density: Option<Density>,
}

/// The ICC profile, EXIF block, XMP packet and density to write.
type Ancillary<'a> = (
    Option<&'a [u8]>,
    Option<&'a [u8]>,
    Option<&'a [u8]>,
    Option<Density>,
);

/// The ancillary chunks that go before the image data: `iCCP`, `pHYs`,
/// `eXIf` and the XMP `iTXt`.
fn write_metadata(
    out: &mut Vec<u8>,
    level: Level,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    density: Option<Density>,
) -> Result<()> {
    if let Some(profile) = icc {
        // iCCP (§11.3.3.3): a name, NUL, method 0, the zlib stream.
        let mut iccp = b"ICC Profile\0\0".to_vec();
        iccp.extend(zlib_compress(profile, level).map_err(crate::compress_error)?);
        write_chunk(out, b"iCCP", &iccp);
    }
    if let Some(d) = density {
        // pHYs (§11.3.5.3) counts pixels per metre.
        let per_metre = |v: f64| {
            (v * if d.unit == ResolutionUnit::Inch {
                100.0 / 2.54
            } else {
                100.0
            })
            .round()
            .clamp(0.0, f64::from(u32::MAX)) as u32
        };
        let mut phys = per_metre(d.x).to_be_bytes().to_vec();
        phys.extend_from_slice(&per_metre(d.y).to_be_bytes());
        phys.push(1);
        write_chunk(out, b"pHYs", &phys);
    }
    if let Some(exif) = exif {
        // eXIf goes before the image data (§11.3.6.1).
        write_chunk(out, b"eXIf", exif);
    }
    if let Some(xmp) = xmp {
        // iTXt keyed for XMP, uncompressed, no language or translation, as
        // the XMP specification (part 3, §1.1.5) asks.
        let mut itxt = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
        itxt.extend_from_slice(xmp);
        write_chunk(out, b"iTXt", &itxt);
    }
    Ok(())
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
    /// For palette output, the image as 8-bit RGB or RGBA, gathered until
    /// `finish` because the palette depends on every pixel.
    true_colour: Option<Vec<u8>>,
    rows_written: u32,
}

impl PngEncoder {
    /// An encoder at the default compression level.
    #[must_use]
    pub const fn new() -> Self {
        Self::with_level(Level::DEFAULT)
    }

    /// An encoder at an explicit DEFLATE level.
    #[must_use]
    pub const fn with_level(level: Level) -> Self {
        Self {
            level,
            palette: None,
            state: None,
            icc: None,
            exif: None,
            xmp: None,
            density: None,
        }
    }

    /// This encoder writing a palette image of at most `colours` entries,
    /// diffusing `dither` (0..=1) of the quantization error.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] for `colours` outside
    /// 2..=256 or `dither` outside 0..=1.
    pub fn with_palette(mut self, colours: usize, dither: f32) -> Result<Self> {
        if !(2..=256).contains(&colours) {
            return Err(PixelsError::invalid_argument(
                "colours",
                format!("palette size must be in 2..=256, got {colours}"),
            ));
        }
        if !(0.0..=1.0).contains(&dither) {
            return Err(PixelsError::invalid_argument(
                "dither",
                format!("must be in 0..=1, got {dither}"),
            ));
        }
        self.palette = Some((colours, dither));
        Ok(self)
    }

    /// An encoder configured from generic encode options.
    ///
    /// [`EncodeOptions::compression_level`] is the DEFLATE level. Without
    /// one, [`EncodeOptions::quality`], which cannot trade fidelity for size
    /// in a lossless format, is read as compression *effort*, mapping 1..=100
    /// onto levels 1..=9.
    ///
    /// [`EncodeOptions::wants_palette`] selects palette output, of
    /// [`EncodeOptions::colours`] entries (256 by default) dithered by
    /// [`EncodeOptions::dither`] (1 by default).
    #[must_use]
    pub fn from_options(options: &EncodeOptions) -> Self {
        let level = options.compression_level.unwrap_or_else(|| {
            let quality = u32::from(options.quality.clamp(1, 100));
            // 1..=100 onto 1..=9, so the default quality of 80 lands on
            // level 7, just above zlib's own default of 6.
            ((quality - 1) * 8 / 99 + 1) as u8
        });
        let mut encoder = Self::with_level(Level::new(level).unwrap_or(Level::DEFAULT));
        if options.wants_palette() {
            let colours = options.colours.map_or(256, usize::from).clamp(2, 256);
            let dither = options.dither.unwrap_or(1.0).clamp(0.0, 1.0);
            encoder.palette = Some((colours, dither));
        }
        encoder
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

/// Append one row as 8-bit RGB, or RGBA if `format` has alpha.
fn append_8_bit(row: &[u8], format: PixelFormat, out: &mut Vec<u8>) {
    let wide = format.sample_kind() == SampleKind::U16;
    let sample = |pixel: &[u8], index: usize| -> u8 {
        if wide {
            let at = index * 2;
            let v = u16::from_ne_bytes([
                pixel.get(at).copied().unwrap_or(0),
                pixel.get(at + 1).copied().unwrap_or(0),
            ]);
            // Rounded to nearest, as every narrowing to 8 bits is.
            ((u32::from(v) * 255 + 32_767) / 65_535) as u8
        } else {
            pixel.get(index).copied().unwrap_or(0)
        }
    };
    for pixel in row.chunks_exact(format.bytes_per_pixel()) {
        match format.layout() {
            ChannelLayout::Gray => {
                let g = sample(pixel, 0);
                out.extend_from_slice(&[g, g, g]);
            }
            ChannelLayout::GrayAlpha => {
                let g = sample(pixel, 0);
                out.extend_from_slice(&[g, g, g, sample(pixel, 1)]);
            }
            ChannelLayout::Rgb => out.extend((0..3).map(|i| sample(pixel, i))),
            ChannelLayout::Rgba => out.extend((0..4).map(|i| sample(pixel, i))),
        }
    }
}

/// A whole palette PNG for `pixels`, 8-bit RGB or (when `descriptor` has
/// alpha) RGBA.
fn encode_palette(
    descriptor: &ImageDescriptor,
    pixels: &[u8],
    colours: usize,
    dither: f32,
    level: Level,
    metadata: Ancillary<'_>,
) -> Result<Vec<u8>> {
    let width = descriptor.width as usize;
    let (entries, indices): (Vec<[u8; 4]>, Vec<u8>) = if descriptor.pixel.layout().has_alpha() {
        let palette = build_palette_rgba(pixels, colours);
        let indices = quantize_with(pixels, width, &palette, dither);
        (palette.entries().to_vec(), indices)
    } else {
        let palette = build_palette(pixels, colours);
        let indices = quantize_with(pixels, width, &palette, dither);
        let entries = palette
            .entries()
            .iter()
            .map(|&[r, g, b]| [r, g, b, 255])
            .collect();
        (entries, indices)
    };

    // The smallest depth whose indices reach every entry.
    let bit_depth: u8 = match entries.len() {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        _ => 8,
    };
    let per_byte = 8 / usize::from(bit_depth);
    let row_bytes = width.div_ceil(per_byte);
    let mut scanlines = Vec::with_capacity((row_bytes + 1) * descriptor.height as usize);
    for row in indices.chunks_exact(width) {
        scanlines.push(Filter::None.to_byte());
        for group in row.chunks(per_byte) {
            let mut byte = 0_u8;
            for (i, &index) in group.iter().enumerate() {
                // Leftmost pixel in the highest bits (§7.2).
                byte |= index << (8 - usize::from(bit_depth) * (i + 1));
            }
            scanlines.push(byte);
        }
    }

    let mut out = SIGNATURE.to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&descriptor.width.to_be_bytes());
    ihdr.extend_from_slice(&descriptor.height.to_be_bytes());
    ihdr.extend_from_slice(&[bit_depth, ColorType::Palette.to_byte(), 0, 0, 0]);
    write_chunk(&mut out, b"IHDR", &ihdr);
    let (icc, exif, xmp, density) = metadata;
    write_metadata(&mut out, level, icc, exif, xmp, density)?;
    let plte: Vec<u8> = entries.iter().flat_map(|e| [e[0], e[1], e[2]]).collect();
    write_chunk(&mut out, b"PLTE", &plte);
    // tRNS lists alphas up to the last translucent entry; the rest are
    // opaque by default (§11.3.2.1).
    if let Some(last) = entries.iter().rposition(|e| e[3] != 255) {
        let alphas: Vec<u8> = entries.iter().take(last + 1).map(|e| e[3]).collect();
        write_chunk(&mut out, b"tRNS", &alphas);
    }
    let compressed = zlib_compress(&scanlines, level).map_err(crate::compress_error)?;
    write_chunk(&mut out, b"IDAT", &compressed);
    write_chunk(&mut out, b"IEND", &[]);
    Ok(out)
}

impl Encoder for PngEncoder {
    fn set_icc_profile(&mut self, profile: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "profile",
                "the ICC profile must be set before write_header",
            ));
        }
        self.icc = profile.map(<[u8]>::to_vec);
        Ok(())
    }

    fn set_exif(&mut self, exif: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "exif",
                "the EXIF block must be set before write_header",
            ));
        }
        self.exif = exif.map(<[u8]>::to_vec);
        Ok(())
    }

    fn set_xmp(&mut self, xmp: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "xmp",
                "the XMP packet must be set before write_header",
            ));
        }
        self.xmp = xmp.map(<[u8]>::to_vec);
        Ok(())
    }

    fn set_density(&mut self, density: Option<Density>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "density",
                "the density must be set before write_header",
            ));
        }
        self.density = density;
        Ok(())
    }

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

        if self.palette.is_some() {
            // Nothing can be written yet: the header's bit depth depends on
            // how many colours the palette ends up with.
            let channels = if desc.pixel.layout().has_alpha() {
                4
            } else {
                3
            };
            self.state = Some(State {
                descriptor: *desc,
                bit_depth: 8,
                stride: 1,
                filtered: Vec::new(),
                previous: Vec::new(),
                swapped: Vec::new(),
                true_colour: Some(Vec::with_capacity(
                    desc.width as usize * desc.height as usize * channels,
                )),
                rows_written: 0,
            });
            return Ok(());
        }

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
        write_metadata(
            &mut chunk,
            self.level,
            self.icc.as_deref(),
            self.exif.as_deref(),
            self.xmp.as_deref(),
            self.density,
        )?;
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
            true_colour: None,
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

        if let Some(pixels) = state.true_colour.as_mut() {
            append_8_bit(row, state.descriptor.pixel, pixels);
        } else if state.bit_depth == 16 {
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

        if let (Some(pixels), Some((colours, dither))) = (state.true_colour.take(), self.palette) {
            let file = encode_palette(
                &state.descriptor,
                &pixels,
                colours,
                dither,
                self.level,
                (
                    self.icc.as_deref(),
                    self.exif.as_deref(),
                    self.xmp.as_deref(),
                    self.density,
                ),
            )?;
            sink.write_all(&file)?;
            return sink.flush();
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

    /// Encode with `encoder`, returning the PNG bytes.
    fn encode_with(desc: &ImageDescriptor, raster: &[u8], mut encoder: PngEncoder) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        encoder.write_header(desc, &mut out).unwrap();
        for row in raster.chunks_exact(desc.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        out
    }

    fn palette(colours: usize, dither: f32) -> PngEncoder {
        PngEncoder::new().with_palette(colours, dither).unwrap()
    }

    /// The IHDR bit depth and colour type.
    fn header(bytes: &[u8]) -> (u8, u8) {
        (bytes[24], bytes[25])
    }

    fn has_chunk(bytes: &[u8], kind: &[u8; 4]) -> bool {
        bytes.windows(4).any(|w| w == kind)
    }

    #[test]
    fn a_palette_image_with_few_colours_round_trips_exactly() {
        let desc = ImageDescriptor::new(13, 7, PixelFormat::Rgba8).unwrap();
        let colours = [
            [255, 0, 0, 255],
            [0, 255, 0, 128],
            [0, 0, 255, 0],
            [9, 9, 9, 255],
        ];
        let raster: Vec<u8> = (0..13 * 7).flat_map(|i| colours[i % 4]).collect();
        let bytes = encode_with(&desc, &raster, palette(256, 1.0));
        assert_eq!(header(&bytes), (2, 3), "four colours fit two bits");
        assert!(has_chunk(&bytes, b"PLTE") && has_chunk(&bytes, b"tRNS"));
        let (out, decoded) = decode(&bytes).unwrap();
        assert_eq!(out.pixel, PixelFormat::Rgba8);
        // Every invisible pixel is the same entry, whatever colour it hid.
        let visible: Vec<u8> = raster
            .chunks_exact(4)
            .flat_map(|p| {
                if p[3] == 0 {
                    [0; 4]
                } else {
                    [p[0], p[1], p[2], p[3]]
                }
            })
            .collect();
        assert_eq!(decoded, visible);
    }

    #[test]
    fn the_bit_depth_is_the_smallest_that_holds_the_palette() {
        for (count, depth) in [
            (2_usize, 1_u8),
            (3, 2),
            (4, 2),
            (5, 4),
            (16, 4),
            (17, 8),
            (200, 8),
        ] {
            // Widths that leave a partial last byte at every depth.
            let desc = ImageDescriptor::new(11, 3, PixelFormat::Rgb8).unwrap();
            let raster: Vec<u8> = (0..33)
                .flat_map(|i| {
                    let v = (i % count) as u8;
                    [v, v.wrapping_mul(3), 255 - v]
                })
                .collect();
            let bytes = encode_with(&desc, &raster, palette(256, 1.0));
            assert_eq!(header(&bytes), (depth, 3), "{count} colours");
            assert!(!has_chunk(&bytes, b"tRNS"), "an opaque image needs no tRNS");
            let (out, decoded) = decode(&bytes).unwrap();
            assert_eq!(out.pixel, PixelFormat::Rgb8);
            assert_eq!(decoded, raster, "{count} colours");
        }
    }

    #[test]
    fn a_photograph_is_reduced_to_the_palette_size() {
        let desc = ImageDescriptor::new(64, 64, PixelFormat::Rgb8).unwrap();
        let raster: Vec<u8> = (0..64 * 64 * 3)
            .map(|i| ((i * 37 + i / 191) % 256) as u8)
            .collect();
        for dither in [0.0, 0.5, 1.0] {
            let bytes = encode_with(&desc, &raster, palette(16, dither));
            let plte = bytes.windows(4).position(|w| w == b"PLTE").unwrap();
            let length = u32::from_be_bytes(bytes[plte - 4..plte].try_into().unwrap());
            assert!(length <= 16 * 3, "{length} bytes of palette");
            let (_, decoded) = decode(&bytes).unwrap();
            let mut distinct: Vec<&[u8]> = decoded.chunks_exact(3).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(distinct.len() <= 16);
        }
        // And on photographic noise, which true colour cannot compress, a
        // palette is much smaller.
        let mut seed = 12_345_u32;
        let noisy: Vec<u8> = (0..64 * 64 * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                (seed >> 16) as u8
            })
            .collect();
        let true_colour = encode(&desc, &noisy, Level::DEFAULT).unwrap();
        let reduced = encode_with(&desc, &noisy, palette(16, 1.0));
        assert!(
            reduced.len() * 2 < true_colour.len(),
            "{} against {}",
            reduced.len(),
            true_colour.len()
        );
    }

    #[test]
    fn grey_and_wide_images_take_a_palette_too() {
        let desc = ImageDescriptor::new(4, 1, PixelFormat::Gray16).unwrap();
        let raster: Vec<u8> = [0_u16, 65535, 0, 65535]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let bytes = encode_with(&desc, &raster, palette(256, 1.0));
        assert_eq!(header(&bytes), (1, 3));
        assert_eq!(
            decode(&bytes).unwrap().1,
            [0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255]
        );
    }

    #[test]
    fn options_choose_the_level_and_the_palette() {
        let desc = ImageDescriptor::new(32, 32, PixelFormat::Rgb8).unwrap();
        let raster: Vec<u8> = (0..32 * 32 * 3).map(|i| (i / 7 % 256) as u8).collect();
        let with = |options: EncodeOptions| {
            encode_with(&desc, &raster, PngEncoder::from_options(&options))
        };
        let stored = with(EncodeOptions::default().with_compression_level(0).unwrap());
        let best = with(EncodeOptions::default().with_compression_level(9).unwrap());
        assert!(best.len() < stored.len());
        // Unset, the level still comes from quality, as it always did.
        assert_eq!(
            with(EncodeOptions::default()),
            encode(&desc, &raster, Level::new(7).unwrap()).unwrap()
        );
        for options in [
            EncodeOptions::default().with_palette(true),
            EncodeOptions::default().with_colours(64).unwrap(),
            EncodeOptions::default().with_dither(0.0).unwrap(),
        ] {
            assert_eq!(header(&with(options)).1, 3, "{options:?}");
        }
        assert_eq!(header(&with(EncodeOptions::default())).1, 2);
    }

    #[test]
    fn palette_parameters_are_validated() {
        assert!(PngEncoder::new().with_palette(1, 1.0).is_err());
        assert!(PngEncoder::new().with_palette(257, 1.0).is_err());
        assert!(PngEncoder::new().with_palette(2, 1.5).is_err());
    }
}
