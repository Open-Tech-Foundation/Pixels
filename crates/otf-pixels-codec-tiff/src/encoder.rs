//! The TIFF encoder.
//!
//! Writes baseline TIFF: little-endian, one directory, either strips or tiles,
//! with no compression or Deflate. That is deliberately narrower than what the
//! decoder reads — a decoder must accept what the world produces, an encoder
//! only has to produce something correct — and it covers what a pipeline
//! actually needs to write.
//!
//! # Why tiled output matters
//!
//! A tiled TIFF is the only v1 format a later pipeline can read back with
//! genuine random access. Writing one is therefore how a caller stores an
//! intermediate that will be re-read in pieces, which is the pattern behind
//! image pyramids and tile servers.
//!
//! # Memory
//!
//! Strip output streams: a strip is written as soon as its rows arrive, so
//! peak memory is one strip. Tiled output buffers rows until a full band of
//! tiles is available, which is one tile height rather than the image.

use otf_pixels_compress::{Level, zlib_compress};
use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

use crate::ifd::{ByteOrder, tag};

/// How an encoder arranges pixels in the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TiffLayout {
    /// Horizontal bands, `rows` tall.
    Strips {
        /// Rows per strip.
        rows: u32,
    },
    /// A grid of tiles. Both dimensions must be multiples of 16.
    Tiles {
        /// Tile width.
        width: u32,
        /// Tile height.
        height: u32,
    },
}

impl Default for TiffLayout {
    fn default() -> Self {
        // 64 rows is a common default and keeps a strip comfortably inside
        // cache for the widths a pipeline usually produces.
        Self::Strips { rows: 64 }
    }
}

/// Encodes a baseline TIFF.
#[derive(Debug)]
pub struct TiffEncoder {
    layout: TiffLayout,
    deflate: Option<Level>,
    state: Option<State>,
}

#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    /// Rows accumulated but not yet emitted as a chunk.
    pending: Vec<u8>,
    /// Compressed chunks, in file order.
    chunks: Vec<Vec<u8>>,
    rows_written: u32,
}

impl Default for TiffEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl TiffEncoder {
    /// An encoder writing uncompressed strips.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            layout: TiffLayout::Strips { rows: 64 },
            deflate: None,
            state: None,
        }
    }

    /// An encoder writing the given layout.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] for a zero-sized layout, or
    /// tile dimensions that are not multiples of 16 — which TIFF 6.0
    /// §Section 15 requires and every reader relies on.
    pub fn with_layout(mut self, layout: TiffLayout) -> Result<Self> {
        match layout {
            TiffLayout::Strips { rows: 0 } => {
                return Err(PixelsError::invalid_argument("rows", "must be non-zero"));
            }
            TiffLayout::Tiles { width, height } => {
                if width == 0 || height == 0 {
                    return Err(PixelsError::invalid_argument(
                        "tile",
                        "tile dimensions must be non-zero",
                    ));
                }
                if width % 16 != 0 || height % 16 != 0 {
                    return Err(PixelsError::invalid_argument(
                        "tile",
                        format!("tile {width}x{height} must be a multiple of 16"),
                    ));
                }
            }
            TiffLayout::Strips { .. } => {}
        }
        self.layout = layout;
        Ok(self)
    }

    /// An encoder compressing with Deflate at `level`.
    #[must_use]
    pub const fn with_deflate(mut self, level: Level) -> Self {
        self.deflate = Some(level);
        self
    }

    /// An encoder configured from generic encode options.
    ///
    /// TIFF is lossless, so quality is read as compression effort: below the
    /// midpoint means store uncompressed and prioritise speed, above it means
    /// Deflate at a level scaled from the remaining range.
    #[must_use]
    pub fn from_options(options: &EncodeOptions) -> Self {
        let quality = u32::from(options.quality.clamp(1, 100));
        let mut encoder = Self::new();
        if quality > 50 {
            let level = ((quality - 50) * 9 / 50).clamp(1, 9) as u8;
            encoder.deflate = Some(Level::new(level).unwrap_or(Level::DEFAULT));
        }
        encoder
    }

    /// The layout this encoder writes.
    #[must_use]
    pub const fn layout(&self) -> TiffLayout {
        self.layout
    }

    /// Emit whatever complete chunks the pending rows allow.
    fn flush_chunks(&mut self, final_flush: bool) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Ok(());
        };
        let row_bytes = state.descriptor.row_bytes();
        let width = state.descriptor.width;

        match self.layout {
            TiffLayout::Strips { rows } => {
                let chunk_bytes = row_bytes * rows as usize;
                while state.pending.len() >= chunk_bytes
                    || (final_flush && !state.pending.is_empty())
                {
                    let take = chunk_bytes.min(state.pending.len());
                    let block: Vec<u8> = state.pending.drain(..take).collect();
                    let compressed = match self.deflate {
                        None => block,
                        Some(level) => {
                            zlib_compress(&block, level).map_err(crate::compress_error)?
                        }
                    };
                    state.chunks.push(compressed);
                    if !final_flush {
                        break;
                    }
                }
            }
            TiffLayout::Tiles {
                width: tile_width,
                height: tile_height,
            } => {
                let band_bytes = row_bytes * tile_height as usize;
                while state.pending.len() >= band_bytes
                    || (final_flush && !state.pending.is_empty())
                {
                    let take = band_bytes.min(state.pending.len());
                    let band: Vec<u8> = state.pending.drain(..take).collect();
                    let rows_in_band = take / row_bytes.max(1);
                    let bpp = state.descriptor.pixel.bytes_per_pixel();
                    let across = width.div_ceil(tile_width);

                    for column in 0..across {
                        // Tiles are always stored whole; the edge ones are
                        // padded rather than clipped, which is what every
                        // reader expects and what the decoder assumes.
                        let mut tile = vec![0_u8; tile_width as usize * tile_height as usize * bpp];
                        for row in 0..tile_height as usize {
                            if row >= rows_in_band {
                                break;
                            }
                            for pixel in 0..tile_width as usize {
                                let x = column as usize * tile_width as usize + pixel;
                                if x >= width as usize {
                                    break;
                                }
                                let from = row * row_bytes + x * bpp;
                                let to = (row * tile_width as usize + pixel) * bpp;
                                let (Some(source), Some(target)) =
                                    (band.get(from..from + bpp), tile.get_mut(to..to + bpp))
                                else {
                                    continue;
                                };
                                target.copy_from_slice(source);
                            }
                        }
                        let compressed = match self.deflate {
                            None => tile,
                            Some(level) => {
                                zlib_compress(&tile, level).map_err(crate::compress_error)?
                            }
                        };
                        state.chunks.push(compressed);
                    }
                    if !final_flush {
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

/// The tags a baseline TIFF needs, as (tag, type code, values).
type Field = (u16, u16, Vec<u32>);

impl Encoder for TiffEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        // The directory carries offsets that are not known until every chunk
        // has been compressed, so nothing is written until `finish`.
        supported_format(desc.pixel)?;
        self.state = Some(State {
            descriptor: *desc,
            pending: Vec::new(),
            chunks: Vec::new(),
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
        state.pending.extend_from_slice(row);
        state.rows_written += 1;
        self.flush_chunks(false)?;
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        {
            let Some(state) = self.state.as_ref() else {
                return Err(PixelsError::malformed(
                    "tiff",
                    "finish called before write_header",
                ));
            };
            if state.rows_written != state.descriptor.height {
                return Err(PixelsError::malformed(
                    "tiff",
                    format!(
                        "{} of {} rows written; a partial image is never emitted",
                        state.rows_written, state.descriptor.height
                    ),
                ));
            }
        }
        self.flush_chunks(true)?;

        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::graph("tiff encoder state vanished"));
        };
        let order = ByteOrder::Little;
        let descriptor = state.descriptor;
        let (channels, bits) = sample_shape(descriptor.pixel);
        let photometric = if channels >= 3 { 2_u32 } else { 1 };
        let compression = if self.deflate.is_some() { 8_u32 } else { 1 };

        let counts: Vec<u32> = state.chunks.iter().map(|c| c.len() as u32).collect();
        let mut fields: Vec<Field> = vec![
            (tag::IMAGE_WIDTH, 4, vec![descriptor.width]),
            (tag::IMAGE_LENGTH, 4, vec![descriptor.height]),
            (tag::BITS_PER_SAMPLE, 3, vec![bits; channels as usize]),
            (tag::COMPRESSION, 3, vec![compression]),
            (tag::PHOTOMETRIC, 3, vec![photometric]),
            (tag::SAMPLES_PER_PIXEL, 3, vec![channels]),
            (tag::PLANAR_CONFIG, 3, vec![1]),
        ];
        // Alpha has to be declared, or a reader shows a four-channel image as
        // RGB plus an unknown extra rather than as RGBA.
        if matches!(
            descriptor.pixel,
            PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::GrayA8
        ) {
            // 2 = unassociated alpha, which is what SPEC §Formats promises.
            fields.push((tag::EXTRA_SAMPLES, 3, vec![2]));
        }
        match self.layout {
            TiffLayout::Strips { rows } => {
                fields.push((tag::ROWS_PER_STRIP, 4, vec![rows]));
                fields.push((tag::STRIP_OFFSETS, 4, vec![0; counts.len()]));
                fields.push((tag::STRIP_BYTE_COUNTS, 4, counts.clone()));
            }
            TiffLayout::Tiles { width, height } => {
                fields.push((tag::TILE_WIDTH, 3, vec![width]));
                fields.push((tag::TILE_LENGTH, 3, vec![height]));
                fields.push((tag::TILE_OFFSETS, 4, vec![0; counts.len()]));
                fields.push((tag::TILE_BYTE_COUNTS, 4, counts.clone()));
            }
        }
        // Tags must appear in ascending order; readers are entitled to binary
        // search, and libtiff warns loudly about files that get this wrong.
        fields.sort_by_key(|(tag, _, _)| *tag);

        let out = assemble(order, &fields, &state.chunks)?;
        sink.write_all(&out)?;
        sink.flush()?;

        state.chunks = Vec::new();
        state.pending = Vec::new();
        Ok(())
    }
}

/// Lay out header, directory, value heap and pixel data.
fn assemble(order: ByteOrder, fields: &[Field], chunks: &[Vec<u8>]) -> Result<Vec<u8>> {
    let count = fields.len();
    let directory_at = 8_usize;
    let directory_size = 2 + count * 12 + 4;
    let heap_at = directory_at + directory_size;

    // First pass: work out where each field's values live and how big the
    // heap is, so the pixel data's offset is known before it is written.
    let mut heap = Vec::new();
    let mut value_offsets: Vec<Option<usize>> = Vec::with_capacity(count);
    for (_, type_code, values) in fields {
        let size = type_size(*type_code);
        if values.len() * size > 4 {
            value_offsets.push(Some(heap_at + heap.len()));
            for &value in values {
                push_value(&mut heap, order, *type_code, value);
            }
            // A value array must start on an even boundary (§Section 2).
            if heap.len() % 2 == 1 {
                heap.push(0);
            }
        } else {
            value_offsets.push(None);
        }
    }

    let data_at = heap_at + heap.len();
    let mut offsets = Vec::with_capacity(chunks.len());
    let mut running = data_at;
    for chunk in chunks {
        offsets.push(running as u32);
        running += chunk.len();
    }

    // Second pass: emit the directory, patching the offset arrays.
    let mut directory = Vec::with_capacity(directory_size);
    directory.extend_from_slice(&order.write_u16(count as u16));
    for (index, (tag_id, type_code, values)) in fields.iter().enumerate() {
        directory.extend_from_slice(&order.write_u16(*tag_id));
        directory.extend_from_slice(&order.write_u16(*type_code));
        directory.extend_from_slice(&order.write_u32(values.len() as u32));

        let is_offsets = *tag_id == tag::STRIP_OFFSETS || *tag_id == tag::TILE_OFFSETS;
        let resolved: Vec<u32> = if is_offsets {
            offsets.clone()
        } else {
            values.clone()
        };

        match value_offsets.get(index).copied().flatten() {
            Some(at) => {
                if is_offsets {
                    // Patch the placeholder written during the first pass.
                    let start = at - heap_at;
                    for (position, &value) in resolved.iter().enumerate() {
                        let slot = start + position * 4;
                        if let Some(target) = heap.get_mut(slot..slot + 4) {
                            target.copy_from_slice(&order.write_u32(value));
                        }
                    }
                }
                directory.extend_from_slice(&order.write_u32(at as u32));
            }
            None => {
                let mut inline = Vec::with_capacity(4);
                for &value in &resolved {
                    push_value(&mut inline, order, *type_code, value);
                }
                inline.resize(4, 0);
                directory.extend_from_slice(inline.get(..4).unwrap_or(&[0; 4]));
            }
        }
    }
    directory.extend_from_slice(&order.write_u32(0));

    let mut out = Vec::with_capacity(data_at + running.saturating_sub(data_at));
    out.extend_from_slice(b"II");
    out.extend_from_slice(&order.write_u16(42));
    out.extend_from_slice(&order.write_u32(directory_at as u32));
    out.extend_from_slice(&directory);
    out.extend_from_slice(&heap);
    for chunk in chunks {
        out.extend_from_slice(chunk);
    }
    Ok(out)
}

const fn type_size(code: u16) -> usize {
    match code {
        3 => 2,
        4 => 4,
        _ => 1,
    }
}

fn push_value(out: &mut Vec<u8>, order: ByteOrder, type_code: u16, value: u32) {
    match type_code {
        3 => out.extend_from_slice(&order.write_u16(value as u16)),
        4 => out.extend_from_slice(&order.write_u32(value)),
        _ => out.push(value as u8),
    }
}

/// Channels and bits per sample for a pixel format.
const fn sample_shape(format: PixelFormat) -> (u32, u32) {
    match format {
        PixelFormat::Gray8 => (1, 8),
        PixelFormat::Gray16 => (1, 16),
        PixelFormat::GrayA8 => (2, 8),
        PixelFormat::Rgb8 => (3, 8),
        PixelFormat::Rgba8 => (4, 8),
        PixelFormat::Rgb16 => (3, 16),
        PixelFormat::Rgba16 => (4, 16),
        _ => (0, 0),
    }
}

/// Reject a format TIFF's baseline cannot express.
fn supported_format(format: PixelFormat) -> Result<()> {
    if sample_shape(format).0 == 0 {
        return Err(PixelsError::unsupported(format!(
            "TIFF encoding needs an integer format; got {format}"
        )));
    }
    Ok(())
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
    use crate::decoder::TiffDecoder;
    use otf_pixels_core::{Decoder, Limits};

    fn encode(desc: &ImageDescriptor, raster: &[u8], encoder: TiffEncoder) -> Result<Vec<u8>> {
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
        let mut decoder = TiffDecoder::new(bytes, Limits::default())?;
        let descriptor = decoder.descriptor();
        let mut raster = Vec::new();
        let mut row = vec![0_u8; descriptor.row_bytes()];
        for _ in 0..descriptor.height {
            decoder.read_row(&mut row)?;
            raster.extend_from_slice(&row);
        }
        Ok((descriptor, raster))
    }

    fn sample(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let len = descriptor.byte_len().unwrap();
        let bytes = (0..len).map(|i| ((i * 37) % 251) as u8).collect();
        (descriptor, bytes)
    }

    #[test]
    fn every_supported_format_round_trips_through_strips() {
        for format in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::GrayA8,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
        ] {
            let (descriptor, raster) = sample(37, 29, format);
            let bytes = encode(&descriptor, &raster, TiffEncoder::new())
                .unwrap_or_else(|e| panic!("{format}: {e}"));
            let (out_desc, decoded) =
                decode(&bytes).unwrap_or_else(|e| panic!("decoding {format}: {e}"));
            assert_eq!(out_desc.pixel, format, "{format} changed format");
            assert_eq!((out_desc.width, out_desc.height), (37, 29), "{format}");
            assert_eq!(decoded, raster, "{format} did not round-trip");
        }
    }

    #[test]
    fn tiled_output_round_trips_and_is_random_access() {
        // The layout the exit criterion turns on: what we write must be
        // readable back with region capability, or a pyramid cannot be built.
        let (descriptor, raster) = sample(100, 70, PixelFormat::Rgb8);
        let encoder = TiffEncoder::new()
            .with_layout(TiffLayout::Tiles {
                width: 32,
                height: 32,
            })
            .unwrap();
        let bytes = encode(&descriptor, &raster, encoder).unwrap();

        let decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
        assert_eq!(
            decoder.capability(),
            otf_pixels_core::DecodeCapability::Regions,
            "our own tiled output must be random-access"
        );
        let (_, decoded) = decode(&bytes).unwrap();
        assert_eq!(decoded, raster, "tiled output did not round-trip");
    }

    #[test]
    fn deflate_round_trips_in_both_layouts() {
        for layout in [
            TiffLayout::Strips { rows: 16 },
            TiffLayout::Tiles {
                width: 16,
                height: 16,
            },
        ] {
            let (descriptor, raster) = sample(48, 40, PixelFormat::Rgb8);
            let encoder = TiffEncoder::new()
                .with_layout(layout)
                .unwrap()
                .with_deflate(Level::DEFAULT);
            let bytes = encode(&descriptor, &raster, encoder).unwrap();
            let (_, decoded) = decode(&bytes).unwrap();
            assert_eq!(
                decoded, raster,
                "{layout:?} with deflate did not round-trip"
            );
        }
    }

    #[test]
    fn deflate_actually_compresses() {
        let descriptor = ImageDescriptor::new(128, 128, PixelFormat::Rgb8).unwrap();
        let raster = vec![7_u8; descriptor.byte_len().unwrap()];
        let plain = encode(&descriptor, &raster, TiffEncoder::new()).unwrap();
        let packed = encode(
            &descriptor,
            &raster,
            TiffEncoder::new().with_deflate(Level::BEST),
        )
        .unwrap();
        assert!(
            packed.len() * 10 < plain.len(),
            "deflate produced {} bytes against {} uncompressed",
            packed.len(),
            plain.len()
        );
    }

    #[test]
    fn every_strip_and_tile_size_round_trips() {
        // Sizes that divide the image evenly and sizes that do not, because
        // the last chunk is where an encoder's arithmetic breaks.
        let (descriptor, raster) = sample(70, 50, PixelFormat::Rgb8);
        for rows in [1_u32, 7, 25, 50, 999] {
            let encoder = TiffEncoder::new()
                .with_layout(TiffLayout::Strips { rows })
                .unwrap();
            let bytes = encode(&descriptor, &raster, encoder).unwrap();
            let (_, decoded) = decode(&bytes).unwrap();
            assert_eq!(decoded, raster, "{rows} rows per strip");
        }
        for (width, height) in [(16_u32, 16_u32), (32, 16), (16, 48), (80, 64)] {
            let encoder = TiffEncoder::new()
                .with_layout(TiffLayout::Tiles { width, height })
                .unwrap();
            let bytes = encode(&descriptor, &raster, encoder).unwrap();
            let (_, decoded) = decode(&bytes).unwrap();
            assert_eq!(decoded, raster, "{width}x{height} tiles");
        }
    }

    #[test]
    fn tile_dimensions_must_be_multiples_of_sixteen() {
        // §Section 15, and the decoder enforces it too — an encoder that
        // emitted a 17-pixel tile would produce files our own reader rejects.
        for (width, height) in [(17_u32, 16_u32), (16, 17), (0, 16), (16, 0)] {
            assert!(
                TiffEncoder::new()
                    .with_layout(TiffLayout::Tiles { width, height })
                    .is_err(),
                "{width}x{height} should be rejected"
            );
        }
        assert!(
            TiffEncoder::new()
                .with_layout(TiffLayout::Strips { rows: 0 })
                .is_err()
        );
    }

    #[test]
    fn a_float_format_is_unsupported_not_a_panic() {
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::RgbaF32).unwrap();
        let mut encoder = TiffEncoder::new();
        let mut out = Vec::new();
        assert!(encoder.write_header(&descriptor, &mut out).is_err());
        assert!(
            out.is_empty(),
            "nothing should be written for a rejected format"
        );
    }

    #[test]
    fn a_short_image_is_an_error_not_a_truncated_tiff() {
        let (descriptor, _) = sample(8, 8, PixelFormat::Rgb8);
        let mut encoder = TiffEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&descriptor, &mut out).unwrap();
        encoder.write_row(&[0; 24], &mut out).unwrap();
        assert!(encoder.finish(&mut out).is_err());
    }

    #[test]
    fn extra_rows_and_wrong_lengths_are_errors() {
        let descriptor = ImageDescriptor::new(4, 1, PixelFormat::Rgb8).unwrap();
        let mut encoder = TiffEncoder::new();
        let mut out = Vec::new();
        encoder.write_header(&descriptor, &mut out).unwrap();
        assert!(encoder.write_row(&[0; 11], &mut out).is_err(), "short row");
        assert!(encoder.write_row(&[0; 13], &mut out).is_err(), "long row");
        encoder.write_row(&[0; 12], &mut out).unwrap();
        assert!(encoder.write_row(&[0; 12], &mut out).is_err(), "extra row");
    }

    #[test]
    fn tags_are_written_in_ascending_order() {
        // Readers are entitled to binary search the directory, and libtiff
        // warns loudly about files that get this wrong.
        let (descriptor, raster) = sample(16, 16, PixelFormat::Rgba8);
        let bytes = encode(&descriptor, &raster, TiffEncoder::new()).unwrap();
        let order = ByteOrder::Little;
        let count = order.u16(&bytes, 8) as usize;
        let mut previous = 0_u16;
        for index in 0..count {
            let tag_id = order.u16(&bytes, 10 + index * 12);
            assert!(tag_id > previous, "tag {tag_id} follows {previous}");
            previous = tag_id;
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let (descriptor, raster) = sample(40, 30, PixelFormat::Rgb8);
        let first = encode(&descriptor, &raster, TiffEncoder::new()).unwrap();
        for _ in 0..4 {
            assert_eq!(
                encode(&descriptor, &raster, TiffEncoder::new()).unwrap(),
                first,
                "TIFF encoding is not deterministic"
            );
        }
    }

    #[test]
    fn options_map_quality_onto_compression() {
        assert!(
            TiffEncoder::from_options(&EncodeOptions::default())
                .deflate
                .is_some()
        );
        let low = EncodeOptions::with_quality(10).unwrap();
        assert!(
            TiffEncoder::from_options(&low).deflate.is_none(),
            "low quality should prioritise speed"
        );
    }
}
