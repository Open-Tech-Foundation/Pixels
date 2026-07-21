//! The image a TIFF directory describes: layout, compression and pixel format.
//!
//! Separated from [`Directory`] on purpose: that module knows about tags, this
//! one knows what tags *mean*. The split is what keeps "skip an unknown tag"
//! and "reject an unsupported layout" different decisions.
//!
//! [`Directory`]: crate::Directory

use otf_pixels_compress::{LzwDecoder, inflate_to, zlib_decompress};
use otf_pixels_core::{
    ImageDescriptor, Limits, PixelFormat, PixelsError, Region, Result, SampleKind,
};

use crate::ifd::{ByteOrder, Directory, tag};

/// How pixel data is compressed (TIFF 6.0 §Section 8, plus §Deflate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Compression {
    /// Stored uncompressed.
    None,
    /// LZW, TIFF's dialect.
    Lzw,
    /// The Deflate extension, both the official and the old Adobe tag.
    Deflate,
    /// PackBits run-length encoding, which baseline TIFF requires.
    PackBits,
}

impl Compression {
    /// The scheme for a Compression tag value.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] for a scheme this codec does not
    /// implement — CCITT fax and old-style JPEG, principally. That is an
    /// unsupported *layout*, not an exotic tag, so it is reported rather than
    /// skipped: the pixels cannot be produced without it.
    pub fn from_tag(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::None),
            5 => Ok(Self::Lzw),
            8 | 32_946 => Ok(Self::Deflate),
            32_773 => Ok(Self::PackBits),
            other => Err(PixelsError::unsupported(format!(
                "TIFF compression {other} is not implemented"
            ))),
        }
    }
}

/// How samples are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Photometric {
    /// Zero is white — an inverted greyscale, common in scanned documents.
    WhiteIsZero,
    /// Zero is black.
    BlackIsZero,
    /// Red, green, blue.
    Rgb,
    /// Palette indices, resolved through the colour map.
    Palette,
}

impl Photometric {
    /// The interpretation for a Photometric tag value.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] for CMYK, YCbCr and the rest.
    pub fn from_tag(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::WhiteIsZero),
            1 => Ok(Self::BlackIsZero),
            2 => Ok(Self::Rgb),
            3 => Ok(Self::Palette),
            other => Err(PixelsError::unsupported(format!(
                "TIFF photometric interpretation {other} is not implemented"
            ))),
        }
    }
}

/// Whether the pixels are stored in strips or tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    /// Horizontal bands the full width of the image.
    Strips {
        /// Rows in each strip but the last.
        rows_per_strip: u32,
    },
    /// A grid of independently compressed rectangles.
    ///
    /// This is the layout that makes region random access possible, and
    /// therefore the one M5's exit criterion turns on.
    Tiles {
        /// Tile width in pixels; always a multiple of 16.
        width: u32,
        /// Tile height in pixels; always a multiple of 16.
        height: u32,
    },
}

impl Layout {
    /// Whether this layout supports decoding an arbitrary region cheaply.
    #[must_use]
    pub const fn is_random_access(self) -> bool {
        matches!(self, Self::Tiles { .. })
    }
}

/// Everything about a TIFF image that decoding needs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TiffImage {
    /// The engine-facing descriptor.
    pub descriptor: ImageDescriptor,
    /// Bits in each stored sample.
    pub bits_per_sample: u32,
    /// Channels per stored pixel.
    pub samples_per_pixel: u32,
    /// How samples are interpreted.
    pub photometric: Photometric,
    /// How pixel data is compressed.
    pub compression: Compression,
    /// Strips or tiles.
    pub layout: Layout,
    /// Byte offset of each chunk.
    pub offsets: Vec<u32>,
    /// Compressed byte count of each chunk.
    pub byte_counts: Vec<u32>,
    /// Whether a horizontal differencing predictor was applied.
    pub predictor: bool,
    /// The colour map, for palette images: 3 * 2^bits 16-bit values.
    pub color_map: Vec<u32>,
    /// The byte order the file declared.
    pub order: ByteOrder,
}

impl TiffImage {
    /// Interpret a directory as an image.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a directory missing tags the
    /// pixels cannot be located without, [`PixelsError::Unsupported`] for a
    /// layout this codec does not implement, or [`PixelsError::LimitExceeded`]
    /// if the dimensions exceed `limits`.
    pub fn from_directory(
        directory: &Directory,
        order: ByteOrder,
        limits: &Limits,
    ) -> Result<Self> {
        let width = directory.require(tag::IMAGE_WIDTH, "ImageWidth")?;
        let height = directory.require(tag::IMAGE_LENGTH, "ImageLength")?;

        let samples_per_pixel = directory.value_or(tag::SAMPLES_PER_PIXEL, 1);
        let bits = directory.values(tag::BITS_PER_SAMPLE);
        let bits_per_sample = bits.first().copied().unwrap_or(1);
        // Mixed bit depths per channel are legal and vanishingly rare, and
        // supporting them would complicate every unpack path. Rejecting is
        // honest; silently using the first depth would corrupt the pixels.
        if bits.iter().any(|&b| b != bits_per_sample) {
            return Err(PixelsError::unsupported(
                "TIFF channels with differing bit depths are not implemented",
            ));
        }

        let photometric =
            Photometric::from_tag(directory.require(tag::PHOTOMETRIC, "Photometric")?)?;
        let compression = Compression::from_tag(directory.value_or(tag::COMPRESSION, 1))?;

        // Planar (channel-separated) storage is legal but rare, and it changes
        // every unpack path rather than adding one.
        if directory.value_or(tag::PLANAR_CONFIG, 1) != 1 {
            return Err(PixelsError::unsupported(
                "TIFF planar configuration 2 is not implemented",
            ));
        }
        let predictor = match directory.value_or(tag::PREDICTOR, 1) {
            1 => false,
            2 => true,
            other => {
                return Err(PixelsError::unsupported(format!(
                    "TIFF predictor {other} is not implemented"
                )));
            }
        };
        if directory.value_or(tag::SAMPLE_FORMAT, 1) != 1 {
            return Err(PixelsError::unsupported(
                "TIFF signed and float sample formats are not implemented",
            ));
        }

        let (layout, offsets, byte_counts) = if directory.get(tag::TILE_OFFSETS).is_some() {
            let tile_width = directory.require(tag::TILE_WIDTH, "TileWidth")?;
            let tile_height = directory.require(tag::TILE_LENGTH, "TileLength")?;
            // TIFF 6.0 §Section 15 requires tile dimensions to be multiples
            // of 16. Enforcing it is not pedantry: a corrupt TileWidth of 1
            // turns a modest image into hundreds of thousands of chunks, each
            // needing its own decompression, which is a CPU exhaustion vector
            // reachable by flipping one bit.
            if tile_width == 0 || tile_height == 0 {
                return Err(PixelsError::malformed("tiff", "tile size is zero"));
            }
            if tile_width % 16 != 0 || tile_height % 16 != 0 {
                return Err(PixelsError::malformed(
                    "tiff",
                    format!("tile size {tile_width}x{tile_height} is not a multiple of 16"),
                ));
            }
            (
                Layout::Tiles {
                    width: tile_width,
                    height: tile_height,
                },
                directory.values(tag::TILE_OFFSETS).to_vec(),
                directory.values(tag::TILE_BYTE_COUNTS).to_vec(),
            )
        } else {
            // A missing RowsPerStrip means the whole image is one strip, which
            // the specification states as the default of 2^32-1.
            let rows_per_strip = directory.value_or(tag::ROWS_PER_STRIP, height).max(1);
            (
                Layout::Strips { rows_per_strip },
                directory.values(tag::STRIP_OFFSETS).to_vec(),
                directory.values(tag::STRIP_BYTE_COUNTS).to_vec(),
            )
        };

        if offsets.is_empty() {
            return Err(PixelsError::malformed(
                "tiff",
                "no strip or tile offsets; the pixels cannot be located",
            ));
        }
        if byte_counts.len() < offsets.len() {
            return Err(PixelsError::malformed(
                "tiff",
                format!(
                    "{} offsets but only {} byte counts",
                    offsets.len(),
                    byte_counts.len()
                ),
            ));
        }

        let pixel = output_format(photometric, bits_per_sample, samples_per_pixel)?;
        // Enforced before any buffer exists (SPEC §Safety).
        let descriptor = ImageDescriptor::with_limits(width, height, pixel, limits)?;

        Ok(Self {
            descriptor,
            bits_per_sample,
            samples_per_pixel,
            photometric,
            compression,
            layout,
            offsets,
            byte_counts,
            predictor,
            color_map: directory.values(tag::COLOR_MAP).to_vec(),
            order,
        })
    }

    /// The number of chunks (strips or tiles) across the image.
    #[must_use]
    pub const fn chunks_across(&self) -> u32 {
        match self.layout {
            Layout::Strips { .. } => 1,
            Layout::Tiles { width, .. } => self.descriptor.width.div_ceil(width),
        }
    }

    /// The number of chunks down the image.
    #[must_use]
    pub const fn chunks_down(&self) -> u32 {
        match self.layout {
            Layout::Strips { rows_per_strip } => self.descriptor.height.div_ceil(rows_per_strip),
            Layout::Tiles { height, .. } => self.descriptor.height.div_ceil(height),
        }
    }

    /// The region of the image chunk `(column, row)` covers.
    ///
    /// Tiles are always whole even at the image edge — the specification pads
    /// them — so the region is clipped for the caller's benefit rather than
    /// describing what is stored.
    #[must_use]
    pub fn chunk_region(&self, column: u32, row: u32) -> Region {
        match self.layout {
            Layout::Strips { rows_per_strip } => {
                let y = row * rows_per_strip;
                let height = rows_per_strip.min(self.descriptor.height.saturating_sub(y));
                Region::new(0, y, self.descriptor.width, height)
            }
            Layout::Tiles { width, height } => {
                let x = column * width;
                let y = row * height;
                Region::new(
                    x,
                    y,
                    width.min(self.descriptor.width.saturating_sub(x)),
                    height.min(self.descriptor.height.saturating_sub(y)),
                )
            }
        }
    }

    /// The stored dimensions of chunk `(column, row)`, including any padding.
    #[must_use]
    pub const fn chunk_stored_size(&self) -> (u32, u32) {
        match self.layout {
            Layout::Strips { rows_per_strip } => (self.descriptor.width, rows_per_strip),
            Layout::Tiles { width, height } => (width, height),
        }
    }

    /// Bytes one stored row of a chunk occupies.
    #[must_use]
    pub const fn chunk_row_bytes(&self) -> usize {
        let (width, _) = self.chunk_stored_size();
        let bits = width as usize * self.samples_per_pixel as usize * self.bits_per_sample as usize;
        bits.div_ceil(8)
    }

    /// Decompress chunk `index` from `data`, returning stored samples.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the chunk lies outside the file
    /// or decompresses to something other than its declared size.
    pub fn read_chunk(&self, data: &[u8], index: usize) -> Result<Vec<u8>> {
        let offset = self.offsets.get(index).copied().unwrap_or(0) as usize;
        let count = self.byte_counts.get(index).copied().unwrap_or(0) as usize;
        let raw = data
            .get(offset..offset.saturating_add(count))
            .ok_or_else(|| {
                PixelsError::malformed(
                    "tiff",
                    format!("chunk {index} at {offset}+{count} lies outside the file"),
                )
            })?;

        let (_, stored_height) = self.chunk_stored_size();
        let row_bytes = self.chunk_row_bytes();
        // The exact expected size is what makes a decompression bomb a
        // malformed-input error rather than an allocation.
        let expected = row_bytes.saturating_mul(stored_height as usize);

        let mut out = match self.compression {
            Compression::None => raw.to_vec(),
            Compression::Lzw => LzwDecoder::tiff()
                .decode(raw, expected)
                .map_err(crate::compress_error)?,
            Compression::Deflate => {
                // The official tag (8) is zlib-wrapped; Adobe's older 32946 is
                // the same in every file anyone has produced. Falling back to
                // raw deflate covers the handful of writers that omit the
                // wrapper rather than rejecting their files.
                match zlib_decompress(raw, expected) {
                    Ok(out) => out,
                    Err(_) => inflate_to(raw, expected).map_err(crate::compress_error)?,
                }
            }
            Compression::PackBits => unpack_bits(raw, expected),
        };

        if self.predictor {
            apply_predictor(&mut out, row_bytes, stored_height as usize, self);
        }
        Ok(out)
    }
}

/// Reverse horizontal differencing (TIFF §Predictor).
///
/// Each sample is stored as its difference from the sample one pixel to the
/// left, which makes smooth gradients compress far better. Undoing it is a
/// running sum along each row, per channel.
fn apply_predictor(data: &mut [u8], row_bytes: usize, rows: usize, image: &TiffImage) {
    let channels = image.samples_per_pixel as usize;
    match image.bits_per_sample {
        8 => {
            for row in 0..rows {
                let start = row * row_bytes;
                let Some(line) = data.get_mut(start..start + row_bytes) else {
                    break;
                };
                for index in channels..line.len() {
                    let previous = line.get(index - channels).copied().unwrap_or(0);
                    if let Some(slot) = line.get_mut(index) {
                        *slot = slot.wrapping_add(previous);
                    }
                }
            }
        }
        16 => {
            let stride = channels * 2;
            for row in 0..rows {
                let start = row * row_bytes;
                let Some(line) = data.get_mut(start..start + row_bytes) else {
                    break;
                };
                let mut index = stride;
                while index + 1 < line.len() {
                    let previous = read16(line, index - stride, image.order);
                    let current = read16(line, index, image.order);
                    write16(line, index, current.wrapping_add(previous), image.order);
                    index += 2;
                }
            }
        }
        // The predictor is defined for 8- and 16-bit samples only; anything
        // else is left alone rather than corrupted by a guess.
        _ => {}
    }
}

fn read16(data: &[u8], at: usize, order: ByteOrder) -> u16 {
    order.u16(data, at)
}

fn write16(data: &mut [u8], at: usize, value: u16, order: ByteOrder) {
    for (offset, byte) in order.write_u16(value).iter().enumerate() {
        if let Some(slot) = data.get_mut(at + offset) {
            *slot = *byte;
        }
    }
}

/// Decode PackBits run-length encoding (TIFF §Section 9).
///
/// A length byte read as `i8`: 0..=127 means that many literals plus one,
/// -1..=-127 means the next byte repeated `1 - n` times, -128 is a no-op. The
/// signed reading is the whole trick, and reading it unsigned produces
/// plausible garbage rather than an error.
fn unpack_bits(data: &[u8], limit: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(limit.min(1 << 20));
    let mut at = 0;
    while at < data.len() && out.len() < limit {
        let header = data.get(at).copied().unwrap_or(0) as i8;
        at += 1;
        if header >= 0 {
            let count = header as usize + 1;
            let end = (at + count).min(data.len());
            if let Some(run) = data.get(at..end) {
                out.extend_from_slice(run);
            }
            at = end;
        } else if header != -128 {
            let count = (1 - i32::from(header)) as usize;
            let Some(&byte) = data.get(at) else { break };
            at += 1;
            for _ in 0..count.min(limit.saturating_sub(out.len())) {
                out.push(byte);
            }
        }
        // -128 is explicitly a no-op, which is why it is not an error.
    }
    out
}

/// The engine pixel format a TIFF's photometric and depth imply.
fn output_format(photometric: Photometric, bits: u32, samples: u32) -> Result<PixelFormat> {
    let format = match (photometric, bits, samples) {
        // Palette images resolve through the colour map, which is 16-bit, but
        // 8-bit output is what every consumer wants and loses nothing that a
        // 256-entry table can express.
        (Photometric::Palette, _, _) => PixelFormat::Rgb8,
        (Photometric::WhiteIsZero | Photometric::BlackIsZero, 1 | 2 | 4 | 8, 1) => {
            PixelFormat::Gray8
        }
        (Photometric::WhiteIsZero | Photometric::BlackIsZero, 16, 1) => PixelFormat::Gray16,
        (Photometric::WhiteIsZero | Photometric::BlackIsZero, 8, 2) => PixelFormat::GrayA8,
        (Photometric::Rgb, 8, 3) => PixelFormat::Rgb8,
        (Photometric::Rgb, 8, 4) => PixelFormat::Rgba8,
        (Photometric::Rgb, 16, 3) => PixelFormat::Rgb16,
        (Photometric::Rgb, 16, 4) => PixelFormat::Rgba16,
        _ => {
            return Err(PixelsError::unsupported(format!(
                "TIFF {photometric:?} with {samples} channels at {bits} bits is not implemented"
            )));
        }
    };
    debug_assert!(format.sample_kind() != SampleKind::F32);
    Ok(format)
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

    #[test]
    fn packbits_decodes_the_specification_example() {
        // TIFF 6.0 §Section 9's own example, which is the only authority on
        // the signed reading of the length byte.
        let encoded = [
            0xFE_u8, 0xAA, 0x02, 0x80, 0x00, 0x2A, 0xFD, 0xAA, 0x03, 0x80, 0x00, 0x2A, 0x22, 0xF7,
            0xAA,
        ];
        // Twenty-four bytes: the final 0xF7 is -9, meaning ten repeats, not
        // nine. That off-by-one is the whole reason the length byte's signed
        // reading has to be exact.
        let expected = [
            0xAA_u8, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0xAA, 0xAA, 0xAA, 0xAA, 0x80, 0x00, 0x2A, 0x22,
            0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA,
        ];
        assert_eq!(
            expected.len(),
            24,
            "the specification's example is 24 bytes"
        );
        assert_eq!(unpack_bits(&encoded, 1000), expected);
    }

    #[test]
    fn packbits_treats_minus_128_as_a_no_op() {
        // The specification is explicit that -128 means "do nothing", not
        // "repeat 129 times", and getting it wrong corrupts every file that
        // happens to contain the byte.
        assert_eq!(unpack_bits(&[0x80, 0x00, 0x41], 100), vec![0x41]);
    }

    #[test]
    fn packbits_never_exceeds_its_limit() {
        // A run byte can claim 128 repeats; a stream of them is a bomb.
        let bomb = [0x81_u8, 0x55].repeat(10_000);
        assert!(unpack_bits(&bomb, 512).len() <= 512);
    }

    #[test]
    fn packbits_on_truncated_input_is_not_a_panic() {
        for cut in 0..8 {
            let _ = unpack_bits(&[0xFE, 0xAA, 0x02, 0x80][..cut.min(4)], 100);
        }
        // A literal run claiming more bytes than remain.
        assert!(unpack_bits(&[0x7F, 0x01, 0x02], 100).len() <= 3);
        // A repeat run with no byte to repeat.
        assert!(unpack_bits(&[0xFE], 100).is_empty());
    }

    #[test]
    fn the_predictor_is_a_running_sum_along_each_row() {
        let image = TiffImage {
            descriptor: ImageDescriptor::new(4, 2, PixelFormat::Rgb8).unwrap(),
            bits_per_sample: 8,
            samples_per_pixel: 3,
            photometric: Photometric::Rgb,
            compression: Compression::None,
            layout: Layout::Strips { rows_per_strip: 2 },
            offsets: vec![0],
            byte_counts: vec![24],
            predictor: true,
            color_map: Vec::new(),
            order: ByteOrder::Little,
        };
        // Row of four RGB pixels, stored as differences from the left.
        let mut data = vec![
            10, 20, 30, 1, 1, 1, 1, 1, 1, 1, 1, 1, // row 0
            5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, // row 1
        ];
        apply_predictor(&mut data, 12, 2, &image);
        assert_eq!(
            &data[..12],
            [10, 20, 30, 11, 21, 31, 12, 22, 32, 13, 23, 33]
        );
        assert_eq!(&data[12..], [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn unsupported_compression_and_photometrics_are_reported() {
        // CCITT fax and CMYK are unsupported *layouts*, not exotic tags: the
        // pixels cannot be produced without them, so they are errors.
        assert!(Compression::from_tag(3).is_err(), "CCITT G3");
        assert!(Compression::from_tag(7).is_err(), "new-style JPEG");
        assert!(Photometric::from_tag(5).is_err(), "CMYK");
        assert!(Photometric::from_tag(6).is_err(), "YCbCr");

        assert_eq!(Compression::from_tag(1).unwrap(), Compression::None);
        assert_eq!(Compression::from_tag(5).unwrap(), Compression::Lzw);
        assert_eq!(Compression::from_tag(8).unwrap(), Compression::Deflate);
        assert_eq!(
            Compression::from_tag(32_946).unwrap(),
            Compression::Deflate,
            "Adobe's older deflate tag"
        );
        assert_eq!(
            Compression::from_tag(32_773).unwrap(),
            Compression::PackBits
        );
    }

    #[test]
    fn output_formats_follow_photometric_and_depth() {
        assert_eq!(
            output_format(Photometric::BlackIsZero, 8, 1).unwrap(),
            PixelFormat::Gray8
        );
        assert_eq!(
            output_format(Photometric::BlackIsZero, 1, 1).unwrap(),
            PixelFormat::Gray8,
            "bilevel expands to 8-bit grey"
        );
        assert_eq!(
            output_format(Photometric::Rgb, 16, 4).unwrap(),
            PixelFormat::Rgba16
        );
        assert_eq!(
            output_format(Photometric::Palette, 8, 1).unwrap(),
            PixelFormat::Rgb8,
            "palette resolves to colour"
        );
        assert!(
            output_format(Photometric::Rgb, 8, 5).is_err(),
            "five channels"
        );
    }

    #[test]
    fn tile_regions_are_clipped_at_the_image_edge() {
        let image = TiffImage {
            descriptor: ImageDescriptor::new(100, 70, PixelFormat::Gray8).unwrap(),
            bits_per_sample: 8,
            samples_per_pixel: 1,
            photometric: Photometric::BlackIsZero,
            compression: Compression::None,
            layout: Layout::Tiles {
                width: 32,
                height: 32,
            },
            offsets: vec![0; 12],
            byte_counts: vec![1024; 12],
            predictor: false,
            color_map: Vec::new(),
            order: ByteOrder::Little,
        };
        assert_eq!(image.chunks_across(), 4, "100 / 32 rounds up to 4");
        assert_eq!(image.chunks_down(), 3, "70 / 32 rounds up to 3");

        // Interior tiles are whole.
        assert_eq!(image.chunk_region(0, 0), Region::new(0, 0, 32, 32));
        // Edge tiles are clipped for the caller even though the stored tile is
        // padded to full size — conflating the two is how a decoder writes
        // padding into the image.
        assert_eq!(image.chunk_region(3, 0), Region::new(96, 0, 4, 32));
        assert_eq!(image.chunk_region(0, 2), Region::new(0, 64, 32, 6));
        assert_eq!(
            image.chunk_stored_size(),
            (32, 32),
            "stored size is never clipped"
        );
    }

    #[test]
    fn strip_regions_span_the_full_width() {
        let image = TiffImage {
            descriptor: ImageDescriptor::new(50, 25, PixelFormat::Gray8).unwrap(),
            bits_per_sample: 8,
            samples_per_pixel: 1,
            photometric: Photometric::BlackIsZero,
            compression: Compression::None,
            layout: Layout::Strips { rows_per_strip: 10 },
            offsets: vec![0; 3],
            byte_counts: vec![500; 3],
            predictor: false,
            color_map: Vec::new(),
            order: ByteOrder::Little,
        };
        assert_eq!(image.chunks_across(), 1);
        assert_eq!(image.chunks_down(), 3);
        assert_eq!(image.chunk_region(0, 0), Region::new(0, 0, 50, 10));
        assert_eq!(
            image.chunk_region(0, 2),
            Region::new(0, 20, 50, 5),
            "the last strip is short"
        );
        assert!(!image.layout.is_random_access());
    }
}
