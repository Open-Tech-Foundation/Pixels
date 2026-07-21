//! The TIFF decoder.
//!
//! # Random access is the point
//!
//! A tiled TIFF stores independently compressed rectangles, each with its own
//! offset, so producing an arbitrary region means decompressing the tiles that
//! region touches and nothing else. [`TiffDecoder`] reports
//! [`DecodeCapability::Regions`] for such a file, and the scheduler then pulls
//! regions rather than rows — which is how a 2 GB scan becomes a thumbnail
//! without ever materializing the full-resolution image.
//!
//! A strip TIFF has no such property: a strip is the full image width, so
//! "the tiles this region touches" is "every strip it crosses", and the
//! decoder reports [`DecodeCapability::Sequential`] instead. Claiming
//! otherwise would be a lie the scheduler would act on.
//!
//! # Why this decoder needs the whole file
//!
//! TIFF's offsets point anywhere: the IFD is commonly at the *end*, and tiles
//! are in no particular order. Random access is therefore incompatible with a
//! forward-only source, and the decoder reads its input into memory once.
//!
//! That is a real cost and it is worth being precise about what it buys: the
//! *pixels* still stream, because only the tiles a region touches are ever
//! decompressed. A 2 GB tiled TIFF costs 2 GB of file buffer and a handful of
//! decompressed tiles, not 2 GB of pixels — which for a 16-bit RGB scan is a
//! factor of six. Memory-mapping the source would remove even that, and is
//! deferred rather than dismissed.

use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Format, ImageDescriptor, Limits, PixelFormat, PixelsError,
    Region, Result, Source, TileMut,
};

use crate::ifd::{ByteOrder, Directory, parse_header, probe as probe_header};
use crate::image::{Layout, Photometric, TiffImage};

/// Decodes a TIFF stream.
#[derive(Debug)]
pub struct TiffDecoder {
    data: Vec<u8>,
    image: TiffImage,
    row: u32,
    /// The most recently decoded chunk, keyed by index.
    ///
    /// One chunk is enough: sequential reads walk chunks in order, and region
    /// reads touch each chunk's rows consecutively. A larger cache belongs to
    /// the scheduler, which already has one.
    cached: Option<(usize, Vec<u8>)>,
}

impl TiffDecoder {
    /// Read and parse a TIFF.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a bad header or directory,
    /// [`PixelsError::Unsupported`] for a layout this codec does not
    /// implement, or [`PixelsError::LimitExceeded`] if the image exceeds
    /// `limits`.
    pub fn new<S: Source>(mut source: S, limits: Limits) -> Result<Self> {
        // TIFF offsets point anywhere, so the whole file is read. See the
        // module docs for what that does and does not cost.
        let mut data = Vec::new();
        let mut buffer = vec![0_u8; 256 * 1024];
        loop {
            let read = source.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            let Some(chunk) = buffer.get(..read) else {
                break;
            };
            data.extend_from_slice(chunk);
        }

        let (order, first_ifd) = parse_header(&data)?;
        let directory = Directory::parse(&data, order, first_ifd)?;
        let image = TiffImage::from_directory(&directory, order, &limits)?;

        Ok(Self {
            data,
            image,
            row: 0,
            cached: None,
        })
    }

    /// The parsed image description.
    #[must_use]
    pub const fn image(&self) -> &TiffImage {
        &self.image
    }

    /// The byte order the file declared.
    #[must_use]
    pub const fn byte_order(&self) -> ByteOrder {
        self.image.order
    }

    /// Ensure chunk `index` is the cached one, decompressing it if not.
    fn ensure_chunk(&mut self, index: usize) -> Result<()> {
        if matches!(&self.cached, Some((cached, _)) if *cached == index) {
            return Ok(());
        }
        let decoded = self.image.read_chunk(&self.data, index)?;
        self.cached = Some((index, decoded));
        Ok(())
    }

    /// Write the part of chunk `(column, row)` that lands inside `region`.
    fn blit_chunk(
        &mut self,
        column: u32,
        chunk_row: u32,
        region: Region,
        out: &mut TileMut<'_>,
    ) -> Result<()> {
        let across = self.image.chunks_across();
        let index = (chunk_row * across + column) as usize;
        if index >= self.image.offsets.len() {
            return Err(PixelsError::malformed(
                "tiff",
                format!(
                    "chunk {index} is beyond the {} declared",
                    self.image.offsets.len()
                ),
            ));
        }

        self.ensure_chunk(index)?;
        // Destructured so the cached chunk and the image description are
        // separate borrows. Cloning either instead — which is what the
        // borrow checker first pushes you toward — would copy the whole
        // decompressed chunk once per output row, making a strip read
        // quadratic in image height and defeating the cache entirely.
        let Self { image, cached, .. } = self;
        let Some((_, data)) = cached.as_ref() else {
            return Err(PixelsError::graph("tiff chunk vanished after decoding"));
        };

        let area = image.chunk_region(column, chunk_row);
        let (stored_width, _) = image.chunk_stored_size();
        let stored_row_bytes = image.chunk_row_bytes();

        // The overlap of this chunk with the requested region.
        let left = area.x.max(region.x);
        let top = area.y.max(region.y);
        let right = (area.x + area.width).min(region.x + region.width);
        let bottom = (area.y + area.height).min(region.y + region.height);
        if right <= left || bottom <= top {
            return Ok(());
        }

        let mut expanded =
            vec![0_u8; (right - left) as usize * image.descriptor.pixel.bytes_per_pixel()];
        for y in top..bottom {
            let within = (y - area.y) as usize;
            let start = within * stored_row_bytes;
            let Some(stored) = data.get(start..start + stored_row_bytes) else {
                continue;
            };
            expand_row(
                image,
                stored,
                (left - area.x) as usize,
                (right - left) as usize,
                stored_width as usize,
                &mut expanded,
            );
            let Some(target) = out.row_mut(y) else {
                continue;
            };
            let bpp = image.descriptor.pixel.bytes_per_pixel();
            let at = (left - region.x) as usize * bpp;
            let Some(slot) = target.get_mut(at..at + expanded.len()) else {
                continue;
            };
            slot.copy_from_slice(&expanded);
        }
        Ok(())
    }

    /// Fill `out` with `region`, decoding only the chunks it touches.
    fn read_region_into(&mut self, region: Region, out: &mut TileMut<'_>) -> Result<()> {
        if region.x + region.width > self.image.descriptor.width
            || region.y + region.height > self.image.descriptor.height
        {
            return Err(PixelsError::invalid_argument(
                "region",
                format!(
                    "{region} is outside a {}x{} image",
                    self.image.descriptor.width, self.image.descriptor.height
                ),
            ));
        }

        let (first_column, last_column, first_row, last_row) = self.chunks_covering(region);
        for chunk_row in first_row..=last_row {
            for column in first_column..=last_column {
                self.blit_chunk(column, chunk_row, region, out)?;
            }
        }
        Ok(())
    }

    /// The inclusive chunk grid coordinates a region touches.
    ///
    /// This is the whole random-access story in four numbers: everything
    /// outside this range is never read, never decompressed, and never paid
    /// for.
    fn chunks_covering(&self, region: Region) -> (u32, u32, u32, u32) {
        match self.image.layout {
            Layout::Strips { rows_per_strip } => {
                let first = region.y / rows_per_strip;
                let last = (region.y + region.height.saturating_sub(1)) / rows_per_strip;
                (0, 0, first, last.min(self.image.chunks_down() - 1))
            }
            Layout::Tiles { width, height } => {
                let first_column = region.x / width;
                let last_column = (region.x + region.width.saturating_sub(1)) / width;
                let first_row = region.y / height;
                let last_row = (region.y + region.height.saturating_sub(1)) / height;
                (
                    first_column,
                    last_column.min(self.image.chunks_across() - 1),
                    first_row,
                    last_row.min(self.image.chunks_down() - 1),
                )
            }
        }
    }
}

/// Expand `count` stored pixels starting at `from` into output format.
fn expand_row(
    image: &TiffImage,
    stored: &[u8],
    from: usize,
    count: usize,
    stored_width: usize,
    out: &mut [u8],
) {
    let bits = image.bits_per_sample as usize;
    let channels = image.samples_per_pixel as usize;
    let format = image.descriptor.pixel;
    let bpp = format.bytes_per_pixel();
    let maximum = if bits >= 16 {
        65535_u32
    } else {
        (1_u32 << bits) - 1
    };

    for index in 0..count {
        let x = from + index;
        if x >= stored_width {
            break;
        }
        let Some(target) = out.get_mut(index * bpp..(index + 1) * bpp) else {
            break;
        };

        // Read every channel of this pixel as a sample in 0..=maximum.
        let mut samples = [0_u32; 4];
        for (channel, slot) in samples.iter_mut().enumerate().take(channels.min(4)) {
            *slot = read_sample(stored, x * channels + channel, bits, image.order);
        }

        match image.photometric {
            Photometric::Palette => {
                // The colour map is 3 * 2^bits 16-bit values, stored as all
                // reds, then all greens, then all blues — not interleaved,
                // which is the thing that catches every first implementation.
                let entries = 1_usize << bits;
                let index = samples[0] as usize;
                for channel in 0..3 {
                    let value = image
                        .color_map
                        .get(channel * entries + index)
                        .copied()
                        .unwrap_or(0);
                    if let Some(slot) = target.get_mut(channel) {
                        // The map is 16-bit; 8-bit output is what consumers
                        // want and a 256-entry table loses nothing by it.
                        *slot = (value >> 8) as u8;
                    }
                }
            }
            _ => {
                for (channel, &sample) in samples.iter().enumerate().take(channels.min(4)) {
                    let mut value = sample;
                    // WhiteIsZero is an inverted greyscale. Inverting only the
                    // colour channels leaves alpha alone, which is why this is
                    // not a blanket negation of the pixel.
                    let is_colour = channel < 3;
                    if image.photometric == Photometric::WhiteIsZero && is_colour {
                        value = maximum.saturating_sub(value);
                    }
                    write_sample(target, channel, value, bits, maximum, format);
                }
            }
        }
    }
}

/// Read stored sample `index` at `bits` per sample.
fn read_sample(data: &[u8], index: usize, bits: usize, order: ByteOrder) -> u32 {
    match bits {
        16 => u32::from(order.u16(data, index * 2)),
        8 => u32::from(data.get(index).copied().unwrap_or(0)),
        1 | 2 | 4 => {
            // Sub-byte samples are packed most-significant-first within each
            // byte, which is the opposite of what an index-first reading gives.
            let per_byte = 8 / bits;
            let byte = data.get(index / per_byte).copied().unwrap_or(0);
            let shift = 8 - bits * (index % per_byte + 1);
            u32::from((byte >> shift) & ((1_u16 << bits) - 1) as u8)
        }
        _ => 0,
    }
}

/// Write one channel of an output pixel, scaling to the format's range.
fn write_sample(
    target: &mut [u8],
    channel: usize,
    value: u32,
    bits: usize,
    maximum: u32,
    format: PixelFormat,
) {
    let widened = if bits >= 8 {
        value
    } else {
        // Scale a sub-byte sample to the full range rather than shifting: a
        // 1-bit white must become 255, not 128.
        (value * 255 + maximum / 2) / maximum.max(1)
    };
    match format.sample_kind() {
        otf_pixels_core::SampleKind::U16 => {
            let scaled = widened as u16;
            for (offset, byte) in scaled.to_ne_bytes().iter().enumerate() {
                if let Some(slot) = target.get_mut(channel * 2 + offset) {
                    *slot = *byte;
                }
            }
        }
        _ => {
            if let Some(slot) = target.get_mut(channel) {
                *slot = widened.min(255) as u8;
            }
        }
    }
}

impl Decoder for TiffDecoder {
    fn descriptor(&self) -> ImageDescriptor {
        self.image.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        // Only a tiled file can answer an arbitrary region cheaply. Claiming
        // otherwise for a strip file would be a lie the scheduler acts on.
        if self.image.layout.is_random_access() {
            DecodeCapability::Regions
        } else {
            DecodeCapability::Sequential
        }
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        if self.row >= self.image.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!(
                    "all {} rows have already been read",
                    self.image.descriptor.height
                ),
            ));
        }
        let row_bytes = self.image.descriptor.row_bytes();
        if out.len() != row_bytes {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {row_bytes}", out.len()),
            ));
        }

        // Decoded straight into the caller's buffer. Allocating a scratch tile
        // per row would be wasteful on every image and catastrophic on a
        // corrupt one, where a bogus ImageWidth within the pixel limit still
        // implies a very large single row.
        let region = Region::new(0, self.row, self.image.descriptor.width, 1);
        let pixel = self.image.descriptor.pixel;
        let mut tile = TileMut::new(region, pixel, row_bytes, out)?;
        self.read_region_into(region, &mut tile)?;
        self.row += 1;
        Ok(())
    }

    fn read_region(&mut self, region: Region, out: &mut TileMut<'_>) -> Result<()> {
        if !self.image.layout.is_random_access() {
            return Err(PixelsError::unsupported(
                "this TIFF is stored in strips; region decode requires tiles",
            ));
        }
        self.read_region_into(region, out)
    }
}

/// Whether `prefix` starts with a TIFF header.
///
/// Detection is by magic bytes only (SPEC §Formats).
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    probe_header(prefix)
}

/// The TIFF entry in a sniffing registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct TiffCodec;

impl Codec for TiffCodec {
    fn format(&self) -> Format {
        Format::Tiff
    }

    fn magic_len(&self) -> usize {
        8
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
    }
}
