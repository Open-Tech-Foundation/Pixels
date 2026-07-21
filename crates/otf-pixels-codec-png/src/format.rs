//! PNG structure: chunks, the header, filters, Adam7 interlacing, palettes.
//!
//! Everything here operates on already-buffered bytes and returns errors for
//! anything malformed. The PNG specification is ISO/IEC 15948; section
//! references below are to it.

use otf_pixels_core::{ImageDescriptor, Limits, PixelFormat, PixelsError, Result, Source};

use crate::checksum::Crc32;

/// The eight-byte PNG signature (§5.2).
///
/// The non-ASCII bytes are deliberate: they catch transfers that mangle
/// line endings or strip the high bit.
pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// A PNG colour type (§11.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorType {
    /// Greyscale.
    Grayscale,
    /// Red, green, blue.
    Rgb,
    /// Palette indices; requires a `PLTE` chunk.
    Palette,
    /// Greyscale with an alpha channel.
    GrayscaleAlpha,
    /// RGB with an alpha channel.
    Rgba,
}

impl ColorType {
    /// The colour type for a header byte.
    fn from_byte(byte: u8) -> Result<Self> {
        match byte {
            0 => Ok(Self::Grayscale),
            2 => Ok(Self::Rgb),
            3 => Ok(Self::Palette),
            4 => Ok(Self::GrayscaleAlpha),
            6 => Ok(Self::Rgba),
            other => Err(PixelsError::malformed(
                "png",
                format!("colour type {other} is not one of 0, 2, 3, 4, 6"),
            )),
        }
    }

    /// The header byte for this colour type.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::Grayscale => 0,
            Self::Rgb => 2,
            Self::Palette => 3,
            Self::GrayscaleAlpha => 4,
            Self::Rgba => 6,
        }
    }

    /// Channels per pixel *in the encoded stream* (palette counts as one).
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Grayscale | Self::Palette => 1,
            Self::GrayscaleAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Whether this colour type permits `depth` (§11.2.2 table).
    #[must_use]
    pub const fn allows_depth(self, depth: u8) -> bool {
        match self {
            Self::Grayscale => matches!(depth, 1 | 2 | 4 | 8 | 16),
            Self::Palette => matches!(depth, 1 | 2 | 4 | 8),
            Self::Rgb | Self::GrayscaleAlpha | Self::Rgba => matches!(depth, 8 | 16),
        }
    }
}

/// A parsed `IHDR` chunk (§11.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Header {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bits per channel: 1, 2, 4, 8 or 16.
    pub bit_depth: u8,
    /// How samples are interpreted.
    pub color_type: ColorType,
    /// Whether the image is Adam7 interlaced.
    pub interlaced: bool,
}

impl Header {
    /// Parse a 13-byte `IHDR` payload, validating it against `limits`.
    ///
    /// Dimension limits are enforced here, before any pixel buffer exists
    /// (SPEC §Safety), so a header claiming 4 billion pixels costs nothing.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for an invalid header, or
    /// [`PixelsError::LimitExceeded`] if the dimensions exceed `limits`.
    pub fn parse(payload: &[u8], limits: &Limits) -> Result<Self> {
        if payload.len() != 13 {
            return Err(PixelsError::malformed(
                "png",
                format!("IHDR must be 13 bytes, got {}", payload.len()),
            ));
        }
        let width = read_u32(payload, 0)?;
        let height = read_u32(payload, 4)?;
        let bit_depth = payload.get(8).copied().unwrap_or(0);
        let color_type = ColorType::from_byte(payload.get(9).copied().unwrap_or(0))?;
        let compression = payload.get(10).copied().unwrap_or(0);
        let filter = payload.get(11).copied().unwrap_or(0);
        let interlace = payload.get(12).copied().unwrap_or(0);

        if width == 0 || height == 0 {
            return Err(PixelsError::malformed(
                "png",
                format!("dimensions must be non-zero, got {width}x{height}"),
            ));
        }
        if !matches!(bit_depth, 1 | 2 | 4 | 8 | 16) {
            return Err(PixelsError::malformed(
                "png",
                format!("bit depth {bit_depth} is not one of 1, 2, 4, 8, 16"),
            ));
        }
        if !color_type.allows_depth(bit_depth) {
            return Err(PixelsError::malformed(
                "png",
                format!("bit depth {bit_depth} is not allowed for colour type {color_type:?}"),
            ));
        }
        if compression != 0 {
            return Err(PixelsError::malformed(
                "png",
                format!("compression method {compression} is not deflate"),
            ));
        }
        if filter != 0 {
            return Err(PixelsError::malformed(
                "png",
                format!("filter method {filter} is not the only defined one"),
            ));
        }
        if !matches!(interlace, 0 | 1) {
            return Err(PixelsError::malformed(
                "png",
                format!("interlace method {interlace} is not none or Adam7"),
            ));
        }
        // Checked before any allocation proportional to the claimed size.
        limits.check(width, height)?;

        Ok(Self {
            width,
            height,
            bit_depth,
            color_type,
            interlaced: interlace == 1,
        })
    }

    /// Bits per pixel in the encoded stream.
    #[must_use]
    pub const fn bits_per_pixel(&self) -> usize {
        self.color_type.channels() * self.bit_depth as usize
    }

    /// Bytes in one filtered scanline of `width` pixels, excluding the filter
    /// byte. Sub-byte depths round up.
    #[must_use]
    pub const fn row_bytes(&self, width: u32) -> usize {
        (width as usize * self.bits_per_pixel()).div_ceil(8)
    }

    /// Bytes per pixel, rounded up — the filter offset (§9.2).
    ///
    /// For depths below 8 this is 1, which is what the filters require.
    #[must_use]
    pub const fn filter_stride(&self) -> usize {
        // `max` is not const-stable, and bits_per_pixel is never zero for a
        // validated header, so the floor is written out longhand.
        let bytes = self.bits_per_pixel().div_ceil(8);
        if bytes == 0 { 1 } else { bytes }
    }

    /// The pixel format this header decodes into.
    ///
    /// The mapping is constrained by SPEC §Pixel formats, which has no
    /// 16-bit grey-with-alpha type. Rather than silently truncating those to
    /// 8 bits, they widen to `Rgba16`: more memory, but no data thrown away by
    /// a decoder the caller did not ask to be lossy.
    ///
    /// Sub-byte greyscale expands to `Gray8` and palettes expand to RGB or
    /// RGBA, because the engine's formats are byte-aligned and unpalettised.
    #[must_use]
    pub const fn output_format(&self, has_transparency: bool) -> PixelFormat {
        let deep = self.bit_depth == 16;
        match self.color_type {
            ColorType::Grayscale => match (has_transparency, deep) {
                (false, false) => PixelFormat::Gray8,
                (false, true) => PixelFormat::Gray16,
                (true, false) => PixelFormat::GrayA8,
                // No GrayA16 exists, so widen rather than truncate.
                (true, true) => PixelFormat::Rgba16,
            },
            ColorType::GrayscaleAlpha => {
                if deep {
                    PixelFormat::Rgba16
                } else {
                    PixelFormat::GrayA8
                }
            }
            ColorType::Rgb => match (has_transparency, deep) {
                (false, false) => PixelFormat::Rgb8,
                (false, true) => PixelFormat::Rgb16,
                (true, false) => PixelFormat::Rgba8,
                (true, true) => PixelFormat::Rgba16,
            },
            ColorType::Rgba => {
                if deep {
                    PixelFormat::Rgba16
                } else {
                    PixelFormat::Rgba8
                }
            }
            // Palette entries are 8-bit by definition (§11.2.3).
            ColorType::Palette => {
                if has_transparency {
                    PixelFormat::Rgba8
                } else {
                    PixelFormat::Rgb8
                }
            }
        }
    }

    /// The engine descriptor for this header.
    ///
    /// # Errors
    ///
    /// Propagates [`ImageDescriptor::with_limits`].
    pub fn descriptor(&self, has_transparency: bool, limits: &Limits) -> Result<ImageDescriptor> {
        ImageDescriptor::with_limits(
            self.width,
            self.height,
            self.output_format(has_transparency),
            limits,
        )
    }

    /// Total filtered bytes the whole image decompresses to.
    ///
    /// This is the exact output limit handed to inflate, which is what turns a
    /// decompression bomb into a malformed-input error (SPEC §Safety).
    #[must_use]
    pub fn filtered_size(&self) -> usize {
        if self.interlaced {
            // Each Adam7 pass is its own filtered raster with its own filter
            // bytes, so the total exceeds the non-interlaced size.
            (0..7)
                .map(|pass| {
                    let (width, height) = adam7_pass_size(pass, self.width, self.height);
                    if width == 0 || height == 0 {
                        0
                    } else {
                        height as usize * (1 + self.row_bytes(width))
                    }
                })
                .sum()
        } else {
            self.height as usize * (1 + self.row_bytes(self.width))
        }
    }
}

/// Read a big-endian `u32` at `offset`.
fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| PixelsError::malformed("png", "truncated 4-byte field"))?;
    Ok(u32::from_be_bytes([
        slice.first().copied().unwrap_or(0),
        slice.get(1).copied().unwrap_or(0),
        slice.get(2).copied().unwrap_or(0),
        slice.get(3).copied().unwrap_or(0),
    ]))
}

/// One PNG chunk (§5.3).
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The four-byte type code, e.g. `IHDR`.
    pub kind: [u8; 4],
    /// The chunk payload, excluding length, type and CRC.
    pub data: Vec<u8>,
}

impl Chunk {
    /// Whether this chunk's type matches `name`.
    #[must_use]
    pub fn is(&self, name: &[u8; 4]) -> bool {
        self.kind == *name
    }

    /// Whether an unrecognised chunk may be skipped (§5.4).
    ///
    /// The fifth bit of the first byte is the ancillary bit: lowercase means
    /// ancillary, so a decoder that does not understand it may ignore it. An
    /// unknown *critical* chunk means the image cannot be rendered correctly,
    /// so it is an error rather than something to skip.
    #[must_use]
    pub fn is_ancillary(&self) -> bool {
        self.kind.first().copied().unwrap_or(0) & 0x20 != 0
    }

    /// The chunk type as a display string, for diagnostics.
    #[must_use]
    pub fn name(&self) -> String {
        String::from_utf8_lossy(&self.kind).into_owned()
    }
}

/// Reads chunks from an in-memory PNG stream.
#[derive(Debug)]
pub struct ChunkReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> ChunkReader<'a> {
    /// Start after the signature, which is verified here.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the signature is absent or wrong.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let head = data
            .get(..8)
            .ok_or_else(|| PixelsError::malformed("png", "shorter than the 8-byte signature"))?;
        if head != SIGNATURE {
            return Err(PixelsError::malformed("png", "signature does not match"));
        }
        Ok(Self { data, position: 8 })
    }

    /// Whether every byte has been consumed.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Read the next chunk, verifying its CRC.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a truncated chunk, a length
    /// beyond the spec's limit, or a CRC mismatch.
    pub fn next_chunk(&mut self) -> Result<Chunk> {
        let length = read_u32(self.data, self.position)? as usize;
        // §5.3: lengths must not exceed 2^31-1. Rejecting here means the
        // allocation below is always sane.
        if length > 0x7FFF_FFFF {
            return Err(PixelsError::malformed(
                "png",
                format!("chunk length {length} exceeds the 2^31-1 maximum"),
            ));
        }
        let kind_at = self.position + 4;
        let data_at = kind_at + 4;
        let crc_at = data_at + length;

        let kind_slice = self
            .data
            .get(kind_at..kind_at + 4)
            .ok_or_else(|| PixelsError::malformed("png", "truncated chunk type"))?;
        let mut kind = [0_u8; 4];
        kind.copy_from_slice(kind_slice);

        let payload = self.data.get(data_at..crc_at).ok_or_else(|| {
            PixelsError::malformed(
                "png",
                format!(
                    "chunk `{}` declares {length} bytes but the stream is shorter",
                    String::from_utf8_lossy(&kind)
                ),
            )
        })?;
        let expected = read_u32(self.data, crc_at)?;

        let mut crc = Crc32::new();
        crc.update(&kind);
        crc.update(payload);
        let actual = crc.finish();
        if actual != expected {
            return Err(PixelsError::malformed(
                "png",
                format!(
                    "chunk `{}` CRC mismatch: declares {expected:#010x}, data is {actual:#010x}",
                    String::from_utf8_lossy(&kind)
                ),
            ));
        }

        self.position = crc_at + 4;
        Ok(Chunk {
            kind,
            data: payload.to_vec(),
        })
    }
}

/// The largest ancillary chunk read whole before being discarded.
///
/// Ancillary chunks are skipped, but a streaming reader still has to walk past
/// them. Reading in bounded pieces means a chunk declaring 2 GiB of text costs
/// time rather than memory.
const SKIP_BUFFER: usize = 32 * 1024;

/// Reads chunks from a forward-only [`Source`], one piece at a time.
///
/// The in-memory [`ChunkReader`] needs the whole file; this one needs only the
/// chunk it is currently walking through, which is what lets a PNG decode in
/// constant memory (SPEC §Guarantees 1). `IDAT` payloads are handed out in
/// pieces rather than collected, so a 2 GiB image never exists as bytes.
#[derive(Debug)]
pub struct ChunkStream<S: Source> {
    source: S,
    /// Payload bytes of the current chunk still unread.
    remaining: usize,
    /// Running CRC over the type and payload seen so far.
    crc: Crc32,
    kind: [u8; 4],
    open: bool,
}

impl<S: Source> ChunkStream<S> {
    /// Start reading chunks from `source`, which must be positioned just past
    /// the signature.
    #[must_use]
    pub const fn new(source: S) -> Self {
        Self {
            source,
            remaining: 0,
            crc: Crc32::new(),
            kind: [0; 4],
            open: false,
        }
    }

    /// Whether the open chunk's payload has been fully read.
    #[must_use]
    pub const fn payload_done(&self) -> bool {
        self.remaining == 0
    }

    /// Whether the open chunk may be skipped if unrecognised (§5.4).
    #[must_use]
    pub const fn is_ancillary(&self) -> bool {
        self.kind[0] & 0x20 != 0
    }

    /// The open chunk's type as a display string, for diagnostics.
    #[must_use]
    pub fn name(&self) -> String {
        String::from_utf8_lossy(&self.kind).into_owned()
    }

    /// Open the next chunk, returning its type.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a truncated header or a length
    /// beyond the specification's 2^31-1 limit, and [`PixelsError::Io`] on
    /// source failure.
    pub fn open_next(&mut self) -> Result<[u8; 4]> {
        let mut head = [0_u8; 8];
        self.source.read_exact(&mut head)?;
        let length = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
        // §5.3: lengths must not exceed 2^31-1.
        if length > 0x7FFF_FFFF {
            return Err(PixelsError::malformed(
                "png",
                format!("chunk length {length} exceeds the 2^31-1 maximum"),
            ));
        }
        let mut kind = [0_u8; 4];
        kind.copy_from_slice(head.get(4..8).unwrap_or(&[0; 4]));

        self.kind = kind;
        self.remaining = length;
        self.crc = Crc32::new();
        self.crc.update(&kind);
        self.open = true;
        Ok(kind)
    }

    /// Read up to `buf.len()` payload bytes of the open chunk.
    ///
    /// Returns zero when the payload is exhausted, at which point
    /// [`ChunkStream::close`] verifies the CRC.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] on source failure, or
    /// [`PixelsError::Malformed`] if the stream ends inside the payload.
    pub fn read_payload(&mut self, buf: &mut [u8]) -> Result<usize> {
        let want = self.remaining.min(buf.len());
        if want == 0 {
            return Ok(0);
        }
        let Some(slot) = buf.get_mut(..want) else {
            return Ok(0);
        };
        self.source.read_exact(slot)?;
        self.crc.update(slot);
        self.remaining -= want;
        Ok(want)
    }

    /// Read the whole remaining payload. For small chunks only.
    ///
    /// `max` is the specification's own cap for the chunk type, so exceeding
    /// it is malformed input rather than a configurable limit.
    ///
    /// # Errors
    ///
    /// As [`ChunkStream::read_payload`], plus [`PixelsError::Malformed`] if
    /// the chunk is larger than `max`.
    pub fn read_payload_to_end(&mut self, max: usize) -> Result<Vec<u8>> {
        if self.remaining > max {
            return Err(PixelsError::malformed(
                "png",
                format!(
                    "chunk `{}` declares {} bytes, above the {max} the specification allows",
                    String::from_utf8_lossy(&self.kind),
                    self.remaining
                ),
            ));
        }
        let mut out = vec![0_u8; self.remaining];
        let mut filled = 0;
        while filled < out.len() {
            let Some(rest) = out.get_mut(filled..) else {
                break;
            };
            match self.read_payload(rest)? {
                0 => break,
                n => filled += n,
            }
        }
        Ok(out)
    }

    /// Discard the rest of the payload in bounded pieces.
    ///
    /// # Errors
    ///
    /// As [`ChunkStream::read_payload`].
    pub fn skip_payload(&mut self) -> Result<()> {
        let mut scratch = vec![0_u8; SKIP_BUFFER.min(self.remaining.max(1))];
        while self.remaining > 0 {
            if self.read_payload(&mut scratch)? == 0 {
                break;
            }
        }
        Ok(())
    }

    /// Finish the open chunk, verifying its CRC.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the payload was not fully read or
    /// the CRC does not match.
    pub fn close(&mut self) -> Result<()> {
        if self.remaining != 0 {
            return Err(PixelsError::malformed(
                "png",
                "chunk closed before its payload was read",
            ));
        }
        let mut trailer = [0_u8; 4];
        self.source.read_exact(&mut trailer)?;
        let expected = u32::from_be_bytes(trailer);
        let actual = self.crc.finish();
        self.open = false;
        if actual != expected {
            return Err(PixelsError::malformed(
                "png",
                format!(
                    "chunk `{}` CRC mismatch: declares {expected:#010x}, data is {actual:#010x}",
                    String::from_utf8_lossy(&self.kind)
                ),
            ));
        }
        Ok(())
    }
}

/// Write one chunk, with its length, type and CRC.
pub fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(payload);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// A PNG scanline filter (§9.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// No transformation.
    None,
    /// Difference from the pixel to the left.
    Sub,
    /// Difference from the pixel above.
    Up,
    /// Difference from the mean of left and above.
    Average,
    /// Difference from the Paeth predictor of left, above and above-left.
    Paeth,
}

impl Filter {
    /// The filter for a leading scanline byte.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a byte above 4.
    pub fn from_byte(byte: u8) -> Result<Self> {
        match byte {
            0 => Ok(Self::None),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Up),
            3 => Ok(Self::Average),
            4 => Ok(Self::Paeth),
            other => Err(PixelsError::malformed(
                "png",
                format!("filter type {other} is not one of 0..=4"),
            )),
        }
    }

    /// The byte for this filter.
    #[must_use]
    pub const fn to_byte(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Sub => 1,
            Self::Up => 2,
            Self::Average => 3,
            Self::Paeth => 4,
        }
    }
}

/// The Paeth predictor (§9.4).
///
/// Chooses whichever of left, above and above-left is closest to `a + b - c`.
const fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i16 + b as i16 - c as i16;
    let pa = (p - a as i16).abs();
    let pb = (p - b as i16).abs();
    let pc = (p - c as i16).abs();
    // Ties resolve toward `a`, then `b`; the order is normative.
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reverse a filter, writing the reconstructed line into `current`.
///
/// `previous` is the already-reconstructed line above, or zeroes for the first
/// line. `stride` is [`Header::filter_stride`].
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] if the lines disagree in length.
pub fn unfilter(filter: Filter, current: &mut [u8], previous: &[u8], stride: usize) -> Result<()> {
    if previous.len() != current.len() {
        return Err(PixelsError::malformed(
            "png",
            "scanlines differ in length while unfiltering",
        ));
    }
    for index in 0..current.len() {
        let raw = current.get(index).copied().unwrap_or(0);
        // Bytes before the first pixel are treated as zero (§9.2).
        let left = index
            .checked_sub(stride)
            .and_then(|i| current.get(i).copied())
            .unwrap_or(0);
        let above = previous.get(index).copied().unwrap_or(0);
        let above_left = index
            .checked_sub(stride)
            .and_then(|i| previous.get(i).copied())
            .unwrap_or(0);
        let value = match filter {
            Filter::None => raw,
            Filter::Sub => raw.wrapping_add(left),
            Filter::Up => raw.wrapping_add(above),
            Filter::Average => {
                // The average is computed in 9 bits then truncated.
                let mean = ((u16::from(left) + u16::from(above)) / 2) as u8;
                raw.wrapping_add(mean)
            }
            Filter::Paeth => raw.wrapping_add(paeth(left, above, above_left)),
        };
        if let Some(slot) = current.get_mut(index) {
            *slot = value;
        }
    }
    Ok(())
}

/// Apply a filter, writing the filtered bytes into `out`.
pub fn apply_filter(
    filter: Filter,
    current: &[u8],
    previous: &[u8],
    stride: usize,
    out: &mut Vec<u8>,
) {
    for index in 0..current.len() {
        let raw = current.get(index).copied().unwrap_or(0);
        let left = index
            .checked_sub(stride)
            .and_then(|i| current.get(i).copied())
            .unwrap_or(0);
        let above = previous.get(index).copied().unwrap_or(0);
        let above_left = index
            .checked_sub(stride)
            .and_then(|i| previous.get(i).copied())
            .unwrap_or(0);
        let value = match filter {
            Filter::None => raw,
            Filter::Sub => raw.wrapping_sub(left),
            Filter::Up => raw.wrapping_sub(above),
            Filter::Average => {
                let mean = ((u16::from(left) + u16::from(above)) / 2) as u8;
                raw.wrapping_sub(mean)
            }
            Filter::Paeth => raw.wrapping_sub(paeth(left, above, above_left)),
        };
        out.push(value);
    }
}

/// Column start and step, then row start and step, for each Adam7 pass (§8.1).
const ADAM7: [(u32, u32, u32, u32); 7] = [
    (0, 8, 0, 8),
    (4, 8, 0, 8),
    (0, 4, 4, 8),
    (2, 4, 0, 4),
    (0, 2, 2, 4),
    (1, 2, 0, 2),
    (0, 1, 1, 2),
];

/// The pixel dimensions of one Adam7 pass over a `width` x `height` image.
///
/// Passes can be empty for small images, which is the classic source of
/// interlace bugs — hence PngSuite's 1x1 through 9x9 cases.
#[must_use]
pub fn adam7_pass_size(pass: usize, width: u32, height: u32) -> (u32, u32) {
    let Some(&(x0, dx, y0, dy)) = ADAM7.get(pass) else {
        return (0, 0);
    };
    let pass_width = if width > x0 {
        (width - x0).div_ceil(dx)
    } else {
        0
    };
    let pass_height = if height > y0 {
        (height - y0).div_ceil(dy)
    } else {
        0
    };
    (pass_width, pass_height)
}

/// Where pixel `(x, y)` of `pass` lands in the full image.
#[must_use]
pub fn adam7_position(pass: usize, x: u32, y: u32) -> (u32, u32) {
    let Some(&(x0, dx, y0, dy)) = ADAM7.get(pass) else {
        return (0, 0);
    };
    (x0 + x * dx, y0 + y * dy)
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

    fn ihdr(width: u32, height: u32, depth: u8, color: u8, interlace: u8) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&width.to_be_bytes());
        payload.extend_from_slice(&height.to_be_bytes());
        payload.extend_from_slice(&[depth, color, 0, 0, interlace]);
        payload
    }

    #[test]
    fn a_valid_header_parses() {
        let header = Header::parse(&ihdr(32, 16, 8, 2, 0), &Limits::default()).unwrap();
        assert_eq!((header.width, header.height), (32, 16));
        assert_eq!(header.bit_depth, 8);
        assert_eq!(header.color_type, ColorType::Rgb);
        assert!(!header.interlaced);
        assert_eq!(header.bits_per_pixel(), 24);
        assert_eq!(header.row_bytes(32), 96);
        assert_eq!(header.filter_stride(), 3);
    }

    #[test]
    fn invalid_headers_are_rejected() {
        let limits = Limits::default();
        // Wrong length.
        assert!(Header::parse(&[0; 12], &limits).is_err());
        assert!(Header::parse(&[0; 14], &limits).is_err());
        // Zero dimensions.
        assert!(Header::parse(&ihdr(0, 8, 8, 0, 0), &limits).is_err());
        assert!(Header::parse(&ihdr(8, 0, 8, 0, 0), &limits).is_err());
        // Bad bit depth (PngSuite xd*).
        assert!(Header::parse(&ihdr(8, 8, 3, 0, 0), &limits).is_err());
        assert!(Header::parse(&ihdr(8, 8, 0, 0, 0), &limits).is_err());
        // Bad colour type (PngSuite xc*).
        assert!(Header::parse(&ihdr(8, 8, 8, 1, 0), &limits).is_err());
        assert!(Header::parse(&ihdr(8, 8, 8, 5, 0), &limits).is_err());
        // Depth/colour combinations the spec forbids.
        assert!(
            Header::parse(&ihdr(8, 8, 1, 2, 0), &limits).is_err(),
            "1-bit RGB"
        );
        assert!(
            Header::parse(&ihdr(8, 8, 16, 3, 0), &limits).is_err(),
            "16-bit palette"
        );
        // Unknown compression, filter or interlace method.
        let mut bad = ihdr(8, 8, 8, 0, 0);
        bad[10] = 1;
        assert!(Header::parse(&bad, &limits).is_err());
        let mut bad = ihdr(8, 8, 8, 0, 0);
        bad[11] = 1;
        assert!(Header::parse(&bad, &limits).is_err());
        assert!(Header::parse(&ihdr(8, 8, 8, 0, 2), &limits).is_err());
    }

    #[test]
    fn max_pixels_is_enforced_at_header_parse() {
        // A hostile header must be rejected before any buffer exists.
        let limits = Limits::default();
        let err = Header::parse(&ihdr(u32::MAX, u32::MAX, 8, 6, 0), &limits).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::LimitExceeded);
        // A tighter limit rejects a modest image too.
        let tight = Limits::default().with_max_pixels(100);
        assert!(Header::parse(&ihdr(20, 20, 8, 0, 0), &tight).is_err());
        assert!(Header::parse(&ihdr(10, 10, 8, 0, 0), &tight).is_ok());
    }

    #[test]
    fn every_legal_depth_and_colour_combination_is_accepted() {
        let limits = Limits::default();
        let combinations: [(u8, u8); 15] = [
            (1, 0),
            (2, 0),
            (4, 0),
            (8, 0),
            (16, 0),
            (8, 2),
            (16, 2),
            (1, 3),
            (2, 3),
            (4, 3),
            (8, 3),
            (8, 4),
            (16, 4),
            (8, 6),
            (16, 6),
        ];
        for (depth, color) in combinations {
            assert!(
                Header::parse(&ihdr(8, 8, depth, color, 0), &limits).is_ok(),
                "depth {depth} colour {color} should be legal"
            );
        }
    }

    #[test]
    fn row_bytes_round_up_for_sub_byte_depths() {
        let limits = Limits::default();
        let header = Header::parse(&ihdr(9, 1, 1, 0, 0), &limits).unwrap();
        assert_eq!(header.row_bytes(9), 2, "9 one-bit pixels need 2 bytes");
        assert_eq!(header.filter_stride(), 1, "sub-byte depths filter per byte");

        let header = Header::parse(&ihdr(3, 1, 4, 0, 0), &limits).unwrap();
        assert_eq!(header.row_bytes(3), 2, "3 four-bit pixels need 2 bytes");

        let header = Header::parse(&ihdr(1, 1, 16, 6, 0), &limits).unwrap();
        assert_eq!(header.row_bytes(1), 8, "one 16-bit RGBA pixel is 8 bytes");
        assert_eq!(header.filter_stride(), 8);
    }

    #[test]
    fn output_formats_follow_colour_type_and_transparency() {
        let limits = Limits::default();
        let case = |depth, color, trns| {
            Header::parse(&ihdr(4, 4, depth, color, 0), &limits)
                .unwrap()
                .output_format(trns)
        };
        assert_eq!(case(8, 0, false), PixelFormat::Gray8);
        assert_eq!(case(16, 0, false), PixelFormat::Gray16);
        assert_eq!(case(8, 0, true), PixelFormat::GrayA8, "tRNS adds alpha");
        assert_eq!(case(8, 2, false), PixelFormat::Rgb8);
        assert_eq!(case(16, 2, false), PixelFormat::Rgb16);
        assert_eq!(case(8, 2, true), PixelFormat::Rgba8);
        assert_eq!(
            case(8, 3, false),
            PixelFormat::Rgb8,
            "palette expands to RGB"
        );
        assert_eq!(case(8, 3, true), PixelFormat::Rgba8);
        assert_eq!(case(8, 4, false), PixelFormat::GrayA8);
        assert_eq!(case(8, 6, false), PixelFormat::Rgba8);
        assert_eq!(case(16, 6, false), PixelFormat::Rgba16);
        // v1 has no 16-bit grey+alpha format, so those widen rather than
        // silently losing the low byte.
        assert_eq!(case(16, 4, false), PixelFormat::Rgba16);
        assert_eq!(case(16, 0, true), PixelFormat::Rgba16);
        // Sub-byte greyscale expands to Gray8, since engine formats are
        // byte-aligned.
        assert_eq!(case(1, 0, false), PixelFormat::Gray8);
        assert_eq!(case(4, 0, false), PixelFormat::Gray8);
    }

    #[test]
    fn the_paeth_predictor_matches_the_specification() {
        // The worked examples from PNG §9.4, including the tie rules.
        assert_eq!(paeth(0, 0, 0), 0);
        assert_eq!(paeth(1, 2, 3), 1, "ties resolve toward the left pixel");
        assert_eq!(paeth(10, 20, 30), 10);
        assert_eq!(paeth(200, 100, 50), 200);
        // p = 50 + 100 - 200 = -50, so the left pixel is nearest.
        assert_eq!(paeth(50, 100, 200), 50);
        // a + b - c exactly equals one of them.
        assert_eq!(paeth(5, 7, 5), 7);
    }

    #[test]
    fn every_filter_round_trips() {
        let filters = [
            Filter::None,
            Filter::Sub,
            Filter::Up,
            Filter::Average,
            Filter::Paeth,
        ];
        let previous: Vec<u8> = (0..32).map(|i| (i * 7 % 251) as u8).collect();
        let original: Vec<u8> = (0..32).map(|i| (i * 13 + 5) as u8).collect();
        for filter in filters {
            for stride in [1, 3, 4, 8] {
                let mut filtered = Vec::new();
                apply_filter(filter, &original, &previous, stride, &mut filtered);
                let mut restored = filtered.clone();
                unfilter(filter, &mut restored, &previous, stride).unwrap();
                assert_eq!(restored, original, "{filter:?} at stride {stride}");
            }
        }
    }

    #[test]
    fn the_first_line_filters_against_zeroes() {
        // Bytes above the first line are zero, and bytes left of the first
        // pixel are zero (§9.2).
        let zeros = vec![0_u8; 8];
        let original: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        for filter in [Filter::Sub, Filter::Up, Filter::Average, Filter::Paeth] {
            let mut filtered = Vec::new();
            apply_filter(filter, &original, &zeros, 3, &mut filtered);
            let mut restored = filtered;
            unfilter(filter, &mut restored, &zeros, 3).unwrap();
            assert_eq!(restored, original, "{filter:?}");
        }
    }

    #[test]
    fn filter_bytes_are_validated() {
        assert_eq!(Filter::from_byte(0).unwrap(), Filter::None);
        assert_eq!(Filter::from_byte(4).unwrap(), Filter::Paeth);
        let err = Filter::from_byte(5).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
        assert!(Filter::from_byte(255).is_err());
        for byte in 0..=4_u8 {
            assert_eq!(Filter::from_byte(byte).unwrap().to_byte(), byte);
        }
    }

    #[test]
    fn unfiltering_mismatched_lines_is_an_error() {
        let mut current = vec![0_u8; 8];
        let err = unfilter(Filter::Up, &mut current, &[0; 4], 1).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
    }

    #[test]
    fn adam7_passes_partition_the_image() {
        // Every pixel must belong to exactly one pass. Small images are where
        // this goes wrong, so they are swept exhaustively.
        for width in 1..=17_u32 {
            for height in 1..=17_u32 {
                let mut seen = vec![0_u32; (width * height) as usize];
                for pass in 0..7 {
                    let (pass_width, pass_height) = adam7_pass_size(pass, width, height);
                    for y in 0..pass_height {
                        for x in 0..pass_width {
                            let (image_x, image_y) = adam7_position(pass, x, y);
                            assert!(image_x < width, "{width}x{height} pass {pass}");
                            assert!(image_y < height, "{width}x{height} pass {pass}");
                            seen[(image_y * width + image_x) as usize] += 1;
                        }
                    }
                }
                assert!(
                    seen.iter().all(|&count| count == 1),
                    "{width}x{height} is not partitioned exactly once per pixel"
                );
            }
        }
    }

    #[test]
    fn small_images_have_empty_adam7_passes() {
        // A 1x1 image lives entirely in pass 0; the rest are empty. Forgetting
        // this is the classic interlace crash.
        //
        // "Empty" means zero in *either* dimension, not (0, 0): pass 1 of a
        // 1x1 is (0, 1), because the single row is in range but no column is.
        assert_eq!(adam7_pass_size(0, 1, 1), (1, 1));
        for pass in 1..7 {
            let (width, height) = adam7_pass_size(pass, 1, 1);
            assert!(
                width == 0 || height == 0,
                "pass {pass} of a 1x1 should be empty, got {width}x{height}"
            );
        }
        // A 5x5 image, by contrast, has every pass non-empty — the passes are
        // uneven, not absent. What must hold at every size is that they cover
        // the image exactly once.
        let sizes: Vec<(u32, u32)> = (0..7).map(|p| adam7_pass_size(p, 5, 5)).collect();
        assert_eq!(sizes[0], (1, 1));
        assert!(
            sizes.iter().all(|&(w, h)| w > 0 && h > 0),
            "5x5 passes: {sizes:?}"
        );
        let covered: u32 = sizes.iter().map(|&(w, h)| w * h).sum();
        assert_eq!(covered, 25, "passes must cover every pixel of a 5x5");

        // Sizes where passes really are empty.
        let covered: u32 = (0..7)
            .map(|p| adam7_pass_size(p, 3, 2))
            .map(|(w, h)| w * h)
            .sum();
        assert_eq!(covered, 6, "passes must cover every pixel of a 3x2");
    }

    #[test]
    fn chunks_round_trip_with_their_crc() {
        let mut out = Vec::new();
        out.extend_from_slice(&SIGNATURE);
        write_chunk(&mut out, b"IHDR", &ihdr(4, 4, 8, 0, 0));
        write_chunk(&mut out, b"IEND", &[]);

        let mut reader = ChunkReader::new(&out).unwrap();
        let first = reader.next_chunk().unwrap();
        assert!(first.is(b"IHDR"));
        assert_eq!(first.data.len(), 13);
        assert!(!first.is_ancillary(), "IHDR is critical");
        let second = reader.next_chunk().unwrap();
        assert!(second.is(b"IEND"));
        assert!(reader.is_finished());
    }

    #[test]
    fn a_corrupted_crc_is_detected() {
        // PngSuite xcrn/xcsn are exactly this case.
        let mut out = Vec::new();
        out.extend_from_slice(&SIGNATURE);
        write_chunk(&mut out, b"IHDR", &ihdr(4, 4, 8, 0, 0));
        let last = out.len() - 1;
        out[last] ^= 0xFF;
        let mut reader = ChunkReader::new(&out).unwrap();
        let err = reader.next_chunk().unwrap_err();
        assert!(err.to_string().contains("CRC"), "{err}");
    }

    #[test]
    fn a_bad_signature_is_rejected() {
        // PngSuite xs1n/xs2n/xs4n/xs7n.
        assert!(ChunkReader::new(&[]).is_err());
        assert!(
            ChunkReader::new(&[0x89, b'P', b'N', b'G']).is_err(),
            "truncated"
        );
        let mut wrong = SIGNATURE;
        wrong[0] = 0x88;
        assert!(ChunkReader::new(&wrong).is_err());
        // The CR/LF bytes catch mangled transfers.
        let mut mangled = SIGNATURE;
        mangled[4] = b'\n';
        assert!(ChunkReader::new(&mangled).is_err());
    }

    #[test]
    fn an_overlong_chunk_length_is_rejected_before_allocating() {
        // PngSuite xlfn: a length field larger than the file.
        let mut out = Vec::new();
        out.extend_from_slice(&SIGNATURE);
        out.extend_from_slice(&0xFFFF_FFFF_u32.to_be_bytes());
        out.extend_from_slice(b"IDAT");
        let mut reader = ChunkReader::new(&out).unwrap();
        let err = reader.next_chunk().unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
        assert!(err.to_string().contains("2^31-1"), "{err}");
    }

    #[test]
    fn ancillary_chunks_are_distinguishable_from_critical_ones() {
        let critical = Chunk {
            kind: *b"IHDR",
            data: Vec::new(),
        };
        let ancillary = Chunk {
            kind: *b"tEXt",
            data: Vec::new(),
        };
        assert!(!critical.is_ancillary());
        assert!(ancillary.is_ancillary());
        assert_eq!(ancillary.name(), "tEXt");
    }

    #[test]
    fn the_filtered_size_accounts_for_interlace_overhead() {
        let limits = Limits::default();
        let plain = Header::parse(&ihdr(8, 8, 8, 0, 0), &limits).unwrap();
        assert_eq!(plain.filtered_size(), 8 * (1 + 8));
        // Interlaced images carry a filter byte per pass row, so they need
        // more space than the same image non-interlaced.
        let interlaced = Header::parse(&ihdr(8, 8, 8, 0, 1), &limits).unwrap();
        assert!(interlaced.filtered_size() > plain.filtered_size());
    }
}
