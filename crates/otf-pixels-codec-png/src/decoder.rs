//! The PNG decoder.
//!
//! # Laziness
//!
//! Construction reads only the signature and `IHDR` — 33 bytes — so the
//! descriptor is available without touching pixel data (SPEC §Guarantees 3).
//! The remaining chunks are read on the first [`Decoder::read_row`].
//!
//! # Memory
//!
//! A non-interlaced PNG **streams**: chunks are walked one piece at a time,
//! inflate runs incrementally, and each scanline is unfiltered and expanded as
//! it arrives. Peak memory is two scanlines plus the 32 KiB inflate window and
//! a read buffer — none of which grow with image height. That is what makes
//! SPEC §Guarantees 1 true for PNG rather than merely claimed.
//!
//! Interlaced PNG is **internally buffered**, as SPEC §Formats says: Adam7
//! scatters each pass across the whole image, so no row is final until every
//! pass has been read. That is ADR-0005's stated allowance for formats that
//! leave no choice.

use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Format, ImageDescriptor, Limits, PixelFormat, PixelsError,
    Result, Source,
};

use crate::format::{
    ChunkReader, ChunkStream, ColorType, Filter, Header, SIGNATURE, adam7_pass_size,
    adam7_position, unfilter,
};
use crate::inflate::{ZlibStream, zlib_decompress};

/// The specification's cap on a `PLTE` chunk: 256 entries of three bytes.
const MAX_PLTE: usize = 256 * 3;
/// The largest `tRNS` chunk any colour type permits: 256 palette alphas.
const MAX_TRNS: usize = 256;
/// How much compressed data is pulled from the source per refill.
const READ_CHUNK: usize = 64 * 1024;

/// Transparency from a `tRNS` chunk (§11.3.2.1).
#[derive(Debug, Clone)]
enum Transparency {
    /// One transparent grey level, in the image's bit depth.
    Gray(u16),
    /// One transparent RGB triple, in the image's bit depth.
    Rgb(u16, u16, u16),
    /// Per-palette-entry alpha; entries beyond the list are opaque.
    Palette(Vec<u8>),
}

/// Decodes a PNG stream.
#[derive(Debug)]
pub struct PngDecoder<S: Source> {
    header: Header,
    descriptor: ImageDescriptor,
    limits: Limits,
    source: Option<S>,
    /// Bytes already read while parsing the header, kept for the full parse.
    prefix: Vec<u8>,
    /// The decoded image in output format, produced on first row read.
    ///
    /// Only used by the interlaced path; a non-interlaced image never
    /// materializes here.
    raster: Option<Vec<u8>>,
    /// Per-row state for the non-interlaced streaming path.
    stream: Option<Box<Streaming<S>>>,
    row: u32,
    /// Set when the `tRNS` chunk was seen, which changes the output format.
    has_transparency: bool,
}

/// Everything the streaming path carries between rows.
///
/// This is the whole memory cost of decoding a non-interlaced PNG: two
/// scanlines, the inflate window, and one read buffer — none of which grow
/// with image height.
#[derive(Debug)]
struct Streaming<S: Source> {
    chunks: ChunkStream<S>,
    zlib: ZlibStream,
    /// Decompressed but not yet consumed filtered bytes.
    filtered: Vec<u8>,
    /// Read cursor into `filtered`.
    at: usize,
    /// The previous reconstructed scanline, which filters predict from.
    previous: Vec<u8>,
    palette: Option<Vec<[u8; 3]>>,
    transparency: Option<Transparency>,
    /// Set once `IEND` is reached or the final `IDAT` is consumed.
    input_done: bool,
}

impl<S: Source> PngDecoder<S> {
    /// Parse the signature and `IHDR`, reading nothing further.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a bad signature or header, or
    /// [`PixelsError::LimitExceeded`] if the dimensions exceed `limits`.
    pub fn new(mut source: S, limits: Limits) -> Result<Self> {
        // Signature plus a complete IHDR chunk: 8 + 4 + 4 + 13 + 4.
        let mut prefix = vec![0_u8; 33];
        source.read_exact(&mut prefix)?;

        let mut reader = ChunkReader::new(&prefix)?;
        let chunk = reader.next_chunk()?;
        if !chunk.is(b"IHDR") {
            return Err(PixelsError::malformed(
                "png",
                format!("first chunk must be IHDR, got `{}`", chunk.name()),
            ));
        }
        let header = Header::parse(&chunk.data, &limits)?;

        // `tRNS` appears later in the stream but changes the output format, so
        // the descriptor cannot be final until the whole stream is read. It is
        // resolved here optimistically and corrected during the full parse;
        // callers see the corrected one because `descriptor()` reads the field.
        let descriptor = header.descriptor(false, &limits)?;
        Ok(Self {
            header,
            descriptor,
            limits,
            source: Some(source),
            prefix,
            raster: None,
            stream: None,
            row: 0,
            has_transparency: false,
        })
    }

    /// The parsed header.
    #[must_use]
    pub const fn header(&self) -> Header {
        self.header
    }

    /// Walk chunks up to the first `IDAT`, resolving the final format.
    ///
    /// `PLTE` and `tRNS` both precede `IDAT` (§5.6), which is what makes
    /// streaming possible at all: everything needed to expand a pixel is
    /// known before the first one arrives.
    fn begin_streaming(&mut self) -> Result<Streaming<S>> {
        let Some(source) = self.source.take() else {
            return Err(PixelsError::graph("png source was already consumed"));
        };
        let mut chunks = ChunkStream::new(source);
        let mut palette: Option<Vec<[u8; 3]>> = None;
        let mut transparency: Option<Transparency> = None;

        loop {
            let kind = chunks.open_next()?;
            match &kind {
                b"IHDR" => {
                    return Err(PixelsError::malformed("png", "more than one IHDR"));
                }
                b"PLTE" => {
                    let data = chunks.read_payload_to_end(MAX_PLTE)?;
                    palette = Some(parse_plte(&data)?);
                    chunks.close()?;
                }
                b"tRNS" => {
                    let data = chunks.read_payload_to_end(MAX_TRNS)?;
                    transparency = Some(parse_trns(&data, self.header.color_type)?);
                    chunks.close()?;
                }
                b"IDAT" => break,
                b"IEND" => {
                    return Err(PixelsError::malformed("png", "no IDAT data"));
                }
                _ => {
                    if !chunks.is_ancillary() {
                        return Err(PixelsError::malformed(
                            "png",
                            format!("unknown critical chunk `{}`", chunks.name()),
                        ));
                    }
                    chunks.skip_payload()?;
                    chunks.close()?;
                }
            }
        }

        if self.header.color_type == ColorType::Palette && palette.is_none() {
            return Err(PixelsError::malformed(
                "png",
                "palette image has no PLTE chunk",
            ));
        }
        self.has_transparency = transparency.is_some();
        self.descriptor = self
            .header
            .descriptor(self.has_transparency, &self.limits)?;

        let row_bytes = self.header.row_bytes(self.header.width);
        Ok(Streaming {
            chunks,
            // The limit is the exact filtered size the header implies, which
            // is what makes a decompression bomb a malformed-input error.
            zlib: ZlibStream::new(self.header.filtered_size()),
            filtered: Vec::new(),
            at: 0,
            previous: vec![0_u8; row_bytes],
            palette,
            transparency,
            input_done: false,
        })
    }

    /// Read every remaining chunk and produce the output-format raster.
    fn decode_image(&mut self) -> Result<Vec<u8>> {
        let mut all = std::mem::take(&mut self.prefix);
        if let Some(mut source) = self.source.take() {
            // A forward-only source is drained once; PNG needs the whole
            // stream because IDAT may be split and tRNS may follow it.
            let mut buffer = [0_u8; 64 * 1024];
            loop {
                let read = source.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                let Some(chunk) = buffer.get(..read) else {
                    break;
                };
                all.extend_from_slice(chunk);
            }
        }

        let mut reader = ChunkReader::new(&all)?;
        let mut compressed: Vec<u8> = Vec::new();
        let mut palette: Option<Vec<[u8; 3]>> = None;
        let mut transparency: Option<Transparency> = None;
        let mut seen_ihdr = false;
        let mut seen_iend = false;

        while !reader.is_finished() {
            let chunk = reader.next_chunk()?;
            match &chunk.kind {
                b"IHDR" => {
                    if seen_ihdr {
                        return Err(PixelsError::malformed("png", "more than one IHDR"));
                    }
                    seen_ihdr = true;
                }
                b"PLTE" => {
                    palette = Some(parse_plte(&chunk.data)?);
                }
                b"tRNS" => {
                    transparency = Some(parse_trns(&chunk.data, self.header.color_type)?);
                }
                b"IDAT" => compressed.extend_from_slice(&chunk.data),
                b"IEND" => {
                    seen_iend = true;
                    break;
                }
                _ => {
                    // Unknown critical chunks mean the image cannot be
                    // rendered correctly; ancillary ones are skipped (§5.4).
                    if !chunk.is_ancillary() {
                        return Err(PixelsError::malformed(
                            "png",
                            format!("unknown critical chunk `{}`", chunk.name()),
                        ));
                    }
                }
            }
        }

        if !seen_iend {
            return Err(PixelsError::malformed(
                "png",
                "stream ends without an IEND chunk",
            ));
        }
        if compressed.is_empty() {
            return Err(PixelsError::malformed("png", "no IDAT data"));
        }
        if self.header.color_type == ColorType::Palette && palette.is_none() {
            return Err(PixelsError::malformed(
                "png",
                "palette image has no PLTE chunk",
            ));
        }

        self.has_transparency = transparency.is_some();
        self.descriptor = self
            .header
            .descriptor(self.has_transparency, &self.limits)?;

        // The limit is the exact filtered size the header implies, which is
        // what makes a decompression bomb a malformed-input error.
        let filtered = zlib_decompress(&compressed, self.header.filtered_size())?;
        let samples = self.unfilter_all(&filtered)?;
        self.expand(&samples, palette.as_deref(), transparency.as_ref())
    }

    /// Reverse filtering, producing unfiltered sample rows in PNG layout.
    ///
    /// For interlaced images the passes are deinterlaced into a single raster
    /// of `height` rows here, so everything downstream sees one image.
    fn unfilter_all(&self, filtered: &[u8]) -> Result<Vec<u8>> {
        let stride = self.header.filter_stride();
        let full_row = self.header.row_bytes(self.header.width);

        if !self.header.interlaced {
            let mut out = vec![0_u8; self.header.height as usize * full_row];
            let mut previous = vec![0_u8; full_row];
            let mut at = 0;
            for y in 0..self.header.height as usize {
                let filter_byte = filtered
                    .get(at)
                    .copied()
                    .ok_or_else(|| PixelsError::malformed("png", "raster ends early"))?;
                let filter = Filter::from_byte(filter_byte)?;
                at += 1;
                let row = filtered
                    .get(at..at + full_row)
                    .ok_or_else(|| PixelsError::malformed("png", "scanline ends early"))?;
                at += full_row;

                let mut current = row.to_vec();
                unfilter(filter, &mut current, &previous, stride)?;
                let start = y * full_row;
                if let Some(slot) = out.get_mut(start..start + full_row) {
                    slot.copy_from_slice(&current);
                }
                previous = current;
            }
            return Ok(out);
        }

        // Adam7: each pass is an independent filtered raster, then its pixels
        // are scattered into their positions in the full image.
        let mut out = vec![0_u8; self.header.height as usize * full_row];
        let mut at = 0;
        for pass in 0..7 {
            let (pass_width, pass_height) =
                adam7_pass_size(pass, self.header.width, self.header.height);
            if pass_width == 0 || pass_height == 0 {
                continue;
            }
            let pass_row = self.header.row_bytes(pass_width);
            let mut previous = vec![0_u8; pass_row];
            for y in 0..pass_height {
                let filter_byte = filtered.get(at).copied().ok_or_else(|| {
                    PixelsError::malformed("png", format!("pass {pass} ends early"))
                })?;
                let filter = Filter::from_byte(filter_byte)?;
                at += 1;
                let row = filtered.get(at..at + pass_row).ok_or_else(|| {
                    PixelsError::malformed("png", format!("pass {pass} scanline ends early"))
                })?;
                at += pass_row;

                let mut current = row.to_vec();
                unfilter(filter, &mut current, &previous, stride)?;
                for x in 0..pass_width {
                    let (image_x, image_y) = adam7_position(pass, x, y);
                    copy_pixel_bits(
                        &current,
                        x as usize,
                        &mut out,
                        image_y as usize * full_row,
                        image_x as usize,
                        self.header.bits_per_pixel(),
                    );
                }
                previous = current;
            }
        }
        Ok(out)
    }

    /// Convert unfiltered PNG samples into the engine's output format.
    fn expand(
        &self,
        samples: &[u8],
        palette: Option<&[[u8; 3]]>,
        transparency: Option<&Transparency>,
    ) -> Result<Vec<u8>> {
        let height = self.header.height as usize;
        let row_bytes = self.header.row_bytes(self.header.width);
        let out_row = self.descriptor.row_bytes();
        let mut out = vec![0_u8; height * out_row];

        for y in 0..height {
            let row = samples
                .get(y * row_bytes..(y + 1) * row_bytes)
                .ok_or_else(|| PixelsError::malformed("png", "sample row missing"))?;
            let Some(slot) = out.get_mut(y * out_row..(y + 1) * out_row) else {
                return Err(PixelsError::graph("output row is missing"));
            };
            self.expand_row(row, palette, transparency, slot)?;
        }
        Ok(out)
    }

    /// Convert one unfiltered PNG scanline into the engine's output format.
    ///
    /// Shared by both paths: the streaming one calls it per scanline as it
    /// arrives, the interlaced one per row of the deinterlaced raster. Having
    /// one implementation is what makes "interlaced and non-interlaced decode
    /// identically" a structural fact rather than a coincidence.
    fn expand_row(
        &self,
        samples: &[u8],
        palette: Option<&[[u8; 3]]>,
        transparency: Option<&Transparency>,
        out: &mut [u8],
    ) -> Result<()> {
        let format = self.descriptor.pixel;
        let width = self.header.width as usize;
        let depth = self.header.bit_depth;
        let channels = self.header.color_type.channels();
        let max = ((1_u32 << depth) - 1) as u16;

        let mut at = 0;
        for x in 0..width {
            let mut channel = [0_u16; 4];
            for (c, slot) in channel.iter_mut().take(channels).enumerate() {
                *slot = read_sample(samples, x * channels + c, depth);
            }
            write_pixel(
                out,
                &mut at,
                format,
                self.header.color_type,
                &channel,
                depth,
                max,
                palette,
                transparency,
            )?;
        }
        Ok(())
    }

    /// The buffered path, used for interlaced images.
    fn read_row_buffered(&mut self, out: &mut [u8]) -> Result<()> {
        if self.raster.is_none() {
            self.raster = Some(self.decode_image()?);
        }
        let Some(raster) = self.raster.as_ref() else {
            return Err(PixelsError::graph("raster vanished after decoding"));
        };
        if self.row >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("all {} rows have already been read", self.descriptor.height),
            ));
        }
        let row_bytes = self.descriptor.row_bytes();
        if out.len() != row_bytes {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {row_bytes}", out.len()),
            ));
        }
        let start = self.row as usize * row_bytes;
        let row = raster
            .get(start..start + row_bytes)
            .ok_or_else(|| PixelsError::malformed("png", "decoded raster is short"))?;
        out.copy_from_slice(row);
        self.row += 1;
        Ok(())
    }
}

impl<S: Source> Streaming<S> {
    /// Pull more compressed bytes and decompress them.
    ///
    /// Returns `false` once the stream is exhausted, so a caller asking for a
    /// scanline that never arrives gets a truncation error rather than a hang.
    fn refill(&mut self) -> Result<bool> {
        if self.input_done {
            return Ok(false);
        }
        let mut buffer = vec![0_u8; READ_CHUNK];

        // Walk forward until this feed produced some output, or the stream
        // ends. An IDAT boundary or a run of ancillary chunks can legitimately
        // yield nothing, so one pass is not enough.
        loop {
            if self.chunks.payload_done() {
                self.chunks.close()?;
                // Find the next IDAT, skipping whatever sits between them.
                loop {
                    let kind = self.chunks.open_next()?;
                    match &kind {
                        b"IDAT" => break,
                        b"IEND" => {
                            // Closed, not just recognised: IEND carries a CRC
                            // like any other chunk, and a stream truncated
                            // inside it is still a truncated stream.
                            self.chunks.skip_payload()?;
                            self.chunks.close()?;
                            self.input_done = true;
                            let tail = self.zlib.finish()?;
                            self.filtered.extend_from_slice(&tail);
                            return Ok(!tail.is_empty());
                        }
                        _ => {
                            if !self.chunks.is_ancillary() {
                                return Err(PixelsError::malformed(
                                    "png",
                                    format!("unknown critical chunk `{}`", self.chunks.name()),
                                ));
                            }
                            self.chunks.skip_payload()?;
                            self.chunks.close()?;
                        }
                    }
                }
            }

            let read = self.chunks.read_payload(&mut buffer)?;
            if read == 0 {
                continue;
            }
            let produced = self.zlib.push(buffer.get(..read).unwrap_or(&[]))?;
            if !produced.is_empty() {
                self.filtered.extend_from_slice(&produced);
                return Ok(true);
            }
        }
    }

    /// Consume whatever follows the last scanline, so the stream is verified.
    ///
    /// The zlib Adler-32 and the trailing `IEND` both sit *after* the final
    /// row. A caller that stops reading at the last row would otherwise never
    /// reach them, and a corrupt checksum would pass unnoticed — which is
    /// exactly what PngSuite's `xcsn0g01` checks.
    fn finalize(&mut self) -> Result<()> {
        while self.refill()? {}
        if !self.input_done {
            return Err(PixelsError::malformed(
                "png",
                "stream ends without an IEND chunk",
            ));
        }
        Ok(())
    }

    /// Reconstruct the next scanline, returning it in PNG sample layout.
    fn next_scanline(&mut self, row_bytes: usize, stride: usize) -> Result<Vec<u8>> {
        let want = row_bytes + 1;
        while self.filtered.len() - self.at < want {
            if !self.refill()? {
                return Err(PixelsError::malformed(
                    "png",
                    "image data ends before the last scanline",
                ));
            }
        }

        let filter_byte = self
            .filtered
            .get(self.at)
            .copied()
            .ok_or_else(|| PixelsError::malformed("png", "scanline ends early"))?;
        let filter = Filter::from_byte(filter_byte)?;
        let start = self.at + 1;
        let mut current = self
            .filtered
            .get(start..start + row_bytes)
            .ok_or_else(|| PixelsError::malformed("png", "scanline ends early"))?
            .to_vec();
        self.at = start + row_bytes;

        // Consumed bytes are dropped rather than accumulated, which is what
        // keeps this buffer at one scanline rather than one image.
        if self.at >= self.filtered.len() {
            self.filtered.clear();
            self.at = 0;
        } else if self.at > READ_CHUNK {
            self.filtered.drain(..self.at);
            self.at = 0;
        }

        unfilter(filter, &mut current, &self.previous, stride)?;
        self.previous.clear();
        self.previous.extend_from_slice(&current);
        Ok(current)
    }
}

/// Parse a `PLTE` payload into RGB triples.
fn parse_plte(data: &[u8]) -> Result<Vec<[u8; 3]>> {
    if data.len() % 3 != 0 || data.is_empty() {
        return Err(PixelsError::malformed(
            "png",
            format!("PLTE length {} is not a positive multiple of 3", data.len()),
        ));
    }
    if data.len() / 3 > 256 {
        return Err(PixelsError::malformed(
            "png",
            "PLTE has more than 256 entries",
        ));
    }
    Ok(data
        .chunks_exact(3)
        .map(|rgb| {
            [
                rgb.first().copied().unwrap_or(0),
                rgb.get(1).copied().unwrap_or(0),
                rgb.get(2).copied().unwrap_or(0),
            ]
        })
        .collect())
}

/// Copy one pixel's bits between rasters, handling sub-byte depths.
fn copy_pixel_bits(
    source_row: &[u8],
    source_x: usize,
    out: &mut [u8],
    row_start: usize,
    dest_x: usize,
    bits_per_pixel: usize,
) {
    if bits_per_pixel >= 8 {
        let bytes = bits_per_pixel / 8;
        for byte in 0..bytes {
            let value = source_row
                .get(source_x * bytes + byte)
                .copied()
                .unwrap_or(0);
            if let Some(slot) = out.get_mut(row_start + dest_x * bytes + byte) {
                *slot = value;
            }
        }
        return;
    }
    // Sub-byte: read the packed field and write it at the destination offset.
    let value = read_bits(source_row, source_x, bits_per_pixel);
    write_bits(out, row_start, dest_x, bits_per_pixel, value);
}

/// Read a packed sub-byte field, most-significant-first within each byte.
fn read_bits(row: &[u8], index: usize, bits: usize) -> u8 {
    let per_byte = 8 / bits;
    let byte = row.get(index / per_byte).copied().unwrap_or(0);
    let shift = 8 - bits * (index % per_byte + 1);
    (byte >> shift) & ((1 << bits) - 1) as u8
}

/// Write a packed sub-byte field.
fn write_bits(out: &mut [u8], row_start: usize, index: usize, bits: usize, value: u8) {
    let per_byte = 8 / bits;
    let offset = row_start + index / per_byte;
    let shift = 8 - bits * (index % per_byte + 1);
    let mask = ((1 << bits) - 1) as u8;
    if let Some(slot) = out.get_mut(offset) {
        *slot = (*slot & !(mask << shift)) | ((value & mask) << shift);
    }
}

/// Read sample `index` of a row at `depth` bits.
fn read_sample(row: &[u8], index: usize, depth: u8) -> u16 {
    match depth {
        16 => {
            // PNG samples are big-endian (§7.1).
            let high = row.get(index * 2).copied().unwrap_or(0);
            let low = row.get(index * 2 + 1).copied().unwrap_or(0);
            u16::from_be_bytes([high, low])
        }
        8 => u16::from(row.get(index).copied().unwrap_or(0)),
        bits => u16::from(read_bits(row, index, bits as usize)),
    }
}

/// Scale a sample from `max` to the full 8-bit range.
///
/// The spec's rule: the value is replicated, not shifted, so 1-bit 1 becomes
/// 255 rather than 128 (§13.13).
const fn scale8(value: u16, max: u16) -> u8 {
    if max == 0 {
        return 0;
    }
    ((value as u32 * 255 + max as u32 / 2) / max as u32) as u8
}

/// Write one output pixel, converting from PNG's layout.
#[allow(
    clippy::too_many_arguments,
    reason = "one pixel conversion needs all of it"
)]
fn write_pixel(
    out: &mut [u8],
    at: &mut usize,
    format: PixelFormat,
    color_type: ColorType,
    channel: &[u16; 4],
    depth: u8,
    max: u16,
    palette: Option<&[[u8; 3]]>,
    transparency: Option<&Transparency>,
) -> Result<()> {
    /// Append one byte.
    fn push(out: &mut [u8], at: &mut usize, value: u8) {
        if let Some(slot) = out.get_mut(*at) {
            *slot = value;
        }
        *at += 1;
    }
    /// Append one native-endian 16-bit sample.
    fn push16(out: &mut [u8], at: &mut usize, value: u16) {
        for byte in value.to_ne_bytes() {
            push(out, at, byte);
        }
    }

    match color_type {
        ColorType::Palette => {
            let index = channel[0] as usize;
            let entries = palette
                .ok_or_else(|| PixelsError::malformed("png", "palette image without a palette"))?;
            let rgb = entries.get(index).copied().ok_or_else(|| {
                PixelsError::malformed(
                    "png",
                    format!(
                        "palette index {index} is beyond the {}-entry palette",
                        entries.len()
                    ),
                )
            })?;
            push(out, at, rgb[0]);
            push(out, at, rgb[1]);
            push(out, at, rgb[2]);
            if format == PixelFormat::Rgba8 {
                let alpha = match transparency {
                    Some(Transparency::Palette(alphas)) => {
                        // Entries past the tRNS list are fully opaque.
                        alphas.get(index).copied().unwrap_or(255)
                    }
                    _ => 255,
                };
                push(out, at, alpha);
            }
        }
        ColorType::Grayscale => {
            let transparent =
                matches!(transparency, Some(Transparency::Gray(key)) if *key == channel[0]);
            match format {
                PixelFormat::Gray8 => push(out, at, scale8(channel[0], max)),
                PixelFormat::Gray16 => push16(out, at, channel[0]),
                PixelFormat::GrayA8 => {
                    push(out, at, scale8(channel[0], max));
                    push(out, at, if transparent { 0 } else { 255 });
                }
                PixelFormat::Rgba16 => {
                    let value = if depth == 16 {
                        channel[0]
                    } else {
                        channel[0] * 257
                    };
                    push16(out, at, value);
                    push16(out, at, value);
                    push16(out, at, value);
                    push16(out, at, if transparent { 0 } else { u16::MAX });
                }
                other => {
                    return Err(PixelsError::unsupported(format!(
                        "greyscale cannot be written as {other}"
                    )));
                }
            }
        }
        ColorType::GrayscaleAlpha => match format {
            PixelFormat::GrayA8 => {
                push(out, at, scale8(channel[0], max));
                push(out, at, scale8(channel[1], max));
            }
            PixelFormat::Rgba16 => {
                push16(out, at, channel[0]);
                push16(out, at, channel[0]);
                push16(out, at, channel[0]);
                push16(out, at, channel[1]);
            }
            other => {
                return Err(PixelsError::unsupported(format!(
                    "grey+alpha cannot be written as {other}"
                )));
            }
        },
        ColorType::Rgb => {
            let transparent = matches!(
                transparency,
                Some(Transparency::Rgb(r, g, b))
                    if *r == channel[0] && *g == channel[1] && *b == channel[2]
            );
            match format {
                PixelFormat::Rgb8 => {
                    for &value in channel.iter().take(3) {
                        push(out, at, scale8(value, max));
                    }
                }
                PixelFormat::Rgb16 => {
                    for &value in channel.iter().take(3) {
                        push16(out, at, value);
                    }
                }
                PixelFormat::Rgba8 => {
                    for &value in channel.iter().take(3) {
                        push(out, at, scale8(value, max));
                    }
                    push(out, at, if transparent { 0 } else { 255 });
                }
                PixelFormat::Rgba16 => {
                    for &value in channel.iter().take(3) {
                        push16(out, at, value);
                    }
                    push16(out, at, if transparent { 0 } else { u16::MAX });
                }
                other => {
                    return Err(PixelsError::unsupported(format!(
                        "RGB cannot be written as {other}"
                    )));
                }
            }
        }
        ColorType::Rgba => match format {
            PixelFormat::Rgba8 => {
                for &value in channel {
                    push(out, at, scale8(value, max));
                }
            }
            PixelFormat::Rgba16 => {
                for &value in channel {
                    push16(out, at, value);
                }
            }
            other => {
                return Err(PixelsError::unsupported(format!(
                    "RGBA cannot be written as {other}"
                )));
            }
        },
    }
    Ok(())
}

/// Parse a `tRNS` payload for `color_type`.
fn parse_trns(data: &[u8], color_type: ColorType) -> Result<Transparency> {
    /// Read a big-endian `u16` at `offset`.
    fn be16(data: &[u8], offset: usize) -> u16 {
        u16::from_be_bytes([
            data.get(offset).copied().unwrap_or(0),
            data.get(offset + 1).copied().unwrap_or(0),
        ])
    }
    match color_type {
        ColorType::Grayscale => {
            if data.len() != 2 {
                return Err(PixelsError::malformed(
                    "png",
                    format!("greyscale tRNS must be 2 bytes, got {}", data.len()),
                ));
            }
            Ok(Transparency::Gray(be16(data, 0)))
        }
        ColorType::Rgb => {
            if data.len() != 6 {
                return Err(PixelsError::malformed(
                    "png",
                    format!("RGB tRNS must be 6 bytes, got {}", data.len()),
                ));
            }
            Ok(Transparency::Rgb(
                be16(data, 0),
                be16(data, 2),
                be16(data, 4),
            ))
        }
        ColorType::Palette => {
            if data.len() > 256 {
                return Err(PixelsError::malformed(
                    "png",
                    format!(
                        "palette tRNS has {} entries, over the 256 maximum",
                        data.len()
                    ),
                ));
            }
            Ok(Transparency::Palette(data.to_vec()))
        }
        // §11.3.2.1: tRNS is forbidden where alpha is already present.
        ColorType::GrayscaleAlpha | ColorType::Rgba => Err(PixelsError::malformed(
            "png",
            "tRNS is not allowed for colour types that already carry alpha",
        )),
    }
}

impl<S: Source + std::fmt::Debug> Decoder for PngDecoder<S> {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        // The raster is fully materialized before the first row is served, so
        // any region could in principle be answered. Declaring `Sequential`
        // keeps the streaming contract of ADR-0005 and costs nothing, since
        // the scheduler pulls rows in order anyway.
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        // Interlaced images need every pass before any row is final, so they
        // take the buffered path; everything else streams.
        if self.header.interlaced {
            return self.read_row_buffered(out);
        }
        if self.stream.is_none() {
            let started = self.begin_streaming()?;
            self.stream = Some(Box::new(started));
        }
        if self.row >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("all {} rows have already been read", self.descriptor.height),
            ));
        }
        let row_bytes = self.descriptor.row_bytes();
        if out.len() != row_bytes {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {row_bytes}", out.len()),
            ));
        }

        let sample_bytes = self.header.row_bytes(self.header.width);
        let stride = self.header.filter_stride();
        let Some(stream) = self.stream.as_mut() else {
            return Err(PixelsError::graph("png stream vanished after starting"));
        };
        let samples = stream.next_scanline(sample_bytes, stride)?;
        let palette = stream.palette.clone();
        let transparency = stream.transparency.clone();

        self.expand_row(&samples, palette.as_deref(), transparency.as_ref(), out)?;
        self.row += 1;

        // The checksum and IEND follow the last scanline, so the stream is
        // only fully verified once it has been read to its end.
        if self.row == self.descriptor.height {
            if let Some(stream) = self.stream.as_mut() {
                stream.finalize()?;
            }
        }
        Ok(())
    }
}

/// Whether `prefix` starts with the PNG signature.
///
/// Detection is by magic bytes only (SPEC §Formats).
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    prefix.get(..8) == Some(&SIGNATURE[..])
}

/// The PNG entry in a sniffing registry.
///
/// Format detection is by magic bytes only (SPEC §Formats); PNG's eight-byte
/// signature is deliberately designed to survive — and detect — the transfer
/// corruptions its §5.2 enumerates, so there is nothing else worth consulting.
#[derive(Debug, Clone, Copy, Default)]
pub struct PngCodec;

impl Codec for PngCodec {
    fn format(&self) -> Format {
        Format::Png
    }

    fn magic_len(&self) -> usize {
        SIGNATURE.len()
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
    }
}
