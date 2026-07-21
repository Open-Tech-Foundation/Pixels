//! The baseline JPEG encoder.
//!
//! # Memory
//!
//! Encoding is streaming, at one MCU row — the same band structure the
//! decoder uses, run the other way. Rows arrive one at a time, accumulate
//! until a whole MCU row is available (8 or 16 of them), and are then
//! colour-converted, transformed, quantized, entropy-coded and written
//! straight to the sink. Nothing accumulates across bands, so peak memory is
//! a band and not the image.
//!
//! That is why the standard Huffman tables are used rather than tables
//! derived from the image's own statistics: optimal tables need a counting
//! pass over every coefficient before the first byte can be written, which
//! means buffering the whole image. The few percent it would save is not
//! worth trading ADR-0005's streaming contract for.
//!
//! # What is written
//!
//! Baseline sequential JPEG: `SOI`, `APP0` (JFIF), `DQT`, `SOF0`, `DHT`,
//! `SOS`, one interleaved scan, `EOI`. No restart markers — they cost bytes
//! and buy resynchronization after corruption, which matters for broadcast
//! and not for a file a pipeline just produced.

use crate::fdct;
use crate::format::{ZIGZAG, marker};
use crate::huffman::HuffmanEncoder;
use crate::tables;
use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

/// How much the chroma channels are subsampled relative to luma.
///
/// Chroma carries far less perceptible detail than luma, so discarding three
/// quarters of it is nearly free visually and saves a substantial fraction of
/// the bytes. It is not free for synthetic images with hard colour edges,
/// which is why it can be turned off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Subsampling {
    /// 4:4:4 — full-resolution chroma.
    None,
    /// 4:2:2 — chroma at half width.
    Horizontal,
    /// 4:2:0 — chroma at half width and half height.
    #[default]
    Both,
}

impl Subsampling {
    /// The luma sampling factors this implies. Chroma is always 1x1.
    const fn factors(self) -> (u32, u32) {
        match self {
            Self::None => (1, 1),
            Self::Horizontal => (2, 1),
            Self::Both => (2, 2),
        }
    }
}

/// Encodes a baseline JPEG stream.
#[derive(Debug)]
pub struct JpegEncoder {
    quality: u8,
    subsampling: Subsampling,
    /// Set by `write_header`; its presence means the header was written.
    state: Option<State>,
}

/// Everything fixed once the descriptor is known.
#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    /// Whether the image is coded as one luminance component or three.
    grayscale: bool,
    /// Luma sampling factors; chroma is always 1x1.
    factors: (u32, u32),
    luma_quant: [u16; 64],
    chroma_quant: [u16; 64],
    luma_dc: HuffmanEncoder,
    luma_ac: HuffmanEncoder,
    chroma_dc: HuffmanEncoder,
    chroma_ac: HuffmanEncoder,
    /// Input rows accumulated for the MCU row being built.
    band: Vec<u8>,
    /// How many rows of `band` hold real pixels.
    band_rows: u32,
    /// Pixel height of one MCU row.
    band_height: u32,
    mcus_per_line: u32,
    /// Padded component planes, reused across bands.
    luma: Plane,
    chroma_blue: Plane,
    chroma_red: Plane,
    /// Full-resolution Cb/Cr for the band, interleaved, before box filtering.
    chroma_full: Vec<u8>,
    /// Per-component DC predictor.
    predictors: [i32; 3],
    writer: BitWriter,
    rows_written: u32,
}

/// One component's padded plane for the MCU row being encoded.
#[derive(Debug, Default)]
struct Plane {
    stride: usize,
    height: usize,
    samples: Vec<u8>,
}

impl Plane {
    fn new(stride: usize, height: usize) -> Self {
        Self {
            stride,
            height,
            samples: vec![128; stride * height],
        }
    }

    /// Copy the 8x8 block whose top-left corner is at `(x, y)`.
    fn block(&self, x: usize, y: usize, out: &mut [u8; 64]) {
        for row in 0..8 {
            let start = (y + row) * self.stride + x;
            let source = self
                .samples
                .get(start..)
                .and_then(|rest| rest.get(..8))
                .unwrap_or(&[128; 8]);
            if let Some(target) = out.get_mut(row * 8..row * 8 + 8) {
                target.copy_from_slice(source);
            }
        }
    }
}

/// A chroma plane, or an empty one for a greyscale image that has no chroma.
fn chroma_plane(grayscale: bool, stride: usize, height: u32) -> Plane {
    if grayscale {
        Plane::default()
    } else {
        Plane::new(stride, height as usize)
    }
}

/// Accumulates entropy-coded bits, stuffing `0xFF` bytes as it goes.
#[derive(Debug, Default)]
struct BitWriter {
    /// Completed bytes, drained to the sink after each MCU row.
    bytes: Vec<u8>,
    accumulator: u32,
    bits: u32,
}

impl BitWriter {
    /// Append `length` low bits of `code`, most significant first.
    fn write(&mut self, code: u32, length: u32) {
        if length == 0 || length > 32 {
            return;
        }
        let mask = if length >= 32 {
            u32::MAX
        } else {
            (1_u32 << length) - 1
        };
        self.accumulator = (self.accumulator << length.min(31)) | (code & mask);
        self.bits += length;

        while self.bits >= 8 {
            let byte = ((self.accumulator >> (self.bits - 8)) & 0xFF) as u8;
            self.bytes.push(byte);
            // `0xFF` introduces a marker, so entropy data escapes a literal
            // one as `FF 00`. Omitting this is the classic encoder bug: the
            // file decodes correctly until the first pixel that happens to
            // produce an `0xFF`.
            if byte == 0xFF {
                self.bytes.push(0x00);
            }
            self.bits -= 8;
        }
        self.accumulator &= (1_u32 << self.bits) - 1;
    }

    /// Pad to a byte boundary with one bits.
    ///
    /// One bits rather than zero: a trailing run of zeros could be read as
    /// the start of a valid code, where a run of ones cannot be — no standard
    /// table assigns an all-ones code.
    fn flush(&mut self) {
        if self.bits > 0 {
            let padding = 8 - self.bits;
            self.write((1 << padding) - 1, padding);
        }
    }
}

/// The number of bits needed to code `value`, and the bits themselves.
///
/// JPEG codes a coefficient as a magnitude category plus that many raw bits,
/// where a negative value is stored as its predecessor in the category's
/// lower half — the mirror of `receive_extend` on the decoding side.
fn magnitude(value: i32) -> (u32, u32) {
    if value == 0 {
        return (0, 0);
    }
    let size = 32 - value.unsigned_abs().leading_zeros();
    let bits = if value < 0 {
        // `value - 1` in `size` bits: -1 becomes 0, -2 and -3 become 1 and 2.
        (value - 1) as u32 & ((1_u32 << size) - 1)
    } else {
        value as u32
    };
    (size, bits)
}

impl JpegEncoder {
    /// An encoder at the default quality.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            quality: EncodeOptions::DEFAULT_QUALITY,
            subsampling: Subsampling::Both,
            state: None,
        }
    }

    /// An encoder at an explicit quality, 1..=100.
    ///
    /// Chroma subsampling follows quality, as it does in every encoder that
    /// exposes one number: 4:4:4 from 90 up, 4:2:0 below. Above 90 the
    /// subsampling, not the quantization, becomes the dominant loss, so
    /// keeping it on would make the quality number stop meaning anything.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] unless `quality` is 1..=100.
    pub fn with_quality(quality: u8) -> Result<Self> {
        if !(1..=100).contains(&quality) {
            return Err(PixelsError::invalid_argument(
                "quality",
                format!("must be in 1..=100, got {quality}"),
            ));
        }
        Ok(Self {
            quality,
            subsampling: if quality >= 90 {
                Subsampling::None
            } else {
                Subsampling::Both
            },
            state: None,
        })
    }

    /// Override the chroma subsampling chosen by quality.
    #[must_use]
    pub const fn with_subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// An encoder configured from generic encode options.
    #[must_use]
    pub fn from_options(options: &EncodeOptions) -> Self {
        Self::with_quality(options.quality).unwrap_or_else(|_| Self::new())
    }

    /// The chosen chroma subsampling.
    #[must_use]
    pub const fn subsampling(&self) -> Subsampling {
        self.subsampling
    }
}

impl Default for JpegEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// How many channels a row of `format` carries, and whether JPEG can take it.
fn channels_of(format: PixelFormat) -> Result<usize> {
    match format {
        PixelFormat::Gray8 => Ok(1),
        PixelFormat::GrayA8 => Ok(2),
        PixelFormat::Rgb8 => Ok(3),
        PixelFormat::Rgba8 => Ok(4),
        // JPEG is 8-bit by definition at baseline, and has no alpha at all.
        other => Err(PixelsError::unsupported(format!(
            "JPEG encoding needs an 8-bit format; got {other}. Convert first."
        ))),
    }
}

/// Read one pixel as RGB, compositing any alpha against black.
///
/// Alpha is composited rather than dropped, for the reason the GIF encoder
/// gives: JPEG has no transparency, so a translucent pixel has to become
/// *something*, and blending against black is what a viewer shows for a
/// flattened image.
fn pixel_rgb(row: &[u8], index: usize, format: PixelFormat) -> [u8; 3] {
    let channels = format.channels();
    let at = index * channels;
    let get = |offset: usize| u32::from(row.get(at + offset).copied().unwrap_or(0));
    let blend = |value: u32, alpha: u32| ((value * alpha + 127) / 255) as u8;

    match format {
        PixelFormat::Gray8 => {
            let value = get(0) as u8;
            [value, value, value]
        }
        PixelFormat::GrayA8 => {
            let value = blend(get(0), get(1));
            [value, value, value]
        }
        PixelFormat::Rgba8 => {
            let alpha = get(3);
            [
                blend(get(0), alpha),
                blend(get(1), alpha),
                blend(get(2), alpha),
            ]
        }
        // Rgb8 and anything `channels_of` already rejected.
        _ => [get(0) as u8, get(1) as u8, get(2) as u8],
    }
}

/// Convert one RGB triple to YCbCr in fixed point.
///
/// The inverse of the decoder's transform, with JFIF's coefficients scaled by
/// 2^16, and fixed point for the same reason (ADR-0011).
fn rgb_to_ycbcr([r, g, b]: [u8; 3]) -> [u8; 3] {
    const HALF: i32 = 1 << 15;
    let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));

    let y = (19_595 * r + 38_470 * g + 7_471 * b + HALF) >> 16;
    let cb = ((-11_056 * r - 21_712 * g + 32_768 * b + HALF) >> 16) + 128;
    let cr = ((32_768 * r - 27_440 * g - 5_328 * b + HALF) >> 16) + 128;
    [
        y.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
    ]
}

impl State {
    /// Fill the component planes from the rows accumulated in `band`.
    ///
    /// Padding is edge replication in both axes: the format codes whole
    /// blocks, so an image whose width or height is not a multiple of the MCU
    /// size has samples that must be invented. Replicating the edge invents
    /// the ones that cost the fewest bits — a constant would put an artificial
    /// edge inside the last block, and the DCT would spend high-frequency
    /// coefficients describing it.
    fn fill_planes(&mut self) {
        let width = self.descriptor.width as usize;
        let format = self.descriptor.pixel;
        let row_bytes = self.descriptor.row_bytes();
        let (h, v) = (self.factors.0 as usize, self.factors.1 as usize);
        let stride = self.luma.stride;
        // The last row that holds real pixels; everything below it repeats.
        let last = (self.band_rows.max(1) - 1) as usize;

        for y in 0..self.luma.height {
            let row = self
                .band
                .get(y.min(last) * row_bytes..)
                .and_then(|rest| rest.get(..row_bytes))
                .unwrap_or(&[]);

            for x in 0..stride {
                let rgb = pixel_rgb(row, x.min(width - 1), format);
                let ycbcr = if self.grayscale {
                    [rgb[0], 128, 128]
                } else {
                    rgb_to_ycbcr(rgb)
                };
                if let Some(slot) = self.luma.samples.get_mut(y * stride + x) {
                    *slot = ycbcr[0];
                }
                if self.grayscale {
                    continue;
                }
                // Full-resolution chroma is kept only until it is averaged
                // down, which is why it lives in scratch rather than a plane.
                if let Some(slot) = self.chroma_full.get_mut((y * stride + x) * 2..) {
                    if let Some(pair) = slot.get_mut(..2) {
                        pair.copy_from_slice(&[ycbcr[1], ycbcr[2]]);
                    }
                }
            }
        }
        if self.grayscale {
            return;
        }

        // Box-filter chroma down to its own resolution. Averaging is what
        // makes subsampling a resampling rather than a decimation: keeping
        // every other sample instead would alias hard edges into the chroma.
        let count = (h * v) as u32;
        for cy in 0..self.chroma_blue.height {
            for cx in 0..self.chroma_blue.stride {
                let (mut blue, mut red) = (0_u32, 0_u32);
                for dy in 0..v {
                    for dx in 0..h {
                        let y = (cy * v + dy).min(self.luma.height - 1);
                        let x = (cx * h + dx).min(stride - 1);
                        let at = (y * stride + x) * 2;
                        blue += u32::from(self.chroma_full.get(at).copied().unwrap_or(128));
                        red += u32::from(self.chroma_full.get(at + 1).copied().unwrap_or(128));
                    }
                }
                let at = cy * self.chroma_blue.stride + cx;
                if let Some(slot) = self.chroma_blue.samples.get_mut(at) {
                    *slot = ((blue + count / 2) / count) as u8;
                }
                if let Some(slot) = self.chroma_red.samples.get_mut(at) {
                    *slot = ((red + count / 2) / count) as u8;
                }
            }
        }
    }

    /// Transform, quantize and entropy-code one MCU row.
    fn encode_band(&mut self) {
        let (h, v) = self.factors;
        let mut samples = [0_u8; 64];
        let mut coefficients = [0_i64; 64];
        let mut quantized = [0_i32; 64];

        let [luma_predictor, blue_predictor, red_predictor] = &mut self.predictors;

        for mcu in 0..self.mcus_per_line {
            // Luma blocks first, in raster order within the MCU: that is the
            // interleave order the scan header declares.
            for block_y in 0..v {
                for block_x in 0..h {
                    let x = ((mcu * h + block_x) * 8) as usize;
                    let y = (block_y * 8) as usize;
                    self.luma.block(x, y, &mut samples);
                    fdct::block(&samples, &mut coefficients);
                    fdct::quantize(&coefficients, &self.luma_quant, &mut quantized);
                    encode_block(
                        &mut self.writer,
                        &quantized,
                        &self.luma_dc,
                        &self.luma_ac,
                        luma_predictor,
                    );
                }
            }
            if self.grayscale {
                continue;
            }

            let x = (mcu * 8) as usize;
            for (plane, predictor) in [
                (&self.chroma_blue, &mut *blue_predictor),
                (&self.chroma_red, &mut *red_predictor),
            ] {
                plane.block(x, 0, &mut samples);
                fdct::block(&samples, &mut coefficients);
                fdct::quantize(&coefficients, &self.chroma_quant, &mut quantized);
                encode_block(
                    &mut self.writer,
                    &quantized,
                    &self.chroma_dc,
                    &self.chroma_ac,
                    predictor,
                );
            }
        }
    }
}

/// Entropy-code one quantized block.
fn encode_block(
    writer: &mut BitWriter,
    block: &[i32; 64],
    dc: &HuffmanEncoder,
    ac: &HuffmanEncoder,
    predictor: &mut i32,
) {
    let value = block.first().copied().unwrap_or(0);
    // DC is coded as a difference from the previous block of the same
    // component, because neighbouring blocks of a photograph have nearly the
    // same average brightness.
    let difference = value.wrapping_sub(*predictor);
    *predictor = value;

    let (size, bits) = magnitude(difference);
    if let Some((code, length)) = dc.code(size as u8) {
        writer.write(code, length);
    }
    writer.write(bits, size);

    let mut run = 0_u32;
    for index in 1..64 {
        let value = ZIGZAG
            .get(index)
            .and_then(|&at| block.get(at))
            .copied()
            .unwrap_or(0);
        if value == 0 {
            run += 1;
            continue;
        }
        // The run length field holds four bits, so longer runs are broken up
        // with the zero-run-length code.
        while run >= 16 {
            if let Some((code, length)) = ac.code(0xF0) {
                writer.write(code, length);
            }
            run -= 16;
        }
        let (size, bits) = magnitude(value);
        if let Some((code, length)) = ac.code(((run as u8) << 4) | size as u8) {
            writer.write(code, length);
        }
        writer.write(bits, size);
        run = 0;
    }
    // A trailing run of zeros is collapsed into one end-of-block code, which
    // is where most of JPEG's compression comes from.
    if run > 0 {
        if let Some((code, length)) = ac.code(0x00) {
            writer.write(code, length);
        }
    }
}

/// Write a marker with no payload.
fn write_marker(code: u8, sink: &mut dyn Sink) -> Result<()> {
    sink.write_all(&[0xFF, code])
}

/// Write a marker segment, prefixing the length its payload implies.
fn write_segment(code: u8, payload: &[u8], sink: &mut dyn Sink) -> Result<()> {
    let Ok(length) = u16::try_from(payload.len() + 2) else {
        return Err(PixelsError::unsupported(format!(
            "a {code:#04x} segment of {} bytes does not fit a 16-bit length",
            payload.len()
        )));
    };
    sink.write_all(&[0xFF, code])?;
    sink.write_all(&length.to_be_bytes())?;
    sink.write_all(payload)
}

/// Write one quantization table, in the zigzag order `DQT` uses.
fn write_quant_table(slot: u8, steps: &[u16; 64], payload: &mut Vec<u8>) {
    // Precision 0 (8-bit) in the high nibble, slot in the low.
    payload.push(slot & 0x0F);
    for &position in &ZIGZAG {
        payload.push(steps.get(position).copied().unwrap_or(1).clamp(1, 255) as u8);
    }
}

/// Write one Huffman table definition.
fn write_huffman_table(
    class: u8,
    slot: u8,
    counts: &[u8; 16],
    values: &[u8],
    payload: &mut Vec<u8>,
) {
    payload.push(((class & 0x0F) << 4) | (slot & 0x0F));
    payload.extend_from_slice(counts);
    payload.extend_from_slice(values);
}

impl Encoder for JpegEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        let channels = channels_of(desc.pixel)?;
        // JPEG dimensions are 16-bit; a larger image cannot be represented at
        // all, so this is a format limit rather than a policy.
        if desc.width > u32::from(u16::MAX) || desc.height > u32::from(u16::MAX) {
            return Err(PixelsError::unsupported(format!(
                "JPEG dimensions are 16-bit; {}x{} does not fit",
                desc.width, desc.height
            )));
        }

        let grayscale = channels <= 2;
        // Subsampling chroma that does not exist would only pad the planes.
        let factors = if grayscale {
            (1, 1)
        } else {
            self.subsampling.factors()
        };
        let luma_quant = tables::scale_quant(&tables::LUMA_QUANT, self.quality);
        let chroma_quant = tables::scale_quant(&tables::CHROMA_QUANT, self.quality);

        let (h, v) = factors;
        let band_height = v * 8;
        let mcus_per_line = desc.width.div_ceil(h * 8);
        let luma_stride = (mcus_per_line * h * 8) as usize;
        let chroma_stride = (mcus_per_line * 8) as usize;

        sink.write_all(&[0xFF, marker::SOI])?;

        // A JFIF header. Nothing here needs it — the density fields are the
        // only content and they say "no units" — but its absence makes some
        // consumers guess at the colour model rather than assume YCbCr.
        write_segment(
            marker::APP0,
            &[
                b'J', b'F', b'I', b'F', 0, // identifier
                1, 2, // version 1.02
                0, // density units: none
                0, 1, 0, 1, // pixel aspect ratio 1:1
                0, 0, // no thumbnail
            ],
            sink,
        )?;

        let mut payload = Vec::new();
        write_quant_table(0, &luma_quant, &mut payload);
        if !grayscale {
            write_quant_table(1, &chroma_quant, &mut payload);
        }
        write_segment(marker::DQT, &payload, sink)?;

        let mut payload = vec![8];
        payload.extend_from_slice(&(desc.height as u16).to_be_bytes());
        payload.extend_from_slice(&(desc.width as u16).to_be_bytes());
        if grayscale {
            payload.push(1);
            payload.extend_from_slice(&[1, 0x11, 0]);
        } else {
            payload.push(3);
            payload.extend_from_slice(&[1, ((h as u8) << 4) | v as u8, 0]);
            payload.extend_from_slice(&[2, 0x11, 1]);
            payload.extend_from_slice(&[3, 0x11, 1]);
        }
        write_segment(marker::SOF0, &payload, sink)?;

        let mut payload = Vec::new();
        write_huffman_table(
            0,
            0,
            &tables::LUMA_DC_COUNTS,
            &tables::LUMA_DC_VALUES,
            &mut payload,
        );
        write_huffman_table(
            1,
            0,
            &tables::LUMA_AC_COUNTS,
            &tables::LUMA_AC_VALUES,
            &mut payload,
        );
        if !grayscale {
            write_huffman_table(
                0,
                1,
                &tables::CHROMA_DC_COUNTS,
                &tables::CHROMA_DC_VALUES,
                &mut payload,
            );
            write_huffman_table(
                1,
                1,
                &tables::CHROMA_AC_COUNTS,
                &tables::CHROMA_AC_VALUES,
                &mut payload,
            );
        }
        write_segment(marker::DHT, &payload, sink)?;

        let mut payload = Vec::new();
        if grayscale {
            payload.push(1);
            payload.extend_from_slice(&[1, 0x00]);
        } else {
            payload.push(3);
            payload.extend_from_slice(&[1, 0x00, 2, 0x11, 3, 0x11]);
        }
        // Spectral selection 0..=63 with no successive approximation: the
        // only shape a baseline scan may take.
        payload.extend_from_slice(&[0, 63, 0]);
        write_segment(marker::SOS, &payload, sink)?;

        self.state = Some(State {
            descriptor: *desc,
            grayscale,
            factors,
            luma_quant,
            chroma_quant,
            luma_dc: HuffmanEncoder::new(&tables::LUMA_DC_COUNTS, &tables::LUMA_DC_VALUES)?,
            luma_ac: HuffmanEncoder::new(&tables::LUMA_AC_COUNTS, &tables::LUMA_AC_VALUES)?,
            chroma_dc: HuffmanEncoder::new(&tables::CHROMA_DC_COUNTS, &tables::CHROMA_DC_VALUES)?,
            chroma_ac: HuffmanEncoder::new(&tables::CHROMA_AC_COUNTS, &tables::CHROMA_AC_VALUES)?,
            band: vec![0; desc.row_bytes() * band_height as usize],
            band_rows: 0,
            band_height,
            mcus_per_line,
            luma: Plane::new(luma_stride, band_height as usize),
            // A greyscale image has no chroma to hold, so none is allocated.
            chroma_full: if grayscale {
                Vec::new()
            } else {
                vec![128; luma_stride * band_height as usize * 2]
            },
            chroma_blue: chroma_plane(grayscale, chroma_stride, band_height / v),
            chroma_red: chroma_plane(grayscale, chroma_stride, band_height / v),
            predictors: [0; 3],
            writer: BitWriter::default(),
            rows_written: 0,
        });
        Ok(())
    }

    fn write_row(&mut self, row: &[u8], sink: &mut dyn Sink) -> Result<()> {
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

        let at = state.band_rows as usize * expected;
        if let Some(slot) = state
            .band
            .get_mut(at..)
            .and_then(|rest| rest.get_mut(..expected))
        {
            slot.copy_from_slice(row);
        }
        state.band_rows += 1;
        state.rows_written += 1;

        if state.band_rows == state.band_height {
            state.fill_planes();
            state.encode_band();
            state.band_rows = 0;
            // Drained here rather than at the end: this is what makes the
            // encoder streaming rather than merely incremental.
            sink.write_all(&state.writer.bytes)?;
            state.writer.bytes.clear();
        }
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "sink",
                "finish called before write_header",
            ));
        };
        if state.rows_written < state.descriptor.height {
            return Err(PixelsError::malformed(
                "jpeg",
                format!(
                    "{} of {} rows were written",
                    state.rows_written, state.descriptor.height
                ),
            ));
        }

        // A final partial MCU row still has to be coded: the format has no
        // way to say "this image ends mid-block". `fill_planes` pads it by
        // replicating the last real row.
        if state.band_rows > 0 {
            state.fill_planes();
            state.encode_band();
            state.band_rows = 0;
        }
        state.writer.flush();
        sink.write_all(&state.writer.bytes)?;
        state.writer.bytes.clear();

        write_marker(marker::EOI, sink)?;
        sink.flush()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_categories_mirror_the_decoder() {
        // The property: `magnitude` and the decoder's `receive_extend` are
        // inverses over every value a coefficient can take.
        for value in [-2047_i32, -255, -8, -3, -2, -1, 1, 2, 3, 8, 255, 2047] {
            let (size, bits) = magnitude(value);
            assert!(size <= 15, "{value}: size {size}");
            // Reproduce the decoder's extension.
            let threshold = 1_i32 << (size - 1);
            let raw = bits as i32;
            let decoded = if raw < threshold {
                raw - (1_i32 << size) + 1
            } else {
                raw
            };
            assert_eq!(decoded, value, "{value} round-tripped as {decoded}");
        }
        assert_eq!(magnitude(0), (0, 0));
    }

    #[test]
    fn the_bit_writer_stuffs_ff_bytes() {
        let mut writer = BitWriter::default();
        writer.write(0xFF, 8);
        assert_eq!(
            writer.bytes,
            vec![0xFF, 0x00],
            "a literal FF must be stuffed"
        );

        let mut writer = BitWriter::default();
        writer.write(0b1010, 4);
        writer.write(0b0101, 4);
        assert_eq!(writer.bytes, vec![0b1010_0101]);
    }

    #[test]
    fn flushing_pads_with_one_bits() {
        let mut writer = BitWriter::default();
        writer.write(0b101, 3);
        writer.flush();
        // Five bits of padding, all ones.
        assert_eq!(writer.bytes, vec![0b1011_1111]);

        // Already aligned: nothing is added.
        let mut writer = BitWriter::default();
        writer.write(0xAB, 8);
        writer.flush();
        assert_eq!(writer.bytes, vec![0xAB]);
    }

    #[test]
    fn colour_conversion_round_trips_through_the_decoder() {
        // Grey stays grey and the primaries land where they should; the exact
        // values are checked against the decoder's inverse rather than
        // hardcoded, because the pair being inverses is the property.
        for rgb in [
            [0, 0, 0],
            [255, 255, 255],
            [128, 128, 128],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [37, 142, 201],
        ] {
            let ycbcr = rgb_to_ycbcr(rgb);
            if rgb[0] == rgb[1] && rgb[1] == rgb[2] {
                assert_eq!(ycbcr[0], rgb[0], "grey should map to its own luma");
                assert_eq!([ycbcr[1], ycbcr[2]], [128, 128], "grey has no chroma");
            }
            let back = crate::decoder::ycbcr_to_rgb(ycbcr);
            for channel in 0..3 {
                assert!(
                    back[channel].abs_diff(rgb[channel]) <= 2,
                    "{rgb:?} -> {ycbcr:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn unsupported_pixel_formats_are_refused_at_the_header() {
        for format in [
            PixelFormat::Gray16,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::RgbF32,
            PixelFormat::RgbaF32,
        ] {
            let descriptor = ImageDescriptor::new(8, 8, format).unwrap();
            let mut sink = Vec::new();
            let error = JpegEncoder::new()
                .write_header(&descriptor, &mut sink)
                .unwrap_err();
            assert_eq!(
                error.code(),
                otf_pixels_core::ErrorCode::Unsupported,
                "{format}"
            );
            assert!(sink.is_empty(), "{format}: bytes were written anyway");
        }
    }

    #[test]
    fn quality_selects_subsampling_but_can_be_overridden() {
        assert_eq!(
            JpegEncoder::with_quality(80).unwrap().subsampling(),
            Subsampling::Both
        );
        assert_eq!(
            JpegEncoder::with_quality(95).unwrap().subsampling(),
            Subsampling::None
        );
        assert_eq!(
            JpegEncoder::with_quality(95)
                .unwrap()
                .with_subsampling(Subsampling::Horizontal)
                .subsampling(),
            Subsampling::Horizontal
        );
        assert!(JpegEncoder::with_quality(0).is_err());
        assert!(JpegEncoder::with_quality(101).is_err());
    }
}
