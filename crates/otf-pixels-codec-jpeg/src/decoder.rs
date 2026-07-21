//! The baseline JPEG decoder.
//!
//! # Shape of the work
//!
//! A JPEG is a sequence of marker segments — quantization tables, Huffman
//! tables, a frame header — followed by one scan of entropy-coded data.
//! [`JpegDecoder::new`] reads up to and including the scan header, so
//! everything the engine needs to plan a pipeline is known before a single
//! coefficient is decoded.
//!
//! The scan is then consumed one **MCU row** at a time. An MCU (minimum coded
//! unit) is the smallest group of blocks that covers the same rectangle in
//! every component, which with 2x2 chroma subsampling is four luma blocks and
//! one of each chroma: 16x16 pixels. Decoding a row of them fills a band of
//! component planes, which is upsampled and colour-converted into interleaved
//! output rows on the spot. Peak memory is that band, not the image.

use crate::entropy::Reader;
use crate::format::{
    AdobeTransform, Frame, Scan, ZIGZAG, adobe_transform, exif_orientation, marker,
};
use crate::huffman::HuffmanTable;
use crate::idct::{self, Scale};
use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Format, ImageDescriptor, Limits, PixelFormat, PixelsError,
    Result, Source,
};

/// The most blocks a single MCU may contain (ITU-T T.81 §B.2.3).
///
/// Sampling factors are capped at 4x4 each, so without this a crafted frame
/// could ask for 48 blocks per MCU across three components. The format says
/// ten; holding it to that bounds the work one MCU can cost.
const MAX_BLOCKS_PER_MCU: u32 = 10;

/// How the frame's components map onto colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Colour {
    /// One component: luminance only.
    Grayscale,
    /// Three components, transformed.
    YCbCr,
    /// Three components already in RGB order.
    Rgb,
}

/// One component's plane for the MCU row currently being decoded.
#[derive(Debug)]
struct Plane {
    /// Samples per row, including the padding that rounds up to whole blocks.
    stride: usize,
    /// Decoded samples for this band.
    samples: Vec<u8>,
    /// Output column to plane column, precomputed to keep upsampling out of
    /// the inner loop.
    columns: Vec<u32>,
    /// Output row within the band to plane row.
    rows: Vec<u32>,
}

/// Decodes a baseline JPEG stream.
pub struct JpegDecoder<S: Source> {
    reader: Reader<S>,
    descriptor: ImageDescriptor,
    frame: Frame,
    scan: Scan,
    colour: Colour,
    /// Quantization tables in zigzag order, by slot.
    quant: [[u16; 64]; 4],
    /// DC and AC Huffman tables, by slot.
    dc_tables: [Option<HuffmanTable>; 4],
    ac_tables: [Option<HuffmanTable>; 4],
    /// MCUs between restart markers; zero means there are none.
    restart_interval: u16,
    /// MCUs left before the next restart marker is due.
    restarts_left: u32,
    /// The EXIF orientation tag, if the file carries one.
    orientation: Option<u8>,
    planes: Vec<Plane>,
    /// Per-component DC predictor, reset at every restart.
    predictors: Vec<i32>,
    /// Interleaved output rows for the MCU row just decoded.
    band: Vec<u8>,
    /// The next row of `band` to serve; equal to the band height when spent.
    band_row: u32,
    /// Pixel height of one MCU row, and so of `band`.
    band_height: u32,
    mcus_per_line: u32,
    mcu_rows: u32,
    /// The next MCU row to decode.
    mcu_row: u32,
    /// How much of full resolution this decode produces.
    scale: Scale,
    /// Output rows already served.
    row: u32,
}

impl<S: Source> std::fmt::Debug for JpegDecoder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JpegDecoder")
            .field("descriptor", &self.descriptor)
            .field("colour", &self.colour)
            .field("components", &self.frame.components)
            .field("scale", &self.scale)
            .field("restart_interval", &self.restart_interval)
            .field("orientation", &self.orientation)
            .field("row", &self.row)
            .finish_non_exhaustive()
    }
}

impl<S: Source> JpegDecoder<S> {
    /// Read every header up to and including the scan header.
    ///
    /// No coefficient is decoded here, so this is the `probe()`/metadata path
    /// as well as the start of a decode (SPEC §Guarantees 3).
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a stream that is not a
    /// well-formed baseline JPEG, [`PixelsError::Unsupported`] for a JPEG
    /// variant this crate does not own (progressive, arithmetic-coded, 12-bit,
    /// CMYK), or [`PixelsError::LimitExceeded`] if the frame exceeds `limits`.
    pub fn new(source: S, limits: Limits) -> Result<Self> {
        Self::with_scale(source, limits, Scale::Full)
    }

    /// Read every header, and decode at a reduced resolution.
    ///
    /// At `M/8` scale the decoder inverse-transforms only the low-frequency
    /// corner of each block, so a thumbnail costs a fraction of the arithmetic
    /// and — the point of it — the full-resolution image is never
    /// materialized for the rest of the pipeline to carry. Entropy decoding is
    /// unchanged: every coefficient is still read, because the format gives no
    /// way to skip one.
    ///
    /// [`Decoder::descriptor`] reports the *scaled* size, so a caller that
    /// asks for a reduced decode is told what it will actually receive.
    ///
    /// # Errors
    ///
    /// As [`JpegDecoder::new`].
    pub fn with_scale(source: S, limits: Limits, scale: Scale) -> Result<Self> {
        let mut reader = Reader::new(source);
        let mut quant = [[0_u16; 64]; 4];
        let mut dc_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut ac_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut restart_interval = 0_u16;
        let mut orientation = None;
        let mut adobe = None;
        let mut frame: Option<Frame> = None;

        let first = reader.next_marker()?;
        if first != marker::SOI {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("stream begins with marker {first:#04x}, not SOI"),
            ));
        }

        let scan = loop {
            let code = reader.next_marker()?;
            match code {
                marker::SOF0 | marker::SOF1 => {
                    if frame.is_some() {
                        return Err(PixelsError::malformed(
                            "jpeg",
                            "stream declares more than one frame",
                        ));
                    }
                    let parsed = Frame::parse(&reader.read_segment()?)?;
                    // Checked here rather than once the scan header arrives,
                    // so an enormous frame costs one segment parse and stops
                    // (SPEC §Safety and limits).
                    limits.check(u32::from(parsed.width), u32::from(parsed.height))?;
                    frame = Some(parsed);
                }
                marker::SOF2 => {
                    // ADR-0004: progressive is a wrapped codec, not ours. The
                    // decoders differ far more than the marker suggests — the
                    // coefficients arrive in spectral bands across many scans,
                    // which rules out the one-band-at-a-time streaming this
                    // decoder is built around.
                    return Err(PixelsError::unsupported(
                        "jpeg: progressive JPEG; this codec decodes baseline only",
                    ));
                }
                marker::DAC => {
                    return Err(PixelsError::unsupported(
                        "jpeg: arithmetic coding; baseline JPEG is Huffman coded",
                    ));
                }
                code if marker::is_frame(code) => {
                    return Err(PixelsError::unsupported(format!(
                        "jpeg: frame type {code:#04x} is not baseline"
                    )));
                }
                marker::DQT => read_quantization_tables(&reader.read_segment()?, &mut quant)?,
                marker::DHT => {
                    read_huffman_tables(&reader.read_segment()?, &mut dc_tables, &mut ac_tables)?;
                }
                marker::DRI => {
                    let payload = reader.read_segment()?;
                    let (Some(&hi), Some(&lo)) = (payload.first(), payload.get(1)) else {
                        return Err(PixelsError::malformed(
                            "jpeg",
                            "DRI segment carries no interval",
                        ));
                    };
                    restart_interval = u16::from_be_bytes([hi, lo]);
                }
                marker::APP1 => {
                    let payload = reader.read_segment()?;
                    // The first EXIF block wins; later ones are thumbnails.
                    orientation = orientation.or_else(|| exif_orientation(&payload));
                }
                marker::APP14 => {
                    let payload = reader.read_segment()?;
                    adobe = adobe_transform(&payload).or(adobe);
                }
                marker::SOS => {
                    let Some(ref frame) = frame else {
                        return Err(PixelsError::malformed(
                            "jpeg",
                            "scan begins before any frame header",
                        ));
                    };
                    break Scan::parse(&reader.read_segment()?, frame)?;
                }
                marker::EOI => {
                    return Err(PixelsError::malformed(
                        "jpeg",
                        "stream ends before any scan",
                    ));
                }
                // Segments that carry no pixels are skipped, not rejected:
                // JFIF density, colour profiles, comments and vendor
                // extensions all live here, and refusing them would fail
                // files every viewer opens.
                marker::APP0..=marker::APP15 | marker::COM => reader.skip_segment()?,
                code if marker::is_standalone(code) => {}
                _ => reader.skip_segment()?,
            }
        };

        let Some(frame) = frame else {
            return Err(PixelsError::malformed("jpeg", "stream has no frame header"));
        };
        check_baseline_scan(&scan)?;

        let colour = colour_model(&frame, adobe)?;
        let pixel = match colour {
            Colour::Grayscale => PixelFormat::Gray8,
            Colour::YCbCr | Colour::Rgb => PixelFormat::Rgb8,
        };
        // The frame's own size drives the MCU grid — that is a property of the
        // encoded data, not of how much of it we intend to produce — while the
        // descriptor reports what the caller will actually receive.
        let (full_width, full_height) = (u32::from(frame.width), u32::from(frame.height));
        let descriptor = ImageDescriptor::with_limits(
            scale.apply(full_width),
            scale.apply(full_height),
            pixel,
            &limits,
        )?;

        if scan.components.len() != frame.components.len() {
            // Every component of a baseline frame is coded in one interleaved
            // scan in practice. The alternative — a scan per component — needs
            // whole-image component planes, which would trade the memory
            // guarantee for a case no encoder in circulation produces.
            return Err(PixelsError::unsupported(
                "jpeg: non-interleaved baseline scan",
            ));
        }
        for component in &scan.components {
            let slot = frame
                .components
                .get(component.index)
                .map_or(0, |c| c.quant as usize);
            if quant
                .get(slot)
                .is_none_or(|table| table.iter().all(|&q| q == 0))
            {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("scan uses quantization table {slot}, which was never defined"),
                ));
            }
            if dc_tables
                .get(component.dc as usize)
                .is_none_or(Option::is_none)
                || ac_tables
                    .get(component.ac as usize)
                    .is_none_or(Option::is_none)
            {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!(
                        "scan uses Huffman tables {}/{}, which were never defined",
                        component.dc, component.ac
                    ),
                ));
            }
        }

        let (h_max, v_max) = (u32::from(frame.h_max()), u32::from(frame.v_max()));
        let blocks: u32 = frame
            .components
            .iter()
            .map(|c| u32::from(c.h) * u32::from(c.v))
            .sum();
        if blocks > MAX_BLOCKS_PER_MCU {
            return Err(PixelsError::malformed(
                "jpeg",
                format!(
                    "an MCU would hold {blocks} blocks; the format allows {MAX_BLOCKS_PER_MCU}"
                ),
            ));
        }

        // How many samples one block becomes: 8 at full resolution, fewer at a
        // reduced scale. Every plane dimension below is in these units, which
        // is the whole of what a scaled decode changes downstream.
        let sample = scale.block_size();
        let mcus_per_line = full_width.div_ceil(h_max * 8);
        let mcu_rows = full_height.div_ceil(v_max * 8);
        let band_height = v_max * sample;

        let mut planes = Vec::with_capacity(frame.components.len());
        for component in &frame.components {
            let (h, v) = (u32::from(component.h), u32::from(component.v));
            let stride = usize::try_from(mcus_per_line * h * sample)
                .map_err(|_| PixelsError::malformed("jpeg", "component plane overflows"))?;
            let height = usize::try_from(v * sample).unwrap_or(32);
            let samples = stride
                .checked_mul(height)
                .ok_or_else(|| PixelsError::malformed("jpeg", "component plane overflows"))?;
            planes.push(Plane {
                stride,
                samples: vec![0_u8; samples],
                // Nearest-neighbour upsampling: output column `x` reads the
                // component sample covering it. Triangle ("fancy") upsampling
                // is smoother and is what libjpeg defaults to, which is why
                // reference comparisons are made against a decoder configured
                // to match rather than against libjpeg's default.
                columns: (0..descriptor.width).map(|x| x * h / h_max).collect(),
                rows: (0..band_height).map(|y| y * v / v_max).collect(),
            });
        }

        let row_bytes = descriptor.row_bytes();
        let band = row_bytes
            .checked_mul(band_height as usize)
            .ok_or_else(|| PixelsError::malformed("jpeg", "MCU row band overflows"))?;

        Ok(Self {
            reader,
            descriptor,
            predictors: vec![0; frame.components.len()],
            frame,
            scan,
            colour,
            quant,
            dc_tables,
            ac_tables,
            restart_interval,
            restarts_left: u32::from(restart_interval),
            orientation,
            planes,
            band: vec![0_u8; band],
            // Nothing decoded yet, so the band is spent and the first
            // `read_row` fills it.
            band_row: band_height,
            band_height,
            mcus_per_line,
            mcu_rows,
            mcu_row: 0,
            scale,
            row: 0,
        })
    }

    /// The resolution this decoder produces, as eighths of full size.
    #[must_use]
    pub const fn scale(&self) -> Scale {
        self.scale
    }

    /// The EXIF orientation tag, 1..=8, if the file carries one.
    ///
    /// Applying it is the caller's job: `auto_orient` is a pipeline decision
    /// (SPEC §Safety and limits), and a decoder that rotated its own output
    /// would leave no way to turn that off.
    #[must_use]
    pub const fn orientation(&self) -> Option<u8> {
        self.orientation
    }

    /// Decode one MCU row into the component planes, then convert it into the
    /// interleaved output band.
    fn fill_band(&mut self) -> Result<()> {
        if self.mcu_row >= self.mcu_rows {
            return Err(PixelsError::malformed(
                "jpeg",
                "more rows were requested than the frame declares",
            ));
        }

        let mut coefficients = [0_i32; 64];
        let scale = self.scale;
        let sample = scale.block_size() as usize;
        for mcu in 0..self.mcus_per_line {
            if self.restart_interval > 0 && self.restarts_left == 0 {
                if !self.reader.restart()? {
                    let met = self.reader.pending_marker();
                    return Err(PixelsError::malformed(
                        "jpeg",
                        match met {
                            Some(code) => format!(
                                "expected a restart marker between MCU intervals, met {code:#04x}"
                            ),
                            None => "expected a restart marker between MCU intervals".to_owned(),
                        },
                    ));
                }
                // Predictors are differential *within* an interval; that is
                // the whole point of restarts, and carrying them across one
                // would corrupt every block after a resynchronization.
                self.predictors.iter_mut().for_each(|p| *p = 0);
                self.restarts_left = u32::from(self.restart_interval);
            }

            for scanned in &self.scan.components {
                let Some(component) = self.frame.components.get(scanned.index) else {
                    continue;
                };
                let (h, v) = (u32::from(component.h), u32::from(component.v));
                let (Some(dc), Some(ac)) = (
                    self.dc_tables
                        .get(scanned.dc as usize)
                        .and_then(Option::as_ref),
                    self.ac_tables
                        .get(scanned.ac as usize)
                        .and_then(Option::as_ref),
                ) else {
                    return Err(PixelsError::malformed(
                        "jpeg",
                        "scan names a Huffman table that was never defined",
                    ));
                };
                let quant = self
                    .quant
                    .get(component.quant as usize)
                    .ok_or_else(|| PixelsError::malformed("jpeg", "component names no table"))?;
                let Some(predictor) = self.predictors.get_mut(scanned.index) else {
                    continue;
                };
                let Some(plane) = self.planes.get_mut(scanned.index) else {
                    continue;
                };

                for block_y in 0..v {
                    for block_x in 0..h {
                        decode_block(
                            &mut self.reader,
                            dc,
                            ac,
                            quant,
                            predictor,
                            &mut coefficients,
                        )?;
                        let x = ((mcu * h) + block_x) as usize * sample;
                        let y = block_y as usize * sample;
                        idct::scaled_block(
                            &coefficients,
                            scale,
                            &mut plane.samples,
                            y * plane.stride + x,
                            plane.stride,
                        );
                    }
                }
            }

            if self.restart_interval > 0 {
                self.restarts_left = self.restarts_left.saturating_sub(1);
            }
        }

        self.mcu_row += 1;
        self.convert_band();
        self.band_row = 0;
        Ok(())
    }

    /// Upsample the component planes and write interleaved pixels into the
    /// output band.
    fn convert_band(&mut self) {
        let width = self.descriptor.width as usize;
        let row_bytes = self.descriptor.row_bytes();

        for y in 0..self.band_height as usize {
            let Some(out) = self
                .band
                .get_mut(y * row_bytes..)
                .and_then(|rest| rest.get_mut(..row_bytes))
            else {
                continue;
            };

            match self.colour {
                Colour::Grayscale => {
                    let Some(plane) = self.planes.first() else {
                        continue;
                    };
                    let source = plane.row(y);
                    for (x, slot) in out.iter_mut().enumerate().take(width) {
                        *slot = plane.sample(source, x);
                    }
                }
                Colour::YCbCr | Colour::Rgb => {
                    let (Some(first), Some(second), Some(third)) =
                        (self.planes.first(), self.planes.get(1), self.planes.get(2))
                    else {
                        continue;
                    };
                    let (a, b, c) = (first.row(y), second.row(y), third.row(y));
                    for (x, pixel) in out.chunks_exact_mut(3).enumerate().take(width) {
                        let samples = [first.sample(a, x), second.sample(b, x), third.sample(c, x)];
                        let rgb = if self.colour == Colour::Rgb {
                            samples
                        } else {
                            ycbcr_to_rgb(samples)
                        };
                        for (slot, value) in pixel.iter_mut().zip(rgb) {
                            *slot = value;
                        }
                    }
                }
            }
        }
    }
}

impl Plane {
    /// The plane row backing output row `y` of the band.
    fn row(&self, y: usize) -> &[u8] {
        let row = self.rows.get(y).copied().unwrap_or(0) as usize;
        self.samples
            .get(row * self.stride..)
            .and_then(|rest| rest.get(..self.stride))
            .unwrap_or(&[])
    }

    /// The sample of `row` covering output column `x`.
    fn sample(&self, row: &[u8], x: usize) -> u8 {
        let column = self.columns.get(x).copied().unwrap_or(0) as usize;
        row.get(column).copied().unwrap_or(0)
    }
}

/// Decode one 8x8 block, dequantizing into natural order as it goes.
fn decode_block<S: Source>(
    reader: &mut Reader<S>,
    dc: &HuffmanTable,
    ac: &HuffmanTable,
    quant: &[u16; 64],
    predictor: &mut i32,
    out: &mut [i32; 64],
) -> Result<()> {
    out.fill(0);

    let magnitude = reader.decode(dc)?;
    if magnitude > 15 {
        return Err(PixelsError::malformed(
            "jpeg",
            format!("DC coefficient claims {magnitude} bits; 15 is the maximum"),
        ));
    }
    let difference = reader.receive_extend(u32::from(magnitude))?;
    // DC is coded as a difference from the previous block of the same
    // component. Wrapping keeps a crafted stream from panicking on overflow;
    // the result is nonsense pixels, which is the correct outcome for
    // nonsense input.
    *predictor = predictor.wrapping_add(difference);
    if let (Some(slot), Some(&step)) = (out.first_mut(), quant.first()) {
        *slot = predictor.saturating_mul(i32::from(step));
    }

    let mut index = 1_usize;
    while index < 64 {
        let symbol = reader.decode(ac)?;
        let (run, size) = ((symbol >> 4) as usize, u32::from(symbol & 0x0F));
        if size == 0 {
            if run != 15 {
                // End of block: every remaining coefficient is zero, which is
                // where JPEG gets most of its compression.
                break;
            }
            // A run of sixteen zeros, coded because the run length field
            // cannot express more than fifteen.
            index += 16;
            continue;
        }
        index += run;
        if index > 63 {
            return Err(PixelsError::malformed(
                "jpeg",
                "coefficient run passes the end of the block",
            ));
        }
        let value = reader.receive_extend(size)?;
        let step = quant.get(index).copied().unwrap_or(0);
        if let Some(slot) = ZIGZAG.get(index).and_then(|&at| out.get_mut(at)) {
            *slot = value.saturating_mul(i32::from(step));
        }
        index += 1;
    }
    Ok(())
}

/// Convert one YCbCr triple to RGB in fixed point.
///
/// The coefficients are JFIF's, scaled by 2^16. Fixed point rather than float
/// for ADR-0011's reason: the same input has to produce the same byte on every
/// target.
pub(crate) fn ycbcr_to_rgb([y, cb, cr]: [u8; 3]) -> [u8; 3] {
    const HALF: i32 = 1 << 15;
    let luma = i32::from(y) << 16;
    let blue = i32::from(cb) - 128;
    let red = i32::from(cr) - 128;

    let r = (luma + 91_881 * red + HALF) >> 16;
    let g = (luma - 22_554 * blue - 46_802 * red + HALF) >> 16;
    let b = (luma + 116_130 * blue + HALF) >> 16;
    [
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    ]
}

/// Decide what the frame's components mean.
fn colour_model(frame: &Frame, adobe: Option<AdobeTransform>) -> Result<Colour> {
    match frame.components.len() {
        1 => Ok(Colour::Grayscale),
        3 => {
            // Component ids 'R', 'G', 'B' are the other way a JPEG says it is
            // not YCbCr, and predate Adobe's marker.
            let labelled_rgb = frame.components.iter().map(|c| c.id).eq([b'R', b'G', b'B']);
            Ok(if adobe == Some(AdobeTransform::None) || labelled_rgb {
                Colour::Rgb
            } else {
                Colour::YCbCr
            })
        }
        // CMYK and YCCK need an ink model and usually an ICC profile to look
        // right; v1 is sRGB-assumed (SPEC §Pixel formats), so guessing here
        // would produce confidently wrong colour.
        count => Err(PixelsError::unsupported(format!(
            "jpeg: {count}-component images (CMYK/YCCK)"
        ))),
    }
}

/// Reject a scan header that is not the single full-spectrum scan baseline
/// requires.
fn check_baseline_scan(scan: &Scan) -> Result<()> {
    if scan.spectral_start != 0 || scan.spectral_end != 63 {
        return Err(PixelsError::malformed(
            "jpeg",
            format!(
                "baseline scan selects coefficients {}..={}; it must select all 64",
                scan.spectral_start, scan.spectral_end
            ),
        ));
    }
    if scan.approx_high != 0 || scan.approx_low != 0 {
        return Err(PixelsError::malformed(
            "jpeg",
            "baseline scan uses successive approximation, which is progressive only",
        ));
    }
    Ok(())
}

/// Read one `DQT` segment, which may define several tables.
fn read_quantization_tables(payload: &[u8], quant: &mut [[u16; 64]; 4]) -> Result<()> {
    let mut at = 0_usize;
    while at < payload.len() {
        let Some(&header) = payload.get(at) else {
            break;
        };
        at += 1;
        let (precision, slot) = (header >> 4, (header & 0x0F) as usize);
        if slot > 3 {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("DQT names table {slot}; only 0..=3 exist"),
            ));
        }
        let wide = match precision {
            0 => false,
            1 => true,
            other => {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("DQT declares precision {other}; only 0 and 1 exist"),
                ));
            }
        };

        let Some(table) = quant.get_mut(slot) else {
            break;
        };
        for entry in table.iter_mut() {
            let value = if wide {
                let (Some(&hi), Some(&lo)) = (payload.get(at), payload.get(at + 1)) else {
                    return Err(PixelsError::malformed("jpeg", "DQT segment is truncated"));
                };
                at += 2;
                u16::from_be_bytes([hi, lo])
            } else {
                let Some(&value) = payload.get(at) else {
                    return Err(PixelsError::malformed("jpeg", "DQT segment is truncated"));
                };
                at += 1;
                u16::from(value)
            };
            *entry = value;
        }
    }
    Ok(())
}

/// Read one `DHT` segment, which may define several tables.
fn read_huffman_tables(
    payload: &[u8],
    dc_tables: &mut [Option<HuffmanTable>; 4],
    ac_tables: &mut [Option<HuffmanTable>; 4],
) -> Result<()> {
    let mut at = 0_usize;
    while at < payload.len() {
        let Some(&header) = payload.get(at) else {
            break;
        };
        at += 1;
        let (class, slot) = (header >> 4, (header & 0x0F) as usize);
        if slot > 3 || class > 1 {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("DHT names class {class} table {slot}; classes are 0..=1, slots 0..=3"),
            ));
        }

        let Some(counts) = payload.get(at..at + 16) else {
            return Err(PixelsError::malformed(
                "jpeg",
                "DHT segment ends inside its code-length counts",
            ));
        };
        let mut lengths = [0_u8; 16];
        lengths.copy_from_slice(counts);
        at += 16;

        let total: usize = lengths.iter().map(|&c| c as usize).sum();
        let Some(values) = payload.get(at..at + total) else {
            return Err(PixelsError::malformed(
                "jpeg",
                "DHT segment ends inside its symbol list",
            ));
        };
        at += total;

        let table = HuffmanTable::new(&lengths, values.to_vec())?;
        let slots = if class == 0 {
            &mut *dc_tables
        } else {
            &mut *ac_tables
        };
        if let Some(entry) = slots.get_mut(slot) {
            *entry = Some(table);
        }
    }
    Ok(())
}

impl<S: Source + std::fmt::Debug> Decoder for JpegDecoder<S> {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
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

        if self.band_row >= self.band_height {
            self.fill_band()?;
        }
        let start = self.band_row as usize * row_bytes;
        let row = self
            .band
            .get(start..)
            .and_then(|rest| rest.get(..row_bytes))
            .ok_or_else(|| PixelsError::malformed("jpeg", "MCU row band is short"))?;
        out.copy_from_slice(row);
        self.band_row += 1;
        self.row += 1;
        Ok(())
    }
}

/// Whether `prefix` starts with a JPEG signature.
///
/// Detection is by magic bytes only (SPEC §Formats).
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    prefix.get(..3) == Some(&crate::format::SIGNATURE[..])
}

/// The JPEG entry in a sniffing registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct JpegCodec;

impl Codec for JpegCodec {
    fn format(&self) -> Format {
        Format::Jpeg
    }

    fn magic_len(&self) -> usize {
        3
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
    }
}
