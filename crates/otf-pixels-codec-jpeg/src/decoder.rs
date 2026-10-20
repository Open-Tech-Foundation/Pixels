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
//!
//! # Progressive frames
//!
//! A progressive frame (`SOF2`) is opened the same way, up to its first scan.
//! On the first row request every scan is decoded into a whole-image buffer of
//! coefficients (`progressive`), and bands are then built from that buffer
//! through the same inverse DCT, upsampling and colour conversion — so a
//! progressive file decodes at reduced scale too.

use crate::entropy::Reader;
use crate::format::{
    AdobeTransform, Frame, IccChunks, Scan, ZIGZAG, adobe_transform, exif_orientation, marker,
};
use crate::huffman::HuffmanTable;
use crate::idct::{self, Scale};
use crate::progressive::{Coefficients, MAX_SCANS, Tables};
use otf_pixels_core::{
    Codec, Comment, DecodeCapability, Decoder, Density, Format, ImageDescriptor, Limits,
    Orientation, PixelFormat, PixelsError, ResolutionUnit, Result, Source, SourceInfo,
};

/// The identifier an XMP `APP1` segment starts with (XMP part 3, §1.1.3).
const XMP_ID: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";

/// The density a JFIF `APP0` segment states: units, then X and Y.
fn jfif_density(payload: &[u8]) -> Option<Density> {
    let rest = payload.strip_prefix(b"JFIF\0")?;
    let (&unit, rest) = rest.get(2..)?.split_first()?;
    let x = u16::from_be_bytes([*rest.first()?, *rest.get(1)?]);
    let y = u16::from_be_bytes([*rest.get(2)?, *rest.get(3)?]);
    let unit = match unit {
        1 => ResolutionUnit::Inch,
        2 => ResolutionUnit::Centimetre,
        // Aspect ratio only: no density.
        _ => return None,
    };
    Some(Density {
        x: f64::from(x),
        y: f64::from(y),
        unit,
    })
}

/// sharp's spelling of a frame's chroma subsampling, from the first
/// component's sampling factors against the others'.
fn chroma_name(frame: &Frame) -> Option<&'static str> {
    let [luma, chroma, ..] = &frame.components[..] else {
        return None;
    };
    let (h, v) = (luma.h / chroma.h.max(1), luma.v / chroma.v.max(1));
    let base = match (h, v) {
        (1, 1) => "4:4:4",
        (2, 1) => "4:2:2",
        (2, 2) => "4:2:0",
        (1, 2) => "4:4:0",
        (4, 1) => "4:1:1",
        (4, 2) => "4:1:0",
        _ => return None,
    };
    Some(if frame.components.len() == 4 {
        match base {
            "4:4:4" => "4:4:4:4",
            "4:2:2" => "4:2:2:4",
            "4:2:0" => "4:2:0:4",
            "4:4:0" => "4:4:0:4",
            "4:1:1" => "4:1:1:4",
            _ => "4:1:0:4",
        }
    } else {
        base
    })
}

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

/// Decodes a JPEG stream.
///
/// Baseline frames stream one MCU row at a time. Progressive frames decode
/// their scans into a coefficient buffer first and are then served the same
/// way; the distinction is not visible through [`Decoder`].
#[derive(Debug)]
pub struct JpegDecoder<S: Source> {
    inner: Box<Engine<S>>,
    /// The ICC profile from the `APP2` segments before the scan.
    icc: Option<Vec<u8>>,
    /// The EXIF block from the first EXIF `APP1` segment.
    exif: Option<Vec<u8>>,
    /// The XMP packet from the first XMP `APP1` segment.
    xmp: Option<Vec<u8>>,
    /// Density, subsampling and the rest of what the header says.
    info: SourceInfo,
}

impl<S: Source> JpegDecoder<S> {
    /// Read every header up to and including the scan header.
    ///
    /// No coefficient is decoded here, baseline or progressive, so this is the
    /// `probe()`/metadata path as well as the start of a decode (SPEC
    /// §Guarantees 3).
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a stream that is not a
    /// well-formed JPEG, [`PixelsError::Unsupported`] for a variant this crate
    /// does not handle (arithmetic coding, lossless, 12-bit, CMYK), or
    /// [`PixelsError::LimitExceeded`] if the frame exceeds `limits`.
    pub fn new(source: S, limits: Limits) -> Result<Self> {
        Self::with_scale(source, limits, Scale::Full)
    }

    /// Read every header, and decode at a reduced resolution.
    ///
    /// At `M/8` scale the decoder inverse-transforms only the low-frequency
    /// corner of each block, so a thumbnail costs a fraction of the arithmetic
    /// and — the point of it — the full-resolution image is never
    /// materialized. Entropy decoding is unchanged: every coefficient is still
    /// read, because the format gives no way to skip one.
    ///
    /// [`Decoder::descriptor`] reports the *scaled* size, so a caller that
    /// asks for a reduced decode is told what it will actually receive.
    ///
    /// # Errors
    ///
    /// As [`JpegDecoder::new`].
    pub fn with_scale(source: S, limits: Limits, scale: Scale) -> Result<Self> {
        let mut inner = Box::new(Engine::with_scale(source, limits, scale)?);
        let icc = inner.icc.take();
        let exif = inner.exif.take();
        let xmp = inner.xmp.take();
        let info = std::mem::take(&mut inner.info);
        Ok(Self {
            inner,
            icc,
            exif,
            xmp,
            info,
        })
    }

    /// The resolution this decoder produces, as eighths of full size.
    #[must_use]
    pub const fn scale(&self) -> Scale {
        self.inner.scale
    }

    /// Whether this stream is a progressive frame.
    #[must_use]
    pub const fn is_progressive(&self) -> bool {
        self.inner.progressive
    }
}

impl<S: Source + std::fmt::Debug> Decoder for JpegDecoder<S> {
    fn descriptor(&self) -> ImageDescriptor {
        self.inner.descriptor()
    }

    /// The ICC profile, joined from its `APP2` segments.
    fn icc_profile(&self) -> Option<&[u8]> {
        self.icc.as_deref()
    }

    /// The EXIF block, from the first EXIF `APP1` segment.
    fn exif(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }

    /// The XMP packet, from the first XMP `APP1` segment. The extended XMP
    /// a packet over 64 KB spills into is not joined.
    fn xmp(&self) -> Option<&[u8]> {
        self.xmp.as_deref()
    }

    /// Progressive, chroma subsampling, density (JFIF, else EXIF) and
    /// `COM` comments before the first scan.
    fn source_info(&self) -> SourceInfo {
        self.info.clone()
    }

    /// The EXIF orientation, from the first EXIF `APP1` segment.
    fn orientation(&self) -> Orientation {
        self.inner.orientation.unwrap_or_default()
    }

    fn capability(&self) -> DecodeCapability {
        self.inner.capability()
    }

    fn reduced_descriptor(&self, target: (u32, u32)) -> Option<ImageDescriptor> {
        self.inner.reduced_descriptor(target)
    }

    fn reduce_to(&mut self, descriptor: ImageDescriptor) -> Result<()> {
        self.inner.reduce_to(descriptor)
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        self.inner.read_row(out)
    }
}

/// The decoder proper, for baseline and progressive frames alike.
struct Engine<S: Source> {
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
    orientation: Option<Orientation>,
    /// The ICC profile, until [`JpegDecoder`] takes it.
    icc: Option<Vec<u8>>,
    /// The EXIF block, until [`JpegDecoder`] takes it.
    exif: Option<Vec<u8>>,
    /// The XMP packet, until [`JpegDecoder`] takes it.
    xmp: Option<Vec<u8>>,
    /// What the header says, until [`JpegDecoder`] takes it.
    info: SourceInfo,
    planes: Vec<Plane>,
    /// One output-width row per component, reused for every row: the
    /// component's samples at output resolution, ready for colour conversion.
    expanded: Vec<Vec<u8>>,
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
    /// Kept so a later re-scale can be re-checked against the same bounds.
    limits: Limits,
    /// Output rows already served.
    row: u32,
    /// Whether the frame is progressive (`SOF2`).
    progressive: bool,
    /// A progressive frame's coefficients, once its scans are decoded.
    coefficients: Option<Coefficients>,
    /// Each component's quantization table, as it stood at the component's
    /// first scan; a progressive file may redefine a slot between scans.
    latched: Vec<Option<[u16; 64]>>,
}

impl<S: Source> std::fmt::Debug for Engine<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
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

impl<S: Source> Engine<S> {
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
    fn with_scale(source: S, limits: Limits, scale: Scale) -> Result<Self> {
        let mut reader = Reader::new(source);
        let mut progressive = false;
        let mut quant = [[0_u16; 64]; 4];
        let mut dc_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut ac_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut restart_interval = 0_u16;
        let mut orientation = None;
        let mut exif: Option<Vec<u8>> = None;
        let mut xmp: Option<Vec<u8>> = None;
        let mut jfif = None;
        let mut comments = Vec::new();
        let mut icc = IccChunks::default();
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
                marker::SOF0 | marker::SOF1 | marker::SOF2 => {
                    if frame.is_some() {
                        return Err(PixelsError::malformed(
                            "jpeg",
                            "stream declares more than one frame",
                        ));
                    }
                    progressive = code == marker::SOF2;
                    let parsed = Frame::parse(&reader.read_segment()?)?;
                    // Checked here rather than once the scan header arrives,
                    // so an enormous frame costs one segment parse and stops
                    // (SPEC §Safety and limits).
                    limits.check(u32::from(parsed.width), u32::from(parsed.height))?;
                    frame = Some(parsed);
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
                    if exif.is_none() {
                        exif = payload.strip_prefix(b"Exif\0\0").map(<[u8]>::to_vec);
                    }
                    if xmp.is_none() {
                        xmp = payload.strip_prefix(XMP_ID).map(<[u8]>::to_vec);
                    }
                }
                marker::APP0 => {
                    let payload = reader.read_segment()?;
                    jfif = jfif.or_else(|| jfif_density(&payload));
                }
                marker::COM => {
                    let payload = reader.read_segment()?;
                    let text = String::from_utf8(payload.clone())
                        .unwrap_or_else(|_| payload.iter().map(|&c| char::from(c)).collect());
                    comments.push(Comment {
                        keyword: String::new(),
                        text: text.trim_end_matches('\0').to_owned(),
                    });
                }
                marker::APP2 => icc.push(&reader.read_segment()?),
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
                // APP3 to APP13 and APP15; the others are read above.
                0xE3..=0xED | marker::APP15 => reader.skip_segment()?,
                code if marker::is_standalone(code) => {}
                _ => reader.skip_segment()?,
            }
        };

        let Some(frame) = frame else {
            return Err(PixelsError::malformed("jpeg", "stream has no frame header"));
        };
        // A progressive scan is checked when it is decoded, against the
        // tables as they stand by then; a baseline one must be complete now.
        if !progressive {
            check_baseline_scan(&scan)?;
        }

        let mut info = SourceInfo::default();
        info.density = jfif.or_else(|| {
            exif.as_deref()
                .and_then(otf_pixels_core::Density::from_exif_block)
        });
        info.progressive = progressive;
        info.bits_per_sample = Some(frame.precision);
        info.chroma_subsampling = chroma_name(&frame);
        info.comments = comments;
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

        if !progressive && scan.components.len() != frame.components.len() {
            // Every component of a baseline frame is coded in one interleaved
            // scan in practice. The alternative — a scan per component — needs
            // whole-image component planes, which would trade the memory
            // guarantee for a case no encoder in circulation produces.
            return Err(PixelsError::unsupported(
                "jpeg: non-interleaved baseline scan",
            ));
        }
        for component in scan.components.iter().filter(|_| !progressive) {
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

        let mcus_per_line = full_width.div_ceil(h_max * 8);
        let mcu_rows = full_height.div_ceil(v_max * 8);

        // Geometry is set up by `apply_scale`, which the planner may call
        // again later: shrink-on-load decides the scale only once the whole
        // pipeline is known, by which point this decoder already exists.
        let mut decoder = Self {
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
            icc: icc.assemble(),
            exif,
            xmp,
            info,
            limits,
            planes: Vec::new(),
            expanded: Vec::new(),
            band: Vec::new(),
            band_row: 0,
            band_height: 0,
            mcus_per_line,
            mcu_rows,
            mcu_row: 0,
            scale: Scale::Full,
            row: 0,
            progressive,
            coefficients: None,
            latched: Vec::new(),
        };
        decoder.apply_scale(scale)?;
        Ok(decoder)
    }

    /// Set the resolution this decoder will produce, rebuilding the geometry
    /// that depends on it.
    ///
    /// Every plane dimension is in samples-per-block, so the scale is the only
    /// thing the rest of the decoder has to know about.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if any row has already been
    /// read — a scaled decode cannot start halfway — or
    /// [`PixelsError::LimitExceeded`] if the scaled size is outside `limits`.
    fn apply_scale(&mut self, scale: Scale) -> Result<()> {
        if self.row > 0 || self.mcu_row > 0 {
            return Err(PixelsError::invalid_argument(
                "scale",
                format!("{} rows have already been decoded", self.row),
            ));
        }
        let full_width = u32::from(self.frame.width);
        let full_height = u32::from(self.frame.height);
        let (h_max, v_max) = (u32::from(self.frame.h_max()), u32::from(self.frame.v_max()));
        let descriptor = ImageDescriptor::with_limits(
            scale.apply(full_width),
            scale.apply(full_height),
            self.descriptor.pixel,
            &self.limits,
        )?;

        // How many samples one block becomes: 8 at full resolution, fewer at a
        // reduced scale.
        let sample = scale.block_size();
        let band_height = v_max * sample;

        let mut planes = Vec::with_capacity(self.frame.components.len());
        for component in &self.frame.components {
            let (h, v) = (u32::from(component.h), u32::from(component.v));
            let stride = usize::try_from(self.mcus_per_line * h * sample)
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

        let band = descriptor
            .row_bytes()
            .checked_mul(band_height as usize)
            .ok_or_else(|| PixelsError::malformed("jpeg", "MCU row band overflows"))?;

        self.descriptor = descriptor;
        self.expanded = vec![vec![0_u8; descriptor.width as usize]; planes.len()];
        self.planes = planes;
        self.band = vec![0_u8; band];
        // Nothing decoded yet, so the band is spent and the first `read_row`
        // fills it.
        self.band_row = band_height;
        self.band_height = band_height;
        self.scale = scale;
        Ok(())
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

        if self.progressive {
            return self.fill_band_progressive();
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

    /// [`Engine::fill_band`] for a progressive frame: the band's blocks come
    /// from the coefficient buffer, decoded in full on the first call.
    fn fill_band_progressive(&mut self) -> Result<()> {
        if self.coefficients.is_none() {
            self.decode_scans()?;
        }
        let Some(stored) = self.coefficients.as_ref() else {
            return Err(PixelsError::graph(
                "jpeg coefficients vanished after decoding",
            ));
        };

        let mut natural = [0_i32; 64];
        let scale = self.scale;
        let sample = scale.block_size() as usize;
        let mcu_row = self.mcu_row as usize;
        for (index, component) in self.frame.components.iter().enumerate() {
            let (h, v) = (usize::from(component.h), usize::from(component.v));
            // A component no scan ever reached has all-zero coefficients, so
            // whichever table dequantizes them makes no difference.
            let quant = self
                .latched
                .get(index)
                .copied()
                .flatten()
                .or_else(|| self.quant.get(component.quant as usize).copied())
                .unwrap_or([0; 64]);
            let Some(plane) = self.planes.get_mut(index) else {
                continue;
            };
            for block_y in 0..v {
                for block_x in 0..h * self.mcus_per_line as usize {
                    let Some(block) = stored.block(index, block_x, mcu_row * v + block_y) else {
                        continue;
                    };
                    for ((&at, &value), &step) in ZIGZAG.iter().zip(block).zip(&quant) {
                        if let Some(slot) = natural.get_mut(at) {
                            *slot = i32::from(value) * i32::from(step);
                        }
                    }
                    idct::scaled_block(
                        &natural,
                        scale,
                        &mut plane.samples,
                        block_y * sample * plane.stride + block_x * sample,
                        plane.stride,
                    );
                }
            }
        }

        self.mcu_row += 1;
        self.convert_band();
        self.band_row = 0;
        Ok(())
    }

    /// Decode every scan of a progressive frame into the coefficient buffer.
    ///
    /// The first scan's header was read when the decoder was built. Tables
    /// and restart intervals may be redefined between scans and take effect
    /// for the scans after them; the frame ends at `EOI`, or where the data
    /// runs out, which leaves a partial download showing what arrived.
    fn decode_scans(&mut self) -> Result<()> {
        let mut coefficients = Coefficients::new(&self.frame, self.mcus_per_line, self.mcu_rows)?;
        let mut latched: Vec<Option<[u16; 64]>> = vec![None; self.frame.components.len()];
        let mut scan = self.scan.clone();
        let mut scans = 0;
        'scans: loop {
            scans += 1;
            if scans > MAX_SCANS {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("progressive frame has more than {MAX_SCANS} scans"),
                ));
            }
            // libjpeg's rule: a component's table is fixed by its first scan.
            for scanned in &scan.components {
                let Some(component) = self.frame.components.get(scanned.index) else {
                    continue;
                };
                if let Some(slot) = latched.get_mut(scanned.index) {
                    if slot.is_none() {
                        *slot = self.quant.get(component.quant as usize).copied();
                    }
                }
            }
            let tables = Tables {
                dc: &self.dc_tables,
                ac: &self.ac_tables,
                restart_interval: self.restart_interval,
            };
            crate::progressive::decode_scan(
                &mut self.reader,
                &self.frame,
                &scan,
                &tables,
                &mut coefficients,
            )?;
            self.reader.align();

            loop {
                // A stream that ends between scans is a partial download:
                // what arrived stands, as it does mid-scan.
                let Ok(code) = self.reader.next_marker() else {
                    break 'scans;
                };
                match code {
                    marker::SOS => {
                        scan = Scan::parse(&self.reader.read_segment()?, &self.frame)?;
                        continue 'scans;
                    }
                    marker::EOI => break 'scans,
                    marker::DHT => read_huffman_tables(
                        &self.reader.read_segment()?,
                        &mut self.dc_tables,
                        &mut self.ac_tables,
                    )?,
                    marker::DQT => {
                        read_quantization_tables(&self.reader.read_segment()?, &mut self.quant)?;
                    }
                    marker::DRI => {
                        let payload = self.reader.read_segment()?;
                        let (Some(&hi), Some(&lo)) = (payload.first(), payload.get(1)) else {
                            return Err(PixelsError::malformed(
                                "jpeg",
                                "DRI segment carries no interval",
                            ));
                        };
                        self.restart_interval = u16::from_be_bytes([hi, lo]);
                    }
                    code if marker::is_frame(code) => {
                        return Err(PixelsError::malformed(
                            "jpeg",
                            "stream declares more than one frame",
                        ));
                    }
                    // A stray restart marker between scans carries nothing.
                    code if marker::is_standalone(code) => {}
                    _ => self.reader.skip_segment()?,
                }
            }
        }
        self.coefficients = Some(coefficients);
        self.latched = latched;
        Ok(())
    }

    /// Upsample the component planes and write interleaved pixels into the
    /// output band.
    ///
    /// Each component's row is first expanded to output width in a buffer of
    /// its own, so colour conversion is one straight loop over equal-length
    /// slices — which the compiler vectorizes — rather than a per-pixel
    /// lookup into each plane.
    fn convert_band(&mut self) {
        let width = self.descriptor.width as usize;
        let row_bytes = self.descriptor.row_bytes();
        let h_max = usize::from(self.frame.h_max());

        for y in 0..self.band_height as usize {
            let Some(out) = self
                .band
                .get_mut(y * row_bytes..)
                .and_then(|rest| rest.get_mut(..row_bytes))
            else {
                continue;
            };
            for ((plane, component), expanded) in self
                .planes
                .iter()
                .zip(&self.frame.components)
                .zip(self.expanded.iter_mut())
            {
                // Whole ratios only; anything else takes the general path.
                let h = usize::from(component.h.max(1));
                let ratio = if h_max % h == 0 { h_max / h } else { 0 };
                plane.expand_row(y, ratio, expanded);
            }

            match self.colour {
                Colour::Grayscale => {
                    if let Some(grey) = self.expanded.first() {
                        for (slot, &value) in out.iter_mut().zip(grey).take(width) {
                            *slot = value;
                        }
                    }
                }
                Colour::Rgb => {
                    let [a, b, c] =
                        [0, 1, 2].map(|i| self.expanded.get(i).map_or(&[][..], Vec::as_slice));
                    for (pixel, ((&r, &g), &b)) in
                        out.chunks_exact_mut(3).zip(a.iter().zip(b).zip(c))
                    {
                        pixel.copy_from_slice(&[r, g, b]);
                    }
                }
                Colour::YCbCr => {
                    let [luma, cb, cr] =
                        [0, 1, 2].map(|i| self.expanded.get(i).map_or(&[][..], Vec::as_slice));
                    for (pixel, ((&y, &cb), &cr)) in
                        out.chunks_exact_mut(3).zip(luma.iter().zip(cb).zip(cr))
                    {
                        pixel.copy_from_slice(&ycbcr_to_rgb([y, cb, cr]));
                    }
                }
            }
        }
    }
}

impl Plane {
    /// Write this plane's samples for output row `y` of the band into
    /// `out`, one per output column. `ratio` is how many output columns one
    /// sample covers, or 0 if that is not a whole number; 1 and 2 — every
    /// common layout — take fast paths, and anything else goes through the
    /// general column table.
    fn expand_row(&self, y: usize, ratio: usize, out: &mut [u8]) {
        let source = self.row(y);
        match ratio {
            1 => {
                let count = out.len().min(source.len());
                if let (Some(to), Some(from)) = (out.get_mut(..count), source.get(..count)) {
                    to.copy_from_slice(from);
                }
            }
            2 => {
                // A fixed two-byte copy, not `fill`: on a slice that short,
                // `fill` is a `memset` call per pixel pair.
                let half = out.len() / 2;
                let mut pairs = out.chunks_exact_mut(2);
                for (pair, &value) in (&mut pairs).zip(source) {
                    pair.copy_from_slice(&[value, value]);
                }
                // An odd width leaves one column, covered by the last sample.
                if let ([last], Some(&value)) = (pairs.into_remainder(), source.get(half)) {
                    *last = value;
                }
            }
            _ => {
                for (x, slot) in out.iter_mut().enumerate() {
                    *slot = self.sample(source, x);
                }
            }
        }
    }

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
///
/// Each constant above 2^15 is split into a shift and a multiplier below it,
/// so every product is 16 by 16 bits: the baseline x86-64 target (SSE2) has
/// no 32-bit vector multiply but does have 16-bit ones, so this form
/// vectorizes and the direct one does not. The results are identical.
///
/// | Constant | As |
/// |---|---|
/// | 91 881 | `(c << 16) + 26 345 c` |
/// | 116 130 | `(c << 17) − 14 942 c` |
/// | 46 802 | `(c << 16) − 18 734 c` |
#[inline]
pub(crate) fn ycbcr_to_rgb([y, cb, cr]: [u8; 3]) -> [u8; 3] {
    const HALF: i32 = 1 << 15;
    let luma = (i32::from(y) << 16) + HALF;
    let blue = i32::from(i16::from(cb) - 128);
    let red = i32::from(i16::from(cr) - 128);

    let r = (luma + (red << 16) + 26_345 * red) >> 16;
    let g = (luma - 22_554 * blue - (red << 16) + 18_734 * red) >> 16;
    let b = (luma + (blue << 17) - 14_942 * blue) >> 16;
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
            let labelled_rgb = frame.components.iter().map(|c| c.id).eq(*b"RGB");
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

impl<S: Source + std::fmt::Debug> Decoder for Engine<S> {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        DecodeCapability::Sequential
    }

    fn reduced_descriptor(&self, target: (u32, u32)) -> Option<ImageDescriptor> {
        // Past the first row the geometry is settled.
        if self.row > 0 || self.mcu_row > 0 {
            return None;
        }
        let full = (u32::from(self.frame.width), u32::from(self.frame.height));
        let scale = Scale::fitting(full, target);
        if scale == self.scale {
            return None;
        }
        ImageDescriptor::new(
            scale.apply(full.0),
            scale.apply(full.1),
            self.descriptor.pixel,
        )
        .ok()
    }

    fn reduce_to(&mut self, descriptor: ImageDescriptor) -> Result<()> {
        let full = (u32::from(self.frame.width), u32::from(self.frame.height));
        // The descriptor has to be one this decoder could have offered, not an
        // arbitrary size: only the four M/8 scales are reachable without
        // resampling, which is the caller's job and not ours.
        let scale = Scale::ALL
            .into_iter()
            .find(|scale| {
                scale.apply(full.0) == descriptor.width && scale.apply(full.1) == descriptor.height
            })
            .ok_or_else(|| {
                PixelsError::invalid_argument(
                    "descriptor",
                    format!(
                        "{}x{} is not an M/8 scale of {}x{}",
                        descriptor.width, descriptor.height, full.0, full.1
                    ),
                )
            })?;
        self.apply_scale(scale)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The split-constant form must equal the direct JFIF formula for every
    /// one of the 16.7 million inputs, not merely for typical ones.
    #[test]
    fn colour_conversion_matches_the_direct_formula_everywhere() {
        fn direct([y, cb, cr]: [u8; 3]) -> [u8; 3] {
            const HALF: i32 = 1 << 15;
            let luma = i32::from(y) << 16;
            let blue = i32::from(cb) - 128;
            let red = i32::from(cr) - 128;
            let r = (luma + 91_881 * red + HALF) >> 16;
            let g = (luma - 22_554 * blue - 46_802 * red + HALF) >> 16;
            let b = (luma + 116_130 * blue + HALF) >> 16;
            [r, g, b].map(|v| v.clamp(0, 255) as u8)
        }
        for y in 0..=255 {
            for cb in 0..=255 {
                for cr in 0..=255 {
                    assert_eq!(ycbcr_to_rgb([y, cb, cr]), direct([y, cb, cr]));
                }
            }
        }
    }
}
