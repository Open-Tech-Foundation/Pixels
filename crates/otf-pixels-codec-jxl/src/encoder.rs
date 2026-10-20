//! The JPEG XL encoder: lossy VarDCT by default, lossless Modular on
//! request (and, until VarDCT carries it, for images with alpha).
//!
//! Lossless:
//!
//! The image becomes integer channels (grey or YCoCg colour, then alpha),
//! each predicted by the clamped gradient and coded per channel with prefix
//! codes fitted to its residuals. An image of one 256x256 group is a single
//! section; a larger one is split into groups, each its own section, behind
//! a table of contents, all sharing the codes of the global section.

use otf_pixels_core::{
    ChannelLayout, EncodeOptions, Encoder, ImageDescriptor, PixelsError, Result, SampleKind, Sink,
};

use crate::bits::BitWriter;
use crate::entropy::{EntropyCode, HybridUint, Token, with_runs};
use crate::headers::{ImageInfo, write_image_header, write_modular_frame_header, write_toc};
use crate::modular::{Channel, Tree, forward_ycocg, residuals, write_header};
use crate::predict::{Predictor, Region};

/// libjxl's cutoffs for the weighted predictor's error (`kWPFixedDC`).
const WP_CUTOFFS: [i32; 33] = [
    -500, -392, -255, -191, -127, -95, -63, -47, -31, -23, -15, -11, -7, -4, -3, -1, 0, 1, 3, 5, 7,
    11, 15, 23, 31, 47, 63, 95, 127, 191, 255, 392, 500,
];

/// The side of a group, the unit a large image is split into.
const GROUP: usize = 256;
/// The side of an LF group, which a Modular frame without squeeze leaves
/// empty.
const LF_GROUP: usize = GROUP * 8;
/// The largest dimension the size header can declare.
const MAX_DIMENSION: u32 = 1 << 30;

/// Encodes a JPEG XL codestream.
#[derive(Debug)]
pub struct JxlEncoder {
    state: Option<State>,
    /// `None` for lossless; otherwise the Butteraugli-style distance.
    distance: Option<f32>,
}

impl Default for JxlEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    pixels: Vec<u8>,
    rows_written: u32,
}

impl JxlEncoder {
    /// A lossy encoder at distance 1, visually lossless.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: None,
            distance: Some(1.0),
        }
    }

    /// A lossless encoder.
    #[must_use]
    pub const fn lossless() -> Self {
        Self {
            state: None,
            distance: None,
        }
    }

    /// A lossy encoder at `distance`: 1 is visually lossless, 0.5 finer,
    /// up to 25 for the coarsest.
    #[must_use]
    pub const fn with_distance(distance: f32) -> Self {
        Self {
            state: None,
            distance: Some(distance),
        }
    }

    /// An encoder configured from generic encode options: lossless if
    /// `lossless`, otherwise lossy at the distance libjxl's `cjxl` gives
    /// `quality` (90 is distance 1).
    #[must_use]
    pub fn from_options(options: &EncodeOptions) -> Self {
        if options.lossless {
            return Self::lossless();
        }
        Self::with_distance(distance_for_quality(options.quality))
    }
}

/// libjxl's quality-to-distance mapping.
fn distance_for_quality(quality: u8) -> f32 {
    let q = f32::from(quality.clamp(1, 100));
    if q >= 30.0 {
        0.1 + (100.0 - q) * 0.09
    } else {
        53.0 / 3000.0 * q * q - 23.0 / 20.0 * q + 25.0
    }
}

impl JxlEncoder {}

/// The header's view of `format`.
fn info_of(descriptor: &ImageDescriptor) -> Result<ImageInfo> {
    let format = descriptor.pixel;
    let bit_depth = match format.sample_kind() {
        SampleKind::U8 => 8,
        SampleKind::U16 => 16,
        _ => {
            return Err(PixelsError::unsupported(format!(
                "JPEG XL encoding takes 8- or 16-bit samples; got {format}. Convert first."
            )));
        }
    };
    Ok(ImageInfo {
        width: descriptor.width,
        height: descriptor.height,
        bit_depth,
        grey: matches!(
            format.layout(),
            ChannelLayout::Gray | ChannelLayout::GrayAlpha
        ),
        alpha: format.layout().has_alpha(),
        xyb: false,
    })
}

/// The image's channels: grey or R, G, B, then alpha.
fn channels_of(descriptor: &ImageDescriptor, pixels: &[u8]) -> Vec<Channel> {
    let format = descriptor.pixel;
    let (width, height) = (descriptor.width as usize, descriptor.height as usize);
    let count = format.channels();
    let wide = format.sample_kind() == SampleKind::U16;
    let mut channels = vec![Channel::new(width, height); count];
    let bytes = if wide { 2 } else { 1 };
    for (i, pixel) in pixels.chunks_exact(count * bytes).enumerate() {
        for (c, channel) in channels.iter_mut().enumerate() {
            let at = c * bytes;
            let value = if wide {
                i32::from(u16::from_ne_bytes([
                    pixel.get(at).copied().unwrap_or(0),
                    pixel.get(at + 1).copied().unwrap_or(0),
                ]))
            } else {
                i32::from(pixel.get(at).copied().unwrap_or(0))
            };
            if let Some(slot) = channel.samples.get_mut(i) {
                *slot = value;
            }
        }
    }
    channels
}

/// The whole lossy codestream for `pixels`, which have no alpha.
fn encode_lossy(descriptor: &ImageDescriptor, pixels: &[u8], distance: f32) -> Result<Vec<u8>> {
    let mut info = info_of(descriptor)?;
    let channels = channels_of(descriptor, pixels);
    let max = if info.bit_depth == 16 {
        65_535.0
    } else {
        255.0
    };
    let n = descriptor.width as usize * descriptor.height as usize;
    let rgb: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let at = |c: usize| {
                channels
                    .get(c)
                    .and_then(|ch| ch.samples.get(i))
                    .copied()
                    .unwrap_or(0) as f32
                    / max
            };
            if info.grey {
                [at(0); 3]
            } else {
                [at(0), at(1), at(2)]
            }
        })
        .collect();
    // XYB carries colour; the header's sRGB says what to decode it to, and
    // eight bits is what the samples were.
    info.grey = false;
    info.xyb = true;
    info.bit_depth = 8;
    Ok(crate::vardct::encode(&rgb, &info, distance))
}

/// The whole lossless codestream for `pixels`.
fn encode(descriptor: &ImageDescriptor, pixels: &[u8]) -> Result<Vec<u8>> {
    let info = info_of(descriptor)?;
    let mut channels = channels_of(descriptor, pixels);
    let rct = !info.grey;
    if rct {
        forward_ycocg(&mut channels);
    }
    let (width, height) = (info.width as usize, info.height as usize);
    // libjxl's `-e3` tree: the weighted predictor everywhere, contexts by
    // its error estimate (property 15).
    let pixels_total = width * height * channels.len();
    let tree = Tree::fixed(
        15,
        &WP_CUTOFFS,
        Predictor::Weighted,
        pixels_total,
        info.bit_depth,
    );
    let tree_tokens = tree.tokens();
    let tree_code = EntropyCode::build(6, &tree_tokens, HybridUint::BUCKETS);

    // Every group's residuals, in raster order of groups; a single group is
    // the whole image, coded in the global section.
    let groups_x = width.div_ceil(GROUP);
    let groups_y = height.div_ceil(GROUP);
    let single = groups_x * groups_y == 1;
    let mut group_tokens: Vec<Vec<Token>> = Vec::with_capacity(groups_x * groups_y);
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let (x0, y0) = (gx * GROUP, gy * GROUP);
            let region = (x0, y0, GROUP.min(width - x0), GROUP.min(height - y0));
            let mut tokens = Vec::with_capacity(region.2 * region.3 * channels.len());
            for (c, channel) in channels.iter().enumerate() {
                let region = Region {
                    channel,
                    x0: region.0,
                    y0: region.1,
                    width: region.2,
                    height: region.3,
                };
                residuals(&region, c as u32, &tree, &mut tokens);
            }
            // Runs of exact predictions (residual 0) become copies.
            group_tokens.push(with_runs(&tokens, 0));
        }
    }
    let all: Vec<Token> = group_tokens.iter().flatten().copied().collect();
    let code = EntropyCode::build(tree.leaves(), &all, HybridUint::DEFAULT);

    // The global section: dequantization defaults, the tree and codes, the
    // image's header and, for a single group, its residuals.
    let mut global = BitWriter::new();
    global.bool(true); // default LF dequantization
    global.bool(true); // a global tree follows
    tree_code.write_header(&mut global);
    for &token in &tree_tokens {
        tree_code.write(&mut global, token);
    }
    code.write_header(&mut global);
    write_header(&mut global, rct);
    let mut sections = Vec::new();
    if single {
        for &token in &all {
            code.write(&mut global, token);
        }
        sections.push(global.finish());
    } else {
        sections.push(global.finish());
        let lf_groups = width.div_ceil(LF_GROUP) * height.div_ceil(LF_GROUP);
        // LF groups and the HF global section hold nothing in a lossless
        // Modular frame.
        sections.extend(std::iter::repeat_n(Vec::new(), lf_groups + 1));
        for tokens in &group_tokens {
            let mut group = BitWriter::new();
            write_header(&mut group, false);
            for &token in tokens {
                code.write(&mut group, token);
            }
            sections.push(group.finish());
        }
    }

    let mut out = BitWriter::new();
    write_image_header(&mut out, &info);
    write_modular_frame_header(&mut out, &info);
    let sizes: Vec<usize> = sections.iter().map(Vec::len).collect();
    write_toc(&mut out, &sizes);
    let mut bytes = out.finish();
    for section in &sections {
        bytes.extend_from_slice(section);
    }
    Ok(bytes)
}

impl Encoder for JxlEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        info_of(desc)?;
        if desc.width == 0
            || desc.height == 0
            || desc.width > MAX_DIMENSION
            || desc.height > MAX_DIMENSION
        {
            return Err(PixelsError::unsupported(format!(
                "JPEG XL dimensions must be 1 to {MAX_DIMENSION}; {}x{} is not",
                desc.width, desc.height
            )));
        }
        let capacity = desc
            .byte_len()
            .ok_or_else(|| PixelsError::malformed("jxl", "image size overflows"))?;
        self.state = Some(State {
            descriptor: *desc,
            pixels: Vec::with_capacity(capacity),
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
        state.pixels.extend_from_slice(row);
        state.rows_written += 1;
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_ref() else {
            return Err(PixelsError::invalid_argument(
                "sink",
                "finish called before write_header",
            ));
        };
        if state.rows_written < state.descriptor.height {
            return Err(PixelsError::malformed(
                "jxl",
                format!(
                    "{} of {} rows were written",
                    state.rows_written, state.descriptor.height
                ),
            ));
        }
        let bytes = match self.distance {
            Some(distance) if !state.descriptor.pixel.layout().has_alpha() => {
                encode_lossy(&state.descriptor, &state.pixels, distance)?
            }
            _ => encode(&state.descriptor, &state.pixels)?,
        };
        sink.write_all(&bytes)?;
        sink.flush()
    }
}
