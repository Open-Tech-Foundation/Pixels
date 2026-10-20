//! Parsing the codestream's headers (ISO/IEC 18181-1 §A, §C.2, §C.3): the
//! image header, frame headers and tables of contents. Field definitions as
//! jxl-oxide (MIT/Apache-2.0) states them; see NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "fixed-size headers and tables, indexed after their lengths are checked"
)]

use otf_pixels_core::Result;

use super::coding::{Decoder, read_permutation};
use super::reader::{BitReader, malformed, unpack_signed};
use crate::bits::{Dist, bits, val};

/// `U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))`.
const DIMENSION: [Dist; 4] = [bits(1, 9), bits(1, 13), bits(1, 18), bits(1, 30)];

/// A sample's depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitDepth {
    pub float: bool,
    pub bits: u32,
    pub exp_bits: u32,
}

impl Default for BitDepth {
    fn default() -> Self {
        Self {
            float: false,
            bits: 8,
            exp_bits: 0,
        }
    }
}

impl BitDepth {
    fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        if r.bool()? {
            let bits = r.u32([val(32), val(16), val(24), bits(1, 6)])?;
            let exp_bits = r.read(4)? + 1;
            Ok(Self {
                float: true,
                bits,
                exp_bits,
            })
        } else {
            let bits = r.u32([val(8), val(10), val(12), bits(1, 6)])?;
            if bits > 31 {
                return Err(malformed("too many bits per sample"));
            }
            Ok(Self {
                float: false,
                bits,
                exp_bits: 0,
            })
        }
    }
}

/// An extra channel's kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtraKind {
    Alpha { premultiplied: bool },
    Depth,
    SpotColour([f32; 4]),
    SelectionMask,
    Black,
    Cfa,
    Thermal,
    Other,
}

/// An extra channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtraChannel {
    pub kind: ExtraKind,
    pub bit_depth: BitDepth,
    pub dim_shift: u32,
}

impl ExtraChannel {
    fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        if r.bool()? {
            return Ok(Self {
                kind: ExtraKind::Alpha {
                    premultiplied: false,
                },
                bit_depth: BitDepth::default(),
                dim_shift: 0,
            });
        }
        let kind = r.enumeration()?;
        let bit_depth = BitDepth::parse(r)?;
        let dim_shift = r.u32([val(0), val(3), val(4), bits(1, 3)])?;
        skip_name(r)?;
        let kind = match kind {
            0 => ExtraKind::Alpha {
                premultiplied: r.bool()?,
            },
            1 => ExtraKind::Depth,
            2 => ExtraKind::SpotColour([r.f16()?, r.f16()?, r.f16()?, r.f16()?]),
            3 => ExtraKind::SelectionMask,
            4 => ExtraKind::Black,
            5 => {
                r.u32([val(1), bits(0, 2), bits(3, 4), bits(19, 8)])?;
                ExtraKind::Cfa
            }
            6 => ExtraKind::Thermal,
            15 | 16 => ExtraKind::Other,
            _ => return Err(malformed("unknown extra channel type")),
        };
        Ok(Self {
            kind,
            bit_depth,
            dim_shift,
        })
    }
}

/// Skip a `Name` bundle.
fn skip_name(r: &mut BitReader<'_>) -> Result<()> {
    let len = r.u32([val(0), bits(0, 4), bits(16, 5), bits(48, 10)])?;
    r.skip(len as usize * 8)
}

/// Skip an `Extensions` bundle.
fn skip_extensions(r: &mut BitReader<'_>) -> Result<()> {
    let flags = r.u64()?;
    let mut total = 0_u64;
    for i in 0..64 {
        if flags >> i & 1 != 0 {
            total = total.saturating_add(r.u64()?);
        }
    }
    r.skip(usize::try_from(total).map_err(|_| malformed("an extension is too large"))?)
}

/// A custom chromaticity, `(x, y)` scaled by a million.
fn customxy(r: &mut BitReader<'_>) -> Result<[f32; 2]> {
    let d = [
        bits(0, 19),
        bits(524_288, 19),
        bits(1_048_576, 20),
        bits(2_097_152, 21),
    ];
    Ok([
        unpack_signed(r.u32(d)?) as f32 / 1e6,
        unpack_signed(r.u32(d)?) as f32 / 1e6,
    ])
}

/// Transfer functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Transfer {
    Gamma(f32),
    Bt709,
    Unknown,
    Linear,
    Srgb,
    Pq,
    Dci,
    Hlg,
}

/// How samples' colours are encoded, when not by an ICC profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColourEncoding {
    pub want_icc: bool,
    /// 0 RGB, 1 grey, 2 XYB, 3 unknown.
    pub colour_space: u32,
    pub white: [f32; 2],
    /// Red, green and blue chromaticities.
    pub primaries: [[f32; 2]; 3],
    pub transfer: Transfer,
}

const D65: [f32; 2] = [0.3127, 0.3290];
const SRGB_PRIMARIES: [[f32; 2]; 3] = [
    [0.639_998_7, 0.330_010_14],
    [0.300_003_8, 0.600_003_36],
    [0.150_002_05, 0.059_997_204],
];

impl Default for ColourEncoding {
    fn default() -> Self {
        Self {
            want_icc: false,
            colour_space: 0,
            white: D65,
            primaries: SRGB_PRIMARIES,
            transfer: Transfer::Srgb,
        }
    }
}

impl ColourEncoding {
    fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        if r.bool()? {
            return Ok(Self::default());
        }
        let want_icc = r.bool()?;
        let colour_space = r.enumeration()?;
        if colour_space > 3 {
            return Err(malformed("unknown colour space"));
        }
        let mut out = Self {
            want_icc,
            colour_space,
            ..Self::default()
        };
        if want_icc {
            return Ok(out);
        }
        if colour_space != 2 {
            out.white = match r.enumeration()? {
                1 => D65,
                2 => customxy(r)?,
                10 => [1.0 / 3.0, 1.0 / 3.0],
                11 => [0.314, 0.351],
                _ => return Err(malformed("unknown white point")),
            };
        }
        if colour_space != 2 && colour_space != 1 {
            out.primaries = match r.enumeration()? {
                1 => SRGB_PRIMARIES,
                2 => [customxy(r)?, customxy(r)?, customxy(r)?],
                9 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
                11 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
                _ => return Err(malformed("unknown primaries")),
            };
        }
        out.transfer = if r.bool()? {
            Transfer::Gamma(r.read(24)? as f32 / 10_000_000.0)
        } else {
            match r.enumeration()? {
                1 => Transfer::Bt709,
                2 => Transfer::Unknown,
                8 => Transfer::Linear,
                13 => Transfer::Srgb,
                16 => Transfer::Pq,
                17 => Transfer::Dci,
                18 => Transfer::Hlg,
                _ => return Err(malformed("unknown transfer function")),
            }
        };
        if r.enumeration()? > 3 {
            return Err(malformed("unknown rendering intent"));
        }
        Ok(out)
    }
}

/// The XYB inverse transform's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpsinInverse {
    pub matrix: [[f32; 3]; 3],
    pub bias: [f32; 3],
    pub quant_bias: [f32; 3],
    pub quant_bias_numerator: f32,
}

impl Default for OpsinInverse {
    fn default() -> Self {
        Self {
            matrix: [
                [11.031_567, -9.866_944, -0.164_623],
                [-3.254_147_4, 4.418_770_3, -0.164_623],
                [-3.658_851_2, 2.712_923, 1.945_928_2],
            ],
            bias: [-0.003_793_073_3; 3],
            quant_bias: [1.0 - 0.054_650_073, 1.0 - 0.070_054_5, 1.0 - 0.049_935_103],
            quant_bias_numerator: 0.145,
        }
    }
}

/// Animation timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationHeader {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    pub num_loops: u32,
    pub have_timecodes: bool,
}

/// The image header.
#[derive(Debug, Clone)]
pub struct ImageHeader {
    pub width: u32,
    pub height: u32,
    pub orientation: u32,
    pub preview: Option<(u32, u32)>,
    pub animation: Option<AnimationHeader>,
    pub bit_depth: BitDepth,
    pub extra: Vec<ExtraChannel>,
    pub xyb: bool,
    pub colour: ColourEncoding,
    pub intensity_target: f32,
    pub opsin: OpsinInverse,
    pub up2: Vec<f32>,
    pub up4: Vec<f32>,
    pub up8: Vec<f32>,
}

/// `SizeHeader` (and `PreviewHeader` with its own distributions).
/// `small` is `None` for `SizeHeader`'s plain `1 + u(5)`.
fn size(r: &mut BitReader<'_>, small: Option<[Dist; 4]>, large: [Dist; 4]) -> Result<(u32, u32)> {
    let read_small = |r: &mut BitReader<'_>| -> Result<u32> {
        Ok(8 * match small {
            Some(d) => r.u32(d)?,
            None => r.read(5)? + 1,
        })
    };
    let div8 = r.bool()?;
    let height = if div8 { read_small(r)? } else { r.u32(large)? };
    let ratio = r.read(3)?;
    let width = if ratio == 0 {
        if div8 { read_small(r)? } else { r.u32(large)? }
    } else {
        let (n, d) =
            [(1, 1), (12, 10), (4, 3), (3, 2), (16, 9), (5, 4), (2, 1)][(ratio - 1) as usize];
        (u64::from(height) * n / d) as u32
    };
    Ok((width, height))
}

fn f16_array(r: &mut BitReader<'_>, n: usize) -> Result<Vec<f32>> {
    (0..n).map(|_| r.f16()).collect()
}

impl ImageHeader {
    /// Parse the signature and image header.
    ///
    /// # Errors
    ///
    /// For a malformed header.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        if r.read(16)? != 0x0AFF {
            return Err(malformed("not a JPEG XL codestream"));
        }
        let (width, height) = size(r, None, DIMENSION)?;
        let mut out = Self {
            width,
            height,
            orientation: 1,
            preview: None,
            animation: None,
            bit_depth: BitDepth::default(),
            extra: Vec::new(),
            xyb: true,
            colour: ColourEncoding::default(),
            intensity_target: 255.0,
            opsin: OpsinInverse::default(),
            up2: super::upsampling::D_UP2.to_vec(),
            up4: super::upsampling::D_UP4.to_vec(),
            up8: super::upsampling::D_UP8.to_vec(),
        };
        let all_default = r.bool()?;
        if !all_default {
            let extra_fields = r.bool()?;
            if extra_fields {
                out.orientation = r.read(3)? + 1;
                if r.bool()? {
                    size(r, None, DIMENSION)?;
                }
                if r.bool()? {
                    out.preview = Some(size(
                        r,
                        Some([val(16), val(32), bits(1, 5), bits(33, 9)]),
                        [bits(1, 6), bits(65, 8), bits(321, 10), bits(1345, 12)],
                    )?);
                }
                if r.bool()? {
                    out.animation = Some(AnimationHeader {
                        tps_numerator: r.u32([val(100), val(1000), bits(1, 10), bits(1, 30)])?,
                        tps_denominator: r.u32([val(1), val(1001), bits(1, 8), bits(1, 10)])?,
                        num_loops: r.u32([val(0), bits(0, 3), bits(0, 16), bits(0, 32)])?,
                        have_timecodes: r.bool()?,
                    });
                }
            }
            out.bit_depth = BitDepth::parse(r)?;
            r.bool()?; // modular_16bit_buffers
            let num_extra = r.u32([val(0), val(1), bits(2, 4), bits(1, 12)])?;
            if num_extra > 256 {
                return Err(malformed("too many extra channels"));
            }
            for _ in 0..num_extra {
                out.extra.push(ExtraChannel::parse(r)?);
            }
            out.xyb = r.bool()?;
            out.colour = ColourEncoding::parse(r)?;
            if extra_fields && !r.bool()? {
                out.intensity_target = r.f16()?;
                r.f16()?; // min_nits
                r.bool()?; // relative_to_max_display
                r.f16()?; // linear_below
            }
            skip_extensions(r)?;
        }
        // CustomTransformData.
        if !r.bool()? {
            if out.xyb && !r.bool()? {
                let m = f16_array(r, 9)?;
                for i in 0..3 {
                    for j in 0..3 {
                        out.opsin.matrix[i][j] = m[i * 3 + j];
                    }
                }
                let b = f16_array(r, 3)?;
                out.opsin.bias = [b[0], b[1], b[2]];
                let q = f16_array(r, 3)?;
                out.opsin.quant_bias = [q[0], q[1], q[2]];
                out.opsin.quant_bias_numerator = r.f16()?;
            }
            let cw_mask = r.read(3)?;
            if cw_mask & 1 != 0 {
                out.up2 = f16_array(r, 15)?;
            }
            if cw_mask & 2 != 0 {
                out.up4 = f16_array(r, 55)?;
            }
            if cw_mask & 4 != 0 {
                out.up8 = f16_array(r, 210)?;
            }
        }
        Ok(out)
    }

    /// Whether colour is grey.
    #[must_use]
    pub const fn grey(&self) -> bool {
        self.colour.colour_space == 1
    }

    /// The first alpha channel, if any.
    #[must_use]
    pub fn alpha(&self) -> Option<usize> {
        self.extra
            .iter()
            .position(|e| matches!(e.kind, ExtraKind::Alpha { .. }))
    }
}

/// Frame kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Regular,
    Lf,
    ReferenceOnly,
    SkipProgressive,
}

/// Blending modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Replace,
    Add,
    Blend,
    MulAdd,
    Mul,
}

/// How a frame blends onto the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendingInfo {
    pub mode: BlendMode,
    pub alpha_channel: u32,
    pub clamp: bool,
    pub source: u32,
}

/// The edge-preserving filter's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Epf {
    pub iters: u32,
    pub sharp_lut: [f32; 8],
    pub channel_scale: [f32; 3],
    pub quant_mul: f32,
    pub pass0_sigma_scale: f32,
    pub pass2_sigma_scale: f32,
    pub border_sad_mul: f32,
    pub sigma_for_modular: f32,
}

/// A frame header with its derived sizes.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub modular: bool,
    pub flags: u64,
    pub do_ycbcr: bool,
    pub jpeg_upsampling: [u32; 3],
    pub upsampling: u32,
    pub ec_upsampling: Vec<u32>,
    pub group_size_shift: u32,
    pub x_qm_scale: u32,
    pub b_qm_scale: u32,
    pub num_passes: u32,
    pub pass_shift: Vec<u32>,
    /// `(downsample, last_pass)` pairs.
    pub downsample: Vec<(u32, u32)>,
    pub lf_level: u32,
    pub x0: i32,
    pub y0: i32,
    pub width: u32,
    pub height: u32,
    pub blending: BlendingInfo,
    pub ec_blending: Vec<BlendingInfo>,
    pub duration: u32,
    pub is_last: bool,
    pub save_as_reference: u32,
    pub save_before_ct: bool,
    pub gaborish: Option<[[f32; 2]; 3]>,
    pub epf: Option<Epf>,
    /// How many channels carry colour: 1 for a grey Modular frame, else 3.
    pub colour_channels: usize,
}

impl FrameHeader {
    pub const NOISE: u64 = 0x1;
    pub const PATCHES: u64 = 0x2;
    pub const SPLINES: u64 = 0x10;
    pub const USE_LF_FRAME: u64 = 0x20;
    pub const SKIP_ADAPTIVE_LF_SMOOTHING: u64 = 0x80;

    /// Parse a frame header.
    ///
    /// # Errors
    ///
    /// For a malformed header.
    #[allow(
        clippy::too_many_lines,
        reason = "one field after another, as specified"
    )]
    pub fn parse(r: &mut BitReader<'_>, image: &ImageHeader) -> Result<Self> {
        let all_default = r.bool()?;
        let mut h = Self {
            frame_type: FrameType::Regular,
            modular: false,
            flags: 0,
            do_ycbcr: false,
            jpeg_upsampling: [0; 3],
            upsampling: 1,
            ec_upsampling: vec![1; image.extra.len()],
            group_size_shift: 1,
            x_qm_scale: if image.xyb { 3 } else { 2 },
            b_qm_scale: 2,
            num_passes: 1,
            pass_shift: Vec::new(),
            downsample: Vec::new(),
            lf_level: 0,
            x0: 0,
            y0: 0,
            width: image.width,
            height: image.height,
            blending: BlendingInfo {
                mode: BlendMode::Replace,
                alpha_channel: 0,
                clamp: false,
                source: 0,
            },
            ec_blending: Vec::new(),
            duration: 0,
            is_last: true,
            save_as_reference: 0,
            save_before_ct: false,
            gaborish: Some([[0.115_169_525, 0.061_248_592]; 3]),
            epf: Some(Epf {
                iters: 2,
                sharp_lut: std::array::from_fn(|i| i as f32 / 7.0),
                channel_scale: [40.0, 5.0, 3.5],
                quant_mul: 0.46,
                pass0_sigma_scale: 0.9,
                pass2_sigma_scale: 6.5,
                border_sad_mul: 2.0 / 3.0,
                sigma_for_modular: 1.0,
            }),
            colour_channels: 3,
        };
        h.ec_blending = vec![h.blending; image.extra.len()];
        if all_default {
            h.colour_channels = 3;
            return Ok(h);
        }
        h.frame_type = match r.read(2)? {
            0 => FrameType::Regular,
            1 => FrameType::Lf,
            2 => FrameType::ReferenceOnly,
            _ => FrameType::SkipProgressive,
        };
        h.modular = r.read(1)? == 1;
        h.flags = r.u64()?;
        if !image.xyb {
            h.do_ycbcr = r.bool()?;
        }
        let use_lf_frame = h.flags & Self::USE_LF_FRAME != 0;
        if h.do_ycbcr && !use_lf_frame {
            for u in &mut h.jpeg_upsampling {
                *u = r.read(2)?;
            }
        }
        if !use_lf_frame {
            h.upsampling = r.u32([val(1), val(2), val(4), val(8)])?;
            for u in &mut h.ec_upsampling {
                *u = r.u32([val(1), val(2), val(4), val(8)])?;
            }
        }
        if h.modular {
            h.group_size_shift = r.read(2)?;
        }
        if image.xyb && !h.modular {
            h.x_qm_scale = r.read(3)?;
            h.b_qm_scale = r.read(3)?;
        } else if !h.modular {
            h.x_qm_scale = 2;
        }
        if h.frame_type != FrameType::ReferenceOnly {
            h.num_passes = r.u32([val(1), val(2), val(3), bits(4, 3)])?;
            if h.num_passes != 1 {
                let num_ds = r.u32([val(0), val(1), val(2), bits(3, 1)])?;
                h.pass_shift = (0..h.num_passes - 1)
                    .map(|_| r.read(2))
                    .collect::<Result<_>>()?;
                let downsample = (0..num_ds)
                    .map(|_| r.u32([val(1), val(2), val(4), val(8)]))
                    .collect::<Result<Vec<_>>>()?;
                let last_pass = (0..num_ds)
                    .map(|_| r.u32([val(0), val(1), val(2), bits(0, 3)]))
                    .collect::<Result<Vec<_>>>()?;
                h.downsample = downsample.into_iter().zip(last_pass).collect();
            }
        }
        if h.frame_type == FrameType::Lf {
            h.lf_level = r.read(2)? + 1;
        }
        let mut have_crop = false;
        if h.frame_type != FrameType::Lf {
            have_crop = r.bool()?;
        }
        if have_crop {
            let pos = [bits(0, 8), bits(256, 11), bits(2304, 14), bits(18688, 30)];
            if h.frame_type != FrameType::ReferenceOnly {
                h.x0 = unpack_signed(r.u32(pos)?);
                h.y0 = unpack_signed(r.u32(pos)?);
            }
            h.width = r.u32(pos)?;
            h.height = r.u32(pos)?;
        }
        let normal = matches!(
            h.frame_type,
            FrameType::Regular | FrameType::SkipProgressive
        );
        let full = !have_crop
            || (h.x0 <= 0
                && h.y0 <= 0
                && i64::from(h.x0) + i64::from(h.width) >= i64::from(image.width)
                && i64::from(h.y0) + i64::from(h.height) >= i64::from(image.height));
        if normal {
            let has_extra = !image.extra.is_empty();
            let blending =
                |r: &mut BitReader<'_>, main: Option<BlendMode>| -> Result<BlendingInfo> {
                    let mode = match r.u32([val(0), val(1), val(2), bits(3, 2)])? {
                        0 => BlendMode::Replace,
                        1 => BlendMode::Add,
                        2 => BlendMode::Blend,
                        3 => BlendMode::MulAdd,
                        4 => BlendMode::Mul,
                        _ => return Err(malformed("unknown blend mode")),
                    };
                    let uses_alpha = matches!(mode, BlendMode::Blend | BlendMode::MulAdd);
                    let alpha_channel = if has_extra && uses_alpha {
                        r.u32([val(0), val(1), val(2), bits(3, 3)])?
                    } else {
                        0
                    };
                    let clamp = if (has_extra && uses_alpha) || mode == BlendMode::Mul {
                        r.bool()?
                    } else {
                        false
                    };
                    let resets = main.unwrap_or(mode) == BlendMode::Replace && full;
                    let source = if resets { 0 } else { r.read(2)? };
                    Ok(BlendingInfo {
                        mode,
                        alpha_channel,
                        clamp,
                        source,
                    })
                };
            h.blending = blending(r, None)?;
            let main = h.blending.mode;
            h.ec_blending = (0..image.extra.len())
                .map(|_| blending(r, Some(main)))
                .collect::<Result<_>>()?;
            if image.animation.is_some() {
                h.duration = r.u32([val(0), val(1), bits(0, 8), bits(0, 32)])?;
                if image.animation.is_some_and(|a| a.have_timecodes) {
                    r.read(32)?;
                }
            }
            h.is_last = r.bool()?;
        } else {
            h.is_last = false;
        }
        if h.frame_type != FrameType::Lf && !h.is_last {
            h.save_as_reference = r.read(2)?;
        }
        let resets = h.blending.mode == BlendMode::Replace && full;
        if h.frame_type == FrameType::ReferenceOnly
            || (resets
                && !h.is_last
                && (h.duration == 0 || h.save_as_reference != 0)
                && h.frame_type != FrameType::Lf)
        {
            h.save_before_ct = r.bool()?;
        } else {
            h.save_before_ct = !normal;
        }
        skip_name(r)?;
        // Restoration filter.
        if !r.bool()? {
            h.gaborish = if r.bool()? {
                if r.bool()? {
                    let mut w = [[0.0; 2]; 3];
                    for c in &mut w {
                        c[0] = r.f16()?;
                        c[1] = r.f16()?;
                    }
                    Some(w)
                } else {
                    Some([[0.115_169_525, 0.061_248_592]; 3])
                }
            } else {
                None
            };
            let iters = r.read(2)?;
            h.epf = if iters == 0 {
                None
            } else {
                let mut epf = Epf {
                    iters,
                    sharp_lut: std::array::from_fn(|i| i as f32 / 7.0),
                    channel_scale: [40.0, 5.0, 3.5],
                    quant_mul: 0.46,
                    pass0_sigma_scale: 0.9,
                    pass2_sigma_scale: 6.5,
                    border_sad_mul: 2.0 / 3.0,
                    sigma_for_modular: 1.0,
                };
                if !h.modular && r.bool()? {
                    for v in &mut epf.sharp_lut {
                        *v = r.f16()?;
                    }
                }
                if r.bool()? {
                    for v in &mut epf.channel_scale {
                        *v = r.f16()?;
                    }
                    r.read(32)?;
                }
                if r.bool()? {
                    if !h.modular {
                        epf.quant_mul = r.f16()?;
                    }
                    epf.pass0_sigma_scale = r.f16()?;
                    epf.pass2_sigma_scale = r.f16()?;
                    epf.border_sad_mul = r.f16()?;
                }
                if h.modular {
                    epf.sigma_for_modular = r.f16()?;
                }
                Some(epf)
            };
            skip_extensions(r)?;
        }
        skip_extensions(r)?;
        h.colour_channels = if h.modular && !h.do_ycbcr && !image.xyb && image.grey() {
            1
        } else {
            3
        };
        Ok(h)
    }

    /// The colour planes' size in samples, after upsampling and LF level
    /// reduce it.
    #[must_use]
    pub fn sample_size(&self, upsampling: u32) -> (u32, u32) {
        let (mut w, mut h) = (self.width, self.height);
        if upsampling > 1 {
            w = w.div_ceil(upsampling);
            h = h.div_ceil(upsampling);
        }
        if self.lf_level > 0 {
            let div = 1 << (3 * self.lf_level);
            w = w.div_ceil(div);
            h = h.div_ceil(div);
        }
        (w, h)
    }

    /// The colour sample size.
    #[must_use]
    pub fn colour_size(&self) -> (u32, u32) {
        self.sample_size(self.upsampling)
    }

    #[must_use]
    pub const fn group_dim(&self) -> u32 {
        128 << self.group_size_shift
    }

    #[must_use]
    pub fn groups(&self) -> (u32, u32) {
        let (w, h) = self.colour_size();
        (w.div_ceil(self.group_dim()), h.div_ceil(self.group_dim()))
    }

    #[must_use]
    pub fn lf_groups(&self) -> (u32, u32) {
        let (w, h) = self.colour_size();
        let d = self.group_dim() * 8;
        (w.div_ceil(d), h.div_ceil(d))
    }

    #[must_use]
    pub fn num_groups(&self) -> u32 {
        let (x, y) = self.groups();
        x.saturating_mul(y)
    }

    #[must_use]
    pub fn num_lf_groups(&self) -> u32 {
        let (x, y) = self.lf_groups();
        x.saturating_mul(y)
    }

    /// Each pass's range of Modular channel shifts, `[min, max)`.
    #[must_use]
    pub fn pass_shifts(&self) -> Vec<(i32, i32)> {
        let mut out = vec![(0, 0); self.num_passes as usize];
        let mut max = 3;
        for &(downsample, last) in &self.downsample {
            let min = downsample.trailing_zeros() as i32;
            if let Some(slot) = out.get_mut(last as usize) {
                *slot = (min, max);
            }
            max = min;
        }
        if let Some(last) = out.last_mut() {
            *last = (0, max);
        }
        out
    }

    /// Whether this frame is shown.
    #[must_use]
    pub fn is_keyframe(&self) -> bool {
        matches!(
            self.frame_type,
            FrameType::Regular | FrameType::SkipProgressive
        ) && (self.is_last || self.duration != 0)
    }

    /// Whether this frame can be referenced later.
    #[must_use]
    pub fn can_reference(&self) -> bool {
        !self.is_last
            && (self.duration == 0 || self.save_as_reference != 0)
            && self.frame_type != FrameType::Lf
    }
}

/// A frame's sections, by byte range in the frame data.
#[derive(Debug, Clone)]
pub struct Toc {
    /// Section ranges in the order the sections are numbered.
    pub sections: Vec<(usize, usize)>,
    /// The bytes all sections take.
    pub total: usize,
    pub single: bool,
}

impl Toc {
    /// Parse a frame's table of contents.
    ///
    /// # Errors
    ///
    /// For a malformed table.
    pub fn parse(r: &mut BitReader<'_>, h: &FrameHeader) -> Result<Self> {
        let groups = h.num_groups();
        let entries = if groups == 1 && h.num_passes == 1 {
            1
        } else {
            2_u64 + u64::from(h.num_lf_groups()) + u64::from(h.num_passes) * u64::from(groups)
        };
        // Every entry's size takes at least 12 bits, so a table longer than
        // the bits left is a lie, refused before anything is sized by it.
        if entries.saturating_mul(12) > r.remaining() as u64 {
            return Err(malformed("a table of contents runs past the codestream"));
        }
        let entries = u32::try_from(entries).map_err(|_| malformed("too many sections"))?;
        let permutation = if r.bool()? {
            let mut d = Decoder::parse(r, 8)?;
            d.begin(r)?;
            let p = read_permutation(r, &mut d, entries, 0)?;
            d.finish()?;
            Some(p)
        } else {
            None
        };
        r.zero_pad_to_byte()?;
        let sizes = (0..entries)
            .map(|_| {
                r.u32([
                    bits(0, 10),
                    bits(1024, 14),
                    bits(17_408, 22),
                    bits(4_211_712, 30),
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        r.zero_pad_to_byte()?;
        // Sections are stored in bitstream order; section `i` is the one
        // stored at position `permutation[i]`.
        let mut stored = Vec::with_capacity(sizes.len());
        let mut at = 0_usize;
        for &size in &sizes {
            stored.push((at, size as usize));
            at += size as usize;
        }
        let ranges = match &permutation {
            Some(p) => p
                .iter()
                .map(|&i| stored.get(i).copied().unwrap_or((0, 0)))
                .collect(),
            None => stored,
        };
        Ok(Self {
            sections: ranges,
            total: at,
            single: entries == 1,
        })
    }
}
