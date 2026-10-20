//! Decoding one frame's sections (ISO/IEC 18181-1 §C.4–§C.9, §G) into
//! floating-point planes: colour in the frame's own space (XYB, or the
//! image's colour space) and extra channels, before any filtering.

#![allow(
    clippy::indexing_slicing,
    reason = "section and plane indices come from the frame's own geometry"
)]

use otf_pixels_core::{PixelsError, Result};

use super::header::{FrameHeader, ImageHeader, Toc};
use super::modular::{GlobalImage, Plane, Shape, Tree};
use super::reader::{BitReader, malformed};

/// A decoded frame, before filters, upsampling or colour conversion.
#[derive(Debug, Clone)]
pub struct FramePlanes {
    /// Colour planes at the colour sample size: X, Y, B for an XYB image,
    /// else R, G, B (a grey image's one plane repeated).
    pub colour: [Vec<f32>; 3],
    pub width: usize,
    pub height: usize,
    /// Extra channels, each with its own size.
    pub extra: Vec<Plane32>,
    /// What the VarDCT filters need, for a VarDCT frame.
    pub vardct: Option<super::vardct::FilterInfo>,
}

/// A float plane.
#[derive(Debug, Clone)]
pub struct Plane32 {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

/// The parts of the frame's data in section order.
pub struct Sections<'a> {
    data: &'a [u8],
    toc: &'a Toc,
    shared: Option<BitReader<'a>>,
}

impl<'a> Sections<'a> {
    #[must_use]
    pub fn new(data: &'a [u8], toc: &'a Toc) -> Self {
        Self {
            data,
            toc,
            shared: toc.single.then(|| BitReader::new(data)),
        }
    }

    /// Run `f` on the reader for section `i`.
    ///
    /// # Errors
    ///
    /// For a section past the data, or whatever `f` returns.
    pub fn with<T>(
        &mut self,
        i: usize,
        f: impl FnOnce(&mut BitReader<'a>) -> Result<T>,
    ) -> Result<T> {
        if let Some(r) = &mut self.shared {
            return f(r);
        }
        let &(at, len) = self
            .toc
            .sections
            .get(i)
            .ok_or_else(|| malformed("a section is missing"))?;
        let bytes = self
            .data
            .get(at..at + len)
            .ok_or_else(|| malformed("a section runs past the frame"))?;
        let mut r = BitReader::new(bytes);
        f(&mut r)
    }
}

/// Integer samples to floats, as their bit depth says.
fn to_float(v: i32, depth: super::header::BitDepth) -> f32 {
    if !depth.float {
        return v as f32 / ((1_i64 << depth.bits) - 1) as f32;
    }
    let bits = depth.bits;
    let exp_bits = depth.exp_bits;
    let sample = v as u32;
    let mantissa_bits = bits - exp_bits - 1;
    let mantissa_mask = (1_u32 << mantissa_bits) - 1;
    let exp_mask = ((1_u32 << (bits - 1)) - 1) ^ mantissa_mask;
    let sign = (sample >> (bits - 1)) & 1;
    let mantissa = sample & mantissa_mask;
    let exp = ((sample & exp_mask) >> mantissa_bits) as i32 - ((1 << (exp_bits - 1)) - 1);
    if bits == 32 && exp_bits == 8 {
        return f32::from_bits(sample);
    }
    let biased = sample & !(1 << (bits - 1));
    if biased == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    let mantissa = if mantissa_bits < 23 {
        mantissa << (23 - mantissa_bits)
    } else {
        mantissa >> (mantissa_bits - 23)
    };
    f32::from_bits((sign << 31) | (((exp + 127).clamp(0, 255) as u32) << 23) | mantissa)
}

/// The extra channels' shifts relative to the colour planes.
fn extra_shifts(image: &ImageHeader, frame: &FrameHeader) -> Vec<u32> {
    let colour = frame.upsampling.trailing_zeros();
    image
        .extra
        .iter()
        .zip(&frame.ec_upsampling)
        .map(|(e, &up)| (up.trailing_zeros() + e.dim_shift).saturating_sub(colour))
        .collect()
}

/// Decode a frame's sections.
///
/// # Errors
///
/// For malformed data or features not implemented.
pub fn decode(
    data: &[u8],
    toc: &Toc,
    image: &ImageHeader,
    frame: &FrameHeader,
    lf_frames: &[Option<FramePlanes>; 5],
    references: &super::render::References,
) -> Result<(FramePlanes, super::features::Features)> {
    let mut sections = Sections::new(data, toc);
    let (cw, ch) = frame.colour_size();
    let (cw, ch) = (cw as usize, ch as usize);
    let group_dim = frame.group_dim() as usize;
    let num_lf = frame.num_lf_groups() as usize;
    let num_groups = frame.num_groups() as usize;
    let (lf_cols, _) = frame.lf_groups();
    let (group_cols, _) = frame.groups();

    // LfGlobal.
    let (features, lf_dequant, vardct_global, tree, mut gimage) = sections.with(0, |r| {
        let features = super::features::Features::parse(r, image, frame, references)?;
        let lf_dequant = if r.bool()? {
            [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0]
        } else {
            [r.f16()?, r.f16()?, r.f16()?]
        };
        let lf_dequant = lf_dequant.map(|v| v / 128.0);
        let vardct_global = if frame.modular {
            None
        } else {
            Some(super::vardct::Global::parse(r)?)
        };
        let mut features = features;
        features.correlation = vardct_global
            .as_ref()
            .map(super::vardct::Global::base_correlation);
        let tree = if r.bool()? {
            let samples = u64::from(frame.width)
                * u64::from(frame.height)
                * (frame.colour_channels + image.extra.len()) as u64;
            Some(Tree::parse(r, (1024 + samples / 16).min(1 << 22) as usize)?)
        } else {
            None
        };
        let mut shapes = Vec::new();
        if frame.modular {
            if frame.jpeg_upsampling.iter().any(|&u| u != 0) {
                return Err(PixelsError::unsupported(
                    "JPEG XL: chroma-subsampled Modular frames",
                ));
            }
            for _ in 0..frame.colour_channels {
                shapes.push(Shape {
                    width: cw,
                    height: ch,
                    hshift: 0,
                    vshift: 0,
                });
            }
        }
        for shift in extra_shifts(image, frame) {
            shapes.push(Shape {
                width: cw.div_ceil(1 << shift),
                height: ch.div_ceil(1 << shift),
                hshift: shift as i32,
                vshift: shift as i32,
            });
        }
        let gimage =
            GlobalImage::parse(r, &shapes, tree.as_ref(), group_dim, image.bit_depth.bits)?;
        Ok((features, lf_dequant, vardct_global, tree, gimage))
    })?;

    let mut vardct = vardct_global
        .as_ref()
        .map(|g| super::vardct::State::new(frame, g, lf_dequant, image));
    let lf_frame = if frame.flags & FrameHeader::USE_LF_FRAME != 0 {
        Some(
            lf_frames[frame.lf_level as usize + 1]
                .as_ref()
                .ok_or_else(|| malformed("a frame uses an LF frame that is not there"))?,
        )
    } else {
        None
    };

    // LF groups.
    for i in 0..num_lf {
        let pos = (i % lf_cols as usize, i / lf_cols as usize);
        sections.with(1 + i, |r| {
            if let (Some(state), Some(global)) = (vardct.as_mut(), vardct_global.as_ref()) {
                if lf_frame.is_none() {
                    state.decode_lf_group(r, global, tree.as_ref(), i, pos)?;
                }
            }
            gimage.decode_group(
                r,
                tree.as_ref(),
                (1 + num_lf + i) as u32,
                group_dim,
                true,
                (0, 0),
                pos,
            )?;
            if let (Some(state), Some(global)) = (vardct.as_mut(), vardct_global.as_ref()) {
                state.decode_hf_metadata(r, global, tree.as_ref(), i, pos)?;
            }
            Ok(())
        })?;
    }
    if let (Some(state), Some(lf)) = (vardct.as_mut(), lf_frame) {
        state.use_lf_frame(lf)?;
    }

    // HfGlobal.
    if let Some(state) = vardct.as_mut() {
        sections.with(1 + num_lf, |r| state.decode_hf_global(r, tree.as_ref()))?;
    } else if !toc.single {
        // A Modular frame's HfGlobal section is empty.
    }

    // Pass groups.
    let shifts = frame.pass_shifts();
    for (pass, &shift) in shifts.iter().enumerate() {
        for g in 0..num_groups {
            let pos = (g % group_cols as usize, g / group_cols as usize);
            sections.with(2 + num_lf + pass * num_groups + g, |r| {
                if let Some(state) = vardct.as_mut() {
                    state.decode_pass_group(r, pass, g, pos)?;
                }
                let stream = 1 + 3 * num_lf + 17 + pass * num_groups + g;
                gimage.decode_group(
                    r,
                    tree.as_ref(),
                    stream as u32,
                    group_dim,
                    false,
                    shift,
                    pos,
                )
            })?;
        }
    }

    let planes = gimage.finish()?;
    let mut planes = planes.into_iter();
    let (colour, vardct_info) = if let Some(state) = vardct {
        let (c, info) = state.finish()?;
        (c, Some(info))
    } else {
        let mut cols: Vec<Plane> = (0..frame.colour_channels)
            .filter_map(|_| planes.next())
            .collect();
        if cols.len() != frame.colour_channels {
            return Err(malformed("a Modular frame lacks colour channels"));
        }
        if image.xyb {
            // Stored as Y, X, B - Y, scaled by the LF dequantisation.
            if cols.len() != 3 {
                return Err(malformed("an XYB frame needs three channels"));
            }
            let b: Vec<f32> = cols[2]
                .data
                .iter()
                .zip(&cols[0].data)
                .map(|(&b, &y)| b.saturating_add(y) as f32 * lf_dequant[2])
                .collect();
            let x: Vec<f32> = cols[1]
                .data
                .iter()
                .map(|&v| v as f32 * lf_dequant[0])
                .collect();
            let y: Vec<f32> = cols[0]
                .data
                .iter()
                .map(|&v| v as f32 * lf_dequant[1])
                .collect();
            ([x, y, b], None)
        } else {
            let f: Vec<Vec<f32>> = cols
                .drain(..)
                .map(|p| {
                    p.data
                        .into_iter()
                        .map(|v| to_float(v, image.bit_depth))
                        .collect()
                })
                .collect();
            if f.len() == 1 {
                let g = f.into_iter().next().unwrap_or_default();
                ([g.clone(), g.clone(), g], None)
            } else {
                let mut it = f.into_iter();
                (
                    [
                        it.next().unwrap_or_default(),
                        it.next().unwrap_or_default(),
                        it.next().unwrap_or_default(),
                    ],
                    None,
                )
            }
        }
    };
    let extra = planes
        .zip(&image.extra)
        .map(|(p, e)| Plane32 {
            width: p.width,
            height: p.height,
            data: p
                .data
                .into_iter()
                .map(|v| to_float(v, e.bit_depth))
                .collect(),
        })
        .collect::<Vec<_>>();
    if extra.len() != image.extra.len() {
        return Err(malformed("a frame lacks extra channels"));
    }
    Ok((
        FramePlanes {
            colour,
            width: cw,
            height: ch,
            extra,
            vardct: vardct_info,
        },
        features,
    ))
}
