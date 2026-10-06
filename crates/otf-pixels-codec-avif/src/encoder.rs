//! The AVIF encoder: pixels to YUV, YUV to an AV1 key frame, and the frame
//! (with an alpha frame, when the image has transparency) into a HEIF
//! container.
//!
//! What is written:
//! - Colour as 8-bit 4:2:0 BT.601 full-range YUV with sRGB primaries and
//!   transfer (`colr` nclx 1/13/6, libavif's default for RGB input); a grey
//!   image as monochrome AV1.
//! - Transparency as an auxiliary monochrome AV1 item (`auxC` alpha URN,
//!   `auxl` reference to the colour item), coded at a finer quantizer than
//!   the colour. An opaque alpha channel is dropped.
//!
//! Lossless AVIF needs 4:4:4 RGB coding at quantizer 0, which this encoder
//! does not produce; asking for it is refused rather than silently lossy.

use crate::av1::encode::frame::{CodedStill, Picture, encode_still, qindex_for_quality};
use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

/// `colr` nclx: BT.709 primaries, sRGB transfer, BT.601 matrix.
const CICP: (u8, u8, u8) = (1, 13, 6);
/// The largest dimension an AV1 sequence header can declare.
const MAX_DIMENSION: u32 = 65_536;

/// Encodes an AVIF still.
#[derive(Debug)]
pub struct AvifEncoder {
    state: Option<State>,
    options: EncodeOptions,
}

#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    pixels: Vec<u8>,
    rows_written: u32,
}

impl Default for AvifEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl AvifEncoder {
    /// An encoder at the default quality.
    #[must_use]
    pub fn new() -> Self {
        Self::from_options(&EncodeOptions::default())
    }

    /// An encoder at `options.quality`.
    #[must_use]
    pub const fn from_options(options: &EncodeOptions) -> Self {
        Self {
            state: None,
            options: *options,
        }
    }
}

impl Encoder for AvifEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        if self.options.lossless {
            return Err(PixelsError::unsupported(
                "lossless AVIF encoding is not implemented; encode lossy or choose PNG or WebP",
            ));
        }
        match desc.pixel {
            PixelFormat::Gray8 | PixelFormat::GrayA8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 => {}
            other => {
                return Err(PixelsError::unsupported(format!(
                    "AVIF encoding needs an 8-bit format; got {other}. Convert first."
                )));
            }
        }
        if desc.width == 0
            || desc.height == 0
            || desc.width > MAX_DIMENSION
            || desc.height > MAX_DIMENSION
        {
            return Err(PixelsError::unsupported(format!(
                "AVIF dimensions must be 1 to {MAX_DIMENSION}; {}x{} is not",
                desc.width, desc.height
            )));
        }
        let capacity = desc
            .byte_len()
            .ok_or_else(|| PixelsError::malformed("avif", "image size overflows"))?;
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
                "avif",
                format!(
                    "{} of {} rows were written",
                    state.rows_written, state.descriptor.height
                ),
            ));
        }
        let bytes = encode(state, self.options.quality)?;
        sink.write_all(&bytes)?;
        sink.flush()
    }
}

/// The planes of an 8-bit image: YUV 4:2:0 (or Y alone for grey) and the
/// alpha channel, if any.
fn to_planes(
    pixels: &[u8],
    format: PixelFormat,
    width: usize,
    height: usize,
) -> (Vec<Vec<u16>>, Option<Vec<u16>>) {
    let channels = format.channels();
    let grey = matches!(format, PixelFormat::Gray8 | PixelFormat::GrayA8);
    let has_alpha = matches!(format, PixelFormat::GrayA8 | PixelFormat::Rgba8);
    let pixel = |i: usize| pixels.get(i * channels..(i + 1) * channels).unwrap_or(&[]);
    let alpha = has_alpha.then(|| {
        (0..width * height)
            .map(|i| u16::from(pixel(i).get(channels - 1).copied().unwrap_or(255)))
            .collect::<Vec<u16>>()
    });
    if grey {
        let y = (0..width * height)
            .map(|i| u16::from(pixel(i).first().copied().unwrap_or(0)))
            .collect();
        return (vec![y], alpha);
    }
    // BT.601 full range: Kr 0.299, Kb 0.114.
    let mut y = Vec::with_capacity(width * height);
    let mut u_full = Vec::with_capacity(width * height);
    let mut v_full = Vec::with_capacity(width * height);
    for i in 0..width * height {
        let p = pixel(i);
        let [r, g, b] = [0, 1, 2].map(|k| f64::from(p.get(k).copied().unwrap_or(0)));
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        y.push(luma.round().clamp(0.0, 255.0) as u16);
        u_full.push((b - luma) / 1.772);
        v_full.push((r - luma) / 1.402);
    }
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    let average = |full: &[f64]| -> Vec<u16> {
        let mut out = Vec::with_capacity(cw * ch);
        for cy in 0..ch {
            for cx in 0..cw {
                let (mut sum, mut n) = (0.0, 0.0);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (x, yy) = (cx * 2 + dx, cy * 2 + dy);
                    if x < width && yy < height {
                        sum += full.get(yy * width + x).copied().unwrap_or(0.0);
                        n += 1.0;
                    }
                }
                out.push((sum / n + 128.0).round().clamp(0.0, 255.0) as u16);
            }
        }
        out
    };
    (vec![y, average(&u_full), average(&v_full)], alpha)
}

fn encode(state: &State, quality: u8) -> Result<Vec<u8>> {
    let (width, height) = (state.descriptor.width, state.descriptor.height);
    let (planes, alpha) = to_planes(
        &state.pixels,
        state.descriptor.pixel,
        width as usize,
        height as usize,
    );
    let qindex = qindex_for_quality(quality);
    let refs: Vec<&[u16]> = planes.iter().map(Vec::as_slice).collect();
    let colour = encode_still(
        &Picture {
            width,
            height,
            planes: &refs,
            cicp: CICP,
            full_range: true,
        },
        qindex,
    )?;
    // Alpha edges show more than chroma errors: code it finer.
    let alpha = match alpha.filter(|a| a.iter().any(|&v| v != 255)) {
        Some(alpha) => Some(encode_still(
            &Picture {
                width,
                height,
                planes: &[&alpha],
                cicp: (2, 2, 2),
                full_range: true,
            },
            (qindex / 2).max(1),
        )?),
        None => None,
    };
    Ok(container(
        width,
        height,
        planes.len() == 1,
        &colour,
        alpha.as_ref(),
    ))
}

/// An ISOBMFF box.
fn bx(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(&(payload.len() as u32 + 8).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// A full box: version and flags, then the payload.
fn full(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut body = vec![version];
    body.extend_from_slice(&flags.to_be_bytes()[1..]);
    body.extend_from_slice(payload);
    bx(kind, &body)
}

/// `av1C` for a coded still (AV1-ISOBMFF §2.3.3).
fn av1c(coded: &CodedStill, mono: bool) -> Vec<u8> {
    let mut payload = vec![
        0x81,               // marker, version 1
        coded.level & 0x1F, // seq_profile 0, seq_level_idx_0
        // tier 0, 8-bit, monochrome, 4:2:0 subsampling, sample position 0.
        (u8::from(mono) << 4) | 0b1100,
        0, // no initial_presentation_delay
    ];
    payload.extend_from_slice(&coded.sequence_header_obu);
    bx(b"av1C", &payload)
}

/// The items' properties, in `ipco` order (1-based indices for `ipma`).
fn container(
    width: u32,
    height: u32,
    mono: bool,
    colour: &CodedStill,
    alpha: Option<&CodedStill>,
) -> Vec<u8> {
    let ftyp = bx(b"ftyp", b"avif\0\0\0\0avifmif1miaf");
    let ispe = {
        let mut p = width.to_be_bytes().to_vec();
        p.extend_from_slice(&height.to_be_bytes());
        full(b"ispe", 0, 0, &p)
    };
    let pixi = |channels: u8| {
        let mut p = vec![channels];
        p.extend(std::iter::repeat_n(8, usize::from(channels)));
        full(b"pixi", 0, 0, &p)
    };
    let colr = {
        let mut p = b"nclx".to_vec();
        for v in [CICP.0, CICP.1, CICP.2] {
            p.extend_from_slice(&u16::from(v).to_be_bytes());
        }
        p.push(0x80); // full_range_flag
        bx(b"colr", &p)
    };
    // ipco: 1 ispe, 2 pixi, 3 av1C, 4 colr; alpha adds 5 pixi, 6 av1C, 7 auxC.
    let mut ipco = [
        ispe,
        pixi(if mono { 1 } else { 3 }),
        av1c(colour, mono),
        colr,
    ]
    .concat();
    let mut associations: Vec<(u16, Vec<u8>)> = vec![(1, vec![1, 2, 0x80 | 3, 4])];
    if let Some(alpha) = alpha {
        let mut urn = crate::meta::URN_ALPHA.as_bytes().to_vec();
        urn.push(0);
        ipco.extend_from_slice(&[pixi(1), av1c(alpha, true), full(b"auxC", 0, 0, &urn)].concat());
        associations.push((2, vec![1, 5, 0x80 | 6, 7]));
    }
    let ipma = {
        let mut p = (associations.len() as u32).to_be_bytes().to_vec();
        for (item, props) in &associations {
            p.extend_from_slice(&item.to_be_bytes());
            p.push(props.len() as u8);
            p.extend_from_slice(props);
        }
        full(b"ipma", 0, 0, &p)
    };
    let iprp = bx(b"iprp", &[bx(b"ipco", &ipco), ipma].concat());

    let hdlr = full(b"hdlr", 0, 0, b"\0\0\0\0pict\0\0\0\0\0\0\0\0\0\0\0\0\0");
    let pitm = full(b"pitm", 0, 0, &1_u16.to_be_bytes());
    let infe = |item: u16| {
        let mut p = item.to_be_bytes().to_vec();
        p.extend_from_slice(&[0, 0]);
        p.extend_from_slice(b"av01");
        p.push(0); // item_name
        full(b"infe", 2, 0, &p)
    };
    let items: Vec<&CodedStill> = std::iter::once(colour).chain(alpha).collect();
    let iinf = {
        let mut p = (items.len() as u16).to_be_bytes().to_vec();
        for i in 0..items.len() {
            p.extend_from_slice(&infe(i as u16 + 1));
        }
        full(b"iinf", 0, 0, &p)
    };
    // auxl: the alpha item (2) is auxiliary to the colour item (1).
    let iref = alpha.map(|_| full(b"iref", 0, 0, &bx(b"auxl", &[0, 2, 0, 1, 0, 1])));
    // iloc v0 with 4-byte offsets and lengths, no base offset.
    let iloc = |mdat_start: u32| {
        let mut p = vec![0x44, 0x00];
        p.extend_from_slice(&(items.len() as u16).to_be_bytes());
        let mut offset = mdat_start;
        for (i, item) in items.iter().enumerate() {
            p.extend_from_slice(&(i as u16 + 1).to_be_bytes());
            p.extend_from_slice(&[0, 0, 0, 1]); // data_reference_index, extent_count
            p.extend_from_slice(&offset.to_be_bytes());
            p.extend_from_slice(&(item.data.len() as u32).to_be_bytes());
            offset += item.data.len() as u32;
        }
        full(b"iloc", 0, 0, &p)
    };
    let meta = |mdat_start: u32| {
        let mut p = [hdlr.clone(), pitm.clone(), iloc(mdat_start), iinf.clone()].concat();
        if let Some(iref) = &iref {
            p.extend_from_slice(iref);
        }
        p.extend_from_slice(&iprp);
        full(b"meta", 0, 0, &p)
    };
    // The meta box's size does not depend on the offsets it records.
    let mdat_start = (ftyp.len() + meta(0).len() + 8) as u32;
    let mdat_payload: Vec<u8> = items
        .iter()
        .flat_map(|item| item.data.iter().copied())
        .collect();
    [ftyp, meta(mdat_start), bx(b"mdat", &mdat_payload)].concat()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_yuv_follows_bt601_full_range() {
        let pixels = [255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255];
        let (planes, alpha) = to_planes(&pixels, PixelFormat::Rgb8, 2, 2);
        assert!(alpha.is_none());
        assert_eq!(planes[0], vec![255, 0, 76, 29]);
        // Chroma of the four averaged: red's V and blue's U pull up.
        let u = (((-76.245_f64) + 255.0 * (1.0 - 0.114)) / 1.772 / 4.0 + 128.0).round();
        assert!(
            (i32::from(planes[1][0]) - u as i32).abs() <= 1,
            "{} vs {u}",
            planes[1][0]
        );
        assert_eq!(planes[1].len(), 1);
    }

    #[test]
    fn grey_keeps_one_plane_and_alpha_is_split_out() {
        let pixels = [10, 200, 20, 255, 30, 0];
        let (planes, alpha) = to_planes(&pixels, PixelFormat::GrayA8, 3, 1);
        assert_eq!(planes, vec![vec![10, 20, 30]]);
        assert_eq!(alpha.unwrap(), vec![200, 255, 0]);
    }

    #[test]
    fn odd_sizes_average_only_the_samples_present() {
        let pixels = [0, 0, 255, 0, 0, 255, 0, 0, 255];
        let (planes, _) = to_planes(&pixels, PixelFormat::Rgb8, 3, 1);
        assert_eq!((planes[1].len(), planes[2].len()), (2, 2));
        assert_eq!(planes[1][0], planes[1][1]);
    }
}
