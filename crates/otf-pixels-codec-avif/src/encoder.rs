//! The AVIF encoder: pixels to YUV, YUV to an AV1 key frame, and the frame
//! (with an alpha frame, when the image has transparency) into a HEIF
//! container.
//!
//! What is written:
//! - Colour as 8-bit BT.601 full-range YUV with sRGB primaries and transfer
//!   (`colr` nclx 1/13/6, libavif's default for RGB input): 4:2:0 in AV1
//!   profile 0 by default, or 4:4:4 in profile 1 when
//!   [`EncodeOptions::chroma_subsampling`] asks; a grey image as monochrome
//!   AV1.
//! - [`EncodeOptions::effort`] (0..=9, default 4) narrows the intra mode
//!   search below 3; above 4 there is no wider search to make yet.
//! - Transparency as an auxiliary monochrome AV1 item (`auxC` alpha URN,
//!   `auxl` reference to the colour item), coded at a finer quantizer than
//!   the colour. An opaque alpha channel is dropped.
//!
//! Lossless AVIF needs 4:4:4 RGB coding at quantizer 0, which this encoder
//! does not produce; asking for it is refused rather than silently lossy.

use crate::av1::encode::frame::{
    CodedStill, DEFAULT_EFFORT, Picture, encode_still_with, qindex_for_quality,
};
use otf_pixels_core::{
    ChromaSubsampling, EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result,
    Sink,
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
    /// The ICC profile for a `colr` of type `prof`, if any.
    icc: Option<Vec<u8>>,
    /// The EXIF block for an `Exif` item, if any.
    exif: Option<Vec<u8>>,
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
            icc: None,
            exif: None,
        }
    }
}

impl Encoder for AvifEncoder {
    fn set_icc_profile(&mut self, profile: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "profile",
                "the ICC profile must be set before write_header",
            ));
        }
        self.icc = profile.map(<[u8]>::to_vec);
        Ok(())
    }

    fn set_exif(&mut self, exif: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "exif",
                "the EXIF block must be set before write_header",
            ));
        }
        self.exif = exif.map(<[u8]>::to_vec);
        Ok(())
    }

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
        let bytes = encode(
            state,
            &self.options,
            self.icc.as_deref(),
            self.exif.as_deref(),
        )?;
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
    subsampled: bool,
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
    if !subsampled {
        let full = |plane: &[f64]| -> Vec<u16> {
            plane
                .iter()
                .map(|&c| (c + 128.0).round().clamp(0.0, 255.0) as u16)
                .collect()
        };
        return (vec![y, full(&u_full), full(&v_full)], alpha);
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

fn encode(
    state: &State,
    options: &EncodeOptions,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let (width, height) = (state.descriptor.width, state.descriptor.height);
    let subsampled = options.chroma_subsampling != Some(ChromaSubsampling::Yuv444);
    let effort = options.effort.unwrap_or(DEFAULT_EFFORT).min(9);
    let (planes, alpha) = to_planes(
        &state.pixels,
        state.descriptor.pixel,
        width as usize,
        height as usize,
        subsampled,
    );
    let qindex = qindex_for_quality(options.quality);
    let refs: Vec<&[u16]> = planes.iter().map(Vec::as_slice).collect();
    let colour = encode_still_with(
        &Picture {
            width,
            height,
            subsampled,
            planes: &refs,
            cicp: CICP,
            full_range: true,
        },
        qindex,
        effort,
    )?;
    // Alpha edges show more than chroma errors: code it finer.
    let alpha = match alpha.filter(|a| a.iter().any(|&v| v != 255)) {
        Some(alpha) => Some(encode_still_with(
            &Picture {
                width,
                height,
                subsampled: true,
                planes: &[&alpha],
                cicp: (2, 2, 2),
                full_range: true,
            },
            (qindex / 2).max(1),
            effort,
        )?),
        None => None,
    };
    Ok(container(
        width,
        height,
        planes.len() == 1,
        &colour,
        alpha.as_ref(),
        icc,
        exif,
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
        0x81, // marker, version 1
        (coded.profile << 5) | (coded.level & 0x1F),
        // tier 0, 8-bit, monochrome, chroma subsampling x and y, sample
        // position 0.
        (u8::from(mono) << 4) | if coded.subsampled { 0b1100 } else { 0 },
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
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
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
    // ipco: 1 ispe, 2 pixi, 3 av1C, 4 colr; alpha adds 5 pixi, 6 av1C, 7 auxC;
    // a profile adds its colr last.
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
    // An ICC profile is a second `colr` on the colour item, next to the
    // nclx, which still says how to turn the YUV into RGB.
    if let Some(profile) = icc {
        ipco.extend_from_slice(&bx(b"colr", &[b"prof".as_slice(), profile].concat()));
        let index = if alpha.is_some() { 8 } else { 5 };
        if let Some((_, props)) = associations.first_mut() {
            props.push(index);
        }
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
    let infe = |item: u16, kind: &[u8; 4]| {
        let mut p = item.to_be_bytes().to_vec();
        p.extend_from_slice(&[0, 0]);
        p.extend_from_slice(kind);
        p.push(0); // item_name
        full(b"infe", 2, 0, &p)
    };
    // Items 1 (colour) and 2 (alpha) are coded images; an EXIF block is the
    // next, its data a zero offset to the TIFF header and then the block
    // (ISO/IEC 23008-12 A.2.1).
    let exif_data = exif.map(|block| [&[0_u8; 4][..], block].concat());
    let mut items: Vec<(&[u8; 4], &[u8])> = std::iter::once(colour)
        .chain(alpha)
        .map(|coded| (b"av01", coded.data.as_slice()))
        .collect();
    if let Some(data) = &exif_data {
        items.push((b"Exif", data));
    }
    let iinf = {
        let mut p = (items.len() as u16).to_be_bytes().to_vec();
        for (i, (kind, _)) in items.iter().enumerate() {
            p.extend_from_slice(&infe(i as u16 + 1, kind));
        }
        full(b"iinf", 0, 0, &p)
    };
    // auxl: the alpha item (2) is auxiliary to the colour item (1); cdsc:
    // the EXIF item describes it.
    let mut references = Vec::new();
    if alpha.is_some() {
        references.extend(bx(b"auxl", &[0, 2, 0, 1, 0, 1]));
    }
    if exif.is_some() {
        let id = items.len() as u16;
        references.extend(bx(
            b"cdsc",
            &[&id.to_be_bytes()[..], &[0, 1, 0, 1]].concat(),
        ));
    }
    let iref = (!references.is_empty()).then(|| full(b"iref", 0, 0, &references));
    // iloc v0 with 4-byte offsets and lengths, no base offset.
    let iloc = |mdat_start: u32| {
        let mut p = vec![0x44, 0x00];
        p.extend_from_slice(&(items.len() as u16).to_be_bytes());
        let mut offset = mdat_start;
        for (i, (_, data)) in items.iter().enumerate() {
            p.extend_from_slice(&(i as u16 + 1).to_be_bytes());
            p.extend_from_slice(&[0, 0, 0, 1]); // data_reference_index, extent_count
            p.extend_from_slice(&offset.to_be_bytes());
            p.extend_from_slice(&(data.len() as u32).to_be_bytes());
            offset += data.len() as u32;
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
        .flat_map(|(_, data)| data.iter().copied())
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
        let (planes, alpha) = to_planes(&pixels, PixelFormat::Rgb8, 2, 2, true);
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
        let (planes, alpha) = to_planes(&pixels, PixelFormat::GrayA8, 3, 1, true);
        assert_eq!(planes, vec![vec![10, 20, 30]]);
        assert_eq!(alpha.unwrap(), vec![200, 255, 0]);
    }

    #[test]
    fn odd_sizes_average_only_the_samples_present() {
        let pixels = [0, 0, 255, 0, 0, 255, 0, 0, 255];
        let (planes, _) = to_planes(&pixels, PixelFormat::Rgb8, 3, 1, true);
        assert_eq!((planes[1].len(), planes[2].len()), (2, 2));
        assert_eq!(planes[1][0], planes[1][1]);
    }

    /// Encode `pixels` with `options`.
    fn encode_with(descriptor: &ImageDescriptor, pixels: &[u8], options: EncodeOptions) -> Vec<u8> {
        let mut encoder = AvifEncoder::from_options(&options);
        let mut out = Vec::new();
        encoder.write_header(descriptor, &mut out).unwrap();
        for row in pixels.chunks_exact(descriptor.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        out
    }

    fn decode(bytes: &[u8]) -> Vec<u8> {
        use otf_pixels_core::{Decoder, Limits};
        let mut decoder = crate::AvifDecoder::new(bytes, Limits::default()).unwrap();
        let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
        let mut out = Vec::new();
        for _ in 0..decoder.descriptor().height {
            decoder.read_row(&mut row).unwrap();
            out.extend_from_slice(&row);
        }
        out
    }

    /// The `av1C` record's profile byte and subsampling byte.
    fn av1c_bytes(file: &[u8]) -> (u8, u8) {
        let at = file.windows(4).position(|w| w == b"av1C").unwrap() + 4;
        (file[at + 1] >> 5, file[at + 2] & 0b1100)
    }

    #[test]
    fn full_resolution_chroma_keeps_every_chroma_sample() {
        let pixels = [255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255];
        let (planes, _) = to_planes(&pixels, PixelFormat::Rgb8, 2, 2, false);
        assert_eq!((planes[1].len(), planes[2].len()), (4, 4));
        assert_eq!(planes[1][0], 128, "white has no chroma");
        assert_eq!(planes[2][2], 255, "red's V is full scale");
    }

    #[test]
    fn four_four_four_is_profile_one_and_sharper_in_colour() {
        // Thin colour stripes, which 4:2:0 averages away.
        let descriptor = ImageDescriptor::new(32, 16, PixelFormat::Rgb8).unwrap();
        let pixels: Vec<u8> = (0..32 * 16)
            .flat_map(|i| if i % 2 == 0 { [255, 0, 0] } else { [0, 0, 255] })
            .collect();
        let options = EncodeOptions::with_quality(90).unwrap();
        let sub = encode_with(&descriptor, &pixels, options);
        let full = encode_with(
            &descriptor,
            &pixels,
            options.with_chroma_subsampling(ChromaSubsampling::Yuv444),
        );
        assert_eq!(av1c_bytes(&sub), (0, 0b1100));
        assert_eq!(av1c_bytes(&full), (1, 0));
        assert_eq!(
            encode_with(
                &descriptor,
                &pixels,
                options.with_chroma_subsampling(ChromaSubsampling::Yuv420)
            ),
            sub
        );
        let error = |bytes: &[u8]| -> u64 {
            decode(bytes)
                .iter()
                .zip(&pixels)
                .map(|(a, b)| u64::from(a.abs_diff(*b)))
                .sum()
        };
        assert!(
            error(&full) * 2 < error(&sub),
            "{} against {}",
            error(&full),
            error(&sub)
        );
    }

    #[test]
    fn grey_and_alpha_stay_profile_zero_under_four_four_four() {
        let descriptor = ImageDescriptor::new(9, 7, PixelFormat::GrayA8).unwrap();
        let pixels: Vec<u8> = (0..9 * 7)
            .flat_map(|i| [(i * 4) as u8, (i * 3) as u8])
            .collect();
        let file = encode_with(
            &descriptor,
            &pixels,
            EncodeOptions::default().with_chroma_subsampling(ChromaSubsampling::Yuv444),
        );
        assert_eq!(av1c_bytes(&file).0, 0);
        assert_eq!(decode(&file).len(), 9 * 7 * 2);
    }

    #[test]
    fn effort_narrows_the_search_and_four_is_the_default() {
        let descriptor = ImageDescriptor::new(40, 24, PixelFormat::Rgb8).unwrap();
        let pixels: Vec<u8> = (0..40 * 24 * 3)
            .map(|i| ((i * 13 + i / 97) % 256) as u8)
            .collect();
        let base = EncodeOptions::default();
        let default = encode_with(&descriptor, &pixels, base);
        for effort in 3..=10 {
            assert_eq!(
                encode_with(&descriptor, &pixels, base.with_effort(effort).unwrap()),
                default,
                "effort {effort}"
            );
        }
        let fast = encode_with(&descriptor, &pixels, base.with_effort(0).unwrap());
        assert_ne!(fast, default);
        assert_eq!(decode(&fast).len(), pixels.len());
    }
}
