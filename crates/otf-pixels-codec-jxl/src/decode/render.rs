//! From frames to the image (ISO/IEC 18181-1 §J, §K, §L): restoration
//! filters, upsampling, colour conversion and blending onto the canvas.

#![allow(
    clippy::indexing_slicing,
    reason = "plane indices stay within each plane's own size"
)]

use otf_pixels_core::Result;

use super::frame::{FramePlanes, Plane32};
use super::header::{BlendMode, BlendingInfo, ExtraKind, FrameHeader, FrameType, ImageHeader, Toc, Transfer};
use super::reader::{BitReader, malformed};

/// A canvas: three colour planes and the extra channels, all full size.
#[derive(Debug, Clone)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub colour: [Vec<f32>; 3],
    pub extra: Vec<Vec<f32>>,
}

impl Canvas {
    fn blank(width: usize, height: usize, extras: usize) -> Self {
        Self {
            width,
            height,
            colour: std::array::from_fn(|_| vec![0.0; width * height]),
            extra: vec![vec![0.0; width * height]; extras],
        }
    }
}

/// The four reference slots frames may save to and read from.
#[derive(Debug, Clone, Default)]
pub struct References {
    pub slots: [Option<Canvas>; 4],
}

/// Mirror an index into `0..n`.
#[inline]
pub fn mirror(mut i: isize, n: usize) -> usize {
    let n = n as isize;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - 1 - i;
        } else {
            return i as usize;
        }
    }
}

/// Upsample a plane by `2^shift` with the image's weights.
fn upsample(data: &[f32], width: usize, height: usize, shift: u32, image: &ImageHeader) -> (Vec<f32>, usize, usize) {
    let mut cur = (data.to_vec(), width, height);
    let mut left = shift;
    while left > 0 {
        let step = if left >= 3 { 3 } else { left };
        left -= step;
        let (k, weights) = match step {
            1 => (2, &image.up2),
            2 => (4, &image.up4),
            _ => (8, &image.up8),
        };
        cur = upsample_once(&cur.0, cur.1, cur.2, k, weights);
    }
    cur
}

fn upsample_once(data: &[f32], width: usize, height: usize, k: usize, weights: &[f32]) -> (Vec<f32>, usize, usize) {
    let n = k / 2;
    let mut kernels = vec![[0.0_f32; 25]; n * n];
    let mut wi = 0;
    for y in 0..5 * n {
        for x in y..5 * n {
            let w = weights.get(wi).copied().unwrap_or(0.0);
            wi += 1;
            kernels[(y / 5) * n + x / 5][(y % 5) * 5 + x % 5] = w;
            kernels[(x / 5) * n + y / 5][(x % 5) * 5 + y % 5] = w;
        }
    }
    let (ow, oh) = (width * k, height * k);
    let mut out = vec![0.0; ow * oh];
    let mut window = [0.0_f32; 25];
    for y in 0..oh {
        let ry = y / k;
        let my = (y % k).min(k - y % k - 1);
        let flip_v = y % k >= n;
        for x in 0..ow {
            let rx = x / k;
            let mx = (x % k).min(k - x % k - 1);
            let flip_h = x % k >= n;
            for iy in 0..5 {
                let sy = mirror(ry as isize + iy as isize - 2, height);
                for ix in 0..5 {
                    let sx = mirror(rx as isize + ix as isize - 2, width);
                    window[iy * 5 + ix] = data[sy * width + sx];
                }
            }
            let kernel = &kernels[my * n + mx];
            let (mut sum, mut lo, mut hi) = (0.0_f32, f32::INFINITY, f32::NEG_INFINITY);
            for iy in 0..5 {
                let ky = if flip_v { 4 - iy } else { iy };
                for ix in 0..5 {
                    let kx = if flip_h { 4 - ix } else { ix };
                    let s = window[iy * 5 + ix];
                    sum += kernel[ky * 5 + kx] * s;
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
            }
            out[y * ow + x] = sum.clamp(lo, hi);
        }
    }
    (out, ow, oh)
}

/// The Gabor-like 3x3 smoothing that undoes the encoder's sharpening.
pub fn gaborish(planes: &mut [Vec<f32>; 3], width: usize, height: usize, weights: [[f32; 2]; 3]) {
    for (plane, [w1, w2]) in planes.iter_mut().zip(weights) {
        let norm = 1.0 / (1.0 + 4.0 * (w1 + w2));
        let src = plane.clone();
        for y in 0..height {
            let up = mirror(y as isize - 1, height) * width;
            let mid = y * width;
            let down = mirror(y as isize + 1, height) * width;
            for x in 0..width {
                let l = mirror(x as isize - 1, width);
                let r = mirror(x as isize + 1, width);
                let edges = src[up + x] + src[down + x] + src[mid + l] + src[mid + r];
                let corners = src[up + l] + src[up + r] + src[down + l] + src[down + r];
                plane[mid + x] = (src[mid + x] + w1 * edges + w2 * corners) * norm;
            }
        }
    }
}

/// XYB to linear sRGB, in place.
fn xyb_to_linear(colour: &mut [Vec<f32>; 3], image: &ImageHeader) {
    let o = &image.opsin;
    let cbrt = o.bias.map(f32::cbrt);
    let scale = 255.0 / image.intensity_target;
    let [x, y, b] = colour;
    for ((x, y), b) in x.iter_mut().zip(y.iter_mut()).zip(b.iter_mut()) {
        let l = *y + *x - cbrt[0];
        let m = *y - *x - cbrt[1];
        let s = *b - cbrt[2];
        let l = (l * l * l + o.bias[0]) * scale;
        let m = (m * m * m + o.bias[1]) * scale;
        let s = (s * s * s + o.bias[2]) * scale;
        let mx = &o.matrix;
        *x = mx[0][0] * l + mx[0][1] * m + mx[0][2] * s;
        *y = mx[1][0] * l + mx[1][1] * m + mx[1][2] * s;
        *b = mx[2][0] * l + mx[2][1] * m + mx[2][2] * s;
    }
}

/// Linear light to the encoded transfer function.
fn encode_transfer(v: f32, transfer: Transfer) -> f32 {
    match transfer {
        Transfer::Linear => v,
        Transfer::Gamma(g) => v.max(0.0).powf(g),
        Transfer::Bt709 => {
            let v = v.max(0.0);
            if v < 0.018_053_97 {
                4.5 * v
            } else {
                1.099_296_8 * v.powf(0.45) - 0.099_296_8
            }
        }
        Transfer::Pq => {
            let (m1, m2, c1, c2, c3) = (0.159_301_76, 78.843_75, 0.835_937_5, 18.851_563, 18.6875);
            let y = (v.max(0.0) * 255.0 / 10_000.0).powf(m1);
            ((c1 + c2 * y) / (1.0 + c3 * y)).powf(m2)
        }
        Transfer::Dci => v.max(0.0).powf(1.0 / 2.6),
        Transfer::Hlg => {
            let (a, b, c) = (0.178_832_77, 0.284_668_92, 0.559_910_7);
            let v = v.max(0.0);
            if v <= 1.0 / 12.0 { (3.0 * v).sqrt() } else { a * (12.0 * v - b).ln() + c }
        }
        Transfer::Srgb | Transfer::Unknown => {
            let s = v.signum();
            let v = v.abs();
            s * if v <= 0.003_130_8 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
        }
    }
}

/// The colour transform to the image's colour space.
fn colour_transform(colour: &mut [Vec<f32>; 3], image: &ImageHeader) {
    if !image.xyb {
        return;
    }
    xyb_to_linear(colour, image);
    // An ICC-described image is delivered as sRGB.
    let transfer = if image.colour.want_icc { Transfer::Srgb } else { image.colour.transfer };
    if transfer != Transfer::Linear {
        for plane in colour.iter_mut() {
            for v in plane.iter_mut() {
                *v = encode_transfer(*v, transfer);
            }
        }
    }
}

fn blend_sample(mode: BlendMode, clamp: bool, base: f32, new: f32, alpha: f32, premultiplied: bool) -> f32 {
    let a = if clamp { alpha.clamp(0.0, 1.0) } else { alpha };
    match mode {
        BlendMode::Replace => new,
        BlendMode::Add => base + new,
        BlendMode::Blend => {
            if premultiplied {
                new + base * (1.0 - a)
            } else {
                new * a + base * (1.0 - a)
            }
        }
        BlendMode::MulAdd => base + new * a,
        BlendMode::Mul => {
            if clamp {
                base * new.clamp(0.0, 1.0)
            } else {
                base * new
            }
        }
    }
}

/// Blend a frame's full-size planes onto the canvas its blending source
/// gives.
#[allow(clippy::too_many_lines, reason = "colour, then each extra channel, by mode")]
fn blend(frame: &FrameHeader, image: &ImageHeader, planes: &Frame, references: &References, width: usize, height: usize) -> Canvas {
    let extras = image.extra.len();
    let mut out = references.slots[frame.blending.source as usize]
        .clone()
        .filter(|c| c.width == width && c.height == height)
        .unwrap_or_else(|| Canvas::blank(width, height, extras));
    let (fw, fh) = (planes.width, planes.height);
    let x_start = frame.x0.max(0) as usize;
    let y_start = frame.y0.max(0) as usize;
    let x_end = (i64::from(frame.x0) + fw as i64).clamp(0, width as i64) as usize;
    let y_end = (i64::from(frame.y0) + fh as i64).clamp(0, height as i64) as usize;
    if x_start >= x_end || y_start >= y_end {
        return out;
    }
    // Each channel blends from its own source.
    let source_of = |info: &BlendingInfo| -> Option<&Canvas> {
        references.slots[info.source as usize]
            .as_ref()
            .filter(|c| c.width == width && c.height == height)
    };
    let alpha_info = |info: &BlendingInfo| -> (Option<usize>, bool) {
        let i = info.alpha_channel as usize;
        match image.extra.get(i).map(|e| e.kind) {
            Some(ExtraKind::Alpha { premultiplied }) => (Some(i), premultiplied),
            _ => (None, false),
        }
    };
    let frame_alpha = |i: Option<usize>, fx: usize, fy: usize| -> f32 {
        i.and_then(|i| planes.extra.get(i)).map_or(1.0, |p| p[fy * fw + fx])
    };
    // Colour.
    {
        let info = frame.blending;
        let base = source_of(&info);
        let (ai, premul) = alpha_info(&info);
        for y in y_start..y_end {
            let fy = (y as i64 - i64::from(frame.y0)) as usize;
            for x in x_start..x_end {
                let fx = (x as i64 - i64::from(frame.x0)) as usize;
                let a = frame_alpha(ai, fx, fy);
                for c in 0..3 {
                    let b = base.map_or(0.0, |s| s.colour[c][y * width + x]);
                    out.colour[c][y * width + x] = blend_sample(info.mode, info.clamp, b, planes.colour[c][fy * fw + fx], a, premul);
                }
            }
        }
    }
    for (e, info) in frame.ec_blending.iter().enumerate().take(extras) {
        let base = source_of(info);
        let (ai, premul) = alpha_info(info);
        let is_alpha = ai == Some(e);
        for y in y_start..y_end {
            let fy = (y as i64 - i64::from(frame.y0)) as usize;
            for x in x_start..x_end {
                let fx = (x as i64 - i64::from(frame.x0)) as usize;
                let b = base.map_or(0.0, |s| s.extra[e][y * width + x]);
                let new = planes.extra[e][fy * fw + fx];
                out.extra[e][y * width + x] = if info.mode == BlendMode::Blend && is_alpha {
                    let a = if info.clamp { new.clamp(0.0, 1.0) } else { new };
                    a + b * (1.0 - a)
                } else {
                    let a = frame_alpha(ai, fx, fy);
                    blend_sample(info.mode, info.clamp, b, new, a, premul)
                };
            }
        }
    }
    out
}

/// A frame at full size, after filters, upsampling and features.
#[derive(Debug, Clone)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub colour: [Vec<f32>; 3],
    pub extra: Vec<Vec<f32>>,
}

fn grow(frame: FramePlanes, header: &FrameHeader, image: &ImageHeader) -> Frame {
    let mut colour = frame.colour;
    let (mut w, mut h) = (frame.width, frame.height);
    if header.upsampling > 1 {
        let shift = header.upsampling.trailing_zeros();
        let mut out_w = w;
        let mut out_h = h;
        colour = colour.map(|p| {
            let (o, ow, oh) = upsample(&p, w, h, shift, image);
            out_w = ow;
            out_h = oh;
            o
        });
        w = out_w;
        h = out_h;
    }
    let (fw, fh) = (header.width as usize, header.height as usize);
    let crop = |p: Vec<f32>, pw: usize, ph: usize| -> Vec<f32> {
        if pw == fw && ph == fh {
            return p;
        }
        let mut out = vec![0.0; fw * fh];
        for y in 0..fh.min(ph) {
            let n = fw.min(pw);
            out[y * fw..y * fw + n].copy_from_slice(&p[y * pw..y * pw + n]);
        }
        out
    };
    let colour = colour.map(|p| crop(p, w, h));
    let extra = frame
        .extra
        .into_iter()
        .zip(&header.ec_upsampling)
        .zip(&image.extra)
        .map(|((Plane32 { width, height, data }, &up), info)| {
            let shift = up.trailing_zeros() + info.dim_shift;
            let (d, ew, eh) = if shift > 0 { upsample(&data, width, height, shift, image) } else { (data, width, height) };
            crop(d, ew, eh)
        })
        .collect();
    Frame {
        width: fw,
        height: fh,
        colour,
        extra,
    }
}

/// The decoded image.
#[derive(Debug, Clone)]
pub struct Image {
    pub header: ImageHeader,
    pub canvas: Canvas,
}

/// Parse the codestream's headers and every frame header, without decoding
/// pixels.
///
/// # Errors
///
/// For a malformed codestream.
pub fn probe(codestream: &[u8]) -> Result<(ImageHeader, Option<Vec<u8>>, Vec<u32>)> {
    let mut r = BitReader::new(codestream);
    let header = ImageHeader::parse(&mut r)?;
    let icc = read_icc(&mut r, &header)?;
    let mut durations = Vec::new();
    skip_preview(&mut r, &header)?;
    loop {
        r.zero_pad_to_byte()?;
        let frame = FrameHeader::parse(&mut r, &header)?;
        let toc = Toc::parse(&mut r, &frame)?;
        r.skip(toc.total * 8)?;
        if frame.is_keyframe() {
            durations.push(frame.duration);
        }
        if frame.is_last {
            break;
        }
    }
    Ok((header, icc, durations))
}

fn read_icc(r: &mut BitReader<'_>, header: &ImageHeader) -> Result<Option<Vec<u8>>> {
    if !header.colour.want_icc {
        return Ok(None);
    }
    let encoded = super::icc::read(r)?;
    Ok(Some(super::icc::decode(&encoded)?))
}

fn skip_preview(r: &mut BitReader<'_>, header: &ImageHeader) -> Result<()> {
    if header.preview.is_none() {
        return Ok(());
    }
    r.zero_pad_to_byte()?;
    let mut preview = header.clone();
    if let Some((w, h)) = header.preview {
        preview.width = w;
        preview.height = h;
    }
    let frame = FrameHeader::parse(r, &preview)?;
    let toc = Toc::parse(r, &frame)?;
    r.skip(toc.total * 8)
}

/// Decode the codestream to its first displayed frame.
///
/// # Errors
///
/// For malformed data or features not implemented.
pub fn decode(codestream: &[u8]) -> Result<Image> {
    let mut r = BitReader::new(codestream);
    let header = ImageHeader::parse(&mut r)?;
    read_icc(&mut r, &header)?;
    skip_preview(&mut r, &header)?;
    let (width, height) = (header.width as usize, header.height as usize);
    let mut references = References::default();
    let mut lf_frames: [Option<FramePlanes>; 5] = Default::default();
    let mut canvas = None;
    loop {
        r.zero_pad_to_byte()?;
        let frame = FrameHeader::parse(&mut r, &header)?;
        let toc = Toc::parse(&mut r, &frame)?;
        let start = r.position() / 8;
        let data = codestream
            .get(start..start + toc.total)
            .ok_or_else(|| malformed("a frame runs past the codestream"))?;
        r.skip(toc.total * 8)?;
        if canvas.is_none() {
            let (mut planes, features) = super::frame::decode(data, &toc, &header, &frame, &lf_frames, &references)?;
            super::filter::restore(&mut planes, &frame)?;
            if frame.frame_type == FrameType::Lf {
                lf_frames[frame.lf_level as usize] = Some(planes);
                continue;
            }
            let mut full = grow(planes, &frame, &header);
            features.apply(&mut full, &frame, &header, &references)?;
            if !frame.save_before_ct {
                colour_transform(&mut full.colour, &header);
            }
            let blended = blend(&frame, &header, &full, &references, width, height);
            if frame.can_reference() || frame.frame_type == FrameType::ReferenceOnly {
                references.slots[frame.save_as_reference as usize] = Some(blended.clone());
            }
            if frame.is_keyframe() {
                canvas = Some(blended);
            }
        }
        if frame.is_last || canvas.is_some() {
            break;
        }
    }
    let mut canvas = canvas.ok_or_else(|| malformed("no frame is displayed"))?;
    // Straight alpha out.
    for (e, info) in header.extra.iter().enumerate() {
        if let ExtraKind::Alpha { premultiplied: true } = info.kind {
            let alpha = canvas.extra[e].clone();
            for plane in &mut canvas.colour {
                for (v, &a) in plane.iter_mut().zip(&alpha) {
                    if a > 0.0 {
                        *v /= a;
                    }
                }
            }
        }
    }
    Ok(Image { header, canvas })
}
