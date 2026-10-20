//! Patches, splines and noise (ISO/IEC 18181-1 §G.1.4–§G.1.6, §K.3–§K.5).
//! Follows the procedure as jxl-oxide (MIT/Apache-2.0) implements it; see
//! NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "plane indices are clipped to the frame before use"
)]
#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "the specification's constants, as written"
)]

use otf_pixels_core::Result;

use super::coding::Decoder;
use super::header::{ExtraKind, FrameHeader, ImageHeader};
use super::reader::{BitReader, malformed, unpack_signed};
use super::render::{Frame, Op, References, mirror};

#[derive(Debug, Clone, Copy)]
struct PatchBlending {
    mode: u32,
    alpha: usize,
    clamp: bool,
}

#[derive(Debug, Clone)]
struct Patch {
    reference: usize,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    /// Each target's position and blending, for colour then each extra channel.
    targets: Vec<(i64, i64, Vec<PatchBlending>)>,
}

#[derive(Debug, Clone)]
struct Spline {
    points: Vec<(f32, f32)>,
    xyb: [[i32; 32]; 3],
    sigma: [i32; 32],
}

/// A frame's patches, splines and noise.
#[derive(Debug, Clone, Default)]
pub struct Features {
    patches: Vec<Patch>,
    splines: Vec<Spline>,
    quant_adjust: i32,
    noise: Option<[f32; 8]>,
    /// VarDCT's base chroma-from-luma correlations, X and B.
    pub correlation: Option<(f32, f32)>,
    /// Frames shown and not shown before this one, seeding the noise.
    pub visible: u64,
    pub invisible: u64,
}

impl Features {
    /// Parse whichever of the three the frame's flags declare.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn parse(
        r: &mut BitReader<'_>,
        image: &ImageHeader,
        frame: &FrameHeader,
        _references: &References,
    ) -> Result<Self> {
        let mut out = Self::default();
        if frame.flags & FrameHeader::PATCHES != 0 {
            out.patches = parse_patches(r, image, frame)?;
        }
        if frame.flags & FrameHeader::SPLINES != 0 {
            (out.splines, out.quant_adjust) = parse_splines(r, frame)?;
        }
        if frame.flags & FrameHeader::NOISE != 0 {
            let mut lut = [0.0; 8];
            for v in &mut lut {
                *v = r.read(10)? as f32 / 1024.0;
            }
            out.noise = Some(lut);
        }
        Ok(out)
    }

    /// Draw the features onto a full-size frame.
    ///
    /// # Errors
    ///
    /// For a patch whose reference is missing.
    pub fn apply(
        &self,
        frame: &mut Frame,
        header: &FrameHeader,
        image: &ImageHeader,
        references: &References,
    ) -> Result<()> {
        for patch in &self.patches {
            let reference = references.slots[patch.reference]
                .as_ref()
                .ok_or_else(|| malformed("a patch names a missing reference frame"))?;
            apply_patch(frame, patch, reference, image)?;
        }
        if !self.splines.is_empty() {
            for s in &self.splines {
                draw_spline(frame, header, s, self.quant_adjust, self.correlation);
            }
        }
        if let Some(lut) = self.noise {
            add_noise(
                frame,
                header,
                &lut,
                self.correlation,
                self.visible,
                self.invisible,
            );
        }
        Ok(())
    }
}

fn parse_patches(
    r: &mut BitReader<'_>,
    image: &ImageHeader,
    frame: &FrameHeader,
) -> Result<Vec<Patch>> {
    let alphas: Vec<usize> = image
        .extra
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e.kind, ExtraKind::Alpha { .. }))
        .map(|(i, _)| i)
        .collect();
    let mut d = Decoder::parse(r, 10)?;
    d.begin(r)?;
    let max_refs = (1 << 24).min(u64::from(frame.width) * u64::from(frame.height) / 16);
    let n = d.read(r, 0)?;
    if u64::from(n) > max_refs {
        return Err(malformed("too many patches"));
    }
    let mut total = 0_u64;
    let mut out = Vec::with_capacity(n as usize);
    for _ in 0..n {
        let reference = d.read(r, 1)? as usize;
        if reference >= 4 {
            return Err(malformed("a patch reference out of range"));
        }
        let x0 = d.read(r, 3)? as usize;
        let y0 = d.read(r, 3)? as usize;
        let width = d.read(r, 2)? as usize + 1;
        let height = d.read(r, 2)? as usize + 1;
        let count = d.read(r, 7)? + 1;
        total += u64::from(count);
        if total > max_refs * 4 {
            return Err(malformed("too many patches"));
        }
        let mut targets = Vec::with_capacity(count as usize);
        let mut prev: Option<(i64, i64)> = None;
        for _ in 0..count {
            let (x, y) = match prev {
                Some((px, py)) => (
                    px + i64::from(unpack_signed(d.read(r, 6)?)),
                    py + i64::from(unpack_signed(d.read(r, 6)?)),
                ),
                None => (i64::from(d.read(r, 4)?), i64::from(d.read(r, 4)?)),
            };
            prev = Some((x, y));
            let mut blending = Vec::with_capacity(image.extra.len() + 1);
            for _ in 0..=image.extra.len() {
                let mode = d.read(r, 5)?;
                if mode > 7 {
                    return Err(malformed("unknown patch blend mode"));
                }
                let alpha = if mode >= 4 && alphas.len() >= 2 {
                    d.read(r, 8)? as usize
                } else {
                    alphas.first().copied().unwrap_or(0)
                };
                let clamp = if mode >= 3 { d.read(r, 9)? != 0 } else { false };
                if mode >= 4 && alpha >= image.extra.len() {
                    return Err(malformed("a patch's alpha channel is out of range"));
                }
                blending.push(PatchBlending { mode, alpha, clamp });
            }
            targets.push((x, y, blending));
        }
        out.push(Patch {
            reference,
            x0,
            y0,
            width,
            height,
            targets,
        });
    }
    d.finish()?;
    Ok(out)
}

fn apply_patch(
    frame: &mut Frame,
    patch: &Patch,
    reference: &super::render::Canvas,
    image: &ImageHeader,
) -> Result<()> {
    let colour = 3;
    let channels = colour + frame.extra.len();
    for (tx, ty, blending) in &patch.targets {
        for idx in 0..channels {
            let info = if idx < colour {
                blending[0]
            } else {
                blending[idx - colour + 1]
            };
            let alpha_ch = info.alpha + colour;
            let op = match info.mode {
                0 => continue,
                1 => Op::Replace,
                2 => Op::Add,
                3 => Op::Mul(info.clamp),
                4 | 5 => {
                    let swapped = info.mode == 5;
                    if idx == alpha_ch {
                        Op::MixAlpha(info.clamp, swapped)
                    } else {
                        let premultiplied = matches!(
                            image.extra.get(info.alpha).map(|e| e.kind),
                            Some(ExtraKind::Alpha {
                                premultiplied: true
                            })
                        );
                        Op::Blend(info.clamp, swapped, premultiplied)
                    }
                }
                _ => {
                    let swapped = info.mode == 7;
                    if idx == alpha_ch {
                        if swapped { Op::Replace } else { continue }
                    } else {
                        Op::MulAdd(info.clamp, swapped)
                    }
                }
            };
            let uses_alpha = info.mode >= 4;
            for py in 0..patch.height {
                let fy = ty + py as i64;
                let ry = patch.y0 + py;
                if fy < 0 || fy >= frame.height as i64 || ry >= reference.height {
                    continue;
                }
                for px in 0..patch.width {
                    let fx = tx + px as i64;
                    let rx = patch.x0 + px;
                    if fx < 0 || fx >= frame.width as i64 || rx >= reference.width {
                        continue;
                    }
                    let f = fy as usize * frame.width + fx as usize;
                    let rf = ry * reference.width + rx;
                    let new = if idx < colour {
                        reference.colour[idx][rf]
                    } else {
                        reference.extra[idx - colour][rf]
                    };
                    let (base_a, new_a) = if uses_alpha && alpha_ch != idx {
                        (
                            frame.extra.get(info.alpha).map_or(0.0, |p| p[f]),
                            reference.extra.get(info.alpha).map(|p| p[rf]),
                        )
                    } else {
                        (0.0, None)
                    };
                    let base = if idx < colour {
                        &mut frame.colour[idx][f]
                    } else {
                        &mut frame.extra[idx - colour][f]
                    };
                    *base = op.apply(*base, new, base_a, new_a);
                }
            }
        }
    }
    Ok(())
}

fn parse_splines(r: &mut BitReader<'_>, frame: &FrameHeader) -> Result<(Vec<Spline>, i32)> {
    let mut d = Decoder::parse(r, 6)?;
    d.begin(r)?;
    let pixels = u64::from(frame.width) * u64::from(frame.height);
    let n = d.read(r, 2)? as u64;
    if n >= (1 << 24).min(pixels / 4) {
        return Err(malformed("too many splines"));
    }
    let n = n as usize + 1;
    let mut starts = Vec::with_capacity(n);
    let mut prev = (i64::from(d.read(r, 1)?), i64::from(d.read(r, 1)?));
    starts.push(prev);
    for _ in 1..n {
        prev.0 += i64::from(unpack_signed(d.read(r, 1)?));
        prev.1 += i64::from(unpack_signed(d.read(r, 1)?));
        starts.push(prev);
    }
    let quant_adjust = unpack_signed(d.read(r, 0)?);
    let mut total_points = 0_u64;
    let mut splines = Vec::with_capacity(n);
    for start in starts {
        let count = d.read(r, 3)?;
        total_points += u64::from(count);
        if total_points > (1 << 20).min(pixels / 2) {
            return Err(malformed("too many spline points"));
        }
        let mut points = vec![(start.0 as f32, start.1 as f32)];
        let (mut cur, mut delta) = (start, (0_i64, 0_i64));
        for _ in 0..count {
            delta.0 += i64::from(unpack_signed(d.read(r, 4)?));
            delta.1 += i64::from(unpack_signed(d.read(r, 4)?));
            let next = (cur.0.saturating_add(delta.0), cur.1.saturating_add(delta.1));
            if next == cur {
                return Err(malformed("repeated spline control points"));
            }
            cur = next;
            points.push((cur.0 as f32, cur.1 as f32));
        }
        let mut xyb = [[0; 32]; 3];
        for c in &mut xyb {
            for v in c.iter_mut() {
                *v = unpack_signed(d.read(r, 5)?);
            }
        }
        let mut sigma = [0; 32];
        for v in &mut sigma {
            *v = unpack_signed(d.read(r, 5)?);
        }
        splines.push(Spline { points, xyb, sigma });
    }
    d.finish()?;
    Ok((splines, quant_adjust))
}

fn continuous_idct(dct: &[f32; 32], t: f32) -> f32 {
    let mut out = dct[0];
    for (i, &v) in dct.iter().enumerate().skip(1) {
        out += std::f32::consts::SQRT_2
            * v
            * ((i as f32) * (std::f32::consts::PI / 32.0) * (t + 0.5)).cos();
    }
    out
}

fn erf(x: f32) -> f32 {
    let ax = x.abs();
    let d1 = ax * 7.77394369e-02 + 2.05260015e-04;
    let d2 = d1 * ax + 2.32120216e-01;
    let d3 = d2 * ax + 2.77820801e-01;
    let d4 = d3 * ax + 1.0;
    let inv = 1.0 / (d4 * d4);
    let r = 1.0 - inv * inv;
    if x < 0.0 { -r } else { r }
}

fn upsampled_points(s: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if s.len() == 1 {
        return s.to_vec();
    }
    let sub = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0, a.1 - b.1);
    let add = |a: (f32, f32), b: (f32, f32)| (a.0 + b.0, a.1 + b.1);
    let mul = |a: (f32, f32), k: f32| (a.0 * k, a.1 * k);
    let mirror_p = |p: (f32, f32), c: (f32, f32)| (2.0 * c.0 - p.0, 2.0 * c.1 - p.1);
    let mut ext = Vec::with_capacity(s.len() + 2);
    ext.push(mirror_p(s[1], s[0]));
    ext.extend_from_slice(s);
    ext.push(mirror_p(s[s.len() - 2], s[s.len() - 1]));
    let mut out = Vec::new();
    for i in 0..ext.len() - 3 {
        let p = &ext[i..i + 4];
        out.push(p[1]);
        let mut t = [0.0_f32; 4];
        for k in 1..4 {
            let d = sub(p[k], p[k - 1]);
            t[k] = t[k - 1] + (d.0 * d.0 + d.1 * d.1).powf(0.25);
        }
        for step in 1..16 {
            let knot = t[1] + (step as f32 / 16.0) * (t[2] - t[1]);
            let mut a = [(0.0, 0.0); 3];
            for k in 0..3 {
                a[k] = add(
                    p[k],
                    mul(sub(p[k + 1], p[k]), (knot - t[k]) / (t[k + 1] - t[k])),
                );
            }
            let mut b = [(0.0, 0.0); 2];
            for k in 0..2 {
                b[k] = add(
                    a[k],
                    mul(sub(a[k + 1], a[k]), (knot - t[k]) / (t[k + 2] - t[k])),
                );
            }
            out.push(add(
                b[0],
                mul(sub(b[1], b[0]), (knot - t[1]) / (t[2] - t[1])),
            ));
        }
    }
    out.push(s[s.len() - 1]);
    out
}

#[allow(
    clippy::needless_range_loop,
    reason = "three channels indexed together"
)]
fn draw_spline(
    frame: &mut Frame,
    header: &FrameHeader,
    s: &Spline,
    quant_adjust: i32,
    correlation: Option<(f32, f32)>,
) {
    const WEIGHTS: [f32; 4] = [0.0042, 0.075, 0.07, 0.3333];
    let qa = quant_adjust as f32;
    let inv_qa = if qa >= 0.0 {
        1.0 / (1.0 + qa / 8.0)
    } else {
        1.0 - qa / 8.0
    };
    let mut xyb = [[0.0_f32; 32]; 3];
    for c in 0..3 {
        for i in 0..32 {
            xyb[c][i] = s.xyb[c][i] as f32 * WEIGHTS[c] * inv_qa;
        }
    }
    let (cx, cb) = correlation.unwrap_or((0.0, 1.0));
    for i in 0..32 {
        xyb[0][i] += cx * xyb[1][i];
        xyb[2][i] += cb * xyb[1][i];
    }
    let mut sigma = [0.0_f32; 32];
    for (o, &q) in sigma.iter_mut().zip(&s.sigma) {
        *o = q as f32 * WEIGHTS[3] * inv_qa;
    }
    // Arc-length sampling at unit steps.
    let up = upsampled_points(&s.points);
    let mut samples = vec![(up[0], 1.0_f32)];
    let mut current = up[0];
    let mut next = 0;
    while next < up.len() {
        let mut prev = current;
        let mut arc = 0.0_f32;
        loop {
            if next >= up.len() {
                samples.push((prev, arc));
                break;
            }
            let n = up[next];
            let len = ((n.0 - prev.0).powi(2) + (n.1 - prev.1).powi(2)).sqrt();
            if arc + len >= 1.0 {
                let k = (1.0 - arc) / len;
                current = (prev.0 + (n.0 - prev.0) * k, prev.1 + (n.1 - prev.1) * k);
                samples.push((current, 1.0));
                break;
            }
            arc += len;
            prev = n;
            next += 1;
        }
    }
    let total = samples.len() as f32 - 2.0 + samples.last().map_or(0.0, |s| s.1);
    let (fw, fh) = (header.width as i32, header.height as i32);
    for (i, &((px, py), length)) in samples.iter().enumerate() {
        let t = 31.0 * (i as f32 / total).min(1.0);
        let sg = continuous_idct(&sigma, t);
        let inv_sigma = 1.0 / sg;
        let values = [
            continuous_idct(&xyb[0], t) * length,
            continuous_idct(&xyb[1], t) * length,
            continuous_idct(&xyb[2], t) * length,
        ];
        let max_colour = values.iter().copied().fold(0.01_f32, f32::max);
        let max_dist = (2.0 * (10.0_f32.ln() * 3.0 + max_colour)).sqrt() * sg.abs();
        let x0 = ((px - max_dist + 0.5).floor() as i32).max(0);
        let x1 = ((px + max_dist + 1.5).floor() as i32)
            .min(fw)
            .min(frame.width as i32);
        let y0 = ((py - max_dist + 0.5).floor() as i32).max(0);
        let y1 = ((py + max_dist + 1.5).floor() as i32)
            .min(fh)
            .min(frame.height as i32);
        for (c, plane) in frame.colour.iter_mut().enumerate() {
            for y in y0..y1 {
                for x in x0..x1 {
                    let (dx, dy) = (x as f32 - px, y as f32 - py);
                    let dist = (dx * dx + dy * dy).sqrt();
                    const SQRT_0125: f32 = 0.353_553_38;
                    let f = erf((0.5 * dist + SQRT_0125) * inv_sigma)
                        - erf((0.5 * dist - SQRT_0125) * inv_sigma);
                    plane[y as usize * frame.width + x as usize] += 0.25 * values[c] * sg * f * f;
                }
            }
        }
    }
}

fn split_mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// libjxl's eight-lane XorShift128+.
struct Rng {
    s0: [u64; 8],
    s1: [u64; 8],
}

impl Rng {
    fn new(seed0: u64, seed1: u64) -> Self {
        let mut s0 = [0; 8];
        let mut s1 = [0; 8];
        s0[0] = split_mix(seed0.wrapping_add(0x9E3779B97F4A7C15));
        s1[0] = split_mix(seed1.wrapping_add(0x9E3779B97F4A7C15));
        for i in 1..8 {
            s0[i] = split_mix(s0[i - 1]);
            s1[i] = split_mix(s1[i - 1]);
        }
        Self { s0, s1 }
    }

    /// Sixteen floats in [1, 2).
    fn batch(&mut self) -> [f32; 16] {
        let mut out = [0.0; 16];
        for i in 0..8 {
            let mut a = self.s0[i];
            let b = self.s1[i];
            let v = a.wrapping_add(b);
            self.s0[i] = b;
            a ^= a << 23;
            self.s1[i] = a ^ (b ^ (a >> 18) ^ (b >> 5));
            for (k, half) in [v as u32, (v >> 32) as u32].into_iter().enumerate() {
                out[i * 2 + k] = f32::from_bits((half >> 9) | 0x3f80_0000);
            }
        }
        out
    }
}

fn add_noise(
    frame: &mut Frame,
    header: &FrameHeader,
    lut: &[f32; 8],
    correlation: Option<(f32, f32)>,
    visible: u64,
    invisible: u64,
) {
    let (w, h) = (frame.width, frame.height);
    let gd = header.group_dim() as usize;
    let seed0 = (visible << 32) + invisible;
    let mut noise = [
        vec![0.0_f32; w * h],
        vec![0.0_f32; w * h],
        vec![0.0_f32; w * h],
    ];
    for gy in (0..h).step_by(gd) {
        for gx in (0..w).step_by(gd) {
            let (gw, gh) = (gd.min(w - gx), gd.min(h - gy));
            let mut rng = Rng::new(seed0, ((gx as u64) << 32) + gy as u64);
            let stride = gw.div_ceil(16) * 16;
            for plane in &mut noise {
                let mut row = Vec::with_capacity(stride);
                for y in 0..gh {
                    row.clear();
                    while row.len() < stride {
                        row.extend_from_slice(&rng.batch());
                    }
                    plane[(gy + y) * w + gx..(gy + y) * w + gx + gw].copy_from_slice(&row[..gw]);
                }
            }
        }
    }
    let conv: Vec<Vec<f32>> = noise
        .iter()
        .map(|p| {
            let mut out = vec![0.0; w * h];
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0.0;
                    for dy in -2..=2_isize {
                        let sy = mirror(y as isize + dy, h);
                        for dx in -2..=2_isize {
                            sum += p[sy * w + mirror(x as isize + dx, w)] * 0.16;
                        }
                    }
                    out[y * w + x] = sum - p[y * w + x] * 4.0;
                }
            }
            out
        })
        .collect();
    let mut lut9 = [0.0; 9];
    lut9[..8].copy_from_slice(lut);
    lut9[8] = lut[7];
    let (cx, cb) = correlation.unwrap_or((0.0, 1.0));
    let [px, py, pb] = &mut frame.colour;
    for i in 0..w * h {
        let (gx, gy) = (px[i], py[i]);
        let strength = |v: f32| {
            let s = (v * 3.0).max(0.0);
            let k = (s as usize).min(7);
            (lut9[k + 1] - lut9[k]) * (s - k as f32) + lut9[k]
        };
        let sx = strength(gx + gy);
        let sy = strength(gy - gx);
        let nx = 0.22 * sx * (0.0078125 * conv[0][i] + 0.9921875 * conv[2][i]);
        let ny = 0.22 * sy * (0.0078125 * conv[1][i] + 0.9921875 * conv[2][i]);
        px[i] += cx * (nx + ny) + nx - ny;
        py[i] += nx + ny;
        pb[i] += cb * (nx + ny);
    }
}
