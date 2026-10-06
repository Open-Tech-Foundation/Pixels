//! Conversion from an embedded ICC profile to sRGB: [`ToSrgb`].
//!
//! Covers the profiles cameras, phones and editors actually embed for RGB
//! and grey images — **matrix/TRC** display profiles (Display P3, Adobe RGB,
//! ProPhoto, BT.2020, ...) and grey `kTRC` profiles. A pixel is linearised
//! through the profile's tone curves, taken to the D50 profile connection
//! space by its colorant matrix, out of it by sRGB's, clipped to the sRGB
//! gamut and encoded with the sRGB curve: the relative-colorimetric
//! matrix-shaper transform ICC.1 §F.3 defines, which is what lcms2 builds
//! for the same pair of profiles. LUT-based (`A2B`-only) profiles, CMYK and
//! Lab-PCS profiles are not converted; [`ToSrgb::from_profile`] says so and
//! the caller keeps the profile with the pixels instead.
//!
//! Alpha is untouched.

#![allow(
    clippy::indexing_slicing,
    reason = "3x3 matrices and fixed-size lookup tables indexed within their bounds"
)]

use otf_pixels_core::{
    ChannelLayout, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, SampleKind, Tile,
    TileMut,
};
use std::sync::{Arc, OnceLock};

/// The ICC profile connection space white, D50 (ICC.1 §7.2.16).
const D50: [f64; 3] = [0.9642, 1.0, 0.8249];
/// Entries in the table that encodes linear light as sRGB.
const ENCODE_TABLE: usize = 1 << 12;

/// A tone curve: `curv` (identity, gamma or sampled) or `para`.
#[derive(Debug, Clone, PartialEq)]
enum Curve {
    Gamma(f64),
    Table(Vec<f64>),
    /// ICC.1 Table 68: function type 0-4 and its parameters `g a b c d e f`.
    Parametric(u16, [f64; 7]),
}

impl Curve {
    /// Device value `x` in 0..=1 to linear light, clamped to 0..=1.
    fn eval(&self, x: f64) -> f64 {
        let y = match self {
            Self::Gamma(g) => x.powf(*g),
            Self::Table(table) => {
                let last = table.len().saturating_sub(1);
                let at = x.clamp(0.0, 1.0) * last as f64;
                let i = (at.floor() as usize).min(last.saturating_sub(1));
                let (a, b) = (
                    table.get(i).copied().unwrap_or(0.0),
                    table.get(i + 1).copied().unwrap_or(1.0),
                );
                a + (b - a) * (at - i as f64)
            }
            Self::Parametric(kind, [g, a, b, c, d, e, f]) => match kind {
                0 => x.powf(*g),
                1 => {
                    if x >= -b / a {
                        (a * x + b).powf(*g)
                    } else {
                        0.0
                    }
                }
                2 => {
                    if x >= -b / a {
                        (a * x + b).powf(*g) + c
                    } else {
                        *c
                    }
                }
                3 => {
                    if x >= *d {
                        (a * x + b).powf(*g)
                    } else {
                        c * x
                    }
                }
                _ => {
                    if x >= *d {
                        (a * x + b).powf(*g) + e
                    } else {
                        c * x + f
                    }
                }
            },
        };
        if y.is_nan() { 0.0 } else { y.clamp(0.0, 1.0) }
    }
}

/// sRGB's curve, IEC 61966-2-1, from linear light.
fn srgb_encode(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB's curve to linear light.
fn srgb_decode(value: f64) -> f64 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

type Matrix = [[f64; 3]; 3];

fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn mat_vec(a: &Matrix, v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

fn mat_inv(m: &Matrix) -> Option<Matrix> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-12 {
        return None;
    }
    Some([
        [
            (e * i - f * h) / det,
            (c * h - b * i) / det,
            (b * f - c * e) / det,
        ],
        [
            (f * g - d * i) / det,
            (a * i - c * g) / det,
            (c * d - a * f) / det,
        ],
        [
            (d * h - e * g) / det,
            (b * g - a * h) / det,
            (a * e - b * d) / det,
        ],
    ])
}

/// sRGB's colorants adapted to D50 with Bradford, as lcms2 builds its sRGB
/// profile: BT.709 primaries, white `x, y = 0.3127, 0.3290`.
fn srgb_colorants() -> Matrix {
    const BRADFORD: Matrix = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let xyz = |x: f64, y: f64| [x / y, 1.0, (1.0 - x - y) / y];
    let primaries = [xyz(0.64, 0.33), xyz(0.30, 0.60), xyz(0.15, 0.06)];
    let m: Matrix = [0, 1, 2].map(|i| [0, 1, 2].map(|j| primaries[j][i]));
    let white = xyz(0.3127, 0.3290);
    let Some(inverse) = mat_inv(&m) else {
        return [[0.0; 3]; 3];
    };
    let s = mat_vec(&inverse, white);
    let rgb_to_xyz: Matrix = [0, 1, 2].map(|i| [0, 1, 2].map(|j| m[i][j] * s[j]));
    let (from, to) = (mat_vec(&BRADFORD, white), mat_vec(&BRADFORD, D50));
    let scale: Matrix =
        [0, 1, 2].map(|i| [0, 1, 2].map(|j| if i == j { to[i] / from[i] } else { 0.0 }));
    let Some(bradford_inverse) = mat_inv(&BRADFORD) else {
        return [[0.0; 3]; 3];
    };
    mat_mul(
        &bradford_inverse,
        &mat_mul(&scale, &mat_mul(&BRADFORD, &rgb_to_xyz)),
    )
}

/// Why a profile is not converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unconvertible {
    /// Not an ICC profile, or a damaged one.
    Malformed(&'static str),
    /// A valid profile this converter does not handle (LUT-based, CMYK,
    /// Lab PCS, ...), named for the caller.
    Unsupported(String),
}

/// What [`ToSrgb::from_profile`] makes of a profile.
#[derive(Debug, Clone)]
pub enum Conversion {
    /// The pixels need this op to become sRGB.
    Convert(ToSrgb),
    /// The profile is sRGB in all but name: the pixels already are.
    AlreadySrgb,
    /// The profile cannot be converted here; keep it with the pixels.
    Keep(Unconvertible),
}

/// Converts pixels in an ICC profile's colour space to sRGB.
#[derive(Debug, Clone)]
pub struct ToSrgb {
    shared: Arc<Shared>,
}

#[derive(Debug)]
struct Shared {
    /// One curve per colour channel: three for RGB, one for grey.
    curves: Vec<Curve>,
    /// Linear source RGB to linear sRGB; unused for grey.
    matrix: Matrix,
    /// Each channel's curve at every 8-bit value.
    lut8: Vec<[f32; 256]>,
    /// The same at every 16-bit value, built on first use.
    lut16: OnceLock<Vec<Vec<f32>>>,
    /// Linear light at `i / (ENCODE_TABLE - 1)` encoded as sRGB.
    encode: Vec<f32>,
}

/// The pieces of a profile this needs, read from its tag table.
struct Tags<'a> {
    data: &'a [u8],
    entries: Vec<([u8; 4], usize, usize)>,
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

fn be16(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

fn s15(data: &[u8], at: usize) -> Option<f64> {
    Some(f64::from(be32(data, at)? as i32) / 65536.0)
}

impl<'a> Tags<'a> {
    fn parse(data: &'a [u8]) -> std::result::Result<Self, Unconvertible> {
        let malformed = Unconvertible::Malformed;
        if data.len() < 132 || data.get(36..40) != Some(b"acsp") {
            return Err(malformed("not an ICC profile"));
        }
        let count = be32(data, 128).ok_or(malformed("truncated tag table"))? as usize;
        if count > 1024 || 132 + count * 12 > data.len() {
            return Err(malformed("tag table runs past the profile"));
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let at = 132 + i * 12;
            let sig: [u8; 4] = data
                .get(at..at + 4)
                .and_then(|s| s.try_into().ok())
                .ok_or(malformed("truncated tag entry"))?;
            let offset = be32(data, at + 4).ok_or(malformed("truncated tag entry"))? as usize;
            let size = be32(data, at + 8).ok_or(malformed("truncated tag entry"))? as usize;
            if offset.checked_add(size).is_none_or(|end| end > data.len()) {
                return Err(malformed("a tag runs past the profile"));
            }
            entries.push((sig, offset, size));
        }
        Ok(Self { data, entries })
    }

    fn get(&self, sig: &[u8; 4]) -> Option<&'a [u8]> {
        let &(_, offset, size) = self.entries.iter().find(|(s, _, _)| s == sig)?;
        self.data.get(offset..offset + size)
    }

    fn xyz(&self, sig: &[u8; 4]) -> std::result::Result<[f64; 3], Unconvertible> {
        let tag = self.get(sig).ok_or_else(|| missing(sig))?;
        if tag.get(..4) != Some(b"XYZ ") {
            return Err(Unconvertible::Malformed("a colorant is not an XYZType"));
        }
        let v = |i: usize| s15(tag, 8 + 4 * i).ok_or(Unconvertible::Malformed("truncated XYZType"));
        Ok([v(0)?, v(1)?, v(2)?])
    }

    fn curve(&self, sig: &[u8; 4]) -> std::result::Result<Curve, Unconvertible> {
        let malformed = Unconvertible::Malformed;
        let tag = self.get(sig).ok_or_else(|| missing(sig))?;
        match tag.get(..4) {
            Some(b"curv") => {
                let n = be32(tag, 8).ok_or(malformed("truncated curv"))? as usize;
                match n {
                    0 => Ok(Curve::Gamma(1.0)),
                    1 => Ok(Curve::Gamma(
                        f64::from(be16(tag, 12).ok_or(malformed("truncated curv"))?) / 256.0,
                    )),
                    _ => {
                        let table: Option<Vec<f64>> = (0..n)
                            .map(|i| be16(tag, 12 + 2 * i).map(|v| f64::from(v) / 65535.0))
                            .collect();
                        Ok(Curve::Table(
                            table.ok_or(malformed("truncated curv table"))?,
                        ))
                    }
                }
            }
            Some(b"para") => {
                let kind = be16(tag, 8).ok_or(malformed("truncated para"))?;
                let count = match kind {
                    0 => 1,
                    1 => 3,
                    2 => 4,
                    3 => 5,
                    4 => 7,
                    _ => return Err(malformed("unknown parametric curve type")),
                };
                let mut params = [0.0; 7];
                for (i, p) in params.iter_mut().enumerate().take(count) {
                    *p = s15(tag, 12 + 4 * i).ok_or(malformed("truncated para"))?;
                }
                if kind != 0 && params[1] == 0.0 {
                    return Err(malformed("parametric curve with a zero slope"));
                }
                Ok(Curve::Parametric(kind, params))
            }
            _ => Err(malformed("a tone curve is neither curv nor para")),
        }
    }
}

fn missing(sig: &[u8; 4]) -> Unconvertible {
    Unconvertible::Unsupported(format!(
        "the profile has no `{}` tag, so it is not a matrix/TRC profile",
        String::from_utf8_lossy(sig)
    ))
}

impl ToSrgb {
    /// What converting pixels in `profile` to sRGB takes.
    ///
    /// `Keep` for anything this converter does not handle, which is never an
    /// error: the profile stays with the pixels, and whatever displays them
    /// can still manage their colour.
    #[must_use]
    pub fn from_profile(profile: &[u8]) -> Conversion {
        match Self::build(profile) {
            Ok(Some(op)) => Conversion::Convert(op),
            Ok(None) => Conversion::AlreadySrgb,
            Err(why) => Conversion::Keep(why),
        }
    }

    fn build(profile: &[u8]) -> std::result::Result<Option<Self>, Unconvertible> {
        let tags = Tags::parse(profile)?;
        let space = profile.get(16..20).unwrap_or_default();
        let pcs = profile.get(20..24).unwrap_or_default();
        if pcs != b"XYZ " {
            return Err(Unconvertible::Unsupported(
                "the profile connects through Lab, not XYZ".into(),
            ));
        }
        let (curves, matrix) = match space {
            b"RGB " => {
                let columns = [tags.xyz(b"rXYZ")?, tags.xyz(b"gXYZ")?, tags.xyz(b"bXYZ")?];
                let source: Matrix = [0, 1, 2].map(|i| [0, 1, 2].map(|j| columns[j][i]));
                let to_srgb = mat_inv(&srgb_colorants())
                    .ok_or(Unconvertible::Malformed("sRGB colorants are singular"))?;
                let matrix = mat_mul(&to_srgb, &source);
                let curves = vec![
                    tags.curve(b"rTRC")?,
                    tags.curve(b"gTRC")?,
                    tags.curve(b"bTRC")?,
                ];
                (curves, matrix)
            }
            b"GRAY" => (
                vec![tags.curve(b"kTRC")?],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            other => {
                return Err(Unconvertible::Unsupported(format!(
                    "a `{}` colour space profile",
                    String::from_utf8_lossy(other).trim_end()
                )));
            }
        };
        if is_srgb(&curves, &matrix) {
            return Ok(None);
        }
        let lut8 = curves
            .iter()
            .map(|curve| std::array::from_fn(|i| curve.eval(i as f64 / 255.0) as f32))
            .collect();
        let encode = (0..ENCODE_TABLE)
            .map(|i| srgb_encode(i as f64 / (ENCODE_TABLE - 1) as f64) as f32)
            .collect();
        Ok(Some(Self {
            shared: Arc::new(Shared {
                curves,
                matrix,
                lut8,
                lut16: OnceLock::new(),
                encode,
            }),
        }))
    }

    /// Whether this converts grey (one colour channel) rather than RGB.
    #[must_use]
    pub fn is_grey(&self) -> bool {
        self.shared.curves.len() == 1
    }

    /// Whether pixels of `format` are what this profile describes: grey for
    /// a grey profile, RGB for an RGB one, 8- or 16-bit or float.
    #[must_use]
    pub fn applies_to(&self, format: PixelFormat) -> bool {
        let grey = matches!(
            format.layout(),
            ChannelLayout::Gray | ChannelLayout::GrayAlpha
        );
        grey == self.is_grey()
    }
}

/// Whether a profile is sRGB to within a fraction of an 8-bit step: an
/// identity matrix and sRGB's curve on every channel.
fn is_srgb(curves: &[Curve], matrix: &Matrix) -> bool {
    let identity =
        (0..3).all(|i| (0..3).all(|j| (matrix[i][j] - f64::from(u8::from(i == j))).abs() < 2e-3));
    let srgb_curve = |curve: &Curve| {
        (0..=32).all(|k| {
            let x = f64::from(k) / 32.0;
            (curve.eval(x) - srgb_decode(x)).abs() < 1e-3
        })
    };
    identity && curves.iter().all(srgb_curve)
}

impl Shared {
    /// Linear light in 0..=1 to sRGB in 0..=1, through the encode table.
    fn encode(&self, linear: f32) -> f32 {
        let at = linear.clamp(0.0, 1.0) * (ENCODE_TABLE - 1) as f32;
        let i = (at as usize).min(ENCODE_TABLE - 2);
        let (a, b) = (
            self.encode.get(i).copied().unwrap_or(0.0),
            self.encode.get(i + 1).copied().unwrap_or(1.0),
        );
        a + (b - a) * (at - i as f32)
    }

    /// Linear device values to sRGB values, clipped to the gamut.
    fn convert(&self, linear: [f32; 3]) -> [f32; 3] {
        let m = &self.matrix;
        [0, 1, 2].map(|i| {
            let v = m[i][0] as f32 * linear[0]
                + m[i][1] as f32 * linear[1]
                + m[i][2] as f32 * linear[2];
            self.encode(v)
        })
    }

    fn lut16(&self) -> &[Vec<f32>] {
        self.lut16.get_or_init(|| {
            self.curves
                .iter()
                .map(|curve| {
                    (0..=u16::MAX)
                        .map(|i| curve.eval(f64::from(i) / 65535.0) as f32)
                        .collect()
                })
                .collect()
        })
    }
}

impl Op for ToSrgb {
    fn name(&self) -> &'static str {
        "to_srgb"
    }

    /// Pointwise, like `modulate`: resolution means nothing to it.
    fn rescaled(&self) -> Option<Arc<dyn Op>> {
        Some(Arc::new(self.clone()))
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let [input] = inputs else {
            return Err(PixelsError::graph("to_srgb takes exactly one input"));
        };
        if !self.applies_to(input.pixel) {
            return Err(PixelsError::invalid_argument(
                "profile",
                format!(
                    "a {} profile cannot describe {} pixels",
                    if self.is_grey() { "grey" } else { "RGB" },
                    input.pixel
                ),
            ));
        }
        Ok(*input)
    }

    fn input_regions(&self, output: Region, _inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let [input] = inputs else {
            return Err(PixelsError::graph("to_srgb takes exactly one input"));
        };
        let format = output.pixel();
        let region = output.region();
        let channels = format.channels();
        let colour = if self.is_grey() { 1 } else { 3 };
        let shared = &*self.shared;
        for y in region.y..region.y + region.height {
            let Some(source) = input.row(y) else { continue };
            let Some(target) = output.row_mut(y) else {
                continue;
            };
            match format.sample_kind() {
                SampleKind::U8 => {
                    for (from, to) in source
                        .chunks_exact(channels)
                        .zip(target.chunks_exact_mut(channels))
                    {
                        let linear: [f32; 3] = std::array::from_fn(|c| {
                            let lut = shared.lut8.get(c.min(colour - 1));
                            let v = from.get(c.min(colour - 1)).copied().unwrap_or(0);
                            lut.and_then(|l| l.get(usize::from(v)))
                                .copied()
                                .unwrap_or(0.0)
                        });
                        let out = if colour == 1 {
                            [shared.encode(linear[0]); 3]
                        } else {
                            shared.convert(linear)
                        };
                        for (c, slot) in to.iter_mut().enumerate() {
                            *slot = if c < colour {
                                (out[c] * 255.0 + 0.5) as u8
                            } else {
                                from.get(c).copied().unwrap_or(255)
                            };
                        }
                    }
                }
                SampleKind::U16 => {
                    let lut = shared.lut16();
                    let bytes = channels * 2;
                    for (from, to) in source
                        .chunks_exact(bytes)
                        .zip(target.chunks_exact_mut(bytes))
                    {
                        let sample = |c: usize| {
                            u16::from_ne_bytes([
                                from.get(2 * c).copied().unwrap_or(0),
                                from.get(2 * c + 1).copied().unwrap_or(0),
                            ])
                        };
                        let linear: [f32; 3] = std::array::from_fn(|c| {
                            let c = c.min(colour - 1);
                            lut.get(c)
                                .and_then(|l| l.get(usize::from(sample(c))))
                                .copied()
                                .unwrap_or(0.0)
                        });
                        let out = if colour == 1 {
                            [shared.encode(linear[0]); 3]
                        } else {
                            shared.convert(linear)
                        };
                        for c in 0..channels {
                            let value = if let Some(v) = out.get(c).filter(|_| c < colour) {
                                (v * 65535.0 + 0.5) as u16
                            } else {
                                sample(c)
                            };
                            if let Some(slot) = to.get_mut(2 * c..2 * c + 2) {
                                slot.copy_from_slice(&value.to_ne_bytes());
                            }
                        }
                    }
                }
                SampleKind::F32 => {
                    let bytes = channels * 4;
                    for (from, to) in source
                        .chunks_exact(bytes)
                        .zip(target.chunks_exact_mut(bytes))
                    {
                        let sample = |c: usize| {
                            let mut b = [0_u8; 4];
                            if let Some(s) = from.get(4 * c..4 * c + 4) {
                                b.copy_from_slice(s);
                            }
                            f32::from_ne_bytes(b)
                        };
                        let linear: [f32; 3] = std::array::from_fn(|c| {
                            let c = c.min(colour - 1);
                            shared
                                .curves
                                .get(c)
                                .map_or(0.0, |curve| curve.eval(f64::from(sample(c))) as f32)
                        });
                        let out = if colour == 1 {
                            [srgb_encode(f64::from(linear[0])) as f32; 3]
                        } else {
                            let m = &shared.matrix;
                            [0, 1, 2].map(|i| {
                                let v = (0..3).map(|k| m[i][k] * f64::from(linear[k])).sum::<f64>();
                                srgb_encode(v.clamp(0.0, 1.0)) as f32
                            })
                        };
                        for c in 0..channels {
                            let value = out.get(c).filter(|_| c < colour).copied().unwrap_or_else(|| sample(c));
                            if let Some(slot) = to.get_mut(4 * c..4 * c + 4) {
                                slot.copy_from_slice(&value.to_ne_bytes());
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;

    /// A minimal profile: header, tag table, tags.
    fn profile(space: &[u8; 4], pcs: &[u8; 4], tags: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut header = vec![0_u8; 128];
        header[16..20].copy_from_slice(space);
        header[20..24].copy_from_slice(pcs);
        header[36..40].copy_from_slice(b"acsp");
        let mut table = (tags.len() as u32).to_be_bytes().to_vec();
        let mut data = Vec::new();
        let base = 128 + 4 + 12 * tags.len();
        for (sig, body) in tags {
            table.extend_from_slice(*sig);
            table.extend_from_slice(&((base + data.len()) as u32).to_be_bytes());
            table.extend_from_slice(&(body.len() as u32).to_be_bytes());
            data.extend_from_slice(body);
            while data.len() % 4 != 0 {
                data.push(0);
            }
        }
        [header, table, data].concat()
    }

    fn xyz(v: [f64; 3]) -> Vec<u8> {
        let mut out = b"XYZ \0\0\0\0".to_vec();
        for x in v {
            out.extend_from_slice(&((x * 65536.0).round() as i32).to_be_bytes());
        }
        out
    }

    fn gamma(g: f64) -> Vec<u8> {
        let mut out = b"curv\0\0\0\0\0\0\0\x01".to_vec();
        out.extend_from_slice(&((g * 256.0).round() as u16).to_be_bytes());
        out
    }

    fn para(kind: u16, params: &[f64]) -> Vec<u8> {
        let mut out = b"para\0\0\0\0".to_vec();
        out.extend_from_slice(&kind.to_be_bytes());
        out.extend_from_slice(&[0, 0]);
        for p in params {
            out.extend_from_slice(&((p * 65536.0).round() as i32).to_be_bytes());
        }
        out
    }

    fn rgb_profile(trc: Vec<u8>) -> Vec<u8> {
        let m = srgb_colorants();
        profile(
            b"RGB ",
            b"XYZ ",
            &[
                (b"rXYZ", xyz([m[0][0], m[1][0], m[2][0]])),
                (b"gXYZ", xyz([m[0][1], m[1][1], m[2][1]])),
                (b"bXYZ", xyz([m[0][2], m[1][2], m[2][2]])),
                (b"rTRC", trc.clone()),
                (b"gTRC", trc.clone()),
                (b"bTRC", trc),
            ],
        )
    }

    #[test]
    fn srgb_colorants_match_the_published_d50_values() {
        // The rXYZ/gXYZ/bXYZ of the ubiquitous sRGB IEC61966-2.1 profiles.
        let m = srgb_colorants();
        let published = [
            [0.4361, 0.3851, 0.1431],
            [0.2225, 0.7169, 0.0606],
            [0.0139, 0.0971, 0.7141],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - published[i][j]).abs() < 2e-4,
                    "{i}{j}: {}",
                    m[i][j]
                );
            }
        }
    }

    #[test]
    fn an_srgb_profile_is_recognised_and_left_alone() {
        let srgb = rgb_profile(para(
            3,
            &[2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045],
        ));
        assert!(matches!(
            ToSrgb::from_profile(&srgb),
            Conversion::AlreadySrgb
        ));
        // Same primaries, a different curve: a conversion.
        assert!(matches!(
            ToSrgb::from_profile(&rgb_profile(gamma(1.8))),
            Conversion::Convert(_)
        ));
    }

    #[test]
    fn parametric_curves_follow_icc_table_68() {
        let curve = |kind, params: [f64; 7]| Curve::Parametric(kind, params);
        assert!((curve(0, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).eval(0.5) - 0.25).abs() < 1e-12);
        // Type 1 is zero below -b/a.
        let t1 = curve(1, [1.0, 2.0, -0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t1.eval(0.2), 0.0);
        assert!((t1.eval(0.5) - 0.5).abs() < 1e-12);
        // Type 2 floors at c.
        assert!((curve(2, [1.0, 2.0, -0.5, 0.1, 0.0, 0.0, 0.0]).eval(0.2) - 0.1).abs() < 1e-12);
        // Type 3 is sRGB's shape: linear below d.
        let t3 = curve(
            3,
            [
                2.4,
                1.0 / 1.055,
                0.055 / 1.055,
                1.0 / 12.92,
                0.04045,
                0.0,
                0.0,
            ],
        );
        for k in 0..=20 {
            let x = f64::from(k) / 20.0;
            assert!((t3.eval(x) - srgb_decode(x)).abs() < 1e-9);
        }
        // Type 4 adds the offsets e and f.
        let t4 = curve(4, [1.0, 1.0, 0.0, 0.5, 0.5, 0.1, 0.05]);
        assert!((t4.eval(0.2) - 0.15).abs() < 1e-12);
        assert!((t4.eval(0.8) - 0.9).abs() < 1e-12);
        // Tables interpolate; the ends are exact.
        let table = Curve::Table(vec![0.0, 0.25, 1.0]);
        assert!((table.eval(0.25) - 0.125).abs() < 1e-12);
        assert_eq!((table.eval(0.0), table.eval(1.0)), (0.0, 1.0));
    }

    #[test]
    fn unconvertible_profiles_are_kept_not_refused() {
        let cmyk = profile(b"CMYK", b"Lab ", &[]);
        assert!(matches!(
            ToSrgb::from_profile(&cmyk),
            Conversion::Keep(Unconvertible::Unsupported(_))
        ));
        let lab_pcs = profile(b"RGB ", b"Lab ", &[]);
        assert!(matches!(
            ToSrgb::from_profile(&lab_pcs),
            Conversion::Keep(Unconvertible::Unsupported(_))
        ));
        // An RGB profile with only an A2B0 LUT has no colorants.
        let lut_only = profile(b"RGB ", b"XYZ ", &[(b"A2B0", b"mft2\0\0\0\0".to_vec())]);
        assert!(matches!(
            ToSrgb::from_profile(&lut_only),
            Conversion::Keep(Unconvertible::Unsupported(_))
        ));
        for broken in [&b"not a profile"[..], &[0_u8; 200][..]] {
            assert!(matches!(
                ToSrgb::from_profile(broken),
                Conversion::Keep(Unconvertible::Malformed(_))
            ));
        }
        // A tag pointing past the end.
        let mut truncated = rgb_profile(gamma(2.2));
        truncated.truncate(truncated.len() - 4);
        assert!(matches!(
            ToSrgb::from_profile(&truncated),
            Conversion::Keep(Unconvertible::Malformed(_))
        ));
    }

    #[test]
    fn grey_profiles_convert_grey_only() {
        let grey = profile(b"GRAY", b"XYZ ", &[(b"kTRC", gamma(1.0))]);
        let Conversion::Convert(op) = ToSrgb::from_profile(&grey) else {
            unreachable!("a grey kTRC profile converts")
        };
        assert!(op.applies_to(PixelFormat::Gray8) && op.applies_to(PixelFormat::GrayA8));
        assert!(!op.applies_to(PixelFormat::Rgb8));
        // Linear grey 0.5 is sRGB 188.
        let encoded = (op.shared.encode(op.shared.lut8[0][128]) * 255.0 + 0.5) as u8;
        assert_eq!(encoded, 188);
    }
}
