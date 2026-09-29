//! YUV to RGB conversion for decoded AV1 planes.
//!
//! AV1 codes Y, U and V; the engine's formats are RGB. The conversion has two
//! parts, and neither is defined bit-exactly by the format the way the AV1
//! reconstruction is, so the aim is the mathematically exact result rather than
//! any one library's approximation of it:
//!
//! - **Chroma upsampling.** 4:2:0 and 4:2:2 chroma is brought to full
//!   resolution with the triangle ("bilinear") filter: each output sample
//!   weighs its nearest chroma sample 3:1 against the next one out, per
//!   subsampled axis (9:3:3:1 in 4:2:0), treating chroma as sited between the
//!   luma samples it covers. Edges replicate. This is the filter libavif and
//!   libyuv apply by default.
//! - **The matrix.** `matrix_coefficients` picks the luma weights `Kr`/`Kb`
//!   (BT.709, BT.601, BT.2020 non-constant-luminance), and the range decides
//!   whether samples span the full code range or the studio range (16..=235
//!   luma, 16..=240 chroma at 8 bits). Coefficients are fixed point with 16
//!   fractional bits (ADR-0011), and the upsampled chroma carries 4 more, so
//!   the result is the exactly rounded one except where the exact value lies
//!   within about 1% of a step from a rounding boundary (a few in 10,000
//!   samples), and never more than one step away.
//!
//! libavif's own output — through libyuv, whose 8-bit matrices keep about six
//! fractional bits — lands up to a few steps away in saturated colours, and
//! further at studio range; the reference suite allows for that per fixture.

use crate::av1::Plane;
use otf_pixels_core::{PixelsError, Result};

/// Fractional bits of the matrix coefficients.
const COEFF_BITS: u32 = 16;
/// Fractional bits of an upsampled chroma sample (weights sum to 16).
const CHROMA_BITS: u32 = 4;

/// An 8-bit YUV to RGB matrix in fixed point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct YuvMatrix {
    /// Luma scale (`255 / 219` at studio range, 1 at full range).
    y_scale: i32,
    /// Luma offset (16 at studio range, 0 at full range).
    y_offset: i32,
    /// V's contribution to R.
    v_to_r: i32,
    /// U's (negated) contribution to G.
    u_to_g: i32,
    /// V's (negated) contribution to G.
    v_to_g: i32,
    /// U's contribution to B.
    u_to_b: i32,
}

impl YuvMatrix {
    /// The matrix for CICP `matrix_coefficients` at full or studio range.
    ///
    /// # Errors
    ///
    /// [`PixelsError::Unsupported`] for a matrix that is not a `Kr`/`Kb`
    /// weighting this module implements: identity (0) is not a YUV matrix and
    /// is handled by the caller, and YCgCo, constant-luminance and ICtCp
    /// matrices are not implemented.
    pub(crate) fn new(matrix_coefficients: u16, full_range: bool) -> Result<Self> {
        let (kr, kb) = match matrix_coefficients {
            1 => (0.2126, 0.0722),
            // 2 is "unspecified"; like libavif, read it as BT.601.
            2 | 5 | 6 => (0.299, 0.114),
            9 => (0.2627, 0.0593),
            other => {
                return Err(PixelsError::unsupported(format!(
                    "avif: YUV to RGB conversion for colour matrix {other} is not implemented"
                )));
            }
        };
        let kg = 1.0 - kr - kb;
        let (y_scale, y_offset, c_scale) = if full_range {
            (1.0, 0, 1.0)
        } else {
            (255.0 / 219.0, 16, 255.0 / 224.0)
        };
        let fixed = |x: f64| (x * f64::from(1_u32 << COEFF_BITS)).round() as i32;
        Ok(Self {
            y_scale: fixed(y_scale),
            y_offset,
            v_to_r: fixed(2.0 * (1.0 - kr) * c_scale),
            u_to_g: fixed(2.0 * kb * (1.0 - kb) / kg * c_scale),
            v_to_g: fixed(2.0 * kr * (1.0 - kr) / kg * c_scale),
            u_to_b: fixed(2.0 * (1.0 - kb) * c_scale),
        })
    }

    /// Convert one sample: 8-bit `y`, and `u`/`v` carrying `CHROMA_BITS`
    /// fractional bits (a plain 8-bit chroma sample shifted up by them).
    fn rgb(&self, y: u16, u: i32, v: i32) -> [u8; 3] {
        const SHIFT: u32 = COEFF_BITS + CHROMA_BITS;
        const HALF: i32 = 1 << (SHIFT - 1);
        let mid = 128 << CHROMA_BITS;
        let (u, v) = (u - mid, v - mid);
        let luma = ((i32::from(y) - self.y_offset) * self.y_scale) << CHROMA_BITS;
        // `>>` floors, so adding half first rounds to nearest (half up) for
        // negative sums too.
        let channel = |sum: i32| ((sum + HALF) >> SHIFT).clamp(0, 255) as u8;
        [
            channel(luma + self.v_to_r * v),
            channel(luma - self.u_to_g * u - self.v_to_g * v),
            channel(luma + self.u_to_b * u),
        ]
    }
}

/// The picture's geometry: its display size and chroma subsampling.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Layout {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) subsampling_x: usize,
    pub(crate) subsampling_y: usize,
}

/// Convert decoded Y/U/V `planes` to an interleaved 8-bit RGB raster of
/// `layout`'s display size.
pub(crate) fn yuv_to_rgb(planes: &[Plane], layout: Layout, matrix: &YuvMatrix) -> Result<Vec<u8>> {
    let [y_plane, u_plane, v_plane] = planes else {
        return Err(PixelsError::malformed("avif", "a colour plane is missing"));
    };
    let Layout {
        width,
        height,
        subsampling_x: sub_x,
        subsampling_y: sub_y,
    } = layout;
    let chroma = Chroma {
        width: (width + sub_x) >> sub_x,
        height: (height + sub_y) >> sub_y,
        sub_x,
        sub_y,
    };
    let mut raster = vec![0_u8; width * height * 3];
    for (y, row) in raster.chunks_exact_mut(width * 3).enumerate() {
        for (x, px) in row.chunks_exact_mut(3).enumerate() {
            let luma = y_plane.get(x, y).unwrap_or(0);
            let u = chroma.upsample(u_plane, x, y);
            let v = chroma.upsample(v_plane, x, y);
            px.copy_from_slice(&matrix.rgb(luma, u, v));
        }
    }
    Ok(raster)
}

/// The displayed chroma plane's size and subsampling, for upsampling.
struct Chroma {
    width: usize,
    height: usize,
    sub_x: usize,
    sub_y: usize,
}

impl Chroma {
    /// The chroma value at luma position `(x, y)`, with `CHROMA_BITS`
    /// fractional bits: the triangle filter along each subsampled axis, a
    /// plain copy along the others.
    fn upsample(&self, plane: &Plane, x: usize, y: usize) -> i32 {
        // Along a subsampled axis, luma sample 2i+0 sits a quarter of a chroma
        // sample before chroma sample i and 2i+1 a quarter after it, so each
        // weighs sample i by 3 and its neighbour on that side by 1.
        let taps = |pos: usize, sub: usize, len: usize| -> [(usize, i32); 2] {
            if sub == 0 {
                return [(pos, 4), (pos, 0)];
            }
            let near = pos >> 1;
            let far = if pos & 1 == 0 {
                near.saturating_sub(1)
            } else {
                (near + 1).min(len - 1)
            };
            [(near.min(len - 1), 3), (far, 1)]
        };
        let sample = |cx: usize, cy: usize| i32::from(plane.get(cx, cy).unwrap_or(0));
        let mut sum = 0;
        for (cy, wy) in taps(y, self.sub_y, self.height) {
            for (cx, wx) in taps(x, self.sub_x, self.width) {
                sum += wy * wx * sample(cx, cy);
            }
        }
        sum
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// The exact conversion in floating point, rounded once at the end.
    fn exact(matrix: u16, full_range: bool, y: f64, u: f64, v: f64) -> [u8; 3] {
        let (kr, kb) = match matrix {
            1 => (0.2126, 0.0722),
            9 => (0.2627, 0.0593),
            _ => (0.299, 0.114),
        };
        let (y, u, v) = if full_range {
            (y / 255.0, (u - 128.0) / 255.0, (v - 128.0) / 255.0)
        } else {
            ((y - 16.0) / 219.0, (u - 128.0) / 224.0, (v - 128.0) / 224.0)
        };
        let r = y + 2.0 * (1.0 - kr) * v;
        let b = y + 2.0 * (1.0 - kb) * u;
        let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
        [r, g, b].map(|c| (c * 255.0).round().clamp(0.0, 255.0) as u8)
    }

    #[test]
    fn fixed_point_agrees_with_exact_arithmetic() {
        // Sweep the 8-bit cube coarsely for every matrix and range. The
        // fixed-point result must be within one step of the exactly rounded
        // one everywhere, and equal to it but for the rare exact value lying
        // right at a rounding boundary, where the 2^-16 coefficient
        // quantisation can tip it.
        for matrix in [1, 6, 9] {
            for full_range in [true, false] {
                let m = YuvMatrix::new(matrix, full_range).unwrap();
                let mut off_by_one = 0;
                let mut total = 0;
                for y in (0..=255_u16).step_by(5) {
                    for u in (0..=255_i32).step_by(7) {
                        for v in (0..=255_i32).step_by(7) {
                            let ours = m.rgb(y, u << CHROMA_BITS, v << CHROMA_BITS);
                            let want = exact(matrix, full_range, y.into(), u.into(), v.into());
                            for (a, b) in ours.iter().zip(&want) {
                                let d = a.abs_diff(*b);
                                assert!(
                                    d <= 1,
                                    "matrix {matrix} full {full_range} yuv {y},{u},{v}"
                                );
                                off_by_one += usize::from(d == 1);
                                total += 1;
                            }
                        }
                    }
                }
                assert!(
                    off_by_one * 1000 < total,
                    "matrix {matrix}: {off_by_one} of {total} samples off by one"
                );
            }
        }
    }

    #[test]
    fn grey_stays_grey_and_the_range_ends_map_to_black_and_white() {
        for matrix in [1, 6, 9] {
            let full = YuvMatrix::new(matrix, true).unwrap();
            let mid = 128 << CHROMA_BITS;
            assert_eq!(full.rgb(0, mid, mid), [0, 0, 0]);
            assert_eq!(full.rgb(255, mid, mid), [255, 255, 255]);
            assert_eq!(full.rgb(77, mid, mid), [77, 77, 77]);
            let studio = YuvMatrix::new(matrix, false).unwrap();
            assert_eq!(studio.rgb(16, mid, mid), [0, 0, 0]);
            assert_eq!(studio.rgb(235, mid, mid), [255, 255, 255]);
        }
    }

    #[test]
    fn unimplemented_matrices_are_refused() {
        for matrix in [0, 3, 4, 7, 8, 10, 12, 13, 14] {
            assert!(YuvMatrix::new(matrix, true).is_err(), "matrix {matrix}");
        }
    }

    #[test]
    fn upsampling_weighs_the_nearer_chroma_sample_three_to_one() {
        // A 4:2:0 chroma row 0, 64: luma column 1 sits a quarter past chroma 0
        // toward chroma 1, so it reads (3*0 + 64) / 4 = 16; column 2 reads
        // (3*64 + 0) / 4 = 48; the edges replicate.
        let mut plane = Plane::new(2, 1);
        plane.set(1, 0, 64);
        let chroma = Chroma {
            width: 2,
            height: 1,
            sub_x: 1,
            sub_y: 1,
        };
        let at = |x| chroma.upsample(&plane, x, 0) >> CHROMA_BITS;
        assert_eq!([at(0), at(1), at(2), at(3)], [0, 16, 48, 64]);
    }
}
