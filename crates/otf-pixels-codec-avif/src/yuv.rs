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
//!   luma, 16..=240 chroma at 8 bits, scaled up by `2^(depth - 8)` for 10 and
//!   12). Coefficients are fixed point with 20 fractional bits (ADR-0011) and
//!   the upsampled chroma carries 4 more, so the result is the exactly rounded
//!   one except where the exact value lies right at a rounding boundary (a few
//!   in 10,000 samples), and never more than one step away.
//!
//! The output is 8-bit for 8-bit input and 16-bit — the full 0..=65535 range
//! `Rgb16` promises — for 10- and 12-bit input: the conversion is computed at
//! the output depth directly, as libavif does, rather than converted and then
//! rescaled. The identity matrix (GBR stored as YUV) goes through the same
//! rescale.
//!
//! libavif's own 8-bit output — through libyuv, whose 8-bit matrices keep about
//! six fractional bits — lands up to a few steps away in saturated colours, and
//! further at studio range; the reference suite allows for that per fixture.

use crate::av1::Plane;
use otf_pixels_core::{PixelsError, Result};

/// Fractional bits of the matrix coefficients.
const COEFF_BITS: u32 = 20;
/// Fractional bits of an upsampled chroma sample (weights sum to 16).
const CHROMA_BITS: u32 = 4;

/// The sample depths of a conversion: the decoded planes' and the raster's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Depths {
    /// The AV1 bit depth: 8, 10 or 12.
    pub(crate) input: u8,
    /// The raster's bits per sample: 8 or 16.
    pub(crate) output: u8,
}

impl Depths {
    /// The raster depth the engine's formats use for `input`: 8 stays 8, and
    /// 10 and 12 widen to 16 (SPEC §Pixel formats has nothing in between).
    pub(crate) fn for_input(input: u8) -> Self {
        Self {
            input,
            output: if input > 8 { 16 } else { 8 },
        }
    }

    fn output_max(self) -> i64 {
        (1_i64 << self.output) - 1
    }
}

/// A YUV to RGB matrix in fixed point, for one pair of depths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct YuvMatrix {
    depths: Depths,
    /// Luma scale, output steps per input step of the luma range.
    y_scale: i64,
    /// Luma offset: `16 << (depth - 8)` at studio range, 0 at full range.
    y_offset: i64,
    /// The chroma zero point, `1 << (depth - 1)`.
    c_mid: i64,
    /// V's contribution to R.
    v_to_r: i64,
    /// U's (negated) contribution to G.
    u_to_g: i64,
    /// V's (negated) contribution to G.
    v_to_g: i64,
    /// U's contribution to B.
    u_to_b: i64,
}

impl YuvMatrix {
    /// The matrix for CICP `matrix_coefficients` at full or studio range.
    ///
    /// # Errors
    ///
    /// [`PixelsError::Unsupported`] for a matrix that is not a `Kr`/`Kb`
    /// weighting this module implements: identity (0) is not a YUV matrix and
    /// has its own path (`identity_to_rgb`), and YCgCo, constant-luminance and
    /// ICtCp matrices are not implemented.
    pub(crate) fn new(matrix_coefficients: u16, full_range: bool, depths: Depths) -> Result<Self> {
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
        let step = f64::from(1_u32 << (depths.input - 8));
        let input_max = f64::from((1_u32 << depths.input) - 1);
        // The input span that maps onto the whole output range, per channel.
        let (y_span, y_offset, c_span) = if full_range {
            (input_max, 0, input_max)
        } else {
            (219.0 * step, 16 << (depths.input - 8), 224.0 * step)
        };
        let out = depths.output_max() as f64;
        let (y_scale, c_scale) = (out / y_span, out / c_span);
        let fixed = |x: f64| (x * f64::from(1_u32 << COEFF_BITS)).round() as i64;
        Ok(Self {
            depths,
            y_scale: fixed(y_scale),
            y_offset,
            c_mid: 1 << (depths.input - 1),
            v_to_r: fixed(2.0 * (1.0 - kr) * c_scale),
            u_to_g: fixed(2.0 * kb * (1.0 - kb) / kg * c_scale),
            v_to_g: fixed(2.0 * kr * (1.0 - kr) / kg * c_scale),
            u_to_b: fixed(2.0 * (1.0 - kb) * c_scale),
        })
    }

    /// Convert one sample: `y` at the input depth, and `u`/`v` carrying
    /// `CHROMA_BITS` fractional bits (a plain chroma sample shifted up by them).
    fn rgb(&self, y: u16, u: i64, v: i64) -> [u16; 3] {
        const SHIFT: u32 = COEFF_BITS + CHROMA_BITS;
        const HALF: i64 = 1 << (SHIFT - 1);
        let mid = self.c_mid << CHROMA_BITS;
        let (u, v) = (u - mid, v - mid);
        let luma = ((i64::from(y) - self.y_offset) * self.y_scale) << CHROMA_BITS;
        let max = self.depths.output_max();
        // `>>` floors, so adding half first rounds to nearest (half up) for
        // negative sums too.
        let channel = |sum: i64| ((sum + HALF) >> SHIFT).clamp(0, max) as u16;
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

/// Convert decoded Y/U/V `planes` to RGB samples at the matrix's output depth,
/// three per pixel in raster order over `layout`'s display size.
pub(crate) fn yuv_to_rgb(planes: &[Plane], layout: Layout, matrix: &YuvMatrix) -> Result<Vec<u16>> {
    let [y_plane, u_plane, v_plane] = planes else {
        return Err(PixelsError::malformed("avif", "a colour plane is missing"));
    };
    let chroma = Chroma {
        luma_width: layout.width,
        width: (layout.width + layout.subsampling_x) >> layout.subsampling_x,
        height: (layout.height + layout.subsampling_y) >> layout.subsampling_y,
        sub_x: layout.subsampling_x,
        sub_y: layout.subsampling_y,
    };
    Ok(samples(layout, |x, y| {
        let luma = y_plane.get(x, y).unwrap_or(0);
        let u = i64::from(chroma.upsample(u_plane, x, y));
        let v = i64::from(chroma.upsample(v_plane, x, y));
        matrix.rgb(luma, u, v)
    }))
}

/// Convert identity-matrix planes — G, B, R stored as Y, U, V, always 4:4:4 —
/// to RGB samples, each rescaled from the input depth to the output depth.
pub(crate) fn identity_to_rgb(
    planes: &[Plane],
    layout: Layout,
    depths: Depths,
) -> Result<Vec<u16>> {
    let [g, b, r] = planes else {
        return Err(PixelsError::malformed("avif", "a colour plane is missing"));
    };
    let scale = Scale::new(depths, true);
    Ok(samples(layout, |x, y| {
        let at = |plane: &Plane| scale.apply(plane.get(x, y).unwrap_or(0));
        [at(r), at(g), at(b)]
    }))
}

/// A single plane — monochrome luma, or an alpha item — as samples at the
/// output depth, one per pixel. Studio-range luma is expanded to the full
/// output range (16 maps to 0, 235 to the maximum, clamped beyond); a
/// monochrome picture has no chroma, so this *is* its YUV to grey conversion.
pub(crate) fn plane_to_grey(
    plane: &Plane,
    layout: Layout,
    depths: Depths,
    full_range: bool,
) -> Vec<u16> {
    let scale = Scale::new(depths, full_range);
    samples(layout, |x, y| [scale.apply(plane.get(x, y).unwrap_or(0))])
}

/// An exact affine rescale of one luma-like sample to the output depth:
/// `round((v - offset) * output_max / span)`, clamped.
#[derive(Debug, Clone, Copy)]
struct Scale {
    offset: i64,
    span: i64,
    output_max: i64,
}

impl Scale {
    fn new(depths: Depths, full_range: bool) -> Self {
        let step = 1_i64 << (depths.input - 8);
        let (offset, span) = if full_range {
            (0, (1_i64 << depths.input) - 1)
        } else {
            (16 * step, 219 * step)
        };
        Self {
            offset,
            span,
            output_max: depths.output_max(),
        }
    }

    fn apply(self, v: u16) -> u16 {
        let num = (i64::from(v) - self.offset) * self.output_max;
        // Round half up, flooring correctly for negative numerators.
        (2 * num + self.span)
            .div_euclid(2 * self.span)
            .clamp(0, self.output_max) as u16
    }
}

/// Collect `N` samples per pixel, in raster order, from a per-pixel function.
fn samples<const N: usize>(layout: Layout, pixel: impl Fn(usize, usize) -> [u16; N]) -> Vec<u16> {
    let mut out = Vec::with_capacity(layout.width * layout.height * N);
    for y in 0..layout.height {
        for x in 0..layout.width {
            out.extend_from_slice(&pixel(x, y));
        }
    }
    out
}

/// The displayed chroma plane's size and subsampling, for upsampling.
struct Chroma {
    /// The picture's luma width, for libyuv's right-edge rule.
    luma_width: usize,
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
        // libyuv, which libavif converts with, copies the last chroma sample
        // into the last column rather than filtering it when the width is odd
        // (its rows end `dst[w - 1] = src[(w - 1) / 2]`); rows have no such
        // rule. Matching it keeps an odd-width image's right edge identical.
        let x_taps = if self.sub_x == 1 && x + 1 == self.luma_width && x & 1 == 0 {
            [(x >> 1, 4), (x >> 1, 0)]
        } else {
            taps(x, self.sub_x, self.width)
        };
        let mut sum = 0;
        for (cy, wy) in taps(y, self.sub_y, self.height) {
            for (cx, wx) in x_taps {
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

    const DEPTHS: [Depths; 3] = [
        Depths {
            input: 8,
            output: 8,
        },
        Depths {
            input: 10,
            output: 16,
        },
        Depths {
            input: 12,
            output: 16,
        },
    ];

    /// The exact conversion in floating point, rounded once at the end.
    fn exact(matrix: u16, full_range: bool, depths: Depths, y: f64, u: f64, v: f64) -> [u16; 3] {
        let (kr, kb) = match matrix {
            1 => (0.2126, 0.0722),
            9 => (0.2627, 0.0593),
            _ => (0.299, 0.114),
        };
        let step = f64::from(1_u32 << (depths.input - 8));
        let max = f64::from((1_u32 << depths.input) - 1);
        let mid = f64::from(1_u32 << (depths.input - 1));
        let (y, u, v) = if full_range {
            (y / max, (u - mid) / max, (v - mid) / max)
        } else {
            (
                (y - 16.0 * step) / (219.0 * step),
                (u - mid) / (224.0 * step),
                (v - mid) / (224.0 * step),
            )
        };
        let r = y + 2.0 * (1.0 - kr) * v;
        let b = y + 2.0 * (1.0 - kb) * u;
        let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
        let out = depths.output_max() as f64;
        [r, g, b].map(|c| (c * out).round().clamp(0.0, out) as u16)
    }

    #[test]
    fn fixed_point_agrees_with_exact_arithmetic() {
        // Sweep the input cube coarsely for every matrix, range and depth
        // pair. The fixed-point result must be within one step of the exactly
        // rounded one everywhere, and equal to it but for the rare exact value
        // lying right at a rounding boundary, where the coefficient
        // quantisation can tip it.
        for depths in DEPTHS {
            let max = (1_u16 << depths.input) - 1;
            for matrix in [1, 6, 9] {
                for full_range in [true, false] {
                    let m = YuvMatrix::new(matrix, full_range, depths).unwrap();
                    let mut off_by_one = 0;
                    let mut total = 0;
                    for y in (0..=max).step_by(usize::from(max / 51)) {
                        for u in (0..=max).step_by(usize::from(max / 36)) {
                            for v in (0..=max).step_by(usize::from(max / 36)) {
                                let (cu, cv) = (i64::from(u), i64::from(v));
                                let ours = m.rgb(y, cu << CHROMA_BITS, cv << CHROMA_BITS);
                                let want =
                                    exact(matrix, full_range, depths, y.into(), u.into(), v.into());
                                for (a, b) in ours.iter().zip(&want) {
                                    let d = a.abs_diff(*b);
                                    assert!(
                                        d <= 1,
                                        "{depths:?} matrix {matrix} full {full_range} yuv {y},{u},{v}"
                                    );
                                    off_by_one += usize::from(d == 1);
                                    total += 1;
                                }
                            }
                        }
                    }
                    assert!(
                        off_by_one * 1000 < total,
                        "{depths:?} matrix {matrix}: {off_by_one} of {total} off by one"
                    );
                }
            }
        }
    }

    #[test]
    fn grey_stays_grey_and_the_range_ends_map_to_black_and_white() {
        for depths in DEPTHS {
            let step = 1_u16 << (depths.input - 8);
            let (max_in, max_out) = ((1_u16 << depths.input) - 1, depths.output_max() as u16);
            let mid = (1_i64 << (depths.input - 1)) << CHROMA_BITS;
            for matrix in [1, 6, 9] {
                let full = YuvMatrix::new(matrix, true, depths).unwrap();
                assert_eq!(full.rgb(0, mid, mid), [0; 3]);
                assert_eq!(full.rgb(max_in, mid, mid), [max_out; 3]);
                let studio = YuvMatrix::new(matrix, false, depths).unwrap();
                assert_eq!(studio.rgb(16 * step, mid, mid), [0; 3]);
                assert_eq!(studio.rgb(235 * step, mid, mid), [max_out; 3]);
            }
        }
        let eight = YuvMatrix::new(6, true, DEPTHS[0]).unwrap();
        assert_eq!(
            eight.rgb(77, 128 << CHROMA_BITS, 128 << CHROMA_BITS),
            [77; 3]
        );
    }

    #[test]
    fn unimplemented_matrices_are_refused() {
        for matrix in [0, 3, 4, 7, 8, 10, 12, 13, 14] {
            assert!(
                YuvMatrix::new(matrix, true, DEPTHS[0]).is_err(),
                "matrix {matrix}"
            );
        }
    }

    #[test]
    fn wide_output_is_full_range_sixteen_bit() {
        // A 10-bit identity picture: 1023 must become 65535 and 512 its exact
        // rescale, round(512 * 65535 / 1023) = 32800, in R, G, B order (the
        // planes are G, B, R).
        let mut planes = [Plane::new(1, 1), Plane::new(1, 1), Plane::new(1, 1)];
        planes[0].set(0, 0, 1023); // G
        planes[1].set(0, 0, 0); // B
        planes[2].set(0, 0, 512); // R
        let layout = Layout {
            width: 1,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        let rgb = identity_to_rgb(&planes, layout, Depths::for_input(10)).unwrap();
        assert_eq!(rgb, [32800, 65535, 0]);
    }

    #[test]
    fn grey_expands_studio_range_and_rescales_depth() {
        let layout = Layout {
            width: 4,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
        };
        let mut plane = Plane::new(4, 1);
        for (x, v) in [0, 16, 126, 235].into_iter().enumerate() {
            plane.set(x, 0, v);
        }
        // Full range at 8 bits is the identity; studio range maps 16..=235
        // onto 0..=255, clamping below 16, with 126 -> round(110*255/219).
        let eight = Depths::for_input(8);
        assert_eq!(
            plane_to_grey(&plane, layout, eight, true),
            [0, 16, 126, 235]
        );
        assert_eq!(
            plane_to_grey(&plane, layout, eight, false),
            [0, 0, 128, 255]
        );
        // A 12-bit full-range maximum lands on the 16-bit maximum.
        let mut wide = Plane::new(4, 1);
        wide.set(0, 0, 4095);
        wide.set(1, 0, 2048);
        let grey = plane_to_grey(&wide, layout, Depths::for_input(12), true);
        assert_eq!(&grey[..2], [65535, 32776]);
    }

    #[test]
    fn upsampling_weighs_the_nearer_chroma_sample_three_to_one() {
        // A 4:2:0 chroma row 0, 64: luma column 1 sits a quarter past chroma 0
        // toward chroma 1, so it reads (3*0 + 64) / 4 = 16; column 2 reads
        // (3*64 + 0) / 4 = 48; the edges replicate.
        let mut plane = Plane::new(2, 1);
        plane.set(1, 0, 64);
        let chroma = Chroma {
            luma_width: 4,
            width: 2,
            height: 1,
            sub_x: 1,
            sub_y: 1,
        };
        let at = |x| chroma.upsample(&plane, x, 0) >> CHROMA_BITS;
        assert_eq!([at(0), at(1), at(2), at(3)], [0, 16, 48, 64]);
    }

    #[test]
    fn an_odd_widths_last_column_copies_its_chroma_as_libyuv_does() {
        // Width 3: chroma 0, 64. Column 2 is centred on chroma 1 with no
        // partner to its right; libyuv copies it, 64, rather than filtering
        // toward chroma 0 (48). Rows keep the filter at an odd height.
        let mut plane = Plane::new(2, 2);
        plane.set(1, 0, 64);
        plane.set(0, 1, 64);
        plane.set(1, 1, 64);
        let chroma = Chroma {
            luma_width: 3,
            width: 2,
            height: 2,
            sub_x: 1,
            sub_y: 1,
        };
        let at = |x, y| chroma.upsample(&plane, x, y) >> CHROMA_BITS;
        assert_eq!([at(0, 0), at(1, 0), at(2, 0)], [0, 16, 64]);
        // Row 2 (the last of an odd height) still filters: 3*64 + 64, /4 = 64
        // at column 2, and (3*64 + 0)/4 = 48 at column 0 from chroma rows 1, 0.
        assert_eq!([at(0, 2), at(2, 2)], [48, 64]);
    }
}
