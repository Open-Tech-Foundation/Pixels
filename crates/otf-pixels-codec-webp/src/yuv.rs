//! YUV 4:2:0 to RGB the way libwebp does it.
//!
//! RFC 9649 says to convert with BT.601 but lets decoders choose how, so
//! decoders differ by a step here and there. libwebp's choice is the one
//! every WebP viewer shows: "fancy" upsampling, which interpolates each
//! chroma sample from its four nearest neighbours with 9:3:3:1 weights, and
//! a 14-bit fixed-point BT.601 studio-range matrix. Reproducing both exactly
//! makes lossy decode comparable to libwebp byte for byte.

#![allow(
    clippy::indexing_slicing,
    reason = "row and column indices are bounded by the plane dimensions the \
              caller allocated and the half-resolution chroma derived from them"
)]

/// `MultHi`: `(v * coeff) >> 8`.
const fn mult_hi(v: i32, coeff: i32) -> i32 {
    (v * coeff) >> 8
}

/// `VP8Clip8`: a 14-bit fixed-point value to 0..=255.
const fn clip8(v: i32) -> u8 {
    const MASK: i32 = (256 << 6) - 1;
    if v & !MASK == 0 {
        (v >> 6) as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

/// libwebp's `VP8YuvToRgb`.
pub const fn yuv_to_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let (y, u, v) = (y as i32, u as i32, v as i32);
    let y = mult_hi(y, 19077);
    [
        clip8(y + mult_hi(v, 26149) - 14234),
        clip8(y - mult_hi(u, 6419) - mult_hi(v, 13320) + 8708),
        clip8(y + mult_hi(u, 33050) - 17685),
    ]
}

/// Chroma for one luma row: libwebp's `UpsampleRgbLinePair`, for one line of
/// the pair. `near` is the chroma row this line sits closer to, `far` the
/// other; both are `(width + 1) / 2` long. The result is one `(u, v)` per
/// luma sample.
fn upsample_line(
    near_u: &[u8],
    near_v: &[u8],
    far_u: &[u8],
    far_v: &[u8],
    width: usize,
    out: &mut Vec<(u8, u8)>,
) {
    out.clear();
    // Each chroma pair (u, v) is interpolated together, as libwebp packs
    // them into one word; the arithmetic is the same per channel.
    let at = |row: &[u8], i: usize| u32::from(row[i]);
    let channel = |near: &[u8], far: &[u8], out_u: &mut Vec<u32>| {
        out_u.clear();
        let last_pair = (width - 1) >> 1;
        let (mut far_l, mut near_l) = (at(far, 0), at(near, 0));
        out_u.push((3 * near_l + far_l + 2) >> 2);
        for x in 1..=last_pair {
            let (far_c, near_c) = (at(far, x), at(near, x));
            let avg = far_l + far_c + near_l + near_c + 8;
            // The two diagonals of the 2x2 neighbourhood.
            let diag_near = (avg + 2 * (far_c + near_l)) >> 3;
            let diag_far = (avg + 2 * (far_l + near_c)) >> 3;
            out_u.push((diag_far + near_l) >> 1);
            out_u.push((diag_near + near_c) >> 1);
            far_l = far_c;
            near_l = near_c;
        }
        if width & 1 == 0 {
            out_u.push((3 * near_l + far_l + 2) >> 2);
        }
    };
    let (mut us, mut vs) = (Vec::with_capacity(width), Vec::with_capacity(width));
    channel(near_u, far_u, &mut us);
    channel(near_v, far_v, &mut vs);
    out.extend(
        us.iter()
            .zip(&vs)
            .take(width)
            .map(|(&u, &v)| (u as u8, v as u8)),
    );
}

/// Convert `width` x `height` YUV 4:2:0 planes to interleaved RGB (or RGBA,
/// with `alpha` filling the fourth channel).
#[allow(clippy::too_many_arguments, reason = "three planes and their geometry")]
pub fn to_rgb(
    y: &[u8],
    y_stride: usize,
    u: &[u8],
    v: &[u8],
    uv_stride: usize,
    width: usize,
    height: usize,
    alpha: Option<&[u8]>,
) -> Vec<u8> {
    let channels = if alpha.is_some() { 4 } else { 3 };
    let mut out = Vec::with_capacity(width * height * channels);
    let mut chroma = Vec::with_capacity(width);
    let uv_width = width.div_ceil(2);
    fn row(plane: &[u8], stride: usize, width: usize, r: usize) -> &[u8] {
        &plane[r * stride..r * stride + width]
    }
    for line in 0..height {
        // Line 0 and an even height's last line see one chroma row only;
        // otherwise line 2k-1 sits nearer chroma row k-1 and line 2k nearer
        // row k.
        let (near, far) = if line == 0 {
            (0, 0)
        } else if line % 2 == 1 {
            let top = (line - 1) / 2;
            let bottom = (top + 1).min(height.div_ceil(2) - 1);
            (top, bottom)
        } else {
            let bottom = line / 2;
            (bottom, bottom - 1)
        };
        upsample_line(
            row(u, uv_stride, uv_width, near),
            row(v, uv_stride, uv_width, near),
            row(u, uv_stride, uv_width, far),
            row(v, uv_stride, uv_width, far),
            width,
            &mut chroma,
        );
        for (x, &(cu, cv)) in chroma.iter().enumerate() {
            out.extend_from_slice(&yuv_to_rgb(y[line * y_stride + x], cu, cv));
            if let Some(alpha) = alpha {
                out.push(alpha[line * width + x]);
            }
        }
    }
    out
}

/// libwebp's `VP8RGBToY`, with its rounding.
const fn rgb_to_y(r: i32, g: i32, b: i32) -> u8 {
    ((16839 * r + 33059 * g + 6420 * b + (1 << 15) + (16 << 16)) >> 16) as u8
}

/// libwebp's `VP8RGBToU`/`VP8RGBToV` over a sum of four samples.
const fn clip_uv(v: i32) -> u8 {
    let v = (v + (1 << 17) + (128 << 18)) >> 18;
    if v & !0xff == 0 {
        v as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

/// Convert interleaved 8-bit samples (`channels` of 1 grey, 2 grey+alpha,
/// 3 RGB, 4 RGBA) to YUV 4:2:0 planes padded to whole macroblocks, as
/// libwebp's encoder does: Y per pixel, U and V from each 2x2 block's sum
/// (an odd edge sample counted twice), and the padding a copy of the last
/// row and column. Returns the alpha plane too when the input has one.
#[allow(
    clippy::many_single_char_names,
    reason = "the conventional channel names"
)]
pub fn from_rgb(
    pixels: &[u8],
    channels: usize,
    width: usize,
    height: usize,
) -> (crate::vp8::encode::Planes, Option<Vec<u8>>) {
    let rgb = |x: usize, y: usize| -> [i32; 3] {
        let at = (y * width + x) * channels;
        match channels {
            1 | 2 => {
                let g = i32::from(pixels[at]);
                [g, g, g]
            }
            _ => [
                i32::from(pixels[at]),
                i32::from(pixels[at + 1]),
                i32::from(pixels[at + 2]),
            ],
        }
    };
    let (mb_w, mb_h) = (width.div_ceil(16) * 16, height.div_ceil(16) * 16);
    let (uv_w, uv_h) = (mb_w / 2, mb_h / 2);
    let mut y_plane = vec![0_u8; mb_w * mb_h];
    let mut u_plane = vec![0_u8; uv_w * uv_h];
    let mut v_plane = vec![0_u8; uv_w * uv_h];
    for y in 0..mb_h {
        for x in 0..mb_w {
            let [r, g, b] = rgb(x.min(width - 1), y.min(height - 1));
            y_plane[y * mb_w + x] = rgb_to_y(r, g, b);
        }
    }
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    for cy in 0..uv_h {
        for cx in 0..uv_w {
            let (sx, sy) = (cx.min(cw - 1), cy.min(ch - 1));
            let mut sum = [0_i32; 3];
            for dy in 0..2 {
                for dx in 0..2 {
                    let px = rgb((2 * sx + dx).min(width - 1), (2 * sy + dy).min(height - 1));
                    for (s, v) in sum.iter_mut().zip(px) {
                        *s += v;
                    }
                }
            }
            let [r, g, b] = sum;
            u_plane[cy * uv_w + cx] = clip_uv(-9719 * r - 19081 * g + 28800 * b);
            v_plane[cy * uv_w + cx] = clip_uv(28800 * r - 24116 * g - 4684 * b);
        }
    }
    let alpha = matches!(channels, 2 | 4).then(|| {
        (0..width * height)
            .map(|i| pixels[i * channels + channels - 1])
            .collect()
    });
    let planes = crate::vp8::encode::Planes {
        y: y_plane,
        u: u_plane,
        v: v_plane,
        width,
        height,
    };
    (planes, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_yuv_and_back_lands_within_a_step_or_two() {
        for &(r, g, b) in &[(0, 0, 0), (255, 255, 255), (200, 30, 90), (12, 180, 240)] {
            let y = rgb_to_y(r, g, b);
            let u = clip_uv(4 * (-9719 * r - 19081 * g + 28800 * b));
            let v = clip_uv(4 * (28800 * r - 24116 * g - 4684 * b));
            let back = yuv_to_rgb(y, u, v);
            for (a, b) in back.iter().zip([r, g, b]) {
                assert!(
                    (i32::from(*a) - b).abs() <= 3,
                    "{:?} -> {back:?}",
                    (r, g, b)
                );
            }
        }
    }

    #[test]
    fn studio_range_extremes_map_to_black_and_white() {
        assert_eq!(yuv_to_rgb(16, 128, 128), [0, 0, 0]);
        assert_eq!(yuv_to_rgb(235, 128, 128), [255, 255, 255]);
    }

    #[test]
    fn flat_chroma_upsamples_to_itself() {
        let mut out = Vec::new();
        upsample_line(
            &[90, 90, 90],
            &[40, 40, 40],
            &[90, 90, 90],
            &[40, 40, 40],
            5,
            &mut out,
        );
        assert_eq!(out, vec![(90, 40); 5]);
    }
}
