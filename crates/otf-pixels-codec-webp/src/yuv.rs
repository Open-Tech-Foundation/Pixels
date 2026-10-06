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

#[cfg(test)]
mod tests {
    use super::*;

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
