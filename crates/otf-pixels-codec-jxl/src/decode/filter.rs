//! Restoration filters (ISO/IEC 18181-1 §J): Gaborish smoothing and the
//! edge-preserving filter. Follows jxl-oxide's (MIT/Apache-2.0) reference
//! loops; see NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "every index is mirrored into the plane"
)]

use otf_pixels_core::Result;

use super::frame::FramePlanes;
use super::header::{Epf, FrameHeader};
use super::render::mirror;

/// Apply the frame's restoration filters to its colour planes.
///
/// # Errors
///
/// Never, for now; kept fallible for filters to come.
pub fn restore(planes: &mut FramePlanes, frame: &FrameHeader) -> Result<()> {
    if let Some(weights) = frame.gaborish {
        super::render::gaborish(&mut planes.colour, planes.width, planes.height, weights);
    }
    if let Some(epf) = &frame.epf {
        let (w, h) = (planes.width, planes.height);
        let sigma: Box<dyn Fn(usize, usize) -> f32> = match &planes.vardct {
            Some(info) => {
                let (sigma, bw) = (&info.sigma, info.width);
                Box::new(move |x, y| sigma.get((y / 8) * bw + x / 8).copied().unwrap_or(0.0))
            }
            None => {
                let s = epf.sigma_for_modular;
                Box::new(move |_, _| s)
            }
        };
        if epf.iters == 3 {
            step(&mut planes.colour, w, h, epf, &*sigma, 0);
        }
        step(&mut planes.colour, w, h, epf, &*sigma, 1);
        if epf.iters >= 2 {
            step(&mut planes.colour, w, h, epf, &*sigma, 2);
        }
    }
    Ok(())
}

const KERNEL_2: [(isize, isize); 12] = [
    (0, -2),
    (-1, -1),
    (0, -1),
    (1, -1),
    (-2, 0),
    (-1, 0),
    (1, 0),
    (2, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (0, 2),
];
const KERNEL_1: [(isize, isize); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

/// One step of the edge-preserving filter.
fn step(
    planes: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    epf: &Epf,
    sigma: &dyn Fn(usize, usize) -> f32,
    which: usize,
) {
    let kernel: &[(isize, isize)] = if which == 0 { &KERNEL_2 } else { &KERNEL_1 };
    let dist: &[(isize, isize)] = match which {
        0 => &[(0, -1), (1, 0), (0, 0), (-1, 0), (0, 1)],
        1 => &[(0, -1), (0, 0), (0, 1), (-1, 0), (1, 0)],
        _ => &[(0, 0)],
    };
    let step_mul = match which {
        0 => epf.pass0_sigma_scale,
        2 => epf.pass2_sigma_scale,
        _ => 1.0,
    };
    let src = planes.clone();
    let at = |c: usize, x: isize, y: isize| src[c][mirror(y, h) * w + mirror(x, w)];
    for y in 0..h {
        let y_border = (y + 1) & 0b110 == 0;
        for x in 0..w {
            let s = sigma(x, y);
            if s < 0.3 {
                continue;
            }
            let x_border = (x + 1) & 0b110 == 0;
            let sm = if y_border || x_border {
                step_mul * epf.border_sad_mul
            } else {
                step_mul
            };
            let neg_inv_sigma = 6.6 * (std::f32::consts::FRAC_1_SQRT_2 - 1.0) / s * sm;
            let (xi, yi) = (x as isize, y as isize);
            let mut sum_w = 1.0_f32;
            let mut sum = [src[0][y * w + x], src[1][y * w + x], src[2][y * w + x]];
            for &(kx, ky) in kernel {
                let mut d = 0.0_f32;
                for c in 0..3 {
                    let mut acc = 0.0;
                    for &(ix, iy) in dist {
                        acc += (at(c, xi + kx + ix, yi + ky + iy) - at(c, xi + ix, yi + iy)).abs();
                    }
                    d += epf.channel_scale[c] * acc;
                }
                let weight = (1.0 + d * neg_inv_sigma).max(0.0);
                sum_w += weight;
                for (c, v) in sum.iter_mut().enumerate() {
                    *v += weight * at(c, xi + kx, yi + ky);
                }
            }
            for c in 0..3 {
                planes[c][y * w + x] = sum[c] / sum_w;
            }
        }
    }
}
