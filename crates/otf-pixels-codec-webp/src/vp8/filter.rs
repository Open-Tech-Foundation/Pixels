//! The VP8 loop filter (RFC 6386 §15), applied to a whole reconstructed frame.
//!
//! Every function takes a plane, the index of the first sample past the edge
//! (`q0`), and `step`, the distance between samples across the edge: 1 to
//! filter a vertical edge, the stride for a horizontal one.

#![allow(
    clippy::indexing_slicing,
    reason = "callers only name edges at least four samples inside an \
              MB-aligned plane, so p3..q3 are always in bounds"
)]

fn at(plane: &[u8], i: usize) -> i32 {
    i32::from(plane[i])
}

const fn s8(v: i32) -> i32 {
    if v < -128 {
        -128
    } else if v > 127 {
        127
    } else {
        v
    }
}

const fn u8c(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v > 255 {
        255
    } else {
        v as u8
    }
}

/// The four samples on each side of the edge: `[p3, p2, p1, p0, q0, q1, q2, q3]`.
fn samples(plane: &[u8], q0: usize, step: usize) -> [i32; 8] {
    [
        at(plane, q0 - 4 * step),
        at(plane, q0 - 3 * step),
        at(plane, q0 - 2 * step),
        at(plane, q0 - step),
        at(plane, q0),
        at(plane, q0 + step),
        at(plane, q0 + 2 * step),
        at(plane, q0 + 3 * step),
    ]
}

fn simple_threshold(s: &[i32; 8], limit: i32) -> bool {
    (s[3] - s[4]).abs() * 2 + ((s[2] - s[5]).abs() >> 1) <= limit
}

fn normal_threshold(s: &[i32; 8], edge: i32, interior: i32) -> bool {
    simple_threshold(s, 2 * edge + interior)
        && (s[0] - s[1]).abs() <= interior
        && (s[1] - s[2]).abs() <= interior
        && (s[2] - s[3]).abs() <= interior
        && (s[7] - s[6]).abs() <= interior
        && (s[6] - s[5]).abs() <= interior
        && (s[5] - s[4]).abs() <= interior
}

fn high_variance(s: &[i32; 8], threshold: i32) -> bool {
    (s[2] - s[3]).abs() > threshold || (s[5] - s[4]).abs() > threshold
}

/// The common adjustment (`filter_common`): `p0`/`q0`, and with the outer
/// taps unused, `p1`/`q1` too.
fn common(plane: &mut [u8], q0: usize, step: usize, s: &[i32; 8], outer_taps: bool) {
    let mut a = 3 * (s[4] - s[3]);
    if outer_taps {
        a += s8(s[2] - s[5]);
    }
    let a = s8(a);
    let f1 = (a + 4).min(127) >> 3;
    let f2 = (a + 3).min(127) >> 3;
    plane[q0 - step] = u8c(s[3] + f2);
    plane[q0] = u8c(s[4] - f1);
    if !outer_taps {
        let a = (f1 + 1) >> 1;
        plane[q0 - 2 * step] = u8c(s[2] + a);
        plane[q0 + step] = u8c(s[5] - a);
    }
}

/// The macroblock-edge adjustment, reaching three samples each side.
fn mb_edge(plane: &mut [u8], q0: usize, step: usize, s: &[i32; 8]) {
    let w = s8(s8(s[2] - s[5]) + 3 * (s[4] - s[3]));
    for (k, weight) in [27, 18, 9].into_iter().enumerate() {
        let a = (weight * w + 63) >> 7;
        plane[q0 - (k + 1) * step] = u8c(s[3 - k] + a);
        plane[q0 + k * step] = u8c(s[4 + k] - a);
    }
}

/// Filter strengths for one macroblock.
#[derive(Debug, Clone, Copy)]
pub struct Strength {
    /// The filter level, `E` for subblock edges (macroblock edges use E + 2).
    pub level: i32,
    /// The interior limit `I`.
    pub interior: i32,
    /// The high-edge-variance threshold.
    pub hev: i32,
}

/// Filter `length` positions along a macroblock edge with the normal filter.
pub fn mb_edge_normal(
    plane: &mut [u8],
    mut q0: usize,
    step: usize,
    along: usize,
    length: usize,
    f: Strength,
) {
    for _ in 0..length {
        let s = samples(plane, q0, step);
        if normal_threshold(&s, f.level + 2, f.interior) {
            if high_variance(&s, f.hev) {
                common(plane, q0, step, &s, true);
            } else {
                mb_edge(plane, q0, step, &s);
            }
        }
        q0 += along;
    }
}

/// Filter `length` positions along a subblock edge with the normal filter.
pub fn subblock_edge_normal(
    plane: &mut [u8],
    mut q0: usize,
    step: usize,
    along: usize,
    length: usize,
    f: Strength,
) {
    for _ in 0..length {
        let s = samples(plane, q0, step);
        if normal_threshold(&s, f.level, f.interior) {
            common(plane, q0, step, &s, high_variance(&s, f.hev));
        }
        q0 += along;
    }
}

/// Filter 16 positions along an edge with the simple filter, at `limit`.
pub fn edge_simple(plane: &mut [u8], mut q0: usize, step: usize, along: usize, limit: i32) {
    for _ in 0..16 {
        let s = samples(plane, q0, step);
        if simple_threshold(&s, limit) {
            common(plane, q0, step, &s, true);
        }
        q0 += along;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_line_is_left_alone_and_a_small_step_is_smoothed() {
        let f = Strength {
            level: 30,
            interior: 10,
            hev: 1,
        };
        let mut flat = [80_u8; 8];
        mb_edge_normal(&mut flat, 4, 1, 0, 1, f);
        assert_eq!(flat, [80; 8]);

        let mut step = [80, 80, 80, 80, 90, 90, 90, 90];
        mb_edge_normal(&mut step, 4, 1, 0, 1, f);
        assert!(step[3] > 80 && step[4] < 90, "{step:?}");
        // A wide filter reaches p2 and q2.
        assert!(step[1] > 80 && step[6] < 90, "{step:?}");
    }

    #[test]
    fn a_large_step_is_an_edge_and_kept() {
        let f = Strength {
            level: 10,
            interior: 5,
            hev: 1,
        };
        let mut edge = [10, 10, 10, 10, 200, 200, 200, 200];
        subblock_edge_normal(&mut edge, 4, 1, 0, 1, f);
        assert_eq!(edge, [10, 10, 10, 10, 200, 200, 200, 200]);
    }
}
