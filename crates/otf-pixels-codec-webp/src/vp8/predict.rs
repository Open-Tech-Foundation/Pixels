//! VP8 intra prediction (RFC 6386 §12), from explicit edge samples.
//!
//! The reference decoder predicts in place in a frame buffer whose border it
//! rewrites before each macroblock; here each predictor is handed its edges
//! instead — the row above (with the four samples beyond it a subblock
//! needs), the column to the left, and the corner — and the caller applies
//! the frame-edge rules when it gathers them.

#![allow(
    clippy::indexing_slicing,
    reason = "fixed-size edge and block arrays indexed by loop bounds within them"
)]

/// Macroblock-level modes (`DC_PRED`..`TM_PRED`, RFC 6386 §11.2).
pub const DC_PRED: u8 = 0;
/// Each row copies the row above.
pub const V_PRED: u8 = 1;
/// Each column copies the column to the left.
pub const H_PRED: u8 = 2;
/// "TrueMotion": left + above - corner.
pub const TM_PRED: u8 = 3;
/// Sixteen 4x4 subblocks, each with its own mode.
pub const B_PRED: u8 = 4;

/// Subblock modes (`B_DC_PRED`..`B_HU_PRED`, §11.3), in the reference
/// decoder's enum order, which the probability tables are indexed by.
pub mod b {
    /// Average of above and left.
    pub const DC: u8 = 0;
    /// TrueMotion.
    pub const TM: u8 = 1;
    /// Vertical, smoothed.
    pub const VE: u8 = 2;
    /// Horizontal, smoothed.
    pub const HE: u8 = 3;
    /// Down-left diagonal.
    pub const LD: u8 = 4;
    /// Down-right diagonal.
    pub const RD: u8 = 5;
    /// Vertical-right.
    pub const VR: u8 = 6;
    /// Vertical-left.
    pub const VL: u8 = 7;
    /// Horizontal-down.
    pub const HD: u8 = 8;
    /// Horizontal-up.
    pub const HU: u8 = 9;
}

/// Edges for an `N`x`N` macroblock-level prediction.
pub struct Edges<const N: usize> {
    /// The row above.
    pub above: [u8; N],
    /// The column to the left.
    pub left: [u8; N],
    /// The sample above-left.
    pub corner: u8,
    /// Whether the row above lies inside the frame (DC averages only what
    /// does).
    pub have_above: bool,
    /// Whether the column to the left lies inside the frame.
    pub have_left: bool,
}

/// Predict an `N`x`N` block (16 for luma, 8 for chroma) in `mode`.
pub fn predict_block<const N: usize>(mode: u8, e: &Edges<N>) -> [[u8; N]; N] {
    let mut out = [[0_u8; N]; N];
    match mode {
        V_PRED => out.fill(e.above),
        H_PRED => {
            for (row, &l) in out.iter_mut().zip(&e.left) {
                row.fill(l);
            }
        }
        TM_PRED => {
            for (row, &l) in out.iter_mut().zip(&e.left) {
                for (s, &a) in row.iter_mut().zip(&e.above) {
                    *s = (i32::from(l) + i32::from(a) - i32::from(e.corner)).clamp(0, 255) as u8;
                }
            }
        }
        _ => {
            // DC: the frame edges stand in for a missing side by repeating
            // the other, and the very first block predicts 128.
            let shift = N.trailing_zeros();
            let sum = |edge: &[u8; N]| edge.iter().map(|&v| u32::from(v)).sum::<u32>();
            let dc = match (e.have_above, e.have_left) {
                (true, true) => (sum(&e.above) + sum(&e.left) + N as u32) >> (shift + 1),
                (true, false) => (sum(&e.above) + (N as u32 >> 1)) >> shift,
                (false, true) => (sum(&e.left) + (N as u32 >> 1)) >> shift,
                (false, false) => 128,
            };
            out = [[dc as u8; N]; N];
        }
    }
    out
}

/// Edges for a 4x4 subblock: `above[0..4]` and the four beyond it,
/// `left[0..4]`, and the corner. Every subblock edge exists — the frame
/// border supplies 127 above and 129 to the left.
pub struct SubEdges {
    /// The row above and the four samples to its right.
    pub above: [u8; 8],
    /// The column to the left.
    pub left: [u8; 4],
    /// The sample above-left.
    pub corner: u8,
}

fn avg2(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b) + 1) >> 1) as u8
}

fn avg3(a: u8, b: u8, c: u8) -> u8 {
    ((u16::from(a) + 2 * u16::from(b) + u16::from(c) + 2) >> 2) as u8
}

/// Predict a 4x4 subblock in subblock `mode` (§12.3).
#[allow(
    clippy::many_single_char_names,
    reason = "the spec's edge sample names"
)]
pub fn predict_subblock(mode: u8, e: &SubEdges) -> [[u8; 4]; 4] {
    let a = e.above;
    let l = e.left;
    let p = e.corner;
    // The edge as one sequence from the bottom-left up and along the top:
    // l[3], l[2], l[1], l[0], p, a[0], ..., a[7].
    let edge = [
        l[3], l[2], l[1], l[0], p, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
    ];
    let mut out = [[0_u8; 4]; 4];
    match mode {
        b::TM => {
            for (r, row) in out.iter_mut().enumerate() {
                for (c, s) in row.iter_mut().enumerate() {
                    *s = (i32::from(l[r]) + i32::from(a[c]) - i32::from(p)).clamp(0, 255) as u8;
                }
            }
        }
        b::VE => {
            let row = [
                avg3(p, a[0], a[1]),
                avg3(a[0], a[1], a[2]),
                avg3(a[1], a[2], a[3]),
                avg3(a[2], a[3], a[4]),
            ];
            out = [row; 4];
        }
        b::HE => {
            let values = [
                avg3(p, l[0], l[1]),
                avg3(l[0], l[1], l[2]),
                avg3(l[1], l[2], l[3]),
                avg3(l[2], l[3], l[3]),
            ];
            for (row, v) in out.iter_mut().zip(values) {
                *row = [v; 4];
            }
        }
        b::LD => {
            for (r, row) in out.iter_mut().enumerate() {
                for (c, s) in row.iter_mut().enumerate() {
                    let i = r + c;
                    *s = if i == 6 {
                        avg3(a[6], a[7], a[7])
                    } else {
                        avg3(a[i], a[i + 1], a[i + 2])
                    };
                }
            }
        }
        b::RD => {
            // Down-right: sample (r, c) sits at edge index 4 - r + c.
            for (r, row) in out.iter_mut().enumerate() {
                for (c, s) in row.iter_mut().enumerate() {
                    let i = 4 + c - r;
                    *s = avg3(edge[i - 1], edge[i], edge[i + 1]);
                }
            }
        }
        b::VR => {
            out[0] = [
                avg2(p, a[0]),
                avg2(a[0], a[1]),
                avg2(a[1], a[2]),
                avg2(a[2], a[3]),
            ];
            out[1] = [
                avg3(l[0], p, a[0]),
                avg3(p, a[0], a[1]),
                avg3(a[0], a[1], a[2]),
                avg3(a[1], a[2], a[3]),
            ];
            out[2] = [avg3(l[1], l[0], p), out[0][0], out[0][1], out[0][2]];
            out[3] = [avg3(l[2], l[1], l[0]), out[1][0], out[1][1], out[1][2]];
        }
        b::VL => {
            out[0] = [
                avg2(a[0], a[1]),
                avg2(a[1], a[2]),
                avg2(a[2], a[3]),
                avg2(a[3], a[4]),
            ];
            out[1] = [
                avg3(a[0], a[1], a[2]),
                avg3(a[1], a[2], a[3]),
                avg3(a[2], a[3], a[4]),
                avg3(a[3], a[4], a[5]),
            ];
            out[2] = [out[0][1], out[0][2], out[0][3], avg3(a[4], a[5], a[6])];
            out[3] = [out[1][1], out[1][2], out[1][3], avg3(a[5], a[6], a[7])];
        }
        b::HD => {
            let p0 = avg2(l[0], p);
            let p1 = avg3(l[0], p, a[0]);
            let p4 = avg2(l[1], l[0]);
            let p5 = avg3(l[1], l[0], p);
            let p6 = avg2(l[2], l[1]);
            let p7 = avg3(l[2], l[1], l[0]);
            out[0] = [p0, p1, avg3(p, a[0], a[1]), avg3(a[0], a[1], a[2])];
            out[1] = [p4, p5, p0, p1];
            out[2] = [p6, p7, p4, p5];
            out[3] = [avg2(l[3], l[2]), avg3(l[3], l[2], l[1]), p6, p7];
        }
        b::HU => {
            let p0 = avg2(l[0], l[1]);
            let p1 = avg3(l[0], l[1], l[2]);
            let p2 = avg2(l[1], l[2]);
            let p3 = avg3(l[1], l[2], l[3]);
            let p4 = avg2(l[2], l[3]);
            let p5 = avg3(l[2], l[3], l[3]);
            let p6 = l[3];
            out[0] = [p0, p1, p2, p3];
            out[1] = [p2, p3, p4, p5];
            out[2] = [p4, p5, p6, p6];
            out[3] = [p6; 4];
        }
        _ => {
            let sum: u32 = a[..4].iter().chain(&l).map(|&v| u32::from(v)).sum();
            out = [[((sum + 4) >> 3) as u8; 4]; 4];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edges() -> SubEdges {
        SubEdges {
            above: [10, 20, 30, 40, 50, 60, 70, 80],
            left: [11, 21, 31, 41],
            corner: 5,
        }
    }

    #[test]
    fn subblock_predictors_match_the_reference_formulas() {
        let e = edges();
        // RD row 0 col 0: (left[0] + 2*corner + above[0] + 2) >> 2.
        assert_eq!(
            predict_subblock(b::RD, &e)[0][0],
            ((11 + 2 * 5 + 10 + 2) >> 2) as u8
        );
        // RD row 3 col 0: (l[3] + 2*l[2] + l[1] + 2) >> 2.
        assert_eq!(
            predict_subblock(b::RD, &e)[3][0],
            ((41 + 62 + 21 + 2) >> 2) as u8
        );
        // LD last sample: (a[6] + 2*a[7] + a[7] + 2) >> 2.
        assert_eq!(
            predict_subblock(b::LD, &e)[3][3],
            ((70 + 160 + 80 + 2) >> 2) as u8
        );
        // HE bottom row repeats l[3]: (l[2] + 2*l[3] + l[3] + 2) >> 2.
        assert_eq!(
            predict_subblock(b::HE, &e)[3],
            [((31 + 82 + 41 + 2) >> 2) as u8; 4]
        );
        // HU fills with l[3] past the diagonal.
        assert_eq!(predict_subblock(b::HU, &e)[3], [41; 4]);
        // DC over the four above and four left.
        let dc = ((10 + 20 + 30 + 40 + 11 + 21 + 31 + 41 + 4) >> 3) as u8;
        assert_eq!(predict_subblock(b::DC, &e), [[dc; 4]; 4]);
    }

    #[test]
    fn dc_at_the_frame_edge_uses_whichever_side_exists() {
        let mut e = Edges::<16> {
            above: [100; 16],
            left: [60; 16],
            corner: 0,
            have_above: false,
            have_left: false,
        };
        assert_eq!(predict_block(DC_PRED, &e)[0][0], 128);
        e.have_above = true;
        assert_eq!(predict_block(DC_PRED, &e)[5][5], 100);
        e.have_left = true;
        assert_eq!(predict_block(DC_PRED, &e)[5][5], 80);
    }
}
