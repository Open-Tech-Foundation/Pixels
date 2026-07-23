//! Intra prediction (spec §7.11.2).
//!
//! An intra block is predicted from the already-reconstructed samples in the
//! row above and the column to its left, then the residual is added. This
//! module owns the prediction; it takes the assembled `AboveRow`/`LeftCol`
//! neighbour arrays and the mode, and returns the predicted block. Building the
//! neighbour arrays from the plane (with the frame-edge and availability rules)
//! is the tile driver's job, which keeps these predictors pure and testable.
//!
//! The modes that need no edge filtering are implemented at any transform size
//! ([`predict_intra_block`]): DC (§7.11.2.5), Paeth (§7.11.2.2), the three
//! Smooth variants (§7.11.2.6), and the two axis-aligned directional modes (V,
//! H, which at exactly 90 and 180 degrees are plain copies). The slanted
//! directional modes need the edge-filter/upsample machinery and report
//! [`PixelsError::unsupported`] until that lands. [`predict_intra_4x4`] is a thin
//! wrapper over the general path, which the lossless 4x4 tile drives.

use otf_pixels_core::{PixelsError, Result};

/// `Sm_Weights_Tx_4x4` (§9.3): the smooth-prediction interpolation weights.
const SM_WEIGHTS_4: [i32; 4] = [255, 149, 85, 64];
/// `Sm_Weights_Tx_8x8` (§9.3).
const SM_WEIGHTS_8: [i32; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
/// `Sm_Weights_Tx_16x16` (§9.3).
const SM_WEIGHTS_16: [i32; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
/// `Sm_Weights_Tx_32x32` (§9.3).
const SM_WEIGHTS_32: [i32; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
/// `Sm_Weights_Tx_64x64` (§9.3).
const SM_WEIGHTS_64: [i32; 64] = [
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133,
    127, 121, 116, 111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38,
    35, 32, 29, 27, 25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

/// `Sm_Weights_Tx[dim]` (§9.3): the interpolation weights for one side length.
fn sm_weights(dim: usize) -> &'static [i32] {
    match dim {
        8 => &SM_WEIGHTS_8,
        16 => &SM_WEIGHTS_16,
        32 => &SM_WEIGHTS_32,
        64 => &SM_WEIGHTS_64,
        _ => &SM_WEIGHTS_4,
    }
}

/// The 13 intra prediction modes (§6.10.2), in their coded order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraMode {
    /// `DC_PRED` (0).
    Dc,
    /// `V_PRED` (1).
    V,
    /// `H_PRED` (2).
    H,
    /// `D45_PRED` (3).
    D45,
    /// `D135_PRED` (4).
    D135,
    /// `D113_PRED` (5).
    D113,
    /// `D157_PRED` (6).
    D157,
    /// `D203_PRED` (7).
    D203,
    /// `D67_PRED` (8).
    D67,
    /// `SMOOTH_PRED` (9).
    Smooth,
    /// `SMOOTH_V_PRED` (10).
    SmoothV,
    /// `SMOOTH_H_PRED` (11).
    SmoothH,
    /// `PAETH_PRED` (12).
    Paeth,
}

impl IntraMode {
    /// The mode for a coded index, or `None` if out of range.
    #[must_use]
    pub fn from_index(index: u8) -> Option<Self> {
        Some(match index {
            0 => Self::Dc,
            1 => Self::V,
            2 => Self::H,
            3 => Self::D45,
            4 => Self::D135,
            5 => Self::D113,
            6 => Self::D157,
            7 => Self::D203,
            8 => Self::D67,
            9 => Self::Smooth,
            10 => Self::SmoothV,
            11 => Self::SmoothH,
            12 => Self::Paeth,
            _ => return None,
        })
    }

    /// Whether this is a directional mode (`is_directional_mode`, §7.11.2): the
    /// eight modes V through D67.
    #[must_use]
    pub fn is_directional(self) -> bool {
        matches!(
            self,
            Self::V
                | Self::H
                | Self::D45
                | Self::D135
                | Self::D113
                | Self::D157
                | Self::D203
                | Self::D67
        )
    }
}

/// `Round2(x, n)` (§4.7).
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// The neighbour samples a 4x4 intra predictor reads: the row above and the
/// column to the left, each `w + h = 8` samples long, plus the shared
/// top-left corner. Assembled by the tile driver per §7.11.2 general process.
#[derive(Debug, Clone, Copy)]
pub struct Neighbours {
    /// `AboveRow[0..8]`.
    pub above: [i32; 8],
    /// `LeftCol[0..8]`.
    pub left: [i32; 8],
    /// `AboveRow[-1]` (equal to `LeftCol[-1]`).
    pub corner: i32,
    /// Whether the above row holds real reconstructed samples.
    pub have_above: bool,
    /// Whether the left column holds real reconstructed samples.
    pub have_left: bool,
}

/// One intra block's assembled neighbours and geometry: the row above, the
/// column to the left, the shared top-left `corner` (`AboveRow[-1]`), the
/// availability flags, and the block size `w` x `h` in samples. `above` must
/// hold at least `w` entries and `left` at least `h`.
#[derive(Debug, Clone, Copy)]
pub struct PredBlock<'a> {
    /// `AboveRow[0..w]`.
    pub above: &'a [i32],
    /// `LeftCol[0..h]`.
    pub left: &'a [i32],
    /// `AboveRow[-1]` (equal to `LeftCol[-1]`).
    pub corner: i32,
    /// Whether the above row holds real reconstructed samples.
    pub have_above: bool,
    /// Whether the left column holds real reconstructed samples.
    pub have_left: bool,
    /// The block width in samples.
    pub w: usize,
    /// The block height in samples.
    pub h: usize,
}

/// Predict an intra block of any size for the modes that need no edge filtering
/// (§7.11.2): DC, Paeth, the three Smooth variants, and the axis-aligned V/H
/// copies. The result is `w * h` samples in row-major order.
///
/// # Errors
///
/// Returns [`PixelsError::unsupported`] for the slanted directional modes, which
/// need the edge-filter and upsample machinery not yet implemented.
pub fn predict_intra_block(mode: IntraMode, b: &PredBlock<'_>, bit_depth: u8) -> Result<Vec<u16>> {
    let (w, h) = (b.w, b.h);
    let max = (1_i32 << bit_depth) - 1;
    let clip1 = |v: i32| v.clamp(0, max) as u16;
    let a = |j: usize| b.above.get(j).copied().unwrap_or(0);
    let l = |i: usize| b.left.get(i).copied().unwrap_or(0);

    let mut pred = vec![0_u16; w * h];
    let put = |pred: &mut Vec<u16>, i: usize, j: usize, v: u16| {
        if let Some(cell) = pred.get_mut(i * w + j) {
            *cell = v;
        }
    };

    match mode {
        IntraMode::Dc => {
            let value = dc_value(b, bit_depth);
            pred.fill(value);
        }
        IntraMode::V => {
            // pAngle 90: a plain copy of the (unfiltered) above row.
            for i in 0..h {
                for j in 0..w {
                    put(&mut pred, i, j, clip1(a(j)));
                }
            }
        }
        IntraMode::H => {
            // pAngle 180: a plain copy of the left column.
            for i in 0..h {
                for j in 0..w {
                    put(&mut pred, i, j, clip1(l(i)));
                }
            }
        }
        IntraMode::Paeth => {
            for i in 0..h {
                for j in 0..w {
                    let base = a(j) + l(i) - b.corner;
                    let p_left = (base - l(i)).abs();
                    let p_top = (base - a(j)).abs();
                    let p_corner = (base - b.corner).abs();
                    let v = if p_left <= p_top && p_left <= p_corner {
                        l(i)
                    } else if p_top <= p_corner {
                        a(j)
                    } else {
                        b.corner
                    };
                    put(&mut pred, i, j, clip1(v));
                }
            }
        }
        IntraMode::Smooth => {
            let wx = sm_weights(w);
            let wy = sm_weights(h);
            let below_left = l(h - 1);
            let above_right = a(w - 1);
            for i in 0..h {
                let wyi = wy.get(i).copied().unwrap_or(0);
                for j in 0..w {
                    let wxj = wx.get(j).copied().unwrap_or(0);
                    let smooth = wyi * a(j)
                        + (256 - wyi) * below_left
                        + wxj * l(i)
                        + (256 - wxj) * above_right;
                    put(&mut pred, i, j, clip1(round2(smooth, 9)));
                }
            }
        }
        IntraMode::SmoothV => {
            let wy = sm_weights(h);
            let below_left = l(h - 1);
            for i in 0..h {
                let wyi = wy.get(i).copied().unwrap_or(0);
                for j in 0..w {
                    let smooth = wyi * a(j) + (256 - wyi) * below_left;
                    put(&mut pred, i, j, clip1(round2(smooth, 8)));
                }
            }
        }
        IntraMode::SmoothH => {
            let wx = sm_weights(w);
            let above_right = a(w - 1);
            for i in 0..h {
                for j in 0..w {
                    let wxj = wx.get(j).copied().unwrap_or(0);
                    let smooth = wxj * l(i) + (256 - wxj) * above_right;
                    put(&mut pred, i, j, clip1(round2(smooth, 8)));
                }
            }
        }
        IntraMode::D45
        | IntraMode::D135
        | IntraMode::D113
        | IntraMode::D157
        | IntraMode::D203
        | IntraMode::D67 => {
            return Err(PixelsError::unsupported(
                "avif: slanted directional intra prediction is not implemented yet",
            ));
        }
    }
    Ok(pred)
}

/// Predict a 4x4 intra block (§7.11.2 for the 4x4 case): a thin wrapper over the
/// size-general [`predict_intra_block`], reshaping the flat result to `[[_; 4];
/// 4]`.
///
/// # Errors
///
/// Returns [`PixelsError::unsupported`] for the slanted directional modes.
pub fn predict_intra_4x4(mode: IntraMode, n: &Neighbours, bit_depth: u8) -> Result<[[u16; 4]; 4]> {
    let block = PredBlock {
        above: &n.above,
        left: &n.left,
        corner: n.corner,
        have_above: n.have_above,
        have_left: n.have_left,
        w: 4,
        h: 4,
    };
    let flat = predict_intra_block(mode, &block, bit_depth)?;
    let mut pred = [[0_u16; 4]; 4];
    for (i, row) in pred.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = flat.get(i * 4 + j).copied().unwrap_or(0);
        }
    }
    Ok(pred)
}

/// The DC prediction value (§7.11.2.5) for a block of any size.
fn dc_value(b: &PredBlock<'_>, bit_depth: u8) -> u16 {
    let max = (1_i32 << bit_depth) - 1;
    let clip1 = |v: i32| v.clamp(0, max) as u16;
    let (w, h) = (b.w, b.h);
    let left_sum: i32 = b.left.iter().take(h).sum();
    let above_sum: i32 = b.above.iter().take(w).sum();
    match (b.have_left, b.have_above) {
        (true, true) => {
            // (sum + (w + h) / 2) / (w + h); already in range, so no Clip1.
            let sum = left_sum + above_sum + ((w + h) >> 1) as i32;
            (sum / (w + h) as i32) as u16
        }
        (true, false) => clip1((left_sum + (h >> 1) as i32) >> h.trailing_zeros()),
        (false, true) => clip1((above_sum + (w >> 1) as i32) >> w.trailing_zeros()),
        (false, false) => 1_u16 << (bit_depth - 1),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    fn neighbours(above: [i32; 8], left: [i32; 8], corner: i32) -> Neighbours {
        Neighbours {
            above,
            left,
            corner,
            have_above: true,
            have_left: true,
        }
    }

    #[test]
    fn mode_indexing_and_directional_classification() {
        assert_eq!(IntraMode::from_index(0), Some(IntraMode::Dc));
        assert_eq!(IntraMode::from_index(12), Some(IntraMode::Paeth));
        assert_eq!(IntraMode::from_index(13), None);
        assert!(IntraMode::V.is_directional());
        assert!(!IntraMode::Dc.is_directional());
        assert!(!IntraMode::Smooth.is_directional());
        assert!(IntraMode::D45.is_directional());
    }

    #[test]
    fn dc_with_no_neighbours_is_the_midpoint() {
        let n = Neighbours {
            above: [0; 8],
            left: [0; 8],
            corner: 0,
            have_above: false,
            have_left: false,
        };
        let pred = predict_intra_4x4(IntraMode::Dc, &n, 8).unwrap();
        assert_eq!(pred, [[128; 4]; 4]);
    }

    #[test]
    fn dc_averages_both_edges() {
        // Above all 100, left all 60: avg = (400 + 240 + 4) / 8 = 80.
        let n = neighbours([100; 8], [60; 8], 100);
        let pred = predict_intra_4x4(IntraMode::Dc, &n, 8).unwrap();
        assert_eq!(pred, [[80; 4]; 4]);
    }

    #[test]
    fn v_copies_the_above_row_down_each_column() {
        let n = neighbours([10, 20, 30, 40, 0, 0, 0, 0], [99; 8], 5);
        let pred = predict_intra_4x4(IntraMode::V, &n, 8).unwrap();
        for row in &pred {
            assert_eq!(row, &[10, 20, 30, 40]);
        }
    }

    #[test]
    fn h_copies_the_left_column_across_each_row() {
        let n = neighbours([99; 8], [10, 20, 30, 40, 0, 0, 0, 0], 5);
        let pred = predict_intra_4x4(IntraMode::H, &n, 8).unwrap();
        for (i, row) in pred.iter().enumerate() {
            assert!(row.iter().all(|&v| v == [10, 20, 30, 40][i]));
        }
    }

    #[test]
    fn paeth_picks_the_closest_predictor() {
        // Flat above=left=corner=50 -> base=50, all distances 0, picks left=50.
        let n = neighbours([50; 8], [50; 8], 50);
        let pred = predict_intra_4x4(IntraMode::Paeth, &n, 8).unwrap();
        assert_eq!(pred, [[50; 4]; 4]);
    }

    #[test]
    fn smooth_of_a_flat_edge_is_that_value() {
        // All neighbours 128: every weighted combination is 128.
        let n = neighbours([128; 8], [128; 8], 128);
        for mode in [IntraMode::Smooth, IntraMode::SmoothV, IntraMode::SmoothH] {
            let pred = predict_intra_4x4(mode, &n, 8).unwrap();
            assert_eq!(pred, [[128; 4]; 4], "mode {mode:?}");
        }
    }

    #[test]
    fn slanted_directional_modes_are_unsupported_for_now() {
        let n = neighbours([100; 8], [100; 8], 100);
        assert!(predict_intra_4x4(IntraMode::D45, &n, 8).is_err());
    }

    #[test]
    fn dc_averages_both_edges_at_a_rectangular_size() {
        // 8 wide, 4 tall. Above all 100 (8 samples), left all 60 (4 samples):
        // sum = 800 + 240 + ((8+4)>>1) = 1046; avg = 1046 / 12 = 87.
        let above = [100; 8];
        let left = [60; 8];
        let b = PredBlock {
            above: &above,
            left: &left,
            corner: 100,
            have_above: true,
            have_left: true,
            w: 8,
            h: 4,
        };
        let pred = predict_intra_block(IntraMode::Dc, &b, 8).unwrap();
        assert_eq!(pred.len(), 32);
        assert!(pred.iter().all(|&v| v == 87));
    }

    #[test]
    fn smooth_of_a_flat_edge_is_that_value_at_8x8() {
        let above = [128; 8];
        let left = [128; 8];
        for mode in [IntraMode::Smooth, IntraMode::SmoothV, IntraMode::SmoothH] {
            let b = PredBlock {
                above: &above,
                left: &left,
                corner: 128,
                have_above: true,
                have_left: true,
                w: 8,
                h: 8,
            };
            let pred = predict_intra_block(mode, &b, 8).unwrap();
            assert_eq!(pred.len(), 64);
            assert!(pred.iter().all(|&v| v == 128), "mode {mode:?}");
        }
    }

    #[test]
    fn v_copies_the_above_row_at_8x8() {
        let above = [10, 20, 30, 40, 50, 60, 70, 80];
        let left = [0; 8];
        let b = PredBlock {
            above: &above,
            left: &left,
            corner: 0,
            have_above: true,
            have_left: true,
            w: 8,
            h: 8,
        };
        let pred = predict_intra_block(IntraMode::V, &b, 8).unwrap();
        // Every one of the 8 rows repeats the above row.
        let expected: [u16; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        for row in pred.chunks(8) {
            assert_eq!(row, &expected[..]);
        }
    }
}
