//! Intra prediction (spec §7.11.2).
//!
//! An intra block is predicted from the already-reconstructed samples in the
//! row above and the column to its left, then the residual is added. This
//! module owns the prediction; it takes the assembled `AboveRow`/`LeftCol`
//! neighbour arrays and the mode, and returns the predicted block. Building the
//! neighbour arrays from the plane (with the frame-edge and availability rules)
//! is the tile driver's job, which keeps these predictors pure and testable.
//!
//! Lossless coding predicts every 4x4 transform block, so 4x4 is what is
//! implemented here. The non-directional modes (DC, Paeth, the three Smooth
//! variants) and the two axis-aligned directional modes (V, H, which at exactly
//! 90 and 180 degrees are plain copies with no edge filtering) are complete.
//! The slanted directional modes need the edge-filter/upsample machinery and
//! report [`PixelsError::unsupported`] until that lands.

use otf_pixels_core::{PixelsError, Result};

/// `Sm_Weights_Tx_4x4` (§9.3): the smooth-prediction interpolation weights.
const SM_WEIGHTS_4: [i32; 4] = [255, 149, 85, 64];

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

/// Predict a 4x4 intra block (§7.11.2 for the 4x4 case).
///
/// # Errors
///
/// Returns [`PixelsError::unsupported`] for the slanted directional modes,
/// which need the edge-filter and upsample machinery not yet implemented.
pub fn predict_intra_4x4(mode: IntraMode, n: &Neighbours, bit_depth: u8) -> Result<[[u16; 4]; 4]> {
    let max = (1_i32 << bit_depth) - 1;
    let clip1 = |v: i32| v.clamp(0, max) as u16;

    // above[j] / left[i] for i, j in 0..4 are the first four of each array.
    let a = |j: usize| n.above.get(j).copied().unwrap_or(0);
    let l = |i: usize| n.left.get(i).copied().unwrap_or(0);

    let mut pred = [[0_u16; 4]; 4];
    match mode {
        IntraMode::Dc => {
            let value = dc_value(n, bit_depth);
            pred = [[value; 4]; 4];
        }
        IntraMode::V => {
            // pAngle 90: a plain copy of the (unfiltered) above row.
            for row in &mut pred {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = clip1(a(j));
                }
            }
        }
        IntraMode::H => {
            // pAngle 180: a plain copy of the left column.
            for (i, row) in pred.iter_mut().enumerate() {
                for cell in row.iter_mut() {
                    *cell = clip1(l(i));
                }
            }
        }
        IntraMode::Paeth => {
            for (i, row) in pred.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let base = a(j) + l(i) - n.corner;
                    let p_left = (base - l(i)).abs();
                    let p_top = (base - a(j)).abs();
                    let p_corner = (base - n.corner).abs();
                    *cell = clip1(if p_left <= p_top && p_left <= p_corner {
                        l(i)
                    } else if p_top <= p_corner {
                        a(j)
                    } else {
                        n.corner
                    });
                }
            }
        }
        IntraMode::Smooth => {
            let wx = SM_WEIGHTS_4;
            let wy = SM_WEIGHTS_4;
            let below_left = l(3);
            let above_right = a(3);
            for (i, row) in pred.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let smooth = wy.get(i).copied().unwrap_or(0) * a(j)
                        + (256 - wy.get(i).copied().unwrap_or(0)) * below_left
                        + wx.get(j).copied().unwrap_or(0) * l(i)
                        + (256 - wx.get(j).copied().unwrap_or(0)) * above_right;
                    *cell = clip1(round2(smooth, 9));
                }
            }
        }
        IntraMode::SmoothV => {
            let w = SM_WEIGHTS_4;
            let below_left = l(3);
            for (i, row) in pred.iter_mut().enumerate() {
                let wi = w.get(i).copied().unwrap_or(0);
                for (j, cell) in row.iter_mut().enumerate() {
                    let smooth = wi * a(j) + (256 - wi) * below_left;
                    *cell = clip1(round2(smooth, 8));
                }
            }
        }
        IntraMode::SmoothH => {
            let w = SM_WEIGHTS_4;
            let above_right = a(3);
            for (i, row) in pred.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let wj = w.get(j).copied().unwrap_or(0);
                    let smooth = wj * l(i) + (256 - wj) * above_right;
                    *cell = clip1(round2(smooth, 8));
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

/// The DC prediction value for a 4x4 block (§7.11.2.5).
fn dc_value(n: &Neighbours, bit_depth: u8) -> u16 {
    let max = (1_i32 << bit_depth) - 1;
    let clip1 = |v: i32| v.clamp(0, max) as u16;
    let left_sum: i32 = n.left.iter().take(4).sum();
    let above_sum: i32 = n.above.iter().take(4).sum();
    match (n.have_left, n.have_above) {
        (true, true) => {
            // (sum + (w + h) / 2) / (w + h) with w = h = 4.
            let sum = left_sum + above_sum + 4;
            (sum / 8) as u16
        }
        (true, false) => clip1((left_sum + 2) >> 2),
        (false, true) => clip1((above_sum + 2) >> 2),
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
}
