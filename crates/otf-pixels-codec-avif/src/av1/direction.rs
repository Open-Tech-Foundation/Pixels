//! Slanted directional intra prediction (spec §7.11.2.4 and its subprocesses).
//!
//! The eight directional modes project the reconstructed edge samples into the
//! block along an angle. Before projection the edge is optionally sharpened (the
//! intra edge filter, §7.11.2.12) and doubled in resolution (the upsample,
//! §7.11.2.11); which of those apply is chosen from the block size, the angle,
//! and whether a neighbouring block used a smooth mode. The projection itself
//! (steps 7–11) interpolates between the two edge samples a fractional angle
//! lands between.
//!
//! The projection ([`predict_directional`]) runs at any transform size;
//! [`predict_directional_4x4`] is a thin wrapper the lossless 4x4 tile drives.
//! The edge arrays are addressed by signed index around a fixed origin so the
//! negative entries (`AboveRow[-1]`, and `[-2]` after upsampling) read naturally
//! without slice indexing.

/// `ANGLE_STEP` (§3): degrees per `angle_delta` unit.
pub const ANGLE_STEP: i32 = 3;
/// `Mode_To_Angle` (§9.3): the base angle of each intra mode, in degrees.
const MODE_TO_ANGLE: [i32; 13] = [0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0];
/// `Intra_Edge_Kernel` (§7.11.2.12): the three edge-filter kernels.
const INTRA_EDGE_KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
/// `Dr_Intra_Derivative` (§9.3): angle-to-slope table, indexed by degrees.
const DR_INTRA_DERIVATIVE: [i32; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0,
    45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0,
    0, 7, 0, 0, 3, 0, 0,
];

/// The base angle of a directional mode in degrees, or `None` if the mode is
/// not directional.
#[must_use]
pub fn mode_base_angle(mode: usize) -> Option<i32> {
    match MODE_TO_ANGLE.get(mode).copied() {
        Some(a) if (1..=8).contains(&mode) => Some(a),
        _ => None,
    }
}

/// One edge array (`AboveRow` or `LeftCol`) addressed by signed index. The
/// origin sits at [`Edge::OFF`], so index -1 (the corner) and -2 (produced by
/// upsampling) are in range. The buffer spans the largest block: a 64x64
/// transform reads up to `AboveRow[w + h - 1]` (index 127) before the offset.
pub struct Edge {
    data: [i32; 320],
}

impl Edge {
    const OFF: isize = 16;

    /// An all-zero edge.
    #[must_use]
    pub fn new() -> Self {
        Self { data: [0; 320] }
    }

    /// The sample at signed index `i` (0 if out of range).
    #[must_use]
    pub fn get(&self, i: isize) -> i32 {
        let idx = i + Self::OFF;
        usize::try_from(idx)
            .ok()
            .and_then(|u| self.data.get(u))
            .copied()
            .unwrap_or(0)
    }

    /// Set the sample at signed index `i`; out-of-range writes are dropped.
    pub fn set(&mut self, i: isize, value: i32) {
        let idx = i + Self::OFF;
        if let Ok(u) = usize::try_from(idx) {
            if let Some(slot) = self.data.get_mut(u) {
                *slot = value;
            }
        }
    }
}

impl Default for Edge {
    fn default() -> Self {
        Self::new()
    }
}

/// Predict a directional block of any size (§7.11.2.4). The result is `w * h`
/// samples in row-major order.
///
/// `above` and `left` hold the raw edge samples: index -1 is the shared corner,
/// indices `0..w+h` the row/column. `avail_above_px`/`avail_left_px` are how far
/// the real frame extends (`maxX - x + 1`, `maxY - y + 1`), which bounds how
/// many samples the edge filter touches. The arrays are consumed (filtered and
/// upsampled in place). The result is the clipped prediction.
#[allow(clippy::too_many_arguments, reason = "mirrors the §7.11.2.4 inputs")]
#[must_use]
pub fn predict_directional(
    p_angle: i32,
    above: &mut Edge,
    left: &mut Edge,
    w: usize,
    h: usize,
    have_left: bool,
    have_above: bool,
    filter_type: bool,
    enable_edge_filter: bool,
    avail_above_px: i32,
    avail_left_px: i32,
    bit_depth: u8,
) -> Vec<u16> {
    let wi = w as i32;
    let hi = h as i32;
    let max = (1_i32 << bit_depth) - 1;
    let clip1 = |v: i32| v.clamp(0, max);

    let mut upsample_above = false;
    let mut upsample_left = false;

    if enable_edge_filter {
        if p_angle != 90 && p_angle != 180 {
            if p_angle > 90 && p_angle < 180 && (wi + hi) >= 24 {
                // Filter corner: three-tap blend written to both corners.
                let s = left.get(0) * 5 + above.get(-1) * 6 + above.get(0) * 5;
                let corner = round2(s, 4);
                above.set(-1, corner);
                left.set(-1, corner);
            }
            if have_above {
                let strength = edge_filter_strength(wi, hi, filter_type, p_angle - 90);
                let num_px = wi.min(avail_above_px) + if p_angle < 90 { hi } else { 0 } + 1;
                intra_edge_filter(above, num_px, strength);
            }
            if have_left {
                let strength = edge_filter_strength(wi, hi, filter_type, p_angle - 180);
                let num_px = hi.min(avail_left_px) + if p_angle > 180 { wi } else { 0 } + 1;
                intra_edge_filter(left, num_px, strength);
            }
        }
        upsample_above = edge_upsample(wi, hi, filter_type, p_angle - 90);
        if upsample_above {
            let num_px = wi + if p_angle < 90 { hi } else { 0 };
            intra_edge_upsample(above, num_px, max);
        }
        upsample_left = edge_upsample(wi, hi, filter_type, p_angle - 180);
        if upsample_left {
            let num_px = hi + if p_angle > 180 { wi } else { 0 };
            intra_edge_upsample(left, num_px, max);
        }
    }

    let ua = i32::from(upsample_above);
    let ul = i32::from(upsample_left);
    let dx = if p_angle < 90 {
        derivative(p_angle)
    } else if p_angle > 90 && p_angle < 180 {
        derivative(180 - p_angle)
    } else {
        0
    };
    let dy = if p_angle > 90 && p_angle < 180 {
        derivative(p_angle - 90)
    } else if p_angle > 180 {
        derivative(270 - p_angle)
    } else {
        0
    };

    let mut pred = vec![0_u16; w * h];
    for i in 0..h {
        let ii = i as i32;
        for j in 0..w {
            let jj = j as i32;
            let value = if p_angle < 90 {
                let idx = (ii + 1) * dx;
                let base = (idx >> (6 - ua)) + (jj << ua);
                let shift = ((idx << ua) >> 1) & 0x1F;
                let max_base_x = (wi + hi - 1) << ua;
                if base < max_base_x {
                    round2(
                        above.get(base as isize) * (32 - shift)
                            + above.get(base as isize + 1) * shift,
                        5,
                    )
                } else {
                    above.get(max_base_x as isize)
                }
            } else if p_angle > 90 && p_angle < 180 {
                let idx = (jj << 6) - (ii + 1) * dx;
                let base = idx >> (6 - ua);
                if base >= -(1 << ua) {
                    let shift = ((idx << ua) >> 1) & 0x1F;
                    round2(
                        above.get(base as isize) * (32 - shift)
                            + above.get(base as isize + 1) * shift,
                        5,
                    )
                } else {
                    let idy = (ii << 6) - (jj + 1) * dy;
                    let base = idy >> (6 - ul);
                    let shift = ((idy << ul) >> 1) & 0x1F;
                    round2(
                        left.get(base as isize) * (32 - shift)
                            + left.get(base as isize + 1) * shift,
                        5,
                    )
                }
            } else if p_angle > 180 {
                let idx = (jj + 1) * dy;
                let base = (idx >> (6 - ul)) + (ii << ul);
                let shift = ((idx << ul) >> 1) & 0x1F;
                round2(
                    left.get(base as isize) * (32 - shift) + left.get(base as isize + 1) * shift,
                    5,
                )
            } else if p_angle == 90 {
                above.get(jj as isize)
            } else {
                left.get(ii as isize)
            };
            if let Some(cell) = pred.get_mut(i * w + j) {
                *cell = clip1(value) as u16;
            }
        }
    }
    pred
}

/// Predict a 4x4 directional block (§7.11.2.4): a thin wrapper over the
/// size-general [`predict_directional`], reshaping the flat result. Retained for
/// the unit tests; the tile drives the general path directly.
#[cfg(test)]
#[allow(clippy::too_many_arguments, reason = "mirrors the §7.11.2.4 inputs")]
#[must_use]
pub fn predict_directional_4x4(
    p_angle: i32,
    above: &mut Edge,
    left: &mut Edge,
    have_left: bool,
    have_above: bool,
    filter_type: bool,
    enable_edge_filter: bool,
    avail_above_px: i32,
    avail_left_px: i32,
    bit_depth: u8,
) -> [[u16; 4]; 4] {
    let flat = predict_directional(
        p_angle,
        above,
        left,
        4,
        4,
        have_left,
        have_above,
        filter_type,
        enable_edge_filter,
        avail_above_px,
        avail_left_px,
        bit_depth,
    );
    let mut pred = [[0_u16; 4]; 4];
    for (i, row) in pred.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = flat.get(i * 4 + j).copied().unwrap_or(0);
        }
    }
    pred
}

/// `INTRA_FILTER_SCALE_BITS` (§3).
const INTRA_FILTER_SCALE_BITS: u32 = 4;

/// `Intra_Filter_Taps` (§9.3): the recursive filter-intra kernels,
/// `[mode][4x2 sub-position][7 taps]`.
const INTRA_FILTER_TAPS: [[[i32; 7]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0],
        [-5, 2, 10, 0, 0, 9, 0],
        [-3, 1, 1, 10, 0, 7, 0],
        [-3, 1, 1, 2, 10, 5, 0],
        [-4, 6, 0, 0, 0, 2, 12],
        [-3, 2, 6, 0, 0, 2, 9],
        [-3, 2, 2, 6, 0, 2, 7],
        [-3, 1, 2, 2, 6, 3, 5],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 4, 0],
        [-2, 0, 0, 0, 16, 2, 0],
        [-10, 16, 0, 0, 0, 0, 10],
        [-6, 0, 16, 0, 0, 0, 6],
        [-4, 0, 0, 16, 0, 0, 4],
        [-2, 0, 0, 0, 16, 0, 2],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0],
        [-8, 0, 8, 0, 0, 16, 0],
        [-8, 0, 0, 8, 0, 16, 0],
        [-8, 0, 0, 0, 8, 16, 0],
        [-4, 4, 0, 0, 0, 0, 16],
        [-4, 0, 4, 0, 0, 0, 16],
        [-4, 0, 0, 4, 0, 0, 16],
        [-4, 0, 0, 0, 4, 0, 16],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0],
        [-1, 3, 8, 0, 0, 6, 0],
        [-1, 2, 3, 8, 0, 4, 0],
        [0, 1, 2, 3, 8, 2, 0],
        [-1, 4, 0, 0, 0, 3, 10],
        [-1, 3, 4, 0, 0, 4, 6],
        [-1, 2, 3, 4, 0, 4, 4],
        [-1, 2, 2, 3, 4, 3, 3],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0],
        [-10, 0, 14, 0, 0, 12, 0],
        [-9, 0, 0, 14, 0, 11, 0],
        [-8, 0, 0, 0, 14, 10, 0],
        [-10, 12, 0, 0, 0, 0, 14],
        [-9, 1, 12, 0, 0, 0, 12],
        [-8, 0, 0, 12, 0, 1, 11],
        [-7, 0, 0, 1, 12, 1, 9],
    ],
];

/// Recursive filter-intra prediction for a block of any size (§7.11.2.3).
/// `above` and `left` hold the raw edge samples (index -1 is the corner). Every
/// 4x2 sub-block is a seven-tap filter of its neighbours, walked left-to-right
/// then top-to-bottom so later sub-blocks read the samples earlier ones
/// produced. The result is `w * h` samples in row-major order.
#[must_use]
pub fn predict_filter_intra(
    filter_mode: usize,
    above: &Edge,
    left: &Edge,
    w: usize,
    h: usize,
    bit_depth: u8,
) -> Vec<u16> {
    let max = (1_i32 << bit_depth) - 1;
    let taps = INTRA_FILTER_TAPS
        .get(filter_mode.min(4))
        .copied()
        .unwrap_or(INTRA_FILTER_TAPS[0]);
    // pred as i32 so intermediate rows feed the next sub-block exactly.
    let mut pred = vec![0_i32; w * h];
    // Read pred[row][col] with signed indices (0 outside the block).
    let get = |pred: &[i32], row: isize, col: isize| -> i32 {
        match (usize::try_from(row), usize::try_from(col)) {
            (Ok(r), Ok(c)) if r < h && c < w => pred.get(r * w + c).copied().unwrap_or(0),
            _ => 0,
        }
    };
    let w4 = w >> 2;
    let h2 = h >> 1;
    for i2 in 0..h2 {
        for j4 in 0..w4 {
            let mut p = [0_i32; 7];
            for (i, slot) in p.iter_mut().enumerate() {
                let ii = i as isize;
                let base_x = (j4 << 2) as isize;
                let base_y = (i2 << 1) as isize;
                *slot = if i < 5 {
                    if i2 == 0 {
                        above.get(base_x + ii - 1)
                    } else if j4 == 0 && i == 0 {
                        left.get(base_y - 1)
                    } else {
                        get(&pred, base_y - 1, base_x + ii - 1)
                    }
                } else if j4 == 0 {
                    left.get(base_y + ii - 5)
                } else {
                    get(&pred, base_y + ii - 5, base_x - 1)
                };
            }
            for i1 in 0..2 {
                for j1 in 0..4 {
                    let row = taps.get((i1 << 2) + j1).copied().unwrap_or([0; 7]);
                    let mut pr = 0;
                    for (tap, &pv) in row.iter().zip(p.iter()) {
                        pr += tap * pv;
                    }
                    let value = round2_signed(pr, INTRA_FILTER_SCALE_BITS).clamp(0, max);
                    let r = (i2 << 1) + i1;
                    let c = (j4 << 2) + j1;
                    if let Some(slot) = pred.get_mut(r * w + c) {
                        *slot = value;
                    }
                }
            }
        }
    }
    pred.iter().map(|&v| v.clamp(0, max) as u16).collect()
}

/// Recursive filter-intra prediction for a 4x4 block (§7.11.2.3): a thin wrapper
/// over the size-general [`predict_filter_intra`], reshaping the flat result.
/// Retained for the unit tests; the tile drives the general path directly.
#[cfg(test)]
#[must_use]
pub fn predict_filter_intra_4x4(
    filter_mode: usize,
    above: &Edge,
    left: &Edge,
    bit_depth: u8,
) -> [[u16; 4]; 4] {
    let flat = predict_filter_intra(filter_mode, above, left, 4, 4, bit_depth);
    let mut out = [[0_u16; 4]; 4];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = flat.get(i * 4 + j).copied().unwrap_or(0);
        }
    }
    out
}

/// `Round2Signed(x, n)` (§4.7).
fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 { round2(x, n) } else { -round2(-x, n) }
}

/// `Round2(x, n)` (§4.7).
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// `Dr_Intra_Derivative[angle]` with a safe fallback.
fn derivative(angle: i32) -> i32 {
    usize::try_from(angle)
        .ok()
        .and_then(|a| DR_INTRA_DERIVATIVE.get(a))
        .copied()
        .unwrap_or(0)
}

/// Intra edge filter strength selection (§7.11.2.9).
#[allow(
    clippy::if_same_then_else,
    reason = "the size buckets are transcribed verbatim from the spec table, some coincide"
)]
fn edge_filter_strength(w: i32, h: i32, filter_type: bool, delta: i32) -> usize {
    let d = delta.abs();
    let blk = w + h;
    let mut strength = 0;
    if !filter_type {
        if blk <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk <= 12 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk <= 32 {
            strength = 1;
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else {
            strength = 3;
        }
    } else if blk <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else {
        strength = 3;
    }
    strength
}

/// Intra edge upsample selection (§7.11.2.10).
fn edge_upsample(w: i32, h: i32, filter_type: bool, delta: i32) -> bool {
    let d = delta.abs();
    let blk = w + h;
    if d <= 0 || d >= 40 {
        false
    } else if !filter_type {
        blk <= 16
    } else {
        blk <= 8
    }
}

/// Intra edge filter (§7.11.2.12): a five-tap smoothing of `edge[0..sz]`,
/// where `edge[i] == buf[i - 1]`, writing back to `buf[i - 1]`.
fn intra_edge_filter(buf: &mut Edge, sz: i32, strength: usize) {
    if strength == 0 || sz <= 0 {
        return;
    }
    let Some(kernel) = INTRA_EDGE_KERNEL.get(strength - 1) else {
        return;
    };
    // Snapshot edge[i] = buf[i - 1] before modifying in place.
    let edge: Vec<i32> = (0..sz)
        .map(|i| buf.get(isize::try_from(i - 1).unwrap_or(0)))
        .collect();
    for i in 1..sz {
        let mut s = 0;
        for (j, &tap) in kernel.iter().enumerate() {
            let k = (i - 2 + j as i32).clamp(0, sz - 1);
            s += tap * edge.get(k as usize).copied().unwrap_or(0);
        }
        buf.set(isize::try_from(i - 1).unwrap_or(0), (s + 8) >> 4);
    }
}

/// Intra edge upsample (§7.11.2.11): doubles the edge resolution in place. On
/// entry entries -1..numPx-1 are valid; on exit -2..2*numPx-2 are valid.
fn intra_edge_upsample(buf: &mut Edge, num_px: i32, max: i32) {
    if num_px <= 0 {
        return;
    }
    let clip1 = |v: i32| v.clamp(0, max);
    // dup has length numPx + 3, extending buf by one at each end.
    let mut dup = vec![0_i32; (num_px + 3) as usize];
    if let Some(first) = dup.first_mut() {
        *first = buf.get(-1);
    }
    for i in -1..num_px {
        if let Some(slot) = dup.get_mut((i + 2) as usize) {
            *slot = buf.get(i as isize);
        }
    }
    if let Some(last) = dup.get_mut((num_px + 2) as usize) {
        *last = buf.get((num_px - 1) as isize);
    }

    buf.set(-2, dup.first().copied().unwrap_or(0));
    for i in 0..num_px {
        let d0 = dup.get(i as usize).copied().unwrap_or(0);
        let d1 = dup.get((i + 1) as usize).copied().unwrap_or(0);
        let d2 = dup.get((i + 2) as usize).copied().unwrap_or(0);
        let d3 = dup.get((i + 3) as usize).copied().unwrap_or(0);
        let s = clip1(round2(-d0 + 9 * d1 + 9 * d2 - d3, 4));
        buf.set((2 * i - 1) as isize, s);
        buf.set((2 * i) as isize, d2);
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

    #[test]
    fn vertical_angle_copies_the_above_row() {
        // pAngle 90 with the edge filter off is a plain copy of AboveRow.
        let mut above = Edge::new();
        let mut left = Edge::new();
        for i in 0..8 {
            above.set(i, 10 + i as i32 * 5);
        }
        let pred =
            predict_directional_4x4(90, &mut above, &mut left, true, true, false, false, 4, 4, 8);
        for row in &pred {
            assert_eq!(row, &[10, 15, 20, 25]);
        }
    }

    #[test]
    fn horizontal_angle_copies_the_left_column() {
        let mut above = Edge::new();
        let mut left = Edge::new();
        for i in 0..8 {
            left.set(i, 20 + i as i32 * 3);
        }
        let pred = predict_directional_4x4(
            180, &mut above, &mut left, true, true, false, false, 4, 4, 8,
        );
        for (i, row) in pred.iter().enumerate() {
            assert!(row.iter().all(|&v| v == 20 + i as u16 * 3));
        }
    }

    #[test]
    fn diagonal_45_projects_the_above_row() {
        // pAngle 45, no edge tools: dx = Dr_Intra_Derivative[45] = 64, so
        // idx = (i+1)*64, base = idx>>6 = i+1, shift 0 -> pred[i][j]=above[i+j+1].
        let mut above = Edge::new();
        let mut left = Edge::new();
        for i in 0..8 {
            above.set(i, i as i32);
        }
        let pred =
            predict_directional_4x4(45, &mut above, &mut left, true, true, false, false, 4, 4, 8);
        assert_eq!(pred[0], [1, 2, 3, 4]);
        assert_eq!(pred[1], [2, 3, 4, 5]);
    }

    #[test]
    fn vertical_angle_copies_the_above_row_at_8x8() {
        // pAngle 90 at 8x8 with the edge filter off: every row copies AboveRow.
        let mut above = Edge::new();
        let mut left = Edge::new();
        for i in 0..16 {
            above.set(i, 5 + i as i32);
        }
        let pred = predict_directional(
            90, &mut above, &mut left, 8, 8, true, true, false, false, 8, 8, 8,
        );
        assert_eq!(pred.len(), 64);
        let expected: [u16; 8] = [5, 6, 7, 8, 9, 10, 11, 12];
        for row in pred.chunks(8) {
            assert_eq!(row, &expected[..]);
        }
    }

    #[test]
    fn filter_intra_of_a_flat_edge_is_that_value() {
        // Every tap row of Intra_Filter_Taps sums to 16 (= 1 << SCALE_BITS), so a
        // constant edge reproduces that constant. Check at 4x4 and 8x8.
        let mut above = Edge::new();
        let mut left = Edge::new();
        for i in -1..16 {
            above.set(i, 100);
            left.set(i, 100);
        }
        let flat = predict_filter_intra(2, &above, &left, 8, 8, 8);
        assert_eq!(flat.len(), 64);
        assert!(flat.iter().all(|&v| v == 100));
        // The 4x4 wrapper agrees with the general path.
        let out = predict_filter_intra_4x4(2, &above, &left, 8);
        assert_eq!(out, [[100; 4]; 4]);
    }

    #[test]
    fn diagonal_45_projects_the_above_row_at_8x8() {
        // Same projection as 4x4, extended: pred[i][j] = above[i+j+1] while
        // i+j+1 < maxBaseX = w+h-1 = 15.
        let mut above = Edge::new();
        let mut left = Edge::new();
        for i in 0..16 {
            above.set(i, i as i32);
        }
        let pred = predict_directional(
            45, &mut above, &mut left, 8, 8, true, true, false, false, 8, 8, 8,
        );
        // Row 0: above[1..9]; row 7: above[8..16] but clamped at maxBaseX=15.
        assert_eq!(&pred[0..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(pred[7 * 8 + 7], 15); // i+j+1 = 15 == maxBaseX -> above[15]
    }
}
