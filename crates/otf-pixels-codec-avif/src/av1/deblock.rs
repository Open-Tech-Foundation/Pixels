//! The deblocking loop filter (spec §7.14).
//!
//! After a frame is reconstructed, the loop filter smooths the block and
//! transform boundaries that the semi-independent coding of super-blocks leaves
//! behind. It runs in two passes over the whole frame — every vertical boundary
//! first, then every horizontal boundary — and at each 4x4 edge decides, from
//! the transform sizes on either side and an adaptive strength derived from the
//! frame's `loop_filter` parameters, whether to apply a narrow (4-tap) or one of
//! the wide (8-/14-tap luma, 6-/8-tap chroma) low-pass filters.
//!
//! This is the still-picture intra subset: every block is intra, so the
//! `applyFilter` test reduces to "there is a transform edge here", the segment
//! id and per-block loop-filter delta are always zero, and the mode type is
//! always the intra type 0. 4:4:4 only, so the chroma planes share the luma
//! mode-info grid with no subsampling shift. The `LoopfilterTxSizes` grid is
//! filled per plane as each transform block is reconstructed.

use super::frame::LoopFilter;
use super::plane::Plane;
use super::transform::TxSize;

/// `MI_SIZE` (§3): the side of the smallest coded block, in samples.
const MI_SIZE: usize = 4;
/// `MAX_LOOP_FILTER` (§3).
const MAX_LOOP_FILTER: i32 = 63;

/// Everything the loop filter reads: the reconstructed planes it filters in
/// place, the frame's `loop_filter` parameters, and the per-4x4 mode-info grids
/// gathered during reconstruction. Grids are row-major, `mi_cols` wide.
pub struct Deblock<'a> {
    /// The reconstructed planes, filtered in place.
    pub planes: &'a mut [Plane],
    /// The frame's loop-filter parameters.
    pub loop_filter: &'a LoopFilter,
    /// Sample bit depth (8/10/12).
    pub bit_depth: u8,
    /// Number of planes (1 monochrome, 3 for 4:4:4).
    pub num_planes: usize,
    /// Frame dimensions in 4x4 units.
    pub mi_rows: usize,
    pub mi_cols: usize,
    /// Visible frame dimensions in luma samples (`FrameWidth`/`FrameHeight`).
    pub frame_width: usize,
    pub frame_height: usize,
    /// `LoopfilterTxSizes[plane][row][col]`: the transform size (a `TxSize`
    /// index) reconstructed at each 4x4 unit, one grid per plane.
    pub lf_tx_sizes: &'a [Vec<u8>],
}

// A full inter-capable filter would also read `Skips`, `MiSizes` (for the block
// edge test) and `YModes`/`RefFrames`/`SegmentIds`/`DeltaLFs` (for the strength
// derivation). In this all-intra 4:4:4 subset those collapse to constants:
// `applyFilter` is just the transform-edge test and the strength comes from the
// frame `loop_filter` level plus the intra reference delta, so they are omitted.

impl Deblock<'_> {
    /// Apply the loop filter to every plane (§7.14.1): all vertical boundaries,
    /// then all horizontal, per plane. A plane with a zero filter level (which a
    /// coded-lossless frame always has) is left untouched.
    pub fn run(&mut self) {
        // Frame-level guard (libaom `av1_loop_filter_frame_init`): if both luma
        // levels are zero the whole loop filter is skipped — even though the
        // per-block intra reference delta could otherwise raise the strength
        // above zero. A coded-lossless or filters-off frame always lands here.
        let luma_vert = self.loop_filter.level.first().copied().unwrap_or(0);
        let luma_horz = self.loop_filter.level.get(1).copied().unwrap_or(0);
        if luma_vert == 0 && luma_horz == 0 {
            return;
        }
        for plane in 0..self.num_planes {
            // Y is always filtered; a chroma plane only if its level is nonzero.
            if plane != 0 && self.level_for(plane) == 0 {
                continue;
            }
            for pass in 0..2 {
                let mut row = 0;
                while row < self.mi_rows {
                    let mut col = 0;
                    while col < self.mi_cols {
                        self.filter_edge(plane, pass, row, col);
                        col += 1;
                    }
                    row += 1;
                }
            }
        }
    }

    /// `loop_filter_level[1 + plane]` for the chroma-plane gate.
    fn level_for(&self, plane: usize) -> u8 {
        self.loop_filter.level.get(1 + plane).copied().unwrap_or(0)
    }

    /// The transform size stored for `plane` at 4x4 unit `(row, col)`.
    fn tx_size(&self, plane: usize, row: usize, col: usize) -> TxSize {
        let idx = row * self.mi_cols + col;
        let raw = self
            .lf_tx_sizes
            .get(plane)
            .and_then(|g| g.get(idx))
            .copied()
            .unwrap_or(0);
        TxSize::from_index(usize::from(raw))
    }

    /// Edge loop filter process (§7.14.2) for one 4x4 boundary.
    fn filter_edge(&mut self, plane: usize, pass: usize, row: usize, col: usize) {
        // 4:4:4: no subsampling, so plane coordinates equal luma coordinates.
        let (dx, dy) = if pass == 0 { (1_usize, 0) } else { (0, 1) };
        let x = col * MI_SIZE;
        let y = row * MI_SIZE;

        // onScreen: both sides of the boundary must lie in the visible area, and
        // the frame's own top/left edge is never an interior boundary.
        if x >= self.frame_width || y >= self.frame_height {
            return;
        }
        if pass == 0 && x == 0 {
            return;
        }
        if pass == 1 && y == 0 {
            return;
        }

        let tx_sz = self.tx_size(plane, row, col);
        let tx_w = tx_sz.width();
        let tx_h = tx_sz.height();

        // The mode-info block on the other side of the boundary.
        let (prev_row, prev_col) = if pass == 0 {
            (row, col.wrapping_sub(1))
        } else {
            (row.wrapping_sub(1), col)
        };
        let prev_tx_sz = self.tx_size(plane, prev_row, prev_col);

        // applyFilter (§7.14.2) is `isTxEdge && (isBlockEdge || !skip || isIntra)`.
        // Every block in this subset is intra, so the parenthesis is always true
        // and the test reduces to "this is a transform edge".
        let is_tx_edge = if pass == 0 {
            x % tx_w == 0
        } else {
            y % tx_h == 0
        };
        if !is_tx_edge {
            return;
        }

        let filter_size = filter_size(tx_sz, prev_tx_sz, pass, plane);

        // Adaptive strength for this block; fall back to the neighbour's when the
        // current block's level is zero (§7.14.2).
        let (mut lvl, mut limit, mut blimit, mut thresh) =
            self.filter_strength(row, col, plane, pass);
        if lvl == 0 {
            let (l2, li2, bl2, th2) = self.filter_strength(prev_row, prev_col, plane, pass);
            lvl = l2;
            limit = li2;
            blimit = bl2;
            thresh = th2;
        }
        if lvl == 0 {
            return;
        }

        let Some(plane_buf) = self.planes.get_mut(plane) else {
            return;
        };
        for i in 0..MI_SIZE {
            let fx = x + dy * i;
            let fy = y + dx * i;
            sample_filter(
                plane_buf,
                fx,
                fy,
                plane,
                limit,
                blimit,
                thresh,
                dx,
                dy,
                filter_size,
                self.bit_depth,
            );
        }
    }

    /// Adaptive filter strength process (§7.14.4/§7.14.5). Segment id and the
    /// per-block loop-filter delta are always zero in this subset, and every
    /// block is intra, so the strength comes from `loop_filter_level` plus the
    /// intra reference delta.
    fn filter_strength(
        &self,
        _row: usize,
        _col: usize,
        plane: usize,
        pass: usize,
    ) -> (i32, i32, i32, i32) {
        let i = if plane == 0 { pass } else { plane + 1 };
        let base = self.loop_filter.level.get(i).copied().unwrap_or(0);
        let mut lvl = i32::from(base).clamp(0, MAX_LOOP_FILTER);
        if self.loop_filter.delta_enabled {
            // ref == INTRA_FRAME (0) for every block; the mode delta applies only
            // to inter modes, so it never contributes here.
            let n_shift = lvl >> 5;
            let intra_delta = self.loop_filter.ref_deltas.first().copied().unwrap_or(0);
            lvl = (lvl + (intra_delta << n_shift)).clamp(0, MAX_LOOP_FILTER);
        }

        let sharpness = i32::from(self.loop_filter.sharpness);
        let shift = if sharpness > 4 {
            2
        } else if sharpness > 0 {
            1
        } else {
            0
        };
        let limit = if sharpness > 0 {
            (lvl >> shift).clamp(1, 9 - sharpness)
        } else {
            (lvl >> shift).max(1)
        };
        let blimit = 2 * (lvl + 2) + limit;
        let thresh = lvl >> 4;
        (lvl, limit, blimit, thresh)
    }
}

/// Filter size process (§7.14.3): the maximum filter width the boundary allows.
fn filter_size(tx_sz: TxSize, prev_tx_sz: TxSize, pass: usize, plane: usize) -> usize {
    let base = if pass == 0 {
        prev_tx_sz.width().min(tx_sz.width())
    } else {
        prev_tx_sz.height().min(tx_sz.height())
    };
    if plane == 0 {
        base.min(16)
    } else {
        base.min(8)
    }
}

/// Read a sample as a signed value, replicating the nearest edge off-frame.
fn get(plane: &Plane, x: usize, y: usize, dx: usize, dy: usize, k: isize) -> i32 {
    let sx = x as isize + dx as isize * k;
    let sy = y as isize + dy as isize * k;
    i32::from(plane.sample_clamped(sx, sy))
}

/// Sample filtering process (§7.14.6): choose and apply the filter for the
/// boundary sample at `(x, y)` running perpendicular to `(dx, dy)`.
#[allow(clippy::too_many_arguments, reason = "mirrors the spec's input list")]
fn sample_filter(
    plane_buf: &mut Plane,
    x: usize,
    y: usize,
    plane: usize,
    limit: i32,
    blimit: i32,
    thresh: i32,
    dx: usize,
    dy: usize,
    filter_size: usize,
    bit_depth: u8,
) {
    let bd = u32::from(bit_depth);
    // q0..q6 on the near side, p0..p6 on the far side of the boundary.
    let q = |k: isize| get(plane_buf, x, y, dx, dy, k);
    let p = |k: isize| get(plane_buf, x, y, dx, dy, -1 - k);
    let (q0, q1, q2, q3) = (q(0), q(1), q(2), q(3));
    let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));

    // filterLen: taps each side used by the masks.
    let filter_len = if filter_size == 4 {
        4
    } else if plane != 0 {
        6
    } else if filter_size == 8 {
        8
    } else {
        16
    };

    let shift = bd - 8;
    let thresh_bd = thresh << shift;
    let hev_mask = ((p1 - p0).abs() > thresh_bd) || ((q1 - q0).abs() > thresh_bd);

    let limit_bd = limit << shift;
    let blimit_bd = blimit << shift;
    let mut mask = false;
    mask |= (p1 - p0).abs() > limit_bd;
    mask |= (q1 - q0).abs() > limit_bd;
    mask |= (p0 - q0).abs() * 2 + (p1 - q1).abs() / 2 > blimit_bd;
    if filter_len >= 6 {
        mask |= (p2 - p1).abs() > limit_bd;
        mask |= (q2 - q1).abs() > limit_bd;
    }
    if filter_len >= 8 {
        mask |= (p3 - p2).abs() > limit_bd;
        mask |= (q3 - q2).abs() > limit_bd;
    }
    let filter_mask = !mask;
    if !filter_mask {
        return;
    }

    // flatMask (filterSize >= 8) and flatMask2 (filterSize >= 16).
    let threshold_bd = 1_i32 << shift;
    let flat_mask = if filter_size >= 8 {
        let mut m = false;
        m |= (p1 - p0).abs() > threshold_bd;
        m |= (q1 - q0).abs() > threshold_bd;
        m |= (p2 - p0).abs() > threshold_bd;
        m |= (q2 - q0).abs() > threshold_bd;
        if filter_len >= 8 {
            m |= (p3 - p0).abs() > threshold_bd;
            m |= (q3 - q0).abs() > threshold_bd;
        }
        !m
    } else {
        false
    };
    let flat_mask2 = if filter_size >= 16 {
        let (q4, q5, q6) = (q(4), q(5), q(6));
        let (p4, p5, p6) = (p(4), p(5), p(6));
        let mut m = false;
        m |= (p6 - p0).abs() > threshold_bd;
        m |= (q6 - q0).abs() > threshold_bd;
        m |= (p5 - p0).abs() > threshold_bd;
        m |= (q5 - q0).abs() > threshold_bd;
        m |= (p4 - p0).abs() > threshold_bd;
        m |= (q4 - q0).abs() > threshold_bd;
        !m
    } else {
        false
    };

    if filter_size == 4 || !flat_mask {
        narrow_filter(plane_buf, x, y, dx, dy, hev_mask, bd);
    } else if filter_size == 8 || !flat_mask2 {
        wide_filter(plane_buf, x, y, plane, dx, dy, 3);
    } else {
        wide_filter(plane_buf, x, y, plane, dx, dy, 4);
    }
}

/// `filter4_clamp`: clamp to the signed `BitDepth` range.
fn filter4_clamp(value: i32, bd: u32) -> i32 {
    let lo = -(1_i32 << (bd - 1));
    let hi = (1_i32 << (bd - 1)) - 1;
    value.clamp(lo, hi)
}

/// `Round2(x, n)`.
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// Narrow filter process (§7.14.6.3): modifies up to two samples each side.
fn narrow_filter(
    plane_buf: &mut Plane,
    x: usize,
    y: usize,
    dx: usize,
    dy: usize,
    hev_mask: bool,
    bd: u32,
) {
    let bias = 0x80 << (bd - 8);
    let read = |buf: &Plane, k: isize| get(buf, x, y, dx, dy, k);
    let q0 = read(plane_buf, 0);
    let q1 = read(plane_buf, 1);
    let p0 = read(plane_buf, -1);
    let p1 = read(plane_buf, -2);
    let ps1 = p1 - bias;
    let ps0 = p0 - bias;
    let qs0 = q0 - bias;
    let qs1 = q1 - bias;
    let mut filter = if hev_mask {
        filter4_clamp(ps1 - qs1, bd)
    } else {
        0
    };
    filter = filter4_clamp(filter + 3 * (qs0 - ps0), bd);
    let filter1 = filter4_clamp(filter + 4, bd) >> 3;
    let filter2 = filter4_clamp(filter + 3, bd) >> 3;
    let oq0 = filter4_clamp(qs0 - filter1, bd) + bias;
    let op0 = filter4_clamp(ps0 + filter2, bd) + bias;
    put(plane_buf, x, y, dx, dy, 0, oq0);
    put(plane_buf, x, y, dx, dy, -1, op0);
    if !hev_mask {
        let f = round2(filter1, 1);
        let oq1 = filter4_clamp(qs1 - f, bd) + bias;
        let op1 = filter4_clamp(ps1 + f, bd) + bias;
        put(plane_buf, x, y, dx, dy, 1, oq1);
        put(plane_buf, x, y, dx, dy, -2, op1);
    }
}

/// Wide filter process (§7.14.6.4): a symmetric low-pass over `2n` samples.
fn wide_filter(
    plane_buf: &mut Plane,
    x: usize,
    y: usize,
    plane: usize,
    dx: usize,
    dy: usize,
    log2_size: u32,
) {
    let n: isize = if log2_size == 4 {
        6
    } else if plane == 0 {
        3
    } else {
        2
    };
    let n2: isize = if log2_size == 3 && plane == 0 { 0 } else { 1 };
    let mut filtered = [0_i32; 12]; // indices i = -n..n-1, offset by n (n <= 6)
    for i in -n..n {
        let mut t = 0;
        for j in -n..=n {
            let pos = (i + j).clamp(-(n + 1), n);
            let tap = if j.abs() <= n2 { 2 } else { 1 };
            t += get(plane_buf, x, y, dx, dy, pos) * tap;
        }
        if let Some(cell) = filtered.get_mut((i + n) as usize) {
            *cell = round2(t, log2_size);
        }
    }
    for i in -n..n {
        let v = filtered.get((i + n) as usize).copied().unwrap_or(0);
        put(plane_buf, x, y, dx, dy, i, v);
    }
}

/// Write a sample at `(x, y) + k * (dx, dy)`, dropping out-of-range writes.
fn put(plane_buf: &mut Plane, x: usize, y: usize, dx: usize, dy: usize, k: isize, value: i32) {
    let sx = x as isize + dx as isize * k;
    let sy = y as isize + dy as isize * k;
    if sx < 0 || sy < 0 {
        return;
    }
    plane_buf.set(sx as usize, sy as usize, value.max(0) as u16);
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::super::frame::LoopFilter;
    use super::*;

    fn lf(level: [u8; 4]) -> LoopFilter {
        LoopFilter {
            level,
            sharpness: 0,
            delta_enabled: true,
            // Default intra ref delta of 1: exercises the strength derivation.
            ref_deltas: [1, 0, 0, 0, -1, -1, -1, -1],
            mode_deltas: [0, 0],
        }
    }

    /// A perfectly flat region has no edges, so every filter is a no-op (the
    /// wide filter has unity DC gain and reproduces a constant exactly).
    #[test]
    fn a_flat_region_is_left_unchanged() {
        let (w, h) = (16, 16);
        let mut plane = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                plane.set(x, y, 137);
            }
        }
        let before = plane.clone();
        // One 8x8 transform grid (index 1) everywhere, so tx edges exist at 8.
        let grid = vec![1_u8; (w / MI_SIZE) * (h / MI_SIZE)];
        let mut planes = [plane];
        Deblock {
            planes: &mut planes,
            loop_filter: &lf([20, 20, 20, 20]),
            bit_depth: 8,
            num_planes: 1,
            mi_rows: h / MI_SIZE,
            mi_cols: w / MI_SIZE,
            frame_width: w,
            frame_height: h,
            lf_tx_sizes: std::slice::from_ref(&grid),
        }
        .run();
        assert_eq!(planes[0].row(0), before.row(0));
        assert_eq!(planes[0].row(7), before.row(7));
        assert_eq!(planes[0].row(8), before.row(8));
    }

    /// With both luma levels zero the whole filter is skipped, even though the
    /// intra reference delta would otherwise raise the strength above zero.
    #[test]
    fn zero_luma_level_skips_filtering() {
        let (w, h) = (16, 16);
        let mut plane = Plane::new(w, h);
        // A hard vertical step at the 8-sample transform edge that the filter
        // would smooth if it ran.
        for y in 0..h {
            for x in 0..w {
                plane.set(x, y, if x < 8 { 40 } else { 200 });
            }
        }
        let before = plane.clone();
        let grid = vec![1_u8; (w / MI_SIZE) * (h / MI_SIZE)];
        let mut planes = [plane];
        Deblock {
            planes: &mut planes,
            loop_filter: &lf([0, 0, 0, 0]),
            bit_depth: 8,
            num_planes: 1,
            mi_rows: h / MI_SIZE,
            mi_cols: w / MI_SIZE,
            frame_width: w,
            frame_height: h,
            lf_tx_sizes: std::slice::from_ref(&grid),
        }
        .run();
        assert_eq!(planes[0].row(0), before.row(0));
    }

    /// A nonzero level does smooth a hard transform-edge step: the samples
    /// either side of the boundary move toward each other.
    #[test]
    fn a_step_edge_is_smoothed() {
        let (w, h) = (16, 16);
        let mut plane = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                plane.set(x, y, if x < 8 { 100 } else { 140 });
            }
        }
        let grid = vec![1_u8; (w / MI_SIZE) * (h / MI_SIZE)];
        let mut planes = [plane];
        Deblock {
            planes: &mut planes,
            loop_filter: &lf([32, 32, 0, 0]),
            bit_depth: 8,
            num_planes: 1,
            mi_rows: h / MI_SIZE,
            mi_cols: w / MI_SIZE,
            frame_width: w,
            frame_height: h,
            lf_tx_sizes: std::slice::from_ref(&grid),
        }
        .run();
        // The last "left" sample rose and the first "right" sample fell.
        let row = planes[0].row(4).unwrap();
        assert!(row[7] > 100, "left edge sample should rise, got {}", row[7]);
        assert!(row[8] < 140, "right edge sample should fall, got {}", row[8]);
    }
}
