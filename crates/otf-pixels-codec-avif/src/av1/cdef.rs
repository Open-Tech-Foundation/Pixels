//! The constrained directional enhancement filter (CDEF, spec §7.15).
//!
//! CDEF runs after deblocking. It works on 8x8 luma blocks: for each block it
//! first detects the dominant edge direction (§7.15.2) by projecting the block
//! onto eight oriented lines and picking the one with the most energy, then
//! applies a non-linear low-pass filter (§7.15.3) whose taps follow that
//! direction. A primary filter along the direction removes ringing while a
//! secondary filter across it preserves the edge; the `constrain` function caps
//! how far any tap may pull a sample so real detail survives. Strengths come
//! from the frame's `cdef_params`, selected per 64x64 block by the `cdef_idx`
//! read during tile decode.
//!
//! Reads are taken from the *input* frame (the deblocked reconstruction) and
//! written to a separate output, so filtering never sees its own results; this
//! module snapshots the planes before it starts. Each 8x8 luma block filters
//! the co-located chroma block — 4x4 for 4:2:0, 4x8 for 4:2:2, 8x8 for 4:4:4 —
//! with the direction remapped for 4:2:2 (`Cdef_Uv_Dir`). A `cdef_idx` of -1,
//! or an 8x8 block whose four 4x4 units are all coded skip, leaves that block
//! untouched.
//!
//! Like the transform DSP, the direction search (§7.15.2) and the filter's tap
//! lookups are transcribed straight from the spec, which reads the fixed-size
//! working arrays and constant tables by index; expressing those through
//! `.get()` would obscure the correspondence, so the module opts into
//! `indexing_slicing` for them. Every such index is a spec constant strictly
//! below its array length (directions 0..8, taps 0..2, `partial`/`cost` sized to
//! the block). The runtime-sized sample planes are still read and written only
//! through [`Plane`]'s checked API.
#![allow(
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    reason = "fixed-size spec tables and direction-search working arrays, indexed \
              by spec-bounded constants; sample planes still use Plane's checked API"
)]

use super::frame::Cdef;
use super::plane::Plane;

/// `MI_SIZE` (§3): the side of the smallest coded block, in samples.
const MI_SIZE: usize = 4;
/// `MI_SIZE_LOG2` (§3).
const MI_SIZE_LOG2: u32 = 2;

/// `Cdef_Pri_Taps` (§7.15.3), indexed by `(priStr >> coeffShift) & 1`.
const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
/// `Cdef_Sec_Taps` (§7.15.3), indexed by `(priStr >> coeffShift) & 1`.
const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];

/// `Div_Table` (§7.15.2), used to normalise the directional costs.
const DIV_TABLE: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// `Cdef_Directions[dir][k]` (§7.15.3): the `(dy, dx)` offset of the `k`-th tap
/// along direction `dir`.
const CDEF_DIRECTIONS: [[[isize; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];

/// `Cdef_Uv_Dir[subX][subY][yDir]` (§7.15.1): the chroma direction for a given
/// luma direction under the plane's subsampling.
const CDEF_UV_DIR: [[[usize; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];

/// Everything the CDEF process reads: the reconstructed planes it filters in
/// place, the frame's `cdef_params`, the per-64x64 `cdef_idx` grid and the
/// per-4x4 `Skips` grid gathered during reconstruction (both row-major,
/// `mi_cols` wide).
pub struct CdefFilter<'a> {
    /// The reconstructed (deblocked) planes, filtered in place.
    pub planes: &'a mut [Plane],
    /// The frame's CDEF parameters.
    pub cdef: &'a Cdef,
    /// `cdef_idx[row][col]`: the strength set for the 64x64 block, or -1 for no
    /// filtering. Only the 64x64-aligned entries are meaningful.
    pub cdef_idx: &'a [i16],
    /// `Skips[row][col]`: the skip flag of the block owning each 4x4 unit.
    pub skips: &'a [u8],
    /// Sample bit depth (8/10/12).
    pub bit_depth: u8,
    /// Number of planes (1 monochrome, 3 otherwise).
    pub num_planes: usize,
    /// Frame dimensions in 4x4 units.
    pub mi_rows: usize,
    pub mi_cols: usize,
    /// Chroma subsampling (0 for 4:4:4).
    pub subsampling_x: usize,
    pub subsampling_y: usize,
}

impl CdefFilter<'_> {
    /// Apply CDEF to the whole frame (§7.15). Steps over every 8x8 luma block,
    /// looks up its 64x64 `cdef_idx`, and filters it. When CDEF is disabled the
    /// `cdef_idx` grid is all -1 and every block is a no-op.
    pub fn run(&mut self) {
        // The filter reads the pre-CDEF frame throughout, so snapshot it. The
        // planes in `self` are the output; they already equal the input, which
        // satisfies the spec's initial "copy CurrFrame to CdefFrame" step.
        let src: Vec<Plane> = self.planes.to_vec();

        // BLOCK_8X8 is 2 units wide; BLOCK_64X64 is 16.
        let step4 = 2;
        let cdef_size4 = 16;
        let mask = !(cdef_size4 - 1);
        let mut r = 0;
        while r < self.mi_rows {
            let mut c = 0;
            while c < self.mi_cols {
                let base = (r & mask) * self.mi_cols + (c & mask);
                let idx = self.cdef_idx.get(base).copied().unwrap_or(-1);
                self.cdef_block(&src, r, c, idx);
                c += step4;
            }
            r += step4;
        }
    }

    /// CDEF block process (§7.15.1) for the 8x8 block at 4x4 unit `(r, c)`.
    fn cdef_block(&mut self, src: &[Plane], r: usize, c: usize, idx: i16) {
        // idx == -1: no filtering; the output already holds the copied input.
        let Ok(idx) = usize::try_from(idx) else {
            return;
        };
        // skip when all four 4x4 units of the block are coded skip.
        if self.skip_at(r, c)
            && self.skip_at(r + 1, c)
            && self.skip_at(r, c + 1)
            && self.skip_at(r + 1, c + 1)
        {
            return;
        }

        let coeff_shift = u32::from(self.bit_depth) - 8;
        let Some(luma) = src.first() else {
            return;
        };
        let (y_dir, var) = direction(luma, r, c, self.bit_depth);

        // Luma (§7.15.1 steps 1-7).
        let mut pri =
            i64::from(self.cdef.y_pri_strength.get(idx).copied().unwrap_or(0)) << coeff_shift;
        let sec = i64::from(self.cdef.y_sec_strength.get(idx).copied().unwrap_or(0)) << coeff_shift;
        let dir = if pri == 0 { 0 } else { y_dir };
        let var_str = if (var >> 6) != 0 {
            floor_log2(var >> 6).min(12)
        } else {
            0
        };
        pri = if var != 0 {
            (pri * (4 + i64::from(var_str)) + 8) >> 4
        } else {
            0
        };
        let damping = i64::from(self.cdef.damping) + i64::from(coeff_shift);
        self.filter(0, src, r, c, pri, sec, damping, dir);

        if self.num_planes == 1 {
            return;
        }

        // Chroma (§7.15.1 steps 9-14): both planes share the luma direction
        // mapped through Cdef_Uv_Dir and a damping one lower than luma.
        let pri_uv =
            i64::from(self.cdef.uv_pri_strength.get(idx).copied().unwrap_or(0)) << coeff_shift;
        let sec_uv =
            i64::from(self.cdef.uv_sec_strength.get(idx).copied().unwrap_or(0)) << coeff_shift;
        let dir_uv = if pri_uv == 0 {
            0
        } else {
            CDEF_UV_DIR[self.subsampling_x][self.subsampling_y][y_dir]
        };
        let damping_uv = i64::from(self.cdef.damping) + i64::from(coeff_shift) - 1;
        self.filter(1, src, r, c, pri_uv, sec_uv, damping_uv, dir_uv);
        self.filter(2, src, r, c, pri_uv, sec_uv, damping_uv, dir_uv);
    }

    /// `Skips[row][col]` as a bool, treating off-grid units as coded skip so a
    /// partial block at the frame edge does not force filtering.
    fn skip_at(&self, row: usize, col: usize) -> bool {
        if row >= self.mi_rows || col >= self.mi_cols {
            return true;
        }
        self.skips
            .get(row * self.mi_cols + col)
            .copied()
            .unwrap_or(1)
            != 0
    }

    /// CDEF filter process (§7.15.3): filter one plane's 8x8 (or subsampled)
    /// region reading from `src` and writing to `self.planes`.
    #[allow(clippy::too_many_arguments, reason = "mirrors the spec's input list")]
    fn filter(
        &mut self,
        plane: usize,
        src: &[Plane],
        r: usize,
        c: usize,
        pri: i64,
        sec: i64,
        damping: i64,
        dir: usize,
    ) {
        let coeff_shift = u32::from(self.bit_depth) - 8;
        let (sub_x, sub_y) = if plane > 0 {
            (self.subsampling_x, self.subsampling_y)
        } else {
            (0, 0)
        };
        let x0 = (c * MI_SIZE) >> sub_x;
        let y0 = (r * MI_SIZE) >> sub_y;
        let w = 8 >> sub_x;
        let h = 8 >> sub_y;

        let tap_sel = ((pri >> coeff_shift) & 1) as usize;
        let pri_taps = CDEF_PRI_TAPS[tap_sel];
        let sec_taps = CDEF_SEC_TAPS[tap_sel];

        let Some(source) = src.get(plane) else {
            return;
        };
        // Collect the whole block first, then commit, so the destination write
        // never disturbs a read (it would not here, but it keeps src/dst clean).
        let mut out = [0_i32; 64];
        for i in 0..h {
            for j in 0..w {
                let x = i32::from(source.get(x0 + j, y0 + i).unwrap_or(0));
                let mut sum = 0_i64;
                let mut max = x;
                let mut min = x;
                for k in 0..2 {
                    for sign in [-1_isize, 1] {
                        // Primary tap along the direction.
                        if let Some(p) =
                            self.get_at(source, x0, y0, i, j, dir, k, sign, sub_x, sub_y)
                        {
                            sum += i64::from(pri_taps[k]) * constrain(p - x, pri, damping);
                            max = max.max(p);
                            min = min.min(p);
                        }
                        // Secondary taps across the direction (±2).
                        for dir_off in [-2_isize, 2] {
                            let sd = ((dir as isize + dir_off) & 7) as usize;
                            if let Some(s) =
                                self.get_at(source, x0, y0, i, j, sd, k, sign, sub_x, sub_y)
                            {
                                sum += i64::from(sec_taps[k]) * constrain(s - x, sec, damping);
                                max = max.max(s);
                                min = min.min(s);
                            }
                        }
                    }
                }
                let bias = i64::from(sum < 0);
                let filtered = i64::from(x) + ((8 + sum - bias) >> 4);
                let clipped = filtered.clamp(i64::from(min), i64::from(max));
                if let Some(cell) = out.get_mut(i * w + j) {
                    *cell = clipped as i32;
                }
            }
        }
        let Some(dest) = self.planes.get_mut(plane) else {
            return;
        };
        for i in 0..h {
            for j in 0..w {
                if let Some(&v) = out.get(i * w + j) {
                    dest.set(x0 + j, y0 + i, v.max(0) as u16);
                }
            }
        }
    }

    /// `cdef_get_at` (§7.15.3): the sample `k` steps along `dir` from `(i, j)`,
    /// or `None` when it falls outside the (single-tile) filter region.
    #[allow(clippy::too_many_arguments, reason = "mirrors the spec's input list")]
    fn get_at(
        &self,
        source: &Plane,
        x0: usize,
        y0: usize,
        i: usize,
        j: usize,
        dir: usize,
        k: usize,
        sign: isize,
        sub_x: usize,
        sub_y: usize,
    ) -> Option<i32> {
        let off = CDEF_DIRECTIONS[dir][k];
        let y = y0 as isize + i as isize + sign * off[0];
        let x = x0 as isize + j as isize + sign * off[1];
        let candidate_r = (y << sub_y) >> MI_SIZE_LOG2;
        let candidate_c = (x << sub_x) >> MI_SIZE_LOG2;
        // is_inside_filter_region: single tile, so the region is the whole frame.
        if candidate_r < 0
            || candidate_r >= self.mi_rows as isize
            || candidate_c < 0
            || candidate_c >= self.mi_cols as isize
        {
            return None;
        }
        source.get(x as usize, y as usize).map(i32::from)
    }
}

/// CDEF direction process (§7.15.2): the dominant edge direction of the 8x8
/// luma block at `(r, c)` and its variance.
fn direction(luma: &Plane, r: usize, c: usize, bit_depth: u8) -> (usize, i64) {
    let shift = u32::from(bit_depth) - 8;
    let x0 = c << MI_SIZE_LOG2;
    let y0 = r << MI_SIZE_LOG2;

    let mut partial = [[0_i64; 15]; 8];
    for i in 0..8_usize {
        for j in 0..8_usize {
            let sample = i32::from(luma.get(x0 + j, y0 + i).unwrap_or(0));
            let x = i64::from((sample >> shift) - 128);
            partial[0][i + j] += x;
            partial[1][i + j / 2] += x;
            partial[2][i] += x;
            partial[3][3 + i - j / 2] += x;
            partial[4][7 + i - j] += x;
            partial[5][3 - i / 2 + j] += x;
            partial[6][j] += x;
            partial[7][i / 2 + j] += x;
        }
    }

    let mut cost = [0_i64; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    for i in 0..7 {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    let mut i = 1;
    while i < 8 {
        for j in 0..5 {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3 {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
        i += 2;
    }

    let mut best_cost = 0_i64;
    let mut y_dir = 0_usize;
    for (d, &value) in cost.iter().enumerate() {
        if value > best_cost {
            best_cost = value;
            y_dir = d;
        }
    }
    let var = (best_cost - cost[(y_dir + 4) & 7]) >> 10;
    (y_dir, var)
}

/// `constrain` (§7.15.3): the tap's contribution, capped so it cannot pull a
/// sample past the strength threshold.
fn constrain(diff: i32, threshold: i64, damping: i64) -> i64 {
    if threshold == 0 {
        return 0;
    }
    let damping_adj = (damping - i64::from(floor_log2(threshold))).max(0);
    let sign = if diff < 0 { -1 } else { 1 };
    let abs = i64::from(diff.unsigned_abs());
    sign * (threshold - (abs >> damping_adj)).clamp(0, abs)
}

/// `FloorLog2(x)` for `x >= 1`: the index of the most significant set bit.
fn floor_log2(x: i64) -> i32 {
    debug_assert!(x >= 1);
    63 - x.max(1).leading_zeros() as i32
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

    fn cdef(pri: u32, sec: u32) -> Cdef {
        Cdef {
            damping: 3,
            bits: 0,
            y_pri_strength: vec![pri],
            y_sec_strength: vec![sec],
            uv_pri_strength: vec![0],
            uv_sec_strength: vec![0],
        }
    }

    /// A flat block has no ringing and no direction; the constrain cap means the
    /// filter cannot move any sample, so a constant plane survives untouched.
    #[test]
    fn a_flat_block_is_unchanged() {
        let mut plane = Plane::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                plane.set(x, y, 128);
            }
        }
        let before = plane.clone();
        let idx = vec![0_i16; 4];
        let skips = vec![0_u8; 4];
        let mut planes = [plane];
        CdefFilter {
            planes: &mut planes,
            cdef: &cdef(15, 4),
            cdef_idx: &idx,
            skips: &skips,
            bit_depth: 8,
            num_planes: 1,
            mi_rows: 2,
            mi_cols: 2,
            subsampling_x: 0,
            subsampling_y: 0,
        }
        .run();
        assert_eq!(planes[0].samples(), before.samples());
    }

    /// A -1 cdef_idx disables filtering for the block even when a strong step
    /// edge is present.
    #[test]
    fn a_negative_index_disables_filtering() {
        let mut plane = Plane::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                plane.set(x, y, if x < 4 { 20 } else { 220 });
            }
        }
        let before = plane.clone();
        let idx = vec![-1_i16; 4];
        let skips = vec![0_u8; 4];
        let mut planes = [plane];
        CdefFilter {
            planes: &mut planes,
            cdef: &cdef(15, 15),
            cdef_idx: &idx,
            skips: &skips,
            bit_depth: 8,
            num_planes: 1,
            mi_rows: 2,
            mi_cols: 2,
            subsampling_x: 0,
            subsampling_y: 0,
        }
        .run();
        assert_eq!(planes[0].samples(), before.samples());
    }

    /// An all-skip block is left untouched even with a valid index and strength.
    #[test]
    fn an_all_skip_block_is_unchanged() {
        let mut plane = Plane::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                plane.set(x, y, if (x + y) % 2 == 0 { 60 } else { 200 });
            }
        }
        let before = plane.clone();
        let idx = vec![0_i16; 4];
        let skips = vec![1_u8; 4];
        let mut planes = [plane];
        CdefFilter {
            planes: &mut planes,
            cdef: &cdef(15, 4),
            cdef_idx: &idx,
            skips: &skips,
            bit_depth: 8,
            num_planes: 1,
            mi_rows: 2,
            mi_cols: 2,
            subsampling_x: 0,
            subsampling_y: 0,
        }
        .run();
        assert_eq!(planes[0].samples(), before.samples());
    }

    /// A small interior ripple over a flat field is pulled toward its
    /// neighbours: removing that low-amplitude ringing is CDEF's whole purpose.
    /// (A *large* spike would instead be a real edge, which `constrain` leaves
    /// alone — the threshold is what keeps genuine detail intact.)
    #[test]
    fn a_small_ripple_is_pulled_toward_its_neighbours() {
        let mut plane = Plane::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                plane.set(x, y, 100);
            }
        }
        plane.set(4, 4, 106);
        let idx = vec![0_i16; 4];
        let skips = vec![0_u8; 4];
        let mut planes = [plane];
        CdefFilter {
            planes: &mut planes,
            cdef: &cdef(15, 4),
            cdef_idx: &idx,
            skips: &skips,
            bit_depth: 8,
            num_planes: 1,
            mi_rows: 2,
            mi_cols: 2,
            subsampling_x: 0,
            subsampling_y: 0,
        }
        .run();
        let v = planes[0].get(4, 4).unwrap();
        assert!(v < 106, "the ripple should be reduced, got {v}");
        assert!(v >= 100, "but not below the surrounding field, got {v}");
    }

    #[test]
    fn floor_log2_matches_bit_position() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(255), 7);
        assert_eq!(floor_log2(256), 8);
    }
}
