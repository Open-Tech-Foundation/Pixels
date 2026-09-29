//! Loop restoration (spec §7.17), the last in-loop post-filter.
//!
//! After deblocking and CDEF, loop restoration reduces the remaining coding
//! error inside restoration units (64–256 luma samples on a side) with one of
//! two filters chosen per unit: a symmetric 7-tap (luma) / 5-tap (chroma) Wiener
//! filter (§7.17.4), or a self-guided filter (§7.17.2) that blends two box-filter
//! passes of different radii toward the source. The per-unit filter type and
//! coefficients are read during tile decode (`read_lr`, §5.11.57–58) using the
//! recentered sub-exponential coding this module also provides.
//!
//! The filter works in horizontal stripes 64 luma samples high, offset up by 8.
//! Within a stripe samples come from the CDEF output; the three rows just above
//! and below come from the pre-CDEF (deblocked) frame instead — so this module
//! is handed *both* frames and picks per sample in `get_source_sample` (§7.17.6).
//! When CDEF is disabled the two frames are identical and the distinction is
//! moot. With super-resolution both frames arrive already upscaled (§7.16), so
//! this module only ever sees `UpscaledWidth`. This is the intra 4:4:4 subset
//! (`subsampling_x`/`y` are zero) and the single-tile path.
//!
//! Like the transform DSP and CDEF, the filters are transcribed straight from
//! the spec's indexed working arrays and constant tables, so the module opts into
//! `indexing_slicing` for them; runtime-sized sample planes still go through
//! [`Plane`]'s checked API.
#![allow(
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    reason = "fixed-size spec tables and small per-block working arrays, indexed \
              by spec-bounded constants; sample planes use Plane's checked API"
)]

use super::plane::Plane;
use super::symbol::SymbolDecoder;
use otf_pixels_core::Result;

/// `RESTORE_NONE` (§7.17).
pub const RESTORE_NONE: u8 = 0;
/// `RESTORE_WIENER` (§7.17).
pub const RESTORE_WIENER: u8 = 1;
/// `RESTORE_SGRPROJ` (§7.17).
pub const RESTORE_SGRPROJ: u8 = 2;
/// `RESTORE_SWITCHABLE` (§7.17).
pub const RESTORE_SWITCHABLE: u8 = 3;

/// `MI_SIZE` (§3).
const MI_SIZE: usize = 4;
/// `MI_SIZE_LOG2` (§3).
const MI_SIZE_LOG2: u32 = 2;
/// `FILTER_BITS` (§3): precision of the Wiener taps.
const FILTER_BITS: i32 = 7;
/// `WIENER_COEFFS` (§3): coded Wiener coefficients per pass.
pub const WIENER_COEFFS: usize = 3;
/// `SGRPROJ_PARAMS_BITS` (§3): bits of the self-guided parameter set index.
pub const SGRPROJ_PARAMS_BITS: u32 = 4;
/// `SGRPROJ_PRJ_SUBEXP_K` (§3).
pub const SGRPROJ_PRJ_SUBEXP_K: i32 = 4;
/// `SGRPROJ_PRJ_BITS` (§3): precision of the projection weights.
pub const SGRPROJ_PRJ_BITS: i32 = 7;
/// `SGRPROJ_RST_BITS` (§3).
const SGRPROJ_RST_BITS: i32 = 4;
/// `SGRPROJ_MTABLE_BITS` (§3).
const SGRPROJ_MTABLE_BITS: i32 = 20;
/// `SGRPROJ_RECIP_BITS` (§3).
const SGRPROJ_RECIP_BITS: i32 = 12;
/// `SGRPROJ_SGR_BITS` (§3).
const SGRPROJ_SGR_BITS: i32 = 8;

/// `Wiener_Taps_Min` (§5.11.58).
pub const WIENER_TAPS_MIN: [i32; 3] = [-5, -23, -17];
/// `Wiener_Taps_Max` (§5.11.58).
pub const WIENER_TAPS_MAX: [i32; 3] = [10, 8, 46];
/// `Wiener_Taps_K` (§5.11.58).
pub const WIENER_TAPS_K: [i32; 3] = [1, 2, 3];
/// `Wiener_Taps_Mid` (§5.11.2): the reset value of `RefLrWiener`.
pub const WIENER_TAPS_MID: [i32; 3] = [3, -7, 15];

/// `Sgrproj_Xqd_Min` (§5.11.58).
pub const SGRPROJ_XQD_MIN: [i32; 2] = [-96, -32];
/// `Sgrproj_Xqd_Max` (§5.11.58).
pub const SGRPROJ_XQD_MAX: [i32; 2] = [31, 95];
/// `Sgrproj_Xqd_Mid` (§5.11.2): the reset value of `RefSgrXqd`.
pub const SGRPROJ_XQD_MID: [i32; 2] = [-32, 31];

/// `Sgr_Params[set][0..4]` (§7.17.3): `{ r0, eps0, r1, eps1 }` per parameter set.
pub const SGR_PARAMS: [[i32; 4]; 16] = [
    [2, 12, 1, 4],
    [2, 15, 1, 6],
    [2, 18, 1, 8],
    [2, 21, 1, 9],
    [2, 24, 1, 10],
    [2, 29, 1, 11],
    [2, 36, 1, 12],
    [2, 45, 1, 13],
    [2, 56, 1, 14],
    [2, 68, 1, 15],
    [0, 0, 1, 5],
    [0, 0, 1, 8],
    [0, 0, 1, 11],
    [0, 0, 1, 14],
    [2, 30, 0, 0],
    [2, 75, 0, 0],
];

/// `count_units_in_frame(unitSize, frameSize)` (§5.11.57).
#[must_use]
pub fn count_units_in_frame(unit_size: usize, frame_size: usize) -> usize {
    ((frame_size + (unit_size >> 1)) / unit_size).max(1)
}

/// `inverse_recenter(r, v)` (§5.9.28).
fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `decode_subexp_bool(numSyms, k)` (§5.11.58): a sub-exponential code read from
/// the arithmetic decoder as uniform literals.
fn decode_subexp_bool(dec: &mut SymbolDecoder<'_>, num_syms: i32, k: i32) -> Result<i32> {
    let mut i = 0_i32;
    let mut mk = 0_i32;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1 << b2;
        if num_syms <= mk + 3 * a {
            let count = (num_syms - mk) as u32;
            let unif = dec.read_ns(count)? as i32;
            return Ok(unif + mk);
        }
        if dec.read_literal(1)? != 0 {
            i += 1;
            mk += a;
        } else {
            let bits = dec.read_literal(b2 as u32)? as i32;
            return Ok(bits + mk);
        }
    }
}

/// `decode_unsigned_subexp_with_ref_bool(mx, k, r)` (§5.11.58).
fn decode_unsigned_subexp_with_ref_bool(
    dec: &mut SymbolDecoder<'_>,
    mx: i32,
    k: i32,
    r: i32,
) -> Result<i32> {
    let v = decode_subexp_bool(dec, mx, k)?;
    if (r << 1) <= mx {
        Ok(inverse_recenter(r, v))
    } else {
        Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
    }
}

/// `decode_signed_subexp_with_ref_bool(low, high, k, r)` (§5.11.58): read a
/// signed value in `[low, high)` coded relative to the running reference `r`.
pub fn decode_signed_subexp_with_ref_bool(
    dec: &mut SymbolDecoder<'_>,
    low: i32,
    high: i32,
    k: i32,
    r: i32,
) -> Result<i32> {
    let x = decode_unsigned_subexp_with_ref_bool(dec, high - low, k, r - low)?;
    Ok(x + low)
}

/// Read one unit's Wiener coefficients (§5.11.58), two passes of three taps,
/// updating the running reference `ref_wiener`. Chroma (`is_chroma`) forces the
/// outermost tap to zero, giving the 5-tap chroma filter.
pub fn read_wiener_unit(
    dec: &mut SymbolDecoder<'_>,
    ref_wiener: &mut [[i32; WIENER_COEFFS]; 2],
    is_chroma: bool,
) -> Result<[[i32; WIENER_COEFFS]; 2]> {
    let mut out = [[0_i32; WIENER_COEFFS]; 2];
    for pass in 0..2 {
        let first = if is_chroma {
            out[pass][0] = 0;
            1
        } else {
            0
        };
        for j in first..WIENER_COEFFS {
            let v = decode_signed_subexp_with_ref_bool(
                dec,
                WIENER_TAPS_MIN[j],
                WIENER_TAPS_MAX[j] + 1,
                WIENER_TAPS_K[j],
                ref_wiener[pass][j],
            )?;
            out[pass][j] = v;
            ref_wiener[pass][j] = v;
        }
    }
    Ok(out)
}

/// Read one unit's self-guided parameters (§5.11.58): the parameter-set index
/// and the two projection weights, updating the running reference `ref_xqd`.
pub fn read_sgrproj_unit(
    dec: &mut SymbolDecoder<'_>,
    ref_xqd: &mut [i32; 2],
) -> Result<(u8, [i32; 2])> {
    let set = dec.read_literal(SGRPROJ_PARAMS_BITS)? as usize;
    let mut xqd = [0_i32; 2];
    for i in 0..2 {
        let radius = SGR_PARAMS[set][i * 2];
        let min = SGRPROJ_XQD_MIN[i];
        let max = SGRPROJ_XQD_MAX[i];
        xqd[i] = if radius != 0 {
            decode_signed_subexp_with_ref_bool(dec, min, max + 1, SGRPROJ_PRJ_SUBEXP_K, ref_xqd[i])?
        } else if i == 1 {
            ((1 << SGRPROJ_PRJ_BITS) - ref_xqd[0]).clamp(min, max)
        } else {
            0
        };
        ref_xqd[i] = xqd[i];
    }
    Ok((set as u8, xqd))
}

/// One plane's loop-restoration parameters, gathered during tile decode: the
/// frame-level filter type, the unit grid geometry and the per-unit filter data.
#[derive(Debug, Clone)]
pub struct PlaneLr {
    /// `FrameRestorationType[plane]`.
    pub frame_restoration_type: u8,
    /// `LoopRestorationSize[plane]` in samples.
    pub unit_size: usize,
    /// Restoration units down the frame for this plane.
    pub unit_rows: usize,
    /// Restoration units across the frame for this plane.
    pub unit_cols: usize,
    /// `LrType[unitRow][unitCol]`, row-major, `unit_cols` wide.
    pub lr_type: Vec<u8>,
    /// `LrWiener[unitRow][unitCol][pass][coeff]`, the three coded Wiener taps
    /// per pass (vertical then horizontal).
    pub wiener: Vec<[[i32; WIENER_COEFFS]; 2]>,
    /// `LrSgrSet[unitRow][unitCol]`.
    pub sgr_set: Vec<u8>,
    /// `LrSgrXqd[unitRow][unitCol][0..2]`, the two projection weights.
    pub sgr_xqd: Vec<[i32; 2]>,
}

impl PlaneLr {
    /// An empty per-plane record sized to its unit grid.
    #[must_use]
    pub fn new(frame_restoration_type: u8, unit_size: usize, rows: usize, cols: usize) -> Self {
        let count = rows * cols;
        Self {
            frame_restoration_type,
            unit_size,
            unit_rows: rows,
            unit_cols: cols,
            lr_type: vec![RESTORE_NONE; count],
            wiener: vec![[[0; WIENER_COEFFS]; 2]; count],
            sgr_set: vec![0; count],
            sgr_xqd: vec![[0; 2]; count],
        }
    }
}

/// Everything the loop-restoration filter reads and writes.
pub struct LoopRestore<'a> {
    /// `LrFrame`: filtered in place, having been initialised to a copy of the
    /// CDEF output by the caller.
    pub planes: &'a mut [Plane],
    /// `UpscaledCurrFrame`: the pre-CDEF (deblocked) reconstruction.
    pub curr: &'a [Plane],
    /// `UpscaledCdefFrame`: the CDEF output.
    pub cdef: &'a [Plane],
    /// Per-plane loop-restoration parameters (length `num_planes`).
    pub lr: &'a [PlaneLr],
    /// Sample bit depth (8/10/12).
    pub bit_depth: u8,
    /// Number of planes (3 for 4:4:4).
    pub num_planes: usize,
    /// Horizontal chroma subsampling (0 for 4:4:4).
    pub subsampling_x: usize,
    /// Vertical chroma subsampling (0 for 4:4:4).
    pub subsampling_y: usize,
    /// `UpscaledWidth` in luma samples.
    pub upscaled_width: usize,
    /// `FrameHeight` in luma samples.
    pub frame_height: usize,
}

/// The per-block filtering geometry that `get_source_sample` needs.
struct BlockRegion {
    plane: usize,
    stripe_start_y: i32,
    stripe_end_y: i32,
    plane_end_x: i32,
    plane_end_y: i32,
}

impl LoopRestore<'_> {
    /// Loop restoration process (§7.17): filter every restoration block. The
    /// caller has already copied the CDEF output into `planes`; blocks whose unit
    /// is `RESTORE_NONE` keep that copy.
    pub fn run(&mut self) {
        let mut y = 0;
        while y < self.frame_height {
            let mut x = 0;
            while x < self.upscaled_width {
                for plane in 0..self.num_planes {
                    if self.lr[plane].frame_restoration_type != RESTORE_NONE {
                        self.loop_restore_block(plane, y >> MI_SIZE_LOG2, x >> MI_SIZE_LOG2);
                    }
                }
                x += MI_SIZE;
            }
            y += MI_SIZE;
        }
    }

    /// Loop restore block process (§7.17.1): resolve the stripe, unit and region
    /// for the 4x4 block at luma `(row, col)`, then dispatch on the unit's type.
    fn loop_restore_block(&mut self, plane: usize, row: usize, col: usize) {
        let luma_y = row * MI_SIZE;
        let stripe_num = (luma_y + 8) / 64;
        let (sub_x, sub_y) = self.subsampling(plane);
        let stripe_start_y = (-8 + stripe_num as i32 * 64) >> sub_y;
        let stripe_end_y = stripe_start_y + (64 >> sub_y) - 1;

        let info = &self.lr[plane];
        let unit_size = info.unit_size;
        let unit_rows = info.unit_rows;
        let unit_cols = info.unit_cols;
        let unit_row = (((row * MI_SIZE + 8) >> sub_y) / unit_size).min(unit_rows - 1);
        let unit_col = (((col * MI_SIZE) >> sub_x) / unit_size).min(unit_cols - 1);

        let plane_end_x = (round2_usize(self.upscaled_width, sub_x) as i32) - 1;
        let plane_end_y = (round2_usize(self.frame_height, sub_y) as i32) - 1;
        let x = (col * MI_SIZE) >> sub_x;
        let y = (row * MI_SIZE) >> sub_y;
        let w = ((MI_SIZE >> sub_x) as i32).min(plane_end_x - x as i32 + 1);
        let h = ((MI_SIZE >> sub_y) as i32).min(plane_end_y - y as i32 + 1);
        if w <= 0 || h <= 0 {
            return;
        }
        let (w, h) = (w as usize, h as usize);

        let region = BlockRegion {
            plane,
            stripe_start_y,
            stripe_end_y,
            plane_end_x,
            plane_end_y,
        };
        let unit_idx = unit_row * unit_cols + unit_col;
        match info.lr_type.get(unit_idx).copied().unwrap_or(RESTORE_NONE) {
            RESTORE_WIENER => self.wiener_filter(&region, unit_idx, x, y, w, h),
            RESTORE_SGRPROJ => self.self_guided_filter(&region, unit_idx, x, y, w, h),
            _ => {}
        }
    }

    fn subsampling(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.subsampling_x, self.subsampling_y)
        }
    }

    /// Get source sample process (§7.17.6): fetch from the CDEF output inside the
    /// stripe, else from the pre-CDEF frame, with the coordinates cropped to the
    /// plane and to two lines beyond the stripe.
    fn get_source_sample(&self, region: &BlockRegion, x: i32, y: i32) -> i32 {
        let plane = region.plane;
        let x = x.clamp(0, region.plane_end_x);
        let y = y.clamp(0, region.plane_end_y);
        let sample = |frame: &[Plane], sy: i32| {
            i32::from(
                frame
                    .get(plane)
                    .map(|p| p.sample_clamped(x as isize, sy as isize))
                    .unwrap_or(0),
            )
        };
        if y < region.stripe_start_y {
            sample(self.curr, (region.stripe_start_y - 2).max(y))
        } else if y > region.stripe_end_y {
            sample(self.curr, (region.stripe_end_y + 2).min(y))
        } else {
            sample(self.cdef, y)
        }
    }

    /// Wiener filter process (§7.17.4): horizontal then vertical 7-tap symmetric
    /// convolution.
    fn wiener_filter(
        &mut self,
        region: &BlockRegion,
        unit_idx: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) {
        let plane = region.plane;
        let coeffs = self.lr[plane].wiener[unit_idx];
        let vfilter = wiener_coefficient(coeffs[0]);
        let hfilter = wiener_coefficient(coeffs[1]);

        // Rounding variables (§7.11.3.2, isCompound = 0).
        let bd = i32::from(self.bit_depth);
        let inter_round0 = if bd == 12 { 5 } else { 3 };
        let inter_round1 = if bd == 12 { 9 } else { 11 };
        let offset = 1 << (bd + FILTER_BITS - inter_round0 - 1);
        let limit = (1 << (bd + 1 + FILTER_BITS - inter_round0)) - 1;

        // intermediate[h+6][w] from the horizontal filter.
        let stride = w;
        let mut intermediate = vec![0_i32; (h + 6) * stride];
        for r in 0..h + 6 {
            for c in 0..w {
                let mut s = 0_i32;
                for t in 0..7 {
                    let px = self.get_source_sample(
                        region,
                        x as i32 + c as i32 + t as i32 - 3,
                        y as i32 + r as i32 - 3,
                    );
                    s += hfilter[t] * px;
                }
                let v = round2(s, inter_round0);
                intermediate[r * stride + c] = v.clamp(-offset, limit - offset);
            }
        }

        // Vertical filter to the output.
        for r in 0..h {
            for c in 0..w {
                let mut s = 0_i32;
                for t in 0..7 {
                    s += vfilter[t] * intermediate[(r + t) * stride + c];
                }
                let v = round2(s, inter_round1);
                self.put(plane, x + c, y + r, self.clip1(v));
            }
        }
    }

    /// Self guided filter process (§7.17.2): blend the two box-filter passes and
    /// the source with the coded projection weights.
    fn self_guided_filter(
        &mut self,
        region: &BlockRegion,
        unit_idx: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
    ) {
        let plane = region.plane;
        let set = usize::from(self.lr[plane].sgr_set[unit_idx]);
        let flt0 = self.box_filter(region, x, y, w, h, set, 0);
        let flt1 = self.box_filter(region, x, y, w, h, set, 1);

        let xqd = self.lr[plane].sgr_xqd[unit_idx];
        let w0 = xqd[0];
        let w1 = xqd[1];
        let w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1;
        let r0 = SGR_PARAMS[set][0];
        let r1 = SGR_PARAMS[set][2];

        for i in 0..h {
            for j in 0..w {
                let u = i32::from(self.cdef_sample(plane, x + j, y + i)) << SGRPROJ_RST_BITS;
                let mut v = w1 * u;
                v += w0 * if r0 != 0 { flt0[i * w + j] } else { u };
                v += w2 * if r1 != 0 { flt1[i * w + j] } else { u };
                let s = round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS);
                self.put(plane, x + j, y + i, self.clip1(s));
            }
        }
    }

    /// Box filter process (§7.17.3): the per-pass guided average, returned as a
    /// flat `h*w` array (all-zero when the pass radius is 0 and thus unused).
    #[allow(clippy::too_many_arguments, reason = "mirrors the spec's input list")]
    fn box_filter(
        &self,
        region: &BlockRegion,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        set: usize,
        pass: usize,
    ) -> Vec<i32> {
        let mut f = vec![0_i32; h * w];
        let r = SGR_PARAMS[set][pass * 2];
        if r == 0 {
            return f;
        }
        let eps = SGR_PARAMS[set][pass * 2 + 1];
        let bd = i32::from(self.bit_depth);
        let n = (2 * r + 1) * (2 * r + 1);
        let n2e = i64::from(n * n * eps);
        let s = ((1_i64 << SGRPROJ_MTABLE_BITS) + n2e / 2) / n2e;
        let one_over_n = ((1_i64 << SGRPROJ_RECIP_BITS) + i64::from(n / 2)) / i64::from(n);

        // A and B carry a one-sample border: index [i+1][j+1] for i in -1..=h.
        let aw = w + 2;
        let mut a_arr = vec![0_i64; (h + 2) * aw];
        let mut b_arr = vec![0_i64; (h + 2) * aw];
        for i in -1..=(h as i32) {
            for j in -1..=(w as i32) {
                let mut acc = 0_i64;
                let mut bcc = 0_i64;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let c = i64::from(self.get_source_sample(
                            region,
                            x as i32 + j + dx,
                            y as i32 + i + dy,
                        ));
                        acc += c * c;
                        bcc += c;
                    }
                }
                let a = round2_i64(acc, 2 * (bd - 8));
                let d = round2_i64(bcc, bd - 8);
                let p = (a * i64::from(n) - d * d).max(0);
                let z = round2_i64(p * s, SGRPROJ_MTABLE_BITS);
                let a2 = if z >= 255 {
                    256
                } else if z == 0 {
                    1
                } else {
                    ((z << SGRPROJ_SGR_BITS) + z / 2) / (z + 1)
                };
                let b2 = ((1_i64 << SGRPROJ_SGR_BITS) - a2) * bcc * one_over_n;
                let idx = ((i + 1) as usize) * aw + (j + 1) as usize;
                a_arr[idx] = a2;
                b_arr[idx] = round2_i64(b2, SGRPROJ_RECIP_BITS);
            }
        }

        for i in 0..h {
            let shift = if pass == 0 && (i & 1) != 0 { 4 } else { 5 };
            for j in 0..w {
                let mut a = 0_i64;
                let mut b = 0_i64;
                for dy in -1_i32..=1 {
                    for dx in -1_i32..=1 {
                        let weight = if pass == 0 {
                            if ((i as i32 + dy) & 1) != 0 {
                                if dx == 0 { 6 } else { 5 }
                            } else {
                                0
                            }
                        } else if dx == 0 || dy == 0 {
                            4
                        } else {
                            3
                        };
                        let idx =
                            ((i as i32 + dy + 1) as usize) * aw + (j as i32 + dx + 1) as usize;
                        a += weight * a_arr[idx];
                        b += weight * b_arr[idx];
                    }
                }
                let v = a * i64::from(self.cdef_sample(region.plane, x + j, y + i)) + b;
                f[i * w + j] = round2_i64(v, SGRPROJ_SGR_BITS + shift - SGRPROJ_RST_BITS) as i32;
            }
        }
        f
    }

    /// A sample of the CDEF output at plane coordinates.
    fn cdef_sample(&self, plane: usize, x: usize, y: usize) -> u16 {
        self.cdef.get(plane).and_then(|p| p.get(x, y)).unwrap_or(0)
    }

    /// Write an output sample into `LrFrame`.
    fn put(&mut self, plane: usize, x: usize, y: usize, value: u16) {
        if let Some(p) = self.planes.get_mut(plane) {
            p.set(x, y, value);
        }
    }

    /// `Clip1(x)` (§4.7): clip to the sample range for the bit depth.
    fn clip1(&self, x: i32) -> u16 {
        let max = (1 << self.bit_depth) - 1;
        x.clamp(0, max) as u16
    }
}

/// Wiener coefficient process (§7.17.5): expand three coded taps into the full
/// symmetric 7-tap unit-DC-gain filter.
fn wiener_coefficient(coeff: [i32; WIENER_COEFFS]) -> [i32; 7] {
    let mut filter = [0_i32; 7];
    filter[3] = 128;
    for i in 0..3 {
        let c = coeff[i];
        filter[i] = c;
        filter[6 - i] = c;
        filter[3] -= 2 * c;
    }
    filter
}

/// `Round2(x, n)` (§4.7) for `i32`.
fn round2(x: i32, n: i32) -> i32 {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

/// `Round2(x, n)` (§4.7) for `i64`.
fn round2_i64(x: i64, n: i32) -> i64 {
    if n == 0 {
        x
    } else {
        (x + (1_i64 << (n - 1))) >> n
    }
}

/// `Round2(x, n)` (§4.7) on a `usize`.
fn round2_usize(x: usize, n: usize) -> usize {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    #[test]
    fn wiener_coefficients_are_symmetric_with_unit_dc_gain() {
        // Luma taps: the full filter mirrors the three coded coefficients and the
        // seven taps sum to 128 (unit DC gain at FILTER_BITS precision).
        let f = wiener_coefficient([3, -7, 15]);
        assert_eq!(f[0], f[6]);
        assert_eq!(f[1], f[5]);
        assert_eq!(f[2], f[4]);
        assert_eq!(f.iter().sum::<i32>(), 128);
        // Chroma: coeff[0] == 0 makes it a 5-tap filter (ends are zero).
        let c = wiener_coefficient([0, -7, 15]);
        assert_eq!(c[0], 0);
        assert_eq!(c[6], 0);
        assert_eq!(c.iter().sum::<i32>(), 128);
    }

    #[test]
    fn inverse_recenter_round_trips_small_values() {
        // The recentering is a bijection on [0, 2r]; check a few points.
        assert_eq!(inverse_recenter(10, 0), 10);
        assert_eq!(inverse_recenter(10, 1), 10 - 1);
        assert_eq!(inverse_recenter(10, 2), 10 + 1);
        assert_eq!(inverse_recenter(10, 25), 25); // v > 2r passes through
    }

    #[test]
    fn count_units_rounds_to_nearest_and_is_at_least_one() {
        assert_eq!(count_units_in_frame(64, 64), 1); // (64 + 32) / 64 = 1
        assert_eq!(count_units_in_frame(64, 96), 2); // (96 + 32) / 64 = 2
        assert_eq!(count_units_in_frame(64, 31), 1); // rounds down to 0, floored to 1
        assert_eq!(count_units_in_frame(256, 10), 1); // never zero
    }

    /// A `w x h` plane whose samples follow `f(x, y)`.
    fn plane_from(w: usize, h: usize, f: impl Fn(usize, usize) -> u16) -> Plane {
        let mut plane = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                plane.set(x, y, f(x, y));
            }
        }
        plane
    }

    /// Run loop restoration over one plane with a single unit of `lr`, the CDEF
    /// output `cdef` and the pre-CDEF frame `curr`, returning the result.
    fn restore(cdef: &Plane, curr: &Plane, lr: PlaneLr) -> Plane {
        let mut planes = [cdef.clone()];
        let cdef_frame = [cdef.clone()];
        let curr_frame = [curr.clone()];
        let lr = [lr];
        LoopRestore {
            planes: &mut planes,
            curr: &curr_frame,
            cdef: &cdef_frame,
            lr: &lr,
            bit_depth: 8,
            num_planes: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            upscaled_width: cdef.width(),
            frame_height: cdef.height(),
        }
        .run();
        let [out] = planes;
        out
    }

    /// A deterministic, non-smooth test pattern.
    fn busy(x: usize, y: usize) -> u16 {
        ((x * 37 + y * 91 + (x * y) % 13) % 256) as u16
    }

    #[test]
    fn a_zero_tap_wiener_unit_is_the_identity() {
        // All three coded taps zero expand to the filter [0,0,0,128,0,0,0]: at
        // FILTER_BITS precision that passes every sample through unchanged.
        let cdef = plane_from(24, 20, busy);
        let mut lr = PlaneLr::new(RESTORE_WIENER, 64, 1, 1);
        lr.lr_type[0] = RESTORE_WIENER;
        lr.wiener[0] = [[0; WIENER_COEFFS]; 2];
        let out = restore(&cdef, &cdef, lr);
        assert_eq!(out.samples(), cdef.samples());
    }

    #[test]
    fn a_smoothing_wiener_unit_changes_a_busy_plane_but_keeps_a_flat_one() {
        let mut lr = PlaneLr::new(RESTORE_WIENER, 64, 1, 1);
        lr.lr_type[0] = RESTORE_WIENER;
        lr.wiener[0] = [WIENER_TAPS_MID; 2];
        // Unit DC gain: a flat plane is a fixed point.
        let flat = plane_from(16, 16, |_, _| 97);
        assert_eq!(restore(&flat, &flat, lr.clone()).samples(), flat.samples());
        // A busy one is actually filtered.
        let busy_plane = plane_from(16, 16, busy);
        assert_ne!(
            restore(&busy_plane, &busy_plane, lr).samples(),
            busy_plane.samples()
        );
    }

    #[test]
    fn a_self_guided_unit_keeps_a_flat_plane_flat() {
        // For every parameter set, both box filters of a flat region return the
        // region itself, so any projection weights reproduce it.
        let flat = plane_from(16, 16, |_, _| 150);
        for set in 0..16 {
            let mut lr = PlaneLr::new(RESTORE_SGRPROJ, 64, 1, 1);
            lr.lr_type[0] = RESTORE_SGRPROJ;
            lr.sgr_set[0] = set;
            lr.sgr_xqd[0] = SGRPROJ_XQD_MID;
            assert_eq!(
                restore(&flat, &flat, lr).samples(),
                flat.samples(),
                "set {set}"
            );
        }
    }

    #[test]
    fn samples_outside_the_stripe_come_from_the_pre_cdef_frame() {
        // Rows 0..56 are stripe 0 (it ends at luma row 55); rows 56.. are
        // stripe 1. A block in stripe 0 must read rows past 55 from the
        // pre-CDEF frame, at most two lines beyond the stripe.
        let cdef = [plane_from(8, 80, |_, _| 10)];
        let curr = [plane_from(8, 80, |_, y| y as u16)];
        let lr = [PlaneLr::new(RESTORE_WIENER, 64, 1, 1)];
        let mut out = cdef.clone();
        let restore = LoopRestore {
            planes: &mut out,
            curr: &curr,
            cdef: &cdef,
            lr: &lr,
            bit_depth: 8,
            num_planes: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            upscaled_width: 8,
            frame_height: 80,
        };
        let region = BlockRegion {
            plane: 0,
            stripe_start_y: -8,
            stripe_end_y: 55,
            plane_end_x: 7,
            plane_end_y: 79,
        };
        // Inside the stripe: the CDEF output.
        assert_eq!(restore.get_source_sample(&region, 3, 55), 10);
        // Below it: the pre-CDEF frame, clamped to stripe_end + 2 = 57.
        assert_eq!(restore.get_source_sample(&region, 3, 56), 56);
        assert_eq!(restore.get_source_sample(&region, 3, 60), 57);
        // Above the frame: cropped to row 0, which is inside the stripe.
        assert_eq!(restore.get_source_sample(&region, 3, -3), 10);
    }

    #[test]
    fn none_type_leaves_the_frame_as_the_cdef_copy() {
        let (w, h) = (16, 16);
        let mut plane = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                plane.set(x, y, (x + y) as u16);
            }
        }
        let cdef = vec![plane.clone()];
        let curr = vec![plane.clone()];
        let mut planes = [plane.clone()];
        let lr = vec![PlaneLr::new(RESTORE_NONE, 64, 1, 1)];
        LoopRestore {
            planes: &mut planes,
            curr: &curr,
            cdef: &cdef,
            lr: &lr,
            bit_depth: 8,
            num_planes: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            upscaled_width: w,
            frame_height: h,
        }
        .run();
        assert_eq!(planes[0].samples(), plane.samples());
    }
}
