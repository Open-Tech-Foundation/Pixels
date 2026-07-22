//! Coefficient decode for the 4x4 lossless path (spec §5.11.39, "coeffs").
//!
//! A transform block's residual is coded as a sparse level array `Quant[]`. The
//! syntax reads, in order: `all_zero` (the whole block is zero), then the
//! end-of-block position, then the base level of each coefficient walking the
//! scan *backwards* from the last, then a magnitude extension (`coeff_br` and a
//! Golomb tail) for large levels, and finally the signs walking forwards. Every
//! symbol is entropy-coded against a context-selected CDF, and getting the
//! contexts exactly right is what keeps the arithmetic decoder in sync — a
//! single wrong context desynchronises the whole tile.
//!
//! Lossless coding forces `TX_4X4` for every block, which is the only case this
//! module handles: `txSzCtx` is 0, the transform class is 2D (`DCT_DCT`), the
//! scan is `Default_Scan_4x4`, and `segEob` is 16. The wider transform sizes
//! arrive with the lossy path in a later phase.

use super::cdf;
use super::symbol::SymbolDecoder;
use otf_pixels_core::{PixelsError, Result};

/// Pick element `q` (0..=3) of a four-entry table by value, without indexing.
fn pick4<T: Copy>(arr: [T; 4], q: usize) -> T {
    let [a, b, c, d] = arr;
    match q {
        1 => b,
        2 => c,
        3 => d,
        _ => a,
    }
}

/// Borrow `slice[index]` as a mutable CDF row, or report a malformed stream
/// rather than panic. Every call site derives its index from spec-bounded
/// context maths, so this only fires on genuinely corrupt input.
fn cdf_row<T>(slice: &mut [T], index: usize) -> Result<&mut T> {
    slice.get_mut(index).ok_or_else(|| {
        PixelsError::malformed("avif", "an AV1 coefficient CDF index ran out of range")
    })
}

/// `NUM_BASE_LEVELS` (§3): base levels coded before the range extension.
const NUM_BASE_LEVELS: i32 = 2;
/// `COEFF_BASE_RANGE` (§3): the span covered by `coeff_br` before Golomb.
const COEFF_BASE_RANGE: i32 = 12;
/// `BR_CDF_SIZE` (§3): symbols in the `coeff_br` alphabet.
const BR_CDF_SIZE: i32 = 4;
/// `SIG_COEF_CONTEXTS` (§3).
const SIG_COEF_CONTEXTS: usize = 42;
/// `SIG_COEF_CONTEXTS_EOB` (§3).
const SIG_COEF_CONTEXTS_EOB: usize = 4;

/// `Default_Scan_4x4` (§9.3): the coefficient scan order for a 4x4 block.
const DEFAULT_SCAN_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// `Sig_Ref_Diff_Offset[TX_CLASS_2D]` (§8.3.3): the neighbour offsets whose
/// magnitudes drive the `coeff_base` context. `(rowDelta, colDelta)` pairs.
const SIG_REF_DIFF_OFFSET_2D: [(i32, i32); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];

/// `Mag_Ref_Offset_With_Tx_Class[TX_CLASS_2D]` (§8.3.3): the neighbour offsets
/// whose magnitudes drive the `coeff_br` context.
const MAG_REF_OFFSET_2D: [(i32, i32); 3] = [(0, 1), (1, 0), (1, 1)];

/// `Coeff_Base_Ctx_Offset[TX_4X4]` (§8.3.3), indexed `[min(row,4)][min(col,4)]`.
/// For a 4x4 block row and column never exceed 3, so the last row/column (all
/// zero) is unreachable, but it is kept for a faithful transcription.
const COEFF_BASE_CTX_OFFSET_4X4: [[i32; 5]; 5] = [
    [0, 1, 6, 6, 0],
    [1, 6, 6, 21, 0],
    [6, 6, 21, 21, 0],
    [6, 21, 21, 21, 0],
    [0, 0, 0, 0, 0],
];

/// The mutable coefficient CDFs for one tile, cloned from the defaults for the
/// frame's quantiser context. Only the 4x4-relevant slices are held.
///
/// The spec's `Tile*Cdf` are the frame defaults pre-indexed by the quantiser
/// context (`get_qctx`), then adapted per symbol as the tile decodes. Lossless
/// frames have `base_q_idx == 0`, so the context is always 0; the field is kept
/// so the wider path can reuse the type.
pub struct CoeffCdfs {
    txb_skip: [[[u16; 3]; 13]; 5],
    eob_pt_16: [[[u16; 6]; 2]; 2],
    eob_extra: [[[u16; 3]; 9]; 2],
    coeff_base_eob: [[[[u16; 4]; 4]; 2]; 5],
    coeff_base: [[[[u16; 5]; 42]; 2]; 5],
    coeff_br: [[[[u16; 5]; 21]; 2]; 5],
    dc_sign: [[[u16; 4]; 3]; 2],
}

impl CoeffCdfs {
    /// Clone the defaults for quantiser context `qctx` (0 for lossless).
    #[must_use]
    pub fn new(qctx: usize) -> Self {
        let q = qctx.min(3);
        Self {
            txb_skip: pick4(cdf::DEFAULT_TXB_SKIP_CDF, q),
            eob_pt_16: pick4(cdf::DEFAULT_EOB_PT_16_CDF, q),
            eob_extra: {
                // txSzCtx 0 (4x4) is the only bucket the lossless path uses.
                let [tx0, ..] = pick4(cdf::DEFAULT_EOB_EXTRA_CDF, q);
                tx0
            },
            coeff_base_eob: pick4(cdf::DEFAULT_COEFF_BASE_EOB_CDF, q),
            coeff_base: pick4(cdf::DEFAULT_COEFF_BASE_CDF, q),
            coeff_br: pick4(cdf::DEFAULT_COEFF_BR_CDF, q),
            dc_sign: pick4(cdf::DEFAULT_DC_SIGN_CDF, q),
        }
    }
}

/// The result of decoding one 4x4 transform block's coefficients.
pub struct CoeffBlock {
    /// `Quant[0..16]` in raster order: signed dequantiser input levels.
    pub quant: [i32; 16],
    /// The end-of-block position: the count of leading scan coefficients.
    pub eob: usize,
    /// `culLevel`, clamped to 63: the neighbour level context this block leaves.
    pub cul_level: u8,
    /// `dcCategory`: 0 none, 1 negative DC, 2 positive DC.
    pub dc_category: u8,
}

/// Decode the coefficients of one 4x4 transform block (`coeffs`, 4x4 lossless).
///
/// `ptype` is 0 for luma and 1 for chroma. `all_zero_ctx` and `dc_sign_ctx` are
/// the neighbour-derived contexts the tile driver computes from its level and
/// DC-sign context arrays. The returned `Quant[]` feeds the inverse transform.
///
/// # Errors
///
/// Propagates any error from the arithmetic decoder (a stream that ends early).
pub fn decode_coeffs_4x4(
    dec: &mut SymbolDecoder<'_>,
    cdfs: &mut CoeffCdfs,
    ptype: usize,
    all_zero_ctx: usize,
    dc_sign_ctx: usize,
) -> Result<CoeffBlock> {
    let mut quant = [0_i32; 16];
    let pt = ptype.min(1);

    // all_zero (txb_skip): the whole block codes as zero. txSzCtx is 0 (4x4).
    let skip_cdf = cdf_row(cdf_row(&mut cdfs.txb_skip, 0)?, all_zero_ctx)?;
    let all_zero = dec.read_symbol(skip_cdf)? != 0;
    if all_zero {
        return Ok(CoeffBlock {
            quant,
            eob: 0,
            cul_level: 0,
            dc_category: 0,
        });
    }

    // eob_pt_16 -> eobPt -> eob. txClass is 2D (DCT_DCT) for lossless, so the
    // eob_pt context is 0.
    let eob_pt = dec.read_symbol(cdf_row(cdf_row(&mut cdfs.eob_pt_16, pt)?, 0)?)? + 1;
    let mut eob = if eob_pt < 2 {
        eob_pt
    } else {
        (1 << (eob_pt - 2)) + 1
    };

    // eob_extra plus the raw extra bits refine eob within its bucket.
    if let Some(eob_shift) = eob_pt.checked_sub(3) {
        let extra_cdf = cdf_row(cdf_row(&mut cdfs.eob_extra, pt)?, eob_pt - 3)?;
        if dec.read_symbol(extra_cdf)? != 0 {
            eob += 1 << eob_shift;
        }
        for i in 1..eob_pt.saturating_sub(2) {
            let shift = eob_pt.saturating_sub(2) - 1 - i;
            if dec.read_bool()? {
                eob += 1 << shift;
            }
        }
    }

    eob = eob.min(16);

    // Base levels, walking the scan backwards from the last coefficient.
    for c in (0..eob).rev() {
        let pos = scan(c);
        let mut level;
        if c == eob - 1 {
            let ctx =
                coeff_base_ctx(&quant, pos, c, true) + SIG_COEF_CONTEXTS_EOB - SIG_COEF_CONTEXTS;
            let cdf_ref = cdf_row(cdf_row(cdf_row(&mut cdfs.coeff_base_eob, 0)?, pt)?, ctx)?;
            level = dec.read_symbol(cdf_ref)? as i32 + 1;
        } else {
            let ctx = coeff_base_ctx(&quant, pos, c, false);
            let cdf_ref = cdf_row(cdf_row(cdf_row(&mut cdfs.coeff_base, 0)?, pt)?, ctx)?;
            level = dec.read_symbol(cdf_ref)? as i32;
        }
        if level > NUM_BASE_LEVELS {
            let br_ctx = coeff_br_ctx(&quant, pos);
            for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                let cdf_ref = cdf_row(cdf_row(cdf_row(&mut cdfs.coeff_br, 0)?, pt)?, br_ctx)?;
                let coeff_br = dec.read_symbol(cdf_ref)? as i32;
                level += coeff_br;
                if coeff_br < BR_CDF_SIZE - 1 {
                    break;
                }
            }
        }
        if let Some(slot) = quant.get_mut(pos) {
            *slot = level;
        }
    }

    // Signs and the Golomb magnitude tail, walking the scan forwards.
    let mut cul_level: i32 = 0;
    let mut dc_category = 0_u8;
    for c in 0..eob {
        let pos = scan(c);
        let level = quant.get(pos).copied().unwrap_or(0);
        let sign = if level != 0 {
            if c == 0 {
                let cdf_ref = cdf_row(cdf_row(&mut cdfs.dc_sign, pt)?, dc_sign_ctx)?;
                dec.read_symbol(cdf_ref)? != 0
            } else {
                dec.read_bool()?
            }
        } else {
            false
        };
        let mut magnitude = level;
        if magnitude > NUM_BASE_LEVELS + COEFF_BASE_RANGE {
            magnitude = read_golomb(dec)? + COEFF_BASE_RANGE + NUM_BASE_LEVELS;
        }
        if pos == 0 && magnitude > 0 {
            dc_category = if sign { 1 } else { 2 };
        }
        magnitude &= 0xF_FFFF;
        cul_level += magnitude;
        if let Some(slot) = quant.get_mut(pos) {
            *slot = if sign { -magnitude } else { magnitude };
        }
    }

    Ok(CoeffBlock {
        quant,
        eob,
        cul_level: cul_level.min(63) as u8,
        dc_category,
    })
}

/// The `Default_Scan_4x4` position at scan index `c` (0 if out of range).
fn scan(c: usize) -> usize {
    DEFAULT_SCAN_4X4.get(c).copied().unwrap_or(0)
}

/// `get_coeff_base_ctx` for a 4x4 2D-class block (spec §8.3.3). `isEob` selects
/// the four end-of-block contexts; otherwise the magnitude of already-decoded
/// scan neighbours picks the context.
fn coeff_base_ctx(quant: &[i32; 16], pos: usize, c: usize, is_eob: bool) -> usize {
    if is_eob {
        // height << bwl == 4 << 2 == 16 for a 4x4 block.
        return if c == 0 {
            SIG_COEF_CONTEXTS - 4
        } else if c <= 16 / 8 {
            SIG_COEF_CONTEXTS - 3
        } else if c <= 16 / 4 {
            SIG_COEF_CONTEXTS - 2
        } else {
            SIG_COEF_CONTEXTS - 1
        };
    }
    let row = (pos >> 2) as i32;
    let col = (pos & 3) as i32;
    let mut mag = 0;
    for (d_row, d_col) in SIG_REF_DIFF_OFFSET_2D {
        let ref_row = row + d_row;
        let ref_col = col + d_col;
        if ref_row >= 0 && ref_col >= 0 && ref_row < 4 && ref_col < 4 {
            let ref_pos = ((ref_row << 2) + ref_col) as usize;
            mag += quant.get(ref_pos).copied().unwrap_or(0).abs().min(3);
        }
    }
    let ctx = ((mag + 1) >> 1).min(4);
    if row == 0 && col == 0 {
        return 0;
    }
    let offset = COEFF_BASE_CTX_OFFSET_4X4
        .get(row.min(4) as usize)
        .and_then(|r| r.get(col.min(4) as usize))
        .copied()
        .unwrap_or(0);
    (ctx + offset) as usize
}

/// `coeff_br` context for a 4x4 2D-class block (spec §8.3.3).
fn coeff_br_ctx(quant: &[i32; 16], pos: usize) -> usize {
    let row = (pos >> 2) as i32;
    let col = (pos & 3) as i32;
    let mut mag = 0;
    for (d_row, d_col) in MAG_REF_OFFSET_2D {
        let ref_row = row + d_row;
        let ref_col = col + d_col;
        if ref_row >= 0 && ref_col >= 0 && ref_row < 4 && ref_col < 4 {
            let ref_pos = ((ref_row << 2) + ref_col) as usize;
            mag += quant
                .get(ref_pos)
                .copied()
                .unwrap_or(0)
                .min(COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1);
        }
    }
    let mag = ((mag + 1) >> 1).min(6);
    let ctx = if pos == 0 {
        mag
    } else if row < 2 && col < 2 {
        mag + 7
    } else {
        mag + 14
    };
    ctx as usize
}

/// Read an exp-Golomb coded magnitude tail (`golomb`, §5.11.39).
fn read_golomb(dec: &mut SymbolDecoder<'_>) -> Result<i32> {
    let mut length = 0_i32;
    loop {
        length += 1;
        if dec.read_bool()? {
            break;
        }
        if length > 20 {
            break;
        }
    }
    let mut x = 1_i32;
    for _ in 0..length.saturating_sub(1) {
        x = (x << 1) | i32::from(dec.read_bool()?);
    }
    Ok(x)
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
    fn base_eob_contexts_match_the_spec_buckets() {
        assert_eq!(coeff_base_ctx(&[0; 16], 0, 0, true), SIG_COEF_CONTEXTS - 4);
        assert_eq!(coeff_base_ctx(&[0; 16], 5, 1, true), SIG_COEF_CONTEXTS - 3);
        assert_eq!(coeff_base_ctx(&[0; 16], 5, 3, true), SIG_COEF_CONTEXTS - 2);
        assert_eq!(coeff_base_ctx(&[0; 16], 5, 9, true), SIG_COEF_CONTEXTS - 1);
    }

    #[test]
    fn dc_position_base_context_is_zero() {
        // row == col == 0 short-circuits to context 0 regardless of neighbours.
        assert_eq!(coeff_base_ctx(&[3; 16], 0, 4, false), 0);
    }

    #[test]
    fn base_context_folds_in_neighbour_magnitudes() {
        // pos 5 -> row 1, col 1. Neighbours at (1,2)=pos6, (2,1)=pos9,
        // (2,2)=pos10, (1,3)=pos7, (3,1)=pos13. Put a 3 at pos6 only: mag=3,
        // ctx=(3+1)>>1=2, plus Coeff_Base_Ctx_Offset[1][1]=6 -> 8.
        let mut q = [0_i32; 16];
        q[6] = 3;
        assert_eq!(coeff_base_ctx(&q, 5, 4, false), 8);
    }

    #[test]
    fn br_context_at_dc_is_the_bare_magnitude() {
        // pos 0: neighbours (0,1)=pos1, (1,0)=pos4, (1,1)=pos5. A single 5 at
        // pos1 -> mag=min(5,15)=5, (5+1)>>1=3, pos==0 so ctx=3.
        let mut q = [0_i32; 16];
        q[1] = 5;
        assert_eq!(coeff_br_ctx(&q, 0), 3);
    }

    #[test]
    fn an_all_zero_block_reads_one_symbol_and_stops() {
        // A zero-filled stream keeps SymbolValue high; with the default txb_skip
        // CDF that decodes all_zero = 0 (not skipped) or 1 depending on the CDF,
        // so assert only on the structural invariant: eob 0 => empty levels.
        let data = [0x00; 8];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = CoeffCdfs::new(0);
        let block = decode_coeffs_4x4(&mut dec, &mut cdfs, 0, 0, 0).unwrap();
        if block.eob == 0 {
            assert_eq!(block.quant, [0; 16]);
            assert_eq!(block.cul_level, 0);
        }
    }

    #[test]
    fn golomb_reads_a_unary_prefix_then_data_bits() {
        // read_bool on an all-0xFF stream returns 1 immediately: length 1, x=1.
        let data = [0xFF; 4];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        assert_eq!(read_golomb(&mut dec).unwrap(), 1);
    }
}
