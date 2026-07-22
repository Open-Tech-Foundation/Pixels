//! Coefficient decode (spec §5.11.39, "coeffs") for every transform size.
//!
//! A transform block's residual is coded as a sparse level array `Quant[]`. The
//! syntax reads, in order: `all_zero` (the whole block is zero), then the
//! end-of-block position (`eob_pt` in one of seven size-dependent alphabets plus
//! refinement bits), then the base level of each coefficient walking the scan
//! *backwards* from the last, then a magnitude extension (`coeff_br` and a
//! Golomb tail) for large levels, and finally the signs walking forwards. Every
//! symbol is entropy-coded against a context-selected CDF, and getting the
//! contexts exactly right is what keeps the arithmetic decoder in sync — a
//! single wrong context desynchronises the whole tile.
//!
//! The contexts depend on the transform size (`txSzCtx`, the coded block's
//! `bwl`) and the transform *class* — whether the transform is separable 2D or a
//! 1D row/column identity transform, which reshapes the neighbour offsets and
//! scan order. The lossless path is the `TX_4X4` / `DCT_DCT` (2D) corner of this
//! and flows through the same entry point.

use super::cdf;
use super::symbol::SymbolDecoder;
use super::transform::{TxSize, TxType};
use otf_pixels_core::{PixelsError, Result};

include!("scan_tables.rs");

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
/// `SIG_COEF_CONTEXTS_2D` (§3).
const SIG_COEF_CONTEXTS_2D: i32 = 26;
/// `SIG_COEF_CONTEXTS_EOB` (§3).
const SIG_COEF_CONTEXTS_EOB: usize = 4;
/// The largest scan a block can code (`TX_32X32`), the `Quant[]` capacity.
const MAX_COEFFS: usize = 1024;

/// Transform class (`get_tx_class`, §8.3.3): 0 = 2D, 1 = horizontal (1D row
/// identity), 2 = vertical (1D column identity). The class selects the neighbour
/// offsets and the scan order.
fn tx_class(tx_type: TxType) -> usize {
    match tx_type {
        TxType::VDct | TxType::VAdst | TxType::VFlipadst => 2,
        TxType::HDct | TxType::HAdst | TxType::HFlipadst => 1,
        _ => 0,
    }
}

/// `Sig_Ref_Diff_Offset[txClass][idx]` (§8.3.3): the neighbour offsets whose
/// magnitudes drive the `coeff_base` context, as `(rowDelta, colDelta)`.
const SIG_REF_DIFF_OFFSET: [[(i32, i32); 5]; 3] = [
    [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)],
    [(0, 1), (1, 0), (0, 2), (0, 3), (0, 4)],
    [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
];

/// `Mag_Ref_Offset_With_Tx_Class[txClass][idx]` (§8.3.3): the neighbour offsets
/// whose magnitudes drive the `coeff_br` context.
const MAG_REF_OFFSET: [[(i32, i32); 3]; 3] = [
    [(0, 1), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (0, 2)],
    [(0, 1), (1, 0), (2, 0)],
];

/// `Coeff_Base_Pos_Ctx_Offset[Min(idx,2)]` (§8.3.3) for the 1D transform classes.
const COEFF_BASE_POS_CTX_OFFSET: [i32; 3] = [
    SIG_COEF_CONTEXTS_2D,
    SIG_COEF_CONTEXTS_2D + 5,
    SIG_COEF_CONTEXTS_2D + 10,
];

// `Coeff_Base_Ctx_Offset[txSz][Min(row,4)][Min(col,4)]` (§8.3.3). The 19 sizes
// share six distinct patterns, named here to make the shape auditable against
// the spec: `SQR` for the square sizes, `TALL`/`WIDE` for the 2:1 and 4:1
// rectangles, and the small 4x4 / 4x8 / 8x4 variants that zero their last
// row or column.
const CBO_4X4: [[i32; 5]; 5] = [
    [0, 1, 6, 6, 0],
    [1, 6, 6, 21, 0],
    [6, 6, 21, 21, 0],
    [6, 21, 21, 21, 0],
    [0, 0, 0, 0, 0],
];
const CBO_SQR: [[i32; 5]; 5] = [
    [0, 1, 6, 6, 21],
    [1, 6, 6, 21, 21],
    [6, 6, 21, 21, 21],
    [6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21],
];
const CBO_NARROW: [[i32; 5]; 5] = [
    [0, 11, 11, 11, 0],
    [11, 11, 11, 11, 0],
    [6, 6, 21, 21, 0],
    [6, 21, 21, 21, 0],
    [21, 21, 21, 21, 0],
];
const CBO_SHORT: [[i32; 5]; 5] = [
    [0, 16, 6, 6, 21],
    [16, 16, 6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [0, 0, 0, 0, 0],
];
const CBO_TALL: [[i32; 5]; 5] = [
    [0, 11, 11, 11, 11],
    [11, 11, 11, 11, 11],
    [6, 6, 21, 21, 21],
    [6, 21, 21, 21, 21],
    [21, 21, 21, 21, 21],
];
const CBO_WIDE: [[i32; 5]; 5] = [
    [0, 16, 6, 6, 21],
    [16, 16, 6, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
    [16, 16, 21, 21, 21],
];
const COEFF_BASE_CTX_OFFSET: [[[i32; 5]; 5]; 19] = [
    CBO_4X4, CBO_SQR, CBO_SQR, CBO_SQR, CBO_SQR, CBO_NARROW, CBO_SHORT, CBO_TALL, CBO_WIDE,
    CBO_TALL, CBO_WIDE, CBO_TALL, CBO_WIDE, CBO_NARROW, CBO_SHORT, CBO_TALL, CBO_WIDE, CBO_TALL,
    CBO_WIDE,
];

/// The mutable coefficient CDFs for one tile, cloned from the defaults for the
/// frame's quantiser context. The spec's `Tile*Cdf` are the frame defaults
/// pre-indexed by the quantiser context (`get_qctx`), then adapted per symbol as
/// the tile decodes.
pub struct CoeffCdfs {
    txb_skip: [[[u16; 3]; 13]; 5],
    eob_pt_16: [[[u16; 6]; 2]; 2],
    eob_pt_32: [[[u16; 7]; 2]; 2],
    eob_pt_64: [[[u16; 8]; 2]; 2],
    eob_pt_128: [[[u16; 9]; 2]; 2],
    eob_pt_256: [[[u16; 10]; 2]; 2],
    eob_pt_512: [[u16; 11]; 2],
    eob_pt_1024: [[u16; 12]; 2],
    eob_extra: [[[[u16; 3]; 9]; 2]; 5],
    coeff_base_eob: [[[[u16; 4]; 4]; 2]; 5],
    coeff_base: [[[[u16; 5]; 42]; 2]; 5],
    coeff_br: [[[[u16; 5]; 21]; 2]; 5],
    dc_sign: [[[u16; 3]; 3]; 2],
}

impl CoeffCdfs {
    /// Clone the defaults for quantiser context `qctx` (0 for lossless).
    #[must_use]
    pub fn new(qctx: usize) -> Self {
        let q = qctx.min(3);
        Self {
            txb_skip: pick4(cdf::DEFAULT_TXB_SKIP_CDF, q),
            eob_pt_16: pick4(cdf::DEFAULT_EOB_PT_16_CDF, q),
            eob_pt_32: pick4(cdf::DEFAULT_EOB_PT_32_CDF, q),
            eob_pt_64: pick4(cdf::DEFAULT_EOB_PT_64_CDF, q),
            eob_pt_128: pick4(cdf::DEFAULT_EOB_PT_128_CDF, q),
            eob_pt_256: pick4(cdf::DEFAULT_EOB_PT_256_CDF, q),
            eob_pt_512: pick4(cdf::DEFAULT_EOB_PT_512_CDF, q),
            eob_pt_1024: pick4(cdf::DEFAULT_EOB_PT_1024_CDF, q),
            eob_extra: pick4(cdf::DEFAULT_EOB_EXTRA_CDF, q),
            coeff_base_eob: pick4(cdf::DEFAULT_COEFF_BASE_EOB_CDF, q),
            coeff_base: pick4(cdf::DEFAULT_COEFF_BASE_CDF, q),
            coeff_br: pick4(cdf::DEFAULT_COEFF_BR_CDF, q),
            dc_sign: pick4(cdf::DEFAULT_DC_SIGN_CDF, q),
        }
    }
}

/// The result of decoding one transform block's coefficients.
pub struct CoeffBlock {
    /// `Quant[]` in the coded block's raster order: signed dequantiser input
    /// levels. Only the first `Tx_Width * Tx_Height` (of the adjusted size) are
    /// meaningful; the tail stays zero.
    pub quant: [i32; MAX_COEFFS],
    /// The end-of-block position: the count of leading scan coefficients.
    pub eob: usize,
    /// `culLevel`, clamped to 63: the neighbour level context this block leaves.
    pub cul_level: u8,
    /// `dcCategory`: 0 none, 1 negative DC, 2 positive DC.
    pub dc_category: u8,
}

/// The scan order (`get_scan`, §5.11.41) for a transform size and type.
fn get_scan(tx_size: TxSize, tx_type: TxType) -> &'static [u16] {
    match tx_size {
        TxSize::Tx16x64 => return &DEFAULT_SCAN_16X32,
        TxSize::Tx64x16 => return &DEFAULT_SCAN_32X16,
        _ => {}
    }
    if tx_size.sqr_up_idx() == 4 {
        // Tx_Size_Sqr_Up == TX_64X64: coded as a 32x32 scan.
        return &DEFAULT_SCAN_32X32;
    }
    match tx_type {
        TxType::Idtx => default_scan(tx_size),
        TxType::VDct | TxType::VAdst | TxType::VFlipadst => mrow_scan(tx_size),
        TxType::HDct | TxType::HAdst | TxType::HFlipadst => mcol_scan(tx_size),
        _ => default_scan(tx_size),
    }
}

/// `get_default_scan(txSz)` (§5.11.41).
fn default_scan(tx_size: TxSize) -> &'static [u16] {
    match tx_size {
        TxSize::Tx4x4 => &DEFAULT_SCAN_4X4,
        TxSize::Tx4x8 => &DEFAULT_SCAN_4X8,
        TxSize::Tx8x4 => &DEFAULT_SCAN_8X4,
        TxSize::Tx8x8 => &DEFAULT_SCAN_8X8,
        TxSize::Tx8x16 => &DEFAULT_SCAN_8X16,
        TxSize::Tx16x8 => &DEFAULT_SCAN_16X8,
        TxSize::Tx16x16 => &DEFAULT_SCAN_16X16,
        TxSize::Tx16x32 => &DEFAULT_SCAN_16X32,
        TxSize::Tx32x16 => &DEFAULT_SCAN_32X16,
        TxSize::Tx4x16 => &DEFAULT_SCAN_4X16,
        TxSize::Tx16x4 => &DEFAULT_SCAN_16X4,
        TxSize::Tx8x32 => &DEFAULT_SCAN_8X32,
        TxSize::Tx32x8 => &DEFAULT_SCAN_32X8,
        _ => &DEFAULT_SCAN_32X32,
    }
}

/// `get_mrow_scan(txSz)` (§5.11.41), used by the `V_*` (row-identity) types.
fn mrow_scan(tx_size: TxSize) -> &'static [u16] {
    match tx_size {
        TxSize::Tx4x4 => &MROW_SCAN_4X4,
        TxSize::Tx4x8 => &MROW_SCAN_4X8,
        TxSize::Tx8x4 => &MROW_SCAN_8X4,
        TxSize::Tx8x8 => &MROW_SCAN_8X8,
        TxSize::Tx8x16 => &MROW_SCAN_8X16,
        TxSize::Tx16x8 => &MROW_SCAN_16X8,
        TxSize::Tx16x16 => &MROW_SCAN_16X16,
        TxSize::Tx4x16 => &MROW_SCAN_4X16,
        _ => &MROW_SCAN_16X4,
    }
}

/// `get_mcol_scan(txSz)` (§5.11.41), used by the `H_*` (column-identity) types.
fn mcol_scan(tx_size: TxSize) -> &'static [u16] {
    match tx_size {
        TxSize::Tx4x4 => &MCOL_SCAN_4X4,
        TxSize::Tx4x8 => &MCOL_SCAN_4X8,
        TxSize::Tx8x4 => &MCOL_SCAN_8X4,
        TxSize::Tx8x8 => &MCOL_SCAN_8X8,
        TxSize::Tx8x16 => &MCOL_SCAN_8X16,
        TxSize::Tx16x8 => &MCOL_SCAN_16X8,
        TxSize::Tx16x16 => &MCOL_SCAN_16X16,
        TxSize::Tx4x16 => &MCOL_SCAN_4X16,
        _ => &MCOL_SCAN_16X4,
    }
}

/// Decode the coefficients of one transform block (`coeffs`, §5.11.39).
///
/// `ptype` is 0 for luma and 1 for chroma. `all_zero_ctx` and `dc_sign_ctx` are
/// the neighbour-derived contexts the tile driver computes. `tx_type` is the
/// already-resolved plane transform type (`DCT_DCT` in the lossless path). The
/// returned `Quant[]` feeds the dequantiser and inverse transform.
///
/// # Errors
///
/// Propagates any error from the arithmetic decoder (a stream that ends early)
/// or a corrupt context index.
pub fn decode_coeffs(
    dec: &mut SymbolDecoder<'_>,
    cdfs: &mut CoeffCdfs,
    tx_size: TxSize,
    tx_type: TxType,
    ptype: usize,
    all_zero_ctx: usize,
    dc_sign_ctx: usize,
) -> Result<CoeffBlock> {
    let mut quant = [0_i32; MAX_COEFFS];
    let pt = ptype.min(1);
    let tx_ctx = tx_size.tx_size_ctx();
    let cls = tx_class(tx_type);

    // all_zero (txb_skip): the whole block codes as zero.
    let skip_cdf = cdf_row(cdf_row(&mut cdfs.txb_skip, tx_ctx)?, all_zero_ctx)?;
    let all_zero = dec.read_symbol(skip_cdf)? != 0;
    if all_zero {
        return Ok(CoeffBlock {
            quant,
            eob: 0,
            cul_level: 0,
            dc_category: 0,
        });
    }

    let scan = get_scan(tx_size, tx_type);
    let eob_ctx = usize::from(cls != 0);

    // eob_pt: the end-of-block bucket, in a size-dependent alphabet.
    let eob_pt = match tx_size.eob_multisize() {
        0 => dec.read_symbol(cdf_row(cdf_row(&mut cdfs.eob_pt_16, pt)?, eob_ctx)?)?,
        1 => dec.read_symbol(cdf_row(cdf_row(&mut cdfs.eob_pt_32, pt)?, eob_ctx)?)?,
        2 => dec.read_symbol(cdf_row(cdf_row(&mut cdfs.eob_pt_64, pt)?, eob_ctx)?)?,
        3 => dec.read_symbol(cdf_row(cdf_row(&mut cdfs.eob_pt_128, pt)?, eob_ctx)?)?,
        4 => dec.read_symbol(cdf_row(cdf_row(&mut cdfs.eob_pt_256, pt)?, eob_ctx)?)?,
        5 => dec.read_symbol(cdf_row(&mut cdfs.eob_pt_512, pt)?)?,
        _ => dec.read_symbol(cdf_row(&mut cdfs.eob_pt_1024, pt)?)?,
    } + 1;

    let mut eob = if eob_pt < 2 {
        eob_pt
    } else {
        (1 << (eob_pt - 2)) + 1
    };

    // eob_extra plus the raw extra bits refine eob within its bucket.
    if let Some(eob_shift) = eob_pt.checked_sub(3) {
        let extra_cdf = cdf_row(
            cdf_row(cdf_row(&mut cdfs.eob_extra, tx_ctx)?, pt)?,
            eob_pt - 3,
        )?;
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

    eob = eob.min(scan.len());

    // Base levels, walking the scan backwards from the last coefficient.
    for c in (0..eob).rev() {
        let pos = scan.get(c).map_or(0, |&p| usize::from(p));
        let mut level;
        if c == eob - 1 {
            let ctx = coeff_base_ctx(tx_size, cls, &quant, pos, c, true) + SIG_COEF_CONTEXTS_EOB
                - SIG_COEF_CONTEXTS;
            let cdf_ref = cdf_row(
                cdf_row(cdf_row(&mut cdfs.coeff_base_eob, tx_ctx)?, pt)?,
                ctx,
            )?;
            level = dec.read_symbol(cdf_ref)? as i32 + 1;
        } else {
            let ctx = coeff_base_ctx(tx_size, cls, &quant, pos, c, false);
            let cdf_ref = cdf_row(cdf_row(cdf_row(&mut cdfs.coeff_base, tx_ctx)?, pt)?, ctx)?;
            level = dec.read_symbol(cdf_ref)? as i32;
        }
        if level > NUM_BASE_LEVELS {
            let br_ctx = coeff_br_ctx(tx_size, cls, &quant, pos);
            let br_bucket = tx_ctx.min(3);
            for _ in 0..(COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) {
                let cdf_ref = cdf_row(
                    cdf_row(cdf_row(&mut cdfs.coeff_br, br_bucket)?, pt)?,
                    br_ctx,
                )?;
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
        let pos = scan.get(c).map_or(0, |&p| usize::from(p));
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

/// `get_coeff_base_ctx` (§8.3.3). `is_eob` selects the four end-of-block
/// contexts; otherwise the magnitude of already-decoded scan neighbours and the
/// coefficient position pick the context.
fn coeff_base_ctx(
    tx_size: TxSize,
    cls: usize,
    quant: &[i32; MAX_COEFFS],
    pos: usize,
    c: usize,
    is_eob: bool,
) -> usize {
    let bwl = tx_size.adjusted_log2_width();
    let width = tx_size.adjusted_width() as i32;
    let height = tx_size.adjusted_height() as i32;
    if is_eob {
        let area = (height as usize) << bwl;
        return if c == 0 {
            SIG_COEF_CONTEXTS - 4
        } else if c <= area / 8 {
            SIG_COEF_CONTEXTS - 3
        } else if c <= area / 4 {
            SIG_COEF_CONTEXTS - 2
        } else {
            SIG_COEF_CONTEXTS - 1
        };
    }
    let row = (pos >> bwl) as i32;
    let col = (pos - ((row as usize) << bwl)) as i32;
    let mut mag = 0;
    for &(d_row, d_col) in offsets_sig(cls) {
        let ref_row = row + d_row;
        let ref_col = col + d_col;
        if ref_row >= 0 && ref_col >= 0 && ref_row < height && ref_col < width {
            let ref_pos = ((ref_row as usize) << bwl) + ref_col as usize;
            mag += quant.get(ref_pos).copied().unwrap_or(0).abs().min(3);
        }
    }
    let ctx = ((mag + 1) >> 1).min(4);
    if cls == 0 {
        if row == 0 && col == 0 {
            return 0;
        }
        let offset = COEFF_BASE_CTX_OFFSET
            .get(tx_size as usize)
            .and_then(|t| t.get(row.min(4) as usize))
            .and_then(|r| r.get(col.min(4) as usize))
            .copied()
            .unwrap_or(0);
        return (ctx + offset) as usize;
    }
    let idx = if cls == 2 { row } else { col };
    let offset = pick3(COEFF_BASE_POS_CTX_OFFSET, idx.min(2) as usize);
    (ctx + offset) as usize
}

/// `coeff_br` context (§8.3.3).
fn coeff_br_ctx(tx_size: TxSize, cls: usize, quant: &[i32; MAX_COEFFS], pos: usize) -> usize {
    let bwl = tx_size.adjusted_log2_width();
    let txw = tx_size.adjusted_width();
    let txh = tx_size.adjusted_height() as i32;
    let row = (pos >> bwl) as i32;
    let col = (pos - ((row as usize) << bwl)) as i32;
    let mut mag = 0;
    for &(d_row, d_col) in offsets_mag(cls) {
        let ref_row = row + d_row;
        let ref_col = col + d_col;
        if ref_row >= 0 && ref_col >= 0 && ref_row < txh && ref_col < (1 << bwl) {
            let ref_pos = ref_row as usize * txw + ref_col as usize;
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
    } else if cls == 0 {
        if row < 2 && col < 2 {
            mag + 7
        } else {
            mag + 14
        }
    } else if cls == 1 {
        if col == 0 { mag + 7 } else { mag + 14 }
    } else if row == 0 {
        mag + 7
    } else {
        mag + 14
    };
    ctx as usize
}

/// `Sig_Ref_Diff_Offset[cls]` without indexing.
fn offsets_sig(cls: usize) -> &'static [(i32, i32); 5] {
    let [two_d, horiz, vert] = &SIG_REF_DIFF_OFFSET;
    match cls {
        1 => horiz,
        2 => vert,
        _ => two_d,
    }
}

/// `Mag_Ref_Offset_With_Tx_Class[cls]` without indexing.
fn offsets_mag(cls: usize) -> &'static [(i32, i32); 3] {
    let [two_d, horiz, vert] = &MAG_REF_OFFSET;
    match cls {
        1 => horiz,
        2 => vert,
        _ => two_d,
    }
}

/// Pick element `q` (0..=2) of a three-entry table by value, without indexing.
fn pick3<T: Copy>(arr: [T; 3], q: usize) -> T {
    let [a, b, c] = arr;
    match q {
        1 => b,
        2 => c,
        _ => a,
    }
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
        let q = [0; MAX_COEFFS];
        assert_eq!(
            coeff_base_ctx(TxSize::Tx4x4, 0, &q, 0, 0, true),
            SIG_COEF_CONTEXTS - 4
        );
        assert_eq!(
            coeff_base_ctx(TxSize::Tx4x4, 0, &q, 5, 1, true),
            SIG_COEF_CONTEXTS - 3
        );
        assert_eq!(
            coeff_base_ctx(TxSize::Tx4x4, 0, &q, 5, 3, true),
            SIG_COEF_CONTEXTS - 2
        );
        assert_eq!(
            coeff_base_ctx(TxSize::Tx4x4, 0, &q, 5, 9, true),
            SIG_COEF_CONTEXTS - 1
        );
    }

    #[test]
    fn dc_position_base_context_is_zero() {
        let q = [3; MAX_COEFFS];
        assert_eq!(coeff_base_ctx(TxSize::Tx4x4, 0, &q, 0, 4, false), 0);
    }

    #[test]
    fn base_context_folds_in_neighbour_magnitudes() {
        // pos 5 -> row 1, col 1 (bwl 2). Put a 3 at pos 6 only: mag=3,
        // ctx=(3+1)>>1=2, plus Coeff_Base_Ctx_Offset[TX_4X4][1][1]=6 -> 8.
        let mut q = [0_i32; MAX_COEFFS];
        q[6] = 3;
        assert_eq!(coeff_base_ctx(TxSize::Tx4x4, 0, &q, 5, 4, false), 8);
    }

    #[test]
    fn br_context_at_dc_is_the_bare_magnitude() {
        // pos 0: neighbours (0,1)=pos1, (1,0)=pos4, (1,1)=pos5. A single 5 at
        // pos1 -> mag=min(5,15)=5, (5+1)>>1=3, pos==0 so ctx=3.
        let mut q = [0_i32; MAX_COEFFS];
        q[1] = 5;
        assert_eq!(coeff_br_ctx(TxSize::Tx4x4, 0, &q, 0), 3);
    }

    #[test]
    fn vertical_class_uses_position_offsets() {
        // A 1D vertical transform (class 2) uses Coeff_Base_Pos_Ctx_Offset, not
        // the 2D table: at pos 0 with no neighbours the ctx is the base offset.
        let q = [0_i32; MAX_COEFFS];
        // pos 8 in an 8x8 (bwl 3) -> row 1, col 0. No neighbours set, mag 0,
        // ctx 0, idx=row=1 -> Coeff_Base_Pos_Ctx_Offset[1] = 31.
        assert_eq!(
            coeff_base_ctx(TxSize::Tx8x8, 2, &q, 8, 4, false),
            (SIG_COEF_CONTEXTS_2D + 5) as usize
        );
    }

    #[test]
    fn scans_are_selected_by_size_and_type() {
        assert_eq!(get_scan(TxSize::Tx4x4, TxType::DctDct).len(), 16);
        assert_eq!(get_scan(TxSize::Tx8x8, TxType::DctDct).len(), 64);
        // 64-wide sizes fall back to the 32x32 scan.
        assert_eq!(get_scan(TxSize::Tx64x64, TxType::DctDct).len(), 1024);
        assert_eq!(get_scan(TxSize::Tx16x64, TxType::DctDct).len(), 512);
        // V_/H_ types switch to the row/column scans (compared by content:
        // scan tables are `const`, so they have no stable address).
        assert_eq!(get_scan(TxSize::Tx8x8, TxType::VDct), &MROW_SCAN_8X8[..]);
        assert_eq!(get_scan(TxSize::Tx8x8, TxType::HDct), &MCOL_SCAN_8X8[..]);
        // The default (2D) scan differs from the row scan for the same size.
        assert_ne!(
            get_scan(TxSize::Tx8x8, TxType::DctDct),
            get_scan(TxSize::Tx8x8, TxType::VDct)
        );
    }

    #[test]
    fn an_all_zero_block_reads_one_symbol_and_stops() {
        let data = [0x00; 8];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = CoeffCdfs::new(0);
        let block =
            decode_coeffs(&mut dec, &mut cdfs, TxSize::Tx4x4, TxType::DctDct, 0, 0, 0).unwrap();
        if block.eob == 0 {
            assert_eq!(block.quant, [0; MAX_COEFFS]);
            assert_eq!(block.cul_level, 0);
        }
    }

    #[test]
    fn golomb_reads_a_unary_prefix_then_data_bits() {
        let data = [0xFF; 4];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        assert_eq!(read_golomb(&mut dec).unwrap(), 1);
    }
}
