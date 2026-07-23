//! Transform-type decode (spec §5.11.47–§5.11.48, "transform_type" /
//! "get_tx_set", and the "compute transform type" function §5.11.40).
//!
//! Every coded transform block has a `PlaneTxType` — the pair of 1D transforms
//! applied to its rows and columns. For an intra (key) frame the luma type is
//! read from the bitstream as `intra_tx_type`, a symbol whose alphabet (the
//! *transform set*) depends on the transform size, then mapped through an
//! inversion table; chroma derives its type from the prediction mode instead.
//! The size also gates whether any symbol is read at all: large transforms and
//! the lossless path are always `DCT_DCT`.
//!
//! This module owns the pure set/mapping logic and the `intra_tx_type` symbol
//! read. The `TxTypes` grid that carries a luma block's type to its co-located
//! chroma (via `compute_tx_type`) lives with the tile decoder that will drive
//! reconstruction; the helpers here take the resolved inputs directly.

use super::cdf;
use super::symbol::SymbolDecoder;
use super::transform::{TxSize, TxType};
use otf_pixels_core::{PixelsError, Result};

/// The intra transform set (`get_tx_set` on an intra frame, §5.11.48). The
/// inter sets never arise in the still-picture subset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntraTxSet {
    /// `TX_SET_DCTONLY`: only `DCT_DCT`; no `intra_tx_type` symbol is coded.
    DctOnly,
    /// `TX_SET_INTRA_1`: seven types, coded with `Tx_Type_Intra_Inv_Set1`.
    Set1,
    /// `TX_SET_INTRA_2`: five types, coded with `Tx_Type_Intra_Inv_Set2`.
    Set2,
}

/// `Tx_Type_Intra_Inv_Set1` (§5.11.47): the `intra_tx_type` symbol to `TxType`
/// map for `TX_SET_INTRA_1`.
const TX_TYPE_INTRA_INV_SET1: [TxType; 7] = [
    TxType::Idtx,
    TxType::DctDct,
    TxType::VDct,
    TxType::HDct,
    TxType::AdstAdst,
    TxType::AdstDct,
    TxType::DctAdst,
];

/// `Tx_Type_Intra_Inv_Set2` (§5.11.47): the map for `TX_SET_INTRA_2`.
const TX_TYPE_INTRA_INV_SET2: [TxType; 5] = [
    TxType::Idtx,
    TxType::DctDct,
    TxType::AdstAdst,
    TxType::AdstDct,
    TxType::DctAdst,
];

/// `Mode_To_Txfm[UVMode]` (§5.11.40): the chroma transform type implied by the
/// prediction mode. Indexed by the intra mode `DC_PRED..UV_CFL_PRED`.
const MODE_TO_TXFM: [TxType; 14] = [
    TxType::DctDct,   // DC_PRED
    TxType::AdstDct,  // V_PRED
    TxType::DctAdst,  // H_PRED
    TxType::DctDct,   // D45_PRED
    TxType::AdstAdst, // D135_PRED
    TxType::AdstDct,  // D113_PRED
    TxType::DctAdst,  // D157_PRED
    TxType::DctAdst,  // D203_PRED
    TxType::AdstDct,  // D67_PRED
    TxType::AdstAdst, // SMOOTH_PRED
    TxType::AdstDct,  // SMOOTH_V_PRED
    TxType::DctAdst,  // SMOOTH_H_PRED
    TxType::AdstAdst, // PAETH_PRED
    TxType::DctDct,   // UV_CFL_PRED
];

/// `Filter_Intra_Mode_To_Intra_Dir` (§5.11.47): the intra direction a filter
/// intra mode maps to for the `intra_tx_type` context (`DC, V, H, D157, DC`).
const FILTER_INTRA_MODE_TO_INTRA_DIR: [usize; 5] = [0, 1, 2, 6, 0];

/// `Tx_Type_In_Set_Intra[set][txType]` (§5.11.40): whether a chroma transform
/// type is permitted by the set, else it falls back to `DCT_DCT`. Rows are
/// `TX_SET_DCTONLY`, `TX_SET_INTRA_1`, `TX_SET_INTRA_2` in that order.
const TX_TYPE_IN_SET_INTRA: [[bool; 16]; 3] = [
    bool_row([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    bool_row([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]),
    bool_row([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
];

/// Turn a spec `{0,1}` row into a `[bool; 16]` at compile time.
#[allow(
    clippy::indexing_slicing,
    reason = "both arrays are fixed [_; 16] and the loop bound is 16"
)]
const fn bool_row(src: [u8; 16]) -> [bool; 16] {
    let mut out = [false; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = src[i] != 0;
        i += 1;
    }
    out
}

/// `get_tx_set(txSz)` for an intra frame (§5.11.48). `reduced_tx_set` is the
/// frame-header flag.
#[must_use]
pub fn intra_tx_set(tx_size: TxSize, reduced_tx_set: bool) -> IntraTxSet {
    // Tx_Size_Sqr_Up >= TX_32X32 (index 3) is always DCT-only for intra.
    if tx_size.sqr_up_idx() >= 3 {
        return IntraTxSet::DctOnly;
    }
    if reduced_tx_set || tx_size.sqr_idx() == 2 {
        // reduced set, or Tx_Size_Sqr == TX_16X16.
        return IntraTxSet::Set2;
    }
    IntraTxSet::Set1
}

/// The intra-frame index into the tx type membership / CDF tables, mirroring the
/// spec's `TX_CLASS`-free numbering (`DctOnly = 0`, `Set1 = 1`, `Set2 = 2`).
fn set_index(set: IntraTxSet) -> usize {
    match set {
        IntraTxSet::DctOnly => 0,
        IntraTxSet::Set1 => 1,
        IntraTxSet::Set2 => 2,
    }
}

/// The `TxType`'s index in `PlaneTxType` order (`§6.10.28`), for set membership.
fn tx_type_index(tx_type: TxType) -> usize {
    tx_type as usize
}

/// The intra direction (`intraDir`) for the `intra_tx_type` context: the luma
/// mode, or the filter-intra mode's mapped direction when filter intra is used.
#[must_use]
pub fn intra_dir(y_mode: usize, filter_intra: Option<usize>) -> usize {
    match filter_intra {
        Some(mode) => FILTER_INTRA_MODE_TO_INTRA_DIR
            .get(mode)
            .copied()
            .unwrap_or(0),
        None => y_mode,
    }
}

/// The adapting `intra_tx_type` CDFs for one tile, cloned from the defaults.
pub struct IntraTxTypeCdfs {
    set1: [[[u16; 8]; 13]; 2],
    set2: [[[u16; 6]; 13]; 3],
}

impl IntraTxTypeCdfs {
    /// Clone the frame defaults (these CDFs do not depend on the quantiser).
    #[must_use]
    pub fn new() -> Self {
        Self {
            set1: cdf::DEFAULT_INTRA_TX_TYPE_SET1_CDF,
            set2: cdf::DEFAULT_INTRA_TX_TYPE_SET2_CDF,
        }
    }
}

impl Default for IntraTxTypeCdfs {
    fn default() -> Self {
        Self::new()
    }
}

fn row_mut<T>(slice: &mut [T], index: usize) -> Result<&mut T> {
    slice
        .get_mut(index)
        .ok_or_else(|| PixelsError::malformed("avif", "an AV1 tx-type CDF index ran out of range"))
}

/// Resolve a luma block's transform type (`transform_type`, §5.11.47).
///
/// Reads the `intra_tx_type` symbol and maps it to a `TxType`, unless the set is
/// `DCT_DCT`-only or the quantiser index is zero (lossless / `set == 0` guards),
/// in which case `DCT_DCT` is returned without consuming a symbol. `intra_dir`
/// is from [`intra_dir`].
///
/// # Errors
///
/// Propagates arithmetic-decoder errors and rejects an out-of-range context.
pub fn read_transform_type(
    dec: &mut SymbolDecoder<'_>,
    cdfs: &mut IntraTxTypeCdfs,
    set: IntraTxSet,
    tx_size: TxSize,
    intra_dir: usize,
    qindex_positive: bool,
) -> Result<TxType> {
    if set == IntraTxSet::DctOnly || !qindex_positive {
        return Ok(TxType::DctDct);
    }
    let sqr = tx_size.sqr_idx() as usize;
    let symbol = match set {
        IntraTxSet::Set1 => {
            let cdf_ref = row_mut(row_mut(&mut cdfs.set1, sqr)?, intra_dir)?;
            dec.read_symbol(cdf_ref)?
        }
        IntraTxSet::Set2 => {
            let cdf_ref = row_mut(row_mut(&mut cdfs.set2, sqr)?, intra_dir)?;
            dec.read_symbol(cdf_ref)?
        }
        IntraTxSet::DctOnly => return Ok(TxType::DctDct),
    };
    let inv: &[TxType] = match set {
        IntraTxSet::Set1 => &TX_TYPE_INTRA_INV_SET1,
        _ => &TX_TYPE_INTRA_INV_SET2,
    };
    Ok(inv.get(symbol).copied().unwrap_or(TxType::DctDct))
}

/// `Mode_To_Txfm[uvMode]` (§5.11.40).
#[must_use]
pub fn mode_to_txfm(uv_mode: usize) -> TxType {
    MODE_TO_TXFM.get(uv_mode).copied().unwrap_or(TxType::DctDct)
}

/// `is_tx_type_in_set` for an intra frame (§5.11.40).
#[must_use]
pub fn is_tx_type_in_set_intra(set: IntraTxSet, tx_type: TxType) -> bool {
    TX_TYPE_IN_SET_INTRA
        .get(set_index(set))
        .and_then(|row| row.get(tx_type_index(tx_type)))
        .copied()
        .unwrap_or(false)
}

/// The chroma transform type (`compute_tx_type` for a plane > 0 intra block,
/// §5.11.40): the mode-implied type if the set permits it, else `DCT_DCT`.
#[must_use]
pub fn chroma_tx_type(uv_mode: usize, set: IntraTxSet) -> TxType {
    if set == IntraTxSet::DctOnly {
        return TxType::DctDct;
    }
    let tx_type = mode_to_txfm(uv_mode);
    if is_tx_type_in_set_intra(set, tx_type) {
        tx_type
    } else {
        TxType::DctDct
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
    fn tx_set_follows_the_size_and_reduced_flag() {
        // Small sizes use the full intra set 1...
        assert_eq!(intra_tx_set(TxSize::Tx4x4, false), IntraTxSet::Set1);
        assert_eq!(intra_tx_set(TxSize::Tx8x8, false), IntraTxSet::Set1);
        // ...16x16 and rectangles with a 16 square drop to set 2...
        assert_eq!(intra_tx_set(TxSize::Tx16x16, false), IntraTxSet::Set2);
        // ...the reduced flag forces set 2...
        assert_eq!(intra_tx_set(TxSize::Tx4x4, true), IntraTxSet::Set2);
        // ...and 32x32 and up are DCT-only.
        assert_eq!(intra_tx_set(TxSize::Tx32x32, false), IntraTxSet::DctOnly);
        assert_eq!(intra_tx_set(TxSize::Tx64x64, false), IntraTxSet::DctOnly);
        assert_eq!(intra_tx_set(TxSize::Tx16x32, false), IntraTxSet::DctOnly);
    }

    #[test]
    fn dct_only_and_zero_qindex_read_no_symbol() {
        // A DctOnly set returns DCT_DCT without touching the decoder.
        let data = [0xFF; 4];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = IntraTxTypeCdfs::new();
        let tt = read_transform_type(
            &mut dec,
            &mut cdfs,
            IntraTxSet::DctOnly,
            TxSize::Tx4x4,
            0,
            true,
        )
        .unwrap();
        assert_eq!(tt, TxType::DctDct);
        // qindex 0 (lossless) also short-circuits even for a real set.
        let tt = read_transform_type(
            &mut dec,
            &mut cdfs,
            IntraTxSet::Set1,
            TxSize::Tx4x4,
            0,
            false,
        )
        .unwrap();
        assert_eq!(tt, TxType::DctDct);
    }

    #[test]
    fn a_read_type_comes_from_the_inversion_table() {
        // With a real set and qindex, a symbol is read and mapped; assert only
        // that the result is one of the set's types (the exact symbol depends on
        // the arithmetic decoder state, which the table maps deterministically).
        let data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = IntraTxTypeCdfs::new();
        let tt = read_transform_type(
            &mut dec,
            &mut cdfs,
            IntraTxSet::Set1,
            TxSize::Tx4x4,
            0,
            true,
        )
        .unwrap();
        assert!(TX_TYPE_INTRA_INV_SET1.contains(&tt));
    }

    #[test]
    fn intra_dir_uses_the_filter_mapping() {
        assert_eq!(intra_dir(9, None), 9); // plain luma mode
        assert_eq!(intra_dir(9, Some(3)), 6); // filter mode 3 -> D157_PRED
        assert_eq!(intra_dir(9, Some(0)), 0); // filter mode 0 -> DC_PRED
    }

    #[test]
    fn chroma_type_respects_set_membership() {
        // V_PRED (mode 1) -> ADST_DCT, which both intra sets permit (index 1 is
        // set in both membership rows), so it survives in each.
        assert_eq!(chroma_tx_type(1, IntraTxSet::Set1), TxType::AdstDct);
        assert_eq!(chroma_tx_type(1, IntraTxSet::Set2), TxType::AdstDct);
        // DC_PRED (mode 0) -> DCT_DCT, always allowed.
        assert_eq!(chroma_tx_type(0, IntraTxSet::Set1), TxType::DctDct);
        // A DctOnly set is always DCT_DCT.
        assert_eq!(chroma_tx_type(1, IntraTxSet::DctOnly), TxType::DctDct);
    }

    #[test]
    fn set_membership_matches_the_spec_rows() {
        // ADST_ADST is in both intra sets; V_DCT only in set 1; FLIPADST in none.
        assert!(is_tx_type_in_set_intra(IntraTxSet::Set1, TxType::AdstAdst));
        assert!(is_tx_type_in_set_intra(IntraTxSet::Set2, TxType::AdstAdst));
        assert!(is_tx_type_in_set_intra(IntraTxSet::Set1, TxType::VDct));
        assert!(!is_tx_type_in_set_intra(IntraTxSet::Set2, TxType::VDct));
        assert!(!is_tx_type_in_set_intra(
            IntraTxSet::Set1,
            TxType::FlipadstFlipadst
        ));
    }
}
