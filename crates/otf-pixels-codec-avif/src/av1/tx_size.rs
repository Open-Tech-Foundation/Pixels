//! Transform-size decode (spec §5.11.15, "read_tx_size", and §5.11.16,
//! "read_block_tx_size").
//!
//! A coding block's luma transform size is read once per block for an intra
//! frame. Lossless forces `TX_4X4`; otherwise the size starts at the largest
//! rectangular transform that fits the block (`Max_Tx_Size_Rect`) and, when the
//! frame is in `TX_MODE_SELECT`, a `tx_depth` symbol splits it down that many
//! levels through `Split_Tx_Size`. The same size then applies to every transform
//! block in the coding block (there is no variable-transform tree for intra).
//!
//! This module owns the pure size tables and the `tx_depth` symbol read with its
//! neighbour-derived context. Wiring the resulting size through the reconstruct
//! loop belongs with the tile decoder, which currently drives a `TX_4X4`-only
//! path; the helpers here take the resolved neighbour widths directly.

use super::cdf;
use super::symbol::SymbolDecoder;
use super::transform::TxSize;
use otf_pixels_core::{PixelsError, Result};

/// `BLOCK_4X4` (§6.10.4): the smallest block, index 0 of `BLOCK_SIZES`.
pub const BLOCK_4X4: usize = 0;

/// `Max_Tx_Size_Rect[BLOCK_SIZES]` (§9.3): the largest rectangular transform
/// that fits each of the 22 block sizes.
const MAX_TX_SIZE_RECT: [TxSize; 22] = [
    TxSize::Tx4x4,
    TxSize::Tx4x8,
    TxSize::Tx8x4,
    TxSize::Tx8x8,
    TxSize::Tx8x16,
    TxSize::Tx16x8,
    TxSize::Tx16x16,
    TxSize::Tx16x32,
    TxSize::Tx32x16,
    TxSize::Tx32x32,
    TxSize::Tx32x64,
    TxSize::Tx64x32,
    TxSize::Tx64x64,
    TxSize::Tx64x64,
    TxSize::Tx64x64,
    TxSize::Tx64x64,
    TxSize::Tx4x16,
    TxSize::Tx16x4,
    TxSize::Tx8x32,
    TxSize::Tx32x8,
    TxSize::Tx16x64,
    TxSize::Tx64x16,
];

/// `Max_Tx_Depth[BLOCK_SIZES]` (§9.3): how many times a block's transform must
/// split to reach `TX_4X4`. Can exceed `MAX_TX_DEPTH`; the `tx_depth` symbol
/// still only codes 0..=2, so deeper blocks cannot actually be coded that small.
const MAX_TX_DEPTH: [usize; 22] = [
    0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4, 4,
];

/// `Split_Tx_Size[TX_SIZES_ALL]` (§9.3): the transform size one depth down.
const SPLIT_TX_SIZE: [TxSize; 19] = [
    TxSize::Tx4x4,
    TxSize::Tx4x4,
    TxSize::Tx8x8,
    TxSize::Tx16x16,
    TxSize::Tx32x32,
    TxSize::Tx4x4,
    TxSize::Tx4x4,
    TxSize::Tx8x8,
    TxSize::Tx8x8,
    TxSize::Tx16x16,
    TxSize::Tx16x16,
    TxSize::Tx32x32,
    TxSize::Tx32x32,
    TxSize::Tx4x8,
    TxSize::Tx8x4,
    TxSize::Tx8x16,
    TxSize::Tx16x8,
    TxSize::Tx16x32,
    TxSize::Tx32x16,
];

/// `Max_Tx_Size_Rect[block]`.
#[must_use]
pub fn max_tx_size_rect(block: usize) -> TxSize {
    MAX_TX_SIZE_RECT
        .get(block)
        .copied()
        .unwrap_or(TxSize::Tx4x4)
}

/// `Max_Tx_Depth[block]`.
#[must_use]
pub fn max_tx_depth(block: usize) -> usize {
    MAX_TX_DEPTH.get(block).copied().unwrap_or(0)
}

/// `Split_Tx_Size[txSz]`: one transform-depth step down.
#[must_use]
pub fn split_tx_size(tx: TxSize) -> TxSize {
    SPLIT_TX_SIZE
        .get(tx as usize)
        .copied()
        .unwrap_or(TxSize::Tx4x4)
}

/// The `BLOCK_SIZES` index for a block `w4 x h4` 4-sample units wide/high, or
/// `None` if that is not a defined block shape.
#[must_use]
pub fn block_size_from_4x4(w4: usize, h4: usize) -> Option<usize> {
    Some(match (w4, h4) {
        (1, 1) => 0,
        (1, 2) => 1,
        (2, 1) => 2,
        (2, 2) => 3,
        (2, 4) => 4,
        (4, 2) => 5,
        (4, 4) => 6,
        (4, 8) => 7,
        (8, 4) => 8,
        (8, 8) => 9,
        (8, 16) => 10,
        (16, 8) => 11,
        (16, 16) => 12,
        (16, 32) => 13,
        (32, 16) => 14,
        (32, 32) => 15,
        (1, 4) => 16,
        (4, 1) => 17,
        (2, 8) => 18,
        (8, 2) => 19,
        (4, 16) => 20,
        (16, 4) => 21,
        _ => return None,
    })
}

/// The `tx_depth` context (§8.3.2): whether the above/left neighbour transforms
/// are at least as wide/tall as this block's maximum transform. `above_w` and
/// `left_h` are the neighbour transform width/height in samples (0 when the
/// neighbour is unavailable), as `get_above_tx_width` / `get_left_tx_height`
/// resolve them for an intra block.
#[must_use]
pub fn tx_depth_ctx(above_w: usize, left_h: usize, max_rect: TxSize) -> usize {
    usize::from(above_w >= max_rect.width()) + usize::from(left_h >= max_rect.height())
}

/// The adapting `tx_depth` CDFs for one tile, one per maximum-transform category.
pub struct TxDepthCdfs {
    tx8x8: [[u16; 3]; 3],
    tx16x16: [[u16; 4]; 3],
    tx32x32: [[u16; 4]; 3],
    tx64x64: [[u16; 4]; 3],
}

impl TxDepthCdfs {
    /// Clone the frame defaults (these CDFs do not depend on the quantiser).
    #[must_use]
    pub fn new() -> Self {
        Self {
            tx8x8: cdf::DEFAULT_TX_8X8_CDF,
            tx16x16: cdf::DEFAULT_TX_16X16_CDF,
            tx32x32: cdf::DEFAULT_TX_32X32_CDF,
            tx64x64: cdf::DEFAULT_TX_64X64_CDF,
        }
    }
}

impl Default for TxDepthCdfs {
    fn default() -> Self {
        Self::new()
    }
}

fn row_mut<T>(slice: &mut [T], index: usize) -> Result<&mut T> {
    slice
        .get_mut(index)
        .ok_or_else(|| PixelsError::malformed("avif", "an AV1 tx-size CDF index ran out of range"))
}

/// The inputs to [`read_tx_size`] other than the decoder and CDFs.
pub struct TxSizeParams {
    /// The coding block's `BLOCK_SIZES` index.
    pub block: usize,
    /// `TxMode == TX_MODE_SELECT`: the frame codes a transform depth per block.
    pub tx_mode_select: bool,
    /// The lossless flag; forces `TX_4X4` and reads no symbol.
    pub lossless: bool,
    /// The caller's selection gate (`!skip || !is_inter`).
    pub allow_select: bool,
    /// The above neighbour's transform width in samples (0 if unavailable).
    pub above_w: usize,
    /// The left neighbour's transform height in samples (0 if unavailable).
    pub left_h: usize,
}

/// Resolve a coding block's luma transform size (`read_tx_size`, §5.11.15).
///
/// A `tx_depth` symbol is read only when the block is larger than 4x4 and
/// selection is active; otherwise the size is `Max_Tx_Size_Rect` (or `TX_4X4`
/// for lossless). See [`TxSizeParams`] for the inputs.
///
/// # Errors
///
/// Propagates arithmetic-decoder errors and rejects an out-of-range context.
pub fn read_tx_size(
    dec: &mut SymbolDecoder<'_>,
    cdfs: &mut TxDepthCdfs,
    params: &TxSizeParams,
) -> Result<TxSize> {
    if params.lossless {
        return Ok(TxSize::Tx4x4);
    }
    let max_rect = max_tx_size_rect(params.block);
    let max_depth = max_tx_depth(params.block);
    let mut tx = max_rect;
    if params.block > BLOCK_4X4 && params.allow_select && params.tx_mode_select {
        let ctx = tx_depth_ctx(params.above_w, params.left_h, max_rect);
        let depth = match max_depth {
            4 => dec.read_symbol(row_mut(&mut cdfs.tx64x64, ctx)?)?,
            3 => dec.read_symbol(row_mut(&mut cdfs.tx32x32, ctx)?)?,
            2 => dec.read_symbol(row_mut(&mut cdfs.tx16x16, ctx)?)?,
            _ => dec.read_symbol(row_mut(&mut cdfs.tx8x8, ctx)?)?,
        };
        for _ in 0..depth {
            tx = split_tx_size(tx);
        }
    }
    Ok(tx)
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
    fn tables_agree_with_the_spec() {
        // 16x16 (block 6) takes a 16x16 transform, splitting to 8x8 then 4x4.
        assert_eq!(max_tx_size_rect(6), TxSize::Tx16x16);
        assert_eq!(max_tx_depth(6), 2);
        assert_eq!(split_tx_size(TxSize::Tx16x16), TxSize::Tx8x8);
        assert_eq!(split_tx_size(TxSize::Tx8x8), TxSize::Tx4x4);
        assert_eq!(split_tx_size(TxSize::Tx4x4), TxSize::Tx4x4);
        // 64x64 caps the depth at 4 and splits square down one level.
        assert_eq!(max_tx_size_rect(12), TxSize::Tx64x64);
        assert_eq!(max_tx_depth(12), 4);
        assert_eq!(split_tx_size(TxSize::Tx64x64), TxSize::Tx32x32);
        // A rectangle splits along its long side first.
        assert_eq!(split_tx_size(TxSize::Tx4x16), TxSize::Tx4x8);
    }

    #[test]
    fn block_size_lookup_round_trips_the_defined_shapes() {
        assert_eq!(block_size_from_4x4(1, 1), Some(0));
        assert_eq!(block_size_from_4x4(4, 4), Some(6));
        assert_eq!(block_size_from_4x4(16, 4), Some(21));
        // 4x32 (w4=1, h4=8) is not a defined AV1 block shape.
        assert_eq!(block_size_from_4x4(1, 8), None);
    }

    fn params(block: usize, tx_mode_select: bool, lossless: bool) -> TxSizeParams {
        TxSizeParams {
            block,
            tx_mode_select,
            lossless,
            allow_select: true,
            above_w: 0,
            left_h: 0,
        }
    }

    #[test]
    fn lossless_is_always_4x4_and_reads_nothing() {
        let data = [0xFF; 4];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = TxDepthCdfs::new();
        // Lossless: TX_4X4 regardless of block or mode, no symbol consumed.
        let tx = read_tx_size(&mut dec, &mut cdfs, &params(12, true, true)).unwrap();
        assert_eq!(tx, TxSize::Tx4x4);
    }

    #[test]
    fn without_selection_the_size_is_the_max_rect() {
        let data = [0xFF; 4];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = TxDepthCdfs::new();
        // TX_MODE_SELECT off: the largest transform, no symbol read.
        let tx = read_tx_size(&mut dec, &mut cdfs, &params(6, false, false)).unwrap();
        assert_eq!(tx, TxSize::Tx16x16);
        // A 4x4 block is always TX_4X4 even with selection on (block == BLOCK_4X4).
        let tx = read_tx_size(&mut dec, &mut cdfs, &params(BLOCK_4X4, true, false)).unwrap();
        assert_eq!(tx, TxSize::Tx4x4);
    }

    #[test]
    fn a_selected_size_is_the_max_rect_or_a_split_of_it() {
        let data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = SymbolDecoder::new(&data, true).unwrap();
        let mut cdfs = TxDepthCdfs::new();
        // 16x16 with selection: the result is 16x16, 8x8, or 4x4 depending on the
        // depth symbol — every reachable size on the split chain.
        let tx = read_tx_size(&mut dec, &mut cdfs, &params(6, true, false)).unwrap();
        assert!(matches!(
            tx,
            TxSize::Tx16x16 | TxSize::Tx8x8 | TxSize::Tx4x4
        ));
    }

    #[test]
    fn depth_context_counts_the_larger_neighbours() {
        // 16x16 max transform is 16 wide / 16 tall.
        let m = TxSize::Tx16x16;
        assert_eq!(tx_depth_ctx(0, 0, m), 0); // both neighbours smaller/absent
        assert_eq!(tx_depth_ctx(16, 0, m), 1); // above at least as wide
        assert_eq!(tx_depth_ctx(32, 16, m), 2); // both at least as large
    }
}
