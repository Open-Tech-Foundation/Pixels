//! The AV1 bitstream decoder, restricted to the still-picture subset.
//!
//! An AVIF still image is an AV1 **key frame**. That single restriction removes
//! inter prediction, reference-frame management, motion vectors, compound and
//! warped and global motion — well over half of AV1's decoder surface — and is
//! what makes owning the codec finite (ADR-0013).
//!
//! The layering mirrors the spec: [`bits`] is the bit reader, [`obu`] the Open
//! Bitstream Unit framing, [`seq`] the sequence header, and [`frame`] the
//! frame (uncompressed) header down to the tile-group boundary. [`still`] ties
//! them together the way an AVIF still is laid out — configuration OBUs plus a
//! coded frame — and hands back the two parsed headers and a locator for the
//! tile data. Reconstruction (the symbol decoder, coefficient parse,
//! prediction, transforms and filters) lands in the phases after this one.

pub mod cdf;

mod bits;
mod coeff;
mod direction;
mod frame;
mod obu;
mod palette;
mod plane;
mod predict;
mod seq;
mod still;
mod symbol;
mod tile;
mod transform;
mod transform_type;

pub use bits::{BitReader, floor_log2};
pub use coeff::{CoeffBlock, CoeffCdfs, decode_coeffs};
pub use frame::{
    Cdef, FilmGrain, FrameHeader, LoopFilter, LoopRestoration, Quantization, Segmentation,
    TileInfo, TxMode,
};
pub use obu::{Obu, ObuHeader, ObuType};
pub use plane::Plane;
pub use predict::{IntraMode, Neighbours, predict_intra_4x4};
pub use seq::{ColorConfig, OperatingPoint, SequenceHeader};
pub use still::{StillPicture, sequence_header_from_config};
pub use symbol::SymbolDecoder;
pub use tile::{DecodedFrame, decode_still};
pub use transform::{
    Dequant, Residual, TxSize, TxType, ac_q, add_residual_4x4, dc_q, dequantize,
    inverse_transform_2d,
};
pub use transform_type::{
    IntraTxSet, IntraTxTypeCdfs, chroma_tx_type, intra_dir, intra_tx_set, is_tx_type_in_set_intra,
    mode_to_txfm, read_transform_type,
};

#[cfg(test)]
#[allow(
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod cdf_tests {
    use super::cdf;

    #[test]
    fn generated_tables_match_the_spec_values() {
        // A one-dimensional table, verbatim from the spec.
        assert_eq!(
            cdf::DEFAULT_CFL_SIGN_CDF,
            [1418, 2123, 13340, 18405, 26972, 28343, 32294, 32768, 0]
        );
        assert_eq!(cdf::DEFAULT_DELTA_Q_CDF, [28160, 32120, 32677, 32768, 0]);
        // A three-dimensional table, first context pair.
        assert_eq!(
            cdf::DEFAULT_INTRA_FRAME_Y_MODE_CDF[0][0],
            [
                15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466,
                32768, 0
            ]
        );
    }

    #[test]
    fn every_cdf_ends_with_the_terminator_and_counter() {
        // The last real cumulative frequency is 1<<15 and the trailing counter
        // is 0, which is exactly what SymbolDecoder::read_symbol relies on.
        fn check(row: &[u16]) {
            let n = row.len();
            assert!(n >= 2, "a CDF row is too short: {row:?}");
            assert_eq!(row[n - 1], 0, "counter not zero in {row:?}");
            assert_eq!(row[n - 2], 1 << 15, "terminator not 32768 in {row:?}");
        }
        for row in &cdf::DEFAULT_ANGLE_DELTA_CDF {
            check(row);
        }
        for ctx in &cdf::DEFAULT_INTRA_FRAME_Y_MODE_CDF {
            for row in ctx {
                check(row);
            }
        }
    }
}
