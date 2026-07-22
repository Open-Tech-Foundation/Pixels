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

mod bits;
mod frame;
mod obu;
mod seq;
mod still;
mod symbol;

pub use bits::{floor_log2, BitReader};
pub use frame::{
    Cdef, FilmGrain, FrameHeader, LoopFilter, LoopRestoration, Quantization, Segmentation, TileInfo,
    TxMode,
};
pub use obu::{Obu, ObuHeader, ObuType};
pub use seq::{ColorConfig, OperatingPoint, SequenceHeader};
pub use still::{sequence_header_from_config, StillPicture};
pub use symbol::SymbolDecoder;
