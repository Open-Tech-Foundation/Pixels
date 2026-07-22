//! AVIF codec for `otf-pixels`, implemented from scratch.
//!
//! AVIF is two specifications stacked: an ISOBMFF/HEIF container (ISO/IEC
//! 23008-12) holding one or more items, and an AV1 bitstream (AOM AV1) coding
//! the pixels. This crate owns both — see ADR-0013, which reverses ADR-0004's
//! decision to wrap the dav1d/rav1e family.
//!
//! # Scope
//!
//! Still images only: an AVIF still is an AV1 **key frame**, which removes
//! inter prediction, reference frame management, motion vectors and compound
//! modes — well over half of AV1's decoder surface. AVIF image *sequences*
//! (the `avis` brand, carrying `moov`/`trak`) are animation and therefore v2
//! per ROADMAP §v2; a sequence decodes its primary item and nothing else,
//! matching what GIF and WebP already do.
//!
//! # Memory
//!
//! Internally buffered, as SPEC §Formats says. The container addresses its
//! payload by absolute file offset through `iloc`, so the bytes must be
//! resident before any of them can be interpreted — there is no prefix of an
//! AVIF that yields a finished row. The external contract stays streaming
//! (ADR-0005): the codec buffers, the caller does not.
//!
//! # Safety
//!
//! Every parser here reads attacker-controlled bytes. Malformed input is a
//! value, never a panic: `unsafe_code = "forbid"` and the workspace ban on
//! `unwrap`/`expect`/`panic!` mean the classic container failures — a box
//! declaring more bytes than its parent holds, an item extent pointing outside
//! the file, a `grid` whose tiles do not tile — are rejected rather than
//! trusted.

mod av1;
mod boxes;
mod decoder;
mod meta;
mod props;

pub use av1::cdf;
pub use av1::{
    BitReader, Cdef, ColorConfig, FilmGrain, FrameHeader, IntraMode, LoopFilter, LoopRestoration,
    Neighbours, Obu, ObuHeader, ObuType, OperatingPoint, Plane, Quantization, Segmentation,
    SequenceHeader, StillPicture, SymbolDecoder, TileInfo, TxMode, add_residual_4x4, floor_log2,
    inverse_wht_4x4, predict_intra_4x4, sequence_header_from_config,
};
pub use boxes::{BoxHeader, FourCc, Reader};
pub use decoder::{AvifCodec, AvifDecoder, AvifInfo, probe};
pub use meta::{Construction, Extent, Item, Meta, Reference, URN_ALPHA, URN_ALPHA_LEGACY};
pub use props::{
    Association, Av1Config, Colour, Extents, PixelInfo, Properties, Property, Subsampling,
};

/// The box type that identifies a file's brands, at offset 4 of every ISOBMFF
/// file.
pub const SIGNATURE_FTYP: [u8; 4] = *b"ftyp";

/// Brands that mean "this file holds an AVIF still image".
///
/// A file is claimed if any of these appears as the major brand or among the
/// compatible brands. `avif` is the still-image brand proper; `mif1` is the
/// HEIF image-file brand that many encoders write as the major brand with
/// `avif` only in the compatible list; `miaf` is the MIAF profile brand; and
/// `MA1A`/`MA1B` are the MIAF AVIF baseline and advanced profiles.
pub const BRANDS_STILL: [[u8; 4]; 5] = [*b"avif", *b"mif1", *b"miaf", *b"MA1A", *b"MA1B"];

/// The brand for an AVIF image sequence.
///
/// Recognised so that sniffing claims the file and the decoder can report what
/// it found, rather than leaving an animation to be mis-sniffed as something
/// else entirely.
pub const BRAND_SEQUENCE: [u8; 4] = *b"avis";
