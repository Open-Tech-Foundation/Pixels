//! TIFF codec for `otf-pixels`, implemented from scratch.
//!
//! Baseline TIFF 6.0: both byte orders, strip and tile layouts, none/LZW/
//! Deflate/PackBits compression, greyscale/RGB/palette photometrics at 1, 8
//! and 16 bits. Exotic tags are skipped rather than treated as errors, which
//! SPEC §Formats requires and TIFF's extensibility makes essential.
//!
//! # Why TIFF is the streaming showcase
//!
//! A *tiled* TIFF stores its pixels as an array of independently compressed
//! rectangles, each with its own offset. That makes it the one v1 format
//! capable of genuine random access: producing an arbitrary region means
//! decompressing the tiles it touches and nothing else.
//!
//! So [`TiffDecoder`] reports [`DecodeCapability::Regions`] for a tiled file,
//! and the scheduler pulls regions rather than rows. Turning a 2 GB tiled
//! scan into a thumbnail then costs the tiles the thumbnail actually needs —
//! which is what ADR-0001's demand-driven design was for, and what the rest of
//! v1 has been building toward.
//!
//! [`DecodeCapability::Regions`]: otf_pixels_core::DecodeCapability::Regions

mod decoder;
mod ifd;
mod image;

pub use decoder::{TiffCodec, TiffDecoder, probe};
pub use ifd::{ByteOrder, Directory, tag};
pub use image::{Compression, Layout, Photometric, TiffImage};

/// Translate a compression failure into this crate's error type.
///
/// `otf-pixels-compress` deliberately does not depend on `otf-pixels-core`
/// (ADR-0012), so the mapping lives here, where the format context is known.
pub(crate) fn compress_error(error: otf_pixels_compress::Error) -> otf_pixels_core::PixelsError {
    otf_pixels_core::PixelsError::malformed("tiff", error.detail().to_owned())
}
