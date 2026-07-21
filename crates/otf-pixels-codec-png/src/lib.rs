//! PNG codec for `otf-pixels`, implemented from scratch.
//!
//! Includes its own DEFLATE implementation per ADR-0010: PNG *is* mostly
//! DEFLATE, so depending on a compression crate would reduce "PNG from
//! scratch" to "PNG container parsing from scratch".
//!
//! Every parser here reads attacker-controlled bytes and returns errors rather
//! than panicking. `unsafe_code = "forbid"` means the classic decompressor
//! failure — an out-of-bounds write through a back-reference — is
//! unrepresentable rather than merely avoided.

mod decoder;
mod encoder;
mod format;

pub use decoder::{PngCodec, PngDecoder, probe};
pub use encoder::PngEncoder;
pub use format::{ColorType, Filter, Header, SIGNATURE};

// Re-exported rather than defined here: ADR-0012 moved the compression
// primitives to `otf-pixels-compress` once TIFF and GIF became consumers of
// them. They stay in this crate's public surface so nothing downstream breaks.
pub use otf_pixels_compress::{
    Adler32, Crc32, Inflater, Level, ZlibStream, deflate, inflate_to, zlib_compress,
    zlib_decompress,
};

/// Translate a compression failure into this crate's error type.
///
/// `otf-pixels-compress` deliberately does not depend on `otf-pixels-core`
/// (ADR-0012), so the mapping to a stable [`ErrorCode`] lives here, where the
/// format context is known.
///
/// [`ErrorCode`]: otf_pixels_core::ErrorCode
pub(crate) fn compress_error(error: otf_pixels_compress::Error) -> otf_pixels_core::PixelsError {
    otf_pixels_core::PixelsError::malformed(error.format(), error.detail().to_owned())
}
