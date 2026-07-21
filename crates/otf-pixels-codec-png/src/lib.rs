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

mod checksum;
mod decoder;
mod deflate;
mod encoder;
mod format;
mod inflate;

pub use checksum::{Adler32, Crc32};
pub use decoder::{PngCodec, PngDecoder, probe};
pub use deflate::{Level, deflate, zlib_compress};
pub use encoder::PngEncoder;
pub use format::{ColorType, Filter, Header, SIGNATURE};
pub use inflate::{inflate_to, zlib_decompress};
