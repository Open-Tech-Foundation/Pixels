//! Compression and checksum primitives for `otf-pixels` codecs.
//!
//! Per ADR-0010 these are written from scratch, and per ADR-0012 they live
//! here rather than inside one codec, because three formats now share them:
//! PNG needs zlib, TIFF needs zlib *and* LZW, GIF needs LZW.
//!
//! # Scope
//!
//! This crate knows about bit streams and byte buffers. It knows nothing about
//! images, pixels or descriptors, and deliberately does not depend on
//! `otf-pixels-core`. That boundary is what lets it be tested directly against
//! reference implementations — the validation ADR-0010 requires — rather than
//! only through a codec.
//!
//! The consequence is a local [`Error`] type. Codecs translate it into their
//! own error at the boundary, which is one small helper per codec and keeps
//! the stable [`ErrorCode`] mapping where it belongs.
//!
//! [`ErrorCode`]: https://docs.rs/otf-pixels-core
//!
//! # Safety
//!
//! Every parser here reads attacker-controlled bytes and returns errors rather
//! than panicking. `unsafe_code = "forbid"` means the classic decompressor
//! failure — an out-of-bounds write through a back-reference — is
//! unrepresentable rather than merely avoided.

mod checksum;
mod deflate;
mod inflate;
mod lzw;

pub use checksum::{Adler32, Crc32};
pub use deflate::{Level, deflate, zlib_compress};
pub use inflate::{Inflater, ZlibStream, inflate_to, zlib_decompress};
pub use lzw::{BitOrder, LzwDecoder, LzwEncoder};

use core::fmt;

/// A compression or checksum failure.
///
/// Always caused by malformed input or by a caller-declared bound being
/// exceeded — never by an internal invariant, which is why there is no
/// "internal error" variant to handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Which format the bytes claimed to be, for the caller's message.
    format: &'static str,
    detail: String,
}

impl Error {
    /// Report bytes that are invalid for `format`.
    #[must_use]
    pub fn malformed(format: &'static str, detail: impl Into<String>) -> Self {
        Self {
            format,
            detail: detail.into(),
        }
    }

    /// The format tag, suitable for a codec's own error type.
    #[must_use]
    pub const fn format(&self) -> &'static str {
        self.format
    }

    /// What went wrong.
    #[must_use]
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed {} data: {}", self.format, self.detail)
    }
}

impl std::error::Error for Error {}

/// The result of a compression operation.
pub type Result<T> = std::result::Result<T, Error>;
