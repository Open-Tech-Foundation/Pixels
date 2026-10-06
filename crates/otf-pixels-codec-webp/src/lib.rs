//! WebP codec for `otf-pixels`, lossy and lossless, owned outright
//! (ADR-0014).
//!
//! # Layers
//!
//! The RIFF container (`riff`); VP8L lossless decode ([`vp8l`]) and encode
//! (`vp8l_encode`); VP8 lossy decode and encode (`vp8`); `ALPH` alpha
//! (`alpha`); and libwebp's YUV/RGB conversions (`yuv`). Decoding matches
//! libwebp exactly, and libwebp checks every file we write.
//!
//! # Memory
//!
//! Internally buffered in both directions, as SPEC §Formats says. The decoder
//! holds the whole file, whose chunks may come in any order the container
//! allows; both encoders make decisions over the whole picture before the
//! first byte is final.

mod alpha;
mod decoder;
mod encoder;
mod riff;
mod vp8;
pub mod vp8l;
mod vp8l_encode;
mod yuv;

pub use decoder::{WebPCodec, WebPDecoder, probe};
pub use encoder::WebPEncoder;

/// The bytes a WebP file begins with: `RIFF`, four length bytes, then `WEBP`.
pub const SIGNATURE_RIFF: [u8; 4] = *b"RIFF";
/// The form type at offset 8.
pub const SIGNATURE_WEBP: [u8; 4] = *b"WEBP";
