//! WebP codec for `otf-pixels`, wrapping [`image-webp`].
//!
//! [`image-webp`]: https://docs.rs/image-webp
//!
//! # Why this one is wrapped
//!
//! ADR-0004 ranks WebP as hard to own: the container holds *two* unrelated
//! codecs — VP8 intra frames for lossy, and a separate dictionary-and-
//! transform format for lossless — so owning WebP means writing two codecs,
//! not one. The trait boundary means a later rewrite is a drop-in.
//!
//! # Memory
//!
//! Internally buffered in both directions, as SPEC §Formats says. The decoder
//! needs to seek within the RIFF container, and the lossless encoder builds a
//! dictionary over the whole image, so neither end can work a row at a time.
//!
//! # Encoding is lossless only
//!
//! The wrapped encoder writes **lossless** WebP and has no quality control, so
//! [`EncodeOptions::quality`] is ignored here. For a photograph that means a
//! considerably larger file than a lossy WebP encoder would produce — the
//! format's headline feature is exactly the one not available. This is a
//! property of the wrapped crate, not a decision, and it is the strongest
//! argument for revisiting WebP ownership.
//!
//! [`EncodeOptions::quality`]: otf_pixels_core::EncodeOptions::quality

mod decoder;
mod encoder;

pub use decoder::{WebPCodec, WebPDecoder, probe};
pub use encoder::WebPEncoder;

/// The bytes a WebP file begins with: `RIFF`, four length bytes, then `WEBP`.
pub const SIGNATURE_RIFF: [u8; 4] = *b"RIFF";
/// The form type at offset 8.
pub const SIGNATURE_WEBP: [u8; 4] = *b"WEBP";
