//! WebP codec for `otf-pixels`, being moved from [`image-webp`] to an owned
//! implementation (ADR-0014).
//!
//! [`image-webp`]: https://docs.rs/image-webp
//!
//! # Ownership, layer by layer
//!
//! The RIFF container (`riff`), the VP8L lossless decoder ([`vp8l`]), the
//! VP8 lossy decoder (`vp8`), `ALPH` alpha (`alpha`) and libwebp's
//! YUV-to-RGB conversion (`yuv`) are owned, and every still image decodes
//! through them, matching libwebp exactly. Animations still decode through
//! `image-webp`, and encoding still uses its lossless encoder, until the
//! owned first-frame compositing and encoders land; the dependency goes
//! when the last of them does.
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

mod alpha;
mod decoder;
mod encoder;
mod riff;
mod vp8;
pub mod vp8l;
mod yuv;

pub use decoder::{WebPCodec, WebPDecoder, probe};
pub use encoder::WebPEncoder;

/// The bytes a WebP file begins with: `RIFF`, four length bytes, then `WEBP`.
pub const SIGNATURE_RIFF: [u8; 4] = *b"RIFF";
/// The form type at offset 8.
pub const SIGNATURE_WEBP: [u8; 4] = *b"WEBP";
