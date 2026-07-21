//! GIF codec for `otf-pixels`, implemented from scratch.
//!
//! Decode covers the whole format: all frames, both interlace layouts,
//! transparency, and every disposal method. Encode is single-frame with
//! palette quantization, which is SPEC §Formats' stated v1 scope — animation
//! *pipelines* are v2.
//!
//! # Frames and the engine
//!
//! [`Decoder`] yields one image, so [`GifDecoder`] presents the first frame
//! composited onto the canvas. Every existing pipeline therefore works on a
//! GIF unchanged, and the engine stays frame-unaware.
//!
//! Callers who want the animation use [`GifDecoder::next_frame`], which walks
//! the remaining frames and applies disposal between them. That keeps
//! "animation pipelines are v2" honest: the decode is complete, and only the
//! graph integration is deferred.
//!
//! [`Decoder`]: otf_pixels_core::Decoder

mod decoder;
mod encoder;
mod format;
mod quantize;

pub use decoder::{Frame, GifCodec, GifDecoder, probe};
pub use encoder::GifEncoder;
pub use format::{Disposal, GraphicControl, ImageDescriptor, Screen};
pub use quantize::{Dither, Palette, build_palette, quantize};

/// Translate a compression failure into this crate's error type.
///
/// `otf-pixels-compress` deliberately does not depend on `otf-pixels-core`
/// (ADR-0012), so the mapping lives here, where the format context is known.
pub(crate) fn compress_error(error: otf_pixels_compress::Error) -> otf_pixels_core::PixelsError {
    otf_pixels_core::PixelsError::malformed("gif", error.detail().to_owned())
}
