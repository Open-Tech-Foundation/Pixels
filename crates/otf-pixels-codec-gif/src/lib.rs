//! GIF codec for `otf-pixels`, implemented from scratch.
//!
//! Decode covers the whole format: all frames, both interlace layouts,
//! transparency, and every disposal method. Encode quantizes to a palette,
//! a still or an animation whose frames hold only what changed.
//!
//! # Frames and the engine
//!
//! [`GifDecoder`]'s rows are the current frame composited onto the canvas:
//! the first, until [`Decoder::advance_frame`] moves to the next with
//! disposal applied. That is the frame protocol the pipeline replays one
//! graph through (ADR-0020); a caller that opens without asking for the
//! animation reads the first frame and never advances.
//!
//! [`GifDecoder::next_frame`] walks the frames as whole-canvas copies, for
//! callers outside the pipeline.
//!
//! [`Decoder::advance_frame`]: otf_pixels_core::Decoder::advance_frame
//! [`Decoder`]: otf_pixels_core::Decoder

mod animate;
mod decoder;
mod encoder;
mod format;

pub use decoder::{Frame, GifCodec, GifDecoder, probe};
pub use encoder::GifEncoder;
pub use format::{Disposal, GraphicControl, ImageDescriptor, Screen};
pub use otf_pixels_compress::{Dither, Palette, build_palette, quantize};

/// Translate a compression failure into this crate's error type.
///
/// `otf-pixels-compress` deliberately does not depend on `otf-pixels-core`
/// (ADR-0012), so the mapping lives here, where the format context is known.
pub(crate) fn compress_error(error: otf_pixels_compress::Error) -> otf_pixels_core::PixelsError {
    otf_pixels_core::PixelsError::malformed("gif", error.detail().to_owned())
}
