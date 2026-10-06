//! The WebP encoder.
//!
//! Lossy by default, through the owned VP8 encoder at
//! [`EncodeOptions::quality`]; [`EncodeOptions::lossless`] selects lossless,
//! which still goes through `image-webp` until the owned VP8L encoder lands
//! (ADR-0014). Either way the whole image is gathered first: both bitstreams
//! make decisions over the whole picture before the first byte is final.

use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

/// Encodes a WebP stream.
#[derive(Debug)]
pub struct WebPEncoder {
    /// Set by `write_header`; its presence means the header was written.
    state: Option<State>,
    options: EncodeOptions,
}

impl Default for WebPEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Everything fixed once the descriptor is known.
#[derive(Debug)]
struct State {
    descriptor: ImageDescriptor,
    colour: image_webp::ColorType,
    /// The whole image, accumulated: the lossless encoder builds a dictionary
    /// over all of it and cannot emit a row at a time.
    pixels: Vec<u8>,
    rows_written: u32,
}

impl WebPEncoder {
    /// An encoder with default settings: lossy at the default quality.
    #[must_use]
    pub fn new() -> Self {
        Self::from_options(&EncodeOptions::default())
    }

    /// An encoder configured from generic encode options: lossy at
    /// `quality`, or lossless when `lossless` is set.
    #[must_use]
    pub const fn from_options(options: &EncodeOptions) -> Self {
        Self {
            state: None,
            options: *options,
        }
    }
}

/// The wrapped encoder's colour type for a pixel format.
fn colour_of(format: PixelFormat) -> Result<image_webp::ColorType> {
    match format {
        PixelFormat::Gray8 => Ok(image_webp::ColorType::L8),
        PixelFormat::GrayA8 => Ok(image_webp::ColorType::La8),
        PixelFormat::Rgb8 => Ok(image_webp::ColorType::Rgb8),
        PixelFormat::Rgba8 => Ok(image_webp::ColorType::Rgba8),
        other => Err(PixelsError::unsupported(format!(
            "WebP encoding needs an 8-bit format; got {other}. Convert first."
        ))),
    }
}

impl Encoder for WebPEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        let colour = colour_of(desc.pixel)?;
        // WebP dimensions are 14-bit in the lossless bitstream; a larger image
        // cannot be represented at all, so this is a format limit.
        const MAX: u32 = 16_383;
        if desc.width > MAX || desc.height > MAX {
            return Err(PixelsError::unsupported(format!(
                "WebP dimensions are at most {MAX}; {}x{} does not fit",
                desc.width, desc.height
            )));
        }
        let capacity = desc
            .byte_len()
            .ok_or_else(|| PixelsError::malformed("webp", "image size overflows"))?;

        // Nothing is written yet: the container length is not known until the
        // compressed body exists.
        self.state = Some(State {
            descriptor: *desc,
            colour,
            pixels: Vec::with_capacity(capacity),
            rows_written: 0,
        });
        Ok(())
    }

    fn write_row(&mut self, row: &[u8], _sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "row",
                "write_row called before write_header",
            ));
        };
        let expected = state.descriptor.row_bytes();
        if row.len() != expected {
            return Err(PixelsError::invalid_argument(
                "row",
                format!("row is {} bytes, expected {expected}", row.len()),
            ));
        }
        if state.rows_written >= state.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "row",
                format!("more than {} rows written", state.descriptor.height),
            ));
        }
        state.pixels.extend_from_slice(row);
        state.rows_written += 1;
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some(state) = self.state.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "sink",
                "finish called before write_header",
            ));
        };
        if state.rows_written < state.descriptor.height {
            return Err(PixelsError::malformed(
                "webp",
                format!(
                    "{} of {} rows were written",
                    state.rows_written, state.descriptor.height
                ),
            ));
        }

        let bytes = if self.options.lossless {
            let mut bytes = Vec::new();
            image_webp::WebPEncoder::new(&mut bytes)
                .encode(
                    &state.pixels,
                    state.descriptor.width,
                    state.descriptor.height,
                    state.colour,
                )
                .map_err(encode_error)?;
            bytes
        } else {
            encode_lossy(state, self.options.quality)?
        };
        sink.write_all(&bytes)?;
        sink.flush()
    }
}

/// A RIFF chunk: FourCC, little-endian size, payload, pad to even.
fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(kind);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        out.push(0);
    }
}

/// Wrap chunks in the `RIFF`/`WEBP` header.
fn riff(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 12);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    out.extend_from_slice(body);
    out
}

/// Lossy WebP: a `VP8 ` frame, and when the image has any transparency, an
/// extended file with an `ALPH` chunk ahead of it.
fn encode_lossy(state: &State, quality: u8) -> Result<Vec<u8>> {
    let (width, height) = (
        state.descriptor.width as usize,
        state.descriptor.height as usize,
    );
    let channels = state.descriptor.pixel.channels();
    let (planes, alpha) = crate::yuv::from_rgb(&state.pixels, channels, width, height);
    let vp8 = crate::vp8::encode::encode(
        &planes,
        crate::vp8::encode::Params::with_quality(quality.clamp(1, 100)),
    )?;
    // An opaque alpha channel is dropped: a simple file says the same thing
    // in fewer bytes and every reader handles it.
    let alpha = alpha.filter(|a| a.iter().any(|&v| v != 255));
    let mut body = Vec::new();
    match alpha {
        None => chunk(&mut body, b"VP8 ", &vp8),
        Some(alpha) => {
            let mut vp8x = vec![0x10, 0, 0, 0];
            vp8x.extend_from_slice(&(width as u32 - 1).to_le_bytes()[..3]);
            vp8x.extend_from_slice(&(height as u32 - 1).to_le_bytes()[..3]);
            chunk(&mut body, b"VP8X", &vp8x);
            // Raw alpha, unfiltered; the owned lossless encoder will compress
            // it once it lands.
            let mut alph = Vec::with_capacity(alpha.len() + 1);
            alph.push(0);
            alph.extend_from_slice(&alpha);
            chunk(&mut body, b"ALPH", &alph);
            chunk(&mut body, b"VP8 ", &vp8);
        }
    }
    Ok(riff(&body))
}

/// Translate the wrapped encoder's failure into this crate's error type.
fn encode_error(error: image_webp::EncodingError) -> PixelsError {
    match error {
        image_webp::EncodingError::IoError(error) => {
            PixelsError::io("encoding a WebP image", error)
        }
        other => PixelsError::invalid_argument("image", other.to_string()),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::ErrorCode;

    #[test]
    fn unsupported_pixel_formats_are_refused_at_the_header() {
        for format in [
            PixelFormat::Gray16,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::RgbF32,
        ] {
            let descriptor = ImageDescriptor::new(4, 4, format).unwrap();
            let mut sink = Vec::new();
            let error = WebPEncoder::new()
                .write_header(&descriptor, &mut sink)
                .unwrap_err();
            assert_eq!(error.code(), ErrorCode::Unsupported, "{format}");
            assert!(sink.is_empty(), "{format}: bytes were written anyway");
        }
    }

    #[test]
    fn the_encoder_contract_is_enforced() {
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::Rgb8).unwrap();
        let row = vec![0_u8; descriptor.row_bytes()];

        let mut encoder = WebPEncoder::new();
        let mut sink = Vec::new();
        assert_eq!(
            encoder.write_row(&row, &mut sink).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            encoder.finish(&mut sink).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );

        encoder.write_header(&descriptor, &mut sink).unwrap();
        assert_eq!(
            encoder
                .write_header(&descriptor, &mut sink)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );

        // Finishing early must not emit a truncated image that looks whole.
        encoder.write_row(&row, &mut sink).unwrap();
        assert_eq!(
            encoder.finish(&mut sink).unwrap_err().code(),
            ErrorCode::Malformed
        );
        assert!(sink.is_empty());
    }

    #[test]
    fn oversized_images_are_refused() {
        let descriptor = ImageDescriptor::new(20_000, 4, PixelFormat::Rgb8).unwrap();
        let error = WebPEncoder::new()
            .write_header(&descriptor, &mut Vec::new())
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::Unsupported, "{error}");
    }
}
