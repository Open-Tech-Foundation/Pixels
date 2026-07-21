//! The WebP encoder, wrapping `image-webp`.
//!
//! Lossless only — see the crate docs. [`EncodeOptions::quality`] is accepted
//! and ignored, which is stated rather than hidden: silently producing a
//! lossless file for a caller who asked for quality 60 is surprising, and the
//! surprise belongs in the documentation rather than in the file size.

use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

/// Encodes a lossless WebP stream.
#[derive(Debug, Default)]
pub struct WebPEncoder {
    /// Set by `write_header`; its presence means the header was written.
    state: Option<State>,
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
    /// An encoder with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self { state: None }
    }

    /// An encoder configured from generic encode options.
    ///
    /// `options` carries only a quality, which lossless WebP has no notion of,
    /// so nothing here reads it.
    #[must_use]
    pub const fn from_options(_options: &EncodeOptions) -> Self {
        Self::new()
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

        let mut bytes = Vec::new();
        image_webp::WebPEncoder::new(&mut bytes)
            .encode(
                &state.pixels,
                state.descriptor.width,
                state.descriptor.height,
                state.colour,
            )
            .map_err(encode_error)?;
        sink.write_all(&bytes)?;
        sink.flush()
    }
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
