//! The WebP decoder, wrapping `image-webp`.

use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Format, ImageDescriptor, Limits, PixelFormat, PixelsError,
    Result, Source,
};

/// The most compressed bytes read before a file is called hostile.
///
/// The wrapped decoder needs to seek within the container, so the stream is
/// held in memory and there is no bound from the image dimensions: a small
/// header can be followed by unlimited chunk data. `max_pixels` bounds the
/// output; this bounds the input.
const MAX_COMPRESSED: usize = 256 * 1024 * 1024;

/// Decodes a WebP stream.
#[derive(Debug)]
pub struct WebPDecoder {
    descriptor: ImageDescriptor,
    /// The decoded image, interleaved.
    pixels: Vec<u8>,
    /// Rows already served.
    row: u32,
}

impl WebPDecoder {
    /// Read the container and decode the image.
    ///
    /// Unlike the streaming codecs this decodes eagerly, because the wrapped
    /// decoder seeks within the RIFF container and a WebP has no prefix that
    /// yields a finished row.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a stream the wrapped decoder
    /// rejects, [`PixelsError::Unsupported`] for a WebP feature it does not
    /// implement, or [`PixelsError::LimitExceeded`] if the image exceeds
    /// `limits`.
    pub fn new<S: Source>(mut source: S, limits: Limits) -> Result<Self> {
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 64 * 1024];
        loop {
            if bytes.len() > MAX_COMPRESSED {
                return Err(PixelsError::malformed(
                    "webp",
                    format!("stream exceeds {MAX_COMPRESSED} bytes"),
                ));
            }
            match source.read(&mut chunk)? {
                0 => break,
                read => {
                    let Some(filled) = chunk.get(..read) else {
                        break;
                    };
                    bytes.extend_from_slice(filled);
                }
            }
        }

        let mut decoder =
            image_webp::WebPDecoder::new(std::io::Cursor::new(bytes)).map_err(decode_error)?;
        let (width, height) = decoder.dimensions();
        // An animation decodes to its first frame, matching what GIF does:
        // ordinary pipelines then work unchanged, and animation pipelines are
        // v2 (SPEC §Formats).
        let pixel = if decoder.has_alpha() {
            PixelFormat::Rgba8
        } else {
            PixelFormat::Rgb8
        };
        // Enforced before the pixel buffer is allocated (SPEC §Safety).
        let descriptor = ImageDescriptor::with_limits(width, height, pixel, &limits)?;

        let wanted = descriptor
            .byte_len()
            .ok_or_else(|| PixelsError::malformed("webp", "image size overflows"))?;
        let reported = decoder
            .output_buffer_size()
            .ok_or_else(|| PixelsError::malformed("webp", "image size overflows"))?;
        if reported != wanted {
            return Err(PixelsError::malformed(
                "webp",
                format!("decoder wants {reported} bytes for a {descriptor} image needing {wanted}"),
            ));
        }

        let mut pixels = vec![0_u8; wanted];
        decoder.read_image(&mut pixels).map_err(decode_error)?;

        Ok(Self {
            descriptor,
            pixels,
            row: 0,
        })
    }
}

/// Translate the wrapped decoder's failure into this crate's error type.
///
/// The split matters: a caller routes on [`PixelsError::Unsupported`] versus
/// [`PixelsError::Malformed`], and collapsing both into "broken image" would
/// send someone hunting for a corrupt file that merely uses a feature this
/// build does not decode.
fn decode_error(error: image_webp::DecodingError) -> PixelsError {
    match error {
        image_webp::DecodingError::IoError(error) => {
            PixelsError::io("decoding a WebP image", error)
        }
        image_webp::DecodingError::UnsupportedFeature(detail) => {
            PixelsError::unsupported(format!("webp: {detail}"))
        }
        other => PixelsError::malformed("webp", other.to_string()),
    }
}

impl Decoder for WebPDecoder {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        // The image is already in memory, but `Sequential` is what the row
        // contract describes; claiming `Regions` would promise a
        // `read_region` this does not implement.
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        if self.row >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("all {} rows have already been read", self.descriptor.height),
            ));
        }
        let row_bytes = self.descriptor.row_bytes();
        if out.len() != row_bytes {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {row_bytes}", out.len()),
            ));
        }
        let start = self.row as usize * row_bytes;
        let row = self
            .pixels
            .get(start..)
            .and_then(|rest| rest.get(..row_bytes))
            .ok_or_else(|| PixelsError::malformed("webp", "decoded image is short"))?;
        out.copy_from_slice(row);
        self.row += 1;
        Ok(())
    }
}

/// Whether `prefix` starts with a WebP signature.
///
/// Detection is by magic bytes only (SPEC §Formats). `RIFF` alone names a
/// container family that also holds WAV and AVI, so the form type at offset 8
/// is what actually identifies a WebP.
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    prefix.get(..4) == Some(&crate::SIGNATURE_RIFF[..])
        && prefix.get(8..12) == Some(&crate::SIGNATURE_WEBP[..])
}

/// The WebP entry in a sniffing registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct WebPCodec;

impl Codec for WebPCodec {
    fn format(&self) -> Format {
        Format::WebP
    }

    fn magic_len(&self) -> usize {
        12
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
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

    #[test]
    fn probe_needs_the_form_type_not_just_riff() {
        let mut header = Vec::from(*b"RIFF");
        header.extend_from_slice(&[0, 0, 0, 0]);
        header.extend_from_slice(b"WEBP");
        assert!(probe(&header));

        // A WAV file is also RIFF, and must not be claimed.
        let mut wav = Vec::from(*b"RIFF");
        wav.extend_from_slice(&[0, 0, 0, 0]);
        wav.extend_from_slice(b"WAVE");
        assert!(!probe(&wav));

        // Short prefixes are declined, never indexed past.
        assert!(!probe(b"RIFF"));
        assert!(!probe(b""));
        assert!(!probe(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn a_stream_that_is_not_a_webp_is_rejected() {
        let error = WebPDecoder::new(&b"not a webp at all"[..], Limits::default()).unwrap_err();
        assert_eq!(
            error.code(),
            otf_pixels_core::ErrorCode::Malformed,
            "{error}"
        );
    }
}
