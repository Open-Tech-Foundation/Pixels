//! The WebP decoder.
//!
//! The container is parsed here (`riff`); the image decodes through the owned
//! VP8L (`vp8l`) or VP8 (`vp8`) decoder, with `ALPH` alpha (`alpha`) and
//! libwebp's YUV-to-RGB conversion (`yuv`). An animation decodes to its first
//! frame, placed on its canvas as libwebp's animation decoder does.

use otf_pixels_core::{
    Animation, Codec, DecodeCapability, Decoder, Format, ImageDescriptor, Limits, Orientation,
    PixelFormat, PixelsError, Result, Source,
};

/// The most compressed bytes read before a file is called hostile.
///
/// The whole file is held in memory, and nothing about the image bounds how
/// much chunk data may follow a small header. `max_pixels` bounds the
/// output; this bounds the input.
const MAX_COMPRESSED: usize = 256 * 1024 * 1024;

/// Decodes a WebP stream.
#[derive(Debug)]
pub struct WebPDecoder {
    descriptor: ImageDescriptor,
    /// The whole file, until the first row is asked for.
    bytes: Vec<u8>,
    /// The decoded image, interleaved, once the first row is asked for.
    pixels: Option<Vec<u8>>,
    /// Rows already served.
    row: u32,
    /// From the `EXIF` chunk, if there is one.
    orientation: Orientation,
    /// The `ICCP` chunk, if there is one.
    icc: Option<Vec<u8>>,
    /// The animation's frames and timing, for an animated file.
    animation: Option<Animation>,
}

impl WebPDecoder {
    /// Read the file and parse its container; the pixels decode when the
    /// first row is read.
    ///
    /// The whole file is read here: chunks may come in any order the
    /// container allows, so the size, alpha, orientation and ICC profile are
    /// only known once every chunk has been seen. Decoding the bitstream is
    /// what costs, and it waits, so opening a WebP to ask its size or
    /// metadata is a header parse. The image then decodes in one piece:
    /// neither bitstream yields finished rows from a prefix of the file
    /// without the decoder holding its whole working state.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a container or bitstream
    /// header it rejects, [`PixelsError::Unsupported`] for a WebP feature it
    /// does not implement, or [`PixelsError::LimitExceeded`] if the image
    /// exceeds `limits`. A bitstream that is damaged past its header is
    /// reported by the first [`Decoder::read_row`].
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

        let container = crate::riff::parse(&bytes)?;
        // An unreadable EXIF block is metadata lost, not an image refused.
        let orientation = container
            .exif
            .and_then(Orientation::from_exif_block)
            .unwrap_or_default();
        let pixel = if container.has_alpha {
            PixelFormat::Rgba8
        } else {
            PixelFormat::Rgb8
        };
        // Enforced before any pixel buffer exists (SPEC §Safety).
        let descriptor =
            ImageDescriptor::with_limits(container.width, container.height, pixel, &limits)?;
        let icc = container.icc.map(<[u8]>::to_vec);
        let animation = Animation::new(container.frame_durations_ms.clone(), container.loop_count);
        Ok(Self {
            descriptor,
            bytes,
            pixels: None,
            row: 0,
            orientation,
            icc,
            animation,
        })
    }

    /// Decode the image onto its canvas, in the output pixel format.
    fn decode_canvas(&self) -> Result<Vec<u8>> {
        let container = crate::riff::parse(&self.bytes)?;
        let frame = container.frame;
        let rgba = decode_rgba(
            container.bitstream,
            frame.width as usize,
            frame.height as usize,
        )?;

        // The frame onto its canvas: a still fills it; an animation's first
        // frame is written into a transparent-black canvas without blending,
        // as libwebp's animation decoder starts every key frame.
        let channels = self.descriptor.pixel.channels();
        let (canvas_width, frame_width) = (container.width as usize, frame.width as usize);
        let mut pixels =
            vec![0_u8; container.width as usize * container.height as usize * channels];
        for (y, source) in rgba.chunks_exact(frame_width * 4).enumerate() {
            let row = (frame.y as usize + y) * canvas_width + frame.x as usize;
            let Some(target) = pixels.get_mut(row * channels..(row + frame_width) * channels)
            else {
                return Err(PixelsError::malformed(
                    "webp",
                    "a frame overruns its canvas",
                ));
            };
            for (out, sample) in target
                .chunks_exact_mut(channels)
                .zip(source.chunks_exact(4))
            {
                out.copy_from_slice(sample.get(..channels).unwrap_or(&[]));
            }
        }
        Ok(pixels)
    }
}

/// Decode one coded image of `width` x `height` to RGBA.
fn decode_rgba(
    bitstream: crate::riff::Bitstream<'_>,
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    match bitstream {
        crate::riff::Bitstream::Lossless(stream) => {
            let argb = crate::vp8l::decode(stream, width, height)?;
            Ok(argb
                .into_iter()
                .flat_map(|p| {
                    let [blue, green, red, alpha] = p.to_le_bytes();
                    [red, green, blue, alpha]
                })
                .collect())
        }
        crate::riff::Bitstream::Lossy { vp8, alpha } => {
            let frame = crate::vp8::decode(vp8)?;
            if (frame.width, frame.height) != (width, height) {
                return Err(PixelsError::malformed(
                    "webp",
                    "the VP8 frame's size differs from the container's",
                ));
            }
            // No ALPH chunk means opaque, as libwebp reports it.
            let alpha = match alpha {
                Some(chunk) => crate::alpha::decode(chunk, width, height)?,
                None => vec![255; width * height],
            };
            Ok(crate::yuv::to_rgb(
                &frame.y,
                frame.y_stride,
                &frame.u,
                &frame.v,
                frame.uv_stride,
                width,
                height,
                Some(&alpha),
            ))
        }
    }
}

impl Decoder for WebPDecoder {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn orientation(&self) -> Orientation {
        self.orientation
    }

    fn icc_profile(&self) -> Option<&[u8]> {
        self.icc.as_deref()
    }

    fn animation(&self) -> Option<Animation> {
        self.animation.clone()
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
        if self.pixels.is_none() {
            self.pixels = Some(self.decode_canvas()?);
            // Decoded, the compressed file is no longer needed.
            self.bytes = Vec::new();
        }
        let start = self.row as usize * row_bytes;
        let row = self
            .pixels
            .as_deref()
            .unwrap_or(&[])
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

    const LOSSY: &[u8] = include_bytes!("../tests/fixtures/lossy/alpha_blocks_q70.webp");

    #[test]
    fn opening_parses_the_header_and_leaves_the_pixels_for_the_first_row() {
        let mut decoder = WebPDecoder::new(LOSSY, Limits::default()).unwrap();
        // Everything metadata needs is known; nothing has been decoded.
        assert_eq!(decoder.descriptor().pixel, PixelFormat::Rgba8);
        assert!(decoder.pixels.is_none());

        let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
        decoder.read_row(&mut row).unwrap();
        assert!(decoder.pixels.is_some());
        assert!(decoder.bytes.is_empty(), "the compressed file is released");
    }

    #[test]
    fn a_bitstream_damaged_past_its_header_fails_on_the_first_row() {
        // The lossless bitstream starts at byte 20 (RIFF, WEBP, VP8L chunk
        // header) with a 5-byte header; everything after it is wrecked, so
        // the container and the image header still parse.
        let mut bytes = include_bytes!("../tests/fixtures/lossless/blocks_m6.webp").to_vec();
        for byte in bytes.iter_mut().skip(25) {
            *byte = 0xFF;
        }
        let mut decoder = WebPDecoder::new(&bytes[..], Limits::default()).unwrap();
        let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
        let error = decoder.read_row(&mut row).unwrap_err();
        assert_eq!(
            error.code(),
            otf_pixels_core::ErrorCode::Malformed,
            "{error}"
        );
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
