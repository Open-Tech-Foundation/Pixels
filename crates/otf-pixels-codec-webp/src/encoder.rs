//! The WebP encoder.
//!
//! Lossy by default, through the owned VP8 encoder at
//! [`EncodeOptions::quality`]; [`EncodeOptions::lossless`] selects the owned
//! VP8L encoder. Either way the whole image is gathered first: both
//! bitstreams make decisions over the whole picture before the first byte is
//! final.

use otf_pixels_core::{
    EncodeOptions, Encoder, ImageDescriptor, PixelFormat, PixelsError, Result, Sink,
};

/// Encodes a WebP stream.
#[derive(Debug)]
pub struct WebPEncoder {
    /// Set by `write_header`; its presence means the header was written.
    state: Option<State>,
    options: EncodeOptions,
    /// The ICC profile to write as `ICCP`, if any.
    icc: Option<Vec<u8>>,
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
            icc: None,
        }
    }
}

/// Refuse pixel formats WebP cannot carry: it is 8 bits per channel.
fn check_format(format: PixelFormat) -> Result<()> {
    match format {
        PixelFormat::Gray8 | PixelFormat::GrayA8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 => Ok(()),
        other => Err(PixelsError::unsupported(format!(
            "WebP encoding needs an 8-bit format; got {other}. Convert first."
        ))),
    }
}

impl Encoder for WebPEncoder {
    fn set_icc_profile(&mut self, profile: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "profile",
                "the ICC profile must be set before write_header",
            ));
        }
        self.icc = profile.map(<[u8]>::to_vec);
        Ok(())
    }

    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called more than once",
            ));
        }
        check_format(desc.pixel)?;
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

        let (body, alpha) = if self.options.lossless {
            encode_lossless(state)
        } else {
            encode_lossy(state, self.options.quality)?
        };
        let (width, height) = (state.descriptor.width, state.descriptor.height);
        // VP8L carries its own alpha; only a lossy ALPH chunk needs VP8X.
        let needs_vp8x = alpha && !self.options.lossless;
        let file = container(width, height, &body, alpha, needs_vp8x, self.icc.as_deref());
        sink.write_all(&file)?;
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

/// The image chunks of a lossy WebP: a `VP8 ` frame, preceded by an `ALPH`
/// chunk when the image has any transparency; and whether it does.
fn encode_lossy(state: &State, quality: u8) -> Result<(Vec<u8>, bool)> {
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
    if let Some(alpha) = &alpha {
        chunk(&mut body, b"ALPH", &encode_alpha(alpha, width, height));
    }
    chunk(&mut body, b"VP8 ", &vp8);
    Ok((body, alpha.is_some()))
}

/// The whole file: the image chunks alone in a simple file, or behind a
/// `VP8X` header (and an `ICCP` chunk) when the alpha needs one or there is a
/// profile.
fn container(
    width: u32,
    height: u32,
    image: &[u8],
    alpha: bool,
    needs_vp8x: bool,
    icc: Option<&[u8]>,
) -> Vec<u8> {
    if !needs_vp8x && icc.is_none() {
        return riff(image);
    }
    // VP8X flags: ICC 0x20, alpha 0x10; then the canvas size less one.
    let flags = if icc.is_some() { 0x20 } else { 0 } | if alpha { 0x10 } else { 0 };
    let mut vp8x = vec![flags, 0, 0, 0];
    vp8x.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
    vp8x.extend_from_slice(&(height - 1).to_le_bytes()[..3]);
    let mut body = Vec::new();
    chunk(&mut body, b"VP8X", &vp8x);
    if let Some(profile) = icc {
        chunk(&mut body, b"ICCP", profile);
    }
    body.extend_from_slice(image);
    riff(&body)
}

/// Interleaved samples as VP8L's ARGB words, grey spread to all three
/// colour channels.
fn to_argb(pixels: &[u8], channels: usize) -> Vec<u32> {
    pixels
        .chunks_exact(channels)
        .map(|p| {
            let (rgb, alpha) = match *p {
                [g] => ([g, g, g], 255),
                [g, a] => ([g, g, g], a),
                [r, g, b] => ([r, g, b], 255),
                [r, g, b, a, ..] => ([r, g, b], a),
                _ => ([0; 3], 255),
            };
            u32::from_be_bytes([alpha, rgb[0], rgb[1], rgb[2]])
        })
        .collect()
}

/// The image chunk of a lossless WebP, one `VP8L`, and whether it has
/// transparency.
fn encode_lossless(state: &State) -> (Vec<u8>, bool) {
    let (width, height) = (
        state.descriptor.width as usize,
        state.descriptor.height as usize,
    );
    let argb = to_argb(&state.pixels, state.descriptor.pixel.channels());
    let has_alpha = argb.iter().any(|&p| p >> 24 != 0xff);
    let mut body = Vec::new();
    chunk(
        &mut body,
        b"VP8L",
        &crate::vp8l_encode::encode(&argb, width, height, has_alpha),
    );
    (body, has_alpha)
}

/// An `ALPH` chunk payload: no filter, lossless VP8L compression, the alpha
/// carried in the green channel of an image stream of implicit size.
fn encode_alpha(alpha: &[u8], width: usize, height: usize) -> Vec<u8> {
    let argb: Vec<u32> = alpha
        .iter()
        .map(|&a| 0xff00_0000 | (u32::from(a) << 8))
        .collect();
    let mut w = crate::vp8l_encode::BitWriter::default();
    crate::vp8l_encode::write_image_stream(&mut w, &argb, width, height);
    let mut out = vec![1]; // compression 1, filter 0, no preprocessing
    out.extend_from_slice(&w.finish());
    out
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
