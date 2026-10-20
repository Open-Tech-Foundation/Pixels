//! The WebP encoder.
//!
//! Lossy by default, through the owned VP8 encoder at
//! [`EncodeOptions::quality`]; [`EncodeOptions::lossless`] selects the owned
//! VP8L encoder. The rest of sharp's WebP options follow libwebp:
//! [`EncodeOptions::effort`] is its `method`, [`EncodeOptions::alpha_quality`]
//! reduces a lossy image's alpha to fewer levels as its `alpha_quality`
//! does, and [`EncodeOptions::near_lossless`] quantizes lossless residuals
//! at `quality` as its `near_lossless` does. Either way the whole image is gathered first: both
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
    /// The EXIF block to write as `EXIF`, if any.
    exif: Option<Vec<u8>>,
    /// The XMP packet to write as `XMP `, if any.
    xmp: Option<Vec<u8>>,
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
            exif: None,
            xmp: None,
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

    fn set_exif(&mut self, exif: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "exif",
                "the EXIF block must be set before write_header",
            ));
        }
        self.exif = exif.map(<[u8]>::to_vec);
        Ok(())
    }

    fn set_xmp(&mut self, xmp: Option<&[u8]>) -> Result<()> {
        if self.state.is_some() {
            return Err(PixelsError::invalid_argument(
                "xmp",
                "the XMP packet must be set before write_header",
            ));
        }
        self.xmp = xmp.map(<[u8]>::to_vec);
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

        let options = self.options;
        let effort = options.effort.unwrap_or(DEFAULT_EFFORT).min(6);
        let lossless = options.lossless || options.near_lossless;
        let (body, alpha) = if lossless {
            let near = options.near_lossless.then_some(options.quality);
            encode_lossless(state, effort, near)
        } else {
            encode_lossy(state, options.quality, effort, options.alpha_quality)?
        };
        let (width, height) = (state.descriptor.width, state.descriptor.height);
        // VP8L carries its own alpha; only a lossy ALPH chunk needs VP8X.
        let needs_vp8x = alpha && !lossless;
        let file = container(
            width,
            height,
            &body,
            alpha,
            needs_vp8x,
            self.icc.as_deref(),
            self.exif.as_deref(),
            self.xmp.as_deref(),
        );
        sink.write_all(&file)?;
        sink.flush()
    }
}

/// libwebp's default `method`, and the search this encoder made before it
/// took an effort.
const DEFAULT_EFFORT: u8 = 4;

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
fn encode_lossy(
    state: &State,
    quality: u8,
    effort: u8,
    alpha_quality: u8,
) -> Result<(Vec<u8>, bool)> {
    let (width, height) = (
        state.descriptor.width as usize,
        state.descriptor.height as usize,
    );
    let channels = state.descriptor.pixel.channels();
    let (planes, alpha) = crate::yuv::from_rgb(&state.pixels, channels, width, height);
    let mut params = crate::vp8::encode::Params::with_quality(quality.clamp(1, 100));
    params.intra4 = effort >= 2;
    let vp8 = crate::vp8::encode::encode(&planes, params)?;
    // An opaque alpha channel is dropped: a simple file says the same thing
    // in fewer bytes and every reader handles it.
    let mut alpha = alpha.filter(|a| a.iter().any(|&v| v != 255));
    let mut body = Vec::new();
    if let Some(alpha) = &mut alpha {
        if alpha_quality < 100 {
            crate::preprocess::quantize_levels(
                alpha,
                crate::preprocess::alpha_levels(alpha_quality),
            );
        }
        chunk(
            &mut body,
            b"ALPH",
            &encode_alpha(alpha, width, height, effort),
        );
    }
    chunk(&mut body, b"VP8 ", &vp8);
    Ok((body, alpha.is_some()))
}

/// The whole file: the image chunks alone in a simple file, or behind a
/// `VP8X` header (with an `ICCP` chunk before them and `EXIF` and `XMP `
/// chunks after) when the alpha needs one or there is metadata.
#[allow(
    clippy::too_many_arguments,
    reason = "the image and each kind of metadata"
)]
fn container(
    width: u32,
    height: u32,
    image: &[u8],
    alpha: bool,
    needs_vp8x: bool,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
) -> Vec<u8> {
    if !needs_vp8x && icc.is_none() && exif.is_none() && xmp.is_none() {
        return riff(image);
    }
    // VP8X flags: ICC 0x20, alpha 0x10, EXIF 0x08, XMP 0x04; then the
    // canvas size less one.
    let flags = if icc.is_some() { 0x20 } else { 0 }
        | if alpha { 0x10 } else { 0 }
        | if exif.is_some() { 0x08 } else { 0 }
        | if xmp.is_some() { 0x04 } else { 0 };
    let mut vp8x = vec![flags, 0, 0, 0];
    vp8x.extend_from_slice(&(width - 1).to_le_bytes()[..3]);
    vp8x.extend_from_slice(&(height - 1).to_le_bytes()[..3]);
    let mut body = Vec::new();
    chunk(&mut body, b"VP8X", &vp8x);
    if let Some(profile) = icc {
        chunk(&mut body, b"ICCP", profile);
    }
    body.extend_from_slice(image);
    if let Some(exif) = exif {
        chunk(&mut body, b"EXIF", exif);
    }
    if let Some(xmp) = xmp {
        chunk(&mut body, b"XMP ", xmp);
    }
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
fn encode_lossless(state: &State, effort: u8, near_lossless: Option<u8>) -> (Vec<u8>, bool) {
    let (width, height) = (
        state.descriptor.width as usize,
        state.descriptor.height as usize,
    );
    let argb = to_argb(&state.pixels, state.descriptor.pixel.channels());
    let mut tuning = crate::vp8l_encode::Tuning::for_effort(effort);
    // libwebp skips near-lossless for icons under 64 pixels both ways.
    if let Some(quality) = near_lossless.filter(|_| width >= 64 || height >= 64) {
        tuning = tuning.near_lossless(quality);
    }
    let has_alpha = argb.iter().any(|&p| p >> 24 != 0xff);
    let mut body = Vec::new();
    chunk(
        &mut body,
        b"VP8L",
        &crate::vp8l_encode::encode(&argb, width, height, has_alpha, tuning),
    );
    (body, has_alpha)
}

/// An `ALPH` chunk payload: no filter, lossless VP8L compression, the alpha
/// carried in the green channel of an image stream of implicit size.
fn encode_alpha(alpha: &[u8], width: usize, height: usize, effort: u8) -> Vec<u8> {
    let argb: Vec<u32> = alpha
        .iter()
        .map(|&a| 0xff00_0000 | (u32::from(a) << 8))
        .collect();
    let mut w = crate::vp8l_encode::BitWriter::default();
    crate::vp8l_encode::write_image_stream(
        &mut w,
        &argb,
        width,
        height,
        crate::vp8l_encode::Tuning::for_effort(effort),
    );
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
    use crate::preprocess::{alpha_levels, quantize_levels};
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

    /// Encode `pixels` with `options`, returning the file.
    fn encode(descriptor: &ImageDescriptor, pixels: &[u8], options: EncodeOptions) -> Vec<u8> {
        let mut encoder = WebPEncoder::from_options(&options);
        let mut out = Vec::new();
        encoder.write_header(descriptor, &mut out).unwrap();
        for row in pixels.chunks_exact(descriptor.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        out
    }

    /// Decode a file to RGBA.
    fn decode(bytes: &[u8]) -> Vec<u8> {
        use otf_pixels_core::{Decoder, Limits};
        let mut decoder = crate::WebPDecoder::new(bytes, Limits::default()).unwrap();
        let descriptor = decoder.descriptor();
        let mut row = vec![0_u8; descriptor.row_bytes()];
        let mut out = Vec::new();
        for _ in 0..descriptor.height {
            decoder.read_row(&mut row).unwrap();
            out.extend_from_slice(&row);
        }
        out
    }

    /// A `width` x `height` RGBA image: noise in colour, a ramp in alpha.
    fn noisy(width: u32, height: u32) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgba8).unwrap();
        let mut seed = 99_u32;
        let mut pixels = Vec::new();
        for _y in 0..height {
            for x in 0..width {
                for _ in 0..3 {
                    seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    pixels.push((seed >> 16) as u8);
                }
                pixels.push((x * 255 / (width - 1)) as u8);
            }
        }
        (descriptor, pixels)
    }

    fn distinct(values: impl Iterator<Item = u8>) -> usize {
        let mut seen = [false; 256];
        values.for_each(|v| seen[usize::from(v)] = true);
        seen.iter().filter(|&&s| s).count()
    }

    #[test]
    fn alpha_levels_follow_libwebp() {
        assert_eq!(alpha_levels(0), 2);
        assert_eq!(alpha_levels(50), 12);
        assert_eq!(alpha_levels(70), 16);
        assert_eq!(alpha_levels(71), 24);
        assert_eq!(alpha_levels(99), 248);
    }

    #[test]
    fn alpha_is_reduced_to_its_levels_keeping_the_ends() {
        let mut ramp: Vec<u8> = (0..=255).collect();
        quantize_levels(&mut ramp, 4);
        assert!(distinct(ramp.iter().copied()) <= 4);
        assert_eq!((ramp[0], ramp[255]), (0, 255), "the ends are fixed");
        // Values already within the level count are left alone.
        let mut few = vec![3, 3, 200, 9];
        quantize_levels(&mut few, 4);
        assert_eq!(few, [3, 3, 200, 9]);
    }

    #[test]
    fn a_lower_alpha_quality_keeps_fewer_alpha_levels() {
        let (descriptor, pixels) = noisy(64, 32);
        let alpha = |quality| {
            let options = EncodeOptions::default()
                .with_alpha_quality(quality)
                .unwrap();
            let decoded = decode(&encode(&descriptor, &pixels, options));
            distinct(decoded.chunks_exact(4).map(|p| p[3]))
        };
        assert_eq!(alpha(100), 64, "lossless alpha keeps every level");
        assert!(alpha(0) <= 2);
        assert!(alpha(50) <= 12);
    }

    #[test]
    fn near_lossless_stays_close_and_is_exact_at_full_quality() {
        let (descriptor, pixels) = noisy(80, 70);
        let options = EncodeOptions::default().with_near_lossless(true);
        let bytes = encode(&descriptor, &pixels, options);
        assert!(
            bytes.windows(4).any(|w| w == b"VP8L"),
            "near-lossless is lossless"
        );
        let decoded = decode(&bytes);
        assert_ne!(decoded, pixels, "quality 80 changes something");
        // Errors are bounded by the largest quantization, 2 at quality 80,
        // with green's compensation carried into red and blue.
        for (got, want) in decoded.chunks_exact(4).zip(pixels.chunks_exact(4)) {
            assert!(
                got.iter().zip(want).all(|(&g, &w)| g.abs_diff(w) <= 2),
                "{got:?} {want:?}"
            );
        }
        let exact = EncodeOptions::with_quality(100)
            .unwrap()
            .with_near_lossless(true);
        assert_eq!(decode(&encode(&descriptor, &pixels, exact)), pixels);
        // An icon is left exact, as libwebp leaves it.
        let (small, small_pixels) = noisy(32, 32);
        assert_eq!(
            decode(&encode(&small, &small_pixels, options)),
            small_pixels
        );
    }

    #[test]
    fn every_effort_decodes_and_the_default_is_four() {
        let (descriptor, pixels) = noisy(48, 40);
        for lossless in [false, true] {
            let base = EncodeOptions::default().with_lossless(lossless);
            let default = encode(&descriptor, &pixels, base);
            assert_eq!(
                default,
                encode(&descriptor, &pixels, base.with_effort(4).unwrap())
            );
            for effort in 0..=10 {
                let bytes = encode(&descriptor, &pixels, base.with_effort(effort).unwrap());
                let decoded = decode(&bytes);
                if lossless {
                    assert_eq!(decoded, pixels, "effort {effort}");
                } else {
                    assert_eq!(decoded.len(), pixels.len());
                }
            }
        }
        // The fast lossy search skips 4x4 prediction, which costs bytes on
        // detail.
        let lossy = EncodeOptions::default();
        assert_ne!(
            encode(&descriptor, &pixels, lossy.with_effort(0).unwrap()),
            encode(&descriptor, &pixels, lossy)
        );
    }
}
