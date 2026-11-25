//! The WebP decoder.
//!
//! The container is parsed here (`riff`); the image decodes through the owned
//! VP8L (`vp8l`) or VP8 (`vp8`) decoder, with `ALPH` alpha (`alpha`) and
//! libwebp's YUV-to-RGB conversion (`yuv`).
//!
//! # Animation
//!
//! An animation's rows are the current frame composited onto its canvas,
//! the first until [`Decoder::advance_frame`] moves on, exactly as libwebp's
//! `WebPAnimDecoder` (`anim_decode.c`) renders it — which is not only the
//! RFC's blend and dispose: whether a frame is a *key frame* decides
//! whether it blends at all, pixels a background disposal just cleared are
//! written rather than blended, and blending is libwebp's integer
//! approximation of source-over. Each of those changes a pixel value, so
//! each is reproduced, not approximated. The canvas starts transparent
//! black and disposal clears to transparent black; the `ANIM` background
//! colour is a hint libwebp's decoder ignores, and so is it here.

use otf_pixels_core::{
    Animation, Codec, DecodeCapability, Decoder, Format, FrameInfo, ImageDescriptor, Limits,
    Orientation, PixelFormat, PixelsError, Result, Source, SourceInfo,
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
    /// What stops this codec's long work (ADR-0021).
    cancel: otf_pixels_core::Cancel,
    descriptor: ImageDescriptor,
    /// The whole file, until the first row is asked for.
    bytes: Vec<u8>,
    /// The decoded image, interleaved, once the first row is asked for.
    pixels: Option<Vec<u8>>,
    /// Rows already served.
    row: u32,
    /// From the `EXIF` chunk, if there is one.
    orientation: Orientation,
    /// The `EXIF` chunk's block.
    exif: Option<Vec<u8>>,
    /// The `ICCP` chunk, if there is one.
    icc: Option<Vec<u8>>,
    /// The `XMP ` chunk, if there is one.
    xmp: Option<Vec<u8>>,
    /// Density (from the EXIF block) and lossless/lossy.
    info: SourceInfo,
    /// The animation's frames and timing, for an animated file.
    animation: Option<Animation>,
    /// Where each `ANMF` payload lies in `bytes`; empty for a still.
    frames: Vec<std::ops::Range<usize>>,
    /// The animation's canvas state, once its first frame is drawn.
    compositor: Option<Compositor>,
}

/// libwebp's `WebPAnimDecoder` state between frames.
#[derive(Debug)]
struct Compositor {
    /// The current frame composited, RGBA.
    canvas: Vec<u8>,
    /// The canvas after the previous frame's disposal: what the next frame
    /// is drawn over (`prev_frame_disposed_`).
    disposed: Vec<u8>,
    /// The frame now on the canvas.
    index: u32,
    /// Its rectangle, disposal and whether it was a key frame.
    previous: (crate::riff::Frame, bool, bool),
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
        let xmp = container.xmp.map(<[u8]>::to_vec);
        let mut info = SourceInfo::default();
        info.density = container
            .exif
            .and_then(otf_pixels_core::Density::from_exif_block);
        info.bits_per_sample = Some(8);
        info.compression = Some(match container.bitstream {
            crate::riff::Bitstream::Lossless(_) => "lossless",
            crate::riff::Bitstream::Lossy { .. } => "lossy",
        });
        let exif = container
            .exif
            .map(|block| block.strip_prefix(b"Exif\0\0").unwrap_or(block).to_vec());
        let animation = Animation::new(container.frame_durations_ms.clone(), container.loop_count);
        let base = bytes.as_ptr() as usize;
        let frames = container
            .anmf
            .iter()
            .map(|payload| {
                let start = payload.as_ptr() as usize - base;
                start..start + payload.len()
            })
            .collect();
        Ok(Self {
            cancel: otf_pixels_core::Cancel::never(),
            descriptor,
            bytes,
            pixels: None,
            row: 0,
            orientation,
            exif,
            icc,
            xmp,
            info,
            animation,
            frames,
            compositor: None,
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

    /// Draw animation frame `index` the way `WebPAnimDecoder::GetNext`
    /// does, and return the canvas in the output pixel format.
    fn draw_frame(&mut self, index: u32) -> Result<Vec<u8>> {
        let range = self
            .frames
            .get(index as usize)
            .cloned()
            .ok_or_else(|| PixelsError::malformed("webp", "the animation has no such frame"))?;
        let payload = self.bytes.get(range).ok_or_else(|| {
            PixelsError::malformed("webp", "an animation frame lies outside the file")
        })?;
        let frame = crate::riff::anmf(payload)?;
        let (width, height) = (self.descriptor.width, self.descriptor.height);
        let rect = frame.rect;
        if u64::from(rect.x) + u64::from(rect.width) > u64::from(width)
            || u64::from(rect.y) + u64::from(rect.height) > u64::from(height)
        {
            return Err(PixelsError::malformed(
                "webp",
                format!(
                    "frame {index}, {}x{} at ({}, {}), does not fit the {width}x{height} canvas",
                    rect.width, rect.height, rect.x, rect.y
                ),
            ));
        }
        let rgba = decode_rgba(frame.bitstream, rect.width as usize, rect.height as usize)?;
        let canvas_len = width as usize * height as usize * 4;
        let full = |r: crate::riff::Frame| (r.width, r.height) == (width, height);

        // IsKeyFrame.
        let key = match &self.compositor {
            None => true,
            Some(state) => {
                let (previous, dispose_background, previous_key) = state.previous;
                ((!frame.has_alpha || !frame.blend) && full(rect))
                    || (dispose_background && (full(previous) || previous_key))
            }
        };
        let mut state = self.compositor.take().unwrap_or_else(|| Compositor {
            canvas: Vec::new(),
            disposed: vec![0; canvas_len],
            index: 0,
            previous: (rect, false, true),
        });
        // A key frame starts from transparent black, any other from the
        // previous frame as disposed.
        state.canvas = if key {
            vec![0; canvas_len]
        } else {
            state.disposed.clone()
        };

        // The frame's pixels into its rectangle, overwriting...
        let stride = width as usize * 4;
        let frame_stride = rect.width as usize * 4;
        for (y, source) in rgba.chunks_exact(frame_stride).enumerate() {
            let at = (rect.y as usize + y) * stride + rect.x as usize * 4;
            if let Some(target) = state.canvas.get_mut(at..at + frame_stride) {
                target.copy_from_slice(source);
            }
        }
        // ...then, on a blended frame that is not a key frame, blended over
        // the disposed canvas: every pixel but those the previous frame's
        // background disposal cleared (FindBlendRangeAtRow).
        if !key && frame.blend {
            let (previous, dispose_background, _) = state.previous;
            for y in rect.y..rect.y + rect.height {
                for x in rect.x..rect.x + rect.width {
                    let cleared = dispose_background
                        && (previous.x..previous.x + previous.width).contains(&x)
                        && (previous.y..previous.y + previous.height).contains(&y);
                    if cleared {
                        continue;
                    }
                    let at = y as usize * stride + x as usize * 4;
                    let pixel = |canvas: &[u8]| -> Option<[u8; 4]> {
                        canvas.get(at..at + 4)?.try_into().ok()
                    };
                    let (Some(src), Some(dst)) = (pixel(&state.canvas), pixel(&state.disposed))
                    else {
                        continue;
                    };
                    let blended = blend_non_premultiplied(src, dst);
                    if let Some(slot) = state.canvas.get_mut(at..at + 4) {
                        slot.copy_from_slice(&blended);
                    }
                }
            }
        }

        // This frame becomes the previous, disposed for the next.
        state.disposed.clone_from(&state.canvas);
        if frame.dispose_background {
            for y in rect.y..rect.y + rect.height {
                let at = y as usize * stride + rect.x as usize * 4;
                if let Some(row) = state.disposed.get_mut(at..at + frame_stride) {
                    row.fill(0);
                }
            }
        }
        state.previous = (rect, frame.dispose_background, key);
        state.index = index;

        let channels = self.descriptor.pixel.channels();
        let out = if channels == 4 {
            state.canvas.clone()
        } else {
            state
                .canvas
                .chunks_exact(4)
                .flat_map(|p| p.iter().take(3).copied())
                .collect()
        };
        self.compositor = Some(state);
        Ok(out)
    }
}

/// libwebp's `BlendPixelNonPremult`: `src` over `dst`, both unassociated
/// RGBA, in its integer arithmetic. Opaque and fully transparent sources are
/// libwebp's early outs (`BlendPixelRowNonPremult` skips the first).
fn blend_non_premultiplied(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let src_a = u32::from(src[3]);
    if src_a == 255 {
        return src;
    }
    if src_a == 0 {
        return dst;
    }
    // dst_a · (255 - src_a) / 255, approximated as libwebp does.
    let dst_factor_a = (u32::from(dst[3]) * (256 - src_a)) >> 8;
    let blend_a = src_a + dst_factor_a;
    let scale = (1_u32 << 24) / blend_a;
    let channel = |s: u8, d: u8| {
        let unscaled = u32::from(s) * src_a + u32::from(d) * dst_factor_a;
        ((u64::from(unscaled) * u64::from(scale)) >> 24) as u8
    };
    [
        channel(src[0], dst[0]),
        channel(src[1], dst[1]),
        channel(src[2], dst[2]),
        blend_a as u8,
    ]
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

    /// The `EXIF` chunk's block.
    fn exif(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }

    /// The `XMP ` chunk.
    fn xmp(&self) -> Option<&[u8]> {
        self.xmp.as_deref()
    }

    /// Lossless or lossy, and the density the EXIF block states.
    fn source_info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn animation(&self) -> Option<Animation> {
        self.animation.clone()
    }

    /// The next frame, composited as libwebp's animation decoder does (see
    /// the module docs).
    fn advance_frame(&mut self) -> Result<Option<FrameInfo>> {
        let cancel = self.cancel.clone();
        cancel.scope(|| {
            if self.frames.len() < 2 {
                self.row = self.descriptor.height;
                return Ok(None);
            }
            if self.compositor.is_none() {
                self.pixels = Some(self.draw_frame(0)?);
            }
            let next = self.compositor.as_ref().map_or(0, |state| state.index) + 1;
            if next as usize >= self.frames.len() {
                self.row = self.descriptor.height;
                return Ok(None);
            }
            self.pixels = Some(self.draw_frame(next)?);
            self.row = 0;
            let duration = self
                .animation
                .as_ref()
                .and_then(|a| a.frame_durations_ms.get(next as usize).copied())
                .unwrap_or(0);
            Ok(Some(FrameInfo::new(next, duration)))
        })
    }

    fn capability(&self) -> DecodeCapability {
        // The image is already in memory, but `Sequential` is what the row
        // contract describes; claiming `Regions` would promise a
        // `read_region` this does not implement.
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        let cancel = self.cancel.clone();
        cancel.scope(|| {
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
                if self.frames.is_empty() {
                    self.pixels = Some(self.decode_canvas()?);
                    // Decoded, the compressed file is no longer needed.
                    self.bytes = Vec::new();
                } else {
                    // Later frames are drawn from the file, so it stays.
                    self.pixels = Some(self.draw_frame(0)?);
                }
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
        })
    }

    fn set_cancel(&mut self, cancel: otf_pixels_core::Cancel) {
        self.cancel = cancel;
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
