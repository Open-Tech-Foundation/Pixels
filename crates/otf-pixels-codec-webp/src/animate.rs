//! Animated WebP encoding (ADR-0020): `ANIM` and `ANMF`, each frame only
//! what changed.
//!
//! # What a frame holds
//!
//! Frame `k` is the bounding box of the pixels that differ from frame
//! `k - 1`, its left and top rounded down to even as `ANMF` requires. Two
//! ways to draw it, chosen per frame as libwebp's `anim_encode` would:
//!
//! - **Blended**, when every pixel the frame draws is opaque: unchanged
//!   pixels become transparent black, so the previous frame shows through
//!   them, and the frame is alpha-blended. Transparent runs cost almost
//!   nothing, so this is the small one.
//! - **Overwritten**, otherwise: the box is written as the frame has it and
//!   replaces what was there, translucency included. Blending a translucent
//!   pixel would mix it with the one beneath.
//!
//! Either way nothing needs disposing, so no frame is cleared to the
//! background. A frame identical to the one before lengthens it instead
//! (sharp's `keepDuplicateFrames` off).
//!
//! Each frame is coded lossy or lossless as a still would be, by the same
//! options. The file is assembled in memory: `VP8X` must say whether any
//! frame has alpha before the first frame is written.

use otf_pixels_core::{EncodeOptions, ImageDescriptor, PixelFormat, PixelsError, Result, Sink};

/// Make the unchanged pixels of a blended frame transparent black, so the
/// previous frame shows through them. Lossless, every one; lossy, only
/// those in an 8x8 block left wholly unchanged, as libwebp's
/// `FlattenSimilarBlocks` does: VP8 codes blocks, and a block half
/// transparent black would smear the black into the pixels it does draw.
fn blank_unchanged(pixels: &mut [u8], unchanged: &[bool], width: usize, lossless: bool) {
    if lossless {
        for (pixel, &same) in pixels.chunks_exact_mut(4).zip(unchanged) {
            if same {
                pixel.fill(0);
            }
        }
        return;
    }
    let height = unchanged.len() / width.max(1);
    for by in (0..height).step_by(8) {
        for bx in (0..width).step_by(8) {
            let rows = by..(by + 8).min(height);
            let columns = bx..(bx + 8).min(width);
            let whole = rows.clone().all(|y| {
                columns
                    .clone()
                    .all(|x| unchanged.get(y * width + x).copied().unwrap_or(false))
            });
            if whole {
                for y in rows {
                    for x in columns.clone() {
                        let at = (y * width + x) * 4;
                        if let Some(pixel) = pixels.get_mut(at..at + 4) {
                            pixel.fill(0);
                        }
                    }
                }
            }
        }
    }
}

/// A frame's rectangle on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// A frame decided but not yet coded: its rectangle, pixels and how it is
/// drawn.
#[derive(Debug)]
struct Planned {
    rect: Rect,
    /// The rectangle's pixels, RGBA.
    pixels: Vec<u8>,
    blend: bool,
    duration_ms: u32,
}

/// The state of one animated WebP being written.
#[derive(Debug)]
pub(crate) struct Animator {
    width: u32,
    height: u32,
    format: PixelFormat,
    loop_count: u32,
    options: EncodeOptions,
    /// The previous frame as given, RGBA.
    previous: Option<Vec<u8>>,
    pending: Option<Planned>,
    current: Option<(u32, Vec<u8>)>,
    rows: u32,
    /// The `ANMF` chunks so far.
    frames: Vec<u8>,
    /// Whether any frame has a pixel less than opaque.
    alpha: bool,
}

impl Animator {
    pub(crate) fn new(
        descriptor: &ImageDescriptor,
        loop_count: u32,
        options: EncodeOptions,
    ) -> Self {
        Self {
            width: descriptor.width,
            height: descriptor.height,
            format: descriptor.pixel,
            loop_count,
            options,
            previous: None,
            pending: None,
            current: None,
            rows: 0,
            frames: Vec::new(),
            alpha: false,
        }
    }

    pub(crate) fn begin_frame(&mut self, duration_ms: u32) -> Result<()> {
        if self.current.is_some() {
            return Err(PixelsError::invalid_argument(
                "frame",
                "begin_frame called with a frame still open",
            ));
        }
        let capacity = self.width as usize * self.height as usize * 4;
        self.current = Some((duration_ms, Vec::with_capacity(capacity)));
        self.rows = 0;
        Ok(())
    }

    pub(crate) fn write_row(&mut self, row: &[u8]) -> Result<()> {
        let channels = self.format.channels();
        let Some((_, rgba)) = self.current.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "row",
                "write_row called outside a frame",
            ));
        };
        if row.len() != self.width as usize * channels || self.rows >= self.height {
            return Err(PixelsError::invalid_argument(
                "row",
                "wrong length, or more rows than the canvas has",
            ));
        }
        for p in row.chunks_exact(channels) {
            let pixel = match *p {
                [g] => [g, g, g, 255],
                [g, a] => [g, g, g, a],
                [r, g, b] => [r, g, b, 255],
                [r, g, b, a] => [r, g, b, a],
                _ => [0, 0, 0, 0],
            };
            rgba.extend_from_slice(&pixel);
        }
        self.rows += 1;
        Ok(())
    }

    pub(crate) fn end_frame(&mut self) -> Result<()> {
        let Some((duration, frame)) = self.current.take() else {
            return Err(PixelsError::invalid_argument(
                "frame",
                "end_frame called with no frame open",
            ));
        };
        if self.rows != self.height {
            return Err(PixelsError::malformed(
                "webp",
                format!("a frame ended after {} of {} rows", self.rows, self.height),
            ));
        }
        self.alpha |= frame.chunks_exact(4).any(|p| p.get(3) != Some(&255));
        let transparent = [0_u8; 4];
        // What is on screen before this frame: the previous frame, or a
        // transparent canvas.
        let changed = |i: usize, previous: Option<&[u8]>| -> bool {
            let now = frame.get(i * 4..i * 4 + 4).unwrap_or(&transparent);
            let before = previous.map_or(&transparent[..], |p| {
                p.get(i * 4..i * 4 + 4).unwrap_or(&transparent)
            });
            // Two transparent pixels look the same whatever their colour.
            !(now == before || (now.get(3) == Some(&0) && before.get(3) == Some(&0)))
        };
        let previous = self.previous.as_deref();
        let found = self.bounds(|i| changed(i, previous));
        let Some(rect) = found else {
            if let (Some(pending), false) =
                (self.pending.as_mut(), self.options.keep_duplicate_frames)
            {
                pending.duration_ms = pending.duration_ms.saturating_add(duration);
                return Ok(());
            }
            // Nothing to draw, but a frame to show: one transparent pixel,
            // blended, changes nothing.
            self.queue(Planned {
                rect: Rect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
                pixels: vec![0; 4],
                blend: true,
                duration_ms: duration,
            })?;
            self.previous = Some(frame);
            return Ok(());
        };
        let stride = self.width as usize;
        let mut pixels = Vec::with_capacity(rect.width as usize * rect.height as usize * 4);
        let mut opaque_changes = true;
        let mut unchanged = Vec::with_capacity(rect.width as usize * rect.height as usize);
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let i = y as usize * stride + x as usize;
                let pixel = frame.get(i * 4..i * 4 + 4).unwrap_or(&transparent);
                let differs = changed(i, previous);
                if differs && pixel.get(3) != Some(&255) {
                    opaque_changes = false;
                }
                unchanged.push(!differs);
                pixels.extend_from_slice(pixel);
            }
        }
        // The first frame is drawn on transparent black: blending gains
        // nothing there, and libwebp does not blend a key frame.
        // Blending is exact only where what the frame draws is opaque: the
        // changed pixels, and any unchanged one it still draws (lossy keeps
        // those in a block with a change). Otherwise the box is overwritten.
        let mut blend = opaque_changes && previous.is_some();
        if blend {
            let lossless = self.options.lossless || self.options.near_lossless;
            let mut blanked = pixels.clone();
            blank_unchanged(&mut blanked, &unchanged, rect.width as usize, lossless);
            if blanked
                .chunks_exact(4)
                .all(|p| matches!(p.get(3), Some(0 | 255)))
            {
                pixels = blanked;
            } else {
                blend = false;
            }
        }
        self.queue(Planned {
            rect,
            pixels,
            blend,
            duration_ms: duration,
        })?;
        self.previous = Some(frame);
        Ok(())
    }

    /// Code the pending frame, and hold `next` in its place: one frame is
    /// held back so a duplicate after it can still lengthen it.
    fn queue(&mut self, next: Planned) -> Result<()> {
        if let Some(done) = self.pending.replace(next) {
            self.code(&done)?;
        }
        Ok(())
    }

    /// The bounding box of the pixels for which `hit` holds, its left and
    /// top rounded down to even.
    fn bounds(&self, mut hit: impl FnMut(usize) -> bool) -> Option<Rect> {
        let (mut left, mut top, mut right, mut bottom) = (u32::MAX, u32::MAX, 0, 0);
        for y in 0..self.height {
            for x in 0..self.width {
                if hit((y * self.width + x) as usize) {
                    left = left.min(x);
                    top = top.min(y);
                    right = right.max(x + 1);
                    bottom = bottom.max(y + 1);
                }
            }
        }
        (left < right).then(|| {
            let (x, y) = (left & !1, top & !1);
            Rect {
                x,
                y,
                width: right - x,
                height: bottom - y,
            }
        })
    }

    /// Append `frame` as an `ANMF` chunk.
    fn code(&mut self, frame: &Planned) -> Result<()> {
        otf_pixels_core::Cancel::checkpoint()?;
        let rect = frame.rect;
        let descriptor = ImageDescriptor::new(rect.width, rect.height, PixelFormat::Rgba8)?;
        let (image, _) = crate::encoder::encode_image(&descriptor, &frame.pixels, &self.options)?;
        let mut payload = Vec::with_capacity(16 + image.len());
        for value in [rect.x / 2, rect.y / 2, rect.width - 1, rect.height - 1] {
            payload.extend_from_slice(&value.to_le_bytes()[..3]);
        }
        payload.extend_from_slice(&frame.duration_ms.min(0x00ff_ffff).to_le_bytes()[..3]);
        // Bit 1 set: do not blend. Bit 0 clear: no disposal.
        payload.push(if frame.blend { 0 } else { 0x02 });
        payload.extend_from_slice(&image);
        crate::encoder::chunk(&mut self.frames, b"ANMF", &payload);
        Ok(())
    }

    /// Code the last frame and write the file: `VP8X`, `ICCP`, `ANIM`, the
    /// frames, `EXIF` and `XMP `.
    pub(crate) fn finish(
        &mut self,
        icc: Option<&[u8]>,
        exif: Option<&[u8]>,
        xmp: Option<&[u8]>,
        sink: &mut dyn Sink,
    ) -> Result<()> {
        if self.current.is_some() {
            return Err(PixelsError::invalid_argument(
                "frame",
                "finish called with a frame still open",
            ));
        }
        let Some(last) = self.pending.take() else {
            return Err(PixelsError::invalid_argument(
                "frame",
                "an animation needs at least one frame",
            ));
        };
        self.code(&last)?;
        // VP8X flags: ICC 0x20, alpha 0x10, EXIF 0x08, XMP 0x04, animation
        // 0x02; then the canvas size less one.
        let flags = 0x02
            | if icc.is_some() { 0x20 } else { 0 }
            | if self.alpha { 0x10 } else { 0 }
            | if exif.is_some() { 0x08 } else { 0 }
            | if xmp.is_some() { 0x04 } else { 0 };
        let mut vp8x = vec![flags, 0, 0, 0];
        vp8x.extend_from_slice(&(self.width - 1).to_le_bytes()[..3]);
        vp8x.extend_from_slice(&(self.height - 1).to_le_bytes()[..3]);
        let mut body = Vec::with_capacity(self.frames.len() + 64);
        crate::encoder::chunk(&mut body, b"VP8X", &vp8x);
        if let Some(profile) = icc {
            crate::encoder::chunk(&mut body, b"ICCP", profile);
        }
        // ANIM: a background colour (transparent; readers may ignore it, as
        // libwebp's decoder does) and the loop count.
        let mut anim = vec![0_u8; 4];
        anim.extend_from_slice(&(self.loop_count.min(u32::from(u16::MAX)) as u16).to_le_bytes());
        crate::encoder::chunk(&mut body, b"ANIM", &anim);
        body.extend_from_slice(&self.frames);
        if let Some(exif) = exif {
            crate::encoder::chunk(&mut body, b"EXIF", exif);
        }
        if let Some(xmp) = xmp {
            crate::encoder::chunk(&mut body, b"XMP ", xmp);
        }
        sink.write_all(&crate::encoder::riff(&body))?;
        sink.flush()
    }
}
