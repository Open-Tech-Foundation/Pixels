//! Animated JPEG XL encoding (ADR-0020): one codestream, a frame per frame.
//!
//! The image header declares an animation whose ticks are milliseconds, so
//! every duration is exact. The first frame covers the canvas. Each later
//! frame is cropped to the pixels that changed and *replaces* them over
//! reference slot 1, where every frame but the last saves the canvas it
//! leaves: libjxl's own way of animating a still background. Replacing
//! writes every channel, alpha included, so nothing ever needs clearing.
//!
//! Frames are lossy or lossless as a still would be: lossy VarDCT unless
//! the image has alpha or lossless is asked for. A frame identical to the
//! one before lengthens it instead (sharp's `keepDuplicateFrames` off).

use otf_pixels_core::{ImageDescriptor, PixelsError, Result, Sink};

use crate::bits::BitWriter;
use crate::headers::{FrameSpec, ImageInfo, write_image_header};

/// The reference slot frames are drawn over and saved to.
const SLOT: u32 = 1;

/// A frame decided but not yet coded.
#[derive(Debug)]
struct Planned {
    /// Its rectangle, `(x, y, width, height)`, or `None` for the canvas.
    crop: Option<(u32, u32, u32, u32)>,
    /// The rectangle's samples, in the image's pixel format.
    pixels: Vec<u8>,
    duration_ms: u32,
}

/// The state of one animated JPEG XL being written.
#[derive(Debug)]
pub(crate) struct Animator {
    descriptor: ImageDescriptor,
    /// `None` for lossless frames.
    distance: Option<f32>,
    keep_duplicates: bool,
    image: ImageInfo,
    previous: Option<Vec<u8>>,
    pending: Option<Planned>,
    current: Option<(u32, Vec<u8>)>,
    rows: u32,
    /// The codestream so far: the image header, then each coded frame.
    out: Vec<u8>,
}

impl Animator {
    pub(crate) fn new(
        descriptor: &ImageDescriptor,
        loop_count: u32,
        distance: Option<f32>,
        keep_duplicates: bool,
    ) -> Result<Self> {
        let distance = distance.filter(|_| !descriptor.pixel.layout().has_alpha());
        let mut image = match distance {
            Some(_) => crate::encoder::lossy_info(descriptor)?,
            None => crate::encoder::lossless_info(descriptor)?,
        };
        image.animation = Some(loop_count);
        let mut header = BitWriter::new();
        write_image_header(&mut header, &image);
        Ok(Self {
            descriptor: *descriptor,
            distance,
            keep_duplicates,
            image,
            previous: None,
            pending: None,
            current: None,
            rows: 0,
            out: header.finish(),
        })
    }

    pub(crate) fn begin_frame(&mut self, duration_ms: u32) -> Result<()> {
        if self.current.is_some() {
            return Err(PixelsError::invalid_argument(
                "frame",
                "begin_frame called with a frame still open",
            ));
        }
        let capacity = self.descriptor.byte_len().unwrap_or_default();
        self.current = Some((duration_ms, Vec::with_capacity(capacity)));
        self.rows = 0;
        Ok(())
    }

    pub(crate) fn write_row(&mut self, row: &[u8]) -> Result<()> {
        let Some((_, pixels)) = self.current.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "row",
                "write_row called outside a frame",
            ));
        };
        if row.len() != self.descriptor.row_bytes() || self.rows >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "row",
                "wrong length, or more rows than the canvas has",
            ));
        }
        pixels.extend_from_slice(row);
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
        if self.rows != self.descriptor.height {
            return Err(PixelsError::malformed(
                "jxl",
                format!(
                    "a frame ended after {} of {} rows",
                    self.rows, self.descriptor.height
                ),
            ));
        }
        let (width, height) = (self.descriptor.width, self.descriptor.height);
        let planned = match self.previous.as_deref() {
            None => Planned {
                crop: None,
                pixels: frame.clone(),
                duration_ms: duration,
            },
            Some(previous) => match self.changed(previous, &frame) {
                Some((_, _, w, h)) if (w, h) == (width, height) => Planned {
                    crop: None,
                    pixels: frame.clone(),
                    duration_ms: duration,
                },
                Some(rect) => Planned {
                    crop: Some(rect),
                    pixels: self.cut(&frame, rect),
                    duration_ms: duration,
                },
                None if !self.keep_duplicates => {
                    if let Some(pending) = self.pending.as_mut() {
                        pending.duration_ms = pending.duration_ms.saturating_add(duration);
                    }
                    return Ok(());
                }
                // A frame of its own that changes nothing: one pixel, put
                // back as it was.
                None => Planned {
                    crop: Some((0, 0, 1, 1)),
                    pixels: self.cut(&frame, (0, 0, 1, 1)),
                    duration_ms: duration,
                },
            },
        };
        if let Some(done) = self.pending.replace(planned) {
            self.code(&done, false)?;
        }
        self.previous = Some(frame);
        Ok(())
    }

    /// The bounding box of the pixels that differ between two frames.
    fn changed(&self, before: &[u8], after: &[u8]) -> Option<(u32, u32, u32, u32)> {
        let bytes = self.descriptor.pixel.bytes_per_pixel();
        let (width, height) = (self.descriptor.width, self.descriptor.height);
        let (mut left, mut top, mut right, mut bottom) = (u32::MAX, u32::MAX, 0, 0);
        for y in 0..height {
            for x in 0..width {
                let at = (y as usize * width as usize + x as usize) * bytes;
                if before.get(at..at + bytes) != after.get(at..at + bytes) {
                    left = left.min(x);
                    top = top.min(y);
                    right = right.max(x + 1);
                    bottom = bottom.max(y + 1);
                }
            }
        }
        (left < right).then(|| (left, top, right - left, bottom - top))
    }

    /// The samples of `rect` of a frame.
    fn cut(&self, frame: &[u8], (x, y, w, h): (u32, u32, u32, u32)) -> Vec<u8> {
        let bytes = self.descriptor.pixel.bytes_per_pixel();
        let stride = self.descriptor.row_bytes();
        let mut out = Vec::with_capacity(w as usize * h as usize * bytes);
        for row in y..y + h {
            let at = row as usize * stride + x as usize * bytes;
            out.extend_from_slice(frame.get(at..at + w as usize * bytes).unwrap_or_default());
        }
        out
    }

    /// Append `frame` to the codestream.
    fn code(&mut self, frame: &Planned, is_last: bool) -> Result<()> {
        otf_pixels_core::Cancel::checkpoint()?;
        let (width, height) = frame
            .crop
            .map_or((self.descriptor.width, self.descriptor.height), |c| {
                (c.2, c.3)
            });
        let descriptor = self.descriptor.resized(width, height)?;
        let spec = FrameSpec {
            crop: frame.crop.map(|(x, y, w, h)| (x as i32, y as i32, w, h)),
            source: SLOT,
            duration: frame.duration_ms,
            is_last,
            save_as_reference: if is_last { 0 } else { SLOT },
        };
        let bytes = match self.distance {
            Some(distance) => crate::encoder::lossy_frame(
                &descriptor,
                &frame.pixels,
                &self.image,
                distance,
                &spec,
            )?,
            None => crate::encoder::lossless_frame(&descriptor, &frame.pixels, &self.image, &spec)?,
        };
        self.out.extend_from_slice(&bytes);
        Ok(())
    }

    /// Code the last frame and hand over the codestream.
    pub(crate) fn finish(&mut self) -> Result<Vec<u8>> {
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
        self.code(&last, true)?;
        Ok(std::mem::take(&mut self.out))
    }
}

/// Write `codestream` to `sink`, as it is or in the container beside its
/// metadata.
pub(crate) fn deliver(
    codestream: &[u8],
    exif: Option<&[u8]>,
    xmp: Option<&[u8]>,
    sink: &mut dyn Sink,
) -> Result<()> {
    if exif.is_some() || xmp.is_some() {
        sink.write_all(&crate::encoder::contained(codestream, exif, xmp))?;
    } else {
        sink.write_all(codestream)?;
    }
    sink.flush()
}
