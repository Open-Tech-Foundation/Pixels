//! The GIF decoder.
//!
//! # Canvas, not image
//!
//! A GIF frame is not the image: it is a rectangle drawn onto a persistent
//! canvas, whose size comes from the logical screen descriptor. A frame can be
//! smaller than the canvas, offset within it, and partially transparent, and
//! the canvas retains whatever earlier frames left behind — subject to the
//! disposal method each declared.
//!
//! Getting that wrong produces images that look right on the first frame and
//! accumulate garbage after it, which is why disposal is modelled explicitly
//! rather than being treated as "clear between frames".
//!
//! # Memory
//!
//! GIF decode is **internally buffered**, at one canvas. Frames are
//! LZW-compressed as a unit and composited onto shared state, so there is no
//! row at which the canvas is final until the frame is complete. SPEC
//! §Formats says "yes (per frame)", and this is what that means: the canvas is
//! bounded by the image, not by the number of frames, and an animation of a
//! thousand frames costs the same as one.

use otf_pixels_compress::LzwDecoder;
use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Format, ImageDescriptor as CoreDescriptor, Limits,
    PixelFormat, PixelsError, Result, Source,
};

use crate::format::{
    Disposal, GraphicControl, ImageDescriptor, SIGNATURE_87A, SIGNATURE_89A, Screen,
    interlaced_pass_rows, interlaced_row, label, read_sub_blocks, skip_sub_blocks,
};

/// The largest sub-block chain accepted for one frame's pixel data.
///
/// LZW output is bounded separately by the frame's pixel count; this bounds
/// the *compressed* side, which a length-prefixed chain does not bound itself.
const MAX_COMPRESSED: usize = 64 * 1024 * 1024;

/// The largest extension payload retained. Comments and application blocks are
/// skipped rather than kept, so this only bounds the ones we read.
const MAX_EXTENSION: usize = 4096;

/// One decoded frame, with the animation metadata that came with it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Frame {
    /// The whole canvas after this frame is composited, as RGBA8.
    pub pixels: Vec<u8>,
    /// Canvas width.
    pub width: u32,
    /// Canvas height.
    pub height: u32,
    /// Delay before the next frame, in hundredths of a second.
    pub delay_centiseconds: u16,
    /// How this frame is disposed of before the next is drawn.
    pub disposal: Disposal,
}

/// Decodes a GIF stream.
///
/// [`Decoder`] presents the first frame; [`GifDecoder::next_frame`] walks the
/// rest. See the crate docs for why the split is where it is.
#[derive(Debug)]
pub struct GifDecoder<S: Source> {
    descriptor: CoreDescriptor,
    screen: Screen,
    source: Option<S>,
    /// The global colour table, if the stream carries one.
    global: Vec<[u8; 3]>,
    /// The canvas, RGBA8, persisting across frames.
    canvas: Vec<u8>,
    /// Whether the first frame has been composited yet.
    started: bool,
    /// Rows of the first frame already served through [`Decoder::read_row`].
    row: u32,
    /// Set when the trailer is reached, so `next_frame` stops.
    finished: bool,
    /// Pending disposal from the frame just drawn.
    pending: Option<Pending>,
    /// The RGBA the canvas reverts to under `Disposal::Background`.
    background: [u8; 4],
}

/// What the frame just drawn asked to happen before the next one.
#[derive(Debug)]
struct Pending {
    disposal: Disposal,
    area: ImageDescriptor,
    /// The rectangle as it was before the frame was drawn, for `Previous`.
    saved: Vec<u8>,
    /// Whether that frame declared a transparent index, which changes what
    /// `Background` means.
    had_transparency: bool,
}

impl<S: Source> GifDecoder<S> {
    /// Parse the header and logical screen descriptor, reading nothing more.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a bad signature or screen
    /// descriptor, or [`PixelsError::LimitExceeded`] if the canvas exceeds
    /// `limits`.
    pub fn new(mut source: S, limits: Limits) -> Result<Self> {
        let mut header = [0_u8; 13];
        source.read_exact(&mut header)?;

        let signature = header.get(..6).unwrap_or(&[]);
        if signature != SIGNATURE_87A && signature != SIGNATURE_89A {
            return Err(PixelsError::malformed(
                "gif",
                "signature is neither GIF87a nor GIF89a",
            ));
        }
        let mut screen_bytes = [0_u8; 7];
        screen_bytes.copy_from_slice(header.get(6..13).unwrap_or(&[0; 7]));
        let screen = Screen::parse(&screen_bytes)?;

        let width = u32::from(screen.width);
        let height = u32::from(screen.height);
        // Enforced before any buffer exists (SPEC §Safety).
        let descriptor = CoreDescriptor::with_limits(width, height, PixelFormat::Rgba8, &limits)?;

        let mut global = Vec::new();
        if screen.global_table_size > 0 {
            global = read_table(&mut source, screen.global_table_size)?;
        }

        let canvas_len = descriptor
            .byte_len()
            .ok_or_else(|| PixelsError::malformed("gif", "canvas size overflows"))?;

        // "Restore to background colour" means the entry the logical screen
        // descriptor names, opaque. A stream with no global table has no
        // background colour to restore to, so transparent is the only
        // available answer.
        let background = global
            .get(screen.background as usize)
            .map_or([0, 0, 0, 0], |c| [c[0], c[1], c[2], 255]);

        Ok(Self {
            descriptor,
            screen,
            source: Some(source),
            global,
            // A GIF canvas begins fully transparent, which is what makes a
            // first frame smaller than the canvas render correctly.
            canvas: vec![0_u8; canvas_len],
            started: false,
            row: 0,
            finished: false,
            pending: None,
            background,
        })
    }

    /// The logical screen descriptor.
    #[must_use]
    pub const fn screen(&self) -> Screen {
        self.screen
    }

    /// Decode the next frame, compositing it onto the canvas.
    ///
    /// Returns `None` once the stream's trailer is reached. Each frame carries
    /// the whole canvas, because that is what a viewer draws — a frame's own
    /// rectangle is meaningless without what it was composited onto.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a malformed block or a frame
    /// that does not fit its canvas, or [`PixelsError::Io`] on source failure.
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.finished {
            return Ok(None);
        }
        let Some(mut source) = self.source.take() else {
            return Ok(None);
        };
        let result = self.decode_next(&mut source);
        self.source = Some(source);
        result
    }

    /// The body of [`GifDecoder::next_frame`], with the source borrowed out.
    fn decode_next(&mut self, source: &mut S) -> Result<Option<Frame>> {
        // Apply the previous frame's disposal before drawing this one. It
        // happens here rather than after drawing because `Previous` needs the
        // saved rectangle, and saving it is only worth doing if a later frame
        // actually arrives.
        self.apply_disposal();

        let mut control = GraphicControl::default();
        loop {
            let mut marker = [0_u8; 1];
            source.read_exact(&mut marker)?;
            match marker[0] {
                label::TRAILER => {
                    self.finished = true;
                    return Ok(None);
                }
                label::EXTENSION => {
                    let mut kind = [0_u8; 1];
                    source.read_exact(&mut kind)?;
                    match kind[0] {
                        label::GRAPHIC_CONTROL => {
                            let payload = read_sub_blocks(source, MAX_EXTENSION)?;
                            control = GraphicControl::parse(&payload);
                        }
                        // Comments, plain text and application blocks carry no
                        // pixels. Skipping them is required, not optional: an
                        // unknown extension must not be an error (§Appendix A).
                        _ => skip_sub_blocks(source)?,
                    }
                }
                label::IMAGE => {
                    let frame = self.decode_image(source, control)?;
                    return Ok(Some(frame));
                }
                other => {
                    return Err(PixelsError::malformed(
                        "gif",
                        format!("unknown block label {other:#04x}"),
                    ));
                }
            }
        }
    }

    /// Undo the previous frame according to its declared disposal.
    fn apply_disposal(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        match pending.disposal {
            // `None` is "unspecified", which viewers treat as leaving the
            // frame in place. Clearing here would break the overwhelmingly
            // common case of an optimised animation drawing only what changed.
            Disposal::None | Disposal::Keep => {}
            Disposal::Background => {
                // §23.c.iv says "restore to background colour", and libgif
                // does exactly that. Browsers instead restore to transparent,
                // which is what an animation with a transparent index is
                // authored against — so the frame's own transparency decides
                // which reading applies. Picking one unconditionally makes
                // one large class of real animations render wrongly.
                let fill = if pending.had_transparency {
                    [0, 0, 0, 0]
                } else {
                    self.background
                };
                self.fill_area(pending.area, &fill);
            }
            Disposal::Previous => {
                self.restore_area(pending.area, &pending.saved);
            }
        }
    }

    /// Fill a frame's rectangle with one RGBA value.
    fn fill_area(&mut self, area: ImageDescriptor, value: &[u8; 4]) {
        let width = self.descriptor.width;
        let height = self.descriptor.height;
        for y in u32::from(area.top)..u32::from(area.top) + u32::from(area.height) {
            if y >= height {
                break;
            }
            for x in u32::from(area.left)..u32::from(area.left) + u32::from(area.width) {
                if x >= width {
                    break;
                }
                let at = ((y * width + x) * 4) as usize;
                if let Some(slot) = self.canvas.get_mut(at..at + 4) {
                    slot.copy_from_slice(value);
                }
            }
        }
    }

    /// Restore a frame's rectangle from a saved copy.
    fn restore_area(&mut self, area: ImageDescriptor, saved: &[u8]) {
        let width = self.descriptor.width;
        let height = self.descriptor.height;
        let area_width = u32::from(area.width) as usize;
        for row in 0..u32::from(area.height) {
            let y = u32::from(area.top) + row;
            if y >= height {
                break;
            }
            for column in 0..u32::from(area.width) {
                let x = u32::from(area.left) + column;
                if x >= width {
                    break;
                }
                let from = ((row as usize * area_width) + column as usize) * 4;
                let to = ((y * width + x) * 4) as usize;
                let (Some(source), Some(target)) =
                    (saved.get(from..from + 4), self.canvas.get_mut(to..to + 4))
                else {
                    continue;
                };
                target.copy_from_slice(source);
            }
        }
    }

    /// Decode one image block and composite it onto the canvas.
    fn decode_image(&mut self, source: &mut S, control: GraphicControl) -> Result<Frame> {
        let mut bytes = [0_u8; 9];
        source.read_exact(&mut bytes)?;
        let image = ImageDescriptor::parse(&bytes)?;

        // A frame must lie within the canvas. Some encoders emit frames that
        // overhang, and viewers clip; rejecting would fail files that display
        // fine, so the compositing loops clip instead.
        let local = if image.local_table_size > 0 {
            read_table(source, image.local_table_size)?
        } else {
            Vec::new()
        };
        // Cloned rather than borrowed: compositing takes `&mut self`, and a
        // 256-entry table is 768 bytes once per frame.
        let palette: Vec<[u8; 3]> = if local.is_empty() {
            self.global.clone()
        } else {
            local
        };
        if palette.is_empty() {
            return Err(PixelsError::malformed(
                "gif",
                "frame has neither a local nor a global colour table",
            ));
        }

        let mut minimum_width = [0_u8; 1];
        source.read_exact(&mut minimum_width)?;
        let compressed = read_sub_blocks(source, MAX_COMPRESSED)?;

        let pixels = u32::from(image.width) as usize * u32::from(image.height) as usize;
        let decoder =
            LzwDecoder::gif(u32::from(minimum_width[0])).map_err(crate::compress_error)?;
        // The limit is the frame's exact pixel count, which is what makes an
        // LZW bomb a malformed-input error rather than an allocation.
        let indices = decoder
            .decode(&compressed, pixels)
            .map_err(crate::compress_error)?;

        // Save the rectangle *before* drawing, for a later `Previous`.
        let saved = if control.disposal == Disposal::Previous {
            self.save_area(image)
        } else {
            Vec::new()
        };

        self.composite(image, &indices, &palette, control.transparent);
        self.pending = Some(Pending {
            disposal: control.disposal,
            area: image,
            saved,
            had_transparency: control.transparent.is_some(),
        });
        self.started = true;

        Ok(Frame {
            pixels: self.canvas.clone(),
            width: self.descriptor.width,
            height: self.descriptor.height,
            delay_centiseconds: control.delay_centiseconds,
            disposal: control.disposal,
        })
    }

    /// Copy a rectangle of the canvas, for `Disposal::Previous`.
    fn save_area(&self, area: ImageDescriptor) -> Vec<u8> {
        let width = self.descriptor.width;
        let height = self.descriptor.height;
        let mut out =
            vec![0_u8; u32::from(area.width) as usize * u32::from(area.height) as usize * 4];
        let area_width = u32::from(area.width) as usize;
        for row in 0..u32::from(area.height) {
            let y = u32::from(area.top) + row;
            if y >= height {
                break;
            }
            for column in 0..u32::from(area.width) {
                let x = u32::from(area.left) + column;
                if x >= width {
                    break;
                }
                let from = ((y * width + x) * 4) as usize;
                let to = ((row as usize * area_width) + column as usize) * 4;
                let (Some(source), Some(target)) =
                    (self.canvas.get(from..from + 4), out.get_mut(to..to + 4))
                else {
                    continue;
                };
                target.copy_from_slice(source);
            }
        }
        out
    }

    /// Draw palette indices onto the canvas, honouring transparency.
    fn composite(
        &mut self,
        image: ImageDescriptor,
        indices: &[u8],
        palette: &[[u8; 3]],
        transparent: Option<u8>,
    ) {
        let canvas_width = self.descriptor.width;
        let canvas_height = self.descriptor.height;
        let frame_width = u32::from(image.width);
        let frame_height = u32::from(image.height);

        for source_row in 0..frame_height {
            // Interlaced frames store rows out of order; the mapping is GIF's
            // four-pass row interlace, which is not PNG's Adam7.
            let target_row = if image.interlaced {
                match deinterlace(source_row, frame_height) {
                    Some(row) => row,
                    None => continue,
                }
            } else {
                source_row
            };
            let y = u32::from(image.top) + target_row;
            if y >= canvas_height {
                continue;
            }

            for column in 0..frame_width {
                let x = u32::from(image.left) + column;
                if x >= canvas_width {
                    continue;
                }
                let index = (source_row * frame_width + column) as usize;
                let Some(&entry) = indices.get(index) else {
                    // A short LZW stream leaves the rest of the frame
                    // untouched, which is what viewers show. Truncating to an
                    // error would reject files that display.
                    continue;
                };
                if Some(entry) == transparent {
                    // Transparent pixels let the canvas show through — the
                    // whole point of frame-to-frame optimisation.
                    continue;
                }
                let colour = palette.get(entry as usize).copied().unwrap_or([0, 0, 0]);
                let at = ((y * canvas_width + x) * 4) as usize;
                if let Some(slot) = self.canvas.get_mut(at..at + 4) {
                    slot.copy_from_slice(&[colour[0], colour[1], colour[2], 255]);
                }
            }
        }
    }

    /// Ensure the first frame has been decoded, for the [`Decoder`] path.
    fn ensure_started(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        match self.next_frame()? {
            Some(_) => Ok(()),
            None => Err(PixelsError::malformed("gif", "stream contains no frames")),
        }
    }
}

/// Map a stored row index to its position in an interlaced frame.
fn deinterlace(stored: u32, height: u32) -> Option<u32> {
    let mut seen = 0;
    for pass in 0..4 {
        let rows = interlaced_pass_rows(pass, height);
        if stored < seen + rows {
            return interlaced_row(pass, stored - seen, height);
        }
        seen += rows;
    }
    None
}

/// Read a colour table of `entries` RGB triples.
fn read_table<S: Source>(source: &mut S, entries: usize) -> Result<Vec<[u8; 3]>> {
    let mut bytes = vec![0_u8; entries * 3];
    source.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(3)
        .map(|rgb| {
            [
                rgb.first().copied().unwrap_or(0),
                rgb.get(1).copied().unwrap_or(0),
                rgb.get(2).copied().unwrap_or(0),
            ]
        })
        .collect())
}

impl<S: Source + std::fmt::Debug> Decoder for GifDecoder<S> {
    fn descriptor(&self) -> CoreDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        self.ensure_started()?;
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
            .canvas
            .get(start..start + row_bytes)
            .ok_or_else(|| PixelsError::malformed("gif", "canvas is short"))?;
        out.copy_from_slice(row);
        self.row += 1;
        Ok(())
    }
}

/// Whether `prefix` starts with a GIF signature.
///
/// Detection is by magic bytes only (SPEC §Formats).
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    let head = prefix.get(..6);
    head == Some(&SIGNATURE_87A[..]) || head == Some(&SIGNATURE_89A[..])
}

/// The GIF entry in a sniffing registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct GifCodec;

impl Codec for GifCodec {
    fn format(&self) -> Format {
        Format::Gif
    }

    fn magic_len(&self) -> usize {
        6
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
    }
}
