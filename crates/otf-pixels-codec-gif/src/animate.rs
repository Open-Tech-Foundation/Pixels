//! Animated GIF encoding (ADR-0020): frames reduced to what changed.
//!
//! # What a frame holds
//!
//! A GIF frame is a rectangle drawn over what the previous frame left on
//! screen. Each frame here is the bounding box of the pixels that changed,
//! and inside it a pixel that did not change is the transparent index, so
//! the one beneath shows through — a frame of an animation that barely moves
//! is a few bytes. "Changed" is sharp's `interFrameMaxError`: any channel
//! further than that from what is on screen.
//!
//! # Transparency
//!
//! GIF transparency is one index: a pixel is drawn or not. A pixel with
//! alpha under one half is transparent, any other is opaque. A frame cannot
//! make a pixel transparent that an earlier one drew; only the earlier
//! frame's *disposal* can, by restoring its rectangle to the background
//! (transparent, as every browser reads it) once it has been shown. So
//! frames are written one behind: when frame `k + 1` arrives and needs a
//! pixel cleared, frame `k`, not yet written, takes "restore to background"
//! and grows to cover that pixel.
//!
//! # Palettes
//!
//! The first frame's palette is the global colour table. A later frame uses
//! it too unless mapping onto it costs more than sharp's
//! `interPaletteMaxError` over the frame's own palette, in which case it
//! carries a local table. Each palette holds `colours - 1` colours and the
//! transparent index.
//!
//! # Time
//!
//! GIF delays are hundredths of a second. Each frame's delay is rounded so
//! that the animation's running total is, rather than each frame on its
//! own: 33 ms frames come out as 3, 4, 3 centiseconds, not 3, 3, 3.

use otf_pixels_compress::{LzwEncoder, Palette, build_palette};
use otf_pixels_core::{ImageDescriptor, PixelFormat, PixelsError, Result, Sink};

use crate::format::{SIGNATURE_89A, label, write_sub_blocks};

/// A rectangle on the canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl Rect {
    fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

/// The bounding box of the pixels for which `hit` holds.
fn bounds(width: u32, height: u32, mut hit: impl FnMut(usize) -> bool) -> Option<Rect> {
    let (mut left, mut top, mut right, mut bottom) = (u32::MAX, u32::MAX, 0, 0);
    for y in 0..height {
        for x in 0..width {
            if hit((y * width + x) as usize) {
                left = left.min(x);
                top = top.min(y);
                right = right.max(x + 1);
                bottom = bottom.max(y + 1);
            }
        }
    }
    (left < right).then(|| Rect {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    })
}

/// A frame decided but not yet written.
#[derive(Debug)]
struct Planned {
    rect: Rect,
    /// The whole canvas, RGBA: opaque where the frame draws, alpha 0 where
    /// it does not.
    draw: Vec<u8>,
    duration_ms: u32,
    dispose_background: bool,
}

/// The state of one animated GIF being written.
#[derive(Debug)]
pub(crate) struct Animator {
    width: u32,
    height: u32,
    format: PixelFormat,
    loop_count: u32,
    colours: usize,
    dither: f32,
    frame_error: f32,
    palette_error: f32,
    keep_duplicates: bool,
    /// What is on screen once the planned frames are shown, RGBA, with
    /// transparent pixels all zero.
    shown: Vec<u8>,
    pending: Option<Planned>,
    /// The frame being received: its duration and RGBA.
    current: Option<(u32, Vec<u8>)>,
    rows: u32,
    /// The global colour table's colours, once the first frame is written.
    global: Option<Palette>,
    /// Running totals, for rounding delays.
    elapsed_ms: u64,
    written_cs: u64,
    /// The bytes not yet handed to the sink: everything, until the header
    /// can be written.
    frames_written: u32,
}

/// Options carried from the encoder.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Settings {
    pub colours: usize,
    pub dither: f32,
    pub frame_error: f32,
    pub palette_error: f32,
    pub keep_duplicates: bool,
}

impl Animator {
    pub(crate) fn new(descriptor: &ImageDescriptor, loop_count: u32, settings: Settings) -> Self {
        let pixels = descriptor.width as usize * descriptor.height as usize;
        Self {
            width: descriptor.width,
            height: descriptor.height,
            format: descriptor.pixel,
            loop_count,
            colours: settings.colours.clamp(2, 256),
            dither: settings.dither,
            frame_error: settings.frame_error,
            palette_error: settings.palette_error,
            keep_duplicates: settings.keep_duplicates,
            shown: vec![0; pixels * 4],
            pending: None,
            current: None,
            rows: 0,
            global: None,
            elapsed_ms: 0,
            written_cs: 0,
            frames_written: 0,
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
        let expected = self.width as usize * self.format.channels();
        let Some((_, rgba)) = self.current.as_mut() else {
            return Err(PixelsError::invalid_argument(
                "row",
                "write_row called outside a frame",
            ));
        };
        if row.len() != expected || self.rows >= self.height {
            return Err(PixelsError::invalid_argument(
                "row",
                "wrong length, or more rows than the canvas has",
            ));
        }
        append_rgba(row, self.format, rgba)?;
        self.rows += 1;
        Ok(())
    }

    pub(crate) fn end_frame(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some((duration, target)) = self.current.take() else {
            return Err(PixelsError::invalid_argument(
                "frame",
                "end_frame called with no frame open",
            ));
        };
        if self.rows != self.height {
            return Err(PixelsError::malformed(
                "gif",
                format!("a frame ended after {} of {} rows", self.rows, self.height),
            ));
        }
        let (width, height) = (self.width, self.height);
        let opaque = |pixels: &[u8], i: usize| pixels.get(i * 4 + 3).is_some_and(|&a| a != 0);

        let Some(mut previous) = self.pending.take() else {
            let mut base = std::mem::take(&mut self.shown);
            self.pending = Some(self.plan(&target, &mut base, duration));
            self.shown = base;
            return Ok(());
        };

        // Pixels this frame leaves transparent that are drawn on screen: only
        // the previous frame's disposal can clear them.
        let clear = bounds(width, height, |i| {
            !opaque(&target, i) && opaque(&self.shown, i)
        });
        let mut base = self.shown.clone();
        if let Some(clear) = clear {
            previous.rect = previous.rect.union(clear);
            previous.dispose_background = true;
            fill(&mut base, width, previous.rect, &[0, 0, 0, 0]);
        }
        let unchanged =
            clear.is_none() && bounds(width, height, |i| self.changed(&target, &base, i)).is_none();
        if unchanged && !self.keep_duplicates {
            // The same picture again: shown for longer, not twice.
            previous.duration_ms = previous.duration_ms.saturating_add(duration);
            self.pending = Some(previous);
            return Ok(());
        }
        self.write_frame(&previous, sink)?;
        self.pending = Some(self.plan(&target, &mut base, duration));
        self.shown = base;
        Ok(())
    }

    pub(crate) fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
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
        self.write_frame(&last, sink)?;
        sink.write_all(&[label::TRAILER])?;
        sink.flush()
    }

    /// Whether pixel `i` of `target` differs from `base` by more than the
    /// inter-frame error.
    fn changed(&self, target: &[u8], base: &[u8], i: usize) -> bool {
        let (Some(&[tr, tg, tb, ta]), Some(&[br, bg, bb, ba])) =
            (target.get(i * 4..i * 4 + 4), base.get(i * 4..i * 4 + 4))
        else {
            return false;
        };
        match (ta != 0, ba != 0) {
            (false, false) => false,
            (true, true) => [(tr, br), (tg, bg), (tb, bb)]
                .iter()
                .any(|&(a, b)| f32::from(a.abs_diff(b)) > self.frame_error),
            _ => true,
        }
    }

    /// The frame that turns `base` into `target`, and `base` updated to
    /// what is then on screen.
    fn plan(&self, target: &[u8], base: &mut [u8], duration_ms: u32) -> Planned {
        let (width, height) = (self.width, self.height);
        let rect = bounds(width, height, |i| self.changed(target, base, i)).unwrap_or(Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        let mut draw = vec![0_u8; base.len()];
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let i = (y * width + x) as usize;
                if self.changed(target, base, i) {
                    let span = i * 4..i * 4 + 4;
                    if let (Some(to), Some(from)) =
                        (draw.get_mut(span.clone()), target.get(span.clone()))
                    {
                        to.copy_from_slice(from);
                    }
                    if let (Some(to), Some(from)) = (base.get_mut(span.clone()), target.get(span)) {
                        to.copy_from_slice(from);
                    }
                }
            }
        }
        Planned {
            rect,
            draw,
            duration_ms,
            dispose_background: false,
        }
    }

    /// Write `frame`, and the file's header before the first.
    fn write_frame(&mut self, frame: &Planned, sink: &mut dyn Sink) -> Result<()> {
        otf_pixels_core::Cancel::checkpoint()?;
        let rect = frame.rect;
        let stride = self.width as usize * 4;
        // The rectangle's pixels, and which of them are drawn.
        let mut rgb = Vec::with_capacity(rect.width as usize * rect.height as usize * 3);
        let mut drawn = Vec::with_capacity(rect.width as usize * rect.height as usize);
        for y in rect.y..rect.y + rect.height {
            for x in rect.x..rect.x + rect.width {
                let at = y as usize * stride + x as usize * 4;
                let pixel = frame.draw.get(at..at + 4).unwrap_or(&[0; 4]);
                rgb.extend_from_slice(pixel.get(..3).unwrap_or(&[0; 3]));
                drawn.push(pixel.get(3).is_some_and(|&a| a != 0));
            }
        }
        let colours: Vec<u8> = rgb
            .chunks_exact(3)
            .zip(&drawn)
            .filter(|(_, d)| **d)
            .flat_map(|(c, _)| c.iter().copied())
            .collect();
        let local = if colours.is_empty() {
            Palette::new(vec![[0, 0, 0]])
        } else {
            build_palette(&colours, self.colours - 1)
        };

        let mut out = Vec::new();
        let palette = match &self.global {
            None => {
                self.write_header(&local, &mut out);
                self.global = Some(local.clone());
                None
            }
            Some(global) => {
                let extra = rms(&colours, global) - rms(&colours, &local);
                (extra > self.palette_error).then_some(local)
            }
        };
        let used = palette
            .as_ref()
            .or(self.global.as_ref())
            .ok_or_else(|| PixelsError::graph("gif: no palette to draw with"))?;
        let transparent = used.len() as u8;
        let table = with_transparent(used);

        let mut indices = dither(&rgb, &drawn, rect.width as usize, used, self.dither);
        for (index, &d) in indices.iter_mut().zip(&drawn) {
            if !d {
                *index = transparent;
            }
        }

        // Graphic control: disposal, transparency, delay.
        self.elapsed_ms += u64::from(frame.duration_ms);
        let target_cs = (self.elapsed_ms + 5) / 10;
        let delay = (target_cs - self.written_cs).min(u64::from(u16::MAX));
        self.written_cs += delay;
        let disposal = if frame.dispose_background { 2 } else { 1 };
        out.extend_from_slice(&[label::EXTENSION, label::GRAPHIC_CONTROL, 4]);
        out.push((disposal << 2) | 1);
        out.extend_from_slice(&(delay as u16).to_le_bytes());
        out.extend_from_slice(&[transparent, 0]);

        out.push(label::IMAGE);
        for v in [rect.x, rect.y, rect.width, rect.height] {
            out.extend_from_slice(&(v as u16).to_le_bytes());
        }
        let bits = table.code_bits();
        match &palette {
            Some(_) => {
                out.push(0x80 | ((bits - 1) as u8));
                for entry in table.padded() {
                    out.extend_from_slice(&entry);
                }
            }
            None => out.push(0),
        }
        let minimum_width = bits.max(2);
        out.push(minimum_width as u8);
        let compressed = LzwEncoder::gif(minimum_width)
            .map_err(crate::compress_error)?
            .encode(&indices);
        write_sub_blocks(&mut out, &compressed);
        sink.write_all(&out)?;
        self.frames_written += 1;
        Ok(())
    }

    /// The signature, logical screen, global table and loop count.
    fn write_header(&self, global: &Palette, out: &mut Vec<u8>) {
        let table = with_transparent(global);
        let bits = table.code_bits();
        out.extend_from_slice(&SIGNATURE_89A);
        out.extend_from_slice(&(self.width as u16).to_le_bytes());
        out.extend_from_slice(&(self.height as u16).to_le_bytes());
        out.push(0x80 | 0x70 | ((bits - 1) as u8));
        // The background is the transparent index: what a restored area is.
        out.push(global.len() as u8);
        out.push(0);
        for entry in table.padded() {
            out.extend_from_slice(&entry);
        }
        // NETSCAPE2.0: the loop count, 0 for forever, as it was read.
        out.extend_from_slice(&[label::EXTENSION, 0xff, 11]);
        out.extend_from_slice(b"NETSCAPE2.0");
        out.extend_from_slice(&[3, 1]);
        out.extend_from_slice(&(self.loop_count.min(u32::from(u16::MAX)) as u16).to_le_bytes());
        out.push(0);
    }
}

/// `palette` with the transparent index after its colours.
fn with_transparent(palette: &Palette) -> Palette {
    let mut entries = palette.entries().to_vec();
    entries.push([0, 0, 0]);
    Palette::new(entries)
}

/// Root-mean-square distance from each colour to its nearest entry.
fn rms(colours: &[u8], palette: &Palette) -> f32 {
    let mut total = 0_f64;
    let mut count = 0_u64;
    for c in colours.chunks_exact(3) {
        let &[r, g, b] = c else { continue };
        let colour = [i32::from(r), i32::from(g), i32::from(b)];
        let entry = palette
            .entries()
            .get(usize::from(palette.nearest(colour)))
            .copied()
            .unwrap_or([0; 3]);
        let distance: i32 = colour
            .iter()
            .zip(entry)
            .map(|(&a, b)| (a - i32::from(b)).pow(2))
            .sum();
        total += f64::from(distance);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64).sqrt() as f32
    }
}

/// Floyd-Steinberg onto `palette` over the drawn pixels only: a pixel left
/// transparent neither takes nor passes on error.
fn dither(rgb: &[u8], drawn: &[bool], width: usize, palette: &Palette, strength: f32) -> Vec<u8> {
    let count = drawn.len();
    let mut out = vec![0_u8; count];
    let mut error = vec![[0_f32; 3]; count];
    let strength = strength.clamp(0.0, 1.0);
    for i in 0..count {
        if !drawn.get(i).copied().unwrap_or(false) {
            continue;
        }
        let pixel = rgb.get(i * 3..i * 3 + 3).unwrap_or(&[0; 3]);
        let carried = error.get(i).copied().unwrap_or([0.0; 3]);
        let mut wanted = [0_f32; 3];
        for ((w, &p), e) in wanted.iter_mut().zip(pixel).zip(&carried) {
            *w = f32::from(p) + e * strength;
        }
        let colour = wanted.map(|v| v.round().clamp(0.0, 255.0) as i32);
        let index = palette.nearest(colour);
        if let Some(slot) = out.get_mut(i) {
            *slot = index;
        }
        if strength == 0.0 {
            continue;
        }
        let entry = palette
            .entries()
            .get(usize::from(index))
            .copied()
            .unwrap_or([0; 3]);
        let mut residual = [0_f32; 3];
        for ((r, &w), &e) in residual.iter_mut().zip(&wanted).zip(&entry) {
            *r = w - f32::from(e);
        }
        let (x, last_column) = (i % width, i % width + 1 == width);
        let mut spread = |at: Option<usize>, weight: f32| {
            if let Some(j) = at.filter(|&j| drawn.get(j).copied().unwrap_or(false)) {
                if let Some(e) = error.get_mut(j) {
                    for (e, r) in e.iter_mut().zip(&residual) {
                        *e += r * weight;
                    }
                }
            }
        };
        spread((!last_column).then_some(i + 1), 7.0 / 16.0);
        spread((x > 0).then(|| i + width - 1), 3.0 / 16.0);
        spread(Some(i + width), 5.0 / 16.0);
        spread((!last_column).then_some(i + width + 1), 1.0 / 16.0);
    }
    out
}

/// Fill `rect` of an RGBA canvas `width` wide with `value`.
fn fill(canvas: &mut [u8], width: u32, rect: Rect, value: &[u8; 4]) {
    for y in rect.y..rect.y + rect.height {
        for x in rect.x..rect.x + rect.width {
            let at = (y * width + x) as usize * 4;
            if let Some(slot) = canvas.get_mut(at..at + 4) {
                slot.copy_from_slice(value);
            }
        }
    }
}

/// One row of `format` as RGBA, transparent (alpha under one half) as all
/// zero and everything else opaque.
fn append_rgba(row: &[u8], format: PixelFormat, out: &mut Vec<u8>) -> Result<()> {
    let push = |out: &mut Vec<u8>, r: u8, g: u8, b: u8, a: u8| {
        if a < 128 {
            out.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            out.extend_from_slice(&[r, g, b, 255]);
        }
    };
    match format {
        PixelFormat::Rgb8 => {
            for p in row.chunks_exact(3) {
                if let &[r, g, b] = p {
                    push(out, r, g, b, 255);
                }
            }
        }
        PixelFormat::Rgba8 => {
            for p in row.chunks_exact(4) {
                if let &[r, g, b, a] = p {
                    push(out, r, g, b, a);
                }
            }
        }
        PixelFormat::Gray8 => row.iter().for_each(|&v| push(out, v, v, v, 255)),
        PixelFormat::GrayA8 => {
            for p in row.chunks_exact(2) {
                if let &[v, a] = p {
                    push(out, v, v, v, a);
                }
            }
        }
        other => {
            return Err(PixelsError::unsupported(format!(
                "GIF encoding needs an 8-bit format; got {other}. Convert first."
            )));
        }
    }
    Ok(())
}
