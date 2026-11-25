//! Test doubles for exercising graph and evaluator contracts.
//!
//! Enabled by the `testing` feature (and always within this crate's own
//! tests). These are deliberately trivial: they let a test assert *when* the
//! engine pulls pixels and *in what order* it runs ops, without dragging a
//! real codec or kernel into the assertion.
//!
//! This module is not part of the stable API surface.

use crate::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Producer, Region, Result, Tile, TileMut,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A producer that counts how many times it was pulled.
///
/// The counter is the laziness proof: it must read zero after any amount of
/// graph construction and metadata querying.
#[derive(Debug)]
pub struct CountingProducer {
    descriptor: ImageDescriptor,
    calls: AtomicUsize,
}

impl CountingProducer {
    /// A producer of all-zero pixels shaped like `descriptor`.
    #[must_use]
    pub const fn new(descriptor: ImageDescriptor) -> Self {
        Self {
            descriptor,
            calls: AtomicUsize::new(0),
        }
    }

    /// How many times [`Producer::produce`] has been called.
    #[must_use]
    pub fn produce_calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Producer for CountingProducer {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn produce(&self, _region: Region, _output: &mut TileMut<'_>) -> Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A producer whose pixels are a predictable ramp: byte `n` holds `n as u8`.
///
/// Distinguishable per position, so a geometry op that transposes or mirrors
/// rows produces a visibly wrong answer rather than an accidentally right one.
#[derive(Debug)]
pub struct RampProducer {
    descriptor: ImageDescriptor,
}

impl RampProducer {
    /// A ramp producer shaped like `descriptor`.
    #[must_use]
    pub const fn new(descriptor: ImageDescriptor) -> Self {
        Self { descriptor }
    }
}

impl Producer for RampProducer {
    fn name(&self) -> &'static str {
        "ramp"
    }

    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn produce(&self, region: Region, output: &mut TileMut<'_>) -> Result<()> {
        let bpp = self.descriptor.pixel.bytes_per_pixel();
        let row_bytes = self.descriptor.row_bytes();
        for y in region.y..region.y.saturating_add(region.height) {
            let start = (region.x - output.region().x) as usize * bpp;
            let row = output
                .row_mut(y)
                .ok_or_else(|| PixelsError::invalid_argument("output", "missing row"))?;
            for x in 0..region.width as usize * bpp {
                let index = start + x;
                let value = (y as usize * row_bytes + region.x as usize * bpp + x) as u8;
                let cell = row
                    .get_mut(index)
                    .ok_or_else(|| PixelsError::invalid_argument("output", "row too short"))?;
                *cell = value;
            }
        }
        Ok(())
    }
}

/// An op that fills its output with a constant, ignoring its input.
#[derive(Debug)]
pub struct ConstantOp {
    value: u8,
}

impl ConstantOp {
    /// An op that writes `value` into every output byte.
    #[must_use]
    pub const fn new(value: u8) -> Self {
        Self { value }
    }
}

impl Op for ConstantOp {
    fn name(&self) -> &'static str {
        "constant"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        inputs
            .first()
            .copied()
            .ok_or_else(|| PixelsError::graph("constant needs one input"))
    }

    fn input_regions(&self, output: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    fn compute(&self, _: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        for row in output.rows_mut() {
            row.fill(self.value);
        }
        Ok(())
    }
}

/// A two-input op that adds its inputs, saturating.
///
/// Exists so tests can join two graph branches and prove the shared prefix
/// below them was evaluated once.
#[derive(Debug)]
pub struct SumOp;

impl Op for SumOp {
    fn name(&self) -> &'static str {
        "sum"
    }

    fn arity(&self) -> usize {
        2
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let (Some(a), Some(b)) = (inputs.first(), inputs.get(1)) else {
            return Err(PixelsError::graph("sum needs two inputs"));
        };
        if a.width != b.width || a.height != b.height || a.pixel != b.pixel {
            return Err(PixelsError::invalid_argument(
                "inputs",
                "sum needs matching inputs",
            ));
        }
        Ok(*a)
    }

    fn input_regions(&self, output: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output, output])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let (Some(a), Some(b)) = (inputs.first(), inputs.get(1)) else {
            return Err(PixelsError::graph("sum needs two input tiles"));
        };
        let region = output.region();
        for y in region.y..region.y.saturating_add(region.height) {
            let (Some(left), Some(right)) = (a.row(y), b.row(y)) else {
                return Err(PixelsError::graph("input tile is missing a row"));
            };
            let Some(out) = output.row_mut(y) else {
                return Err(PixelsError::graph("output tile is missing a row"));
            };
            for (cell, (l, r)) in out.iter_mut().zip(left.iter().zip(right)) {
                *cell = l.saturating_add(*r);
            }
        }
        Ok(())
    }
}

/// An op that always fails, to prove failures abort the whole pipeline.
#[derive(Debug)]
pub struct FailingOp;

impl Op for FailingOp {
    fn name(&self) -> &'static str {
        "failing"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        inputs
            .first()
            .copied()
            .ok_or_else(|| PixelsError::graph("failing needs one input"))
    }

    fn input_regions(&self, output: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
        Err(PixelsError::malformed("testing", "this op always fails"))
    }
}

/// An animated decoder with no format behind it: `frames` frames of
/// `descriptor`'s shape, every byte of frame *k* equal to `k`, each frame
/// showing for `(k + 1) * 10` ms.
///
/// It implements [`crate::Decoder::advance_frame`] exactly as the trait
/// describes, so a pipeline or encoder test can run the frame protocol
/// without a real codec.
#[derive(Debug)]
pub struct FrameStubDecoder {
    descriptor: ImageDescriptor,
    frames: u32,
    frame: u32,
    row: u32,
}

impl FrameStubDecoder {
    /// An animation of `frames` frames, at least 1.
    #[must_use]
    pub const fn new(descriptor: ImageDescriptor, frames: u32) -> Self {
        Self {
            descriptor,
            frames: if frames == 0 { 1 } else { frames },
            frame: 0,
            row: 0,
        }
    }

    /// The duration this stub gives frame `index`.
    #[must_use]
    pub const fn duration_of(index: u32) -> u32 {
        (index + 1) * 10
    }
}

impl crate::Decoder for FrameStubDecoder {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn animation(&self) -> Option<crate::Animation> {
        crate::Animation::new((0..self.frames).map(Self::duration_of).collect(), 0)
    }

    fn advance_frame(&mut self) -> Result<Option<crate::FrameInfo>> {
        if self.frame + 1 >= self.frames {
            // Exhausted: no rows remain either.
            self.row = self.descriptor.height;
            return Ok(None);
        }
        self.frame += 1;
        self.row = 0;
        Ok(Some(crate::FrameInfo::new(
            self.frame,
            Self::duration_of(self.frame),
        )))
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        if out.len() != self.descriptor.row_bytes() {
            return Err(PixelsError::invalid_argument(
                "out",
                "row buffer is the wrong length",
            ));
        }
        if self.row >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                "every row of this frame has been read",
            ));
        }
        self.row += 1;
        out.fill(self.frame as u8);
        Ok(())
    }
}

/// Every frame `decoder` yields, from the current one on, with its timing:
/// frame 0's from [`crate::Decoder::animation`] (or 0 ms for a still).
///
/// # Errors
///
/// As [`crate::Decoder::read_row`] and [`crate::Decoder::advance_frame`].
pub fn read_frames(decoder: &mut dyn crate::Decoder) -> Result<Vec<(crate::FrameInfo, Vec<u8>)>> {
    let descriptor = decoder.descriptor();
    let first = decoder
        .animation()
        .and_then(|a| a.frame_durations_ms.first().copied())
        .unwrap_or(0);
    let mut info = crate::FrameInfo::new(0, first);
    let mut frames = Vec::new();
    loop {
        let len = descriptor
            .byte_len()
            .ok_or_else(|| PixelsError::graph("the canvas does not fit in memory"))?;
        let mut pixels = vec![0; len];
        for row in pixels.chunks_exact_mut(descriptor.row_bytes()) {
            decoder.read_row(row)?;
        }
        frames.push((info, pixels));
        match decoder.advance_frame()? {
            Some(next) => info = next,
            None => return Ok(frames),
        }
    }
}

/// Where a [`FrameRecorder`] is in the frame protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecorderState {
    Fresh,
    Animated,
    Header,
    InFrame,
    Finished,
}

/// An animated encoder with no format behind it, which records each frame
/// and refuses every call out of the order [`crate::Encoder::set_animation`]
/// documents.
///
/// It is the protocol's reference: a pipeline test that drives it without
/// an error drives a real animated encoder correctly too. Bytes go nowhere;
/// [`FrameRecorder::frames`] holds what was written.
#[derive(Debug)]
pub struct FrameRecorder {
    state: RecorderState,
    descriptor: Option<ImageDescriptor>,
    loop_count: Option<u32>,
    rows: u32,
    frames: Vec<(u32, Vec<u8>)>,
}

impl Default for FrameRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameRecorder {
    /// A recorder that has seen nothing.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: RecorderState::Fresh,
            descriptor: None,
            loop_count: None,
            rows: 0,
            frames: Vec::new(),
        }
    }

    /// The frames written, as `(duration_ms, pixels)`.
    #[must_use]
    pub fn frames(&self) -> &[(u32, Vec<u8>)] {
        &self.frames
    }

    /// The loop count [`crate::Encoder::set_animation`] was given.
    #[must_use]
    pub const fn loop_count(&self) -> Option<u32> {
        self.loop_count
    }

    /// Whether [`crate::Encoder::finish`] succeeded.
    #[must_use]
    pub fn finished(&self) -> bool {
        self.state == RecorderState::Finished
    }

    fn misuse(call: &str, state: RecorderState) -> PixelsError {
        PixelsError::invalid_argument(
            "frame protocol",
            format!("`{call}` is not allowed in state {state:?}"),
        )
    }
}

impl crate::Encoder for FrameRecorder {
    fn supports_animation(&self) -> bool {
        true
    }

    fn set_animation(&mut self, loop_count: u32) -> Result<()> {
        if self.state != RecorderState::Fresh {
            return Err(Self::misuse("set_animation", self.state));
        }
        self.loop_count = Some(loop_count);
        self.state = RecorderState::Animated;
        Ok(())
    }

    fn write_header(&mut self, desc: &ImageDescriptor, _: &mut dyn crate::Sink) -> Result<()> {
        if self.state != RecorderState::Animated {
            return Err(Self::misuse("write_header", self.state));
        }
        self.descriptor = Some(*desc);
        self.state = RecorderState::Header;
        Ok(())
    }

    fn begin_frame(&mut self, duration_ms: u32, _: &mut dyn crate::Sink) -> Result<()> {
        if self.state != RecorderState::Header {
            return Err(Self::misuse("begin_frame", self.state));
        }
        self.frames.push((duration_ms, Vec::new()));
        self.rows = 0;
        self.state = RecorderState::InFrame;
        Ok(())
    }

    fn write_row(&mut self, row: &[u8], _: &mut dyn crate::Sink) -> Result<()> {
        let descriptor = match (self.state, self.descriptor) {
            (RecorderState::InFrame, Some(d)) => d,
            _ => return Err(Self::misuse("write_row", self.state)),
        };
        if row.len() != descriptor.row_bytes() || self.rows >= descriptor.height {
            return Err(PixelsError::invalid_argument(
                "row",
                "wrong length, or more rows than the canvas has",
            ));
        }
        self.rows += 1;
        if let Some((_, pixels)) = self.frames.last_mut() {
            pixels.extend_from_slice(row);
        }
        Ok(())
    }

    fn end_frame(&mut self, _: &mut dyn crate::Sink) -> Result<()> {
        let descriptor = match (self.state, self.descriptor) {
            (RecorderState::InFrame, Some(d)) => d,
            _ => return Err(Self::misuse("end_frame", self.state)),
        };
        if self.rows != descriptor.height {
            return Err(PixelsError::malformed(
                "testing",
                format!(
                    "frame ended after {} of {} rows",
                    self.rows, descriptor.height
                ),
            ));
        }
        self.state = RecorderState::Header;
        Ok(())
    }

    fn finish(&mut self, _: &mut dyn crate::Sink) -> Result<()> {
        if self.state != RecorderState::Header || self.frames.is_empty() {
            return Err(Self::misuse("finish", self.state));
        }
        self.state = RecorderState::Finished;
        Ok(())
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;
    use crate::{Decoder, Encoder, ErrorCode, FrameInfo, PixelFormat};

    fn canvas() -> ImageDescriptor {
        ImageDescriptor::new(3, 2, PixelFormat::Rgb8).unwrap()
    }

    #[test]
    fn the_stub_decoder_yields_every_frame_composited_to_the_canvas() {
        let mut decoder = FrameStubDecoder::new(canvas(), 3);
        assert_eq!(
            decoder.animation().unwrap().frame_durations_ms,
            [10, 20, 30]
        );
        let frames = read_frames(&mut decoder).unwrap();
        assert_eq!(frames.len(), 3);
        for (k, (info, pixels)) in frames.iter().enumerate() {
            assert_eq!(*info, FrameInfo::new(k as u32, (k as u32 + 1) * 10));
            assert_eq!(pixels.len(), 3 * 2 * 3);
            assert!(pixels.iter().all(|&b| b == k as u8));
        }
        // Exhausted stays exhausted.
        assert_eq!(decoder.advance_frame().unwrap(), None);
        assert!(decoder.read_row(&mut [0; 9]).is_err());
    }

    #[test]
    fn advancing_skips_the_rest_of_a_frame() {
        let mut decoder = FrameStubDecoder::new(canvas(), 2);
        let mut row = [9; 9];
        decoder.read_row(&mut row).unwrap();
        assert_eq!(row, [0; 9]);
        // One row of frame 0 left unread; frame 1 starts at its top.
        assert_eq!(
            decoder.advance_frame().unwrap(),
            Some(FrameInfo::new(1, 20))
        );
        decoder.read_row(&mut row).unwrap();
        decoder.read_row(&mut row).unwrap();
        assert_eq!(row, [1; 9]);
        assert!(decoder.read_row(&mut row).is_err(), "only two rows");
    }

    #[test]
    fn a_still_reads_as_one_frame() {
        let mut decoder = FrameStubDecoder::new(canvas(), 1);
        assert_eq!(decoder.animation(), None);
        let frames = read_frames(&mut decoder).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, FrameInfo::new(0, 0));
    }

    /// Copy every frame from `decoder` into `encoder` by the protocol.
    fn transcode(decoder: &mut dyn Decoder, encoder: &mut dyn Encoder) -> Result<()> {
        let mut sink = Vec::new();
        let descriptor = decoder.descriptor();
        encoder.set_animation(decoder.animation().map_or(0, |a| a.loop_count))?;
        encoder.write_header(&descriptor, &mut sink)?;
        for (info, pixels) in read_frames(decoder)? {
            encoder.begin_frame(info.duration_ms, &mut sink)?;
            for row in pixels.chunks_exact(descriptor.row_bytes()) {
                encoder.write_row(row, &mut sink)?;
            }
            encoder.end_frame(&mut sink)?;
        }
        encoder.finish(&mut sink)
    }

    #[test]
    fn the_recorder_accepts_the_protocol_and_keeps_every_frame() {
        let mut recorder = FrameRecorder::new();
        transcode(&mut FrameStubDecoder::new(canvas(), 4), &mut recorder).unwrap();
        assert!(recorder.finished());
        assert_eq!(recorder.loop_count(), Some(0));
        let durations: Vec<u32> = recorder.frames().iter().map(|f| f.0).collect();
        assert_eq!(durations, [10, 20, 30, 40]);
        for (k, (_, pixels)) in recorder.frames().iter().enumerate() {
            assert!(pixels.iter().all(|&b| b == k as u8), "frame {k}");
        }
    }

    #[test]
    fn the_recorder_refuses_calls_out_of_order() {
        let mut sink = Vec::new();
        let row = [0; 9];
        let code = |r: Result<()>| r.unwrap_err().code();

        // Header before set_animation.
        let mut r = FrameRecorder::new();
        assert_eq!(
            code(r.write_header(&canvas(), &mut sink)),
            ErrorCode::InvalidArgument
        );

        // A frame before the header, a row outside a frame, set_animation twice.
        let mut r = FrameRecorder::new();
        r.set_animation(1).unwrap();
        assert!(r.set_animation(1).is_err());
        assert!(r.begin_frame(10, &mut sink).is_err());
        r.write_header(&canvas(), &mut sink).unwrap();
        assert!(r.write_row(&row, &mut sink).is_err());

        // No frames at all cannot finish.
        assert!(r.finish(&mut sink).is_err());

        // A short frame cannot end, nor a frame open finish.
        r.begin_frame(10, &mut sink).unwrap();
        assert!(r.begin_frame(10, &mut sink).is_err(), "already open");
        r.write_row(&row, &mut sink).unwrap();
        assert_eq!(code(r.end_frame(&mut sink)), ErrorCode::Malformed);
        assert!(r.finish(&mut sink).is_err());
        r.write_row(&row, &mut sink).unwrap();
        assert!(r.write_row(&row, &mut sink).is_err(), "a third row");
        r.end_frame(&mut sink).unwrap();
        assert!(r.end_frame(&mut sink).is_err(), "none open");
        r.finish(&mut sink).unwrap();
        assert!(r.finished());
    }
}
