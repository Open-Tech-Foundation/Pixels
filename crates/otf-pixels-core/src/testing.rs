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
        Self { descriptor, calls: AtomicUsize::new(0) }
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
        inputs.first().copied().ok_or_else(|| PixelsError::graph("constant needs one input"))
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
            return Err(PixelsError::invalid_argument("inputs", "sum needs matching inputs"));
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
        inputs.first().copied().ok_or_else(|| PixelsError::graph("failing needs one input"))
    }

    fn input_regions(&self, output: Region, _: &[ImageDescriptor]) -> Result<Vec<Region>> {
        Ok(vec![output])
    }

    fn compute(&self, _: &[Tile<'_>], _: &mut TileMut<'_>) -> Result<()> {
        Err(PixelsError::malformed("testing", "this op always fails"))
    }
}
