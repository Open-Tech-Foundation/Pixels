//! The operation trait and the pixel producers that feed graphs.

use crate::{ImageDescriptor, Region, Result, Tile, TileMut};
use core::fmt;

/// The tile **shape** an op wants its input delivered in.
///
/// The scheduler negotiates tile shapes from this declaration (ADR-0003):
/// runs of [`Sequential`] ops move full-width strips, matching how codecs
/// produce and consume rows, while [`Spatial`] segments switch to square tiles
/// to bound redundant border work. A rolling line-cache is inserted at the
/// seam between the two. Declaring [`Spatial`] when [`Sequential`] would do
/// costs throughput; declaring [`Sequential`] when the op actually reads
/// neighbours is a correctness bug.
///
/// # Shape, not order
///
/// This says nothing about the *order* tiles are produced in. An op that
/// mirrors vertically reads no neighbours and wants full-width strips, so it
/// is [`Sequential`] — even though it consumes those strips bottom-up.
///
/// Order is not declared at all: the scheduler derives it from
/// [`Op::input_regions`] and inserts a materialization buffer only where a
/// non-forward demand sequence meets a forward-only source (ADR-0009). Keeping
/// order out of this enum is deliberate — it is a property of the *seam*
/// between an op and its upstream, not of the op, so the same op streams or
/// buffers depending on what it is connected to.
///
/// [`Sequential`]: AccessPattern::Sequential
/// [`Spatial`]: AccessPattern::Spatial
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessPattern {
    /// Each output pixel depends on input pixels from a single row.
    ///
    /// Wants full-width strips. Pointwise ops (`modulate`, `flatten`, channel
    /// extraction) and row-preserving geometry remaps (`crop`, `flop`, `flip`)
    /// are sequential — including `flip`, which reverses row *order* but still
    /// reads one input row per output row.
    Sequential,
    /// Output pixels depend on a neighbourhood spanning multiple input rows.
    ///
    /// Wants square tiles. Convolution and resize with filter support are
    /// spatial.
    Spatial,
}

/// A node in the op graph.
///
/// Ops are immutable, shared (`Arc<dyn Op>`), and evaluated concurrently, hence
/// `Send + Sync`. An op holds its own parameters; the graph supplies the input
/// descriptors, which is what lets one op instance be evaluated against
/// different input shapes.
///
/// # Relationship to ARCHITECTURE §Layer 3
///
/// [`Op::input_regions`] and [`Op::output_descriptor`] take the input
/// descriptors explicitly, where ARCHITECTURE writes them in shorthand as
/// `input_region(out_region)` and `output_descriptor()`. The semantics are
/// unchanged — an op still cannot know its input shapes without being told
/// them, and passing them keeps ops free of duplicated graph state.
pub trait Op: Send + Sync + fmt::Debug {
    /// A short, stable name for this op, used in diagnostics.
    fn name(&self) -> &'static str;

    /// The number of inputs this op consumes.
    fn arity(&self) -> usize {
        1
    }

    /// Compute the output shape from the input shapes.
    ///
    /// This runs at **graph-build time**, not evaluation time: descriptors flow
    /// forward as the graph is chained, which is what makes
    /// [`Image::metadata`] free of pixel work (SPEC §Guarantees 3).
    ///
    /// [`Image::metadata`]: crate::Image::metadata
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if the op's parameters are
    /// incompatible with these inputs (for example a crop outside the image),
    /// or [`PixelsError::Unsupported`] if the op cannot handle the input pixel
    /// format.
    ///
    /// [`PixelsError::InvalidArgument`]: crate::PixelsError::InvalidArgument
    /// [`PixelsError::Unsupported`]: crate::PixelsError::Unsupported
    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor>;

    /// A copy of this op fit for a graph rebuilt at a **reduced input
    /// resolution**, or `None` if this op must not be.
    ///
    /// This is what licenses shrink-on-load: a JPEG can be decoded at 1/8 and
    /// fed to `resize` with the same result. Returning `Some` asserts two
    /// separate things, which is why they are one method — an op that got only
    /// the first right would be silently wrong:
    ///
    /// 1. **The op means the same thing** against a smaller input. Three kinds
    ///    do not, and all three still produce a correctly-shaped image:
    ///    ops carrying coordinates in input pixels (`crop(1000, 1000, ..)`
    ///    names a different part of a source eight times smaller), ops
    ///    carrying a distance in input pixels (a 3x3 convolution over a 1/8
    ///    decode is eight times the blur relative to content), and ops whose
    ///    second input is a separate image that would not be reduced with it.
    /// 2. **The returned instance carries no state bound to the old input.**
    ///    Ops may memoize tables keyed to the shape they first saw — `resize`
    ///    builds its filter weights that way — and reusing such an instance
    ///    against a new shape is at best an error and at worst a resample
    ///    against the wrong scale.
    ///
    /// The default is `None`, so an op is presumed unsafe until it says
    /// otherwise. Declaring it wrongly does not corrupt memory or change the
    /// output *shape*; it silently changes the picture, which is worse, so the
    /// conservative default is the right one.
    fn rescaled(&self) -> Option<std::sync::Arc<dyn Op>> {
        None
    }

    /// The input regions needed to produce `output`.
    ///
    /// This is the inverse mapping that demand propagation walks backwards
    /// (ARCHITECTURE §Layer 4). The returned vector has one region per input,
    /// in input order. A pointwise op returns `output` unchanged; a 5×5
    /// convolution returns it grown by 2px; a resize returns the scaled region
    /// plus filter support.
    ///
    /// Returned regions must be clamped to their input's bounds — an op asking
    /// for pixels outside its input is a defect, and edge handling
    /// (clamp, reflect, …) is the op's own business.
    ///
    /// # Errors
    ///
    /// Returns an error if `output` is not a region this op can produce.
    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>>;

    /// How this op reads its inputs; drives tile negotiation (ADR-0003).
    ///
    /// Defaults to [`AccessPattern::Sequential`], the safe-and-fast case for
    /// pointwise ops. Override it for anything reading a neighbourhood.
    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Sequential
    }

    /// Fill `output` from `inputs` — the kernel entry point.
    ///
    /// `inputs` holds one tile per input, covering exactly the regions
    /// [`Op::input_regions`] asked for. `output` covers the region that was
    /// passed to it.
    ///
    /// Per ADR-0002 the pixel format is a runtime value here: dispatch **once**
    /// on it into a monomorphized kernel (see [`dispatch_sample!`]) rather than
    /// branching per pixel.
    ///
    /// [`dispatch_sample!`]: crate::dispatch_sample
    ///
    /// # Errors
    ///
    /// Returns an error if the supplied tiles do not match what
    /// [`Op::input_regions`] requested, or if the pixel format is one this op
    /// does not implement.
    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()>;
}

/// A source of pixels at the root of a graph.
///
/// Producers sit where decoders meet the graph. `produce` takes `&self` because
/// graph nodes are shared across threads; a producer wrapping a single-pass
/// streaming decoder therefore owns whatever interior mutability it needs (see
/// [`DecodedSource`]).
///
/// [`DecodedSource`]: crate::DecodedSource
pub trait Producer: Send + Sync + fmt::Debug {
    /// A short, stable name for this producer, used in diagnostics.
    fn name(&self) -> &'static str;

    /// The shape of the image this producer yields.
    ///
    /// Known from the header alone; answering this must not decode pixels.
    fn descriptor(&self) -> ImageDescriptor;

    /// Whether this producer can serve arbitrary regions, or only forward ones.
    ///
    /// This is the upstream half of ADR-0009's seam analysis: a producer that
    /// can only go forward forces the scheduler to materialize whenever demand
    /// is not forward-monotonic, while one serving arbitrary regions lets the
    /// same pipeline stream.
    ///
    /// Defaults to [`DecodeCapability::Sequential`], the conservative answer —
    /// over-declaring it costs a buffer, under-declaring it is a correctness
    /// bug.
    ///
    /// [`DecodeCapability::Sequential`]: crate::DecodeCapability::Sequential
    fn capability(&self) -> crate::DecodeCapability {
        crate::DecodeCapability::Sequential
    }

    /// Fill `output` with the pixels of `region`.
    ///
    /// `region` is always within `descriptor().region()`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] on invalid input bytes,
    /// [`PixelsError::Io`] on source failure, or
    /// [`PixelsError::InvalidArgument`] if `output` does not cover `region`.
    ///
    /// [`PixelsError::Malformed`]: crate::PixelsError::Malformed
    /// [`PixelsError::Io`]: crate::PixelsError::Io
    /// [`PixelsError::InvalidArgument`]: crate::PixelsError::InvalidArgument
    fn produce(&self, region: Region, output: &mut TileMut<'_>) -> Result<()>;

    /// What this producer would emit if asked for `target` or larger, when it
    /// can reach that size more cheaply than by producing full resolution.
    ///
    /// **Pure**: nothing is committed, and calling this must not change what
    /// the producer subsequently emits. The planner asks first, checks the
    /// whole graph still holds, and only then calls [`Producer::reduce_to`] —
    /// so a producer that reduced itself here would corrupt pipelines the
    /// planner went on to reject.
    ///
    /// The returned descriptor is never smaller than `target` in either axis:
    /// decoding below the requested size and enlarging afterwards would
    /// discard detail and then invent it back.
    ///
    /// `None` — the default — means this producer has only one resolution.
    fn reduced_descriptor(&self, target: (u32, u32)) -> Option<ImageDescriptor> {
        let _ = target;
        None
    }

    /// Commit to emitting `descriptor`, which [`Producer::reduced_descriptor`]
    /// must have returned.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] if this producer cannot reduce, or
    /// [`PixelsError::InvalidArgument`] if pixels have already been produced —
    /// the resolution is fixed from the first [`Producer::produce`] onward,
    /// because rows already emitted cannot be retracted.
    ///
    /// [`PixelsError::Unsupported`]: crate::PixelsError::Unsupported
    /// [`PixelsError::InvalidArgument`]: crate::PixelsError::InvalidArgument
    fn reduce_to(&self, descriptor: ImageDescriptor) -> Result<()> {
        let _ = descriptor;
        Err(crate::PixelsError::unsupported(format!(
            "producer `{}` has only one resolution",
            self.name()
        )))
    }
}
