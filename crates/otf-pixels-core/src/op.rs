//! The operation trait and the pixel producers that feed graphs.

use crate::{ImageDescriptor, Region, Result, Tile, TileMut};
use core::fmt;

/// How an op reads its input relative to the output it is producing.
///
/// The scheduler negotiates tile shapes from this declaration (ADR-0003):
/// runs of [`Sequential`] ops move full-width strips, matching how codecs
/// produce and consume rows, while [`Spatial`] segments switch to square tiles
/// to bound redundant border work. A rolling line-cache is inserted at the
/// seam between the two. Declaring [`Spatial`] when [`Sequential`] would do
/// costs throughput; declaring [`Sequential`] when the op actually reads
/// neighbours is a correctness bug.
///
/// [`Sequential`]: AccessPattern::Sequential
/// [`Spatial`]: AccessPattern::Spatial
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccessPattern {
    /// Each output pixel depends only on input pixels in the same row order.
    ///
    /// Pointwise ops (`modulate`, `flatten`, channel extraction) and pure
    /// geometry remaps that preserve row order (`crop`, `flop`) are sequential.
    Sequential,
    /// Output pixels depend on a neighbourhood of input pixels.
    ///
    /// Convolution, resize with filter support, and rotation are spatial.
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
    fn input_regions(
        &self,
        output: Region,
        inputs: &[ImageDescriptor],
    ) -> Result<Vec<Region>>;

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
}
