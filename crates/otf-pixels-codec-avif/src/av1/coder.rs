//! The seam between tile syntax and entropy coding.
//!
//! The tile decoder walks the AV1 syntax — partitions, modes, transform
//! blocks — and at each symbol asks a [`TileCoder`] for it. Decoding answers
//! by reading the arithmetic-coded stream. Encoding answers with the
//! encoder's own decision and writes it, so the encoder is the decoder
//! driven by decisions: it reconstructs, predicts and adapts its CDFs with
//! the very code that will later decode its output, and the two cannot
//! drift apart.

use super::coeff::{CoeffBlock, CoeffCdfs, TxTypeCtx, decode_coeffs};
use super::symbol::SymbolDecoder;
use super::tile::TileState;
use super::transform::TxSize;
use otf_pixels_core::Result;

/// What a symbol codes, for an encoder to decide it. A decoder ignores this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Site {
    /// `partition` for the square block of side `bsize4` 4x4 units at `(r, c)`.
    Partition { r: usize, c: usize, bsize4: usize },
    /// `split_or_horz` / `split_or_vert` for a block crossing the frame edge.
    SplitOr { r: usize, c: usize, bsize4: usize },
    /// The block's `skip`.
    Skip,
    /// The 64x64 block's `cdef_idx`.
    CdefIdx,
    /// The luma intra mode.
    YMode,
    /// `angle_delta_y`, offset by `MAX_ANGLE_DELTA`.
    AngleDeltaY,
    /// The chroma intra mode.
    UvMode,
    /// `angle_delta_uv`, offset by `MAX_ANGLE_DELTA`.
    AngleDeltaUv,
    /// `cfl_alpha_signs`.
    CflSign,
    /// `cfl_alpha_u` or `cfl_alpha_v` magnitude less one.
    CflAlpha,
    /// `tx_depth`.
    TxDepth,
    /// Anything an encoder of this subset never codes: palettes,
    /// filter-intra, segment ids, restoration units, deltas.
    Other,
}

/// What the encoder needs to code one transform block's coefficients: where
/// it is and what the decoder predicted for it.
#[allow(dead_code, reason = "read by the encoder's coder, which lands next")]
pub(crate) struct CoeffJob<'a> {
    /// Plane index: 0 luma, 1 U, 2 V.
    pub plane: usize,
    /// Top-left sample in the plane.
    pub x: usize,
    /// Top-left sample in the plane.
    pub y: usize,
    /// The prediction, `width * height` row-major.
    pub prediction: &'a [u16],
    /// The block's DC and AC quantizer steps.
    pub dc_q: i64,
    /// See `dc_q`.
    pub ac_q: i64,
}

/// Answers the tile syntax's symbols: by reading them, or by deciding and
/// writing them.
pub(crate) trait TileCoder {
    /// A symbol coded with `cdf` (adapting it).
    ///
    /// # Errors
    ///
    /// A decoder reports a malformed or exhausted stream.
    fn symbol(&mut self, cdf: &mut [u16], site: Site) -> Result<usize>;

    /// An `n`-bit literal.
    ///
    /// # Errors
    ///
    /// As [`TileCoder::symbol`].
    fn literal(&mut self, n: u32, site: Site) -> Result<u32>;

    /// A non-symmetric value in `0..n`.
    ///
    /// # Errors
    ///
    /// As [`TileCoder::symbol`].
    fn ns(&mut self, n: u32) -> Result<u32>;

    /// Called as each block starts, before any of its symbols; an encoder
    /// decides the block's modes here, with the reconstruction so far.
    fn plan_block(&mut self, _state: &TileState, _r: usize, _c: usize, _bw4: usize, _bh4: usize) {}

    /// One transform block's coefficients.
    ///
    /// # Errors
    ///
    /// As [`TileCoder::symbol`].
    #[allow(clippy::too_many_arguments, reason = "the coefficient syntax's inputs")]
    fn coeffs(
        &mut self,
        cdfs: &mut CoeffCdfs,
        tx_size: TxSize,
        tx: TxTypeCtx<'_>,
        ptype: usize,
        all_zero_ctx: usize,
        dc_sign_ctx: usize,
        job: &CoeffJob<'_>,
    ) -> Result<CoeffBlock>;
}

impl TileCoder for SymbolDecoder<'_> {
    fn symbol(&mut self, cdf: &mut [u16], _site: Site) -> Result<usize> {
        self.read_symbol(cdf)
    }

    fn literal(&mut self, n: u32, _site: Site) -> Result<u32> {
        self.read_literal(n)
    }

    fn ns(&mut self, n: u32) -> Result<u32> {
        self.read_ns(n)
    }

    fn coeffs(
        &mut self,
        cdfs: &mut CoeffCdfs,
        tx_size: TxSize,
        tx: TxTypeCtx<'_>,
        ptype: usize,
        all_zero_ctx: usize,
        dc_sign_ctx: usize,
        _job: &CoeffJob<'_>,
    ) -> Result<CoeffBlock> {
        decode_coeffs(self, cdfs, tx_size, tx, ptype, all_zero_ctx, dc_sign_ctx)
    }
}
