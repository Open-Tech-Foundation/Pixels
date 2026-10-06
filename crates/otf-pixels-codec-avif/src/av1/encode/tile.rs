//! The encoder's [`TileCoder`]: the tile syntax asks, this decides and
//! writes.
//!
//! Decisions are deliberately simple and always legal:
//! - 64x64 superblocks always split; 32x32 and 16x16 blocks split when their
//!   luma varies more than the quantizer can represent cheaply; 8x8 is the
//!   smallest block, so every block has its own chroma.
//! - Each block's luma and chroma modes are the intra predictions closest to
//!   the source (sum of absolute differences), with no angle deltas, CfL,
//!   palette or filter-intra.
//! - Residuals go through the forward transform derived from the decoder's
//!   inverse, a dead-zone quantizer, and the coefficient writer.
//!
//! Anything the subset never codes is written as symbol 0, which is legal at
//! every site.

use super::super::coder::{CoeffJob, Site, TileCoder};
use super::super::coeff::{CoeffBlock, CoeffCdfs, TxTypeCtx, encode_coeffs};
use super::super::symbol::SymbolEncoder;
use super::super::tile::TileState;
use super::super::transform::dsp::{ForwardBasis, quantize};
use super::super::transform::{TxSize, TxType};
use super::super::transform_type::chroma_tx_type;
use otf_pixels_core::Result;

/// `PARTITION_NONE` and `PARTITION_SPLIT`.
const PARTITION_NONE: usize = 0;
const PARTITION_SPLIT: usize = 3;
/// `MAX_ANGLE_DELTA`: the angle-delta symbol meaning "no delta".
const NO_ANGLE_DELTA: usize = 3;
/// Intra modes `DC_PRED..=PAETH_PRED`.
const INTRA_MODES: usize = 13;

/// One plane of the picture to encode, padded by edge replication to cover
/// whole superblocks (the decoder's plane size).
#[derive(Debug, Clone)]
pub(crate) struct SourcePlane {
    pub data: Vec<u16>,
    pub width: usize,
    pub height: usize,
}

impl SourcePlane {
    /// `samples` (`width * height`, row-major) padded to `padded_width` by
    /// `padded_height`.
    pub(crate) fn padded(
        samples: &[u16],
        width: usize,
        height: usize,
        padded_width: usize,
        padded_height: usize,
    ) -> Self {
        let mut data = Vec::with_capacity(padded_width * padded_height);
        for y in 0..padded_height {
            let row = y.min(height.saturating_sub(1)) * width;
            for x in 0..padded_width {
                data.push(
                    samples
                        .get(row + x.min(width.saturating_sub(1)))
                        .copied()
                        .unwrap_or(0),
                );
            }
        }
        Self {
            data,
            width: padded_width,
            height: padded_height,
        }
    }

    fn at(&self, x: usize, y: usize) -> i32 {
        let (x, y) = (x.min(self.width - 1), y.min(self.height - 1));
        i32::from(self.data.get(y * self.width + x).copied().unwrap_or(0))
    }
}

/// Encoder tuning.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Tuning {
    /// The AC quantizer step (for the split decision).
    pub ac_q: i64,
    /// Dead-zone rounding points for DC and AC levels, as a fraction of a
    /// step.
    pub dc_bias: f64,
    pub ac_bias: f64,
}

/// Decides and writes one tile.
pub(crate) struct TileEncoder<'a> {
    enc: SymbolEncoder,
    source: &'a [SourcePlane],
    tuning: Tuning,
    /// The current block's modes, chosen in `plan_block`.
    y_mode: usize,
    uv_mode: usize,
    bases: Vec<(TxSize, TxType, ForwardBasis)>,
}

impl<'a> TileEncoder<'a> {
    pub(crate) fn new(source: &'a [SourcePlane], tuning: Tuning) -> Self {
        Self {
            enc: SymbolEncoder::new(false),
            source,
            tuning,
            y_mode: 0,
            uv_mode: 0,
            bases: Vec::new(),
        }
    }

    /// The tile's coded bytes.
    pub(crate) fn finish(self) -> Vec<u8> {
        self.enc.finish()
    }

    /// Whether the square block of `bsize4` 4x4 units at `(r, c)` should split.
    fn should_split(&self, r: usize, c: usize, bsize4: usize) -> bool {
        if bsize4 >= 16 {
            return true;
        }
        if bsize4 <= 2 {
            return false;
        }
        let Some(luma) = self.source.first() else {
            return false;
        };
        let side = bsize4 * 4;
        let (x0, y0) = (c * 4, r * 4);
        let (mut sum, mut sum_sq) = (0_i64, 0_i64);
        for y in y0..y0 + side {
            for x in x0..x0 + side {
                let v = i64::from(luma.at(x, y));
                sum += v;
                sum_sq += v * v;
            }
        }
        let n = (side * side) as i64;
        let variance = (sum_sq - sum * sum / n) / n;
        // The quantizer step in sample units is about ac_q / 8: split when the
        // block's spread is well beyond what one large transform resolves.
        let step = (self.tuning.ac_q as f64 / 8.0).max(1.0);
        variance as f64 > step * step * 0.6 + 4.0
    }

    /// The mode whose prediction of `planes` is closest to the source.
    fn best_mode(
        &self,
        state: &TileState,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        planes: &[usize],
    ) -> usize {
        let mut best = (u64::MAX, 0);
        for mode in 0..INTRA_MODES {
            let mut cost = 0_u64;
            for &plane in planes {
                let (Some(pred), Some(src)) = (
                    state.preview_prediction(r, c, bw4, bh4, plane, mode),
                    self.source.get(plane),
                ) else {
                    return 0;
                };
                let sub = usize::from(
                    plane > 0
                        && self
                            .source
                            .first()
                            .is_some_and(|luma| src.width < luma.width),
                );
                let (x0, y0) = ((c * 4) >> sub, (r * 4) >> sub);
                let w = ((bw4 * 4) >> sub).max(4);
                let h = pred.len() / w;
                for y in 0..h {
                    for x in 0..w {
                        let p = i32::from(pred.get(y * w + x).copied().unwrap_or(0));
                        cost += u64::from((src.at(x0 + x, y0 + y) - p).unsigned_abs());
                    }
                }
            }
            // A slight preference for DC, the cheapest mode to code.
            if mode != 0 {
                cost += cost / 64;
            }
            if cost < best.0 {
                best = (cost, mode);
            }
        }
        best.1
    }

    fn basis(&mut self, tx_size: TxSize, tx_type: TxType) -> &ForwardBasis {
        if let Some(i) = self
            .bases
            .iter()
            .position(|(s, t, _)| *s == tx_size && *t == tx_type)
        {
            let (_, _, basis) = self.bases.swap_remove(i);
            self.bases.push((tx_size, tx_type, basis));
        } else {
            self.bases
                .push((tx_size, tx_type, ForwardBasis::new(tx_size, tx_type)));
        }
        // The basis just pushed (or moved) to the end.
        let (_, _, basis) = self.bases.last_mut().unwrap_or_else(|| unreachable!());
        basis
    }
}

impl TileCoder for TileEncoder<'_> {
    fn symbol(&mut self, cdf: &mut [u16], site: Site) -> Result<usize> {
        let value = match site {
            Site::Partition { r, c, bsize4 } => {
                if self.should_split(r, c, bsize4) {
                    PARTITION_SPLIT
                } else {
                    PARTITION_NONE
                }
            }
            // Split a block crossing the frame edge: its quarters fit better.
            Site::SplitOr { .. } => 1,
            Site::YMode => self.y_mode,
            Site::UvMode => self.uv_mode,
            Site::AngleDeltaY | Site::AngleDeltaUv => NO_ANGLE_DELTA,
            _ => 0,
        };
        self.enc.write_symbol(cdf, value);
        Ok(value)
    }

    fn literal(&mut self, n: u32, _site: Site) -> Result<u32> {
        self.enc.write_literal(n, 0);
        Ok(0)
    }

    fn ns(&mut self, n: u32) -> Result<u32> {
        // Value 0 of `ns(n)` is `w - 1` zero bits.
        let w = u32::BITS - n.leading_zeros();
        self.enc.write_literal(w.saturating_sub(1), 0);
        Ok(0)
    }

    fn plan_block(&mut self, state: &TileState, r: usize, c: usize, bw4: usize, bh4: usize) {
        self.y_mode = self.best_mode(state, r, c, bw4, bh4, &[0]);
        self.uv_mode = if self.source.len() > 1 {
            self.best_mode(state, r, c, bw4, bh4, &[1, 2])
        } else {
            0
        };
    }

    fn coeffs(
        &mut self,
        cdfs: &mut CoeffCdfs,
        tx_size: TxSize,
        tx: TxTypeCtx<'_>,
        ptype: usize,
        all_zero_ctx: usize,
        dc_sign_ctx: usize,
        job: &CoeffJob<'_>,
    ) -> Result<CoeffBlock> {
        let tx_type = if ptype == 0 || tx.lossless {
            TxType::DctDct
        } else {
            chroma_tx_type(tx.uv_mode, tx.set)
        };
        let (w, h) = (tx_size.width(), tx_size.height());
        let source = self.source.get(job.plane);
        let mut residual = vec![0_i32; w * h];
        if let Some(src) = source {
            for (k, out) in residual.iter_mut().enumerate() {
                let p = i32::from(job.prediction.get(k).copied().unwrap_or(0));
                *out = src.at(job.x + k % w, job.y + k / w) - p;
            }
        }
        let tuning = self.tuning;
        let coefficients = self.basis(tx_size, tx_type).forward(&residual);
        let levels: Vec<i32> = coefficients
            .iter()
            .enumerate()
            .map(|(i, &coefficient)| {
                let (q, bias) = if i == 0 {
                    (job.dc_q, tuning.dc_bias)
                } else {
                    (job.ac_q, tuning.ac_bias)
                };
                quantize(coefficient, q, tx_size, bias)
            })
            .collect();
        encode_coeffs(
            &mut self.enc,
            cdfs,
            tx_size,
            tx,
            tx_type,
            ptype,
            all_zero_ctx,
            dc_sign_ctx,
            &levels,
        )
    }
}
