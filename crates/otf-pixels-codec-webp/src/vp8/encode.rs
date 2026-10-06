//! VP8 key-frame encoding: the lossy half of WebP output.
//!
//! The encoder mirrors the decoder in this module step for step, which is
//! what keeps the two in sync: every macroblock is predicted from the
//! encoder's own reconstruction, made with the decoder's inverse transforms,
//! so the picture the encoder reasons about is the one any decoder will see.
//!
//! Per macroblock it chooses between 16x16 prediction (four modes) and
//! sixteen 4x4 subblocks (ten modes each), and among four chroma modes, by
//! rate-distortion cost: squared error plus `lambda` times the bits the
//! tokens would take under the current probabilities. A second pass counts
//! what was actually chosen and rewrites the coefficient probabilities where
//! that pays for its own signalling.

#![allow(
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    reason = "block loops index the coefficient, context and edge arrays in \
              parallel by the spec's block number; planes are allocated to whole \
              macroblocks"
)]

use super::predict::{self, B_PRED, DC_PRED, H_PRED, TM_PRED, V_PRED, b};
use super::tables::{
    AC_Q_LOOKUP, COEFF_UPDATE_PROBS, DC_Q_LOOKUP, DEFAULT_COEFF_PROBS, KF_B_MODE_PROBS,
};
use super::transform::{idct_add, inverse_wht};
use super::{
    B_MODE_TREE, BANDS, CoeffProbs, KF_UV_MODE_PROBS, KF_Y_MODE_PROBS, KF_Y_MODE_TREE,
    UV_MODE_TREE, ZIGZAG,
};
use super::{block_edges, sub_edges};
use otf_pixels_core::{PixelsError, Result};

/// The largest coded coefficient magnitude.
const MAX_LEVEL: i32 = 2047;

/// Encoder settings beyond quality, mostly to reach every corner of the
/// format — the decoder is checked against libwebp on files that use them.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// 1..=100, mapped to the quantizer as libwebp maps it.
    pub quality: u8,
    /// Token partitions as a power of two, 0..=3.
    pub partitions_log2: u8,
    /// Use the simple loop filter instead of the normal one.
    pub simple_filter: bool,
    /// Loop-filter sharpness, 0..=7.
    pub sharpness: u8,
    /// Four segments by local activity, with relative quantizer and filter
    /// offsets, instead of one quantizer for the whole frame.
    pub segments: bool,
    /// Code loop-filter deltas for the intra reference and for 4x4
    /// prediction.
    pub filter_deltas: bool,
}

impl Params {
    /// Settings for `quality`, everything else at its default.
    pub const fn with_quality(quality: u8) -> Self {
        Self {
            quality,
            partitions_log2: 0,
            simple_filter: false,
            sharpness: 0,
            segments: false,
            filter_deltas: false,
        }
    }
}

/// YUV 4:2:0 planes padded to whole macroblocks, ready to encode.
pub struct Planes {
    /// Luma, 16 per macroblock column wide.
    pub y: Vec<u8>,
    /// Chroma, 8 per macroblock column wide.
    pub u: Vec<u8>,
    /// Chroma, 8 per macroblock column wide.
    pub v: Vec<u8>,
    /// Visible width.
    pub width: usize,
    /// Visible height.
    pub height: usize,
}

/// Where booleans go: an arithmetic coder, or a counter of their cost.
trait BoolSink {
    fn put(&mut self, probability: u8, bit: bool);
    /// A coefficient-tree node, which the probability optimiser also counts.
    fn node(&mut self, probability: u8, bit: bool, _at: (usize, usize, usize, usize)) {
        self.put(probability, bit);
    }
    fn literal(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.put(128, (value >> i) & 1 == 1);
        }
    }
    fn signed(&mut self, n: u32, value: i32) {
        self.put(128, value != 0);
        if value != 0 {
            self.literal(n, value.unsigned_abs());
            self.put(128, value < 0);
        }
    }
    fn tree(&mut self, tree: &[i8], probs: &[u8], value: u8) {
        // Find the path to the leaf, then emit it from the root.
        fn path(tree: &[i8], node: usize, value: u8, out: &mut Vec<(usize, bool)>) -> bool {
            for bit in [false, true] {
                let next = tree[node + usize::from(bit)];
                out.push((node, bit));
                if next <= 0 {
                    if next.unsigned_abs() == value {
                        return true;
                    }
                } else if path(tree, next as usize, value, out) {
                    return true;
                }
                out.pop();
            }
            false
        }
        let mut steps = Vec::new();
        path(tree, 0, value, &mut steps);
        for (node, bit) in steps {
            self.put(probs[node >> 1], bit);
        }
    }
}

/// The RFC 6386 §7.3 boolean encoder.
pub struct BoolEncoder {
    out: Vec<u8>,
    range: u32,
    bottom: u32,
    bit_count: i32,
}

impl BoolEncoder {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            range: 255,
            bottom: 0,
            bit_count: 24,
        }
    }

    fn add_one(&mut self) {
        for byte in self.out.iter_mut().rev() {
            if *byte == 255 {
                *byte = 0;
            } else {
                *byte += 1;
                return;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        let mut c = self.bit_count;
        let mut v = self.bottom;
        if v & (1 << (32 - c)) != 0 {
            self.add_one();
        }
        v <<= c & 7;
        c >>= 3;
        for _ in 0..c {
            v <<= 8;
        }
        for _ in 0..4 {
            self.out.push((v >> 24) as u8);
            v <<= 8;
        }
        self.out
    }
}

impl BoolSink for BoolEncoder {
    fn put(&mut self, probability: u8, bit: bool) {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        if bit {
            self.bottom = self.bottom.wrapping_add(split);
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            if self.bottom & (1 << 31) != 0 {
                self.add_one();
            }
            self.bottom <<= 1;
            self.bit_count -= 1;
            if self.bit_count == 0 {
                self.out.push((self.bottom >> 24) as u8);
                self.bottom &= (1 << 24) - 1;
                self.bit_count = 8;
            }
        }
    }
}

/// `-log2` of a probability out of 256, in 1/256 bits, for coding a 0 with
/// probability `p` (and a 1 at `256 - p`).
fn bit_cost(probability: u8, bit: bool) -> u32 {
    let p = if bit {
        256 - u32::from(probability)
    } else {
        u32::from(probability)
    };
    // log2(256 / p) * 256, by integer bit length plus a linear interpolation:
    // accurate to a few percent, which is all a mode decision needs.
    let shift = 31 - p.leading_zeros();
    let frac = ((p << 8) >> shift) - 256; // 0..256
    let log2_p = shift * 256 + frac;
    8 * 256 - log2_p
}

/// Counts what booleans would cost.
#[derive(Default)]
struct Cost(u32);

impl BoolSink for Cost {
    fn put(&mut self, probability: u8, bit: bool) {
        self.0 += bit_cost(probability, bit);
    }
}

/// Zero and one counts for every coefficient probability.
type BranchCounts = [[[[[u32; 2]; 11]; 3]; 8]; 4];

/// Counts coefficient-tree branches for the probability optimiser.
struct Stats {
    counts: Box<BranchCounts>,
}

impl BoolSink for Stats {
    fn put(&mut self, _probability: u8, _bit: bool) {}
    fn node(
        &mut self,
        _probability: u8,
        bit: bool,
        (t, band, ctx, node): (usize, usize, usize, usize),
    ) {
        self.counts[t][band][ctx][node][usize::from(bit)] += 1;
    }
}

/// Code one block's quantized `levels` (raster order) from scan position
/// `first`, mirroring the decoder's `read_block`. Returns whether any token
/// was coded past `first`.
fn write_block<S: BoolSink>(
    sink: &mut S,
    probs: &CoeffProbs,
    kind: usize,
    first: usize,
    ctx: usize,
    levels: &[i16; 16],
) -> bool {
    let last = (first..16).rev().find(|&n| levels[ZIGZAG[n]] != 0);
    let mut n = first;
    let mut c = ctx;
    let at = |n: usize, c: usize, node: usize| (kind, BANDS[n], c, node);
    let p = |n: usize, c: usize| &probs[kind][BANDS[n]][c];
    let Some(last) = last else {
        sink.node(p(n, c)[0], false, at(n, c, 0));
        return false;
    };
    sink.node(p(n, c)[0], true, at(n, c, 0));
    loop {
        let value = i32::from(levels[ZIGZAG[n]]);
        let v = value.abs();
        let pr = p(n, c);
        if v == 0 {
            sink.node(pr[1], false, at(n, c, 1));
            n += 1;
            c = 0;
            continue;
        }
        sink.node(pr[1], true, at(n, c, 1));
        if v == 1 {
            sink.node(pr[2], false, at(n, c, 2));
            c = 1;
        } else {
            sink.node(pr[2], true, at(n, c, 2));
            if v <= 4 {
                sink.node(pr[3], false, at(n, c, 3));
                if v == 2 {
                    sink.node(pr[4], false, at(n, c, 4));
                } else {
                    sink.node(pr[4], true, at(n, c, 4));
                    sink.node(pr[5], v == 4, at(n, c, 5));
                }
            } else {
                sink.node(pr[3], true, at(n, c, 3));
                let (category, base, probs): (usize, i32, &[u8]) = match v {
                    5..=6 => (0, 5, &[159]),
                    7..=10 => (1, 7, &[165, 145]),
                    11..=18 => (2, 11, &[173, 148, 140]),
                    19..=34 => (3, 19, &[176, 155, 140, 135]),
                    35..=66 => (4, 35, &[180, 157, 141, 134, 130]),
                    _ => (
                        5,
                        67,
                        &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129],
                    ),
                };
                if category < 2 {
                    sink.node(pr[6], false, at(n, c, 6));
                    sink.node(pr[7], category == 1, at(n, c, 7));
                } else if category < 4 {
                    sink.node(pr[6], true, at(n, c, 6));
                    sink.node(pr[8], false, at(n, c, 8));
                    sink.node(pr[9], category == 3, at(n, c, 9));
                } else {
                    sink.node(pr[6], true, at(n, c, 6));
                    sink.node(pr[8], true, at(n, c, 8));
                    sink.node(pr[10], category == 5, at(n, c, 10));
                }
                let extra = (v - base) as u32;
                let len = probs.len() as u32;
                for (i, &q) in probs.iter().enumerate() {
                    sink.put(q, (extra >> (len - 1 - i as u32)) & 1 == 1);
                }
            }
            c = 2;
        }
        sink.put(128, value < 0);
        n += 1;
        if n == 16 {
            return true;
        }
        let pr = p(n, c);
        if n > last {
            sink.node(pr[0], false, at(n, c, 0));
            return true;
        }
        sink.node(pr[0], true, at(n, c, 0));
    }
}

/// libwebp's `FTransform`: the forward 4x4 DCT of `src - pred`.
fn forward_dct(src: &[[u8; 4]; 4], pred: &[[u8; 4]; 4]) -> [i32; 16] {
    let mut tmp = [0_i32; 16];
    for i in 0..4 {
        let d: [i32; 4] = core::array::from_fn(|k| i32::from(src[i][k]) - i32::from(pred[i][k]));
        let (a0, a1, a2, a3) = (d[0] + d[3], d[1] + d[2], d[1] - d[2], d[0] - d[3]);
        tmp[i * 4] = (a0 + a1) * 8;
        tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
        tmp[2 + i * 4] = (a0 - a1) * 8;
        tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
    }
    let mut out = [0_i32; 16];
    for i in 0..4 {
        let (a0, a1) = (tmp[i] + tmp[12 + i], tmp[4 + i] + tmp[8 + i]);
        let (a2, a3) = (tmp[4 + i] - tmp[8 + i], tmp[i] - tmp[12 + i]);
        out[i] = (a0 + a1 + 7) >> 4;
        out[4 + i] = ((a2 * 2217 + a3 * 5352 + 12000) >> 16) + i32::from(a3 != 0);
        out[8 + i] = (a0 - a1 + 7) >> 4;
        out[12 + i] = (a3 * 2217 - a2 * 5352 + 51000) >> 16;
    }
    out
}

/// libwebp's `FTransformWHT` over the sixteen luma DCs, raster order.
fn forward_wht(dc: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0_i32; 16];
    for i in 0..4 {
        let r = &dc[i * 4..i * 4 + 4];
        let (a0, a1, a2, a3) = (r[0] + r[2], r[1] + r[3], r[1] - r[3], r[0] - r[2]);
        tmp[i * 4] = a0 + a1;
        tmp[1 + i * 4] = a3 + a2;
        tmp[2 + i * 4] = a3 - a2;
        tmp[3 + i * 4] = a0 - a1;
    }
    let mut out = [0_i32; 16];
    for i in 0..4 {
        let (a0, a1) = (tmp[i] + tmp[8 + i], tmp[4 + i] + tmp[12 + i]);
        let (a2, a3) = (tmp[4 + i] - tmp[12 + i], tmp[i] - tmp[8 + i]);
        out[i] = (a0 + a1) >> 1;
        out[4 + i] = (a3 + a2) >> 1;
        out[8 + i] = (a3 - a2) >> 1;
        out[12 + i] = (a0 - a1) >> 1;
    }
    out
}

/// One block kind's quantizer: steps, and libwebp's rounding biases (out of
/// 256 of a step), for DC and AC.
#[derive(Clone, Copy)]
struct Quant {
    step: [i32; 2],
    bias: [i32; 2],
}

impl Quant {
    /// Quantize `coeffs` (raster) from scan position `first`, returning the
    /// levels and their dequantized values as the decoder will see them.
    fn quantize(&self, coeffs: &[i32; 16], first: usize) -> ([i16; 16], [i16; 16]) {
        let mut levels = [0_i16; 16];
        let mut dequant = [0_i16; 16];
        for n in first..16 {
            let pos = ZIGZAG[n];
            let k = usize::from(n > 0);
            let c = coeffs[pos];
            let level = ((c.abs() * 256 + self.step[k] * self.bias[k]) / (self.step[k] * 256))
                .min(MAX_LEVEL);
            let level = if c < 0 { -level } else { level };
            levels[pos] = level as i16;
            dequant[pos] = (level * self.step[k]) as i16;
        }
        (levels, dequant)
    }
}

/// One segment's quantizers, in the decoder's dequantization.
#[derive(Clone, Copy)]
struct SegmentQuant {
    y: Quant,
    y2: Quant,
    uv: Quant,
    /// Lagrangian multiplier: squared-error units per bit.
    lambda: u32,
}

fn segment_quant(q: i32) -> SegmentQuant {
    let dc = |q: i32| i32::from(DC_Q_LOOKUP[q.clamp(0, 127) as usize]);
    let ac = |q: i32| i32::from(AC_Q_LOOKUP[q.clamp(0, 127) as usize]);
    let y_ac = ac(q);
    SegmentQuant {
        y: Quant {
            step: [dc(q), y_ac],
            bias: [96, 110],
        },
        y2: Quant {
            step: [dc(q) * 2, (ac(q) * 155 / 100).max(8)],
            bias: [96, 108],
        },
        uv: Quant {
            step: [dc(q).min(132), ac(q)],
            bias: [110, 115],
        },
        // ~0.85 step² per bit, the usual high-rate choice; scaled for the
        // 1/256-bit costs below.
        lambda: (y_ac * y_ac * 218 / 256).max(1) as u32,
    }
}

/// libwebp's quality-to-quantizer curve (`QualityToCompression`).
fn quality_to_q(quality: u8) -> i32 {
    let c = f64::from(quality.clamp(1, 100)) / 100.0;
    let linear = if c < 0.75 {
        c * (2.0 / 3.0)
    } else {
        2.0 * c - 1.0
    };
    let v = linear.cbrt();
    ((127.0 * (1.0 - v)).round() as i32).clamp(0, 127)
}

/// What one macroblock was coded as.
#[derive(Clone)]
struct Macroblock {
    y_mode: u8,
    modes: [u8; 16],
    uv_mode: u8,
    segment: u8,
    /// Levels: Y 0..16, U 16..20, V 20..24, Y2 24.
    levels: Box<[[i16; 16]; 25]>,
}

impl Macroblock {
    fn all_zero(&self) -> bool {
        self.levels
            .iter()
            .all(|block| block.iter().all(|&l| l == 0))
    }
}

/// Non-zero contexts, as the decoder tracks them.
#[derive(Clone, Copy, Default)]
struct Contexts {
    left: [bool; 9],
    above: [bool; 9],
}

fn sse4(a: &[[u8; 4]; 4], b: &[[u8; 4]; 4]) -> u32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            (d * d) as u32
        })
        .sum()
}

fn block_at(plane: &[u8], stride: usize, x: usize, y: usize) -> [[u8; 4]; 4] {
    core::array::from_fn(|r| {
        let at = (y + r) * stride + x;
        [plane[at], plane[at + 1], plane[at + 2], plane[at + 3]]
    })
}

fn put_block(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[[u8; 4]; 4]) {
    for (r, row) in block.iter().enumerate() {
        let at = (y + r) * stride + x;
        plane[at..at + 4].copy_from_slice(row);
    }
}

/// Encode `planes` as a VP8 key frame.
///
/// # Errors
///
/// Returns [`PixelsError::Unsupported`] when the modes alone overflow the
/// first partition's 19-bit size, which only an enormous image reaches.
pub fn encode(planes: &Planes, params: Params) -> Result<Vec<u8>> {
    let mb_cols = planes.width.div_ceil(16);
    let mb_rows = planes.height.div_ceil(16);
    let base_q = quality_to_q(params.quality);

    // Segments by activity: busier macroblocks hide quantization, so they
    // take a coarser quantizer and calm ones a finer one.
    let (segment_of, segment_q): (Vec<u8>, [i32; 4]) = if params.segments {
        let stride = mb_cols * 16;
        let activity: Vec<u32> = (0..mb_rows * mb_cols)
            .map(|i| {
                let (mx, my) = (i % mb_cols, i / mb_cols);
                let mut sum = 0_u32;
                for y in 0..16 {
                    for x in 1..16 {
                        let at = (my * 16 + y) * stride + mx * 16 + x;
                        sum += u32::from(planes.y[at].abs_diff(planes.y[at - 1]));
                    }
                }
                sum
            })
            .collect();
        let mut sorted = activity.clone();
        sorted.sort_unstable();
        let cut = |f: usize| sorted[(sorted.len() * f / 4).min(sorted.len() - 1)];
        let cuts = [cut(1), cut(2), cut(3)];
        let segments = activity
            .iter()
            .map(|&a| cuts.iter().filter(|&&c| a > c).count() as u8)
            .collect();
        (segments, [-6, -2, 2, 6])
    } else {
        (vec![0; mb_rows * mb_cols], [0; 4])
    };
    let quants: [SegmentQuant; 4] = core::array::from_fn(|s| {
        segment_quant(base_q + if params.segments { segment_q[s] } else { 0 })
    });

    // Pass 1: decide and reconstruct.
    let mut recon_y = vec![0_u8; planes.y.len()];
    let mut recon_u = vec![0_u8; planes.u.len()];
    let mut recon_v = vec![0_u8; planes.v.len()];
    let mut mbs = Vec::with_capacity(mb_rows * mb_cols);
    let mut above_modes = vec![[b::DC; 4]; mb_cols];
    let mut above_nz = vec![[false; 9]; mb_cols];
    let default_probs: Box<CoeffProbs> = Box::new(DEFAULT_COEFF_PROBS);
    for my in 0..mb_rows {
        let mut left_modes = [b::DC; 4];
        let mut left_nz = [false; 9];
        for mx in 0..mb_cols {
            let segment = segment_of[my * mb_cols + mx];
            let quant = &quants[usize::from(segment)];
            let mut ctx = Contexts {
                left: left_nz,
                above: above_nz[mx],
            };
            let mb = encode_macroblock(
                planes,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
                mb_cols,
                mx,
                my,
                quant,
                &default_probs,
                &mut ctx,
                &above_modes[mx],
                &left_modes,
                segment,
            );
            above_modes[mx] = [mb.modes[12], mb.modes[13], mb.modes[14], mb.modes[15]];
            left_modes = [mb.modes[3], mb.modes[7], mb.modes[11], mb.modes[15]];
            left_nz = ctx.left;
            above_nz[mx] = ctx.above;
            mbs.push(mb);
        }
    }

    // Pass 2: optimise the coefficient probabilities for what was chosen.
    let skip_count = mbs.iter().filter(|mb| mb.all_zero()).count();
    let use_skip = skip_count > 0;
    let mut stats = Stats {
        counts: Box::new([[[[[0; 2]; 11]; 3]; 8]; 4]),
    };
    write_all_tokens(&mut stats, &mbs, mb_cols, &default_probs, use_skip);
    let mut probs = default_probs.clone();
    let mut updates = Vec::new();
    for t in 0..4 {
        for band in 0..8 {
            for c in 0..3 {
                for node in 0..11 {
                    let [zeros, ones] = stats.counts[t][band][c][node];
                    let old = probs[t][band][c][node];
                    let total = zeros + ones;
                    if total == 0 {
                        continue;
                    }
                    let new = ((zeros * 256 + total / 2) / total).clamp(1, 255) as u8;
                    let cost = |p: u8| zeros * bit_cost(p, false) + ones * bit_cost(p, true);
                    let signalling = bit_cost(COEFF_UPDATE_PROBS[t][band][c][node], true) + 8 * 256
                        - bit_cost(COEFF_UPDATE_PROBS[t][band][c][node], false);
                    if cost(old) > cost(new) + signalling {
                        probs[t][band][c][node] = new;
                        updates.push((t, band, c, node));
                    }
                }
            }
        }
    }

    // Loop filter strength from the quantizer: stronger where coarser.
    let filter_level =
        |q: i32| -> i32 { (i32::from(AC_Q_LOOKUP[q.clamp(0, 127) as usize]) * 3 / 8).clamp(0, 63) };
    let level = filter_level(base_q);

    // Partition 0: header and modes.
    let segment_probs = probs_for_segments(&mbs);
    let mut head = BoolEncoder::new();
    head.literal(2, 0); // color space, clamping
    head.put(128, params.segments);
    if params.segments {
        head.put(128, true); // update map
        head.put(128, true); // update data
        head.put(128, false); // relative values
        for q in segment_q {
            head.signed(7, q);
        }
        for q in segment_q {
            // Coarser segments filter harder.
            head.signed(6, q);
        }
        // Segment tree probabilities from the actual distribution.
        for p in segment_probs {
            head.put(128, true);
            head.literal(8, u32::from(p));
        }
    }
    head.put(128, params.simple_filter);
    head.literal(6, level as u32);
    head.literal(3, u32::from(params.sharpness));
    head.put(128, params.filter_deltas);
    if params.filter_deltas {
        head.put(128, true); // update
        for i in 0..4 {
            head.signed(6, if i == 0 { 2 } else { 0 });
        }
        for i in 0..4 {
            head.signed(6, if i == 0 { -2 } else { 0 });
        }
    }
    head.literal(2, u32::from(params.partitions_log2));
    head.literal(7, base_q as u32);
    for _ in 0..5 {
        head.put(128, false); // no quantizer deltas
    }
    head.put(128, false); // refresh_entropy_probs
    for t in 0..4 {
        for band in 0..8 {
            for c in 0..3 {
                for node in 0..11 {
                    let update = updates.contains(&(t, band, c, node));
                    head.put(COEFF_UPDATE_PROBS[t][band][c][node], update);
                    if update {
                        head.literal(8, u32::from(probs[t][band][c][node]));
                    }
                }
            }
        }
    }
    let skip_prob = prob_of((mbs.len() - skip_count) as u32, mbs.len() as u32);
    head.put(128, use_skip);
    if use_skip {
        head.literal(8, u32::from(skip_prob));
    }
    let mut above_modes = vec![[b::DC; 4]; mb_cols];
    for (i, mb) in mbs.iter().enumerate() {
        let mx = i % mb_cols;
        if params.segments {
            let p = segment_probs;
            let s = mb.segment;
            head.put(p[0], s >= 2);
            if s >= 2 {
                head.put(p[2], s == 3);
            } else {
                head.put(p[1], s == 1);
            }
        }
        if use_skip {
            head.put(skip_prob, mb.all_zero());
        }
        head.tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, mb.y_mode);
        if mb.y_mode == B_PRED {
            let left_modes = if mx == 0 {
                [b::DC; 4]
            } else {
                let l = &mbs[i - 1].modes;
                [l[3], l[7], l[11], l[15]]
            };
            for k in 0..16 {
                let above = if k < 4 {
                    above_modes[mx][k]
                } else {
                    mb.modes[k - 4]
                };
                let left = if k & 3 == 0 {
                    left_modes[k >> 2]
                } else {
                    mb.modes[k - 1]
                };
                head.tree(
                    &B_MODE_TREE,
                    &KF_B_MODE_PROBS[usize::from(above)][usize::from(left)],
                    mb.modes[k],
                );
            }
        }
        above_modes[mx] = [mb.modes[12], mb.modes[13], mb.modes[14], mb.modes[15]];
        head.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, mb.uv_mode);
    }
    let first = head.finish();
    if first.len() >= 1 << 19 {
        return Err(PixelsError::unsupported(
            "webp: the image's modes overflow VP8's first partition; encode it lossless or smaller",
        ));
    }

    // Token partitions.
    let count = 1_usize << params.partitions_log2;
    let mut partitions: Vec<BoolEncoder> = (0..count).map(|_| BoolEncoder::new()).collect();
    let mut above_nz = vec![[false; 9]; mb_cols];
    for my in 0..mb_rows {
        let mut left_nz = [false; 9];
        let sink = &mut partitions[my % count];
        for mx in 0..mb_cols {
            let mb = &mbs[my * mb_cols + mx];
            let mut ctx = Contexts {
                left: left_nz,
                above: above_nz[mx],
            };
            write_macroblock_tokens(sink, mb, &probs, &mut ctx, use_skip);
            left_nz = ctx.left;
            above_nz[mx] = ctx.above;
        }
    }
    let partitions: Vec<Vec<u8>> = partitions.into_iter().map(BoolEncoder::finish).collect();

    let mut out =
        Vec::with_capacity(10 + first.len() + partitions.iter().map(Vec::len).sum::<usize>());
    let tag = (first.len() as u32) << 5 | 1 << 4; // key frame, version 0, shown
    out.extend_from_slice(&tag.to_le_bytes()[..3]);
    out.extend_from_slice(&[0x9d, 0x01, 0x2a]);
    out.extend_from_slice(&(planes.width as u16).to_le_bytes());
    out.extend_from_slice(&(planes.height as u16).to_le_bytes());
    out.extend_from_slice(&first);
    for p in &partitions[..partitions.len() - 1] {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes()[..3]);
    }
    for p in &partitions {
        out.extend_from_slice(p);
    }
    Ok(out)
}

/// The probability, out of 256, that an event with `zeros` of `total`
/// occurrences codes as 0.
fn prob_of(zeros: u32, total: u32) -> u8 {
    if total == 0 {
        return 128;
    }
    ((zeros * 256 + total / 2) / total).clamp(1, 255) as u8
}

fn probs_for_segments(mbs: &[Macroblock]) -> [u8; 3] {
    let count = |f: &dyn Fn(u8) -> bool| mbs.iter().filter(|mb| f(mb.segment)).count() as u32;
    let total = mbs.len() as u32;
    let low = count(&|s| s < 2);
    [
        prob_of(low, total),
        prob_of(count(&|s| s == 0), low),
        prob_of(count(&|s| s == 2), total - low),
    ]
}

fn write_all_tokens<S: BoolSink>(
    sink: &mut S,
    mbs: &[Macroblock],
    mb_cols: usize,
    probs: &CoeffProbs,
    use_skip: bool,
) {
    let mut above_nz = vec![[false; 9]; mb_cols];
    let mut left_nz = [false; 9];
    for (i, mb) in mbs.iter().enumerate() {
        let mx = i % mb_cols;
        if mx == 0 {
            left_nz = [false; 9];
        }
        let mut ctx = Contexts {
            left: left_nz,
            above: above_nz[mx],
        };
        write_macroblock_tokens(sink, mb, probs, &mut ctx, use_skip);
        left_nz = ctx.left;
        above_nz[mx] = ctx.above;
    }
}

/// One macroblock's tokens, with the decoder's context updates.
fn write_macroblock_tokens<S: BoolSink>(
    sink: &mut S,
    mb: &Macroblock,
    probs: &CoeffProbs,
    ctx: &mut Contexts,
    use_skip: bool,
) {
    let has_y2 = mb.y_mode != B_PRED;
    if use_skip && mb.all_zero() {
        ctx.left[..8].fill(false);
        ctx.above[..8].fill(false);
        if has_y2 {
            ctx.left[8] = false;
            ctx.above[8] = false;
        }
        return;
    }
    let first_y = if has_y2 {
        let c = usize::from(ctx.left[8]) + usize::from(ctx.above[8]);
        let nz = write_block(sink, probs, 1, 0, c, &mb.levels[24]);
        ctx.left[8] = nz;
        ctx.above[8] = nz;
        1
    } else {
        0
    };
    let y_kind = if has_y2 { 0 } else { 3 };
    for i in 0..16 {
        let (row, col) = (i >> 2, i & 3);
        let c = usize::from(ctx.left[row]) + usize::from(ctx.above[col]);
        let nz = write_block(sink, probs, y_kind, first_y, c, &mb.levels[i]);
        ctx.left[row] = nz;
        ctx.above[col] = nz;
    }
    for i in 16..24 {
        let base = if i < 20 { 4 } else { 6 };
        let k = i & 3;
        let (l, a) = (base + (k >> 1), base + (k & 1));
        let c = usize::from(ctx.left[l]) + usize::from(ctx.above[a]);
        let nz = write_block(sink, probs, 2, 0, c, &mb.levels[i]);
        ctx.left[l] = nz;
        ctx.above[a] = nz;
    }
}

/// A chroma mode's score, mode, levels for U and V, reconstruction, and the
/// contexts it leaves.
type UvCandidate = (u64, u8, [[i16; 16]; 8], [[[u8; 8]; 8]; 2], Contexts);

/// Choose and code one macroblock, writing its reconstruction.
#[allow(
    clippy::too_many_arguments,
    reason = "the frame state one macroblock touches"
)]
fn encode_macroblock(
    planes: &Planes,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
    mb_cols: usize,
    mx: usize,
    my: usize,
    q: &SegmentQuant,
    probs: &CoeffProbs,
    ctx: &mut Contexts,
    above_modes: &[u8; 4],
    left_modes: &[u8; 4],
    segment: u8,
) -> Macroblock {
    let stride = mb_cols * 16;
    let (x0, y0) = (mx * 16, my * 16);
    let lambda = q.lambda;

    // --- 16x16 candidates.
    let edges16 = block_edges::<16>(recon_y, stride, mx, my);
    let try16 = |mode: u8| -> (u64, u8, [[i16; 16]; 25], [[u8; 16]; 16], Contexts) {
        let pred = predict::predict_block(mode, &edges16);
        let mut levels = [[0_i16; 16]; 25];
        let mut dequant = [[0_i16; 16]; 16];
        let mut dcs = [0_i32; 16];
        let mut acs = [[0_i32; 16]; 16];
        for i in 0..16 {
            let (bx, by) = (i & 3, i >> 2);
            let src = block_at(&planes.y, stride, x0 + bx * 4, y0 + by * 4);
            let p: [[u8; 4]; 4] =
                core::array::from_fn(|r| core::array::from_fn(|c| pred[by * 4 + r][bx * 4 + c]));
            let coeffs = forward_dct(&src, &p);
            dcs[i] = coeffs[0];
            acs[i] = coeffs;
        }
        let (y2_levels, y2_dequant) = q.y2.quantize(&forward_wht(&dcs), 0);
        levels[24] = y2_levels;
        let recon_dcs = inverse_wht(&y2_dequant);
        let mut recon = pred;
        let mut sse = 0_u64;
        for i in 0..16 {
            let (bx, by) = (i & 3, i >> 2);
            let (l, d) = q.y.quantize(&acs[i], 1);
            levels[i] = l;
            dequant[i] = d;
            dequant[i][0] = recon_dcs[i];
            let mut block: [[u8; 4]; 4] =
                core::array::from_fn(|r| core::array::from_fn(|c| pred[by * 4 + r][bx * 4 + c]));
            idct_add(&dequant[i], &mut block);
            let src = block_at(&planes.y, stride, x0 + bx * 4, y0 + by * 4);
            sse += u64::from(sse4(&src, &block));
            for r in 0..4 {
                recon[by * 4 + r][bx * 4..bx * 4 + 4].copy_from_slice(&block[r]);
            }
        }
        let mut cost = Cost::default();
        let mut trial = *ctx;
        let c = usize::from(trial.left[8]) + usize::from(trial.above[8]);
        let nz = write_block(&mut cost, probs, 1, 0, c, &levels[24]);
        trial.left[8] = nz;
        trial.above[8] = nz;
        for i in 0..16 {
            let (row, col) = (i >> 2, i & 3);
            let c = usize::from(trial.left[row]) + usize::from(trial.above[col]);
            let nz = write_block(&mut cost, probs, 0, 1, c, &levels[i]);
            trial.left[row] = nz;
            trial.above[col] = nz;
        }
        cost.tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, mode);
        let score = sse * 256 + u64::from(lambda) * u64::from(cost.0);
        (score, mode, levels, recon, trial)
    };
    let (score16, mode16, levels16, recon16, ctx16) = [V_PRED, H_PRED, TM_PRED]
        .into_iter()
        .map(try16)
        .fold(
            try16(DC_PRED),
            |best, c| if c.0 < best.0 { c } else { best },
        );

    // --- 4x4 candidates, subblock by subblock, reconstructing as we go.
    let mut levels4 = [[0_i16; 16]; 25];
    let mut modes4 = [b::DC; 16];
    let mut ctx4 = *ctx;
    let mut score4 = 0_u64;
    {
        let mut cost = Cost::default();
        cost.tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS, B_PRED);
        score4 += u64::from(lambda) * u64::from(cost.0);
    }
    for i in 0..16 {
        let (bx, by) = (i & 3, i >> 2);
        let edges = sub_edges(recon_y, mb_cols, mx, my, bx, by);
        let src = block_at(&planes.y, stride, x0 + bx * 4, y0 + by * 4);
        let above = if i < 4 { above_modes[i] } else { modes4[i - 4] };
        let left = if i & 3 == 0 {
            left_modes[i >> 2]
        } else {
            modes4[i - 1]
        };
        let (row, col) = (by, bx);
        let c = usize::from(ctx4.left[row]) + usize::from(ctx4.above[col]);
        let try4 = |mode: u8| -> (u64, u8, [i16; 16], [[u8; 4]; 4], bool) {
            let pred = predict::predict_subblock(mode, &edges);
            let (l, d) = q.y.quantize(&forward_dct(&src, &pred), 0);
            let mut block = pred;
            idct_add(&d, &mut block);
            let mut cost = Cost::default();
            let nz = write_block(&mut cost, probs, 3, 0, c, &l);
            cost.tree(
                &B_MODE_TREE,
                &KF_B_MODE_PROBS[usize::from(above)][usize::from(left)],
                mode,
            );
            let score = u64::from(sse4(&src, &block)) * 256 + u64::from(lambda) * u64::from(cost.0);
            (score, mode, l, block, nz)
        };
        let (score, mode, l, block, nz) = (1..10_u8)
            .map(try4)
            .fold(try4(b::DC), |best, c| if c.0 < best.0 { c } else { best });
        score4 += score;
        modes4[i] = mode;
        levels4[i] = l;
        ctx4.left[row] = nz;
        ctx4.above[col] = nz;
        put_block(recon_y, stride, x0 + bx * 4, y0 + by * 4, &block);
        if score4 >= score16 {
            break; // already worse than 16x16
        }
    }

    let (y_mode, modes, mut levels, mut chosen_ctx) = if score4 < score16 {
        (B_PRED, modes4, levels4, ctx4)
    } else {
        for (r, row) in recon16.iter().enumerate() {
            let at = (y0 + r) * stride + x0;
            recon_y[at..at + 16].copy_from_slice(row);
        }
        let implied = match mode16 {
            V_PRED => b::VE,
            H_PRED => b::HE,
            TM_PRED => b::TM,
            _ => b::DC,
        };
        (mode16, [implied; 16], levels16, ctx16)
    };

    // --- chroma.
    let uv_stride = mb_cols * 8;
    let edges_u = block_edges::<8>(recon_u, uv_stride, mx, my);
    let edges_v = block_edges::<8>(recon_v, uv_stride, mx, my);
    let try_uv = |mode: u8| -> UvCandidate {
        let mut trial = chosen_ctx;
        let mut sse = 0_u64;
        let mut cost = Cost::default();
        let mut uv_levels = [[0_i16; 16]; 8];
        let mut recon = [[[0_u8; 8]; 8]; 2];
        for (p, (edges, plane)) in [(&edges_u, &planes.u), (&edges_v, &planes.v)]
            .into_iter()
            .enumerate()
        {
            let pred = predict::predict_block(mode, edges);
            let mut out = pred;
            for k in 0..4 {
                let (bx, by) = (k & 1, k >> 1);
                let src = block_at(plane, uv_stride, mx * 8 + bx * 4, my * 8 + by * 4);
                let pb: [[u8; 4]; 4] = core::array::from_fn(|r| {
                    core::array::from_fn(|c| pred[by * 4 + r][bx * 4 + c])
                });
                let (l, d) = q.uv.quantize(&forward_dct(&src, &pb), 0);
                let mut block = pb;
                idct_add(&d, &mut block);
                sse += u64::from(sse4(&src, &block));
                let base = if p == 0 { 4 } else { 6 };
                let (li, ai) = (base + (k >> 1), base + (k & 1));
                let c = usize::from(trial.left[li]) + usize::from(trial.above[ai]);
                let nz = write_block(&mut cost, probs, 2, 0, c, &l);
                trial.left[li] = nz;
                trial.above[ai] = nz;
                uv_levels[p * 4 + k] = l;
                for r in 0..4 {
                    out[by * 4 + r][bx * 4..bx * 4 + 4].copy_from_slice(&block[r]);
                }
            }
            recon[p] = out;
        }
        cost.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS, mode);
        let score = sse * 256 + u64::from(lambda) * u64::from(cost.0);
        (score, mode, uv_levels, recon, trial)
    };
    let (_, uv_mode, uv_levels, uv_recon, uv_ctx) = [V_PRED, H_PRED, TM_PRED]
        .into_iter()
        .map(try_uv)
        .fold(
            try_uv(DC_PRED),
            |best, c| if c.0 < best.0 { c } else { best },
        );
    for (p, plane) in [recon_u, recon_v].into_iter().enumerate() {
        for (r, row) in uv_recon[p].iter().enumerate() {
            let at = (my * 8 + r) * uv_stride + mx * 8;
            plane[at..at + 8].copy_from_slice(row);
        }
    }
    levels[16..24].copy_from_slice(&uv_levels);
    chosen_ctx = uv_ctx;

    let mb = Macroblock {
        y_mode,
        modes,
        uv_mode,
        segment,
        levels: Box::new(levels),
    };
    // A macroblock with nothing to code is skipped, which resets the
    // contexts it would otherwise have set — mirror that now.
    if mb.all_zero() {
        chosen_ctx.left[..8].fill(false);
        chosen_ctx.above[..8].fill(false);
        if y_mode != B_PRED {
            chosen_ctx.left[8] = false;
            chosen_ctx.above[8] = false;
        }
    }
    *ctx = chosen_ctx;
    mb
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    fn planes(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Planes {
        let (w, h) = (width.div_ceil(16) * 16, height.div_ceil(16) * 16);
        let y = (0..w * h).map(|i| f(i % w, i / w)).collect();
        Planes {
            y,
            u: vec![128; w * h / 4],
            v: vec![128; w * h / 4],
            width,
            height,
        }
    }

    fn luma_error(p: &Planes, quality: u8, params: impl Fn(&mut Params)) -> (u32, f64) {
        let mut prm = Params::with_quality(quality);
        params(&mut prm);
        let frame = crate::vp8::decode(&encode(p, prm).unwrap()).unwrap();
        let stride = frame.y_stride;
        let (mut max, mut se) = (0_u32, 0_f64);
        for y in 0..p.height {
            for x in 0..p.width {
                let d = u32::from(frame.y[y * stride + x].abs_diff(p.y[y * stride + x]));
                max = max.max(d);
                se += f64::from(d * d);
            }
        }
        (max, se / (p.width * p.height) as f64)
    }

    /// Writes one lossy WebP per encoder option, each with our own decode of
    /// it, so `scripts/check-webp-interop.sh` can have libwebp decode the same
    /// file and demand the identical result. These options reach decoder
    /// paths no file written through Pillow does: several token partitions,
    /// the simple loop filter, sharpness, relative segment values and filter
    /// deltas. Inert unless `OTF_EMIT_DIR` is set.
    #[test]
    fn emit_option_variants_for_external_verification() {
        let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
            return;
        };
        let (w, h) = (83_usize, 61_usize);
        let rgb: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                let ring = ((x as f64 - 40.0).hypot(y as f64 - 30.0) * 0.4).sin();
                [(x * 3) as u8, (128.0 + 100.0 * ring) as u8, (y * 4) as u8]
            })
            .collect();
        type Variant = (&'static str, fn(&mut Params));
        let variants: [Variant; 8] = [
            ("plain", |_| {}),
            ("partitions2", |p| p.partitions_log2 = 1),
            ("partitions8", |p| p.partitions_log2 = 3),
            ("simple_filter", |p| p.simple_filter = true),
            ("sharpness5", |p| p.sharpness = 5),
            ("segments", |p| p.segments = true),
            ("filter_deltas", |p| p.filter_deltas = true),
            ("everything", |p| {
                p.partitions_log2 = 2;
                p.simple_filter = true;
                p.sharpness = 3;
                p.segments = true;
                p.filter_deltas = true;
            }),
        ];
        // A flat image too, whose empty macroblocks take the skip flag.
        let flat: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                if i % w < w / 2 {
                    [40, 90, 160]
                } else {
                    [200, 180, 20]
                }
            })
            .collect();
        for (image, quality) in [(&rgb, 20_u8), (&rgb, 75), (&flat, 50)] {
            for (name, set) in variants {
                let rgb = image;
                let (planes, _) = crate::yuv::from_rgb(rgb, 3, w, h);
                let mut params = Params::with_quality(quality);
                set(&mut params);
                let vp8 = encode(&planes, params).unwrap();
                let mut file = b"RIFF".to_vec();
                file.extend_from_slice(
                    &(vp8.len() as u32 + 12 + (vp8.len() as u32 & 1)).to_le_bytes(),
                );
                file.extend_from_slice(b"WEBPVP8 ");
                file.extend_from_slice(&(vp8.len() as u32).to_le_bytes());
                file.extend_from_slice(&vp8);
                if vp8.len() % 2 == 1 {
                    file.push(0);
                }
                let frame = crate::vp8::decode(&vp8).unwrap();
                let ours = crate::yuv::to_rgb(
                    &frame.y,
                    frame.y_stride,
                    &frame.u,
                    &frame.v,
                    frame.uv_stride,
                    w,
                    h,
                    None,
                );
                let flatness = if std::ptr::eq(image, &flat) {
                    "_flat"
                } else {
                    ""
                };
                let base = format!("{dir}/vp8_{name}{flatness}_q{quality}");
                std::fs::write(format!("{base}.webp"), &file).unwrap();
                std::fs::write(format!("{base}.ours"), &ours).unwrap();
                std::fs::write(format!("{base}.src"), rgb).unwrap();
            }
        }
    }

    #[test]
    fn high_quality_reconstructs_closely() {
        let p = planes(48, 32, |x, y| (x * 5 + y * 3) as u8);
        let (max, mse) = luma_error(&p, 100, |_| {});
        assert!(max <= 4 && mse < 2.0, "max {max} mse {mse}");
    }
}
