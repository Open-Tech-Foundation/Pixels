//! Lossy VarDCT frames (ISO/IEC 18181-1 §I).
//!
//! The image is converted to XYB, cut into 8x8 blocks and transformed by
//! the DCT. Each block's DC — its *LF* — is quantized and coded as a
//! Modular image, an eighth of the size, alongside the block's metadata;
//! the rest — its *HF* — is quantized against the default tables and coded
//! by count of nonzeros and then in zig-zag order, with the decoder's
//! contexts.
//!
//! This is the smallest useful subset of VarDCT: every block is DCT8, one
//! quantizer for the whole frame, the default tables and block contexts,
//! no chroma-from-luma beyond the default B-from-Y, and no adaptive LF
//! smoothing, Gaborish or edge-preserving filter.

#![allow(
    clippy::indexing_slicing,
    reason = "block and coefficient indices are within the padded planes and \
              the 64-entry tables they were sized from"
)]

use std::f32::consts::{PI, SQRT_2};

use crate::bits::{BitWriter, bits, val};
use crate::entropy::{EntropyCode, HybridUint, Token, with_runs};
use crate::headers::{ImageInfo, write_image_header, write_toc, write_vardct_frame_header};
use crate::modular::{Channel, Tree, residuals, write_header};
use crate::predict::{Predictor, Region};

/// Pixels per side of a group, and of an LF group.
const GROUP: usize = 256;
const LF_GROUP: usize = 2048;
/// Blocks per side of a group.
const GROUP_BLOCKS: usize = GROUP / 8;

/// The opsin absorbance matrix and bias XYB is built on.
const OPSIN: [[f32; 3]; 3] = [
    [0.30, 0.622, 0.078],
    [0.23, 0.692, 0.078],
    [0.243_422_69, 0.204_767_44, 0.551_809_87],
];
const OPSIN_BIAS: f32 = 0.003_793_073_3;

/// The decoder's LF dequantization multipliers for X, Y and B.
const M_LF: [f32; 3] = [1.0 / 32.0, 1.0 / 4.0, 1.0 / 2.0];
/// The quantization matrix scale for X, Y, B at the default
/// `x_qm_scale` 3 and `b_qm_scale` 2.
const QM_SCALE: [f32; 3] = [0.8, 1.0, 1.0];
/// The default DCT8 band parameters for X, Y and B.
const DCT8_PARAMS: [[f32; 6]; 3] = [
    [3150.0, 0.0, -0.4, -0.4, -0.4, -2.0],
    [560.0, 0.0, -0.3, -0.3, -0.3, -0.3],
    [512.0, -2.0, -1.0, 0.0, -1.0, -2.0],
];

/// The default block contexts' map, for (channel order Y, X, B) x 13
/// orders; DCT8 is order 0.
const BLOCK_CTX: [u32; 3] = [0, 7, 7];
const BLOCK_CLUSTERS: u32 = 15;
const COEFF_FREQ_CONTEXT: [u32; 63] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20,
    20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27,
    27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const COEFF_NUM_NONZERO_CONTEXT: [u32; 63] = [
    0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152, 152, 180,
    180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206, 206, 206, 206,
];

/// The DCT8 coefficient order, zig-zag, as the decoder's `(dx, dy)` pairs.
/// The decoder's coefficient grid is transposed against our row-major
/// blocks: an order entry `(a, b)` is our coefficient `[a * 8 + b]`,
/// vertical frequency `a`, horizontal `b`.
fn natural_order() -> [(usize, usize); 64] {
    let mut out = [(0, 0); 64];
    let mut idx = 1;
    for dist in 1_usize..16 {
        let margin = dist.saturating_sub(8);
        for order in margin..dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            if x == 0 && y == 0 {
                continue;
            }
            out[idx] = (x, y);
            idx += 1;
        }
    }
    out
}

/// The decoder's dequantization weights for DCT8: `1 / weight` multiplies
/// a quantized coefficient. Row-major, `[channel][y * 8 + x]`.
fn dct8_weights() -> [[f32; 64]; 3] {
    let mult = |x: f32| if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) };
    let mut out = [[0.0; 64]; 3];
    for (c, params) in DCT8_PARAMS.iter().enumerate() {
        let mut bands = [0.0_f32; 6];
        bands[0] = params[0];
        for i in 1..6 {
            bands[i] = bands[i - 1] * mult(params[i]);
        }
        for y in 0..8 {
            for x in 0..8 {
                let (dx, dy) = (x as f32 / 7.0, y as f32 / 7.0);
                let pos = (dx * dx + dy * dy).sqrt() * 5.0 / (SQRT_2 + 1e-6);
                let i = pos as usize;
                let frac = pos - i as f32;
                let weight = if i + 1 < 6 {
                    bands[i] * (bands[i + 1] / bands[i]).powf(frac)
                } else {
                    bands[5]
                };
                out[c][y * 8 + x] = weight;
            }
        }
    }
    out
}

/// The 8-point DCT the decoder inverts: `X_k = 1/8 sum x_n cos(...)`, AC
/// terms scaled by the square root of two.
fn dct8(input: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0; 8];
    for (k, o) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (n, &x) in input.iter().enumerate() {
            sum += x * ((k * (2 * n + 1)) as f32 * PI / 16.0).cos();
        }
        *o = sum / 8.0 * if k > 0 { SQRT_2 } else { 1.0 };
    }
    out
}

/// The 2-D DCT of an 8x8 block, `[y * 8 + x]` in both: rows, then columns.
fn dct8x8(block: &[f32; 64]) -> [f32; 64] {
    let mut rows = [0.0; 64];
    for y in 0..8 {
        let row: [f32; 8] = std::array::from_fn(|x| block[y * 8 + x]);
        let t = dct8(&row);
        rows[y * 8..y * 8 + 8].copy_from_slice(&t);
    }
    let mut out = [0.0; 64];
    for x in 0..8 {
        let col: [f32; 8] = std::array::from_fn(|y| rows[y * 8 + x]);
        let t = dct8(&col);
        for y in 0..8 {
            out[y * 8 + x] = t[y];
        }
    }
    out
}

/// sRGB to linear light.
fn linear(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// An image as XYB planes, padded with its edge to whole blocks.
struct Xyb {
    /// Blocks across and down.
    bw: usize,
    bh: usize,
    /// X, Y and B, each `(bw * 8) x (bh * 8)`.
    planes: [Vec<f32>; 3],
}

impl Xyb {
    /// Convert `rgb` (0..=1 non-linear sRGB, `width * height` triples).
    fn new(rgb: &[[f32; 3]], width: usize, height: usize) -> Self {
        let (bw, bh) = (width.div_ceil(8), height.div_ceil(8));
        let stride = bw * 8;
        let mut planes = [
            vec![0.0; stride * bh * 8],
            vec![0.0; stride * bh * 8],
            vec![0.0; stride * bh * 8],
        ];
        let bias_cbrt = OPSIN_BIAS.cbrt();
        for y in 0..bh * 8 {
            for x in 0..stride {
                let [r, g, b] = rgb[y.min(height - 1) * width + x.min(width - 1)].map(linear);
                let mix = OPSIN.map(|row| row[0] * r + row[1] * g + row[2] * b + OPSIN_BIAS);
                let [l, m, s] = mix.map(|v| v.max(0.0).cbrt() - bias_cbrt);
                let at = y * stride + x;
                planes[0][at] = (l - m) / 2.0;
                planes[1][at] = (l + m) / 2.0;
                planes[2][at] = s;
            }
        }
        Self { bw, bh, planes }
    }

    /// The DCT of block `(bx, by)` of channel `c`.
    fn block(&self, c: usize, bx: usize, by: usize) -> [f32; 64] {
        let stride = self.bw * 8;
        let mut block = [0.0; 64];
        for y in 0..8 {
            for x in 0..8 {
                block[y * 8 + x] = self.planes[c][(by * 8 + y) * stride + bx * 8 + x];
            }
        }
        dct8x8(&block)
    }
}

/// How hard to quantize: the frame's quantizer and each block's multiplier.
#[derive(Debug, Clone, Copy)]
struct Quantizer {
    global_scale: u32,
    quant_lf: u32,
    hf_mul: u32,
}

impl Quantizer {
    /// The quantizer for a Butteraugli-style `distance` (1 is visually
    /// lossless, larger is coarser).
    fn for_distance(distance: f32) -> Self {
        let distance = distance.clamp(0.05, 25.0);
        // The HF step of the lowest Y frequency grows with distance; the LF
        // step is kept finer, as banding in flat areas shows first.
        let y_step = 0.0026 * distance;
        let global_scale = ((65_536.0 / (560.0 * y_step)) as u32).clamp(1, 73_727);
        let lf_step = 0.0011 * distance.sqrt();
        let quant_lf = ((128.0 / (global_scale as f32 * lf_step)) as u32).clamp(1, 65_536);
        Self {
            global_scale,
            quant_lf,
            hf_mul: 1,
        }
    }

    fn lf_step(&self, c: usize) -> f32 {
        M_LF[c] * 512.0 / (self.global_scale as f32 * self.quant_lf as f32)
    }

    fn hf_scale(&self, c: usize) -> f32 {
        65_536.0 / (self.global_scale as f32 * self.hf_mul as f32) * QM_SCALE[c]
    }

    fn write(&self, w: &mut BitWriter) {
        w.u32(
            self.global_scale,
            [bits(1, 11), bits(2049, 11), bits(4097, 12), bits(8193, 16)],
        );
        w.u32(
            self.quant_lf,
            [val(16), bits(1, 5), bits(1, 8), bits(1, 16)],
        );
    }
}

/// Quantize with a small dead zone, which costs little and saves many
/// near-zero coefficients.
fn quantize(value: f32, step: f32) -> i32 {
    let q = value / step;
    let rounded = (q.abs() + 0.4).floor();
    if q < 0.0 {
        -rounded as i32
    } else {
        rounded as i32
    }
}

/// Every block's quantized LF and HF, `[c][block]`, for X, Y and B (B
/// with Y subtracted, as the decoder adds it back).
struct Quantized {
    lf: [Vec<i32>; 3],
    hf: [Vec<[i32; 64]>; 3],
}

fn quantize_image(xyb: &Xyb, quantizer: &Quantizer) -> Quantized {
    let weights = dct8_weights();
    let blocks = xyb.bw * xyb.bh;
    let mut lf = [vec![0; blocks], vec![0; blocks], vec![0; blocks]];
    let mut hf = [
        vec![[0; 64]; blocks],
        vec![[0; 64]; blocks],
        vec![[0; 64]; blocks],
    ];
    for by in 0..xyb.bh {
        for bx in 0..xyb.bw {
            let i = by * xyb.bw + bx;
            let y = xyb.block(1, bx, by);
            for c in 0..3 {
                let mut coeffs = if c == 1 { y } else { xyb.block(c, bx, by) };
                if c == 2 {
                    for (b, y) in coeffs.iter_mut().zip(&y) {
                        *b -= y;
                    }
                }
                lf[c][i] = quantize(coeffs[0], quantizer.lf_step(c));
                let scale = quantizer.hf_scale(c);
                for k in 1..64 {
                    hf[c][i][k] = quantize(coeffs[k], scale / weights[c][k]);
                }
            }
        }
    }
    Quantized { lf, hf }
}

/// The HF tokens of the blocks `[bx0, bx0 + w) x [by0, by0 + h)`, as a pass
/// group codes them.
fn hf_tokens(
    q: &Quantized,
    bw: usize,
    (bx0, by0, w, h): (usize, usize, usize, usize),
) -> Vec<Token> {
    let order = natural_order();
    let mut tokens = Vec::new();
    // Each channel's nonzero counts of the row above, overwritten in turn.
    let mut nonzeros_row = [vec![0_u32; w], vec![0_u32; w], vec![0_u32; w]];
    for y in 0..h {
        for x in 0..w {
            let i = (by0 + y) * bw + bx0 + x;
            for (loop_c, c) in [1_usize, 0, 2].into_iter().enumerate() {
                let block_ctx = BLOCK_CTX[loop_c];
                let row = &mut nonzeros_row[c];
                let predicted = match (x, y) {
                    (0, 0) => 32,
                    (_, 0) => row[x - 1],
                    (0, _) => row[x],
                    _ => (row[x] + row[x - 1] + 1) >> 1,
                };
                let bucket = if predicted >= 8 {
                    4 + predicted / 2
                } else {
                    predicted
                };
                let coeffs = &q.hf[c][i];
                let nonzeros = order[1..]
                    .iter()
                    .filter(|&&(fx, fy)| coeffs[fx * 8 + fy] != 0)
                    .count() as u32;
                tokens.push(Token::new(block_ctx + bucket * BLOCK_CLUSTERS, nonzeros));
                row[x] = nonzeros;
                if nonzeros == 0 {
                    continue;
                }
                let base = 37 * BLOCK_CLUSTERS + block_ctx * 458;
                let mut left = nonzeros;
                let mut prev_nonzero = u32::from(nonzeros <= 4);
                for (k, &(fx, fy)) in order[1..].iter().enumerate() {
                    let ctx = (COEFF_NUM_NONZERO_CONTEXT[(left - 1) as usize]
                        + COEFF_FREQ_CONTEXT[k])
                        * 2
                        + prev_nonzero;
                    let value = coeffs[fx * 8 + fy];
                    tokens.push(Token::signed(base + ctx, value));
                    prev_nonzero = u32::from(value != 0);
                    if value != 0 {
                        left -= 1;
                        if left == 0 {
                            break;
                        }
                    }
                }
            }
        }
    }
    tokens
}

/// The LF group at `(lx, ly)` in blocks, of `(w, h)` blocks: its LF image
/// tokens and its metadata tokens, each one Modular stream.
fn lf_group_streams(
    q: &Quantized,
    bw: usize,
    (bx0, by0, w, h): (usize, usize, usize, usize),
    quantizer: &Quantizer,
    tree: &Tree,
) -> (Vec<Token>, Vec<Token>) {
    // The LF image: Y, X, B.
    let mut lf_tokens = Vec::new();
    for (index, c) in [1_usize, 0, 2].into_iter().enumerate() {
        let mut channel = Channel::new(w, h);
        for y in 0..h {
            for x in 0..w {
                channel.samples[y * w + x] = q.lf[c][(by0 + y) * bw + bx0 + x];
            }
        }
        residuals(
            &Region {
                channel: &channel,
                x0: 0,
                y0: 0,
                width: w,
                height: h,
            },
            index as u32,
            tree,
            &mut lf_tokens,
        );
    }
    // The metadata: no chroma from luma, every block DCT8 at one
    // multiplier, no EPF sharpness.
    let cfl = Channel::new(w.div_ceil(8), h.div_ceil(8));
    let mut info = Channel::new(w * h, 2);
    for b in 0..w * h {
        info.samples[w * h + b] = quantizer.hf_mul as i32 - 1;
    }
    let sharpness = Channel::new(w, h);
    let mut meta_tokens = Vec::new();
    for (index, channel) in [&cfl, &cfl, &info, &sharpness].into_iter().enumerate() {
        residuals(
            &Region {
                channel,
                x0: 0,
                y0: 0,
                width: channel.width,
                height: channel.height,
            },
            index as u32,
            tree,
            &mut meta_tokens,
        );
    }
    (with_runs(&lf_tokens, 0), with_runs(&meta_tokens, 0))
}

/// The whole codestream of a VarDCT image from non-linear sRGB samples in
/// 0..=1, at `distance`.
pub fn encode(rgb: &[[f32; 3]], info: &ImageInfo, distance: f32) -> Vec<u8> {
    let (width, height) = (info.width as usize, info.height as usize);
    let xyb = Xyb::new(rgb, width, height);
    let quantizer = Quantizer::for_distance(distance);
    let q = quantize_image(&xyb, &quantizer);
    let (bw, bh) = (xyb.bw, xyb.bh);

    // Modular streams (LF and metadata) share one tree and one code.
    let tree = Tree::per_channel(4, Predictor::Gradient);
    let lf_groups: Vec<(usize, usize, usize, usize)> = (0..height.div_ceil(LF_GROUP))
        .flat_map(|ly| {
            (0..width.div_ceil(LF_GROUP)).map(move |lx| {
                let (bx0, by0) = (lx * LF_GROUP / 8, ly * LF_GROUP / 8);
                (
                    bx0,
                    by0,
                    (LF_GROUP / 8).min(bw - bx0),
                    (LF_GROUP / 8).min(bh - by0),
                )
            })
        })
        .collect();
    let streams: Vec<(Vec<Token>, Vec<Token>)> = lf_groups
        .iter()
        .map(|&g| lf_group_streams(&q, bw, g, &quantizer, &tree))
        .collect();
    let all_modular: Vec<Token> = streams
        .iter()
        .flat_map(|(a, b)| a.iter().chain(b))
        .copied()
        .collect();
    let tree_tokens = tree.tokens();
    let tree_code = EntropyCode::build(6, &tree_tokens, HybridUint::BUCKETS);
    let modular_code = EntropyCode::build(tree.leaves(), &all_modular, HybridUint::DEFAULT);

    // HF, per group.
    let groups: Vec<(usize, usize, usize, usize)> = (0..height.div_ceil(GROUP))
        .flat_map(|gy| {
            (0..width.div_ceil(GROUP)).map(move |gx| {
                let (bx0, by0) = (gx * GROUP_BLOCKS, gy * GROUP_BLOCKS);
                (
                    bx0,
                    by0,
                    GROUP_BLOCKS.min(bw - bx0),
                    GROUP_BLOCKS.min(bh - by0),
                )
            })
        })
        .collect();
    let hf: Vec<Vec<Token>> = groups.iter().map(|&g| hf_tokens(&q, bw, g)).collect();
    let all_hf: Vec<Token> = hf.iter().flatten().copied().collect();
    let hf_code = EntropyCode::build(495 * BLOCK_CLUSTERS as usize, &all_hf, HybridUint::DEFAULT);

    // LfGlobal.
    let mut lf_global = BitWriter::new();
    lf_global.bool(true); // default LF dequantization
    quantizer.write(&mut lf_global);
    lf_global.bool(true); // default block contexts
    lf_global.bool(true); // default channel correlation
    lf_global.bool(true); // a global tree follows
    tree_code.write_header(&mut lf_global);
    for &token in &tree_tokens {
        tree_code.write(&mut lf_global, token);
    }
    modular_code.write_header(&mut lf_global);

    // LF groups.
    let mut lf_sections = Vec::with_capacity(lf_groups.len());
    for (&(_, _, w, h), (lf_tokens, meta_tokens)) in lf_groups.iter().zip(&streams) {
        let mut s = BitWriter::new();
        s.write(2, 0); // no extra LF precision
        write_header(&mut s, false);
        for &token in lf_tokens {
            modular_code.write(&mut s, token);
        }
        let blocks = w * h;
        s.write(
            blocks.next_power_of_two().trailing_zeros(),
            (blocks - 1) as u64,
        );
        write_header(&mut s, false);
        for &token in meta_tokens {
            modular_code.write(&mut s, token);
        }
        lf_sections.push(s);
    }

    // HfGlobal: default matrices, one preset, natural orders.
    let mut hf_global = BitWriter::new();
    hf_global.bool(true);
    hf_global.write(groups.len().next_power_of_two().trailing_zeros(), 0);
    hf_global.u32(0, [val(0x5F), val(0x13), val(0), bits(0, 13)]);
    hf_code.write_header(&mut hf_global);

    let mut pass_sections = Vec::with_capacity(groups.len());
    for tokens in &hf {
        let mut s = BitWriter::new();
        for &token in tokens {
            hf_code.write(&mut s, token);
        }
        pass_sections.push(s);
    }

    let mut out = BitWriter::new();
    write_image_header(&mut out, info);
    write_vardct_frame_header(&mut out, info);
    if groups.len() == 1 {
        // One section: everything, bit after bit.
        let mut all = lf_global;
        for s in lf_sections.iter().chain([&hf_global]).chain(&pass_sections) {
            all.append(s);
        }
        let all = all.finish();
        write_toc(&mut out, &[all.len()]);
        let mut bytes = out.finish();
        bytes.extend_from_slice(&all);
        return bytes;
    }
    let sections: Vec<Vec<u8>> = std::iter::once(lf_global)
        .chain(lf_sections)
        .chain([hf_global])
        .chain(pass_sections)
        .map(BitWriter::finish)
        .collect();
    let sizes: Vec<usize> = sections.iter().map(Vec::len).collect();
    write_toc(&mut out, &sizes);
    let mut bytes = out.finish();
    for s in &sections {
        bytes.extend_from_slice(s);
    }
    bytes
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    #[test]
    fn the_order_is_jpegs_zigzag() {
        let order = natural_order();
        assert_eq!(
            &order[..6],
            &[(0, 0), (1, 0), (0, 1), (0, 2), (1, 1), (2, 0)]
        );
        let mut seen = order.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 64);
    }

    #[test]
    fn a_flat_block_is_all_dc() {
        let coeffs = dct8x8(&[0.25; 64]);
        assert!((coeffs[0] - 0.25).abs() < 1e-6);
        assert!(coeffs[1..].iter().all(|c| c.abs() < 1e-6));
    }

    #[test]
    fn the_weights_start_at_the_first_band() {
        let w = dct8_weights();
        assert!((w[0][0] - 3150.0).abs() < 1e-3);
        assert!((w[1][0] - 560.0).abs() < 1e-3);
        assert!(
            w[1][63] < w[1][0],
            "higher frequencies are quantized harder"
        );
    }

    #[test]
    fn white_and_black_are_grey_in_xyb() {
        let xyb = Xyb::new(&[[1.0; 3], [0.0; 3]], 2, 1);
        assert!(xyb.planes[0][0].abs() < 1e-3, "white has no X");
        assert!(xyb.planes[1][0] > 0.8);
        assert!(xyb.planes[1][1].abs() < 1e-3, "black is zero");
    }
}
