//! VarDCT decoding (ISO/IEC 18181-1 §C.4.4, §C.6, §C.8, §I): LF
//! coefficients, block metadata, dequantization matrices, coefficient
//! orders, HF coefficients and the inverse transforms of every block type.
//! Follows the procedure as jxl-oxide (MIT/Apache-2.0) implements it; see
//! NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "block and coefficient indices are checked against the frame's block grid"
)]
#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "the specification's constants, as written"
)]

use otf_pixels_core::{PixelsError, Result};

use super::coding::{Decoder, read_permutation};
use super::frame::FramePlanes;
use super::header::{FrameHeader, ImageHeader};
use super::modular::{Shape, Tree, decode_stream};
use super::reader::{BitReader, malformed, unpack_signed};
use crate::bits::{bits, val};

/// The block context model of LfGlobal.
#[derive(Debug, Clone)]
struct BlockContext {
    qf_thresholds: Vec<u32>,
    lf_thresholds: [Vec<i32>; 3],
    map: Vec<u8>,
    clusters: u32,
}

/// Chroma-from-luma parameters.
#[derive(Debug, Clone, Copy)]
struct Correlation {
    colour_factor: u32,
    base_x: f32,
    base_b: f32,
    x_factor_lf: u32,
    b_factor_lf: u32,
}

/// LfGlobal's VarDCT part.
#[derive(Debug, Clone)]
pub struct Global {
    global_scale: u32,
    quant_lf: u32,
    block_ctx: BlockContext,
    cfl: Correlation,
}

impl Global {
    /// The base chroma-from-luma correlations, X and B.
    #[must_use]
    pub const fn base_correlation(&self) -> (f32, f32) {
        (self.cfl.base_x, self.cfl.base_b)
    }

    /// # Errors
    ///
    /// For malformed data.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        let global_scale = r.u32([bits(1, 11), bits(2049, 11), bits(4097, 12), bits(8193, 16)])?;
        let quant_lf = r.u32([val(16), bits(1, 5), bits(1, 8), bits(1, 16)])?;
        let block_ctx = if r.bool()? {
            BlockContext {
                qf_thresholds: Vec::new(),
                lf_thresholds: [Vec::new(), Vec::new(), Vec::new()],
                map: vec![
                    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14,
                    14, 14, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,
                ],
                clusters: 15,
            }
        } else {
            let mut size = 1;
            let mut lf_thresholds = [Vec::new(), Vec::new(), Vec::new()];
            for t in &mut lf_thresholds {
                let n = r.read(4)?;
                size *= n + 1;
                for _ in 0..n {
                    t.push(unpack_signed(r.u32([
                        bits(0, 4),
                        bits(16, 8),
                        bits(272, 16),
                        bits(65808, 32),
                    ])?));
                }
            }
            let n = r.read(4)?;
            size *= n + 1;
            let qf_thresholds = (0..n)
                .map(|_| {
                    r.u32([bits(0, 2), bits(4, 3), bits(12, 5), bits(44, 8)])
                        .map(|t| t + 1)
                })
                .collect::<Result<Vec<_>>>()?;
            if size > 64 {
                return Err(malformed("too many block contexts"));
            }
            let (clusters, map) = super::coding::read_clusters(r, size * 39)?;
            if clusters > 16 {
                return Err(malformed("too many block context clusters"));
            }
            BlockContext {
                qf_thresholds,
                lf_thresholds,
                map,
                clusters,
            }
        };
        let cfl = if r.bool()? {
            Correlation {
                colour_factor: 84,
                base_x: 0.0,
                base_b: 1.0,
                x_factor_lf: 128,
                b_factor_lf: 128,
            }
        } else {
            Correlation {
                colour_factor: r.u32([val(84), val(256), bits(2, 8), bits(258, 16)])?,
                base_x: r.f16()?,
                base_b: r.f16()?,
                x_factor_lf: r.read(8)?,
                b_factor_lf: r.read(8)?,
            }
        };
        Ok(Self {
            global_scale,
            quant_lf,
            block_ctx,
            cfl,
        })
    }
}

/// The 27 block types, by their number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Kind {
    Dct8 = 0,
    Hornuss,
    Dct2,
    Dct4,
    Dct16,
    Dct32,
    Dct16x8,
    Dct8x16,
    Dct32x8,
    Dct8x32,
    Dct32x16,
    Dct16x32,
    Dct4x8,
    Dct8x4,
    Afv0,
    Afv1,
    Afv2,
    Afv3,
    Dct64,
    Dct64x32,
    Dct32x64,
    Dct128,
    Dct128x64,
    Dct64x128,
    Dct256,
    Dct256x128,
    Dct128x256,
}

const KINDS: [Kind; 27] = [
    Kind::Dct8,
    Kind::Hornuss,
    Kind::Dct2,
    Kind::Dct4,
    Kind::Dct16,
    Kind::Dct32,
    Kind::Dct16x8,
    Kind::Dct8x16,
    Kind::Dct32x8,
    Kind::Dct8x32,
    Kind::Dct32x16,
    Kind::Dct16x32,
    Kind::Dct4x8,
    Kind::Dct8x4,
    Kind::Afv0,
    Kind::Afv1,
    Kind::Afv2,
    Kind::Afv3,
    Kind::Dct64,
    Kind::Dct64x32,
    Kind::Dct32x64,
    Kind::Dct128,
    Kind::Dct128x64,
    Kind::Dct64x128,
    Kind::Dct256,
    Kind::Dct256x128,
    Kind::Dct128x256,
];

impl Kind {
    /// Width and height in 8x8 blocks.
    const fn size(self) -> (usize, usize) {
        use Kind::*;
        match self {
            Dct8 | Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => (1, 1),
            Dct16 => (2, 2),
            Dct32 => (4, 4),
            Dct16x8 => (1, 2),
            Dct8x16 => (2, 1),
            Dct32x8 => (1, 4),
            Dct8x32 => (4, 1),
            Dct32x16 => (2, 4),
            Dct16x32 => (4, 2),
            Dct64 => (8, 8),
            Dct64x32 => (4, 8),
            Dct32x64 => (8, 4),
            Dct128 => (16, 16),
            Dct128x64 => (8, 16),
            Dct64x128 => (16, 8),
            Dct256 => (32, 32),
            Dct256x128 => (16, 32),
            Dct128x256 => (32, 16),
        }
    }

    /// The dequantization matrix set index.
    const fn matrix(self) -> usize {
        use Kind::*;
        match self {
            Dct8 => 0,
            Hornuss => 1,
            Dct2 => 2,
            Dct4 => 3,
            Dct16 => 4,
            Dct32 => 5,
            Dct16x8 | Dct8x16 => 6,
            Dct32x8 | Dct8x32 => 7,
            Dct32x16 | Dct16x32 => 8,
            Dct4x8 | Dct8x4 => 9,
            Afv0 | Afv1 | Afv2 | Afv3 => 10,
            Dct64 => 11,
            Dct64x32 | Dct32x64 => 12,
            Dct128 => 13,
            Dct128x64 | Dct64x128 => 14,
            Dct256 => 15,
            Dct256x128 | Dct128x256 => 16,
        }
    }

    /// The coefficient order set index.
    const fn order(self) -> usize {
        use Kind::*;
        match self {
            Dct8 => 0,
            Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3 => 1,
            Dct16 => 2,
            Dct32 => 3,
            Dct16x8 | Dct8x16 => 4,
            Dct32x8 | Dct8x32 => 5,
            Dct32x16 | Dct16x32 => 6,
            Dct64 => 7,
            Dct64x32 | Dct32x64 => 8,
            Dct128 => 9,
            Dct128x64 | Dct64x128 => 10,
            Dct256 => 11,
            Dct256x128 | Dct128x256 => 12,
        }
    }

    /// Whether the stored coefficients are transposed against the block.
    const fn transposed(self) -> bool {
        use Kind::*;
        if matches!(
            self,
            Hornuss | Dct2 | Dct4 | Dct4x8 | Dct8x4 | Afv0 | Afv1 | Afv2 | Afv3
        ) {
            false
        } else {
            let (w, h) = self.size();
            h >= w
        }
    }

    /// Whether the block is a single 8x8 transform that takes its LF as is.
    const fn small(self) -> bool {
        use Kind::*;
        matches!(
            self,
            Hornuss | Dct2 | Dct4 | Dct8x4 | Dct4x8 | Dct8 | Afv0 | Afv1 | Afv2 | Afv3
        )
    }
}

const fn matrix_size(index: usize) -> (usize, usize) {
    match index {
        4 => (16, 16),
        5 => (32, 32),
        6 => (16, 8),
        7 => (32, 8),
        8 => (32, 16),
        11 => (64, 64),
        12 => (64, 32),
        13 => (128, 128),
        14 => (128, 64),
        15 => (256, 256),
        16 => (256, 128),
        _ => (8, 8),
    }
}

/// How a dequantization matrix is given.
#[derive(Debug, Clone)]
enum MatrixParams {
    Hornuss([[f32; 3]; 3]),
    Dct2([[f32; 6]; 3]),
    Dct4([[f32; 2]; 3], [Vec<f32>; 3]),
    Dct4x8([[f32; 1]; 3], [Vec<f32>; 3]),
    Afv([[f32; 9]; 3], [Vec<f32>; 3], [Vec<f32>; 3]),
    Dct([Vec<f32>; 3]),
    Raw([Vec<f32>; 3]),
}

const SEQ_A: [f32; 7] = [
    -1.025,
    -0.78,
    -0.65012,
    -0.19041574,
    -0.20819396,
    -0.421064,
    -0.32733846,
];
const SEQ_B: [f32; 7] = [
    -0.30419582,
    -0.36330363,
    -0.3566038,
    -0.34430745,
    -0.33699593,
    -0.30180866,
    -0.27321684,
];
const SEQ_C: [f32; 7] = [-1.2, -1.2, -0.8, -0.7, -0.7, -0.4, -0.5];
const DCT4X8_PARAMS: [[f32; 4]; 3] = [
    [2198.0505, -0.96269625, -0.7619425, -0.65511405],
    [764.36554, -0.926302, -0.967523, -0.2784529],
    [527.10754, -1.4594386, -1.4500821, -1.5843723],
];
const DCT4_PARAMS: [[f32; 4]; 3] = [
    [2200.0, 0.0, 0.0, 0.0],
    [392.0, 0.0, 0.0, 0.0],
    [112.0, -0.25, -0.25, -0.5],
];

fn common_seq(a: f32, b: f32, c: f32) -> MatrixParams {
    let with = |first: f32, seq: &[f32; 7]| {
        let mut v = vec![first];
        v.extend_from_slice(seq);
        v
    };
    MatrixParams::Dct([with(a, &SEQ_A), with(b, &SEQ_B), with(c, &SEQ_C)])
}

fn default_params(index: usize) -> MatrixParams {
    let v = <[f32]>::to_vec;
    match index {
        0 => MatrixParams::Dct([
            v(&[3150.0, 0.0, -0.4, -0.4, -0.4, -2.0]),
            v(&[560.0, 0.0, -0.3, -0.3, -0.3, -0.3]),
            v(&[512.0, -2.0, -1.0, 0.0, -1.0, -2.0]),
        ]),
        1 => MatrixParams::Hornuss([
            [280.0, 3160.0, 3160.0],
            [60.0, 864.0, 864.0],
            [18.0, 200.0, 200.0],
        ]),
        2 => MatrixParams::Dct2([
            [3840.0, 2560.0, 1280.0, 640.0, 480.0, 300.0],
            [960.0, 640.0, 320.0, 180.0, 140.0, 120.0],
            [640.0, 320.0, 128.0, 64.0, 32.0, 16.0],
        ]),
        3 => MatrixParams::Dct4([[1.0; 2]; 3], DCT4_PARAMS.map(|p| p.to_vec())),
        4 => MatrixParams::Dct([
            v(&[
                8996.873,
                -1.3000778,
                -0.4942453,
                -0.43909377,
                -0.6350102,
                -0.9017726,
                -1.6162099,
            ]),
            v(&[
                3191.4836,
                -0.67424583,
                -0.80745816,
                -0.4492584,
                -0.3586544,
                -0.3132239,
                -0.37615025,
            ]),
            v(&[
                1157.504, -2.0531423, -1.4, -0.5068713, -0.4270873, -1.4856834, -4.920914,
            ]),
        ]),
        5 => MatrixParams::Dct([
            v(&[
                15718.408,
                -1.025,
                -0.98,
                -0.9012,
                -0.4,
                -0.48819396,
                -0.421064,
                -0.27,
            ]),
            v(&[
                7305.7637,
                -0.8041958,
                -0.76330364,
                -0.5566038,
                -0.49785304,
                -0.43699592,
                -0.40180868,
                -0.27321684,
            ]),
            v(&[
                3803.5317,
                -3.0607336,
                -2.041327,
                -2.023565,
                -0.54953897,
                -0.4,
                -0.4,
                -0.3,
            ]),
        ]),
        6 => MatrixParams::Dct([
            v(&[7240.7734, -0.7, -0.7, -0.2, -0.2, -0.2, -0.5]),
            v(&[1448.1547, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2]),
            v(&[506.85413, -1.4, -0.2, -0.5, -0.5, -1.5, -3.6]),
        ]),
        7 => MatrixParams::Dct([
            v(&[
                16283.249, -1.7812846, -1.6309059, -1.0382179, -0.85, -0.7, -0.9, -1.2360638,
            ]),
            v(&[
                5089.1577, -0.3200494, -0.3536285, -0.3034, -0.61, -0.5, -0.5, -0.6,
            ]),
            v(&[
                3397.7761,
                -0.32132736,
                -0.3450762,
                -0.7034,
                -0.9,
                -1.0,
                -1.0,
                -1.1754606,
            ]),
        ]),
        8 => MatrixParams::Dct([
            v(&[
                13844.971, -0.971138, -0.658, -0.42026, -0.22712, -0.2206, -0.226, -0.6,
            ]),
            v(&[
                4798.964,
                -0.6112531,
                -0.8377079,
                -0.7901486,
                -0.26927274,
                -0.38272768,
                -0.22924222,
                -0.20719099,
            ]),
            v(&[1807.2369, -1.2, -1.2, -0.7, -0.7, -0.7, -0.4, -0.5]),
        ]),
        9 => MatrixParams::Dct4x8([[1.0]; 3], DCT4X8_PARAMS.map(|p| p.to_vec())),
        10 => MatrixParams::Afv(
            [
                [3072.0, 3072.0, 256.0, 256.0, 256.0, 414.0, 0.0, 0.0, 0.0],
                [1024.0, 1024.0, 50.0, 50.0, 50.0, 58.0, 0.0, 0.0, 0.0],
                [384.0, 384.0, 12.0, 12.0, 12.0, 22.0, -0.25, -0.25, -0.25],
            ],
            DCT4X8_PARAMS.map(|p| p.to_vec()),
            DCT4_PARAMS.map(|p| p.to_vec()),
        ),
        11 => common_seq(23966.166, 8380.191, 4493.024),
        12 => common_seq(15358.898, 5597.3604, 2919.9617),
        13 => common_seq(47932.332, 16760.383, 8986.048),
        14 => common_seq(30717.797, 11194.721, 5839.9233),
        15 => common_seq(95864.664, 33520.766, 17972.096),
        _ => common_seq(61435.594, 24209.441, 12979.847),
    }
}

fn interpolate(pos: f32, max: f32, bands: &[f32]) -> f32 {
    if bands.len() == 1 {
        return bands[0];
    }
    let scaled = pos * (bands.len() - 1) as f32 / max;
    let i = (scaled as usize).min(bands.len() - 2);
    let frac = scaled - i as f32;
    let (a, b) = (bands[i], bands[i + 1]);
    a * (b / a).powf(frac)
}

fn mult(x: f32) -> f32 {
    if x > 0.0 { 1.0 + x } else { 1.0 / (1.0 - x) }
}

fn dct_weights(params: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let mut bands = Vec::with_capacity(params.len());
    let mut last = params[0];
    bands.push(last);
    for &p in &params[1..] {
        last *= mult(p);
        if last <= 0.0 {
            return Err(malformed("a dequantization band is not positive"));
        }
        bands.push(last);
    }
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 / (width - 1) as f32;
            let dy = y as f32 / (height - 1) as f32;
            out.push(interpolate(
                (dx * dx + dy * dy).sqrt(),
                std::f32::consts::SQRT_2 + 1e-6,
                &bands,
            ));
        }
    }
    Ok(out)
}

impl MatrixParams {
    fn parse(
        r: &mut BitReader<'_>,
        index: usize,
        tree: Option<&Tree>,
        stream: u32,
        bit_depth: u32,
    ) -> Result<Self> {
        fn fixed<const N: usize>(r: &mut BitReader<'_>) -> Result<[[f32; N]; 3]> {
            let mut out = [[0.0; N]; 3];
            for v in out.iter_mut().flatten() {
                *v = r.f16()?;
            }
            Ok(out)
        }
        fn dct_params(r: &mut BitReader<'_>) -> Result<[Vec<f32>; 3]> {
            let n = r.read(4)? as usize + 1;
            let mut out = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            for v in out.iter_mut().flatten() {
                *v = r.f16()?;
            }
            for c in &mut out {
                c[0] *= 64.0;
            }
            Ok(out)
        }
        let mode = r.read(3)?;
        if (1..=5).contains(&mode) && !matches!(index, 0 | 1 | 2 | 3 | 9 | 10) {
            return Err(malformed(
                "a dequantization mode does not fit its block type",
            ));
        }
        Ok(match mode {
            0 => default_params(index),
            1 => Self::Hornuss(fixed(r)?),
            2 => Self::Dct2(fixed(r)?),
            3 => Self::Dct4(fixed(r)?, dct_params(r)?),
            4 => Self::Dct4x8(fixed(r)?, dct_params(r)?),
            5 => {
                let mut p = fixed::<9>(r)?;
                for c in &mut p {
                    for v in &mut c[..6] {
                        *v *= 64.0;
                    }
                }
                Self::Afv(p, dct_params(r)?, dct_params(r)?)
            }
            6 => Self::Dct(dct_params(r)?),
            _ => {
                let (w, h) = matrix_size(index);
                let denominator = r.f16()?;
                let shapes = [Shape {
                    width: w,
                    height: h,
                    hshift: 0,
                    vshift: 0,
                }; 3];
                let planes = decode_stream(r, &shapes, stream, tree, bit_depth)?;
                if planes.len() != 3 {
                    return Err(malformed("a raw dequantization matrix lacks channels"));
                }
                let mut out = [Vec::new(), Vec::new(), Vec::new()];
                for (o, p) in out.iter_mut().zip(planes) {
                    *o = p.data.iter().map(|&v| v as f32 * denominator).collect();
                }
                Self::Raw(out)
            }
        })
    }

    /// The reciprocal weights, row-major at the matrix's size.
    #[allow(clippy::too_many_lines, reason = "one layout per matrix kind")]
    fn weights(self, index: usize) -> Result<[Vec<f32>; 3]> {
        let raw = matches!(self, Self::Raw(_));
        let mut w: [Vec<f32>; 3] = match self {
            Self::Dct(p) => {
                let (w, h) = matrix_size(index);
                [
                    dct_weights(&p[0], w, h)?,
                    dct_weights(&p[1], w, h)?,
                    dct_weights(&p[2], w, h)?,
                ]
            }
            Self::Hornuss(p) => p.map(|p| {
                let mut out = vec![p[0]; 64];
                out[0] = 1.0;
                out[1] = p[1];
                out[8] = p[1];
                out[9] = p[2];
                out
            }),
            Self::Dct2(p) => p.map(|p| {
                let mut out = vec![0.0; 64];
                out[0] = 1.0;
                for (i, v) in p.into_iter().enumerate() {
                    let dim = 1 << (i / 2);
                    if i % 2 == 0 {
                        for y in 0..dim {
                            for x in dim..dim * 2 {
                                out[y * 8 + x] = v;
                                out[x * 8 + y] = v;
                            }
                        }
                    } else {
                        for y in dim..dim * 2 {
                            for x in dim..dim * 2 {
                                out[y * 8 + x] = v;
                            }
                        }
                    }
                }
                out
            }),
            Self::Dct4(p, dct) => {
                let mut out = [Vec::new(), Vec::new(), Vec::new()];
                for ((o, p), dct) in out.iter_mut().zip(p).zip(dct) {
                    let m = dct_weights(&dct, 4, 4)?;
                    *o = vec![0.0; 64];
                    for y in 0..4 {
                        for x in 0..4 {
                            let v = m[y * 4 + x];
                            o[y * 16 + x * 2] = v;
                            o[y * 16 + x * 2 + 1] = v;
                            o[(y * 2 + 1) * 8 + x * 2] = v;
                            o[(y * 2 + 1) * 8 + x * 2 + 1] = v;
                        }
                    }
                    o[1] /= p[0];
                    o[8] /= p[0];
                    o[9] /= p[1];
                }
                out
            }
            Self::Dct4x8(p, dct) => {
                let mut out = [Vec::new(), Vec::new(), Vec::new()];
                for ((o, p), dct) in out.iter_mut().zip(p).zip(dct) {
                    let m = dct_weights(&dct, 8, 4)?;
                    *o = m
                        .chunks_exact(8)
                        .flat_map(|row| [row, row])
                        .flatten()
                        .copied()
                        .collect();
                    o[8] /= p[0];
                }
                out
            }
            Self::Afv(p, dct, dct4) => {
                const FREQS: [f32; 16] = [
                    0.0, 0.0, 0.8517779, 5.3777843, 0.0, 0.0, 4.734748, 5.4492455, 1.659827, 4.0,
                    7.275749, 10.423227, 2.6629324, 7.6306577, 8.962389, 12.971662,
                ];
                let (lo, hi) = (FREQS[2], FREQS[15]);
                let mut out = [Vec::new(), Vec::new(), Vec::new()];
                for (((o, p), dct), dct4) in out.iter_mut().zip(p).zip(dct).zip(dct4) {
                    let w48 = dct_weights(&dct, 8, 4)?;
                    let w44 = dct_weights(&dct4, 4, 4)?;
                    let mut bands = [p[5], 0.0, 0.0, 0.0];
                    for i in 1..4 {
                        bands[i] = bands[i - 1] * mult(p[5 + i]);
                    }
                    *o = vec![0.0; 64];
                    for y in 0..4 {
                        for x in 0..4 {
                            o[16 * y + 2 * x] = match (x, y) {
                                (0, 0) => 1.0,
                                (0, 1) => p[2],
                                (1, 0) => p[3],
                                (1, 1) => p[4],
                                _ => interpolate(FREQS[y * 4 + x] - lo, hi - lo + 1e-6, &bands),
                            };
                        }
                    }
                    for y in 0..4 {
                        for x in 0..8 {
                            o[16 * y + 8 + x] = if x == 0 && y == 0 {
                                p[0]
                            } else {
                                w48[y * 8 + x]
                            };
                        }
                        for x in 0..4 {
                            o[16 * y + 2 * x + 1] = if x == 0 && y == 0 {
                                p[1]
                            } else {
                                w44[y * 4 + x]
                            };
                        }
                    }
                }
                out
            }
            Self::Raw(m) => m,
        };
        if !raw {
            for v in w.iter_mut().flatten() {
                *v = 1.0 / *v;
            }
        }
        if w.iter().flatten().any(|&v| !(v > 0.0 && v < 1e8)) {
            return Err(malformed("a dequantization weight is out of range"));
        }
        Ok(w)
    }
}

/// One pass's coefficient orders and HF code.
#[derive(Debug, Clone)]
struct Pass {
    /// By order set and channel: (x, y) of each coefficient, in order.
    orders: Vec<[Vec<(u16, u16)>; 3]>,
    decoder: Decoder,
}

/// The order sets' sizes, (width, height) in coefficients.
const ORDER_SIZES: [(usize, usize); 13] = [
    (8, 8),
    (8, 8),
    (16, 16),
    (32, 32),
    (16, 8),
    (32, 8),
    (32, 16),
    (64, 64),
    (64, 32),
    (128, 128),
    (128, 64),
    (256, 256),
    (256, 128),
];

/// The natural (zig-zag by LLF first) coefficient order of an order set.
fn natural_order((bw, bh): (usize, usize)) -> Vec<(u16, u16)> {
    let y_scale = bw / bh;
    let (lbw, lbh) = (bw / 8, bh / 8);
    let mut out = Vec::with_capacity(bw * bh);
    for i in 0..lbw * lbh {
        out.push(((i % lbw) as u16, (i / lbw) as u16));
    }
    for dist in 1..2 * bw {
        let margin = dist.saturating_sub(bw);
        for order in margin..dist - margin {
            let (x, y) = if dist % 2 == 1 {
                (order, dist - 1 - order)
            } else {
                (dist - 1 - order, order)
            };
            if (x < lbw && y < lbw) || y % y_scale != 0 {
                continue;
            }
            out.push((x as u16, (y / y_scale) as u16));
        }
    }
    out
}

/// A varblock's place in the block grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Block {
    Empty,
    Covered,
    Start { kind: Kind, hf_mul: i32 },
}

/// What the restoration filters need from a VarDCT frame.
#[derive(Debug, Clone)]
pub struct FilterInfo {
    /// The edge-preserving filter's sigma for each 8x8 block.
    pub sigma: Vec<f32>,
    /// Blocks per row.
    pub width: usize,
}

/// Chroma subsampling shifts of the X (Cb), Y and B (Cr) planes.
fn jpeg_shifts(ups: [u32; 3]) -> [(usize, usize); 3] {
    let h = ups.iter().any(|&u| u == 1 || u == 2);
    let v = ups.iter().any(|&u| u == 1 || u == 3);
    std::array::from_fn(|c| {
        let (hs, vs) = match ups[c] {
            0 => (h, v),
            1 => (false, false),
            2 => (false, v),
            _ => (h, false),
        };
        (usize::from(hs), usize::from(vs))
    })
}

/// A VarDCT frame being decoded.
#[derive(Debug)]
pub struct State {
    global: Global,
    /// Colour sample size.
    width: usize,
    height: usize,
    /// The block grid, rounded for subsampling.
    bw: usize,
    bh: usize,
    shifts: [(usize, usize); 3],
    subsampled: bool,
    group_dim: usize,
    num_lf: usize,
    num_groups: usize,
    pass_shift: Vec<u32>,
    lf_dequant: [f32; 3],
    x_qm_scale: u32,
    b_qm_scale: u32,
    quant_bias: [f32; 3],
    quant_bias_numerator: f32,
    bit_depth: u32,
    skip_lf_smoothing: bool,
    epf: Option<(f32, [f32; 8])>,
    /// LF per channel at block resolution (shifted).
    lf: [Vec<f32>; 3],
    lf_quant: Option<[Vec<i32>; 3]>,
    /// Whether the LF came from an LF frame, already final.
    lf_from_frame: bool,
    blocks: Vec<Block>,
    /// Chroma-from-luma factors per 64x64 tile.
    x_from_y: Vec<i32>,
    b_from_y: Vec<i32>,
    tiles_w: usize,
    sigma: Vec<f32>,
    matrices: Vec<[Vec<f32>; 3]>,
    num_presets: u32,
    passes: Vec<Pass>,
    coeffs: [Vec<i32>; 3],
}

impl State {
    #[must_use]
    pub fn new(
        frame: &FrameHeader,
        global: &Global,
        lf_dequant: [f32; 3],
        image: &ImageHeader,
    ) -> Self {
        let (w, h) = frame.colour_size();
        let (width, height) = (w as usize, h as usize);
        let shifts = jpeg_shifts(frame.jpeg_upsampling);
        let subsampled = shifts.iter().any(|&(h, v)| h + v > 0);
        let mut bw = width.div_ceil(8);
        let mut bh = height.div_ceil(8);
        if shifts.iter().any(|s| s.0 > 0) {
            bw = bw.div_ceil(2) * 2;
        }
        if shifts.iter().any(|s| s.1 > 0) {
            bh = bh.div_ceil(2) * 2;
        }
        let lf: [Vec<f32>; 3] =
            std::array::from_fn(|c| vec![0.0; (bw >> shifts[c].0) * (bh >> shifts[c].1)]);
        let coeffs =
            std::array::from_fn(|c| vec![0; (bw >> shifts[c].0) * (bh >> shifts[c].1) * 64]);
        let tiles_w = width.div_ceil(64);
        let tiles = tiles_w * height.div_ceil(64);
        let epf = frame.epf.map(|e| {
            (
                e.quant_mul * 65536.0 / global.global_scale as f32,
                e.sharp_lut,
            )
        });
        Self {
            global: global.clone(),
            width,
            height,
            bw,
            bh,
            shifts,
            subsampled,
            group_dim: frame.group_dim() as usize,
            num_lf: frame.num_lf_groups() as usize,
            num_groups: frame.num_groups() as usize,
            pass_shift: frame.pass_shift.clone(),
            lf_dequant: lf_dequant.map(|m| m * 128.0),
            x_qm_scale: frame.x_qm_scale,
            b_qm_scale: frame.b_qm_scale,
            quant_bias: image.opsin.quant_bias,
            quant_bias_numerator: image.opsin.quant_bias_numerator,
            bit_depth: image.bit_depth.bits,
            skip_lf_smoothing: frame.flags & FrameHeader::SKIP_ADAPTIVE_LF_SMOOTHING != 0,
            epf,
            lf,
            lf_quant: None,
            lf_from_frame: false,
            blocks: vec![Block::Empty; bw * bh],
            x_from_y: vec![0; tiles],
            b_from_y: vec![0; tiles],
            tiles_w,
            sigma: vec![1.0; bw * bh],
            matrices: Vec::new(),
            num_presets: 1,
            passes: Vec::new(),
            coeffs,
        }
    }

    /// The LF group's block rectangle: origin and size in blocks.
    fn lf_rect(&self, (lx, ly): (usize, usize)) -> (usize, usize, usize, usize) {
        let d = self.group_dim;
        let (x0, y0) = (lx * d, ly * d);
        (
            x0,
            y0,
            d.min(self.bw.saturating_sub(x0)),
            d.min(self.bh.saturating_sub(y0)),
        )
    }

    /// Decode an LF group's quantized LF coefficients.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn decode_lf_group(
        &mut self,
        r: &mut BitReader<'_>,
        _g: &Global,
        tree: Option<&Tree>,
        i: usize,
        pos: (usize, usize),
    ) -> Result<()> {
        let extra_precision = r.read(2)?;
        let (x0, y0, w, h) = self.lf_rect(pos);
        // Stored as Y, X, B.
        let order = [1, 0, 2];
        let shapes: Vec<Shape> = order
            .iter()
            .map(|&c| {
                let (hs, vs) = self.shifts[c];
                Shape {
                    width: w.div_ceil(1 << hs),
                    height: h.div_ceil(1 << vs),
                    hshift: hs as i32,
                    vshift: vs as i32,
                }
            })
            .collect();
        let planes = decode_stream(r, &shapes, (1 + i) as u32, tree, self.bit_depth)?;
        if planes.len() != 3 {
            return Err(malformed("LF coefficients lack channels"));
        }
        let scale_inv = f64::from(self.global.global_scale) * f64::from(self.global.quant_lf);
        let precision = f64::from(1_u32 << (9 - extra_precision));
        let lf_quant = self.lf_quant.get_or_insert_with(|| {
            std::array::from_fn(|c| {
                vec![0; (self.bw >> self.shifts[c].0) * (self.bh >> self.shifts[c].1)]
            })
        });
        for (plane, &c) in planes.iter().zip(&order) {
            let (hs, vs) = self.shifts[c];
            let cw = self.bw >> hs;
            let scale = (f64::from(self.lf_dequant[c]) * precision / scale_inv) as f32;
            let (bx, by) = (x0 >> hs, y0 >> vs);
            for y in 0..plane.height {
                for x in 0..plane.width {
                    let at = (by + y) * cw + bx + x;
                    if at < self.lf[c].len() && bx + x < cw {
                        let q = plane.data[y * plane.width + x];
                        lf_quant[c][at] = q;
                        self.lf[c][at] = q as f32 * scale;
                    }
                }
            }
        }
        Ok(())
    }

    /// Decode an LF group's block types, multipliers, chroma-from-luma
    /// factors and filter sharpness.
    ///
    /// # Errors
    ///
    /// For malformed data, or blocks that do not tile the group.
    pub fn decode_hf_metadata(
        &mut self,
        r: &mut BitReader<'_>,
        _g: &Global,
        tree: Option<&Tree>,
        i: usize,
        pos: (usize, usize),
    ) -> Result<()> {
        let (x0, y0, w, h) = self.lf_rect(pos);
        let nb_blocks = 1 + r.read((w * h).next_power_of_two().trailing_zeros())? as usize;
        let (tw, th) = ((w * 8).div_ceil(64), (h * 8).div_ceil(64));
        let shape = |width, height| Shape {
            width,
            height,
            hshift: 0,
            vshift: 0,
        };
        let shapes = [
            shape(tw, th),
            shape(tw, th),
            shape(nb_blocks, 2),
            shape(w, h),
        ];
        let planes = decode_stream(
            r,
            &shapes,
            (1 + 2 * self.num_lf + i) as u32,
            tree,
            self.bit_depth,
        )?;
        let [xfy, bfy, info, sharpness] = &planes[..] else {
            return Err(malformed("block metadata lacks channels"));
        };
        let (tx0, ty0) = (x0 / 8, y0 / 8);
        for y in 0..th {
            for x in 0..tw {
                let at = (ty0 + y) * self.tiles_w + tx0 + x;
                if tx0 + x < self.tiles_w && at < self.x_from_y.len() {
                    self.x_from_y[at] = xfy.at(x, y);
                    self.b_from_y[at] = bfy.at(x, y);
                }
            }
        }
        let mut next = 0_usize;
        for y in 0..h {
            let mut x = 0;
            while x < w {
                if self.blocks[(y0 + y) * self.bw + x0 + x] != Block::Empty {
                    x += 1;
                    continue;
                }
                if next >= nb_blocks {
                    return Err(malformed("block types do not fill the LF group"));
                }
                let kind = *KINDS
                    .get(info.at(next, 0) as usize)
                    .ok_or_else(|| malformed("unknown block type"))?;
                let hf_mul = info.at(next, 1).wrapping_add(1);
                if hf_mul <= 0 {
                    return Err(malformed("a block's HF multiplier is not positive"));
                }
                let (dw, dh) = kind.size();
                if (x % 32) + dw > 32 || (y % 32) + dh > 32 || x + dw > w || y + dh > h {
                    return Err(malformed("a block crosses a group border"));
                }
                for dy in 0..dh {
                    for dx in 0..dw {
                        let at = (y0 + y + dy) * self.bw + x0 + x + dx;
                        if self.blocks[at] != Block::Empty {
                            return Err(malformed("blocks overlap"));
                        }
                        self.blocks[at] = if dx == 0 && dy == 0 {
                            Block::Start { kind, hf_mul }
                        } else {
                            Block::Covered
                        };
                        if let Some((quant_mul, lut)) = self.epf {
                            let s = sharpness.at(x + dx, y + dy);
                            let s = usize::try_from(s)
                                .ok()
                                .filter(|&s| s < 8)
                                .ok_or_else(|| malformed("bad filter sharpness"))?;
                            self.sigma[at] = quant_mul / hf_mul as f32 * lut[s];
                        }
                    }
                }
                next += 1;
                x += dw;
            }
        }
        Ok(())
    }

    /// Take the LF from a decoded LF frame.
    ///
    /// # Errors
    ///
    /// If its size does not match.
    pub fn use_lf_frame(&mut self, lf: &FramePlanes) -> Result<()> {
        for c in 0..3 {
            let (hs, vs) = self.shifts[c];
            let (cw, ch) = (self.bw >> hs, self.bh >> vs);
            for y in 0..ch.min(lf.height) {
                for x in 0..cw.min(lf.width) {
                    self.lf[c][y * cw + x] = lf.colour[c][y * lf.width + x];
                }
            }
        }
        self.lf_quant = None;
        self.lf_from_frame = true;
        Ok(())
    }

    /// Decode HfGlobal: dequantization matrices, presets and passes.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn decode_hf_global(&mut self, r: &mut BitReader<'_>, tree: Option<&Tree>) -> Result<()> {
        let all_default = r.bool()?;
        let stream = 1 + 3 * self.num_lf as u32;
        self.matrices = (0..17)
            .map(|i| {
                let params = if all_default {
                    default_params(i)
                } else {
                    MatrixParams::parse(r, i, tree, stream + i as u32, self.bit_depth)?
                };
                params.weights(i)
            })
            .collect::<Result<_>>()?;
        self.num_presets = 1 + r.read(self.num_groups.next_power_of_two().trailing_zeros())?;
        let contexts = 495 * self.num_presets * self.global.block_ctx.clusters;
        for _ in 0..self.pass_shift.len() + 1 {
            let mut used = r.u32([val(0x5F), val(0x13), val(0), bits(0, 13)])?;
            let mut orders: Vec<[Vec<(u16, u16)>; 3]> = Vec::with_capacity(13);
            let mut d = if used != 0 {
                Some(Decoder::parse(r, 8)?)
            } else {
                None
            };
            if let Some(d) = d.as_mut() {
                d.begin(r)?;
            }
            for &size in &ORDER_SIZES {
                let natural = natural_order(size);
                if used & 1 != 0 {
                    let d = d
                        .as_mut()
                        .ok_or_else(|| malformed("coefficient orders without a code"))?;
                    let n = (size.0 * size.1) as u32;
                    let mut set: [Vec<(u16, u16)>; 3] = Default::default();
                    for o in &mut set {
                        let perm = read_permutation(r, d, n, n / 64)?;
                        *o = perm.into_iter().map(|i| natural[i]).collect();
                    }
                    orders.push(set);
                } else {
                    orders.push([natural.clone(), natural.clone(), natural]);
                }
                used >>= 1;
            }
            if let Some(d) = &d {
                d.finish()?;
            }
            let decoder = Decoder::parse(r, contexts)?;
            self.passes.push(Pass { orders, decoder });
        }
        Ok(())
    }

    /// Decode a pass group's HF coefficients.
    ///
    /// # Errors
    ///
    /// For malformed data.
    #[allow(clippy::too_many_lines, reason = "the coefficient loop, as specified")]
    pub fn decode_pass_group(
        &mut self,
        r: &mut BitReader<'_>,
        pass: usize,
        _g: usize,
        (gx, gy): (usize, usize),
    ) -> Result<()> {
        const FREQ_CTX: [u32; 63] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
            19, 20, 20, 21, 21, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26,
            26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
        ];
        const NONZERO_CTX: [u32; 63] = [
            0, 31, 62, 62, 93, 93, 93, 93, 123, 123, 123, 123, 152, 152, 152, 152, 152, 152, 152,
            152, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
            206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
        ];
        let Some(p) = self.passes.get(pass) else {
            return Err(malformed("a pass has no HF code"));
        };
        let mut d = p.decoder.clone();
        let coeff_shift = self.pass_shift.get(pass).copied().unwrap_or(0);
        let bctx = &self.global.block_ctx;
        let nbc = bctx.clusters;
        let lf_mul = (bctx.lf_thresholds[0].len() + 1)
            * (bctx.lf_thresholds[1].len() + 1)
            * (bctx.lf_thresholds[2].len() + 1);
        let hf_mul_n = bctx.qf_thresholds.len() + 1;
        let hfp = r.read(self.num_presets.next_power_of_two().trailing_zeros())?;
        if hfp >= self.num_presets {
            return Err(malformed("an HF preset out of range"));
        }
        let base = 495 * nbc * hfp;
        d.begin(r)?;
        let gd = self.group_dim / 8;
        let (bx0, by0) = (gx * gd, gy * gd);
        let bw = gd.min(self.bw.saturating_sub(bx0));
        let bh = gd.min(self.bh.saturating_sub(by0));
        let mut nz_rows: [Vec<u32>; 3] =
            std::array::from_fn(|c| vec![0; bw.div_ceil(1 << self.shifts[c].0) + 1]);
        for y in 0..bh {
            for x in 0..bw {
                let Block::Start { kind, hf_mul: qf } = self.blocks[(by0 + y) * self.bw + bx0 + x]
                else {
                    continue;
                };
                let (w8, h8) = kind.size();
                let num_blocks = (w8 * h8) as u32;
                let log = num_blocks.trailing_zeros();
                let order_id = kind.order();
                let lf_idx = if let Some(lq) = &self.lf_quant {
                    let mut idx = 0;
                    for c in [0, 2, 1] {
                        let t = &bctx.lf_thresholds[c];
                        idx *= t.len() + 1;
                        let (hs, vs) = self.shifts[c];
                        let q = lq[c][((by0 + y) >> vs) * (self.bw >> hs) + ((bx0 + x) >> hs)];
                        idx += t.iter().filter(|&&th| q > th).count();
                    }
                    idx
                } else {
                    0
                };
                let hf_idx = bctx
                    .qf_thresholds
                    .iter()
                    .filter(|&&t| qf > t as i32)
                    .count();
                for ci in 0..3 {
                    let c = [1, 0, 2][ci];
                    let (hs, vs) = self.shifts[c];
                    let (sx, sy) = (x >> hs, y >> vs);
                    if hs != 0 || vs != 0 {
                        if (sx << hs) != x || (sy << vs) != y {
                            continue;
                        }
                        if !matches!(
                            self.blocks[(by0 + sy) * self.bw + bx0 + sx],
                            Block::Start { .. }
                        ) {
                            continue;
                        }
                    }
                    let idx = ((ci * 13 + order_id) * hf_mul_n + hf_idx) * lf_mul + lf_idx;
                    let block_ctx = u32::from(
                        *bctx
                            .map
                            .get(idx)
                            .ok_or_else(|| malformed("a block context out of range"))?,
                    );
                    let predicted = if sy == 0 {
                        if sx == 0 { 32 } else { nz_rows[c][sx - 1] }
                    } else if sx == 0 {
                        nz_rows[c][sx]
                    } else {
                        (nz_rows[c][sx] + nz_rows[c][sx - 1] + 1) >> 1
                    };
                    let bucket = if predicted >= 8 {
                        4 + predicted / 2
                    } else {
                        predicted
                    };
                    let nz_ctx = block_ctx + bucket * nbc;
                    let mut non_zeros = d.read(r, base + nz_ctx)?;
                    if non_zeros > (63 << log) {
                        return Err(malformed("too many nonzero coefficients"));
                    }
                    let nz_val = (non_zeros + num_blocks - 1) >> log;
                    for dx in 0..w8 {
                        if let Some(v) = nz_rows[c].get_mut(sx + dx) {
                            *v = nz_val;
                        }
                    }
                    if non_zeros == 0 {
                        continue;
                    }
                    let mut prev_nonzero = u32::from(non_zeros <= num_blocks * 4);
                    let order = &p.orders[order_id][c];
                    let ctx_base = base + block_ctx * 458 + 37 * nbc;
                    let cw = (self.bw >> hs) * 8;
                    let (ox, oy) = (((bx0 >> hs) + sx) * 8, ((by0 >> vs) + sy) * 8);
                    let transposed = kind.transposed();
                    for (k, &(ax, ay)) in order.iter().enumerate().skip(num_blocks as usize) {
                        let k = k - num_blocks as usize;
                        let ctx = (NONZERO_CTX[((non_zeros - 1) >> log) as usize]
                            + FREQ_CTX[k >> log])
                            * 2
                            + prev_nonzero;
                        if ctx >= 458 {
                            return Err(malformed("too many zero HF coefficients"));
                        }
                        let u = d.read(r, ctx_base + ctx)?;
                        if u == 0 {
                            prev_nonzero = 0;
                            continue;
                        }
                        let v = unpack_signed(u) << coeff_shift;
                        let (dx, dy) = if transposed { (ay, ax) } else { (ax, ay) };
                        let at = (oy + usize::from(dy)) * cw + ox + usize::from(dx);
                        let slot = self.coeffs[c]
                            .get_mut(at)
                            .ok_or_else(|| malformed("a coefficient outside the frame"))?;
                        *slot = slot.wrapping_add(v);
                        prev_nonzero = 1;
                        non_zeros -= 1;
                        if non_zeros == 0 {
                            break;
                        }
                    }
                }
            }
        }
        d.finish()
    }

    /// Dequantize and transform every block, returning the colour planes at
    /// the colour sample size and the filters' block data.
    ///
    /// # Errors
    ///
    /// For a block whose matrices are missing.
    pub fn finish(mut self) -> Result<([Vec<f32>; 3], FilterInfo)> {
        if self.matrices.is_empty() {
            return Err(malformed("a VarDCT frame lacks HfGlobal"));
        }
        if !self.subsampled && !self.lf_from_frame {
            self.lf_correlation();
        }
        if !self.skip_lf_smoothing && !self.subsampled && !self.lf_from_frame {
            self.lf_smoothing();
        }
        let qm_scale = [
            0.8_f32.powi(self.x_qm_scale as i32 - 2),
            1.0,
            0.8_f32.powi(self.b_qm_scale as i32 - 2),
        ];
        let mut out: [Vec<f32>; 3] = std::array::from_fn(|c| {
            vec![0.0; (self.bw >> self.shifts[c].0) * (self.bh >> self.shifts[c].1) * 64]
        });
        let cfl = self.global.cfl;
        let mut block = [Vec::new(), Vec::new(), Vec::new()];
        for by in 0..self.bh {
            for bx in 0..self.bw {
                let Block::Start { kind, hf_mul } = self.blocks[by * self.bw + bx] else {
                    continue;
                };
                let (w8, h8) = kind.size();
                let (cw, ch) = (w8 * 8, h8 * 8);
                let mul_base = 65536.0 / (self.global.global_scale as f32 * hf_mul as f32);
                let mut active = [false; 3];
                for c in [1, 0, 2] {
                    let (hs, vs) = self.shifts[c];
                    let (sx, sy) = (bx >> hs, by >> vs);
                    if hs != 0 || vs != 0 {
                        if (sx << hs) != bx || (sy << vs) != by {
                            continue;
                        }
                        if !matches!(self.blocks[sy * self.bw + sx], Block::Start { .. }) {
                            continue;
                        }
                    }
                    active[c] = true;
                    let stride = (self.bw >> hs) * 8;
                    let m = &self.matrices[kind.matrix()][c];
                    let transposed = kind.transposed();
                    let (mw, _) = matrix_size(kind.matrix());
                    let mul = mul_base * qm_scale[c];
                    let b = &mut block[c];
                    b.clear();
                    b.resize(cw * ch, 0.0);
                    for y in 0..ch {
                        for x in 0..cw {
                            let q = self.coeffs[c][(sy * 8 + y) * stride + sx * 8 + x];
                            if q == 0 {
                                continue;
                            }
                            let mut v = q as f32;
                            if v.abs() <= 1.0 {
                                v *= self.quant_bias[c];
                            } else {
                                v -= self.quant_bias_numerator / v;
                            }
                            // The matrix is stored at (width, height) of the
                            // wide orientation.
                            let w = if transposed {
                                m[x * mw + y]
                            } else {
                                m[y * cw + x]
                            };
                            b[y * cw + x] = v * w * mul;
                        }
                    }
                }
                if !self.subsampled {
                    let [bx_, by_, bb_] = &mut block;
                    for y in 0..ch {
                        for x in 0..cw {
                            let (px, py) = (bx * 8 + x, by * 8 + y);
                            let tile = (py / 64) * self.tiles_w + px / 64;
                            let kx = cfl.base_x
                                + self.x_from_y.get(tile).copied().unwrap_or(0) as f32
                                    / cfl.colour_factor as f32;
                            let kb = cfl.base_b
                                + self.b_from_y.get(tile).copied().unwrap_or(0) as f32
                                    / cfl.colour_factor as f32;
                            let yv = by_[y * cw + x];
                            bx_[y * cw + x] += kx * yv;
                            bb_[y * cw + x] += kb * yv;
                        }
                    }
                }
                for c in 0..3 {
                    if !active[c] {
                        continue;
                    }
                    let (hs, vs) = self.shifts[c];
                    let (sx, sy) = (bx >> hs, by >> vs);
                    let lw = self.bw >> hs;
                    let b = &mut block[c];
                    // The LLF coefficients from the LF image.
                    if kind.small() {
                        b[0] = self.lf[c][sy * lw + sx];
                    } else {
                        let mut llf = vec![0.0; w8 * h8];
                        for y in 0..h8 {
                            for x in 0..w8 {
                                llf[y * w8 + x] = self.lf[c]
                                    .get((sy + y) * lw + sx + x)
                                    .copied()
                                    .unwrap_or(0.0);
                            }
                        }
                        dct_2d(&mut llf, w8, h8, false);
                        let (lbw, lbh) =
                            (w8.trailing_zeros() as usize, h8.trailing_zeros() as usize);
                        for y in 0..h8 {
                            for x in 0..w8 {
                                b[y * cw + x] =
                                    llf[y * w8 + x] / (scale_f(y, 5 - lbh) * scale_f(x, 5 - lbw));
                            }
                        }
                    }
                    transform(b, cw, ch, kind);
                    let stride = lw * 8;
                    for y in 0..ch {
                        let row = (sy * 8 + y) * stride + sx * 8;
                        if let Some(dst) = out[c].get_mut(row..row + cw) {
                            dst.copy_from_slice(&b[y * cw..(y + 1) * cw]);
                        }
                    }
                }
            }
        }
        // Chroma upsampling, then the colour sample size.
        let full_w = self.bw * 8;
        let full_h = self.bh * 8;
        let mut planes: [Vec<f32>; 3] = Default::default();
        for (c, plane) in out.into_iter().enumerate() {
            let (hs, vs) = self.shifts[c];
            let (mut p, w, h) = (plane, full_w >> hs, full_h >> vs);
            // Upsample from the valid samples only, not the block padding.
            let (vw, vh) = (
                self.width.div_ceil(1 << hs).min(w),
                self.height.div_ceil(1 << vs).min(h),
            );
            if (vw, vh) != (w, h) {
                let mut valid = Vec::with_capacity(vw * vh);
                for y in 0..vh {
                    valid.extend_from_slice(&p[y * w..y * w + vw]);
                }
                p = valid;
            }
            let (mut w, h) = (vw, vh);
            if hs > 0 {
                p = upsample_h(&p, w, h);
                w *= 2;
            }
            let mut h = h;
            if vs > 0 {
                p = upsample_v(&p, w, h);
                h *= 2;
            }
            let mut cropped = vec![0.0; self.width * self.height];
            for y in 0..self.height.min(h) {
                let n = self.width.min(w);
                cropped[y * self.width..y * self.width + n].copy_from_slice(&p[y * w..y * w + n]);
            }
            planes[c] = cropped;
        }
        Ok((
            planes,
            FilterInfo {
                sigma: self.sigma,
                width: self.bw,
            },
        ))
    }

    /// Chroma from luma on the LF image.
    fn lf_correlation(&mut self) {
        let cfl = self.global.cfl;
        let kx = cfl.base_x + (cfl.x_factor_lf as f32 - 128.0) / cfl.colour_factor as f32;
        let kb = cfl.base_b + (cfl.b_factor_lf as f32 - 128.0) / cfl.colour_factor as f32;
        let [x, y, b] = &mut self.lf;
        for ((x, &y), b) in x.iter_mut().zip(y.iter()).zip(b.iter_mut()) {
            *x += kx * y;
            *b += kb * y;
        }
    }

    /// Adaptive smoothing of the LF image.
    fn lf_smoothing(&mut self) {
        const SELF: f32 = 0.052262735;
        const SIDE: f32 = 0.2034514;
        const DIAG: f32 = 0.03348292;
        let (w, h) = (self.bw, self.bh);
        if w <= 2 || h <= 2 {
            return;
        }
        let scale_inv = f64::from(self.global.global_scale) * f64::from(self.global.quant_lf);
        let lf_scale = self
            .lf_dequant
            .map(|m| (512.0 * f64::from(m) / scale_inv) as f32);
        let src = self.lf.clone();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let mut wa = [0.0; 3];
                let mut gap = 0.5_f32;
                for c in 0..3 {
                    let s = &src[c];
                    let at = y * w + x;
                    let side = s[at - 1] + s[at + 1] + s[at - w] + s[at + w];
                    let diag = s[at - w - 1] + s[at - w + 1] + s[at + w - 1] + s[at + w + 1];
                    wa[c] = s[at] * SELF + side * SIDE + diag * DIAG;
                    gap = gap.max((wa[c] - s[at]).abs() / lf_scale[c]);
                }
                let scale = (3.0 - 4.0 * gap).max(0.0);
                for c in 0..3 {
                    let at = y * w + x;
                    let s = src[c][at];
                    self.lf[c][at] = (wa[c] - s) * scale + s;
                }
            }
        }
    }
}

fn upsample_h(p: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * 2 * h];
    for y in 0..h {
        let row = &p[y * w..(y + 1) * w];
        for x in 0..w {
            let prev = row[x.saturating_sub(1)];
            let cur = row[x];
            let next = row[(x + 1).min(w - 1)];
            out[y * 2 * w + 2 * x] = 0.25 * prev + 0.75 * cur;
            out[y * 2 * w + 2 * x + 1] = 0.75 * cur + 0.25 * next;
        }
    }
    out
}

fn upsample_v(p: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * 2 * h];
    for y in 0..h {
        let prev = y.saturating_sub(1);
        let next = (y + 1).min(h - 1);
        for x in 0..w {
            let cur = p[y * w + x];
            out[2 * y * w + x] = 0.25 * p[prev * w + x] + 0.75 * cur;
            out[(2 * y + 1) * w + x] = 0.75 * cur + 0.25 * p[next * w + x];
        }
    }
    out
}

const SCALE_F: [f32; 32] = [
    1.0000000000000000,
    0.9996047255830407,
    0.9984194528776054,
    0.9964458326264695,
    0.9936866130906366,
    0.9901456355893141,
    0.9858278282666936,
    0.9807391980963174,
    0.9748868211368796,
    0.9682788310563117,
    0.9609244059440204,
    0.9528337534340876,
    0.9440180941651672,
    0.9344896436056892,
    0.9242615922757944,
    0.9133480844001980,
    0.9017641950288744,
    0.8895259056651056,
    0.8766500784429904,
    0.8631544288990163,
    0.8490574973847023,
    0.8343786191696513,
    0.8191378932865928,
    0.8033561501721485,
    0.7870549181591013,
    0.7702563888779096,
    0.7529833816270532,
    0.7352593067735488,
    0.7171081282466044,
    0.6985543251889097,
    0.6796228528314652,
    0.6603391026591464,
];

fn scale_f(c: usize, logb: usize) -> f32 {
    SCALE_F[(c << logb).min(31)]
}

/// `cos(π (2i + 1) k / 2n)` for `n` a power of two up to 256, `k`-major.
fn cos_table(n: usize) -> &'static [f32] {
    static TABLES: [std::sync::OnceLock<Vec<f32>>; 9] = [const { std::sync::OnceLock::new() }; 9];
    let log = n.trailing_zeros() as usize;
    TABLES[log.min(8)].get_or_init(|| {
        let tau = std::f64::consts::PI / (2 * n) as f64;
        let mut t = Vec::with_capacity(n * n);
        for k in 0..n {
            for i in 0..n {
                t.push(((2 * i + 1) as f64 * k as f64 * tau).cos() as f32);
            }
        }
        t
    })
}

/// One-dimensional DCT-II (forward: DC is the mean, AC scaled by √2/N)
/// or its inverse, in place.
fn dct_1d(data: &mut [f32], scratch: &mut [f32], inverse: bool) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let table = cos_table(n);
    let out = &mut scratch[..n];
    if inverse {
        out.fill(data[0]);
        for (k, &v) in data.iter().enumerate().skip(1) {
            if v == 0.0 {
                continue;
            }
            let v = v * std::f32::consts::SQRT_2;
            for (o, &c) in out.iter_mut().zip(&table[k * n..(k + 1) * n]) {
                *o += v * c;
            }
        }
    } else {
        let scale = std::f32::consts::SQRT_2 / n as f32;
        for (k, o) in out.iter_mut().enumerate() {
            let s: f32 = data
                .iter()
                .zip(&table[k * n..(k + 1) * n])
                .map(|(&v, &c)| v * c)
                .sum();
            *o = if k == 0 { s / n as f32 } else { s * scale };
        }
    }
    data.copy_from_slice(out);
}

/// A separable 2D DCT of a `w` x `h` row-major block.
fn dct_2d(data: &mut [f32], w: usize, h: usize, inverse: bool) {
    let mut scratch = vec![0.0; w.max(h)];
    if w > 1 {
        for row in data.chunks_exact_mut(w) {
            dct_1d(row, &mut scratch, inverse);
        }
    }
    if h > 1 {
        let mut col = vec![0.0; h];
        for x in 0..w {
            for y in 0..h {
                col[y] = data[y * w + x];
            }
            dct_1d(&mut col, &mut scratch, inverse);
            for y in 0..h {
                data[y * w + x] = col[y];
            }
        }
    }
}

/// The inverse of each block type's transform, in place on a `w` x `h`
/// block of dequantized coefficients (LLF already in place).
fn transform(b: &mut [f32], w: usize, h: usize, kind: Kind) {
    match kind {
        Kind::Dct2 => {
            aux_idct2(b, 2);
            aux_idct2(b, 4);
            aux_idct2(b, 8);
        }
        Kind::Dct4 => {
            aux_idct2(b, 2);
            let mut out = [0.0; 64];
            for y in 0..2 {
                for x in 0..2 {
                    let mut s = [0.0; 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            s[ix * 4 + iy] = b[(y + iy * 2) * 8 + x + ix * 2];
                        }
                    }
                    dct_2d(&mut s, 4, 4, true);
                    for iy in 0..4 {
                        for ix in 0..4 {
                            out[(y * 4 + iy) * 8 + x * 4 + ix] = s[iy * 4 + ix];
                        }
                    }
                }
            }
            b.copy_from_slice(&out);
        }
        Kind::Hornuss => {
            aux_idct2(b, 2);
            let mut out = [0.0; 64];
            for y in 0..2 {
                for x in 0..2 {
                    let mut s = [0.0_f32; 16];
                    for iy in 0..4 {
                        for ix in 0..4 {
                            s[iy * 4 + ix] = b[(y + iy * 2) * 8 + x + ix * 2];
                        }
                    }
                    let residual: f32 = s[1..].iter().sum();
                    let avg = s[0] - residual / 16.0;
                    s[0] = s[5];
                    s[5] = 0.0;
                    for v in &mut s {
                        *v += avg;
                    }
                    for iy in 0..4 {
                        for ix in 0..4 {
                            out[(y * 4 + iy) * 8 + x * 4 + ix] = s[iy * 4 + ix];
                        }
                    }
                }
            }
            b.copy_from_slice(&out);
        }
        Kind::Dct4x8 | Kind::Dct8x4 => {
            let (c0, c1) = (b[0], b[8]);
            b[0] = c0 + c1;
            b[8] = c0 - c1;
            let mut s = [0.0; 64];
            for half in 0..2 {
                let part = &mut s[half * 32..half * 32 + 32];
                for iy in 0..4 {
                    for ix in 0..8 {
                        part[iy * 8 + ix] = b[(iy * 2 + half) * 8 + ix];
                    }
                }
                dct_2d(part, 8, 4, true);
            }
            if kind == Kind::Dct8x4 {
                for y in 0..8 {
                    for x in 0..8 {
                        b[x * 8 + y] = s[y * 8 + x];
                    }
                }
            } else {
                b.copy_from_slice(&s);
            }
        }
        Kind::Afv0 | Kind::Afv1 | Kind::Afv2 | Kind::Afv3 => {
            afv(b, kind as usize - Kind::Afv0 as usize)
        }
        _ => dct_2d(b, w, h, true),
    }
}

fn aux_idct2(b: &mut [f32], size: usize) {
    let half = size / 2;
    let mut s = [[0.0_f32; 8]; 8];
    for y in 0..half {
        for x in 0..half {
            let c00 = b[y * 8 + x];
            let c01 = b[y * 8 + x + half];
            let c10 = b[(y + half) * 8 + x];
            let c11 = b[(y + half) * 8 + x + half];
            s[2 * y][2 * x] = c00 + c01 + c10 + c11;
            s[2 * y][2 * x + 1] = c00 + c01 - c10 - c11;
            s[2 * y + 1][2 * x] = c00 - c01 + c10 - c11;
            s[2 * y + 1][2 * x + 1] = c00 - c01 - c10 + c11;
        }
    }
    for y in 0..size {
        b[y * 8..y * 8 + size].copy_from_slice(&s[y][..size]);
    }
}

fn afv(b: &mut [f32], n: usize) {
    let (flip_x, flip_y) = (n % 2, n / 2);
    let mut coeff = [0.0_f32; 16];
    coeff[0] = (b[0] + b[1] + b[8]) * 4.0;
    for (i, v) in coeff.iter_mut().enumerate().skip(1) {
        *v = b[(2 * (i / 4)) * 8 + 2 * (i % 4)];
    }
    let mut samples = [0.0_f32; 16];
    for (c, basis) in coeff.iter().zip(&AFV_BASIS) {
        for (s, &bv) in samples.iter_mut().zip(basis) {
            *s += c * bv;
        }
    }
    let mut s44 = [0.0; 16];
    s44[0] = b[0] - b[1] + b[8];
    for iy in 0..4 {
        for ix in 0..4 {
            if ix | iy != 0 {
                s44[ix * 4 + iy] = b[(2 * iy) * 8 + 2 * ix + 1];
            }
        }
    }
    dct_2d(&mut s44, 4, 4, true);
    let mut s48 = [0.0; 32];
    s48[0] = b[0] - b[8];
    for iy in 0..4 {
        for ix in 0..8 {
            if ix | iy != 0 {
                s48[iy * 8 + ix] = b[(2 * iy + 1) * 8 + ix];
            }
        }
    }
    dct_2d(&mut s48, 8, 4, true);
    for iy in 0..4 {
        let ay = if flip_y == 0 { iy } else { 3 - iy };
        for ix in 0..4 {
            let ax = if flip_x == 0 { ix } else { 3 - ix };
            b[(flip_y * 4 + iy) * 8 + flip_x * 4 + ix] = samples[ay * 4 + ax];
        }
    }
    for iy in 0..4 {
        for ix in 0..4 {
            b[(flip_y * 4 + iy) * 8 + (1 - flip_x) * 4 + ix] = s44[iy * 4 + ix];
        }
    }
    for iy in 0..4 {
        let y = (1 - flip_y) * 4 + iy;
        b[y * 8..y * 8 + 8].copy_from_slice(&s48[iy * 8..iy * 8 + 8]);
    }
}

const AFV_BASIS: [[f32; 16]; 16] = [
    [0.25; 16],
    [
        0.876902929799142,
        0.2206518106944235,
        -0.10140050393753763,
        -0.1014005039375375,
        0.2206518106944236,
        -0.10140050393753777,
        -0.10140050393753772,
        -0.10140050393753763,
        -0.10140050393753758,
        -0.10140050393753769,
        -0.1014005039375375,
        -0.10140050393753768,
        -0.10140050393753768,
        -0.10140050393753759,
        -0.10140050393753763,
        -0.10140050393753741,
    ],
    [
        0.0,
        0.0,
        0.40670075830260755,
        0.44444816619734445,
        0.0,
        0.0,
        0.19574399372042936,
        0.2929100136981264,
        -0.40670075830260716,
        -0.19574399372042872,
        0.0,
        0.11379074460448091,
        -0.44444816619734384,
        -0.29291001369812636,
        -0.1137907446044814,
        0.0,
    ],
    [
        0.0,
        0.0,
        -0.21255748058288748,
        0.3085497062849767,
        0.0,
        0.4706702258572536,
        -0.1621205195722993,
        0.0,
        -0.21255748058287047,
        -0.16212051957228327,
        -0.47067022585725277,
        -0.1464291867126764,
        0.3085497062849487,
        0.0,
        -0.14642918671266536,
        0.4251149611657548,
    ],
    [
        0.0,
        -0.7071067811865474,
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        -0.4105377591765233,
        0.6235485373547691,
        -0.06435071657946274,
        -0.06435071657946266,
        0.6235485373547694,
        -0.06435071657946284,
        -0.0643507165794628,
        -0.06435071657946274,
        -0.06435071657946272,
        -0.06435071657946279,
        -0.06435071657946266,
        -0.06435071657946277,
        -0.06435071657946277,
        -0.06435071657946273,
        -0.06435071657946274,
        -0.0643507165794626,
    ],
    [
        0.0,
        0.0,
        -0.4517556589999482,
        0.15854503551840063,
        0.0,
        -0.04038515160822202,
        0.0074182263792423875,
        0.39351034269210167,
        -0.45175565899994635,
        0.007418226379244351,
        0.1107416575309343,
        0.08298163094882051,
        0.15854503551839705,
        0.3935103426921022,
        0.0829816309488214,
        -0.45175565899994796,
    ],
    [
        0.0,
        0.0,
        -0.304684750724869,
        0.5112616136591823,
        0.0,
        0.0,
        -0.290480129728998,
        -0.06578701549142804,
        0.304684750724884,
        0.2904801297290076,
        0.0,
        -0.23889773523344604,
        -0.5112616136592012,
        0.06578701549142545,
        0.23889773523345467,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.3017929516615495,
        0.25792362796341184,
        0.0,
        0.16272340142866204,
        0.09520022653475037,
        0.0,
        0.3017929516615503,
        0.09520022653475055,
        -0.16272340142866173,
        -0.35312385449816297,
        0.25792362796341295,
        0.0,
        -0.3531238544981624,
        -0.6035859033230976,
    ],
    [
        0.0,
        0.0,
        0.40824829046386274,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4082482904638628,
        -0.4082482904638635,
        0.0,
        0.0,
        -0.40824829046386296,
        0.0,
        0.4082482904638634,
        0.408248290463863,
        0.0,
    ],
    [
        0.0,
        0.0,
        0.1747866975480809,
        0.0812611176717539,
        0.0,
        0.0,
        -0.3675398009862027,
        -0.307882213957909,
        -0.17478669754808135,
        0.3675398009862011,
        0.0,
        0.4826689115059883,
        -0.08126111767175039,
        0.30788221395790305,
        -0.48266891150598584,
        0.0,
    ],
    [
        0.0,
        0.0,
        -0.21105601049335784,
        0.18567180916109802,
        0.0,
        0.0,
        0.49215859013738733,
        -0.38525013709251915,
        0.21105601049335806,
        -0.49215859013738905,
        0.0,
        0.17419412659916217,
        -0.18567180916109904,
        0.3852501370925211,
        -0.1741941265991621,
        0.0,
    ],
    [
        0.0,
        0.0,
        -0.14266084808807264,
        -0.3416446842253372,
        0.0,
        0.7367497537172237,
        0.24627107722075148,
        -0.08574019035519306,
        -0.14266084808807344,
        0.24627107722075137,
        0.14883399227113567,
        -0.04768680350229251,
        -0.3416446842253373,
        -0.08574019035519267,
        -0.047686803502292804,
        -0.14266084808807242,
    ],
    [
        0.0,
        0.0,
        -0.13813540350758585,
        0.3302282550303788,
        0.0,
        0.08755115000587084,
        -0.07946706605909573,
        -0.4613374887461511,
        -0.13813540350758294,
        -0.07946706605910261,
        0.49724647109535086,
        0.12538059448563663,
        0.3302282550303805,
        -0.4613374887461554,
        0.12538059448564315,
        -0.13813540350758452,
    ],
    [
        0.0,
        0.0,
        -0.17437602599651067,
        0.0702790691196284,
        0.0,
        -0.2921026642334881,
        0.3623817333531167,
        0.0,
        -0.1743760259965108,
        0.36238173335311646,
        0.29210266423348785,
        -0.4326608024727445,
        0.07027906911962818,
        0.0,
        -0.4326608024727457,
        0.34875205199302267,
    ],
    [
        0.0,
        0.0,
        0.11354987314994337,
        -0.07417504595810355,
        0.0,
        0.19402893032594343,
        -0.435190496523228,
        0.21918684838857466,
        0.11354987314994257,
        -0.4351904965232251,
        0.5550443808910661,
        -0.25468277124066463,
        -0.07417504595810233,
        0.2191868483885728,
        -0.25468277124066413,
        0.1135498731499429,
    ],
];

/// For the type system: an unsupported-feature error.
#[allow(dead_code, reason = "kept for features not yet implemented")]
fn unsupported(what: &str) -> PixelsError {
    PixelsError::unsupported(format!("JPEG XL: {what}"))
}
