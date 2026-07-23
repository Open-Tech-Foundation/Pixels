//! Inverse transforms and dequantization (spec §7.12–§7.13).
//!
//! A transform block is reconstructed by dequantizing the coefficient levels
//! (`dequantize`), running a 2D inverse transform (`inverse_transform_2d`), and
//! adding the result to the prediction (`add_residual`). The `dsp` submodule
//! below holds the full lossy machinery — the DCT/ADST/identity butterfly
//! network and the 2D driver — transcribed from the spec's ordered steps.
//!
//! The lossless path is just the qindex-0 corner of that machinery: `dc_q` and
//! `ac_q` are both 4, `dqDenom` is 1, and the 4x4 Walsh–Hadamard transform
//! divides the dequantiser's "times 4" straight back out, which is why lossless
//! is bit-exact. It flows through the same `dequantize` + `inverse_transform_2d`
//! pair as the lossy path, with `Lossless == 1`.

/// Reconstruct a transform block by adding the `Residual` to the prediction and
/// clipping to the sample range (`Clip1`, the reconstruct process §7.12.3 step
/// 3), honouring the transform type's `flipUD`/`flipLR`. `prediction` is
/// row-major `residual.width * residual.height`; the result is the same shape.
///
/// The residual is stored pre-flip, so `Residual[i][j]` is added to the output
/// at `(flipUD ? h-1-i : i, flipLR ? w-1-j : j)` — the prediction already sits
/// there. The lossless path is the `DCT_DCT` (no-flip) 4x4 corner of this.
#[must_use]
pub fn add_residual(
    prediction: &[u16],
    residual: &Residual,
    tx_type: TxType,
    bit_depth: u8,
) -> Vec<u16> {
    let w = residual.width;
    let h = residual.height;
    let max = i64::from((1_u32 << bit_depth) - 1);
    let flip_ud = tx_type.flip_ud();
    let flip_lr = tx_type.flip_lr();
    let mut out = prediction.to_vec();
    for i in 0..h {
        for j in 0..w {
            let xx = if flip_lr { w - j - 1 } else { j };
            let yy = if flip_ud { h - i - 1 } else { i };
            let idx = yy * w + xx;
            let pred = prediction.get(idx).copied().unwrap_or(0);
            let value = i64::from(pred) + i64::from(residual.at(i, j));
            if let Some(cell) = out.get_mut(idx) {
                *cell = value.clamp(0, max) as u16;
            }
        }
    }
    out
}

/// Reconstruct a 4x4 block (§7.12.3 step 3): a thin wrapper over [`add_residual`]
/// for the `DCT_DCT` (no-flip) case the lossless 4x4 tile drives.
#[must_use]
pub fn add_residual_4x4(
    prediction: &[[u16; 4]; 4],
    residual: &Residual,
    bit_depth: u8,
) -> [[u16; 4]; 4] {
    let mut flat = [0_u16; 16];
    for (i, row) in prediction.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if let Some(cell) = flat.get_mut(i * 4 + j) {
                *cell = v;
            }
        }
    }
    let out = add_residual(&flat, residual, TxType::DctDct, bit_depth);
    let mut result = [[0_u16; 4]; 4];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = out.get(i * 4 + j).copied().unwrap_or(0);
        }
    }
    result
}

pub use dsp::{Dequant, Residual, TxSize, TxType, ac_q, dc_q, dequantize, inverse_transform_2d};

/// The lossy inverse-transform machinery (spec §7.13): the DCT/ADST/identity
/// butterfly network, the 2D transform driver, and the dequantiser lookups.
///
/// This lives in its own module because the 1D transforms are transcribed
/// straight from the spec's ordered `B`/`H` butterfly steps, which read and
/// write a fixed-length working array `T` by index. Expressing that through
/// `.get()` would obscure the correspondence with the spec, so the module opts
/// into `indexing_slicing`: every index is a spec constant bounded below the
/// array length (`T` is 64 long, the largest transform).
mod dsp {
    #![allow(
        clippy::indexing_slicing,
        clippy::needless_range_loop,
        reason = "fixed-size DSP working arrays; every index is a spec-bounded \
                  constant strictly below the array length, and the permutation \
                  loops index a separate snapshot from the array they write"
    )]

    /// `Round2(x, n)` (§4.7): rounding right shift, `x` unchanged when `n == 0`.
    fn round2(x: i64, n: u32) -> i64 {
        if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
    }

    /// `Clip3(-(1<<(bits-1)), (1<<(bits-1))-1, x)`: clamp to a signed `bits` range.
    fn clip_signed(x: i64, bits: u32) -> i64 {
        let lo = -(1_i64 << (bits - 1));
        let hi = (1_i64 << (bits - 1)) - 1;
        x.clamp(lo, hi)
    }

    /// `Cos128_Lookup` (§7.13.2.1): `round(4096 * cos(angle * pi / 128))` for
    /// `angle` in `0..=64`.
    const COS128_LOOKUP: [i64; 65] = [
        4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857,
        3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102,
        3035, 2967, 2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931,
        1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401,
        301, 201, 101, 0,
    ];

    /// `cos128(angle)` (§7.13.2.1), reducing the angle modulo 256.
    fn cos128(angle: i32) -> i64 {
        let a = (angle & 255) as usize;
        match a {
            0..=64 => COS128_LOOKUP[a],
            65..=128 => -COS128_LOOKUP[128 - a],
            129..=192 => -COS128_LOOKUP[a - 128],
            _ => COS128_LOOKUP[256 - a],
        }
    }

    /// `sin128(angle) = cos128(angle - 64)` (§7.13.2.1).
    fn sin128(angle: i32) -> i64 {
        cos128(angle - 64)
    }

    /// `brev(numBits, x)` (§7.13.2.1): bit-reversal of the low `num_bits` of `x`.
    fn brev(num_bits: u32, x: usize) -> usize {
        let mut t = 0;
        for i in 0..num_bits {
            let bit = (x >> i) & 1;
            t += bit << (num_bits - 1 - i);
        }
        t
    }

    /// `B(a, b, angle, flip, r)`: butterfly rotation (§7.13.2.1). The `r` clamp is
    /// a conformance requirement on the inputs, not an operation, so it is unused.
    fn butterfly(t: &mut [i64; 64], a: usize, b: usize, angle: i32, flip: bool) {
        let (ta, tb) = (t[a], t[b]);
        let x = ta * cos128(angle) - tb * sin128(angle);
        let y = ta * sin128(angle) + tb * cos128(angle);
        t[a] = round2(x, 12);
        t[b] = round2(y, 12);
        if flip {
            t.swap(a, b);
        }
    }

    /// `H(a, b, flip, r)`: Hadamard rotation (§7.13.2.1). A flip swaps the pair.
    fn hadamard(t: &mut [i64; 64], a: usize, b: usize, flip: bool, r: u32) {
        let (a, b) = if flip { (b, a) } else { (a, b) };
        let (x, y) = (t[a], t[b]);
        t[a] = clip_signed(x + y, r);
        t[b] = clip_signed(x - y, r);
    }

    /// Inverse DCT array permutation process (§7.13.2.2).
    fn dct_permute(t: &mut [i64; 64], n: u32) {
        let len = 1usize << n;
        let copy = *t;
        for i in 0..len {
            t[i] = copy[brev(n, i)];
        }
    }

    /// Inverse DCT process (§7.13.2.3) for a length-`2^n` array (`2 <= n <= 6`).
    #[allow(
        clippy::too_many_lines,
        reason = "a faithful transcription of the spec's 31 ordered butterfly steps"
    )]
    fn inverse_dct(t: &mut [i64; 64], n: u32, r: u32) {
        dct_permute(t, n);
        // Each numbered block matches the like-numbered step of §7.13.2.3.
        if n == 6 {
            for i in 0..16 {
                butterfly(t, 32 + i, 63 - i, 63 - 4 * brev(4, i) as i32, false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                butterfly(t, 16 + i, 31 - i, 6 + ((brev(3, 7 - i) as i32) << 3), false);
            }
        }
        if n == 6 {
            for i in 0..16 {
                hadamard(t, 32 + i * 2, 33 + i * 2, i & 1 == 1, r);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                butterfly(t, 8 + i, 15 - i, 12 + ((brev(2, 3 - i) as i32) << 4), false);
            }
        }
        if n >= 5 {
            for i in 0..8 {
                hadamard(t, 16 + 2 * i, 17 + 2 * i, i & 1 == 1, r);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..2 {
                    butterfly(
                        t,
                        62 - i * 4 - j,
                        33 + i * 4 + j,
                        60 - 16 * brev(2, i) as i32 + 64 * j as i32,
                        true,
                    );
                }
            }
        }
        if n >= 3 {
            for i in 0..2 {
                butterfly(t, 4 + i, 7 - i, 56 - 32 * i as i32, false);
            }
        }
        if n >= 4 {
            for i in 0..4 {
                hadamard(t, 8 + 2 * i, 9 + 2 * i, i & 1 == 1, r);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..2 {
                    butterfly(
                        t,
                        30 - 4 * i - j,
                        17 + 4 * i + j,
                        24 + ((j as i32) << 6) + (((1 - i) as i32) << 5),
                        true,
                    );
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                for j in 0..2 {
                    hadamard(t, 32 + i * 4 + j, 35 + i * 4 - j, i & 1 == 1, r);
                }
            }
        }
        for i in 0..2 {
            butterfly(t, 2 * i, 2 * i + 1, 32 + 16 * i as i32, i == 0);
        }
        if n >= 3 {
            for i in 0..2 {
                hadamard(t, 4 + 2 * i, 5 + 2 * i, i == 1, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                butterfly(t, 14 - i, 9 + i, 48 + 64 * i as i32, true);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                for j in 0..2 {
                    hadamard(t, 16 + 4 * i + j, 19 + 4 * i - j, i & 1 == 1, r);
                }
            }
        }
        if n == 6 {
            for i in 0..2 {
                for j in 0..4 {
                    butterfly(
                        t,
                        61 - i * 8 - j,
                        34 + i * 8 + j,
                        56 - i as i32 * 32 + (j as i32 >> 1) * 64,
                        true,
                    );
                }
            }
        }
        for i in 0..2 {
            hadamard(t, i, 3 - i, false, r);
        }
        if n >= 3 {
            butterfly(t, 6, 5, 32, true);
        }
        if n >= 4 {
            for i in 0..2 {
                for j in 0..2 {
                    hadamard(t, 8 + 4 * i + j, 11 + 4 * i - j, i == 1, r);
                }
            }
        }
        if n >= 5 {
            for i in 0..4 {
                butterfly(t, 29 - i, 18 + i, 48 + (i as i32 >> 1) * 64, true);
            }
        }
        if n == 6 {
            for i in 0..4 {
                for j in 0..4 {
                    hadamard(t, 32 + 8 * i + j, 39 + 8 * i - j, i & 1 == 1, r);
                }
            }
        }
        if n >= 3 {
            for i in 0..4 {
                hadamard(t, i, 7 - i, false, r);
            }
        }
        if n >= 4 {
            for i in 0..2 {
                butterfly(t, 13 - i, 10 + i, 32, true);
            }
        }
        if n >= 5 {
            for i in 0..2 {
                for j in 0..4 {
                    hadamard(t, 16 + i * 8 + j, 23 + i * 8 - j, i == 1, r);
                }
            }
        }
        if n == 6 {
            for i in 0..8 {
                butterfly(t, 59 - i, 36 + i, if i < 4 { 48 } else { 112 }, true);
            }
        }
        if n >= 4 {
            for i in 0..8 {
                hadamard(t, i, 15 - i, false, r);
            }
        }
        if n >= 5 {
            for i in 0..4 {
                butterfly(t, 27 - i, 20 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..8 {
                hadamard(t, 32 + i, 47 - i, false, r);
                hadamard(t, 48 + i, 63 - i, true, r);
            }
        }
        if n >= 5 {
            for i in 0..16 {
                hadamard(t, i, 31 - i, false, r);
            }
        }
        if n == 6 {
            for i in 0..8 {
                butterfly(t, 55 - i, 40 + i, 32, true);
            }
        }
        if n == 6 {
            for i in 0..32 {
                hadamard(t, i, 63 - i, false, r);
            }
        }
    }

    /// Inverse ADST input array permutation (§7.13.2.4), `3 <= n <= 4`.
    fn adst_permute_in(t: &mut [i64; 64], n: u32) {
        let n0 = 1usize << n;
        let copy = *t;
        for i in 0..n0 {
            let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
            t[i] = copy[idx];
        }
    }

    /// Inverse ADST output array permutation (§7.13.2.5), `3 <= n <= 4`.
    fn adst_permute_out(t: &mut [i64; 64], n: u32) {
        let n0 = 1usize << n;
        let copy = *t;
        for i in 0..n0 {
            let a = (i >> 3) & 1;
            let b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
            let c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
            let d = (i & 1) ^ ((i >> 1) & 1);
            let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
            t[i] = if i & 1 == 1 { -copy[idx] } else { copy[idx] };
        }
    }

    /// Inverse ADST4 process (§7.13.2.6).
    fn inverse_adst4(t: &mut [i64; 64]) {
        const SINPI_1_9: i64 = 1321;
        const SINPI_2_9: i64 = 2482;
        const SINPI_3_9: i64 = 3344;
        const SINPI_4_9: i64 = 3803;
        let (t0, t1, t2, t3) = (t[0], t[1], t[2], t[3]);
        let mut s = [
            SINPI_1_9 * t0,
            SINPI_2_9 * t0,
            SINPI_3_9 * t1,
            SINPI_4_9 * t2,
            SINPI_1_9 * t2,
            SINPI_2_9 * t3,
            SINPI_4_9 * t3,
        ];
        let a7 = t0 - t2;
        let b7 = a7 + t3;
        s[0] += s[3];
        s[1] -= s[4];
        s[3] = s[2];
        s[2] = SINPI_3_9 * b7;
        s[0] += s[5];
        s[1] -= s[6];
        let x0 = s[0] + s[3];
        let x1 = s[1] + s[3];
        let x2 = s[2];
        let x3 = s[0] + s[1] - s[3];
        t[0] = round2(x0, 12);
        t[1] = round2(x1, 12);
        t[2] = round2(x2, 12);
        t[3] = round2(x3, 12);
    }

    /// Inverse ADST8 process (§7.13.2.7).
    fn inverse_adst8(t: &mut [i64; 64], r: u32) {
        adst_permute_in(t, 3);
        for i in 0..4 {
            butterfly(t, 2 * i, 2 * i + 1, 60 - 16 * i as i32, true);
        }
        for i in 0..4 {
            hadamard(t, i, 4 + i, false, r);
        }
        for i in 0..2 {
            butterfly(t, 4 + 3 * i, 5 + i, 48 - 32 * i as i32, true);
        }
        for i in 0..2 {
            for j in 0..2 {
                hadamard(t, 4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..2 {
            butterfly(t, 2 + 4 * i, 3 + 4 * i, 32, true);
        }
        adst_permute_out(t, 3);
    }

    /// Inverse ADST16 process (§7.13.2.8).
    fn inverse_adst16(t: &mut [i64; 64], r: u32) {
        adst_permute_in(t, 4);
        for i in 0..8 {
            butterfly(t, 2 * i, 2 * i + 1, 62 - 8 * i as i32, true);
        }
        for i in 0..8 {
            hadamard(t, i, 8 + i, false, r);
        }
        for i in 0..2 {
            butterfly(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i as i32, true);
            butterfly(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i as i32, true);
        }
        for i in 0..4 {
            for j in 0..2 {
                hadamard(t, 8 * j + i, 4 + 8 * j + i, false, r);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                butterfly(
                    t,
                    4 + 8 * j + 3 * i,
                    5 + 8 * j + i,
                    48 - 32 * i as i32,
                    true,
                );
            }
        }
        for i in 0..2 {
            for j in 0..4 {
                hadamard(t, 4 * j + i, 2 + 4 * j + i, false, r);
            }
        }
        for i in 0..4 {
            butterfly(t, 2 + 4 * i, 3 + 4 * i, 32, true);
        }
        adst_permute_out(t, 4);
    }

    /// Inverse ADST process (§7.13.2.9) for a length-`2^n` array (`2 <= n <= 4`).
    fn inverse_adst(t: &mut [i64; 64], n: u32, r: u32) {
        match n {
            2 => inverse_adst4(t),
            3 => inverse_adst8(t, r),
            _ => inverse_adst16(t, r),
        }
    }

    /// Inverse identity transform process (§7.13.2.15), `2 <= n <= 5`.
    fn inverse_identity(t: &mut [i64; 64], n: u32) {
        let len = 1usize << n;
        for cell in t.iter_mut().take(len) {
            *cell = match n {
                2 => round2(*cell * 5793, 12),
                3 => *cell * 2,
                4 => round2(*cell * 11586, 12),
                _ => *cell * 4,
            };
        }
    }

    /// Inverse Walsh–Hadamard transform (§7.13.2.10) on the first four elements.
    fn inverse_wht(t: &mut [i64; 64], shift: u32) {
        let mut a = t[0] >> shift;
        let mut c = t[1] >> shift;
        let mut d = t[2] >> shift;
        let mut b = t[3] >> shift;
        a += c;
        d -= b;
        let e = (a - d) >> 1;
        b = e - b;
        c = e - c;
        a -= b;
        d += c;
        t[0] = a;
        t[1] = b;
        t[2] = c;
        t[3] = d;
    }

    /// Which 1D transform a row or column pass runs, before flipping.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Dct,
        Adst,
        Identity,
    }

    fn apply_1d(t: &mut [i64; 64], kind: Kind, n: u32, r: u32) {
        match kind {
            Kind::Dct => inverse_dct(t, n, r),
            Kind::Adst => inverse_adst(t, n, r),
            Kind::Identity => inverse_identity(t, n),
        }
    }

    /// The 19 transform sizes (`TX_SIZES_ALL`, §6.10.28 order).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[allow(missing_docs, reason = "each variant is a self-describing WxH size")]
    pub enum TxSize {
        Tx4x4,
        Tx8x8,
        Tx16x16,
        Tx32x32,
        Tx64x64,
        Tx4x8,
        Tx8x4,
        Tx8x16,
        Tx16x8,
        Tx16x32,
        Tx32x16,
        Tx32x64,
        Tx64x32,
        Tx4x16,
        Tx16x4,
        Tx8x32,
        Tx32x8,
        Tx16x64,
        Tx64x16,
    }

    const TX_WIDTH_LOG2: [u32; 19] = [2, 3, 4, 5, 6, 2, 3, 3, 4, 4, 5, 5, 6, 2, 4, 3, 5, 4, 6];
    const TX_HEIGHT_LOG2: [u32; 19] = [2, 3, 4, 5, 6, 3, 2, 4, 3, 5, 4, 6, 5, 4, 2, 5, 3, 6, 4];
    const TRANSFORM_ROW_SHIFT: [u32; 19] =
        [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];
    // `Tx_Size_Sqr[txSz]` / `Tx_Size_Sqr_Up[txSz]` as the square size's index
    // (TX_4X4=0..TX_64X64=4): the square tx with side Min(w,h) resp. Max(w,h).
    const TX_SIZE_SQR: [u32; 19] = [0, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2];
    const TX_SIZE_SQR_UP: [u32; 19] = [0, 1, 2, 3, 4, 1, 1, 2, 2, 3, 3, 4, 4, 2, 2, 3, 3, 4, 4];
    // `Adjusted_Tx_Size[txSz]`: the coded tx size, mapping any 64-wide/high size
    // down to its 32 counterpart (only the top-left 32x32 of coeffs is coded).
    const ADJUSTED_TX_SIZE: [usize; 19] = [
        0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 3, 3, 13, 14, 15, 16, 9, 10,
    ];

    impl TxSize {
        /// `Tx_Width_Log2[txSz]`.
        #[must_use]
        pub fn log2_width(self) -> u32 {
            TX_WIDTH_LOG2[self as usize]
        }

        /// `Tx_Height_Log2[txSz]`.
        #[must_use]
        pub fn log2_height(self) -> u32 {
            TX_HEIGHT_LOG2[self as usize]
        }

        /// `Tx_Size_Sqr[txSz]` as the square size's index (`Min(w,h)`).
        #[must_use]
        pub fn sqr_idx(self) -> u32 {
            TX_SIZE_SQR[self as usize]
        }

        /// `Tx_Size_Sqr_Up[txSz]` as the square size's index (`Max(w,h)`).
        #[must_use]
        pub fn sqr_up_idx(self) -> u32 {
            TX_SIZE_SQR_UP[self as usize]
        }

        /// `txSzCtx = (Tx_Size_Sqr + Tx_Size_Sqr_Up + 1) >> 1`, the coefficient
        /// CDF bucket (0..=4).
        #[must_use]
        pub fn tx_size_ctx(self) -> usize {
            ((self.sqr_idx() + self.sqr_up_idx() + 1) >> 1) as usize
        }

        /// `Tx_Width_Log2[Adjusted_Tx_Size[txSz]]`: the coded block's width log2,
        /// which drives the coefficient position maths (`bwl`).
        #[must_use]
        pub fn adjusted_log2_width(self) -> u32 {
            TX_WIDTH_LOG2[ADJUSTED_TX_SIZE[self as usize]]
        }

        /// `Tx_Height[Adjusted_Tx_Size[txSz]]`: the coded block's height.
        #[must_use]
        pub fn adjusted_height(self) -> usize {
            1 << TX_HEIGHT_LOG2[ADJUSTED_TX_SIZE[self as usize]]
        }

        /// `Tx_Width[Adjusted_Tx_Size[txSz]]`: the coded block's width.
        #[must_use]
        pub fn adjusted_width(self) -> usize {
            1 << self.adjusted_log2_width()
        }

        /// `segEob` (§5.11.39): the number of scan positions the block can code.
        #[must_use]
        pub fn seg_eob(self) -> usize {
            match self {
                TxSize::Tx16x64 | TxSize::Tx64x16 => 512,
                _ => (self.width() * self.height()).min(1024),
            }
        }

        /// `eobMultisize` (§5.11.39): selects the `eob_pt_*` alphabet (0..=6).
        #[must_use]
        pub fn eob_multisize(self) -> usize {
            (self.log2_width().min(5) + self.log2_height().min(5) - 4) as usize
        }

        /// The transform width in samples, `1 << Tx_Width_Log2[txSz]`.
        #[must_use]
        pub fn width(self) -> usize {
            1 << self.log2_width()
        }

        /// The transform height in samples, `1 << Tx_Height_Log2[txSz]`.
        #[must_use]
        pub fn height(self) -> usize {
            1 << self.log2_height()
        }

        fn row_shift(self) -> u32 {
            TRANSFORM_ROW_SHIFT[self as usize]
        }

        /// `dqDenom` (§7.12.3): the shared denominator of the dequantiser.
        fn dq_denom(self) -> i64 {
            match self {
                TxSize::Tx32x32
                | TxSize::Tx16x32
                | TxSize::Tx32x16
                | TxSize::Tx16x64
                | TxSize::Tx64x16 => 2,
                TxSize::Tx64x64 | TxSize::Tx32x64 | TxSize::Tx64x32 => 4,
                _ => 1,
            }
        }
    }

    /// `PlaneTxType`: the 16 transform types (§6.10.28 order). The first word
    /// names the column (vertical) transform, the second the row (horizontal).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    #[allow(
        missing_docs,
        reason = "each variant names its column_row transform pair per §6.10.28"
    )]
    pub enum TxType {
        DctDct,
        AdstDct,
        DctAdst,
        AdstAdst,
        FlipadstDct,
        DctFlipadst,
        FlipadstFlipadst,
        AdstFlipadst,
        FlipadstAdst,
        Idtx,
        VDct,
        HDct,
        VAdst,
        HAdst,
        VFlipadst,
        HFlipadst,
    }

    impl TxType {
        /// The row (horizontal) 1D transform — the type's second word.
        fn row_kind(self) -> Kind {
            match self {
                TxType::DctDct | TxType::AdstDct | TxType::FlipadstDct | TxType::HDct => Kind::Dct,
                TxType::Idtx | TxType::VDct | TxType::VAdst | TxType::VFlipadst => Kind::Identity,
                _ => Kind::Adst,
            }
        }

        /// The column (vertical) 1D transform — the type's first word.
        fn col_kind(self) -> Kind {
            match self {
                TxType::DctDct | TxType::DctAdst | TxType::DctFlipadst | TxType::VDct => Kind::Dct,
                TxType::Idtx | TxType::HDct | TxType::HAdst | TxType::HFlipadst => Kind::Identity,
                _ => Kind::Adst,
            }
        }

        /// `flipUD` (§7.12.3): the column transform is a flipped ADST.
        #[must_use]
        pub fn flip_ud(self) -> bool {
            matches!(
                self,
                TxType::FlipadstDct
                    | TxType::FlipadstAdst
                    | TxType::VFlipadst
                    | TxType::FlipadstFlipadst
            )
        }

        /// `flipLR` (§7.12.3): the row transform is a flipped ADST.
        #[must_use]
        pub fn flip_lr(self) -> bool {
            matches!(
                self,
                TxType::DctFlipadst
                    | TxType::AdstFlipadst
                    | TxType::HFlipadst
                    | TxType::FlipadstFlipadst
            )
        }
    }

    /// A reconstructed residual block, `width` by `height` samples in raster
    /// order. Values are pre-flip: the caller applies `flip_ud`/`flip_lr` when
    /// adding to the prediction (§7.12.3 step 3).
    pub struct Residual {
        /// The residual width in samples.
        pub width: usize,
        /// The residual height in samples.
        pub height: usize,
        values: [i32; 64 * 64],
    }

    impl Residual {
        /// The residual at row `i`, column `j` (0 outside the block).
        #[must_use]
        pub fn at(&self, i: usize, j: usize) -> i32 {
            if i < self.height && j < self.width {
                self.values[i * self.width + j]
            } else {
                0
            }
        }
    }

    /// 2D inverse transform process (§7.13.3). `dequant` holds the dequantised
    /// coefficients `Dequant[i][j]` in raster order over the `min(32,w)` by
    /// `min(32,h)` populated region; entries beyond it are treated as zero.
    #[must_use]
    pub fn inverse_transform_2d(
        dequant: &Dequant,
        tx_size: TxSize,
        tx_type: TxType,
        lossless: bool,
        bit_depth: u8,
    ) -> Residual {
        let log2w = tx_size.log2_width();
        let log2h = tx_size.log2_height();
        let w = 1usize << log2w;
        let h = 1usize << log2h;
        let row_shift = if lossless { 0 } else { tx_size.row_shift() };
        let col_shift = if lossless { 0 } else { 4 };
        let row_clamp = u32::from(bit_depth) + 8;
        let col_clamp = (u32::from(bit_depth) + 6).max(16);
        let rect_scale = log2w.abs_diff(log2h) == 1;

        let mut residual = [0_i64; 64 * 64];
        let mut t = [0_i64; 64];

        // Row transforms.
        for i in 0..h {
            for (j, cell) in t.iter_mut().enumerate().take(w) {
                *cell = if i < 32 && j < 32 {
                    dequant.at(i, j)
                } else {
                    0
                };
            }
            if rect_scale {
                for cell in t.iter_mut().take(w) {
                    *cell = round2(*cell * 2896, 12);
                }
            }
            if lossless {
                inverse_wht(&mut t, 2);
            } else {
                apply_1d(&mut t, tx_type.row_kind(), log2w, row_clamp);
            }
            for j in 0..w {
                residual[i * w + j] = round2(t[j], row_shift);
            }
        }

        // Clamp between the passes.
        for value in residual.iter_mut().take(w * h) {
            *value = clip_signed(*value, col_clamp);
        }

        // Column transforms.
        for j in 0..w {
            for (i, cell) in t.iter_mut().enumerate().take(h) {
                *cell = residual[i * w + j];
            }
            if lossless {
                inverse_wht(&mut t, 0);
            } else {
                apply_1d(&mut t, tx_type.col_kind(), log2h, col_clamp);
            }
            for i in 0..h {
                residual[i * w + j] = round2(t[i], col_shift);
            }
        }

        let mut values = [0_i32; 64 * 64];
        for (out, &v) in values.iter_mut().zip(residual.iter()).take(w * h) {
            *out = v as i32;
        }
        Residual {
            width: w,
            height: h,
            values,
        }
    }

    /// The dequantised coefficient block `Dequant[i][j]`, raster order over the
    /// populated `tw` by `th` region (`tw = min(32,w)`, `th = min(32,h)`).
    pub struct Dequant {
        width: usize,
        height: usize,
        values: [i64; 32 * 32],
    }

    impl Dequant {
        fn at(&self, i: usize, j: usize) -> i64 {
            if i < self.height && j < self.width {
                self.values[i * self.width + j]
            } else {
                0
            }
        }
    }

    /// Dequantise one transform block (§7.12.3 step 1). `quant` holds `Quant[]`
    /// in raster order over the `tw` by `th` region; `dc_quant`/`ac_quant` are
    /// the plane's DC/AC quantiser steps. No quantiser matrix is applied.
    #[must_use]
    pub fn dequantize(
        quant: &[i32],
        tx_size: TxSize,
        dc_quant: i64,
        ac_quant: i64,
        bit_depth: u8,
    ) -> Dequant {
        let tw = tx_size.width().min(32);
        let th = tx_size.height().min(32);
        let denom = tx_size.dq_denom();
        let mut values = [0_i64; 32 * 32];
        for (idx, out) in values.iter_mut().enumerate().take(tw * th) {
            let level = quant.get(idx).copied().unwrap_or(0);
            let q = if idx == 0 { dc_quant } else { ac_quant };
            let dq = i64::from(level) * q;
            let sign = if dq < 0 { -1 } else { 1 };
            let dq2 = sign * ((dq.abs() & 0xFF_FFFF) / denom);
            *out = clip_signed(dq2, 8 + u32::from(bit_depth));
        }
        Dequant {
            width: tw,
            height: th,
            values,
        }
    }

    /// `Dc_Qlookup[(BitDepth-8)>>1][Clip3(0,255,b)]` (§7.12.2).
    #[must_use]
    pub fn dc_q(bit_depth: u8, b: i32) -> i64 {
        let row = usize::from(bit_depth.saturating_sub(8) >> 1).min(2);
        let col = b.clamp(0, 255) as usize;
        i64::from(DC_QLOOKUP[row][col])
    }

    /// `Ac_Qlookup[(BitDepth-8)>>1][Clip3(0,255,b)]` (§7.12.2).
    #[must_use]
    pub fn ac_q(bit_depth: u8, b: i32) -> i64 {
        let row = usize::from(bit_depth.saturating_sub(8) >> 1).min(2);
        let col = b.clamp(0, 255) as usize;
        i64::from(AC_QLOOKUP[row][col])
    }

    include!("quant_tables.rs");

    #[cfg(test)]
    #[allow(
        clippy::unwrap_used,
        clippy::panic,
        reason = "tests operate on known-good values and assert shapes directly"
    )]
    mod dsp_tests {
        use super::*;

        fn dequant_from(vals: &[(usize, i64)], tw: usize, th: usize) -> Dequant {
            let mut values = [0_i64; 32 * 32];
            for &(idx, v) in vals {
                values[idx] = v;
            }
            Dequant {
                width: tw,
                height: th,
                values,
            }
        }

        #[test]
        fn cos_and_sin_hit_the_reference_points() {
            assert_eq!(cos128(0), 4096);
            assert_eq!(cos128(64), 0);
            assert_eq!(cos128(128), -4096);
            assert_eq!(sin128(64), 4096);
            assert_eq!(sin128(0), 0);
        }

        #[test]
        fn brev_reverses_bits() {
            assert_eq!(brev(4, 1), 8);
            assert_eq!(brev(4, 0b0011), 0b1100);
            assert_eq!(brev(3, 0b001), 0b100);
        }

        #[test]
        fn identity_identity_scales_a_dc_block() {
            // 8x8 IDTX: each pass multiplies by 2, then colShift = 4 halves twice.
            // A single DC term becomes (dc*2*2) >> 4 spread only at [0][0].
            let dequant = dequant_from(&[(0, 32)], 8, 8);
            let res = inverse_transform_2d(&dequant, TxSize::Tx8x8, TxType::Idtx, false, 8);
            assert_eq!(res.width, 8);
            assert_eq!(res.height, 8);
            // Row pass: T[0]=32*2=64, rowShift=1 -> 32. Col pass: 32*2=64,
            // colShift=4 -> 4. Only column 0, row 0 is non-zero.
            assert_eq!(res.at(0, 0), 4);
            assert_eq!(res.at(0, 1), 0);
            assert_eq!(res.at(1, 0), 0);
        }

        #[test]
        fn dct_of_a_dc_only_block_is_flat() {
            // A DCT with only the DC coefficient set reconstructs a constant
            // block: every sample equal, no spatial variation.
            let dequant = dequant_from(&[(0, 512)], 8, 8);
            let res = inverse_transform_2d(&dequant, TxSize::Tx8x8, TxType::DctDct, false, 8);
            let first = res.at(0, 0);
            assert!(first != 0, "DC should reconstruct a non-zero level");
            for i in 0..8 {
                for j in 0..8 {
                    assert_eq!(res.at(i, j), first, "DCT DC block must be flat");
                }
            }
        }

        #[test]
        fn adst_dc_block_is_not_flat_but_symmetric_is_valid() {
            // ADST is not flat for a DC input; just assert it runs and produces
            // a populated 4x4 block without panicking.
            let dequant = dequant_from(&[(0, 256)], 4, 4);
            let res = inverse_transform_2d(&dequant, TxSize::Tx4x4, TxType::AdstAdst, false, 8);
            assert_eq!(res.width, 4);
            assert_eq!(res.height, 4);
        }

        #[test]
        fn lossless_dc_divides_the_dequantiser_back_out() {
            // Lossless is bit-exact because the qindex-0 dequant (times 4) and
            // the WHT's shift = 2 cancel: a DC level of 16 -> dequant 64 -> a
            // flat residual of 4, integrally, with no rounding loss.
            let dequant = dequant_from(&[(0, 64)], 4, 4);
            let res = inverse_transform_2d(&dequant, TxSize::Tx4x4, TxType::DctDct, true, 8);
            for i in 0..4 {
                for j in 0..4 {
                    assert_eq!(res.at(i, j), 4, "lossless DC must be flat and integral");
                }
            }
        }

        #[test]
        fn dequantize_applies_dc_and_ac_steps() {
            // level 3 at DC with dc=10, level 2 at pos 1 with ac=5.
            let quant = [3_i32, 2, 0, 0];
            let dq = dequantize(&quant, TxSize::Tx4x4, 10, 5, 8);
            assert_eq!(dq.at(0, 0), 30);
            assert_eq!(dq.at(0, 1), 10);
        }

        #[test]
        fn quant_lookups_hit_known_entries() {
            assert_eq!(dc_q(8, 0), 4);
            assert_eq!(ac_q(8, 0), 4);
            assert_eq!(dc_q(8, 255), 1336);
            assert_eq!(ac_q(8, 255), 1828);
            assert_eq!(dc_q(10, 0), 4);
            assert_eq!(dc_q(10, 255), 5347);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// Reconstruct a lossless 4x4 residual from raw coefficient levels, the way
    /// the tile driver does: qindex-0 dequant (`dc == ac == 4`) then the WHT.
    fn lossless_residual(quant: &[i32; 16]) -> Residual {
        let dq = dequantize(quant, TxSize::Tx4x4, 4, 4, 8);
        inverse_transform_2d(&dq, TxSize::Tx4x4, TxType::DctDct, true, 8)
    }

    #[test]
    fn all_zero_coefficients_give_a_zero_residual() {
        let residual = lossless_residual(&[0; 16]);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(residual.at(i, j), 0);
            }
        }
    }

    #[test]
    fn a_dc_only_coefficient_spreads_evenly() {
        // A single DC level of 8 spreads to a flat block of 2.
        let mut quant = [0_i32; 16];
        quant[0] = 8;
        let residual = lossless_residual(&quant);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(residual.at(i, j), 2, "DC residual should be flat");
            }
        }
    }

    #[test]
    fn add_residual_clips_to_the_sample_range() {
        let pred = [[250_u16; 4]; 4];
        // A DC of 400 spreads to a flat +100; 250 + 100 clips to 255.
        let mut hi = [0_i32; 16];
        hi[0] = 400;
        let out = add_residual_4x4(&pred, &lossless_residual(&hi), 8);
        assert_eq!(out[0][0], 255);
        // A DC of -1200 spreads to a flat -300; 250 - 300 clips to 0.
        let mut lo = [0_i32; 16];
        lo[0] = -1200;
        let out = add_residual_4x4(&pred, &lossless_residual(&lo), 8);
        assert_eq!(out[1][1], 0);
    }

    #[test]
    fn a_divisible_dc_reconstructs_integrally() {
        let mut quant = [0_i32; 16];
        quant[0] = 16;
        let residual = lossless_residual(&quant);
        assert_eq!(residual.at(0, 0), 4);
    }

    #[test]
    fn flip_ud_places_the_residual_vertically_mirrored() {
        // With a uniform prediction, adding a residual under FLIPADST_DCT
        // (flipUD, no flipLR) must land Residual[i][j] at output row h-1-i,
        // i.e. the vertical mirror of the no-flip result — regardless of the
        // residual's values. Small residual + mid prediction avoids clipping.
        let pred = [128_u16; 16];
        let mut quant = [0_i32; 16];
        quant[1] = 8;
        quant[4] = -8;
        let res = lossless_residual(&quant);
        let no_flip = add_residual(&pred, &res, TxType::DctDct, 8);
        let flipped = add_residual(&pred, &res, TxType::FlipadstDct, 8);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(flipped[(3 - i) * 4 + j], no_flip[i * 4 + j]);
            }
        }
    }

    #[test]
    fn flip_lr_places_the_residual_horizontally_mirrored() {
        let pred = [128_u16; 16];
        let mut quant = [0_i32; 16];
        quant[1] = 8;
        quant[4] = -8;
        let res = lossless_residual(&quant);
        let no_flip = add_residual(&pred, &res, TxType::DctDct, 8);
        // DCT_FLIPADST is flipLR, no flipUD.
        let flipped = add_residual(&pred, &res, TxType::DctFlipadst, 8);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(flipped[i * 4 + (3 - j)], no_flip[i * 4 + j]);
            }
        }
    }
}
