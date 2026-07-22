//! Inverse transforms (spec §7.13).
//!
//! A transform block is reconstructed by dequantizing the coefficient levels,
//! running a 2D inverse transform, and adding the result to the prediction.
//! This module owns the transform half; the lossless path is implemented first
//! because it is the whole of the first reconstruction target.
//!
//! Lossless AVIF (`CodedLossless`) uses only the 4x4 Walsh–Hadamard transform.
//! The dequantizer multiplies every level by 4 (the qindex-0 step, which is 4
//! at every bit depth), and the row WHT's `shift = 2` divides it straight back
//! out — which is exactly why a lossless decode is bit-exact. The DCT, ADST and
//! identity transforms that lossy coding needs land in a later phase.
//!
//! The workspace forbids slice indexing in production code, so the 4x4 blocks
//! here are handled with array destructuring and iterators rather than `[i]`.

/// `Clip3(low, high, x)` (§4.7).
fn clip3(low: i64, high: i64, x: i64) -> i64 {
    x.clamp(low, high)
}

/// The inverse Walsh–Hadamard transform of a 4-element vector (§7.13.2.10),
/// returned as a new array. `shift` pre-scales the inputs: the row pass uses
/// `shift = 2`, the column pass `shift = 0`.
fn inverse_wht4([x0, x1, x2, x3]: [i64; 4], shift: u32) -> [i64; 4] {
    let mut a = x0 >> shift;
    let mut c = x1 >> shift;
    let mut d = x2 >> shift;
    let mut b = x3 >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

/// Transpose a 4x4 block. Written out so no element is reached by index.
fn transpose(
    [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]]: [[i64; 4]; 4],
) -> [[i64; 4]; 4] {
    [[a, e, i, m], [b, f, j, n], [c, g, k, o], [d, h, l, p]]
}

/// The lossless 4x4 inverse transform: dequantize the 16 coefficient levels and
/// run the 2D WHT, returning the residual as a row-major 4x4 block (§7.12.3,
/// §7.13.3 with `Lossless == 1`).
///
/// `quant` is the coefficient level array in row-major order. The residual is
/// what gets added to the prediction and clipped to the sample range.
#[must_use]
pub fn inverse_wht_4x4(quant: &[i32; 16]) -> [[i32; 4]; 4] {
    // Dequantize: every level times the qindex-0 step of 4. dqDenom is 1 and
    // there is no quantizer matrix in the lossless path.
    let mut block = [[0_i64; 4]; 4];
    for (row, chunk) in block.iter_mut().zip(quant.chunks_exact(4)) {
        for (cell, &level) in row.iter_mut().zip(chunk) {
            *cell = i64::from(level) * 4;
        }
    }

    // Row transforms with shift = 2 (rowShift = 0 leaves the result unchanged).
    for row in &mut block {
        *row = inverse_wht4(*row, 2);
    }

    // Clip between the passes to colClampRange = Max(BitDepth + 6, 16) bits,
    // which is 16 bits for 8/10/12-bit lossless.
    for row in &mut block {
        for value in row.iter_mut() {
            *value = clip3(-(1 << 15), (1 << 15) - 1, *value);
        }
    }

    // Column transforms with shift = 0: transpose, WHT each row, transpose back.
    let mut columns = transpose(block);
    for col in &mut columns {
        *col = inverse_wht4(*col, 0);
    }
    let block = transpose(columns);

    let mut out = [[0_i32; 4]; 4];
    for (orow, brow) in out.iter_mut().zip(block) {
        for (cell, value) in orow.iter_mut().zip(brow) {
            // The lossless residual fits in 1 + BitDepth bits by conformance.
            *cell = value as i32;
        }
    }
    out
}

/// Add a 4x4 residual to a 4x4 prediction and clip to the sample range
/// (`Clip1`, the final step of the reconstruct process, §7.12.3 step 3).
#[must_use]
pub fn add_residual_4x4(
    prediction: &[[u16; 4]; 4],
    residual: &[[i32; 4]; 4],
    bit_depth: u8,
) -> [[u16; 4]; 4] {
    let max = (1_i64 << bit_depth) - 1;
    let mut out = [[0_u16; 4]; 4];
    for ((orow, prow), rrow) in out.iter_mut().zip(prediction).zip(residual) {
        for ((cell, &pred), &res) in orow.iter_mut().zip(prow).zip(rrow) {
            let value = i64::from(pred) + i64::from(res);
            *cell = clip3(0, max, value) as u16;
        }
    }
    out
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

    #[test]
    fn all_zero_coefficients_give_a_zero_residual() {
        let residual = inverse_wht_4x4(&[0; 16]);
        assert_eq!(residual, [[0; 4]; 4]);
    }

    #[test]
    fn a_dc_only_coefficient_spreads_evenly() {
        // A single DC level of 8 spreads to a flat block.
        let mut quant = [0_i32; 16];
        quant[0] = 8;
        let residual = inverse_wht_4x4(&quant);
        let first = residual[0][0];
        for row in &residual {
            for &v in row {
                assert_eq!(v, first, "DC residual should be flat: {residual:?}");
            }
        }
        assert_eq!(first, 2);
    }

    #[test]
    fn add_residual_clips_to_the_sample_range() {
        let pred = [[250_u16; 4]; 4];
        let mut residual = [[0_i32; 4]; 4];
        residual[0][0] = 100; // 350 -> clip to 255
        residual[1][1] = -300; // negative -> clip to 0
        let out = add_residual_4x4(&pred, &residual, 8);
        assert_eq!(out[0][0], 255);
        assert_eq!(out[1][1], 0);
        assert_eq!(out[2][2], 250);
    }

    #[test]
    fn a_divisible_dc_reconstructs_integrally() {
        let mut quant = [0_i32; 16];
        quant[0] = 16;
        let residual = inverse_wht_4x4(&quant);
        assert_eq!(residual, [[4; 4]; 4]);
    }
}
