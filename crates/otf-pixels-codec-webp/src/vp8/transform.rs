//! The VP8 inverse transforms (RFC 6386 §14.3, §14.4): the 4x4 DCT
//! approximation and the Walsh-Hadamard transform of the second-order block.

/// `cos(pi/8) * sqrt(2) - 1` in Q16.
const COS_MINUS_ONE: i32 = 20091;
/// `sin(pi/8) * sqrt(2)` in Q16.
const SIN: i32 = 35468;

const fn mul_cos(a: i32) -> i32 {
    ((a * COS_MINUS_ONE) >> 16) + a
}

const fn mul_sin(a: i32) -> i32 {
    (a * SIN) >> 16
}

/// Inverse-transform `coeffs` (row-major) and add the result to the 4x4
/// prediction `block`, clamping each sample.
pub fn idct_add(coeffs: &[i16; 16], block: &mut [[u8; 4]; 4]) {
    let c = coeffs.map(i32::from);
    // Columns first, as the reference decoder does.
    let mut tmp = [0_i32; 16];
    for i in 0..4 {
        let a = c[i] + c[8 + i];
        let b = c[i] - c[8 + i];
        let cc = mul_sin(c[4 + i]) - mul_cos(c[12 + i]);
        let d = mul_cos(c[4 + i]) + mul_sin(c[12 + i]);
        tmp[i] = a + d;
        tmp[4 + i] = b + cc;
        tmp[8 + i] = b - cc;
        tmp[12 + i] = a - d;
    }
    for (row, out) in block.iter_mut().enumerate() {
        let t = &tmp[row * 4..row * 4 + 4];
        let a = t[0] + t[2];
        let b = t[0] - t[2];
        let cc = mul_sin(t[1]) - mul_cos(t[3]);
        let d = mul_cos(t[1]) + mul_sin(t[3]);
        let residual = [a + d, b + cc, b - cc, a - d];
        for (sample, r) in out.iter_mut().zip(residual) {
            *sample = (i32::from(*sample) + ((r + 4) >> 3)).clamp(0, 255) as u8;
        }
    }
}

/// The inverse Walsh-Hadamard transform: the second-order block's 16
/// coefficients become the DC of each luma subblock, in raster order.
pub fn inverse_wht(input: &[i16; 16]) -> [i16; 16] {
    let c = input.map(i32::from);
    let mut tmp = [0_i32; 16];
    for i in 0..4 {
        let a1 = c[i] + c[12 + i];
        let b1 = c[4 + i] + c[8 + i];
        let c1 = c[4 + i] - c[8 + i];
        let d1 = c[i] - c[12 + i];
        tmp[i] = a1 + b1;
        tmp[4 + i] = c1 + d1;
        tmp[8 + i] = a1 - b1;
        tmp[12 + i] = d1 - c1;
    }
    let mut out = [0_i16; 16];
    for row in 0..4 {
        let t = &tmp[row * 4..row * 4 + 4];
        let a1 = t[0] + t[3];
        let b1 = t[1] + t[2];
        let c1 = t[1] - t[2];
        let d1 = t[0] - t[3];
        let values = [a1 + b1, c1 + d1, a1 - b1, d1 - c1];
        for (k, v) in values.into_iter().enumerate() {
            out[row * 4 + k] = ((v + 3) >> 3) as i16;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dc_only_block_adds_its_rounded_eighth_everywhere() {
        let mut coeffs = [0_i16; 16];
        coeffs[0] = 100; // (100 + 4) >> 3 = 13 after both passes
        let mut block = [[50_u8; 4]; 4];
        idct_add(&coeffs, &mut block);
        assert_eq!(block, [[63; 4]; 4]);
    }

    #[test]
    fn sums_clamp_to_the_sample_range() {
        let mut coeffs = [0_i16; 16];
        coeffs[0] = -4000;
        let mut block = [[10_u8; 4]; 4];
        idct_add(&coeffs, &mut block);
        assert_eq!(block, [[0; 4]; 4]);
    }

    #[test]
    fn the_wht_spreads_a_dc_evenly() {
        let mut input = [0_i16; 16];
        input[0] = 80;
        assert_eq!(inverse_wht(&input), [10; 16]);
    }
}
