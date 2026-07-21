//! The inverse DCT, in fixed point.
//!
//! # Why fixed point
//!
//! ADR-0011: integer arithmetic gives the same answer on every machine, which
//! is what lets the engine promise byte-identical output run to run and across
//! thread counts. A float IDCT would be a per-target result, and "the same
//! image twice" would stop being a testable property.
//!
//! The transform is the usual even/odd decomposition of the 8-point IDCT
//! (Loeffler-Ligtenberg-Moerkotte), applied to columns then rows. Constants
//! are scaled by 2^12; the column pass keeps two extra bits of precision for
//! the row pass to spend, so the passes shift right by 10 and 17. The two
//! constant scales and the two shifts leave a net 2^-3, which is exactly the
//! 1/8 a normalized 2-D IDCT ends on.
//!
//! Accuracy is within the ±1 tolerance the JPEG conformance spec allows, which
//! is why decoded output is compared against reference decoders with a
//! tolerance rather than for equality — no two IDCTs agree exactly, and the
//! format does not require them to.

/// Fixed-point scale for the transform constants.
///
/// Shared with the forward transform in [`crate::fdct`]: the two directions
/// are built from the same eight constants, and letting them drift apart
/// would be a silent accuracy bug in whichever one was not being tested.
pub(crate) const SCALE: u32 = 12;

/// Right shift applied after the column pass, keeping two bits of headroom.
const COLUMN_SHIFT: u32 = SCALE - 2;

/// Right shift applied after the row pass, spending that headroom.
const ROW_SHIFT: u32 = SCALE + 5;

/// The rounding term for a right shift of `bits`.
pub(crate) const fn rounding(bits: u32) -> i64 {
    1 << (bits - 1)
}

/// Round a float constant into [`SCALE`]-bit fixed point.
const fn fixed(value: f64) -> i64 {
    (value * (1_i64 << SCALE) as f64 + 0.5) as i64
}

// The eight distinct constants the decomposition needs.
pub(crate) const C_0_298: i64 = fixed(0.298_631_336);
pub(crate) const C_0_390: i64 = fixed(0.390_180_644);
pub(crate) const C_0_541: i64 = fixed(0.541_196_100);
pub(crate) const C_0_765: i64 = fixed(0.765_366_865);
pub(crate) const C_0_899: i64 = fixed(0.899_976_223);
pub(crate) const C_1_175: i64 = fixed(1.175_875_602);
pub(crate) const C_1_501: i64 = fixed(1.501_321_110);
pub(crate) const C_1_847: i64 = fixed(1.847_759_065);
pub(crate) const C_1_961: i64 = fixed(1.961_570_560);
pub(crate) const C_2_053: i64 = fixed(2.053_119_869);
pub(crate) const C_2_562: i64 = fixed(2.562_915_447);
pub(crate) const C_3_072: i64 = fixed(3.072_711_026);

/// One 8-point inverse DCT, returning values scaled by 2^`SCALE`.
///
/// Splitting even and odd inputs is what makes this eleven multiplies instead
/// of sixty-four: the even half is a 4-point transform, and the odd half's
/// four outputs share three of their products.
fn transform([s0, s1, s2, s3, s4, s5, s6, s7]: [i64; 8]) -> [i64; 8] {
    // Even part: inputs 0, 2, 4, 6.
    let shared = (s2 + s6) * C_0_541;
    let even2 = shared - (s6 * C_1_847);
    let even3 = shared + (s2 * C_0_765);
    let sum = (s0 + s4) << SCALE;
    let difference = (s0 - s4) << SCALE;

    let x0 = sum + even3;
    let x3 = sum - even3;
    let x1 = difference + even2;
    let x2 = difference - even2;

    // Odd part: inputs 1, 3, 5, 7. The four cross sums below are each used
    // twice, which is the saving the decomposition exists for.
    let a = s7 + s3;
    let b = s5 + s1;
    let c = s7 + s1;
    let d = s5 + s3;
    let common = (c + d) * C_1_175;

    let p1 = common - (c * C_0_899);
    let p2 = common - (d * C_2_562);
    let p3 = -(a * C_1_961);
    let p4 = -(b * C_0_390);

    let y0 = (s7 * C_0_298) + p1 + p3;
    let y1 = (s5 * C_2_053) + p2 + p4;
    let y2 = (s3 * C_3_072) + p2 + p3;
    let y3 = (s1 * C_1_501) + p1 + p4;

    [
        x0 + y3,
        x1 + y2,
        x2 + y1,
        x3 + y0,
        x3 - y0,
        x2 - y1,
        x1 - y2,
        x0 - y3,
    ]
}

/// Inverse-transform one dequantized block into 8 rows of `out`, level-shifted
/// and clamped to `u8`.
///
/// `out` is written at `offset` with `stride` bytes between rows, so blocks
/// land directly in a component plane with no intermediate copy. Rows that
/// fall outside `out` are dropped rather than being an error: the bottom and
/// right edges of an image are padded to whole blocks by the format, and that
/// padding has nowhere to go.
pub fn block(coefficients: &[i32; 64], out: &mut [u8], offset: usize, stride: usize) {
    let mut columns = [0_i64; 64];

    for column in 0..8 {
        let input: [i64; 8] =
            std::array::from_fn(|row| i64::from(*coefficients.get(column + row * 8).unwrap_or(&0)));

        // An all-zero AC column — the common case in flat areas — is a
        // constant, so the whole transform collapses to one broadcast.
        let [dc, rest @ ..] = input;
        if rest.iter().all(|&value| value == 0) {
            let flat = ((dc << SCALE) + rounding(COLUMN_SHIFT)) >> COLUMN_SHIFT;
            for row in 0..8 {
                if let Some(slot) = columns.get_mut(column + row * 8) {
                    *slot = flat;
                }
            }
            continue;
        }

        let output = transform(input);
        for (row, value) in output.into_iter().enumerate() {
            if let Some(slot) = columns.get_mut(column + row * 8) {
                *slot = (value + rounding(COLUMN_SHIFT)) >> COLUMN_SHIFT;
            }
        }
    }

    for row in 0..8 {
        let input: [i64; 8] =
            std::array::from_fn(|column| *columns.get(column + row * 8).unwrap_or(&0));
        let output = transform(input);

        let Some(target) = out.get_mut(offset + row * stride..) else {
            continue;
        };
        for (slot, value) in target.iter_mut().take(8).zip(output) {
            // Undo the column pass's headroom and both constant scales, then
            // re-centre: JPEG codes samples as signed values around zero, so
            // the 128 the encoder subtracted comes back here.
            let shifted = (value + rounding(ROW_SHIFT)) >> ROW_SHIFT;
            *slot = (shifted + 128).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// The textbook floating-point IDCT, to check the fixed-point one against.
    fn reference(coefficients: &[i32; 64]) -> [f64; 64] {
        let mut out = [0.0; 64];
        for y in 0..8 {
            for x in 0..8 {
                let mut sum = 0.0;
                for v in 0..8 {
                    for u in 0..8 {
                        let cu = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                        let cv = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                        sum += cu
                            * cv
                            * f64::from(coefficients[v * 8 + u])
                            * (((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI) / 16.0).cos()
                            * (((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI) / 16.0).cos();
                    }
                }
                out[y * 8 + x] = sum / 4.0 + 128.0;
            }
        }
        out
    }

    fn decode(coefficients: &[i32; 64]) -> Vec<u8> {
        let mut out = vec![0_u8; 64];
        block(coefficients, &mut out, 0, 8);
        out
    }

    #[test]
    fn a_dc_only_block_is_a_flat_grey() {
        // DC of 8*16 is a level of 16 above the 128 mid-point.
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 8 * 16;
        assert!(decode(&coefficients).iter().all(|&v| v == 144));

        // A zero block is the mid-grey the level shift centres on.
        assert!(decode(&[0; 64]).iter().all(|&v| v == 128));
    }

    #[test]
    fn output_matches_the_reference_idct_within_one_step() {
        // A spread of blocks: single frequencies, a ramp, and a noisy one.
        let mut cases: Vec<[i32; 64]> = Vec::new();
        for index in [0, 1, 8, 9, 27, 63] {
            let mut block = [0_i32; 64];
            block[index] = 200;
            cases.push(block);
            let mut negative = [0_i32; 64];
            negative[index] = -300;
            cases.push(negative);
        }
        let mut ramp = [0_i32; 64];
        for (index, slot) in ramp.iter_mut().enumerate() {
            *slot = (index as i32 % 7) * 20 - 60;
        }
        ramp[0] = 400;
        cases.push(ramp);

        let mut noisy = [0_i32; 64];
        let mut state = 12_345_u32;
        for slot in &mut noisy {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *slot = ((state >> 16) as i32 % 512) - 256;
        }
        cases.push(noisy);

        for coefficients in cases {
            let ours = decode(&coefficients);
            let theirs = reference(&coefficients);
            for (index, (&got, &want)) in ours.iter().zip(theirs.iter()).enumerate() {
                let want = want.clamp(0.0, 255.0);
                assert!(
                    (f64::from(got) - want).abs() <= 1.0,
                    "sample {index}: got {got}, reference {want:.3}"
                );
            }
        }
    }

    #[test]
    fn the_flat_column_shortcut_agrees_with_the_full_transform() {
        // Only the DC and one AC row are set, so most columns take the
        // shortcut and one does not. Both must land on the same answer as the
        // reference transform.
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 300;
        coefficients[3] = -120;
        let ours = decode(&coefficients);
        let theirs = reference(&coefficients);
        for (&got, &want) in ours.iter().zip(theirs.iter()) {
            assert!((f64::from(got) - want.clamp(0.0, 255.0)).abs() <= 1.0);
        }
    }

    #[test]
    fn extreme_coefficients_clamp_instead_of_wrapping() {
        // A maximally negative and a maximally positive DC, which without
        // clamping would wrap around the byte range.
        let mut coefficients = [0_i32; 64];
        coefficients[0] = -32_768;
        assert!(decode(&coefficients).iter().all(|&v| v == 0));
        coefficients[0] = 32_767;
        assert!(decode(&coefficients).iter().all(|&v| v == 255));
    }

    #[test]
    fn blocks_write_at_a_stride_and_clip_at_the_end_of_the_buffer() {
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 8 * 16;

        // A 16-wide plane, block written to the right half.
        let mut plane = vec![0_u8; 16 * 8];
        block(&coefficients, &mut plane, 8, 16);
        for row in 0..8 {
            assert_eq!(&plane[row * 16..row * 16 + 8], &[0_u8; 8]);
            assert_eq!(&plane[row * 16 + 8..row * 16 + 16], &[144_u8; 8]);
        }

        // A plane with only three rows: the rest of the block is dropped.
        let mut short = vec![0_u8; 8 * 3];
        block(&coefficients, &mut short, 0, 8);
        assert!(short.iter().all(|&v| v == 144));
    }
}
