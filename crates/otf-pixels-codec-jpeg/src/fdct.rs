//! The forward DCT and quantization, in fixed point.
//!
//! The mirror image of [`crate::idct`], sharing its constants, and fixed point
//! for the same reason (ADR-0011): the same pixels must encode to the same
//! bytes on every target, or "encode twice, get the same file" stops being a
//! testable property.
//!
//! # Scaling
//!
//! [`block`] returns coefficients **eight times** a true DCT, which is the
//! convention every integer FDCT in this lineage uses: the two passes each
//! leave a factor of sqrt(8), and removing it would cost a shift per
//! coefficient to no purpose. [`quantize`] divides it out along with the
//! quantization step, so the factor never escapes this module.

use crate::idct::{
    C_0_298, C_0_390, C_0_541, C_0_765, C_0_899, C_1_175, C_1_501, C_1_847, C_1_961, C_2_053,
    C_2_562, C_3_072, SCALE, rounding,
};

/// Extra bits of precision the first pass keeps for the second to spend.
const PASS1: u32 = 2;

/// What the second pass divides the coefficients by, on top of the
/// quantization step: the sqrt(8) each pass leaves behind.
const OVERSCALE: i32 = 8;

/// Divide by `2^bits`, rounding to nearest.
const fn descale(value: i64, bits: u32) -> i64 {
    (value + rounding(bits)) >> bits
}

/// The eight outputs of one 8-point forward DCT, before scaling.
///
/// The two even outputs that need no multiply are kept separate from the six
/// that do, because they carry a different scale and the caller shifts them
/// differently. Folding them together would mean multiplying by one.
struct Parts {
    /// Outputs 0 and 4, at input scale.
    plain: [i64; 2],
    /// Outputs 2, 6, 1, 3, 5, 7, at input scale times `2^SCALE`.
    scaled: [i64; 6],
}

/// One 8-point forward DCT, split even and odd.
///
/// Same decomposition as the inverse, run backwards: the input butterflies
/// separate the even and odd halves, the even half is a 4-point transform,
/// and the odd half's four outputs share three products.
fn parts([s0, s1, s2, s3, s4, s5, s6, s7]: [i64; 8]) -> Parts {
    let t0 = s0 + s7;
    let t7 = s0 - s7;
    let t1 = s1 + s6;
    let t6 = s1 - s6;
    let t2 = s2 + s5;
    let t5 = s2 - s5;
    let t3 = s3 + s4;
    let t4 = s3 - s4;

    // Even part: a 4-point DCT of the sums.
    let t10 = t0 + t3;
    let t13 = t0 - t3;
    let t11 = t1 + t2;
    let t12 = t1 - t2;

    let shared = (t12 + t13) * C_0_541;
    let out2 = shared + (t13 * C_0_765);
    let out6 = shared - (t12 * C_1_847);

    // Odd part: the four cross sums below are each used twice.
    let a = t4 + t7;
    let b = t5 + t6;
    let c = t4 + t6;
    let d = t5 + t7;
    let common = (c + d) * C_1_175;

    let p1 = -(a * C_0_899);
    let p2 = -(b * C_2_562);
    let p3 = common - (c * C_1_961);
    let p4 = common - (d * C_0_390);

    let out7 = (t4 * C_0_298) + p1 + p3;
    let out5 = (t5 * C_2_053) + p2 + p4;
    let out3 = (t6 * C_3_072) + p2 + p3;
    let out1 = (t7 * C_1_501) + p1 + p4;

    Parts {
        plain: [t10 + t11, t10 - t11],
        scaled: [out2, out6, out1, out3, out5, out7],
    }
}

/// Reassemble the eight outputs into transform order.
fn ordered(parts: &Parts, plain: impl Fn(i64) -> i64, scaled: impl Fn(i64) -> i64) -> [i64; 8] {
    let [dc, four] = parts.plain;
    let [two, six, one, three, five, seven] = parts.scaled;
    [
        plain(dc),
        scaled(one),
        scaled(two),
        scaled(three),
        plain(four),
        scaled(five),
        scaled(six),
        scaled(seven),
    ]
}

/// Forward-transform one 8x8 block of samples into DCT coefficients.
///
/// Samples are level-shifted here — JPEG codes them centred on zero, so the
/// 128 an unsigned byte carries is subtracted before the transform rather
/// than being left for the DC coefficient to express.
///
/// The result is eight times a true DCT; [`quantize`] removes that.
pub fn block(samples: &[u8; 64], out: &mut [i64; 64]) {
    let mut rows = [0_i64; 64];

    for row in 0..8 {
        let input: [i64; 8] = std::array::from_fn(|column| {
            i64::from(*samples.get(row * 8 + column).unwrap_or(&128)) - 128
        });
        let computed = parts(input);
        let values = ordered(
            &computed,
            |value| value << PASS1,
            |value| descale(value, SCALE - PASS1),
        );
        for (column, value) in values.into_iter().enumerate() {
            if let Some(slot) = rows.get_mut(row * 8 + column) {
                *slot = value;
            }
        }
    }

    for column in 0..8 {
        let input: [i64; 8] = std::array::from_fn(|row| *rows.get(row * 8 + column).unwrap_or(&0));
        let computed = parts(input);
        let values = ordered(
            &computed,
            |value| descale(value, PASS1),
            |value| descale(value, SCALE + PASS1),
        );
        for (row, value) in values.into_iter().enumerate() {
            if let Some(slot) = out.get_mut(row * 8 + column) {
                *slot = value;
            }
        }
    }
}

/// Divide each coefficient by its quantization step, rounding to nearest.
///
/// This is the only lossy step in the whole encoder. Rounding to nearest —
/// rather than truncating, which is cheaper and what a naive implementation
/// does — is worth roughly half a step of error per coefficient, and the
/// error is what quality *means* here.
pub fn quantize(coefficients: &[i64; 64], steps: &[u16; 64], out: &mut [i32; 64]) {
    for (index, slot) in out.iter_mut().enumerate() {
        let coefficient = *coefficients.get(index).unwrap_or(&0);
        let step = i64::from(*steps.get(index).unwrap_or(&1)).max(1) * i64::from(OVERSCALE);
        let half = step / 2;
        // Rounding away from zero on both sides: `(-3) / 2` truncating
        // towards zero would bias every negative coefficient upwards.
        let quantized = if coefficient >= 0 {
            (coefficient + half) / step
        } else {
            -((-coefficient + half) / step)
        };
        *slot = quantized.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i32;
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

    /// The textbook floating-point forward DCT, to check the fixed-point one
    /// against. Returns true DCT coefficients, not the 8x-scaled ones.
    fn reference(samples: &[u8; 64]) -> [f64; 64] {
        let mut out = [0.0; 64];
        for v in 0..8 {
            for u in 0..8 {
                let cu = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                let cv = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                let mut sum = 0.0;
                for y in 0..8 {
                    for x in 0..8 {
                        sum += (f64::from(samples[y * 8 + x]) - 128.0)
                            * (((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI) / 16.0).cos()
                            * (((2 * y + 1) as f64 * v as f64 * std::f64::consts::PI) / 16.0).cos();
                    }
                }
                out[v * 8 + u] = cu * cv * sum / 4.0;
            }
        }
        out
    }

    fn forward(samples: &[u8; 64]) -> [i64; 64] {
        let mut out = [0_i64; 64];
        block(samples, &mut out);
        out
    }

    #[test]
    fn a_flat_block_has_only_a_dc_coefficient() {
        let coefficients = forward(&[144; 64]);
        // 144 is 16 above the mid-point, and the DC of a flat block is
        // 8 x level; the 8x output scale makes that 8 x 8 x 16.
        assert_eq!(coefficients[0], 8 * 8 * 16);
        assert!(
            coefficients[1..].iter().all(|&c| c == 0),
            "a flat block has no detail to code"
        );

        // The mid-grey a level shift centres on encodes to nothing at all.
        assert!(forward(&[128; 64]).iter().all(|&c| c == 0));
    }

    #[test]
    fn output_matches_the_reference_dct() {
        let mut cases: Vec<[u8; 64]> = vec![[0; 64], [255; 64]];

        // A horizontal ramp, a vertical ramp, and a diagonal one.
        cases.push(std::array::from_fn(|i| ((i % 8) * 36) as u8));
        cases.push(std::array::from_fn(|i| ((i / 8) * 36) as u8));
        cases.push(std::array::from_fn(|i| ((i % 8 + i / 8) * 18) as u8));
        // A hard edge, which puts energy in every frequency.
        cases.push(std::array::from_fn(|i| if i % 8 < 4 { 0 } else { 255 }));
        // Alternating pixels: the highest frequency the block can hold.
        cases.push(std::array::from_fn(|i| if i % 2 == 0 { 0 } else { 255 }));

        let mut state = 987_654_321_u32;
        let mut noisy = [0_u8; 64];
        for slot in &mut noisy {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *slot = (state >> 16) as u8;
        }
        cases.push(noisy);

        for samples in cases {
            let ours = forward(&samples);
            let theirs = reference(&samples);
            for (index, (&got, &want)) in ours.iter().zip(theirs.iter()).enumerate() {
                // Ours is 8x a true DCT by construction.
                let got = got as f64 / 8.0;
                assert!(
                    (got - want).abs() <= 1.0,
                    "coefficient {index}: got {got:.3}, reference {want:.3}"
                );
            }
        }
    }

    #[test]
    fn a_block_survives_the_round_trip_through_quantization() {
        // At quality 100 every step is 1, so the round trip is limited by the
        // transforms rather than by the quantizer — which is the tightest
        // this codec can ever be, and so the strongest check that the forward
        // and inverse transforms are actually inverses.
        let steps = crate::tables::scale_quant(&crate::tables::LUMA_QUANT, 100);
        let mut state = 24_680_u32;
        let samples: [u8; 64] = std::array::from_fn(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            // A mid-range block: extremes clamp on the way back and would
            // hide error rather than reveal it.
            (64 + (state >> 16) % 128) as u8
        });

        let mut coefficients = [0_i64; 64];
        block(&samples, &mut coefficients);
        let mut quantized = [0_i32; 64];
        quantize(&coefficients, &steps, &mut quantized);

        // Dequantize and invert, the way a decoder would.
        let dequantized: [i32; 64] = std::array::from_fn(|i| quantized[i] * i32::from(steps[i]));
        let mut back = [0_u8; 64];
        crate::idct::block(&dequantized, &mut back, 0, 8);

        for (index, (&got, &want)) in back.iter().zip(samples.iter()).enumerate() {
            assert!(
                got.abs_diff(want) <= 2,
                "sample {index}: {got} came back from {want}"
            );
        }
    }

    #[test]
    fn quantization_rounds_to_nearest_and_is_symmetric() {
        let steps = [10_u16; 64];
        // The 8x output scale means a coefficient of 8*n quantizes as n/step.
        let coefficients: [i64; 64] = std::array::from_fn(|i| match i {
            0 => 8 * 14,  // 1.4 steps -> 1
            1 => 8 * 15,  // 1.5 steps -> 2, away from zero
            2 => -8 * 14, // -1.4 -> -1
            3 => -8 * 15, // -1.5 -> -2, symmetric with the positive case
            4 => 8 * 4,   // 0.4 -> 0
            _ => 0,
        });
        let mut out = [0_i32; 64];
        quantize(&coefficients, &steps, &mut out);
        assert_eq!(&out[..5], &[1, 2, -1, -2, 0]);
    }

    #[test]
    fn quantization_cannot_divide_by_zero() {
        // A zero step would be a division by zero; the table scaler never
        // produces one, but this is the place where it would be fatal.
        let mut out = [0_i32; 64];
        quantize(&[8 * 100; 64], &[0; 64], &mut out);
        assert!(out.iter().all(|&value| value == 100));
    }
}
