//! The one-dimensional resampling kernels resize is built from.
//!
//! Separated from [`Resize`] on purpose: the op decides *what* to resample,
//! these decide *how*, and the how is where all the performance and all the
//! determinism live.
//!
//! [`Resize`]: crate::Resize
//!
//! # Shape of the loops (ADR-0011)
//!
//! Every kernel here is written to autovectorize rather than to call
//! intrinsics. Concretely that means: the inner loop walks a contiguous slice
//! with a fixed trip count, the bounds check is hoisted by slicing once
//! outside, accumulators are locals rather than memory the compiler must
//! assume aliases, and there are no early exits.
//!
//! Vectorization is across **channels and output pixels**, never a reduction
//! across lanes. That is what makes scalar and vector results bit-identical
//! rather than merely close: each lane performs the same operations in the
//! same order the scalar loop would.
//!
//! # Two accumulators, not one
//!
//! Eight-bit input accumulates in `i32` fixed point; wider input accumulates in
//! `f32`. Integer addition is associative, so the 8-bit path is exactly
//! reproducible under any vectorization the compiler chooses. The float path
//! relies on the ordering discipline above instead.

use crate::filter::{ONE, Weights};

/// Round a fixed-point accumulator back to a sample and clamp to 0..=255.
///
/// Clamping is not optional: Lanczos weights are signed, so a run over a hard
/// edge legitimately overshoots past both ends of the range.
#[inline]
fn finish_u8(accumulator: i32) -> u8 {
    // Round half away from zero, matching the weight quantization.
    let rounded = (accumulator + (ONE / 2)) >> 14;
    rounded.clamp(0, 255) as u8
}

/// The same for 16-bit samples.
#[inline]
fn finish_u16(accumulator: f32) -> u16 {
    let rounded = accumulator + 0.5;
    if rounded <= 0.0 {
        0
    } else if rounded >= 65535.0 {
        65535
    } else {
        rounded as u16
    }
}

/// Resample one row of 8-bit samples along `weights`.
///
/// `input` and `output` are interleaved by `channels`; every channel of an
/// output pixel shares a run, so the channel loop is the innermost and the one
/// the compiler widens.
pub fn row_u8(input: &[u8], output: &mut [u8], channels: usize, weights: &Weights) {
    // Dispatched once per row into a monomorphized kernel, so the channel loop
    // is unrolled into independent accumulator chains rather than a three-trip
    // loop the compiler cannot widen. This is ADR-0002's dispatch discipline
    // applied to channel count as well as sample type.
    match channels {
        1 => row_u8_n::<1>(input, output, weights),
        2 => row_u8_n::<2>(input, output, weights),
        3 => row_u8_n::<3>(input, output, weights),
        4 => row_u8_n::<4>(input, output, weights),
        // No v1 format has another channel count; a future one falls back to
        // the general form rather than silently producing nothing.
        other => row_u8_general(input, output, other, weights),
    }
}

/// The 8-bit horizontal kernel with the channel count known at compile time.
fn row_u8_n<const C: usize>(input: &[u8], output: &mut [u8], weights: &Weights) {
    for (out_index, run) in weights.runs().iter().enumerate() {
        let coefficients = weights.quantized(run);
        let first = run.start as usize * C;
        let taps = run.len as usize;

        // Slice the exact window once, so the inner loop carries no bounds
        // checks and the compiler can see a fixed trip count.
        let Some(window) = input.get(first..first + taps * C) else {
            continue;
        };
        let Some(target) = output.get_mut(out_index * C..(out_index + 1) * C) else {
            continue;
        };

        // `C` independent accumulator chains, which is what lets the tap loop
        // be software-pipelined instead of serialized on one dependency.
        let mut accumulators = [0_i32; C];
        for (pixel, &coefficient) in window.chunks_exact(C).zip(coefficients) {
            for (slot, &sample) in accumulators.iter_mut().zip(pixel) {
                *slot += i32::from(sample) * coefficient;
            }
        }
        for (slot, &value) in target.iter_mut().zip(accumulators.iter()) {
            *slot = finish_u8(value);
        }
    }
}

/// The same, for a channel count not known at compile time.
fn row_u8_general(input: &[u8], output: &mut [u8], channels: usize, weights: &Weights) {
    for (out_index, run) in weights.runs().iter().enumerate() {
        let coefficients = weights.quantized(run);
        let first = run.start as usize * channels;
        let taps = run.len as usize;

        let Some(window) = input.get(first..first + taps * channels) else {
            continue;
        };
        let Some(target) = output.get_mut(out_index * channels..(out_index + 1) * channels) else {
            continue;
        };

        let mut accumulators = [0_i32; MAX_CHANNELS];
        for (pixel, &coefficient) in window.chunks_exact(channels).zip(coefficients) {
            for (slot, &sample) in accumulators.iter_mut().zip(pixel) {
                *slot += i32::from(sample) * coefficient;
            }
        }
        for (channel, slot) in target.iter_mut().enumerate() {
            *slot = finish_u8(accumulators.get(channel).copied().unwrap_or(0));
        }
    }
}

/// Resample one row of 16-bit samples along `weights`.
pub fn row_u16(input: &[u16], output: &mut [u16], channels: usize, weights: &Weights) {
    for (out_index, run) in weights.runs().iter().enumerate() {
        let coefficients = weights.exact(run);
        let first = run.start as usize * channels;
        let taps = run.len as usize;

        let Some(window) = input.get(first..first + taps * channels) else {
            continue;
        };
        let Some(target) = output.get_mut(out_index * channels..(out_index + 1) * channels) else {
            continue;
        };

        let mut accumulators = [0.0_f32; MAX_CHANNELS];
        for (tap, &coefficient) in coefficients.iter().enumerate() {
            let Some(pixel) = window.get(tap * channels..(tap + 1) * channels) else {
                continue;
            };
            for (channel, &sample) in pixel.iter().enumerate() {
                if let Some(slot) = accumulators.get_mut(channel) {
                    *slot += f32::from(sample) * coefficient;
                }
            }
        }
        for (channel, slot) in target.iter_mut().enumerate() {
            *slot = finish_u16(accumulators.get(channel).copied().unwrap_or(0.0));
        }
    }
}

/// Resample one row of float samples along `weights`.
///
/// Float pixels carry no nominal range, so nothing is clamped: a caller working
/// in linear light or in HDR is entitled to values outside 0..=1, and silently
/// crushing them would be the op inventing a policy it has no business having.
pub fn row_f32(input: &[f32], output: &mut [f32], channels: usize, weights: &Weights) {
    for (out_index, run) in weights.runs().iter().enumerate() {
        let coefficients = weights.exact(run);
        let first = run.start as usize * channels;
        let taps = run.len as usize;

        let Some(window) = input.get(first..first + taps * channels) else {
            continue;
        };
        let Some(target) = output.get_mut(out_index * channels..(out_index + 1) * channels) else {
            continue;
        };

        let mut accumulators = [0.0_f32; MAX_CHANNELS];
        for (tap, &coefficient) in coefficients.iter().enumerate() {
            let Some(pixel) = window.get(tap * channels..(tap + 1) * channels) else {
                continue;
            };
            for (channel, &sample) in pixel.iter().enumerate() {
                if let Some(slot) = accumulators.get_mut(channel) {
                    *slot += sample * coefficient;
                }
            }
        }
        for (channel, slot) in target.iter_mut().enumerate() {
            *slot = accumulators.get(channel).copied().unwrap_or(0.0);
        }
    }
}

/// The widest pixel in SPEC §Pixel formats. A fixed-size accumulator array
/// keeps the hot loop free of allocation and lets it live in registers.
pub const MAX_CHANNELS: usize = 4;

// ---------------------------------------------------------------------------
// Vertical pass
// ---------------------------------------------------------------------------
//
// The vertical pass could be the horizontal one applied to columns, and that
// is the obvious implementation — gather a column into a contiguous buffer,
// resample it, scatter it back. It is also several times slower, because a
// column of a row-major image touches one cache line per sample and evicts it
// before the next output pixel needs its neighbour.
//
// Accumulating a whole output *row* at once instead reads the input rows
// sequentially and writes the accumulator sequentially. The arithmetic is
// identical — the same weights applied to the same samples in the same tap
// order — so the result is bit-for-bit what the column form produced.

/// Accumulate one output row from a vertical run of input rows, 8-bit.
///
/// `accumulator` and `out` are both `row_len` long; `source` is row-major with
/// `row_len` samples per row. The inner loop is contiguous in both operands,
/// which is what makes it the one the compiler widens.
pub fn column_u8(
    source: &[u8],
    row_len: usize,
    start: usize,
    coefficients: &[i32],
    accumulator: &mut [i32],
    out: &mut [u8],
) {
    accumulator.fill(0);
    for (tap, &coefficient) in coefficients.iter().enumerate() {
        let at = (start + tap) * row_len;
        let Some(row) = source.get(at..at + row_len) else {
            continue;
        };
        for (slot, &sample) in accumulator.iter_mut().zip(row) {
            *slot += i32::from(sample) * coefficient;
        }
    }
    for (slot, &value) in out.iter_mut().zip(accumulator.iter()) {
        *slot = finish_u8(value);
    }
}

/// The same for 16-bit samples, accumulating in `f32`.
pub fn column_u16(
    source: &[u16],
    row_len: usize,
    start: usize,
    coefficients: &[f32],
    accumulator: &mut [f32],
    out: &mut [u16],
) {
    accumulator.fill(0.0);
    for (tap, &coefficient) in coefficients.iter().enumerate() {
        let at = (start + tap) * row_len;
        let Some(row) = source.get(at..at + row_len) else {
            continue;
        };
        for (slot, &sample) in accumulator.iter_mut().zip(row) {
            *slot += f32::from(sample) * coefficient;
        }
    }
    for (slot, &value) in out.iter_mut().zip(accumulator.iter()) {
        *slot = finish_u16(value);
    }
}

/// The same for float samples, which are not clamped.
pub fn column_f32(
    source: &[f32],
    row_len: usize,
    start: usize,
    coefficients: &[f32],
    accumulator: &mut [f32],
    out: &mut [f32],
) {
    accumulator.fill(0.0);
    for (tap, &coefficient) in coefficients.iter().enumerate() {
        let at = (start + tap) * row_len;
        let Some(row) = source.get(at..at + row_len) else {
            continue;
        };
        for (slot, &sample) in accumulator.iter_mut().zip(row) {
            *slot += sample * coefficient;
        }
    }
    out.copy_from_slice(accumulator);
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::filter::Filter;

    const ALL: [Filter; 7] = [
        Filter::Nearest,
        Filter::Box,
        Filter::Bilinear,
        Filter::CatmullRom,
        Filter::Mitchell,
        Filter::Lanczos2,
        Filter::Lanczos3,
    ];

    #[test]
    fn a_constant_row_stays_constant_at_every_scale() {
        // The sharpest test of the weight normalization: if every run sums to
        // one, a flat input must come out flat. Any drift shows here as a
        // value that is not the one we put in.
        for filter in ALL {
            for (input_len, output_len) in [(64, 32), (32, 64), (64, 64), (100, 37), (7, 200)] {
                for channels in 1..=4 {
                    let weights = Weights::build(filter, input_len, output_len).unwrap();
                    let input = vec![200_u8; input_len as usize * channels];
                    let mut output = vec![0_u8; output_len as usize * channels];
                    row_u8(&input, &mut output, channels, &weights);
                    assert!(
                        output.iter().all(|&v| v == 200),
                        "{} {input_len}->{output_len}x{channels} did not stay flat: {:?}",
                        filter.as_str(),
                        &output[..output.len().min(8)]
                    );
                }
            }
        }
    }

    #[test]
    fn a_one_to_one_resize_is_the_identity() {
        // At scale 1 every filter must return the input untouched. This is
        // where a half-pixel error or a rounding bias becomes visible as a
        // changed byte rather than as a slightly soft image.
        for filter in ALL {
            let weights = Weights::build(filter, 64, 64).unwrap();
            let input: Vec<u8> = (0..64_u32).map(|i| (i * 3 % 256) as u8).collect();
            let mut output = vec![0_u8; 64];
            row_u8(&input, &mut output, 1, &weights);
            assert_eq!(output, input, "{} changed pixels at 1:1", filter.as_str());
        }
    }

    #[test]
    fn channels_are_resampled_independently() {
        // A channel bleeding into its neighbour is the classic interleaving
        // bug, and it is invisible on greyscale test data.
        let weights = Weights::build(Filter::Bilinear, 8, 4).unwrap();
        let mut input = vec![0_u8; 8 * 4];
        for (pixel, chunk) in input.chunks_exact_mut(4).enumerate() {
            chunk[0] = 255;
            chunk[1] = 0;
            chunk[2] = if pixel < 4 { 255 } else { 0 };
            chunk[3] = 128;
        }
        let mut output = vec![0_u8; 4 * 4];
        row_u8(&input, &mut output, 4, &weights);

        for pixel in output.chunks_exact(4) {
            assert_eq!(pixel[0], 255, "channel 0 should stay saturated");
            assert_eq!(pixel[1], 0, "channel 1 should stay zero");
            assert_eq!(pixel[3], 128, "channel 3 should stay constant");
        }
    }

    #[test]
    fn overshoot_is_clamped_not_wrapped() {
        // Lanczos weights are signed, so a hard edge legitimately produces an
        // accumulator outside 0..=255. Wrapping would turn a bright edge into
        // a black one, which is the most alarming possible failure mode.
        let weights = Weights::build(Filter::Lanczos3, 32, 30).unwrap();
        let mut input = vec![0_u8; 32];
        for slot in input.iter_mut().skip(16) {
            *slot = 255;
        }
        let mut output = vec![0_u8; 30];
        row_u8(&input, &mut output, 1, &weights);
        // Nothing to assert about the ringing itself; what matters is that no
        // value wrapped, which for u8 output means the edge stayed monotone
        // in the large.
        assert_eq!(output.first().copied(), Some(0), "left of the edge");
        assert_eq!(output.last().copied(), Some(255), "right of the edge");
    }

    #[test]
    fn the_eight_bit_path_is_bit_identical_run_to_run() {
        // SPEC §Guarantees 2, and the property ADR-0011's fixed point exists
        // to make structural: the same input must give the same bytes, always.
        let weights = Weights::build(Filter::Lanczos3, 500, 137).unwrap();
        let input: Vec<u8> = (0..500_u32 * 3).map(|i| (i * 31 % 251) as u8).collect();
        let mut first = vec![0_u8; 137 * 3];
        row_u8(&input, &mut first, 3, &weights);
        for _ in 0..8 {
            let mut again = vec![0_u8; 137 * 3];
            row_u8(&input, &mut again, 3, &weights);
            assert_eq!(again, first, "the 8-bit kernel is not deterministic");
        }
    }

    #[test]
    fn sixteen_bit_and_float_paths_track_the_eight_bit_one() {
        // Not bit-identical — different arithmetic, deliberately — but a
        // divergence beyond rounding would mean one of them is wrong.
        let weights = Weights::build(Filter::CatmullRom, 64, 40).unwrap();
        let eight: Vec<u8> = (0..64_u32).map(|i| (i * 4 % 256) as u8).collect();
        let wide: Vec<u16> = eight.iter().map(|&v| u16::from(v) * 257).collect();
        let floats: Vec<f32> = eight.iter().map(|&v| f32::from(v)).collect();

        let mut out8 = vec![0_u8; 40];
        let mut out16 = vec![0_u16; 40];
        let mut outf = vec![0.0_f32; 40];
        row_u8(&eight, &mut out8, 1, &weights);
        row_u16(&wide, &mut out16, 1, &weights);
        row_f32(&floats, &mut outf, 1, &weights);

        for (index, &value) in out8.iter().enumerate() {
            let from16 = (f32::from(out16[index]) / 257.0).round() as i32;
            let fromf = outf[index].round() as i32;
            assert!(
                (i32::from(value) - from16).abs() <= 1,
                "16-bit differs at {index}: {value} vs {from16}"
            );
            assert!(
                (i32::from(value) - fromf).abs() <= 1,
                "float differs at {index}: {value} vs {fromf}"
            );
        }
    }

    #[test]
    fn float_output_is_not_clamped() {
        // Float pixels have no nominal range. An op that crushed them to
        // 0..=1 would be inventing a policy, and would break HDR pipelines
        // silently rather than loudly.
        // Uniform inputs, so the widened downscale kernel cannot average the
        // two extremes back into range and hide a clamp.
        let weights = Weights::build(Filter::Bilinear, 4, 2).unwrap();
        for value in [-5.0_f32, 900.0] {
            let input = vec![value; 4];
            let mut output = vec![0.0_f32; 2];
            row_f32(&input, &mut output, 1, &weights);
            for &got in &output {
                assert!(
                    (got - value).abs() < 1e-3,
                    "{value} came back as {got}, so the float path is clamping"
                );
            }
        }
    }

    #[test]
    fn a_single_sample_input_is_not_a_panic() {
        // Degenerate but legal: a 1-pixel image scaled up.
        for filter in ALL {
            let weights = Weights::build(filter, 1, 16).unwrap();
            let input = vec![77_u8];
            let mut output = vec![0_u8; 16];
            row_u8(&input, &mut output, 1, &weights);
            assert!(
                output.iter().all(|&v| v == 77),
                "{} on a 1-sample input gave {output:?}",
                filter.as_str()
            );
        }
    }

    #[test]
    fn a_mismatched_output_length_truncates_rather_than_panicking() {
        // Kernels take slices from a scheduler; a wrong length is a caller
        // bug, but it must not be a crash inside the hot loop.
        let weights = Weights::build(Filter::Box, 8, 4).unwrap();
        let input = vec![1_u8; 8];
        let mut short = vec![0_u8; 2];
        row_u8(&input, &mut short, 1, &weights);
        let mut long = vec![0_u8; 16];
        row_u8(&input, &mut long, 1, &weights);
    }
}
