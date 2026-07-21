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

/// How much of an image a decode produces, as eighths of full size.
///
/// # What it buys
///
/// A downsampled block can be produced directly from its coefficients, so a
/// thumbnail never materializes the full-resolution image for the rest of the
/// pipeline to carry: at 1/8 the resize, colour conversion and every later op
/// see one sixty-fourth of the pixels. The reduced transform is also cheaper
/// than the full one at 1/8 and 1/4, though at 1/2 it is not — it folds all
/// sixty-four coefficients into sixteen outputs. The saving that matters is
/// downstream, not in the transform.
///
/// This is a *resolution* reduction, not random access: rows still arrive top
/// to bottom, and every coefficient is still entropy-decoded. A scaled decode
/// is therefore a decoder configuration rather than a
/// [`DecodeCapability`](otf_pixels_core::DecodeCapability).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[non_exhaustive]
pub enum Scale {
    /// 1/8 scale: the DC coefficient alone.
    Eighth,
    /// 1/4 scale.
    Quarter,
    /// 1/2 scale.
    Half,
    /// Full resolution.
    #[default]
    Full,
}

impl Scale {
    /// Every scale, coarsest first.
    pub const ALL: [Self; 4] = [Self::Eighth, Self::Quarter, Self::Half, Self::Full];

    /// The side of the block this scale produces, in samples: `M` of `M/8`.
    #[must_use]
    pub const fn block_size(self) -> u32 {
        match self {
            Self::Eighth => 1,
            Self::Quarter => 2,
            Self::Half => 4,
            Self::Full => 8,
        }
    }

    /// The size `source` decodes to at this scale.
    ///
    /// Rounded up, as the format requires: an image 61 pixels wide is 8 blocks
    /// across, and at 1/8 scale those 8 blocks are 8 samples.
    #[must_use]
    pub const fn apply(self, source: u32) -> u32 {
        let scaled = source.div_ceil(8 / self.block_size());
        // A one-pixel image still has a block, and so still has a sample.
        if scaled == 0 { 1 } else { scaled }
    }

    /// The coarsest scale still at least `target` in both axes.
    ///
    /// Never returns a scale that would decode *below* the target: enlarging
    /// afterwards would throw away detail and then invent it back, which is
    /// worse than decoding one step larger and shrinking properly.
    #[must_use]
    pub fn fitting(source: (u32, u32), target: (u32, u32)) -> Self {
        Self::ALL
            .into_iter()
            .find(|scale| scale.apply(source.0) >= target.0 && scale.apply(source.1) >= target.1)
            .unwrap_or(Self::Full)
    }
}

/// Basis tables for the reduced transforms: `BOX_M[u][x]` maps coefficient
/// `u` onto reduced output `x`, scaled by `2^SCALE`.
///
/// # What these are
///
/// Each entry is `C(u) * (M/8) * sum over the full-resolution samples that
/// output `x` covers, of cos((2t+1)*u*pi/16)`. That makes the reduced
/// transform *exactly* the box average of the full one — the average of the
/// `8/M` samples it replaces — rather than an approximation of it.
///
/// The obvious alternative, inverse-transforming only the top-left `MxM`
/// coefficients, is not the same thing and is measurably worse: it discards
/// the high-frequency coefficients outright instead of folding their
/// contribution into the average, so detailed images come out visibly wrong.
/// Against libjpeg's own scaled decode on a noise fixture at 1/2, truncation
/// was ten times further from the true downsample than this is.
///
/// Note `u = 4` contributing nothing at 1/2 scale, and every odd-indexed
/// coefficient contributing nothing at 1/8: those are the terms that average
/// to zero over their group. libjpeg's reduced transforms drop exactly the
/// same terms, which is a useful sign that both are computing the same thing.
///
/// Hardcoded rather than computed, because `cos` is not available in a `const`
/// context and because a value computed at run time could differ in its last
/// bit between targets — which ADR-0011 forbids. The tests check every entry
/// against the real cosines, so a transcription slip fails loudly.
const BOX_1: [[i64; 1]; 8] = [[2896], [0], [0], [0], [0], [0], [0], [0]];

/// The 1/4-scale basis.
const BOX_2: [[i64; 2]; 8] = [
    [2896, 2896],
    [2624, -2624],
    [0, 0],
    [-922, 922],
    [0, 0],
    [616, -616],
    [0, 0],
    [-522, 522],
];

/// The 1/2-scale basis.
const BOX_4: [[i64; 4]; 8] = [
    [2896, 2896, 2896, 2896],
    [3711, 1537, -1537, -3711],
    [2676, -2676, -2676, 2676],
    [1303, -3146, 3146, -1303],
    [0, 0, 0, 0],
    [-871, 2102, -2102, 871],
    [-1108, 1108, 1108, -1108],
    [-738, -306, 306, 738],
];

/// Inverse-transform a block at a reduced scale, into `MxM` samples.
///
/// Both passes multiply by a `2^SCALE` constant and the transform carries the
/// 1/4 every 2-D IDCT ends on, so the result is descaled by `2 * SCALE + 2` —
/// the same total the full transform applies, which is what makes the two
/// agree on a flat block.
fn reduced<const M: usize>(
    coefficients: &[i32; 64],
    basis: &[[i64; M]; 8],
    out: &mut [u8],
    offset: usize,
    stride: usize,
) {
    // Rows first. Every coefficient row contributes, unlike a truncating
    // transform: that is the whole difference between this and a crop of the
    // low frequencies.
    let mut rows = [[0_i64; M]; 8];
    for (v, row) in rows.iter_mut().enumerate() {
        for (u, weights) in basis.iter().enumerate() {
            // Most coefficients in a real block are zero, and skipping them
            // is worth more than any arrangement of the loop below.
            let coefficient = i64::from(coefficients.get(v * 8 + u).copied().unwrap_or(0));
            if coefficient == 0 {
                continue;
            }
            for (slot, &weight) in row.iter_mut().zip(weights.iter()) {
                *slot += coefficient * weight;
            }
        }
    }

    for y in 0..M {
        let Some(target) = out.get_mut(offset + y * stride..) else {
            continue;
        };
        for (x, slot) in target.iter_mut().take(M).enumerate() {
            let mut sum = 0_i64;
            for (v, weights) in basis.iter().enumerate() {
                let value = rows.get(v).and_then(|row| row.get(x)).copied().unwrap_or(0);
                if value != 0 {
                    sum += value * weights.get(y).copied().unwrap_or(0);
                }
            }
            let shifted = (sum + rounding(2 * SCALE + 2)) >> (2 * SCALE + 2);
            *slot = (shifted + 128).clamp(0, 255) as u8;
        }
    }
}

/// Inverse-transform one dequantized block at `scale`.
///
/// Writes `scale.block_size()` rows of that many samples at `offset`, so a
/// caller's plane geometry scales with the same factor and nothing else in the
/// decoder needs to know a reduced transform happened.
pub fn scaled_block(
    coefficients: &[i32; 64],
    scale: Scale,
    out: &mut [u8],
    offset: usize,
    stride: usize,
) {
    match scale {
        Scale::Eighth => reduced(coefficients, &BOX_1, out, offset, stride),
        Scale::Quarter => reduced(coefficients, &BOX_2, out, offset, stride),
        Scale::Half => reduced(coefficients, &BOX_4, out, offset, stride),
        // The full transform is separable and specialized; the general form
        // above would be correct but several times slower.
        Scale::Full => block(coefficients, out, offset, stride),
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
    fn the_basis_tables_match_the_averages_they_stand_for() {
        // The tables are hardcoded because `cos` is not const and because a
        // value computed at run time could differ in its last bit between
        // targets. This is what stops that from being a place a typo hides.
        let expect = |m: usize, u: usize, x: usize| -> i64 {
            let group = 8 / m;
            let c = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
            let sum: f64 = (0..group)
                .map(|k| {
                    let t = x * group + k;
                    (((2 * t + 1) as f64 * u as f64 * std::f64::consts::PI) / 16.0).cos()
                })
                .sum();
            (c * (m as f64 / 8.0) * sum * f64::from(1_i32 << SCALE)).round() as i64
        };
        for (u, row) in BOX_1.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                assert_eq!(value, expect(1, u, x), "BOX_1[{u}][{x}]");
            }
        }
        for (u, row) in BOX_2.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                assert_eq!(value, expect(2, u, x), "BOX_2[{u}][{x}]");
            }
        }
        for (u, row) in BOX_4.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                assert_eq!(value, expect(4, u, x), "BOX_4[{u}][{x}]");
            }
        }
    }

    #[test]
    fn a_scaled_block_averages_to_what_the_full_block_averages_to() {
        // The defining property of a reduced IDCT: it is the full block
        // low-pass filtered, so the mean must survive. A scaling error shows
        // up here immediately, because the mean is carried entirely by the DC
        // coefficient the reduced transform keeps.
        let mut state = 5_150_u32;
        for _ in 0..40 {
            let mut coefficients = [0_i32; 64];
            for slot in &mut coefficients {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *slot = ((state >> 20) as i32 % 50) - 25;
            }
            coefficients[0] = 200;

            let mut full = vec![0_u8; 64];
            block(&coefficients, &mut full, 0, 8);
            // Clamping would break the premise rather than the transform: a
            // sample pinned at 0 or 255 has had energy discarded, and the
            // means would then legitimately differ. The AC range above keeps
            // the block clear of both ends, and this is what proves it.
            assert!(
                full.iter().all(|&v| v > 0 && v < 255),
                "the test block clamped; the mean comparison would be vacuous"
            );
            let mean = full.iter().map(|&v| u32::from(v)).sum::<u32>() as f64 / 64.0;

            for scale in [Scale::Eighth, Scale::Quarter, Scale::Half] {
                let m = scale.block_size() as usize;
                let mut small = vec![0_u8; m * m];
                scaled_block(&coefficients, scale, &mut small, 0, m);
                let got = small.iter().map(|&v| u32::from(v)).sum::<u32>() as f64 / (m * m) as f64;
                assert!(
                    (got - mean).abs() <= 1.5,
                    "{scale:?}: mean {got:.2} against the full block's {mean:.2}"
                );
            }
        }
    }

    #[test]
    fn a_flat_block_stays_flat_at_every_scale() {
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 8 * 16;
        for scale in Scale::ALL {
            let m = scale.block_size() as usize;
            let mut out = vec![0_u8; m * m];
            scaled_block(&coefficients, scale, &mut out, 0, m);
            assert!(
                out.iter().all(|&v| v == 144),
                "{scale:?}: {out:?} is not a flat 144"
            );
        }
    }

    #[test]
    fn a_scaled_block_is_a_box_downsample_of_the_full_one() {
        // Not "close to" but *is*: the basis is defined as the average over
        // the samples each reduced output replaces, so the only difference
        // from box-averaging the full transform is fixed-point rounding in
        // two places. This is the property the whole design rests on, and the
        // one a truncating transform would fail outright.
        let mut coefficients = [0_i32; 64];
        coefficients[0] = 400;
        coefficients[1] = -180;
        coefficients[8] = 120;
        coefficients[9] = 60;
        coefficients[2] = 40;
        coefficients[16] = -30;

        let mut full = vec![0_u8; 64];
        block(&coefficients, &mut full, 0, 8);

        for scale in [Scale::Quarter, Scale::Half] {
            let m = scale.block_size() as usize;
            let factor = 8 / m;
            let mut small = vec![0_u8; m * m];
            scaled_block(&coefficients, scale, &mut small, 0, m);

            for y in 0..m {
                for x in 0..m {
                    let mut sum = 0_u32;
                    for dy in 0..factor {
                        for dx in 0..factor {
                            sum += u32::from(full[(y * factor + dy) * 8 + x * factor + dx]);
                        }
                    }
                    let boxed = sum as f64 / (factor * factor) as f64;
                    let got = f64::from(small[y * m + x]);
                    assert!(
                        (got - boxed).abs() <= 1.5,
                        "{scale:?} at ({x},{y}): {got} against box average {boxed:.1}"
                    );
                }
            }
        }
    }

    #[test]
    fn scales_map_sizes_the_way_the_format_counts_blocks() {
        assert_eq!(Scale::Eighth.apply(64), 8);
        assert_eq!(Scale::Quarter.apply(64), 16);
        assert_eq!(Scale::Half.apply(64), 32);
        assert_eq!(Scale::Full.apply(64), 64);
        // Rounded up: 61 pixels is 8 blocks, and 8 blocks is 8 samples at 1/8.
        assert_eq!(Scale::Eighth.apply(61), 8);
        assert_eq!(Scale::Quarter.apply(61), 16);
        // Never zero, however small the source.
        assert_eq!(Scale::Eighth.apply(1), 1);
        assert_eq!(Scale::Eighth.apply(3), 1);
    }

    #[test]
    fn fitting_never_decodes_below_the_target() {
        let source = (4000_u32, 3000_u32);
        // A 200x150 thumbnail fits inside the 1/8 decode (500x375).
        assert_eq!(Scale::fitting(source, (200, 150)), Scale::Eighth);
        // 600 wide does not fit 1/8's 500, so the next step up is used.
        assert_eq!(Scale::fitting(source, (600, 450)), Scale::Quarter);
        assert_eq!(Scale::fitting(source, (1200, 900)), Scale::Half);
        assert_eq!(Scale::fitting(source, (3000, 2250)), Scale::Full);
        // A target larger than the source cannot be met by any scale, so the
        // most detailed one is the right answer rather than an error.
        assert_eq!(Scale::fitting(source, (9000, 9000)), Scale::Full);

        // Both axes have to fit, not just one.
        assert_eq!(Scale::fitting(source, (200, 400)), Scale::Quarter);
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
