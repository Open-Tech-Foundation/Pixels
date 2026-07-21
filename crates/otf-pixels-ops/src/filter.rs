//! Resampling filters and the weight tables resize builds from them.
//!
//! A separable resize is two one-dimensional passes, and each pass is the same
//! thing: for every output position, a short run of input samples multiplied by
//! weights that sum to one. Everything specific to a filter lives in
//! [`Filter::weight`]; everything specific to a *scale* lives in [`Weights`].
//!
//! # Why the weights are precomputed
//!
//! Evaluating `sinc` per pixel would dominate the cost and, worse, would put a
//! transcendental function inside the loop we want vectorized. Computing the
//! table once per pass costs `output_length` evaluations instead of
//! `output_length × input_length`, and leaves an inner loop of nothing but
//! multiply-accumulate.
//!
//! # Fixed point
//!
//! Per ADR-0011, eight-bit paths use `i32` fixed-point weights. Quantization
//! happens once, here, and the residual is corrected so each run still sums to
//! exactly [`ONE`]: an uncorrected table drifts the output brightness by a
//! fraction of a level, which is visible as banding on a gradient.

use otf_pixels_core::{PixelsError, Result};

/// Fixed-point scale for quantized weights: one unit of `1.0`.
///
/// 14 bits leaves room for a full 8-bit sample (8 bits) times the largest
/// plausible coefficient sum, accumulated over a Lanczos3 support of up to a
/// few dozen taps, without leaving `i32`. Lanczos weights are signed and can
/// overshoot, so the headroom is not merely the positive case.
pub const ONE: i32 = 1 << 14;

/// A resampling filter kernel (SPEC §Core ops).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Filter {
    /// Nearest neighbour. Fastest, blockiest; the only filter that preserves
    /// exact sample values, which is why it is the right default for masks
    /// and palettes rather than photographs.
    Nearest,
    /// Box average over the source footprint. The correct choice for large
    /// downscales, where it is both fast and alias-free.
    Box,
    /// Linear interpolation between the two nearest samples.
    Bilinear,
    /// Catmull-Rom cubic. Sharper than Mitchell, mild ringing.
    CatmullRom,
    /// Mitchell-Netravali cubic (B=C=1/3). The usual compromise between
    /// blurring and ringing.
    Mitchell,
    /// Lanczos windowed sinc, 2 lobes.
    Lanczos2,
    /// Lanczos windowed sinc, 3 lobes. The default: sharpest of these at the
    /// cost of some ringing on hard edges.
    #[default]
    Lanczos3,
}

impl Filter {
    /// The filter's radius in **output** units before scale is applied.
    ///
    /// The actual support in input pixels is this scaled by the downsampling
    /// ratio, because downscaling must average over everything it discards or
    /// it aliases.
    #[must_use]
    pub const fn support(self) -> f32 {
        match self {
            Self::Nearest => 0.5,
            Self::Box => 0.5,
            Self::Bilinear => 1.0,
            Self::CatmullRom | Self::Mitchell => 2.0,
            Self::Lanczos2 => 2.0,
            Self::Lanczos3 => 3.0,
        }
    }

    /// The filter's weight at signed distance `x` from the sample centre.
    #[must_use]
    pub fn weight(self, x: f32) -> f32 {
        let t = x.abs();
        match self {
            Self::Nearest => {
                if t <= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Box => {
                if t < 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Bilinear => {
                if t < 1.0 {
                    1.0 - t
                } else {
                    0.0
                }
            }
            Self::CatmullRom => cubic(t, 0.0, 0.5),
            Self::Mitchell => cubic(t, 1.0 / 3.0, 1.0 / 3.0),
            Self::Lanczos2 => lanczos(t, 2.0),
            Self::Lanczos3 => lanczos(t, 3.0),
        }
    }

    /// A short, stable name for diagnostics and benchmark output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Box => "box",
            Self::Bilinear => "bilinear",
            Self::CatmullRom => "catmull-rom",
            Self::Mitchell => "mitchell",
            Self::Lanczos2 => "lanczos2",
            Self::Lanczos3 => "lanczos3",
        }
    }
}

/// The Mitchell-Netravali cubic family, of which Catmull-Rom is `B=0, C=1/2`.
fn cubic(t: f32, b: f32, c: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    if t < 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * t3 + (-18.0 + 12.0 * b + 6.0 * c) * t2 + (6.0 - 2.0 * b))
            / 6.0
    } else if t < 2.0 {
        ((-b - 6.0 * c) * t3
            + (6.0 * b + 30.0 * c) * t2
            + (-12.0 * b - 48.0 * c) * t
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// A sinc windowed by a wider sinc — the Lanczos kernel.
fn lanczos(t: f32, lobes: f32) -> f32 {
    if t < f32::EPSILON {
        return 1.0;
    }
    if t >= lobes {
        return 0.0;
    }
    sinc(t) * sinc(t / lobes)
}

/// The normalized sinc, `sin(pi x) / (pi x)`.
fn sinc(x: f32) -> f32 {
    let pi_x = std::f32::consts::PI * x;
    pi_x.sin() / pi_x
}

/// The input samples and weights contributing to one output position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    /// First input index this output position reads.
    pub start: u32,
    /// How many consecutive input samples it reads.
    pub len: u32,
    /// Offset of this run's weights within [`Weights::quantized`].
    pub at: usize,
}

/// Precomputed weights for one resize pass along one axis.
///
/// Built once per pass and shared by every row (or column) it is applied to,
/// which is what turns a resize into a multiply-accumulate loop.
#[derive(Debug, Clone)]
pub struct Weights {
    runs: Vec<Run>,
    /// Fixed-point weights, concatenated run by run. Each run sums to [`ONE`].
    quantized: Vec<i32>,
    /// The same weights unquantized, for the 16-bit and float paths.
    exact: Vec<f32>,
    /// The longest run, which bounds the accumulator loop.
    max_len: u32,
}

impl Weights {
    /// Build the weight table mapping `input_len` samples onto `output_len`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if either length is zero, or
    /// if the filter support at this scale would exceed what `u32` can index.
    pub fn build(filter: Filter, input_len: u32, output_len: u32) -> Result<Self> {
        if input_len == 0 || output_len == 0 {
            return Err(PixelsError::invalid_argument(
                "size",
                format!("cannot resample {input_len} samples to {output_len}"),
            ));
        }

        let scale = f64::from(output_len) / f64::from(input_len);
        // Downscaling widens the kernel in input space: an output pixel must
        // average everything that maps onto it, or the discarded samples alias
        // back as moire. Upscaling leaves the kernel at its natural width.
        let filter_scale = if scale < 1.0 { 1.0 / scale } else { 1.0 };
        let support = f64::from(filter.support()) * filter_scale;

        let mut runs = Vec::with_capacity(output_len as usize);
        let mut quantized = Vec::new();
        let mut exact = Vec::new();
        let mut max_len = 0_u32;
        let mut row: Vec<f32> = Vec::new();

        for out in 0..output_len {
            // Centre of this output pixel projected into input coordinates,
            // measured between samples rather than at them — the half-pixel
            // offset is what keeps the image from drifting by half a pixel.
            let centre = (f64::from(out) + 0.5) / scale;
            let first = ((centre - support) + 0.5).floor().max(0.0);
            let last = ((centre + support) + 0.5).ceil().min(f64::from(input_len));
            let start = first as u32;
            let len = (last - first).max(1.0) as u32;
            let len = len.min(input_len - start.min(input_len - 1));

            row.clear();
            let mut sum = 0.0_f32;
            for i in 0..len {
                let sample = f64::from(start + i) + 0.5;
                // Distance measured in *filter* space, so a widened kernel
                // still evaluates its own profile.
                let distance = ((sample - centre) / filter_scale) as f32;
                let w = filter.weight(distance);
                row.push(w);
                sum += w;
            }

            // A run whose weights cancel to nothing would divide by zero and
            // produce a black pixel; fall back to a single nearest sample.
            if sum.abs() < 1e-6 {
                row.clear();
                row.push(1.0);
                sum = 1.0;
            }

            let at = quantized.len();
            let mut total = 0_i32;
            for w in &mut row {
                *w /= sum;
                exact.push(*w);
                // Round half away from zero: weights are signed for Lanczos.
                let q = if *w >= 0.0 {
                    (*w * ONE as f32 + 0.5) as i32
                } else {
                    (*w * ONE as f32 - 0.5) as i32
                };
                quantized.push(q);
                total += q;
            }
            // Quantization residue goes to the largest weight, so every run
            // sums to exactly ONE. Without this the image drifts a fraction of
            // a level darker or lighter, which shows as banding on gradients.
            if total != ONE {
                let biggest = quantized
                    .get(at..)
                    .and_then(|run| {
                        run.iter()
                            .enumerate()
                            .max_by_key(|&(_, w)| *w)
                            .map(|(i, _)| at + i)
                    })
                    .unwrap_or(at);
                if let Some(slot) = quantized.get_mut(biggest) {
                    *slot += ONE - total;
                }
            }

            let len = row.len() as u32;
            max_len = max_len.max(len);
            runs.push(Run { start, len, at });
        }

        Ok(Self {
            runs,
            quantized,
            exact,
            max_len,
        })
    }

    /// The runs, one per output position.
    #[must_use]
    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// The fixed-point weights of `run`.
    #[must_use]
    pub fn quantized(&self, run: &Run) -> &[i32] {
        self.quantized
            .get(run.at..run.at + run.len as usize)
            .unwrap_or(&[])
    }

    /// The exact weights of `run`.
    #[must_use]
    pub fn exact(&self, run: &Run) -> &[f32] {
        self.exact
            .get(run.at..run.at + run.len as usize)
            .unwrap_or(&[])
    }

    /// The longest run in the table.
    #[must_use]
    pub const fn max_len(&self) -> u32 {
        self.max_len
    }

    /// The sub-table covering `out_len` output positions from `out_start`,
    /// rebased so run starts are relative to input index `in_start`.
    ///
    /// This is how a tile uses the *image's* weights rather than its own.
    /// Building a fresh table from the tile's dimensions would resample at the
    /// tile's scale instead of the image's, so an output pixel would depend on
    /// where the tile boundaries fell — which SPEC §Guarantees 2 forbids.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::graph`] if the requested outputs fall outside
    /// this table, or if a run would start before `in_start` — either means
    /// the tile is not the footprint `input_regions` asked for.
    pub fn for_tile(&self, out_start: u32, out_len: u32, in_start: u32) -> Result<Self> {
        let from = out_start as usize;
        let to = from + out_len as usize;
        let slice = self.runs.get(from..to).ok_or_else(|| {
            PixelsError::graph(format!(
                "resize weights cover {} outputs, tile wants {from}..{to}",
                self.runs.len()
            ))
        })?;

        let mut runs = Vec::with_capacity(slice.len());
        let mut quantized = Vec::new();
        let mut exact = Vec::new();
        let mut max_len = 0_u32;
        for run in slice {
            let start = run.start.checked_sub(in_start).ok_or_else(|| {
                PixelsError::graph(format!(
                    "resize tile starts at input {in_start} but a run needs {}",
                    run.start
                ))
            })?;
            let at = quantized.len();
            quantized.extend_from_slice(self.quantized(run));
            exact.extend_from_slice(self.exact(run));
            max_len = max_len.max(run.len);
            runs.push(Run {
                start,
                len: run.len,
                at,
            });
        }
        Ok(Self {
            runs,
            quantized,
            exact,
            max_len,
        })
    }

    /// The first and last input index any output position reads.
    ///
    /// This is what `input_regions` needs: the footprint of a whole pass.
    #[must_use]
    pub fn footprint(&self, from: u32, len: u32) -> (u32, u32) {
        let mut lo = u32::MAX;
        let mut hi = 0_u32;
        for run in self.runs.iter().skip(from as usize).take(len as usize) {
            lo = lo.min(run.start);
            hi = hi.max(run.start + run.len);
        }
        if lo == u32::MAX {
            (0, 0)
        } else {
            (lo, hi - lo)
        }
    }
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
    fn every_filter_peaks_at_the_centre_and_is_zero_past_its_support() {
        // Deliberately *not* "is 1.0 at the centre": Mitchell peaks at 8/9,
        // because a cubic with B=1/3 trades peak height for smoothness. What
        // every kernel must share is that the centre is the maximum — a kernel
        // peaking off-centre shifts the image — and that it vanishes outside
        // its declared support, which is what makes the support honest.
        for filter in ALL {
            let centre = filter.weight(0.0);
            for step in 1..40 {
                let x = step as f32 * 0.1;
                assert!(
                    filter.weight(x) <= centre + 1e-6,
                    "{} peaks at {x}, not at the centre",
                    filter.as_str()
                );
            }
            let past = filter.support() + 0.01;
            assert!(
                filter.weight(past).abs() < 1e-6,
                "{} is non-zero past its support",
                filter.as_str()
            );
        }
    }

    #[test]
    fn every_filter_is_symmetric() {
        // An asymmetric kernel shifts the image, which is the kind of bug that
        // looks like "slightly soft" rather than like a failure.
        for filter in ALL {
            for step in 0..40 {
                let x = step as f32 * 0.1;
                let (l, r) = (filter.weight(-x), filter.weight(x));
                assert!(
                    (l - r).abs() < 1e-6,
                    "{} is asymmetric at {x}: {l} vs {r}",
                    filter.as_str()
                );
            }
        }
    }

    #[test]
    fn every_run_sums_to_exactly_one() {
        // The property the residue correction exists for. A run that sums to
        // ONE-1 darkens the image by a fraction of a level everywhere, which
        // is invisible per pixel and obvious on a gradient.
        for filter in ALL {
            for (input, output) in [(100, 50), (50, 100), (7, 7), (1, 64), (64, 1), (999, 37)] {
                let weights = Weights::build(filter, input, output).unwrap();
                for (index, run) in weights.runs().iter().enumerate() {
                    let sum: i32 = weights.quantized(run).iter().sum();
                    assert_eq!(
                        sum,
                        ONE,
                        "{} {input}->{output} run {index} sums to {sum}, not {ONE}",
                        filter.as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn exact_weights_sum_to_one_too() {
        for filter in ALL {
            for (input, output) in [(100, 50), (50, 100), (33, 17)] {
                let weights = Weights::build(filter, input, output).unwrap();
                for run in weights.runs() {
                    let sum: f32 = weights.exact(run).iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-4,
                        "{} {input}->{output} exact run sums to {sum}",
                        filter.as_str()
                    );
                }
            }
        }
    }

    #[test]
    fn every_run_stays_inside_the_input() {
        // An op reading outside its input is a defect (Op::input_regions), so
        // clamping is this table's job and not the kernel's.
        for filter in ALL {
            for (input, output) in [(10, 1), (1, 10), (37, 999), (999, 37), (2, 3)] {
                let weights = Weights::build(filter, input, output).unwrap();
                for run in weights.runs() {
                    assert!(
                        run.start + run.len <= input,
                        "{} {input}->{output} reads {}..{} of {input}",
                        filter.as_str(),
                        run.start,
                        run.start + run.len
                    );
                    assert!(run.len > 0, "empty run");
                }
            }
        }
    }

    #[test]
    fn one_to_one_resize_is_the_identity() {
        // The strongest single check on the half-pixel convention: at scale 1
        // every output must read exactly its own input sample at full weight.
        // Off-by-a-half-pixel shows up here and nowhere else so cleanly.
        for filter in ALL {
            let weights = Weights::build(filter, 64, 64).unwrap();
            for (index, run) in weights.runs().iter().enumerate() {
                let quantized = weights.quantized(run);
                let peak = quantized
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, w)| *w)
                    .map(|(i, _)| run.start as usize + i)
                    .unwrap();
                assert_eq!(
                    peak,
                    index,
                    "{} at 1:1 centres output {index} on input {peak}",
                    filter.as_str()
                );
            }
        }
    }

    #[test]
    fn downscaling_widens_the_kernel() {
        // Averaging over everything discarded is what stops a downscale
        // aliasing. A kernel that stayed at its natural width would sample
        // rather than filter.
        let half = Weights::build(Filter::Bilinear, 100, 50).unwrap();
        let same = Weights::build(Filter::Bilinear, 100, 100).unwrap();
        assert!(
            half.max_len() > same.max_len(),
            "downscale support {} is not wider than 1:1 support {}",
            half.max_len(),
            same.max_len()
        );
    }

    #[test]
    fn the_footprint_covers_every_run_it_spans() {
        let weights = Weights::build(Filter::Lanczos3, 200, 97).unwrap();
        let (start, len) = weights.footprint(10, 20);
        for run in weights.runs().iter().skip(10).take(20) {
            assert!(run.start >= start, "run starts before the footprint");
            assert!(
                run.start + run.len <= start + len,
                "run ends after the footprint"
            );
        }
    }

    #[test]
    fn an_empty_footprint_is_reported_as_empty() {
        let weights = Weights::build(Filter::Box, 10, 10).unwrap();
        assert_eq!(weights.footprint(0, 0), (0, 0));
    }

    #[test]
    fn a_zero_length_axis_is_an_error_not_a_panic() {
        assert!(Weights::build(Filter::Box, 0, 10).is_err());
        assert!(Weights::build(Filter::Box, 10, 0).is_err());
    }

    #[test]
    fn building_a_table_is_deterministic() {
        // SPEC §Guarantees 2. Float weights make this worth pinning: the same
        // inputs must give the same bits, run to run.
        for filter in ALL {
            let a = Weights::build(filter, 1000, 173).unwrap();
            let b = Weights::build(filter, 1000, 173).unwrap();
            for (ra, rb) in a.runs().iter().zip(b.runs()) {
                assert_eq!(a.quantized(ra), b.quantized(rb));
                assert_eq!(a.exact(ra), b.exact(rb));
            }
        }
    }
}
