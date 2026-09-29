//! The super-resolution upscaling process (spec §7.16).
//!
//! With `use_superres`, a frame is coded at a reduced width — `FrameWidth`, which
//! is `8 / SuperresDenom` of the display width — and every tile-level step
//! (reconstruction, deblocking, CDEF) runs at that width. Between CDEF and loop
//! restoration the frame is stretched back horizontally to `UpscaledWidth` with
//! a normative 8-tap filter whose phase advances in 1/16384-sample steps, so the
//! upscale is bit-exact rather than a matter of taste. Loop restoration then
//! runs on the upscaled frame, which is why both the CDEF output and the
//! pre-CDEF frame it borrows stripe-boundary rows from are upscaled.
//!
//! Only the width changes; rows are filtered independently. The source column is
//! clamped to the coded area, `MiCols * MI_SIZE` samples (per plane), which
//! reaches a few samples past `FrameWidth` into the decoded padding when the
//! width is not a multiple of 8.
//!
//! The filter table is indexed by a phase in `0..64` and a tap in `0..8`, both
//! spec-bounded, so the module opts into `indexing_slicing` for it; samples are
//! read and written through [`Plane`]'s checked API.
#![allow(
    clippy::indexing_slicing,
    reason = "Upscale_Filter is indexed by a 6-bit phase and a tap below 8; \
              sample planes use Plane's checked API"
)]

use super::plane::Plane;

/// `MI_SIZE` (§3).
const MI_SIZE: usize = 4;
/// `SUPERRES_NUM` (§3): the numerator of the upscaling ratio.
pub const SUPERRES_NUM: usize = 8;
/// `SUPERRES_SCALE_BITS` (§3): fractional bits of the source position.
const SUPERRES_SCALE_BITS: i64 = 14;
/// `SUPERRES_SCALE_MASK` (§3).
const SUPERRES_SCALE_MASK: i64 = (1 << SUPERRES_SCALE_BITS) - 1;
/// `SUPERRES_EXTRA_BITS` (§3): `SUPERRES_SCALE_BITS - SUPERRES_FILTER_BITS`.
const SUPERRES_EXTRA_BITS: i64 = 8;
/// `SUPERRES_FILTER_TAPS` (§3).
const SUPERRES_FILTER_TAPS: usize = 8;
/// `SUPERRES_FILTER_OFFSET` (§3): the tap that sits on the source sample.
const SUPERRES_FILTER_OFFSET: i64 = 3;
/// `FILTER_BITS` (§3): the precision of the filter taps (they sum to 128).
const FILTER_BITS: u32 = 7;

/// `Upscale_Filter[SUPERRES_FILTER_SHIFTS][SUPERRES_FILTER_TAPS]` (§7.16).
const UPSCALE_FILTER: [[i32; SUPERRES_FILTER_TAPS]; 64] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// The frame geometry the upscale needs.
#[derive(Debug, Clone, Copy)]
pub struct Superres {
    /// `FrameWidth`: the coded (downscaled) luma width.
    pub frame_width: usize,
    /// `UpscaledWidth`: the luma width after upscaling.
    pub upscaled_width: usize,
    /// `FrameHeight` in luma samples.
    pub frame_height: usize,
    /// `MiCols`: the coded width in 4x4 units, which bounds the source column.
    pub mi_cols: usize,
    /// Horizontal chroma subsampling (0 for 4:4:4).
    pub subsampling_x: usize,
    /// Vertical chroma subsampling (0 for 4:4:4).
    pub subsampling_y: usize,
    /// Sample bit depth (8/10/12).
    pub bit_depth: u8,
}

impl Superres {
    /// The upscaling process (§7.16): stretch each of `planes` horizontally from
    /// `FrameWidth` to `UpscaledWidth`. Each output plane is the upscaled width
    /// wide and as tall as its input; rows past `FrameHeight` are left zero, as
    /// nothing downstream reads them.
    #[must_use]
    pub fn upscale(&self, planes: &[Plane]) -> Vec<Plane> {
        planes
            .iter()
            .enumerate()
            .map(|(plane, input)| self.upscale_plane(plane, input))
            .collect()
    }

    fn upscale_plane(&self, plane: usize, input: &Plane) -> Plane {
        let (sub_x, sub_y) = if plane == 0 {
            (0, 0)
        } else {
            (self.subsampling_x, self.subsampling_y)
        };
        let downscaled_w = round2(self.frame_width, sub_x) as i64;
        let upscaled_w = round2(self.upscaled_width, sub_x) as i64;
        let plane_h = round2(self.frame_height, sub_y);

        let step_x = ((downscaled_w << SUPERRES_SCALE_BITS) + upscaled_w / 2) / upscaled_w;
        let err = upscaled_w * step_x - (downscaled_w << SUPERRES_SCALE_BITS);
        // `/` truncates toward zero in both the spec and Rust, and the first
        // numerator is negative whenever the frame is actually upscaled.
        let mut initial_subpel_x = (-((upscaled_w - downscaled_w) << (SUPERRES_SCALE_BITS - 1))
            + upscaled_w / 2)
            / upscaled_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2;
        initial_subpel_x &= SUPERRES_SCALE_MASK;
        let mi_w = (self.mi_cols >> sub_x) as i64;
        let max_x = mi_w * MI_SIZE as i64 - 1;
        let max_value = (1_i32 << self.bit_depth) - 1;

        let mut output = Plane::new(upscaled_w as usize, input.height());
        for y in 0..plane_h {
            for x in 0..upscaled_w {
                let src_x = -(1 << SUPERRES_SCALE_BITS) + initial_subpel_x + x * step_x;
                let src_x_px = src_x >> SUPERRES_SCALE_BITS;
                let src_x_subpel = ((src_x & SUPERRES_SCALE_MASK) >> SUPERRES_EXTRA_BITS) as usize;
                let taps = &UPSCALE_FILTER[src_x_subpel];
                let mut sum = 0_i32;
                for (k, &tap) in taps.iter().enumerate() {
                    let sample_x = (src_x_px + k as i64 - SUPERRES_FILTER_OFFSET).clamp(0, max_x);
                    let px = input.get(sample_x as usize, y).unwrap_or(0);
                    sum += i32::from(px) * tap;
                }
                let value = ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, max_value);
                output.set(x as usize, y, value as u16);
            }
        }
        output
    }
}

/// `Round2(x, n)` (§4.7) for the subsampling shifts, `n` in `{0, 1}`.
fn round2(x: usize, n: usize) -> usize {
    if n == 0 { x } else { (x + (1 << (n - 1))) >> n }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    fn geometry(frame_width: usize, upscaled_width: usize, height: usize) -> Superres {
        Superres {
            frame_width,
            upscaled_width,
            frame_height: height,
            mi_cols: 2 * frame_width.div_ceil(8),
            subsampling_x: 0,
            subsampling_y: 0,
            bit_depth: 8,
        }
    }

    fn plane_from(w: usize, h: usize, f: impl Fn(usize, usize) -> u16) -> Plane {
        let mut plane = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                plane.set(x, y, f(x, y));
            }
        }
        plane
    }

    #[test]
    fn every_filter_phase_has_unit_dc_gain() {
        for (phase, taps) in UPSCALE_FILTER.iter().enumerate() {
            assert_eq!(taps.iter().sum::<i32>(), 128, "phase {phase}");
        }
    }

    #[test]
    fn a_flat_plane_upscales_to_a_flat_plane_of_the_upscaled_width() {
        // 48 wide coded at denominator 16 (8/16 of 96 upscaled).
        let sr = geometry(48, 96, 8);
        let input = plane_from(48, 8, |_, _| 77);
        let out = sr.upscale(&[input]);
        assert_eq!(out[0].width(), 96);
        assert_eq!(out[0].height(), 8);
        assert!(out[0].samples().iter().all(|&s| s == 77));
    }

    #[test]
    fn a_ramp_upscales_monotonically_within_range() {
        // A smooth horizontal ramp must stay non-decreasing (the filter's
        // ringing is far below one step here) and cover the same range.
        let sr = geometry(40, 60, 4);
        let input = plane_from(40, 4, |x, _| (x * 6) as u16);
        let out = &sr.upscale(&[input])[0];
        for y in 0..4 {
            let row = out.row(y).unwrap();
            assert!(row.windows(2).all(|w| w[0] <= w[1]), "row {y}: {row:?}");
            assert!(row[0] <= 6 && row[59] >= 228, "row {y}: {row:?}");
        }
    }

    #[test]
    fn the_source_column_clamps_to_the_coded_area_not_the_frame_width() {
        // FrameWidth 36 decodes MiCols = 10 (40 samples). Columns 36..40 are
        // decoded padding that the spec's clamp (MiCols * MI_SIZE - 1) reads,
        // so changing them must change the right edge of the upscaled row.
        let sr = geometry(36, 72, 1);
        assert_eq!(sr.mi_cols, 10);
        let base = plane_from(40, 1, |_, _| 100);
        let bright = plane_from(40, 1, |x, _| if x >= 36 { 250 } else { 100 });
        let a = &sr.upscale(&[base])[0];
        let b = &sr.upscale(&[bright])[0];
        assert_eq!(a.get(0, 0), b.get(0, 0));
        assert_ne!(a.get(71, 0), b.get(71, 0));
    }
}
