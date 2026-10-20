//! Modular prediction as the decoder computes it (ISO/IEC 18181-1 §H.3,
//! H.4): the neighbourhood of each sample, the sixteen properties a tree
//! decides on, and the self-correcting ("weighted") predictor.
//!
//! The encoder must reproduce the decoder's arithmetic exactly, edges
//! included, or every later sample decodes wrong. Neighbours missing at an
//! edge are replaced the way the decoder's row state does it: west by north
//! at the start of a row, north by west on the first row, and so on.

#![allow(
    clippy::indexing_slicing,
    reason = "sample indices are within the channel region; arrays are fixed size"
)]

use crate::modular::Channel;

/// The neighbours of one sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct Neighbours {
    pub w: i32,
    pub n: i32,
    pub nw: i32,
    pub ne: i32,
    pub nn: i32,
    pub ww: i32,
}

/// A region of a channel, read as the decoder sees it: an image of its own.
#[derive(Debug, Clone, Copy)]
pub struct Region<'a> {
    pub channel: &'a Channel,
    pub x0: usize,
    pub y0: usize,
    pub width: usize,
    pub height: usize,
}

impl Region<'_> {
    /// The sample at `(x, y)` of the region.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> i32 {
        self.channel.samples[(self.y0 + y) * self.channel.width + self.x0 + x]
    }

    /// The neighbours of `(x, y)`, with the decoder's edge rules.
    #[must_use]
    pub fn neighbours(&self, x: usize, y: usize) -> Neighbours {
        let w = if x > 0 {
            self.at(x - 1, y)
        } else if y > 0 {
            self.at(x, y - 1)
        } else {
            0
        };
        let n = if y > 0 { self.at(x, y - 1) } else { w };
        let nw = if x > 0 && y > 0 {
            self.at(x - 1, y - 1)
        } else {
            w
        };
        let ne = if y > 0 && x + 1 < self.width {
            self.at(x + 1, y - 1)
        } else {
            n
        };
        let nn = if y > 1 { self.at(x, y - 2) } else { n };
        let ww = if x > 1 { self.at(x - 2, y) } else { w };
        Neighbours {
            w,
            n,
            nw,
            ne,
            nn,
            ww,
        }
    }
}

/// The predictors this encoder uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predictor {
    /// The clamped gradient, predictor 5.
    Gradient,
    /// The self-correcting predictor, predictor 6.
    Weighted,
}

impl Predictor {
    /// The predictor's number in a tree.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::Gradient => 5,
            Self::Weighted => 6,
        }
    }
}

/// The clamped gradient predictor: `W + N - NW`, kept between `W` and `N`.
#[must_use]
pub fn gradient(nb: &Neighbours) -> i32 {
    let (lo, hi) = (nb.w.min(nb.n), nb.w.max(nb.n));
    (i64::from(nb.w) + i64::from(nb.n) - i64::from(nb.nw)).clamp(i64::from(lo), i64::from(hi))
        as i32
}

/// `1 << 24` divided by 0 to 64, for the weighted predictor's divisions.
const DIV: [u32; 65] = {
    let mut out = [0_u32; 65];
    let mut i = 1;
    while i <= 64 {
        out[i] = ((1_u32 << 24) / i as u32) as u32;
        i += 1;
    }
    out
};

/// The default weighted-predictor parameters (the `WpHeader` defaults).
const P1: i64 = 16;
const P2: i64 = 10;
const P3: [i64; 5] = [7, 7, 7, 0, 0];
const WEIGHTS: [u32; 4] = [13, 12, 12, 12];

/// One weighted prediction.
#[derive(Debug, Clone, Copy)]
pub struct Weighted {
    /// The prediction, in eighths.
    prediction: i64,
    /// The largest neighbouring error, property 15.
    pub max_error: i32,
    subpred: [i64; 4],
}

impl Weighted {
    /// The prediction, as a sample.
    #[must_use]
    pub const fn value(&self) -> i32 {
        ((self.prediction + 3) >> 3) as i32
    }
}

/// The self-correcting predictor's state over one channel region, in
/// raster order.
#[derive(Debug, Clone)]
pub struct WeightedState {
    width: usize,
    x: usize,
    y: usize,
    true_err_row: Vec<i32>,
    subpred_err_row: Vec<[u32; 4]>,
    err_w: i32,
    err_nw: i32,
    err_n: i32,
    err_ne: i32,
    sub_nw_ww: [u32; 4],
    sub_n_w: [u32; 4],
    sub_ne: [u32; 4],
}

impl WeightedState {
    /// Fresh state for a region `width` samples wide.
    #[must_use]
    pub fn new(width: usize) -> Self {
        Self {
            width,
            x: 0,
            y: 0,
            true_err_row: vec![0; width],
            subpred_err_row: vec![[0; 4]; width],
            err_w: 0,
            err_nw: 0,
            err_n: 0,
            err_ne: 0,
            sub_nw_ww: [0; 4],
            sub_n_w: [0; 4],
            sub_ne: [0; 4],
        }
    }

    /// The prediction for the current sample.
    #[must_use]
    pub fn predict(&self, nb: &Neighbours) -> Weighted {
        let (ew, enw, en, ene) = (
            i64::from(self.err_w),
            i64::from(self.err_nw),
            i64::from(self.err_n),
            i64::from(self.err_ne),
        );
        let n3 = i64::from(nb.n) << 3;
        let nw3 = i64::from(nb.nw) << 3;
        let ne3 = i64::from(nb.ne) << 3;
        let w3 = i64::from(nb.w) << 3;
        let nn3 = i64::from(nb.nn) << 3;
        let subpred = [
            w3 + ne3 - n3,
            n3 - (((ew + en + ene) * P1) >> 5),
            w3 - (((ew + en + enw) * P2) >> 5),
            n3 - ((enw * P3[0]
                + en * P3[1]
                + ene * P3[2]
                + (nn3 - n3) * P3[3]
                + (nw3 - w3) * P3[4])
                >> 5),
        ];
        let mut weight = [0_u32; 4];
        for i in 0..4 {
            let err_sum = self.sub_nw_ww[i]
                .wrapping_add(self.sub_n_w[i])
                .wrapping_add(self.sub_ne[i]);
            let shift = ((u64::from(err_sum) + 1) >> 5).checked_ilog2().unwrap_or(0);
            weight[i] = 4 + ((WEIGHTS[i] * DIV[(err_sum >> shift) as usize + 1]) >> shift);
        }
        let sum: u32 = weight.iter().sum();
        let log_weight = (u64::from(sum) >> 4).ilog2();
        for w in &mut weight {
            *w >>= log_weight;
        }
        let sum: u32 = weight.iter().sum();
        let mut s = (i64::from(sum) >> 1) - 1;
        for (p, w) in subpred.iter().zip(weight) {
            s += p * i64::from(w);
        }
        let mut prediction = (s * i64::from(DIV[sum as usize])) >> 24;
        if (en ^ ew) | (en ^ enw) <= 0 {
            let lo = n3.min(w3).min(ne3);
            let hi = n3.max(w3).max(ne3);
            prediction = prediction.clamp(lo, hi);
        }
        let mut max_error = ew;
        for e in [en, enw, ene] {
            if e.abs() > max_error.abs() {
                max_error = e;
            }
        }
        Weighted {
            prediction,
            max_error: max_error as i32,
            subpred,
        }
    }

    /// Record the current sample's true value and move to the next.
    pub fn record(&mut self, pred: &Weighted, sample: i32) {
        let sample = i64::from(sample) << 3;
        let true_err = (pred.prediction - sample) as i32;
        let mut sub_err = [0_u32; 4];
        for (e, p) in sub_err.iter_mut().zip(pred.subpred) {
            *e = ((p.abs_diff(sample) + 3) >> 3) as u32;
        }
        self.true_err_row[self.x] = true_err;
        self.subpred_err_row[self.x] = sub_err;
        self.x += 1;
        if self.x >= self.width {
            self.y += 1;
            self.x = 0;
            self.err_w = 0;
            self.err_n = self.true_err_row[0];
            self.err_nw = self.err_n;
            self.sub_n_w = self.subpred_err_row[0];
            self.sub_nw_ww = self.sub_n_w;
            if self.width <= 1 {
                self.err_ne = self.err_n;
                self.sub_ne = self.sub_n_w;
            } else {
                self.err_ne = self.true_err_row[1];
                self.sub_ne = self.subpred_err_row[1];
            }
        } else {
            self.err_w = true_err;
            self.err_nw = self.err_n;
            self.err_n = self.err_ne;
            self.sub_nw_ww = self.sub_n_w;
            self.sub_n_w = self.sub_ne;
            for (a, b) in self.sub_n_w.iter_mut().zip(sub_err) {
                *a = a.wrapping_add(b);
            }
            if self.x + 1 >= self.width {
                self.err_ne = self.err_n;
                self.sub_ne = self.sub_n_w;
            } else if self.y != 0 {
                self.err_ne = self.true_err_row[self.x + 1];
                self.sub_ne = self.subpred_err_row[self.x + 1];
            }
        }
    }
}

/// The sixteen properties of a sample (§H.4.1), given its neighbours, the
/// previous sample's property 9 on this row, and the weighted predictor's
/// error.
#[must_use]
pub fn properties(
    channel: u32,
    x: usize,
    y: usize,
    nb: &Neighbours,
    prev_grad: i32,
    max_error: i32,
) -> [i32; 16] {
    let w_nw = nb.w.wrapping_sub(nb.nw);
    [
        channel as i32,
        0,
        y as i32,
        x as i32,
        nb.n.unsigned_abs() as i32,
        nb.w.unsigned_abs() as i32,
        nb.n,
        nb.w,
        nb.w.wrapping_sub(prev_grad),
        w_nw.wrapping_add(nb.n),
        w_nw,
        nb.nw.wrapping_sub(nb.n),
        nb.n.wrapping_sub(nb.ne),
        nb.n.wrapping_sub(nb.nn),
        nb.w.wrapping_sub(nb.ww),
        max_error,
    ]
}
