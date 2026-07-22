//! Sample planes for reconstruction.
//!
//! AV1 reconstruction is the most index-dense code in this crate by a wide
//! margin: prediction reads a block's above row and left column, transforms
//! write a block back, and every one of those touches samples by coordinate.
//! The workspace forbids raw slice indexing in production code, so rather than
//! spread `.get()?` chains across every predictor and transform, the bounds
//! reasoning is concentrated here — the same tactic `TileMut` uses for the
//! engine's own hot paths.
//!
//! A [`Plane`] stores one component's samples as `u16` (enough for 8-, 10- and
//! 12-bit) in a tight row-major buffer. Reads that stray outside the plane are
//! answered by replicating the nearest edge, which is exactly what AV1 intra
//! prediction wants when a neighbour is off the frame.

/// One component's reconstructed samples.
#[derive(Debug, Clone)]
pub struct Plane {
    data: Vec<u16>,
    width: usize,
    height: usize,
}

impl Plane {
    /// A zero-filled plane of the given size.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0; width.saturating_mul(height)],
            width,
            height,
        }
    }

    /// Plane width in samples.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Plane height in samples.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// The sample at `(x, y)`, or `None` if outside the plane.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<u16> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.data.get(y * self.width + x).copied()
    }

    /// Set the sample at `(x, y)`; out-of-range writes are dropped.
    pub fn set(&mut self, x: usize, y: usize, value: u16) {
        if x < self.width && y < self.height {
            if let Some(slot) = self.data.get_mut(y * self.width + x) {
                *slot = value;
            }
        }
    }

    /// A read-only view of row `y`, or `None` if out of range.
    #[must_use]
    pub fn row(&self, y: usize) -> Option<&[u16]> {
        if y >= self.height {
            return None;
        }
        self.data.get(y * self.width..y * self.width + self.width)
    }

    /// The sample at signed `(x, y)`, clamping the coordinates to the nearest
    /// edge. This is how intra prediction reads neighbours that may lie off the
    /// frame: the edge sample is replicated rather than treated as an error.
    ///
    /// The plane must be non-empty; a zero-sized plane yields 0.
    #[must_use]
    pub fn sample_clamped(&self, x: isize, y: isize) -> u16 {
        if self.width == 0 || self.height == 0 {
            return 0;
        }
        let cx = x.clamp(0, self.width as isize - 1) as usize;
        let cy = y.clamp(0, self.height as isize - 1) as usize;
        self.data.get(cy * self.width + cx).copied().unwrap_or(0)
    }

    /// Copy the whole plane out as a row-major `u16` buffer. Used by tests and
    /// by the plane-to-RGB conversion.
    #[must_use]
    pub fn samples(&self) -> &[u16] {
        &self.data
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

    #[test]
    fn set_and_get_round_trip_within_bounds() {
        let mut p = Plane::new(4, 3);
        p.set(2, 1, 300);
        assert_eq!(p.get(2, 1), Some(300));
        assert_eq!(p.get(0, 0), Some(0));
        assert_eq!(p.get(4, 0), None);
        assert_eq!(p.get(0, 3), None);
    }

    #[test]
    fn out_of_range_writes_are_dropped() {
        let mut p = Plane::new(2, 2);
        p.set(5, 5, 999);
        assert_eq!(p.samples(), &[0, 0, 0, 0]);
    }

    #[test]
    fn clamped_sampling_replicates_the_edge() {
        let mut p = Plane::new(3, 2);
        for y in 0..2 {
            for x in 0..3 {
                p.set(x, y, (10 * y + x) as u16);
            }
        }
        // Inside.
        assert_eq!(p.sample_clamped(1, 1), 11);
        // Left/above of the plane clamps to (0, 0).
        assert_eq!(p.sample_clamped(-4, -9), 0);
        // Past the right/bottom clamps to the far corner (x=2, y=1) = 12.
        assert_eq!(p.sample_clamped(99, 99), 12);
        // Mixed: past the right edge on row 0.
        assert_eq!(p.sample_clamped(99, 0), 2);
    }

    #[test]
    fn row_returns_exactly_the_width() {
        let mut p = Plane::new(3, 2);
        p.set(0, 1, 7);
        p.set(2, 1, 9);
        assert_eq!(p.row(1), Some(&[7, 0, 9][..]));
        assert_eq!(p.row(2), None);
    }
}
