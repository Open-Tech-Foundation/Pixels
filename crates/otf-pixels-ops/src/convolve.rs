//! [`Convolve`] — arbitrary small kernels, with blur and sharpen presets.
//!
//! # Edges
//!
//! Samples outside the image are taken as the nearest edge pixel (clamp).
//! That is the choice that does not invent detail: zero-padding darkens
//! borders, and reflection fabricates structure that was never there. It is
//! also what makes the op's demand exactly the output region grown by the
//! radius and clamped — nothing outside the input is ever requested, which
//! `Op::input_regions` requires.
//!
//! # Fixed point
//!
//! Per ADR-0011 the 8-bit path quantizes the kernel into `i32` once, at
//! construction, and accumulates in `i32`. Wider formats accumulate in `f32`.
//! The kernel is normalized by its own sum, so a blur preserves brightness
//! and an edge detector summing to zero is left alone.

use otf_pixels_core::{
    AccessPattern, ImageDescriptor, Op, PixelsError, Region, Result, SampleKind, Tile, TileMut,
};

use crate::resample::MAX_CHANNELS;

/// Fixed-point scale for quantized kernel taps.
const ONE: i32 = 1 << 14;

/// A convolution kernel: odd-sized, square or rectangular, small.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    width: u32,
    height: u32,
    /// The taps, row-major, normalized so they sum to 1 (or to 0 if the
    /// original did).
    taps: Vec<f32>,
    /// The same taps in fixed point, for the 8-bit path.
    quantized: Vec<i32>,
}

impl Kernel {
    /// The largest kernel accepted, per axis.
    ///
    /// "Arbitrary small kernels" (SPEC §Core ops) needs a number attached to
    /// it. Beyond this a separable or frequency-domain implementation is the
    /// right answer, and silently accepting a 999-tap kernel would be an
    /// invitation to a very slow surprise.
    pub const MAX_SIZE: u32 = 63;

    /// Build a kernel from `width * height` taps in row-major order.
    ///
    /// Taps are normalized by their sum, so `[1; 9]` and `[1.0/9.0; 9]` are the
    /// same blur. A kernel summing to zero — an edge detector — is left
    /// unnormalized rather than divided by zero.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if the dimensions are even or
    /// zero, exceed [`Kernel::MAX_SIZE`], disagree with the tap count, or if
    /// any tap is not finite.
    pub fn new(width: u32, height: u32, taps: &[f32]) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(PixelsError::invalid_argument(
                "kernel",
                format!("kernel {width}x{height} is empty"),
            ));
        }
        if width % 2 == 0 || height % 2 == 0 {
            // An even kernel has no centre tap, so "the pixel this output
            // corresponds to" is ambiguous and the image shifts by half a
            // pixel. Refusing is better than picking a convention silently.
            return Err(PixelsError::invalid_argument(
                "kernel",
                format!("kernel {width}x{height} must have odd dimensions"),
            ));
        }
        if width > Self::MAX_SIZE || height > Self::MAX_SIZE {
            return Err(PixelsError::invalid_argument(
                "kernel",
                format!(
                    "kernel {width}x{height} exceeds the {}x{} maximum",
                    Self::MAX_SIZE,
                    Self::MAX_SIZE
                ),
            ));
        }
        let expected = width as usize * height as usize;
        if taps.len() != expected {
            return Err(PixelsError::invalid_argument(
                "kernel",
                format!("{width}x{height} needs {expected} taps, got {}", taps.len()),
            ));
        }
        if taps.iter().any(|t| !t.is_finite()) {
            return Err(PixelsError::invalid_argument(
                "kernel",
                "every tap must be finite",
            ));
        }

        let sum: f32 = taps.iter().sum();
        // A zero-sum kernel is an edge detector, not a mistake; dividing by
        // its sum would be dividing by zero.
        let divisor = if sum.abs() < 1e-6 { 1.0 } else { sum };
        let normalized: Vec<f32> = taps.iter().map(|t| t / divisor).collect();
        let quantized = normalized
            .iter()
            .map(|&t| {
                if t >= 0.0 {
                    (t * ONE as f32 + 0.5) as i32
                } else {
                    (t * ONE as f32 - 0.5) as i32
                }
            })
            .collect();

        Ok(Self {
            width,
            height,
            taps: normalized,
            quantized,
        })
    }

    /// A square kernel of `size` taps per side.
    ///
    /// # Errors
    ///
    /// As [`Kernel::new`].
    pub fn square(size: u32, taps: &[f32]) -> Result<Self> {
        Self::new(size, size, taps)
    }

    /// A box blur of `size` by `size`.
    ///
    /// # Errors
    ///
    /// As [`Kernel::new`].
    pub fn blur(size: u32) -> Result<Self> {
        let count = size as usize * size as usize;
        Self::square(size, &vec![1.0; count])
    }

    /// A Gaussian blur approximated at `sigma`, sized to 3 sigma each way.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `sigma` is not finite and
    /// positive, or if the implied kernel exceeds [`Kernel::MAX_SIZE`].
    pub fn gaussian(sigma: f32) -> Result<Self> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(PixelsError::invalid_argument(
                "sigma",
                format!("must be finite and positive, got {sigma}"),
            ));
        }
        // 3 sigma captures ~99.7% of the distribution; truncating closer
        // leaves a visible discontinuity at the kernel edge.
        let radius = (sigma * 3.0).ceil().max(1.0) as u32;
        let size = radius * 2 + 1;
        let mut taps = Vec::with_capacity((size * size) as usize);
        let denominator = 2.0 * sigma * sigma;
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - radius as f32;
                let dy = y as f32 - radius as f32;
                taps.push((-(dx * dx + dy * dy) / denominator).exp());
            }
        }
        Self::square(size, &taps)
    }

    /// A 3x3 sharpen: the identity plus a Laplacian scaled by `amount`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `amount` is not finite.
    pub fn sharpen(amount: f32) -> Result<Self> {
        if !amount.is_finite() {
            return Err(PixelsError::invalid_argument(
                "amount",
                format!("must be finite, got {amount}"),
            ));
        }
        let a = amount;
        Self::square(3, &[0.0, -a, 0.0, -a, 1.0 + 4.0 * a, -a, 0.0, -a, 0.0])
    }

    /// The kernel width in taps.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The kernel height in taps.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// How far the kernel reaches from its centre, horizontally.
    #[must_use]
    pub const fn radius_x(&self) -> u32 {
        self.width / 2
    }

    /// How far the kernel reaches from its centre, vertically.
    #[must_use]
    pub const fn radius_y(&self) -> u32 {
        self.height / 2
    }

    /// The normalized taps, row-major.
    #[must_use]
    pub fn taps(&self) -> &[f32] {
        &self.taps
    }
}

/// Apply a convolution kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct Convolve {
    kernel: Kernel,
}

impl Convolve {
    /// Convolve with `kernel`.
    #[must_use]
    pub const fn new(kernel: Kernel) -> Self {
        Self { kernel }
    }

    /// The kernel this op applies.
    #[must_use]
    pub const fn kernel(&self) -> &Kernel {
        &self.kernel
    }
}

impl Op for Convolve {
    fn name(&self) -> &'static str {
        "convolve"
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`convolve` takes one input, got none"))?;
        Ok(*input)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`convolve` takes one input, got none"))?;

        // Grown by the radius and clamped: edge handling is this op's business
        // (Op::input_regions), so it never asks for pixels that do not exist.
        let x = output.x.saturating_sub(self.kernel.radius_x());
        let y = output.y.saturating_sub(self.kernel.radius_y());
        let right = (output.x + output.width + self.kernel.radius_x()).min(input.width);
        let bottom = (output.y + output.height + self.kernel.radius_y()).min(input.height);
        Ok(vec![Region::new(
            x,
            y,
            right.saturating_sub(x),
            bottom.saturating_sub(y),
        )])
    }

    fn access_pattern(&self) -> AccessPattern {
        AccessPattern::Spatial
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let input = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`convolve` takes one input tile, got none"))?;
        if input.pixel() != output.pixel() {
            return Err(PixelsError::graph(format!(
                "`convolve` input is {} but output is {}",
                input.pixel(),
                output.pixel()
            )));
        }

        let format = output.pixel();
        let channels = format.channels();
        if channels > MAX_CHANNELS {
            return Err(PixelsError::unsupported(format!(
                "convolve handles up to {MAX_CHANNELS} channels, {format} has {channels}"
            )));
        }
        let region = output.region();
        let source = input.region();
        let sample = format.sample_kind();

        for y in region.y..region.y + region.height {
            let mut row_out: Vec<u8> =
                Vec::with_capacity(region.width as usize * format.bytes_per_pixel());
            for x in region.x..region.x + region.width {
                match sample {
                    SampleKind::U8 => {
                        let mut accumulators = [0_i32; MAX_CHANNELS];
                        self.gather_u8(input, source, x, y, channels, &mut accumulators);
                        for slot in accumulators.iter().take(channels) {
                            let value = (*slot + ONE / 2) >> 14;
                            row_out.push(value.clamp(0, 255) as u8);
                        }
                    }
                    SampleKind::U16 => {
                        let mut accumulators = [0.0_f32; MAX_CHANNELS];
                        self.gather_wide(input, source, (x, y), channels, 2, &mut accumulators);
                        for slot in accumulators.iter().take(channels) {
                            let value = slot.clamp(0.0, 65535.0) + 0.5;
                            row_out.extend_from_slice(&(value as u16).to_ne_bytes());
                        }
                    }
                    SampleKind::F32 => {
                        let mut accumulators = [0.0_f32; MAX_CHANNELS];
                        self.gather_wide(input, source, (x, y), channels, 4, &mut accumulators);
                        for slot in accumulators.iter().take(channels) {
                            row_out.extend_from_slice(&slot.to_ne_bytes());
                        }
                    }
                }
            }
            if let Some(target) = output.row_mut(y) {
                let len = target.len().min(row_out.len());
                if let (Some(to), Some(from)) = (target.get_mut(..len), row_out.get(..len)) {
                    to.copy_from_slice(from);
                }
            }
        }
        Ok(())
    }
}

impl Convolve {
    /// Accumulate the 8-bit neighbourhood of one pixel in fixed point.
    fn gather_u8(
        &self,
        input: &Tile<'_>,
        source: Region,
        x: u32,
        y: u32,
        channels: usize,
        accumulators: &mut [i32; MAX_CHANNELS],
    ) {
        for ky in 0..self.kernel.height {
            let sy = clamp_coordinate(y, ky, self.kernel.radius_y(), source.y, source.height);
            let Some(row) = input.row(sy) else { continue };
            for kx in 0..self.kernel.width {
                let sx = clamp_coordinate(x, kx, self.kernel.radius_x(), source.x, source.width);
                let tap = self
                    .kernel
                    .quantized
                    .get((ky * self.kernel.width + kx) as usize)
                    .copied()
                    .unwrap_or(0);
                let at = (sx - source.x) as usize * channels;
                let Some(pixel) = row.get(at..at + channels) else {
                    continue;
                };
                for (channel, &value) in pixel.iter().enumerate() {
                    if let Some(slot) = accumulators.get_mut(channel) {
                        *slot += i32::from(value) * tap;
                    }
                }
            }
        }
    }

    /// Accumulate a 16-bit or float neighbourhood in `f32`.
    fn gather_wide(
        &self,
        input: &Tile<'_>,
        source: Region,
        at: (u32, u32),
        channels: usize,
        sample_bytes: usize,
        accumulators: &mut [f32; MAX_CHANNELS],
    ) {
        let (x, y) = at;
        for ky in 0..self.kernel.height {
            let sy = clamp_coordinate(y, ky, self.kernel.radius_y(), source.y, source.height);
            let Some(row) = input.row(sy) else { continue };
            for kx in 0..self.kernel.width {
                let sx = clamp_coordinate(x, kx, self.kernel.radius_x(), source.x, source.width);
                let tap = self
                    .kernel
                    .taps
                    .get((ky * self.kernel.width + kx) as usize)
                    .copied()
                    .unwrap_or(0.0);
                let base = (sx - source.x) as usize * channels * sample_bytes;
                for channel in 0..channels {
                    let at = base + channel * sample_bytes;
                    let value = if sample_bytes == 2 {
                        f32::from(u16::from_ne_bytes([
                            row.get(at).copied().unwrap_or(0),
                            row.get(at + 1).copied().unwrap_or(0),
                        ]))
                    } else {
                        let mut bytes = [0_u8; 4];
                        for (slot, offset) in bytes.iter_mut().zip(0..4) {
                            *slot = row.get(at + offset).copied().unwrap_or(0);
                        }
                        f32::from_ne_bytes(bytes)
                    };
                    if let Some(slot) = accumulators.get_mut(channel) {
                        *slot += value * tap;
                    }
                }
            }
        }
    }
}

/// Offset a coordinate by a kernel tap, clamping to the available window.
///
/// Clamping to the *tile* is correct because `input_regions` already clamped
/// the tile to the image: at an image edge the tile stops there too, so
/// clamping to the tile is clamping to the image.
const fn clamp_coordinate(centre: u32, tap: u32, radius: u32, start: u32, len: u32) -> u32 {
    let wanted = centre as i64 + tap as i64 - radius as i64;
    let low = start as i64;
    let high = start as i64 + len as i64 - 1;
    if wanted < low {
        start
    } else if wanted > high {
        (start + len).saturating_sub(1)
    } else {
        wanted as u32
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
    use otf_pixels_core::{PixelFormat, TileBuf};

    fn apply(
        op: &dyn Op,
        input: &ImageDescriptor,
        bytes: &[u8],
    ) -> Result<(ImageDescriptor, Vec<u8>)> {
        let out_desc = op.output_descriptor(std::slice::from_ref(input))?;
        let source = TileBuf::from_vec(input.region(), input.pixel, bytes.to_vec())?;
        let mut target = TileBuf::for_image(&out_desc)?;
        op.compute(&[source.as_tile()?], &mut target.as_tile_mut()?)?;
        Ok((out_desc, target.into_bytes()))
    }

    fn flat(width: u32, height: u32, format: PixelFormat, value: u8) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let bytes = vec![value; descriptor.byte_len().unwrap()];
        (descriptor, bytes)
    }

    #[test]
    fn the_identity_kernel_changes_nothing() {
        let kernel = Kernel::square(3, &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let descriptor = ImageDescriptor::new(9, 7, PixelFormat::Rgb8).unwrap();
        let bytes: Vec<u8> = (0..descriptor.byte_len().unwrap())
            .map(|i| (i * 7 % 251) as u8)
            .collect();
        let (_, out) = apply(&Convolve::new(kernel), &descriptor, &bytes).unwrap();
        assert_eq!(out, bytes, "the identity kernel changed the image");
    }

    #[test]
    fn a_blur_preserves_a_flat_image() {
        // Normalization exists so a blur does not darken. On a flat field any
        // drift is immediately visible as a changed value.
        for size in [3_u32, 5, 7] {
            let kernel = Kernel::blur(size).unwrap();
            let (desc, bytes) = flat(16, 16, PixelFormat::Rgb8, 173);
            let (_, out) = apply(&Convolve::new(kernel), &desc, &bytes).unwrap();
            assert!(
                out.iter().all(|&v| v == 173),
                "{size}x{size} blur changed a flat field: {:?}",
                &out[..6]
            );
        }
    }

    #[test]
    fn a_blur_preserves_flatness_at_the_edges_too() {
        // Clamped edges are what make this true; zero-padding would darken
        // the border, which is the classic convolution bug.
        let kernel = Kernel::blur(5).unwrap();
        let (desc, bytes) = flat(8, 8, PixelFormat::Gray8, 200);
        let (_, out) = apply(&Convolve::new(kernel), &desc, &bytes).unwrap();
        assert!(
            out.iter().all(|&v| v == 200),
            "corners darkened: {:?}",
            &out[..8]
        );
    }

    #[test]
    fn a_gaussian_is_normalized_and_symmetric() {
        let kernel = Kernel::gaussian(1.5).unwrap();
        let sum: f32 = kernel.taps().iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "gaussian sums to {sum}");

        let w = kernel.width() as usize;
        for y in 0..w {
            for x in 0..w {
                let mirrored = kernel.taps()[y * w + (w - 1 - x)];
                assert!(
                    (kernel.taps()[y * w + x] - mirrored).abs() < 1e-6,
                    "gaussian is not symmetric"
                );
            }
        }
    }

    #[test]
    fn a_blur_actually_blurs() {
        // A hard edge must soften. Without this the tests above would all pass
        // for an op that did nothing at all.
        let descriptor = ImageDescriptor::new(8, 1, PixelFormat::Gray8).unwrap();
        let mut bytes = vec![0_u8; 8];
        for slot in bytes.iter_mut().skip(4) {
            *slot = 255;
        }
        let kernel = Kernel::blur(3).unwrap();
        let (_, out) = apply(&Convolve::new(kernel), &descriptor, &bytes).unwrap();
        assert!(
            out[3] > 0 && out[3] < 255,
            "the edge did not soften: {out:?}"
        );
        assert!(
            out[0] == 0,
            "far from the edge should be unchanged: {out:?}"
        );
    }

    #[test]
    fn an_edge_detector_is_not_normalized_away() {
        // A zero-sum kernel would divide by zero if normalized blindly.
        let kernel = Kernel::square(3, &[0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0]).unwrap();
        assert!(
            kernel.taps().iter().all(|t| t.is_finite()),
            "zero-sum kernel produced non-finite taps"
        );
        let (desc, bytes) = flat(8, 8, PixelFormat::Gray8, 128);
        let (_, out) = apply(&Convolve::new(kernel), &desc, &bytes).unwrap();
        // A flat field has no edges, so an edge detector returns zero.
        assert!(out.iter().all(|&v| v == 0), "got {:?}", &out[..8]);
    }

    #[test]
    fn sharpening_overshoot_is_clamped_not_wrapped() {
        let descriptor = ImageDescriptor::new(8, 1, PixelFormat::Gray8).unwrap();
        let mut bytes = vec![10_u8; 8];
        bytes[4] = 250;
        let kernel = Kernel::sharpen(3.0).unwrap();
        let (_, out) = apply(&Convolve::new(kernel), &descriptor, &bytes).unwrap();
        // A strong sharpen around a spike overshoots hard in both directions:
        // the peak saturates and its neighbours undershoot below zero. Both
        // must clamp. Wrapping would turn the bright pixel black and the dark
        // ones white, which is the most alarming possible failure mode.
        assert_eq!(out[4], 255, "the peak did not clamp high: {out:?}");
        assert_eq!(out[3], 0, "the neighbour did not clamp low: {out:?}");
    }

    #[test]
    fn even_and_oversized_kernels_are_rejected() {
        assert!(Kernel::new(2, 3, &[0.0; 6]).is_err(), "even width");
        assert!(Kernel::new(3, 4, &[0.0; 12]).is_err(), "even height");
        assert!(Kernel::new(0, 3, &[]).is_err(), "zero width");
        let huge = Kernel::MAX_SIZE + 2;
        assert!(
            Kernel::new(huge, 3, &vec![0.0; (huge * 3) as usize]).is_err(),
            "oversized"
        );
    }

    #[test]
    fn a_tap_count_mismatch_is_an_error() {
        assert!(Kernel::new(3, 3, &[0.0; 8]).is_err());
        assert!(Kernel::new(3, 3, &[0.0; 10]).is_err());
    }

    #[test]
    fn non_finite_taps_and_sigmas_are_rejected() {
        assert!(Kernel::square(3, &[f32::NAN; 9]).is_err());
        assert!(Kernel::gaussian(0.0).is_err());
        assert!(Kernel::gaussian(-1.0).is_err());
        assert!(Kernel::gaussian(f32::NAN).is_err());
        assert!(Kernel::sharpen(f32::INFINITY).is_err());
    }

    #[test]
    fn demand_is_grown_by_the_radius_and_clamped() {
        let input = ImageDescriptor::new(20, 20, PixelFormat::Gray8).unwrap();
        let op = Convolve::new(Kernel::blur(5).unwrap());

        // Interior: grown by 2 in every direction.
        let demand = op
            .input_regions(Region::new(10, 10, 4, 4), std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(demand[0], Region::new(8, 8, 8, 8));

        // Corner: clamped rather than reaching outside the image.
        let corner = op
            .input_regions(Region::new(0, 0, 4, 4), std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(corner[0], Region::new(0, 0, 6, 6));

        // Every tile of every size must stay inside.
        for y in 0..input.height {
            for x in 0..input.width {
                let r = op
                    .input_regions(Region::new(x, y, 1, 1), std::slice::from_ref(&input))
                    .unwrap()[0];
                assert!(
                    r.x + r.width <= input.width && r.y + r.height <= input.height,
                    "demand {r} leaves the image"
                );
            }
        }
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        let kernel = Kernel::blur(3).unwrap();
        let op = Convolve::new(kernel);
        let descriptor = ImageDescriptor::new(17, 13, PixelFormat::Rgb8).unwrap();
        let bytes: Vec<u8> = (0..descriptor.byte_len().unwrap())
            .map(|i| (i * 31 % 251) as u8)
            .collect();
        let (out_desc, whole) = apply(&op, &descriptor, &bytes).unwrap();
        let source = TileBuf::from_vec(descriptor.region(), descriptor.pixel, bytes).unwrap();

        for (tw, th) in [(4_u32, 4_u32), (1, 13), (17, 1), (5, 3)] {
            let mut target = TileBuf::for_image(&out_desc).unwrap();
            let mut y = 0;
            while y < out_desc.height {
                let h = th.min(out_desc.height - y);
                let mut x = 0;
                while x < out_desc.width {
                    let w = tw.min(out_desc.width - x);
                    let region = Region::new(x, y, w, h);
                    let demand = op
                        .input_regions(region, std::slice::from_ref(&descriptor))
                        .unwrap();
                    let mut cut = TileBuf::zeroed(demand[0], descriptor.pixel).unwrap();
                    otf_pixels_core::copy_region(
                        &source.as_tile().unwrap(),
                        &mut cut.as_tile_mut().unwrap(),
                        demand[0],
                    )
                    .unwrap();
                    let mut sub = TileBuf::zeroed(region, out_desc.pixel).unwrap();
                    op.compute(&[cut.as_tile().unwrap()], &mut sub.as_tile_mut().unwrap())
                        .unwrap();
                    otf_pixels_core::copy_region(
                        &sub.as_tile().unwrap(),
                        &mut target.as_tile_mut().unwrap(),
                        region,
                    )
                    .unwrap();
                    x += w;
                }
                y += h;
            }
            assert_eq!(
                target.bytes(),
                whole.as_slice(),
                "tiling at {tw}x{th} changed the pixels"
            );
        }
    }

    #[test]
    fn wide_formats_convolve_too() {
        for format in [PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::RgbF32] {
            let descriptor = ImageDescriptor::new(8, 8, format).unwrap();
            let bytes = vec![0_u8; descriptor.byte_len().unwrap()];
            let op = Convolve::new(Kernel::blur(3).unwrap());
            let (_, out) = apply(&op, &descriptor, &bytes).unwrap();
            assert_eq!(out.len(), bytes.len(), "{format}");
        }
    }
}
