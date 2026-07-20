//! Regions, image descriptors, and the safety limits checked against them.

use crate::{ColorModel, Limit, PixelFormat, PixelsError, Result};
use core::fmt;

/// An axis-aligned rectangle of pixels, in image coordinates.
///
/// The origin is the top-left corner; `x` grows right and `y` grows down. A
/// region with zero width or height is *empty* and legal to represent — it is
/// what demand propagation produces when an op needs nothing from an input.
///
/// Edge accessors return [`u64`] so that a region touching the far edge of the
/// coordinate space cannot overflow during intersection math.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Region {
    /// Distance from the left edge of the image, in pixels.
    pub x: u32,
    /// Distance from the top edge of the image, in pixels.
    pub y: u32,
    /// Width in pixels; may be zero.
    pub width: u32,
    /// Height in pixels; may be zero.
    pub height: u32,
}

impl Region {
    /// The empty region at the origin.
    pub const EMPTY: Self = Self {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    /// Construct a region from its origin and size.
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// A region covering `width` × `height` pixels at the origin.
    #[must_use]
    pub const fn from_size(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    /// The x coordinate one past the right edge.
    #[must_use]
    pub const fn right(self) -> u64 {
        self.x as u64 + self.width as u64
    }

    /// The y coordinate one past the bottom edge.
    #[must_use]
    pub const fn bottom(self) -> u64 {
        self.y as u64 + self.height as u64
    }

    /// Whether the region contains no pixels.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// The number of pixels in the region.
    #[must_use]
    pub const fn pixel_count(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Whether `other` lies entirely within `self`.
    ///
    /// An empty region is contained by any region, including another empty one.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        if other.is_empty() {
            return true;
        }
        if self.is_empty() {
            return false;
        }
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    /// The overlap between two regions, or [`Region::EMPTY`] if they are
    /// disjoint.
    #[must_use]
    pub fn intersect(self, other: Self) -> Self {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if u64::from(x) >= right || u64::from(y) >= bottom {
            return Self::EMPTY;
        }
        // Both differences are positive and bounded by a u32 edge, so the
        // truncating casts below cannot lose information.
        Self {
            x,
            y,
            width: (right - u64::from(x)) as u32,
            height: (bottom - u64::from(y)) as u32,
        }
    }

    /// Move the region by a signed offset, clamping at the coordinate origin.
    #[must_use]
    pub fn translate(self, dx: i64, dy: i64) -> Self {
        let shift = |v: u32, d: i64| -> u32 {
            let moved = i64::from(v).saturating_add(d);
            moved.clamp(0, i64::from(u32::MAX)) as u32
        };
        Self {
            x: shift(self.x, dx),
            y: shift(self.y, dy),
            ..self
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}+{}+{}", self.width, self.height, self.x, self.y)
    }
}

/// Safety limits applied before any pixel memory is allocated.
///
/// See SPEC §Safety. Limits are checked at header parse time, so a hostile
/// header that claims enormous dimensions is rejected before the engine
/// commits memory to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    /// Maximum total pixels (width × height) in a single image.
    ///
    /// Defaults to [`Limits::DEFAULT_MAX_PIXELS`].
    pub max_pixels: u64,
}

impl Limits {
    /// The default `max_pixels`: 268 megapixels, matching Sharp.
    pub const DEFAULT_MAX_PIXELS: u64 = 268_402_689;

    /// Limits with every check disabled.
    ///
    /// Only appropriate for fully trusted input.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_pixels: u64::MAX,
        }
    }

    /// The default limits with `max_pixels` replaced.
    ///
    /// This exists because [`Limits`] is `#[non_exhaustive]`: downstream crates
    /// cannot write `Limits { max_pixels, .. }`, so without a setter the field
    /// would be readable but unconfigurable outside this crate.
    #[must_use]
    pub const fn with_max_pixels(mut self, max_pixels: u64) -> Self {
        self.max_pixels = max_pixels;
        self
    }

    /// Check a dimension pair against these limits.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::LimitExceeded`] if the pixel count exceeds
    /// [`Limits::max_pixels`], or [`PixelsError::InvalidArgument`] if either
    /// dimension is zero.
    pub fn check(&self, width: u32, height: u32) -> Result<()> {
        if width == 0 {
            return Err(PixelsError::invalid_argument("width", "must be non-zero"));
        }
        if height == 0 {
            return Err(PixelsError::invalid_argument("height", "must be non-zero"));
        }
        let pixels = u64::from(width) * u64::from(height);
        if pixels > self.max_pixels {
            return Err(PixelsError::limit_exceeded(
                Limit::MaxPixels,
                pixels,
                self.max_pixels,
            ));
        }
        Ok(())
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_pixels: Self::DEFAULT_MAX_PIXELS,
        }
    }
}

/// The shape of an image at a point in the graph.
///
/// Descriptors flow **forward at graph-build time**: every node computes its
/// output descriptor from its inputs' descriptors when it is constructed. That
/// is what makes [`metadata()`] free — no pixels are touched to answer it
/// (ARCHITECTURE §Layer 3).
///
/// [`metadata()`]: crate::Image::metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ImageDescriptor {
    /// Width in pixels; always non-zero.
    pub width: u32,
    /// Height in pixels; always non-zero.
    pub height: u32,
    /// Interleaved pixel format of the samples.
    pub pixel: PixelFormat,
    /// Color model the samples are interpreted in (sRGB-assumed in v1).
    pub color: ColorModel,
}

impl ImageDescriptor {
    /// Construct a descriptor, validating it against the default [`Limits`].
    ///
    /// # Errors
    ///
    /// See [`Limits::check`].
    pub fn new(width: u32, height: u32, pixel: PixelFormat) -> Result<Self> {
        Self::with_limits(width, height, pixel, &Limits::default())
    }

    /// Construct a descriptor, validating it against explicit `limits`.
    ///
    /// # Errors
    ///
    /// See [`Limits::check`].
    pub fn with_limits(
        width: u32,
        height: u32,
        pixel: PixelFormat,
        limits: &Limits,
    ) -> Result<Self> {
        limits.check(width, height)?;
        Ok(Self {
            width,
            height,
            pixel,
            color: ColorModel::Srgb,
        })
    }

    /// The region covering the whole image.
    #[must_use]
    pub const fn region(&self) -> Region {
        Region::from_size(self.width, self.height)
    }

    /// Bytes in one densely packed row of this image.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.width as usize * self.pixel.bytes_per_pixel()
    }

    /// Bytes in the whole image when densely packed.
    ///
    /// Returns [`None`] on overflow, which on a 32-bit target is reachable for
    /// large-but-legal dimensions. Callers sizing an allocation must treat
    /// [`None`] as "too large for this platform" rather than unwrapping.
    #[must_use]
    pub fn byte_len(&self) -> Option<usize> {
        (self.height as usize).checked_mul(self.row_bytes())
    }

    /// The same descriptor with different dimensions, revalidated.
    ///
    /// # Errors
    ///
    /// See [`Limits::check`].
    pub fn resized(&self, width: u32, height: u32) -> Result<Self> {
        Limits::default().check(width, height)?;
        Ok(Self {
            width,
            height,
            ..*self
        })
    }
}

impl fmt::Display for ImageDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} {}", self.width, self.height, self.pixel)
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
    use crate::ErrorCode;

    #[test]
    fn intersect_of_overlapping_regions() {
        let a = Region::new(0, 0, 10, 10);
        let b = Region::new(5, 5, 10, 10);
        assert_eq!(a.intersect(b), Region::new(5, 5, 5, 5));
        assert_eq!(
            a.intersect(b),
            b.intersect(a),
            "intersection is commutative"
        );
    }

    #[test]
    fn intersect_of_disjoint_regions_is_empty() {
        let a = Region::new(0, 0, 4, 4);
        let b = Region::new(10, 10, 4, 4);
        assert!(a.intersect(b).is_empty());
        // Touching edges do not overlap.
        assert!(
            Region::new(0, 0, 4, 4)
                .intersect(Region::new(4, 0, 4, 4))
                .is_empty()
        );
    }

    #[test]
    fn intersect_at_the_coordinate_limit_does_not_overflow() {
        let a = Region::new(u32::MAX - 4, 0, 4, 4);
        let b = Region::new(u32::MAX - 2, 0, 2, 4);
        assert_eq!(a.intersect(b), Region::new(u32::MAX - 2, 0, 2, 4));
        assert_eq!(
            Region::new(u32::MAX, u32::MAX, u32::MAX, u32::MAX).right(),
            8_589_934_590
        );
    }

    #[test]
    fn contains_handles_empty_regions() {
        let full = Region::from_size(10, 10);
        assert!(full.contains(Region::new(2, 2, 3, 3)));
        assert!(full.contains(full));
        assert!(!full.contains(Region::new(8, 8, 4, 4)));
        assert!(full.contains(Region::EMPTY), "empty fits anywhere");
        assert!(!Region::EMPTY.contains(full));
        assert!(Region::EMPTY.contains(Region::EMPTY));
    }

    #[test]
    fn translate_clamps_instead_of_wrapping() {
        assert_eq!(
            Region::new(5, 5, 2, 2).translate(-10, -10),
            Region::new(0, 0, 2, 2)
        );
        assert_eq!(
            Region::new(5, 5, 2, 2).translate(3, 4),
            Region::new(8, 9, 2, 2)
        );
        assert_eq!(
            Region::new(0, 0, 2, 2).translate(i64::MAX, i64::MAX).x,
            u32::MAX
        );
    }

    #[test]
    fn max_pixels_is_checked_before_allocation() {
        let limits = Limits { max_pixels: 100 };
        assert!(limits.check(10, 10).is_ok());
        let err = limits.check(10, 11).unwrap_err();
        assert_eq!(err.code(), ErrorCode::LimitExceeded);
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        let limits = Limits::default();
        assert_eq!(
            limits.check(0, 10).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            limits.check(10, 0).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn hostile_dimensions_cannot_overflow_the_limit_check() {
        // u32::MAX * u32::MAX would wrap in 32-bit math; the check is u64.
        let err = Limits::default().check(u32::MAX, u32::MAX).unwrap_err();
        assert_eq!(err.code(), ErrorCode::LimitExceeded);
    }

    #[test]
    fn default_max_pixels_matches_spec() {
        assert_eq!(Limits::default().max_pixels, 268_402_689);
        assert!(Limits::unlimited().check(u32::MAX, u32::MAX).is_ok());
    }

    #[test]
    fn max_pixels_is_configurable_without_a_struct_literal() {
        // `Limits` is non_exhaustive, so downstream crates need this setter to
        // configure the limit at all.
        let limits = Limits::default().with_max_pixels(16);
        assert_eq!(limits.max_pixels, 16);
        assert!(limits.check(4, 4).is_ok());
        assert_eq!(
            limits.check(4, 5).unwrap_err().code(),
            ErrorCode::LimitExceeded
        );
    }

    #[test]
    fn descriptor_reports_packed_sizes() {
        let desc = ImageDescriptor::new(4, 3, PixelFormat::Rgb8).unwrap();
        assert_eq!(desc.row_bytes(), 12);
        assert_eq!(desc.byte_len(), Some(36));
        assert_eq!(desc.region(), Region::from_size(4, 3));
        assert_eq!(desc.color, ColorModel::Srgb);
    }

    #[test]
    fn descriptor_construction_enforces_limits() {
        assert_eq!(
            ImageDescriptor::new(0, 4, PixelFormat::Rgb8)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        let desc = ImageDescriptor::new(4, 4, PixelFormat::Rgb8).unwrap();
        assert_eq!(desc.resized(2, 2).unwrap().width, 2);
        assert!(desc.resized(0, 2).is_err());
    }
}
