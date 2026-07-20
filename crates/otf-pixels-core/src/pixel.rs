//! Pixel formats and the sample types kernels are monomorphized over.
//!
//! The v1 format set is fixed by SPEC §Pixel formats. Alpha is **unassociated
//! (straight)** at every API boundary; ops needing premultiplied alpha convert
//! internally and convert back.
//!
//! # Typing model (ADR-0002)
//!
//! [`PixelFormat`] is a runtime value: the graph and the public API are
//! dynamic, so `Image` carries no type parameter. Kernels are generic over
//! [`Sample`] and are selected **once per tile** by matching on the format —
//! see [`dispatch_sample!`]. One match per tile is noise next to a fully
//! specialized inner loop.
//!
//! [`dispatch_sample!`]: crate::dispatch_sample

use core::fmt;

/// The numeric type of one channel sample.
///
/// Deliberately **exhaustive**, unlike most enums here: kernels match on it to
/// select a monomorphization, so downstream op crates must be able to match it
/// without a wildcard arm — see [`dispatch_sample!`]. A wildcard would silently
/// swallow a new sample type at runtime; an exhaustive match turns adding one
/// into the compile error it deserves to be.
///
/// [`dispatch_sample!`]: crate::dispatch_sample
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleKind {
    /// 8-bit unsigned integer, full range 0..=255.
    U8,
    /// 16-bit unsigned integer, native endian in memory, full range 0..=65535.
    U16,
    /// 32-bit IEEE-754 float, nominal range 0.0..=1.0 (used for filter math).
    F32,
}

impl SampleKind {
    /// Size of one sample in bytes.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }
}

/// The channel layout of a pixel, independent of sample type.
///
/// Exhaustive for the same reason as [`SampleKind`]: kernels dispatch on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Single luminance channel.
    Gray,
    /// Luminance plus unassociated alpha.
    GrayAlpha,
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue, plus unassociated alpha.
    Rgba,
}

impl ChannelLayout {
    /// Number of channels in this layout.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Whether this layout carries an alpha channel.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }
}

/// An interleaved pixel format: a [`ChannelLayout`] over a [`SampleKind`].
///
/// The v1 set is exactly the formats listed in SPEC §Pixel formats. Samples
/// are interleaved (`RGBRGB…`, not planar) and, for `U16`, stored in **native
/// endianness** in memory — codecs convert at their own boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// 8-bit grayscale.
    Gray8,
    /// 16-bit grayscale.
    Gray16,
    /// 8-bit grayscale with alpha.
    GrayA8,
    /// 8-bit RGB.
    Rgb8,
    /// 8-bit RGBA.
    Rgba8,
    /// 16-bit RGB.
    Rgb16,
    /// 16-bit RGBA.
    Rgba16,
    /// 32-bit float RGB.
    RgbF32,
    /// 32-bit float RGBA.
    RgbaF32,
}

impl PixelFormat {
    /// Every pixel format supported in v1.
    pub const ALL: &'static [Self] = &[
        Self::Gray8,
        Self::Gray16,
        Self::GrayA8,
        Self::Rgb8,
        Self::Rgba8,
        Self::Rgb16,
        Self::Rgba16,
        Self::RgbF32,
        Self::RgbaF32,
    ];

    /// The channel layout of this format.
    #[must_use]
    pub const fn layout(self) -> ChannelLayout {
        match self {
            Self::Gray8 | Self::Gray16 => ChannelLayout::Gray,
            Self::GrayA8 => ChannelLayout::GrayAlpha,
            Self::Rgb8 | Self::Rgb16 | Self::RgbF32 => ChannelLayout::Rgb,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 => ChannelLayout::Rgba,
        }
    }

    /// The sample type of this format.
    #[must_use]
    pub const fn sample_kind(self) -> SampleKind {
        match self {
            Self::Gray8 | Self::GrayA8 | Self::Rgb8 | Self::Rgba8 => SampleKind::U8,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 => SampleKind::U16,
            Self::RgbF32 | Self::RgbaF32 => SampleKind::F32,
        }
    }

    /// Number of channels per pixel.
    #[must_use]
    pub const fn channels(self) -> usize {
        self.layout().channels()
    }

    /// Whether this format carries an alpha channel.
    #[must_use]
    pub const fn has_alpha(self) -> bool {
        self.layout().has_alpha()
    }

    /// Size of one pixel in bytes.
    ///
    /// This is exact: v1 formats are byte-aligned and interleaved, so there is
    /// no sub-byte packing to account for.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.sample_kind().size()
    }

    /// A short, stable, lowercase name for this format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gray8 => "gray8",
            Self::Gray16 => "gray16",
            Self::GrayA8 => "graya8",
            Self::Rgb8 => "rgb8",
            Self::Rgba8 => "rgba8",
            Self::Rgb16 => "rgb16",
            Self::Rgba16 => "rgba16",
            Self::RgbF32 => "rgbf32",
            Self::RgbaF32 => "rgbaf32",
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The color model a descriptor's samples are interpreted in.
///
/// v1 is sRGB-assumed (SPEC §Pixel formats); ICC transforms are v2. The enum
/// exists so that the v2 color pipeline is an added variant rather than a
/// breaking descriptor change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorModel {
    /// sRGB primaries and transfer function; the v1 assumption for all input.
    #[default]
    Srgb,
}

/// A channel sample type that kernels can be monomorphized over.
///
/// Implemented for exactly `u8`, `u16` and `f32` — the sample types of the v1
/// [`PixelFormat`] set. This trait is the type-level half of ADR-0002: ops
/// match on a runtime [`PixelFormat`] once per tile, then call a generic
/// kernel that the compiler specializes.
///
/// This trait is sealed: implementing it for other types would let a kernel
/// reinterpret tile bytes as a type no codec produces.
pub trait Sample: sealed::Sealed + Copy + Send + Sync + 'static {
    /// The [`SampleKind`] this type corresponds to.
    const KIND: SampleKind;
    /// The value representing full intensity: opaque alpha, white luminance.
    ///
    /// This is **not** the type's maximum representable value. For `f32` the
    /// nominal range is `0.0..=1.0`, so `FULL_SCALE` is `1.0` while
    /// [`f32::MAX`] is about `3.4e38`.
    ///
    /// It is named `FULL_SCALE` rather than `MAX` precisely because of that
    /// gap: `u8`, `u16` and `f32` all have an *inherent* `MAX` constant, and
    /// inherent constants win name resolution over trait ones. A kernel
    /// generic over `S: Sample` writing `S::MAX` would therefore compile,
    /// resolve to the inherent constant, and be silently wrong for floats.
    const FULL_SCALE: Self;
    /// The value representing zero intensity.
    const ZERO: Self;
}

impl Sample for u8 {
    const KIND: SampleKind = SampleKind::U8;
    const FULL_SCALE: Self = u8::MAX;
    const ZERO: Self = 0;
}

impl Sample for u16 {
    const KIND: SampleKind = SampleKind::U16;
    const FULL_SCALE: Self = u16::MAX;
    const ZERO: Self = 0;
}

impl Sample for f32 {
    const KIND: SampleKind = SampleKind::F32;
    const FULL_SCALE: Self = 1.0;
    const ZERO: Self = 0.0;
}

mod sealed {
    /// Prevents downstream implementations of [`super::Sample`].
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
}

/// Dispatch once on a [`SampleKind`] into a kernel generic over [`Sample`].
///
/// This is the per-tile dispatch of ADR-0002: the match runs once, the body is
/// instantiated once per sample type, and the inner loop is fully specialized.
///
/// The macro binds a type alias (conventionally `S`) inside the body:
///
/// ```
/// use otf_pixels_core::{dispatch_sample, PixelFormat, Sample};
///
/// fn full_scale_as_f64(format: PixelFormat) -> f64 {
///     dispatch_sample!(format.sample_kind(), S => {
///         // `S` is a concrete type here: u8, u16, or f32.
///         fn widen<T: Sample + Into<f64>>(v: T) -> f64 { v.into() }
///         widen(<S as Sample>::FULL_SCALE)
///     })
/// }
///
/// assert_eq!(full_scale_as_f64(PixelFormat::Rgb8), 255.0);
/// assert_eq!(full_scale_as_f64(PixelFormat::Rgb16), 65535.0);
/// assert_eq!(full_scale_as_f64(PixelFormat::RgbF32), 1.0);
/// ```
#[macro_export]
macro_rules! dispatch_sample {
    ($kind:expr, $ty:ident => $body:block) => {
        match $kind {
            $crate::SampleKind::U8 => {
                type $ty = u8;
                $body
            }
            $crate::SampleKind::U16 => {
                type $ty = u16;
                $body
            }
            $crate::SampleKind::F32 => {
                type $ty = f32;
                $body
            }
        }
    };
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_pixel_matches_layout_times_sample_size() {
        for &format in PixelFormat::ALL {
            let expected = format.channels() * format.sample_kind().size();
            assert_eq!(format.bytes_per_pixel(), expected, "{format}");
            assert!(format.bytes_per_pixel() > 0, "{format}");
        }
    }

    #[test]
    fn v1_format_set_has_expected_sizes() {
        assert_eq!(PixelFormat::Gray8.bytes_per_pixel(), 1);
        assert_eq!(PixelFormat::Gray16.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::GrayA8.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgb16.bytes_per_pixel(), 6);
        assert_eq!(PixelFormat::Rgba16.bytes_per_pixel(), 8);
        assert_eq!(PixelFormat::RgbF32.bytes_per_pixel(), 12);
        assert_eq!(PixelFormat::RgbaF32.bytes_per_pixel(), 16);
    }

    #[test]
    fn alpha_layouts_are_reported_consistently() {
        for &format in PixelFormat::ALL {
            assert_eq!(format.has_alpha(), format.layout().has_alpha(), "{format}");
        }
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(PixelFormat::GrayA8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
        assert!(!PixelFormat::Gray8.has_alpha());
    }

    #[test]
    fn format_names_are_unique_and_stable() {
        let mut seen = std::collections::HashSet::new();
        for &format in PixelFormat::ALL {
            assert!(seen.insert(format.as_str()), "duplicate name {format}");
        }
        assert_eq!(PixelFormat::Rgba8.as_str(), "rgba8");
        assert_eq!(
            PixelFormat::ALL.len(),
            9,
            "SPEC §Pixel formats lists 9 v1 formats"
        );
    }

    #[test]
    fn dispatch_selects_the_matching_sample_type() {
        fn sample_size(format: PixelFormat) -> usize {
            dispatch_sample!(format.sample_kind(), S => { size_of::<S>() })
        }
        for &format in PixelFormat::ALL {
            assert_eq!(sample_size(format), format.sample_kind().size(), "{format}");
        }
    }

    #[test]
    fn sample_constants_match_their_kind() {
        assert_eq!(<u8 as Sample>::KIND, SampleKind::U8);
        assert_eq!(<u16 as Sample>::KIND, SampleKind::U16);
        assert_eq!(<f32 as Sample>::KIND, SampleKind::F32);
        assert_eq!(<u8 as Sample>::FULL_SCALE, 255);
        assert_eq!(<u16 as Sample>::FULL_SCALE, 65535);
        assert_eq!(<u8 as Sample>::ZERO, 0);
    }

    #[test]
    fn full_scale_is_the_nominal_range_not_the_type_maximum() {
        // The whole reason the constant is not called `MAX`: for floats the
        // nominal full-intensity value and the type's maximum differ wildly,
        // and an inherent `MAX` would shadow a trait one in generic code.
        assert!((<f32 as Sample>::FULL_SCALE - 1.0).abs() < f32::EPSILON);
        let (full_scale, type_max) = (<f32 as Sample>::FULL_SCALE, f32::MAX);
        assert!(
            full_scale < type_max,
            "{full_scale} should be far below {type_max}"
        );
        // For the integer formats the two coincide, which is what makes the
        // shadowing bug invisible until a float kernel is written.
        assert_eq!(<u8 as Sample>::FULL_SCALE, u8::MAX);
    }
}
