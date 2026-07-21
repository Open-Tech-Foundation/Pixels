//! Container formats and the codec traits every format plugs in behind.
//!
//! Ownership per format — implemented from scratch versus wrapped — is
//! recorded in ADR-0004 and is invisible here on purpose: everything sits
//! behind [`Decoder`] and [`Encoder`], so a format can be rewritten without an
//! API change.

use crate::{ImageDescriptor, PixelFormat, PixelsError, Region, Result, Sink, TileMut};
use core::fmt;

/// A container format.
///
/// Format is **data**, not a type parameter: `output(format, options)` is the
/// single encode terminal, which keeps the API surface stable as formats are
/// added and maps cleanly onto host bindings (ADR-0006). Formats not yet
/// implemented are still nameable, so an unsupported request is a catchable
/// [`PixelsError::Unsupported`] rather than a compile error in a host binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// Uncompressed pixels described entirely by the caller.
    Raw,
    /// Portable Network Graphics.
    Png,
    /// JPEG.
    Jpeg,
    /// Graphics Interchange Format.
    Gif,
    /// Tagged Image File Format.
    Tiff,
    /// WebP.
    WebP,
    /// AV1 Image File Format.
    Avif,
}

impl Format {
    /// A short, stable, lowercase name for this format.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Gif => "gif",
            Self::Tiff => "tiff",
            Self::WebP => "webp",
            Self::Avif => "avif",
        }
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a decoder can produce without a full decode.
///
/// The scheduler uses this to decide whether it may request arbitrary regions
/// from a source or must pull rows in order (ARCHITECTURE §Layer 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeCapability {
    /// Emits rows in top-to-bottom order as bytes arrive.
    ///
    /// Out-of-order region requests are not supported; the engine reads
    /// forward. This covers non-interlaced PNG, baseline JPEG, GIF frames,
    /// strip TIFF and raw.
    Sequential,
    /// Can produce an arbitrary region without decoding the whole image.
    ///
    /// This covers tiled TIFF, and is what makes the giant-image thumbnail
    /// path constant-memory.
    ///
    /// A JPEG decoding at a reduced `M/8` scale is *not* this, though an
    /// earlier version of this comment said it was: scaled decode lowers the
    /// resolution but still emits rows in order, and every coefficient is
    /// still entropy-decoded. It is a decoder configuration, not a capability,
    /// and claiming otherwise would be a lie the scheduler acts on.
    Regions,
}

/// Header-only facts about an image.
///
/// Answering this must not decode pixels (SPEC §Guarantees 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Metadata {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// The container format the image was read from.
    pub format: Format,
    /// The pixel format the decoder will produce.
    pub pixel: PixelFormat,
}

impl Metadata {
    /// Build metadata from a decoded header descriptor and its container.
    #[must_use]
    pub const fn new(desc: &ImageDescriptor, format: Format) -> Self {
        Self {
            width: desc.width,
            height: desc.height,
            format,
            pixel: desc.pixel,
        }
    }
}

/// Encoder tuning shared across formats.
///
/// Fields that a format has no notion of are ignored by that format rather
/// than rejected, so the same options value can be handed to any encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Lossy quality, 1–100. Ignored by lossless formats.
    pub quality: u8,
}

impl EncodeOptions {
    /// The default quality used when none is given.
    pub const DEFAULT_QUALITY: u8 = 80;

    /// Options with the given quality.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] unless `quality` is in 1..=100.
    pub fn with_quality(quality: u8) -> Result<Self> {
        if !(1..=100).contains(&quality) {
            return Err(PixelsError::invalid_argument(
                "quality",
                format!("must be in 1..=100, got {quality}"),
            ));
        }
        Ok(Self { quality })
    }
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quality: Self::DEFAULT_QUALITY,
        }
    }
}

/// Format sniffing.
///
/// Detection is by **magic bytes only** — extensions and MIME types are
/// ignored (SPEC §Formats), because they are attacker-controlled hints rather
/// than facts about the bytes.
pub trait Codec {
    /// The format this codec handles.
    fn format(&self) -> Format;

    /// The number of leading bytes [`Codec::probe`] needs to decide.
    fn magic_len(&self) -> usize;

    /// Whether `prefix` looks like this format.
    ///
    /// `prefix` may be shorter than [`Codec::magic_len`] if the source ended
    /// early; an implementation must then return `false`, never index blindly.
    fn probe(&self, prefix: &[u8]) -> bool;
}

/// Decodes a byte stream into rows of pixels.
///
/// Implementations must return [`PixelsError::Malformed`] for **every** input
/// they cannot parse. Panicking on hostile bytes is a defect, not a caller
/// error (ARCHITECTURE §Failure model); every parser is fuzzed in CI from the
/// first codec onward.
///
/// Dimension limits are enforced at header parse, before any pixel buffer is
/// allocated, so a header claiming enormous dimensions costs nothing.
pub trait Decoder: Send + fmt::Debug {
    /// The shape of the image, known after the header is parsed.
    fn descriptor(&self) -> ImageDescriptor;

    /// What this decoder can produce without a full decode.
    fn capability(&self) -> DecodeCapability {
        DecodeCapability::Sequential
    }

    /// Decode the next row, top to bottom, into `out`.
    ///
    /// `out` is exactly [`ImageDescriptor::row_bytes`] long. Each call advances
    /// the cursor by one row; the caller reads `descriptor().height` rows.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] on invalid or truncated input,
    /// [`PixelsError::Io`] on source failure, or
    /// [`PixelsError::InvalidArgument`] if `out` is the wrong length or every
    /// row has already been read.
    fn read_row(&mut self, out: &mut [u8]) -> Result<()>;

    /// Decode an arbitrary region into `out`.
    ///
    /// Only meaningful when [`Decoder::capability`] is
    /// [`DecodeCapability::Regions`]; the default implementation reports that
    /// this decoder is sequential.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] for sequential decoders, and
    /// otherwise as [`Decoder::read_row`].
    fn read_region(&mut self, region: Region, out: &mut TileMut<'_>) -> Result<()> {
        let _ = (region, out);
        Err(PixelsError::unsupported(
            "this decoder is sequential; region decode requires DecodeCapability::Regions",
        ))
    }

    /// What this decoder would produce if asked for `target` or larger, when
    /// the format lets it reach that size more cheaply than a full decode.
    ///
    /// JPEG is the motivating case — the low-frequency corner of a DCT block
    /// is a smaller version of that block, so 1/8, 1/4 and 1/2 come almost
    /// free — but nothing here is JPEG-specific: a pyramidal TIFF or a WebP
    /// with scaled decode fits the same shape.
    ///
    /// **Pure**: nothing is committed. The planner asks before it knows
    /// whether the reduction is legal for the pipeline as a whole.
    ///
    /// The returned descriptor is never smaller than `target` in either axis.
    fn reduced_descriptor(&self, target: (u32, u32)) -> Option<ImageDescriptor> {
        let _ = target;
        None
    }

    /// Commit to producing `descriptor`, which
    /// [`Decoder::reduced_descriptor`] must have returned.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] if this decoder has one
    /// resolution, or [`PixelsError::InvalidArgument`] if any row has already
    /// been read — the resolution is fixed from the first row onward.
    fn reduce_to(&mut self, descriptor: ImageDescriptor) -> Result<()> {
        let _ = descriptor;
        Err(PixelsError::unsupported(
            "this decoder has only one resolution",
        ))
    }
}

/// Encodes rows of pixels into a byte stream.
///
/// The sink is passed to each call rather than held by the encoder, which
/// keeps [`Encoder`] object-safe and lets the caller retain ownership of the
/// destination. Encoders write incrementally as rows arrive; they must not
/// buffer the whole image unless the format leaves no choice (ADR-0005).
pub trait Encoder: Send {
    /// Begin an image, writing any container header.
    ///
    /// Must be called exactly once, before any [`Encoder::write_row`].
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] on sink failure, or
    /// [`PixelsError::Unsupported`] if the encoder cannot represent `desc`
    /// (for example an alpha channel in a format without one).
    fn write_header(&mut self, desc: &ImageDescriptor, sink: &mut dyn Sink) -> Result<()>;

    /// Write the next row, top to bottom.
    ///
    /// `row` is exactly [`ImageDescriptor::row_bytes`] long.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] on sink failure, or
    /// [`PixelsError::InvalidArgument`] if `row` is the wrong length or more
    /// rows are written than the header declared.
    fn write_row(&mut self, row: &[u8], sink: &mut dyn Sink) -> Result<()>;

    /// Finish the image, writing any trailer and flushing the sink.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] on sink failure, or
    /// [`PixelsError::Malformed`] if fewer rows were written than the header
    /// declared — a partial image is never silently emitted
    /// (ARCHITECTURE §Failure model).
    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()>;
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
    fn quality_is_validated_at_the_boundary() {
        assert_eq!(EncodeOptions::with_quality(1).unwrap().quality, 1);
        assert_eq!(EncodeOptions::with_quality(100).unwrap().quality, 100);
        assert_eq!(
            EncodeOptions::with_quality(0).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            EncodeOptions::with_quality(101).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(EncodeOptions::default().quality, 80);
    }

    #[test]
    fn format_names_are_stable_and_unique() {
        let all = [
            Format::Raw,
            Format::Png,
            Format::Jpeg,
            Format::Gif,
            Format::Tiff,
            Format::WebP,
            Format::Avif,
        ];
        let mut seen = std::collections::HashSet::new();
        for f in all {
            assert!(seen.insert(f.as_str()), "duplicate name {f}");
        }
        assert_eq!(Format::WebP.as_str(), "webp");
    }

    #[test]
    fn metadata_mirrors_the_descriptor() {
        let desc = ImageDescriptor::new(7, 5, PixelFormat::Rgba8).unwrap();
        let meta = Metadata::new(&desc, Format::Png);
        assert_eq!((meta.width, meta.height), (7, 5));
        assert_eq!(meta.pixel, PixelFormat::Rgba8);
        assert_eq!(meta.format, Format::Png);
    }

    #[test]
    fn sequential_decoders_decline_region_decode() {
        #[derive(Debug)]
        struct Seq;
        impl Decoder for Seq {
            fn descriptor(&self) -> ImageDescriptor {
                ImageDescriptor::new(1, 1, PixelFormat::Gray8).unwrap_or(ImageDescriptor {
                    width: 1,
                    height: 1,
                    pixel: PixelFormat::Gray8,
                    color: crate::ColorModel::Srgb,
                })
            }
            fn read_row(&mut self, _: &mut [u8]) -> Result<()> {
                Ok(())
            }
        }
        let mut buf = crate::TileBuf::zeroed(Region::from_size(1, 1), PixelFormat::Gray8).unwrap();
        let mut tile = buf.as_tile_mut().unwrap();
        let err = Seq
            .read_region(Region::from_size(1, 1), &mut tile)
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::Unsupported);
        assert_eq!(Seq.capability(), DecodeCapability::Sequential);
    }
}
