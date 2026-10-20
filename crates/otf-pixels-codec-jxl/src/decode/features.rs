//! Patches, splines and noise (ISO/IEC 18181-1 §G.1.4–§G.1.6, §K.3–§K.5).

use otf_pixels_core::{PixelsError, Result};

use super::header::{FrameHeader, ImageHeader};
use super::reader::BitReader;
use super::render::References;

/// A frame's patches, splines and noise.
#[derive(Debug, Clone, Default)]
pub struct Features {}

impl Features {
    /// Parse whichever of the three the frame's flags declare.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn parse(
        _r: &mut BitReader<'_>,
        _image: &ImageHeader,
        frame: &FrameHeader,
        _references: &References,
    ) -> Result<Self> {
        if frame.flags & (FrameHeader::PATCHES | FrameHeader::SPLINES | FrameHeader::NOISE) != 0 {
            return Err(PixelsError::unsupported("JPEG XL: patches, splines and noise"));
        }
        Ok(Self {})
    }

    /// Draw the features onto a full-size frame.
    ///
    /// # Errors
    ///
    /// For a patch whose reference is missing.
    #[allow(clippy::unused_self, reason = "to be implemented")]
    pub fn apply(
        &self,
        _frame: &mut super::render::Frame,
        _header: &FrameHeader,
        _image: &ImageHeader,
        _references: &References,
    ) -> Result<()> {
        Ok(())
    }
}
