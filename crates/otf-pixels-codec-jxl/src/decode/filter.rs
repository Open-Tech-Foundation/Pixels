//! Restoration filters (ISO/IEC 18181-1 §J): Gaborish smoothing and the
//! edge-preserving filter.

use otf_pixels_core::{PixelsError, Result};

use super::frame::FramePlanes;
use super::header::FrameHeader;

/// Apply the frame's restoration filters to its colour planes.
///
/// # Errors
///
/// For a filter not implemented.
pub fn restore(planes: &mut FramePlanes, frame: &FrameHeader) -> Result<()> {
    if let Some(weights) = frame.gaborish {
        super::render::gaborish(&mut planes.colour, planes.width, planes.height, weights);
    }
    if frame.epf.is_some() {
        return Err(PixelsError::unsupported("JPEG XL: the edge-preserving filter"));
    }
    Ok(())
}
