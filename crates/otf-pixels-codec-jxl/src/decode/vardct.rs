//! VarDCT decoding (ISO/IEC 18181-1 §C.4.4, §C.6, §I).

use otf_pixels_core::{PixelsError, Result};

use super::frame::FramePlanes;
use super::header::{FrameHeader, ImageHeader};
use super::modular::Tree;
use super::reader::BitReader;

/// What the restoration filters need from a VarDCT frame.
#[derive(Debug, Clone)]
pub struct FilterInfo {}

/// LfGlobal's VarDCT part.
#[derive(Debug, Clone)]
pub struct Global {}

impl Global {
    /// # Errors
    ///
    /// Always, for now.
    pub fn parse(_r: &mut BitReader<'_>) -> Result<Self> {
        Err(PixelsError::unsupported("JPEG XL: VarDCT"))
    }
}

/// A VarDCT frame being decoded.
#[derive(Debug)]
pub struct State {}

#[allow(clippy::unused_self, clippy::needless_pass_by_ref_mut, reason = "to be implemented")]
impl State {
    #[must_use]
    pub fn new(_frame: &FrameHeader, _global: &Global, _lf_dequant: [f32; 3], _image: &ImageHeader) -> Self {
        Self {}
    }

    pub fn decode_lf_group(&mut self, _r: &mut BitReader<'_>, _g: &Global, _t: Option<&Tree>, _i: usize, _pos: (usize, usize)) -> Result<()> {
        Ok(())
    }

    pub fn decode_hf_metadata(&mut self, _r: &mut BitReader<'_>, _g: &Global, _t: Option<&Tree>, _i: usize, _pos: (usize, usize)) -> Result<()> {
        Ok(())
    }

    pub fn use_lf_frame(&mut self, _lf: &FramePlanes) -> Result<()> {
        Ok(())
    }

    pub fn decode_hf_global(&mut self, _r: &mut BitReader<'_>, _t: Option<&Tree>) -> Result<()> {
        Ok(())
    }

    pub fn decode_pass_group(&mut self, _r: &mut BitReader<'_>, _pass: usize, _g: usize, _pos: (usize, usize)) -> Result<()> {
        Ok(())
    }

    pub fn finish(self) -> Result<([Vec<f32>; 3], FilterInfo)> {
        Err(PixelsError::unsupported("JPEG XL: VarDCT"))
    }
}
