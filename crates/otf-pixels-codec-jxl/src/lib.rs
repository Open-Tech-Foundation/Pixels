//! JPEG XL codec for `otf-pixels`, implemented from scratch.
//!
//! JPEG XL (ISO/IEC 18181) codes an image either as *Modular*, integer
//! channels predicted and entropy coded, which is how it is lossless, or as
//! *VarDCT*, the lossy mode. This crate owns the format outright, checked
//! against libjxl's `djxl`; see ADR-0019.
//!
//! # Scope
//!
//! [`JxlDecoder`] reads bare codestreams and the ISOBMFF container, with
//! every coding tool: Modular (all predictors, MA trees, RCT, palette and
//! squeeze), VarDCT (every block type, chroma from luma, LF frames,
//! progressive passes, JPEG recompression with chroma subsampling), ANS and
//! prefix codes with LZ77, patches, splines, noise, Gaborish, the
//! edge-preserving filter, upsampling, extra channels and blending. Rows
//! are the first displayed frame; an animation's frames and timing are
//! reported. XYB images are delivered as sRGB (or their declared transfer
//! function), so their ICC profile, if any, is not reported; other images
//! report theirs. Lossless files decode exactly to libjxl's pixels, lossy
//! ones within a level (`tests/reference.rs`).
//!
//! [`JxlEncoder`] writes lossy VarDCT images (DCT8 blocks in XYB) or
//! lossless Modular ones, grey or RGB, at 8 or 16 bits, as a bare
//! codestream; images with alpha are lossless for now.
//!
//! # Memory
//!
//! Internally buffered both ways. The encoder fits its codes to every
//! residual before the first is written; the decoder holds the file and
//! decodes it whole on the first row, as JPEG XL's groups may come in any
//! order and frames blend onto a canvas.

mod bits;
mod decode;
mod decoder;
mod encoder;
mod entropy;
mod headers;
mod modular;
mod predict;
mod vardct;

pub use decoder::{JxlCodec, JxlDecoder, probe};
pub use encoder::JxlEncoder;
