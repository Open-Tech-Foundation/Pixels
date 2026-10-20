//! JPEG XL codec for `otf-pixels`, implemented from scratch.
//!
//! JPEG XL (ISO/IEC 18181) codes an image either as *Modular*, integer
//! channels predicted and entropy coded, which is how it is lossless, or as
//! *VarDCT*, the lossy mode. This crate owns the format outright, checked
//! against libjxl's `djxl`; see ADR-0019.
//!
//! # Scope
//!
//! Encoding only, so far: [`JxlEncoder`] writes lossy VarDCT images (DCT8
//! blocks in XYB) or lossless Modular ones, grey or RGB, at 8 or 16 bits,
//! as a bare codestream; images with alpha are lossless for now. Decoding
//! is to follow.
//!
//! # Memory
//!
//! Internally buffered, as the other encoders that choose their codes from
//! the whole image are: the codes are fitted to every residual before the
//! first is written.

mod bits;
mod encoder;
mod entropy;
mod headers;
mod modular;
mod predict;
mod vardct;

pub use encoder::JxlEncoder;
