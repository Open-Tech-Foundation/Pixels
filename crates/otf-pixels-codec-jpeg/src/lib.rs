//! JPEG codec for `otf-pixels`, implemented from scratch.
//!
//! Baseline JPEG — 8-bit samples, sequential DCT, Huffman coding — is decoded
//! and encoded. Progressive JPEG is decoded too (ADR-0018): its scans fill a
//! coefficient buffer that is then turned into pixels by the same path.
//! Arithmetic coding, lossless, 12-bit and CMYK report
//! [`PixelsError::Unsupported`].
//!
//! [`PixelsError::Unsupported`]: otf_pixels_core::PixelsError::Unsupported
//!
//! # Memory
//!
//! Decode is streaming, at one MCU row. A JPEG's entropy stream is a single
//! run of bits with no per-row structure, but it is *ordered*: every block of
//! MCU row `n` precedes every block of row `n + 1`. So the decoder keeps one
//! band of component planes — 8 or 16 pixel rows — converts it to interleaved
//! output, serves those rows, and reuses the band. Nothing scales with image
//! height, which is what makes a thumbnail of a 24 MP photograph affordable.
//!
//! # Safety
//!
//! Every parser here reads attacker-controlled bytes. Malformed input is a
//! value, never a panic: `unsafe_code = "forbid"` and the workspace's ban on
//! `unwrap`/`expect`/`panic!` mean the classic decoder failures — a Huffman
//! table indexing past its symbols, an MCU count overflowing, a block written
//! outside its plane — are unrepresentable rather than merely avoided.

mod decoder;
mod encoder;
mod entropy;
mod fdct;
mod format;
mod huffman;
mod idct;
mod progressive;
mod progressive_encode;
mod tables;

pub use decoder::{JpegCodec, JpegDecoder, probe};
pub use encoder::{JpegEncoder, Subsampling};
pub use format::{Component, Frame, SIGNATURE, Scan, ScanComponent};
pub use idct::Scale;
