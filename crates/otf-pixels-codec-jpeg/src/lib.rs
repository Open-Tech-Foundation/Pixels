//! Baseline JPEG codec for `otf-pixels`, implemented from scratch.
//!
//! "Baseline" here means what the format means by it: 8-bit samples, sequential
//! DCT, Huffman entropy coding. Progressive JPEG is a different enough decoder
//! that ADR-0004 puts it behind a wrapped codec instead; this crate reports it
//! as [`PixelsError::Unsupported`] rather than half-decoding it.
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
mod entropy;
mod format;
mod huffman;
mod idct;

pub use decoder::{JpegCodec, JpegDecoder, probe};
pub use format::{Component, Frame, SIGNATURE, Scan, ScanComponent};
