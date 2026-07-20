//! Operation kernels for `otf-pixels`.
//!
//! Each op implements [`Op`] and is chained onto an
//! [`Image`] to build graph structure. Ops do no work when chained — they
//! declare their output shape and their demand, and run only when a terminal
//! pulls pixels through them.
//!
//! [`Op`]: otf_pixels_core::Op
//! [`Image`]: otf_pixels_core::Image
//!
//! # M1 scope
//!
//! M1 ships the geometry ops needed by the round-trip exit criterion:
//! [`Crop`], [`Flip`] and [`Flop`]. The full v1 op set from SPEC §Core ops —
//! resize, rotate, modulate, convolve, composite, channel ops — lands in M4,
//! together with their SIMD paths.
//!
//! # Writing an op
//!
//! An op declares four things (ARCHITECTURE §Layer 3):
//!
//! - `output_descriptor` — its output shape, computed at graph-build time.
//! - `input_regions` — the inverse mapping demand propagation walks backwards.
//! - `access_pattern` — `Sequential` or `Spatial`, driving tile negotiation
//!   (ADR-0003).
//! - `compute` — the kernel itself.
//!
//! The first three are what make an op work correctly under M2's scheduler
//! rather than only under M1's whole-image evaluator, so they must be right
//! even while the evaluator only ever asks for whole images.

mod geometry;

pub use geometry::{Crop, Flip, Flop};
