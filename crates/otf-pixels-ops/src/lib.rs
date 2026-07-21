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
//! # Scope
//!
//! The full v1 op set from SPEC §Core ops is here: [`Resize`], [`Crop`],
//! [`Rotate`], [`Flip`], [`Flop`], [`Modulate`], [`Convolve`], [`Composite`],
//! [`ExtractChannel`] and [`Flatten`].
//!
//! # Two kinds of op
//!
//! **Layout** ops — crop, flip, flop, rotate — move whole pixels without
//! inspecting them. They copy opaque byte runs, so one implementation is
//! correct for every pixel format including ones added later, and they do not
//! use `dispatch_sample!`: there is no arithmetic to specialize.
//!
//! **Arithmetic** ops — resize, modulate, convolve, composite, flatten — read
//! sample values, so they dispatch once per tile on the sample type and run a
//! monomorphized inner loop (ADR-0002). Per ADR-0011 those loops are written
//! to autovectorize rather than to call intrinsics, and 8-bit paths use `i32`
//! fixed point so that vectorization cannot change the result.
//!
//! # Tiling is not observable
//!
//! Every op here produces the same pixels whether it runs in one tile or in
//! many. That is not automatic — a resize whose weight tables were built per
//! tile would resample at the tile's scale — so each op with a non-trivial
//! demand mapping asserts it directly.
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

mod composite;
mod convolve;
mod filter;
mod geometry;
mod pointwise;
mod resample;
mod resize;
mod rotate;

pub use composite::{Blend, Composite};
pub use convolve::{Convolve, Kernel};
pub use filter::{Filter, Run, Weights};
pub use geometry::{Crop, Flip, Flop};
pub use pointwise::{ExtractChannel, Flatten, Modulate};
pub use resize::{Fit, Resize, ResizeOptions};
pub use rotate::{Quarter, Rotate};
