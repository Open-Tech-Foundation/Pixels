//! Core engine for `otf-pixels`: the op graph, tiles, codec traits, and the
//! M1 reference evaluator.
//!
//! Most users want the [`otf-pixels`] facade instead — this crate is the
//! substrate it and the op/codec crates are built on. Depend on it directly
//! when implementing a codec or an op.
//!
//! [`otf-pixels`]: https://docs.rs/otf-pixels
//!
//! # The model
//!
//! An [`Image`] is a handle onto a node of an immutable lazy DAG. Chaining ops
//! builds graph structure and computes descriptors; it reads no pixels. Pixels
//! move only when a terminal pulls them, at which point demand propagates
//! *backwards* through [`Op::input_regions`] and pixels flow *forwards* through
//! [`Op::compute`] (ADR-0001).
//!
//! ```
//! use otf_pixels_core::{Format, Image, ImageDescriptor, PixelFormat, TileBuf, evaluate};
//! use otf_pixels_core::{BufferSource, Producer, Region};
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), otf_pixels_core::PixelsError> {
//! let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Gray8)?;
//! let pixels = TileBuf::from_vec(descriptor.region(), PixelFormat::Gray8, vec![1, 2, 3, 4])?;
//! let source = BufferSource::new(descriptor, Arc::new(pixels))?;
//!
//! // Construction and chaining do no pixel work.
//! let image = Image::from_producer(Arc::new(source), Format::Raw);
//! assert_eq!(image.metadata()?.width, 2);
//!
//! // A terminal pulls.
//! assert_eq!(evaluate(&image)?.bytes(), &[1, 2, 3, 4]);
//! # Ok(())
//! # }
//! ```
//!
//! # Errors never panic
//!
//! Every fallible path returns [`PixelsError`]. Malformed input is a value, not
//! a panic — this crate forbids `unsafe` and denies `unwrap`/`expect`/`panic!`
//! outside tests, because a hostile image must not be able to take down a
//! process embedding the engine (ARCHITECTURE §Failure model).
//!
//! # Concurrency
//!
//! The core is synchronous (ADR-0005). [`Image`] is `Send + Sync` and cheap to
//! clone, so an async host integrates by running pipelines on its own worker
//! threads and meeting the engine at the [`Source`]/[`Sink`] boundary.

mod cache;
mod codec;
mod error;
mod eval;
mod geometry;
mod graph;
mod io;
mod op;
mod pixel;
mod plan;
mod pool;
mod schedule;
mod source;
mod tile;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use cache::{CacheStats, TileCache, TileKey};
pub use codec::{Codec, DecodeCapability, Decoder, EncodeOptions, Encoder, Format, Metadata};
pub use error::{ErrorCode, Limit, PixelsError, Result};
pub use eval::{demand, evaluate, evaluate_rows};
pub use geometry::{ImageDescriptor, Limits, Region};
pub use graph::{Image, Node, NodeId};
pub use io::{Sink, Source};
pub use op::{AccessPattern, Op, Producer};
pub use pixel::{ChannelLayout, ColorModel, PixelFormat, Sample, SampleKind};
pub use plan::{NodePlan, Plan, PlanOptions, TileShape};
pub use pool::ThreadPool;
pub use schedule::{RunStats, Scheduler, SchedulerOptions, evaluate_tiled};
pub use source::{BufferSource, DecodedSource};
pub use tile::{Tile, TileBuf, TileMut, copy_region};
