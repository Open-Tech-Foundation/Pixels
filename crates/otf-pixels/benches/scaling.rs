//! Scaling benchmark across cores — the third M2 exit criterion.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --bench scaling
//! ```
//!
//! # Why this is not `criterion`
//!
//! Criterion answers "how long does this take, precisely?" for a small
//! operation run many times. The question here is different: how does one
//! fixed unit of work redistribute as workers are added? That means holding
//! the work constant and varying the thread count, reporting speedup and
//! parallel efficiency against the single-threaded baseline. A bespoke harness
//! states that directly, and keeps the dependency tree empty until M4, where
//! comparative numbers against other libraries actually need criterion's rigor.
//!
//! # Reading the output
//!
//! `efficiency` is speedup divided by thread count: 1.00 is perfect scaling,
//! 0.50 means half of each added core is wasted. Expect it to fall off as
//! threads approach core count, and to fall off sooner for pipelines whose
//! per-tile work is small relative to scheduling overhead — that is the real
//! result, not a defect to hide.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "a benchmark binary reports to stdout and operates on known-good values"
)]

use otf_pixels::{
    EncodeOptions, Format, Image, ImageDescriptor, PixelFormat, PlanOptions, RawFormat,
    SchedulerOptions,
};
use std::time::{Duration, Instant};

/// A raw source that generates rows on demand.
///
/// Keeps the fixture out of the measurement: no multi-hundred-megabyte buffer
/// is allocated, and no page-cache effects are being timed.
#[derive(Debug)]
struct SyntheticRows {
    row_bytes: usize,
    rows: u64,
    emitted: u64,
    cursor: usize,
}

impl SyntheticRows {
    fn new(descriptor: &ImageDescriptor) -> Self {
        Self {
            row_bytes: descriptor.row_bytes(),
            rows: u64::from(descriptor.height),
            emitted: 0,
            cursor: 0,
        }
    }
}

impl std::io::Read for SyntheticRows {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut written = 0;
        while written < buf.len() && self.emitted < self.rows {
            buf[written] = (self
                .emitted
                .wrapping_mul(31)
                .wrapping_add(self.cursor as u64)
                % 251) as u8;
            written += 1;
            self.cursor += 1;
            if self.cursor == self.row_bytes {
                self.cursor = 0;
                self.emitted += 1;
            }
        }
        Ok(written)
    }
}

/// A sink that discards, so I/O is not part of the measurement.
struct Discard {
    bytes: u64,
}

impl std::io::Write for Discard {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Where a workload's pixels come from.
///
/// This is the axis that matters most for scaling. A forward-only stream must
/// be decoded serially by construction (ADR-0005), so it caps speedup by
/// Amdahl's law no matter how good the scheduler is. A memory source has no
/// such stage, and isolates what the scheduler itself achieves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    /// Forward-only stream: decode is a serial stage.
    Stream,
    /// Random-access memory buffer: no serial stage.
    Memory,
}

/// One pipeline to measure.
struct Workload {
    name: &'static str,
    descriptor: ImageDescriptor,
    source: SourceKind,
    build: fn(Image) -> Image,
}

/// Run `workload` once on `threads` workers, returning wall time.
fn run_once(workload: &Workload, threads: usize) -> Duration {
    let image = match workload.source {
        SourceKind::Stream => Image::from_raw_stream(
            RawFormat::packed(workload.descriptor),
            SyntheticRows::new(&workload.descriptor),
        )
        .unwrap(),
        SourceKind::Memory => {
            let len = workload.descriptor.byte_len().unwrap();
            Image::from_raw(
                workload.descriptor,
                (0..len).map(|i| (i % 251) as u8).collect(),
            )
            .unwrap()
        }
    };
    let pipeline = (workload.build)(image);
    let options = SchedulerOptions::default()
        .with_threads(threads)
        .with_plan(PlanOptions::default().with_strip_rows(32));

    let mut sink = Discard { bytes: 0 };
    let start = Instant::now();
    pipeline
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options)
        .write(&mut sink)
        .unwrap();
    let elapsed = start.elapsed();
    assert!(sink.bytes > 0, "the pipeline produced nothing");
    elapsed
}

/// Best of `samples` runs, which suppresses scheduler and page-fault noise
/// better than a mean does for wall-clock work of this size.
fn best_of(workload: &Workload, threads: usize, samples: usize) -> Duration {
    (0..samples)
        .map(|_| run_once(workload, threads))
        .min()
        .unwrap_or_default()
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let thread_counts: Vec<usize> = [1, 2, 4, 8, 16]
        .into_iter()
        .filter(|&t| t <= cores.max(1) * 2)
        .collect();

    let workloads = [
        Workload {
            name: "stream + identity (scheduler overhead floor)",
            descriptor: ImageDescriptor::new(1024, 8192, PixelFormat::Rgba8).unwrap(),
            source: SourceKind::Stream,
            build: |image| image,
        },
        Workload {
            name: "stream + flop x4 (serial decode, parallel ops)",
            descriptor: ImageDescriptor::new(1024, 4096, PixelFormat::Rgba8).unwrap(),
            source: SourceKind::Stream,
            build: |image| image.flop().flop().flop().flop(),
        },
        Workload {
            name: "memory + flop x4 (no serial stage)",
            descriptor: ImageDescriptor::new(1024, 4096, PixelFormat::Rgba8).unwrap(),
            source: SourceKind::Memory,
            build: |image| image.flop().flop().flop().flop(),
        },
        Workload {
            name: "memory + crop + flop (light work per tile)",
            descriptor: ImageDescriptor::new(1024, 8192, PixelFormat::Rgba8).unwrap(),
            source: SourceKind::Memory,
            build: |image| image.crop(0, 0, 1000, 8000).flop(),
        },
    ];

    println!("otf-pixels M2 scaling benchmark");
    println!("available parallelism: {cores}");

    for workload in &workloads {
        let pixels = u64::from(workload.descriptor.width) * u64::from(workload.descriptor.height);
        let megapixels = pixels as f64 / 1e6;
        println!("\n{}", workload.name);
        println!(
            "  {}x{} {} ({megapixels:.1} MP)",
            workload.descriptor.width, workload.descriptor.height, workload.descriptor.pixel
        );
        println!(
            "  {:>7}  {:>10}  {:>10}  {:>8}  {:>10}",
            "threads", "time", "MP/s", "speedup", "efficiency"
        );

        let mut baseline = Duration::ZERO;
        for (index, &threads) in thread_counts.iter().enumerate() {
            let elapsed = best_of(workload, threads, 3);
            if index == 0 {
                baseline = elapsed;
            }
            let seconds = elapsed.as_secs_f64();
            let throughput = if seconds > 0.0 {
                megapixels / seconds
            } else {
                f64::NAN
            };
            let speedup = if seconds > 0.0 {
                baseline.as_secs_f64() / seconds
            } else {
                f64::NAN
            };
            let efficiency = speedup / threads as f64;
            println!(
                "  {threads:>7}  {:>9.1?}  {throughput:>10.1}  {speedup:>7.2}x  {efficiency:>9.2}",
                elapsed
            );
        }
    }

    println!(
        "\nefficiency = speedup / threads; 1.00 is perfect scaling.\n\
         \n\
         Two distinct ceilings are visible, and neither is a scheduler defect:\n\
         \n\
         1. Stream workloads are capped by Amdahl's law. A forward-only source\n\
            must be decoded serially (ADR-0005), so that stage does not shrink\n\
            as workers are added. Random-access sources (memory today, tiled\n\
            TIFF in M5) have no such stage and scale substantially better.\n\
         2. Memory workloads flatten past ~4 threads because M1/M2 ops are all\n\
            byte movement -- flop is close to a memcpy -- so they saturate\n\
            memory bandwidth rather than compute. M4's arithmetic kernels\n\
            (resize, convolve) should extend the useful range."
    );
}
