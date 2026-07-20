//! End-to-end tests for the M2 exit criteria.
//!
//! ROADMAP M2 states them as:
//!
//! > pipelines produce byte-identical output vs M1 evaluator; constant-memory
//! > test on a synthetic huge raw source; scaling benchmark across cores.
//!
//! The scaling benchmark lives in `benches/scaling.rs`, since it measures
//! rather than asserts. The other two are here.
//!
//! The differential test is the important one: the M1 evaluator is
//! deliberately naive and therefore obviously correct, so any disagreement is
//! a scheduler bug by definition.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{
    EncodeOptions, Format, Image, ImageDescriptor, PixelFormat, PlanOptions, RawFormat, Result,
    SchedulerOptions,
};

fn gray(width: u32, height: u32) -> ImageDescriptor {
    ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap()
}

fn ramp_bytes(descriptor: &ImageDescriptor) -> Vec<u8> {
    (0..descriptor.byte_len().unwrap())
        .map(|i| (i % 251) as u8)
        .collect()
}

/// Scheduler options with explicit tiling, so tests can sweep them.
fn options(threads: usize, strip_rows: u32) -> SchedulerOptions {
    SchedulerOptions::default().with_threads(threads).with_plan(
        PlanOptions::default()
            .with_strip_rows(strip_rows)
            .with_square_size(16),
    )
}

// ---------------------------------------------------------------------------
// Exit criterion 1: byte-identical output versus the M1 evaluator.
// ---------------------------------------------------------------------------

/// Every pipeline shape M1 can express, as (name, builder).
fn pipelines(descriptor: ImageDescriptor) -> Vec<(&'static str, Image)> {
    let build = || Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    vec![
        ("identity", build()),
        ("crop", build().crop(1, 2, 8, 9)),
        ("flip", build().flip()),
        ("flop", build().flop()),
        ("flip+flop", build().flip().flop()),
        ("crop+flip", build().crop(2, 3, 7, 11).flip()),
        ("flip+crop", build().flip().crop(2, 3, 7, 11)),
        ("flop+crop+flip", build().flop().crop(0, 1, 12, 13).flip()),
        ("double flip", build().flip().flip()),
        ("crop to one pixel", build().crop(5, 5, 1, 1)),
        ("crop full width", build().crop(0, 4, descriptor.width, 6)),
    ]
}

#[test]
fn every_pipeline_matches_the_reference_evaluator() {
    let descriptor = gray(16, 24);
    for (name, image) in pipelines(descriptor) {
        let expected = image
            .clone()
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference();
        let actual = image.output(Format::Raw, EncodeOptions::default()).bytes();
        assert_eq!(
            actual.unwrap(),
            expected.unwrap(),
            "pipeline `{name}` diverged"
        );
    }
}

#[test]
fn agreement_holds_across_thread_counts_and_tile_sizes() {
    // The scheduler must not depend on how it is tuned. Sweeping both dials
    // over every pipeline is what makes this a real differential test rather
    // than a single lucky configuration.
    let descriptor = gray(19, 29);
    for (name, image) in pipelines(descriptor) {
        let expected = image
            .clone()
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference()
            .unwrap();
        for threads in [1, 2, 4, 8] {
            for strip_rows in [1, 2, 7, 64, 10_000] {
                let actual = image
                    .clone()
                    .output(Format::Raw, EncodeOptions::default())
                    .scheduler_options(options(threads, strip_rows))
                    .bytes()
                    .unwrap();
                assert_eq!(
                    actual, expected,
                    "`{name}` diverged at threads={threads} strip_rows={strip_rows}"
                );
            }
        }
    }
}

#[test]
fn agreement_holds_for_every_pixel_format() {
    for &pixel in PixelFormat::ALL {
        let descriptor = ImageDescriptor::new(9, 13, pixel).unwrap();
        let build = || Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
        let expected = build()
            .crop(1, 1, 7, 9)
            .flip()
            .flop()
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference()
            .unwrap();
        let actual = build()
            .crop(1, 1, 7, 9)
            .flip()
            .flop()
            .output(Format::Raw, EncodeOptions::default())
            .scheduler_options(options(4, 3))
            .bytes()
            .unwrap();
        assert_eq!(actual, expected, "{pixel} diverged");
    }
}

#[test]
fn agreement_holds_over_a_streaming_source() {
    // A forward-only source is where ordering mistakes would show up: the
    // scheduler must pull it in order even while running tiles in parallel.
    let descriptor = gray(32, 97);
    let pixels = ramp_bytes(&descriptor);
    let build = || {
        Image::from_raw_stream(
            RawFormat::packed(descriptor),
            std::io::Cursor::new(pixels.clone()),
        )
        .unwrap()
    };
    let expected = build()
        .crop(4, 4, 20, 60)
        .output(Format::Raw, EncodeOptions::default())
        .bytes_via_reference()
        .unwrap();
    for threads in [1, 4, 8] {
        let actual = build()
            .crop(4, 4, 20, 60)
            .output(Format::Raw, EncodeOptions::default())
            .scheduler_options(options(threads, 8))
            .bytes()
            .unwrap();
        assert_eq!(actual, expected, "threads={threads}");
    }
}

#[test]
fn reversal_over_a_streaming_source_matches_the_reference() {
    // ADR-0009's materialization path, end to end and byte-exact.
    let descriptor = gray(16, 64);
    let pixels = ramp_bytes(&descriptor);
    let build = || {
        Image::from_raw_stream(
            RawFormat::packed(descriptor),
            std::io::Cursor::new(pixels.clone()),
        )
        .unwrap()
    };
    let expected = build()
        .flip()
        .output(Format::Raw, EncodeOptions::default())
        .bytes_via_reference()
        .unwrap();
    let actual = build()
        .flip()
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(4, 8))
        .bytes()
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn repeated_runs_are_byte_identical() {
    // SPEC §Guarantees 2: determinism, now under real concurrency.
    let descriptor = gray(24, 40);
    let build = || Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    let first = build()
        .crop(2, 2, 20, 30)
        .flip()
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(8, 4))
        .bytes()
        .unwrap();
    for run in 0..25 {
        let again = build()
            .crop(2, 2, 20, 30)
            .flip()
            .output(Format::Raw, EncodeOptions::default())
            .scheduler_options(options(8, 4))
            .bytes()
            .unwrap();
        assert_eq!(again, first, "run {run} differed");
    }
}

// ---------------------------------------------------------------------------
// Exit criterion 2: constant memory on a synthetic huge raw source.
// ---------------------------------------------------------------------------

/// A raw source that generates rows on demand rather than storing them.
///
/// This is how a "huge" image is tested without needing the memory to hold
/// one: if the engine ever buffers the whole thing, the test's own memory
/// blows up rather than the fixture's.
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

    /// Byte `i` of row `r`, the value the pipeline is expected to carry.
    fn byte_at(row: u64, index: usize) -> u8 {
        (row.wrapping_mul(31).wrapping_add(index as u64) % 251) as u8
    }
}

impl std::io::Read for SyntheticRows {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let mut written = 0;
        while written < buf.len() {
            if self.emitted >= self.rows {
                break;
            }
            let byte = Self::byte_at(self.emitted, self.cursor);
            buf[written] = byte;
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

/// Resident set size in bytes, on platforms that report it.
#[cfg(target_os = "linux")]
fn resident_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}

#[cfg(not(target_os = "linux"))]
fn resident_bytes() -> Option<u64> {
    None
}

/// A sink that verifies and discards, so the *test* never holds the image.
struct Checksum {
    bytes: u64,
    sum: u64,
}

impl std::io::Write for Checksum {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for byte in buf {
            self.sum = self.sum.wrapping_mul(31).wrapping_add(u64::from(*byte));
        }
        self.bytes += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn a_huge_streaming_source_runs_in_constant_memory() {
    // 512 x 200_000 Gray8 is ~98 MB of pixels. Nothing may hold it: not the
    // fixture (rows are generated), not the sink (bytes are checksummed), and
    // above all not the engine (SPEC §Guarantees 1).
    let descriptor = gray(512, 200_000);
    let total_bytes = descriptor.byte_len().unwrap() as u64;
    assert!(
        total_bytes > 90_000_000,
        "fixture is not large enough to be meaningful"
    );

    let before = resident_bytes();
    let image = Image::from_raw_stream(
        RawFormat::packed(descriptor),
        SyntheticRows::new(&descriptor),
    )
    .unwrap();

    let mut sink = Checksum { bytes: 0, sum: 0 };
    image
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(4, 64))
        .write(&mut sink)
        .unwrap();

    assert_eq!(sink.bytes, total_bytes, "not every pixel reached the sink");

    if let (Some(before), Some(after)) = (before, resident_bytes()) {
        let growth = after.saturating_sub(before);
        assert!(
            growth < total_bytes / 4,
            "resident memory grew by {growth} bytes processing a {total_bytes} byte image; \
             the pipeline is not streaming"
        );
    }
}

#[test]
fn memory_does_not_scale_with_image_height() {
    // The sharper statement: ten times the rows, same peak memory. Measuring
    // growth relative to a baseline run cancels out fixed startup cost.
    let measure = |height: u32| -> u64 {
        let descriptor = gray(256, height);
        let image = Image::from_raw_stream(
            RawFormat::packed(descriptor),
            SyntheticRows::new(&descriptor),
        )
        .unwrap();
        let before = resident_bytes().unwrap_or(0);
        let mut sink = Checksum { bytes: 0, sum: 0 };
        image
            .output(Format::Raw, EncodeOptions::default())
            .scheduler_options(options(2, 32))
            .write(&mut sink)
            .unwrap();
        resident_bytes().unwrap_or(0).saturating_sub(before)
    };

    // Warm up so allocator growth is not attributed to the measured runs.
    let _ = measure(10_000);
    let small = measure(10_000);
    let large = measure(100_000);
    if resident_bytes().is_some() {
        // 10x the rows must not cost anything like 10x the memory.
        assert!(
            large < small.max(1_000_000) * 3,
            "memory scaled with height: {small} bytes for 10k rows, {large} for 100k"
        );
    }
}

#[test]
fn a_huge_source_still_produces_correct_pixels() {
    // Constant memory is worthless if the pixels are wrong. Crop a window out
    // of a tall synthetic image and check every byte against the generator.
    let descriptor = gray(64, 50_000);
    let image = Image::from_raw_stream(
        RawFormat::packed(descriptor),
        SyntheticRows::new(&descriptor),
    )
    .unwrap();
    let out = image
        .crop(8, 49_000, 16, 4)
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(4, 64))
        .bytes()
        .unwrap();

    let mut expected = Vec::new();
    for row in 49_000_u64..49_004 {
        for x in 8_usize..24 {
            expected.push(SyntheticRows::byte_at(row, x));
        }
    }
    assert_eq!(out, expected, "a cropped window of a huge image is wrong");
}

// ---------------------------------------------------------------------------
// Streaming behaviour of the pipeline as a whole.
// ---------------------------------------------------------------------------

#[test]
fn output_reaches_the_sink_progressively() {
    // A streaming engine must hand bytes over as it goes, not at the end.
    // The sink records how much had arrived by the time it first saw data.
    struct Progressive {
        chunks: Vec<usize>,
    }
    impl std::io::Write for Progressive {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.chunks.push(buf.len());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let descriptor = gray(64, 1000);
    let image = Image::from_raw_stream(
        RawFormat::packed(descriptor),
        SyntheticRows::new(&descriptor),
    )
    .unwrap();
    let mut sink = Progressive { chunks: Vec::new() };
    image
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(2, 16))
        .write(&mut sink)
        .unwrap();

    assert_eq!(
        sink.chunks.len(),
        1000,
        "rows should be written one at a time"
    );
    assert!(
        sink.chunks.iter().all(|&n| n == 64),
        "each write should be one row"
    );
}

#[test]
fn laziness_still_holds_with_the_scheduler() {
    // SPEC §Guarantees 3, re-checked now that the scheduler drives execution.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct Counting {
        inner: SyntheticRows,
        reads: Arc<AtomicUsize>,
    }
    impl std::io::Read for Counting {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = std::io::Read::read(&mut self.inner, buf)?;
            self.reads.fetch_add(n, Ordering::Relaxed);
            Ok(n)
        }
    }

    let descriptor = gray(32, 500);
    let reads = Arc::new(AtomicUsize::new(0));
    let source = Counting {
        inner: SyntheticRows::new(&descriptor),
        reads: Arc::clone(&reads),
    };
    let image = Image::from_raw_stream(RawFormat::packed(descriptor), source).unwrap();

    let pipeline = image.crop(0, 0, 16, 100).flop();
    assert_eq!(
        reads.load(Ordering::Relaxed),
        0,
        "chaining read source bytes"
    );
    assert_eq!(pipeline.metadata().unwrap().width, 16);
    assert_eq!(
        reads.load(Ordering::Relaxed),
        0,
        "metadata read source bytes"
    );

    let out = pipeline
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(2, 16))
        .bytes()
        .unwrap();
    assert_eq!(out.len(), 16 * 100);
    let consumed = reads.load(Ordering::Relaxed);
    assert!(consumed > 0, "the terminal read nothing");
    // Only the first 100 rows are needed, so the rest must not be read.
    assert!(
        consumed < descriptor.byte_len().unwrap(),
        "read {consumed} bytes for a 100-row crop of a 500-row image"
    );
}

#[test]
fn errors_from_a_huge_source_stay_errors() -> Result<()> {
    // Truncation partway through a large stream is still a value, not a crash.
    let descriptor = gray(64, 10_000);
    let truncated = vec![0_u8; 64 * 500];
    let image = Image::from_raw_stream(
        RawFormat::packed(descriptor),
        std::io::Cursor::new(truncated),
    )?;
    let err = image
        .output(Format::Raw, EncodeOptions::default())
        .scheduler_options(options(4, 32))
        .bytes()
        .unwrap_err();
    assert_eq!(err.code(), otf_pixels::ErrorCode::Malformed);
    Ok(())
}
