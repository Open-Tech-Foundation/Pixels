//! End-to-end tests for the M1 exit criteria.
//!
//! ROADMAP M1 states the exit condition as:
//!
//! > raw → crop/flip → raw round-trips; graph laziness proven by test.
//!
//! Each criterion gets an explicit test here, at the public API boundary, so
//! that "M1 is done" is a claim the test suite makes rather than one a human
//! asserts. The malformed-input and safety-limit tests cover the guarantees
//! from ARCHITECTURE §Failure model and SPEC §Safety, which apply from the
//! first codec onward and so are M1's responsibility too.

#![cfg(feature = "raw")]
// The M1 criteria are all expressed over the raw codec, which is how they
// reach pixels at all. Without it there is nothing to assert, so the suite
// compiles out rather than failing to build.
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{
    EncodeOptions, ErrorCode, Format, Image, ImageDescriptor, Limits, PixelFormat, PixelsError,
    RawFormat, Region, Result,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pixel values for a `width` × `height` image where byte `n` holds `n as u8`.
fn ramp_bytes(descriptor: &ImageDescriptor) -> Vec<u8> {
    (0..descriptor.byte_len().unwrap())
        .map(|i| i as u8)
        .collect()
}

fn gray(width: u32, height: u32) -> ImageDescriptor {
    ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap()
}

fn raw_out(image: Image) -> Result<Vec<u8>> {
    image.output(Format::Raw, EncodeOptions::default()).bytes()
}

// ---------------------------------------------------------------------------
// Exit criterion 1: raw → crop/flip → raw round-trips.
// ---------------------------------------------------------------------------

#[test]
fn raw_round_trips_unchanged_through_an_empty_pipeline() {
    let descriptor = gray(8, 5);
    let pixels = ramp_bytes(&descriptor);
    let out = raw_out(Image::from_raw(descriptor, pixels.clone()).unwrap()).unwrap();
    assert_eq!(out, pixels, "raw in, raw out, byte for byte");
}

#[test]
fn raw_round_trips_through_crop_and_flip() {
    // 4x4 ramp:  0  1  2  3
    //            4  5  6  7
    //            8  9 10 11
    //           12 13 14 15
    let descriptor = gray(4, 4);
    let image = Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();

    // Window is [[5,6],[9,10]]; flipping it vertically gives [[9,10],[5,6]].
    let out = raw_out(image.crop(1, 1, 2, 2).flip()).unwrap();
    assert_eq!(out, [9, 10, 5, 6]);
}

#[test]
fn raw_round_trips_through_every_v1_pixel_format() {
    for &pixel in PixelFormat::ALL {
        let descriptor = ImageDescriptor::new(4, 3, pixel).unwrap();
        let pixels = ramp_bytes(&descriptor);
        let image = Image::from_raw(descriptor, pixels.clone()).unwrap();
        // flip twice and flop twice: an identity built from real op passes.
        let out = raw_out(image.flip().flip().flop().flop()).unwrap();
        assert_eq!(out, pixels, "{pixel} did not survive the round trip");
    }
}

#[test]
fn a_streaming_raw_source_round_trips_to_a_streaming_sink() {
    // The full streaming path: reader in, writer out, no Vec at either end.
    let descriptor = gray(6, 4);
    let pixels = ramp_bytes(&descriptor);
    let source = std::io::Cursor::new(pixels.clone());
    let image = Image::from_raw_stream(RawFormat::packed(descriptor), source).unwrap();

    let mut sink = Vec::new();
    image
        .output(Format::Raw, EncodeOptions::default())
        .write(&mut sink)
        .unwrap();
    assert_eq!(sink, pixels);
}

#[test]
fn crop_and_flip_compose_in_the_order_written() {
    let descriptor = gray(4, 4);
    let build = || Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();

    // crop-then-flip and flip-then-crop select different pixels, which is what
    // makes this a real test of ordering rather than of commutativity.
    let crop_first = raw_out(build().crop(0, 0, 2, 2).flip()).unwrap();
    assert_eq!(crop_first, [4, 5, 0, 1]);

    let flip_first = raw_out(build().flip().crop(0, 0, 2, 2)).unwrap();
    assert_eq!(flip_first, [12, 13, 8, 9]);
}

#[test]
fn output_dimensions_follow_the_pipeline() {
    let descriptor = gray(10, 8);
    let image = Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    let cropped = image.crop(2, 3, 4, 5);

    let meta = cropped.metadata().unwrap();
    assert_eq!((meta.width, meta.height), (4, 5));
    // The encoded byte count matches the descriptor, so the encoder and the
    // graph agree about the shape.
    assert_eq!(raw_out(cropped).unwrap().len(), 4 * 5);
}

// ---------------------------------------------------------------------------
// Exit criterion 2: graph laziness proven by test.
// ---------------------------------------------------------------------------

/// A source that counts how many bytes have actually been read from it.
#[derive(Debug)]
struct CountingSource {
    data: std::io::Cursor<Vec<u8>>,
    bytes_read: Arc<AtomicUsize>,
}

impl std::io::Read for CountingSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = std::io::Read::read(&mut self.data, buf)?;
        self.bytes_read.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}

#[test]
fn no_source_bytes_are_read_before_a_terminal() {
    // SPEC §Guarantees 3: no source bytes are read before metadata() (header
    // only) or a terminal (pixels). Raw's header is empty, so a correct
    // implementation reads *zero* bytes until the terminal runs.
    let descriptor = gray(16, 16);
    let bytes_read = Arc::new(AtomicUsize::new(0));
    let source = CountingSource {
        data: std::io::Cursor::new(ramp_bytes(&descriptor)),
        bytes_read: Arc::clone(&bytes_read),
    };

    let image = Image::from_raw_stream(RawFormat::packed(descriptor), source).unwrap();
    assert_eq!(
        bytes_read.load(Ordering::Relaxed),
        0,
        "construction read source bytes"
    );

    // Chaining a whole pipeline still reads nothing.
    let pipeline = image.crop(2, 2, 8, 8).flip().flop();
    assert_eq!(
        bytes_read.load(Ordering::Relaxed),
        0,
        "chaining read source bytes"
    );

    // Metadata is answered from the descriptor, not the stream.
    let meta = pipeline.metadata().unwrap();
    assert_eq!((meta.width, meta.height), (8, 8));
    assert_eq!(
        bytes_read.load(Ordering::Relaxed),
        0,
        "metadata read source bytes"
    );

    // Only the terminal pulls.
    let out = raw_out(pipeline).unwrap();
    assert_eq!(out.len(), 64);
    assert!(
        bytes_read.load(Ordering::Relaxed) > 0,
        "the terminal read nothing"
    );
}

#[test]
fn building_a_pipeline_and_dropping_it_reads_nothing() {
    // A pipeline that is never pulled must cost nothing at all.
    let descriptor = gray(32, 32);
    let bytes_read = Arc::new(AtomicUsize::new(0));
    let source = CountingSource {
        data: std::io::Cursor::new(ramp_bytes(&descriptor)),
        bytes_read: Arc::clone(&bytes_read),
    };
    {
        let image = Image::from_raw_stream(RawFormat::packed(descriptor), source).unwrap();
        let _discarded = image
            .crop(0, 0, 4, 4)
            .flip()
            .output(Format::Raw, EncodeOptions::default());
    }
    assert_eq!(bytes_read.load(Ordering::Relaxed), 0);
}

#[test]
fn cloning_a_pipeline_shares_it_rather_than_re_reading() {
    // SPEC: `Image` is Clone (cheap; shares graph nodes). Two clones pulled
    // separately must produce identical output.
    let descriptor = gray(4, 4);
    let image = Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    let pipeline = image.crop(1, 1, 2, 2);
    let first = raw_out(pipeline.clone()).unwrap();
    let second = raw_out(pipeline).unwrap();
    assert_eq!(first, second);
    assert_eq!(first, [5, 6, 9, 10]);
}

// ---------------------------------------------------------------------------
// Failure model: malformed input is a value, never a panic.
// ---------------------------------------------------------------------------

#[test]
fn every_truncation_of_a_raw_stream_is_an_error_not_a_panic() {
    let descriptor = gray(4, 4);
    let full = ramp_bytes(&descriptor);
    for len in 0..full.len() {
        let source = std::io::Cursor::new(full[..len].to_vec());
        let image = Image::from_raw_stream(RawFormat::packed(descriptor), source).unwrap();
        let err = raw_out(image).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed, "truncated to {len} bytes");
    }
    // The untruncated stream still succeeds, so the test is not vacuous.
    let source = std::io::Cursor::new(full.clone());
    let image = Image::from_raw_stream(RawFormat::packed(descriptor), source).unwrap();
    assert_eq!(raw_out(image).unwrap(), full);
}

#[test]
fn a_failing_source_surfaces_as_an_io_error() {
    #[derive(Debug)]
    struct Broken;
    impl std::io::Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("device disappeared"))
        }
    }
    let image = Image::from_raw_stream(RawFormat::packed(gray(4, 4)), Broken).unwrap();
    assert_eq!(raw_out(image).unwrap_err().code(), ErrorCode::Io);
}

#[test]
fn a_failing_sink_surfaces_as_an_io_error() {
    struct Broken;
    impl std::io::Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let descriptor = gray(4, 4);
    let image = Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    let err = image
        .output(Format::Raw, EncodeOptions::default())
        .write(Broken)
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::Io);
}

#[test]
fn invalid_crop_windows_are_errors_at_every_boundary() {
    let descriptor = gray(4, 4);
    let build = || Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();

    let cases: [(u32, u32, u32, u32, &str); 5] = [
        (0, 0, 5, 4, "wider than the image"),
        (0, 0, 4, 5, "taller than the image"),
        (4, 0, 1, 1, "origin past the right edge"),
        (0, 4, 1, 1, "origin past the bottom edge"),
        (3, 3, 2, 2, "extends past the far corner"),
    ];
    for (x, y, w, h, why) in cases {
        let err = raw_out(build().crop(x, y, w, h)).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument, "{why}");
    }

    // A window filling the image exactly is legal.
    assert!(raw_out(build().crop(0, 0, 4, 4)).is_ok());
    // Zero-sized windows are rejected too.
    assert!(raw_out(build().crop(0, 0, 0, 1)).is_err());
    assert!(raw_out(build().crop(0, 0, 1, 0)).is_err());
}

// ---------------------------------------------------------------------------
// SPEC §Safety: limits are enforced before pixel allocation.
// ---------------------------------------------------------------------------

#[test]
fn max_pixels_is_enforced_before_any_allocation() {
    // A caller forwarding untrusted dimensions must get an error, not an
    // attempted 16-exabyte allocation.
    let err =
        RawFormat::from_dimensions(u32::MAX, u32::MAX, PixelFormat::Rgba8, &Limits::default())
            .unwrap_err();
    assert_eq!(err.code(), ErrorCode::LimitExceeded);

    // Just over the default limit fails; just under succeeds.
    let limit = Limits::default().max_pixels;
    assert!(ImageDescriptor::new(1, (limit + 1) as u32, PixelFormat::Gray8).is_err());
    assert!(ImageDescriptor::new(1, limit as u32, PixelFormat::Gray8).is_ok());
}

#[test]
fn limit_errors_report_what_was_asked_for() {
    let limits = Limits::default().with_max_pixels(100);
    let err = RawFormat::from_dimensions(20, 20, PixelFormat::Gray8, &limits).unwrap_err();
    match err {
        PixelsError::LimitExceeded {
            requested, allowed, ..
        } => {
            assert_eq!(requested, 400);
            assert_eq!(allowed, 100);
        }
        other => panic!("expected LimitExceeded, got {other}"),
    }
}

#[test]
fn zero_sized_images_are_rejected() {
    assert!(ImageDescriptor::new(0, 4, PixelFormat::Gray8).is_err());
    assert!(ImageDescriptor::new(4, 0, PixelFormat::Gray8).is_err());
}

// ---------------------------------------------------------------------------
// The M1 evaluator is the oracle M2 is diffed against, so its determinism is
// itself a tested property.
// ---------------------------------------------------------------------------

#[test]
fn evaluation_is_deterministic_across_repeated_runs() {
    // SPEC §Guarantees 2: identical input + pipeline + version yields
    // byte-identical output. This is the baseline M2 must reproduce exactly.
    let descriptor = ImageDescriptor::new(7, 5, PixelFormat::Rgba8).unwrap();
    let pixels = ramp_bytes(&descriptor);
    let run = || {
        raw_out(
            Image::from_raw(descriptor, pixels.clone())
                .unwrap()
                .crop(1, 0, 5, 4)
                .flip()
                .flop(),
        )
        .unwrap()
    };
    let first = run();
    for _ in 0..8 {
        assert_eq!(run(), first, "evaluation is not deterministic");
    }
}

#[test]
fn a_pipeline_is_evaluable_from_multiple_threads() {
    // SPEC: `Image` is Send + Sync. The engine core is synchronous, and hosts
    // integrate by running pipelines on their own worker threads (ADR-0005),
    // so a shared pipeline must be pullable concurrently.
    let descriptor = gray(8, 8);
    let image = Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    let pipeline = image.crop(2, 2, 4, 4).flip();
    let expected = raw_out(pipeline.clone()).unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let pipeline = pipeline.clone();
            std::thread::spawn(move || raw_out(pipeline).unwrap())
        })
        .collect();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[test]
fn regions_and_descriptors_agree_about_the_output() {
    let descriptor = gray(9, 7);
    let image = Image::from_raw(descriptor, ramp_bytes(&descriptor)).unwrap();
    let cropped = image.crop(1, 2, 3, 4);
    let out = cropped.descriptor().unwrap();
    assert_eq!(out.region(), Region::from_size(3, 4));
    assert_eq!(out.byte_len().unwrap(), raw_out(cropped).unwrap().len());
}
