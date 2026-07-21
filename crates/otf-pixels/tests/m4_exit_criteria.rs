//! End-to-end tests for the M4 exit criteria.
//!
//! ROADMAP M4 states them as:
//!
//! > benchmark vs `image` + `fast_image_resize`; publish numbers.
//!
//! plus, from the milestone body, a scalar/SIMD exact-equality gate. The
//! benchmark lives in `benches/ops.rs`, because it measures rather than
//! asserts, and its numbers are published in the README.
//!
//! The exact-equality gate needs restating, because ADR-0011 changed what it
//! can mean. There is no separate scalar path to compare against: kernels are
//! written in one vectorizable form, and 8-bit arithmetic is fixed-point
//! precisely so that whatever the compiler does to that form cannot change the
//! result. What is left to assert — and what this suite asserts — is the
//! property the gate existed to protect:
//!
//! - the same input gives byte-identical output, run to run and across
//!   optimisation of the same binary
//! - output does not depend on thread count, tile shape, or how the scheduler
//!   chose to cut the image up
//! - the scheduler agrees with the M1 whole-image evaluator, which is naive
//!   enough to be obviously correct
//!
//! A vectorization *regression* is a performance fact, not a correctness one,
//! and is visible in the benchmark rather than here — ADR-0011 names that
//! asymmetry rather than pretending it away.

#![cfg(feature = "raw")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{
    Blend, EncodeOptions, Filter, Fit, Format, Image, ImageDescriptor, Kernel, Modulate,
    PixelFormat, PlanOptions, ResizeOptions, Result, SchedulerOptions,
};

/// A deterministic image with structure at several scales, so a resize bug
/// that only affects one frequency still shows up.
fn source(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
    let descriptor = ImageDescriptor::new(width, height, format).unwrap();
    let len = descriptor.byte_len().unwrap();
    let row_bytes = descriptor.row_bytes();
    let bytes = (0..len)
        .map(|i| {
            let x = (i % row_bytes) as u32;
            let y = (i / row_bytes) as u32;
            // A gradient, a checker and a fine stripe superimposed.
            let gradient = (x * 255 / row_bytes.max(1) as u32) as u8;
            let checker = if ((x / 8) + (y / 8)) % 2 == 0 { 40 } else { 0 };
            let stripe = if x % 3 == 0 { 25 } else { 0 };
            gradient.wrapping_add(checker).wrapping_add(stripe)
        })
        .collect();
    (descriptor, bytes)
}

/// Run a pipeline to raw bytes.
fn run(image: Image) -> Result<Vec<u8>> {
    image.output(Format::Raw, EncodeOptions::default()).bytes()
}

/// Every pipeline shape worth checking, as a name and a builder.
type Pipeline = (&'static str, fn(Image) -> Image);

const PIPELINES: [Pipeline; 8] = [
    ("resize down", |i| i.resize(97, 61)),
    ("resize up", |i| i.resize(301, 233)),
    ("thumbnail", |i| i.thumbnail(64, 64)),
    ("rotate 90", |i| i.rotate(90)),
    ("blur", |i| i.blur(1.5)),
    ("sharpen", |i| i.sharpen(0.8)),
    ("resize then rotate", |i| i.resize(80, 60).rotate(270)),
    ("crop, resize, flip", |i| {
        i.crop(10, 10, 100, 80).resize(50, 40).flip()
    }),
];

#[test]
fn every_pipeline_is_byte_identical_run_to_run() {
    // SPEC §Guarantees 2, the version that survives ADR-0011: the arithmetic
    // is deterministic, so repeated runs cannot differ.
    let (descriptor, bytes) = source(160, 120, PixelFormat::Rgb8);
    for (name, build) in PIPELINES {
        let first = run(build(Image::from_raw(descriptor, bytes.clone()).unwrap()))
            .unwrap_or_else(|e| panic!("`{name}`: {e}"));
        for attempt in 0..4 {
            let again = run(build(Image::from_raw(descriptor, bytes.clone()).unwrap())).unwrap();
            assert_eq!(again, first, "`{name}` differed on attempt {attempt}");
        }
    }
}

#[test]
fn every_pipeline_is_independent_of_thread_count() {
    // Parallel evaluation must not be observable in the output. A kernel that
    // accumulated across tiles, or a weight table built per worker, would fail
    // here and nowhere else.
    let (descriptor, bytes) = source(200, 150, PixelFormat::Rgba8);
    for (name, build) in PIPELINES {
        let single = build(Image::from_raw(descriptor, bytes.clone()).unwrap())
            .output(Format::Raw, EncodeOptions::default())
            .threads(1)
            .bytes()
            .unwrap_or_else(|e| panic!("`{name}` on 1 thread: {e}"));

        for threads in [2, 4, 8] {
            let many = build(Image::from_raw(descriptor, bytes.clone()).unwrap())
                .output(Format::Raw, EncodeOptions::default())
                .threads(threads)
                .bytes()
                .unwrap_or_else(|e| panic!("`{name}` on {threads} threads: {e}"));
            assert_eq!(many, single, "`{name}` differed on {threads} threads");
        }
    }
}

#[test]
fn every_pipeline_is_independent_of_tile_shape() {
    // The scheduler is free to negotiate tile shapes (ADR-0003). That freedom
    // must not reach the pixels, which is exactly what a per-tile filter table
    // would break.
    let (descriptor, bytes) = source(180, 140, PixelFormat::Rgb8);
    for (name, build) in PIPELINES {
        let reference = run(build(Image::from_raw(descriptor, bytes.clone()).unwrap())).unwrap();

        for tile in [16_u32, 32, 64, 256] {
            // Both dials, since a pipeline can negotiate either shape.
            let plan = PlanOptions::default()
                .with_square_size(tile)
                .with_strip_rows(tile);
            let options = SchedulerOptions::default().with_plan(plan);
            let actual = build(Image::from_raw(descriptor, bytes.clone()).unwrap())
                .output(Format::Raw, EncodeOptions::default())
                .scheduler_options(options)
                .bytes()
                .unwrap_or_else(|e| panic!("`{name}` at tile {tile}: {e}"));
            assert_eq!(actual, reference, "`{name}` differed at tile size {tile}");
        }
    }
}

#[test]
fn the_scheduler_agrees_with_the_reference_evaluator() {
    // The M1 whole-image evaluator is naive and therefore obviously correct.
    // Any disagreement is a scheduler or demand-mapping bug by definition —
    // which for M4 means an `input_regions` that does not match what the
    // kernel actually reads.
    let (descriptor, bytes) = source(150, 110, PixelFormat::Rgb8);
    for (name, build) in PIPELINES {
        let scheduled = run(build(Image::from_raw(descriptor, bytes.clone()).unwrap()))
            .unwrap_or_else(|e| panic!("`{name}` scheduled: {e}"));

        let reference = build(Image::from_raw(descriptor, bytes.clone()).unwrap())
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference()
            .unwrap_or_else(|e| panic!("`{name}` via reference: {e}"));

        assert_eq!(scheduled, reference, "`{name}` disagreed with the oracle");
    }
}

#[test]
fn every_filter_produces_the_declared_size() {
    let (descriptor, bytes) = source(128, 96, PixelFormat::Rgb8);
    for filter in [
        Filter::Nearest,
        Filter::Box,
        Filter::Bilinear,
        Filter::CatmullRom,
        Filter::Mitchell,
        Filter::Lanczos2,
        Filter::Lanczos3,
    ] {
        let options = ResizeOptions::default().with_filter(filter);
        let image = Image::from_raw(descriptor, bytes.clone())
            .unwrap()
            .resize_with(53, 37, options);
        let meta = image.descriptor().unwrap();
        assert_eq!((meta.width, meta.height), (53, 37), "{}", filter.as_str());

        let out = run(image).unwrap_or_else(|e| panic!("{}: {e}", filter.as_str()));
        assert_eq!(out.len(), meta.byte_len().unwrap(), "{}", filter.as_str());
    }
}

#[test]
fn metadata_is_free_of_pixel_work() {
    // SPEC §Guarantees 3, extended to the ops that change shape: a resize must
    // report its output size without resampling anything.
    let (descriptor, bytes) = source(4000, 3000, PixelFormat::Rgb8);
    let image = Image::from_raw(descriptor, bytes)
        .unwrap()
        .resize(200, 150)
        .rotate(90)
        .blur(2.0);
    let meta = image.descriptor().unwrap();
    // Rotated after resize, so the shape is transposed.
    assert_eq!((meta.width, meta.height), (150, 200));
}

#[test]
fn a_thumbnail_preserves_aspect_and_refuses_to_enlarge() {
    let (wide, bytes) = source(400, 100, PixelFormat::Rgb8);
    let meta = Image::from_raw(wide, bytes)
        .unwrap()
        .thumbnail(200, 200)
        .descriptor()
        .unwrap();
    assert_eq!((meta.width, meta.height), (200, 50), "aspect not preserved");

    let (small, bytes) = source(30, 20, PixelFormat::Rgb8);
    let meta = Image::from_raw(small, bytes)
        .unwrap()
        .thumbnail(200, 200)
        .descriptor()
        .unwrap();
    assert_eq!(
        (meta.width, meta.height),
        (30, 20),
        "small image was enlarged"
    );
}

#[test]
fn compositing_two_lazy_branches_evaluates_both() {
    // The two-input join: both branches stay lazy and neither runs until the
    // terminal pulls. What matters is that the result is correct, not that it
    // is deferred, so this checks the pixels.
    let base = ImageDescriptor::new(20, 20, PixelFormat::Rgba8).unwrap();
    let base_bytes = [0_u8, 0, 0, 255].repeat(400);
    let overlay = ImageDescriptor::new(10, 10, PixelFormat::Rgba8).unwrap();
    let overlay_bytes = [255_u8, 0, 0, 255].repeat(100);

    let out = Image::from_raw(base, base_bytes).unwrap().composite(
        Image::from_raw(overlay, overlay_bytes).unwrap(),
        5,
        5,
    );
    let bytes = run(out).unwrap();

    let pixel = |x: usize, y: usize| -> &[u8] { &bytes[(y * 20 + x) * 4..(y * 20 + x) * 4 + 4] };
    assert_eq!(pixel(0, 0), [0, 0, 0, 255], "outside the overlay");
    assert_eq!(pixel(7, 7), [255, 0, 0, 255], "inside the overlay");
    assert_eq!(pixel(19, 19), [0, 0, 0, 255], "past the overlay");
}

#[test]
fn a_composite_of_two_branches_over_a_pipeline_still_agrees_with_the_oracle() {
    // The hardest demand case in the op set: a two-input op whose second input
    // is itself the output of a shape-changing op.
    let base = ImageDescriptor::new(64, 64, PixelFormat::Rgba8).unwrap();
    let base_bytes: Vec<u8> = (0..base.byte_len().unwrap())
        .map(|i| (i % 251) as u8)
        .collect();
    let overlay = ImageDescriptor::new(80, 80, PixelFormat::Rgba8).unwrap();
    let overlay_bytes: Vec<u8> = (0..overlay.byte_len().unwrap())
        .map(|i| ((i * 7) % 241) as u8)
        .collect();

    let build = || {
        Image::from_raw(base, base_bytes.clone())
            .unwrap()
            .composite_with(
                Image::from_raw(overlay, overlay_bytes.clone())
                    .unwrap()
                    .resize(32, 32),
                8,
                8,
                Blend::Over,
            )
    };

    let scheduled = run(build()).unwrap();
    let reference = build()
        .output(Format::Raw, EncodeOptions::default())
        .bytes_via_reference()
        .unwrap();
    assert_eq!(scheduled, reference, "composite over a resize disagreed");
}

#[test]
fn ops_that_reject_their_arguments_surface_at_the_terminal() {
    // Errors raised mid-chain are captured and reported once, at the terminal,
    // rather than panicking where they occur.
    let (descriptor, bytes) = source(32, 32, PixelFormat::Rgb8);

    type Case = (&'static str, Box<dyn Fn(Image) -> Image>);

    let cases: [Case; 4] = [
        ("resize to zero", Box::new(|i: Image| i.resize(0, 10))),
        ("rotate 45", Box::new(|i: Image| i.rotate(45))),
        (
            "blur with a negative sigma",
            Box::new(|i: Image| i.blur(-1.0)),
        ),
        (
            "extract a channel that does not exist",
            Box::new(|i: Image| i.extract_channel(9)),
        ),
    ];

    for (name, build) in cases {
        let result = run(build(Image::from_raw(descriptor, bytes.clone()).unwrap()));
        assert!(result.is_err(), "`{name}` should have failed");
    }
}

#[test]
fn a_modulation_round_trips_through_the_pipeline() {
    // Brightness is invertible up to rounding, which is a cheap end-to-end
    // check that the op is wired into the graph correctly rather than skipped.
    let (descriptor, bytes) = source(40, 30, PixelFormat::Rgb8);
    let dimmed = run(Image::from_raw(descriptor, bytes.clone())
        .unwrap()
        .modulate(Modulate::identity().with_brightness(0.5).unwrap()))
    .unwrap();
    assert_ne!(dimmed, bytes, "modulate did nothing");

    let identity = run(Image::from_raw(descriptor, bytes.clone())
        .unwrap()
        .modulate(Modulate::identity()))
    .unwrap();
    assert_eq!(identity, bytes, "the identity modulation changed pixels");
}

#[test]
fn a_convolution_over_a_pipeline_agrees_with_the_oracle() {
    // Convolve's demand grows the region and clamps it; a mismatch between
    // what it asks for and what it reads shows up only under tiling.
    let (descriptor, bytes) = source(96, 72, PixelFormat::Rgb8);
    for kernel in [
        Kernel::blur(3).unwrap(),
        Kernel::blur(7).unwrap(),
        Kernel::gaussian(2.0).unwrap(),
        Kernel::sharpen(1.0).unwrap(),
    ] {
        let build = || {
            Image::from_raw(descriptor, bytes.clone())
                .unwrap()
                .convolve(kernel.clone())
        };
        let scheduled = run(build()).unwrap();
        let reference = build()
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference()
            .unwrap();
        assert_eq!(
            scheduled,
            reference,
            "a {}x{} kernel disagreed with the oracle",
            kernel.width(),
            kernel.height()
        );
    }
}

#[test]
fn resize_handles_the_degenerate_shapes() {
    // One-pixel images, single rows and single columns are where an
    // off-by-one in the weight tables becomes a panic rather than a blur.
    for (w, h, tw, th) in [
        (1_u32, 1_u32, 1_u32, 1_u32),
        (1, 1, 32, 32),
        (100, 1, 10, 1),
        (1, 100, 1, 10),
        (3, 3, 1, 1),
        (2, 2, 999, 999),
    ] {
        let (descriptor, bytes) = source(w, h, PixelFormat::Rgb8);
        for filter in [Filter::Nearest, Filter::Box, Filter::Lanczos3] {
            let options = ResizeOptions::default().with_filter(filter);
            let out = run(Image::from_raw(descriptor, bytes.clone())
                .unwrap()
                .resize_with(tw, th, options))
            .unwrap_or_else(|e| panic!("{w}x{h} -> {tw}x{th} {}: {e}", filter.as_str()));
            let expected = ImageDescriptor::new(tw, th, PixelFormat::Rgb8)
                .unwrap()
                .byte_len()
                .unwrap();
            assert_eq!(out.len(), expected, "{w}x{h} -> {tw}x{th}");
        }
    }
}

#[test]
fn fit_inside_never_exceeds_the_box() {
    // The invariant the mode is named for. Rounding a scaled dimension up
    // would break it by a pixel, which is exactly the kind of thing a layout
    // depending on it would trip over.
    for (w, h) in [(1000_u32, 3_u32), (3, 1000), (999, 998), (7, 13)] {
        let descriptor = ImageDescriptor::new(w, h, PixelFormat::Gray8).unwrap();
        let bytes = vec![0_u8; descriptor.byte_len().unwrap()];
        let options = ResizeOptions::default().with_fit(Fit::Inside);
        let meta = Image::from_raw(descriptor, bytes)
            .unwrap()
            .resize_with(100, 100, options)
            .descriptor()
            .unwrap();
        assert!(
            meta.width <= 100 && meta.height <= 100,
            "{w}x{h} fitted to {}x{}, which leaves the box",
            meta.width,
            meta.height
        );
    }
}
