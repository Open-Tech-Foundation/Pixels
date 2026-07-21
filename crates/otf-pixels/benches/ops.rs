//! Comparative benchmark — the M4 exit criterion.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --bench ops
//! ```
//!
//! # What is being compared, and what is not
//!
//! Against `image` and `fast_image_resize`, on the same machine, the same
//! pixels and the same target sizes. Quoting published figures from three
//! different machines would be close to meaningless, which is why these are
//! dev-dependencies rather than a table copied from a README.
//!
//! The comparison is **not** apples to apples in one direction that matters,
//! and pretending otherwise would be the easy way to flatter ourselves:
//!
//! - `fast_image_resize` is hand-written SIMD with runtime dispatch. We are
//!   autovectorized safe Rust by ADR-0011. It should be faster at resize, and
//!   by roughly the margin that ADR predicted. If it is *much* faster, the
//!   autovectorization assumption needs revisiting — which is the number this
//!   benchmark exists to produce.
//! - Both of them resize a buffer that is already in memory. We run a
//!   demand-driven graph that could equally be reading a 2 GB TIFF through a
//!   pipe. On a single in-memory resize that machinery is pure overhead, and
//!   this benchmark charges us for all of it.
//!
//! So: single-op numbers are the honest worst case for our design, and the
//! pipeline numbers below are where the design is supposed to pay. Both are
//! reported.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "a benchmark binary reports to stdout and operates on known-good values"
)]

use std::time::{Duration, Instant};

use otf_pixels::{
    EncodeOptions, Filter, Format, Image, ImageDescriptor, PixelFormat, ResizeOptions,
};

/// Source image size. Roughly a 12 MP photo, which is the case that matters.
const WIDTH: u32 = 4000;
const HEIGHT: u32 = 3000;

/// Thumbnail target, the overwhelmingly common real workload.
const THUMB_W: u32 = 400;
const THUMB_H: u32 = 300;

/// How long to run each case before reporting.
const BUDGET: Duration = Duration::from_millis(1500);

fn main() {
    let pixels = WIDTH as usize * HEIGHT as usize;
    println!(
        "otf-pixels M4 benchmark — {WIDTH}x{HEIGHT} RGB8 ({:.1} MP), {}",
        pixels as f64 / 1e6,
        std::env::consts::ARCH
    );
    println!(
        "Each case runs for {:?}; the best iteration is reported.\n",
        BUDGET
    );

    let source = make_source();

    resize_comparison(&source);
    filter_sweep(&source);
    pipeline_cases(&source);

    println!(
        "\nNotes: `image` and `fast_image_resize` resize an in-memory buffer; \
         otf-pixels runs a demand-driven graph, so single-op numbers charge us \
         for machinery that only pays off on pipelines and streaming sources."
    );
}

/// A deterministic RGB8 image with structure at several scales.
fn make_source() -> Vec<u8> {
    let mut bytes = vec![0_u8; WIDTH as usize * HEIGHT as usize * 3];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let at = (y * WIDTH as usize + x) * 3;
            let checker = if ((x / 32) + (y / 32)) % 2 == 0 {
                60
            } else {
                0
            };
            bytes[at] = ((x * 255 / WIDTH as usize) as u8).wrapping_add(checker);
            bytes[at + 1] = ((y * 255 / HEIGHT as usize) as u8).wrapping_add(checker);
            bytes[at + 2] = ((x ^ y) % 256) as u8;
        }
    }
    bytes
}

/// Run `body` repeatedly for [`BUDGET`], returning the fastest iteration.
///
/// The minimum, not the mean: the fastest run is the one least perturbed by
/// scheduling noise, and it is the figure that reproduces.
fn measure(mut body: impl FnMut()) -> Duration {
    // One untimed run to fault in pages and warm caches, so the first
    // iteration is not measuring the allocator.
    body();
    let start = Instant::now();
    let mut best = Duration::MAX;
    while start.elapsed() < BUDGET {
        let iteration = Instant::now();
        body();
        best = best.min(iteration.elapsed());
    }
    best
}

/// Print one result line, with throughput and a ratio against a baseline.
fn report(label: &str, elapsed: Duration, baseline: Option<Duration>) {
    let megapixels = (WIDTH as f64 * HEIGHT as f64) / 1e6;
    let mps = megapixels / elapsed.as_secs_f64();
    match baseline {
        Some(base) => {
            let ratio = elapsed.as_secs_f64() / base.as_secs_f64();
            println!(
                "  {label:<34} {:>8.2} ms  {mps:>7.1} MP/s   {ratio:>5.2}x",
                elapsed.as_secs_f64() * 1e3
            );
        }
        None => println!(
            "  {label:<34} {:>8.2} ms  {mps:>7.1} MP/s   {:>5}",
            elapsed.as_secs_f64() * 1e3,
            "1.00x"
        ),
    }
}

/// Resize the same image to the same size, three ways.
fn resize_comparison(source: &[u8]) {
    println!("Lanczos3 downscale {WIDTH}x{HEIGHT} -> {THUMB_W}x{THUMB_H}, RGB8");
    println!(
        "  {:<34} {:>8}  {:>10}   {:>5}",
        "", "time", "throughput", "vs ours"
    );

    let descriptor = ImageDescriptor::new(WIDTH, HEIGHT, PixelFormat::Rgb8).unwrap();

    // The floor: what an identity pipeline costs. Everything above this is
    // moving 36 MB in and out, not filtering, and it is charged to every
    // number below — so it is reported rather than left implicit.
    let floor = measure(|| {
        let out = Image::from_raw(descriptor, source.to_vec())
            .unwrap()
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        std::hint::black_box(out);
    });
    report("otf-pixels (identity, no resize)", floor, None);

    let ours = measure(|| {
        let out = Image::from_raw(descriptor, source.to_vec())
            .unwrap()
            .resize(THUMB_W, THUMB_H)
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        std::hint::black_box(out);
    });
    report("otf-pixels", ours, None);

    // `image`: the general-purpose Rust imaging library, scalar.
    let theirs = measure(|| {
        let buffer: image::RgbImage =
            image::ImageBuffer::from_raw(WIDTH, HEIGHT, source.to_vec()).unwrap();
        let out = image::imageops::resize(
            &buffer,
            THUMB_W,
            THUMB_H,
            image::imageops::FilterType::Lanczos3,
        );
        std::hint::black_box(out);
    });
    report("image", theirs, Some(ours));

    // `fast_image_resize`: hand-written SIMD with runtime dispatch.
    let fir = measure(|| {
        let src = fast_image_resize::images::Image::from_vec_u8(
            WIDTH,
            HEIGHT,
            source.to_vec(),
            fast_image_resize::PixelType::U8x3,
        )
        .unwrap();
        let mut dst = fast_image_resize::images::Image::new(
            THUMB_W,
            THUMB_H,
            fast_image_resize::PixelType::U8x3,
        );
        let mut resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new().resize_alg(
            fast_image_resize::ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
        );
        resizer.resize(&src, &mut dst, &options).unwrap();
        std::hint::black_box(dst);
    });
    report("fast_image_resize", fir, Some(ours));
    println!();
}

/// Our own filters against each other, so the cost of quality is visible.
fn filter_sweep(source: &[u8]) {
    println!("otf-pixels filters, same downscale");
    let descriptor = ImageDescriptor::new(WIDTH, HEIGHT, PixelFormat::Rgb8).unwrap();
    let mut baseline = None;
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
        let elapsed = measure(|| {
            let out = Image::from_raw(descriptor, source.to_vec())
                .unwrap()
                .resize_with(THUMB_W, THUMB_H, options)
                .output(Format::Raw, EncodeOptions::default())
                .bytes()
                .unwrap();
            std::hint::black_box(out);
        });
        if baseline.is_none() {
            baseline = Some(elapsed);
        }
        report(filter.as_str(), elapsed, baseline);
    }
    println!();
}

/// Multi-op pipelines, where a graph engine is supposed to earn its overhead.
fn pipeline_cases(source: &[u8]) {
    println!("otf-pixels pipelines (one pass over the graph, not one pass per op)");
    let descriptor = ImageDescriptor::new(WIDTH, HEIGHT, PixelFormat::Rgb8).unwrap();

    /// A named pipeline shape, applied to whatever source it is given.
    type Case = (&'static str, fn(Image) -> Image);

    let cases: [Case; 4] = [
        ("resize", |i| i.resize(THUMB_W, THUMB_H)),
        ("crop then resize", |i| {
            i.crop(500, 400, 3000, 2200).resize(THUMB_W, THUMB_H)
        }),
        ("resize, rotate, sharpen", |i| {
            i.resize(THUMB_W, THUMB_H).rotate(90).sharpen(0.5)
        }),
        ("resize then blur", |i| i.resize(THUMB_W, THUMB_H).blur(1.5)),
    ];

    let mut baseline = None;
    for (name, build) in cases {
        let elapsed = measure(|| {
            let image = Image::from_raw(descriptor, source.to_vec()).unwrap();
            let out = build(image)
                .output(Format::Raw, EncodeOptions::default())
                .bytes()
                .unwrap();
            std::hint::black_box(out);
        });
        if baseline.is_none() {
            baseline = Some(elapsed);
        }
        report(name, elapsed, baseline);
    }
}
