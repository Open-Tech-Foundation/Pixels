//! Giant tiled TIFF to thumbnail — the M5 exit criterion.
//!
//! Run with:
//!
//! ```text
//! cargo bench -p otf-pixels --bench thumbnail
//! ```
//!
//! # What is compared
//!
//! libvips, on the same machine and the same file. libvips is the reference
//! for this workload — it is the engine whose demand-driven design ADR-0001
//! follows — so it is the number worth being measured against.
//!
//! The comparison runs `vips thumbnail`, which means it needs `libvips-tools`
//! on `PATH`. If that is absent the benchmark says so and reports our own
//! figures alone, rather than silently omitting the row or inventing one.
//!
//! # What is being measured
//!
//! Thumbnailing a tiled TIFF far larger than the thumbnail. This is the
//! workload the whole engine was built for: the scheduler asks for regions,
//! the decoder answers them from individual tiles, and the full-resolution
//! image never exists. A design that materialized the image would show up here
//! as time proportional to the source rather than to the output.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::print_stdout,
    reason = "a benchmark binary reports to stdout and operates on known-good values"
)]

use std::process::Command;
use std::time::{Duration, Instant};

use otf_pixels::{
    EncodeOptions, Format, Image, ImageDescriptor, PixelFormat, TiffEncoder, TiffLayout,
};
use otf_pixels_core::Encoder;

/// Source size. Large enough that a full decode is obviously the wrong plan.
const WIDTH: u32 = 8192;
const HEIGHT: u32 = 6144;
/// Tile edge, the usual choice for a tiled scan.
const TILE: u32 = 256;
/// Thumbnail target.
const THUMB: u32 = 256;
/// How long to run each case.
const BUDGET: Duration = Duration::from_millis(2000);

fn main() {
    let megapixels = (WIDTH as f64 * HEIGHT as f64) / 1e6;
    println!(
        "otf-pixels M5 benchmark — {WIDTH}x{HEIGHT} tiled TIFF ({megapixels:.1} MP, \
         {TILE}x{TILE} tiles) → {THUMB}px thumbnail"
    );

    let path = std::env::temp_dir().join(format!("otf-pixels-m5-bench-{}.tif", std::process::id()));
    let bytes = build_fixture();
    std::fs::write(&path, &bytes).unwrap();
    println!(
        "Fixture: {:.1} MB on disk, {:.1} MB as a decoded raster.\n",
        bytes.len() as f64 / 1e6,
        (WIDTH as f64 * HEIGHT as f64 * 3.0) / 1e6
    );

    let ours = measure(|| {
        let out = Image::open(&path)
            .unwrap()
            .thumbnail(THUMB, THUMB)
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        std::hint::black_box(out);
    });
    report("otf-pixels", ours, None);

    match vips_thumbnail(&path) {
        Some(elapsed) => report("libvips", elapsed, Some(ours)),
        None => println!(
            "  {:<28} skipped — `vips` is not on PATH.\n  {:<28} Install libvips-tools to \
             produce this number; CI does.",
            "libvips", ""
        ),
    }

    println!(
        "\nBoth read the same tiled file. The point of the comparison is that \
         neither should\nmaterialize the {:.0} MB raster to produce a {THUMB}px \
         thumbnail — time proportional to\nthe source rather than the output is \
         what failure looks like.",
        (WIDTH as f64 * HEIGHT as f64 * 3.0) / 1e6
    );

    std::fs::remove_file(&path).ok();
}

/// Build the tiled TIFF fixture with our own encoder.
fn build_fixture() -> Vec<u8> {
    let descriptor = ImageDescriptor::new(WIDTH, HEIGHT, PixelFormat::Rgb8).unwrap();
    let mut encoder = TiffEncoder::new()
        .with_layout(TiffLayout::Tiles {
            width: TILE,
            height: TILE,
        })
        .unwrap();
    let mut out: Vec<u8> = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();

    let mut row = vec![0_u8; descriptor.row_bytes()];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = x as usize * 3;
            // Structure at several scales, so a resize bug that only affects
            // one frequency is still visible if anyone looks at the output.
            let checker = if ((x / 64) + (y / 64)) % 2 == 0 {
                60
            } else {
                0
            };
            row[at] = ((x * 255 / WIDTH) as u8).wrapping_add(checker);
            row[at + 1] = ((y * 255 / HEIGHT) as u8).wrapping_add(checker);
            row[at + 2] = ((x ^ y) % 256) as u8;
        }
        encoder.write_row(&row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

/// Time `vips thumbnail`, or `None` if the tool is not installed.
fn vips_thumbnail(path: &std::path::Path) -> Option<Duration> {
    let output = std::env::temp_dir().join("otf-pixels-m5-vips.v");
    // A probe run first: it also confirms the tool works before it is timed.
    let probe = Command::new("vips")
        .arg("thumbnail")
        .arg(path)
        .arg(&output)
        .arg(THUMB.to_string())
        .status();
    match probe {
        Ok(status) if status.success() => {}
        _ => return None,
    }

    let start = Instant::now();
    let mut best = Duration::MAX;
    while start.elapsed() < BUDGET {
        let iteration = Instant::now();
        let status = Command::new("vips")
            .arg("thumbnail")
            .arg(path)
            .arg(&output)
            .arg(THUMB.to_string())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        best = best.min(iteration.elapsed());
    }
    std::fs::remove_file(&output).ok();
    Some(best)
}

/// Run `body` repeatedly for [`BUDGET`], returning the fastest iteration.
fn measure(mut body: impl FnMut()) -> Duration {
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

/// Print one result line.
fn report(label: &str, elapsed: Duration, baseline: Option<Duration>) {
    let megapixels = (WIDTH as f64 * HEIGHT as f64) / 1e6;
    let mps = megapixels / elapsed.as_secs_f64();
    let ratio = baseline.map_or(1.0, |base| elapsed.as_secs_f64() / base.as_secs_f64());
    println!(
        "  {label:<28} {:>8.1} ms  {mps:>7.1} MP/s   {ratio:>5.2}x",
        elapsed.as_secs_f64() * 1e3
    );
}
