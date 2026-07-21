//! Emits our own JPEGs so libjpeg can verify them.
//!
//! Our decoder cannot validate our encoder: a shared misreading of the
//! specification would round-trip perfectly and still be wrong.
//! `tests/reference.rs` checks one direction by decoding what libjpeg
//! produced; this checks the other, by producing files for libjpeg to read.
//!
//! Alongside each JPEG, the source raster is written out. The comparison
//! cannot be exact — JPEG is lossy, and that is the point of it — so the
//! checking script compares with a tolerance derived from the quality each
//! file was written at, which is encoded in its name.
//!
//! Inert unless `OTF_EMIT_DIR` is set, so an ordinary `cargo test` is
//! unaffected. `scripts/check-jpeg-interop.sh` drives it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_jpeg::{JpegEncoder, Subsampling};
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

#[test]
fn emit_jpegs_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }

    // Sizes chosen to catch the classic off-by-ones: images smaller than one
    // MCU, images whose width or height is one past an MCU boundary, and
    // images that are exactly a whole number of MCUs.
    let sizes = [
        (1_u32, 1_u32),
        (7, 3),
        (16, 16),
        (17, 17),
        (31, 15),
        (64, 48),
        (129, 5),
    ];
    let subsamplings = [
        ("444", Subsampling::None),
        ("422", Subsampling::Horizontal),
        ("420", Subsampling::Both),
    ];
    let formats = [("rgb", PixelFormat::Rgb8), ("gray", PixelFormat::Gray8)];

    for &(width, height) in &sizes {
        for &(kind, format) in &formats {
            let raster = source_image(width, height, format);
            let descriptor = ImageDescriptor::new(width, height, format).unwrap();

            for &(label, subsampling) in &subsamplings {
                // Subsampling has no meaning for a single-component image, so
                // emitting three copies of it would only slow the check down.
                if format == PixelFormat::Gray8 && subsampling != Subsampling::None {
                    continue;
                }
                for quality in [25_u8, 50, 75, 90, 100] {
                    let mut encoder = JpegEncoder::with_quality(quality)
                        .unwrap()
                        .with_subsampling(subsampling);
                    let mut jpeg: Vec<u8> = Vec::new();
                    encoder.write_header(&descriptor, &mut jpeg).unwrap();
                    for row in raster.chunks_exact(descriptor.row_bytes()) {
                        encoder.write_row(row, &mut jpeg).unwrap();
                    }
                    encoder.finish(&mut jpeg).unwrap();

                    let name = format!("{dir}/{width}x{height}_{kind}_{label}_q{quality}");
                    std::fs::write(format!("{name}.jpg"), &jpeg).unwrap();
                    std::fs::write(format!("{name}.raw"), &raster).unwrap();
                }
            }
        }
    }
}

/// A source image with both smooth regions and hard edges, so the comparison
/// exercises the low and high frequencies rather than only one.
fn source_image(width: u32, height: u32, format: PixelFormat) -> Vec<u8> {
    let channels = format.channels();
    let mut out = Vec::with_capacity((width * height) as usize * channels);
    for y in 0..height {
        for x in 0..width {
            // A gradient, with a darker band every twelve pixels.
            let edge = if (x / 12 + y / 12) % 2 == 0 { 0 } else { 60 };
            let r = (x * 200 / width.max(1)) as u8;
            let g = (y * 200 / height.max(1)) as u8;
            let b = ((x + y) * 200 / (width + height).max(1)) as u8;
            match channels {
                1 => out.push(r.saturating_sub(edge)),
                _ => out.extend_from_slice(&[
                    r.saturating_sub(edge),
                    g.saturating_sub(edge),
                    b.saturating_sub(edge),
                ]),
            }
        }
    }
    out
}
