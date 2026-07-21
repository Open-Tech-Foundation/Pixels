//! Emits our own WebP files so libwebp can verify them.
//!
//! Our decoder cannot validate our encoder — both sit on the same wrapped
//! crate, so a fault in it would round-trip perfectly and still produce a file
//! nothing else reads. `tests/reference.rs` checks the decode direction
//! against libwebp; this checks the encode direction, by handing libwebp our
//! files.
//!
//! The encoder is lossless, so the comparison is exact. That is the whole
//! value of testing this direction here: there is no tolerance to hide behind.
//!
//! Inert unless `OTF_EMIT_DIR` is set, so an ordinary `cargo test` is
//! unaffected. `scripts/check-webp-interop.sh` drives it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_webp::WebPEncoder;
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

#[test]
fn emit_webp_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }

    // Sizes chosen to catch the classic off-by-ones, including a single pixel
    // and a single row.
    let sizes = [
        (1_u32, 1_u32),
        (1, 17),
        (17, 1),
        (23, 19),
        (64, 48),
        (129, 5),
    ];
    let formats = [
        ("rgb", PixelFormat::Rgb8),
        ("rgba", PixelFormat::Rgba8),
        ("gray", PixelFormat::Gray8),
        ("graya", PixelFormat::GrayA8),
    ];

    for &(width, height) in &sizes {
        for &(kind, format) in &formats {
            let descriptor = ImageDescriptor::new(width, height, format).unwrap();
            let raster = source_image(width, height, format);

            let mut encoder = WebPEncoder::new();
            let mut webp: Vec<u8> = Vec::new();
            encoder.write_header(&descriptor, &mut webp).unwrap();
            for row in raster.chunks_exact(descriptor.row_bytes()) {
                encoder.write_row(row, &mut webp).unwrap();
            }
            encoder.finish(&mut webp).unwrap();

            let name = format!("{dir}/{width}x{height}_{kind}");
            std::fs::write(format!("{name}.webp"), &webp).unwrap();
            std::fs::write(format!("{name}.raw"), &raster).unwrap();
        }
    }
}

/// A deterministic image with both flat runs and per-pixel variation, so the
/// lossless coder's predictors and its literal path are both exercised.
fn source_image(width: u32, height: u32, format: PixelFormat) -> Vec<u8> {
    let channels = format.channels();
    let mut out = Vec::with_capacity((width * height) as usize * channels);
    let mut state = 0x1234_5678_u32;
    for y in 0..height {
        for x in 0..width {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let noise = (state >> 24) as u8;
            let flat = ((x / 8 + y / 8) % 2 == 0) as u8;
            let value = if flat == 1 { 200 } else { noise };
            match channels {
                1 => out.push(value),
                2 => out.extend_from_slice(&[value, 255]),
                3 => out.extend_from_slice(&[value, value.wrapping_add(40), 90]),
                _ => out.extend_from_slice(&[value, value.wrapping_add(40), 90, 255]),
            }
        }
    }
    out
}
