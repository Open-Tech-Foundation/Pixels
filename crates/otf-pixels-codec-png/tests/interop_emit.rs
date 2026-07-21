//! Emits our own zlib streams and PNG files so external tools can verify them.
//!
//! Our decoder cannot validate our encoder: a shared misreading of RFC 1951 or
//! of the PNG specification would round-trip perfectly and still be wrong.
//! `tests/zlib_reference.rs` and `tests/pngsuite.rs` check one direction, by
//! reading what real zlib and real libpng produced; this checks the other, by
//! producing files for them to read.
//!
//! It is inert unless `OTF_EMIT_DIR` is set, so an ordinary `cargo test` is
//! unaffected. `scripts/check-deflate-interop.sh` and
//! `scripts/check-png-interop.sh` drive it and run the comparisons.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_png::{Level, PngEncoder, zlib_compress};
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

#[test]
fn emit_streams_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("text", b"the quick brown fox. ".repeat(300)),
        ("zeros", vec![0; 10_000]),
        (
            "incompressible",
            (0..8192).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
        ),
        ("longrun", vec![0xAB; 70_000]),
        ("binary", (0..=255_u8).cycle().take(20_000).collect()),
        ("min_match", vec![7, 7, 7]),
        ("max_match", std::iter::repeat_n(b'z', 258 * 3).collect()),
    ];
    for (name, data) in cases {
        for level in [0_u8, 1, 6, 9] {
            let compressed = zlib_compress(&data, Level::new(level).unwrap()).unwrap();
            std::fs::write(format!("{dir}/{name}_{level}.zlib"), &compressed).unwrap();
            std::fs::write(format!("{dir}/{name}_{level}.raw"), &data).unwrap();
        }
    }
}

/// Emit PNGs for libpng to read, alongside the raw raster each encodes.
///
/// The companion raster is written as `<name>.rgba`: 8-bit RGBA, which is the
/// one form the comparison script can produce from any Pillow decoding. For
/// 16-bit cases the low byte is dropped, matching `tests/pngsuite.rs`.
#[test]
fn emit_pngs_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }

    // Sizes chosen to catch the classic off-by-ones: a single pixel, a single
    // row, a single column, and a size that is not a multiple of anything.
    let sizes = [(1_u32, 1_u32), (17, 1), (1, 17), (23, 19), (64, 64)];
    let formats = [
        PixelFormat::Gray8,
        PixelFormat::Gray16,
        PixelFormat::GrayA8,
        PixelFormat::Rgb8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba8,
        PixelFormat::Rgba16,
    ];

    for &(width, height) in &sizes {
        for format in formats {
            for level in [0_u8, 1, 6, 9] {
                let descriptor = ImageDescriptor::new(width, height, format).unwrap();
                let raster = pattern(&descriptor);
                let mut encoder = PngEncoder::with_level(Level::new(level).unwrap());
                let mut png: Vec<u8> = Vec::new();
                encoder.write_header(&descriptor, &mut png).unwrap();
                for row in raster.chunks_exact(descriptor.row_bytes()) {
                    encoder.write_row(row, &mut png).unwrap();
                }
                encoder.finish(&mut png).unwrap();

                let name = format!("{dir}/{format}_{width}x{height}_l{level}");
                std::fs::write(format!("{name}.png"), &png).unwrap();
                std::fs::write(format!("{name}.rgba"), to_rgba8(&descriptor, &raster)).unwrap();
            }
        }
    }
}

/// A deterministic raster with gradients, hard edges and saturated values, so
/// every filter is plausible on some row and 16-bit byte order is observable.
fn pattern(descriptor: &ImageDescriptor) -> Vec<u8> {
    let mut raster = vec![0_u8; descriptor.byte_len().unwrap()];
    for (index, byte) in raster.iter_mut().enumerate() {
        *byte = match index % 5 {
            0 => (index % 251) as u8,
            1 => 0xFF,
            2 => 0x00,
            3 => ((index / 7) % 256) as u8,
            _ => ((index * 31) % 97) as u8,
        };
    }
    raster
}

/// Reduce our raster to 8-bit RGBA, the form the comparison script produces
/// from Pillow. 16-bit samples lose their low byte; see `tests/pngsuite.rs`
/// for why that is the only reduction both sides can agree on.
fn to_rgba8(descriptor: &ImageDescriptor, raster: &[u8]) -> Vec<u8> {
    let pixels = (descriptor.width as usize) * (descriptor.height as usize);
    let bpp = descriptor.pixel.bytes_per_pixel();
    let mut out = Vec::with_capacity(pixels * 4);
    // Our 16-bit samples are native-endian, so the high byte is at +1 on a
    // little-endian host.
    let high = if cfg!(target_endian = "little") { 1 } else { 0 };
    for index in 0..pixels {
        let p = &raster[index * bpp..(index + 1) * bpp];
        let (r, g, b, a) = match descriptor.pixel {
            PixelFormat::Gray8 => (p[0], p[0], p[0], 255),
            PixelFormat::GrayA8 => (p[0], p[0], p[0], p[1]),
            PixelFormat::Gray16 => (p[high], p[high], p[high], 255),
            PixelFormat::Rgb8 => (p[0], p[1], p[2], 255),
            PixelFormat::Rgba8 => (p[0], p[1], p[2], p[3]),
            PixelFormat::Rgb16 => (p[high], p[2 + high], p[4 + high], 255),
            PixelFormat::Rgba16 => (p[high], p[2 + high], p[4 + high], p[6 + high]),
            other => panic!("unexpected format {other}"),
        };
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}
