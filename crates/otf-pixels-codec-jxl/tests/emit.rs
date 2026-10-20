//! Writes sample files to `OTF_EMIT_DIR`, for checking with libjxl's `djxl`
//! (scripts/check-jxl-interop.sh). Without the variable it only checks the
//! files are written.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use otf_pixels_codec_jxl::JxlEncoder;
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

/// A width x height image in `format` whose samples are a mix of gradient
/// and noise, as raw bytes; with `flat`, blocks of one colour instead, so
/// exact predictions run long.
fn image(width: u32, height: u32, format: PixelFormat, flat: bool) -> (ImageDescriptor, Vec<u8>) {
    let descriptor = ImageDescriptor::new(width, height, format).unwrap();
    let mut seed = 1_u32;
    let wide = format.bytes_per_pixel() / format.channels() == 2;
    let mut bytes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for c in 0..format.channels() as u32 {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let v = if flat {
                    ((x / 37 + y / 23) % 3 * 90 + c * 7) as u16
                } else {
                    (x * 3 + y * 2 + c * 50 + ((seed >> 16) & 15)) as u16
                };
                if wide {
                    bytes.extend_from_slice(&(v.wrapping_mul(257)).to_ne_bytes());
                } else {
                    bytes.push(v as u8);
                }
            }
        }
    }
    (descriptor, bytes)
}

/// Encode a raw image.
fn encode(descriptor: &ImageDescriptor, bytes: &[u8]) -> Vec<u8> {
    let mut encoder = JxlEncoder::new();
    let mut out = Vec::new();
    encoder.write_header(descriptor, &mut out).unwrap();
    for row in bytes.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

#[test]
fn emit_samples() {
    let dir = std::env::var("OTF_EMIT_DIR").ok();
    for (width, height) in [(1, 1), (17, 9), (256, 256), (300, 40), (513, 300)] {
        for format in [
            PixelFormat::Gray8,
            PixelFormat::GrayA8,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Gray16,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
        ] {
            for flat in [false, true] {
                let (descriptor, bytes) = image(width, height, format, flat);
                let jxl = encode(&descriptor, &bytes);
                assert_eq!(jxl[..2], [0xFF, 0x0A]);
                if let Some(dir) = &dir {
                    let kind = if flat { "flat" } else { "noise" };
                    let stem = format!("{dir}/{width}x{height}{kind}_{format}");
                    std::fs::write(format!("{stem}.jxl"), &jxl).unwrap();
                    std::fs::write(format!("{stem}.raw"), &bytes).unwrap();
                }
            }
        }
    }
}
