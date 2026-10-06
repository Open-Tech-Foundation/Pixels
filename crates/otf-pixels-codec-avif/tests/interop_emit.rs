//! Emits our own AVIF files so libavif can verify them.
//!
//! Our decoder cannot validate our encoder: a misreading of the format shared
//! by both would round-trip perfectly and still produce files nothing else
//! reads. So `scripts/check-avif-interop.sh` hands these to `avifdec` with
//! every AV1 decoder it has, and requires each to decode them to our pixels
//! (colour within the rounding of two YUV-to-RGB conversions, alpha exact).
//!
//! Inert unless `OTF_EMIT_DIR` is set, so an ordinary `cargo test` is
//! unaffected.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_avif::{AvifDecoder, AvifEncoder};
use otf_pixels_core::{Decoder, EncodeOptions, Encoder, ImageDescriptor, Limits, PixelFormat};

fn encode(descriptor: &ImageDescriptor, raster: &[u8], quality: u8) -> Vec<u8> {
    let mut encoder = AvifEncoder::from_options(&EncodeOptions::with_quality(quality).unwrap());
    let mut avif: Vec<u8> = Vec::new();
    encoder.write_header(descriptor, &mut avif).unwrap();
    for row in raster.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut avif).unwrap();
    }
    encoder.finish(&mut avif).unwrap();
    avif
}

fn decode(avif: &[u8]) -> Vec<u8> {
    let mut decoder = AvifDecoder::new(avif, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut out = vec![0_u8; descriptor.row_bytes() * descriptor.height as usize];
    for row in out.chunks_mut(descriptor.row_bytes()) {
        decoder.read_row(row).unwrap();
    }
    out
}

/// Gradients, a hard-edged disc, fine stripes, and (with alpha) a ramp and a
/// fully transparent corner.
fn scene(width: u32, height: u32, channels: usize) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut out = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        for x in 0..w {
            let disc = (x as i64 - w as i64 / 2).pow(2) + (y as i64 - h as i64 / 2).pow(2)
                < (w.min(h) as i64 / 3).pow(2);
            let stripes = (x / 2 + y / 3) % 2 == 0 && y > h / 2;
            let r = (x * 255 / w.max(1)) as u8;
            let g = if disc {
                220
            } else {
                (y * 255 / h.max(1)) as u8
            };
            let b = if stripes { 200 } else { 40 };
            let px = [r, g, b];
            match channels {
                1 => out.push(((u16::from(r) + u16::from(g)) / 2) as u8),
                3 => out.extend_from_slice(&px),
                _ => {
                    out.extend_from_slice(&px);
                    out.push(if x < w / 4 && y < h / 4 {
                        0
                    } else {
                        (128 + x * 127 / w.max(1)) as u8
                    });
                }
            }
        }
    }
    out
}

#[test]
fn emit_avif_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    let cases = [
        (1_u32, 1_u32, PixelFormat::Rgb8, 60_u8),
        (17, 9, PixelFormat::Rgb8, 80),
        (96, 64, PixelFormat::Rgb8, 50),
        (200, 133, PixelFormat::Rgba8, 75),
        (64, 48, PixelFormat::Gray8, 90),
        (4200, 16, PixelFormat::Rgb8, 40), // wider than one tile may be
        (130, 70, PixelFormat::Rgb8, 100),
        (130, 70, PixelFormat::Rgb8, 1),
    ];
    for (width, height, pixel, quality) in cases {
        let descriptor = ImageDescriptor::new(width, height, pixel).unwrap();
        let raster = scene(width, height, pixel.channels());
        let avif = encode(&descriptor, &raster, quality);
        let kind = match pixel {
            PixelFormat::Gray8 => "gray",
            PixelFormat::Rgba8 => "rgba",
            _ => "rgb",
        };
        let name = format!("{width}x{height}_{kind}_q{quality}");
        std::fs::write(dir.join(format!("{name}.avif")), &avif).unwrap();
        std::fs::write(dir.join(format!("{name}.ours")), decode(&avif)).unwrap();
    }
}
