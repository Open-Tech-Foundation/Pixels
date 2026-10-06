//! Emits our own WebP files so libwebp can verify them.
//!
//! Our decoder cannot validate our encoder: a misreading of the format shared
//! by both would round-trip perfectly and still produce files nothing else
//! reads. So libwebp gets the last word, in two ways:
//!
//! - Lossless files must decode, in libwebp, to exactly the pixels we put in.
//! - Lossy files must decode, in libwebp, to exactly what our own decoder
//!   makes of them — VP8 decoding is exact by specification, so any
//!   difference is a bitstream one of us misreads — and their alpha, which
//!   is coded losslessly, must come back exact.
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

use otf_pixels_codec_webp::{WebPDecoder, WebPEncoder};
use otf_pixels_core::{Decoder, EncodeOptions, Encoder, ImageDescriptor, Limits, PixelFormat};

fn encode(descriptor: &ImageDescriptor, raster: &[u8], options: EncodeOptions) -> Vec<u8> {
    let mut encoder = WebPEncoder::from_options(&options);
    let mut webp: Vec<u8> = Vec::new();
    encoder.write_header(descriptor, &mut webp).unwrap();
    for row in raster.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut webp).unwrap();
    }
    encoder.finish(&mut webp).unwrap();
    webp
}

fn decode(webp: &[u8]) -> Vec<u8> {
    let mut decoder = WebPDecoder::new(webp, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut out = vec![0_u8; descriptor.row_bytes() * descriptor.height as usize];
    for row in out.chunks_mut(descriptor.row_bytes()) {
        decoder.read_row(row).unwrap();
    }
    out
}

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

            let lossless = EncodeOptions::default().with_lossless(true);
            let webp = encode(&descriptor, &raster, lossless);

            let name = format!("{dir}/{width}x{height}_{kind}");
            std::fs::write(format!("{name}.webp"), &webp).unwrap();
            std::fs::write(format!("{name}.raw"), &raster).unwrap();

            for quality in [10_u8, 80] {
                let options = EncodeOptions::with_quality(quality).unwrap();
                let raster = with_varying_alpha(raster.clone(), format);
                let webp = encode(&descriptor, &raster, options);
                let name = format!("{dir}/{width}x{height}_{kind}_lossy_q{quality}");
                std::fs::write(format!("{name}.webp"), &webp).unwrap();
                std::fs::write(format!("{name}.raw"), &raster).unwrap();
                std::fs::write(format!("{name}.ours"), decode(&webp)).unwrap();
            }
        }
    }
}

/// The same image with a gradient in its alpha channel, if it has one, so
/// lossy files carry a real `ALPH` chunk rather than an opaque one dropped.
fn with_varying_alpha(mut raster: Vec<u8>, format: PixelFormat) -> Vec<u8> {
    let channels = format.channels();
    if channels % 2 == 0 {
        for (i, pixel) in raster.chunks_exact_mut(channels).enumerate() {
            pixel[channels - 1] = (i * 7 % 256) as u8;
        }
    }
    raster
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
