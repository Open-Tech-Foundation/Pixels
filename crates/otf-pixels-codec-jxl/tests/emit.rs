//! Writes sample files to `OTF_EMIT_DIR`, for checking with libjxl's `djxl`
//! (scripts/check-jxl-interop.sh). Without the variable it only checks the
//! files are written.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

mod common;

use otf_pixels_codec_jxl::{JxlDecoder, JxlEncoder};
use otf_pixels_core::{Encoder, ImageDescriptor, Limits, PixelFormat};

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
    let mut encoder = JxlEncoder::lossless();
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

/// Lossy files: `<name>_lossy.jxl` with the 8-bit RGB source as `.rgb`, for
/// the interop script to decode and hold to a PSNR floor.
#[test]
fn emit_lossy_samples() {
    let dir = std::env::var("OTF_EMIT_DIR").ok();
    for (width, height) in [
        (1, 1),
        (17, 9),
        (256, 256),
        (300, 40),
        (513, 300),
        (2100, 24),
    ] {
        for format in [PixelFormat::Rgb8, PixelFormat::Gray8, PixelFormat::Rgb16] {
            let descriptor = ImageDescriptor::new(width, height, format).unwrap();
            // A smooth scene with edges: what VarDCT is for.
            let channels = format.channels() as u32;
            let wide = format.bytes_per_pixel() / format.channels() == 2;
            let mut bytes = Vec::new();
            let mut rgb = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    let disc = ((x as i32 - 150).pow(2) + (y as i32 - 100).pow(2)) < 60 * 60;
                    let base = [
                        (x * 255 / width.max(1)) as u8,
                        (y * 255 / height.max(1)) as u8,
                        if disc { 230 } else { 40 },
                    ];
                    let px: Vec<u8> = if channels == 1 {
                        vec![base[0] / 2 + base[1] / 2]
                    } else {
                        base.to_vec()
                    };
                    for &v in &px {
                        if wide {
                            bytes.extend_from_slice(&(u16::from(v) * 257).to_ne_bytes());
                        } else {
                            bytes.push(v);
                        }
                    }
                    let grey = px[0];
                    rgb.extend(if channels == 1 { [grey; 3] } else { base });
                }
            }
            let mut encoder = JxlEncoder::with_distance(1.0);
            let mut jxl = Vec::new();
            encoder.write_header(&descriptor, &mut jxl).unwrap();
            for row in bytes.chunks_exact(descriptor.row_bytes()) {
                encoder.write_row(row, &mut jxl).unwrap();
            }
            encoder.finish(&mut jxl).unwrap();
            assert_eq!(jxl[..2], [0xFF, 0x0A]);
            if let Some(dir) = &dir {
                let stem = format!("{dir}/{width}x{height}_{format}_lossy");
                std::fs::write(format!("{stem}.jxl"), &jxl).unwrap();
                std::fs::write(format!("{stem}.rgb"), &rgb).unwrap();
            }
        }
    }
}

#[test]
fn emit_animations() {
    // Each scene lossless, whose frames djxl must give back exactly, and the
    // 8-bit RGB one lossy, whose frames it must give within a level of ours.
    // A `.meta` line: width, height, channels, bits, frames, loop count, then
    // each duration.
    let dir = std::env::var("OTF_EMIT_DIR").ok();
    for scene in common::scenes() {
        let lossy = scene.name == "rgb8";
        for (mode, encoder) in [
            ("lossless", JxlEncoder::lossless()),
            ("lossy", JxlEncoder::new()),
        ] {
            if mode == "lossy" && !lossy {
                continue;
            }
            let jxl = common::encode(&scene, &scene.frames, &scene.durations, encoder);
            let mut decoder = JxlDecoder::new(&jxl[..], Limits::default()).unwrap();
            let ours: Vec<u8> = otf_pixels_core::testing::read_frames(&mut decoder)
                .unwrap()
                .into_iter()
                .flat_map(|(_, pixels)| pixels)
                .collect();
            let Some(dir) = &dir else { continue };
            let expected = if mode == "lossless" {
                scene.frames.concat()
            } else {
                ours
            };
            let stem = format!("{dir}/anim_{}_{mode}", scene.name);
            std::fs::write(format!("{stem}.jxl"), &jxl).unwrap();
            std::fs::write(format!("{stem}.frames"), expected).unwrap();
            let pixel = scene.descriptor.pixel;
            let durations: Vec<String> = scene.durations.iter().map(u32::to_string).collect();
            let meta = format!(
                "{} {} {} {} {} {} {}",
                scene.descriptor.width,
                scene.descriptor.height,
                pixel.channels(),
                pixel.bytes_per_pixel() / pixel.channels() * 8,
                scene.frames.len(),
                scene.loop_count,
                durations.join(" ")
            );
            std::fs::write(format!("{stem}.meta"), meta).unwrap();
        }
    }
}
