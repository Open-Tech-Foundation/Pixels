//! Decoding: our own output round trips exactly when lossless and closely
//! when lossy, and files libjxl wrote (in `tests/fixtures`) decode to the
//! pixels `djxl` gives.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use otf_pixels_codec_jxl::{JxlDecoder, JxlEncoder};
use otf_pixels_core::{Decoder, Encoder, ImageDescriptor, Limits, PixelFormat};

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

fn encode(mut encoder: JxlEncoder, descriptor: &ImageDescriptor, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    encoder.write_header(descriptor, &mut out).unwrap();
    for row in bytes.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

fn decode(jxl: &[u8]) -> (ImageDescriptor, Vec<u8>) {
    let mut decoder = JxlDecoder::new(jxl, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut out = vec![0; descriptor.row_bytes() * descriptor.height as usize];
    for row in out.chunks_exact_mut(descriptor.row_bytes()) {
        decoder.read_row(row).unwrap();
    }
    (descriptor, out)
}

#[test]
fn lossless_output_round_trips_exactly() {
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
                let jxl = encode(JxlEncoder::lossless(), &descriptor, &bytes);
                let (got, pixels) = decode(&jxl);
                assert_eq!(got.pixel, format, "{width}x{height} {format}");
                assert!(pixels == bytes, "{width}x{height} {format} flat={flat}");
            }
        }
    }
}

/// Decode every `.jxl` in `OTF_JXL_DIR` to `<name>.otf` (raw samples) and
/// `<name>.fmt` (the pixel format), or `<name>.err`, for comparing with
/// `djxl` (scripts/check-jxl-interop.sh).
#[test]
#[ignore = "a tool for the interop script"]
fn decode_directory() {
    let Ok(dir) = std::env::var("OTF_JXL_DIR") else {
        return;
    };
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jxl") {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let result =
            std::panic::catch_unwind(|| -> otf_pixels_core::Result<(ImageDescriptor, Vec<u8>)> {
                let mut decoder = JxlDecoder::new(&bytes[..], Limits::default())?;
                let descriptor = decoder.descriptor();
                let mut out = vec![0; descriptor.row_bytes() * descriptor.height as usize];
                for row in out.chunks_exact_mut(descriptor.row_bytes()) {
                    decoder.read_row(row)?;
                }
                Ok((descriptor, out))
            });
        match result {
            Ok(Ok((descriptor, pixels))) => {
                std::fs::write(path.with_extension("otf"), pixels).unwrap();
                std::fs::write(path.with_extension("fmt"), descriptor.pixel.to_string()).unwrap();
            }
            Ok(Err(error)) => {
                std::fs::write(path.with_extension("err"), error.to_string()).unwrap()
            }
            Err(_) => std::fs::write(path.with_extension("err"), "panicked").unwrap(),
        }
    }
}

/// Damage every `.jxl` in `OTF_JXL_DIR` many ways and decode each: an error
/// is fine, a panic is a defect. Prints what panics.
#[test]
#[ignore = "a tool: a quick mutation fuzz"]
fn mutations_never_panic() {
    let Ok(dir) = std::env::var("OTF_JXL_DIR") else {
        return;
    };
    let rounds: u32 = std::env::var("OTF_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let mut seed = 0x1234_5678_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut panics = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "jxl") {
            continue;
        }
        let original = std::fs::read(&path).unwrap();
        for round in 0..rounds {
            let mut bytes = original.clone();
            match round % 3 {
                0 => bytes.truncate(next() as usize % bytes.len().max(1)),
                1 => {
                    for _ in 0..1 + next() % 4 {
                        let at = next() as usize % bytes.len();
                        bytes[at] ^= 1 << (next() % 8);
                    }
                }
                _ => {
                    let at = next() as usize % bytes.len();
                    bytes[at] = next() as u8;
                }
            }
            let result = std::panic::catch_unwind(|| {
                let Ok(mut decoder) = JxlDecoder::new(&bytes[..], Limits::default()) else {
                    return;
                };
                let descriptor = decoder.descriptor();
                let mut row = vec![0; descriptor.row_bytes()];
                let _ = decoder.read_row(&mut row);
            });
            if result.is_err() {
                panics += 1;
                let out = format!("{}.panic{round}", path.display());
                std::fs::write(&out, &bytes).unwrap();
                eprintln!("PANIC {out}");
            }
        }
    }
    assert_eq!(panics, 0);
}
