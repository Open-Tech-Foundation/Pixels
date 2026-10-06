//! The owned VP8L decoder against libwebp, exactly.
//!
//! Each file in `tests/fixtures/lossless` was written by libwebp (through
//! Pillow) at a method and quality chosen to make its encoder reach for a
//! particular tool — every transform, colour indexing at each bundling width,
//! the colour cache, meta prefix codes on the larger images — and its raster
//! is libwebp's own decode of those bytes. Lossless means any difference at
//! all is a bug. Regenerate with `scripts/regenerate-webp-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_webp::WebPDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits};

fn dir() -> String {
    format!("{}/tests/fixtures/lossless", env!("CARGO_MANIFEST_DIR"))
}

fn cases() -> Vec<(String, u32, u32, usize)> {
    std::fs::read_to_string(format!("{}/REFERENCE", dir()))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (
                f[0].to_owned(),
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
            )
        })
        .collect()
}

fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(Vec<u8>, otf_pixels_core::ImageDescriptor)> {
    let mut decoder = WebPDecoder::new(bytes, Limits::default())?;
    let descriptor = decoder.descriptor();
    let mut out = vec![0_u8; descriptor.row_bytes() * descriptor.height as usize];
    for row in out.chunks_mut(descriptor.row_bytes()) {
        decoder.read_row(row)?;
    }
    Ok((out, descriptor))
}

#[test]
fn every_lossless_fixture_matches_libwebp_exactly() {
    let cases = cases();
    assert!(cases.len() >= 20, "the corpus went missing");
    for (name, width, height, channels) in cases {
        let file = std::fs::read(format!("{}/{name}.webp", dir())).unwrap();
        let expected = std::fs::read(format!("{}/{name}.raw", dir())).unwrap();
        let (ours, descriptor) = decode(&file).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            (descriptor.width, descriptor.height),
            (width, height),
            "{name}"
        );
        assert_eq!(descriptor.pixel.channels(), channels, "{name}: alpha");
        if let Some(i) = ours.iter().zip(&expected).position(|(a, b)| a != b) {
            panic!(
                "{name}: pixel {} differs: ours {} vs libwebp {}",
                i / channels,
                ours[i],
                expected[i]
            );
        }
        assert_eq!(ours.len(), expected.len(), "{name}: length");
    }
}

#[test]
fn every_truncation_of_a_lossless_file_is_an_error_never_a_panic() {
    for name in ["smooth_m4", "palette5", "noise_alpha_m2"] {
        let file = std::fs::read(format!("{}/{name}.webp", dir())).unwrap();
        for len in 0..file.len() {
            if let Err(error) = decode(&file[..len]) {
                assert_eq!(
                    error.code(),
                    ErrorCode::Malformed,
                    "{name} at {len}: {error}"
                );
            }
        }
    }
}

#[test]
fn corrupted_lossless_files_are_errors_or_pixels_never_panics() {
    let file = std::fs::read(format!("{}/smooth_alpha_m4.webp", dir())).unwrap();
    for at in (20..file.len()).step_by(7) {
        for flip in [0x01_u8, 0x80, 0xff] {
            let mut bad = file.clone();
            bad[at] ^= flip;
            let _ = decode(&bad);
        }
    }
}
