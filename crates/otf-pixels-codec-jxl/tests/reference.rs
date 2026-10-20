//! JPEG XL decode checked against libjxl, not against ourselves.
//!
//! The decoder is owned end to end (ADR-0019). The fixtures are written by
//! libjxl's `cjxl` across its coding tools and the reference rasters are what
//! libjxl's `djxl` makes of them; a lossless fixture must match to the byte,
//! a lossy one within the manifest's tolerance, since the two decoders round
//! floats differently.
//!
//! Regenerate the fixtures and manifest with `scripts/regenerate-jxl-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_jxl::JxlDecoder;
use otf_pixels_core::{Decoder, Limits};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

fn decode(bytes: &[u8]) -> (otf_pixels_core::ImageDescriptor, Vec<u8>, JxlDecoder) {
    let mut decoder = JxlDecoder::new(bytes, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut out = vec![0; descriptor.row_bytes() * descriptor.height as usize];
    for row in out.chunks_exact_mut(descriptor.row_bytes()) {
        decoder.read_row(row).unwrap();
    }
    (descriptor, out, decoder)
}

#[test]
fn every_fixture_decodes_to_libjxls_pixels() {
    let manifest = String::from_utf8(fixture("REFERENCE")).unwrap();
    let mut checked = 0;
    for line in manifest
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let [name, width, height, channels, tolerance, bits] = f[..] else {
            panic!("bad manifest line {line}");
        };
        let (width, height, channels): (u32, u32, usize) = (
            width.parse().unwrap(),
            height.parse().unwrap(),
            channels.parse().unwrap(),
        );
        let tolerance: u32 = tolerance.parse().unwrap();
        let (descriptor, pixels, _) = decode(&fixture(&format!("{name}.jxl")));
        assert_eq!(
            (descriptor.width, descriptor.height),
            (width, height),
            "{name}"
        );
        assert_eq!(descriptor.pixel.channels(), channels, "{name}");
        let expected = fixture(&format!("{name}.raw"));
        assert_eq!(pixels.len(), expected.len(), "{name}");
        let worst = if bits == "16" {
            pixels
                .chunks_exact(2)
                .zip(expected.chunks_exact(2))
                .map(|(a, b)| {
                    u16::from_ne_bytes([a[0], a[1]]).abs_diff(u16::from_le_bytes([b[0], b[1]]))
                })
                .max()
                .map_or(0, u32::from)
        } else {
            pixels
                .iter()
                .zip(&expected)
                .map(|(a, b)| u32::from(a.abs_diff(*b)))
                .max()
                .unwrap_or(0)
        };
        assert!(
            worst <= tolerance,
            "{name}: off by {worst}, tolerance {tolerance}"
        );
        checked += 1;
    }
    assert!(checked >= 20, "the manifest lists {checked} fixtures");
}

#[test]
fn an_animation_reports_its_frames_and_decodes_the_first() {
    let (_, _, decoder) = decode(&fixture("anim.jxl"));
    let animation = decoder.animation().expect("three frames");
    assert_eq!(animation.frame_durations_ms, [100, 100, 100]);
    assert_eq!(animation.loop_count, 0);
}

#[test]
fn a_still_image_has_no_animation_and_no_profile_for_xyb() {
    let (_, _, decoder) = decode(&fixture("photo_d1.jxl"));
    assert!(decoder.animation().is_none());
    // XYB is delivered as sRGB, so no profile describes the samples.
    assert!(decoder.icc_profile().is_none());
}

#[test]
fn damaged_files_are_errors_not_panics() {
    for name in [
        "photo_d1.jxl",
        "photo_lossless_e9.jxl",
        "text_d1_patches.jxl",
        "photo_jpeg.jxl",
    ] {
        let bytes = fixture(name);
        for cut in [
            0,
            1,
            2,
            12,
            bytes.len() / 3,
            bytes.len() / 2,
            bytes.len() - 1,
        ] {
            let Ok(mut decoder) = JxlDecoder::new(&bytes[..cut], Limits::default()) else {
                continue;
            };
            let mut row = vec![0; decoder.descriptor().row_bytes()];
            assert!(
                decoder.read_row(&mut row).is_err(),
                "{name} cut at {cut} decoded"
            );
        }
        let mut flipped = bytes.clone();
        for i in (20..flipped.len()).step_by(7) {
            flipped[i] ^= 0x5A;
        }
        if let Ok(mut decoder) = JxlDecoder::new(&flipped[..], Limits::default()) {
            let mut row = vec![0; decoder.descriptor().row_bytes()];
            let _ = decoder.read_row(&mut row);
        }
    }
}

#[test]
fn limits_are_enforced_at_the_header() {
    let limits = Limits::default().with_max_pixels(1000);
    assert!(JxlDecoder::new(&fixture("photo_d1.jxl")[..], limits).is_err());
}
