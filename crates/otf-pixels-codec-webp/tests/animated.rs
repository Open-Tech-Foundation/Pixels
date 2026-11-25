//! Animated WebP decodes frame by frame, as libwebp's animation decoder
//! renders it: the first on a transparent-black canvas, every later one
//! blended and disposed by `anim_decode.c`'s rules. Each frame is compared
//! with libwebp's, byte for byte.
//!
//! `anim_flags` is assembled frame by frame to reach every blend and
//! disposal case the compositor distinguishes, key frames included; the
//! `anim_moving_*` files are Pillow's encoder choosing sub-frames itself.
//!
//! Of the original four, two come from Pillow's animation encoder, which always codes a
//! full-canvas first frame; two are assembled by the regeneration script with
//! the first frame at an offset on a larger canvas, which the format allows
//! and that encoder never writes. Every raster is libwebp's own decode.
//! Regenerate with `scripts/regenerate-webp-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_webp::WebPDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits};

fn dir() -> String {
    format!("{}/tests/fixtures/animated", env!("CARGO_MANIFEST_DIR"))
}

fn cases() -> Vec<(String, u32, u32, usize)> {
    frame_cases()
        .into_iter()
        .map(|(name, w, h, c, _)| (name, w, h, c))
        .collect()
}

/// The manifest with each animation's frame count.
fn frame_cases() -> Vec<(String, u32, u32, usize, usize)> {
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
                f[4].parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn every_frame_matches_libwebps_animation_decoder() {
    let cases = frame_cases();
    assert!(
        cases.iter().any(|c| c.0 == "anim_flags"),
        "the corpus went missing"
    );
    for (name, width, height, channels, count) in cases {
        let file = std::fs::read(format!("{}/{name}.webp", dir())).unwrap();
        let expected = std::fs::read(format!("{}/{name}.frames.raw", dir())).unwrap();
        let frame_len = (width * height) as usize * channels;
        assert_eq!(expected.len(), frame_len * count, "{name}: reference size");

        let mut decoder = WebPDecoder::new(&file[..], Limits::default()).unwrap();
        let durations = decoder.animation().unwrap().frame_durations_ms;
        let frames = otf_pixels_core::testing::read_frames(&mut decoder)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(frames.len(), count, "{name}: frame count");
        for (k, ((info, ours), theirs)) in frames
            .iter()
            .zip(expected.chunks_exact(frame_len))
            .enumerate()
        {
            assert_eq!(info.index as usize, k, "{name}");
            assert_eq!(info.duration_ms, durations[k], "{name} frame {k}");
            if let Some(i) = ours.iter().zip(theirs).position(|(a, b)| a != b) {
                let p = i / channels;
                panic!(
                    "{name} frame {k}: pixel ({}, {}) channel {} is {} in ours, {} in libwebp's",
                    p as u32 % width,
                    p as u32 / width,
                    i % channels,
                    ours[i],
                    theirs[i]
                );
            }
        }
        assert_eq!(
            decoder.advance_frame().unwrap(),
            None,
            "{name}: past the end"
        );
    }
}

#[test]
fn a_still_has_no_next_frame() {
    let file = std::fs::read(format!("{}/../lossless/blocks_m6.webp", dir())).unwrap();
    let mut decoder = WebPDecoder::new(&file[..], Limits::default()).unwrap();
    assert_eq!(decoder.advance_frame().unwrap(), None);
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
fn every_animation_decodes_to_libwebps_first_frame() {
    let cases = cases();
    assert!(cases.len() >= 4, "the corpus went missing");
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
fn an_animation_reports_the_timing_libwebp_wrote() {
    // Pillow's libwebp wrote three frames of 100, 250 and 40 ms, looping 3
    // times (see the regeneration script).
    let bytes = std::fs::read(format!("{}/anim_timing.webp", dir())).unwrap();
    let decoder = WebPDecoder::new(&bytes[..], Limits::default()).unwrap();
    let animation = otf_pixels_core::Decoder::animation(&decoder).unwrap();
    assert_eq!(animation.frame_count, 3);
    assert_eq!(animation.loop_count, 3);
    assert_eq!(animation.frame_durations_ms, vec![100, 250, 40]);
    // The other fixtures loop forever; a still has no animation.
    let bytes = std::fs::read(format!("{}/anim_lossless.webp", dir())).unwrap();
    let decoder = WebPDecoder::new(&bytes[..], Limits::default()).unwrap();
    let animation = otf_pixels_core::Decoder::animation(&decoder).unwrap();
    assert_eq!((animation.frame_count, animation.loop_count), (2, 0));
}

#[test]
fn every_truncation_of_an_animation_is_an_error_never_a_panic() {
    for name in ["anim_offset_lossless", "anim_offset_lossy_alpha"] {
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

/// Decode every frame, as the pipeline would.
fn decode_frames(bytes: &[u8]) -> otf_pixels_core::Result<usize> {
    let mut decoder = WebPDecoder::new(bytes, Limits::default())?;
    Ok(otf_pixels_core::testing::read_frames(&mut decoder)?.len())
}

#[test]
fn every_truncation_and_corruption_of_later_frames_is_an_error_never_a_panic() {
    for name in ["anim_flags", "anim_moving_mixed"] {
        let file = std::fs::read(format!("{}/{name}.webp", dir())).unwrap();
        for len in (0..file.len()).step_by(3) {
            if let Err(error) = decode_frames(&file[..len]) {
                assert_eq!(
                    error.code(),
                    ErrorCode::Malformed,
                    "{name} at {len}: {error}"
                );
            }
        }
        for at in (20..file.len()).step_by(5) {
            for flip in [0x01_u8, 0x80, 0xff] {
                let mut bad = file.clone();
                bad[at] ^= flip;
                let _ = decode_frames(&bad);
            }
        }
    }
}

#[test]
fn corrupted_animations_are_errors_or_pixels_never_panics() {
    let file = std::fs::read(format!("{}/anim_lossy_alpha.webp", dir())).unwrap();
    for at in (20..file.len()).step_by(7) {
        for flip in [0x01_u8, 0x80, 0xff] {
            let mut bad = file.clone();
            bad[at] ^= flip;
            let _ = decode(&bad);
        }
    }
}
