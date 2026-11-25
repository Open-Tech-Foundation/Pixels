//! Animated JPEG XL encoding, read back by this crate's decoder (checked
//! against libjxl frame by frame in `tests/reference.rs`);
//! `scripts/check-jxl-interop.sh` hands the same files to `djxl`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

mod common;

use otf_pixels_codec_jxl::{JxlDecoder, JxlEncoder};
use otf_pixels_core::{
    Decoder, EncodeOptions, Encoder, ErrorCode, ImageDescriptor, Limits, PixelFormat,
};

fn decode(bytes: &[u8]) -> (otf_pixels_core::Animation, ImageDescriptor, Vec<Vec<u8>>) {
    let mut decoder = JxlDecoder::new(bytes, Limits::default()).unwrap();
    let animation = decoder.animation().expect("animated");
    let descriptor = decoder.descriptor();
    let frames = otf_pixels_core::testing::read_frames(&mut decoder)
        .unwrap()
        .into_iter()
        .map(|(_, pixels)| pixels)
        .collect();
    (animation, descriptor, frames)
}

#[test]
fn lossless_frames_come_back_exactly_with_their_timing() {
    for scene in common::scenes() {
        let bytes = common::encode(
            &scene,
            &scene.frames,
            &scene.durations,
            JxlEncoder::lossless(),
        );
        let (animation, descriptor, frames) = decode(&bytes);
        assert_eq!(descriptor.pixel, scene.descriptor.pixel, "{}", scene.name);
        assert_eq!(animation.loop_count, scene.loop_count, "{}", scene.name);
        assert_eq!(
            animation.frame_durations_ms, scene.durations,
            "{}",
            scene.name
        );
        assert_eq!(frames.len(), scene.frames.len(), "{}", scene.name);
        for (k, (ours, want)) in frames.iter().zip(&scene.frames).enumerate() {
            assert!(ours == want, "{} frame {k} differs", scene.name);
        }
    }
}

#[test]
fn lossy_frames_are_as_close_as_stills() {
    let scene = common::scenes().remove(1);
    assert_eq!(scene.name, "rgb8");
    let bytes = common::encode(&scene, &scene.frames, &scene.durations, JxlEncoder::new());
    let (animation, _, frames) = decode(&bytes);
    assert_eq!(animation.frame_durations_ms, scene.durations);
    let error = |a: &[u8], b: &[u8]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(x.abs_diff(*y)))
            .sum::<f64>()
            / a.len() as f64
    };
    for (k, (ours, want)) in frames.iter().zip(&scene.frames).enumerate() {
        let mut still = JxlEncoder::new();
        let mut one = Vec::new();
        still.write_header(&scene.descriptor, &mut one).unwrap();
        for row in want.chunks_exact(scene.descriptor.row_bytes()) {
            still.write_row(row, &mut one).unwrap();
        }
        still.finish(&mut one).unwrap();
        let mut decoder = JxlDecoder::new(&one[..], Limits::default()).unwrap();
        let alone = otf_pixels_core::testing::read_frames(&mut decoder)
            .unwrap()
            .remove(0)
            .1;
        let (animated, by_itself) = (error(ours, want), error(&alone, want));
        assert!(
            animated <= by_itself * 1.25 + 0.5,
            "frame {k}: {animated} against {by_itself} as a still"
        );
    }
}

#[test]
fn repeated_frames_merge_unless_kept_and_later_frames_are_cropped() {
    let scene = common::scenes().remove(1);
    let frames = [
        scene.frames[0].clone(),
        scene.frames[0].clone(),
        scene.frames[1].clone(),
    ];
    let merged = common::encode(&scene, &frames, &[100, 50, 70], JxlEncoder::lossless());
    let (animation, _, decoded) = decode(&merged);
    assert_eq!(animation.frame_durations_ms, [150, 70]);
    assert_eq!(decoded[1], frames[2]);

    let keep = JxlEncoder::from_options(
        &EncodeOptions::default()
            .with_lossless(true)
            .with_keep_duplicate_frames(true),
    );
    let kept = common::encode(&scene, &frames, &[100, 50, 70], keep);
    let (animation, _, decoded) = decode(&kept);
    assert_eq!(animation.frame_durations_ms, [100, 50, 70]);
    for (ours, want) in decoded.iter().zip(&frames) {
        assert_eq!(ours, want);
    }

    // Five frames of a small moving square cost far less than five frames.
    let all = common::encode(
        &scene,
        &scene.frames,
        &scene.durations,
        JxlEncoder::lossless(),
    );
    let first = common::encode(
        &scene,
        &scene.frames[..1],
        &scene.durations[..1],
        JxlEncoder::lossless(),
    );
    assert!(
        all.len() < first.len() * 3,
        "{} bytes against {} for one frame",
        all.len(),
        first.len()
    );
}

#[test]
fn metadata_goes_in_the_container_and_the_protocol_is_enforced() {
    let scene = common::scenes().remove(1);
    let mut encoder = JxlEncoder::lossless();
    let mut out = Vec::new();
    encoder.set_animation(0).unwrap();
    encoder.set_xmp(Some(b"<x:xmpmeta/>")).unwrap();
    encoder.write_header(&scene.descriptor, &mut out).unwrap();
    assert!(encoder.set_xmp(None).is_err());
    assert!(
        encoder.write_row(&[0; 3], &mut out).is_err(),
        "a row outside a frame"
    );
    encoder.begin_frame(10, &mut out).unwrap();
    for row in scene.frames[0].chunks_exact(scene.descriptor.row_bytes()) {
        encoder.write_row(row, &mut out).unwrap();
    }
    encoder.end_frame(&mut out).unwrap();
    encoder.finish(&mut out).unwrap();
    let decoder = JxlDecoder::new(&out[..], Limits::default()).unwrap();
    assert_eq!(decoder.xmp(), Some(&b"<x:xmpmeta/>"[..]));

    let mut still = JxlEncoder::new();
    let mut sink = Vec::new();
    still.write_header(&scene.descriptor, &mut sink).unwrap();
    assert_eq!(
        still.begin_frame(1, &mut sink).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    let mut empty = JxlEncoder::lossless();
    empty.set_animation(0).unwrap();
    empty.write_header(&scene.descriptor, &mut sink).unwrap();
    assert!(empty.finish(&mut sink).is_err(), "no frames");
}

#[test]
fn still_output_is_unchanged_by_the_animation_support() {
    let cases = [
        (
            JxlEncoder::new(),
            PixelFormat::Rgb8,
            37,
            29,
            0xa8a2_6490_b249_292f_u64,
        ),
        (
            JxlEncoder::lossless(),
            PixelFormat::Rgba8,
            37,
            29,
            0x817b_6e4e_c49d_d85b,
        ),
        (
            JxlEncoder::lossless(),
            PixelFormat::Gray16,
            300,
            270,
            0x0153_19be_19e4_388e,
        ),
        (
            JxlEncoder::new(),
            PixelFormat::Rgb8,
            300,
            270,
            0x8875_8c8e_8991_584f,
        ),
    ];
    for (mut encoder, format, w, h, pinned) in cases {
        let descriptor = ImageDescriptor::new(w, h, format).unwrap();
        let raster: Vec<u8> = (0..descriptor.byte_len().unwrap())
            .map(|i| (i * 37 % 251) as u8)
            .collect();
        let mut out = Vec::new();
        encoder.write_header(&descriptor, &mut out).unwrap();
        for row in raster.chunks_exact(descriptor.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        let hash = out.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        assert_eq!(hash, pinned, "{format} {w}x{h}: still bytes changed");
    }
}

#[test]
fn frames_differing_only_under_transparency_are_both_kept() {
    // Lossless JPEG XL keeps the colour of a transparent pixel, so two
    // frames that look alike but differ there are two frames; identical
    // ones are one.
    let descriptor = ImageDescriptor::new(22, 39, PixelFormat::Rgba8).unwrap();
    let blank = vec![0_u8; descriptor.byte_len().unwrap()];
    let mut tinted = blank.clone();
    tinted[0] = 200; // red under alpha 0
    for (second, frames) in [(blank.clone(), 1), (tinted, 2)] {
        let scene = common::Scene {
            name: "transparent",
            descriptor,
            frames: vec![blank.clone(), second],
            durations: vec![10, 10],
            loop_count: 0,
        };
        let bytes = common::encode(
            &scene,
            &scene.frames,
            &scene.durations,
            JxlEncoder::lossless(),
        );
        let mut decoder = JxlDecoder::new(&bytes[..], Limits::default()).unwrap();
        let decoded = otf_pixels_core::testing::read_frames(&mut decoder).unwrap();
        assert_eq!(decoded.len(), frames);
        for (got, want) in decoded.iter().zip(&scene.frames) {
            assert_eq!(&got.1, want);
        }
    }
}
