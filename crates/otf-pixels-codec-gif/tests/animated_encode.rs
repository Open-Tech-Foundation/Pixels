//! Animated GIF encoding, read back by this crate's decoder.
//!
//! `scripts/check-gif-interop.sh` hands the same files to libgif; this
//! checks the encoder's own promises: every frame exact for scenes a
//! palette can hold, timing kept to the running total, duplicate frames
//! merged, disposal clearing what transparency cannot, and frames that hold
//! only what changed.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

mod common;

use otf_pixels_codec_gif::{GifDecoder, GifEncoder};
use otf_pixels_core::{
    Decoder, EncodeOptions, Encoder, ErrorCode, ImageDescriptor, Limits, PixelFormat,
};

fn decode(bytes: &[u8]) -> (Option<otf_pixels_core::Animation>, Vec<Vec<u8>>) {
    let mut decoder = GifDecoder::new(bytes, Limits::default()).unwrap();
    let animation = decoder.animation();
    let frames = otf_pixels_core::testing::read_frames(&mut decoder)
        .unwrap()
        .into_iter()
        .map(|(_, pixels)| pixels)
        .collect();
    (animation, frames)
}

#[test]
fn every_frame_comes_back_exactly_with_its_timing() {
    for scene in common::scenes() {
        let bytes = common::encode(
            &scene.descriptor,
            &scene.frames,
            &scene.durations,
            scene.loop_count,
            &common::exact(),
        );
        let (animation, frames) = decode(&bytes);
        let animation = animation.unwrap_or_else(|| panic!("{}: not animated", scene.name));
        assert_eq!(animation.loop_count, scene.loop_count, "{}", scene.name);
        assert_eq!(frames.len(), scene.frames.len(), "{}", scene.name);
        for (k, (ours, want)) in frames.iter().zip(&scene.frames).enumerate() {
            if let Some(i) = ours.iter().zip(want).position(|(a, b)| a != b) {
                panic!(
                    "{} frame {k}: pixel {} is {:?}, wanted {:?}",
                    scene.name,
                    i / 4,
                    &ours[i / 4 * 4..i / 4 * 4 + 4],
                    &want[i / 4 * 4..i / 4 * 4 + 4]
                );
            }
        }
        // Centiseconds, rounded on the running total.
        let total: u32 = scene.durations.iter().sum();
        assert_eq!(
            animation.duration_ms(),
            u64::from((total + 5) / 10 * 10),
            "{}",
            scene.name
        );
    }
}

#[test]
fn delays_round_the_running_total_not_each_frame() {
    let scene = &common::scenes()[0];
    let bytes = common::encode(
        &scene.descriptor,
        &scene.frames,
        &scene.durations,
        0,
        &common::exact(),
    );
    let (animation, _) = decode(&bytes);
    // 100, 40, 70, 33, 33, 34 ms: totals 100, 140, 210, 243, 276, 310.
    assert_eq!(
        animation.unwrap().frame_durations_ms,
        [100, 40, 70, 30, 40, 30]
    );
}

#[test]
fn a_repeated_frame_lengthens_the_one_before_unless_kept() {
    let scene = &common::scenes()[0];
    let frames = [
        scene.frames[0].clone(),
        scene.frames[0].clone(),
        scene.frames[1].clone(),
        scene.frames[1].clone(),
        scene.frames[1].clone(),
    ];
    let durations = [100, 50, 100, 20, 30];
    let merged = common::encode(&scene.descriptor, &frames, &durations, 0, &common::exact());
    let (animation, decoded) = decode(&merged);
    assert_eq!(decoded.len(), 2);
    assert_eq!(animation.unwrap().frame_durations_ms, [150, 150]);

    let kept_options = common::exact().with_keep_duplicate_frames(true);
    let kept = common::encode(&scene.descriptor, &frames, &durations, 0, &kept_options);
    let (animation, decoded) = decode(&kept);
    assert_eq!(decoded.len(), 5);
    assert_eq!(animation.unwrap().frame_durations_ms, durations);
    for (ours, want) in decoded.iter().zip(&frames) {
        assert_eq!(ours, want);
    }
}

#[test]
fn frames_hold_only_what_changed() {
    // Twenty frames of one pixel moving across a busy, still picture: each
    // later frame is a tiny rectangle, so the file is a fraction of twenty
    // whole frames.
    let descriptor = ImageDescriptor::new(64, 48, PixelFormat::Rgb8).unwrap();
    let mut base = Vec::new();
    for i in 0..64 * 48 {
        base.extend_from_slice(&[(i % 7 * 30) as u8, (i % 5 * 40) as u8, (i % 3 * 80) as u8]);
    }
    let frames: Vec<Vec<u8>> = (0..20)
        .map(|k| {
            let mut f = base.clone();
            let at = (10 * 64 + k * 3) * 3;
            f[at..at + 3].copy_from_slice(&[255, 255, 255]);
            f
        })
        .collect();
    let animated = common::encode(&descriptor, &frames, &[50; 20], 0, &common::exact());
    let mut still = GifEncoder::from_options(&common::exact());
    let mut one = Vec::new();
    still.write_header(&descriptor, &mut one).unwrap();
    for row in base.chunks_exact(descriptor.row_bytes()) {
        still.write_row(row, &mut one).unwrap();
    }
    still.finish(&mut one).unwrap();
    assert!(
        animated.len() < one.len() * 2,
        "{} bytes for 20 frames against {} for one",
        animated.len(),
        one.len()
    );
    let (_, decoded) = decode(&animated);
    for (k, (ours, want)) in decoded.iter().zip(&frames).enumerate() {
        let rgb: Vec<u8> = ours
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        assert_eq!(&rgb, want, "frame {k}");
    }
}

#[test]
fn an_inter_frame_error_leaves_near_misses_showing() {
    // The second frame is the first, every channel one brighter: within an
    // error of 1, nothing is redrawn and it merges into the first.
    let descriptor = ImageDescriptor::new(8, 8, PixelFormat::Rgb8).unwrap();
    let a = vec![100_u8; 8 * 8 * 3];
    let b = vec![101_u8; 8 * 8 * 3];
    let loose = common::exact().with_inter_frame_max_error(1.0).unwrap();
    let (_, decoded) = decode(&common::encode(
        &descriptor,
        &[a.clone(), b.clone()],
        &[10, 10],
        0,
        &loose,
    ));
    assert_eq!(decoded.len(), 1);
    let (_, decoded) = decode(&common::encode(
        &descriptor,
        &[a, b],
        &[10, 10],
        0,
        &common::exact(),
    ));
    assert_eq!(decoded.len(), 2);
    assert!(
        decoded[1]
            .chunks_exact(4)
            .all(|p| p == [101, 101, 101, 255])
    );
}

#[test]
fn a_close_enough_global_palette_is_reused() {
    // Frame 1 is all 100, frame 2 all 101: mapping frame 2 onto the global
    // palette costs an RMS of 1.7, inside the default interPaletteMaxError
    // of 3, so it is drawn in the global palette's 100 rather than given a
    // table of its own; at 0 it keeps its colour.
    let descriptor = ImageDescriptor::new(8, 8, PixelFormat::Rgb8).unwrap();
    let frames = [vec![100_u8; 8 * 8 * 3], vec![101_u8; 8 * 8 * 3]];
    let reuse = EncodeOptions::default().with_dither(0.0).unwrap();
    let reused = common::encode(&descriptor, &frames, &[10, 10], 0, &reuse);
    let (_, decoded) = decode(&reused);
    assert!(
        decoded[1]
            .chunks_exact(4)
            .all(|p| p == [100, 100, 100, 255])
    );
    let own = common::encode(&descriptor, &frames, &[10, 10], 0, &common::exact());
    let (_, decoded) = decode(&own);
    assert!(
        decoded[1]
            .chunks_exact(4)
            .all(|p| p == [101, 101, 101, 255])
    );
    assert!(own.len() > reused.len(), "a local table costs bytes");
}

#[test]
fn a_photographic_animation_stays_close_and_dithers_only_drawn_pixels() {
    let descriptor = ImageDescriptor::new(48, 32, PixelFormat::Rgb8).unwrap();
    let frames: Vec<Vec<u8>> = (0..4_u32)
        .map(|k| {
            let mut f = Vec::new();
            for y in 0..32_u32 {
                for x in 0..48_u32 {
                    f.extend_from_slice(&[
                        (x * 5 + k * 9) as u8,
                        (y * 7) as u8,
                        ((x + y) * 3) as u8,
                    ]);
                }
            }
            f
        })
        .collect();
    let bytes = common::encode(&descriptor, &frames, &[40; 4], 0, &EncodeOptions::default());
    let (_, decoded) = decode(&bytes);
    assert_eq!(decoded.len(), 4);
    for (k, (ours, want)) in decoded.iter().zip(&frames).enumerate() {
        let error: u64 = ours
            .chunks_exact(4)
            .zip(want.chunks_exact(3))
            .map(|(a, b)| (0..3).map(|c| u64::from(a[c].abs_diff(b[c]))).sum::<u64>())
            .sum();
        let mean = error as f64 / (48.0 * 32.0 * 3.0);
        assert!(mean < 8.0, "frame {k}: mean error {mean}");
        assert!(
            ours.chunks_exact(4).all(|p| p[3] == 255),
            "frame {k} has holes"
        );
    }
}

#[test]
fn the_frame_protocol_is_enforced() {
    let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Rgb8).unwrap();
    let mut sink = Vec::new();
    let code = |r: otf_pixels_core::Result<()>| r.unwrap_err().code();

    let mut still = GifEncoder::new();
    still.write_header(&descriptor, &mut sink).unwrap();
    assert_eq!(
        code(still.begin_frame(10, &mut sink)),
        ErrorCode::InvalidArgument
    );
    assert_eq!(code(still.set_animation(0)), ErrorCode::InvalidArgument);

    let mut animated = GifEncoder::new();
    animated.set_animation(0).unwrap();
    animated.write_header(&descriptor, &mut sink).unwrap();
    assert!(
        animated.write_row(&[0; 6], &mut sink).is_err(),
        "a row outside a frame"
    );
    assert!(animated.finish(&mut sink).is_err(), "no frames");
    animated.begin_frame(10, &mut sink).unwrap();
    assert!(animated.begin_frame(10, &mut sink).is_err(), "already open");
    animated.write_row(&[0; 6], &mut sink).unwrap();
    assert_eq!(code(animated.end_frame(&mut sink)), ErrorCode::Malformed);
}

#[test]
fn still_output_is_unchanged_by_the_animation_support() {
    // The bytes a still encoded to before animation existed, pinned.
    let descriptor = ImageDescriptor::new(5, 3, PixelFormat::Rgb8).unwrap();
    let raster: Vec<u8> = (0..45).map(|i| (i * 37 % 251) as u8).collect();
    let mut encoder = GifEncoder::new();
    let mut out = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();
    for row in raster.chunks_exact(15) {
        encoder.write_row(row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    assert_eq!(fnv(&out), STILL_HASH, "still GIF bytes changed");
}

/// FNV-1a 64 of the still above, taken before animation support.
const STILL_HASH: u64 = 0x4ef2_dbf6_1291_0017;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}
