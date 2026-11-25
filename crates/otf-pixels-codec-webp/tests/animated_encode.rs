//! Animated WebP encoding, read back by this crate's decoder — which is
//! libwebp's animation decoder frame for frame (`tests/animated.rs`).
//! `scripts/check-webp-interop.sh` hands the same files to libwebp itself.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

mod common;

use otf_pixels_codec_webp::{WebPDecoder, WebPEncoder};
use otf_pixels_core::{
    Decoder, EncodeOptions, Encoder, ErrorCode, ImageDescriptor, Limits, PixelFormat,
};

fn decode(
    bytes: &[u8],
) -> (
    Option<otf_pixels_core::Animation>,
    ImageDescriptor,
    Vec<Vec<u8>>,
) {
    let mut decoder = WebPDecoder::new(bytes, Limits::default()).unwrap();
    let animation = decoder.animation();
    let descriptor = decoder.descriptor();
    // An animation with no translucent pixel decodes as RGB: widened here.
    let frames = otf_pixels_core::testing::read_frames(&mut decoder)
        .unwrap()
        .into_iter()
        .map(|(_, pixels)| match descriptor.pixel {
            PixelFormat::Rgb8 => pixels
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
            _ => pixels,
        })
        .collect();
    (animation, descriptor, frames)
}

/// What a viewer shows: every transparent pixel the same.
fn shown(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|p| {
            if p[3] == 0 {
                [0, 0, 0, 0]
            } else {
                [p[0], p[1], p[2], p[3]]
            }
        })
        .collect()
}

#[test]
fn lossless_frames_come_back_exactly_with_their_timing() {
    for scene in common::scenes() {
        let bytes = common::encode(&scene, &scene.frames, &scene.durations, &common::lossless());
        let (animation, descriptor, frames) = decode(&bytes);
        let animation = animation.unwrap_or_else(|| panic!("{}: not animated", scene.name));
        let translucent = scene
            .frames
            .iter()
            .flatten()
            .skip(3)
            .step_by(4)
            .any(|&a| a != 255);
        let expected = if translucent {
            PixelFormat::Rgba8
        } else {
            PixelFormat::Rgb8
        };
        assert_eq!(
            descriptor.pixel, expected,
            "{}: alpha only if a frame has it",
            scene.name
        );
        assert_eq!(animation.loop_count, scene.loop_count, "{}", scene.name);
        assert_eq!(
            animation.frame_durations_ms, scene.durations,
            "{}",
            scene.name
        );
        assert_eq!(frames.len(), scene.frames.len(), "{}", scene.name);
        for (k, (ours, want)) in frames.iter().zip(&scene.frames).enumerate() {
            let (ours, want) = (shown(ours), shown(want));
            if let Some(i) = ours.iter().zip(&want).position(|(a, b)| a != b) {
                panic!(
                    "{} frame {k}: pixel {} is {:?}, wanted {:?}",
                    scene.name,
                    i / 4,
                    &ours[i / 4 * 4..i / 4 * 4 + 4],
                    &want[i / 4 * 4..i / 4 * 4 + 4]
                );
            }
        }
    }
}

/// Mean error over opaque pixels of `ours` against `want`, both RGBA.
fn mean_error(ours: &[u8], want: &[u8]) -> f64 {
    let (mut error, mut count) = (0_u64, 0_u64);
    for (a, b) in ours.chunks_exact(4).zip(want.chunks_exact(4)) {
        if b[3] == 255 {
            error += (0..3).map(|c| u64::from(a[c].abs_diff(b[c]))).sum::<u64>();
            count += 3;
        }
    }
    error as f64 / count.max(1) as f64
}

#[test]
fn lossy_frames_are_as_close_as_stills_and_keep_their_alpha() {
    // Each frame's error against what it was is no worse than the same
    // picture coded as a still: drawing only the change, and blanking only
    // whole unchanged blocks, costs no quality.
    for scene in common::scenes() {
        let options = EncodeOptions::default();
        let bytes = common::encode(&scene, &scene.frames, &scene.durations, &options);
        let (_, _, frames) = decode(&bytes);
        assert_eq!(frames.len(), scene.frames.len(), "{}", scene.name);
        for (k, (ours, want)) in frames.iter().zip(&scene.frames).enumerate() {
            for (a, b) in ours.chunks_exact(4).zip(want.chunks_exact(4)) {
                assert_eq!(
                    a[3], b[3],
                    "{} frame {k}: alpha is coded losslessly",
                    scene.name
                );
            }
            let mut still = WebPEncoder::from_options(&options);
            let mut one = Vec::new();
            still.write_header(&scene.descriptor, &mut one).unwrap();
            for row in want.chunks_exact(scene.descriptor.row_bytes()) {
                still.write_row(row, &mut one).unwrap();
            }
            still.finish(&mut one).unwrap();
            let (_, _, alone) = decode(&one);
            let (animated, by_itself) = (mean_error(ours, want), mean_error(&alone[0], want));
            assert!(
                animated <= by_itself * 1.25 + 0.5,
                "{} frame {k}: {animated} against {by_itself} as a still",
                scene.name
            );
        }
    }
}

#[test]
fn repeated_frames_merge_unless_kept() {
    let scene = &common::scenes()[0];
    let frames = [
        scene.frames[0].clone(),
        scene.frames[0].clone(),
        scene.frames[1].clone(),
        scene.frames[1].clone(),
    ];
    let durations = [100, 50, 70, 30];
    let (animation, _, decoded) = decode(&common::encode(
        scene,
        &frames,
        &durations,
        &common::lossless(),
    ));
    assert_eq!(decoded.len(), 2);
    assert_eq!(animation.unwrap().frame_durations_ms, [150, 100]);

    let keep = common::lossless().with_keep_duplicate_frames(true);
    let (animation, _, decoded) = decode(&common::encode(scene, &frames, &durations, &keep));
    assert_eq!(animation.unwrap().frame_durations_ms, durations);
    for (ours, want) in decoded.iter().zip(&frames) {
        assert_eq!(shown(ours), shown(want));
    }
}

#[test]
fn later_frames_hold_only_the_change() {
    // Six frames of a small square over a large still backdrop: the file is
    // little more than one frame of it.
    let scene = &common::scenes()[0];
    let all = common::encode(scene, &scene.frames, &scene.durations, &common::lossless());
    let first = common::encode(
        scene,
        &scene.frames[..1],
        &scene.durations[..1],
        &common::lossless(),
    );
    // Six whole frames would be six times one; each later frame here is a
    // small rectangle and its header.
    assert!(
        all.len() < first.len() * 3,
        "{} bytes for six frames against {} for one",
        all.len(),
        first.len()
    );
}

#[test]
fn the_container_carries_metadata_and_the_frame_protocol_is_enforced() {
    let scene = &common::scenes()[2];
    let mut encoder = WebPEncoder::from_options(&common::lossless());
    let mut out = Vec::new();
    encoder.set_animation(0).unwrap();
    encoder.set_xmp(Some(b"<x:xmpmeta/>")).unwrap();
    encoder.write_header(&scene.descriptor, &mut out).unwrap();
    assert!(
        encoder.set_xmp(None).is_err(),
        "metadata goes before the header"
    );
    assert!(encoder.finish(&mut out).is_err(), "no frames");
    encoder.begin_frame(10, &mut out).unwrap();
    encoder
        .write_row(&scene.frames[0][..scene.descriptor.row_bytes()], &mut out)
        .unwrap();
    assert_eq!(
        encoder.end_frame(&mut out).unwrap_err().code(),
        ErrorCode::Malformed
    );

    let bytes = {
        let mut encoder = WebPEncoder::from_options(&common::lossless());
        let mut out = Vec::new();
        encoder.set_animation(0).unwrap();
        encoder.set_xmp(Some(b"<x:xmpmeta/>")).unwrap();
        encoder.write_header(&scene.descriptor, &mut out).unwrap();
        for frame in &scene.frames {
            encoder.begin_frame(10, &mut out).unwrap();
            for row in frame.chunks_exact(scene.descriptor.row_bytes()) {
                encoder.write_row(row, &mut out).unwrap();
            }
            encoder.end_frame(&mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        out
    };
    let decoder = WebPDecoder::new(&bytes[..], Limits::default()).unwrap();
    assert_eq!(decoder.xmp(), Some(&b"<x:xmpmeta/>"[..]));

    let mut still = WebPEncoder::new();
    let mut sink = Vec::new();
    still.write_header(&scene.descriptor, &mut sink).unwrap();
    assert_eq!(
        still.begin_frame(1, &mut sink).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

#[test]
fn still_output_is_unchanged_by_the_animation_support() {
    for (options, pinned) in [
        (EncodeOptions::default(), STILL_LOSSY),
        (common::lossless(), STILL_LOSSLESS),
    ] {
        let descriptor = ImageDescriptor::new(19, 13, PixelFormat::Rgba8).unwrap();
        let raster: Vec<u8> = (0..19 * 13 * 4).map(|i| (i * 37 % 251) as u8).collect();
        let mut encoder = WebPEncoder::from_options(&options);
        let mut out = Vec::new();
        encoder.write_header(&descriptor, &mut out).unwrap();
        for row in raster.chunks_exact(descriptor.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.finish(&mut out).unwrap();
        assert_eq!(fnv(&out), pinned, "still WebP bytes changed ({options:?})");
    }
}

/// FNV-1a 64 of the stills above, taken before animation support.
const STILL_LOSSY: u64 = 0xf07a_eb77_c742_01d7;
const STILL_LOSSLESS: u64 = 0x91ae_c453_b716_89ea;

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}
