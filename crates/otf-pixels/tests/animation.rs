//! Animated input through the public facade.
//!
//! Opened plainly, an animation is its first frame, as sharp does by
//! default, and [`Image::animation`] reports what the file holds. Opened
//! with [`OpenOptions::animated`], every frame is processed; the frame
//! pipeline itself is checked in `animated_pipeline.rs`.

#![cfg(all(feature = "gif", feature = "webp", feature = "png", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, ErrorCode, Format, Image, Limits, OpenOptions};

fn codec_fixture(codec: &str, name: &str) -> String {
    format!(
        "{}/../otf-pixels-codec-{codec}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn an_animated_gif_reports_its_frames_and_processes_the_first() {
    let image = Image::open(codec_fixture("gif", "animation.gif")).unwrap();
    let animation = image.animation().unwrap().clone();
    assert_eq!(animation.frame_count, 4);
    assert_eq!(animation.loop_count, 0);
    assert_eq!(animation.frame_durations_ms, vec![100; 4]);
    // The pipeline runs on the first frame; the report survives the ops.
    let resized = image.resize(8, 8);
    assert_eq!(resized.animation(), Some(&animation));
    let png = resized
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let back = Image::from_stream(std::io::Cursor::new(png)).unwrap();
    assert_eq!(back.metadata().unwrap().width, 8);
    // What was written is a still.
    assert_eq!(back.animation(), None);
}

#[test]
fn an_animated_webp_reports_its_timing() {
    let image = Image::open(codec_fixture("webp", "animated/anim_timing.webp")).unwrap();
    let animation = image.animation().unwrap();
    assert_eq!(animation.frame_durations_ms, vec![100, 250, 40]);
    assert_eq!((animation.loop_count, animation.duration_ms()), (3, 390));
}

#[test]
fn stills_report_no_animation() {
    for (codec, name) in [("gif", "static.gif"), ("webp", "lossless/blocks_m6.webp")] {
        let image = Image::open(codec_fixture(codec, name)).unwrap();
        assert_eq!(image.animation(), None, "{name}");
    }
}

#[test]
fn an_animated_pipeline_says_which_still_to_write_rather_than_flattening() {
    let animated = OpenOptions::default().with_animated(true);
    for (codec, name, frames) in [
        ("gif", "animation.gif", 4),
        ("webp", "animated/anim_timing.webp", 3),
    ] {
        let image = Image::open_with(codec_fixture(codec, name), animated).unwrap();
        assert!(image.is_animated(), "{name}");
        assert_eq!(image.frame_count(), frames, "{name}");
        let error = image
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::Unsupported, "{name}");
        for hint in ["first_frame()", "frame(i)", "into_strip()"] {
            assert!(error.to_string().contains(hint), "{error}");
        }
    }
    // A still opens either way, and is not animated.
    let still = Image::open_with(codec_fixture("gif", "static.gif"), animated).unwrap();
    assert!(!still.is_animated());
    assert_eq!(still.frame_count(), 1);
}

#[test]
fn limits_are_set_per_open() {
    // The GIF canvas is larger than 64 pixels.
    let tight = OpenOptions::default().with_limits(Limits::default().with_max_pixels(64));
    let error = Image::open_with(codec_fixture("gif", "animation.gif"), tight).unwrap_err();
    assert_eq!(error.code(), ErrorCode::LimitExceeded);
    assert!(Image::open(codec_fixture("gif", "animation.gif")).is_ok());
}
