//! The multi-frame pipeline (ADR-0020) through the facade.
//!
//! An animated pipeline replays one graph per frame. That is checked here
//! against the decoders' own frames: whatever a pipeline does to frame `k`
//! must be exactly what the same ops do to that frame opened as a still.
//! No built-in encoder writes animations yet, so frames are written through
//! [`FrameWriter`], an encoder that only records them, by
//! [`Output::write_with_encoder`].

#![cfg(all(
    feature = "gif",
    feature = "webp",
    feature = "jxl",
    feature = "png",
    feature = "raw"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{
    Decoder, EncodeOptions, Encoder, ErrorCode, Format, GifDecoder, Image, ImageDescriptor,
    JxlDecoder, Limits, OpenOptions, Output, PixelFormat, Result, Sink, Trim, WebPDecoder,
};

fn codec_fixture(codec: &str, name: &str) -> String {
    format!(
        "{}/../otf-pixels-codec-{codec}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

const GIF: (&str, &str) = ("gif", "animation.gif");
const WEBP: (&str, &str) = ("webp", "animated/anim_flags.webp");
const JXL: (&str, &str) = ("jxl", "anim_sparse.jxl");
/// Frames that cover only part of a transparent canvas, each a different
/// part.
const PARTIAL: (&str, &str) = ("webp", "animated/anim_partial_key.webp");

fn animated() -> OpenOptions {
    OpenOptions::default().with_animated(true)
}

fn open(fixture: (&str, &str), options: OpenOptions) -> Image {
    Image::open_with(codec_fixture(fixture.0, fixture.1), options).unwrap()
}

/// Every frame the format's own decoder gives, as RGBA8 stills.
fn decoder_frames(fixture: (&str, &str)) -> (ImageDescriptor, Vec<Vec<u8>>) {
    let bytes = std::fs::read(codec_fixture(fixture.0, fixture.1)).unwrap();
    let mut decoder: Box<dyn Decoder> = match fixture.0 {
        "gif" => Box::new(GifDecoder::new(&bytes[..], Limits::default()).unwrap()),
        "webp" => Box::new(WebPDecoder::new(&bytes[..], Limits::default()).unwrap()),
        "jxl" => Box::new(JxlDecoder::new(&bytes[..], Limits::default()).unwrap()),
        other => panic!("no fixture codec {other}"),
    };
    let descriptor = decoder.descriptor();
    let frames = otf_pixels_core::testing::read_frames(decoder.as_mut())
        .unwrap()
        .into_iter()
        .map(|(_, pixels)| pixels)
        .collect();
    (descriptor, frames)
}

fn still(descriptor: ImageDescriptor, pixels: &[u8]) -> Image {
    Image::from_raw(descriptor, pixels.to_vec()).unwrap()
}

fn raw(image: Image) -> Vec<u8> {
    image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap()
}

/// An animated encoder that writes, per frame, its duration (`u32` little
/// endian) and its raw rows; [`frames`] reads that back.
#[derive(Debug, Default)]
struct FrameWriter {
    loop_count: Option<u32>,
    frame_bytes: usize,
    in_frame: usize,
}

impl Encoder for FrameWriter {
    fn supports_animation(&self) -> bool {
        true
    }
    fn set_animation(&mut self, loop_count: u32) -> Result<()> {
        self.loop_count = Some(loop_count);
        Ok(())
    }
    fn write_header(&mut self, desc: &ImageDescriptor, sink: &mut dyn Sink) -> Result<()> {
        assert!(self.loop_count.is_some(), "set_animation comes first");
        self.frame_bytes = desc.byte_len().unwrap();
        sink.write_all(&self.loop_count.unwrap_or(0).to_le_bytes())
    }
    fn begin_frame(&mut self, duration_ms: u32, sink: &mut dyn Sink) -> Result<()> {
        self.in_frame = 0;
        sink.write_all(&duration_ms.to_le_bytes())
    }
    fn write_row(&mut self, row: &[u8], sink: &mut dyn Sink) -> Result<()> {
        self.in_frame += row.len();
        sink.write_all(row)
    }
    fn end_frame(&mut self, _: &mut dyn Sink) -> Result<()> {
        assert_eq!(self.in_frame, self.frame_bytes, "a frame ended short");
        Ok(())
    }
    fn finish(&mut self, _: &mut dyn Sink) -> Result<()> {
        Ok(())
    }
}

/// What a [`FrameWriter`] wrote: the loop count, then `(duration, pixels)`
/// per frame.
fn frames(bytes: &[u8], frame_bytes: usize) -> (u32, Vec<(u32, Vec<u8>)>) {
    let loop_count = u32::from_le_bytes(bytes[..4].try_into().unwrap());
    let frames = bytes[4..]
        .chunks(4 + frame_bytes)
        .map(|chunk| {
            assert_eq!(chunk.len(), 4 + frame_bytes, "a frame was cut short");
            (
                u32::from_le_bytes(chunk[..4].try_into().unwrap()),
                chunk[4..].to_vec(),
            )
        })
        .collect();
    (loop_count, frames)
}

/// Write `image` frame by frame and read the frames back.
fn write_frames(output: Output, frame_bytes: usize) -> (u32, Vec<(u32, Vec<u8>)>) {
    let mut bytes = Vec::new();
    output
        .write_with_encoder(&mut FrameWriter::default(), &mut bytes)
        .unwrap();
    frames(&bytes, frame_bytes)
}

#[test]
fn every_frame_gets_the_same_ops_as_that_frame_opened_alone() {
    for fixture in [GIF, WEBP, JXL] {
        let (descriptor, expected) = decoder_frames(fixture);
        let pipeline = |image: Image| image.resize(17, 11).flop().blur(0.8);
        let image = pipeline(open(fixture, animated()));
        let timing = image.animation().unwrap().clone();
        let frame_bytes = image.descriptor().unwrap().byte_len().unwrap();
        let (loop_count, written) = write_frames(
            image.output(Format::Raw, EncodeOptions::default()),
            frame_bytes,
        );
        assert_eq!(loop_count, timing.loop_count, "{fixture:?}");
        assert_eq!(written.len(), expected.len(), "{fixture:?}: frame count");
        for (k, ((duration, pixels), frame)) in written.iter().zip(&expected).enumerate() {
            assert_eq!(
                *duration, timing.frame_durations_ms[k],
                "{fixture:?} frame {k}"
            );
            let alone = raw(pipeline(still(descriptor, frame)));
            assert!(
                *pixels == alone,
                "{fixture:?} frame {k} differs from it opened alone"
            );
        }
    }
}

#[test]
fn the_scheduler_and_the_reference_evaluator_agree_frame_for_frame() {
    for fixture in [GIF, WEBP, JXL] {
        let pipeline = || {
            open(fixture, animated())
                .rotate_with(20.0, otf_pixels::RotateOptions::default())
                .resize(23, 19)
                .output(Format::Raw, EncodeOptions::default())
        };
        let mut scheduled = Vec::new();
        pipeline()
            .threads(4)
            .write_with_encoder(&mut FrameWriter::default(), &mut scheduled)
            .unwrap();
        let reference = pipeline()
            .bytes_via_reference_with(&mut FrameWriter::default())
            .unwrap();
        assert!(scheduled == reference, "{fixture:?}");
    }
}

#[test]
fn a_still_overlay_from_a_stream_is_drawn_on_every_frame() {
    // The watermark is a PNG read from a forward-only stream: replaying it
    // per frame would fail on the second, so it is evaluated once.
    let mark = Image::from_raw(
        ImageDescriptor::new(6, 5, PixelFormat::Rgba8).unwrap(),
        [250, 250, 0, 200].repeat(30),
    )
    .unwrap()
    .output(Format::Png, EncodeOptions::default())
    .bytes()
    .unwrap();
    let watermark = || Image::from_stream(std::io::Cursor::new(mark.clone())).unwrap();
    let (descriptor, expected) = decoder_frames(GIF);
    let image = open(GIF, animated()).composite(watermark(), 3, 4);
    let frame_bytes = image.descriptor().unwrap().byte_len().unwrap();
    let (_, written) = write_frames(
        image.output(Format::Raw, EncodeOptions::default()),
        frame_bytes,
    );
    assert_eq!(written.len(), 4);
    for (k, (_, pixels)) in written.iter().enumerate() {
        let alone = raw(still(descriptor, &expected[k]).composite(watermark(), 3, 4));
        assert!(*pixels == alone, "frame {k}");
    }
}

#[test]
fn an_animated_overlay_must_be_made_a_still_first() {
    let error = open(GIF, OpenOptions::default())
        .composite(open(GIF, animated()), 0, 0)
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported);
    assert!(error.to_string().contains("first_frame()"), "{error}");
    let fine =
        open(GIF, OpenOptions::default()).composite(open(GIF, animated()).first_frame(), 0, 0);
    assert!(
        fine.output(Format::Png, EncodeOptions::default())
            .bytes()
            .is_ok()
    );
}

#[test]
fn frame_picks_one_still_with_the_ops_so_far() {
    for fixture in [GIF, WEBP, JXL] {
        let (descriptor, expected) = decoder_frames(fixture);
        let count = expected.len() as u32;
        for k in [0, 1, count - 1] {
            let picked = open(fixture, animated()).resize(9, 7).frame(k);
            assert!(!picked.is_animated());
            assert_eq!(picked.frame_count(), 1);
            // A still format takes it.
            assert!(
                picked
                    .clone()
                    .output(Format::Png, EncodeOptions::default())
                    .bytes()
                    .is_ok()
            );
            let alone = raw(still(descriptor, &expected[k as usize]).resize(9, 7));
            assert!(raw(picked) == alone, "{fixture:?} frame {k}");
        }
        assert_eq!(
            raw(open(fixture, animated()).first_frame()),
            raw(still(descriptor, &expected[0])),
            "{fixture:?}"
        );
        let past = open(fixture, animated())
            .frame(count)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap_err();
        assert_eq!(past.code(), ErrorCode::InvalidArgument, "{fixture:?}");
    }
    // A still has only frame 0.
    let still_gif = || Image::open(codec_fixture("gif", "static.gif")).unwrap();
    assert_eq!(raw(still_gif().first_frame()), raw(still_gif()));
    assert!(still_gif().frame(1).descriptor().is_err());
}

#[test]
fn into_strip_stacks_every_frame_and_later_ops_see_the_strip() {
    for fixture in [GIF, WEBP, JXL] {
        let (descriptor, expected) = decoder_frames(fixture);
        let strip = open(fixture, animated()).flip().into_strip();
        assert!(!strip.is_animated());
        let shape = strip.descriptor().unwrap();
        assert_eq!(
            (shape.width, shape.height),
            (descriptor.width, descriptor.height * expected.len() as u32),
            "{fixture:?}"
        );
        let stacked: Vec<u8> = expected
            .iter()
            .flat_map(|frame| raw(still(descriptor, frame).flip()))
            .collect();
        assert!(raw(strip.clone()) == stacked, "{fixture:?}");
        // After the strip, one image: a resize scales the whole sheet.
        let half = strip.resize(descriptor.width / 2, shape.height / 2);
        assert_eq!(half.descriptor().unwrap().height, shape.height / 2);
        assert!(
            half.output(Format::Png, EncodeOptions::default())
                .bytes()
                .is_ok()
        );
    }
}

#[test]
fn trim_cuts_the_box_that_holds_every_frame() {
    // Each frame covers a different part of a transparent canvas: alone,
    // each would trim to a different box; the animation keeps one for all.
    let (descriptor, expected) = decoder_frames(PARTIAL);
    let trim = Trim::default().with_background([0, 0, 0, 0]);
    let trimmed = open(PARTIAL, animated()).trim(trim);
    assert!(trimmed.is_animated(), "trim keeps the frames");
    let (x, y) = trimmed.trim_offset().unwrap();
    let cut = trimmed.descriptor().unwrap();

    // The union of each frame's own box.
    let (mut left, mut top, mut right, mut bottom) = (u32::MAX, u32::MAX, 0, 0);
    for frame in &expected {
        let one = still(descriptor, frame).trim(trim);
        let (fx, fy) = one.trim_offset().unwrap();
        let size = one.descriptor().unwrap();
        left = left.min(fx);
        top = top.min(fy);
        right = right.max(fx + size.width);
        bottom = bottom.max(fy + size.height);
    }
    assert_eq!(
        (x, y, cut.width, cut.height),
        (left, top, right - left, bottom - top)
    );
    assert!(
        cut.width < descriptor.width || cut.height < descriptor.height,
        "nothing trimmed"
    );

    let (_, written) = write_frames(
        trimmed.output(Format::Raw, EncodeOptions::default()),
        cut.byte_len().unwrap(),
    );
    for (k, (_, pixels)) in written.iter().enumerate() {
        let alone = raw(still(descriptor, &expected[k]).crop(x, y, cut.width, cut.height));
        assert!(*pixels == alone, "frame {k}");
    }
}

#[test]
fn stats_cover_every_frame_as_the_strip_does() {
    for fixture in [GIF, WEBP] {
        let animated_stats = open(fixture, animated()).stats().unwrap();
        let strip_stats = open(fixture, animated()).into_strip().stats().unwrap();
        assert_eq!(animated_stats, strip_stats, "{fixture:?}");
        let first = open(fixture, OpenOptions::default()).stats().unwrap();
        assert_ne!(
            animated_stats, first,
            "{fixture:?}: only the first frame was counted"
        );
    }
}

#[test]
fn a_frame_range_selects_frames_and_their_timing() {
    let (descriptor, expected) = decoder_frames(WEBP);
    let total = expected.len() as u32;
    let all = open(WEBP, animated())
        .animation()
        .unwrap()
        .frame_durations_ms
        .clone();

    let range = open(WEBP, animated().with_frames(2, Some(3)));
    assert_eq!(range.frame_count(), 3);
    let (_, written) = write_frames(
        range.output(Format::Raw, EncodeOptions::default()),
        descriptor.byte_len().unwrap(),
    );
    assert_eq!(written.len(), 3);
    for (k, (duration, pixels)) in written.iter().enumerate() {
        assert_eq!(*duration, all[2 + k]);
        assert!(*pixels == expected[2 + k], "frame {}", 2 + k);
    }
    // From a frame to the end.
    assert_eq!(
        open(WEBP, animated().with_frames(4, None)).frame_count(),
        total - 4
    );
    // One frame of an animation is a still, with or without `animated`.
    for options in [
        animated().with_frames(5, Some(1)),
        OpenOptions::default().with_frames(5, None),
    ] {
        let one = open(WEBP, options);
        assert!(!one.is_animated());
        assert!(raw(one) == expected[5]);
    }

    let refused = |options: OpenOptions| {
        Image::open_with(codec_fixture(WEBP.0, WEBP.1), options)
            .unwrap_err()
            .code()
    };
    assert_eq!(
        refused(animated().with_frames(total, None)),
        ErrorCode::InvalidArgument
    );
    assert_eq!(
        refused(animated().with_frames(1, Some(total))),
        ErrorCode::InvalidArgument
    );
    assert_eq!(
        refused(animated().with_frames(0, Some(0))),
        ErrorCode::InvalidArgument
    );
    assert_eq!(
        refused(OpenOptions::default().with_frames(0, Some(2))),
        ErrorCode::InvalidArgument,
        "two frames need `animated`"
    );
    // A format without frames has only frame 0.
    let still_bytes = still(descriptor, &expected[0])
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let error = Image::from_stream_with(
        std::io::Cursor::new(still_bytes),
        OpenOptions::default().with_frames(1, None),
    )
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidArgument);
}

#[test]
fn animation_limits_are_checked_at_open_and_only_when_frames_are_asked_for() {
    let tight = Limits::default().with_max_frames(3);
    let error =
        Image::open_with(codec_fixture(GIF.0, GIF.1), animated().with_limits(tight)).unwrap_err();
    assert_eq!(error.code(), ErrorCode::LimitExceeded);
    // Three of the four frames fit; the first frame alone always does.
    assert!(
        Image::open_with(
            codec_fixture(GIF.0, GIF.1),
            animated().with_limits(tight).with_frames(1, None)
        )
        .is_ok()
    );
    assert!(
        Image::open_with(
            codec_fixture(GIF.0, GIF.1),
            OpenOptions::default().with_limits(tight)
        )
        .is_ok()
    );
    let pixels = Limits::default().with_max_animation_pixels(20 * 20 * 3);
    assert_eq!(
        Image::open_with(codec_fixture(GIF.0, GIF.1), animated().with_limits(pixels))
            .unwrap_err()
            .code(),
        ErrorCode::LimitExceeded
    );
}

#[test]
fn clones_of_an_animated_pipeline_run_independently() {
    // Each pass decodes the file afresh, so one output does not use up the
    // frames another needs.
    let image = open(GIF, animated()).resize(10, 10);
    let frame_bytes = image.descriptor().unwrap().byte_len().unwrap();
    let first = write_frames(
        image.clone().output(Format::Raw, EncodeOptions::default()),
        frame_bytes,
    );
    let second = write_frames(
        image.clone().output(Format::Raw, EncodeOptions::default()),
        frame_bytes,
    );
    assert_eq!(first, second);
    let _ = image.stats().unwrap();
    let third = write_frames(
        image.output(Format::Raw, EncodeOptions::default()),
        frame_bytes,
    );
    assert_eq!(first, third);
}

/// RGBA with every pixel GIF cannot draw (alpha under one half) as zero.
fn as_gif_shows(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|p| {
            if p[3] < 128 {
                [0, 0, 0, 0]
            } else {
                [p[0], p[1], p[2], 255]
            }
        })
        .collect()
}

#[test]
fn an_animation_written_as_gif_reads_back_frame_for_frame() {
    // The GIF fixture's frames fit a palette, so with no dithering and no
    // palette reuse each frame must come back exactly as the pipeline made
    // it, timing and loop count included.
    let options = EncodeOptions::default()
        .with_colours(256)
        .unwrap()
        .with_dither(0.0)
        .unwrap()
        .with_inter_palette_max_error(0.0)
        .unwrap();
    let pipeline = || open(GIF, animated()).flop();
    let timing = pipeline().animation().unwrap().clone();
    let frame_bytes = pipeline().descriptor().unwrap().byte_len().unwrap();
    let (_, made) = write_frames(pipeline().output(Format::Raw, options), frame_bytes);

    let gif = pipeline().output(Format::Gif, options).bytes().unwrap();
    let back = Image::from_stream_with(std::io::Cursor::new(gif.clone()), animated()).unwrap();
    assert!(back.is_animated());
    assert_eq!(back.animation().unwrap().loop_count, timing.loop_count);
    assert_eq!(
        back.animation().unwrap().frame_durations_ms,
        timing.frame_durations_ms
    );
    let (_, read) = write_frames(back.output(Format::Raw, options), frame_bytes);
    assert_eq!(read.len(), made.len());
    for (k, ((_, ours), (_, theirs))) in read.iter().zip(&made).enumerate() {
        assert!(*ours == as_gif_shows(theirs), "frame {k}");
    }
}

#[test]
fn a_webp_animation_converts_to_gif() {
    let gif = open(WEBP, animated())
        .resize(30, 22)
        .output(Format::Gif, EncodeOptions::default())
        .bytes()
        .unwrap();
    let back = Image::from_stream_with(std::io::Cursor::new(gif), animated()).unwrap();
    let source = open(WEBP, animated());
    assert_eq!(back.frame_count(), source.frame_count());
    assert_eq!(
        back.animation().unwrap().duration_ms(),
        source.animation().unwrap().duration_ms()
    );
    assert_eq!(
        (
            back.descriptor().unwrap().width,
            back.descriptor().unwrap().height
        ),
        (30, 22)
    );
}

#[test]
fn an_animation_written_as_lossless_webp_reads_back_exactly() {
    // From every animated format: lossless WebP keeps each frame exactly,
    // and its timing and loop count.
    let lossless = EncodeOptions::default().with_lossless(true);
    for fixture in [GIF, WEBP, JXL] {
        let pipeline = || open(fixture, animated()).resize(21, 15);
        let timing = pipeline().animation().unwrap().clone();
        let frame_bytes = pipeline().descriptor().unwrap().byte_len().unwrap();
        let (_, made) = write_frames(pipeline().output(Format::Raw, lossless), frame_bytes);

        let webp = pipeline().output(Format::WebP, lossless).bytes().unwrap();
        let back = Image::from_stream_with(std::io::Cursor::new(webp), animated()).unwrap();
        assert!(back.is_animated(), "{fixture:?}");
        assert_eq!(
            back.animation().unwrap().frame_durations_ms,
            timing.frame_durations_ms
        );
        assert_eq!(back.animation().unwrap().loop_count, timing.loop_count);
        let back = back.to_pixel_format(PixelFormat::Rgba8);
        let (_, read) = write_frames(back.output(Format::Raw, lossless), frame_bytes);
        assert_eq!(read.len(), made.len(), "{fixture:?}");
        for (k, ((_, ours), (_, theirs))) in read.iter().zip(&made).enumerate() {
            let visible = |p: &[u8]| -> Vec<u8> {
                p.chunks_exact(4)
                    .flat_map(|q| {
                        if q[3] == 0 {
                            [0; 4]
                        } else {
                            [q[0], q[1], q[2], q[3]]
                        }
                    })
                    .collect()
            };
            assert!(visible(ours) == visible(theirs), "{fixture:?} frame {k}");
        }
    }
}

#[test]
fn a_lossy_webp_animation_is_written_by_default() {
    let webp = open(GIF, animated())
        .output(Format::WebP, EncodeOptions::default())
        .bytes()
        .unwrap();
    let back = Image::from_stream_with(std::io::Cursor::new(webp), animated()).unwrap();
    assert_eq!(back.frame_count(), 4);
    assert_eq!(back.source_metadata().unwrap().compression, Some("lossy"));
}

#[test]
fn an_animation_written_as_lossless_jpeg_xl_reads_back_exactly() {
    let lossless = EncodeOptions::default().with_lossless(true);
    for fixture in [GIF, WEBP, JXL] {
        let pipeline = || open(fixture, animated()).resize(19, 13);
        let timing = pipeline().animation().unwrap().clone();
        let frame_bytes = pipeline().descriptor().unwrap().byte_len().unwrap();
        let (_, made) = write_frames(pipeline().output(Format::Raw, lossless), frame_bytes);
        let jxl = pipeline().output(Format::Jxl, lossless).bytes().unwrap();
        let back = Image::from_stream_with(std::io::Cursor::new(jxl), animated()).unwrap();
        assert_eq!(
            back.animation().unwrap().frame_durations_ms,
            timing.frame_durations_ms
        );
        assert_eq!(back.animation().unwrap().loop_count, timing.loop_count);
        let (_, read) = write_frames(back.output(Format::Raw, lossless), frame_bytes);
        assert_eq!(read.len(), made.len(), "{fixture:?}");
        for (k, ((_, ours), (_, theirs))) in read.iter().zip(&made).enumerate() {
            assert!(ours == theirs, "{fixture:?} frame {k}");
        }
    }
}
