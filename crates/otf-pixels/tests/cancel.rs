//! Cooperative cancellation (ADR-0021) through the facade and inside every
//! codec that does its work in one piece.
//!
//! Timing is not what is tested — a slow machine would make that flaky —
//! but where a cancel lands: an encoder that cancels after row 10 or frame 2
//! stops the run there, and each buffering codec handed a cancelled flag
//! fails from inside its own work.

#![cfg(all(
    feature = "gif",
    feature = "webp",
    feature = "jxl",
    feature = "avif",
    feature = "jpeg",
    feature = "png",
    feature = "raw"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use std::time::Duration;

use otf_pixels::{
    AvifDecoder, AvifEncoder, Cancel, Decoder, EncodeOptions, Encoder, ErrorCode, Format,
    GifDecoder, GifEncoder, Image, ImageDescriptor, JpegDecoder, JpegEncoder, JxlDecoder,
    JxlEncoder, Limits, OpenOptions, PixelFormat, Result, Sink, Trim, WebPDecoder, WebPEncoder,
};

fn picture(width: u32, height: u32) -> (ImageDescriptor, Vec<u8>) {
    let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();
    let bytes = (0..descriptor.byte_len().unwrap())
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    (descriptor, bytes)
}

fn image() -> Image {
    let (descriptor, bytes) = picture(300, 200);
    Image::from_raw(descriptor, bytes).unwrap()
}

fn cancelled() -> Cancel {
    let cancel = Cancel::new();
    cancel.cancel();
    cancel
}

fn code<T>(result: Result<T>) -> ErrorCode {
    match result {
        Ok(_) => panic!("expected the run to be cancelled"),
        Err(error) => error.code(),
    }
}

/// An encoder that records rows and frames, and cancels `cancel` once it
/// has seen `after` of them.
struct Canceller {
    cancel: Cancel,
    after: u32,
    seen: u32,
    animated: bool,
}

impl Encoder for Canceller {
    fn supports_animation(&self) -> bool {
        true
    }
    fn set_animation(&mut self, _: u32) -> Result<()> {
        self.animated = true;
        Ok(())
    }
    fn write_header(&mut self, _: &ImageDescriptor, _: &mut dyn Sink) -> Result<()> {
        Ok(())
    }
    fn begin_frame(&mut self, _: u32, _: &mut dyn Sink) -> Result<()> {
        self.seen += 1;
        if self.seen == self.after {
            self.cancel.cancel();
        }
        Ok(())
    }
    fn write_row(&mut self, _: &[u8], _: &mut dyn Sink) -> Result<()> {
        if !self.animated {
            self.seen += 1;
            if self.seen == self.after {
                self.cancel.cancel();
            }
        }
        Ok(())
    }
    fn end_frame(&mut self, _: &mut dyn Sink) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self, _: &mut dyn Sink) -> Result<()> {
        Ok(())
    }
}

#[test]
fn a_cancelled_image_fails_every_terminal_with_cancelled() {
    let cancel = cancelled();
    for format in [Format::Raw, Format::Png, Format::WebP, Format::Jpeg] {
        let result = image()
            .with_cancel(&cancel)
            .resize(100, 80)
            .output(format, EncodeOptions::default())
            .bytes();
        assert_eq!(code(result), ErrorCode::Cancelled, "{format}");
    }
    assert_eq!(
        code(image().with_cancel(&cancel).stats()),
        ErrorCode::Cancelled
    );
    let trimmed = image().with_cancel(&cancel).trim(Trim::default());
    assert_eq!(code(trimmed.descriptor()), ErrorCode::Cancelled);
    let reference = image()
        .with_cancel(&cancel)
        .output(Format::Raw, EncodeOptions::default())
        .bytes_via_reference();
    assert_eq!(code(reference), ErrorCode::Cancelled);
    // The same pipeline without the flag runs.
    assert!(
        image()
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .is_ok()
    );
}

#[test]
fn a_cancel_mid_run_stops_the_run_there() {
    // Row 10 of 200 cancels; the scheduler stops at its next batch, long
    // before the last row.
    let cancel = Cancel::new();
    let mut encoder = Canceller {
        cancel: cancel.clone(),
        after: 10,
        seen: 0,
        animated: false,
    };
    let result = image()
        .with_cancel(&cancel)
        .blur(2.0)
        .output(Format::Raw, EncodeOptions::default())
        .threads(2)
        .write_with_encoder(&mut encoder, Vec::new());
    assert_eq!(code(result), ErrorCode::Cancelled);
    assert!(encoder.seen < 200, "{} rows were written", encoder.seen);
}

#[test]
fn an_animation_cancelled_at_frame_two_writes_no_more() {
    let gif = format!(
        "{}/../otf-pixels-codec-gif/tests/fixtures/animation.gif",
        env!("CARGO_MANIFEST_DIR")
    );
    let cancel = Cancel::new();
    let mut encoder = Canceller {
        cancel: cancel.clone(),
        after: 2,
        seen: 0,
        animated: false,
    };
    let result = Image::open_with(&gif, OpenOptions::default().with_animated(true))
        .unwrap()
        .with_cancel(&cancel)
        .output(Format::Raw, EncodeOptions::default())
        .write_with_encoder(&mut encoder, Vec::new());
    assert_eq!(code(result), ErrorCode::Cancelled);
    assert_eq!(encoder.seen, 2, "frames begun before the cancel landed");

    // Strips and single frames of a cancelled animation are cancelled too.
    let strip = Image::open_with(&gif, OpenOptions::default().with_animated(true))
        .unwrap()
        .with_cancel(&cancelled())
        .into_strip()
        .output(Format::Png, EncodeOptions::default())
        .bytes();
    assert_eq!(code(strip), ErrorCode::Cancelled);
}

#[test]
fn a_deadline_cancels_and_says_so() {
    let past = Cancel::after(Duration::ZERO);
    let result = image()
        .with_cancel(&past)
        .output(Format::Png, EncodeOptions::default())
        .bytes();
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::Cancelled);
    assert!(error.to_string().contains("deadline"), "{error}");
    assert_eq!(ErrorCode::Cancelled.as_str(), "cancelled");
    // A generous deadline does not get in the way.
    let later = Cancel::after(Duration::from_secs(600));
    assert!(
        image()
            .with_cancel(&later)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .is_ok()
    );
}

#[test]
fn an_overlay_composited_after_the_flag_carries_it() {
    let cancel = cancelled();
    let mark = image()
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let overlay = Image::from_stream(std::io::Cursor::new(mark)).unwrap();
    let result = image()
        .with_cancel(&cancel)
        .composite(overlay, 0, 0)
        .output(Format::Raw, EncodeOptions::default())
        .bytes();
    assert_eq!(code(result), ErrorCode::Cancelled);
}

/// Encode `descriptor`/`bytes` with `encoder`, handing it `cancel` first.
fn encode_with(mut encoder: Box<dyn Encoder>, cancel: Option<Cancel>) -> Result<Vec<u8>> {
    let (descriptor, bytes) = picture(96, 64);
    if let Some(cancel) = cancel {
        encoder.set_cancel(cancel);
    }
    let mut out = Vec::new();
    encoder.write_header(&descriptor, &mut out)?;
    for row in bytes.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut out)?;
    }
    encoder.finish(&mut out)?;
    Ok(out)
}

#[test]
fn every_buffering_encoder_stops_inside_its_own_work() {
    let progressive = EncodeOptions::default().with_progressive(true);
    type Make = fn() -> Box<dyn Encoder>;
    let encoders: Vec<(&str, Make)> = vec![
        ("gif", || Box::new(GifEncoder::new())),
        ("webp lossy", || Box::new(WebPEncoder::new())),
        ("webp lossless", || {
            Box::new(WebPEncoder::from_options(
                &EncodeOptions::default().with_lossless(true),
            ))
        }),
        ("jxl lossy", || Box::new(JxlEncoder::new())),
        ("jxl lossless", || Box::new(JxlEncoder::lossless())),
        ("avif", || Box::new(AvifEncoder::new())),
    ];
    for (name, make) in encoders {
        assert!(encode_with(make(), None).is_ok(), "{name} without a flag");
        assert_eq!(
            code(encode_with(make(), Some(cancelled()))),
            ErrorCode::Cancelled,
            "{name}"
        );
    }
    let jpeg = || Box::new(JpegEncoder::from_options(&progressive)) as Box<dyn Encoder>;
    assert_eq!(
        code(encode_with(jpeg(), Some(cancelled()))),
        ErrorCode::Cancelled,
        "progressive jpeg"
    );
}

#[test]
fn every_buffering_decoder_stops_inside_its_own_work() {
    let files: Vec<(&str, Vec<u8>)> = vec![
        (
            "gif",
            encode_with(Box::new(GifEncoder::new()), None).unwrap(),
        ),
        (
            "webp lossy",
            encode_with(Box::new(WebPEncoder::new()), None).unwrap(),
        ),
        (
            "webp lossless",
            encode_with(
                Box::new(WebPEncoder::from_options(
                    &EncodeOptions::default().with_lossless(true),
                )),
                None,
            )
            .unwrap(),
        ),
        (
            "jxl",
            encode_with(Box::new(JxlEncoder::new()), None).unwrap(),
        ),
        (
            "avif",
            encode_with(Box::new(AvifEncoder::new()), None).unwrap(),
        ),
        (
            "jpeg progressive",
            encode_with(
                Box::new(JpegEncoder::from_options(
                    &EncodeOptions::default().with_progressive(true),
                )),
                None,
            )
            .unwrap(),
        ),
    ];
    for (name, bytes) in files {
        let open = || -> Box<dyn Decoder> {
            let source = &bytes[..];
            match name {
                "gif" => Box::new(GifDecoder::new(source, Limits::default()).unwrap()),
                "jxl" => Box::new(JxlDecoder::new(source, Limits::default()).unwrap()),
                "avif" => Box::new(AvifDecoder::new(source, Limits::default()).unwrap()),
                "jpeg progressive" => {
                    Box::new(JpegDecoder::new(source, Limits::default()).unwrap())
                }
                _ => Box::new(WebPDecoder::new(source, Limits::default()).unwrap()),
            }
        };
        let mut row = vec![0; open().descriptor().row_bytes()];
        assert!(open().read_row(&mut row).is_ok(), "{name} without a flag");
        let mut decoder = open();
        decoder.set_cancel(cancelled());
        assert_eq!(
            code(decoder.read_row(&mut row)),
            ErrorCode::Cancelled,
            "{name}"
        );
    }
}
