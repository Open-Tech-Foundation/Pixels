//! End-to-end tests for the M3 exit criteria.
//!
//! ROADMAP M3 states them as:
//!
//! > decodes PNG test suite (PngSuite) correctly; fuzz-clean.
//!
//! Both are asserted where the evidence lives — `otf-pixels-codec-png`'s
//! `tests/pngsuite.rs` against libpng, and its `tests/fuzz.rs` plus `fuzz/`
//! for the no-panic property. What is left, and what this suite covers, is the
//! milestone's other half: that PNG is a *pipeline* format and not merely a
//! codec that happens to compile. A decoder nothing can reach is not done.
//!
//! So these are the claims a user actually depends on:
//!
//! - a PNG can be opened by content, not by file name
//! - ops compose over a decoded PNG exactly as over raw pixels
//! - a PNG round-trips through the engine unchanged
//! - decoding a PNG holds memory proportional to a row, not to the image
//! - malformed input is a value at every stage of the chain

#![cfg(all(feature = "png", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, ErrorCode, Format, Image, ImageDescriptor, PixelFormat, Result};

/// A deterministic image with structure, so a wrong row order is visible.
fn source(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
    let descriptor = ImageDescriptor::new(width, height, format).unwrap();
    let len = descriptor.byte_len().unwrap();
    let bytes = (0..len)
        .map(|i| {
            let row = i / descriptor.row_bytes();
            ((i * 7 + row * 31) % 251) as u8
        })
        .collect();
    (descriptor, bytes)
}

/// Encode `bytes` as PNG through the facade.
fn to_png(descriptor: ImageDescriptor, bytes: Vec<u8>) -> Result<Vec<u8>> {
    Image::from_raw(descriptor, bytes)?
        .output(Format::Png, EncodeOptions::default())
        .bytes()
}

#[test]
fn every_pixel_format_png_can_represent_round_trips_through_the_engine() {
    // The end-to-end claim: raw in, PNG out, PNG in, raw out, same bytes.
    // Anything wrong in the encoder, the decoder, the filters, DEFLATE or the
    // format mapping shows up here as a mismatch.
    for format in [
        PixelFormat::Gray8,
        PixelFormat::Gray16,
        PixelFormat::GrayA8,
        PixelFormat::Rgb8,
        PixelFormat::Rgb16,
        PixelFormat::Rgba8,
        PixelFormat::Rgba16,
    ] {
        let (descriptor, original) = source(29, 23, format);
        let png = to_png(descriptor, original.clone())
            .unwrap_or_else(|e| panic!("encoding {format}: {e}"));

        let back = Image::from_stream(std::io::Cursor::new(png))
            .unwrap_or_else(|e| panic!("opening {format}: {e}"))
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap_or_else(|e| panic!("decoding {format}: {e}"));

        assert_eq!(back, original, "{format} did not survive a PNG round trip");
    }
}

#[test]
fn a_png_is_identified_by_its_content_not_its_name() {
    let (descriptor, bytes) = source(8, 8, PixelFormat::Rgb8);
    let png = to_png(descriptor, bytes).unwrap();

    let dir = std::env::temp_dir().join(format!("otf-pixels-m3-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // Every one of these is a lie about the format. The bytes are the truth.
    for name in ["image.jpg", "image.gif", "image", "image.txt"] {
        let path = dir.join(name);
        std::fs::write(&path, &png).unwrap();
        let image = Image::open(&path).unwrap_or_else(|e| panic!("opening {name}: {e}"));
        assert_eq!(
            image.metadata().unwrap().format,
            Format::Png,
            "`{name}` was identified by its extension"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ops_compose_over_a_decoded_png_exactly_as_over_raw_pixels() {
    // A decoded PNG must be an ordinary graph source. If it is, then running
    // the same pipeline over the raw pixels and over the PNG of those pixels
    // gives the same answer — for every pipeline shape.
    let (descriptor, original) = source(40, 32, PixelFormat::Rgba8);
    let png = to_png(descriptor, original.clone()).unwrap();

    /// A named pipeline shape, applied to whatever source it is given.
    type Pipeline = (&'static str, fn(Image) -> Image);

    let pipelines: [Pipeline; 5] = [
        ("crop", |i| i.crop(3, 5, 20, 15)),
        ("flip", Image::flip),
        ("flop", Image::flop),
        ("crop then flip", |i| i.crop(1, 1, 30, 20).flip()),
        ("flip then flop then crop", |i| {
            i.flip().flop().crop(4, 4, 16, 16)
        }),
    ];

    for (name, build) in pipelines {
        let from_png = build(Image::from_stream(std::io::Cursor::new(png.clone())).unwrap())
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap_or_else(|e| panic!("`{name}` over PNG: {e}"));

        let from_raw = build(Image::from_raw(descriptor, original.clone()).unwrap())
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap_or_else(|e| panic!("`{name}` over raw: {e}"));

        assert_eq!(from_png, from_raw, "`{name}` differed over PNG");
    }
}

#[test]
fn the_scheduler_and_the_reference_evaluator_agree_over_a_png_source() {
    // The M2 differential check, extended to a real codec. The M1 evaluator is
    // naive and therefore obviously correct, so a disagreement here is a bug
    // in how the scheduler pulls from a streaming decoder.
    let (descriptor, bytes) = source(64, 48, PixelFormat::Rgb8);
    let png = to_png(descriptor, bytes).unwrap();

    let scheduled = Image::from_stream(std::io::Cursor::new(png.clone()))
        .unwrap()
        .crop(2, 3, 50, 40)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();

    let reference = Image::from_stream(std::io::Cursor::new(png))
        .unwrap()
        .crop(2, 3, 50, 40)
        .output(Format::Raw, EncodeOptions::default())
        .bytes_via_reference()
        .unwrap();

    assert_eq!(
        scheduled, reference,
        "the scheduler disagreed with the oracle"
    );
}

#[test]
fn decoding_a_tall_png_does_not_hold_the_image() {
    // The constant-memory criterion for a real format. A source that counts
    // what it has handed over proves the decoder produces rows before it has
    // read the file — which a buffering decoder cannot do.
    use otf_pixels::Source;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct Metered {
        data: Vec<u8>,
        at: Arc<AtomicUsize>,
    }

    impl Source for Metered {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            let from = self.at.load(Ordering::SeqCst);
            let take = (self.data.len() - from).min(buf.len());
            buf[..take].copy_from_slice(&self.data[from..from + take]);
            self.at.store(from + take, Ordering::SeqCst);
            Ok(take)
        }
    }

    // Tall and genuinely noisy: the smooth ramp `source` produces compresses
    // to almost nothing, which would leave the file no bigger than a row and
    // the measurement meaningless.
    let descriptor = ImageDescriptor::new(96, 3000, PixelFormat::Rgb8).unwrap();
    let len = descriptor.byte_len().unwrap();
    let mut noise: u32 = 0x1234_5678;
    let bytes: Vec<u8> = (0..len)
        .map(|_| {
            noise ^= noise << 13;
            noise ^= noise >> 17;
            noise ^= noise << 5;
            (noise & 0xFF) as u8
        })
        .collect();
    let png = to_png(descriptor, bytes).unwrap();
    let row_bytes = descriptor.row_bytes();
    assert!(
        png.len() > row_bytes * 100,
        "the fixture is {} bytes against a {row_bytes}-byte row, too small to be evidence",
        png.len()
    );

    let at = Arc::new(AtomicUsize::new(0));
    let metered = Metered {
        data: png.clone(),
        at: Arc::clone(&at),
    };

    use otf_pixels::Decoder;
    let mut decoder = otf_pixels::PngDecoder::new(metered, otf_pixels::Limits::default()).unwrap();
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    decoder.read_row(&mut row).unwrap();

    let after_one_row = at.load(Ordering::SeqCst);
    assert!(
        after_one_row < png.len() / 2,
        "reading one row consumed {after_one_row} of {} bytes, which is not streaming",
        png.len()
    );
}

#[test]
fn a_png_pipeline_reports_malformed_input_at_every_stage() {
    let (descriptor, bytes) = source(16, 16, PixelFormat::Rgb8);
    let png = to_png(descriptor, bytes).unwrap();

    // Every prefix of a real PNG. Some are recognised and then fail; the ones
    // shorter than the signature are not recognised at all. Neither may panic,
    // and neither may succeed.
    for cut in (1..png.len()).step_by(7) {
        let truncated = png[..cut].to_vec();
        let result = Image::from_stream(std::io::Cursor::new(truncated))
            .and_then(|i| i.output(Format::Raw, EncodeOptions::default()).bytes());
        assert!(result.is_err(), "a {cut}-byte prefix decoded successfully");
    }

    // Corrupting a byte in the middle of the compressed data must also be a
    // reported error rather than a plausible-looking image.
    let mut corrupt = png.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xFF;
    let result = Image::from_stream(std::io::Cursor::new(corrupt))
        .and_then(|i| i.output(Format::Raw, EncodeOptions::default()).bytes());
    assert!(result.is_err(), "a corrupted PNG decoded successfully");
}

#[test]
fn limits_are_enforced_from_the_header_before_any_pixels() {
    // SPEC §Safety: a header claiming more than the caller allows costs
    // nothing, because it is refused before a buffer exists.
    let (descriptor, bytes) = source(200, 200, PixelFormat::Rgb8);
    let png = to_png(descriptor, bytes).unwrap();

    let limits = otf_pixels::Limits::default().with_max_pixels(1000);
    let error = otf_pixels::PngDecoder::new(&png[..], limits).unwrap_err();
    assert_eq!(error.code(), ErrorCode::LimitExceeded, "{error}");
}

#[test]
fn compression_level_changes_the_bytes_but_never_the_pixels() {
    // `EncodeOptions::quality` maps onto DEFLATE effort. It must be a
    // size/time knob only: PNG is lossless, so the pixels are not negotiable.
    let (descriptor, original) = source(48, 48, PixelFormat::Rgba8);

    let mut sizes = Vec::new();
    for quality in [1_u8, 50, 100] {
        let options = EncodeOptions::with_quality(quality).unwrap();
        let png = Image::from_raw(descriptor, original.clone())
            .unwrap()
            .output(Format::Png, options)
            .bytes()
            .unwrap();

        let back = Image::from_stream(std::io::Cursor::new(png.clone()))
            .unwrap()
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert_eq!(back, original, "quality {quality} changed the pixels");
        sizes.push(png.len());
    }

    assert!(
        sizes.first() >= sizes.last(),
        "more effort produced a larger file: {sizes:?}"
    );
}
