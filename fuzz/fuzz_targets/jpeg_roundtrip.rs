//! Encode arbitrary pixels, then decode them back. The property is that the
//! stream we produced is one we can parse, at the shape we declared.
//!
//! Unlike PNG's round-trip target this cannot assert the pixels survive
//! exactly — JPEG is lossy, and demanding equality would assert the codec is
//! broken. What it *can* assert is everything structural: that our own
//! encoder never emits a stream our own decoder rejects, that the dimensions
//! and pixel format come back unchanged, and that every row is there. A
//! missing byte-stuff, a miscounted MCU or a bad segment length all show up
//! as a decode failure here, over every raster rather than the handful in the
//! test suite.
//!
//! Fidelity is checked where it can be checked properly: against libjpeg, in
//! `scripts/check-jpeg-interop.sh`.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_jpeg::{JpegDecoder, JpegEncoder, Subsampling};
use otf_pixels_core::{Decoder, Encoder, ImageDescriptor, Limits, PixelFormat};

/// The formats a JPEG encoder accepts. The alpha-bearing ones are composited
/// on the way in, so they come back with fewer channels by design.
const FORMATS: [PixelFormat; 4] = [
    PixelFormat::Gray8,
    PixelFormat::GrayA8,
    PixelFormat::Rgb8,
    PixelFormat::Rgba8,
];

const SUBSAMPLINGS: [Subsampling; 3] = [
    Subsampling::None,
    Subsampling::Horizontal,
    Subsampling::Both,
];

fuzz_target!(|data: &[u8]| {
    // The first four bytes choose the shape and settings; the rest are pixels.
    let Some((&w, rest)) = data.split_first() else {
        return;
    };
    let Some((&h, rest)) = rest.split_first() else {
        return;
    };
    let Some((&f, rest)) = rest.split_first() else {
        return;
    };
    let Some((&q, pixels)) = rest.split_first() else {
        return;
    };
    let format = FORMATS[f as usize % FORMATS.len()];
    let subsampling = SUBSAMPLINGS[(f as usize / FORMATS.len()) % SUBSAMPLINGS.len()];
    let width = u32::from(w) % 64 + 1;
    let height = u32::from(h) % 64 + 1;
    let quality = q % 100 + 1;

    let Ok(descriptor) = ImageDescriptor::new(width, height, format) else {
        return;
    };
    let Some(len) = descriptor.byte_len() else {
        return;
    };
    if pixels.len() < len {
        return;
    }
    let raster = &pixels[..len];

    let Ok(encoder) = JpegEncoder::with_quality(quality) else {
        return;
    };
    let mut encoder = encoder.with_subsampling(subsampling);
    let mut jpeg: Vec<u8> = Vec::new();
    encoder.write_header(&descriptor, &mut jpeg).expect("header");
    for row in raster.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut jpeg).expect("row");
    }
    encoder.finish(&mut jpeg).expect("finish");

    let mut decoder =
        JpegDecoder::new(&jpeg[..], Limits::default()).expect("our own JPEG must parse");
    let decoded = decoder.descriptor();
    assert_eq!(
        (decoded.width, decoded.height),
        (width, height),
        "shape changed through the round trip"
    );
    // Alpha cannot survive a format that has none, so the channel count is
    // expected to drop; the colour model must not change beyond that.
    let expected = if format.channels() <= 2 {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    assert_eq!(decoded.pixel, expected, "pixel format changed");

    let mut row = vec![0_u8; decoded.row_bytes()];
    for index in 0..height {
        decoder
            .read_row(&mut row)
            .unwrap_or_else(|e| panic!("our own JPEG must decode row {index}: {e}"));
    }
});
