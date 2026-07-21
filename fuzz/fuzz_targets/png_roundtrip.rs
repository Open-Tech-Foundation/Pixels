//! Encode arbitrary pixels, then decode them back. The property is that the
//! pixels survive exactly.
//!
//! Unlike the decode targets this *can* assert on content, because the input
//! is a raster we chose rather than bytes an attacker chose. It is the
//! differential check the corpus tests cannot express: every raster, not just
//! the handful in the test suite.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_png::{Level, PngDecoder, PngEncoder};
use otf_pixels_core::{Decoder, Encoder, ImageDescriptor, Limits, PixelFormat};

const FORMATS: [PixelFormat; 7] = [
    PixelFormat::Gray8,
    PixelFormat::Gray16,
    PixelFormat::GrayA8,
    PixelFormat::Rgb8,
    PixelFormat::Rgb16,
    PixelFormat::Rgba8,
    PixelFormat::Rgba16,
];

fuzz_target!(|data: &[u8]| {
    // The first three bytes choose the shape; the rest are the pixels.
    let Some((&w, rest)) = data.split_first() else {
        return;
    };
    let Some((&h, rest)) = rest.split_first() else {
        return;
    };
    let Some((&f, pixels)) = rest.split_first() else {
        return;
    };
    let format = FORMATS[f as usize % FORMATS.len()];
    let width = u32::from(w) % 64 + 1;
    let height = u32::from(h) % 64 + 1;

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

    let mut encoder = PngEncoder::with_level(Level::FAST);
    let mut png: Vec<u8> = Vec::new();
    encoder.write_header(&descriptor, &mut png).expect("header");
    for row in raster.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut png).expect("row");
    }
    encoder.finish(&mut png).expect("finish");

    let mut decoder = PngDecoder::new(&png[..], Limits::default()).expect("our own PNG must parse");
    let mut back = Vec::with_capacity(len);
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    for _ in 0..height {
        decoder.read_row(&mut row).expect("our own PNG must decode");
        back.extend_from_slice(&row);
    }
    assert_eq!(back, raster, "{format} {width}x{height} did not round-trip");
});
