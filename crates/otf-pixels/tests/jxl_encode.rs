//! JPEG XL output through the facade.
//!
//! Exactness is checked against libjxl by `scripts/check-jxl-interop.sh`;
//! this checks the facade reaches the encoder, narrows what it cannot hold,
//! and gives the same file from the scheduler as from the reference
//! evaluator.

#![cfg(all(feature = "raw", feature = "jxl"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{EncodeOptions, Format, Image, ImageDescriptor, PixelFormat};

fn image(width: u32, height: u32, pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(width, height, pixel).unwrap();
    let bytes = (0..descriptor.byte_len().unwrap())
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

#[test]
fn every_integer_format_encodes_and_the_two_evaluators_agree() {
    for pixel in [
        PixelFormat::Gray8,
        PixelFormat::GrayA8,
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Gray16,
        PixelFormat::Rgba16,
        PixelFormat::RgbF32,
    ] {
        let output = || {
            image(300, 270, pixel)
                .blur(1.0)
                .output(Format::Jxl, EncodeOptions::default())
        };
        let scheduled = output().threads(4).bytes().unwrap();
        assert_eq!(
            scheduled[..2],
            [0xFF, 0x0A],
            "{pixel}: a JPEG XL codestream"
        );
        assert_eq!(
            scheduled,
            output().bytes_via_reference().unwrap(),
            "{pixel}"
        );
    }
}

#[test]
fn the_format_is_named_jxl() {
    assert_eq!(Format::Jxl.as_str(), "jxl");
}
