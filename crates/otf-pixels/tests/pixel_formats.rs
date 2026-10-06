//! Every pixel format into every encoder: output narrows to what the format
//! holds rather than refusing, and `to_pixel_format` converts on request.

#![cfg(all(
    feature = "png",
    feature = "jpeg",
    feature = "webp",
    feature = "tiff",
    feature = "avif",
    feature = "gif",
    feature = "raw"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, Format, Image, ImageDescriptor, PixelFormat};

const ALL: [PixelFormat; 9] = [
    PixelFormat::Gray8,
    PixelFormat::GrayA8,
    PixelFormat::Rgb8,
    PixelFormat::Rgba8,
    PixelFormat::Gray16,
    PixelFormat::Rgb16,
    PixelFormat::Rgba16,
    PixelFormat::RgbF32,
    PixelFormat::RgbaF32,
];

/// A flat mid-grey image (half intensity, half alpha) in `pixel`.
fn mid_grey(pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(16, 16, pixel).unwrap();
    let sample: Vec<u8> = match pixel {
        PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            32_896_u16.to_ne_bytes().to_vec()
        }
        PixelFormat::RgbF32 | PixelFormat::RgbaF32 => (128.0_f32 / 255.0).to_ne_bytes().to_vec(),
        _ => vec![128],
    };
    let bytes = sample.repeat(16 * 16 * pixel.channels());
    Image::from_raw(descriptor, bytes).unwrap()
}

#[test]
fn every_pixel_format_encodes_to_every_format() {
    for pixel in ALL {
        for (format, options) in [
            (Format::Png, EncodeOptions::default()),
            (Format::Jpeg, EncodeOptions::default()),
            (Format::WebP, EncodeOptions::default().with_lossless(true)),
            (Format::Avif, EncodeOptions::default()),
            (Format::Gif, EncodeOptions::default()),
            (Format::Tiff, EncodeOptions::default()),
        ] {
            let bytes = mid_grey(pixel)
                .output(format, options)
                .bytes()
                .unwrap_or_else(|e| panic!("{pixel} as {format}: {e}"));
            let back = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
            let back_pixel = back.metadata().unwrap().pixel;
            let raw = back
                .output(Format::Raw, EncodeOptions::default())
                .bytes()
                .unwrap();
            // Mid-grey comes back mid-grey: 128 at 8 bits (a step or two of
            // lossy coding aside), 32896 at 16. JPEG has no alpha, and GIF
            // only all-or-nothing transparency; both composite a partly
            // transparent pixel over black, as libvips does: 64.
            let flattens = matches!(format, Format::Jpeg | Format::Gif);
            let expected = if flattens && pixel.has_alpha() {
                64.0
            } else {
                128.0
            };
            let wide_back = back_pixel.sample_kind() != otf_pixels::SampleKind::U8;
            let first = if wide_back {
                f64::from(u16::from_ne_bytes([raw[0], raw[1]])) / 257.0
            } else {
                f64::from(raw[0])
            };
            assert!(
                (first - expected).abs() <= 3.0,
                "{pixel} as {format}: {first}"
            );
        }
    }
}

#[test]
fn narrowing_keeps_the_layout_and_the_lossless_formats_keep_depth() {
    let reopen = |pixel: PixelFormat, format: Format| {
        let options = EncodeOptions::default().with_lossless(format == Format::WebP);
        let bytes = mid_grey(pixel).output(format, options).bytes().unwrap();
        Image::from_stream(std::io::Cursor::new(bytes))
            .unwrap()
            .metadata()
            .unwrap()
            .pixel
    };
    assert_eq!(
        reopen(PixelFormat::Rgba16, Format::WebP),
        PixelFormat::Rgba8
    );
    assert_eq!(
        reopen(PixelFormat::Gray16, Format::Avif),
        PixelFormat::Gray8
    );
    assert_eq!(reopen(PixelFormat::Rgb16, Format::Png), PixelFormat::Rgb16);
    assert_eq!(
        reopen(PixelFormat::RgbaF32, Format::Png),
        PixelFormat::Rgba16
    );
    assert_eq!(
        reopen(PixelFormat::RgbF32, Format::Tiff),
        PixelFormat::Rgb16
    );
}

#[test]
fn to_pixel_format_converts_on_request() {
    let grey = mid_grey(PixelFormat::Rgba8).to_pixel_format(PixelFormat::Gray8);
    assert_eq!(grey.metadata().unwrap().pixel, PixelFormat::Gray8);
    let raw = grey
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    assert!(raw.iter().all(|&v| v == 128));
    // Converting to the format it already has is free.
    let same = mid_grey(PixelFormat::Rgb8).to_pixel_format(PixelFormat::Rgb8);
    assert_eq!(same.metadata().unwrap().pixel, PixelFormat::Rgb8);
}
