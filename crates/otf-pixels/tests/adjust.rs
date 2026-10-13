//! sharp's per-pixel adjustments through the facade: `negate`, `gamma`,
//! `linear`, `threshold`, `tint`, `greyscale`, `recomb`, `modulate`'s
//! lightness, `ensureAlpha`, `removeAlpha` and named `extractChannel`.
//!
//! Exact values are checked in `otf-pixels-ops`; this checks the methods are
//! wired to them, change the shape where sharp does, and give the same pixels
//! through the scheduler as through the reference evaluator.

#![cfg(feature = "raw")]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    Channel, EncodeOptions, ErrorCode, Format, Image, ImageDescriptor, Modulate, PixelFormat,
};

/// A 37x23 image in `pixel`, every pixel different.
fn image(pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(37, 23, pixel).unwrap();
    let bytes: Vec<u8> = (0..descriptor.row_bytes() * 23)
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

/// One pixel image.
fn pixel(format: PixelFormat, bytes: &[u8]) -> Image {
    Image::from_raw(ImageDescriptor::new(1, 1, format).unwrap(), bytes.to_vec()).unwrap()
}

fn raw(image: Image) -> Vec<u8> {
    image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap()
}

#[test]
fn each_method_gives_sharps_values() {
    let rgba = || pixel(PixelFormat::Rgba8, &[200, 50, 50, 40]);
    assert_eq!(raw(rgba().negate(true)), [55, 205, 205, 215]);
    assert_eq!(raw(rgba().negate(false)), [55, 205, 205, 40]);
    assert_eq!(raw(rgba().greyscale()), [95, 95, 95, 40]);
    assert_eq!(raw(rgba().threshold(128, true)), [0, 0, 0, 40]);
    assert_eq!(raw(rgba().threshold(128, false)), [255, 0, 0, 40]);
    assert_eq!(raw(rgba().linear(0.5, 10.0)), [110, 35, 35, 40]);
    assert_eq!(
        raw(rgba().recomb([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])),
        [50, 50, 200, 40]
    );
    assert_eq!(raw(rgba().gamma(1.0)), [200, 50, 50, 40]);
    let lighter = Modulate::identity().with_lightness(20.0).unwrap();
    assert_eq!(raw(rgba().modulate(lighter)), [251, 101, 101, 40]);
}

#[test]
fn shape_changes_follow_sharp() {
    let grey = pixel(PixelFormat::Gray8, &[128]);
    let tinted = grey.clone().tint(200, 100, 100);
    assert_eq!(tinted.descriptor().unwrap().pixel, PixelFormat::Rgb8);

    let with_alpha = pixel(PixelFormat::Rgb8, &[1, 2, 3]).ensure_alpha(0.5);
    assert_eq!(with_alpha.descriptor().unwrap().pixel, PixelFormat::Rgba8);
    assert_eq!(raw(with_alpha), [1, 2, 3, 128]);

    let opaque = pixel(PixelFormat::Rgba8, &[1, 2, 3, 4]).remove_alpha();
    assert_eq!(raw(opaque), [1, 2, 3]);
    // Without alpha, removing it changes nothing.
    assert_eq!(raw(grey.remove_alpha()), [128]);
}

#[test]
fn channels_are_extracted_by_name() {
    let rgba = || pixel(PixelFormat::Rgba8, &[10, 20, 30, 40]);
    for (channel, value) in [
        (Channel::Red, 10),
        (Channel::Green, 20),
        (Channel::Blue, 30),
        (Channel::Alpha, 40),
    ] {
        assert_eq!(raw(rgba().extract_channel_named(channel)), [value]);
    }
    let grey_alpha = || pixel(PixelFormat::GrayA8, &[7, 9]);
    assert_eq!(raw(grey_alpha().extract_channel_named(Channel::Alpha)), [9]);
    let missing = grey_alpha()
        .extract_channel_named(Channel::Blue)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(missing.code(), ErrorCode::InvalidArgument);
    let no_alpha = pixel(PixelFormat::Rgb8, &[1, 2, 3])
        .extract_channel_named(Channel::Alpha)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(no_alpha.code(), ErrorCode::InvalidArgument);
}

#[test]
fn a_chain_of_every_adjustment_matches_the_reference_at_every_depth() {
    for format in [
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Gray8,
        PixelFormat::Rgba16,
        PixelFormat::RgbaF32,
    ] {
        let pipeline = || {
            image(format)
                .gamma(1.8)
                .linear_per_channel([1.1, 0.9, 1.0], [5.0, -5.0, 0.0])
                .tint(180, 140, 100)
                .recomb([
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131],
                ])
                .modulate(Modulate::identity().with_lightness(-5.0).unwrap())
                .ensure_alpha(0.75)
                .negate(false)
                .greyscale()
                .threshold(100, false)
                .output(Format::Raw, EncodeOptions::default())
        };
        assert_eq!(
            pipeline().threads(4).bytes().unwrap(),
            pipeline().bytes_via_reference().unwrap(),
            "{format}"
        );
    }
}

#[test]
fn invalid_parameters_surface_at_the_terminal() {
    for image in [
        pixel(PixelFormat::Rgb8, &[1, 2, 3]).gamma(0.0),
        pixel(PixelFormat::Rgb8, &[1, 2, 3]).linear(f32::NAN, 0.0),
        pixel(PixelFormat::Rgb8, &[1, 2, 3]).ensure_alpha(2.0),
        pixel(PixelFormat::Rgb8, &[1, 2, 3]).recomb([[f32::INFINITY; 3]; 3]),
    ] {
        let error = image
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
    }
}
