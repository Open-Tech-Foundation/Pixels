//! sharp's `rotate(angle)` for any angle, and `affine`, through the facade.
//!
//! The resampling itself is checked in `otf-pixels-ops`; this checks
//! `Image::rotate` and `Image::rotate_with` reach it (or the exact quarter
//! turn) where sharp would, add alpha for a translucent background, stream
//! from a forward-only decoder, and give the same pixels through the
//! scheduler as through the reference evaluator.

#![cfg(all(feature = "raw", feature = "png"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    AffineOptions, EncodeOptions, Format, Image, ImageDescriptor, Interpolator, PixelFormat,
    RotateOptions,
};

/// A 37x23 image in `pixel`, every pixel different.
fn image(pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(37, 23, pixel).unwrap();
    let bytes: Vec<u8> = (0..descriptor.row_bytes() * 23)
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

fn raw(image: Image) -> Vec<u8> {
    image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap()
}

fn size(image: &Image) -> (u32, u32) {
    let descriptor = image.descriptor().unwrap();
    (descriptor.width, descriptor.height)
}

#[test]
fn any_integer_angle_rotates_onto_the_bounding_box() {
    // 37·cos30 + 23·sin30 = 43.54; 37·sin30 + 23·cos30 = 38.42.
    let rotated = image(PixelFormat::Rgb8).rotate(30);
    assert_eq!(size(&rotated), (44, 39));
    let pixels = raw(rotated);
    assert_eq!(&pixels[..3], [0, 0, 0], "sharp's default is opaque black");
    // Whole turns on top change nothing.
    assert_eq!(raw(image(PixelFormat::Rgb8).rotate(390)), pixels);
    assert_eq!(raw(image(PixelFormat::Rgb8).rotate(-330)), pixels);
}

#[test]
fn quarter_turns_stay_exact_whatever_the_options() {
    let options = RotateOptions::default()
        .with_background([1, 2, 3, 0])
        .with_interpolator(Interpolator::Bicubic);
    for degrees in [0, 90, 180, 270, -90, 450] {
        let exact = image(PixelFormat::Rgb8).rotate(degrees);
        let via = image(PixelFormat::Rgb8).rotate_with(f64::from(degrees), options);
        // No alpha added either: a quarter turn uncovers no background.
        assert_eq!(via.descriptor().unwrap().pixel, PixelFormat::Rgb8);
        assert_eq!(raw(via), raw(exact), "{degrees}");
    }
}

#[test]
fn a_translucent_background_adds_alpha_as_sharp_does() {
    let rotated = image(PixelFormat::Rgb8)
        .rotate_with(12.5, RotateOptions::default().with_background([9, 8, 7, 0]));
    assert_eq!(rotated.descriptor().unwrap().pixel, PixelFormat::Rgba8);
    let (width, height) = size(&rotated);
    let pixels = raw(rotated);
    assert_eq!(&pixels[..4], [0, 0, 0, 0], "a transparent corner");
    // The middle is the opaque image.
    let at = (((height / 2) * width + width / 2) * 4) as usize;
    assert_eq!(pixels[at + 3], 255);
}

#[test]
fn a_non_finite_angle_fails_at_the_terminal() {
    for degrees in [f64::NAN, f64::INFINITY] {
        let result = image(PixelFormat::Rgb8)
            .rotate_with(degrees, RotateOptions::default())
            .output(Format::Raw, EncodeOptions::default())
            .bytes();
        assert!(result.is_err(), "{degrees}");
    }
}

#[test]
fn the_scheduler_matches_the_reference_at_every_depth() {
    for format in [
        PixelFormat::Rgb8,
        PixelFormat::GrayA8,
        PixelFormat::Rgba16,
        PixelFormat::RgbaF32,
    ] {
        for interpolator in [
            Interpolator::Nearest,
            Interpolator::Bilinear,
            Interpolator::Bicubic,
        ] {
            let pipeline = || {
                image(format)
                    .resize(120, 80)
                    .rotate_with(
                        -37.0,
                        RotateOptions::default().with_interpolator(interpolator),
                    )
                    .blur(1.0)
                    .output(Format::Raw, EncodeOptions::default())
            };
            assert_eq!(
                pipeline().threads(4).bytes().unwrap(),
                pipeline().bytes_via_reference().unwrap(),
                "{format} {interpolator:?}"
            );
        }
    }
}

#[test]
fn a_rotation_streams_from_a_forward_only_decoder() {
    // An output row crosses the input diagonally, reading its rows out of
    // order; the scheduler has to buffer a PNG decode, not fail.
    let png = image(PixelFormat::Rgb8)
        .resize(400, 300)
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let decoded = Image::from_stream(std::io::Cursor::new(png.clone())).unwrap();
    let in_memory = Image::from_stream(std::io::Cursor::new(png))
        .unwrap()
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let in_memory = Image::from_raw(
        ImageDescriptor::new(400, 300, PixelFormat::Rgb8).unwrap(),
        in_memory,
    )
    .unwrap();
    assert_eq!(raw(decoded.rotate(200)), raw(in_memory.rotate(200)));
}

#[test]
fn affine_shears_onto_the_bounding_box() {
    // X = x + y/2: 37 + 23/2 = 48.5, rounded up.
    let sheared = image(PixelFormat::Rgb8).affine([1.0, 0.5, 0.0, 1.0], AffineOptions::default());
    assert_eq!(size(&sheared), (49, 23));
    let pixels = raw(sheared);
    // The top-right corner is uncovered: sharp's opaque black.
    assert_eq!(&pixels[48 * 3..49 * 3], [0, 0, 0]);
}

#[test]
fn affine_mirrors_exactly() {
    let mirrored = image(PixelFormat::Rgba8).affine(
        [-1.0, 0.0, 0.0, 1.0],
        AffineOptions::default().with_interpolator(Interpolator::Bilinear),
    );
    // Premultiplied resampling drops the colour of a fully transparent
    // pixel, so compare the rest.
    let got = raw(mirrored);
    let expected = raw(image(PixelFormat::Rgba8).flop());
    for (g, e) in got.chunks_exact(4).zip(expected.chunks_exact(4)) {
        if e[3] > 0 {
            assert_eq!(g, e);
        }
    }
}

#[test]
fn affine_with_a_translucent_background_adds_alpha() {
    let sheared = image(PixelFormat::Rgb8).affine(
        [1.0, 0.0, 0.4, 1.0],
        AffineOptions::default().with_background([0, 0, 0, 0]),
    );
    assert_eq!(sheared.descriptor().unwrap().pixel, PixelFormat::Rgba8);
    let (width, _) = size(&sheared);
    let pixels = raw(sheared);
    // Y = 0.4x + y: the bottom-left corner is uncovered and transparent.
    let height = pixels.len() / 4 / width as usize;
    let at = (height - 1) * width as usize * 4;
    assert_eq!(pixels[at + 3], 0);
}

#[test]
fn a_singular_matrix_fails_at_the_terminal() {
    let result = image(PixelFormat::Rgb8)
        .affine([1.0, 2.0, 0.5, 1.0], AffineOptions::default())
        .output(Format::Raw, EncodeOptions::default())
        .bytes();
    assert!(result.is_err());
}

#[test]
fn affine_through_the_scheduler_matches_the_reference() {
    for format in [PixelFormat::Rgb8, PixelFormat::GrayA8, PixelFormat::Rgba16] {
        for options in [
            AffineOptions::default(),
            AffineOptions::default()
                .with_interpolator(Interpolator::Bilinear)
                .with_input_offset(0.5, 0.25)
                .with_output_offset(-1.5, 3.0)
                .with_background([10, 20, 30, 128]),
        ] {
            let pipeline = || {
                image(format)
                    .resize(120, 80)
                    .affine([0.9, 0.4, -0.3, 1.2], options)
                    .blur(1.0)
                    .output(Format::Raw, EncodeOptions::default())
            };
            assert_eq!(
                pipeline().threads(4).bytes().unwrap(),
                pipeline().bytes_via_reference().unwrap(),
                "{format} {options:?}"
            );
        }
    }
}
