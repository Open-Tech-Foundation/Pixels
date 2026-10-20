//! sharp's `extend` and `trim` through the facade.
//!
//! Exact margins and boxes are checked in `otf-pixels-ops`; this checks the
//! methods are wired to them, change the shape where sharp does, stream from
//! a forward-only decoder, and give the same pixels through the scheduler as
//! through the reference evaluator.

#![cfg(all(feature = "raw", feature = "png"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    EncodeOptions, ErrorCode, Extend, ExtendWith, Format, Image, ImageDescriptor, PixelFormat, Trim,
};

/// A 37x23 image in `pixel`, every pixel different.
fn image(pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(37, 23, pixel).unwrap();
    let bytes: Vec<u8> = (0..descriptor.row_bytes() * 23)
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

/// A `width` x `height` RGB image of white with a black block at
/// `(x, y, w, h)`.
fn framed(width: u32, height: u32, block: (u32, u32, u32, u32)) -> Image {
    let (bx, by, bw, bh) = block;
    let mut bytes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let inside = (bx..bx + bw).contains(&x) && (by..by + bh).contains(&y);
            bytes.extend(if inside { [0, 0, 0] } else { [255, 255, 255] });
        }
    }
    Image::from_raw(
        ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap(),
        bytes,
    )
    .unwrap()
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
fn extend_grows_the_canvas_by_each_margin() {
    let extended = image(PixelFormat::Rgb8).extend(Extend::new(1, 2, 3, 4));
    assert_eq!(size(&extended), (37 + 6, 23 + 4));
    let pixels = raw(extended);
    assert_eq!(&pixels[..3], [0, 0, 0], "sharp's default is opaque black");
}

#[test]
fn a_translucent_background_adds_alpha_as_sharp_does() {
    let extended =
        image(PixelFormat::Rgb8).extend(Extend::uniform(2).with_background([9, 8, 7, 0]));
    assert_eq!(extended.descriptor().unwrap().pixel, PixelFormat::Rgba8);
    let pixels = raw(extended);
    assert_eq!(&pixels[..4], [9, 8, 7, 0]);
    // The image itself is opaque.
    let at = ((2 * 41) + 2) * 4;
    assert_eq!(pixels[at + 3], 255);

    // A margin copied from the image has no colour of its own to need it.
    let copied = image(PixelFormat::Rgb8).extend(
        Extend::uniform(2)
            .extend_with(ExtendWith::Copy)
            .with_background([0, 0, 0, 0]),
    );
    assert_eq!(copied.descriptor().unwrap().pixel, PixelFormat::Rgb8);
}

#[test]
fn every_method_matches_the_reference_at_every_depth() {
    for format in [
        PixelFormat::Rgb8,
        PixelFormat::GrayA8,
        PixelFormat::Rgba16,
        PixelFormat::RgbaF32,
    ] {
        for with in [
            ExtendWith::Background,
            ExtendWith::Copy,
            ExtendWith::Repeat,
            ExtendWith::Mirror,
        ] {
            let pipeline = || {
                image(format)
                    .extend(Extend::new(40, 3, 50, 70).extend_with(with))
                    .blur(1.5)
                    .output(Format::Raw, EncodeOptions::default())
            };
            assert_eq!(
                pipeline().threads(4).bytes().unwrap(),
                pipeline().bytes_via_reference().unwrap(),
                "{format} {with:?}"
            );
        }
    }
}

#[test]
fn a_mirrored_margin_streams_from_a_forward_only_decoder() {
    // The top margin of a mirror reads the image's rows in reverse, which a
    // PNG decoder cannot serve; the scheduler has to buffer, not fail.
    let png = image(PixelFormat::Rgb8)
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let decoded = || Image::from_stream(std::io::Cursor::new(png.clone())).unwrap();
    let options = Extend::new(30, 0, 30, 0).extend_with(ExtendWith::Mirror);
    assert_eq!(
        raw(decoded().extend(options.clone())),
        raw(image(PixelFormat::Rgb8).extend(options))
    );
}

#[test]
fn trim_cuts_to_the_content_and_says_where() {
    let trimmed = framed(30, 20, (5, 4, 10, 8)).trim(Trim::default());
    assert_eq!(size(&trimmed), (10, 8));
    assert_eq!(trimmed.trim_offset(), Some((5, 4)));
    assert!(
        raw(trimmed).iter().all(|&v| v == 0),
        "only the block is left"
    );
}

#[test]
fn trim_runs_after_the_ops_before_it() {
    let trimmed = framed(30, 20, (5, 4, 10, 8))
        .extend(Extend::uniform(6).with_background([255, 255, 255, 255]))
        .flop()
        .trim(Trim::default());
    assert_eq!(size(&trimmed), (10, 8));
    assert_eq!(trimmed.trim_offset(), Some((42 - 6 - 15, 10)));

    // And the ops after it see the trimmed image.
    let after = trimmed.extend(Extend::uniform(1));
    assert_eq!(size(&after), (12, 10));
    assert_eq!(
        after
            .clone()
            .output(Format::Raw, EncodeOptions::default())
            .threads(2)
            .bytes()
            .unwrap(),
        after
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference()
            .unwrap()
    );
}

#[test]
fn trimming_a_decoded_file_keeps_its_format() {
    let png = framed(16, 16, (4, 4, 8, 8))
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let trimmed = Image::from_stream(std::io::Cursor::new(png))
        .unwrap()
        .trim(Trim::default());
    let metadata = trimmed.metadata().unwrap();
    assert_eq!((metadata.width, metadata.height), (8, 8));
    assert_eq!(metadata.format, Format::Png);
}

#[test]
fn an_image_of_only_background_is_left_alone() {
    let plain = framed(10, 10, (0, 0, 0, 0));
    let trimmed = plain.clone().trim(Trim::default());
    assert_eq!(size(&trimmed), (10, 10));
    assert_eq!(trimmed.trim_offset(), Some((0, 0)));
    assert_eq!(raw(trimmed), raw(plain));
    assert_eq!(framed(4, 4, (0, 0, 0, 0)).trim_offset(), None);
}

#[test]
fn trim_errors_surface_at_the_terminal() {
    let tiny = Image::from_raw(
        ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap(),
        vec![0; 4],
    )
    .unwrap()
    .trim(Trim::default());
    let error = tiny
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");

    // An error earlier in the chain is the one reported, and nothing runs.
    let error = image(PixelFormat::Rgb8)
        .crop(100, 100, 1, 1)
        .trim(Trim::default())
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
}
