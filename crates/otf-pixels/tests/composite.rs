//! sharp's `composite` for watermarking through the facade: gravity, several
//! overlays, tiling, blend modes and overlays in another format.
//!
//! The blend formulas and placements are checked in `otf-pixels-ops`; this
//! checks the facade wires them up, streams them from forward-only decoders,
//! and gives the same pixels through the scheduler as through the reference
//! evaluator.

#![cfg(all(feature = "raw", feature = "png"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    Blend, EncodeOptions, ErrorCode, Format, Image, ImageDescriptor, Overlay, PixelFormat, Position,
};

/// A `width` x `height` image in `pixel`, every pixel different.
fn varied(width: u32, height: u32, pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(width, height, pixel).unwrap();
    let bytes: Vec<u8> = (0..descriptor.byte_len().unwrap())
        .map(|i| (i * 31 % 251) as u8)
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

/// A `width` x `height` image of one `pixel` value.
fn solid(width: u32, height: u32, format: PixelFormat, pixel: &[u8]) -> Image {
    let descriptor = ImageDescriptor::new(width, height, format).unwrap();
    Image::from_raw(descriptor, pixel.repeat((width * height) as usize)).unwrap()
}

fn raw(image: Image) -> Vec<u8> {
    image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap()
}

/// The scheduler's and the reference evaluator's output, which must match.
fn both(image: &Image) -> (Vec<u8>, Vec<u8>) {
    let output = || image.clone().output(Format::Raw, EncodeOptions::default());
    (
        output().threads(4).bytes().unwrap(),
        output().bytes_via_reference().unwrap(),
    )
}

#[test]
fn a_translucent_watermark_goes_in_the_corner_of_an_opaque_photo() {
    let photo = solid(40, 30, PixelFormat::Rgb8, &[0, 0, 0]);
    let mark = solid(8, 6, PixelFormat::Rgba8, &[255, 255, 255, 128]);
    let marked = photo.composite_overlays([Overlay::new(mark).with_gravity(Position::SouthEast)]);
    assert_eq!(marked.descriptor().unwrap().pixel, PixelFormat::Rgb8);
    let pixels = raw(marked);
    let at = |x: usize, y: usize| &pixels[(y * 40 + x) * 3..(y * 40 + x) * 3 + 3];
    assert_eq!(at(39, 29), [128, 128, 128], "inside the watermark");
    assert_eq!(at(32, 24), [128, 128, 128], "its top-left corner");
    assert_eq!(at(31, 29), [0, 0, 0], "left of it");
    assert_eq!(at(39, 23), [0, 0, 0], "above it");
}

#[test]
fn the_default_overlay_is_centred_as_in_sharp() {
    let base = solid(9, 9, PixelFormat::Gray8, &[0]);
    let dot = solid(3, 3, PixelFormat::Gray8, &[255]);
    let pixels = raw(base.composite_overlays([Overlay::new(dot)]));
    for y in 0..9 {
        for x in 0..9 {
            let inside = (3..6).contains(&x) && (3..6).contains(&y);
            assert_eq!(
                pixels[y * 9 + x],
                if inside { 255 } else { 0 },
                "({x}, {y})"
            );
        }
    }
}

#[test]
fn several_overlays_are_drawn_in_order() {
    let base = solid(4, 1, PixelFormat::Gray8, &[0]);
    let pixels = raw(base.composite_overlays([
        Overlay::new(solid(3, 1, PixelFormat::Gray8, &[100])).at(0, 0),
        Overlay::new(solid(2, 1, PixelFormat::Gray8, &[200])).at(1, 0),
        Overlay::new(solid(1, 1, PixelFormat::Gray8, &[50]))
            .with_gravity(Position::East)
            .with_blend(Blend::Lighten),
    ]));
    assert_eq!(pixels, [100, 200, 200, 50]);
    // No overlays at all is no change.
    let plain = varied(5, 5, PixelFormat::Rgb8);
    assert_eq!(
        raw(plain.clone().composite_overlays(Vec::new())),
        raw(plain)
    );
}

#[test]
fn a_tiled_overlay_covers_the_image() {
    let base = solid(7, 2, PixelFormat::Gray8, &[0]);
    let pattern = Image::from_raw(
        ImageDescriptor::new(3, 1, PixelFormat::Gray8).unwrap(),
        vec![1, 2, 3],
    )
    .unwrap();
    let tiled = |overlay: Overlay| raw(base.clone().composite_overlays([overlay]));
    assert_eq!(
        tiled(
            Overlay::new(pattern.clone())
                .tiled(true)
                .with_gravity(Position::NorthWest)
        ),
        [1, 2, 3, 1, 2, 3, 1, 1, 2, 3, 1, 2, 3, 1]
    );
    assert_eq!(
        tiled(Overlay::new(pattern).tiled(true).at(2, 0)),
        [3, 1, 2, 3, 1, 2, 3, 3, 1, 2, 3, 1, 2, 3]
    );
}

#[test]
fn a_tiled_watermark_streams_from_decoded_files() {
    // Both inputs come from forward-only PNG decoders, and every band of the
    // base reads the overlay from the top again.
    let png = |image: Image| {
        image
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap()
    };
    let photo = png(varied(97, 61, PixelFormat::Rgb8));
    let mark = png(varied(13, 7, PixelFormat::Rgba8));
    let open = |bytes: &Vec<u8>| Image::from_stream(std::io::Cursor::new(bytes.clone())).unwrap();
    let decoded = open(&photo).composite_overlays([Overlay::new(open(&mark))
        .tiled(true)
        .with_blend(Blend::Multiply)]);
    let in_memory = varied(97, 61, PixelFormat::Rgb8).composite_overlays([Overlay::new(varied(
        13,
        7,
        PixelFormat::Rgba8,
    ))
    .tiled(true)
    .with_blend(Blend::Multiply)]);
    assert_eq!(raw(decoded), raw(in_memory));
}

#[test]
fn every_blend_matches_the_reference_across_formats() {
    let blends = [
        Blend::Clear,
        Blend::Source,
        Blend::Over,
        Blend::In,
        Blend::Out,
        Blend::Atop,
        Blend::Dest,
        Blend::DestOver,
        Blend::DestIn,
        Blend::DestOut,
        Blend::DestAtop,
        Blend::Xor,
        Blend::Add,
        Blend::Saturate,
        Blend::Multiply,
        Blend::Screen,
        Blend::Overlay,
        Blend::Darken,
        Blend::Lighten,
        Blend::ColourDodge,
        Blend::ColourBurn,
        Blend::HardLight,
        Blend::SoftLight,
        Blend::Difference,
        Blend::Exclusion,
    ];
    for (base, overlay) in [
        (PixelFormat::Rgb8, PixelFormat::Rgba8),
        (PixelFormat::Rgba8, PixelFormat::Rgba8),
        (PixelFormat::GrayA8, PixelFormat::Rgb8),
        (PixelFormat::Rgba16, PixelFormat::Gray8),
        (PixelFormat::RgbF32, PixelFormat::RgbaF32),
    ] {
        for blend in blends {
            let image = varied(41, 29, base).blur(1.0).composite_overlays([
                Overlay::new(varied(9, 8, overlay))
                    .with_gravity(Position::South)
                    .with_blend(blend),
                Overlay::new(varied(5, 6, overlay))
                    .tiled(true)
                    .with_blend(blend),
            ]);
            let (scheduled, reference) = both(&image);
            assert_eq!(scheduled, reference, "{base} under {overlay}, {blend:?}");
        }
    }
}

#[test]
fn a_blend_that_can_erase_adds_alpha() {
    let base = solid(3, 1, PixelFormat::Rgb8, &[9, 9, 9]);
    let eraser = solid(1, 1, PixelFormat::Gray8, &[0]);
    let erased = base.composite_overlays([Overlay::new(eraser).with_blend(Blend::DestOut)]);
    assert_eq!(erased.descriptor().unwrap().pixel, PixelFormat::Rgba8);
    assert_eq!(raw(erased), [9, 9, 9, 255, 0, 0, 0, 0, 9, 9, 9, 255]);
}

#[test]
fn an_overlay_error_surfaces_at_the_terminal() {
    let broken = varied(4, 4, PixelFormat::Rgb8).resize(0, 1);
    let error = varied(8, 8, PixelFormat::Rgb8)
        .composite_overlays([Overlay::new(broken)])
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
}
