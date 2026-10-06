//! ICC profiles through the public facade: read from every container that
//! has a place for one, carried through a pipeline, and written back.
//!
//! The fixtures come from `scripts/regenerate-icc-reference.py`: the same
//! profile embedded by Pillow (PNG, JPEG baseline and progressive, WebP,
//! TIFF) and by libavif (AVIF), so what is checked is that we find the bytes
//! other writers put there.

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
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, Format, Image, ImageDescriptor, OpenOptions, PixelFormat};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/icc/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn p3() -> Vec<u8> {
    std::fs::read(fixture("display_p3.icc")).unwrap()
}

/// Opened as stored: no conversion, so the profile is still the file's.
fn open(name: &str) -> Image {
    Image::open_with(fixture(name), as_stored()).unwrap()
}

fn reopen(bytes: Vec<u8>) -> Image {
    Image::from_stream_with(std::io::Cursor::new(bytes), as_stored()).unwrap()
}

fn as_stored() -> OpenOptions {
    OpenOptions::default().with_to_srgb(false)
}

fn raw(image: Image) -> Vec<u8> {
    image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap()
}

/// Our sRGB conversion of a fixture against lcms2's: the worst difference.
fn worst_against_lcms(name: &str, image: Image) -> u8 {
    let theirs = std::fs::read(fixture(&format!("{name}.srgb.raw"))).unwrap();
    let ours = raw(image);
    assert_eq!(ours.len(), theirs.len(), "{name}");
    ours.iter()
        .zip(&theirs)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap()
}

#[test]
fn opening_converts_to_srgb_as_lcms2_does() {
    for name in [
        "display_p3",
        "adobe_rgb",
        "prophoto",
        "rec2020",
        "grey_gamma22",
        "display_p3_alpha",
    ] {
        let image = Image::open(fixture(&format!("{name}.png"))).unwrap();
        // Converted pixels are sRGB, so the profile is gone.
        assert_eq!(image.icc_profile(), None, "{name}");
        let worst = worst_against_lcms(name, image);
        assert!(worst <= 1, "{name}: differs from lcms2 by up to {worst}");
    }
}

#[test]
fn conversion_can_be_deferred_and_applied_later() {
    let stored = open("display_p3.png");
    let as_stored = raw(open("display_p3.png"));
    assert_ne!(
        as_stored,
        std::fs::read(fixture("display_p3.srgb.raw")).unwrap()
    );
    assert!(worst_against_lcms("display_p3", stored.to_srgb()) <= 1);
    // A crop first, then the conversion: it is pointwise, so order is free.
    let cropped = open("display_p3.png").crop(10, 5, 20, 10).to_srgb();
    let whole = raw(Image::open(fixture("display_p3.png"))
        .unwrap()
        .crop(10, 5, 20, 10));
    assert_eq!(raw(cropped), whole);
}

#[test]
fn sixteen_bit_pixels_convert_like_eight_bit_ones() {
    // The 8-bit card widened to 16 bits converts to the same colours.
    let card = raw(open("display_p3.png"));
    let wide: Vec<u8> = card
        .iter()
        .flat_map(|&v| (u16::from(v) * 257).to_ne_bytes())
        .collect();
    let descriptor = ImageDescriptor::new(96, 64, PixelFormat::Rgb16).unwrap();
    let converted = raw(Image::from_raw(descriptor, wide)
        .unwrap()
        .with_icc_profile(Some(p3()))
        .to_srgb());
    let theirs = std::fs::read(fixture("display_p3.srgb.raw")).unwrap();
    for (pair, &expected) in converted.chunks_exact(2).zip(&theirs) {
        let v = u16::from_ne_bytes([pair[0], pair[1]]);
        assert!(
            (f64::from(v) / 257.0 - f64::from(expected)).abs() <= 1.0,
            "{v} vs {expected}"
        );
    }
}

#[test]
fn a_profile_that_cannot_be_converted_stays_with_the_pixels() {
    // An RGB profile on grey pixels describes nothing they hold.
    let grey = ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap();
    let image = Image::from_raw(grey, vec![100; 16])
        .unwrap()
        .with_icc_profile(Some(p3()))
        .to_srgb();
    assert_eq!(image.icc_profile(), Some(p3().as_slice()));
    assert_eq!(raw(image), vec![100; 16]);
    // Something that is not a profile at all.
    let rgb = ImageDescriptor::new(2, 2, PixelFormat::Rgb8).unwrap();
    let junk = Image::from_raw(rgb, vec![7; 12])
        .unwrap()
        .with_icc_profile(Some(b"junk".to_vec()))
        .to_srgb();
    assert_eq!(junk.icc_profile(), Some(&b"junk"[..]));
}

#[test]
fn every_container_yields_its_embedded_profile() {
    for name in [
        "display_p3.png",
        "display_p3.jpg",
        "display_p3_progressive.jpg",
        "display_p3.webp",
        "display_p3.tif",
        "display_p3.avif",
    ] {
        assert_eq!(open(name).icc_profile(), Some(p3().as_slice()), "{name}");
    }
}

#[test]
fn the_profile_survives_a_pipeline_into_every_format_that_holds_one() {
    let outputs = [
        (Format::Png, EncodeOptions::default()),
        (Format::Jpeg, EncodeOptions::default()),
        (Format::WebP, EncodeOptions::default()),
        (Format::WebP, EncodeOptions::default().with_lossless(true)),
        (Format::Tiff, EncodeOptions::default()),
        (Format::Avif, EncodeOptions::default()),
    ];
    for (format, options) in outputs {
        let bytes = open("display_p3.png")
            .resize(48, 32)
            .flop()
            .output(format, options)
            .bytes()
            .unwrap();
        assert_eq!(
            reopen(bytes).icc_profile(),
            Some(p3().as_slice()),
            "{format}"
        );
    }
}

#[test]
fn an_image_with_alpha_keeps_its_profile_in_webp_and_avif() {
    for format in [Format::WebP, Format::Avif, Format::Png] {
        let bytes = open("display_p3_alpha.png")
            .output(format, EncodeOptions::default())
            .bytes()
            .unwrap();
        let back = reopen(bytes);
        assert_eq!(back.icc_profile(), Some(p3().as_slice()), "{format}");
        assert_eq!(
            back.metadata().unwrap().pixel,
            PixelFormat::Rgba8,
            "{format}"
        );
    }
}

#[test]
fn formats_without_a_place_for_a_profile_drop_it_quietly() {
    for format in [Format::Gif, Format::Raw] {
        assert!(
            open("display_p3.png")
                .output(format, EncodeOptions::default())
                .bytes()
                .is_ok(),
            "{format}"
        );
    }
}

#[test]
fn a_profile_can_be_declared_or_dropped_by_hand() {
    // A profile larger than one JPEG segment (64 KiB) is split and rejoined.
    let large: Vec<u8> = (0..150_000_u32).map(|i| (i % 253) as u8).collect();
    let descriptor = ImageDescriptor::new(8, 8, PixelFormat::Rgb8).unwrap();
    let image = || Image::from_raw(descriptor, vec![90; 8 * 8 * 3]).unwrap();
    assert_eq!(image().icc_profile(), None);
    for format in [
        Format::Jpeg,
        Format::Png,
        Format::WebP,
        Format::Tiff,
        Format::Avif,
    ] {
        let bytes = image()
            .with_icc_profile(Some(large.clone()))
            .output(format, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert_eq!(
            reopen(bytes).icc_profile(),
            Some(large.as_slice()),
            "{format}"
        );
    }
    let stripped = open("display_p3.png")
        .with_icc_profile(None)
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    assert_eq!(reopen(stripped).icc_profile(), None);
}
