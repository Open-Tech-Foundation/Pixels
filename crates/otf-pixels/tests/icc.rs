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

use otf_pixels::{EncodeOptions, Format, Image, ImageDescriptor, PixelFormat};

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/icc/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn p3() -> Vec<u8> {
    std::fs::read(fixture("display_p3.icc")).unwrap()
}

fn open(name: &str) -> Image {
    Image::open(fixture(name)).unwrap()
}

fn reopen(bytes: Vec<u8>) -> Image {
    Image::from_stream(std::io::Cursor::new(bytes)).unwrap()
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
