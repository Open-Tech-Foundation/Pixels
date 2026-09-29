//! AVIF through the public facade: open a real avifenc file, run it through a
//! pipeline, and get libavif's pixels back.
//!
//! The codec crate verifies the decode in depth (its reference, plane and
//! unsupported suites). What this checks is the wiring a user actually goes
//! through — magic-byte detection, the lazy decoder behind `Image`, a pipeline
//! op, an encoder at the end — on the kind of file they actually have: 4:2:0
//! chroma, a BT.601 matrix, every in-loop filter on.

#![cfg(all(feature = "avif", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, ErrorCode, Format, Image, PixelFormat};

fn fixture(name: &str) -> String {
    format!(
        "{}/../otf-pixels-codec-avif/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn a_420_avif_opens_and_crops_to_libavifs_pixels() {
    // photo_420: 96x64, 4:2:0, BT.601 full range; libavif's raster is the
    // reference, within the 2-step gap between libyuv's conversion and ours.
    let (width, left, top, w, h) = (96_usize, 10, 7, 40, 30);
    let image = Image::open(fixture("photo_420.avif")).unwrap();
    let metadata = image.metadata().unwrap();
    assert_eq!(metadata.format, Format::Avif);
    assert_eq!((metadata.width, metadata.height), (96, 64));
    assert_eq!(metadata.pixel, PixelFormat::Rgb8);

    let ours = image
        .crop(left as u32, top as u32, w as u32, h as u32)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let theirs = std::fs::read(fixture("photo_420.raw")).unwrap();
    assert_eq!(ours.len(), w * h * 3);
    for y in 0..h {
        for x in 0..w * 3 {
            let a = ours[y * w * 3 + x];
            let b = theirs[(top + y) * width * 3 + left * 3 + x];
            assert!(a.abs_diff(b) <= 2, "({x}, {y}): {a} vs {b}");
        }
    }
}

#[test]
fn an_avif_using_an_unimplemented_tool_fails_cleanly_through_the_facade() {
    // Opening only reads the container; the refusal comes when pixels are
    // pulled, and it is a catchable Unsupported rather than a wrong image.
    let image = Image::open(fixture("unsupported/delta_q.avif")).unwrap();
    let err = image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::Unsupported, "{err}");
}
