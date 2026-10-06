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
fn an_avif_with_per_superblock_deltas_matches_libavif() {
    // photo_420_deltaq: avifenc's key-frame visual-quality mode varies the
    // quantizer and the deblocking strength per superblock (delta_q and
    // delta_lf). The codec crate checks the planes against libaom exactly;
    // this is the file a user has, opened and converted the ordinary way.
    let image = Image::open(fixture("photo_420_deltaq.avif")).unwrap();
    let metadata = image.metadata().unwrap();
    assert_eq!((metadata.width, metadata.height), (256, 192));
    let ours = image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let theirs = std::fs::read(fixture("photo_420_deltaq.raw")).unwrap();
    assert_eq!(ours.len(), theirs.len());
    for (index, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert!(a.abs_diff(*b) <= 2, "byte {index}: {a} vs {b}");
    }
}

#[test]
fn a_ten_bit_avif_decodes_to_full_range_sixteen_bit_rgb() {
    // photo_420_10bit: 10-bit 4:2:0. The engine has no 10-bit format, so it
    // arrives as Rgb16 over the full 0..=65535 range, native-endian — within
    // one 16-bit step of libavif's (little-endian) reference.
    let image = Image::open(fixture("photo_420_10bit.avif")).unwrap();
    assert_eq!(image.metadata().unwrap().pixel, PixelFormat::Rgb16);
    let ours = image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let theirs = std::fs::read(fixture("photo_420_10bit.raw")).unwrap();
    assert_eq!(ours.len(), 96 * 64 * 3 * 2);
    let mut brightest = 0;
    for (a, b) in ours.chunks_exact(2).zip(theirs.chunks_exact(2)) {
        let (a, b) = (
            u16::from_ne_bytes([a[0], a[1]]),
            u16::from_le_bytes([b[0], b[1]]),
        );
        assert!(a.abs_diff(b) <= 1, "{a} vs {b}");
        brightest = brightest.max(a);
    }
    // Rescaled, not left in 10-bit units.
    assert!(brightest > 1023 * 16, "brightest sample {brightest}");
}

#[test]
fn an_avif_with_alpha_decodes_to_rgba_with_libavifs_alpha() {
    // photo_alpha: the transparency is a second, monochrome AV1 image; it
    // must come back as the fourth channel, matching libavif's exactly (the
    // colour within its one-step conversion gap).
    let image = Image::open(fixture("photo_alpha.avif")).unwrap();
    assert_eq!(image.metadata().unwrap().pixel, PixelFormat::Rgba8);
    let ours = image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let theirs = std::fs::read(fixture("photo_alpha.raw")).unwrap();
    assert_eq!(ours.len(), theirs.len());
    for (a, b) in ours.chunks_exact(4).zip(theirs.chunks_exact(4)) {
        assert_eq!(a[3], b[3], "alpha");
        for c in 0..3 {
            assert!(a[c].abs_diff(b[c]) <= 1, "{a:?} vs {b:?}");
        }
    }
}

#[test]
fn an_avif_using_an_unimplemented_tool_fails_cleanly_through_the_facade() {
    // Opening only reads the container; the refusal comes when pixels are
    // pulled, and it is a catchable Unsupported rather than a wrong image.
    let image = Image::open(fixture("unsupported/qmatrix.avif")).unwrap();
    let err = image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(err.code(), ErrorCode::Unsupported, "{err}");
}
