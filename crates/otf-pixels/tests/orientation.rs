//! Auto-orientation through the public facade (SPEC §Safety and limits).
//!
//! Each fixture stores one picture sideways in some way and says how to turn
//! it upright: EXIF `Orientation` 1–8 in TIFF, PNG, WebP and JPEG, and libavif's
//! `irot`/`imir` in AVIF. The expected rasters come from Pillow — its
//! `exif_transpose` for EXIF, its own rotate and transpose for the AVIF
//! properties — so our orientation code is checked against an independent
//! reading of the same metadata, not against itself.
//! `scripts/regenerate-orientation-reference.py` writes them.

#![cfg(all(
    feature = "raw",
    feature = "tiff",
    feature = "png",
    feature = "webp",
    feature = "jpeg",
    feature = "avif"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, Format, Image, OpenOptions, Orientation};

fn fixture(name: &str) -> String {
    format!(
        "{}/tests/fixtures/orientation/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

/// Decode `image` to packed RGB8 bytes.
fn raw(image: Image) -> Vec<u8> {
    image
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap()
}

/// Open `name` with default options and compare it with its reference raster.
///
/// `stored` is the size as written; orientations that transpose swap it.
/// `tolerance` is zero for every lossless format.
fn check(name: &str, stored: (u32, u32), transposes: bool, tolerance: u8) {
    let image = Image::open(fixture(name)).unwrap();
    let metadata = image.metadata().unwrap();
    let expected_size = if transposes {
        (stored.1, stored.0)
    } else {
        stored
    };
    assert_eq!(
        (metadata.width, metadata.height),
        expected_size,
        "{name}: upright size"
    );

    let ours = raw(image);
    let theirs = std::fs::read(fixture(&format!("{name}.raw"))).unwrap();
    assert_eq!(ours.len(), theirs.len(), "{name}: raster length");
    for (index, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert!(
            a.abs_diff(*b) <= tolerance,
            "{name}: byte {index} is {a}, reference {b}"
        );
    }
}

#[test]
fn every_exif_orientation_in_a_tiff_is_applied() {
    for n in 1..=8 {
        check(&format!("exif{n}.tif"), (7, 5), n >= 5, 0);
    }
}

#[test]
fn every_exif_orientation_in_a_png_is_applied() {
    for n in 1..=8 {
        check(&format!("exif{n}.png"), (7, 5), n >= 5, 0);
    }
}

#[test]
fn every_exif_orientation_in_a_webp_is_applied() {
    for n in 1..=8 {
        check(&format!("exif{n}.webp"), (7, 5), n >= 5, 0);
    }
}

#[test]
fn every_exif_orientation_in_a_jpeg_is_applied() {
    // Lossy, so within libjpeg's rounding of the same flat blocks.
    for n in 1..=8 {
        check(&format!("exif{n}.jpg"), (24, 16), n >= 5, 2);
    }
}

#[test]
fn avif_irot_and_imir_are_applied() {
    for (name, transposes) in [
        ("irot1", true),
        ("irot2", false),
        ("irot3", true),
        ("imir0", false),
        ("imir1", false),
        ("irot1_imir0", true),
        ("irot1_imir1", true),
        ("irot3_imir1", true),
    ] {
        check(&format!("avif_{name}.avif"), (7, 5), transposes, 0);
    }
}

#[test]
fn auto_orient_off_leaves_the_pixels_as_stored() {
    let stored = std::fs::read(fixture("exif1.tif.raw")).unwrap();
    for name in [
        "exif6.tif",
        "exif5.png",
        "exif7.webp",
        "avif_irot1_imir1.avif",
    ] {
        let options = OpenOptions::default().with_auto_orient(false);
        let image = Image::open_with(fixture(name), options).unwrap();
        let metadata = image.metadata().unwrap();
        assert_eq!((metadata.width, metadata.height), (7, 5), "{name}");
        assert_eq!(raw(image), stored, "{name}");
    }
}

#[test]
fn orient_by_hand_matches_auto_orient() {
    // The escape hatch for pixels opened unoriented: the same transform,
    // applied explicitly, gives the same result.
    let options = OpenOptions::default().with_auto_orient(false);
    for n in 1..=8_u16 {
        let name = fixture(&format!("exif{n}.tif"));
        let orientation = Orientation::from_exif(n).unwrap();
        let by_hand = Image::open_with(&name, options)
            .unwrap()
            .orient(orientation);
        assert_eq!(raw(by_hand), raw(Image::open(&name).unwrap()), "exif {n}");
    }
}

#[test]
fn auto_orient_is_on_by_default_and_from_stream_applies_it() {
    assert!(OpenOptions::default().auto_orient);
    let bytes = std::fs::read(fixture("exif6.webp")).unwrap();
    let image = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
    let metadata = image.metadata().unwrap();
    assert_eq!((metadata.width, metadata.height), (5, 7));
}

#[test]
fn an_oriented_jpeg_keeps_shrink_on_load() {
    // Rotation and mirroring carry no pixel lengths, so they rescale and the
    // planner can still decode a thumbnail at reduced scale. A phone photo is
    // exactly the image that is both rotated and thumbnailed.
    let plain = std::fs::read(format!(
        "{}/../otf-pixels-codec-jpeg/tests/fixtures/gradient444.jpg",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    // An APP1 EXIF segment declaring orientation 6, right after the SOI.
    let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0".to_vec();
    let mut tagged = vec![0xFF, 0xD8, 0xFF, 0xE1];
    tagged.extend_from_slice(&u16::try_from(exif.len() + 2).unwrap().to_be_bytes());
    tagged.append(&mut exif);
    tagged.extend_from_slice(&plain[2..]);

    let image = Image::from_stream(std::io::Cursor::new(tagged)).unwrap();
    let upright = image.metadata().unwrap();
    let stored = Image::from_stream(std::io::Cursor::new(plain))
        .unwrap()
        .metadata()
        .unwrap();
    assert_eq!(
        (upright.width, upright.height),
        (stored.height, stored.width)
    );

    let (w, h) = (upright.width / 8, upright.height / 8);
    let mut sink = Vec::new();
    let stats = image
        .resize(w, h)
        .output(Format::Raw, EncodeOptions::default())
        .write_with_stats(&mut sink)
        .unwrap();
    assert!(stats.reduction.is_some(), "no shrink-on-load: {stats:?}");
    assert_eq!(sink.len(), (w * h * 3) as usize);
}
