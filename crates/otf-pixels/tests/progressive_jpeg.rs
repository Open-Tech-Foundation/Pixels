//! Progressive JPEG through the facade: the same sniffing, `Image` API and
//! shrink-on-load as a baseline file.

#![cfg(all(feature = "jpeg", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, Format, Image};

/// A progressive JPEG goes through the same sniffing and the same `Image`
/// API as a baseline one.
#[test]
fn a_progressive_jpeg_decodes_through_the_facade() {
    let path = format!(
        "{}/../otf-pixels-codec-jpeg/tests/fixtures/progressive.jpg",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));

    let image = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
    let metadata = image.metadata().unwrap();
    assert_eq!(metadata.format, Format::Jpeg, "sniffing missed the JPEG");
    assert_eq!((metadata.width, metadata.height), (48, 32));

    // And it composes: a progressive source resizes like any other.
    let thumbnail = image
        .resize(16, 16)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    assert_eq!(thumbnail.len(), 16 * 16 * 3);
}

/// Shrink-on-load fires for a progressive source too: its coefficients are
/// inverse-transformed at reduced size, exactly as a baseline file's are.
#[test]
fn a_progressive_source_shrinks_on_load() {
    let path = format!(
        "{}/../otf-pixels-codec-jpeg/tests/fixtures/progressive.jpg",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap();
    let stats = Image::from_stream(std::io::Cursor::new(bytes))
        .unwrap()
        .resize(6, 4)
        .output(Format::Raw, EncodeOptions::default())
        .write_with_stats(&mut Vec::new())
        .unwrap();
    let reduction = stats
        .reduction
        .expect("a 48x32 -> 6x4 resize should shrink on load");
    assert_eq!(reduction.from, (48, 32));
    assert!(reduction.to.0 < 48, "{reduction:?}");
}
