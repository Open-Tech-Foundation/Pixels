//! Progressive JPEG through the facade: the wrapped decoder behind the same
//! sniffing and `Image` API as a baseline file.

#![cfg(all(feature = "jpeg-progressive", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, Format, Image};

/// A progressive JPEG reaches the wrapped decoder through the same
/// sniffing and the same `Image` API as a baseline one.
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

/// Shrink-on-load does not fire for a progressive source: the wrapped
/// decoder has one resolution, and claiming otherwise would be a promise
/// nothing could keep.
#[test]
fn a_progressive_source_does_not_shrink_on_load() {
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
    assert!(stats.reduction.is_none(), "{:?}", stats.reduction);
}

