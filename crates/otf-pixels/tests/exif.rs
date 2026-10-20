//! sharp's `keepExif`: an EXIF block read from JPEG, PNG, WebP and AVIF is
//! dropped from outputs by default and written, with its orientation made
//! upright, when an output keeps it.

#![cfg(all(
    feature = "raw",
    feature = "png",
    feature = "jpeg",
    feature = "webp",
    feature = "avif",
    feature = "gif"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    EncodeOptions, Format, Image, ImageDescriptor, OpenOptions, Orientation, PixelFormat,
};

/// A little-endian EXIF block: IFD0 with Make "Pixels" and `orientation`.
fn exif_block(orientation: u16) -> Vec<u8> {
    let mut b = b"II*\0\x08\0\0\0".to_vec();
    b.extend_from_slice(&2_u16.to_le_bytes());
    // Make: ASCII, 7 bytes, at offset 8 + 2 + 24 + 4 = 38.
    b.extend_from_slice(&[0x0F, 0x01, 2, 0]);
    b.extend_from_slice(&7_u32.to_le_bytes());
    b.extend_from_slice(&38_u32.to_le_bytes());
    // Orientation: SHORT, 1, inline.
    b.extend_from_slice(&[0x12, 0x01, 3, 0]);
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&orientation.to_le_bytes());
    b.extend_from_slice(&[0, 0]);
    b.extend_from_slice(&0_u32.to_le_bytes()); // no IFD1
    b.extend_from_slice(b"Pixels\0");
    b
}

/// A 6x4 RGB image carrying `exif`.
fn source(exif: Vec<u8>) -> Image {
    let descriptor = ImageDescriptor::new(6, 4, PixelFormat::Rgb8).unwrap();
    let bytes = (0..6 * 4 * 3).map(|i| (i * 9) as u8).collect();
    Image::from_raw(descriptor, bytes)
        .unwrap()
        .with_exif(Some(exif))
}

fn encode(image: Image, format: Format, keep: bool) -> Vec<u8> {
    image
        .output(format, EncodeOptions::default().with_keep_exif(keep))
        .bytes()
        .unwrap()
}

fn open(bytes: Vec<u8>) -> Image {
    Image::from_stream(std::io::Cursor::new(bytes)).unwrap()
}

#[test]
fn every_format_with_a_place_for_exif_keeps_it_when_asked() {
    for format in [Format::Jpeg, Format::Png, Format::WebP, Format::Avif] {
        let block = exif_block(1);
        let kept = open(encode(source(block.clone()), format, true));
        assert_eq!(kept.exif(), Some(block.as_slice()), "{format}");
        let dropped = open(encode(source(block), format, false));
        assert_eq!(dropped.exif(), None, "{format} strips EXIF by default");
    }
}

#[test]
fn exif_rides_along_with_alpha_palettes_and_lossless() {
    let block = exif_block(1);
    let rgba = || {
        let descriptor = ImageDescriptor::new(5, 5, PixelFormat::Rgba8).unwrap();
        Image::from_raw(descriptor, (0..100).map(|i| (i * 5) as u8).collect())
            .unwrap()
            .with_exif(Some(block.clone()))
    };
    for options in [
        (Format::Png, EncodeOptions::default().with_palette(true)),
        (Format::WebP, EncodeOptions::default().with_lossless(true)),
        (Format::WebP, EncodeOptions::default()),
        (Format::Avif, EncodeOptions::default()),
    ] {
        let bytes = rgba()
            .output(options.0, options.1.with_keep_exif(true))
            .bytes()
            .unwrap();
        let image = open(bytes);
        assert_eq!(image.exif(), Some(block.as_slice()), "{options:?}");
        assert_eq!(image.descriptor().unwrap().pixel.channels(), 4);
    }
}

#[test]
fn an_auto_oriented_image_keeps_an_upright_orientation_tag() {
    // Written as stored, then read back: the file says "rotate 90".
    let jpeg = encode(source(exif_block(6)), Format::Jpeg, true);

    let upright = open(jpeg.clone());
    assert_eq!(upright.descriptor().unwrap().width, 4, "turned upright");
    let exif = upright.exif().unwrap();
    assert_eq!(
        Orientation::from_exif_block(exif),
        Some(Orientation::Normal)
    );
    assert!(exif.ends_with(b"Pixels\0"), "the other tags are untouched");
    let rewritten = open(encode(upright, Format::Png, true));
    assert_eq!(
        Orientation::from_exif_block(rewritten.exif().unwrap()),
        Some(Orientation::Normal),
        "a viewer will not rotate it a second time"
    );

    // Opened as stored, the tag is left saying what to do.
    let options = OpenOptions::default().with_auto_orient(false);
    let stored = Image::from_stream_with(std::io::Cursor::new(jpeg), options).unwrap();
    assert_eq!(stored.descriptor().unwrap().width, 6);
    assert_eq!(
        Orientation::from_exif_block(stored.exif().unwrap()),
        Some(Orientation::Rotate90)
    );
}

#[test]
fn exif_survives_the_pipeline_and_can_be_replaced() {
    let block = exif_block(1);
    let processed = source(block.clone())
        .resize(3, 2)
        .blur(0.5)
        .composite(source(exif_block(3)), 0, 0)
        .flop();
    assert_eq!(
        processed.exif(),
        Some(block.as_slice()),
        "the base's EXIF wins"
    );
    let replaced = processed.with_exif(None);
    assert_eq!(open(encode(replaced, Format::Jpeg, true)).exif(), None);
}

#[test]
fn formats_without_a_place_for_exif_ignore_it() {
    for format in [Format::Gif, Format::Raw] {
        let bytes = encode(source(exif_block(1)), format, true);
        assert!(!bytes.is_empty(), "{format}");
    }
}

#[test]
fn a_block_too_large_for_jpeg_is_dropped_not_fatal() {
    let mut huge = exif_block(1);
    huge.resize(70_000, 0);
    let jpeg = encode(source(huge.clone()), Format::Jpeg, true);
    assert_eq!(open(jpeg).exif(), None);
    // PNG has room for it.
    assert_eq!(
        open(encode(source(huge.clone()), Format::Png, false)).exif(),
        None
    );
    let png = encode(source(huge), Format::Png, true);
    assert!(png.len() > 70_000);
}
