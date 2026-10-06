//! `eXIf` orientation, and the chunks before `IDAT` being read at construction.
//!
//! Each case splices a chunk into a PngSuite file and checks what the decoder
//! reports — and, because reading `eXIf` moved into the constructor, that the
//! pixels still decode exactly as the untouched file's do.

#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_png::PngDecoder;
use otf_pixels_compress::Crc32;
use otf_pixels_core::{Decoder, ErrorCode, Limits, Orientation, PixelFormat};

fn pngsuite(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/pngsuite/{name}.png",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

fn chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(payload.len()).unwrap().to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(payload);
    out.extend_from_slice(&crc.finish().to_be_bytes());
    out
}

/// A bare (identifier-less) EXIF block declaring orientation `value`,
/// followed by `padding` bytes standing in for a thumbnail.
fn exif(value: u8, padding: usize) -> Vec<u8> {
    let mut block = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\0\0\0\0".to_vec();
    block[18] = value;
    block.resize(block.len() + padding, 0xAB);
    block
}

/// `png` with `inserted` placed immediately before the chunk named `before`.
fn splice(png: &[u8], before: &[u8; 4], inserted: &[u8]) -> Vec<u8> {
    let at = png.windows(4).position(|w| w == before).unwrap() - 4;
    [&png[..at], inserted, &png[at..]].concat()
}

fn decode(bytes: &[u8]) -> (PngDecoder<&[u8]>, Vec<u8>) {
    let mut decoder = PngDecoder::new(bytes, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut pixels = vec![0_u8; descriptor.row_bytes() * descriptor.height as usize];
    for row in pixels.chunks_mut(descriptor.row_bytes()) {
        decoder.read_row(row).unwrap();
    }
    (decoder, pixels)
}

#[test]
fn exif_before_the_image_data_is_reported_and_the_pixels_are_unchanged() {
    // One non-interlaced and one interlaced file: the two decode paths pick
    // up from the constructor differently.
    for name in ["basn2c08", "basi2c08"] {
        let plain = pngsuite(name);
        let (_, expected) = decode(&plain);
        let tagged = splice(&plain, b"IDAT", &chunk(b"eXIf", &exif(6, 0)));
        let (decoder, pixels) = decode(&tagged);
        assert_eq!(decoder.orientation(), Orientation::Rotate90, "{name}");
        assert_eq!(pixels, expected, "{name}");
    }
}

#[test]
fn a_file_without_exif_is_upright() {
    let plain = pngsuite("basn2c08");
    let (decoder, _) = decode(&plain);
    assert_eq!(decoder.orientation(), Orientation::Normal);
}

#[test]
fn exif_after_the_image_data_is_not_seen() {
    let plain = pngsuite("basn2c08");
    let tagged = splice(&plain, b"IEND", &chunk(b"eXIf", &exif(6, 0)));
    let (decoder, pixels) = decode(&tagged);
    assert_eq!(decoder.orientation(), Orientation::Normal);
    assert_eq!(pixels, decode(&plain).1);
}

#[test]
fn a_large_exif_chunk_is_read_past_without_disturbing_the_stream() {
    // Larger than the prefix the decoder reads, so the rest is skipped — and
    // the CRC still has to check out across the whole chunk.
    let plain = pngsuite("basn2c08");
    let tagged = splice(&plain, b"IDAT", &chunk(b"eXIf", &exif(3, 200_000)));
    let (decoder, pixels) = decode(&tagged);
    assert_eq!(decoder.orientation(), Orientation::Rotate180);
    assert_eq!(pixels, decode(&plain).1);
}

#[test]
fn broken_exif_is_metadata_declined_not_an_image_refused() {
    let plain = pngsuite("basn2c08");
    for payload in [b"garbage".to_vec(), exif(9, 0), Vec::new()] {
        let tagged = splice(&plain, b"IDAT", &chunk(b"eXIf", &payload));
        let (decoder, pixels) = decode(&tagged);
        assert_eq!(decoder.orientation(), Orientation::Normal);
        assert_eq!(pixels, decode(&plain).1);
    }
}

#[test]
fn a_corrupt_exif_crc_is_still_malformed() {
    let plain = pngsuite("basn2c08");
    let mut bad = chunk(b"eXIf", &exif(6, 0));
    let last = bad.len() - 1;
    bad[last] ^= 0xFF;
    let tagged = splice(&plain, b"IDAT", &bad);
    let error = PngDecoder::new(&tagged[..], Limits::default()).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Malformed, "{error}");
}

#[test]
fn transparency_is_in_the_descriptor_from_construction() {
    // tRNS precedes IDAT, so the format is final before any row is read.
    let bytes = pngsuite("tbrn2c08");
    let decoder = PngDecoder::new(&bytes[..], Limits::default()).unwrap();
    assert_eq!(decoder.descriptor().pixel, PixelFormat::Rgba8);
}
