//! What this decoder refuses, and how it says so.
//!
//! Two kinds of refusal, and the difference between them is part of the API:
//!
//! - [`ErrorCode::Unsupported`] — the file is a valid JPEG that this codec
//!   does not own. Progressive, arithmetic-coded, 12-bit and CMYK files are
//!   real images a caller may reasonably hold, and the error says so, because
//!   a host binding routes on it (SPEC §Embedding notes) and because a wrapped
//!   codec can take over later without any caller changing.
//! - [`ErrorCode::Malformed`] — the bytes do not describe an image at all.
//!
//! Getting these the wrong way round is a real defect: reporting a progressive
//! JPEG as corrupt sends a user looking for a broken file that is fine.
//!
//! [`ErrorCode::Unsupported`]: otf_pixels_core::ErrorCode::Unsupported
//! [`ErrorCode::Malformed`]: otf_pixels_core::ErrorCode::Malformed

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_jpeg::JpegDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits, PixelsError};

fn read_fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{name}.jpg", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

fn open(bytes: &[u8]) -> Result<JpegDecoder<&[u8]>, PixelsError> {
    JpegDecoder::new(bytes, Limits::default())
}

/// A baseline frame header, as a starting point for building broken ones.
///
/// 16x16, three components, 2x2 luma sampling — the commonest layout there is.
fn frame_segment(marker: u8, precision: u8, components: &[[u8; 3]]) -> Vec<u8> {
    let mut payload = vec![precision];
    payload.extend_from_slice(&16_u16.to_be_bytes());
    payload.extend_from_slice(&16_u16.to_be_bytes());
    payload.push(components.len() as u8);
    for component in components {
        payload.extend_from_slice(component);
    }

    let mut bytes = vec![0xFF, 0xD8, 0xFF, marker];
    bytes.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    bytes.extend_from_slice(&payload);
    bytes
}

/// The standard three-component layout: Y at 2x2, Cb and Cr at 1x1.
fn ycbcr() -> Vec<[u8; 3]> {
    vec![[1, 0x22, 0], [2, 0x11, 1], [3, 0x11, 1]]
}

#[test]
fn progressive_jpeg_is_unsupported_not_malformed() {
    let error = open(&read_fixture("progressive")).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported, "{error}");
    assert!(
        error.to_string().contains("progressive"),
        "the message should name the reason: {error}"
    );
}

#[test]
fn cmyk_jpeg_is_unsupported_not_malformed() {
    let error = open(&read_fixture("cmyk")).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported, "{error}");
}

#[test]
fn arithmetic_coding_and_twelve_bit_are_unsupported() {
    // DAC — arithmetic conditioning — appearing at all means the scan is
    // arithmetic coded.
    let mut bytes = frame_segment(0xC0, 8, &ycbcr());
    bytes.extend_from_slice(&[0xFF, 0xCC, 0x00, 0x03, 0x00]);
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    let error = open(&bytes).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported, "{error}");

    // 12-bit samples in an otherwise baseline-looking frame.
    let bytes = frame_segment(0xC0, 12, &ycbcr());
    let error = open(&bytes).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported, "{error}");

    // Lossless (SOF3) and differential frame types are JPEG, just not ours.
    for marker in [0xC3_u8, 0xC5, 0xC9, 0xCF] {
        let bytes = frame_segment(marker, 8, &ycbcr());
        let error = open(&bytes).unwrap_err();
        assert_eq!(
            error.code(),
            ErrorCode::Unsupported,
            "{marker:#04x}: {error}"
        );
    }
}

#[test]
fn a_stream_that_is_not_a_jpeg_is_malformed() {
    for bytes in [
        &b""[..],
        &b"\xFF\xD8"[..],
        &b"not a jpeg at all"[..],
        &b"\x89PNG\r\n\x1a\n"[..],
        // A JPEG signature and nothing after it.
        &b"\xFF\xD8\xFF"[..],
    ] {
        let error = open(bytes).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed, "{bytes:?}: {error}");
    }
}

#[test]
fn a_frame_with_no_scan_is_malformed() {
    let mut bytes = frame_segment(0xC0, 8, &ycbcr());
    bytes.extend_from_slice(&[0xFF, 0xD9]);
    let error = open(&bytes).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Malformed, "{error}");
}

#[test]
fn degenerate_sampling_factors_are_malformed() {
    // A zero sampling factor would make an MCU zero blocks wide, and every
    // division by it a division by zero.
    let error = open(&frame_segment(
        0xC0,
        8,
        &[[1, 0x02, 0], [2, 0x11, 1], [3, 0x11, 1]],
    ))
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::Malformed, "{error}");

    // 4x4 on all three components is 48 blocks per MCU, well past the ten the
    // format permits, and a way to make one MCU arbitrarily expensive.
    let error = open(&frame_segment(
        0xC0,
        8,
        &[[1, 0x44, 0], [2, 0x44, 1], [3, 0x44, 1]],
    ))
    .unwrap_err();
    assert_eq!(error.code(), ErrorCode::Malformed, "{error}");
}

/// A scan naming tables that no `DQT` or `DHT` defined must be refused before
/// decoding, not decoded against zeroes.
#[test]
fn a_scan_naming_undefined_tables_is_malformed() {
    let mut bytes = frame_segment(0xC0, 8, &ycbcr());
    // SOS for all three components, no DQT or DHT anywhere.
    bytes.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x0C, 0x03]);
    bytes.extend_from_slice(&[1, 0x00, 2, 0x11, 3, 0x11]);
    bytes.extend_from_slice(&[0x00, 0x3F, 0x00]);
    let error = open(&bytes).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Malformed, "{error}");
    assert!(
        error.to_string().contains("never defined"),
        "the message should say what is missing: {error}"
    );
}

#[test]
fn the_row_contract_is_enforced() {
    let bytes = read_fixture("gradient444");
    let mut decoder = open(&bytes).unwrap();
    let row_bytes = decoder.descriptor().row_bytes();

    // A buffer of the wrong length is a caller error, and must be rejected
    // before it can be partially filled.
    let mut wrong = vec![0_u8; row_bytes - 1];
    assert_eq!(
        decoder.read_row(&mut wrong).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    let mut wrong = vec![0_u8; row_bytes + 1];
    assert_eq!(
        decoder.read_row(&mut wrong).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );

    // Reading past the last row is an error rather than a repeat of the last.
    let mut row = vec![0_u8; row_bytes];
    for _ in 0..decoder.descriptor().height {
        decoder.read_row(&mut row).unwrap();
    }
    assert_eq!(
        decoder.read_row(&mut row).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

/// Metadata must come from the header alone, with no scan decoded.
#[test]
fn headers_are_readable_without_the_scan() {
    let whole = read_fixture("gradient420");
    // Cut the file immediately after the scan header, so it contains every
    // header and not one byte of entropy data.
    let sos = whole
        .windows(2)
        .position(|pair| pair == [0xFF, 0xDA])
        .expect("the fixture has a scan");
    let length = usize::from(u16::from_be_bytes([whole[sos + 2], whole[sos + 3]]));
    let decoder = open(&whole[..sos + 2 + length]).expect("headers parse without the scan");
    let descriptor = decoder.descriptor();
    assert_eq!((descriptor.width, descriptor.height), (64, 48));
    assert_eq!(descriptor.pixel, otf_pixels_core::PixelFormat::Rgb8);
}
