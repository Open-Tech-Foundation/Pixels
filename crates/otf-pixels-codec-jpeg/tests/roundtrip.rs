//! Encode, then decode, and check what survived.
//!
//! # What a round trip can and cannot prove
//!
//! It cannot prove correctness on its own: a shared misreading of the
//! specification round-trips perfectly and is still wrong. That is what
//! `tests/reference.rs` (libjpeg's output, decoded by us) and
//! `scripts/check-jpeg-interop.sh` (our output, decoded by libjpeg) are for.
//!
//! What it *can* prove is the thing neither of those isolates: that the
//! encoder's forward transform, quantizer and entropy coder are the exact
//! inverses of the decoder's, at every quality and subsampling, over image
//! shapes that land awkwardly against the MCU grid. A round trip that loses
//! more than the quantizer explains means one of those pairs disagrees.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_jpeg::{JpegDecoder, JpegEncoder, Subsampling};
use otf_pixels_core::{
    Decoder, EncodeOptions, Encoder, ErrorCode, ImageDescriptor, Limits, PixelFormat,
};

/// Encode a raster, returning the JPEG bytes.
fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    encoder: JpegEncoder,
) -> Vec<u8> {
    let descriptor = ImageDescriptor::new(width, height, format).unwrap();
    let mut encoder = encoder;
    let mut out = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();
    for row in pixels.chunks_exact(descriptor.row_bytes()) {
        encoder.write_row(row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

/// Decode JPEG bytes back to a raster.
fn decode(bytes: &[u8]) -> (Vec<u8>, u32, u32, PixelFormat) {
    let mut decoder = JpegDecoder::new(bytes, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut pixels = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row).unwrap();
        pixels.extend_from_slice(&row);
    }
    (
        pixels,
        descriptor.width,
        descriptor.height,
        descriptor.pixel,
    )
}

/// Largest and mean absolute difference between two rasters.
fn difference(ours: &[u8], theirs: &[u8]) -> (u32, f64) {
    let mut worst = 0_u32;
    let mut total = 0_u64;
    for (&got, &want) in ours.iter().zip(theirs) {
        let delta = u32::from(got.abs_diff(want));
        worst = worst.max(delta);
        total += u64::from(delta);
    }
    (worst, total as f64 / ours.len().max(1) as f64)
}

/// A smooth two-axis gradient — what a DCT codes well.
fn gradient(width: u32, height: u32, channels: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height) as usize * channels);
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / width.max(2).saturating_sub(1).max(1)) as u8;
            let g = (y * 255 / height.max(2).saturating_sub(1).max(1)) as u8;
            let b = ((x + y) * 255 / (width + height).saturating_sub(2).max(1)) as u8;
            match channels {
                1 => out.push(((u32::from(r) + u32::from(g)) / 2) as u8),
                _ => out.extend_from_slice(&[r, g, b]),
            }
        }
    }
    out
}

#[test]
fn a_gradient_survives_the_round_trip_at_high_quality() {
    // 4:4:4 at quality 95: no chroma subsampling, fine quantization steps.
    // What is left is transform error, and it must be small.
    for &(width, height) in &[(64_u32, 64_u32), (61, 37), (17, 3), (8, 8), (1, 1)] {
        let source = gradient(width, height, 3);
        let bytes = encode(
            &source,
            width,
            height,
            PixelFormat::Rgb8,
            JpegEncoder::with_quality(95).unwrap(),
        );
        let (decoded, w, h, pixel) = decode(&bytes);

        assert_eq!((w, h), (width, height), "{width}x{height}: shape");
        assert_eq!(pixel, PixelFormat::Rgb8);
        assert_eq!(decoded.len(), source.len());

        let (worst, mean) = difference(&decoded, &source);
        assert!(
            worst <= 12 && mean <= 2.0,
            "{width}x{height}: worst {worst}, mean {mean:.3}"
        );
    }
}

#[test]
fn grayscale_round_trips_as_grayscale() {
    let (width, height) = (61_u32, 37_u32);
    let source = gradient(width, height, 1);
    let bytes = encode(
        &source,
        width,
        height,
        PixelFormat::Gray8,
        JpegEncoder::with_quality(95).unwrap(),
    );
    let (decoded, .., pixel) = decode(&bytes);

    // One component in, one component out — not an RGB image with equal
    // channels, which would triple the bytes for nothing.
    assert_eq!(pixel, PixelFormat::Gray8);
    assert_eq!(decoded.len(), source.len());
    let (worst, mean) = difference(&decoded, &source);
    assert!(worst <= 8 && mean <= 1.5, "worst {worst}, mean {mean:.3}");
}

#[test]
fn every_subsampling_round_trips() {
    let (width, height) = (48_u32, 32_u32);
    let source = gradient(width, height, 3);

    for subsampling in [
        Subsampling::None,
        Subsampling::Horizontal,
        Subsampling::Both,
    ] {
        let bytes = encode(
            &source,
            width,
            height,
            PixelFormat::Rgb8,
            JpegEncoder::with_quality(92)
                .unwrap()
                .with_subsampling(subsampling),
        );
        let (decoded, w, h, _) = decode(&bytes);
        assert_eq!((w, h), (width, height), "{subsampling:?}");

        // Chroma subsampling costs colour accuracy on a gradient but very
        // little: the chroma of a gradient is itself a gradient.
        let (worst, mean) = difference(&decoded, &source);
        assert!(
            worst <= 16 && mean <= 3.0,
            "{subsampling:?}: worst {worst}, mean {mean:.3}"
        );
    }
}

#[test]
fn quality_trades_size_against_fidelity_monotonically() {
    let (width, height) = (64_u32, 64_u32);
    let source = gradient(width, height, 3);
    let mut previous_size = 0_usize;
    let mut previous_error = f64::MAX;

    for quality in [10_u8, 30, 50, 75, 90, 100] {
        let bytes = encode(
            &source,
            width,
            height,
            PixelFormat::Rgb8,
            JpegEncoder::with_quality(quality)
                .unwrap()
                // Held fixed, so the trend measures quantization alone rather
                // than the subsampling switch that quality would otherwise
                // flip at 90.
                .with_subsampling(Subsampling::None),
        );
        let (decoded, ..) = decode(&bytes);
        let (_, error) = difference(&decoded, &source);

        assert!(
            bytes.len() > previous_size,
            "quality {quality}: {} bytes is not more than {previous_size}",
            bytes.len()
        );
        assert!(
            error <= previous_error,
            "quality {quality}: error {error:.3} is worse than {previous_error:.3}"
        );
        previous_size = bytes.len();
        previous_error = error;
    }
    // The best quality this encoder offers should be visually lossless.
    assert!(
        previous_error < 1.0,
        "quality 100 error {previous_error:.3}"
    );
}

/// Alpha has to become something; JPEG has no way to keep it.
#[test]
fn alpha_is_composited_rather_than_dropped() {
    let (width, height) = (16_u32, 16_u32);
    // A half-transparent red: composited against black this is dark red, and
    // dropping alpha instead would leave it bright red.
    let source: Vec<u8> = (0..(width * height))
        .flat_map(|_| [255_u8, 0, 0, 128])
        .collect();
    let bytes = encode(
        &source,
        width,
        height,
        PixelFormat::Rgba8,
        JpegEncoder::with_quality(95).unwrap(),
    );
    let (decoded, .., pixel) = decode(&bytes);
    assert_eq!(pixel, PixelFormat::Rgb8, "JPEG has no alpha channel");

    let red = u32::from(decoded[0]);
    assert!(
        (120..=136).contains(&red),
        "expected red composited to about 128, got {red}"
    );
    assert!(decoded[1] < 16 && decoded[2] < 16, "colour drifted");
}

#[test]
fn encoding_is_deterministic() {
    let (width, height) = (40_u32, 24_u32);
    let source = gradient(width, height, 3);
    let first = encode(
        &source,
        width,
        height,
        PixelFormat::Rgb8,
        JpegEncoder::new(),
    );
    let second = encode(
        &source,
        width,
        height,
        PixelFormat::Rgb8,
        JpegEncoder::new(),
    );
    assert_eq!(first, second, "the same pixels encoded to different bytes");
}

/// The encoder writes as it goes rather than buffering the image.
#[test]
fn rows_reach_the_sink_before_the_image_is_complete() {
    let (width, height) = (64_u32, 128_u32);
    let source = gradient(width, height, 3);
    let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();

    let mut encoder = JpegEncoder::new();
    let mut out = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();
    let after_header = out.len();

    let row_bytes = descriptor.row_bytes();
    for row in source.chunks_exact(row_bytes).take(height as usize / 2) {
        encoder.write_row(row, &mut out).unwrap();
    }
    let halfway = out.len();
    assert!(
        halfway > after_header,
        "no entropy data was written after half the rows: \
         the encoder is buffering the image, not streaming it"
    );

    for row in source.chunks_exact(row_bytes).skip(height as usize / 2) {
        encoder.write_row(row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    assert!(out.len() > halfway);

    // And the result still decodes to the right shape.
    let (_, w, h, _) = decode(&out);
    assert_eq!((w, h), (width, height));
}

#[test]
fn the_encoder_contract_is_enforced() {
    let descriptor = ImageDescriptor::new(8, 8, PixelFormat::Rgb8).unwrap();
    let row = vec![0_u8; descriptor.row_bytes()];

    // Rows before a header.
    let mut encoder = JpegEncoder::new();
    let mut out = Vec::new();
    assert_eq!(
        encoder.write_row(&row, &mut out).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    assert_eq!(
        encoder.finish(&mut out).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );

    // A second header.
    let mut encoder = JpegEncoder::new();
    let mut out = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();
    assert_eq!(
        encoder
            .write_header(&descriptor, &mut out)
            .unwrap_err()
            .code(),
        ErrorCode::InvalidArgument
    );

    // A row of the wrong length.
    let short = vec![0_u8; descriptor.row_bytes() - 1];
    assert_eq!(
        encoder.write_row(&short, &mut out).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );

    // Finishing early must not emit a truncated image that looks whole.
    for _ in 0..4 {
        encoder.write_row(&row, &mut out).unwrap();
    }
    assert_eq!(
        encoder.finish(&mut out).unwrap_err().code(),
        ErrorCode::Malformed
    );

    // More rows than the header declared.
    for _ in 4..8 {
        encoder.write_row(&row, &mut out).unwrap();
    }
    assert_eq!(
        encoder.write_row(&row, &mut out).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    encoder.finish(&mut out).unwrap();
}

#[test]
fn generic_encode_options_reach_the_encoder() {
    let (width, height) = (32_u32, 32_u32);
    let source = gradient(width, height, 3);
    let low = encode(
        &source,
        width,
        height,
        PixelFormat::Rgb8,
        JpegEncoder::from_options(&EncodeOptions::with_quality(20).unwrap()),
    );
    let high = encode(
        &source,
        width,
        height,
        PixelFormat::Rgb8,
        JpegEncoder::from_options(&EncodeOptions::with_quality(95).unwrap()),
    );
    assert!(
        low.len() < high.len(),
        "quality did not reach the encoder: {} vs {} bytes",
        low.len(),
        high.len()
    );
}

/// Our own output must satisfy our own decoder's validation, including the
/// parts that reject malformed files.
#[test]
fn emitted_files_are_well_formed_streams() {
    let source = gradient(23, 19, 3);
    let bytes = encode(&source, 23, 19, PixelFormat::Rgb8, JpegEncoder::new());

    assert!(otf_pixels_codec_jpeg::probe(&bytes), "bad signature");
    assert_eq!(&bytes[..2], &[0xFF, 0xD8], "no SOI");
    assert_eq!(&bytes[bytes.len() - 2..], &[0xFF, 0xD9], "no EOI");

    // Every 0xFF in the entropy stream must be stuffed or a real marker.
    // Walking the segment structure is what proves the stuffing is right.
    let mut at = 2_usize;
    let mut saw_scan = false;
    while at + 1 < bytes.len() {
        assert_eq!(bytes[at], 0xFF, "expected a marker at {at}");
        let code = bytes[at + 1];
        at += 2;
        if code == 0xD9 {
            break;
        }
        let length = usize::from(u16::from_be_bytes([bytes[at], bytes[at + 1]]));
        assert!(length >= 2, "segment at {at} declares length {length}");
        at += length;
        if code == 0xDA {
            // The scan runs to the next marker that is not a stuffed 0xFF.
            saw_scan = true;
            while at + 1 < bytes.len() {
                if bytes[at] == 0xFF && bytes[at + 1] != 0x00 {
                    break;
                }
                at += 1;
            }
        }
    }
    assert!(saw_scan, "no scan was written");
}
