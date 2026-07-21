//! WebP decode checked against libwebp, not against ourselves.
//!
//! The codec is wrapped (ADR-0004), so the VP8 bitstream is not what is on
//! trial — libwebp and `image-webp` are both mature. What is on trial is the
//! adaptation: dimensions, pixel format, whether alpha was detected, row
//! order, and that a row served is the row asked for. Those are what a
//! wrapper gets wrong, and a reference raster is what catches them.
//!
//! Reference rasters come from libwebp via Pillow; regenerate them with
//! `scripts/regenerate-webp-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_webp::WebPDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits, PixelFormat};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str, extension: &str) -> Vec<u8> {
    let path = format!("{}/{name}.{extension}", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// One fixture as the manifest describes it.
struct Reference {
    name: String,
    width: u32,
    height: u32,
    channels: usize,
    lossless: bool,
}

fn references() -> Vec<Reference> {
    let path = format!("{}/REFERENCE", fixture_dir());
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {path}: {e}; run the regeneration script"));
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            Reference {
                name: f[0].to_owned(),
                width: f[1].parse().unwrap(),
                height: f[2].parse().unwrap(),
                channels: f[3].parse().unwrap(),
                lossless: f[4] == "1",
            }
        })
        .collect()
}

fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(Vec<u8>, otf_pixels_core::ImageDescriptor)> {
    let mut decoder = WebPDecoder::new(bytes, Limits::default())?;
    let descriptor = decoder.descriptor();
    let mut pixels = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row)?;
        pixels.extend_from_slice(&row);
    }
    Ok((pixels, descriptor))
}

/// Lossless means lossless: any difference at all is ours.
#[test]
fn lossless_fixtures_decode_exactly() {
    let mut compared = 0;
    for reference in references().into_iter().filter(|r| r.lossless) {
        let (ours, descriptor) = decode(&read_fixture(&reference.name, "webp"))
            .unwrap_or_else(|e| panic!("{}: {e}", reference.name));
        let theirs = read_fixture(&reference.name, "raw");

        assert_eq!(
            (descriptor.width, descriptor.height),
            (reference.width, reference.height),
            "{}: dimensions",
            reference.name
        );
        assert_eq!(
            descriptor.pixel,
            if reference.channels == 4 {
                PixelFormat::Rgba8
            } else {
                PixelFormat::Rgb8
            },
            "{}: alpha detection",
            reference.name
        );
        assert_eq!(
            ours, theirs,
            "{}: a lossless decode differs from libwebp's",
            reference.name
        );
        compared += 1;
    }
    assert!(compared >= 5, "only {compared} lossless fixtures compared");
}

/// Lossy fixtures: two decoders of the same VP8 stream need not agree to the
/// last step, but they must agree about the picture.
#[test]
fn lossy_fixtures_decode_within_tolerance() {
    let mut compared = 0;
    for reference in references().into_iter().filter(|r| !r.lossless) {
        let (ours, descriptor) = decode(&read_fixture(&reference.name, "webp"))
            .unwrap_or_else(|e| panic!("{}: {e}", reference.name));
        let theirs = read_fixture(&reference.name, "raw");

        assert_eq!(
            (descriptor.width, descriptor.height),
            (reference.width, reference.height),
            "{}: dimensions",
            reference.name
        );
        assert_eq!(ours.len(), theirs.len(), "{}: raster size", reference.name);

        let mut worst = 0_u32;
        let mut total = 0_u64;
        for (&got, &want) in ours.iter().zip(&theirs) {
            let delta = u32::from(got.abs_diff(want));
            worst = worst.max(delta);
            total += u64::from(delta);
        }
        let mean = total as f64 / ours.len().max(1) as f64;
        assert!(
            worst <= 8 && mean <= 1.0,
            "{}: worst {worst}, mean {mean:.3}",
            reference.name
        );
        compared += 1;
    }
    assert!(compared >= 3, "only {compared} lossy fixtures compared");
}

/// Greyscale has no native WebP mode, so it comes back as RGB. Recorded
/// because a caller who put one channel in gets three out, which is a real
/// difference from PNG or JPEG and not an accident here.
#[test]
fn greyscale_comes_back_as_rgb() {
    let (_, descriptor) = decode(&read_fixture("grey_lossless", "webp")).unwrap();
    assert_eq!(descriptor.pixel, PixelFormat::Rgb8);
}

#[test]
fn rows_are_served_in_order_and_bounded() {
    let bytes = read_fixture("gradient_lossless", "webp");
    let (whole, descriptor) = decode(&bytes).unwrap();

    let mut decoder = WebPDecoder::new(&bytes[..], Limits::default()).unwrap();
    let row_bytes = descriptor.row_bytes();
    let mut row = vec![0_u8; row_bytes];
    for index in 0..descriptor.height as usize {
        decoder.read_row(&mut row).unwrap();
        assert_eq!(
            row,
            whole[index * row_bytes..(index + 1) * row_bytes],
            "row {index}"
        );
    }
    // One row past the end is an error, not a repeat and not a panic.
    assert_eq!(
        decoder.read_row(&mut row).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    // A buffer of the wrong length is refused too.
    let mut short = vec![0_u8; row_bytes - 1];
    let mut decoder = WebPDecoder::new(&bytes[..], Limits::default()).unwrap();
    assert_eq!(
        decoder.read_row(&mut short).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

#[test]
fn decoding_is_deterministic() {
    for reference in references() {
        let bytes = read_fixture(&reference.name, "webp");
        let (first, _) = decode(&bytes).unwrap();
        let (second, _) = decode(&bytes).unwrap();
        assert_eq!(first, second, "{}: two decodes disagree", reference.name);
    }
}

#[test]
fn probe_recognises_fixtures_and_nothing_else() {
    for reference in references() {
        let bytes = read_fixture(&reference.name, "webp");
        assert!(
            otf_pixels_codec_webp::probe(&bytes),
            "{}: not recognised",
            reference.name
        );
    }
    assert!(!otf_pixels_codec_webp::probe(b"\x89PNG\r\n\x1a\n"));
    assert!(!otf_pixels_codec_webp::probe(b"GIF89a"));
}

/// An oversized header must be refused before the pixel buffer is allocated.
#[test]
fn limits_are_enforced() {
    let bytes = read_fixture("gradient_lossless", "webp");
    let limits = Limits::default().with_max_pixels(16);
    let error = WebPDecoder::new(&bytes[..], limits).unwrap_err();
    assert_eq!(error.code(), ErrorCode::LimitExceeded, "{error}");
}

/// Damage is reported or absorbed, never a panic.
///
/// Note what is *not* asserted: that every truncation fails. A VP8 stream cut
/// short still decodes to a partial picture, and the wrapped decoder returns
/// it — the same choice our own JPEG decoder makes for a truncated scan, and
/// for the same reason: a partially downloaded photograph should show the part
/// that arrived. What must hold is that the container header is not optional,
/// and that nothing panics whatever the bytes say.
#[test]
fn damaged_streams_are_reported_not_panics() {
    let whole = read_fixture("blocks_lossy", "webp");
    // A cut inside the RIFF/VP8 headers leaves nothing decodable.
    for cut in [0, 1, 8, 12, 20] {
        let result = decode(&whole[..cut.min(whole.len())]);
        assert!(result.is_err(), "truncation to {cut} bytes decoded anyway");
    }

    // Flip bits throughout the body and require a value either way.
    let mut rng = 0x5EED_u64;
    for _ in 0..200 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut damaged = whole.clone();
        let at = (rng >> 33) as usize % damaged.len();
        damaged[at] ^= 1 << ((rng >> 13) % 8);
        // Either answer is fine; a panic is not.
        let _ = decode(&damaged);
    }
}
