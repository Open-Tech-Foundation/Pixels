//! AVIF decode checked against libaom/libavif, not against ourselves.
//!
//! The codec is owned end to end (ADR-0013), so the AV1 bitstream *is* on
//! trial. The reference rasters are produced by libavif's `avifdec`, and a
//! lossless fixture must match one to the byte. A fixture that exercised a tool
//! this decoder does not implement would report `Unsupported` and be skipped
//! rather than asserted; the whole lossless corpus currently decodes.
//!
//! Regenerate the fixtures and manifest with `scripts/regenerate-avif-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_avif::AvifDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits, PixelFormat};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str, extension: &str) -> Vec<u8> {
    let path = format!("{}/{name}.{extension}", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// One fixture as the manifest describes it: `name width height channels tolerance`.
struct Reference {
    name: String,
    width: u32,
    height: u32,
    channels: usize,
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
            }
        })
        .collect()
}

fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(Vec<u8>, otf_pixels_core::ImageDescriptor)> {
    let mut decoder = AvifDecoder::new(bytes, Limits::default())?;
    let descriptor = decoder.descriptor();
    let mut pixels = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row)?;
        pixels.extend_from_slice(&row);
    }
    Ok((pixels, descriptor))
}

/// Exact means exact: for every fixture this decoder can handle — the lossless
/// ones and the filters-off lossy ("nofilter") ones — the raster must equal
/// libavif's to the byte. The nofilter fixtures are genuinely lossy (DCT/ADST,
/// larger transforms, chroma-from-luma) but code every in-loop post-filter off,
/// so a filter-free reconstruct still reproduces them exactly.
#[test]
fn reference_fixtures_decode_exactly() {
    let mut compared = 0;
    let mut lossy_compared = 0;
    for reference in references() {
        let result = decode(&read_fixture(&reference.name, "avif"));
        let (ours, descriptor) = match result {
            Ok(pair) => pair,
            Err(e) if e.code() == ErrorCode::Unsupported => {
                // A tool this phase does not implement (e.g. palette). The
                // decode refused cleanly rather than producing a wrong raster.
                continue;
            }
            Err(e) => panic!("{}: {e}", reference.name),
        };
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
            "{}: pixel format",
            reference.name
        );
        assert_eq!(
            ours, theirs,
            "{}: a decode differs from libavif's",
            reference.name
        );
        compared += 1;
        if reference.name.contains("nofilter") {
            lossy_compared += 1;
        }
    }
    assert!(compared >= 2, "only {compared} fixtures decoded");
    assert!(
        lossy_compared >= 1,
        "no filters-off lossy fixture was compared — the lossy path regressed \
         to Unsupported or the manifest lost its nofilter entries"
    );
}
