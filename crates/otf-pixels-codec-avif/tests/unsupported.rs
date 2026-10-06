//! Files this decoder must refuse rather than decode wrong.
//!
//! Each fixture in `tests/fixtures/unsupported` is a real avifenc encode using
//! one coding tool the AV1 decoder does not implement yet. Several of these
//! change what is coded — extra per-block symbols, or a different
//! dequantisation — so a decoder that ignores them does not fail: it produces a
//! plausible, wrong image. The contract is a clean `Unsupported` instead, and
//! each file must be refused *for its own tool*, which the expected message
//! fragment pins down (otherwise a fixture that stopped exercising its tool
//! would still pass on some other refusal).
//!
//! When a tool lands, its file moves into the reference corpus with a raster.
//! Regenerate with `scripts/regenerate-avif-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_avif::AvifDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits};

/// `(fixture, fragment the refusal must mention)`.
const CASES: &[(&str, &str)] = &[("film_grain", "film grain")];

fn decode(bytes: &[u8]) -> otf_pixels_core::Result<()> {
    let mut decoder = AvifDecoder::new(bytes, Limits::default())?;
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    for _ in 0..decoder.descriptor().height {
        decoder.read_row(&mut row)?;
    }
    Ok(())
}

#[test]
fn unimplemented_tools_are_refused_not_decoded_wrong() {
    let dir = format!("{}/tests/fixtures/unsupported", env!("CARGO_MANIFEST_DIR"));
    for (name, fragment) in CASES {
        let path = format!("{dir}/{name}.avif");
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
        match decode(&bytes) {
            Ok(()) => panic!("{name}: decoded, but it uses a tool this decoder lacks"),
            Err(e) => {
                assert_eq!(e.code(), ErrorCode::Unsupported, "{name}: {e}");
                assert!(
                    e.to_string().contains(fragment),
                    "{name}: refused for the wrong reason: {e}"
                );
            }
        }
    }
}

#[test]
fn every_unsupported_fixture_is_covered() {
    // A file added by the regeneration script without a case here would sit
    // unchecked; so would a case whose file was removed.
    let dir = format!("{}/tests/fixtures/unsupported", env!("CARGO_MANIFEST_DIR"));
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".avif").map(str::to_owned))
        .collect();
    files.sort();
    let mut cases: Vec<String> = CASES.iter().map(|(name, _)| (*name).to_owned()).collect();
    cases.sort();
    assert_eq!(files, cases);
}
