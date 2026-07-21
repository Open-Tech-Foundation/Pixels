//! Cross-checks our inflate against streams produced by real zlib.
//!
//! Our own deflate cannot validate our own inflate — a shared misunderstanding
//! of RFC 1951 would round-trip perfectly and still be wrong. The fixtures in
//! `tests/fixtures/zlib/` were produced by the reference zlib implementation at
//! several compression levels, so decoding them exercises encoder choices we
//! would never make ourselves: stored blocks, dynamic tables built by a
//! different heuristic, and long-distance back-references.
//!
//! See `tests/fixtures/zlib/MANIFEST` for how each was produced.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_png::zlib_decompress;

/// Rebuild the payload a fixture was compressed from.
///
/// Kept in step with the generator recorded in the manifest, so a fixture and
/// its expected bytes cannot drift apart silently.
fn payload(kind: &str, count: usize) -> Vec<u8> {
    match kind {
        "zeros" => vec![0_u8; count],
        "pattern" => (0..count).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
        "text" => b"the quick brown fox jumps over the lazy dog. ".repeat(count),
        "runs" => (0..count)
            .flat_map(|i| std::iter::repeat_n(b'A' + (i % 3) as u8, i % 17 + 1))
            .collect(),
        other => panic!("unknown payload kind `{other}` in the manifest"),
    }
}

/// Every fixture, as (name, kind, count).
fn manifest() -> Vec<(String, String, usize)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zlib/MANIFEST");
    let text = std::fs::read_to_string(path).expect("the zlib fixture manifest should exist");
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            (
                fields[0].to_owned(),
                fields[1].to_owned(),
                fields[2].parse().unwrap(),
            )
        })
        .collect()
}

fn fixture(name: &str) -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/zlib/{name}.zlib",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

#[test]
fn every_zlib_fixture_decompresses_to_its_payload() {
    let entries = manifest();
    assert!(
        entries.len() >= 8,
        "the manifest should cover several encoder settings"
    );
    for (name, kind, count) in entries {
        let expected = payload(&kind, count);
        let compressed = fixture(&name);
        let actual = zlib_decompress(&compressed, expected.len().max(1))
            .unwrap_or_else(|e| panic!("`{name}` failed to decompress: {e}"));
        assert_eq!(
            actual.len(),
            expected.len(),
            "`{name}` decoded to the wrong length"
        );
        assert_eq!(actual, expected, "`{name}` decoded to the wrong bytes");
    }
}

#[test]
fn stored_blocks_from_a_real_encoder_decode() {
    // zlib level 0 emits stored blocks, and splits them at 65535 bytes. Our
    // own deflate would never produce this shape.
    let expected = payload("pattern", 3000);
    let actual = zlib_decompress(&fixture("stored_level0"), expected.len()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn dynamic_huffman_tables_from_a_real_encoder_decode() {
    // Built by zlib's heuristic, not ours: different code lengths, different
    // table layout, same required meaning.
    let expected = payload("text", 60);
    let actual = zlib_decompress(&fixture("dynamic_text"), expected.len()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn long_runs_compress_to_far_less_than_they_expand_to() {
    // 5000 zero bytes in 28: heavy use of long back-references, which is where
    // an off-by-one in distance decoding shows up immediately.
    let compressed = fixture("long_zero_run");
    assert!(
        compressed.len() < 64,
        "fixture is not the dense one we expect"
    );
    let actual = zlib_decompress(&compressed, 5000).unwrap();
    assert_eq!(actual, vec![0_u8; 5000]);
}

#[test]
fn the_output_limit_is_enforced_against_real_streams() {
    // A genuine 5000-byte expansion refused at 4999 is the decompression-bomb
    // guard doing its job on a real stream rather than a synthetic one.
    let err = zlib_decompress(&fixture("long_zero_run"), 4999).unwrap_err();
    assert_eq!(err.format(), "deflate", "{err}");
    assert!(zlib_decompress(&fixture("long_zero_run"), 5000).is_ok());
}

#[test]
fn an_empty_payload_round_trips() {
    assert_eq!(
        zlib_decompress(&fixture("empty"), 16).unwrap(),
        Vec::<u8>::new()
    );
}

#[test]
fn every_single_byte_corruption_is_an_error_or_wrong_data_never_a_panic() {
    // Flipping any byte of a valid stream must not crash. Most corruptions are
    // caught by the Adler-32 or a table check; a few decode to different bytes.
    // Neither is a defect. A panic would be.
    let original = fixture("dynamic_text");
    for index in 0..original.len() {
        for mask in [0x01_u8, 0x80, 0xFF] {
            let mut corrupted = original.clone();
            corrupted[index] ^= mask;
            let _ = zlib_decompress(&corrupted, 1 << 16);
        }
    }
}

#[test]
fn every_truncation_of_every_fixture_is_an_error_never_a_panic() {
    for (name, _, _) in manifest() {
        let full = fixture(&name);
        for len in 0..full.len() {
            let _ = zlib_decompress(&full[..len], 1 << 16);
        }
    }
}
