//! Emits our own zlib streams so an external tool can verify them.
//!
//! Our decoder cannot validate our encoder: a shared misreading of RFC 1951
//! would round-trip perfectly and still be wrong. `tests/zlib_reference.rs`
//! checks one direction by decoding streams real zlib produced; this checks
//! the other, by producing streams for real zlib to decode.
//!
//! It is inert unless `OTF_EMIT_DIR` is set, so an ordinary `cargo test` is
//! unaffected. `scripts/check-deflate-interop.sh` drives it and runs the
//! comparison.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_png::{Level, zlib_compress};

#[test]
fn emit_streams_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("text", b"the quick brown fox. ".repeat(300)),
        ("zeros", vec![0; 10_000]),
        (
            "incompressible",
            (0..8192).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
        ),
        ("longrun", vec![0xAB; 70_000]),
        ("binary", (0..=255_u8).cycle().take(20_000).collect()),
        ("min_match", vec![7, 7, 7]),
        ("max_match", std::iter::repeat_n(b'z', 258 * 3).collect()),
    ];
    for (name, data) in cases {
        for level in [0_u8, 1, 6, 9] {
            let compressed = zlib_compress(&data, Level::new(level).unwrap()).unwrap();
            std::fs::write(format!("{dir}/{name}_{level}.zlib"), &compressed).unwrap();
            std::fs::write(format!("{dir}/{name}_{level}.raw"), &data).unwrap();
        }
    }
}
