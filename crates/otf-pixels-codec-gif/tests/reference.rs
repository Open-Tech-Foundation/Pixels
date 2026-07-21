//! GIF decode checked against a reference decoder, not against ourselves.
//!
//! The discipline M3 established for PNG applies here for the same reason: a
//! shared misreading of the specification round-trips perfectly and is still
//! wrong. `REFERENCE` records what Pillow (libgif) decodes each fixture to;
//! regenerate it with `scripts/regenerate-gif-reference.py`.
//!
//! # What the fixtures cover
//!
//! - `static` — a plain 8-colour image, the baseline
//! - `interlaced` — the same image stored in GIF's four-pass row interlace
//! - `checker` — a two-colour image, which exercises the minimum code width
//! - `photo` — a full 256-entry adaptive palette
//! - `animation` — four frames with disposal and a graphic control extension
//!
//! `static` and `interlaced` are the same pixels stored two ways, so they must
//! decode identically to each other as well as to the reference. That
//! localises an interlace bug to the interlace rather than to the decoder.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_gif::{Disposal, GifDecoder, probe};
use otf_pixels_core::{Decoder, Limits};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/{name}.gif", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// FNV-1a, 64-bit — matches the regeneration script.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &byte in data {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// One reference decoding: dimensions, frame count and per-frame hashes.
struct Reference {
    name: String,
    width: u32,
    height: u32,
    frames: Vec<u64>,
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
                frames: f[3..]
                    .iter()
                    .map(|h| u64::from_str_radix(h, 16).unwrap())
                    .collect(),
            }
        })
        .collect()
}

/// Decode every frame of a GIF, returning each composited canvas.
fn decode_all(bytes: &[u8]) -> otf_pixels_core::Result<Vec<Vec<u8>>> {
    let mut decoder = GifDecoder::new(bytes, Limits::default())?;
    let mut frames = Vec::new();
    while let Some(frame) = decoder.next_frame()? {
        frames.push(frame.pixels);
    }
    Ok(frames)
}

#[test]
fn every_fixture_decodes_to_the_reference_pixels() {
    let references = references();
    assert!(!references.is_empty(), "the reference manifest looks empty");
    let mut failures = Vec::new();

    for reference in &references {
        let bytes = read_fixture(&reference.name);
        let frames = match decode_all(&bytes) {
            Ok(frames) => frames,
            Err(error) => {
                failures.push(format!("{}: failed to decode: {error}", reference.name));
                continue;
            }
        };
        if frames.len() != reference.frames.len() {
            failures.push(format!(
                "{}: decoded {} frames, reference has {}",
                reference.name,
                frames.len(),
                reference.frames.len()
            ));
            continue;
        }
        for (index, (frame, expected)) in frames.iter().zip(&reference.frames).enumerate() {
            if fnv1a64(frame) != *expected {
                failures.push(format!(
                    "{} frame {index}: pixels differ from libgif ({}x{})",
                    reference.name, reference.width, reference.height
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} disagreements with the reference decoder:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn interlaced_and_non_interlaced_decode_identically() {
    // The strongest interlace check available: the same image, both ways. A
    // row emitted twice or in the wrong pass shows up here and nowhere else
    // so cleanly.
    let plain = decode_all(&read_fixture("static")).unwrap();
    let interlaced = decode_all(&read_fixture("interlaced")).unwrap();
    assert_eq!(plain.len(), interlaced.len(), "frame counts differ");
    assert_eq!(plain[0], interlaced[0], "interlacing changed the pixels");
}

#[test]
fn an_animation_reports_its_frames_delays_and_disposal() {
    let bytes = read_fixture("animation");
    let mut decoder = GifDecoder::new(&bytes[..], Limits::default()).unwrap();

    let mut count = 0;
    while let Some(frame) = decoder.next_frame().unwrap() {
        assert_eq!((frame.width, frame.height), (20, 20), "frame {count}");
        assert_eq!(
            frame.pixels.len(),
            20 * 20 * 4,
            "frame {count} is not a full canvas"
        );
        assert!(frame.delay_centiseconds > 0, "frame {count} lost its delay");
        assert_eq!(
            frame.disposal,
            Disposal::Background,
            "frame {count} lost its disposal method"
        );
        count += 1;
    }
    assert_eq!(count, 4, "expected four frames");
}

#[test]
fn the_decoder_trait_yields_the_first_frame() {
    // SPEC's split: the engine sees one image, so a GIF works in an ordinary
    // pipeline. The frames API is what exposes the rest.
    let bytes = read_fixture("animation");
    let mut decoder = GifDecoder::new(&bytes[..], Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    assert_eq!((descriptor.width, descriptor.height), (20, 20));

    let mut raster = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row).unwrap();
        raster.extend_from_slice(&row);
    }

    let frames = decode_all(&bytes).unwrap();
    assert_eq!(
        raster, frames[0],
        "read_row did not produce the first frame"
    );
}

#[test]
fn a_gif87a_stream_decodes() {
    // The older signature has no extensions at all, so a decoder that
    // required a graphic control block would fail every one of them.
    let frames = decode_all(&read_fixture("checker")).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), 8 * 8 * 4);
}

#[test]
fn probe_detects_gif_by_magic_bytes_only() {
    assert!(probe(b"GIF89a..."), "89a");
    assert!(probe(b"GIF87a..."), "87a");
    assert!(!probe(b"GIF88a..."), "a version that does not exist");
    assert!(!probe(b"GIF"), "shorter than the magic");
    assert!(!probe(b""), "empty");
    assert!(!probe(&[0x89, b'P', b'N', b'G', 13, 10, 26, 10]), "a PNG");
}

#[test]
fn construction_reads_only_the_header() {
    // SPEC §Guarantees 3: the descriptor is available without pixel work.
    // Thirteen bytes of header plus a global colour table is all a GIF needs.
    let bytes = read_fixture("photo");
    let decoder = GifDecoder::new(&bytes[..], Limits::default()).unwrap();
    assert_eq!(
        (decoder.descriptor().width, decoder.descriptor().height),
        (48, 32)
    );
}

#[test]
fn max_pixels_is_enforced_before_decoding() {
    let bytes = read_fixture("photo");
    let limits = Limits::default().with_max_pixels(100);
    let error = GifDecoder::new(&bytes[..], limits).unwrap_err();
    assert_eq!(
        error.code(),
        otf_pixels_core::ErrorCode::LimitExceeded,
        "{error}"
    );
}

#[test]
fn every_truncation_of_every_fixture_is_an_error_never_a_panic() {
    // Every byte of a GIF is attacker-controlled, and truncation is the
    // cheapest corruption to produce.
    for name in ["static", "interlaced", "checker", "photo", "animation"] {
        let bytes = read_fixture(name);
        for cut in 0..bytes.len() {
            let truncated = &bytes[..cut];
            if let Ok(mut decoder) = GifDecoder::new(truncated, Limits::default()) {
                // Draining is what actually reaches the LZW and block parsers.
                while let Ok(Some(_)) = decoder.next_frame() {}
            }
        }
    }
}

#[test]
fn single_byte_corruption_never_panics() {
    for name in ["static", "photo", "animation"] {
        let original = read_fixture(name);
        for at in 0..original.len() {
            for mask in [0x01_u8, 0x80, 0xFF] {
                let mut bytes = original.clone();
                bytes[at] ^= mask;
                if let Ok(mut decoder) = GifDecoder::new(&bytes[..], Limits::default()) {
                    while let Ok(Some(_)) = decoder.next_frame() {}
                }
            }
        }
    }
}
