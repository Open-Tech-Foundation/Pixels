//! In-tree fuzzing: mutate real JPEGs and assert the decoder never panics.
//!
//! # Why this exists alongside `fuzz/`
//!
//! Same reason as the PNG harness it mirrors: `cargo fuzz` is the real fuzzer
//! but needs nightly and a time budget, so it cannot gate an ordinary
//! `cargo test`. This runs the same property — *no input panics* — on stable,
//! in a second, over a deterministic corpus.
//!
//! # Why JPEG in particular
//!
//! JPEG has more places to go wrong than PNG does. Huffman tables are
//! attacker-supplied and index a symbol list; sampling factors multiply into
//! MCU geometry; the entropy stream has no length and ends only when a marker
//! appears; and coefficients are multiplied by attacker-supplied quantization
//! steps before reaching the IDCT. Each of those is an arithmetic overflow or
//! an out-of-bounds index in a language that permits one, and a panic here.
//!
//! # What is asserted
//!
//! Only that decoding **terminates with a value**: `Ok` or a `PixelsError`,
//! never a panic and never an unbounded allocation. A mutated JPEG has no
//! correct decoding — that is what `tests/reference.rs` is for.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_jpeg::JpegDecoder;
use otf_pixels_core::{Decoder, Limits};

/// A deterministic 64-bit PRNG: `xorshift64*`.
///
/// Reproducibility matters more than statistical quality: a failing run must
/// be replayable from its seed alone, with no corpus file to lose.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next() % bound as u64) as usize
        }
    }
}

fn seed_corpus() -> Vec<(String, Vec<u8>)> {
    let dir = format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"));
    let mut corpus: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {dir}: {e}"))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? != "jpg" {
                return None;
            }
            let name = path.file_stem()?.to_str()?.to_owned();
            Some((name, std::fs::read(&path).ok()?))
        })
        .collect();
    // `read_dir` order is filesystem-dependent; sorting makes runs comparable
    // across machines, which is the whole point of a fixed seed.
    corpus.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(corpus.len() >= 10, "seed corpus looks incomplete");
    corpus
}

/// Apply one random mutation, chosen to reach different parsers.
fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) -> String {
    if bytes.is_empty() {
        bytes.push(0xFF);
        return "seeded an empty input".to_owned();
    }
    match rng.below(7) {
        // Flip a bit: corrupts a segment length, a sampling factor or a
        // Huffman code without changing the file's shape.
        0 => {
            let at = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[at] ^= 1 << bit;
            format!("flipped bit {bit} of byte {at}")
        }
        // Extreme byte values find the off-by-ones a random byte misses.
        1 => {
            let at = rng.below(bytes.len());
            let value = [0x00_u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF][rng.below(6)];
            bytes[at] = value;
            format!("set byte {at} to {value:#04x}")
        }
        // Truncate: every parser must handle running out of input mid-item,
        // and for JPEG that includes running out mid-scan.
        2 => {
            let at = rng.below(bytes.len());
            bytes.truncate(at);
            format!("truncated to {at} bytes")
        }
        // Splice out a run, desynchronising the segment stream.
        3 => {
            let start = rng.below(bytes.len());
            let len = rng.below(bytes.len() - start).min(64);
            bytes.drain(start..start + len);
            format!("removed {len} bytes at {start}")
        }
        // Insert a run, which is how a segment length comes to overstate.
        4 => {
            let at = rng.below(bytes.len());
            let len = rng.below(64) + 1;
            let filler = (rng.next() & 0xFF) as u8;
            for _ in 0..len {
                bytes.insert(at, filler);
            }
            format!("inserted {len} copies of {filler:#04x} at {at}")
        }
        // Plant a marker mid-stream, which ends a scan early or starts a
        // segment where the parser expected entropy data.
        5 if bytes.len() >= 2 => {
            let at = rng.below(bytes.len() - 1);
            let code = [0xC0_u8, 0xC2, 0xC4, 0xD8, 0xD9, 0xDA, 0xDB, 0xDD][rng.below(8)];
            bytes[at] = 0xFF;
            bytes[at + 1] = code;
            format!("planted marker FF{code:02X} at {at}")
        }
        // Overwrite a big-endian 16-bit field: how a segment claims a length
        // near 65535 and tries to provoke an allocation or a read past the end.
        _ => {
            let at = rng.below(bytes.len());
            let value = [0xFFFF_u16, 0x0000, 0x0001, 0x8000][rng.below(4)];
            for (offset, byte) in value.to_be_bytes().iter().enumerate() {
                if let Some(slot) = bytes.get_mut(at + offset) {
                    *slot = *byte;
                }
            }
            format!("wrote {value:#06x} big-endian at {at}")
        }
    }
}

/// Decode `bytes` fully, returning whatever the decoder did with it.
///
/// The limit is deliberately small: a mutated frame header can legitimately
/// claim enormous dimensions, and refusing that is the behaviour under test,
/// not a reason to allocate gigabytes in CI.
fn decode_fully(bytes: &[u8]) -> otf_pixels_core::Result<u32> {
    let limits = Limits::default().with_max_pixels(4_000_000);
    let mut decoder = JpegDecoder::new(bytes, limits)?;
    let descriptor = decoder.descriptor();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    let mut rows = 0;
    while rows < descriptor.height {
        decoder.read_row(&mut row)?;
        rows += 1;
    }
    Ok(rows)
}

/// Run `body`, converting a panic into an error naming what was being tried.
fn catch(what: &str, body: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    let previous = std::panic::take_hook();
    // Silence the default handler: an expected-to-be-caught panic printing a
    // backtrace makes a passing run look like a failing one.
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(body);
    std::panic::set_hook(previous);

    result.err().map(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".to_owned());
        format!("{what}: {message}")
    })
}

#[test]
fn mutated_jpegs_never_panic() {
    let corpus = seed_corpus();
    let mut rng = Rng(0x5EED_0000_0000_0003);
    let mut failures = Vec::new();

    // Each seed is mutated repeatedly, so later iterations explore inputs
    // several steps away from anything valid.
    for (name, original) in &corpus {
        let mut bytes = original.clone();
        for iteration in 0..60 {
            let what = mutate(&mut rng, &mut bytes);
            let context = format!("{name} iteration {iteration}: {what}");
            let input = bytes.clone();
            if let Some(failure) = catch(&context, move || {
                let _ = decode_fully(&input);
            }) {
                failures.push(failure);
            }
            // Restart from the original periodically, so the corpus does not
            // degenerate into uniformly-random bytes that stop reaching the
            // interesting parsers.
            if iteration % 10 == 9 {
                bytes.clone_from(original);
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} mutations panicked:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn arbitrary_bytes_never_panic() {
    // Random bytes behind a valid signature reach the segment parser directly,
    // which mutation of a real file rarely does.
    let mut rng = Rng(0x5EED_0000_0000_0004);
    let mut failures = Vec::new();

    for iteration in 0..2000 {
        let len = rng.below(512) + 1;
        let mut data = vec![0xFF_u8, 0xD8];
        data.extend((0..len).map(|_| (rng.next() & 0xFF) as u8));
        let context = format!("arbitrary iteration {iteration} over {len} bytes");
        let input = data.clone();
        if let Some(failure) = catch(&context, move || {
            let _ = decode_fully(&input);
        }) {
            failures.push(failure);
        }
    }

    assert!(
        failures.is_empty(),
        "{} inputs panicked:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// A truncated scan must yield the rows that did arrive, not an error and not
/// a hang.
///
/// This is the one malformed case with a *useful* answer: a partially
/// downloaded photograph decodes to a partial photograph in every viewer, and
/// the decoder feeding zero bits past the end is what produces that.
#[test]
fn a_truncated_scan_still_yields_rows() {
    let path = format!(
        "{}/tests/fixtures/gradient444.jpg",
        env!("CARGO_MANIFEST_DIR")
    );
    let whole = std::fs::read(&path).unwrap();
    let cut = whole.len() * 2 / 3;
    let rows = decode_fully(&whole[..cut]).expect("a truncated scan decodes");
    assert!(rows > 0, "no rows decoded from a truncated scan");
}

/// A header claiming more pixels than the limit allows must be refused before
/// anything is allocated for them.
#[test]
fn an_oversized_frame_is_refused_at_the_header() {
    // 65500x65500 is the largest a JPEG frame header can express, and is far
    // beyond the limit below.
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 8];
    bytes.extend_from_slice(&65_500_u16.to_be_bytes());
    bytes.extend_from_slice(&65_500_u16.to_be_bytes());
    bytes.push(3);
    for id in 1..=3_u8 {
        bytes.extend_from_slice(&[id, 0x11, 0]);
    }
    bytes.extend_from_slice(&[0xFF, 0xD9]);

    let limits = Limits::default().with_max_pixels(4_000_000);
    let error = JpegDecoder::new(&bytes[..], limits).unwrap_err();
    assert_eq!(
        error.code(),
        otf_pixels_core::ErrorCode::LimitExceeded,
        "{error}"
    );
}
