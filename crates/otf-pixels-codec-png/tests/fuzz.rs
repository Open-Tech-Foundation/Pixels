//! In-tree fuzzing: mutate real PNGs and assert the decoder never panics.
//!
//! # Why this exists alongside `fuzz/`
//!
//! `cargo fuzz` is the real fuzzer, but it needs a nightly toolchain and a
//! time budget, so it cannot gate an ordinary `cargo test`. This harness runs
//! the same property — *no input panics* — on stable, in a second, over a
//! deterministic corpus derived from PngSuite. It catches the shallow
//! regressions immediately; `cargo fuzz` finds the deep ones overnight.
//!
//! # What is asserted
//!
//! Only that decoding **terminates with a value**: `Ok` or a [`PixelsError`],
//! never a panic, never an unbounded allocation. A mutated PNG has no correct
//! decoding to compare against — that is what `tests/pngsuite.rs` is for — so
//! asserting anything about the pixels here would only encode whatever the
//! decoder happens to do today.
//!
//! Panics are caught rather than allowed to abort the run, so a failure names
//! the exact mutation instead of just the seed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_png::{PngDecoder, inflate_to, zlib_decompress};
use otf_pixels_core::{Decoder, Limits};

/// A deterministic 64-bit PRNG: `xorshift64*`.
///
/// Reproducibility matters more than statistical quality here — a failing run
/// must be replayable from its seed alone, with no corpus file to lose.
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
    let dir = format!("{}/tests/fixtures/pngsuite", env!("CARGO_MANIFEST_DIR"));
    let mut corpus: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {dir}: {e}"))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension()? != "png" {
                return None;
            }
            let name = path.file_stem()?.to_str()?.to_owned();
            Some((name, std::fs::read(&path).ok()?))
        })
        .collect();
    // `read_dir` order is filesystem-dependent; sorting makes runs comparable
    // across machines, which is the whole point of a fixed seed.
    corpus.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(corpus.len() > 50, "seed corpus looks incomplete");
    corpus
}

/// Apply one random mutation, chosen to reach different parsers.
fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>) -> String {
    if bytes.is_empty() {
        bytes.push(0);
        return "seeded an empty input".to_owned();
    }
    match rng.below(6) {
        // Flip a bit: the cheapest way to corrupt a length, a CRC or a
        // Huffman code without changing the file's shape.
        0 => {
            let at = rng.below(bytes.len());
            let bit = rng.below(8);
            bytes[at] ^= 1 << bit;
            format!("flipped bit {bit} of byte {at}")
        }
        // Replace a byte with an extreme value, which finds the off-by-ones
        // that a random byte usually misses.
        1 => {
            let at = rng.below(bytes.len());
            let value = [0x00_u8, 0x01, 0x7F, 0x80, 0xFE, 0xFF][rng.below(6)];
            bytes[at] = value;
            format!("set byte {at} to {value:#04x}")
        }
        // Truncate: every parser must handle running out of input mid-item.
        2 => {
            let at = rng.below(bytes.len());
            bytes.truncate(at);
            format!("truncated to {at} bytes")
        }
        // Splice out a run, which desynchronises the chunk stream.
        3 => {
            let start = rng.below(bytes.len());
            let len = rng.below(bytes.len() - start).min(64);
            bytes.drain(start..start + len);
            format!("removed {len} bytes at {start}")
        }
        // Insert a run, which is how a length field comes to overstate.
        4 => {
            let at = rng.below(bytes.len());
            let len = rng.below(64) + 1;
            let filler = (rng.next() & 0xFF) as u8;
            for _ in 0..len {
                bytes.insert(at, filler);
            }
            format!("inserted {len} copies of {filler:#04x} at {at}")
        }
        // Overwrite a 4-byte big-endian field, which is how a chunk claims a
        // length near `usize::MAX` and tries to provoke an allocation.
        _ => {
            let at = rng.below(bytes.len());
            let value = [0xFFFF_FFFF_u32, 0x7FFF_FFFF, 0x8000_0000, 0][rng.below(4)];
            for (offset, byte) in value.to_be_bytes().iter().enumerate() {
                if let Some(slot) = bytes.get_mut(at + offset) {
                    *slot = *byte;
                }
            }
            format!("wrote {value:#010x} big-endian at {at}")
        }
    }
}

/// Decode `bytes` fully, returning whatever the decoder did with it.
///
/// The limit is deliberately small: a mutated header can legitimately claim
/// enormous dimensions, and refusing that is the behaviour under test, not a
/// reason to allocate gigabytes in CI.
fn decode_fully(bytes: &[u8]) -> otf_pixels_core::Result<usize> {
    let limits = Limits::default().with_max_pixels(4_000_000);
    let mut decoder = PngDecoder::new(bytes, limits)?;
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    let mut rows = 0;
    // Re-read the descriptor each iteration: tRNS can change the row length
    // once the stream is parsed, and a fuzzer will find that seam.
    while rows < decoder.descriptor().height {
        let wanted = decoder.descriptor().row_bytes();
        if row.len() != wanted {
            row = vec![0_u8; wanted];
        }
        decoder.read_row(&mut row)?;
        rows += 1;
    }
    Ok(rows as usize)
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
fn mutated_pngs_never_panic() {
    let corpus = seed_corpus();
    let mut rng = Rng(0x5EED_0000_0000_0001);
    let mut failures = Vec::new();

    // Each seed is mutated repeatedly, so later iterations explore inputs that
    // are already several steps away from anything valid.
    for (name, original) in &corpus {
        let mut bytes = original.clone();
        for iteration in 0..40 {
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
            if iteration % 8 == 7 {
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
fn arbitrary_bytes_never_panic_in_inflate() {
    // The decompressor is reached with attacker-controlled bytes on every
    // image, and it is the component where a bad back-reference would be a
    // memory-safety bug in a language that allowed one.
    let mut rng = Rng(0x5EED_0000_0000_0002);
    let mut failures = Vec::new();

    for iteration in 0..2000 {
        let len = rng.below(512) + 1;
        let data: Vec<u8> = (0..len).map(|_| (rng.next() & 0xFF) as u8).collect();
        let context = format!("inflate iteration {iteration} over {len} bytes");
        let input = data.clone();
        if let Some(failure) = catch(&context, move || {
            let _ = inflate_to(&input, 1 << 20);
            let _ = zlib_decompress(&input, 1 << 20);
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

#[test]
fn a_decompression_bomb_is_bounded_not_allocated() {
    // A large run of zeroes compresses to a tiny fraction of its size. The
    // decoder must refuse it by the *output* limit, not discover the problem
    // after allocating — the property the bound in `inflate_to` exists for.
    const BOMB: usize = 32 * 1024 * 1024;
    let bomb = {
        use otf_pixels_codec_png::{Level, zlib_compress};
        zlib_compress(&vec![0_u8; BOMB], Level::BEST).unwrap()
    };
    // Our deflate emits fixed-Huffman blocks only (a documented simplification
    // in `deflate`), so it reaches roughly 150:1 here rather than zlib's
    // ~1000:1. Any ratio this large makes the point; the exact figure is not
    // the property under test.
    assert!(
        bomb.len() * 100 < BOMB,
        "the bomb should be far smaller than its expansion: {} bytes",
        bomb.len()
    );

    let error = zlib_decompress(&bomb, 1 << 20).unwrap_err();
    assert_eq!(error.format(), "deflate", "{error}");
}
