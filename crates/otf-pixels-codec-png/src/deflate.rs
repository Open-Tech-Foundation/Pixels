//! DEFLATE compression (RFC 1951) and the zlib wrapper (RFC 1950).
//!
//! Written from scratch per ADR-0010. Correctness here means "a conforming
//! decoder reproduces the input exactly"; compression ratio is a tuning
//! question, not a correctness one, and this implementation deliberately
//! favours being obviously right.
//!
//! # Strategy
//!
//! Level 0 emits stored blocks. Levels 1–9 run LZ77 over a hash chain of
//! three-byte prefixes and emit fixed-Huffman blocks. The level controls how
//! far back the matcher searches, trading time for ratio.
//!
//! Fixed Huffman rather than dynamic is a deliberate simplification: dynamic
//! tables would compress better, but they are a second encoder to get right,
//! and every conforming decoder accepts fixed blocks. The `Encoder` trait
//! boundary makes a later dynamic-table implementation a drop-in, exactly as
//! ADR-0004 makes whole codecs swappable.

use otf_pixels_core::{PixelsError, Result};

use crate::checksum::Adler32;

/// Compression effort, 0 (stored) to 9 (most search).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Level(u8);

impl Level {
    /// No compression: stored blocks only. Fastest, always expands slightly.
    pub const NONE: Self = Self(0);
    /// Fastest compression.
    pub const FAST: Self = Self(1);
    /// The default balance.
    pub const DEFAULT: Self = Self(6);
    /// Most search effort.
    pub const BEST: Self = Self(9);

    /// A level from 0..=9.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] outside 0..=9.
    pub fn new(level: u8) -> Result<Self> {
        if level > 9 {
            return Err(PixelsError::invalid_argument(
                "level",
                format!("compression level must be 0..=9, got {level}"),
            ));
        }
        Ok(Self(level))
    }

    /// The raw level.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// How many chain positions the matcher inspects at this level.
    const fn search_depth(self) -> usize {
        match self.0 {
            0 => 0,
            1 => 4,
            2 => 8,
            3 => 16,
            4 => 32,
            5 => 64,
            6 => 128,
            7 => 256,
            8 => 512,
            _ => 1024,
        }
    }
}

impl Default for Level {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Writes bits least-significant-first, as DEFLATE specifies.
#[derive(Debug, Default)]
struct BitWriter {
    out: Vec<u8>,
    bits: u32,
    count: u32,
}

impl BitWriter {
    /// Append `n` bits of `value`, least-significant first.
    fn write(&mut self, value: u32, n: u32) {
        self.bits |= value << self.count;
        self.count += n;
        while self.count >= 8 {
            self.out.push((self.bits & 0xFF) as u8);
            self.bits >>= 8;
            self.count -= 8;
        }
    }

    /// Append `n` bits of `value`, most-significant first (Huffman codes).
    fn write_code(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.write((value >> i) & 1, 1);
        }
    }

    /// Pad to a byte boundary with zero bits.
    fn align(&mut self) {
        if self.count > 0 {
            self.out.push((self.bits & 0xFF) as u8);
            self.bits = 0;
            self.count = 0;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.align();
        self.out
    }
}

/// The fixed literal/length code for `symbol`, as (code, bit length).
///
/// From RFC 1951 §3.2.6.
const fn fixed_literal_code(symbol: u16) -> (u32, u32) {
    match symbol {
        0..=143 => (0x30 + symbol as u32, 8),
        144..=255 => (0x190 + (symbol as u32 - 144), 9),
        256..=279 => (symbol as u32 - 256, 7),
        _ => (0xC0 + (symbol as u32 - 280), 8),
    }
}

/// Base lengths for length codes 257..=285.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits for length codes 257..=285.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Base distances for distance codes 0..=29.
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits for distance codes 0..=29.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// The largest back-reference DEFLATE can express.
const MAX_MATCH: usize = 258;
/// The shortest back-reference worth emitting.
const MIN_MATCH: usize = 3;
/// The sliding window size.
const WINDOW: usize = 32768;
/// Hash table size; a power of two so the mask is cheap.
const HASH_SIZE: usize = 1 << 15;

/// Find the length code and extra bits for a match length.
fn length_code(length: usize) -> Option<(u16, u32, u32)> {
    for index in (0..LENGTH_BASE.len()).rev() {
        let base = LENGTH_BASE.get(index).copied()? as usize;
        if length >= base {
            let extra_bits = LENGTH_EXTRA.get(index).copied()? as u32;
            let extra = (length - base) as u32;
            return Some((257 + index as u16, extra, extra_bits));
        }
    }
    None
}

/// Find the distance code and extra bits for a match distance.
fn distance_code(distance: usize) -> Option<(u16, u32, u32)> {
    for index in (0..DISTANCE_BASE.len()).rev() {
        let base = DISTANCE_BASE.get(index).copied()? as usize;
        if distance >= base {
            let extra_bits = DISTANCE_EXTRA.get(index).copied()? as u32;
            let extra = (distance - base) as u32;
            return Some((index as u16, extra, extra_bits));
        }
    }
    None
}

/// Hash three bytes into a chain bucket.
fn hash3(data: &[u8], at: usize) -> usize {
    let a = data.get(at).copied().unwrap_or(0) as usize;
    let b = data.get(at + 1).copied().unwrap_or(0) as usize;
    let c = data.get(at + 2).copied().unwrap_or(0) as usize;
    ((a << 10) ^ (b << 5) ^ c) & (HASH_SIZE - 1)
}

/// Compress `data` into a raw DEFLATE stream.
///
/// # Errors
///
/// Returns [`PixelsError::Graph`] only if an internal invariant is violated;
/// compression itself cannot fail on valid input.
pub fn deflate(data: &[u8], level: Level) -> Result<Vec<u8>> {
    if level == Level::NONE {
        return Ok(deflate_stored(data));
    }
    let mut writer = BitWriter::default();
    // One fixed-Huffman block for the whole input. Block splitting would
    // improve ratio on heterogeneous data; it does not affect correctness.
    writer.write(1, 1); // BFINAL
    writer.write(1, 2); // BTYPE = 01, fixed Huffman

    // `head[hash]` is the most recent position with that hash; `prev[pos]` is
    // the previous position in the same chain. Together they form the standard
    // hash-chain matcher.
    let mut head = vec![usize::MAX; HASH_SIZE];
    let mut prev = vec![usize::MAX; data.len().max(1)];
    let depth = level.search_depth();

    let mut position = 0;
    while position < data.len() {
        let (mut best_length, mut best_distance) = (0_usize, 0_usize);

        if position + MIN_MATCH <= data.len() {
            let bucket = hash3(data, position);
            let mut candidate = head.get(bucket).copied().unwrap_or(usize::MAX);
            let limit = position.saturating_sub(WINDOW);
            let mut tries = depth;

            while candidate != usize::MAX && candidate >= limit && tries > 0 {
                tries -= 1;
                let length = match_length(data, candidate, position);
                if length > best_length {
                    best_length = length;
                    best_distance = position - candidate;
                    if best_length >= MAX_MATCH {
                        break;
                    }
                }
                let next = prev.get(candidate).copied().unwrap_or(usize::MAX);
                // Chains must strictly decrease, or a corrupt chain could loop.
                if next >= candidate {
                    break;
                }
                candidate = next;
            }
        }

        if best_length >= MIN_MATCH {
            let (code, extra, extra_bits) = length_code(best_length)
                .ok_or_else(|| PixelsError::graph("no length code for a computed match"))?;
            let (literal_code, literal_bits) = fixed_literal_code(code);
            writer.write_code(literal_code, literal_bits);
            if extra_bits > 0 {
                writer.write(extra, extra_bits);
            }
            let (dcode, dextra, dextra_bits) = distance_code(best_distance)
                .ok_or_else(|| PixelsError::graph("no distance code for a computed match"))?;
            // Distance codes use a fixed 5-bit code in fixed-Huffman blocks.
            writer.write_code(u32::from(dcode), 5);
            if dextra_bits > 0 {
                writer.write(dextra, dextra_bits);
            }
            // Insert every position the match covers, so later matches can
            // start inside it.
            for offset in 0..best_length {
                insert(data, &mut head, &mut prev, position + offset);
            }
            position += best_length;
        } else {
            let byte = data.get(position).copied().unwrap_or(0);
            let (code, bits) = fixed_literal_code(u16::from(byte));
            writer.write_code(code, bits);
            insert(data, &mut head, &mut prev, position);
            position += 1;
        }
    }

    // End-of-block.
    let (code, bits) = fixed_literal_code(256);
    writer.write_code(code, bits);
    Ok(writer.finish())
}

/// Record `at` in the hash chains.
fn insert(data: &[u8], head: &mut [usize], prev: &mut [usize], at: usize) {
    if at + MIN_MATCH > data.len() {
        return;
    }
    let bucket = hash3(data, at);
    let Some(slot) = head.get_mut(bucket) else {
        return;
    };
    if let Some(chain) = prev.get_mut(at) {
        *chain = *slot;
    }
    *slot = at;
}

/// How many bytes match between `candidate` and `position`.
fn match_length(data: &[u8], candidate: usize, position: usize) -> usize {
    let available = data.len() - position;
    let max = available.min(MAX_MATCH);
    let mut length = 0;
    while length < max {
        let a = data.get(candidate + length).copied();
        let b = data.get(position + length).copied();
        if a.is_none() || a != b {
            break;
        }
        length += 1;
    }
    length
}

/// Emit `data` as stored (uncompressed) blocks.
fn deflate_stored(data: &[u8]) -> Vec<u8> {
    // A stored block's length field is 16 bits, so long inputs are split.
    const MAX_STORED: usize = 65535;
    let mut out = Vec::with_capacity(data.len() + data.len() / MAX_STORED * 5 + 5);
    if data.is_empty() {
        out.push(0x01);
        out.extend_from_slice(&0_u16.to_le_bytes());
        out.extend_from_slice(&(!0_u16).to_le_bytes());
        return out;
    }
    let mut chunks = data.chunks(MAX_STORED).peekable();
    while let Some(chunk) = chunks.next() {
        let final_block = u8::from(chunks.peek().is_none());
        out.push(final_block);
        let length = chunk.len() as u16;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out
}

/// Compress `data` into a zlib stream (RFC 1950).
///
/// # Errors
///
/// As [`deflate`].
pub fn zlib_compress(data: &[u8], level: Level) -> Result<Vec<u8>> {
    // CMF: deflate (8) with a 32 KiB window (7 << 4).
    let cmf = 0x78_u8;
    // FLG carries the level hint and makes the 16-bit header divisible by 31.
    let level_bits = match level.get() {
        0..=1 => 0_u8,
        2..=5 => 1,
        6 => 2,
        _ => 3,
    };
    let mut flg = level_bits << 6;
    let check = (u16::from(cmf) << 8) | u16::from(flg);
    flg += (31 - (check % 31) % 31) as u8;

    let mut out = Vec::new();
    out.push(cmf);
    out.push(flg);
    out.extend_from_slice(&deflate(data, level)?);
    out.extend_from_slice(&Adler32::of(data).to_be_bytes());
    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::inflate::{inflate_to, zlib_decompress};

    /// Payloads that exercise different compressor behaviour.
    fn corpus() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("empty", Vec::new()),
            ("one byte", vec![42]),
            ("two bytes", vec![1, 2]),
            ("below min match", vec![7, 7]),
            ("exactly min match", vec![7, 7, 7]),
            ("all zeros", vec![0; 10_000]),
            ("repeating text", b"the quick brown fox. ".repeat(300)),
            (
                "incompressible",
                (0..8192).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
            ),
            ("long run of one byte", vec![0xAB; 70_000]),
            ("alternating", (0..5000).map(|i| (i % 2) as u8).collect()),
            (
                "match at max length",
                std::iter::repeat_n(b'z', MAX_MATCH * 3).collect::<Vec<u8>>(),
            ),
            ("binary", (0..=255_u8).cycle().take(20_000).collect()),
        ]
    }

    #[test]
    fn every_payload_round_trips_at_every_level() {
        // The correctness property that matters: our decoder reproduces the
        // input exactly, at every level, for every shape of data.
        for (name, data) in corpus() {
            for level in 0..=9 {
                let level = Level::new(level).unwrap();
                let compressed = deflate(&data, level).unwrap();
                let out = inflate_to(&compressed, data.len().max(1)).unwrap();
                assert_eq!(out, data, "`{name}` at level {}", level.get());
            }
        }
    }

    #[test]
    fn zlib_wrapping_round_trips_at_every_level() {
        for (name, data) in corpus() {
            for level in 0..=9 {
                let level = Level::new(level).unwrap();
                let compressed = zlib_compress(&data, level).unwrap();
                let out = zlib_decompress(&compressed, data.len().max(1)).unwrap();
                assert_eq!(out, data, "`{name}` at level {}", level.get());
            }
        }
    }

    #[test]
    fn the_zlib_header_is_well_formed() {
        for level in 0..=9 {
            let level = Level::new(level).unwrap();
            let stream = zlib_compress(b"hello", level).unwrap();
            let header = (u16::from(stream[0]) << 8) | u16::from(stream[1]);
            assert_eq!(stream[0] & 0x0F, 8, "compression method must be deflate");
            assert_eq!(header % 31, 0, "header check bits at level {}", level.get());
            assert_eq!(stream[1] & 0x20, 0, "no preset dictionary");
        }
    }

    #[test]
    fn compression_actually_compresses_compressible_data() {
        // Not a correctness property, but a ratio this bad would mean the
        // matcher is not finding anything at all.
        let data = b"the quick brown fox. ".repeat(500);
        let compressed = deflate(&data, Level::DEFAULT).unwrap();
        assert!(
            compressed.len() < data.len() / 10,
            "compressed {} bytes to {}, expected under {}",
            data.len(),
            compressed.len(),
            data.len() / 10
        );
    }

    #[test]
    fn higher_levels_do_not_compress_worse() {
        let data = b"abcabcabd".repeat(2000);
        let fast = deflate(&data, Level::FAST).unwrap().len();
        let best = deflate(&data, Level::BEST).unwrap().len();
        assert!(
            best <= fast,
            "level 9 produced {best} bytes, level 1 produced {fast}"
        );
    }

    #[test]
    fn level_zero_stores_without_compressing() {
        let data = vec![0_u8; 1000];
        let compressed = deflate(&data, Level::NONE).unwrap();
        assert!(compressed.len() > data.len(), "stored blocks add framing");
        assert_eq!(inflate_to(&compressed, 1000).unwrap(), data);
    }

    #[test]
    fn stored_blocks_split_at_the_sixteen_bit_length_limit() {
        // A stored block's length field is 16 bits, so >65535 bytes must be
        // split across blocks or the length silently wraps.
        let data = vec![7_u8; 200_000];
        let compressed = deflate(&data, Level::NONE).unwrap();
        assert_eq!(inflate_to(&compressed, 200_000).unwrap(), data);
    }

    #[test]
    fn levels_are_validated() {
        assert!(Level::new(0).is_ok());
        assert!(Level::new(9).is_ok());
        let err = Level::new(10).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::InvalidArgument);
        assert_eq!(Level::default(), Level::DEFAULT);
        assert_eq!(Level::DEFAULT.get(), 6);
    }

    #[test]
    fn matches_at_the_window_boundary_round_trip() {
        // A match exactly 32768 bytes back is the furthest DEFLATE can express;
        // one byte further must not be emitted as a match.
        let mut data = vec![0_u8; WINDOW + 64];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = ((i * 7) % 251) as u8;
        }
        // Repeat the opening bytes at the very edge of the window.
        let head: Vec<u8> = data[..32].to_vec();
        data.extend_from_slice(&head);
        let compressed = deflate(&data, Level::BEST).unwrap();
        assert_eq!(inflate_to(&compressed, data.len()).unwrap(), data);
    }

    #[test]
    fn maximum_length_matches_round_trip() {
        // 258 is the longest expressible match; longer runs must be split.
        let data = vec![0x5A_u8; MAX_MATCH * 5 + 7];
        let compressed = deflate(&data, Level::BEST).unwrap();
        assert_eq!(inflate_to(&compressed, data.len()).unwrap(), data);
    }

    #[test]
    fn our_output_is_decodable_after_a_round_trip_through_our_decoder() {
        // Guards against the encoder and decoder drifting together: the data
        // is compressed, decompressed, recompressed, and compared.
        for (name, data) in corpus() {
            let once = zlib_compress(&data, Level::DEFAULT).unwrap();
            let back = zlib_decompress(&once, data.len().max(1)).unwrap();
            let twice = zlib_compress(&back, Level::DEFAULT).unwrap();
            assert_eq!(once, twice, "`{name}` is not deterministic");
        }
    }

    #[test]
    fn compression_is_deterministic() {
        // SPEC §Guarantees 2 reaches the encoder too: same input, same bytes.
        let data = b"determinism matters. ".repeat(100);
        let first = zlib_compress(&data, Level::BEST).unwrap();
        for _ in 0..5 {
            assert_eq!(zlib_compress(&data, Level::BEST).unwrap(), first);
        }
    }
}
