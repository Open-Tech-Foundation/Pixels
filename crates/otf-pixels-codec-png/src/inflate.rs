//! DEFLATE decompression (RFC 1951) and the zlib wrapper (RFC 1950).
//!
//! Written from scratch per ADR-0010. Every path returns an error rather than
//! panicking: this code parses attacker-controlled bytes, and
//! `unsafe_code = "forbid"` plus explicit bounds checks mean the classic
//! decompressor failure — an out-of-bounds write through a back-reference —
//! is unrepresentable rather than merely avoided.
//!
//! # Shape
//!
//! A DEFLATE stream is a sequence of blocks, each either stored (raw),
//! fixed-Huffman (a table the spec hardcodes) or dynamic-Huffman (a table the
//! block carries). Literal/length and distance codes decode into either a
//! literal byte or a back-reference into the last 32 KiB of output.
//!
//! # Bounded output
//!
//! [`inflate_to`] takes a byte limit. A short input can expand enormously — a
//! decompression bomb — so the caller states how much output it is prepared to
//! accept, which for PNG is exactly the filtered raster size derived from the
//! header (SPEC §Safety).

use otf_pixels_core::{PixelsError, Result};

use crate::checksum::Adler32;

/// Reads bits least-significant-first, as DEFLATE specifies.
#[derive(Debug)]
struct BitReader<'a> {
    data: &'a [u8],
    /// Index of the next byte to load.
    position: usize,
    /// Bits not yet consumed, right-aligned.
    bits: u64,
    /// How many bits in `bits` are valid.
    count: u32,
}

impl<'a> BitReader<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            position: 0,
            bits: 0,
            count: 0,
        }
    }

    /// Ensure at least `want` bits are buffered, if the input has them.
    fn fill(&mut self, want: u32) {
        while self.count < want {
            let Some(&byte) = self.data.get(self.position) else {
                break;
            };
            self.bits |= u64::from(byte) << self.count;
            self.position += 1;
            self.count += 8;
        }
    }

    /// Consume `n` bits (`n <= 32`).
    fn take(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n);
        if self.count < n {
            return Err(truncated());
        }
        // `n <= 32`, so the mask fits and the cast cannot lose bits.
        let mask = (1_u64 << n) - 1;
        let value = (self.bits & mask) as u32;
        self.bits >>= n;
        self.count -= n;
        Ok(value)
    }

    /// Look at up to `n` buffered bits without consuming them.
    fn peek(&mut self, n: u32) -> u32 {
        self.fill(n);
        let mask = (1_u64 << n) - 1;
        (self.bits & mask) as u32
    }

    /// Drop `n` already-peeked bits.
    fn skip(&mut self, n: u32) -> Result<()> {
        if self.count < n {
            return Err(truncated());
        }
        self.bits >>= n;
        self.count -= n;
        Ok(())
    }

    /// Discard buffered bits back to a byte boundary.
    fn align(&mut self) {
        let extra = self.count % 8;
        self.bits >>= extra;
        self.count -= extra;
    }

    /// Take `n` whole bytes, which must be byte-aligned already.
    fn take_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(n.min(1 << 16));
        for _ in 0..n {
            // Buffered bits are consumed first, then raw input.
            if self.count >= 8 {
                out.push((self.bits & 0xFF) as u8);
                self.bits >>= 8;
                self.count -= 8;
            } else if let Some(&byte) = self.data.get(self.position) {
                self.position += 1;
                out.push(byte);
            } else {
                return Err(truncated());
            }
        }
        Ok(out)
    }
}

/// The error for a stream that ended mid-symbol.
fn truncated() -> PixelsError {
    PixelsError::malformed("deflate", "stream ended in the middle of a symbol")
}

/// The maximum code length DEFLATE permits.
const MAX_BITS: usize = 15;

/// A canonical Huffman decoding table.
///
/// Decoding walks the code length by length rather than using a large lookup
/// table: slower per symbol, but obviously correct and with no table-size
/// arithmetic to get wrong on hostile input.
#[derive(Debug, Clone)]
struct Huffman {
    /// `counts[n]` is how many codes have length `n`.
    counts: [u16; MAX_BITS + 1],
    /// Symbols ordered by code length, then by symbol value.
    symbols: Vec<u16>,
}

impl Huffman {
    /// Build a table from per-symbol code lengths (zero meaning "unused").
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0_u16; MAX_BITS + 1];
        for &length in lengths {
            let length = length as usize;
            if length > MAX_BITS {
                return Err(PixelsError::malformed(
                    "deflate",
                    format!("code length {length} exceeds the {MAX_BITS}-bit maximum"),
                ));
            }
            if let Some(slot) = counts.get_mut(length) {
                *slot += 1;
            }
        }
        // Length zero means "no code", so it never participates.
        if let Some(slot) = counts.get_mut(0) {
            *slot = 0;
        }

        // Reject over-subscribed tables: a set of code lengths that cannot form
        // a prefix code would otherwise decode garbage deterministically.
        let mut left = 1_i32;
        for length in 1..=MAX_BITS {
            left <<= 1;
            left -= i32::from(counts.get(length).copied().unwrap_or(0));
            if left < 0 {
                return Err(PixelsError::malformed(
                    "deflate",
                    "Huffman code lengths are over-subscribed",
                ));
            }
        }

        // Offsets of each length's run within `symbols`.
        let mut offsets = [0_usize; MAX_BITS + 2];
        for length in 1..=MAX_BITS {
            let next = offsets.get(length).copied().unwrap_or(0)
                + counts.get(length).copied().unwrap_or(0) as usize;
            if let Some(slot) = offsets.get_mut(length + 1) {
                *slot = next;
            }
        }
        let mut symbols = vec![0_u16; lengths.len()];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let Some(offset) = offsets.get_mut(length as usize) else {
                continue;
            };
            if let Some(slot) = symbols.get_mut(*offset) {
                *slot = symbol as u16;
            }
            *offset += 1;
        }
        Ok(Self { counts, symbols })
    }

    /// Decode one symbol from `reader`.
    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0_i32;
        let mut first = 0_i32;
        let mut index = 0_i32;
        // Peek the maximum, then consume exactly the bits actually used.
        let peeked = reader.peek(MAX_BITS as u32);
        for length in 1..=MAX_BITS {
            // DEFLATE codes are stored most-significant-bit first within the
            // LSB-first bit stream, so the code is rebuilt bit by bit.
            code |= ((peeked >> (length - 1)) & 1) as i32;
            let count = i32::from(self.counts.get(length).copied().unwrap_or(0));
            if code - first < count {
                reader.skip(length as u32)?;
                let position = (index + (code - first)) as usize;
                return self
                    .symbols
                    .get(position)
                    .copied()
                    .ok_or_else(|| PixelsError::malformed("deflate", "invalid Huffman symbol"));
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(PixelsError::malformed(
            "deflate",
            "no Huffman code matched within 15 bits",
        ))
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
/// The order code-length codes appear in a dynamic block header.
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// The fixed literal/length table from RFC 1951 §3.2.6.
fn fixed_literal_table() -> Result<Huffman> {
    let mut lengths = [0_u8; 288];
    for (symbol, slot) in lengths.iter_mut().enumerate() {
        *slot = match symbol {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    Huffman::new(&lengths)
}

/// The fixed distance table: 30 codes of 5 bits each.
fn fixed_distance_table() -> Result<Huffman> {
    Huffman::new(&[5_u8; 30])
}

/// Decompress a raw DEFLATE stream, refusing to exceed `limit` output bytes.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for any invalid or truncated stream, and
/// for output exceeding `limit` — a decompression bomb is malformed input, not
/// a resource the caller must survive.
pub fn inflate_to(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(data);
    let mut out: Vec<u8> = Vec::new();

    loop {
        let final_block = reader.take(1)? == 1;
        let kind = reader.take(2)?;
        match kind {
            0 => inflate_stored(&mut reader, &mut out, limit)?,
            1 => {
                let literals = fixed_literal_table()?;
                let distances = fixed_distance_table()?;
                inflate_block(&mut reader, &literals, &distances, &mut out, limit)?;
            }
            2 => {
                let (literals, distances) = read_dynamic_tables(&mut reader)?;
                inflate_block(&mut reader, &literals, &distances, &mut out, limit)?;
            }
            _ => {
                return Err(PixelsError::malformed("deflate", "reserved block type 3"));
            }
        }
        if final_block {
            break;
        }
    }
    Ok(out)
}

/// Copy a stored (uncompressed) block.
fn inflate_stored(reader: &mut BitReader<'_>, out: &mut Vec<u8>, limit: usize) -> Result<()> {
    reader.align();
    let length = reader.take(16)? as usize;
    let complement = reader.take(16)? as usize;
    if length ^ 0xFFFF != complement {
        return Err(PixelsError::malformed(
            "deflate",
            "stored block length does not match its complement",
        ));
    }
    check_limit(out.len(), length, limit)?;
    let bytes = reader.take_bytes(length)?;
    out.extend_from_slice(&bytes);
    Ok(())
}

/// Read the code-length-coded tables of a dynamic block.
fn read_dynamic_tables(reader: &mut BitReader<'_>) -> Result<(Huffman, Huffman)> {
    let literal_count = reader.take(5)? as usize + 257;
    let distance_count = reader.take(5)? as usize + 1;
    let code_length_count = reader.take(4)? as usize + 4;
    if literal_count > 288 || distance_count > 30 {
        return Err(PixelsError::malformed(
            "deflate",
            "dynamic block declares too many codes",
        ));
    }

    let mut code_lengths = [0_u8; 19];
    for index in 0..code_length_count {
        let bits = reader.take(3)? as u8;
        let Some(&position) = CODE_LENGTH_ORDER.get(index) else {
            break;
        };
        if let Some(slot) = code_lengths.get_mut(position) {
            *slot = bits;
        }
    }
    let code_length_table = Huffman::new(&code_lengths)?;

    // The two tables are coded as one run, so repeats may straddle the seam.
    let total = literal_count + distance_count;
    let mut lengths = vec![0_u8; total];
    let mut index = 0;
    while index < total {
        let symbol = code_length_table.decode(reader)?;
        match symbol {
            0..=15 => {
                if let Some(slot) = lengths.get_mut(index) {
                    *slot = symbol as u8;
                }
                index += 1;
            }
            16 => {
                // Repeat the previous length 3..=6 times.
                let previous = index
                    .checked_sub(1)
                    .and_then(|i| lengths.get(i).copied())
                    .ok_or_else(|| {
                        PixelsError::malformed("deflate", "repeat code with no previous length")
                    })?;
                let repeat = reader.take(2)? as usize + 3;
                fill(&mut lengths, &mut index, previous, repeat, total)?;
            }
            17 => {
                let repeat = reader.take(3)? as usize + 3;
                fill(&mut lengths, &mut index, 0, repeat, total)?;
            }
            18 => {
                let repeat = reader.take(7)? as usize + 11;
                fill(&mut lengths, &mut index, 0, repeat, total)?;
            }
            _ => {
                return Err(PixelsError::malformed(
                    "deflate",
                    "invalid code length symbol",
                ));
            }
        }
    }

    let (literal_lengths, distance_lengths) = lengths.split_at(literal_count);
    let literals = Huffman::new(literal_lengths)?;
    let distances = Huffman::new(distance_lengths)?;
    Ok((literals, distances))
}

/// Write `value` into `lengths` `repeat` times, refusing to overrun.
fn fill(
    lengths: &mut [u8],
    index: &mut usize,
    value: u8,
    repeat: usize,
    total: usize,
) -> Result<()> {
    if *index + repeat > total {
        return Err(PixelsError::malformed(
            "deflate",
            "code length repeat runs past the end of the table",
        ));
    }
    for _ in 0..repeat {
        if let Some(slot) = lengths.get_mut(*index) {
            *slot = value;
        }
        *index += 1;
    }
    Ok(())
}

/// Reject output that would exceed `limit`.
fn check_limit(current: usize, adding: usize, limit: usize) -> Result<()> {
    if current.saturating_add(adding) > limit {
        return Err(PixelsError::malformed(
            "deflate",
            format!("stream expands beyond the {limit} byte limit implied by the image header"),
        ));
    }
    Ok(())
}

/// Decode one Huffman-coded block into `out`.
fn inflate_block(
    reader: &mut BitReader<'_>,
    literals: &Huffman,
    distances: &Huffman,
    out: &mut Vec<u8>,
    limit: usize,
) -> Result<()> {
    loop {
        let symbol = literals.decode(reader)?;
        match symbol {
            // A literal byte.
            0..=255 => {
                check_limit(out.len(), 1, limit)?;
                out.push(symbol as u8);
            }
            // End of block.
            256 => return Ok(()),
            // A back-reference.
            257..=285 => {
                let index = symbol as usize - 257;
                let base = LENGTH_BASE
                    .get(index)
                    .copied()
                    .ok_or_else(|| PixelsError::malformed("deflate", "invalid length code"))?;
                let extra = LENGTH_EXTRA.get(index).copied().unwrap_or(0);
                let length = base as usize + reader.take(u32::from(extra))? as usize;

                let distance_symbol = distances.decode(reader)? as usize;
                let distance_base = DISTANCE_BASE
                    .get(distance_symbol)
                    .copied()
                    .ok_or_else(|| PixelsError::malformed("deflate", "invalid distance code"))?;
                let distance_extra = DISTANCE_EXTRA.get(distance_symbol).copied().unwrap_or(0);
                let distance =
                    distance_base as usize + reader.take(u32::from(distance_extra))? as usize;

                // The reference must land inside what has already been emitted.
                // This is the check whose absence is the classic decompressor
                // out-of-bounds read.
                if distance == 0 || distance > out.len() {
                    return Err(PixelsError::malformed(
                        "deflate",
                        format!(
                            "back-reference of distance {distance} points before the start of \
                             the {} bytes decoded so far",
                            out.len()
                        ),
                    ));
                }
                check_limit(out.len(), length, limit)?;

                // Copied byte by byte on purpose: overlapping references are
                // legal and are how DEFLATE encodes runs, so the source may
                // include bytes this very loop is writing.
                let start = out.len() - distance;
                for offset in 0..length {
                    let byte = out.get(start + offset).copied().ok_or_else(|| {
                        PixelsError::malformed("deflate", "back-reference read out of range")
                    })?;
                    out.push(byte);
                }
            }
            _ => {
                return Err(PixelsError::malformed(
                    "deflate",
                    format!("literal/length symbol {symbol} is out of range"),
                ));
            }
        }
    }
}

/// Decompress a zlib stream (RFC 1950), verifying its Adler-32.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for a bad header, an unsupported
/// compression method, a checksum mismatch, or any DEFLATE error.
pub fn zlib_decompress(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let (&cmf, &flg) = match (data.first(), data.get(1)) {
        (Some(cmf), Some(flg)) => (cmf, flg),
        _ => {
            return Err(PixelsError::malformed(
                "zlib",
                "stream is shorter than its 2-byte header",
            ));
        }
    };
    if cmf & 0x0F != 8 {
        return Err(PixelsError::malformed(
            "zlib",
            format!("compression method {} is not deflate", cmf & 0x0F),
        ));
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(PixelsError::malformed(
            "zlib",
            "header check bits are wrong",
        ));
    }
    if flg & 0x20 != 0 {
        // A preset dictionary would change what the back-references mean, and
        // PNG forbids it (PNG spec §10.3).
        return Err(PixelsError::malformed(
            "zlib",
            "preset dictionaries are not supported",
        ));
    }

    let body = data.get(2..).unwrap_or(&[]);
    let out = inflate_to(body, limit)?;

    // The trailing Adler-32 is the last four bytes of the stream. The deflate
    // decoder does not report how much input it consumed, so the checksum is
    // read from the end.
    let Some(trailer) = data.len().checked_sub(4).and_then(|at| data.get(at..)) else {
        return Err(PixelsError::malformed(
            "zlib",
            "stream is missing its Adler-32 trailer",
        ));
    };
    let expected = u32::from_be_bytes([
        trailer.first().copied().unwrap_or(0),
        trailer.get(1).copied().unwrap_or(0),
        trailer.get(2).copied().unwrap_or(0),
        trailer.get(3).copied().unwrap_or(0),
    ]);
    let actual = Adler32::of(&out);
    if actual != expected {
        return Err(PixelsError::malformed(
            "zlib",
            format!("Adler-32 mismatch: stream declares {expected:#010x}, data is {actual:#010x}"),
        ));
    }
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

    /// A stored-block DEFLATE stream wrapping `payload`.
    fn stored_stream(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x01];
        let length = payload.len() as u16;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn stored_blocks_round_trip() {
        let payload = b"the quick brown fox";
        let out = inflate_to(&stored_stream(payload), 1024).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn an_empty_stored_block_yields_nothing() {
        assert_eq!(
            inflate_to(&stored_stream(b""), 16).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn a_stored_block_with_a_bad_complement_is_rejected() {
        let mut stream = stored_stream(b"abc");
        stream[3] ^= 0xFF;
        let err = inflate_to(&stream, 1024).unwrap_err();
        assert!(err.to_string().contains("complement"), "{err}");
    }

    #[test]
    fn fixed_huffman_decodes_a_known_stream() {
        // zlib's output for "hello" at the default level, minus the wrapper:
        // a single fixed-Huffman block.
        let stream = [0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x07, 0x00];
        assert_eq!(inflate_to(&stream, 64).unwrap(), b"hello");
    }

    #[test]
    fn zlib_wrapped_streams_verify_their_checksum() {
        // zlib -9 output for "hello world".
        let stream = [
            0x78, 0xDA, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0x28, 0xCF, 0x2F, 0xCA, 0x49, 0x01,
            0x00, 0x1A, 0x0B, 0x04, 0x5D,
        ];
        assert_eq!(zlib_decompress(&stream, 64).unwrap(), b"hello world");
    }

    #[test]
    fn a_corrupted_adler_is_reported() {
        let mut stream = vec![
            0x78, 0xDA, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0x28, 0xCF, 0x2F, 0xCA, 0x49, 0x01,
            0x00, 0x1A, 0x0B, 0x04, 0x5D,
        ];
        let last = stream.len() - 1;
        stream[last] ^= 0xFF;
        let err = zlib_decompress(&stream, 64).unwrap_err();
        assert!(err.to_string().contains("Adler-32"), "{err}");
    }

    #[test]
    fn zlib_headers_are_validated() {
        assert!(zlib_decompress(&[], 16).is_err(), "empty");
        assert!(zlib_decompress(&[0x78], 16).is_err(), "one byte");
        // Compression method 7 is not deflate.
        assert!(zlib_decompress(&[0x77, 0x00, 0x00], 16).is_err());
        // Header check bits that do not divide by 31.
        assert!(zlib_decompress(&[0x78, 0x00, 0x00], 16).is_err());
        // Preset dictionary flag set (0x78, 0x3F is divisible by 31).
        let err = zlib_decompress(&[0x78, 0x3F, 0x00], 16).unwrap_err();
        assert!(err.to_string().contains("dictionar"), "{err}");
    }

    #[test]
    fn reserved_block_type_three_is_rejected() {
        // BFINAL=1, BTYPE=3 packs to 0b111 in the first byte.
        let err = inflate_to(&[0x07], 16).unwrap_err();
        assert!(err.to_string().contains("reserved"), "{err}");
    }

    #[test]
    fn a_back_reference_before_the_start_is_rejected() {
        // A fixed-Huffman block whose very first symbol is a length code:
        // nothing has been emitted, so the distance points before the start of
        // the output. This is the classic decompressor out-of-bounds read, and
        // the bytes are hand-assembled because no real encoder emits it.
        //
        // BFINAL=1, BTYPE=01, symbol 257 (7-bit code 0000001), distance code 0.
        let err = inflate_to(&[0x03, 0x02], 1024).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
        assert!(err.to_string().contains("back-reference"), "{err}");
    }

    #[test]
    fn output_beyond_the_limit_is_malformed_not_an_allocation() {
        // A decompression bomb: 65535 zero bytes from a tiny stored block.
        let bomb = stored_stream(&vec![0_u8; 65535]);
        let err = inflate_to(&bomb, 1024).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
        assert!(err.to_string().contains("limit"), "{err}");
        // The same stream within a generous limit is fine.
        assert_eq!(inflate_to(&bomb, 65535).unwrap().len(), 65535);
    }

    #[test]
    fn every_truncation_of_a_valid_stream_is_an_error_not_a_panic() {
        let full = [
            0x78, 0xDA, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0x28, 0xCF, 0x2F, 0xCA, 0x49, 0x01,
            0x00, 0x1A, 0x0B, 0x04, 0x5D,
        ];
        for len in 0..full.len() {
            // Must not panic. Some prefixes decode a valid shorter stream, so
            // success is acceptable; a crash is not.
            let _ = zlib_decompress(&full[..len], 4096);
        }
        assert!(
            zlib_decompress(&full, 4096).is_ok(),
            "the untruncated stream still works"
        );
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        // A cheap deterministic sweep; the real corpus fuzzing is in the
        // crate's fuzz tests. This is here so the module is never committed in
        // a state that crashes on trivial garbage.
        let mut state = 0x1234_5678_u32;
        for _ in 0..2000 {
            let mut bytes = Vec::new();
            for _ in 0..32 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bytes.push((state >> 24) as u8);
            }
            let _ = inflate_to(&bytes, 4096);
            let _ = zlib_decompress(&bytes, 4096);
        }
    }

    #[test]
    fn over_subscribed_huffman_tables_are_rejected() {
        // Three one-bit codes cannot form a prefix code.
        assert!(Huffman::new(&[1, 1, 1]).is_err());
        // Two one-bit codes can.
        assert!(Huffman::new(&[1, 1]).is_ok());
        // A length beyond 15 bits is out of spec.
        assert!(Huffman::new(&[16]).is_err());
    }

    #[test]
    fn overlapping_back_references_encode_runs() {
        // A run is encoded as a copy from distance 1, so the source overlaps
        // the destination and includes bytes the copy loop is itself writing.
        // Produced by zlib, not by hand.
        let stream = [0x4B, 0x4C, 0x84, 0x00, 0x00];
        assert_eq!(inflate_to(&stream, 64).unwrap(), b"aaaaaaaa");
    }
}
