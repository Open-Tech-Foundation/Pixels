//! LZW, in both the GIF and TIFF dialects.
//!
//! # One implementation, two dialects
//!
//! GIF (§Appendix F) and TIFF (§Section 13) both use variable-width LZW with a
//! clear code and an end code, and they differ in exactly two ways:
//!
//! - **Bit order.** GIF packs codes least-significant-bit first; TIFF packs
//!   most-significant-bit first.
//! - **When the code width grows.** GIF widens *after* the table reaches
//!   `2^width` entries; TIFF widens one code *early*, a documented quirk of
//!   the original implementation that every conforming decoder must match.
//!
//! Everything else — the table, the deferred-clear behaviour, the special
//! first code — is identical. Two copies would be two chances to get the
//! table logic wrong, with nothing testing that they agree, so this is one
//! implementation with the differences named as [`BitOrder`].
//!
//! # The KwKwK case
//!
//! LZW's one genuinely subtle case: a code can refer to an entry that is only
//! being defined by this very code. It happens whenever the input contains
//! `KwKwK` — a sequence whose second occurrence begins before its first
//! finished being added. A decoder that looks the code up naively finds
//! nothing and either errors or reads uninitialized memory. It is handled
//! explicitly below, and tested directly, because it is rare enough in small
//! fixtures to escape casual testing and common enough in real images to break
//! immediately in production.

use crate::{Error, Result};

/// Which end of a byte codes are packed from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitOrder {
    /// Least-significant bit first, as GIF packs codes.
    LsbFirst,
    /// Most-significant bit first, as TIFF packs codes.
    ///
    /// TIFF also widens the code one entry early; the two go together in
    /// practice, so [`BitOrder`] selects both.
    MsbFirst,
}

impl BitOrder {
    /// Whether this dialect increases the code width one code early.
    const fn early_change(self) -> bool {
        matches!(self, Self::MsbFirst)
    }
}

/// The largest code width either dialect permits.
const MAX_WIDTH: u32 = 12;

/// The table size at which a decoder moves to the next code width.
///
/// TIFF widens one entry early — a documented quirk of the original
/// implementation that every conforming decoder must reproduce.
const fn widen_at(width: u32, order: BitOrder) -> u16 {
    let full = 1_u16 << width;
    if order.early_change() {
        full.saturating_sub(1)
    } else {
        full
    }
}
/// The largest table any dialect can reach.
const MAX_CODES: usize = 1 << MAX_WIDTH;

/// Reads variable-width codes from a byte stream.
#[derive(Debug)]
struct CodeReader<'a> {
    data: &'a [u8],
    position: usize,
    bits: u32,
    count: u32,
    order: BitOrder,
}

impl<'a> CodeReader<'a> {
    const fn new(data: &'a [u8], order: BitOrder) -> Self {
        Self {
            data,
            position: 0,
            bits: 0,
            count: 0,
            order,
        }
    }

    /// Read one code of `width` bits, or `None` at end of input.
    fn next(&mut self, width: u32) -> Option<u16> {
        while self.count < width {
            let &byte = self.data.get(self.position)?;
            self.position += 1;
            match self.order {
                BitOrder::LsbFirst => self.bits |= u32::from(byte) << self.count,
                BitOrder::MsbFirst => self.bits = (self.bits << 8) | u32::from(byte),
            }
            self.count += 8;
        }
        let code = match self.order {
            BitOrder::LsbFirst => {
                let mask = (1_u32 << width) - 1;
                let value = self.bits & mask;
                self.bits >>= width;
                self.count -= width;
                value
            }
            BitOrder::MsbFirst => {
                let shift = self.count - width;
                let value = (self.bits >> shift) & ((1_u32 << width) - 1);
                self.count -= width;
                // Keep only the bits still unconsumed, so `bits` cannot grow
                // past 32 on a long run of whole bytes.
                self.bits &= (1_u32 << self.count).wrapping_sub(1);
                value
            }
        };
        Some(code as u16)
    }
}

/// Writes variable-width codes to a byte stream.
#[derive(Debug)]
struct CodeWriter {
    out: Vec<u8>,
    bits: u32,
    count: u32,
    order: BitOrder,
}

impl CodeWriter {
    const fn new(order: BitOrder) -> Self {
        Self {
            out: Vec::new(),
            bits: 0,
            count: 0,
            order,
        }
    }

    fn push(&mut self, code: u16, width: u32) {
        match self.order {
            BitOrder::LsbFirst => {
                self.bits |= u32::from(code) << self.count;
                self.count += width;
                while self.count >= 8 {
                    self.out.push((self.bits & 0xFF) as u8);
                    self.bits >>= 8;
                    self.count -= 8;
                }
            }
            BitOrder::MsbFirst => {
                self.bits = (self.bits << width) | u32::from(code);
                self.count += width;
                while self.count >= 8 {
                    let shift = self.count - 8;
                    self.out.push(((self.bits >> shift) & 0xFF) as u8);
                    self.count -= 8;
                    self.bits &= (1_u32 << self.count).wrapping_sub(1);
                }
            }
        }
    }

    /// Flush any partial byte and return the stream.
    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            let byte = match self.order {
                BitOrder::LsbFirst => (self.bits & 0xFF) as u8,
                // Pad with zeroes on the right, which is what both
                // specifications' encoders emit.
                BitOrder::MsbFirst => ((self.bits << (8 - self.count)) & 0xFF) as u8,
            };
            self.out.push(byte);
        }
        self.out
    }
}

/// Decompresses an LZW stream.
#[derive(Debug)]
pub struct LzwDecoder {
    order: BitOrder,
    minimum_width: u32,
}

impl LzwDecoder {
    /// A decoder for `order`, with `minimum_width` bits per initial code.
    ///
    /// GIF carries the minimum width in the image block; TIFF fixes it at 8.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `minimum_width` is outside 2..=11. A width of 1 has
    /// no room for both the clear and end codes, and 12 leaves none for the
    /// table.
    pub fn new(order: BitOrder, minimum_width: u32) -> Result<Self> {
        if !(2..=11).contains(&minimum_width) {
            return Err(Error::malformed(
                "lzw",
                format!("minimum code width {minimum_width} is outside 2..=11"),
            ));
        }
        Ok(Self {
            order,
            minimum_width,
        })
    }

    /// A decoder for GIF's dialect at `minimum_width`.
    ///
    /// # Errors
    ///
    /// As [`LzwDecoder::new`].
    pub fn gif(minimum_width: u32) -> Result<Self> {
        Self::new(BitOrder::LsbFirst, minimum_width)
    }

    /// A decoder for TIFF's dialect, whose minimum width is always 8.
    #[must_use]
    pub const fn tiff() -> Self {
        Self {
            order: BitOrder::MsbFirst,
            minimum_width: 8,
        }
    }

    /// Decompress `data`, refusing to produce more than `limit` bytes.
    ///
    /// The bound is what makes a decompression bomb a malformed-input error
    /// rather than an allocation the caller has to survive: LZW expands by up
    /// to 4096:1, so a caller states what it is prepared to accept.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] for an invalid code, a stream that ends mid-symbol
    /// without an end code, or output exceeding `limit`.
    pub fn decode(&self, data: &[u8], limit: usize) -> Result<Vec<u8>> {
        let clear = 1_u16 << self.minimum_width;
        let end = clear + 1;
        // Codes below this are literals. It is *not* 256: GIF derives the
        // minimum width from the palette size, so a 4-colour image has
        // literals 0..=3 and its table starts at code 6. Assuming 256 here
        // decodes small-palette images as gibberish while leaving 8-bit ones
        // perfect, which is exactly the bug that survives casual testing.
        let literals = clear;

        let mut reader = CodeReader::new(data, self.order);
        let mut out: Vec<u8> = Vec::new();

        // The table stores each entry as (prefix code, final byte), so an
        // entry costs four bytes instead of a growing Vec. Expanding walks the
        // prefix chain backwards, which is why `scratch` is reversed after.
        let mut prefix = vec![0_u16; MAX_CODES];
        let mut suffix = vec![0_u8; MAX_CODES];
        let mut scratch: Vec<u8> = Vec::with_capacity(MAX_CODES);

        let mut width = self.minimum_width + 1;
        let mut next = end + 1;
        let mut previous: Option<u16> = None;

        // Running out of input without an end code is common enough in real
        // files — many encoders omit it — that treating it as malformed would
        // reject images every other decoder accepts. So the loop ends on
        // either an end code or exhausted input, and both are success.
        while let Some(code) = reader.next(width) {
            if code == clear {
                width = self.minimum_width + 1;
                next = end + 1;
                previous = None;
                continue;
            }
            if code == end {
                break;
            }

            // Expand this code into `scratch`, handling the KwKwK case where
            // the code being read is the one this step is about to define.
            scratch.clear();
            let first = if code < next {
                expand(code, literals, &prefix, &suffix, &mut scratch)?
            } else if code == next {
                let Some(previous_code) = previous else {
                    return Err(Error::malformed(
                        "lzw",
                        "the first code after a clear cannot be a forward reference",
                    ));
                };
                // KwKwK: the entry is `previous` followed by its own first
                // byte, which is exactly what this branch reconstructs.
                let first = expand(previous_code, literals, &prefix, &suffix, &mut scratch)?;
                scratch.push(first);
                first
            } else {
                return Err(Error::malformed(
                    "lzw",
                    format!("code {code} is beyond the {next} entries defined so far"),
                ));
            };

            if out.len() + scratch.len() > limit {
                return Err(Error::malformed(
                    "lzw",
                    format!("stream expands beyond the {limit} byte limit"),
                ));
            }
            out.extend_from_slice(&scratch);

            // Define the new entry: the previous string plus this one's first
            // byte. Nothing is defined for the first code after a clear.
            let room = (next as usize) < MAX_CODES;
            if let Some(previous_code) = previous.filter(|_| room) {
                if let (Some(p), Some(s)) =
                    (prefix.get_mut(next as usize), suffix.get_mut(next as usize))
                {
                    *p = previous_code;
                    *s = first;
                }
                next += 1;
                if next >= widen_at(width, self.order) && width < MAX_WIDTH {
                    width += 1;
                }
            }
            previous = Some(code);
        }

        Ok(out)
    }
}

/// Expand `code` into `out`, returning the first byte of the expansion.
///
/// `literals` is the number of single-byte codes, which is `2^minimum_width`
/// and not necessarily 256. Walking the prefix chain backwards and reversing
/// is why the length is bounded by the table size rather than by the output.
fn expand(
    code: u16,
    literals: u16,
    prefix: &[u16],
    suffix: &[u8],
    out: &mut Vec<u8>,
) -> Result<u8> {
    let start = out.len();
    let mut current = code;
    // A corrupt table could in principle form a cycle; the table size is a
    // hard bound on any legitimate chain, so exceeding it is malformed input.
    for _ in 0..=MAX_CODES {
        if current < literals {
            out.push(current as u8);
            let Some(slice) = out.get_mut(start..) else {
                break;
            };
            slice.reverse();
            return Ok(current as u8);
        }
        let index = current as usize;
        let (Some(&byte), Some(&parent)) = (suffix.get(index), prefix.get(index)) else {
            return Err(Error::malformed("lzw", "code refers outside the table"));
        };
        out.push(byte);
        current = parent;
    }
    Err(Error::malformed("lzw", "code chain does not terminate"))
}

/// Compresses to an LZW stream.
#[derive(Debug)]
pub struct LzwEncoder {
    order: BitOrder,
    minimum_width: u32,
}

impl LzwEncoder {
    /// An encoder for `order`, with `minimum_width` bits per initial code.
    ///
    /// # Errors
    ///
    /// As [`LzwDecoder::new`].
    pub fn new(order: BitOrder, minimum_width: u32) -> Result<Self> {
        if !(2..=11).contains(&minimum_width) {
            return Err(Error::malformed(
                "lzw",
                format!("minimum code width {minimum_width} is outside 2..=11"),
            ));
        }
        Ok(Self {
            order,
            minimum_width,
        })
    }

    /// An encoder for GIF's dialect at `minimum_width`.
    ///
    /// # Errors
    ///
    /// As [`LzwDecoder::new`].
    pub fn gif(minimum_width: u32) -> Result<Self> {
        Self::new(BitOrder::LsbFirst, minimum_width)
    }

    /// An encoder for TIFF's dialect.
    #[must_use]
    pub const fn tiff() -> Self {
        Self {
            order: BitOrder::MsbFirst,
            minimum_width: 8,
        }
    }

    /// Compress `data`.
    ///
    /// The output always begins with a clear code and ends with an end code,
    /// which every conforming decoder accepts and some require.
    #[must_use]
    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        let clear = 1_u16 << self.minimum_width;
        let end = clear + 1;
        let mut writer = CodeWriter::new(self.order);
        let mut width = self.minimum_width + 1;

        // The dictionary maps (prefix code, next byte) to a code. A flat map
        // over `prefix * 256 + byte` would be 1 MB; a hash map of the pairs
        // actually used is smaller and no slower at these sizes.
        let mut table: std::collections::HashMap<(u16, u8), u16> = std::collections::HashMap::new();
        let mut next = end + 1;

        writer.push(clear, width);

        let mut current: Option<u16> = None;
        for &byte in data {
            let combined = match current {
                None => {
                    current = Some(u16::from(byte));
                    continue;
                }
                Some(code) => (code, byte),
            };

            if let Some(&found) = table.get(&combined) {
                current = Some(found);
                continue;
            }

            // Emit what we had, then define the extended string.
            if let Some(code) = current {
                writer.push(code, width);
            }
            if (next as usize) < MAX_CODES {
                table.insert(combined, next);
                next += 1;
                // Strictly greater, where the decoder uses "at least". The
                // asymmetry is not a choice: a decoder cannot define an entry
                // until it has read the *following* code and knows its first
                // byte, so its table is one behind the encoder's at the moment
                // the width is decided. Using the same comparison on both
                // sides makes an encoder and decoder agree with each other and
                // disagree with the rest of the world — which passes every
                // round-trip test and fails on the first real image.
                if next > widen_at(width, self.order) && width < MAX_WIDTH {
                    width += 1;
                }
            } else {
                // The table is full: reset, exactly as the decoder will when
                // it sees the clear code.
                writer.push(clear, width);
                table.clear();
                width = self.minimum_width + 1;
                next = end + 1;
            }
            current = Some(u16::from(byte));
        }

        if let Some(code) = current {
            writer.push(code, width);
        }
        writer.push(end, width);
        writer.finish()
    }
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

    fn round_trip(order: BitOrder, width: u32, data: &[u8]) {
        let encoder = LzwEncoder::new(order, width).unwrap();
        let decoder = LzwDecoder::new(order, width).unwrap();
        let compressed = encoder.encode(data);
        let back = decoder
            .decode(&compressed, data.len() * 4 + 1024)
            .unwrap_or_else(|e| panic!("{order:?} width {width}: {e}"));
        assert_eq!(back, data, "{order:?} width {width} did not round-trip");
    }

    #[test]
    fn both_dialects_round_trip_everything() {
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            vec![0],
            vec![1, 2, 3],
            vec![7; 1000],
            b"the quick brown fox jumps over the lazy dog. ".repeat(60),
            (0..=255_u8).cycle().take(5000).collect(),
            (0..4000).map(|i| ((i * 37) % 251) as u8).collect(),
        ];
        for order in [BitOrder::LsbFirst, BitOrder::MsbFirst] {
            for data in &cases {
                round_trip(order, 8, data);
            }
        }
    }

    #[test]
    fn every_gif_minimum_width_round_trips() {
        // GIF carries the minimum width per image, derived from the palette
        // size, so every legal width has to work.
        for width in 2..=8_u32 {
            // The alphabet is 2^width, which is 256 at width 8 and therefore
            // does not fit a u8 — compute it wider and narrow the result.
            let alphabet = 1_u16 << width;
            let data: Vec<u8> = (0..3000).map(|i| ((i * 7) % alphabet) as u8).collect();
            round_trip(BitOrder::LsbFirst, width, &data);
        }
    }

    #[test]
    fn the_kwkwk_case_decodes_correctly() {
        // The one genuinely subtle case in LZW: a code referring to the entry
        // this very step defines. It needs input whose second occurrence of a
        // run begins before the first finished being added — a long run of one
        // byte is the shortest way to force it.
        for order in [BitOrder::LsbFirst, BitOrder::MsbFirst] {
            for length in [3_usize, 4, 5, 10, 100, 4000] {
                let data = vec![0xAB_u8; length];
                round_trip(order, 8, &data);
            }
            // The classic textbook case, spelled out.
            round_trip(order, 8, b"ababababab");
            round_trip(order, 8, b"aaaaaaaaaaaaaaaa");
        }
    }

    #[test]
    fn a_stream_longer_than_the_table_resets_cleanly() {
        // Past 4096 entries the encoder must emit a clear and start again, and
        // the decoder must follow it. Random-ish data fills the table fastest.
        for order in [BitOrder::LsbFirst, BitOrder::MsbFirst] {
            let data: Vec<u8> = (0..200_000_u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
                .collect();
            round_trip(order, 8, &data);
        }
    }

    #[test]
    fn a_forward_reference_is_an_error_not_a_panic() {
        // A code beyond the table is the classic corrupt-LZW case, and the
        // one that reads uninitialized memory in a careless implementation.
        let decoder = LzwDecoder::gif(8).unwrap();
        // 0x100 is clear at width 9; 0x1FF is far beyond anything defined.
        let mut writer = CodeWriter::new(BitOrder::LsbFirst);
        writer.push(0x100, 9);
        writer.push(0x1FF, 9);
        let stream = writer.finish();
        let error = decoder.decode(&stream, 1 << 20).unwrap_err();
        assert!(error.detail().contains("beyond"), "{error}");
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        // Every LZW stream a codec sees is attacker-controlled.
        let mut seed = 0x1234_5678_u32;
        for order in [BitOrder::LsbFirst, BitOrder::MsbFirst] {
            for width in [2_u32, 8, 11] {
                let decoder = LzwDecoder::new(order, width).unwrap();
                for _ in 0..500 {
                    let len = (seed % 200) as usize + 1;
                    let data: Vec<u8> = (0..len)
                        .map(|_| {
                            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            (seed >> 16) as u8
                        })
                        .collect();
                    let _ = decoder.decode(&data, 1 << 16);
                }
            }
        }
    }

    #[test]
    fn every_truncation_of_a_valid_stream_is_an_error_or_a_short_read() {
        // Truncation must never panic. It may legitimately produce a partial
        // decode, because a missing end code is tolerated.
        let data = b"the quick brown fox. ".repeat(40);
        let stream = LzwEncoder::gif(8).unwrap().encode(&data);
        let decoder = LzwDecoder::gif(8).unwrap();
        for cut in 0..stream.len() {
            let _ = decoder.decode(&stream[..cut], 1 << 20);
        }
    }

    #[test]
    fn the_output_limit_is_enforced() {
        // LZW expands by up to 4096:1, so the bound is what stands between a
        // 2 KB file and a gigabyte of memory.
        let data = vec![0_u8; 1_000_000];
        let stream = LzwEncoder::gif(8).unwrap().encode(&data);
        assert!(stream.len() < 100_000, "the bomb should be small");
        let error = LzwDecoder::gif(8)
            .unwrap()
            .decode(&stream, 4096)
            .unwrap_err();
        assert!(error.detail().contains("limit"), "{error}");
    }

    #[test]
    fn an_out_of_range_minimum_width_is_rejected() {
        for width in [0_u32, 1, 12, 99] {
            assert!(
                LzwDecoder::new(BitOrder::LsbFirst, width).is_err(),
                "{width}"
            );
            assert!(
                LzwEncoder::new(BitOrder::LsbFirst, width).is_err(),
                "{width}"
            );
        }
    }

    #[test]
    fn a_clear_code_resets_the_table_mid_stream() {
        // Decoders must follow an encoder that clears early, which some do to
        // bound their own memory.
        let mut writer = CodeWriter::new(BitOrder::LsbFirst);
        writer.push(0x100, 9); // clear
        writer.push(b'A'.into(), 9);
        writer.push(b'B'.into(), 9);
        writer.push(0x100, 9); // clear again
        writer.push(b'C'.into(), 9);
        writer.push(0x101, 9); // end
        let stream = writer.finish();
        let out = LzwDecoder::gif(8).unwrap().decode(&stream, 1024).unwrap();
        assert_eq!(out, b"ABC");
    }

    #[test]
    fn the_two_dialects_pack_bits_differently() {
        // If both dialects produced the same bytes, one of them would be
        // wrong, and every round-trip test above would still pass.
        let data = b"hello world, this is a test of bit ordering".repeat(4);
        let lsb = LzwEncoder::new(BitOrder::LsbFirst, 8)
            .unwrap()
            .encode(&data);
        let msb = LzwEncoder::new(BitOrder::MsbFirst, 8)
            .unwrap()
            .encode(&data);
        assert_ne!(lsb, msb, "the two bit orders produced identical streams");
    }
}
