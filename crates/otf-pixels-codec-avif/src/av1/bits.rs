//! The AV1 bitstream reader.
//!
//! AV1 is read most-significant-bit first (AV1 spec §4.10.2). This is a
//! crate-local reader by the same convention the rest of the workspace follows:
//! JPEG and inflate each carry their own, because a bit order and a set of
//! variable-length primitives are format decisions, not shared infrastructure.
//!
//! Every primitive here is fallible. The spec is written as if the stream never
//! runs out — a conformant one does not — but this decoder reads
//! attacker-controlled bytes under `unsafe_code = "forbid"` and a ban on
//! `panic!`, so reading past the end is a returned error, never a trap.

use otf_pixels_core::{PixelsError, Result};

/// A most-significant-bit-first reader over an AV1 byte slice.
///
/// Position is tracked in bits so `byte_alignment` and the byte-granular
/// primitives (`leb128`, `le`) can assert and act on alignment.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// The next bit to read, counted from the front of `data`. Bit 0 is the
    /// most significant bit of byte 0.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap a byte slice. Reading starts at its first bit.
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// The current position, in bits from the start.
    #[must_use]
    pub fn bit_position(&self) -> usize {
        self.pos
    }

    /// The total length of the underlying slice, in bits.
    #[must_use]
    pub fn bit_len(&self) -> usize {
        self.data.len().saturating_mul(8)
    }

    /// Bits remaining before the end of the slice.
    #[must_use]
    pub fn bits_left(&self) -> usize {
        self.bit_len().saturating_sub(self.pos)
    }

    /// Whether the reader sits exactly on a byte boundary.
    #[must_use]
    pub fn is_byte_aligned(&self) -> bool {
        self.pos % 8 == 0
    }

    /// The current position in whole bytes — only meaningful when byte-aligned.
    #[must_use]
    pub fn byte_position(&self) -> usize {
        self.pos / 8
    }

    /// Read a single bit.
    fn read_bit(&mut self) -> Result<u32> {
        let byte = self.pos / 8;
        let Some(&value) = self.data.get(byte) else {
            return Err(PixelsError::malformed(
                "avif",
                "the AV1 bitstream ended in the middle of a value",
            ));
        };
        // Bit 0 of a byte is its most significant bit (spec §4.10.2).
        let shift = 7 - (self.pos % 8);
        self.pos += 1;
        Ok(u32::from((value >> shift) & 1))
    }

    /// `f(n)` — read `n` bits as an unsigned integer, MSB first (§4.10.2).
    ///
    /// `n` is at most 32; the AV1 syntax never reads a wider `f(n)` in one call.
    pub fn f(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(PixelsError::malformed(
                "avif",
                "an AV1 fixed-width read wider than 32 bits is a decoder bug",
            ));
        }
        let mut value: u32 = 0;
        for _ in 0..n {
            // Shifting a u32 left by up to 31 and or-ing one bit never
            // overflows; the width guard above keeps the loop within 32 steps.
            value = (value << 1) | self.read_bit()?;
        }
        Ok(value)
    }

    /// `f(n)` for values that may need the full 64 bits (`le` uses it).
    fn f64(&mut self, n: u32) -> Result<u64> {
        if n > 64 {
            return Err(PixelsError::malformed(
                "avif",
                "an AV1 fixed-width read wider than 64 bits is a decoder bug",
            ));
        }
        let mut value: u64 = 0;
        for _ in 0..n {
            value = (value << 1) | u64::from(self.read_bit()?);
        }
        Ok(value)
    }

    /// Read a boolean flag — `f(1)` reported as `bool`.
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.f(1)? != 0)
    }

    /// `uvlc()` — unsigned variable-length code (§4.10.3).
    ///
    /// A run of zero bits terminated by a one, then that many value bits. A run
    /// of 32 or more leading zeros is the spec's saturation case and yields
    /// `u32::MAX`.
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut leading_zeros: u32 = 0;
        loop {
            if self.flag()? {
                break;
            }
            leading_zeros += 1;
            if leading_zeros >= 32 {
                return Ok(u32::MAX);
            }
        }
        let value = self.f(leading_zeros)?;
        // value + 2^leading_zeros - 1, computed without overflow: leading_zeros
        // is < 32 here, and value < 2^leading_zeros, so the sum fits in u32.
        Ok(value + ((1_u32 << leading_zeros) - 1))
    }

    /// `le(n)` — an `n`-byte little-endian unsigned integer (§4.10.4).
    ///
    /// Must be byte-aligned, which the syntax guarantees at every call site.
    pub fn le(&mut self, n: u32) -> Result<u64> {
        if !self.is_byte_aligned() {
            return Err(PixelsError::malformed(
                "avif",
                "an AV1 le() read was not byte-aligned",
            ));
        }
        let mut value: u64 = 0;
        for i in 0..n {
            let byte = self.f64(8)?;
            value |= byte << (i * 8);
        }
        Ok(value)
    }

    /// `leb128()` — a little-endian base-128 unsigned integer (§4.10.5).
    ///
    /// At most eight bytes; a ninth continuation bit is malformed. Returns the
    /// value and the number of bytes consumed so callers can bound a payload.
    pub fn leb128(&mut self) -> Result<u64> {
        if !self.is_byte_aligned() {
            return Err(PixelsError::malformed(
                "avif",
                "an AV1 leb128() read was not byte-aligned",
            ));
        }
        let mut value: u64 = 0;
        for i in 0..8 {
            let byte = self.f(8)?;
            // Seven payload bits per byte, low group first.
            value |= u64::from(byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(PixelsError::malformed(
            "avif",
            "an AV1 leb128 value ran past its eight-byte maximum",
        ))
    }

    /// `su(n)` — a signed integer in `n+1` bits, sign last (§4.10.6).
    pub fn su(&mut self, n: u32) -> Result<i32> {
        let value = self.f(n + 1)? as i32;
        let sign_mask = 1_i32 << n;
        if value & sign_mask != 0 {
            Ok(value - 2 * sign_mask)
        } else {
            Ok(value)
        }
    }

    /// `ns(n)` — a non-symmetric unsigned integer over `[0, n)` (§4.10.7).
    ///
    /// Uses one fewer bit for the smaller half of the range, so it is not a
    /// plain `f`. `n == 0` reads nothing and yields 0.
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = floor_log2(n) + 1;
        let m = (1_u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.f(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// Advance to the next byte boundary, requiring the skipped bits be zero
    /// (`byte_alignment()`, §5.3.5). AV1 mandates the padding be zero.
    pub fn byte_alignment(&mut self) -> Result<()> {
        while !self.is_byte_aligned() {
            if self.f(1)? != 0 {
                return Err(PixelsError::malformed(
                    "avif",
                    "an AV1 byte-alignment pad bit was not zero",
                ));
            }
        }
        Ok(())
    }

    /// Skip `n` bits without interpreting them.
    pub fn skip_bits(&mut self, n: usize) -> Result<()> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bit_len());
        let Some(end) = end else {
            return Err(PixelsError::malformed(
                "avif",
                "an AV1 skip ran past the end of the bitstream",
            ));
        };
        self.pos = end;
        Ok(())
    }
}

/// `FloorLog2(x)` (§4.7): the index of the most significant set bit. `x` must
/// be non-zero, which every AV1 call site guarantees.
#[must_use]
pub fn floor_log2(x: u32) -> u32 {
    // 31 - leading_zeros is the MSB index; for x >= 1 it is well-defined.
    31 - x.leading_zeros()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unusual_byte_groupings,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::ErrorCode;

    #[test]
    fn f_reads_most_significant_bit_first() {
        // 0b1011_0010, 0b0100_0000
        let data = [0xB2, 0x40];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(3).unwrap(), 0b101);
        assert_eq!(r.f(5).unwrap(), 0b10010);
        assert_eq!(r.f(2).unwrap(), 0b01);
        assert_eq!(r.bit_position(), 10);
    }

    #[test]
    fn f_of_zero_reads_nothing() {
        let data = [0xFF];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(0).unwrap(), 0);
        assert_eq!(r.bit_position(), 0);
    }

    #[test]
    fn reading_past_the_end_is_an_error_not_a_panic() {
        let data = [0xFF];
        let mut r = BitReader::new(&data);
        assert_eq!(r.f(8).unwrap(), 0xFF);
        let err = r.f(1).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
    }

    #[test]
    fn uvlc_decodes_the_exponential_golomb_shape() {
        // 1                -> 0
        // 010              -> 1
        // 011              -> 2
        // 00100            -> 3
        // Pack: 1 010 011 00100 = 1010_0110_0100_...
        let data = [0b1010_0110, 0b0100_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.uvlc().unwrap(), 0);
        assert_eq!(r.uvlc().unwrap(), 1);
        assert_eq!(r.uvlc().unwrap(), 2);
        assert_eq!(r.uvlc().unwrap(), 3);
    }

    #[test]
    fn uvlc_saturates_at_thirty_two_leading_zeros() {
        // 32 zero bits with no terminating one: four zero bytes, then more.
        let data = [0x00, 0x00, 0x00, 0x00, 0x80];
        let mut r = BitReader::new(&data);
        assert_eq!(r.uvlc().unwrap(), u32::MAX);
    }

    #[test]
    fn leb128_reads_little_endian_base_128() {
        // 0xE5 0x8E 0x26 -> 624485, the canonical LEB128 example.
        let data = [0xE5, 0x8E, 0x26];
        let mut r = BitReader::new(&data);
        assert_eq!(r.leb128().unwrap(), 624_485);
        assert_eq!(r.byte_position(), 3);
    }

    #[test]
    fn leb128_rejects_a_ninth_continuation_byte() {
        let data = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
        let mut r = BitReader::new(&data);
        let err = r.leb128().unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
    }

    #[test]
    fn le_reads_little_endian_bytes() {
        let data = [0x34, 0x12];
        let mut r = BitReader::new(&data);
        assert_eq!(r.le(2).unwrap(), 0x1234);
    }

    #[test]
    fn su_recovers_negative_values() {
        // su(3) reads 4 bits. 0b1111 -> -1; 0b0111 -> 7; 0b1000 -> -8.
        let data = [0b1111_0111, 0b1000_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.su(3).unwrap(), -1);
        assert_eq!(r.su(3).unwrap(), 7);
        assert_eq!(r.su(3).unwrap(), -8);
    }

    #[test]
    fn ns_uses_one_fewer_bit_for_the_low_half() {
        // n = 3: w = 2, m = 1. v = f(1); v<1 -> value v; else read one more.
        // Bits 0 -> 0. Bits 10 -> (1<<1)-1+0 = 1. Bits 11 -> (1<<1)-1+1 = 2.
        let data = [0b0_10_11_000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.ns(3).unwrap(), 0);
        assert_eq!(r.ns(3).unwrap(), 1);
        assert_eq!(r.ns(3).unwrap(), 2);
    }

    #[test]
    fn ns_of_a_power_of_two_is_plain_fixed_width() {
        // n = 4: w = 2, m = 0, so every value is read in 2 bits.
        let data = [0b00_01_10_11];
        let mut r = BitReader::new(&data);
        assert_eq!(r.ns(4).unwrap(), 0);
        assert_eq!(r.ns(4).unwrap(), 1);
        assert_eq!(r.ns(4).unwrap(), 2);
        assert_eq!(r.ns(4).unwrap(), 3);
    }

    #[test]
    fn byte_alignment_requires_zero_padding() {
        let mut r = BitReader::new(&[0b101_00000]);
        assert_eq!(r.f(3).unwrap(), 0b101);
        r.byte_alignment().unwrap();
        assert!(r.is_byte_aligned());
        assert_eq!(r.byte_position(), 1);

        let mut bad = BitReader::new(&[0b101_00001]);
        assert_eq!(bad.f(3).unwrap(), 0b101);
        assert_eq!(
            bad.byte_alignment().unwrap_err().code(),
            ErrorCode::Malformed
        );
    }

    #[test]
    fn floor_log2_is_the_top_set_bit() {
        assert_eq!(floor_log2(1), 0);
        assert_eq!(floor_log2(2), 1);
        assert_eq!(floor_log2(3), 1);
        assert_eq!(floor_log2(255), 7);
        assert_eq!(floor_log2(256), 8);
    }
}
