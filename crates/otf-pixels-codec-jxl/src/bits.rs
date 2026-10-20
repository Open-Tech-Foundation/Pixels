//! The JPEG XL bit writer and the field encodings headers are built from
//! (ISO/IEC 18181-1 §B, "Bit stream").
//!
//! JPEG XL packs bits least significant first: the first bit written is bit
//! 0 of the first byte. Fields are a few fixed shapes — `u(n)`, `Bool`,
//! `U32` (a 2-bit selector choosing one of four distributions), `U64`, `F16`
//! and `Enum` — and every header is a sequence of them.

/// A distribution a `U32` field's selector can choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dist {
    /// Exactly this value, with no further bits.
    Value(u32),
    /// `offset + u(bits)`.
    Bits(u32, u32),
}

impl Dist {
    /// The bits that code `value` under this distribution, as `(count,
    /// value)`, if it can.
    const fn code(self, value: u32) -> Option<(u32, u32)> {
        match self {
            Self::Value(v) => {
                if v == value {
                    Some((0, 0))
                } else {
                    None
                }
            }
            Self::Bits(offset, bits) => {
                if value < offset {
                    return None;
                }
                let rest = value - offset;
                if bits < 32 && rest >> bits != 0 {
                    return None;
                }
                Some((bits, rest))
            }
        }
    }
}

/// Shorthand for [`Dist::Value`].
pub const fn val(v: u32) -> Dist {
    Dist::Value(v)
}

/// Shorthand for [`Dist::Bits`].
pub const fn bits(offset: u32, bits: u32) -> Dist {
    Dist::Bits(offset, bits)
}

/// The distributions of an `Enum` field.
pub const ENUM: [Dist; 4] = [val(0), val(1), bits(2, 4), bits(18, 6)];

/// Accumulates bits, least significant first.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Bits not yet a whole byte, low bits first.
    pending: u64,
    pending_bits: u32,
}

impl BitWriter {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the low `count` bits of `value` (`count` at most 56).
    pub fn write(&mut self, count: u32, value: u64) {
        debug_assert!(count <= 56);
        if count == 0 {
            return;
        }
        let masked = if count >= 64 {
            value
        } else {
            value & ((1_u64 << count) - 1)
        };
        self.pending |= masked << self.pending_bits;
        self.pending_bits += count;
        while self.pending_bits >= 8 {
            self.bytes.push(self.pending as u8);
            self.pending >>= 8;
            self.pending_bits -= 8;
        }
    }

    /// Append a `Bool`.
    pub fn bool(&mut self, value: bool) {
        self.write(1, u64::from(value));
    }

    /// Append a `U32` with the first of `dists` that can hold `value`.
    ///
    /// # Panics
    ///
    /// Never in practice: callers pass values their field's distributions
    /// hold. In debug builds an unrepresentable value asserts.
    pub fn u32(&mut self, value: u32, dists: [Dist; 4]) {
        for (selector, dist) in dists.iter().enumerate() {
            if let Some((count, rest)) = dist.code(value) {
                self.write(2, selector as u64);
                self.write(count, u64::from(rest));
                return;
            }
        }
        debug_assert!(false, "{value} fits none of {dists:?}");
    }

    /// Append an `Enum`.
    pub fn enumeration(&mut self, value: u32) {
        self.u32(value, ENUM);
    }

    /// Append a `U64`.
    pub fn u64(&mut self, value: u64) {
        match value {
            0 => self.write(2, 0),
            1..=16 => {
                self.write(2, 1);
                self.write(4, value - 1);
            }
            17..=272 => {
                self.write(2, 2);
                self.write(8, value - 17);
            }
            _ => {
                self.write(2, 3);
                self.write(12, value & 0xFFF);
                let mut rest = value >> 12;
                let mut shift = 12;
                while rest != 0 {
                    self.write(1, 1);
                    if shift == 60 {
                        self.write(4, rest & 0xF);
                        return;
                    }
                    self.write(8, rest & 0xFF);
                    rest >>= 8;
                    shift += 8;
                }
                self.write(1, 0);
            }
        }
    }

    /// Pad with zero bits to a byte boundary.
    pub fn zero_pad_to_byte(&mut self) {
        if self.pending_bits > 0 {
            self.write(8 - self.pending_bits, 0);
        }
    }

    /// Bits written so far.
    #[must_use]
    #[cfg(test)]
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.pending_bits as usize
    }

    /// Append another writer's bits, unaligned.
    pub fn append(&mut self, other: &Self) {
        if self.pending_bits == 0 {
            self.bytes.extend_from_slice(&other.bytes);
        } else {
            for &byte in &other.bytes {
                self.write(8, u64::from(byte));
            }
        }
        self.write(other.pending_bits, other.pending);
    }

    /// The bytes, padded with zero bits to a whole byte.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.zero_pad_to_byte();
        self.bytes
    }
}

/// `PackSigned`: 0, -1, 1, -2, 2, ... as 0, 1, 2, 3, 4, ...
#[must_use]
pub const fn pack_signed(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        ((-(value as i64) as u32) << 1) - 1
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;

    #[test]
    fn bits_go_least_significant_first() {
        let mut w = BitWriter::new();
        w.write(1, 1);
        w.write(3, 0b010);
        w.write(4, 0b1111);
        w.write(2, 0b11);
        assert_eq!(w.finish(), [0b1111_0101, 0b11]);
    }

    #[test]
    fn the_signature_is_ff_0a() {
        let mut w = BitWriter::new();
        w.write(16, 0x0AFF);
        assert_eq!(w.finish(), [0xFF, 0x0A]);
    }

    #[test]
    fn u32_picks_the_first_distribution_that_fits() {
        // An Enum: 13 is 2 + u(4) = 11, selector 2.
        let mut w = BitWriter::new();
        w.enumeration(13);
        assert_eq!(w.finish(), [0b10 | (11 << 2)]);
        let mut w = BitWriter::new();
        w.u32(1, ENUM);
        assert_eq!(w.finish(), [0b01]);
    }

    #[test]
    fn u64_takes_each_shape() {
        for (value, expected_bits) in [(0_u64, 2), (5, 6), (200, 10), (4096, 2 + 12 + 1 + 8 + 1)] {
            let mut w = BitWriter::new();
            w.u64(value);
            assert_eq!(w.bit_len(), expected_bits, "{value}");
        }
    }

    #[test]
    fn signed_values_interleave() {
        let packed: Vec<u32> = [0, -1, 1, -2, 2, i32::MIN + 1]
            .iter()
            .map(|&v| pack_signed(v))
            .collect();
        assert_eq!(&packed[..5], [0, 1, 2, 3, 4]);
        assert_eq!(packed[5], u32::MAX - 2);
    }
}
