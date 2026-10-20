//! The bit reader the decoder parses with: least significant bit first, as
//! the codestream is written (ISO/IEC 18181-1 §B), with the same field
//! encodings as [`crate::bits::BitWriter`]. Running past the end is an
//! error, never a panic: input is untrusted.

use otf_pixels_core::{PixelsError, Result};

use crate::bits::Dist;

/// The error for input that ends too soon or breaks a rule.
pub fn malformed(detail: impl Into<String>) -> PixelsError {
    PixelsError::malformed("jxl", detail.into())
}

/// Reads bits from a byte slice.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position of the next unread bit.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader at the start of `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Bits read so far.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Bits left.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    /// The `count` (at most 32) next bits, without consuming them; bits past
    /// the end read as zero.
    #[must_use]
    pub fn peek(&self, count: u32) -> u32 {
        let mut value = 0_u64;
        let byte = self.pos / 8;
        for i in 0..6 {
            if let Some(&b) = self.data.get(byte + i) {
                value |= u64::from(b) << (8 * i);
            }
        }
        let value = value >> (self.pos % 8);
        if count == 0 {
            0
        } else {
            (value & ((1_u64 << count) - 1)) as u32
        }
    }

    /// Consume `count` bits.
    ///
    /// # Errors
    ///
    /// If fewer than `count` remain.
    pub fn skip(&mut self, count: usize) -> Result<()> {
        if count > self.remaining() {
            return Err(malformed("the codestream ends early"));
        }
        self.pos += count;
        Ok(())
    }

    /// Read `count` bits, at most 32.
    ///
    /// # Errors
    ///
    /// If fewer than `count` remain.
    pub fn read(&mut self, count: u32) -> Result<u32> {
        let value = self.peek(count);
        self.skip(count as usize)?;
        Ok(value)
    }

    /// Read a `Bool`.
    ///
    /// # Errors
    ///
    /// At the end of input.
    pub fn bool(&mut self) -> Result<bool> {
        Ok(self.read(1)? == 1)
    }

    /// Read a `U32` with distributions `dists`.
    ///
    /// # Errors
    ///
    /// At the end of input.
    pub fn u32(&mut self, dists: [Dist; 4]) -> Result<u32> {
        let selector = self.read(2)? as usize;
        match dists.get(selector).copied() {
            Some(Dist::Value(v)) => Ok(v),
            Some(Dist::Bits(offset, bits)) => Ok(self.read(bits)?.wrapping_add(offset)),
            None => Err(malformed("bad U32 selector")),
        }
    }

    /// Read an `Enum`.
    ///
    /// # Errors
    ///
    /// At the end of input.
    pub fn enumeration(&mut self) -> Result<u32> {
        self.u32(crate::bits::ENUM)
    }

    /// Read a `U64`.
    ///
    /// # Errors
    ///
    /// At the end of input.
    pub fn u64(&mut self) -> Result<u64> {
        Ok(match self.read(2)? {
            0 => 0,
            1 => u64::from(self.read(4)?) + 1,
            2 => u64::from(self.read(8)?) + 17,
            _ => {
                let mut value = u64::from(self.read(12)?);
                let mut shift = 12;
                while self.bool()? {
                    if shift == 60 {
                        value |= u64::from(self.read(4)?) << shift;
                        break;
                    }
                    value |= u64::from(self.read(8)?) << shift;
                    shift += 8;
                }
                value
            }
        })
    }

    /// Read an `F16` as an `f32`.
    ///
    /// # Errors
    ///
    /// At the end of input, or for an infinity or NaN.
    pub fn f16(&mut self) -> Result<f32> {
        let v = self.read(16)?;
        let sign = (v & 0x8000) << 16;
        let exponent = (v >> 10) & 0x1F;
        let mantissa = v & 0x3FF;
        if v & 0x7FFF == 0 {
            return Ok(f32::from_bits(sign));
        }
        if exponent == 0x1F {
            return Err(malformed("an F16 is not finite"));
        }
        if exponent == 0 {
            let value = mantissa as f32 / 1024.0 / 16_384.0;
            return Ok(if sign != 0 { -value } else { value });
        }
        Ok(f32::from_bits(
            sign | ((exponent + 112) << 23) | (mantissa << 13),
        ))
    }

    /// Skip to the next byte boundary, which must be zero padding.
    ///
    /// # Errors
    ///
    /// If the padding bits are not zero.
    pub fn zero_pad_to_byte(&mut self) -> Result<()> {
        let pad = (8 - self.pos % 8) % 8;
        if self.read(pad as u32)? != 0 {
            return Err(malformed("padding bits are not zero"));
        }
        Ok(())
    }
}

/// `UnpackSigned`, the inverse of [`crate::bits::pack_signed`].
#[must_use]
pub const fn unpack_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value >> 1) as i32
    } else {
        -((value >> 1) as i32) - 1
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;
    use crate::bits::{BitWriter, ENUM, bits, pack_signed, val};

    #[test]
    fn everything_the_writer_writes_reads_back() {
        let mut w = BitWriter::new();
        w.write(5, 19);
        w.bool(true);
        w.u32(300, [val(0), bits(1, 4), bits(17, 8), bits(273, 12)]);
        w.enumeration(13);
        for v in [0_u64, 7, 200, 1 << 40] {
            w.u64(v);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read(5).unwrap(), 19);
        assert!(r.bool().unwrap());
        assert_eq!(
            r.u32([val(0), bits(1, 4), bits(17, 8), bits(273, 12)])
                .unwrap(),
            300
        );
        assert_eq!(r.u32(ENUM).unwrap(), 13);
        for v in [0_u64, 7, 200, 1 << 40] {
            assert_eq!(r.u64().unwrap(), v);
        }
        assert!(r.read(32).is_err(), "past the end is an error");
    }

    #[test]
    fn f16_and_signed_values_decode() {
        let mut w = BitWriter::new();
        w.write(16, 0x3C00); // 1.0
        w.write(16, 0xC000); // -2.0
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.f16().unwrap(), 1.0);
        assert_eq!(r.f16().unwrap(), -2.0);
        for v in [0, -1, 1, -1000, 77] {
            assert_eq!(unpack_signed(pack_signed(v)), v);
        }
    }
}
