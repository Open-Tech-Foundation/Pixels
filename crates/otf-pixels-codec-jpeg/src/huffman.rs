//! Canonical Huffman tables, built from the counts-and-values form `DHT`
//! carries.
//!
//! JPEG never transmits codes. It transmits how many codes exist of each
//! length 1..=16, then the symbols in code order, and both ends reconstruct
//! the same canonical assignment from that. Decoding therefore needs no tree:
//! codes of a given length occupy a contiguous numeric range, so a code is
//! recognised by comparing it against that length's maximum.

use otf_pixels_core::{PixelsError, Result};

/// The number of leading bits the direct lookup table covers.
///
/// Eight is the usual choice: it catches the overwhelming majority of real
/// codes in one array index, and costs 512 bytes per table.
const LOOKUP_BITS: u32 = 8;

/// A decoding table for one Huffman code set.
#[derive(Debug, Clone)]
pub struct HuffmanTable {
    /// `(length << 8) | symbol` for every [`LOOKUP_BITS`]-bit prefix that
    /// resolves within that many bits; `0` where it does not.
    lookup: [u16; 1 << LOOKUP_BITS],
    /// The smallest code of each length, indexed by length.
    mincode: [i32; 17],
    /// The largest code of each length, or `-1` if there are none.
    maxcode: [i32; 17],
    /// Index into [`HuffmanTable::values`] of the first symbol of each length.
    valptr: [i32; 17],
    /// Symbols in code order.
    values: Vec<u8>,
}

impl HuffmanTable {
    /// Build a table from a `DHT` segment's counts and symbols.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the counts and symbol list
    /// disagree, or if the code set is over-subscribed — more codes of some
    /// length than that length can represent, which describes no prefix code
    /// and would otherwise decode into arbitrary symbols.
    pub fn new(counts: &[u8; 16], values: Vec<u8>) -> Result<Self> {
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        if total != values.len() {
            return Err(PixelsError::malformed(
                "jpeg",
                format!(
                    "Huffman table declares {total} codes but carries {} symbols",
                    values.len()
                ),
            ));
        }
        if total == 0 {
            return Err(PixelsError::malformed("jpeg", "Huffman table is empty"));
        }

        let mut mincode = [0_i32; 17];
        let mut maxcode = [-1_i32; 17];
        let mut valptr = [0_i32; 17];
        let mut lookup = [0_u16; 1 << LOOKUP_BITS];

        let mut code: i32 = 0;
        let mut index: i32 = 0;
        for length in 1..=16_usize {
            let count = i32::from(*counts.get(length - 1).unwrap_or(&0));
            if let (Some(min), Some(max), Some(ptr)) = (
                mincode.get_mut(length),
                maxcode.get_mut(length),
                valptr.get_mut(length),
            ) {
                *min = code;
                *ptr = index;
                // `maxcode` stays -1 for a length with no codes, so the
                // `code <= maxcode` test can never match at that length.
                *max = if count > 0 { code + count - 1 } else { -1 };
            }

            // Fill the direct-lookup entries for the short codes.
            if length <= LOOKUP_BITS as usize {
                let shift = LOOKUP_BITS as usize - length;
                for step in 0..count {
                    let Some(&symbol) = values.get((index + step) as usize) else {
                        break;
                    };
                    let prefix = ((code + step) as usize) << shift;
                    for slot in prefix..prefix + (1 << shift) {
                        if let Some(entry) = lookup.get_mut(slot) {
                            *entry = ((length as u16) << 8) | u16::from(symbol);
                        }
                    }
                }
            }

            code += count;
            index += count;
            if code > (1 << length) {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("Huffman table is over-subscribed at code length {length}"),
                ));
            }
            code <<= 1;
        }

        Ok(Self {
            lookup,
            mincode,
            maxcode,
            valptr,
            values,
        })
    }

    /// The symbol for an already-resolved short code, given the next
    /// [`LOOKUP_BITS`] bits of the stream.
    ///
    /// Returns the code's length and its symbol, or `None` if the code is
    /// longer than the table covers.
    #[must_use]
    pub fn lookup(&self, prefix: u8) -> Option<(u32, u8)> {
        match self.lookup.get(prefix as usize).copied().unwrap_or(0) {
            0 => None,
            entry => Some((u32::from(entry >> 8), (entry & 0xFF) as u8)),
        }
    }

    /// The symbol for `code`, which is `length` bits long, if one exists.
    #[must_use]
    pub fn resolve(&self, length: usize, code: i32) -> Option<u8> {
        let max = *self.maxcode.get(length)?;
        if max < 0 || code > max {
            return None;
        }
        let min = *self.mincode.get(length)?;
        let base = *self.valptr.get(length)?;
        let at = base.checked_add(code.checked_sub(min)?)?;
        self.values.get(usize::try_from(at).ok()?).copied()
    }

    /// The longest code length this table defines, for bounding a decode loop.
    #[must_use]
    pub fn max_length(&self) -> usize {
        (1..=16)
            .rev()
            .find(|&l| self.maxcode.get(l).is_some_and(|&m| m >= 0))
            .unwrap_or(16)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::ErrorCode;

    /// Three symbols with code lengths 1, 2, 3: codes `0`, `10`, `110`.
    fn simple() -> HuffmanTable {
        let mut counts = [0_u8; 16];
        counts[0] = 1;
        counts[1] = 1;
        counts[2] = 1;
        HuffmanTable::new(&counts, vec![b'a', b'b', b'c']).unwrap()
    }

    #[test]
    fn canonical_codes_resolve_to_their_symbols() {
        let table = simple();
        assert_eq!(table.resolve(1, 0b0), Some(b'a'));
        assert_eq!(table.resolve(2, 0b10), Some(b'b'));
        assert_eq!(table.resolve(3, 0b110), Some(b'c'));
        // `111` is the unused code point; nothing decodes to it.
        assert_eq!(table.resolve(3, 0b111), None);
        // A length with no codes never matches, whatever the code.
        assert_eq!(table.resolve(4, 0), None);
    }

    #[test]
    fn the_lookup_table_agrees_with_the_slow_path() {
        let table = simple();
        // `0xxxxxxx` is 'a' in one bit.
        assert_eq!(table.lookup(0b0000_0000), Some((1, b'a')));
        assert_eq!(table.lookup(0b0111_1111), Some((1, b'a')));
        // `10xxxxxx` is 'b' in two bits.
        assert_eq!(table.lookup(0b1011_1111), Some((2, b'b')));
        // `110xxxxx` is 'c' in three.
        assert_eq!(table.lookup(0b1101_0101), Some((3, b'c')));
        // `111xxxxx` is the unused code point: no entry.
        assert_eq!(table.lookup(0b1110_0000), None);
    }

    #[test]
    fn long_codes_fall_out_of_the_lookup_table() {
        // One symbol at each length 1..=16 — the deepest legal table.
        let counts = [1_u8; 16];
        let values: Vec<u8> = (0..16).collect();
        let table = HuffmanTable::new(&counts, values).unwrap();
        assert_eq!(table.max_length(), 16);
        // Length 9 is past LOOKUP_BITS, so it must resolve the slow way.
        // Codes are 0, 10, 110, ... so the 9-bit code is 0b111111110.
        assert_eq!(table.resolve(9, 0b1_1111_1110), Some(8));
        assert_eq!(table.lookup(0b1111_1111), None);
    }

    #[test]
    fn mismatched_counts_and_symbols_are_malformed() {
        let mut counts = [0_u8; 16];
        counts[0] = 2;
        assert_eq!(
            HuffmanTable::new(&counts, vec![b'a']).unwrap_err().code(),
            ErrorCode::Malformed
        );
        assert_eq!(
            HuffmanTable::new(&[0; 16], vec![]).unwrap_err().code(),
            ErrorCode::Malformed
        );
    }

    #[test]
    fn over_subscribed_tables_are_rejected() {
        // Three codes of length 1, where only two exist. Accepting this would
        // let a crafted table decode past the end of its symbol list.
        let mut counts = [0_u8; 16];
        counts[0] = 3;
        assert_eq!(
            HuffmanTable::new(&counts, vec![b'a', b'b', b'c'])
                .unwrap_err()
                .code(),
            ErrorCode::Malformed
        );
    }

    #[test]
    fn a_full_table_of_short_codes_is_accepted() {
        // 256 codes of length 8 exactly fills the code space — legal, and the
        // boundary the over-subscription check must not trip on.
        let mut counts = [0_u8; 16];
        counts[7] = 255;
        let values: Vec<u8> = (0..255).collect();
        let table = HuffmanTable::new(&counts, values).unwrap();
        assert_eq!(table.lookup(0), Some((8, 0)));
        assert_eq!(table.lookup(254), Some((8, 254)));
    }
}
