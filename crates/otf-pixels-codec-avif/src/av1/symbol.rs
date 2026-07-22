//! The AV1 multi-symbol arithmetic decoder (spec §8.2.3–8.2.6).
//!
//! This is the entropy engine every tile symbol flows through. It is a range
//! decoder over 15-bit cumulative distribution functions, with the twist that
//! AV1 works with the *inverse* CDF (`f = 32768 - cdf[i]`) and gives every
//! symbol a floor probability of [`EC_MIN_PROB`] so the range never collapses.
//! Each `read_symbol` optionally adapts the CDF toward the symbol it just
//! decoded, which is why the CDF is passed by mutable reference.
//!
//! The transcription is taken verbatim from the AV1 specification's symbol
//! decoding process and cross-checked against libaom's `entdec.c`; the two
//! agree on the inverse-CDF arithmetic and on the renormalization
//! (`SymbolValue = paddedData ^ (((SymbolValue + 1) << bits) - 1)`). Numeric
//! conformance against real streams is established by the libaom differential
//! harness, which is the only trustworthy check for an entropy coder.
//!
//! Every read is fallible: a stream that runs out mid-renormalization is a
//! returned error, and `SymbolMaxBits` accounting means the padding-zero region
//! past the real bytes is entered deliberately, never by reading off the end.

use super::bits::{BitReader, floor_log2};
use otf_pixels_core::{PixelsError, Result};

/// Bits of CDF precision dropped during the range update (§3, `EC_PROB_SHIFT`).
const EC_PROB_SHIFT: u32 = 6;
/// The floor probability every symbol is guaranteed (§3, `EC_MIN_PROB`).
const EC_MIN_PROB: u32 = 4;

/// A range decoder over an AV1 tile's symbol data.
pub struct SymbolDecoder<'a> {
    reader: BitReader<'a>,
    /// `SymbolValue` — the decoded position within the current range, stored in
    /// the spec's inverted form.
    value: u32,
    /// `SymbolRange` — the current range, always in `[2^15, 2^16)`.
    range: u32,
    /// `SymbolMaxBits` — real bits still available. Goes negative once the
    /// decoder enters the implicit zero-padding past the end of the data.
    max_bits: i64,
    /// Whether per-symbol CDF adaptation is switched off for the frame.
    disable_cdf_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// Initialise the decoder over a tile's `sz`-byte symbol partition
    /// (`init_symbol`, §8.2.2). The position must be byte-aligned, which the
    /// syntax guarantees at every call site.
    pub fn new(data: &'a [u8], disable_cdf_update: bool) -> Result<Self> {
        let sz = data.len();
        let mut reader = BitReader::new(data);
        let num_bits = u32::try_from(sz.saturating_mul(8).min(15)).unwrap_or(15);
        let buf = reader.f(num_bits)?;
        let padded_buf = buf << (15 - num_bits);
        let value = ((1_u32 << 15) - 1) ^ padded_buf;
        let max_bits = (8 * sz as i64) - 15;
        Ok(Self {
            reader,
            value,
            range: 1 << 15,
            max_bits,
            disable_cdf_update,
        })
    }

    /// Decode one symbol against `cdf`, adapting it unless updates are disabled
    /// (`read_symbol`, §8.2.6). `cdf` has `N + 1` entries: `N` cumulative
    /// frequencies with `cdf[N-1] == 1 << 15`, then an adaptation counter.
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<usize> {
        let len = cdf.len();
        let Some(n) = len.checked_sub(1).filter(|&n| n >= 1) else {
            return Err(PixelsError::malformed(
                "avif",
                "an AV1 CDF must hold at least one symbol and a counter",
            ));
        };

        // decode_symbol: walk the inverse CDF until SymbolValue lands in a
        // symbol's interval. cur decreases as the symbol index rises, so the
        // first index whose cur is at or below the value is the answer.
        let mut cur = self.range;
        let mut prev = cur;
        let mut symbol = n - 1;
        for (k, &c) in cdf.iter().take(n).enumerate() {
            prev = cur;
            let f = (1_u32 << 15) - u32::from(c);
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n as u32 - 1 - k as u32);
            if self.value >= cur {
                symbol = k;
                break;
            }
        }

        self.range = prev - cur;
        self.value -= cur;
        self.renormalize()?;

        if !self.disable_cdf_update {
            update_cdf(cdf, symbol, n);
        }
        Ok(symbol)
    }

    /// Renormalize the range and refill the value with new bits (§8.2.6 steps
    /// 1–7). `bits` new bits enter; past the real data they are implicit zeros.
    fn renormalize(&mut self) -> Result<()> {
        let bits = 15 - floor_log2(self.range);
        self.range <<= bits;
        let available = self.max_bits.max(0);
        let num_bits = if i64::from(bits) < available {
            bits
        } else {
            // available < bits <= 15, so it fits in u32.
            available as u32
        };
        let new_data = self.reader.f(num_bits)?;
        let padded_data = new_data << (bits - num_bits);
        self.value = padded_data ^ (((self.value + 1) << bits) - 1);
        self.max_bits -= i64::from(bits);
        Ok(())
    }

    /// Decode one equiprobable bit (`read_bool`, §8.2.3). The transient CDF's
    /// adaptation is never observed, so it is skipped.
    pub fn read_bool(&mut self) -> Result<bool> {
        let mut cdf = [1_u16 << 14, 1_u16 << 15, 0];
        let saved = self.disable_cdf_update;
        self.disable_cdf_update = true;
        let symbol = self.read_symbol(&mut cdf);
        self.disable_cdf_update = saved;
        Ok(symbol? != 0)
    }

    /// Decode an `n`-bit literal, most-significant bit first (`read_literal`,
    /// §8.2.5).
    pub fn read_literal(&mut self, n: u32) -> Result<u32> {
        let mut x = 0;
        for _ in 0..n {
            x = 2 * x + u32::from(self.read_bool()?);
        }
        Ok(x)
    }

    /// Decode a non-symmetric `NS(n)` value in `0..n` (`read_ns`, §8.2.4). Used
    /// for the palette colour-index map's first sample, which is uniform over
    /// the palette size.
    pub fn read_ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = floor_log2(n) + 1;
        let m = (1 << w) - n;
        let v = self.read_literal(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.read_literal(1)?;
        Ok((v << 1) - m + extra)
    }

    /// The number of real bits still available (may be negative once the
    /// decoder is in the padding region). Exposed for the tile-exit checks.
    #[must_use]
    pub fn max_bits(&self) -> i64 {
        self.max_bits
    }
}

/// Adapt `cdf` toward `symbol` (`update_cdf`, §8.2.6). The counter at `cdf[n]`
/// slows adaptation as a symbol is seen more often.
fn update_cdf(cdf: &mut [u16], symbol: usize, n: usize) {
    let count = cdf.get(n).copied().unwrap_or(0);
    let rate = 3 + u32::from(count > 15) + u32::from(count > 31) + floor_log2(n as u32).min(2);
    let mut tmp: u32 = 0;
    for (i, slot) in cdf.iter_mut().take(n.saturating_sub(1)).enumerate() {
        if i == symbol {
            tmp = 1 << 15;
        }
        let ci = u32::from(*slot);
        let updated = if tmp < ci {
            ci - ((ci - tmp) >> rate)
        } else {
            ci + ((tmp - ci) >> rate)
        };
        // updated stays within [0, 1<<15], so the cast never truncates.
        *slot = updated as u16;
    }
    if let Some(counter) = cdf.get_mut(n) {
        if *counter < 32 {
            *counter += 1;
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// A binary CDF: `cdf[0]` is P(symbol 0), then the mandatory `1<<15` and the
    /// adaptation counter.
    fn cdf_binary(c0: u16) -> [u16; 3] {
        [c0, 1 << 15, 0]
    }

    /// A genuine three-symbol CDF (two cumulative splits + counter).
    fn cdf3(c0: u16, c1: u16) -> [u16; 4] {
        [c0, c1, 1 << 15, 0]
    }

    #[test]
    fn init_state_matches_the_spec() {
        let data = [0xAB, 0xCD, 0xEF];
        let dec = SymbolDecoder::new(&data, false).unwrap();
        assert_eq!(dec.range, 1 << 15);
        // numBits = min(24,15) = 15; buf = top 15 bits of 0xABCD. paddedBuf is
        // buf << (15 - 15) = buf.
        let padded_buf = u32::from(0xABCD_u16) >> 1;
        assert_eq!(dec.value, ((1 << 15) - 1) ^ padded_buf);
        assert_eq!(dec.max_bits, 8 * 3 - 15);
    }

    #[test]
    fn a_tiny_buffer_starts_in_the_padding_region() {
        // One byte: only 8 real bits, so SymbolMaxBits is negative from the
        // start and renormalization reads no further real bits.
        let dec = SymbolDecoder::new(&[0x00], false).unwrap();
        assert_eq!(dec.max_bits, 8 - 15);
    }

    #[test]
    fn a_cdf_certain_of_the_first_symbol_decodes_it_when_the_value_is_high() {
        // An all-zero buffer makes init's SymbolValue = 0x7FFF ^ 0 = 0x7FFF,
        // which sits in the high interval. With cdf[0] = 32767 the low-value
        // sliver belongs to symbol 1, so a high value pins the result to 0, and
        // the all-zero stream keeps it there.
        let mut dec = SymbolDecoder::new(&[0x00; 6], true).unwrap();
        for _ in 0..8 {
            assert_eq!(dec.read_symbol(&mut cdf_binary(32767)).unwrap(), 0);
        }
    }

    #[test]
    fn a_cdf_certain_of_the_last_symbol_decodes_it_when_the_value_is_low() {
        // An all-0xFF buffer makes init's SymbolValue = 0x7FFF ^ 0x7FFF = 0,
        // the lowest value. With cdf[0] = 1 almost the entire range belongs to
        // symbol 1, and value 0 lands squarely in it.
        let mut dec = SymbolDecoder::new(&[0xFF; 6], true).unwrap();
        for _ in 0..8 {
            assert_eq!(dec.read_symbol(&mut cdf_binary(1)).unwrap(), 1);
        }
    }

    #[test]
    fn read_literal_composes_read_bool() {
        // read_literal(n) must equal n read_bool calls, MSB first, on the same
        // stream. Run each on its own decoder over identical data.
        let data = [0x3C, 0xA7, 0x91, 0x08, 0x55];
        let mut a = SymbolDecoder::new(&data, false).unwrap();
        let literal = a.read_literal(5).unwrap();

        let mut b = SymbolDecoder::new(&data, false).unwrap();
        let mut composed = 0;
        for _ in 0..5 {
            composed = 2 * composed + u32::from(b.read_bool().unwrap());
        }
        assert_eq!(literal, composed);
    }

    #[test]
    fn decoding_is_deterministic_for_the_same_input() {
        let data = [0x9E, 0x42, 0x17, 0xCB, 0x30, 0x8A];
        let decode_all = || {
            let mut dec = SymbolDecoder::new(&data, false).unwrap();
            let mut out = Vec::new();
            for _ in 0..12 {
                out.push(dec.read_symbol(&mut cdf3(1 << 13, 3 << 13)).unwrap());
            }
            out
        };
        assert_eq!(decode_all(), decode_all());
    }

    #[test]
    fn adaptation_moves_the_cdf_toward_the_decoded_symbol() {
        // A high initial value (zero buffer) decodes symbol 0 here; confirm
        // cdf[0] climbs toward 1<<15 as that symbol is reinforced.
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut dec = SymbolDecoder::new(&data, false).unwrap();
        let mut cdf = cdf_binary(1 << 14);
        let before = cdf[0];
        let symbol = dec.read_symbol(&mut cdf).unwrap();
        assert_eq!(symbol, 0);
        // Seeing symbol 0 pushes cdf[0] upward (toward certainty of 0).
        assert!(cdf[0] > before, "{} !> {}", cdf[0], before);
        // The counter advanced.
        assert_eq!(cdf[2], 1);
    }

    #[test]
    fn a_malformed_cdf_is_rejected_not_panicked() {
        let mut dec = SymbolDecoder::new(&[0x00, 0x11], false).unwrap();
        let mut too_short = [1_u16 << 15];
        assert!(dec.read_symbol(&mut too_short).is_err());
    }
}
