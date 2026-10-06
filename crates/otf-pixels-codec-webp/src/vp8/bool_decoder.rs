//! The VP8 boolean entropy decoder (RFC 6386 §7.3).

/// Bytes the decoder may read past its data, as zeros, before the stream is
/// called truncated. The arithmetic coder's two-byte window runs ahead of the
/// symbols it has decoded, so a complete stream can touch the first byte or
/// two beyond its end; a truncated one runs much further.
const OVERRUN_SLACK: usize = 2;

/// Decodes booleans, each with an 8-bit probability of being 0.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    /// Next byte of `data` to shift in.
    position: usize,
    /// Zero bytes shifted in past the end.
    overrun: usize,
    range: u32,
    value: u32,
    bit_count: u32,
}

impl<'a> BoolDecoder<'a> {
    /// Start decoding `data`.
    pub fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            data,
            position: 0,
            overrun: 0,
            range: 255,
            value: 0,
            bit_count: 0,
        };
        decoder.value = (decoder.next_byte() << 8) | decoder.next_byte();
        decoder
    }

    fn next_byte(&mut self) -> u32 {
        if let Some(&byte) = self.data.get(self.position) {
            self.position += 1;
            u32::from(byte)
        } else {
            self.overrun += 1;
            0
        }
    }

    /// Whether the decoder has read so far past its data that the stream must
    /// have been cut short.
    pub const fn exhausted(&self) -> bool {
        self.overrun > OVERRUN_SLACK
    }

    /// One boolean, 0 with probability `probability / 256`.
    pub fn read(&mut self, probability: u8) -> bool {
        let split = 1 + (((self.range - 1) * u32::from(probability)) >> 8);
        let big_split = split << 8;
        let bit = if self.value >= big_split {
            self.range -= split;
            self.value -= big_split;
            true
        } else {
            self.range = split;
            false
        };
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.bit_count += 1;
            if self.bit_count == 8 {
                self.bit_count = 0;
                self.value |= self.next_byte();
            }
        }
        bit
    }

    /// An `n`-bit unsigned literal, most significant bit first.
    pub fn literal(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |acc, _| (acc << 1) | u32::from(self.read(128)))
    }

    /// A flag, then if set an `n`-bit magnitude and a sign.
    pub fn maybe_signed(&mut self, n: u32) -> i32 {
        if !self.read(128) {
            return 0;
        }
        let magnitude = self.literal(n) as i32;
        if self.read(128) {
            -magnitude
        } else {
            magnitude
        }
    }

    /// A value coded with `tree` (RFC 6386 §8.1): non-negative entries are
    /// the index of the next node pair, others are a negated leaf.
    pub fn tree(&mut self, tree: &[i8], probs: &[u8]) -> u8 {
        let mut i = 0_i8;
        loop {
            let bit = usize::from(self.read(probs.get(i as usize >> 1).copied().unwrap_or(128)));
            i = tree.get(i as usize + bit).copied().unwrap_or(0);
            if i <= 0 {
                return i.unsigned_abs();
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    #[test]
    fn an_all_zero_stream_decodes_zeros_until_it_runs_out() {
        let mut d = BoolDecoder::new(&[0, 0, 0, 0]);
        assert_eq!(d.literal(16), 0);
        assert!(!d.exhausted());
        for _ in 0..64 {
            d.read(128);
        }
        assert!(d.exhausted());
    }

    #[test]
    fn literals_read_most_significant_bit_first() {
        // With probability 128 each bit splits the range evenly, so the
        // stream's own bits come back out: 0xa5 = 1010_0101.
        let mut d = BoolDecoder::new(&[0xa5, 0x00, 0x00]);
        assert_eq!(d.literal(8), 0xa5);
    }
}
