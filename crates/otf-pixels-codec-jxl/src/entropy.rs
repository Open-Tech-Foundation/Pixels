//! Entropy coding (ISO/IEC 18181-1 §C): the prefix-code form.
//!
//! A JPEG XL entropy-coded stream codes integers by *context*. Each value
//! becomes a token and some raw bits (a hybrid integer), contexts map to
//! clusters, and each cluster has its own code. The stream's header names
//! all of that; then the tokens follow, interleaved with nothing.
//!
//! This writes the prefix-code variant (Brotli's code format), which every
//! decoder must accept alongside ANS. LZ77 is not used, and contexts are not
//! clustered: each context is its own cluster, so a stream may use at most
//! eight, the most the simple context map can name.

#![allow(
    clippy::indexing_slicing,
    reason = "symbol indices are below the alphabet size the tables were \
              built for, and code-length symbols below 18"
)]

use crate::bits::{BitWriter, pack_signed};

/// How a value splits into a token and raw bits.
///
/// Values below `1 << split_exponent` are their own token. Above, the token
/// carries the value's magnitude, its `msb` bits below the leading one and
/// its `lsb` lowest bits; the bits between are raw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridUint {
    split_exponent: u32,
    msb: u32,
    lsb: u32,
}

impl HybridUint {
    /// libjxl's usual configuration for Modular residuals.
    pub const DEFAULT: Self = Self::new(4, 2, 0);

    /// Every value is a power-of-two bucket and its offset: tokens are tiny,
    /// which suits the few small values a tree stream holds.
    pub const BUCKETS: Self = Self::new(0, 0, 0);

    /// A configuration; `msb + lsb` must not exceed `split_exponent`.
    #[must_use]
    pub const fn new(split_exponent: u32, msb: u32, lsb: u32) -> Self {
        Self {
            split_exponent,
            msb,
            lsb,
        }
    }

    /// The token for `value`, and the raw bits that follow it, as
    /// `(token, count, bits)`.
    #[must_use]
    pub fn encode(&self, value: u32) -> (u32, u32, u32) {
        let split = 1_u32 << self.split_exponent;
        if value < split {
            return (value, 0, 0);
        }
        let n = 31 - value.leading_zeros();
        let m = value - (1 << n);
        let token = split
            + ((n - self.split_exponent) << (self.msb + self.lsb))
            + ((m >> (n - self.msb)) << self.lsb)
            + (m & ((1 << self.lsb) - 1));
        let count = n - self.msb - self.lsb;
        let bits = (value >> self.lsb) & ((1_u32 << count) - 1);
        (token, count, bits)
    }

    /// Write this configuration, for a code whose alphabet is
    /// `log_alphabet_size` bits (15 for prefix codes).
    fn write(&self, w: &mut BitWriter, log_alphabet_size: u32) {
        let ceil_log2_plus_one = |x: u32| 32 - x.leading_zeros();
        w.write(
            ceil_log2_plus_one(log_alphabet_size),
            u64::from(self.split_exponent),
        );
        if self.split_exponent != log_alphabet_size {
            w.write(ceil_log2_plus_one(self.split_exponent), u64::from(self.msb));
            w.write(
                ceil_log2_plus_one(self.split_exponent - self.msb),
                u64::from(self.lsb),
            );
        }
    }
}

/// One value to code, in a context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub ctx: u32,
    pub value: u32,
}

impl Token {
    /// A value.
    #[must_use]
    pub const fn new(ctx: u32, value: u32) -> Self {
        Self { ctx, value }
    }

    /// A signed value, packed.
    #[must_use]
    pub const fn signed(ctx: u32, value: i32) -> Self {
        Self::new(ctx, pack_signed(value))
    }
}

/// Code lengths for `freq`, no longer than `limit`, minimising the coded
/// size (package-merge). Unused symbols get length 0; a lone used symbol
/// gets length 1.
fn limited_lengths(freq: &[u64], limit: u32) -> Vec<u8> {
    let mut lengths = vec![0_u8; freq.len()];
    let mut used: Vec<(u64, usize)> = freq
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0)
        .map(|(s, &f)| (f, s))
        .collect();
    match used.len() {
        0 => return lengths,
        1 => {
            lengths[used[0].1] = 1;
            return lengths;
        }
        _ => {}
    }
    used.sort_unstable();
    // Each item is a weight and the symbols it covers.
    let leaves: Vec<(u64, Vec<usize>)> = used.iter().map(|&(f, s)| (f, vec![s])).collect();
    let mut list = leaves.clone();
    for _ in 1..limit {
        let packages: Vec<(u64, Vec<usize>)> = list
            .chunks_exact(2)
            .map(|pair| {
                let mut symbols = pair[0].1.clone();
                symbols.extend_from_slice(&pair[1].1);
                (pair[0].0 + pair[1].0, symbols)
            })
            .collect();
        let mut merged = Vec::with_capacity(leaves.len() + packages.len());
        let (mut a, mut b) = (leaves.iter().peekable(), packages.into_iter().peekable());
        loop {
            match (a.peek(), b.peek()) {
                (Some(x), Some(y)) if x.0 <= y.0 => {
                    merged.push(a.next().cloned().unwrap_or_default())
                }
                (Some(_), Some(_)) | (None, Some(_)) => merged.push(b.next().unwrap_or_default()),
                (Some(_), None) => merged.push(a.next().cloned().unwrap_or_default()),
                (None, None) => break,
            }
        }
        list = merged;
    }
    for (_, symbols) in list.iter().take(2 * used.len() - 2) {
        for &s in symbols {
            lengths[s] += 1;
        }
    }
    lengths
}

/// Canonical codes for `lengths`, bit-reversed for a least-significant-first
/// stream: codes are assigned in order of length, then of symbol.
fn canonical_codes(lengths: &[u8]) -> Vec<u16> {
    let mut count = [0_u16; 17];
    for &l in lengths {
        count[usize::from(l)] += 1;
    }
    count[0] = 0;
    let mut next = [0_u16; 17];
    let mut code = 0_u16;
    for len in 1..17 {
        code = (code + count[len - 1]) << 1;
        next[len] = code;
    }
    lengths
        .iter()
        .map(|&l| {
            if l == 0 {
                return 0;
            }
            let c = next[usize::from(l)];
            next[usize::from(l)] += 1;
            c.reverse_bits() >> (16 - u32::from(l))
        })
        .collect()
}

/// One cluster's prefix code over an alphabet of `alphabet` symbols.
#[derive(Debug, Clone)]
struct PrefixCode {
    alphabet: usize,
    lengths: Vec<u8>,
    codes: Vec<u16>,
}

/// The code lengths Brotli gives the 18 code-length symbols' own lengths
/// (0 to 5): `(bit count, bits)`.
const CODE_LENGTH_CODE: [(u32, u64); 6] = [(2, 0), (4, 7), (3, 3), (2, 2), (2, 1), (4, 15)];

/// The order code-length code lengths are written in.
const CODE_LENGTH_ORDER: [usize; 18] =
    [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];

impl PrefixCode {
    fn new(freq: &[u64]) -> Self {
        let alphabet = freq.iter().rposition(|&f| f > 0).map_or(1, |last| last + 1);
        let used = freq.iter().filter(|&&f| f > 0).count();
        let lengths = if used <= 1 {
            // A single symbol costs nothing to code.
            vec![0; alphabet]
        } else if used <= 4 {
            simple_lengths(&freq[..alphabet])
        } else {
            limited_lengths(&freq[..alphabet], 15)
        };
        let codes = canonical_codes(&lengths);
        Self {
            alphabet,
            lengths,
            codes,
        }
    }

    /// Write the alphabet size: 1, or `1 + 2^n + u(n)`.
    fn write_count(&self, w: &mut BitWriter) {
        if self.alphabet == 1 {
            w.write(1, 0);
            return;
        }
        let rest = (self.alphabet - 1) as u32;
        let n = 31 - rest.leading_zeros();
        w.write(1, 1);
        w.write(4, u64::from(n));
        w.write(n, u64::from(rest - (1 << n)));
    }

    /// Write the code itself (Brotli's simple or complex form).
    fn write(&self, w: &mut BitWriter) {
        if self.alphabet == 1 {
            return;
        }
        let used: Vec<usize> = (0..self.alphabet)
            .filter(|&s| self.lengths[s] > 0)
            .collect();
        let alphabet_bits = (self.alphabet as u32).next_power_of_two().trailing_zeros();
        if used.len() <= 4 {
            w.write(2, 1); // simple
            if used.is_empty() {
                // The only symbol is the last of the alphabet.
                w.write(2, 0);
                w.write(alphabet_bits, (self.alphabet - 1) as u64);
                return;
            }
            // Shortest codes first, as the format assigns lengths by
            // position.
            let mut order = used.clone();
            order.sort_by_key(|&s| (self.lengths[s], s));
            w.write(2, (order.len() - 1) as u64);
            for &s in &order {
                w.write(alphabet_bits, s as u64);
            }
            if order.len() == 4 {
                w.bool(self.lengths[order[0]] == 1);
            }
            return;
        }

        // Complex: the code lengths, coded by a code of their own.
        let last = used.last().copied().unwrap_or(0);
        let mut length_freq = [0_u64; 18];
        for &l in &self.lengths[..=last] {
            length_freq[usize::from(l)] += 1;
        }
        let length_lengths = limited_lengths(&length_freq, 5);
        let distinct = length_freq.iter().filter(|&&f| f > 0).count();
        let length_codes = canonical_codes(&length_lengths);
        w.write(2, 0); // hskip 0
        if distinct == 1 {
            // One length for every symbol: the decoder reads every entry,
            // then codes each length in no bits.
            for &s in &CODE_LENGTH_ORDER {
                let (count, bits) = CODE_LENGTH_CODE[usize::from(length_lengths[s])];
                w.write(count, bits);
            }
        } else {
            // Up to the last nonzero entry, where the code completes.
            let end = CODE_LENGTH_ORDER
                .iter()
                .rposition(|&s| length_lengths[s] > 0)
                .map_or(0, |i| i + 1);
            for &s in &CODE_LENGTH_ORDER[..end] {
                let (count, bits) = CODE_LENGTH_CODE[usize::from(length_lengths[s])];
                w.write(count, bits);
            }
        }
        if distinct > 1 {
            for &l in &self.lengths[..=last] {
                let l = usize::from(l);
                w.write(u32::from(length_lengths[l]), u64::from(length_codes[l]));
            }
        }
    }

    fn emit(&self, w: &mut BitWriter, symbol: u32) {
        let s = symbol as usize;
        w.write(u32::from(self.lengths[s]), u64::from(self.codes[s]));
    }
}

/// Lengths for 2 to 4 used symbols, in the shapes the simple form allows.
fn simple_lengths(freq: &[u64]) -> Vec<u8> {
    let mut lengths = vec![0_u8; freq.len()];
    let mut used: Vec<(u64, usize)> = freq
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0)
        .map(|(s, &f)| (f, s))
        .collect();
    used.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let shape: &[u8] = match used.len() {
        2 => &[1, 1],
        3 => &[1, 2, 2],
        _ => {
            // All two bits, or 1-2-3-3, whichever is smaller.
            let f: Vec<u64> = used.iter().map(|&(f, _)| f).collect();
            let flat = 2 * f.iter().sum::<u64>();
            let skewed = f[0] + 2 * f[1] + 3 * (f[2] + f[3]);
            if skewed < flat {
                &[1, 2, 3, 3]
            } else {
                &[2, 2, 2, 2]
            }
        }
    };
    for (&(_, s), &l) in used.iter().zip(shape) {
        lengths[s] = l;
    }
    lengths
}

/// A whole entropy-coded stream's codes: one per context.
#[derive(Debug, Clone)]
pub struct EntropyCode {
    config: HybridUint,
    codes: Vec<PrefixCode>,
}

impl EntropyCode {
    /// Codes for `num_ctx` contexts (at most 8), fitted to `tokens`.
    #[must_use]
    pub fn build(num_ctx: usize, tokens: &[Token], config: HybridUint) -> Self {
        let mut freq: Vec<Vec<u64>> = vec![Vec::new(); num_ctx];
        for token in tokens {
            let (symbol, _, _) = config.encode(token.value);
            let table = &mut freq[token.ctx as usize];
            if table.len() <= symbol as usize {
                table.resize(symbol as usize + 1, 0);
            }
            table[symbol as usize] += 1;
        }
        let codes = freq.iter().map(|f| PrefixCode::new(f)).collect();
        Self { config, codes }
    }

    /// Write the stream's header: no LZ77, the context map, and the codes.
    pub fn write_header(&self, w: &mut BitWriter) {
        w.bool(false); // no LZ77
        let num_ctx = self.codes.len();
        if num_ctx > 1 {
            // A simple map, each context its own cluster.
            let bits = (32 - ((num_ctx - 1) as u32).leading_zeros()).min(3);
            w.bool(true);
            w.write(2, u64::from(bits));
            for ctx in 0..num_ctx {
                w.write(bits, ctx as u64);
            }
        }
        w.bool(true); // prefix codes
        for _ in &self.codes {
            self.config.write(w, 15);
        }
        for code in &self.codes {
            code.write_count(w);
        }
        for code in &self.codes {
            code.write(w);
        }
    }

    /// Write one token.
    pub fn write(&self, w: &mut BitWriter, token: Token) {
        let (symbol, count, bits) = self.config.encode(token.value);
        self.codes[token.ctx as usize].emit(w, symbol);
        w.write(count, u64::from(bits));
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

    /// Decode `value` from a token and its bits, as the decoder does.
    fn decode(config: HybridUint, token: u32, bits: u32) -> u32 {
        let split = 1 << config.split_exponent;
        if token < split {
            return token;
        }
        let n = config.split_exponent - (config.msb + config.lsb)
            + ((token - split) >> (config.msb + config.lsb));
        let low = token & ((1 << config.lsb) - 1);
        let t = (token >> config.lsb) & ((1 << config.msb) - 1) | (1 << config.msb);
        (((t << n) | bits) << config.lsb) | low
    }

    #[test]
    fn hybrid_integers_round_trip_through_the_decoders_formula() {
        for config in [
            HybridUint::DEFAULT,
            HybridUint::BUCKETS,
            HybridUint::new(4, 1, 1),
            HybridUint::new(3, 0, 2),
        ] {
            for value in (0..5000).chain([65_535, 1 << 20, u32::MAX >> 1]) {
                let (token, count, bits) = config.encode(value);
                assert!(bits < (1 << count) || count == 0, "{value}");
                assert_eq!(decode(config, token, bits), value, "{config:?} {value}");
            }
        }
    }

    #[test]
    fn limited_lengths_are_complete_and_bounded() {
        let mut freq = vec![0_u64; 40];
        let (mut a, mut b) = (1_u64, 1_u64);
        for slot in &mut freq {
            *slot = a;
            (a, b) = (b, a + b);
        }
        for limit in [15, 8, 6] {
            let lengths = limited_lengths(&freq, limit);
            assert!(lengths.iter().all(|&l| l >= 1 && u32::from(l) <= limit));
            let kraft: f64 = lengths.iter().map(|&l| 0.5_f64.powi(i32::from(l))).sum();
            assert!((kraft - 1.0).abs() < 1e-9, "{limit}: {kraft}");
        }
    }

    #[test]
    fn canonical_codes_follow_length_then_symbol() {
        // Lengths 2,1,3,3: symbol 1 is "0", symbol 0 "10", 2 "110", 3 "111";
        // written least significant first, so reversed.
        let codes = canonical_codes(&[2, 1, 3, 3]);
        assert_eq!(codes, [0b01, 0b0, 0b011, 0b111]);
    }

    #[test]
    fn simple_shapes_follow_the_frequencies() {
        assert_eq!(simple_lengths(&[5, 0, 9]), [1, 0, 1]);
        assert_eq!(simple_lengths(&[1, 50, 2]), [2, 1, 2]);
        assert_eq!(simple_lengths(&[100, 1, 1, 1]), [1, 2, 3, 3]);
        assert_eq!(simple_lengths(&[5, 5, 5, 5]), [2, 2, 2, 2]);
    }
}
