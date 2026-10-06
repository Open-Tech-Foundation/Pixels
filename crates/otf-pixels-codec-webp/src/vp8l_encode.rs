//! VP8L encoding: lossless WebP, and the compressed form of an `ALPH` chunk.
//!
//! The encoder chooses between two shapes. An image of 256 colours or fewer
//! is colour-indexed — pixels become palette indices, packed two, four or
//! eight to a pixel when the palette is small enough — since nothing else
//! comes close for such content. Anything else is decorrelated with the
//! subtract-green and predictor transforms, each block taking the predictor
//! whose residuals cost least. Either way the result is coded with LZ77 back
//! references, a colour cache sized by an entropy estimate, and canonical
//! prefix codes.

#![allow(
    clippy::indexing_slicing,
    reason = "histograms are sized by their alphabets, pixel and block indices \
              by the image dimensions they were derived from"
)]

use crate::vp8l::{DISTANCE_MAP, div_round_up};

/// The literal-length code lengths' own code order (§3.7.2.1.2).
const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];
/// Shortest back reference worth coding, in pixels.
const MIN_MATCH: usize = 3;
/// Longest back reference VP8L can code.
const MAX_MATCH: usize = 4096;
/// The furthest back a reference can reach: distance code 39's largest value
/// less the 120 short codes.
const MAX_DISTANCE: usize = 1_048_576 - 120;
/// Hash-chain candidates examined per position.
const CHAIN_LIMIT: usize = 32;
/// Predictor and entropy block size, as log2 of the side.
const BLOCK_BITS: u32 = 4;

/// Writes bits least-significant first.
#[derive(Default)]
pub struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    count: u32,
}

impl BitWriter {
    /// Append the low `n` bits of `value`.
    pub fn put(&mut self, n: u32, value: u32) {
        debug_assert!(n <= 32);
        self.acc |= u64::from(value & ((1_u64 << n) - 1) as u32) << self.count;
        self.count += n;
        while self.count >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.count -= 8;
        }
    }

    /// The bytes written, the last one zero-padded.
    pub fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// Code lengths for `freqs`, at most `limit` bits. A symbol that never
/// occurs gets length 0. Lengths too long are cured by flattening the
/// distribution — raising every count to a floor — and rebuilding.
fn huffman_lengths(freqs: &[u32], limit: u8) -> Vec<u8> {
    let mut floor = 1_u32;
    loop {
        let lengths = huffman_unlimited(freqs, floor);
        if lengths.iter().all(|&l| l <= limit) {
            return lengths;
        }
        floor *= 2;
    }
}

fn huffman_unlimited(freqs: &[u32], floor: u32) -> Vec<u8> {
    let mut lengths = vec![0_u8; freqs.len()];
    let used: Vec<usize> = (0..freqs.len()).filter(|&s| freqs[s] > 0).collect();
    match used.len() {
        0 => return lengths,
        1 => {
            lengths[used[0]] = 1;
            return lengths;
        }
        _ => {}
    }
    // Nodes: leaves first, then internal nodes; a min-heap by weight.
    let mut weight: Vec<u64> = used
        .iter()
        .map(|&s| u64::from(freqs[s].max(floor)))
        .collect();
    let mut parent = vec![usize::MAX; used.len()];
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> = (0..used.len())
        .map(|i| std::cmp::Reverse((weight[i], i)))
        .collect();
    while heap.len() > 1 {
        let (Some(std::cmp::Reverse((wa, a))), Some(std::cmp::Reverse((wb, b)))) =
            (heap.pop(), heap.pop())
        else {
            break;
        };
        let node = weight.len();
        weight.push(wa + wb);
        parent.push(usize::MAX);
        parent[a] = node;
        parent[b] = node;
        heap.push(std::cmp::Reverse((wa + wb, node)));
    }
    for (i, &symbol) in used.iter().enumerate() {
        let mut depth = 0_u8;
        let mut n = i;
        while parent[n] != usize::MAX {
            n = parent[n];
            depth = depth.saturating_add(1);
        }
        lengths[symbol] = depth;
    }
    lengths
}

/// Canonical codes for `lengths`, bit-reversed for the LSB-first writer.
fn canonical_codes(lengths: &[u8]) -> Vec<u32> {
    let mut count = [0_u32; 16];
    for &l in lengths {
        count[usize::from(l)] += 1;
    }
    count[0] = 0;
    let mut next = [0_u32; 16];
    let mut code = 0;
    for len in 1..16 {
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
            c.reverse_bits() >> (32 - u32::from(l))
        })
        .collect()
}

/// A prefix code ready to write symbols with.
struct Code {
    lengths: Vec<u8>,
    codes: Vec<u32>,
    /// A code with one used symbol spends no bits on it (§3.7.2.1).
    single: bool,
}

impl Code {
    fn new(freqs: &[u32]) -> Self {
        let lengths = huffman_lengths(freqs, 15);
        let codes = canonical_codes(&lengths);
        let single = lengths.iter().filter(|&&l| l > 0).count() <= 1;
        Self {
            lengths,
            codes,
            single,
        }
    }

    fn symbol(&self, w: &mut BitWriter, symbol: usize) {
        if !self.single {
            w.put(u32::from(self.lengths[symbol]), self.codes[symbol]);
        }
    }

    /// Write this code's lengths (§3.7.2.1).
    fn write_header(&self, w: &mut BitWriter) {
        let used: Vec<usize> = (0..self.lengths.len())
            .filter(|&s| self.lengths[s] > 0)
            .collect();
        // One or two symbols below 256 fit the simple form, whose lengths
        // are all 1 (a lone symbol costs no bits).
        if used.len() <= 2 && used.iter().all(|&s| s < 256) {
            let symbols: Vec<usize> = if used.is_empty() { vec![0] } else { used };
            w.put(1, 1);
            w.put(1, (symbols.len() - 1) as u32);
            if symbols[0] < 2 {
                w.put(1, 0);
                w.put(1, symbols[0] as u32);
            } else {
                w.put(1, 1);
                w.put(8, symbols[0] as u32);
            }
            if let Some(&second) = symbols.get(1) {
                w.put(8, second as u32);
            }
            return;
        }
        w.put(1, 0);
        // Run-length tokens over the lengths: (token, extra bits, extra value).
        let mut tokens: Vec<(u8, u32, u32)> = Vec::new();
        let mut i = 0;
        let mut previous = 8_u8;
        let lengths = &self.lengths;
        while i < lengths.len() {
            let value = lengths[i];
            let run = lengths[i..].iter().take_while(|&&l| l == value).count();
            if value == 0 && run >= 3 {
                let run = run.min(138);
                if run >= 11 {
                    tokens.push((18, 7, (run - 11) as u32));
                } else {
                    tokens.push((17, 3, (run - 3) as u32));
                }
                i += run;
            } else if value != 0 && value == previous && run >= 3 {
                let run = run.min(6);
                tokens.push((16, 2, (run - 3) as u32));
                i += run;
            } else {
                tokens.push((value, 0, 0));
                if value != 0 {
                    previous = value;
                }
                i += 1;
            }
        }
        let mut freqs = [0_u32; 19];
        for &(t, _, _) in &tokens {
            freqs[usize::from(t)] += 1;
        }
        let mut length_lengths = huffman_lengths(&freqs, 7);
        // A lone token kind still needs a decodable code: give it a partner
        // so the tree is complete, as every reader expects of normal codes.
        if length_lengths.iter().filter(|&&l| l > 0).count() == 1 {
            let only = length_lengths.iter().position(|&l| l > 0).unwrap_or(0);
            length_lengths[only] = 1;
            length_lengths[if only == 0 { 1 } else { 0 }] = 1;
        }
        let length_codes = canonical_codes(&length_lengths);
        let count = CODE_LENGTH_ORDER
            .iter()
            .rposition(|&s| length_lengths[s] > 0)
            .map_or(4, |p| (p + 1).max(4));
        w.put(4, (count - 4) as u32);
        for &s in &CODE_LENGTH_ORDER[..count] {
            w.put(3, u32::from(length_lengths[s]));
        }
        w.put(1, 0); // max_symbol: the whole alphabet
        for (t, bits, extra) in tokens {
            let t = usize::from(t);
            w.put(u32::from(length_lengths[t]), length_codes[t]);
            if bits > 0 {
                w.put(bits, extra);
            }
        }
    }
}

/// A coded pixel: a literal, a cache index, or a back reference.
#[derive(Clone, Copy)]
enum Token {
    Literal(u32),
    Cache(u32),
    Copy { length: usize, distance_code: usize },
}

/// The prefix code and extra bits for a length or distance (§3.6.2.2).
fn prefix_encode(value: usize) -> (usize, u32, u32) {
    if value <= 4 {
        return (value - 1, 0, 0);
    }
    let v = value - 1;
    let high = usize::BITS - 1 - v.leading_zeros();
    let second = (v >> (high - 1)) & 1;
    let extra_bits = high - 1;
    let code = 2 * high as usize + second;
    (code, extra_bits, (v & ((1 << extra_bits) - 1)) as u32)
}

/// The distance code for a back reference `distance` pixels back.
fn distance_code(distance: usize, width: usize) -> usize {
    for (i, &(dx, dy)) in DISTANCE_MAP.iter().enumerate() {
        let d = i64::from(dx) + i64::from(dy) * width as i64;
        if d == distance as i64 {
            return i + 1;
        }
    }
    distance + 120
}

/// LZ77 over pixels with a hash chain, greedy with one step of lookahead.
fn lz77(pixels: &[u32], width: usize) -> Vec<(usize, usize)> {
    // Each entry: (length, distance); length 0 marks a literal.
    let n = pixels.len();
    let mut out = Vec::with_capacity(n);
    const HASH_BITS: u32 = 16;
    let hash = |i: usize| -> usize {
        let a = pixels[i].wrapping_mul(0x9e37_79b1);
        let b = pixels[i + 1].wrapping_mul(0x85eb_ca6b);
        let c = pixels[i + 2].wrapping_mul(0xc2b2_ae35);
        ((a ^ b.rotate_left(7) ^ c.rotate_left(13)) >> (32 - HASH_BITS)) as usize
    };
    let mut head = vec![usize::MAX; 1 << HASH_BITS];
    let mut prev = vec![usize::MAX; n];
    let insert = |i: usize, head: &mut [usize], prev: &mut [usize]| {
        if i + 2 < n {
            let h = hash(i);
            prev[i] = head[h];
            head[h] = i;
        }
    };
    let best_match = |i: usize, head: &[usize], prev: &[usize]| -> (usize, usize) {
        if i + MIN_MATCH > n {
            return (0, 0);
        }
        let mut best = (0, 0);
        let mut candidate = head[hash(i)];
        let mut tries = 0;
        let limit = (n - i).min(MAX_MATCH);
        while candidate != usize::MAX && tries < CHAIN_LIMIT && i - candidate <= MAX_DISTANCE {
            let len = (0..limit)
                .take_while(|&k| pixels[candidate + k] == pixels[i + k])
                .count();
            if len > best.0 {
                best = (len, i - candidate);
                if len == limit {
                    break;
                }
            }
            candidate = prev[candidate];
            tries += 1;
        }
        // The pixel directly above is the commonest long match; try it
        // whatever the hash chain held.
        if i >= width {
            let len = (0..limit)
                .take_while(|&k| pixels[i - width + k] == pixels[i + k])
                .count();
            if len > best.0 {
                best = (len, width);
            }
        }
        best
    };
    let mut i = 0;
    while i < n {
        let (len, dist) = best_match(i, &head, &prev);
        if len >= MIN_MATCH {
            // Lazy: a longer match one pixel on beats this one.
            insert(i, &mut head, &mut prev);
            let (next_len, _) = if i + 1 < n {
                best_match(i + 1, &head, &prev)
            } else {
                (0, 0)
            };
            if next_len > len + 1 {
                out.push((0, 0));
                i += 1;
                continue;
            }
            for k in 1..len {
                insert(i + k, &mut head, &mut prev);
            }
            out.push((len, dist));
            i += len;
        } else {
            insert(i, &mut head, &mut prev);
            out.push((0, 0));
            i += 1;
        }
    }
    out
}

fn cache_key(pixel: u32, bits: u32) -> usize {
    (pixel.wrapping_mul(0x1e35_a7bd) >> (32 - bits)) as usize
}

/// Turn LZ77 output into tokens, with a colour cache of `cache_bits`.
fn tokenize(
    pixels: &[u32],
    width: usize,
    matches: &[(usize, usize)],
    cache_bits: u32,
) -> Vec<Token> {
    let mut cache = vec![0_u32; if cache_bits > 0 { 1 << cache_bits } else { 0 }];
    let mut tokens = Vec::with_capacity(matches.len());
    let mut at = 0;
    for &(len, dist) in matches {
        if len == 0 {
            let p = pixels[at];
            if cache_bits > 0 && cache[cache_key(p, cache_bits)] == p {
                tokens.push(Token::Cache(cache_key(p, cache_bits) as u32));
            } else {
                tokens.push(Token::Literal(p));
            }
            if cache_bits > 0 {
                cache[cache_key(p, cache_bits)] = p;
            }
            at += 1;
        } else {
            tokens.push(Token::Copy {
                length: len,
                distance_code: distance_code(dist, width),
            });
            if cache_bits > 0 {
                for &p in &pixels[at..at + len] {
                    cache[cache_key(p, cache_bits)] = p;
                }
            }
            at += len;
        }
    }
    tokens
}

/// Symbol histograms for the five codes of one group.
struct Histograms {
    green: Vec<u32>,
    red: Vec<u32>,
    blue: Vec<u32>,
    alpha: Vec<u32>,
    distance: Vec<u32>,
}

fn histograms(tokens: &[Token], cache_bits: u32) -> Histograms {
    let cache_size = if cache_bits > 0 { 1 << cache_bits } else { 0 };
    let mut h = Histograms {
        green: vec![0; 256 + 24 + cache_size],
        red: vec![0; 256],
        blue: vec![0; 256],
        alpha: vec![0; 256],
        distance: vec![0; 40],
    };
    for &t in tokens {
        match t {
            Token::Literal(p) => {
                h.green[((p >> 8) & 0xff) as usize] += 1;
                h.red[((p >> 16) & 0xff) as usize] += 1;
                h.blue[(p & 0xff) as usize] += 1;
                h.alpha[(p >> 24) as usize] += 1;
            }
            Token::Cache(k) => h.green[256 + 24 + k as usize] += 1,
            Token::Copy {
                length,
                distance_code,
            } => {
                h.green[256 + prefix_encode(length).0] += 1;
                h.distance[prefix_encode(distance_code).0] += 1;
            }
        }
    }
    h
}

/// Shannon cost of a histogram in bits, a fast stand-in for its coded size.
fn entropy(freqs: &[u32]) -> f64 {
    let total: u64 = freqs.iter().map(|&f| u64::from(f)).sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    freqs
        .iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let f = f64::from(f);
            -f * (f / t).log2()
        })
        .sum()
}

/// Write `pixels` as an entropy-coded image (§3.8.3): colour cache, (for the
/// top level) no meta codes, one prefix code group, then the tokens.
fn write_coded_image(
    w: &mut BitWriter,
    pixels: &[u32],
    width: usize,
    top_level: bool,
    compress: bool,
) {
    let matches = if compress {
        lz77(pixels, width)
    } else {
        vec![(0, 0); pixels.len()]
    };
    // Colour cache size by estimated cost: none, small, medium, large.
    let candidates: &[u32] = if compress { &[0, 4, 7, 10] } else { &[0] };
    let (cache_bits, tokens) = candidates
        .iter()
        .map(|&bits| {
            let tokens = tokenize(pixels, width, &matches, bits);
            let h = histograms(&tokens, bits);
            let cost = entropy(&h.green)
                + entropy(&h.red)
                + entropy(&h.blue)
                + entropy(&h.alpha)
                + entropy(&h.distance)
                + f64::from(bits) * 64.0;
            (cost, bits, tokens)
        })
        .fold(None::<(f64, u32, Vec<Token>)>, |best, c| match best {
            Some(b) if b.0 <= c.0 => Some(b),
            _ => Some(c),
        })
        .map_or((0, Vec::new()), |(_, bits, tokens)| (bits, tokens));

    if cache_bits > 0 {
        w.put(1, 1);
        w.put(4, cache_bits);
    } else {
        w.put(1, 0);
    }
    if top_level {
        w.put(1, 0); // one prefix code group everywhere
    }
    let h = histograms(&tokens, cache_bits);
    let codes = [
        Code::new(&h.green),
        Code::new(&h.red),
        Code::new(&h.blue),
        Code::new(&h.alpha),
        Code::new(&h.distance),
    ];
    for code in &codes {
        code.write_header(w);
    }
    for &t in &tokens {
        match t {
            Token::Literal(p) => {
                codes[0].symbol(w, ((p >> 8) & 0xff) as usize);
                codes[1].symbol(w, ((p >> 16) & 0xff) as usize);
                codes[2].symbol(w, (p & 0xff) as usize);
                codes[3].symbol(w, (p >> 24) as usize);
            }
            Token::Cache(k) => codes[0].symbol(w, 256 + 24 + k as usize),
            Token::Copy {
                length,
                distance_code,
            } => {
                let (code, bits, extra) = prefix_encode(length);
                codes[0].symbol(w, 256 + code);
                w.put(bits, extra);
                let (code, bits, extra) = prefix_encode(distance_code);
                codes[4].symbol(w, code);
                w.put(bits, extra);
            }
        }
    }
}

/// Per-channel wrapping subtraction.
const fn sub_pixels(a: u32, b: u32) -> u32 {
    let alpha_green = (a | 0x00ff_00ff).wrapping_sub(b & 0xff00_ff00);
    let red_blue = (a | 0xff00_ff00).wrapping_sub(b & 0x00ff_00ff);
    (alpha_green & 0xff00_ff00) | (red_blue & 0x00ff_00ff)
}

/// The cost proxy for a residual: how far each channel is from zero.
fn residual_cost(r: u32) -> u32 {
    r.to_le_bytes()
        .iter()
        .map(|&b| u32::from((b as i8).unsigned_abs()))
        .sum()
}

/// Write `pixels` (ARGB) as a VP8L image stream: transforms, then the coded
/// image. Used both for a whole lossless file (after its header) and for an
/// `ALPH` chunk's alpha, whose size is implicit.
pub fn write_image_stream(w: &mut BitWriter, pixels: &[u32], width: usize, height: usize) {
    let mut palette: Vec<u32> = pixels.to_vec();
    palette.sort_unstable();
    palette.dedup();
    if palette.len() <= 256 {
        // Colour indexing (§3.5.4).
        w.put(1, 1);
        w.put(2, 3);
        w.put(8, (palette.len() - 1) as u32);
        let deltas: Vec<u32> = palette
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                if i == 0 {
                    p
                } else {
                    sub_pixels(p, palette[i - 1])
                }
            })
            .collect();
        write_coded_image(w, &deltas, deltas.len(), false, false);
        let bits = match palette.len() {
            0..=2 => 3,
            3..=4 => 2,
            5..=16 => 1,
            _ => 0,
        };
        let packed_width = div_round_up(width, bits);
        let per = 1 << bits;
        let index_bits = 8 >> bits;
        let mut packed = vec![0_u32; packed_width * height];
        for y in 0..height {
            for x in 0..width {
                let index = palette.binary_search(&pixels[y * width + x]).unwrap_or(0) as u32;
                packed[y * packed_width + (x >> bits)] |=
                    index << (8 + (x & (per - 1)) * index_bits);
            }
        }
        for p in &mut packed {
            *p |= 0xff00_0000;
        }
        w.put(1, 0); // no more transforms
        write_coded_image(w, &packed, packed_width, true, true);
        return;
    }

    // Subtract green (§3.5.3).
    w.put(1, 1);
    w.put(2, 2);
    let green_subtracted: Vec<u32> = pixels
        .iter()
        .map(|&p| {
            let g = (p >> 8) & 0xff;
            sub_pixels(p, (g << 16) | g)
        })
        .collect();

    // Predictor (§3.5.1): per 16x16 block, the mode whose residuals cost
    // least.
    let blocks_wide = div_round_up(width, BLOCK_BITS);
    let blocks_high = div_round_up(height, BLOCK_BITS);
    let src = &green_subtracted;
    let predict_at = |mode: u32, i: usize| -> u32 {
        let (x, y) = (i % width, i / width);
        if i == 0 {
            0xff00_0000
        } else if y == 0 {
            src[i - 1]
        } else if x == 0 {
            src[i - width]
        } else {
            crate::vp8l::predict(
                mode,
                src[i - 1],
                src[i - width],
                src[i - width + 1],
                src[i - width - 1],
            )
        }
    };
    let mut modes = vec![0_u32; blocks_wide * blocks_high];
    for by in 0..blocks_high {
        for bx in 0..blocks_wide {
            let mut best = (u32::MAX, 0);
            for mode in 0..14 {
                let mut cost = 0;
                for y in (by << BLOCK_BITS)..((by + 1) << BLOCK_BITS).min(height) {
                    for x in (bx << BLOCK_BITS)..((bx + 1) << BLOCK_BITS).min(width) {
                        let i = y * width + x;
                        cost += residual_cost(sub_pixels(src[i], predict_at(mode, i)));
                    }
                }
                if cost < best.0 {
                    best = (cost, mode);
                }
            }
            modes[by * blocks_wide + bx] = best.1;
        }
    }
    let residuals: Vec<u32> = (0..src.len())
        .map(|i| {
            let (x, y) = (i % width, i / width);
            let mode = modes[(y >> BLOCK_BITS) * blocks_wide + (x >> BLOCK_BITS)];
            sub_pixels(src[i], predict_at(mode, i))
        })
        .collect();
    w.put(1, 1);
    w.put(2, 0);
    w.put(3, BLOCK_BITS - 2);
    let mode_image: Vec<u32> = modes.iter().map(|&m| 0xff00_0000 | (m << 8)).collect();
    write_coded_image(w, &mode_image, blocks_wide, false, true);
    w.put(1, 0); // no more transforms
    write_coded_image(w, &residuals, width, true, true);
}

/// A complete `VP8L` chunk payload for `pixels` (ARGB, row-major).
pub fn encode(pixels: &[u32], width: usize, height: usize, has_alpha: bool) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.put(8, 0x2f);
    w.put(14, (width - 1) as u32);
    w.put(14, (height - 1) as u32);
    w.put(1, u32::from(has_alpha));
    w.put(3, 0);
    write_image_stream(&mut w, pixels, width, height);
    w.finish()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    fn round_trip(pixels: &[u32], width: usize, height: usize) {
        let data = encode(pixels, width, height, true);
        let decoded = crate::vp8l::decode(&data, width, height).unwrap();
        assert_eq!(decoded, pixels, "{width}x{height}");
    }

    #[test]
    fn prefix_encoding_inverts_the_decoder() {
        for value in [1, 2, 3, 4, 5, 6, 7, 12, 13, 100, 4096, 786_433, 1_048_576] {
            let (code, bits, extra) = prefix_encode(value);
            let decoded = if code < 4 {
                code + 1
            } else {
                let eb = (code - 2) >> 1;
                let offset = (2 + (code & 1)) << eb;
                assert_eq!(eb as u32, bits);
                offset + extra as usize + 1
            };
            assert_eq!(decoded, value);
        }
    }

    #[test]
    fn huffman_lengths_respect_the_limit_and_the_kraft_sum() {
        let freqs: Vec<u32> = (0..40).map(|i| 1 << (i % 20)).collect();
        let lengths = huffman_lengths(&freqs, 15);
        assert!(lengths.iter().all(|&l| (1..=15).contains(&l)));
        let kraft: f64 = lengths.iter().map(|&l| 0.5_f64.powi(i32::from(l))).sum();
        assert!((kraft - 1.0).abs() < 1e-9, "{kraft}");
    }

    #[test]
    fn images_of_every_shape_round_trip() {
        // Palette sizes on each side of every bundling width, and a busy one.
        for colours in [1_u32, 2, 3, 4, 5, 16, 17, 256, 4000] {
            let (w, h) = (37, 23);
            let pixels: Vec<u32> = (0..w * h)
                .map(|i| {
                    let c = ((i * 7919) % colours as usize) as u32;
                    0x8000_0000 | c.wrapping_mul(0x0103_0507)
                })
                .collect();
            round_trip(&pixels, w, h);
        }
        round_trip(&[0xff12_3456], 1, 1);
        let gradient: Vec<u32> = (0..64 * 40)
            .map(|i| 0xff00_0000 | ((i % 64) as u32 * 0x0003_0201) | ((i / 64) as u32) << 16)
            .collect();
        round_trip(&gradient, 64, 40);
        let column: Vec<u32> = (0..50).map(|i| 0xff00_0000 | (i * 5) as u32).collect();
        round_trip(&column, 1, 50);
        round_trip(&column, 50, 1);
    }
}
