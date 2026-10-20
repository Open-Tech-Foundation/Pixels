//! Entropy decoding (ISO/IEC 18181-1 §C): hybrid integers over prefix or
//! ANS codes, by context through a cluster map, with LZ77.
//!
//! Follows the decoding procedure as jxl-oxide (MIT/Apache-2.0) implements
//! it; see NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "table indices are masked to the table size, symbols checked \
              against the alphabet when the tables are built"
)]

use otf_pixels_core::Result;

use super::reader::{BitReader, malformed};

/// `ceil(log2(x + 1))`.
const fn add_log2_ceil(x: u32) -> u32 {
    if x >= 0x8000_0000 {
        32
    } else {
        (x + 1).next_power_of_two().trailing_zeros()
    }
}

/// How a token and raw bits make a value.
#[derive(Debug, Clone, Copy)]
struct IntegerConfig {
    split_exponent: u32,
    msb: u32,
    lsb: u32,
}

impl IntegerConfig {
    fn parse(r: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Self> {
        let split_exponent = r.read(add_log2_ceil(log_alphabet_size))?;
        let (msb, lsb) = if split_exponent == log_alphabet_size {
            (0, 0)
        } else {
            let msb = r.read(add_log2_ceil(split_exponent))?;
            if msb > split_exponent {
                return Err(malformed("bad hybrid integer configuration"));
            }
            let lsb = r.read(add_log2_ceil(split_exponent - msb))?;
            (msb, lsb)
        };
        if msb + lsb > split_exponent {
            return Err(malformed("bad hybrid integer configuration"));
        }
        Ok(Self {
            split_exponent,
            msb,
            lsb,
        })
    }

    fn value(&self, r: &mut BitReader<'_>, token: u32) -> Result<u32> {
        let split = 1_u32 << self.split_exponent;
        if token < split {
            return Ok(token);
        }
        let n = (self.split_exponent - (self.msb + self.lsb)
            + ((token - split) >> (self.msb + self.lsb)))
            & 31;
        let rest = r.read(n)?;
        let low = token & ((1 << self.lsb) - 1);
        let top = ((token >> self.lsb) & ((1 << self.msb) - 1)) | (1 << self.msb);
        Ok((((u64::from(top) << n) | u64::from(rest)) << self.lsb | u64::from(low)) as u32)
    }
}

/// A prefix code, as a lookup table on the next `bits` bits.
#[derive(Debug, Clone)]
struct PrefixCode {
    bits: u32,
    /// `(symbol, length)` by the next `bits` bits.
    table: Vec<(u16, u8)>,
}

impl PrefixCode {
    fn single(symbol: u16) -> Self {
        Self {
            bits: 0,
            table: vec![(symbol, 0)],
        }
    }

    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let max = lengths.iter().copied().max().unwrap_or(0);
        let used: Vec<usize> = (0..lengths.len()).filter(|&s| lengths[s] > 0).collect();
        if used.len() == 1 {
            return Ok(Self::single(used[0] as u16));
        }
        let bits = u32::from(max);
        let size = 1_usize << bits;
        let mut table = vec![(0_u16, 0_u8); size];
        let mut code = 0_u32;
        let mut filled = 0_usize;
        for len in 1..=max {
            for (symbol, &l) in lengths.iter().enumerate() {
                if l != len {
                    continue;
                }
                let reversed = code.reverse_bits() >> (32 - u32::from(len));
                let step = 1_usize << len;
                let mut at = reversed as usize;
                while at < size {
                    table[at] = (symbol as u16, len);
                    at += step;
                }
                filled += size >> len;
                code += 1;
            }
            code <<= 1;
        }
        if filled != size {
            return Err(malformed("an incomplete prefix code"));
        }
        Ok(Self { bits, table })
    }

    fn read(&self, r: &mut BitReader<'_>) -> Result<u32> {
        let (symbol, len) = self.table[r.peek(self.bits) as usize];
        r.skip(usize::from(len))?;
        Ok(u32::from(symbol))
    }

    /// Parse a code over `alphabet_size` symbols (Brotli's format).
    fn parse(r: &mut BitReader<'_>, alphabet_size: u32) -> Result<Self> {
        if alphabet_size == 1 {
            return Ok(Self::single(0));
        }
        if alphabet_size > 1 << 15 {
            return Err(malformed("a prefix code alphabet is too large"));
        }
        let hskip = r.read(2)?;
        if hskip == 1 {
            return Self::parse_simple(r, alphabet_size);
        }
        const ORDER: [usize; 18] = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut cl_lengths = [0_u8; 18];
        let (mut acc, mut nonzero, mut nonzero_sym) = (0_u32, 0, 0);
        for &idx in ORDER.iter().skip(hskip as usize) {
            // Brotli's fixed code for code-length-code lengths.
            let len = match r.read(2)? {
                0 => 0,
                1 => 4,
                2 => 3,
                _ => {
                    if r.bool()? {
                        if r.bool()? { 5 } else { 1 }
                    } else {
                        2
                    }
                }
            };
            cl_lengths[idx] = len;
            if len != 0 {
                nonzero += 1;
                nonzero_sym = idx;
                acc += 32 >> len;
                if acc == 32 {
                    break;
                }
                if acc > 32 {
                    return Err(malformed("bad code-length code"));
                }
            }
        }
        let cl_code = if nonzero == 1 {
            Self::single(nonzero_sym as u16)
        } else if acc != 32 {
            return Err(malformed("incomplete code-length code"));
        } else {
            Self::from_lengths(&cl_lengths)?
        };

        let mut lengths = vec![0_u8; alphabet_size as usize];
        let mut acc = 0_u32;
        let (mut prev, mut last_nonzero, mut last_repeat) = (8_u8, 8_u8, 0_usize);
        let mut i = 0;
        while i < lengths.len() {
            let sym = cl_code.read(r)? as u8;
            match sym {
                0..=15 => {
                    lengths[i] = sym;
                    i += 1;
                    if sym != 0 {
                        last_nonzero = sym;
                        acc += 32_768 >> sym;
                        if acc > 32_768 {
                            return Err(malformed("an over-full prefix code"));
                        }
                        if acc == 32_768 {
                            break;
                        }
                    }
                }
                16 | 17 => {
                    let (extra, value) = if sym == 16 { (2, last_nonzero) } else { (3, 0) };
                    let mut count = r.read(extra)? as usize + 3;
                    if prev == sym {
                        let mul = if sym == 16 { 3 } else { 7 };
                        let sub = if sym == 16 { 8 } else { 16 };
                        count = (count + last_repeat * mul).saturating_sub(sub);
                        last_repeat += count;
                    } else {
                        last_repeat = count;
                    }
                    for _ in 0..count {
                        if i >= lengths.len() {
                            return Err(malformed("a code-length repeat runs past the alphabet"));
                        }
                        lengths[i] = value;
                        i += 1;
                        if value != 0 {
                            acc += 32_768 >> value;
                        }
                    }
                    if acc > 32_768 {
                        return Err(malformed("an over-full prefix code"));
                    }
                    if acc == 32_768 {
                        break;
                    }
                }
                _ => return Err(malformed("bad code-length symbol")),
            }
            prev = sym;
        }
        if acc != 32_768 {
            return Err(malformed("an incomplete prefix code"));
        }
        Self::from_lengths(&lengths)
    }

    fn parse_simple(r: &mut BitReader<'_>, alphabet_size: u32) -> Result<Self> {
        let bits = alphabet_size.next_power_of_two().trailing_zeros();
        let nsym = r.read(2)? + 1;
        let mut syms = [0_usize; 4];
        for s in syms.iter_mut().take(nsym as usize) {
            *s = r.read(bits)? as usize;
            if *s >= alphabet_size as usize {
                return Err(malformed("a prefix symbol is past the alphabet"));
            }
        }
        if nsym == 1 {
            return Ok(Self::single(syms[0] as u16));
        }
        let shape: &[u8] = match nsym {
            2 => &[1, 1],
            3 => &[1, 2, 2],
            _ => {
                if r.bool()? {
                    &[1, 2, 3, 3]
                } else {
                    &[2, 2, 2, 2]
                }
            }
        };
        let mut lengths = vec![0_u8; alphabet_size as usize];
        for (&s, &l) in syms.iter().zip(shape) {
            if lengths[s] != 0 {
                return Err(malformed("a repeated prefix symbol"));
            }
            lengths[s] = l;
        }
        Self::from_lengths(&lengths)
    }
}

/// An ANS distribution with its alias table.
#[derive(Debug, Clone)]
struct AnsCode {
    log_bucket_size: u32,
    /// `(dist, alias symbol, alias offset, cutoff)` per bucket.
    buckets: Vec<(u32, u32, u32, u32)>,
    dist: Vec<u32>,
    single: Option<u32>,
}

impl AnsCode {
    fn read_u8(r: &mut BitReader<'_>) -> Result<u32> {
        Ok(if r.bool()? {
            let n = r.read(3)?;
            (1 << n) + r.read(n)?
        } else {
            0
        })
    }

    fn read_prefix(r: &mut BitReader<'_>) -> Result<u32> {
        Ok(match r.read(3)? {
            0 => 10,
            1 => {
                for v in [4, 0, 11, 13] {
                    if r.bool()? {
                        return Ok(v);
                    }
                }
                12
            }
            2 => 7,
            3 => {
                if r.bool()? { 1 } else { 3 }
            }
            4 => 6,
            5 => 8,
            6 => 9,
            _ => {
                if r.bool()? { 2 } else { 5 }
            }
        })
    }

    fn parse(r: &mut BitReader<'_>, log_alphabet_size: u32) -> Result<Self> {
        let table_size = 1_usize << log_alphabet_size;
        let log_bucket_size = 12 - log_alphabet_size;
        let bucket_size = 1_u32 << log_bucket_size;
        let mut dist = vec![0_u32; table_size];
        let alphabet_size;
        if r.bool()? {
            if r.bool()? {
                let (v0, v1) = (Self::read_u8(r)? as usize, Self::read_u8(r)? as usize);
                if v0 == v1 || v0.max(v1) >= table_size {
                    return Err(malformed("bad ANS histogram"));
                }
                alphabet_size = v0.max(v1) + 1;
                let p = r.read(12)?;
                dist[v0] = p;
                dist[v1] = 4096 - p;
            } else {
                let v = Self::read_u8(r)? as usize;
                if v >= table_size {
                    return Err(malformed("bad ANS histogram"));
                }
                alphabet_size = v + 1;
                dist[v] = 4096;
            }
        } else if r.bool()? {
            alphabet_size = Self::read_u8(r)? as usize + 1;
            if alphabet_size > table_size {
                return Err(malformed("bad ANS histogram"));
            }
            let base = 4096 / alphabet_size as u32;
            let leftover = 4096 % alphabet_size;
            for (i, d) in dist.iter_mut().enumerate().take(alphabet_size) {
                *d = base + u32::from(i < leftover);
            }
        } else {
            let mut len = 0;
            while len < 3 && r.bool()? {
                len += 1;
            }
            let shift = (r.read(len)? + (1 << len) - 1) as i32;
            if shift > 13 {
                return Err(malformed("bad ANS histogram shift"));
            }
            alphabet_size = Self::read_u8(r)? as usize + 3;
            if alphabet_size > table_size {
                return Err(malformed("bad ANS histogram"));
            }
            let mut repeat = vec![false; table_size];
            let mut omit: Option<(u32, usize)> = None;
            let mut i = 0;
            while i < alphabet_size {
                dist[i] = Self::read_prefix(r)?;
                if dist[i] == 13 {
                    let count = Self::read_u8(r)? as usize + 4;
                    if i + count > alphabet_size {
                        return Err(malformed("bad ANS histogram repeat"));
                    }
                    for flag in &mut repeat[i..i + count] {
                        *flag = true;
                    }
                    i += count;
                    continue;
                }
                match omit {
                    Some((log, _)) if dist[i] <= log => {}
                    _ => omit = Some((dist[i], i)),
                }
                i += 1;
            }
            let (_, omit_pos) = omit.ok_or_else(|| malformed("bad ANS histogram"))?;
            if repeat.get(omit_pos + 1) == Some(&true) {
                return Err(malformed("bad ANS histogram"));
            }
            let mut acc = 0_u32;
            let mut prev = 0_u32;
            for i in 0..table_size {
                if repeat[i] {
                    dist[i] = prev;
                } else {
                    let code = dist[i];
                    if code == 0 || i == omit_pos {
                        prev = 0;
                        continue;
                    }
                    if code > 1 {
                        let zeros = (code - 1) as i32;
                        let count = (shift - ((12 - zeros) >> 1)).clamp(0, zeros);
                        dist[i] = (1 << zeros) + (r.read(count as u32)? << (zeros - count));
                    }
                    prev = dist[i];
                }
                acc += dist[i];
                if acc > 4096 {
                    return Err(malformed("an over-full ANS histogram"));
                }
            }
            dist[omit_pos] = 4096 - acc;
        }
        if let Some(single) = dist.iter().position(|&d| d == 4096) {
            return Ok(Self {
                log_bucket_size,
                buckets: Vec::new(),
                dist,
                single: Some(single as u32),
            });
        }
        // Vose's alias method, exactly as the decoder's procedure builds it.
        let mut cutoff: Vec<u32> = dist.clone();
        let mut alias_symbol: Vec<u32> = (0..table_size as u32)
            .map(|i| if (i as usize) < alphabet_size { i } else { 0 })
            .collect();
        let mut alias_offset = vec![0_u32; table_size];
        let mut under = Vec::new();
        let mut over = Vec::new();
        for (i, &d) in dist.iter().enumerate() {
            match d.cmp(&bucket_size) {
                std::cmp::Ordering::Less => under.push(i),
                std::cmp::Ordering::Greater => over.push(i),
                std::cmp::Ordering::Equal => {}
            }
        }
        while let (Some(o), Some(u)) = (over.pop(), under.pop()) {
            let by = bucket_size - cutoff[u];
            cutoff[o] -= by;
            alias_symbol[u] = o as u32;
            alias_offset[u] = cutoff[o];
            match cutoff[o].cmp(&bucket_size) {
                std::cmp::Ordering::Less => under.push(o),
                std::cmp::Ordering::Greater => over.push(o),
                std::cmp::Ordering::Equal => {}
            }
        }
        let buckets = (0..table_size)
            .map(|i| {
                if cutoff[i] == bucket_size {
                    (dist[i], i as u32, 0, 0)
                } else {
                    (dist[i], alias_symbol[i], alias_offset[i].wrapping_sub(cutoff[i]), cutoff[i])
                }
            })
            .collect();
        Ok(Self {
            log_bucket_size,
            buckets,
            dist,
            single: None,
        })
    }

    fn read(&self, r: &mut BitReader<'_>, state: &mut u32) -> Result<u32> {
        let idx = *state & 0xFFF;
        let (symbol, offset, freq) = if let Some(single) = self.single {
            (single, idx, 4096)
        } else {
            let i = (idx >> self.log_bucket_size) as usize;
            let pos = idx & ((1 << self.log_bucket_size) - 1);
            let (dist, alias, alias_offset, cut) = self.buckets[i];
            if pos >= cut {
                let freq = self.dist[alias as usize];
                (alias, alias_offset.wrapping_add(pos), freq)
            } else {
                (i as u32, pos, dist)
            }
        };
        let next = (*state >> 12) * freq + offset;
        *state = if next < 1 << 16 {
            (next << 16) | r.read(16)?
        } else {
            next
        };
        Ok(symbol)
    }
}

#[derive(Debug, Clone)]
enum Coder {
    Prefix(Vec<PrefixCode>),
    Ans {
        codes: Vec<AnsCode>,
        state: u32,
        started: bool,
    },
}

/// LZ77 copies within a stream.
#[derive(Debug, Clone)]
struct Lz77 {
    min_symbol: u32,
    min_length: u32,
    length_config: IntegerConfig,
    window: Vec<u32>,
    to_copy: u32,
    copy_pos: u32,
    decoded: u32,
}

/// The special distances a Modular stream's copies use: `(dx, dy)`.
const SPECIAL_DISTANCES: [[i8; 2]; 120] = [
    [0, 1], [1, 0], [1, 1], [-1, 1], [0, 2], [2, 0], [1, 2], [-1, 2], [2, 1], [-2, 1], [2, 2],
    [-2, 2], [0, 3], [3, 0], [1, 3], [-1, 3], [3, 1], [-3, 1], [2, 3], [-2, 3], [3, 2], [-3, 2],
    [0, 4], [4, 0], [1, 4], [-1, 4], [4, 1], [-4, 1], [3, 3], [-3, 3], [2, 4], [-2, 4], [4, 2],
    [-4, 2], [0, 5], [3, 4], [-3, 4], [4, 3], [-4, 3], [5, 0], [1, 5], [-1, 5], [5, 1], [-5, 1],
    [2, 5], [-2, 5], [5, 2], [-5, 2], [4, 4], [-4, 4], [3, 5], [-3, 5], [5, 3], [-5, 3], [0, 6],
    [6, 0], [1, 6], [-1, 6], [6, 1], [-6, 1], [2, 6], [-2, 6], [6, 2], [-6, 2], [4, 5], [-4, 5],
    [5, 4], [-5, 4], [3, 6], [-3, 6], [6, 3], [-6, 3], [0, 7], [7, 0], [1, 7], [-1, 7], [5, 5],
    [-5, 5], [7, 1], [-7, 1], [4, 6], [-4, 6], [6, 4], [-6, 4], [2, 7], [-2, 7], [7, 2], [-7, 2],
    [3, 7], [-3, 7], [7, 3], [-7, 3], [5, 6], [-5, 6], [6, 5], [-6, 5], [8, 0], [4, 7], [-4, 7],
    [7, 4], [-7, 4], [8, 1], [8, 2], [6, 6], [-6, 6], [8, 3], [5, 7], [-5, 7], [7, 5], [-7, 5],
    [8, 4], [6, 7], [-6, 7], [7, 6], [-7, 6], [8, 5], [7, 7], [-7, 7], [8, 6], [8, 7],
];

/// An entropy-coded stream's decoder.
#[derive(Debug, Clone)]
pub struct Decoder {
    lz77: Option<Lz77>,
    clusters: Vec<u8>,
    configs: Vec<IntegerConfig>,
    coder: Coder,
}

impl Decoder {
    /// Parse a stream's header for `num_dist` contexts.
    ///
    /// # Errors
    ///
    /// For a malformed header.
    pub fn parse(r: &mut BitReader<'_>, num_dist: u32) -> Result<Self> {
        let lz77 = if r.bool()? {
            let min_symbol = r.u32([
                crate::bits::val(224),
                crate::bits::val(512),
                crate::bits::val(4096),
                crate::bits::bits(8, 15),
            ])?;
            let min_length = r.u32([
                crate::bits::val(3),
                crate::bits::val(4),
                crate::bits::bits(5, 2),
                crate::bits::bits(9, 8),
            ])?;
            let length_config = IntegerConfig::parse(r, 8)?;
            Some(Lz77 {
                min_symbol,
                min_length,
                length_config,
                window: Vec::new(),
                to_copy: 0,
                copy_pos: 0,
                decoded: 0,
            })
        } else {
            None
        };
        let num_dist = num_dist + u32::from(lz77.is_some());
        let (num_clusters, clusters) = read_clusters(r, num_dist)?;
        let prefix = r.bool()?;
        let log_alphabet_size = if prefix { 15 } else { r.read(2)? + 5 };
        let configs = (0..num_clusters)
            .map(|_| IntegerConfig::parse(r, log_alphabet_size))
            .collect::<Result<Vec<_>>>()?;
        let coder = if prefix {
            let counts = (0..num_clusters)
                .map(|_| -> Result<u32> {
                    if r.bool()? {
                        let n = r.read(4)?;
                        Ok(1 + (1 << n) + r.read(n)?)
                    } else {
                        Ok(1)
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Coder::Prefix(
                counts
                    .into_iter()
                    .map(|count| PrefixCode::parse(r, count))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            Coder::Ans {
                codes: (0..num_clusters)
                    .map(|_| AnsCode::parse(r, log_alphabet_size))
                    .collect::<Result<Vec<_>>>()?,
                state: 0,
                started: false,
            }
        };
        Ok(Self {
            lz77,
            clusters,
            configs,
            coder,
        })
    }

    /// Start a stream: ANS reads its initial state here.
    ///
    /// # Errors
    ///
    /// At the end of input.
    pub fn begin(&mut self, r: &mut BitReader<'_>) -> Result<()> {
        if let Coder::Ans { state, started, .. } = &mut self.coder {
            *state = r.read(32)?;
            *started = true;
        }
        Ok(())
    }

    /// Check a stream ended where it should: ANS's state returns to its
    /// initial value.
    ///
    /// # Errors
    ///
    /// If it did not.
    pub fn finish(&self) -> Result<()> {
        match self.coder {
            Coder::Ans { state, started: true, .. } if state != 0x13_0000 => {
                Err(malformed("an ANS stream ended in the wrong state"))
            }
            _ => Ok(()),
        }
    }

    fn symbol(&mut self, r: &mut BitReader<'_>, cluster: usize) -> Result<u32> {
        match &mut self.coder {
            Coder::Prefix(codes) => codes
                .get(cluster)
                .ok_or_else(|| malformed("a context names no code"))?
                .read(r),
            Coder::Ans {
                codes,
                state,
                started,
            } => {
                if !*started {
                    *state = r.read(32)?;
                    *started = true;
                }
                codes
                    .get(cluster)
                    .ok_or_else(|| malformed("a context names no code"))?
                    .read(r, state)
            }
        }
    }

    /// Read a value in context `ctx`.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn read(&mut self, r: &mut BitReader<'_>, ctx: u32) -> Result<u32> {
        self.read_with_multiplier(r, ctx, 0)
    }

    /// Read a value in context `ctx`, LZ77 distances scaled by
    /// `dist_multiplier` (a Modular channel's width) where nonzero.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn read_with_multiplier(&mut self, r: &mut BitReader<'_>, ctx: u32, dist_multiplier: u32) -> Result<u32> {
        let cluster = usize::from(
            *self
                .clusters
                .get(ctx as usize)
                .ok_or_else(|| malformed("a context out of range"))?,
        );
        let Some(mut lz) = self.lz77.take() else {
            let token = self.symbol(r, cluster)?;
            return self.configs[cluster].value(r, token);
        };
        let result = (|| -> Result<u32> {
            let value = if lz.to_copy > 0 {
                let v = lz.window[(lz.copy_pos & 0xF_FFFF) as usize];
                lz.copy_pos += 1;
                lz.to_copy -= 1;
                v
            } else {
                let token = self.symbol(r, cluster)?;
                if token >= lz.min_symbol {
                    if lz.decoded == 0 {
                        return Err(malformed("an LZ77 copy before any value"));
                    }
                    let dist_cluster = usize::from(*self.clusters.last().unwrap_or(&0));
                    let length = lz.length_config.value(r, token - lz.min_symbol)?;
                    lz.to_copy = length
                        .checked_add(lz.min_length)
                        .ok_or_else(|| malformed("an LZ77 length overflows"))?;
                    let token = self.symbol(r, dist_cluster)?;
                    let distance = self.configs[dist_cluster].value(r, token)?;
                    let distance = if dist_multiplier == 0 {
                        distance
                    } else if distance < 120 {
                        let [dx, dy] = SPECIAL_DISTANCES[distance as usize];
                        (i32::from(dx) + dist_multiplier as i32 * i32::from(dy) - 1).max(0) as u32
                    } else {
                        distance - 120
                    };
                    let distance = ((1 << 20) - 1).min(distance).saturating_add(1).min(lz.decoded);
                    lz.copy_pos = lz.decoded - distance;
                    let v = lz.window[(lz.copy_pos & 0xF_FFFF) as usize];
                    lz.copy_pos += 1;
                    lz.to_copy -= 1;
                    v
                } else {
                    self.configs[cluster].value(r, token)?
                }
            };
            let at = (lz.decoded & 0xF_FFFF) as usize;
            if lz.window.len() <= at {
                lz.window.push(value);
            } else {
                lz.window[at] = value;
            }
            lz.decoded += 1;
            Ok(value)
        })();
        self.lz77 = Some(lz);
        result
    }
}

/// Read a context map: each of `num_dist` contexts' cluster.
fn read_clusters(r: &mut BitReader<'_>, num_dist: u32) -> Result<(u32, Vec<u8>)> {
    if num_dist == 1 {
        return Ok((1, vec![0]));
    }
    let clusters = if r.bool()? {
        let bits = r.read(2)?;
        (0..num_dist)
            .map(|_| r.read(bits).map(|v| v as u8))
            .collect::<Result<Vec<_>>>()?
    } else {
        let mtf = r.bool()?;
        let mut decoder = Decoder::parse(r, 1)?;
        if num_dist <= 2 && decoder.lz77.is_some() {
            return Err(malformed("LZ77 in a small context map"));
        }
        decoder.begin(r)?;
        let mut out = (0..num_dist)
            .map(|_| {
                let v = decoder.read(r, 0)?;
                u8::try_from(v).map_err(|_| malformed("a cluster past 255"))
            })
            .collect::<Result<Vec<_>>>()?;
        decoder.finish()?;
        if mtf {
            let mut order: Vec<u8> = (0..=255).collect();
            for c in &mut out {
                let i = usize::from(*c);
                let v = order[i];
                order.copy_within(0..i, 1);
                order[0] = v;
                *c = v;
            }
        }
        out
    };
    let count = u32::from(*clusters.iter().max().unwrap_or(&0)) + 1;
    let mut seen = vec![false; count as usize];
    for &c in &clusters {
        seen[usize::from(c)] = true;
    }
    if seen.iter().any(|&s| !s) {
        return Err(malformed("a context map leaves a cluster unused"));
    }
    Ok((count, clusters))
}

/// Read a permutation of `size` entries whose first `skip` are fixed
/// (Lehmer-coded).
///
/// # Errors
///
/// For an invalid permutation.
pub fn read_permutation(r: &mut BitReader<'_>, d: &mut Decoder, size: u32, skip: u32) -> Result<Vec<usize>> {
    let ctx = |x: u32| add_log2_ceil(x).min(7);
    let end = d.read(r, ctx(size))?;
    if end > size - skip {
        return Err(malformed("bad permutation length"));
    }
    let mut lehmer = Vec::with_capacity(end as usize);
    let mut prev = 0;
    for i in 0..end {
        let v = d.read(r, ctx(prev))?;
        if v >= size - skip - i {
            return Err(malformed("bad permutation entry"));
        }
        lehmer.push(v);
        prev = v;
    }
    let mut rest: Vec<usize> = (skip as usize..size as usize).collect();
    let mut out: Vec<usize> = (0..skip as usize).collect();
    for v in lehmer {
        out.push(rest.remove(v as usize));
    }
    out.extend(rest);
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;
    use crate::bits::BitWriter;
    use crate::entropy::{EntropyCode, HybridUint, Token, with_runs};

    /// Our encoder's streams decode back.
    #[test]
    fn our_streams_round_trip() {
        let mut tokens: Vec<Token> = Vec::new();
        for i in 0..4000_u32 {
            let ctx = i % 11;
            let value = if i % 50 < 20 { 0 } else { i.wrapping_mul(2_654_435_761) >> (20 + ctx) };
            tokens.push(Token::new(ctx, value));
        }
        let tokens = with_runs(&tokens, 0);
        let code = EntropyCode::build(11, &tokens, HybridUint::DEFAULT);
        let mut w = BitWriter::new();
        code.write_header(&mut w);
        for &t in &tokens {
            code.write(&mut w, t);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let mut d = Decoder::parse(&mut r, 11).unwrap();
        d.begin(&mut r).unwrap();
        // Copies expand: compare value by value.
        let mut expected = Vec::new();
        for t in &tokens {
            if t.copy {
                let last = *expected.last().unwrap();
                expected.extend(std::iter::repeat_n(last, t.value as usize));
            } else {
                expected.push(t.value);
            }
        }
        let mut ctxs = Vec::new();
        for t in &tokens {
            if t.copy {
                ctxs.extend(std::iter::repeat_n(t.ctx, t.value as usize));
            } else {
                ctxs.push(t.ctx);
            }
        }
        for (i, (&want, &ctx)) in expected.iter().zip(&ctxs).enumerate() {
            assert_eq!(d.read_with_multiplier(&mut r, ctx, 64).unwrap(), want, "value {i}");
        }
        d.finish().unwrap();
    }
}
