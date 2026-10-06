//! VP8L, the WebP lossless bitstream (RFC 9649 §3).
//!
//! An image stream is an optional chain of reversible transforms (predictor,
//! colour, subtract-green, colour indexing) over a pixel array coded with
//! canonical prefix codes, LZ77 back references into the pixels already
//! decoded, and a small hash cache of recent colours. The transforms' own data
//! are smaller images coded the same way, so decoding recurses.
//!
//! Pixels are ARGB packed in a `u32` as the specification describes them:
//! alpha in the top byte, then red, green, blue.

#![allow(
    clippy::indexing_slicing,
    reason = "every index is bounded by construction: transform sub-images are \
              sized by the same div_round_up their lookups use, the colour table is \
              padded to 256, distance codes are range-checked before the map, back \
              references are checked against the pixels decoded so far, and code \
              lengths are below 16 by the token alphabet. The truncation and \
              corruption tests hold the decoder to that"
)]

use otf_pixels_core::{PixelsError, Result};

/// The most code-length bits any prefix code may use.
const MAX_CODE_LENGTH: usize = 15;
/// Bits resolved by one lookup in a prefix code's fast table.
const FAST_BITS: u32 = 8;
/// The literal-length code lengths' own code order (§3.7.2.1.2).
const CODE_LENGTH_ORDER: [usize; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];
/// Back-reference length prefix codes (§3.6.2.2).
const LENGTH_CODES: usize = 24;
/// Distance prefix codes.
const DISTANCE_CODES: usize = 40;
/// The 120 short distance codes' `(dx, dy)` neighbour offsets (§3.6.2.2.1).
pub(crate) const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (0, 2),
    (2, 0),
    (1, 2),
    (-1, 2),
    (2, 1),
    (-2, 1),
    (2, 2),
    (-2, 2),
    (0, 3),
    (3, 0),
    (1, 3),
    (-1, 3),
    (3, 1),
    (-3, 1),
    (2, 3),
    (-2, 3),
    (3, 2),
    (-3, 2),
    (0, 4),
    (4, 0),
    (1, 4),
    (-1, 4),
    (4, 1),
    (-4, 1),
    (3, 3),
    (-3, 3),
    (2, 4),
    (-2, 4),
    (4, 2),
    (-4, 2),
    (0, 5),
    (3, 4),
    (-3, 4),
    (4, 3),
    (-4, 3),
    (5, 0),
    (1, 5),
    (-1, 5),
    (5, 1),
    (-5, 1),
    (2, 5),
    (-2, 5),
    (5, 2),
    (-5, 2),
    (4, 4),
    (-4, 4),
    (3, 5),
    (-3, 5),
    (5, 3),
    (-5, 3),
    (0, 6),
    (6, 0),
    (1, 6),
    (-1, 6),
    (6, 1),
    (-6, 1),
    (2, 6),
    (-2, 6),
    (6, 2),
    (-6, 2),
    (4, 5),
    (-4, 5),
    (5, 4),
    (-5, 4),
    (3, 6),
    (-3, 6),
    (6, 3),
    (-6, 3),
    (0, 7),
    (7, 0),
    (1, 7),
    (-1, 7),
    (5, 5),
    (-5, 5),
    (7, 1),
    (-7, 1),
    (4, 6),
    (-4, 6),
    (6, 4),
    (-6, 4),
    (2, 7),
    (-2, 7),
    (7, 2),
    (-7, 2),
    (3, 7),
    (-3, 7),
    (7, 3),
    (-7, 3),
    (5, 6),
    (-5, 6),
    (6, 5),
    (-6, 5),
    (8, 0),
    (4, 7),
    (-4, 7),
    (7, 4),
    (-7, 4),
    (8, 1),
    (8, 2),
    (6, 6),
    (-6, 6),
    (8, 3),
    (5, 7),
    (-5, 7),
    (7, 5),
    (-7, 5),
    (8, 4),
    (6, 7),
    (-6, 7),
    (7, 6),
    (-7, 6),
    (8, 5),
    (7, 7),
    (-7, 7),
    (8, 6),
    (8, 7),
];

fn malformed(detail: impl Into<String>) -> PixelsError {
    PixelsError::malformed("webp", detail.into())
}

/// Reads bits least-significant first, as VP8L packs them.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data`.
    position: usize,
}

impl<'a> BitReader<'a> {
    /// Read from the start of `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// The next `n` (at most 32) bits without consuming them; bits past the
    /// end read as zero, which [`BitReader::consume`] then refuses.
    fn peek(&self, n: u32) -> u32 {
        let byte = self.position >> 3;
        let mut word = 0_u64;
        for (i, &b) in self.data.iter().skip(byte).take(5).enumerate() {
            word |= u64::from(b) << (8 * i);
        }
        let shifted = word >> (self.position & 7);
        (shifted & ((1_u64 << n) - 1)) as u32
    }

    fn consume(&mut self, n: u32) -> Result<()> {
        self.position += n as usize;
        if self.position > self.data.len() * 8 {
            return Err(malformed("the lossless bitstream ends early"));
        }
        Ok(())
    }

    /// Read `n` bits, at most 32.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] past the end of the data.
    pub fn read(&mut self, n: u32) -> Result<u32> {
        let value = self.peek(n);
        self.consume(n)?;
        Ok(value)
    }

    fn flag(&mut self) -> Result<bool> {
        Ok(self.read(1)? == 1)
    }
}

/// One canonical prefix code.
enum PrefixCode {
    /// A single used symbol, which costs no bits (§3.7.2.1).
    Single(u16),
    /// Two or more symbols.
    Tree {
        /// `(symbol << 4) | length` for every `FAST_BITS`-bit window whose
        /// code is that short, 0 where a longer code continues.
        fast: Box<[u16; 1 << FAST_BITS]>,
        /// Codes of each length, for the canonical walk past the fast table.
        counts: [u16; MAX_CODE_LENGTH + 1],
        /// Symbols in canonical order.
        symbols: Vec<u16>,
    },
}

impl PrefixCode {
    /// Build a code from per-symbol lengths, which must describe a complete
    /// tree unless exactly one symbol is used.
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0_u16; MAX_CODE_LENGTH + 1];
        for &length in lengths {
            counts[usize::from(length)] += 1;
        }
        counts[0] = 0;
        let used: usize = counts.iter().map(|&c| usize::from(c)).sum();
        if used == 0 {
            return Err(malformed("a prefix code has no symbols"));
        }
        if used == 1 {
            let symbol = lengths.iter().position(|&l| l != 0).unwrap_or(0);
            return Ok(Self::Single(symbol as u16));
        }
        // Kraft: the lengths must fill the code space exactly.
        let mut space = 1_i64 << MAX_CODE_LENGTH;
        for (length, &count) in counts.iter().enumerate().skip(1) {
            space -= i64::from(count) << (MAX_CODE_LENGTH - length);
        }
        if space != 0 {
            return Err(malformed(
                "a prefix code's lengths do not form a complete tree",
            ));
        }

        let mut offsets = [0_usize; MAX_CODE_LENGTH + 2];
        for length in 1..=MAX_CODE_LENGTH {
            offsets[length + 1] = offsets[length] + usize::from(counts[length]);
        }
        let mut symbols = vec![0_u16; used];
        let mut next = offsets;
        for (symbol, &length) in lengths.iter().enumerate() {
            if length != 0 {
                let slot = &mut next[usize::from(length)];
                symbols[*slot] = symbol as u16;
                *slot += 1;
            }
        }

        // Fast table: walk the canonical codes in order, placing each short
        // one at every window whose low bits are its code bit-reversed.
        let mut fast = Box::new([0_u16; 1 << FAST_BITS]);
        let mut code = 0_u32;
        let mut index = 0;
        for length in 1..=MAX_CODE_LENGTH as u32 {
            for _ in 0..counts[length as usize] {
                if length <= FAST_BITS {
                    let reversed = code.reverse_bits() >> (32 - length);
                    let entry = (symbols[index] << 4) | length as u16;
                    let mut slot = reversed as usize;
                    while slot < 1 << FAST_BITS {
                        fast[slot] = entry;
                        slot += 1 << length;
                    }
                }
                code += 1;
                index += 1;
            }
            code <<= 1;
        }
        Ok(Self::Tree {
            fast,
            counts,
            symbols,
        })
    }

    fn read(&self, bits: &mut BitReader<'_>) -> Result<u16> {
        match self {
            Self::Single(symbol) => Ok(*symbol),
            Self::Tree {
                fast,
                counts,
                symbols,
            } => {
                let entry = fast[bits.peek(FAST_BITS) as usize];
                if entry != 0 {
                    bits.consume(u32::from(entry & 15))?;
                    return Ok(entry >> 4);
                }
                // The canonical walk (as in zlib's `puff`), one bit at a time.
                let (mut code, mut first, mut index) = (0_i32, 0_i32, 0_i32);
                for &count in counts.iter().skip(1) {
                    code |= bits.read(1)? as i32;
                    let count = i32::from(count);
                    if code - count < first {
                        return symbols
                            .get((index + code - first) as usize)
                            .copied()
                            .ok_or_else(|| malformed("a prefix code walked off its symbols"));
                    }
                    index += count;
                    first = (first + count) << 1;
                    code <<= 1;
                }
                Err(malformed("a prefix code was read past its longest length"))
            }
        }
    }
}

/// Read one prefix code's lengths over an `alphabet` and build it
/// (§3.7.2.1). `build` false parses and discards, for groups nothing uses.
fn read_prefix_code(
    bits: &mut BitReader<'_>,
    alphabet: usize,
    build: bool,
) -> Result<Option<PrefixCode>> {
    let mut lengths = vec![0_u8; alphabet];
    if bits.flag()? {
        // Simple code: one or two symbols of length 1.
        let two = bits.flag()?;
        let first_bits = if bits.flag()? { 8 } else { 1 };
        let symbol = bits.read(first_bits)? as usize;
        *lengths
            .get_mut(symbol)
            .ok_or_else(|| malformed("a simple code's symbol is out of range"))? = 1;
        if two {
            let symbol = bits.read(8)? as usize;
            *lengths
                .get_mut(symbol)
                .ok_or_else(|| malformed("a simple code's symbol is out of range"))? = 1;
        }
    } else {
        let mut length_lengths = [0_u8; 19];
        let count = 4 + bits.read(4)? as usize;
        for &position in CODE_LENGTH_ORDER.iter().take(count) {
            length_lengths[position] = bits.read(3)? as u8;
        }
        let length_code = PrefixCode::new(&length_lengths)?;
        let mut max_tokens = if bits.flag()? {
            let width = 2 + 2 * bits.read(3)?;
            let max = 2 + bits.read(width)? as usize;
            if max > alphabet {
                return Err(malformed(
                    "a prefix code declares more symbols than its alphabet",
                ));
            }
            max
        } else {
            alphabet
        };
        let mut symbol = 0;
        let mut previous = 8_u8;
        while symbol < alphabet {
            if max_tokens == 0 {
                break;
            }
            max_tokens -= 1;
            let token = length_code.read(bits)?;
            if token < 16 {
                lengths[symbol] = token as u8;
                symbol += 1;
                if token != 0 {
                    previous = token as u8;
                }
                continue;
            }
            let (repeat, value) = match token {
                16 => (3 + bits.read(2)? as usize, previous),
                17 => (3 + bits.read(3)? as usize, 0),
                _ => (11 + bits.read(7)? as usize, 0),
            };
            let run = lengths
                .get_mut(symbol..symbol + repeat)
                .ok_or_else(|| malformed("a code-length run overruns the alphabet"))?;
            run.fill(value);
            symbol += repeat;
        }
    }
    if build {
        PrefixCode::new(&lengths).map(Some)
    } else {
        // Still validated: a broken unused code is still a broken stream.
        PrefixCode::new(&lengths).map(|_| None)
    }
}

/// The five codes for one block: green-length-cache, red, blue, alpha, distance.
struct Group {
    codes: [PrefixCode; 5],
}

/// A transform and the width it was read at.
enum Transform {
    Predictor { bits: u32, modes: Vec<u32> },
    Color { bits: u32, elements: Vec<u32> },
    SubtractGreen,
    ColorIndexing { bits: u32, table: Vec<u32> },
}

pub(crate) const fn div_round_up(value: usize, bits: u32) -> usize {
    (value + (1 << bits) - 1) >> bits
}

/// Decode a VP8L file's image stream after its 5-byte header: the image's
/// `width * height` ARGB pixels.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for any stream that breaks the format.
pub fn decode(data: &[u8], width: usize, height: usize) -> Result<Vec<u32>> {
    let mut bits = BitReader::new(
        data.get(5..)
            .ok_or_else(|| malformed("VP8L header cut short"))?,
    );
    decode_image_stream(&mut bits, width, height, true)
}

/// Decode an image stream of the given size (§3.8). Only the top-level
/// (`level0`) image carries transforms and meta prefix codes; an `ALPH`
/// chunk's stream is one, with its size implicit.
///
/// # Errors
///
/// As [`decode`].
pub fn decode_image_stream(
    bits: &mut BitReader<'_>,
    width: usize,
    height: usize,
    level0: bool,
) -> Result<Vec<u32>> {
    let mut transforms: Vec<(Transform, usize)> = Vec::new();
    let mut coded_width = width;
    if level0 {
        let mut seen = [false; 4];
        while bits.flag()? {
            let kind = bits.read(2)? as usize;
            if std::mem::replace(&mut seen[kind], true) {
                return Err(malformed("a transform appears twice"));
            }
            let transform = match kind {
                0 | 1 => {
                    let block_bits = bits.read(3)? + 2;
                    let sub = decode_image_stream(
                        bits,
                        div_round_up(coded_width, block_bits),
                        div_round_up(height, block_bits),
                        false,
                    )?;
                    if kind == 0 {
                        Transform::Predictor {
                            bits: block_bits,
                            modes: sub,
                        }
                    } else {
                        Transform::Color {
                            bits: block_bits,
                            elements: sub,
                        }
                    }
                }
                2 => Transform::SubtractGreen,
                _ => {
                    let size = bits.read(8)? as usize + 1;
                    let mut table = decode_image_stream(bits, size, 1, false)?;
                    // Stored as deltas, each channel wrapping independently.
                    for i in 1..table.len() {
                        table[i] = add_pixels(table[i], table[i - 1]);
                    }
                    let bundle = match size {
                        0..=2 => 3,
                        3..=4 => 2,
                        5..=16 => 1,
                        _ => 0,
                    };
                    // Unused indices decode to transparent black.
                    table.resize(256, 0);
                    Transform::ColorIndexing {
                        bits: bundle,
                        table,
                    }
                }
            };
            let read_width = coded_width;
            if let Transform::ColorIndexing { bits: bundle, .. } = transform {
                coded_width = div_round_up(coded_width, bundle);
            }
            transforms.push((transform, read_width));
        }
    }

    let cache_bits = if bits.flag()? {
        let cache_bits = bits.read(4)?;
        if !(1..=11).contains(&cache_bits) {
            return Err(malformed(format!(
                "color cache bits {cache_bits} outside 1..=11"
            )));
        }
        cache_bits
    } else {
        0
    };

    // Meta prefix codes: an entropy image naming each block's group.
    let mut entropy: Option<(u32, usize, Vec<u32>)> = None;
    let mut group_count = 1;
    if level0 && bits.flag()? {
        let block_bits = bits.read(3)? + 2;
        let entropy_width = div_round_up(coded_width, block_bits);
        let image =
            decode_image_stream(bits, entropy_width, div_round_up(height, block_bits), false)?;
        group_count = image
            .iter()
            .map(|&p| ((p >> 8) & 0xffff) as usize)
            .max()
            .unwrap_or(0)
            + 1;
        entropy = Some((block_bits, entropy_width, image));
    }
    let mut used = vec![entropy.is_none(); group_count];
    if let Some((_, _, image)) = &entropy {
        for &p in image {
            used[((p >> 8) & 0xffff) as usize] = true;
        }
    }
    let cache_size = if cache_bits > 0 {
        1_usize << cache_bits
    } else {
        0
    };
    let alphabets = [
        256 + LENGTH_CODES + cache_size,
        256,
        256,
        256,
        DISTANCE_CODES,
    ];
    let mut groups: Vec<Option<Group>> = Vec::with_capacity(group_count.min(4096));
    for &needed in &used {
        let mut codes = Vec::with_capacity(5);
        for &alphabet in &alphabets {
            codes.push(read_prefix_code(bits, alphabet, needed)?);
        }
        groups.push(if needed {
            let codes: Vec<PrefixCode> = codes.into_iter().flatten().collect();
            let codes: [PrefixCode; 5] = codes
                .try_into()
                .map_err(|_| malformed("a prefix code group is incomplete"))?;
            Some(Group { codes })
        } else {
            None
        });
    }

    let mut pixels = decode_pixels(
        bits,
        coded_width,
        height,
        cache_bits,
        &groups,
        entropy.as_ref(),
    )?;

    for (transform, read_width) in transforms.iter().rev() {
        pixels = inverse(transform, pixels, *read_width, height);
    }
    Ok(pixels)
}

/// Per-channel addition, wrapping each byte.
const fn add_pixels(a: u32, b: u32) -> u32 {
    let alpha_green = (a & 0xff00_ff00).wrapping_add(b & 0xff00_ff00);
    let red_blue = (a & 0x00ff_00ff).wrapping_add(b & 0x00ff_00ff);
    (alpha_green & 0xff00_ff00) | (red_blue & 0x00ff_00ff)
}

/// A length or distance from its prefix code and extra bits (§3.6.2.2).
fn prefix_value(bits: &mut BitReader<'_>, code: u16) -> Result<usize> {
    let code = u32::from(code);
    if code < 4 {
        return Ok(code as usize + 1);
    }
    let extra = (code - 2) >> 1;
    let offset = (2 + (code & 1)) << extra;
    Ok((offset + bits.read(extra)? + 1) as usize)
}

fn decode_pixels(
    bits: &mut BitReader<'_>,
    width: usize,
    height: usize,
    cache_bits: u32,
    groups: &[Option<Group>],
    entropy: Option<&(u32, usize, Vec<u32>)>,
) -> Result<Vec<u32>> {
    let total = width
        .checked_mul(height)
        .ok_or_else(|| malformed("the image size overflows"))?;
    let mut pixels: Vec<u32> = Vec::with_capacity(total);
    let mut cache = vec![0_u32; if cache_bits > 0 { 1 << cache_bits } else { 0 }];
    let mut cached = 0;
    let group_at = |position: usize| -> Result<&Group> {
        let index = match entropy {
            None => 0,
            Some((block_bits, entropy_width, image)) => {
                let (x, y) = (position % width, position / width);
                let at = (y >> block_bits) * entropy_width + (x >> block_bits);
                ((image.get(at).copied().unwrap_or(0) >> 8) & 0xffff) as usize
            }
        };
        groups
            .get(index)
            .and_then(Option::as_ref)
            .ok_or_else(|| malformed("a block names a prefix code group that is missing"))
    };

    while pixels.len() < total {
        let group = group_at(pixels.len())?;
        let symbol = group.codes[0].read(bits)?;
        if symbol < 256 {
            let red = group.codes[1].read(bits)?;
            let blue = group.codes[2].read(bits)?;
            let alpha = group.codes[3].read(bits)?;
            pixels.push(
                (u32::from(alpha) << 24)
                    | (u32::from(red) << 16)
                    | (u32::from(symbol) << 8)
                    | u32::from(blue),
            );
        } else if usize::from(symbol) < 256 + LENGTH_CODES {
            let length = prefix_value(bits, symbol - 256)?;
            let distance_symbol = group.codes[4].read(bits)?;
            let code = prefix_value(bits, distance_symbol)?;
            let distance = if code > 120 {
                code - 120
            } else {
                let (dx, dy) = DISTANCE_MAP[code - 1];
                (i64::from(dx) + i64::from(dy) * width as i64).max(1) as usize
            };
            let start = pixels
                .len()
                .checked_sub(distance)
                .ok_or_else(|| malformed("a back reference points before the image"))?;
            if pixels.len() + length > total {
                return Err(malformed("a back reference runs past the image"));
            }
            for i in 0..length {
                let pixel = pixels[start + i];
                pixels.push(pixel);
            }
        } else {
            let index = usize::from(symbol) - 256 - LENGTH_CODES;
            let pixel = *cache
                .get(index)
                .ok_or_else(|| malformed("a color cache index is out of range"))?;
            pixels.push(pixel);
        }
        if cache_bits > 0 {
            for &pixel in &pixels[cached..] {
                let key = pixel.wrapping_mul(0x1e35_a7bd) >> (32 - cache_bits);
                cache[key as usize] = pixel;
            }
            cached = pixels.len();
        }
    }
    Ok(pixels)
}

fn channel(pixel: u32, shift: u32) -> i32 {
    ((pixel >> shift) & 0xff) as i32
}

fn per_channel(f: impl Fn(i32, i32, i32) -> i32, a: u32, b: u32, c: u32) -> u32 {
    [24, 16, 8, 0].iter().fold(0, |out, &shift| {
        let value = f(channel(a, shift), channel(b, shift), channel(c, shift));
        out | ((value.clamp(0, 255) as u32) << shift)
    })
}

fn average2(a: u32, b: u32) -> u32 {
    per_channel(|a, b, _| (a + b) / 2, a, b, 0)
}

fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let distance = |pixel: u32| -> i32 {
        [24, 16, 8, 0]
            .iter()
            .map(|&s| {
                let estimate = channel(left, s) + channel(top, s) - channel(top_left, s);
                (estimate - channel(pixel, s)).abs()
            })
            .sum()
    };
    if distance(left) < distance(top) {
        left
    } else {
        top
    }
}

/// Predictor `mode` from left, top, top-right and top-left (§3.5.1).
pub(crate) fn predict(mode: u32, l: u32, t: u32, tr: u32, tl: u32) -> u32 {
    match mode {
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => average2(average2(l, tr), t),
        6 => average2(l, tl),
        7 => average2(l, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average2(average2(l, tl), average2(t, tr)),
        11 => select(l, t, tl),
        12 => per_channel(|l, t, tl| l + t - tl, l, t, tl),
        13 => per_channel(|a, tl, _| a + (a - tl) / 2, average2(l, t), tl, 0),
        // 0, and the two values a 4-bit mode can hold beyond 13, which
        // libwebp also treats as black.
        _ => 0xff00_0000,
    }
}

fn color_delta(t: u32, c: u32) -> i32 {
    (i32::from(t as u8 as i8) * i32::from(c as u8 as i8)) >> 5
}

/// Undo one transform: `pixels` at the width it produced, back to `width`.
fn inverse(transform: &Transform, mut pixels: Vec<u32>, width: usize, height: usize) -> Vec<u32> {
    match transform {
        Transform::SubtractGreen => {
            for pixel in &mut pixels {
                let green = (*pixel >> 8) & 0xff;
                *pixel = add_pixels(*pixel, (green << 16) | green);
            }
            pixels
        }
        Transform::Color { bits, elements } => {
            let blocks_wide = div_round_up(width, *bits);
            for (i, pixel) in pixels.iter_mut().enumerate() {
                let (x, y) = (i % width, i / width);
                let element = elements[(y >> bits) * blocks_wide + (x >> bits)];
                let (green, red, blue) =
                    ((*pixel >> 8) & 0xff, (*pixel >> 16) & 0xff, *pixel & 0xff);
                let new_red = (red as i32 + color_delta(element, green)) & 0xff;
                let new_blue = (blue as i32
                    + color_delta(element >> 8, green)
                    + color_delta(element >> 16, new_red as u32))
                    & 0xff;
                *pixel = (*pixel & 0xff00_ff00) | ((new_red as u32) << 16) | new_blue as u32;
            }
            pixels
        }
        Transform::Predictor { bits, modes } => {
            let blocks_wide = div_round_up(width, *bits);
            for i in 0..pixels.len() {
                let (x, y) = (i % width, i / width);
                let prediction = if i == 0 {
                    0xff00_0000
                } else if y == 0 {
                    pixels[i - 1]
                } else if x == 0 {
                    pixels[i - width]
                } else {
                    let mode = (modes[(y >> bits) * blocks_wide + (x >> bits)] >> 8) & 0xf;
                    // The top-right of the last column is, in scan order,
                    // the first pixel of the current row (§3.5.1).
                    predict(
                        mode,
                        pixels[i - 1],
                        pixels[i - width],
                        pixels[i - width + 1],
                        pixels[i - width - 1],
                    )
                };
                pixels[i] = add_pixels(pixels[i], prediction);
            }
            pixels
        }
        Transform::ColorIndexing { bits, table } => {
            let packed_width = div_round_up(width, *bits);
            let per_byte = 1 << bits;
            let index_bits = 8 >> bits;
            let mask = (1_u32 << index_bits) - 1;
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height {
                for x in 0..width {
                    let packed = pixels[y * packed_width + (x >> bits)];
                    let shift = (x & (per_byte - 1)) as u32 * index_bits;
                    let index = ((packed >> 8) >> shift) & mask;
                    out.push(table[index as usize]);
                }
            }
            out
        }
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

    #[test]
    fn bits_are_read_least_significant_first() {
        let mut bits = BitReader::new(&[0b1011_0100, 0xff]);
        assert_eq!(bits.read(2).unwrap(), 0b00);
        assert_eq!(bits.read(3).unwrap(), 0b101);
        assert_eq!(bits.read(5).unwrap(), 0b11_101);
        assert!(bits.read(7).is_err(), "only six bits remain");
    }

    #[test]
    fn a_prefix_code_decodes_canonically() {
        // Lengths 1, 2, 3, 3 give codes 0, 10, 110, 111 — sent MSB first,
        // so in LSB-first bytes symbol 2 (110) arrives as bits 1, 1, 0.
        let code = PrefixCode::new(&[1, 2, 3, 3]).unwrap();
        // Bits in order: 0 | 1 0 | 1 1 0 | 1 1 1.
        let mut bits = BitReader::new(&[0b1101_1010, 0b1]);
        let decoded: Vec<u16> = (0..4).map(|_| code.read(&mut bits).unwrap()).collect();
        assert_eq!(decoded, [0, 1, 2, 3]);
    }

    #[test]
    fn long_codes_take_the_canonical_walk() {
        // 2 symbols of length 1..=10, one of each up to length 11 and two
        // of 12: lengths past FAST_BITS must still decode.
        let mut lengths = vec![0_u8; 16];
        for (i, l) in (1..=11).enumerate() {
            lengths[i] = l;
        }
        lengths[11] = 12;
        lengths[12] = 12;
        let code = PrefixCode::new(&lengths).unwrap();
        // Symbol 12 is the last code: twelve 1 bits.
        let mut bits = BitReader::new(&[0xff, 0x0f]);
        assert_eq!(code.read(&mut bits).unwrap(), 12);
    }

    #[test]
    fn incomplete_and_empty_codes_are_rejected_and_one_symbol_costs_nothing() {
        assert!(PrefixCode::new(&[1, 2]).is_err(), "incomplete");
        assert!(PrefixCode::new(&[1, 1, 1]).is_err(), "over-subscribed");
        assert!(PrefixCode::new(&[0, 0]).is_err(), "empty");
        let single = PrefixCode::new(&[0, 0, 7]).unwrap();
        let mut bits = BitReader::new(&[]);
        assert_eq!(single.read(&mut bits).unwrap(), 2);
    }

    #[test]
    fn prefix_values_follow_the_table() {
        let mut bits = BitReader::new(&[0b1, 0, 0, 0]);
        assert_eq!(prefix_value(&mut bits, 3).unwrap(), 4);
        // Code 4: one extra bit, offset 2 -> 5..6.
        assert_eq!(prefix_value(&mut bits, 4).unwrap(), 6);
        // Code 39: 18 extra bits, offset 3 << 18 -> 786433.. (all zero bits).
        let mut zeros = BitReader::new(&[0; 4]);
        assert_eq!(prefix_value(&mut zeros, 39).unwrap(), 786_433);
    }

    #[test]
    fn predictors_follow_their_definitions() {
        let (l, t, tr, tl) = (0x10_20_30_40, 0x30_40_50_60, 0xff_00_ff_00, 0x00_00_00_00);
        assert_eq!(predict(0, l, t, tr, tl), 0xff00_0000);
        assert_eq!(predict(7, l, t, tr, tl), 0x20_30_40_50);
        // Mode 12 clamps per channel: 0x10 + 0x30 - 0 etc.
        assert_eq!(predict(12, l, t, tr, tl), 0x40_60_80_a0);
        // Mode 13 halves (a - tl) truncating toward zero, as C does:
        // 2 + (2 - 5) / 2 is 2 + -1, not 2 + -2.
        assert_eq!(
            predict(13, 0x00_00_00_02, 0x00_00_00_02, 0, 0x00_00_00_05),
            1
        );
        assert_eq!(predict(14, l, t, tr, tl), 0xff00_0000);
    }

    #[test]
    fn color_indexing_unpacks_bundled_pixels() {
        // Two colours: one bit per pixel, eight per packed pixel.
        let mut table = vec![0xff00_0000, 0xffff_ffff];
        table.resize(256, 0);
        let transform = Transform::ColorIndexing { bits: 3, table };
        let packed = vec![0b1010_0101 << 8];
        let out = inverse(&transform, packed, 8, 1);
        let ones: Vec<bool> = out.iter().map(|&p| p == 0xffff_ffff).collect();
        assert_eq!(ones, [true, false, true, false, false, true, false, true]);
    }

    #[test]
    fn truncated_streams_are_malformed_not_panics() {
        for len in 0..12 {
            let data = vec![0xa5_u8; len];
            let mut bits = BitReader::new(&data);
            assert!(
                decode_image_stream(&mut bits, 4, 4, true).is_err(),
                "{len} bytes"
            );
        }
    }
}
