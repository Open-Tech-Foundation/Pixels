//! DEFLATE decompression (RFC 1951) and the zlib wrapper (RFC 1950).
//!
//! Written from scratch per ADR-0010. Every path returns an error rather than
//! panicking: this code parses attacker-controlled bytes, and
//! `unsafe_code = "forbid"` plus explicit bounds checks mean the classic
//! decompressor failure — an out-of-bounds write through a back-reference —
//! is unrepresentable rather than merely avoided.
//!
//! # Shape
//!
//! A DEFLATE stream is a sequence of blocks, each either stored (raw),
//! fixed-Huffman (a table the spec hardcodes) or dynamic-Huffman (a table the
//! block carries). Literal/length and distance codes decode into either a
//! literal byte or a back-reference into the last 32 KiB of output.
//!
//! # Incremental by construction
//!
//! [`Inflater`] is the real decompressor: bytes are fed in as they arrive and
//! output is drained as it is produced, so a caller never needs the whole
//! stream in memory. That is what lets PNG decode row by row rather than
//! image by image (SPEC §Guarantees 1).
//!
//! Resuming mid-symbol is handled by *checkpointing* rather than by a symbol
//! -level state machine: before each step the bit reader's position is saved,
//! and if the input runs out the step is rewound and retried when more
//! arrives. The decode logic is therefore written once, in its natural
//! straight-line form, and is identical whether the input arrives in one piece
//! or in a thousand.
//!
//! # Bounded output
//!
//! Every entry point takes a byte limit. A short input can expand enormously —
//! a decompression bomb — so the caller states how much output it is prepared
//! to accept, which for PNG is exactly the filtered raster size derived from
//! the header (SPEC §Safety).

use otf_pixels_core::{PixelsError, Result};

use crate::checksum::Adler32;

/// The largest back-reference distance DEFLATE permits, and therefore the
/// amount of already-emitted output that must stay reachable.
const WINDOW: usize = 32 * 1024;

/// Why a decode step stopped early.
///
/// `NeedInput` is not an error: it means the step was rewound and should be
/// retried once more bytes arrive. Keeping it distinct from a real failure is
/// what makes "the stream is truncated" and "the stream is not finished yet"
/// different answers rather than the same one.
#[derive(Debug)]
enum Halt {
    /// The input ran out mid-item; the reader has been rewound.
    NeedInput,
    /// The stream is invalid and no amount of further input will help.
    Fatal(PixelsError),
}

impl From<PixelsError> for Halt {
    fn from(error: PixelsError) -> Self {
        Self::Fatal(error)
    }
}

/// The result of a step that may pause for more input.
type Step<T> = std::result::Result<T, Halt>;

/// Reads bits least-significant-first, as DEFLATE specifies.
///
/// Owns its input so that more can be appended mid-stream. Consumed bytes are
/// dropped by [`BitReader::compact`], which is what keeps a long stream from
/// accumulating in memory.
#[derive(Debug, Default)]
struct BitReader {
    data: Vec<u8>,
    /// Index of the next byte to load.
    position: usize,
    /// Bits not yet consumed, right-aligned.
    bits: u64,
    /// How many bits in `bits` are valid.
    count: u32,
    /// Whether the caller has promised there is no more input.
    ended: bool,
}

/// A saved reader position, so a step that runs out of input can be rewound.
#[derive(Debug, Clone, Copy)]
struct Checkpoint {
    position: usize,
    bits: u64,
    count: u32,
}

impl BitReader {
    /// Append more input.
    fn feed(&mut self, more: &[u8]) {
        self.data.extend_from_slice(more);
    }

    /// Declare that no further input will arrive.
    const fn end(&mut self) {
        self.ended = true;
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            position: self.position,
            bits: self.bits,
            count: self.count,
        }
    }

    fn restore(&mut self, at: Checkpoint) {
        self.position = at.position;
        self.bits = at.bits;
        self.count = at.count;
    }

    /// Drop input that has been consumed and can never be rewound to.
    ///
    /// Safe only at a checkpoint boundary, which is where the caller calls it.
    fn compact(&mut self) {
        if self.position > 0 {
            self.data.drain(..self.position);
            self.position = 0;
        }
    }

    /// Take back every unconsumed byte, including whole bytes that were
    /// pulled into the bit buffer but never used.
    ///
    /// Whole bytes in `bits` are real input; a partial byte is the padding
    /// that ends a DEFLATE stream. Forgetting the former is how a trailer
    /// comes to look one byte short.
    fn drain_unconsumed(&mut self) -> Vec<u8> {
        // Realign first. Mid-byte, `bits` holds the tail of a partially read
        // byte in its low positions, so popping eight bits from the low end
        // would yield a byte straddling two real ones. That padding is exactly
        // what ends a DEFLATE stream, so discarding it is also correct.
        self.align();
        let mut out = Vec::new();
        while self.count >= 8 {
            out.push((self.bits & 0xFF) as u8);
            self.bits >>= 8;
            self.count -= 8;
        }
        if let Some(rest) = self.data.get(self.position..) {
            out.extend_from_slice(rest);
        }
        self.position = self.data.len();
        out
    }

    /// Ensure at least `want` bits are buffered, if the input has them.
    fn fill(&mut self, want: u32) {
        while self.count < want {
            let Some(&byte) = self.data.get(self.position) else {
                break;
            };
            self.bits |= u64::from(byte) << self.count;
            self.position += 1;
            self.count += 8;
        }
    }

    /// Report a shortage as either "wait" or "the stream is truncated".
    fn short(&self) -> Halt {
        if self.ended {
            Halt::Fatal(truncated())
        } else {
            Halt::NeedInput
        }
    }

    /// Consume `n` bits (`n <= 32`).
    fn take(&mut self, n: u32) -> Step<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.fill(n);
        if self.count < n {
            return Err(self.short());
        }
        // `n <= 32`, so the mask fits and the cast cannot lose bits.
        let mask = (1_u64 << n) - 1;
        let value = (self.bits & mask) as u32;
        self.bits >>= n;
        self.count -= n;
        Ok(value)
    }

    /// Look at up to `n` buffered bits without consuming them.
    fn peek(&mut self, n: u32) -> u32 {
        self.fill(n);
        let mask = (1_u64 << n) - 1;
        (self.bits & mask) as u32
    }

    /// Drop `n` already-peeked bits.
    fn skip(&mut self, n: u32) -> Step<()> {
        if self.count < n {
            return Err(self.short());
        }
        self.bits >>= n;
        self.count -= n;
        Ok(())
    }

    /// Discard buffered bits back to a byte boundary.
    fn align(&mut self) {
        let extra = self.count % 8;
        self.bits >>= extra;
        self.count -= extra;
    }

    /// Take up to `n` whole bytes, which must be byte-aligned already.
    ///
    /// Returns fewer than `n` when the input is exhausted, so a stored block
    /// can be copied across as many feeds as it takes.
    fn take_bytes_upto(&mut self, n: usize, out: &mut Vec<u8>) -> usize {
        let mut taken = 0;
        while taken < n {
            // Buffered bits are consumed first, then raw input.
            if self.count >= 8 {
                out.push((self.bits & 0xFF) as u8);
                self.bits >>= 8;
                self.count -= 8;
            } else if let Some(&byte) = self.data.get(self.position) {
                self.position += 1;
                out.push(byte);
            } else {
                break;
            }
            taken += 1;
        }
        taken
    }
}

/// The error for a stream that ended mid-symbol.
fn truncated() -> PixelsError {
    PixelsError::malformed("deflate", "stream ended in the middle of a symbol")
}

/// The maximum code length DEFLATE permits.
const MAX_BITS: usize = 15;

/// A canonical Huffman decoding table.
///
/// Decoding walks the code length by length rather than using a large lookup
/// table: slower per symbol, but obviously correct and with no table-size
/// arithmetic to get wrong on hostile input.
#[derive(Debug, Clone)]
struct Huffman {
    /// `counts[n]` is how many codes have length `n`.
    counts: [u16; MAX_BITS + 1],
    /// Symbols ordered by code length, then by symbol value.
    symbols: Vec<u16>,
}

impl Huffman {
    /// Build a table from per-symbol code lengths (zero meaning "unused").
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0_u16; MAX_BITS + 1];
        for &length in lengths {
            let length = length as usize;
            if length > MAX_BITS {
                return Err(PixelsError::malformed(
                    "deflate",
                    format!("code length {length} exceeds the {MAX_BITS}-bit maximum"),
                ));
            }
            if let Some(slot) = counts.get_mut(length) {
                *slot += 1;
            }
        }
        // Length zero means "no code", so it never participates.
        if let Some(slot) = counts.get_mut(0) {
            *slot = 0;
        }

        // Reject an over-subscribed table: more codes than the tree can hold.
        let mut left = 1_i32;
        for length in 1..=MAX_BITS {
            left <<= 1;
            left -= i32::from(counts.get(length).copied().unwrap_or(0));
            if left < 0 {
                return Err(PixelsError::malformed(
                    "deflate",
                    "Huffman table is over-subscribed",
                ));
            }
        }

        let mut offsets = [0_u16; MAX_BITS + 2];
        for length in 1..=MAX_BITS {
            let next = offsets.get(length).copied().unwrap_or(0)
                + counts.get(length).copied().unwrap_or(0);
            if let Some(slot) = offsets.get_mut(length + 1) {
                *slot = next;
            }
        }

        let total: usize = counts.iter().map(|&c| c as usize).sum();
        let mut symbols = vec![0_u16; total];
        let mut cursor = offsets;
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            let length = length as usize;
            let Some(at) = cursor.get_mut(length) else {
                continue;
            };
            let index = *at as usize;
            *at += 1;
            if let Some(slot) = symbols.get_mut(index) {
                // Symbols are `u16`; a DEFLATE alphabet never exceeds 288.
                *slot = symbol as u16;
            }
        }

        Ok(Self { counts, symbols })
    }

    /// Decode one symbol from `reader`.
    fn decode(&self, reader: &mut BitReader) -> Step<u16> {
        // A short buffer would let zero padding masquerade as real bits, so
        // ask for the maximum first and pause if the stream cannot supply it.
        reader.fill(MAX_BITS as u32);
        if reader.count < MAX_BITS as u32 && !reader.ended {
            return Err(Halt::NeedInput);
        }

        let mut code = 0_i32;
        let mut first = 0_i32;
        let mut index = 0_i32;
        // Peek the maximum, then consume exactly the bits actually used.
        let peeked = reader.peek(MAX_BITS as u32);
        for length in 1..=MAX_BITS {
            // DEFLATE codes are stored most-significant-bit first within the
            // LSB-first bit stream, so the code is rebuilt bit by bit.
            code |= ((peeked >> (length - 1)) & 1) as i32;
            let count = i32::from(self.counts.get(length).copied().unwrap_or(0));
            if code - first < count {
                reader.skip(length as u32)?;
                let position = (index + (code - first)) as usize;
                return self.symbols.get(position).copied().ok_or_else(|| {
                    Halt::Fatal(PixelsError::malformed("deflate", "invalid Huffman symbol"))
                });
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(Halt::Fatal(PixelsError::malformed(
            "deflate",
            "no Huffman code matched within 15 bits",
        )))
    }
}

/// Base lengths for length codes 257..=285.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits for length codes 257..=285.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Base distances for distance codes 0..=29.
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits for distance codes 0..=29.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order code-length codes appear in a dynamic block header.
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// The fixed literal/length table from RFC 1951 §3.2.6.
fn fixed_literal_table() -> Result<Huffman> {
    let mut lengths = [0_u8; 288];
    for (symbol, slot) in lengths.iter_mut().enumerate() {
        *slot = match symbol {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    Huffman::new(&lengths)
}

/// The fixed distance table: 30 codes of 5 bits each.
fn fixed_distance_table() -> Result<Huffman> {
    Huffman::new(&[5_u8; 30])
}

/// Where the decoder is in the block structure.
#[derive(Debug)]
enum State {
    /// Between blocks, about to read a three-bit block header.
    BlockHeader,
    /// Inside a stored block with `remaining` bytes still to copy.
    Stored { remaining: usize, last: bool },
    /// Inside a Huffman-coded block.
    Coded {
        literals: Box<Huffman>,
        distances: Box<Huffman>,
        last: bool,
    },
    /// The final block has ended.
    Done,
}

/// An incremental DEFLATE decompressor.
///
/// Feed bytes with [`Inflater::feed`], drain output with
/// [`Inflater::take_output`], and call [`Inflater::end_of_input`] when the
/// stream is over. Output beyond the configured limit is a malformed-input
/// error, never an allocation.
#[derive(Debug)]
pub struct Inflater {
    reader: BitReader,
    state: State,
    /// Retained history plus not-yet-drained output.
    window: Vec<u8>,
    /// Index into `window` of the first byte not yet handed to the caller.
    pending: usize,
    /// Total bytes ever produced, which is what `limit` bounds.
    produced: usize,
    limit: usize,
}

impl Inflater {
    /// A decompressor that will produce at most `limit` bytes.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            reader: BitReader::default(),
            state: State::BlockHeader,
            window: Vec::new(),
            pending: 0,
            produced: 0,
            limit,
        }
    }

    /// Supply more compressed bytes.
    pub fn feed(&mut self, data: &[u8]) {
        self.reader.feed(data);
    }

    /// Declare that no further compressed bytes will arrive.
    pub const fn end_of_input(&mut self) {
        self.reader.end();
    }

    /// Whether the final block has been decoded.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        matches!(self.state, State::Done)
    }

    /// How many bytes have been produced in total.
    #[must_use]
    pub const fn produced(&self) -> usize {
        self.produced
    }

    /// How many output bytes are currently held — history plus undrained.
    ///
    /// Exposed so callers can assert the streaming guarantee rather than
    /// trust it: a drained inflater retains the 32 KiB window and no more.
    #[must_use]
    pub fn retained(&self) -> usize {
        self.window.len()
    }

    /// Take back compressed bytes that were fed in but are not part of the
    /// DEFLATE stream — for zlib, that is the Adler-32 trailer.
    fn drain_unconsumed_input(&mut self) -> Vec<u8> {
        self.reader.drain_unconsumed()
    }

    /// Take everything decoded since the last call.
    ///
    /// Draining is what bounds memory: the decompressor keeps only the 32 KiB
    /// of history that back-references can still reach, so a caller that
    /// drains regularly holds a fixed amount regardless of stream length.
    pub fn take_output(&mut self) -> Vec<u8> {
        // Copied, not moved: delivered bytes are still back-reference history
        // until they fall out of the window. Moving them out is how a
        // long-range reference comes to point at nothing.
        let out = self.window.get(self.pending..).unwrap_or(&[]).to_vec();
        if self.window.len() > WINDOW {
            self.window.drain(..self.window.len() - WINDOW);
        }
        self.pending = self.window.len();
        out
    }

    /// Decode as far as the buffered input allows.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for an invalid stream, for output
    /// exceeding the limit, or for a stream that ended mid-symbol after
    /// [`Inflater::end_of_input`].
    pub fn decode(&mut self) -> Result<()> {
        loop {
            if matches!(self.state, State::Done) {
                return Ok(());
            }
            let at = self.reader.checkpoint();
            match self.step() {
                Ok(()) => {
                    // Only past a completed step is consumed input unreachable.
                    self.reader.compact();
                }
                Err(Halt::NeedInput) => {
                    self.reader.restore(at);
                    return Ok(());
                }
                Err(Halt::Fatal(error)) => return Err(error),
            }
        }
    }

    /// Perform one unit of work: a block header, a stored run, or a symbol.
    fn step(&mut self) -> Step<()> {
        match &self.state {
            State::Done => Ok(()),
            State::BlockHeader => {
                let last = self.reader.take(1)? == 1;
                let kind = self.reader.take(2)?;
                self.state = match kind {
                    0 => {
                        self.reader.align();
                        let length = self.reader.take(16)? as usize;
                        let complement = self.reader.take(16)? as usize;
                        if length ^ 0xFFFF != complement {
                            return Err(Halt::Fatal(PixelsError::malformed(
                                "deflate",
                                "stored block length does not match its complement",
                            )));
                        }
                        self.check_limit(length)?;
                        State::Stored {
                            remaining: length,
                            last,
                        }
                    }
                    1 => State::Coded {
                        literals: Box::new(fixed_literal_table()?),
                        distances: Box::new(fixed_distance_table()?),
                        last,
                    },
                    2 => {
                        let (literals, distances) = read_dynamic_tables(&mut self.reader)?;
                        State::Coded {
                            literals: Box::new(literals),
                            distances: Box::new(distances),
                            last,
                        }
                    }
                    _ => {
                        return Err(Halt::Fatal(PixelsError::malformed(
                            "deflate",
                            "reserved block type 3",
                        )));
                    }
                };
                Ok(())
            }
            State::Stored { remaining, last } => {
                let (remaining, last) = (*remaining, *last);
                if remaining == 0 {
                    self.state = if last {
                        State::Done
                    } else {
                        State::BlockHeader
                    };
                    return Ok(());
                }
                let taken = self.reader.take_bytes_upto(remaining, &mut self.window);
                self.produced += taken;
                if taken == 0 {
                    return Err(self.reader.short());
                }
                self.state = State::Stored {
                    remaining: remaining - taken,
                    last,
                };
                Ok(())
            }
            State::Coded { .. } => self.step_coded(),
        }
    }

    /// Decode one literal or back-reference from the current coded block.
    fn step_coded(&mut self) -> Step<()> {
        let State::Coded {
            literals,
            distances,
            last,
        } = &self.state
        else {
            return Ok(());
        };
        // Cloning the table handles would fight the borrow checker for no
        // gain; the tables are read-only, so they are taken out and put back.
        let literals = literals.clone();
        let distances = distances.clone();
        let last = *last;

        let symbol = literals.decode(&mut self.reader)?;
        match symbol {
            // A literal byte.
            0..=255 => {
                self.check_limit(1)?;
                self.window.push(symbol as u8);
                self.produced += 1;
            }
            // End of block.
            256 => {
                self.state = if last {
                    State::Done
                } else {
                    State::BlockHeader
                };
            }
            // A back-reference.
            257..=285 => {
                let index = symbol as usize - 257;
                let base = LENGTH_BASE.get(index).copied().ok_or_else(|| {
                    Halt::Fatal(PixelsError::malformed("deflate", "invalid length code"))
                })?;
                let extra = LENGTH_EXTRA.get(index).copied().unwrap_or(0);
                let length = base as usize + self.reader.take(u32::from(extra))? as usize;

                let distance_symbol = distances.decode(&mut self.reader)? as usize;
                let distance_base =
                    DISTANCE_BASE.get(distance_symbol).copied().ok_or_else(|| {
                        Halt::Fatal(PixelsError::malformed("deflate", "invalid distance code"))
                    })?;
                let distance_extra = DISTANCE_EXTRA.get(distance_symbol).copied().unwrap_or(0);
                let distance =
                    distance_base as usize + self.reader.take(u32::from(distance_extra))? as usize;

                // The reference must land inside what has already been
                // emitted. This is the check whose absence is the classic
                // decompressor out-of-bounds read. Comparing against the
                // retained window rather than total output is what makes it
                // still correct once old output has been drained away.
                if distance == 0 || distance > self.window.len() {
                    return Err(Halt::Fatal(PixelsError::malformed(
                        "deflate",
                        format!(
                            "back-reference of distance {distance} points before the start of \
                             the {} bytes decoded so far",
                            self.produced
                        ),
                    )));
                }
                self.check_limit(length)?;

                // Copied byte by byte on purpose: overlapping references are
                // legal and are how DEFLATE encodes runs, so the source may
                // include bytes this very loop is writing.
                let start = self.window.len() - distance;
                for offset in 0..length {
                    let byte = self.window.get(start + offset).copied().ok_or_else(|| {
                        Halt::Fatal(PixelsError::malformed(
                            "deflate",
                            "back-reference read out of range",
                        ))
                    })?;
                    self.window.push(byte);
                }
                self.produced += length;
            }
            _ => {
                return Err(Halt::Fatal(PixelsError::malformed(
                    "deflate",
                    format!("literal/length symbol {symbol} is out of range"),
                )));
            }
        }
        Ok(())
    }

    /// Reject output that would exceed the limit.
    fn check_limit(&self, adding: usize) -> Step<()> {
        if self.produced.saturating_add(adding) > self.limit {
            return Err(Halt::Fatal(PixelsError::malformed(
                "deflate",
                format!(
                    "stream expands beyond the {} byte limit implied by the image header",
                    self.limit
                ),
            )));
        }
        Ok(())
    }
}

/// Read the code-length-coded tables of a dynamic block.
fn read_dynamic_tables(reader: &mut BitReader) -> Step<(Huffman, Huffman)> {
    let literal_count = reader.take(5)? as usize + 257;
    let distance_count = reader.take(5)? as usize + 1;
    let code_length_count = reader.take(4)? as usize + 4;
    if literal_count > 288 || distance_count > 30 {
        return Err(Halt::Fatal(PixelsError::malformed(
            "deflate",
            "dynamic block declares too many codes",
        )));
    }

    let mut code_lengths = [0_u8; 19];
    for index in 0..code_length_count {
        let bits = reader.take(3)? as u8;
        let Some(&position) = CODE_LENGTH_ORDER.get(index) else {
            break;
        };
        if let Some(slot) = code_lengths.get_mut(position) {
            *slot = bits;
        }
    }
    let code_length_table = Huffman::new(&code_lengths)?;

    // The two tables are coded as one run, so repeats may straddle the seam.
    let total = literal_count + distance_count;
    let mut lengths = vec![0_u8; total];
    let mut index = 0;
    while index < total {
        let symbol = code_length_table.decode(reader)?;
        match symbol {
            0..=15 => {
                if let Some(slot) = lengths.get_mut(index) {
                    *slot = symbol as u8;
                }
                index += 1;
            }
            16 => {
                // Repeat the previous length 3..=6 times.
                let previous = index
                    .checked_sub(1)
                    .and_then(|i| lengths.get(i).copied())
                    .ok_or_else(|| {
                        Halt::Fatal(PixelsError::malformed(
                            "deflate",
                            "repeat code with no previous length",
                        ))
                    })?;
                let repeat = reader.take(2)? as usize + 3;
                fill(&mut lengths, &mut index, previous, repeat, total)?;
            }
            17 => {
                let repeat = reader.take(3)? as usize + 3;
                fill(&mut lengths, &mut index, 0, repeat, total)?;
            }
            18 => {
                let repeat = reader.take(7)? as usize + 11;
                fill(&mut lengths, &mut index, 0, repeat, total)?;
            }
            _ => {
                return Err(Halt::Fatal(PixelsError::malformed(
                    "deflate",
                    "invalid code length symbol",
                )));
            }
        }
    }

    let (literal_lengths, distance_lengths) = lengths.split_at(literal_count);
    let literals = Huffman::new(literal_lengths)?;
    let distances = Huffman::new(distance_lengths)?;
    Ok((literals, distances))
}

/// Write `value` into `lengths` `repeat` times, refusing to overrun.
fn fill(lengths: &mut [u8], index: &mut usize, value: u8, repeat: usize, total: usize) -> Step<()> {
    if *index + repeat > total {
        return Err(Halt::Fatal(PixelsError::malformed(
            "deflate",
            "code length repeat runs past the end of the table",
        )));
    }
    for _ in 0..repeat {
        if let Some(slot) = lengths.get_mut(*index) {
            *slot = value;
        }
        *index += 1;
    }
    Ok(())
}

/// Decompress a raw DEFLATE stream, refusing to exceed `limit` output bytes.
///
/// This is [`Inflater`] used all at once, which is what a caller that already
/// holds the whole stream wants.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for any invalid or truncated stream, and
/// for output exceeding `limit` — a decompression bomb is malformed input, not
/// a resource the caller must survive.
pub fn inflate_to(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut inflater = Inflater::new(limit);
    inflater.feed(data);
    inflater.end_of_input();
    inflater.decode()?;
    if !inflater.is_finished() {
        return Err(truncated());
    }
    Ok(inflater.take_output())
}

/// An incremental zlib (RFC 1950) decompressor.
///
/// Wraps [`Inflater`] with the two-byte header, the running Adler-32 and the
/// four-byte trailer. The checksum is computed as output is produced, so
/// verifying it costs nothing extra and does not require retaining the output.
#[derive(Debug)]
pub struct ZlibStream {
    header: Vec<u8>,
    inflater: Inflater,
    adler: Adler32,
    /// The trailer, accumulated once the deflate stream is finished.
    trailer: Vec<u8>,
    ended: bool,
}

impl ZlibStream {
    /// A decompressor that will produce at most `limit` bytes.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            header: Vec::with_capacity(2),
            inflater: Inflater::new(limit),
            adler: Adler32::new(),
            trailer: Vec::with_capacity(4),
            ended: false,
        }
    }

    /// Supply more compressed bytes and return whatever they decoded to.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a bad header, an unsupported
    /// compression method, or any DEFLATE error.
    pub fn push(&mut self, mut data: &[u8]) -> Result<Vec<u8>> {
        // The two-byte header must be complete before anything else happens,
        // and it may itself be split across feeds.
        while self.header.len() < 2 {
            let Some((&byte, rest)) = data.split_first() else {
                return Ok(Vec::new());
            };
            self.header.push(byte);
            data = rest;
            if self.header.len() == 2 {
                validate_zlib_header(&self.header)?;
            }
        }

        if self.inflater.is_finished() {
            self.collect_trailer(data);
            return Ok(Vec::new());
        }

        self.inflater.feed(data);
        self.inflater.decode()?;
        let out = self.inflater.take_output();
        self.adler.update(&out);

        // Anything the deflate stream did not consume is the trailer. Taking
        // it from the reader rather than slicing `data` keeps this correct
        // when the trailer straddles two feeds.
        if self.inflater.is_finished() {
            let leftover = self.inflater.drain_unconsumed_input();
            self.collect_trailer(&leftover);
        }
        Ok(out)
    }

    /// Keep up to four trailing bytes, which carry the Adler-32.
    fn collect_trailer(&mut self, data: &[u8]) {
        for &byte in data {
            if self.trailer.len() < 4 {
                self.trailer.push(byte);
            }
        }
    }

    /// Declare the input over and verify the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the stream ended mid-symbol, if
    /// the trailer is missing, or if the Adler-32 does not match.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        if self.ended {
            return Ok(Vec::new());
        }
        self.ended = true;
        if self.header.len() < 2 {
            return Err(PixelsError::malformed(
                "zlib",
                "stream is shorter than its 2-byte header",
            ));
        }
        self.inflater.end_of_input();
        self.inflater.decode()?;
        let out = self.inflater.take_output();
        self.adler.update(&out);
        if !self.inflater.is_finished() {
            return Err(truncated());
        }
        let leftover = self.inflater.drain_unconsumed_input();
        self.collect_trailer(&leftover);

        if self.trailer.len() < 4 {
            return Err(PixelsError::malformed(
                "zlib",
                "stream is missing its Adler-32 trailer",
            ));
        }
        let expected = u32::from_be_bytes([
            self.trailer.first().copied().unwrap_or(0),
            self.trailer.get(1).copied().unwrap_or(0),
            self.trailer.get(2).copied().unwrap_or(0),
            self.trailer.get(3).copied().unwrap_or(0),
        ]);
        let actual = self.adler.finish();
        if actual != expected {
            return Err(PixelsError::malformed(
                "zlib",
                format!(
                    "Adler-32 mismatch: stream declares {expected:#010x}, data is {actual:#010x}"
                ),
            ));
        }
        Ok(out)
    }
}

/// Validate the two-byte zlib header (RFC 1950 §2.2).
fn validate_zlib_header(header: &[u8]) -> Result<()> {
    let (&cmf, &flg) = match (header.first(), header.get(1)) {
        (Some(cmf), Some(flg)) => (cmf, flg),
        _ => {
            return Err(PixelsError::malformed(
                "zlib",
                "stream is shorter than its 2-byte header",
            ));
        }
    };
    if cmf & 0x0F != 8 {
        return Err(PixelsError::malformed(
            "zlib",
            format!("compression method {} is not deflate", cmf & 0x0F),
        ));
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(PixelsError::malformed(
            "zlib",
            "header check bits are wrong",
        ));
    }
    if flg & 0x20 != 0 {
        // A preset dictionary would change what the back-references mean, and
        // PNG forbids it (PNG spec §10.3).
        return Err(PixelsError::malformed(
            "zlib",
            "preset dictionaries are not supported",
        ));
    }
    Ok(())
}

/// Decompress a zlib stream (RFC 1950), verifying its Adler-32.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for a bad header, an unsupported
/// compression method, a checksum mismatch, or any DEFLATE error.
pub fn zlib_decompress(data: &[u8], limit: usize) -> Result<Vec<u8>> {
    let mut stream = ZlibStream::new(limit);
    let mut out = stream.push(data)?;
    out.extend_from_slice(&stream.finish()?);
    Ok(out)
}
#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// A stored-block DEFLATE stream wrapping `payload`.
    fn stored_stream(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0x01];
        let length = payload.len() as u16;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(&(!length).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn stored_blocks_round_trip() {
        let payload = b"the quick brown fox";
        let out = inflate_to(&stored_stream(payload), 1024).unwrap();
        assert_eq!(out, payload);
    }

    #[test]
    fn an_empty_stored_block_yields_nothing() {
        assert_eq!(
            inflate_to(&stored_stream(b""), 16).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn a_stored_block_with_a_bad_complement_is_rejected() {
        let mut stream = stored_stream(b"abc");
        stream[3] ^= 0xFF;
        let err = inflate_to(&stream, 1024).unwrap_err();
        assert!(err.to_string().contains("complement"), "{err}");
    }

    #[test]
    fn fixed_huffman_decodes_a_known_stream() {
        // zlib's output for "hello" at the default level, minus the wrapper:
        // a single fixed-Huffman block.
        let stream = [0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x07, 0x00];
        assert_eq!(inflate_to(&stream, 64).unwrap(), b"hello");
    }

    #[test]
    fn zlib_wrapped_streams_verify_their_checksum() {
        // zlib -9 output for "hello world".
        let stream = [
            0x78, 0xDA, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0x28, 0xCF, 0x2F, 0xCA, 0x49, 0x01,
            0x00, 0x1A, 0x0B, 0x04, 0x5D,
        ];
        assert_eq!(zlib_decompress(&stream, 64).unwrap(), b"hello world");
    }

    #[test]
    fn a_corrupted_adler_is_reported() {
        let mut stream = vec![
            0x78, 0xDA, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0x28, 0xCF, 0x2F, 0xCA, 0x49, 0x01,
            0x00, 0x1A, 0x0B, 0x04, 0x5D,
        ];
        let last = stream.len() - 1;
        stream[last] ^= 0xFF;
        let err = zlib_decompress(&stream, 64).unwrap_err();
        assert!(err.to_string().contains("Adler-32"), "{err}");
    }

    #[test]
    fn zlib_headers_are_validated() {
        assert!(zlib_decompress(&[], 16).is_err(), "empty");
        assert!(zlib_decompress(&[0x78], 16).is_err(), "one byte");
        // Compression method 7 is not deflate.
        assert!(zlib_decompress(&[0x77, 0x00, 0x00], 16).is_err());
        // Header check bits that do not divide by 31.
        assert!(zlib_decompress(&[0x78, 0x00, 0x00], 16).is_err());
        // Preset dictionary flag set (0x78, 0x3F is divisible by 31).
        let err = zlib_decompress(&[0x78, 0x3F, 0x00], 16).unwrap_err();
        assert!(err.to_string().contains("dictionar"), "{err}");
    }

    #[test]
    fn reserved_block_type_three_is_rejected() {
        // BFINAL=1, BTYPE=3 packs to 0b111 in the first byte.
        let err = inflate_to(&[0x07], 16).unwrap_err();
        assert!(err.to_string().contains("reserved"), "{err}");
    }

    #[test]
    fn a_back_reference_before_the_start_is_rejected() {
        // A fixed-Huffman block whose very first symbol is a length code:
        // nothing has been emitted, so the distance points before the start of
        // the output. This is the classic decompressor out-of-bounds read, and
        // the bytes are hand-assembled because no real encoder emits it.
        //
        // BFINAL=1, BTYPE=01, symbol 257 (7-bit code 0000001), distance code 0.
        let err = inflate_to(&[0x03, 0x02], 1024).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
        assert!(err.to_string().contains("back-reference"), "{err}");
    }

    #[test]
    fn output_beyond_the_limit_is_malformed_not_an_allocation() {
        // A decompression bomb: 65535 zero bytes from a tiny stored block.
        let bomb = stored_stream(&vec![0_u8; 65535]);
        let err = inflate_to(&bomb, 1024).unwrap_err();
        assert_eq!(err.code(), otf_pixels_core::ErrorCode::Malformed);
        assert!(err.to_string().contains("limit"), "{err}");
        // The same stream within a generous limit is fine.
        assert_eq!(inflate_to(&bomb, 65535).unwrap().len(), 65535);
    }

    #[test]
    fn every_truncation_of_a_valid_stream_is_an_error_not_a_panic() {
        let full = [
            0x78, 0xDA, 0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x57, 0x28, 0xCF, 0x2F, 0xCA, 0x49, 0x01,
            0x00, 0x1A, 0x0B, 0x04, 0x5D,
        ];
        for len in 0..full.len() {
            // Must not panic. Some prefixes decode a valid shorter stream, so
            // success is acceptable; a crash is not.
            let _ = zlib_decompress(&full[..len], 4096);
        }
        assert!(
            zlib_decompress(&full, 4096).is_ok(),
            "the untruncated stream still works"
        );
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        // A cheap deterministic sweep; the real corpus fuzzing is in the
        // crate's fuzz tests. This is here so the module is never committed in
        // a state that crashes on trivial garbage.
        let mut state = 0x1234_5678_u32;
        for _ in 0..2000 {
            let mut bytes = Vec::new();
            for _ in 0..32 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                bytes.push((state >> 24) as u8);
            }
            let _ = inflate_to(&bytes, 4096);
            let _ = zlib_decompress(&bytes, 4096);
        }
    }

    #[test]
    fn over_subscribed_huffman_tables_are_rejected() {
        // Three one-bit codes cannot form a prefix code.
        assert!(Huffman::new(&[1, 1, 1]).is_err());
        // Two one-bit codes can.
        assert!(Huffman::new(&[1, 1]).is_ok());
        // A length beyond 15 bits is out of spec.
        assert!(Huffman::new(&[16]).is_err());
    }

    #[test]
    fn overlapping_back_references_encode_runs() {
        // A run is encoded as a copy from distance 1, so the source overlaps
        // the destination and includes bytes the copy loop is itself writing.
        // Produced by zlib, not by hand.
        let stream = [0x4B, 0x4C, 0x84, 0x00, 0x00];
        assert_eq!(inflate_to(&stream, 64).unwrap(), b"aaaaaaaa");
    }

    /// A zlib stream of `data` at the given level, via our own compressor.
    fn compress(data: &[u8], level: u8) -> Vec<u8> {
        crate::deflate::zlib_compress(data, crate::deflate::Level::new(level).unwrap()).unwrap()
    }

    /// The raw DEFLATE body of that stream, for tests that drive `Inflater`
    /// directly rather than through the zlib wrapper.
    fn deflate_body(data: &[u8], level: u8) -> Vec<u8> {
        compress(data, level).split_off(2)
    }

    #[test]
    fn feeding_one_byte_at_a_time_decodes_identically() {
        // The property that makes streaming real: how the input is chunked
        // must not change the output. One byte at a time is the worst case,
        // because every symbol is interrupted.
        for level in [0_u8, 1, 6, 9] {
            let original = b"the quick brown fox jumps over the lazy dog. ".repeat(120);
            let stream = compress(&original, level);

            let mut zlib = ZlibStream::new(1 << 20);
            let mut out = Vec::new();
            for byte in &stream {
                out.extend_from_slice(&zlib.push(std::slice::from_ref(byte)).unwrap());
            }
            out.extend_from_slice(&zlib.finish().unwrap());
            assert_eq!(
                out, original,
                "level {level} differed when fed byte by byte"
            );
        }
    }

    #[test]
    fn every_chunk_size_decodes_identically() {
        let original: Vec<u8> = (0..40_000).map(|i| ((i * 7) % 251) as u8).collect();
        let stream = compress(&original, 6);
        for chunk in [1, 2, 3, 7, 64, 1024, 65_536] {
            let mut zlib = ZlibStream::new(1 << 20);
            let mut out = Vec::new();
            for piece in stream.chunks(chunk) {
                out.extend_from_slice(&zlib.push(piece).unwrap());
            }
            out.extend_from_slice(&zlib.finish().unwrap());
            assert_eq!(out, original, "chunk size {chunk} differed");
        }
    }

    #[test]
    fn a_drained_inflater_retains_only_its_window() {
        // The whole point of the restructure: decoding a stream far larger
        // than the window must not accumulate it. A caller that drains holds
        // the window and nothing more.
        let original = vec![0_u8; 8 * 1024 * 1024];
        let stream = deflate_body(&original, 9);

        let mut inflater = Inflater::new(16 * 1024 * 1024);
        let mut total = 0_usize;
        for piece in stream.chunks(4096) {
            inflater.feed(piece);
            inflater.decode().unwrap();
            total += inflater.take_output().len();
            assert!(
                inflater.retained() <= WINDOW + 4096,
                "retained {} bytes after {total} of output",
                inflater.retained()
            );
        }
        inflater.end_of_input();
        inflater.decode().unwrap();
        total += inflater.take_output().len();
        assert_eq!(total, original.len());
        assert_eq!(inflater.produced(), original.len());
    }

    #[test]
    fn a_back_reference_reaching_across_a_drain_still_resolves() {
        // Draining discards delivered output, so a back-reference pointing
        // into it must still find its bytes in the retained window. A run
        // longer than the window is where that breaks if it is going to.
        let original = b"abcdefgh".repeat(200_000);
        let stream = deflate_body(&original, 9);

        let mut inflater = Inflater::new(4 * 1024 * 1024);
        let mut out = Vec::new();
        for piece in stream.chunks(777) {
            inflater.feed(piece);
            inflater.decode().unwrap();
            out.extend_from_slice(&inflater.take_output());
        }
        inflater.end_of_input();
        inflater.decode().unwrap();
        out.extend_from_slice(&inflater.take_output());
        assert_eq!(out, original);
    }

    #[test]
    fn an_unfinished_stream_is_not_reported_as_complete() {
        // Half a stream must read as "not done", never as a short success —
        // that distinction is what stops a truncated PNG decoding to a
        // partial image.
        let stream = deflate_body(&vec![7_u8; 100_000], 6);
        let mut inflater = Inflater::new(1 << 20);
        inflater.feed(&stream[..stream.len() / 2]);
        inflater.decode().unwrap();
        assert!(!inflater.is_finished());

        inflater.end_of_input();
        assert!(inflater.decode().is_err() || !inflater.is_finished());
    }

    #[test]
    fn the_limit_is_enforced_incrementally_not_at_the_end() {
        // A bomb must be refused while decoding, not after the allocation it
        // was trying to provoke.
        let stream = deflate_body(&vec![0_u8; 4 * 1024 * 1024], 9);
        let mut inflater = Inflater::new(1024);
        inflater.feed(&stream);
        let error = inflater.decode().unwrap_err();
        assert_eq!(
            error.code(),
            otf_pixels_core::ErrorCode::Malformed,
            "{error}"
        );
        assert!(
            inflater.produced() <= 1024,
            "produced {} bytes",
            inflater.produced()
        );
    }
}
