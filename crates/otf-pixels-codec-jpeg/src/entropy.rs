//! The byte and bit layer beneath the scan: buffered reads, byte stuffing,
//! and marker detection.
//!
//! # Why one reader does both jobs
//!
//! Entropy-coded data is not a segment with a length — it runs until the next
//! marker, and the only way to find that marker is to decode until the bits
//! run out. So the same reader has to serve `read_exact` for segment payloads
//! and bit-at-a-time reads for the scan, and has to notice a marker mid-scan
//! and stop feeding bits rather than decode the marker as data.
//!
//! # Byte stuffing
//!
//! `0xFF` is the marker prefix, so entropy data escapes a literal `0xFF` as
//! `FF 00`. A `0xFF` followed by anything else is a real marker and ends the
//! scan; a run of `0xFF` bytes is padding before one.

use crate::huffman::HuffmanTable;
use otf_pixels_core::{PixelsError, Result, Source};

/// Bytes pulled from the source per refill.
const BUFFER: usize = 16 * 1024;

/// A buffered, marker-aware reader over a JPEG stream.
pub struct Reader<S: Source> {
    source: S,
    buffer: Box<[u8]>,
    /// How far into `buffer` reading has got.
    position: usize,
    /// How much of `buffer` holds bytes from the source.
    filled: usize,
    /// Set once the source has returned end-of-input.
    drained: bool,
    /// Bits pulled from the entropy stream but not yet consumed, right-aligned.
    accumulator: u32,
    /// How many bits of `accumulator` are live.
    bits: u32,
    /// A marker met while feeding bits, held until the scan loop asks for it.
    marker: Option<u8>,
    /// Every byte pulled so far, when the caller may need to replay them.
    ///
    /// Progressive JPEG is decoded by a wrapped codec (ADR-0004) that wants
    /// the stream from byte zero, but nothing announces "progressive" until
    /// the `SOF2` marker — by which point the header is already consumed and a
    /// forward-only source cannot give it back. Recording until the frame type
    /// is known is what makes the handover possible; the tape is dropped the
    /// moment a baseline frame is confirmed, so an ordinary decode carries
    /// nothing.
    tape: Option<Vec<u8>>,
}

impl<S: Source> std::fmt::Debug for Reader<S> {
    /// Deliberately omits the buffer: 16 KiB of entropy data in a panic
    /// message or a test failure is noise, not information.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("buffered", &(self.filled - self.position))
            .field("bits", &self.bits)
            .field("marker", &self.marker)
            .finish_non_exhaustive()
    }
}

impl<S: Source> Reader<S> {
    /// Wrap `source`.
    pub fn new(source: S) -> Self {
        Self {
            source,
            buffer: vec![0_u8; BUFFER].into_boxed_slice(),
            position: 0,
            filled: 0,
            drained: false,
            accumulator: 0,
            bits: 0,
            marker: None,
            tape: None,
        }
    }

    /// Start recording pulled bytes, so the stream can be replayed.
    pub fn record(&mut self) {
        self.tape = Some(Vec::new());
    }

    /// Stop recording and discard what was recorded.
    pub fn forget(&mut self) {
        self.tape = None;
    }

    /// Consume the reader, returning the bytes needed to replay the stream
    /// from where recording began, and the source positioned after them.
    ///
    /// The replay is everything pulled while recording plus everything read
    /// from the source into the buffer but not yet pulled — otherwise the
    /// buffered tail would be lost.
    pub fn into_replay(self) -> (Vec<u8>, S) {
        let mut replay = self.tape.unwrap_or_default();
        if let Some(pending) = self.buffer.get(self.position..self.filled) {
            replay.extend_from_slice(pending);
        }
        (replay, self.source)
    }

    /// The next byte of the stream, or `None` at end of input.
    fn pull(&mut self) -> Result<Option<u8>> {
        if self.position >= self.filled {
            if self.drained {
                return Ok(None);
            }
            let read = self.source.read(&mut self.buffer)?;
            if read == 0 {
                self.drained = true;
                return Ok(None);
            }
            self.position = 0;
            self.filled = read.min(self.buffer.len());
        }
        let byte = self.buffer.get(self.position).copied();
        self.position += 1;
        if let (Some(tape), Some(byte)) = (self.tape.as_mut(), byte) {
            tape.push(byte);
        }
        Ok(byte)
    }

    /// Fill `out` completely.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the stream ends first — a
    /// truncated JPEG is malformed input, not an I/O fault.
    pub fn read_exact(&mut self, out: &mut [u8]) -> Result<()> {
        for (index, slot) in out.iter_mut().enumerate() {
            match self.pull()? {
                Some(byte) => *slot = byte,
                None => {
                    return Err(PixelsError::malformed(
                        "jpeg",
                        format!("stream ended after {index} of {} expected bytes", out.len()),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Read a big-endian `u16`.
    ///
    /// # Errors
    ///
    /// As [`Reader::read_exact`].
    pub fn read_u16(&mut self) -> Result<u16> {
        let mut bytes = [0_u8; 2];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_be_bytes(bytes))
    }

    /// Advance to the next marker and return its code.
    ///
    /// Any bytes before the marker are skipped, which is what recovers a
    /// stream whose scan ended in padding or garbage.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the stream ends before a marker
    /// is found.
    pub fn next_marker(&mut self) -> Result<u8> {
        // A marker already met while decoding bits is the next one; taking it
        // here is what lets the scan loop hand control back to the segment
        // loop without re-reading anything.
        if let Some(code) = self.marker.take() {
            return Ok(code);
        }
        let mut seen_prefix = false;
        loop {
            let Some(byte) = self.pull()? else {
                return Err(PixelsError::malformed(
                    "jpeg",
                    "stream ended while looking for a marker",
                ));
            };
            match byte {
                0xFF => seen_prefix = true,
                // `FF 00` is a stuffed data byte, not a marker; so is a `00`
                // with no prefix. Either way, keep looking.
                0x00 => seen_prefix = false,
                code if seen_prefix => return Ok(code),
                _ => {}
            }
        }
    }

    /// Skip a segment whose two-byte length has not yet been read.
    ///
    /// # Errors
    ///
    /// As [`Reader::read_exact`], plus [`PixelsError::Malformed`] if the
    /// declared length is below the two bytes it counts itself.
    pub fn skip_segment(&mut self) -> Result<()> {
        let length = self.read_u16()?;
        let Some(payload) = length.checked_sub(2) else {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("segment declares length {length}, below its own length field"),
            ));
        };
        for _ in 0..payload {
            if self.pull()?.is_none() {
                return Err(PixelsError::malformed(
                    "jpeg",
                    "stream ended inside a skipped segment",
                ));
            }
        }
        Ok(())
    }

    /// Read a segment's payload: everything after its two-byte length.
    ///
    /// # Errors
    ///
    /// As [`Reader::skip_segment`].
    pub fn read_segment(&mut self) -> Result<Vec<u8>> {
        let length = self.read_u16()?;
        let Some(payload) = length.checked_sub(2) else {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("segment declares length {length}, below its own length field"),
            ));
        };
        let mut bytes = vec![0_u8; payload as usize];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    /// The next byte of entropy-coded data, unstuffing `FF 00`.
    ///
    /// Returns `None` once a marker has been reached; the marker is retained
    /// for [`Reader::next_marker`].
    fn entropy_byte(&mut self) -> Result<Option<u8>> {
        if self.marker.is_some() {
            return Ok(None);
        }
        let Some(byte) = self.pull()? else {
            // Running out of bytes mid-scan is a truncated file. Treating it
            // as an end-of-image marker lets the rows decoded so far stand,
            // which is what every viewer does with a partial download.
            self.marker = Some(crate::format::marker::EOI);
            return Ok(None);
        };
        if byte != 0xFF {
            return Ok(Some(byte));
        }
        loop {
            match self.pull()? {
                // A stuffed literal `0xFF`.
                Some(0x00) => return Ok(Some(0xFF)),
                // Padding before a marker: keep looking for the marker itself.
                Some(0xFF) => {}
                Some(code) => {
                    self.marker = Some(code);
                    return Ok(None);
                }
                None => {
                    self.marker = Some(crate::format::marker::EOI);
                    return Ok(None);
                }
            }
        }
    }

    /// Ensure at least `wanted` bits are buffered, padding with zeros past a
    /// marker.
    fn fill(&mut self, wanted: u32) -> Result<()> {
        while self.bits < wanted {
            // Past a marker there is no more data for this scan. Zeros are
            // what libjpeg feeds too: the alternative is refusing to show the
            // part of a truncated image that did arrive.
            let byte = self.entropy_byte()?.unwrap_or(0);
            self.accumulator = (self.accumulator << 8) | u32::from(byte);
            self.bits += 8;
        }
        Ok(())
    }

    /// The next `count` bits without consuming them.
    fn peek(&self, count: u32) -> u32 {
        if count == 0 || count > self.bits {
            return 0;
        }
        (self.accumulator >> (self.bits - count)) & ((1 << count) - 1)
    }

    /// Drop `count` bits already peeked at.
    fn consume(&mut self, count: u32) {
        self.bits = self.bits.saturating_sub(count);
        self.accumulator &= (1_u32 << self.bits) - 1;
    }

    /// Read `count` bits as an unsigned value.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the source fails.
    pub fn get_bits(&mut self, count: u32) -> Result<u32> {
        if count == 0 {
            return Ok(0);
        }
        self.fill(count)?;
        let value = self.peek(count);
        self.consume(count);
        Ok(value)
    }

    /// Decode one Huffman-coded symbol.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the bits match no code in
    /// `table`.
    pub fn decode(&mut self, table: &HuffmanTable) -> Result<u8> {
        // 16 is the longest legal JPEG code, so one fill covers any symbol.
        self.fill(16)?;
        if let Some((length, symbol)) = table.lookup(self.peek(8) as u8) {
            self.consume(length);
            return Ok(symbol);
        }
        // Longer than the lookup covers: extend one bit at a time against
        // each length's maximum code.
        let mut code = self.peek(8) as i32;
        for length in 9..=table.max_length() {
            code = (code << 1) | ((self.peek(length as u32) & 1) as i32);
            if let Some(symbol) = table.resolve(length, code) {
                self.consume(length as u32);
                return Ok(symbol);
            }
        }
        Err(PixelsError::malformed(
            "jpeg",
            "entropy data contains a code no Huffman table defines",
        ))
    }

    /// Read a `magnitude`-bit coefficient and sign-extend it.
    ///
    /// JPEG codes a coefficient as its bit length plus that many raw bits,
    /// where a leading zero means the value is negative and offset — the
    /// asymmetry that keeps zero out of the alphabet.
    ///
    /// # Errors
    ///
    /// As [`Reader::get_bits`].
    pub fn receive_extend(&mut self, magnitude: u32) -> Result<i32> {
        if magnitude == 0 {
            return Ok(0);
        }
        let raw = self.get_bits(magnitude)? as i32;
        let threshold = 1_i32 << (magnitude - 1);
        Ok(if raw < threshold {
            raw - (1_i32 << magnitude) + 1
        } else {
            raw
        })
    }

    /// Discard buffered bits, returning to a byte boundary.
    pub fn align(&mut self) {
        self.accumulator = 0;
        self.bits = 0;
    }

    /// Consume a restart marker at an interval boundary.
    ///
    /// Returns whether one was found. The bit buffer is dropped first: a
    /// restart interval is byte-aligned by construction, so whatever bits are
    /// left are the encoder's padding.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the source fails.
    pub fn restart(&mut self) -> Result<bool> {
        self.align();
        if self.marker.is_none() {
            // The bits ran out exactly at the boundary without the marker
            // being met yet, so read forward to find it.
            while self.entropy_byte()?.is_some() {}
        }
        match self.marker {
            Some(code) if crate::format::marker::is_restart(code) => {
                self.marker = None;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// The marker the scan stopped at, if it has already been met.
    #[must_use]
    pub const fn pending_marker(&self) -> Option<u8> {
        self.marker
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
    use crate::format::marker;
    use otf_pixels_core::ErrorCode;

    fn reader(bytes: &[u8]) -> Reader<std::io::Cursor<Vec<u8>>> {
        Reader::new(std::io::Cursor::new(bytes.to_vec()))
    }

    #[test]
    fn bits_are_read_most_significant_first() {
        let mut r = reader(&[0b1011_0010, 0b0100_0000]);
        assert_eq!(r.get_bits(1).unwrap(), 1);
        assert_eq!(r.get_bits(3).unwrap(), 0b011);
        assert_eq!(r.get_bits(4).unwrap(), 0b0010);
        assert_eq!(r.get_bits(2).unwrap(), 0b01);
    }

    #[test]
    fn stuffed_ff_bytes_are_data_and_bare_ones_are_markers() {
        // FF 00 is a literal 0xFF byte of entropy data.
        let mut r = reader(&[0xFF, 0x00, 0x55]);
        assert_eq!(r.get_bits(8).unwrap(), 0xFF);
        assert_eq!(r.get_bits(8).unwrap(), 0x55);
        assert_eq!(r.pending_marker(), None);

        // FF D9 is EOI, and ends the bit stream.
        let mut r = reader(&[0x55, 0xFF, 0xD9]);
        assert_eq!(r.get_bits(8).unwrap(), 0x55);
        // Past the marker the stream reads as zeros rather than as 0xFFD9.
        assert_eq!(r.get_bits(8).unwrap(), 0);
        assert_eq!(r.pending_marker(), Some(marker::EOI));
    }

    #[test]
    fn padding_before_a_marker_is_skipped() {
        let mut r = reader(&[0x55, 0xFF, 0xFF, 0xFF, 0xD0]);
        assert_eq!(r.get_bits(8).unwrap(), 0x55);
        assert_eq!(r.get_bits(1).unwrap(), 0);
        assert_eq!(r.pending_marker(), Some(marker::RST0));
    }

    #[test]
    fn a_truncated_scan_reads_as_zeros_rather_than_failing() {
        let mut r = reader(&[0b1111_0000]);
        assert_eq!(r.get_bits(8).unwrap(), 0b1111_0000);
        assert_eq!(r.get_bits(16).unwrap(), 0);
        assert_eq!(r.pending_marker(), Some(marker::EOI));
    }

    #[test]
    fn receive_extend_recovers_negative_coefficients() {
        // Magnitude 3: 0b111 is +7, 0b000 is -7, 0b100 is +4, 0b011 is -4.
        let mut r = reader(&[0b1110_0010, 0b0011_0000]);
        assert_eq!(r.receive_extend(3).unwrap(), 7);
        assert_eq!(r.receive_extend(3).unwrap(), -7);
        assert_eq!(r.receive_extend(3).unwrap(), 4);
        assert_eq!(r.receive_extend(3).unwrap(), -4);
        assert_eq!(r.receive_extend(0).unwrap(), 0);
    }

    #[test]
    fn huffman_symbols_decode_through_both_paths() {
        // Codes: 'a' = 0, 'b' = 10, 'c' = 110.
        let mut counts = [0_u8; 16];
        counts[0] = 1;
        counts[1] = 1;
        counts[2] = 1;
        let table = HuffmanTable::new(&counts, vec![b'a', b'b', b'c']).unwrap();

        // 0 10 110 0 -> a b c a
        let mut r = reader(&[0b0101_1000]);
        assert_eq!(r.decode(&table).unwrap(), b'a');
        assert_eq!(r.decode(&table).unwrap(), b'b');
        assert_eq!(r.decode(&table).unwrap(), b'c');
        assert_eq!(r.decode(&table).unwrap(), b'a');

        // A 16-bit code exercises the slow path.
        let deep = HuffmanTable::new(&[1_u8; 16], (0..16).collect()).unwrap();
        // The 16-bit code is fifteen ones then a zero.
        let mut r = reader(&[0xFF, 0x00, 0xFE]);
        assert_eq!(r.decode(&deep).unwrap(), 15);
    }

    #[test]
    fn an_undefined_code_is_malformed_rather_than_a_wrong_symbol() {
        // A table with a single one-bit code leaves `1` undefined.
        let mut counts = [0_u8; 16];
        counts[0] = 1;
        let table = HuffmanTable::new(&counts, vec![b'a']).unwrap();
        let mut r = reader(&[0b1111_1111, 0x00]);
        assert_eq!(r.decode(&table).unwrap_err().code(), ErrorCode::Malformed);
    }

    #[test]
    fn markers_are_found_across_stuffed_data() {
        let mut r = reader(&[0x12, 0xFF, 0x00, 0x34, 0xFF, 0xDA]);
        assert_eq!(r.next_marker().unwrap(), marker::SOS);

        // A marker already met while reading bits is handed back, not re-read.
        let mut r = reader(&[0x55, 0xFF, 0xD9, 0xFF, 0xD8]);
        assert_eq!(r.get_bits(16).unwrap(), 0x5500);
        assert_eq!(r.next_marker().unwrap(), marker::EOI);
        assert_eq!(r.next_marker().unwrap(), marker::SOI);
    }

    #[test]
    fn restart_markers_are_consumed_at_interval_boundaries() {
        // Data, then RST0, then more data.
        let mut r = reader(&[0xAA, 0xFF, 0xD0, 0xBB]);
        assert_eq!(r.get_bits(4).unwrap(), 0xA);
        // Restart drops the half-read byte and swallows the marker.
        assert!(r.restart().unwrap());
        assert_eq!(r.pending_marker(), None);
        assert_eq!(r.get_bits(8).unwrap(), 0xBB);

        // A missing restart marker is reported, not invented.
        let mut r = reader(&[0xAA, 0xFF, 0xD9]);
        assert_eq!(r.get_bits(8).unwrap(), 0xAA);
        assert!(!r.restart().unwrap());
        assert_eq!(r.pending_marker(), Some(marker::EOI));
    }

    #[test]
    fn segments_are_read_and_skipped_by_their_declared_length() {
        // Length 5 counts itself, so the payload is three bytes.
        let mut r = reader(&[0x00, 0x05, 1, 2, 3, 0xFF, 0xD9]);
        assert_eq!(r.read_segment().unwrap(), vec![1, 2, 3]);
        assert_eq!(r.next_marker().unwrap(), marker::EOI);

        let mut r = reader(&[0x00, 0x05, 1, 2, 3, 0xFF, 0xD9]);
        r.skip_segment().unwrap();
        assert_eq!(r.next_marker().unwrap(), marker::EOI);

        // A length below its own field would rewind the stream.
        let mut r = reader(&[0x00, 0x01, 1, 2]);
        assert_eq!(r.read_segment().unwrap_err().code(), ErrorCode::Malformed);

        // A length past the end of the stream is truncation.
        let mut r = reader(&[0x00, 0x40, 1, 2]);
        assert_eq!(r.read_segment().unwrap_err().code(), ErrorCode::Malformed);
    }
}
