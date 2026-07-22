//! The ISOBMFF box layer: the grammar every other AVIF structure is written in.
//!
//! An AVIF file is an ISO base media file (ISO/IEC 14496-12) carrying still
//! images rather than tracks. Everything is a *box*: a length, a four-character
//! type, and a payload that is either more boxes or leaf data. This module
//! implements only that grammar — what the boxes *mean* is [`crate::meta`] and
//! [`crate::props`].
//!
//! # Bounds
//!
//! Every read here is checked and returns [`PixelsError::Malformed`] rather
//! than panicking, because every byte is attacker-controlled. The two classic
//! ISOBMFF parser failures are a box whose declared size exceeds its parent's
//! payload, and a box whose declared size is smaller than its own header —
//! the second of which makes a naive parser loop forever. [`Reader::next_box`]
//! rejects both.

use core::fmt;
use otf_pixels_core::{PixelsError, Result};

/// A four-character box or brand identifier.
///
/// Compared as bytes, not as text: the specification defines these as four
/// octets, and some real brands (`MA1A`) are case-sensitive in a way a
/// lowercased comparison would lose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCc(pub [u8; 4]);

impl FourCc {
    /// The identifier for these four bytes.
    #[must_use]
    pub const fn new(bytes: &[u8; 4]) -> Self {
        Self(*bytes)
    }
}

impl fmt::Display for FourCc {
    /// Renders printable ASCII as itself and anything else as an escape, so a
    /// malformed-box message names the type without emitting control bytes
    /// into a log.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            if byte.is_ascii_graphic() || byte == b' ' {
                write!(f, "{}", byte as char)?;
            } else {
                write!(f, "\\x{byte:02x}")?;
            }
        }
        Ok(())
    }
}

/// A box's type and the extent of its payload within the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoxHeader {
    /// The four-character type.
    pub kind: FourCc,
    /// Offset of the payload's first byte, from the start of the file buffer.
    pub payload_start: usize,
    /// Length of the payload in bytes, excluding the header.
    pub payload_len: usize,
}

impl BoxHeader {
    /// The offset one past this box's last byte.
    #[must_use]
    pub const fn end(&self) -> usize {
        // Both fields were bounds-checked against the buffer when the header
        // was parsed, so this cannot overflow a `usize`.
        self.payload_start.saturating_add(self.payload_len)
    }
}

/// A checked cursor over a byte range of the file.
///
/// Holds the whole file buffer plus the window this reader is allowed to
/// touch, so a [`BoxHeader`]'s absolute offsets stay meaningful when a child
/// reader is handed to a nested parser.
#[derive(Debug, Clone, Copy)]
pub struct Reader<'a> {
    /// The complete file, so absolute offsets from `iloc` resolve.
    file: &'a [u8],
    /// One past the last byte this cursor may read, absolute into `file`.
    end: usize,
    /// The cursor, as an absolute offset into `file`.
    pos: usize,
}

impl<'a> Reader<'a> {
    /// A reader over the whole of `file`.
    #[must_use]
    pub const fn new(file: &'a [u8]) -> Self {
        Self {
            file,
            end: file.len(),
            pos: 0,
        }
    }

    /// A reader over `file[start..end]`, clamped to the file.
    ///
    /// Used to resolve an `iloc` extent, which names an absolute range that a
    /// hostile file may place outside the data it actually shipped.
    #[must_use]
    pub fn window(file: &'a [u8], start: usize, end: usize) -> Self {
        let end = end.min(file.len());
        let start = start.min(end);
        Self {
            file,
            end,
            pos: start,
        }
    }

    /// The complete file this reader was cut from.
    #[must_use]
    pub const fn file(&self) -> &'a [u8] {
        self.file
    }

    /// The cursor's absolute offset within the file.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Bytes between the cursor and the end of this reader's window.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.end.saturating_sub(self.pos)
    }

    /// Whether the cursor has reached the end of the window.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.remaining() == 0
    }

    /// The window's bytes from the cursor onward.
    #[must_use]
    pub fn rest(&self) -> &'a [u8] {
        self.file.get(self.pos..self.end).unwrap_or(&[])
    }

    /// Take `len` bytes, advancing the cursor.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the window holds fewer than `len`
    /// bytes from the cursor.
    pub fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let stop = self.pos.checked_add(len).ok_or_else(|| {
            PixelsError::malformed("avif", format!("a {len}-byte read overflows the file offset"))
        })?;
        if stop > self.end {
            return Err(PixelsError::malformed(
                "avif",
                format!(
                    "a {len}-byte read at offset {} runs past the end of its box, which holds {}",
                    self.pos,
                    self.remaining()
                ),
            ));
        }
        let bytes = self.file.get(self.pos..stop).ok_or_else(|| {
            PixelsError::malformed("avif", format!("offset {stop} is outside the file"))
        })?;
        self.pos = stop;
        Ok(bytes)
    }

    /// Advance the cursor by `len` bytes without returning them.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn skip(&mut self, len: usize) -> Result<()> {
        self.take(len).map(|_| ())
    }

    /// Read one byte.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn u8(&mut self) -> Result<u8> {
        self.take(1)?
            .first()
            .copied()
            .ok_or_else(|| PixelsError::malformed("avif", "a one-byte read returned nothing"))
    }

    /// Read a big-endian `u16`. ISOBMFF is big-endian throughout.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn u16(&mut self) -> Result<u16> {
        let bytes: [u8; 2] = self.array()?;
        Ok(u16::from_be_bytes(bytes))
    }

    /// Read a big-endian `u32`.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self.array()?;
        Ok(u32::from_be_bytes(bytes))
    }

    /// Read a big-endian `u64`.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn u64(&mut self) -> Result<u64> {
        let bytes: [u8; 8] = self.array()?;
        Ok(u64::from_be_bytes(bytes))
    }

    /// Read a big-endian unsigned integer of `size` bytes, where `size` is 0,
    /// 4 or 8.
    ///
    /// `iloc` encodes its offset and length field widths this way, and a width
    /// of zero means the field is absent and reads as zero.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a width the specification does
    /// not allow, or as [`Reader::take`].
    pub fn uint(&mut self, size: u8) -> Result<u64> {
        match size {
            0 => Ok(0),
            4 => self.u32().map(u64::from),
            8 => self.u64(),
            other => Err(PixelsError::malformed(
                "avif",
                format!("field width {other} is not one of 0, 4 or 8"),
            )),
        }
    }

    /// Read a four-character code.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn fourcc(&mut self) -> Result<FourCc> {
        self.array().map(FourCc)
    }

    /// Read a fixed-size array.
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self.take(N)?;
        let mut out = [0_u8; N];
        // `take` returned exactly `N` bytes, so the lengths agree.
        if bytes.len() != N {
            return Err(PixelsError::malformed(
                "avif",
                format!("a {N}-byte read returned {} bytes", bytes.len()),
            ));
        }
        out.copy_from_slice(bytes);
        Ok(out)
    }

    /// Read a null-terminated UTF-8 string.
    ///
    /// Used by `infe` for item names and by `auxC` for the auxiliary type URN.
    /// An unterminated string consumes the rest of the box, which is what
    /// real files with a missing terminator intend and costs nothing to allow.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the bytes are not UTF-8.
    pub fn cstring(&mut self) -> Result<&'a str> {
        let rest = self.rest();
        let len = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
        let bytes = self.take(len)?;
        // Step over the terminator when there was one.
        if len < rest.len() {
            self.skip(1)?;
        }
        core::str::from_utf8(bytes)
            .map_err(|_| PixelsError::malformed("avif", "a box string is not valid UTF-8"))
    }

    /// Read the version and flags of a full box.
    ///
    /// # Errors
    ///
    /// As [`Reader::take`].
    pub fn full_box(&mut self) -> Result<(u8, u32)> {
        let word = self.u32()?;
        // Version is the top octet, flags the low 24 bits.
        let version = u8::try_from(word >> 24).unwrap_or(0);
        Ok((version, word & 0x00ff_ffff))
    }

    /// Read the next box header, positioning the cursor at its payload.
    ///
    /// Returns `None` at the end of the window. A trailing run shorter than a
    /// header is treated as padding and ends iteration rather than failing:
    /// real files pad, and refusing them buys no safety.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if a box declares a size smaller
    /// than its own header — which would make iteration loop forever — or one
    /// that runs past the end of the enclosing box.
    pub fn next_box(&mut self) -> Option<Result<BoxHeader>> {
        // A header is 8 bytes; anything shorter is trailing padding.
        if self.remaining() < 8 {
            return None;
        }
        Some(self.read_box_header())
    }

    /// The fallible half of [`Reader::next_box`].
    fn read_box_header(&mut self) -> Result<BoxHeader> {
        let start = self.pos;
        let size32 = self.u32()?;
        let kind = self.fourcc()?;

        // Size 1 means a 64-bit size follows the type; size 0 means the box
        // runs to the end of the enclosing box.
        let (total, header_len) = match size32 {
            1 => {
                let large = self.u64()?;
                let total = usize::try_from(large).map_err(|_| {
                    PixelsError::malformed(
                        "avif",
                        format!("box '{kind}' declares {large} bytes, more than this platform can address"),
                    )
                })?;
                (total, 16_usize)
            }
            0 => (self.end.saturating_sub(start), 8_usize),
            n => (usize::try_from(n).unwrap_or(0), 8_usize),
        };

        // A `uuid` box carries a 16-byte user type before its payload. We do
        // not interpret any, but the header length must account for it so the
        // payload extent is right.
        let header_len = if kind == FourCc::new(b"uuid") {
            self.skip(16)?;
            header_len.saturating_add(16)
        } else {
            header_len
        };

        if total < header_len {
            return Err(PixelsError::malformed(
                "avif",
                format!("box '{kind}' declares {total} bytes, less than its own {header_len}-byte header"),
            ));
        }
        let payload_len = total.saturating_sub(header_len);
        let payload_start = self.pos;
        let end = payload_start.checked_add(payload_len).ok_or_else(|| {
            PixelsError::malformed("avif", format!("box '{kind}' extends past the address space"))
        })?;
        if end > self.end {
            return Err(PixelsError::malformed(
                "avif",
                format!(
                    "box '{kind}' declares {total} bytes at offset {start}, running {} past the end of its parent",
                    end.saturating_sub(self.end)
                ),
            ));
        }

        self.pos = end;
        Ok(BoxHeader {
            kind,
            payload_start,
            payload_len,
        })
    }

    /// A reader over `header`'s payload.
    #[must_use]
    pub fn payload(&self, header: &BoxHeader) -> Reader<'a> {
        Reader::window(self.file, header.payload_start, header.end())
    }

    /// Find the first child box of type `kind`, if any.
    ///
    /// Scans from the cursor and leaves the cursor where it stopped, so this
    /// is for one-shot lookups rather than repeated probing of one container.
    ///
    /// # Errors
    ///
    /// As [`Reader::next_box`].
    pub fn find(&mut self, kind: &[u8; 4]) -> Result<Option<Reader<'a>>> {
        let wanted = FourCc::new(kind);
        while let Some(header) = self.next_box() {
            let header = header?;
            if header.kind == wanted {
                return Ok(Some(self.payload(&header)));
            }
        }
        Ok(None)
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

    /// Build a box: 4-byte size, 4-byte type, payload.
    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let total = u32::try_from(8 + payload.len()).unwrap();
        out.extend_from_slice(&total.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn reads_a_flat_sequence_of_boxes() {
        let mut file = boxed(b"ftyp", b"avif0000");
        file.extend_from_slice(&boxed(b"mdat", &[1, 2, 3]));

        let mut reader = Reader::new(&file);
        let first = reader.next_box().unwrap().unwrap();
        assert_eq!(first.kind, FourCc::new(b"ftyp"));
        assert_eq!(first.payload_len, 8);
        assert_eq!(reader.payload(&first).rest(), b"avif0000");

        let second = reader.next_box().unwrap().unwrap();
        assert_eq!(second.kind, FourCc::new(b"mdat"));
        assert_eq!(reader.payload(&second).rest(), &[1, 2, 3]);

        assert!(reader.next_box().is_none());
    }

    #[test]
    fn nested_boxes_are_bounded_by_their_parent() {
        let inner = boxed(b"hdlr", b"pict");
        let outer = boxed(b"meta", &inner);

        let mut reader = Reader::new(&outer);
        let meta = reader.next_box().unwrap().unwrap();
        let mut children = reader.payload(&meta);
        let hdlr = children.next_box().unwrap().unwrap();
        assert_eq!(hdlr.kind, FourCc::new(b"hdlr"));
        assert_eq!(children.payload(&hdlr).rest(), b"pict");
        assert!(children.next_box().is_none());
    }

    /// A box declaring less than its own header would leave the cursor where
    /// it was and iterate forever. This is the single most important bound in
    /// the module.
    #[test]
    fn a_box_smaller_than_its_header_is_rejected() {
        let mut file = Vec::new();
        file.extend_from_slice(&3_u32.to_be_bytes());
        file.extend_from_slice(b"junk");
        file.extend_from_slice(&[0; 16]);

        let mut reader = Reader::new(&file);
        let error = reader.next_box().unwrap().unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("less than its own"), "{error}");
    }

    #[test]
    fn a_box_running_past_its_parent_is_rejected() {
        // Declares 400 bytes inside a file holding 16.
        let mut file = Vec::new();
        file.extend_from_slice(&400_u32.to_be_bytes());
        file.extend_from_slice(b"meta");
        file.extend_from_slice(&[0; 8]);

        let mut reader = Reader::new(&file);
        let error = reader.next_box().unwrap().unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("past the end"), "{error}");
    }

    #[test]
    fn size_zero_runs_to_the_end_of_the_parent() {
        let mut file = Vec::new();
        file.extend_from_slice(&0_u32.to_be_bytes());
        file.extend_from_slice(b"mdat");
        file.extend_from_slice(&[7; 12]);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        assert_eq!(header.payload_len, 12);
        assert_eq!(reader.payload(&header).rest(), &[7; 12]);
        assert!(reader.next_box().is_none());
    }

    #[test]
    fn a_sixty_four_bit_size_is_honoured() {
        let mut file = Vec::new();
        file.extend_from_slice(&1_u32.to_be_bytes());
        file.extend_from_slice(b"mdat");
        file.extend_from_slice(&20_u64.to_be_bytes());
        file.extend_from_slice(&[9; 4]);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        assert_eq!(header.payload_len, 4);
        assert_eq!(reader.payload(&header).rest(), &[9; 4]);
    }

    #[test]
    fn a_uuid_box_accounts_for_its_user_type() {
        let mut payload = Vec::from([0xAB; 16]);
        payload.extend_from_slice(b"data");
        let file = boxed(b"uuid", &payload);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        assert_eq!(header.kind, FourCc::new(b"uuid"));
        assert_eq!(reader.payload(&header).rest(), b"data");
    }

    #[test]
    fn trailing_padding_ends_iteration_rather_than_failing() {
        let mut file = boxed(b"ftyp", b"avif");
        file.extend_from_slice(&[0, 0, 0]);

        let mut reader = Reader::new(&file);
        assert!(reader.next_box().unwrap().is_ok());
        assert!(reader.next_box().is_none());
    }

    #[test]
    fn scalar_reads_are_bounded() {
        let file = [0x01, 0x02, 0x03];
        let mut reader = Reader::new(&file);
        assert_eq!(reader.u16().unwrap(), 0x0102);
        // One byte left, but a u32 wants four.
        let error = reader.u32().unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        // The failed read did not advance the cursor past the window.
        assert_eq!(reader.remaining(), 1);
    }

    #[test]
    fn full_box_splits_version_from_flags() {
        let file = [0x01, 0x00, 0x00, 0x0F];
        let mut reader = Reader::new(&file);
        let (version, flags) = reader.full_box().unwrap();
        assert_eq!(version, 1);
        assert_eq!(flags, 0x0F);
    }

    #[test]
    fn uint_honours_the_declared_width() {
        let file = [0x00, 0x00, 0x01, 0x00];
        let mut reader = Reader::new(&file);
        assert_eq!(reader.uint(0).unwrap(), 0);
        assert_eq!(reader.uint(4).unwrap(), 256);

        let mut reader = Reader::new(&file);
        let error = reader.uint(3).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
    }

    #[test]
    fn cstring_stops_at_the_terminator_and_survives_a_missing_one() {
        let file = *b"alpha\0beta";
        let mut reader = Reader::new(&file);
        assert_eq!(reader.cstring().unwrap(), "alpha");
        // No terminator on the tail; it reads to the end of the window.
        assert_eq!(reader.cstring().unwrap(), "beta");
        assert!(reader.is_empty());
    }

    #[test]
    fn a_window_outside_the_file_is_clamped_rather_than_panicking() {
        let file = [1, 2, 3, 4];
        let reader = Reader::window(&file, 100, 200);
        assert!(reader.is_empty());
        assert_eq!(reader.rest(), &[] as &[u8]);
    }

    #[test]
    fn find_locates_a_child_by_type() {
        let mut payload = boxed(b"hdlr", b"pict");
        payload.extend_from_slice(&boxed(b"pitm", &[0, 0, 0, 0, 0, 1]));
        let file = boxed(b"meta", &payload);

        let mut reader = Reader::new(&file);
        let meta = reader.next_box().unwrap().unwrap();
        let found = reader.payload(&meta).find(b"pitm").unwrap();
        assert!(found.is_some());
        let missing = reader.payload(&meta).find(b"iloc").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn fourcc_display_escapes_unprintable_bytes() {
        assert_eq!(FourCc::new(b"ftyp").to_string(), "ftyp");
        assert_eq!(FourCc::new(&[0x00, 0x41, 0x1b, 0x42]).to_string(), "\\x00A\\x1bB");
    }
}
