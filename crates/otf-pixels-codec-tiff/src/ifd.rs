//! TIFF's image file directory: tags, types, and the endianness they arrive in.
//!
//! A TIFF is a header naming a byte order and pointing at an IFD; an IFD is a
//! count followed by twelve-byte entries, each a tag, a type, a count and
//! either a value or an offset to one. Everything about the image — its size,
//! its layout, where its pixels live — is a tag.
//!
//! # Both endiannesses are real
//!
//! `II` (Intel, little) and `MM` (Motorola, big) are both common in the wild;
//! scanners and Adobe tools disagree. A decoder that assumed one would fail
//! half the files it met, so byte order is a value threaded through every
//! read rather than a compile-time choice.
//!
//! # Unknown tags are skipped
//!
//! SPEC §Formats: "exotic tags are skipped, not errors". TIFF's extensibility
//! is the whole point of the format, and every real file carries tags a given
//! reader does not know. Only tags that change how pixels are *laid out* can
//! be fatal when unsupported.

use otf_pixels_core::{PixelsError, Result};

/// Byte order, from the two-character header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// `II` — least significant byte first.
    Little,
    /// `MM` — most significant byte first.
    Big,
}

impl ByteOrder {
    /// Read a 16-bit value at `at`.
    #[must_use]
    pub fn u16(self, data: &[u8], at: usize) -> u16 {
        let a = data.get(at).copied().unwrap_or(0);
        let b = data.get(at + 1).copied().unwrap_or(0);
        match self {
            Self::Little => u16::from_le_bytes([a, b]),
            Self::Big => u16::from_be_bytes([a, b]),
        }
    }

    /// Read a 32-bit value at `at`.
    #[must_use]
    pub fn u32(self, data: &[u8], at: usize) -> u32 {
        let mut bytes = [0_u8; 4];
        for (slot, offset) in bytes.iter_mut().zip(0..4) {
            *slot = data.get(at + offset).copied().unwrap_or(0);
        }
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }

    /// Write a 16-bit value.
    #[must_use]
    pub const fn write_u16(self, value: u16) -> [u8; 2] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }

    /// Write a 32-bit value.
    #[must_use]
    pub const fn write_u32(self, value: u32) -> [u8; 4] {
        match self {
            Self::Little => value.to_le_bytes(),
            Self::Big => value.to_be_bytes(),
        }
    }
}

/// The baseline tags this codec understands (TIFF 6.0 §Section 8).
pub mod tag {
    /// Image width in pixels.
    pub const IMAGE_WIDTH: u16 = 256;
    /// Image height in pixels.
    pub const IMAGE_LENGTH: u16 = 257;
    /// Bits per sample, one entry per channel.
    pub const BITS_PER_SAMPLE: u16 = 258;
    /// Compression scheme.
    pub const COMPRESSION: u16 = 259;
    /// How samples are interpreted: greyscale, RGB, palette.
    pub const PHOTOMETRIC: u16 = 262;
    /// Byte offset of each strip.
    pub const STRIP_OFFSETS: u16 = 273;
    /// Channels per pixel.
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    /// Rows per strip.
    pub const ROWS_PER_STRIP: u16 = 278;
    /// Compressed byte count of each strip.
    pub const STRIP_BYTE_COUNTS: u16 = 279;
    /// The colour map, for palette images.
    pub const COLOR_MAP: u16 = 320;
    /// How channels are arranged: interleaved or planar.
    pub const PLANAR_CONFIG: u16 = 284;
    /// Predictor applied before compression.
    pub const PREDICTOR: u16 = 317;
    /// Tile width in pixels.
    pub const TILE_WIDTH: u16 = 322;
    /// Tile height in pixels.
    pub const TILE_LENGTH: u16 = 323;
    /// Byte offset of each tile.
    pub const TILE_OFFSETS: u16 = 324;
    /// Compressed byte count of each tile.
    pub const TILE_BYTE_COUNTS: u16 = 325;
    /// Extra channel semantics, which is where alpha is declared.
    pub const EXTRA_SAMPLES: u16 = 338;
    /// Sample format: unsigned, signed or float.
    pub const SAMPLE_FORMAT: u16 = 339;
}

/// A TIFF field type, and how many bytes one value of it occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// 8-bit unsigned.
    Byte,
    /// NUL-terminated string.
    Ascii,
    /// 16-bit unsigned.
    Short,
    /// 32-bit unsigned.
    Long,
    /// Two 32-bit values, numerator and denominator.
    Rational,
    /// A type this codec does not interpret, with its declared size.
    Other(u16, usize),
}

impl FieldType {
    /// The type for a field's type code.
    #[must_use]
    pub const fn from_code(code: u16) -> Self {
        match code {
            1 => Self::Byte,
            2 => Self::Ascii,
            3 => Self::Short,
            4 => Self::Long,
            5 => Self::Rational,
            // Signed variants and floats occupy known sizes even though we do
            // not interpret them; knowing the size is what lets an unknown tag
            // be skipped rather than desynchronise the directory.
            6 => Self::Other(code, 1),
            7 => Self::Other(code, 1),
            8 => Self::Other(code, 2),
            9 => Self::Other(code, 4),
            10 => Self::Other(code, 8),
            11 => Self::Other(code, 4),
            12 => Self::Other(code, 8),
            other => Self::Other(other, 0),
        }
    }

    /// Bytes per value of this type.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::Byte | Self::Ascii => 1,
            Self::Short => 2,
            Self::Long => 4,
            Self::Rational => 8,
            Self::Other(_, size) => size,
        }
    }
}

/// One directory entry, with its values already resolved.
#[derive(Debug, Clone)]
pub struct Entry {
    /// The tag this entry carries.
    pub tag: u16,
    /// The field type.
    pub field_type: FieldType,
    /// The values, widened to `u32`. Types wider than 32 bits are not
    /// interpreted, so their entries carry no values.
    pub values: Vec<u32>,
}

impl Entry {
    /// The first value, if any.
    #[must_use]
    pub fn first(&self) -> Option<u32> {
        self.values.first().copied()
    }
}

/// A parsed image file directory.
#[derive(Debug, Clone)]
pub struct Directory {
    entries: Vec<Entry>,
    /// Offset of the next IFD, or zero if this is the last.
    next: u32,
}

/// The largest value array read from one tag.
///
/// A tiled 2 GB TIFF legitimately has hundreds of thousands of tile offsets,
/// so this cannot be small — but it must exist, because the count is a 32-bit
/// field an attacker controls.
const MAX_VALUES: usize = 16 * 1024 * 1024;

impl Directory {
    /// Parse the IFD at `offset` within `data`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a directory that runs past the
    /// end of the data or declares an implausible value count.
    pub fn parse(data: &[u8], order: ByteOrder, offset: usize) -> Result<Self> {
        let count = order.u16(data, offset) as usize;
        if data.len() < offset + 2 + count * 12 + 4 {
            return Err(PixelsError::malformed(
                "tiff",
                format!("directory of {count} entries runs past the end of the file"),
            ));
        }

        let mut entries = Vec::with_capacity(count);
        for index in 0..count {
            let at = offset + 2 + index * 12;
            let tag = order.u16(data, at);
            let field_type = FieldType::from_code(order.u16(data, at + 2));
            let value_count = order.u32(data, at + 4) as usize;
            let size = field_type.size();

            if size == 0 || value_count > MAX_VALUES {
                // An unknown type or an implausible count: keep the tag so a
                // caller can see it was present, but read no values. Skipping
                // rather than failing is what SPEC §Formats requires.
                entries.push(Entry {
                    tag,
                    field_type,
                    values: Vec::new(),
                });
                continue;
            }

            let total = value_count.saturating_mul(size);
            // Values of four bytes or fewer live in the entry itself; anything
            // larger is an offset. Getting this backwards is the classic TIFF
            // parsing bug, and it silently reads the offset as data.
            let values_at = if total <= 4 {
                at + 8
            } else {
                order.u32(data, at + 8) as usize
            };

            let mut values = Vec::with_capacity(value_count.min(4096));
            for value_index in 0..value_count {
                let value_at = values_at + value_index * size;
                if value_at + size > data.len() {
                    break;
                }
                let value = match field_type {
                    FieldType::Byte | FieldType::Ascii => {
                        u32::from(data.get(value_at).copied().unwrap_or(0))
                    }
                    FieldType::Short => u32::from(order.u16(data, value_at)),
                    FieldType::Long => order.u32(data, value_at),
                    // A rational's numerator is the useful half for the tags
                    // we read (resolution), and we read none of them for
                    // pixels, so the denominator is dropped rather than lost.
                    FieldType::Rational => order.u32(data, value_at),
                    FieldType::Other(..) => break,
                };
                values.push(value);
            }

            entries.push(Entry {
                tag,
                field_type,
                values,
            });
        }

        let next = order.u32(data, offset + 2 + count * 12);
        Ok(Self { entries, next })
    }

    /// The entry for `tag`, if present.
    #[must_use]
    pub fn get(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.tag == tag)
    }

    /// The first value of `tag`, if present.
    #[must_use]
    pub fn value(&self, tag: u16) -> Option<u32> {
        self.get(tag).and_then(Entry::first)
    }

    /// The first value of `tag`, or `default` if absent.
    #[must_use]
    pub fn value_or(&self, tag: u16, default: u32) -> u32 {
        self.value(tag).unwrap_or(default)
    }

    /// Every value of `tag`, or an empty slice if absent.
    #[must_use]
    pub fn values(&self, tag: u16) -> &[u32] {
        self.get(tag).map_or(&[], |entry| &entry.values)
    }

    /// The first value of `tag`, or a malformed-input error naming it.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the tag is absent or empty.
    pub fn require(&self, tag: u16, name: &str) -> Result<u32> {
        self.value(tag)
            .ok_or_else(|| PixelsError::malformed("tiff", format!("missing required tag {name}")))
    }

    /// The offset of the next directory, or `None` if this is the last.
    #[must_use]
    pub const fn next_offset(&self) -> Option<u32> {
        if self.next == 0 {
            None
        } else {
            Some(self.next)
        }
    }

    /// How many entries the directory holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the directory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Parse the eight-byte TIFF header, returning the byte order and first IFD.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for a bad byte-order mark or magic
/// number.
pub fn parse_header(data: &[u8]) -> Result<(ByteOrder, usize)> {
    let order = match data.get(..2) {
        Some(b"II") => ByteOrder::Little,
        Some(b"MM") => ByteOrder::Big,
        _ => {
            return Err(PixelsError::malformed(
                "tiff",
                "byte-order mark is neither II nor MM",
            ));
        }
    };
    // 42 is the magic, and reading it *in the declared order* is what proves
    // the byte-order mark was honest.
    let magic = order.u16(data, 2);
    if magic != 42 {
        return Err(PixelsError::malformed(
            "tiff",
            format!("magic number is {magic}, not 42"),
        ));
    }
    Ok((order, order.u32(data, 4) as usize))
}

/// Whether `prefix` starts with a TIFF header.
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    parse_header(prefix).is_ok()
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

    /// Build a minimal TIFF with the given entries, in the given byte order.
    fn build(order: ByteOrder, entries: &[(u16, FieldType, Vec<u32>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(if order == ByteOrder::Little {
            b"II"
        } else {
            b"MM"
        });
        out.extend_from_slice(&order.write_u16(42));
        out.extend_from_slice(&order.write_u32(8));

        // Directory, then any values that did not fit inline.
        let mut directory = Vec::new();
        directory.extend_from_slice(&order.write_u16(entries.len() as u16));
        let heap_start = 8 + 2 + entries.len() * 12 + 4;
        let mut heap = Vec::new();

        for (tag, field_type, values) in entries {
            directory.extend_from_slice(&order.write_u16(*tag));
            let code = match field_type {
                FieldType::Byte => 1,
                FieldType::Ascii => 2,
                FieldType::Short => 3,
                FieldType::Long => 4,
                FieldType::Rational => 5,
                FieldType::Other(code, _) => *code,
            };
            directory.extend_from_slice(&order.write_u16(code));
            directory.extend_from_slice(&order.write_u32(values.len() as u32));

            let size = field_type.size();
            let total = values.len() * size;
            let mut encoded = Vec::new();
            for &value in values {
                match size {
                    1 => encoded.push(value as u8),
                    2 => encoded.extend_from_slice(&order.write_u16(value as u16)),
                    _ => encoded.extend_from_slice(&order.write_u32(value)),
                }
            }
            if total <= 4 {
                encoded.resize(4, 0);
                directory.extend_from_slice(&encoded);
            } else {
                directory.extend_from_slice(&order.write_u32((heap_start + heap.len()) as u32));
                heap.extend_from_slice(&encoded);
            }
        }
        directory.extend_from_slice(&order.write_u32(0));

        out.extend_from_slice(&directory);
        out.extend_from_slice(&heap);
        out
    }

    #[test]
    fn both_byte_orders_parse() {
        // Half the TIFFs in the world are big-endian; assuming one would fail
        // them all.
        for order in [ByteOrder::Little, ByteOrder::Big] {
            let bytes = build(order, &[(tag::IMAGE_WIDTH, FieldType::Long, vec![640])]);
            let (parsed_order, offset) = parse_header(&bytes).unwrap();
            assert_eq!(parsed_order, order);
            let directory = Directory::parse(&bytes, order, offset).unwrap();
            assert_eq!(directory.value(tag::IMAGE_WIDTH), Some(640), "{order:?}");
        }
    }

    #[test]
    fn a_wrong_magic_number_is_rejected() {
        let mut bytes = build(ByteOrder::Little, &[]);
        bytes[2] = 43;
        assert!(parse_header(&bytes).is_err());
    }

    #[test]
    fn a_bad_byte_order_mark_is_rejected() {
        let mut bytes = build(ByteOrder::Little, &[]);
        bytes[0] = b'X';
        assert!(parse_header(&bytes).is_err());
    }

    #[test]
    fn small_values_live_inline_and_large_ones_are_offsets() {
        // The classic TIFF parsing bug is getting this backwards, which reads
        // the offset itself as data and produces plausible nonsense.
        for order in [ByteOrder::Little, ByteOrder::Big] {
            // Two shorts fit in four bytes; three do not.
            let inline = build(
                order,
                &[(tag::BITS_PER_SAMPLE, FieldType::Short, vec![8, 8])],
            );
            let (o, at) = parse_header(&inline).unwrap();
            let directory = Directory::parse(&inline, o, at).unwrap();
            assert_eq!(directory.values(tag::BITS_PER_SAMPLE), &[8, 8], "{order:?}");

            let offset = build(
                order,
                &[(tag::BITS_PER_SAMPLE, FieldType::Short, vec![8, 8, 8])],
            );
            let (o, at) = parse_header(&offset).unwrap();
            let directory = Directory::parse(&offset, o, at).unwrap();
            assert_eq!(
                directory.values(tag::BITS_PER_SAMPLE),
                &[8, 8, 8],
                "{order:?}"
            );
        }
    }

    #[test]
    fn every_field_type_width_reads_correctly() {
        let bytes = build(
            ByteOrder::Little,
            &[
                (100, FieldType::Byte, vec![1, 2, 3]),
                (101, FieldType::Short, vec![1000, 2000]),
                (102, FieldType::Long, vec![100_000]),
            ],
        );
        let (order, at) = parse_header(&bytes).unwrap();
        let directory = Directory::parse(&bytes, order, at).unwrap();
        assert_eq!(directory.values(100), &[1, 2, 3]);
        assert_eq!(directory.values(101), &[1000, 2000]);
        assert_eq!(directory.values(102), &[100_000]);
    }

    #[test]
    fn an_unknown_tag_is_kept_but_not_fatal() {
        // SPEC §Formats: exotic tags are skipped, not errors. TIFF's
        // extensibility is the point of the format.
        let bytes = build(
            ByteOrder::Little,
            &[
                (60_000, FieldType::Long, vec![7]),
                (tag::IMAGE_WIDTH, FieldType::Long, vec![32]),
            ],
        );
        let (order, at) = parse_header(&bytes).unwrap();
        let directory = Directory::parse(&bytes, order, at).unwrap();
        assert_eq!(directory.value(60_000), Some(7));
        assert_eq!(
            directory.value(tag::IMAGE_WIDTH),
            Some(32),
            "an unknown tag desynchronised the directory"
        );
    }

    #[test]
    fn an_unknown_field_type_does_not_desynchronise_the_directory() {
        // Entries are a fixed twelve bytes whatever their type, so a type we
        // cannot read must not stop us reading the tags after it.
        let bytes = build(
            ByteOrder::Little,
            &[
                (60_001, FieldType::Other(31_000, 0), vec![]),
                (tag::IMAGE_LENGTH, FieldType::Long, vec![48]),
            ],
        );
        let (order, at) = parse_header(&bytes).unwrap();
        let directory = Directory::parse(&bytes, order, at).unwrap();
        assert_eq!(directory.value(tag::IMAGE_LENGTH), Some(48));
    }

    #[test]
    fn a_missing_required_tag_names_itself() {
        let bytes = build(ByteOrder::Little, &[]);
        let (order, at) = parse_header(&bytes).unwrap();
        let directory = Directory::parse(&bytes, order, at).unwrap();
        let error = directory
            .require(tag::IMAGE_WIDTH, "ImageWidth")
            .unwrap_err();
        assert!(error.to_string().contains("ImageWidth"), "{error}");
    }

    #[test]
    fn a_directory_running_past_the_file_is_an_error() {
        let mut bytes = build(
            ByteOrder::Little,
            &[(tag::IMAGE_WIDTH, FieldType::Long, vec![1])],
        );
        // Claim a hundred entries in a file that holds one.
        bytes[8] = 100;
        let (order, at) = parse_header(&bytes).unwrap();
        assert!(Directory::parse(&bytes, order, at).is_err());
    }

    #[test]
    fn a_value_offset_past_the_file_truncates_rather_than_panicking() {
        let mut bytes = build(
            ByteOrder::Little,
            &[(tag::STRIP_OFFSETS, FieldType::Long, vec![1, 2, 3])],
        );
        // Point the value array at the far end of the address space.
        let at = 8 + 2 + 8;
        bytes[at..at + 4].copy_from_slice(&0xFFFF_FF00_u32.to_le_bytes());
        let (order, start) = parse_header(&bytes).unwrap();
        let directory = Directory::parse(&bytes, order, start).unwrap();
        assert!(
            directory.values(tag::STRIP_OFFSETS).is_empty(),
            "an out-of-range offset should yield no values"
        );
    }

    #[test]
    fn arbitrary_bytes_never_panic() {
        let mut seed = 0x2468_ACE0_u32;
        for _ in 0..2000 {
            let len = (seed % 300) as usize + 8;
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 16) as u8
                })
                .collect();
            if let Ok((order, offset)) = parse_header(&data) {
                let _ = Directory::parse(&data, order, offset);
            }
        }
    }

    #[test]
    fn probe_accepts_only_a_real_header() {
        assert!(probe(b"II\x2a\x00\x08\x00\x00\x00"));
        assert!(probe(b"MM\x00\x2a\x00\x00\x00\x08"));
        assert!(!probe(b"II\x2b\x00\x08\x00\x00\x00"), "BigTIFF is not v1");
        assert!(!probe(b"GIF89a"));
        assert!(!probe(b""));
    }
}
