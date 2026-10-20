//! EXIF blocks as trees of image file directories.
//!
//! An EXIF block is TIFF-structured: a byte-order mark, then IFD0, whose
//! entries may point at the Exif, GPS and Interoperability sub-directories.
//! Most formats carry the block whole and never look inside, so they need
//! nothing here. TIFF is the exception: its EXIF is not a block but tags in
//! the file's own directory, so turning one into the other means reading
//! the directories, keeping the metadata and dropping the image structure,
//! and laying the result out again with fresh offsets. That is this module.
//!
//! Values are kept as the bytes they were stored as, in the block's byte
//! order; nothing is interpreted beyond what relocating them needs. A
//! maker note's internal offsets are relative to wherever its maker chose,
//! and move with it as they do in every tool that rewrites EXIF.

/// Tags whose value is the offset of a sub-directory.
const SUB_IFDS: [u16; 3] = [0x8769, 0x8825, 0xA005];

/// The width in bytes of one value of each TIFF field type, 0 if unknown.
const fn type_size(kind: u16) -> usize {
    match kind {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => 0,
    }
}

/// One directory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExifEntry {
    /// The tag number.
    pub tag: u16,
    /// The TIFF field type.
    pub kind: u16,
    /// How many values of `kind`.
    pub count: u32,
    /// The values, in the block's byte order. Empty for a sub-directory.
    pub data: Vec<u8>,
    /// The directory this entry points at, for the Exif, GPS and
    /// Interoperability pointers.
    pub sub: Option<Vec<ExifEntry>>,
}

/// A parsed EXIF block: its byte order and IFD0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exif {
    /// Whether values are big-endian (`MM`).
    pub big_endian: bool,
    /// IFD0's entries, in tag order.
    pub ifd0: Vec<ExifEntry>,
}

struct Reader<'a> {
    data: &'a [u8],
    big_endian: bool,
}

impl Reader<'_> {
    fn u16(&self, at: usize) -> Option<u16> {
        let b = self.data.get(at..at.checked_add(2)?)?;
        let b = [*b.first()?, *b.get(1)?];
        Some(if self.big_endian {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    }

    fn u32(&self, at: usize) -> Option<u32> {
        let b = self.data.get(at..at.checked_add(4)?)?;
        let b = [*b.first()?, *b.get(1)?, *b.get(2)?, *b.get(3)?];
        Some(if self.big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    }

    /// The directory at `offset`, its sub-directories `depth` deep at most.
    fn ifd(&self, offset: usize, depth: u32) -> Option<Vec<ExifEntry>> {
        let count = usize::from(self.u16(offset)?);
        let mut out = Vec::with_capacity(count.min(512));
        for i in 0..count {
            let at = offset.checked_add(2)?.checked_add(i.checked_mul(12)?)?;
            let (tag, kind, n) = (self.u16(at)?, self.u16(at + 2)?, self.u32(at + 4)?);
            let size = type_size(kind);
            if size == 0 {
                continue;
            }
            if SUB_IFDS.contains(&tag) {
                if depth == 0 {
                    continue;
                }
                let target = usize::try_from(self.u32(at + 8)?).ok()?;
                if let Some(sub) = self.ifd(target, depth - 1) {
                    out.push(ExifEntry {
                        tag,
                        kind: 4,
                        count: 1,
                        data: Vec::new(),
                        sub: Some(sub),
                    });
                }
                continue;
            }
            let len = size.checked_mul(usize::try_from(n).ok()?)?;
            let data = if len <= 4 {
                self.data.get(at + 8..at + 8 + len)?.to_vec()
            } else {
                let from = usize::try_from(self.u32(at + 8)?).ok()?;
                match self.data.get(from..from.checked_add(len)?) {
                    Some(d) => d.to_vec(),
                    // A value past the end is dropped, not the block.
                    None => continue,
                }
            };
            out.push(ExifEntry {
                tag,
                kind,
                count: n,
                data,
                sub: None,
            });
        }
        out.sort_by_key(|e| e.tag);
        out.dedup_by_key(|e| e.tag);
        Some(out)
    }
}

impl Exif {
    /// Parse a TIFF-structured block, with or without the `Exif\0\0`
    /// identifier. Only IFD0 and its sub-directories are read; IFD1, the
    /// thumbnail, is dropped. `None` for a block that is not TIFF.
    #[must_use]
    pub fn parse(block: &[u8]) -> Option<Self> {
        let tiff = block.strip_prefix(b"Exif\0\0").unwrap_or(block);
        Self::parse_directory(tiff, None)
    }

    /// Parse the directory at `offset` of a TIFF-structured `tiff`, or IFD0
    /// where `offset` is `None`. A TIFF file's own directory reads this way.
    #[must_use]
    pub fn parse_directory(tiff: &[u8], offset: Option<usize>) -> Option<Self> {
        let big_endian = match tiff.get(..2)? {
            b"MM" => true,
            b"II" => false,
            _ => return None,
        };
        let r = Reader {
            data: tiff,
            big_endian,
        };
        if r.u16(2)? != 42 {
            return None;
        }
        let offset = match offset {
            Some(o) => o,
            None => usize::try_from(r.u32(4)?).ok()?,
        };
        Some(Self {
            big_endian,
            ifd0: r.ifd(offset, 2)?,
        })
    }

    /// An IFD0 entry.
    #[must_use]
    pub fn get(&self, tag: u16) -> Option<&ExifEntry> {
        self.ifd0.iter().find(|e| e.tag == tag)
    }

    /// A SHORT or LONG value of an IFD0 entry.
    #[must_use]
    pub fn integer(&self, tag: u16) -> Option<u32> {
        let e = self.get(tag)?;
        match e.kind {
            3 => {
                let b = [*e.data.first()?, *e.data.get(1)?];
                Some(u32::from(if self.big_endian {
                    u16::from_be_bytes(b)
                } else {
                    u16::from_le_bytes(b)
                }))
            }
            4 => {
                let b = [
                    *e.data.first()?,
                    *e.data.get(1)?,
                    *e.data.get(2)?,
                    *e.data.get(3)?,
                ];
                Some(if self.big_endian {
                    u32::from_be_bytes(b)
                } else {
                    u32::from_le_bytes(b)
                })
            }
            _ => None,
        }
    }

    /// A RATIONAL value of an IFD0 entry, as a float.
    #[must_use]
    pub fn rational(&self, tag: u16) -> Option<f64> {
        let e = self.get(tag)?;
        if e.kind != 5 {
            return None;
        }
        let word = |at: usize| -> Option<u32> {
            let b = [
                *e.data.get(at)?,
                *e.data.get(at + 1)?,
                *e.data.get(at + 2)?,
                *e.data.get(at + 3)?,
            ];
            Some(if self.big_endian {
                u32::from_be_bytes(b)
            } else {
                u32::from_le_bytes(b)
            })
        };
        let (n, d) = (word(0)?, word(4)?);
        (d != 0).then(|| f64::from(n) / f64::from(d))
    }

    /// The same block in the other byte order if `big_endian` differs:
    /// every value is swapped by its type's width (a RATIONAL as two words).
    /// UNDEFINED values, a maker note among them, are left as they are.
    #[must_use]
    pub fn with_byte_order(mut self, big_endian: bool) -> Self {
        if self.big_endian != big_endian {
            swap_entries(&mut self.ifd0);
            self.big_endian = big_endian;
        }
        self
    }

    /// Keep only IFD0 entries `keep` accepts; sub-directories always stay.
    #[must_use]
    pub fn retain(mut self, keep: impl Fn(u16) -> bool) -> Self {
        self.ifd0.retain(|e| e.sub.is_some() || keep(e.tag));
        self
    }

    /// Lay the block out again: the byte-order mark, IFD0 at offset 8, each
    /// directory followed by its out-of-line values and then its
    /// sub-directories.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(if self.big_endian { b"MM" } else { b"II" });
        out.extend_from_slice(&self.u16(42));
        out.extend_from_slice(&self.u32(8));
        self.write_ifd(&self.ifd0, &mut out, 0);
        out
    }

    /// The bytes `entries` take as a directory placed at `base` in a file
    /// whose offsets count from `origin`: for splicing into a TIFF file's
    /// own directory, the encoder writes the returned sub-directory blob at
    /// `base` and points entries at it.
    #[must_use]
    pub fn sub_directory_bytes(&self, entries: &[ExifEntry], base: u32) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_ifd(entries, &mut out, base);
        out
    }

    fn u16(&self, v: u16) -> [u8; 2] {
        if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    fn u32(&self, v: u32) -> [u8; 4] {
        if self.big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    }

    /// Append a directory at the end of `out`; offsets are `base` plus the
    /// position in `out`.
    fn write_ifd(&self, entries: &[ExifEntry], out: &mut Vec<u8>, base: u32) {
        let start = out.len();
        let n = entries.len().min(usize::from(u16::MAX));
        let entries = entries.get(..n).unwrap_or(entries);
        let table = 2 + 12 * n + 4;
        let pos = |len: usize| base.saturating_add(u32::try_from(len).unwrap_or(u32::MAX));
        // Out-of-line values first, after the table; sub-directories after.
        let mut values_len = 0;
        for e in entries {
            if e.sub.is_none() && e.data.len() > 4 {
                values_len += e.data.len() + (e.data.len() & 1);
            }
        }
        let mut value_at = start + table;
        let mut sub_at = start + table + values_len;
        let mut subs = Vec::new();
        out.extend_from_slice(&self.u16(n as u16));
        for e in entries {
            out.extend_from_slice(&self.u16(e.tag));
            if let Some(sub) = &e.sub {
                out.extend_from_slice(&self.u16(4));
                out.extend_from_slice(&self.u32(1));
                out.extend_from_slice(&self.u32(pos(sub_at)));
                let mut blob = Vec::new();
                self.write_ifd(sub, &mut blob, pos(sub_at));
                sub_at += blob.len();
                subs.push(blob);
                continue;
            }
            out.extend_from_slice(&self.u16(e.kind));
            out.extend_from_slice(&self.u32(e.count));
            if e.data.len() <= 4 {
                let mut inline = [0_u8; 4];
                for (slot, &b) in inline.iter_mut().zip(&e.data) {
                    *slot = b;
                }
                out.extend_from_slice(&inline);
            } else {
                out.extend_from_slice(&self.u32(pos(value_at)));
                value_at += e.data.len() + (e.data.len() & 1);
            }
        }
        out.extend_from_slice(&self.u32(0));
        for e in entries {
            if e.sub.is_none() && e.data.len() > 4 {
                out.extend_from_slice(&e.data);
                if e.data.len() & 1 == 1 {
                    out.push(0);
                }
            }
        }
        for blob in subs {
            out.extend_from_slice(&blob);
        }
    }
}

/// Swap every value of `entries` (and their sub-directories) end for end.
fn swap_entries(entries: &mut [ExifEntry]) {
    for e in entries {
        if let Some(sub) = &mut e.sub {
            swap_entries(sub);
            continue;
        }
        let width = match e.kind {
            3 | 8 => 2,
            4 | 9 | 11 | 5 | 10 => 4,
            12 => 8,
            _ => continue,
        };
        for word in e.data.chunks_exact_mut(width) {
            word.reverse();
        }
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

    fn sample(big_endian: bool) -> Exif {
        let e = Exif {
            big_endian,
            ifd0: Vec::new(),
        };
        let make = ExifEntry {
            tag: 0x010F,
            kind: 2,
            count: 6,
            data: b"Canon\0".to_vec(),
            sub: None,
        };
        let orient = ExifEntry {
            tag: 0x0112,
            kind: 3,
            count: 1,
            data: e.u16(6).to_vec(),
            sub: None,
        };
        let mut xres = e.u32(300).to_vec();
        xres.extend_from_slice(&e.u32(1));
        let xres = ExifEntry {
            tag: 0x011A,
            kind: 5,
            count: 1,
            data: xres,
            sub: None,
        };
        let exposure = ExifEntry {
            tag: 0x829A,
            kind: 5,
            count: 1,
            data: [e.u32(1), e.u32(250)].concat(),
            sub: None,
        };
        let exif_ifd = ExifEntry {
            tag: 0x8769,
            kind: 4,
            count: 1,
            data: Vec::new(),
            sub: Some(vec![exposure]),
        };
        Exif {
            big_endian,
            ifd0: vec![make, orient, xres, exif_ifd],
        }
    }

    #[test]
    fn a_block_round_trips_through_its_tree() {
        for big_endian in [false, true] {
            let exif = sample(big_endian);
            let bytes = exif.to_bytes();
            let back = Exif::parse(&bytes).unwrap();
            assert_eq!(back, exif);
            assert_eq!(back.integer(0x0112), Some(6));
            assert_eq!(back.rational(0x011A), Some(300.0));
            assert_eq!(
                crate::Orientation::from_exif_block(&bytes),
                crate::Orientation::from_exif(6)
            );
            let mut prefixed = b"Exif\0\0".to_vec();
            prefixed.extend_from_slice(&bytes);
            assert_eq!(Exif::parse(&prefixed).unwrap(), exif);
        }
    }

    #[test]
    fn a_block_converts_between_byte_orders() {
        let big = sample(true);
        let little = big.clone().with_byte_order(false);
        assert_eq!(little, sample(false));
        assert_eq!(little.with_byte_order(true), big);
    }

    #[test]
    fn retain_keeps_sub_directories() {
        let exif = sample(false).retain(|t| t == 0x0112);
        assert_eq!(
            exif.ifd0.iter().map(|e| e.tag).collect::<Vec<_>>(),
            [0x0112, 0x8769]
        );
    }

    #[test]
    fn hostile_blocks_never_panic() {
        let bytes = sample(true).to_bytes();
        for cut in 0..bytes.len() {
            let _ = Exif::parse(&bytes[..cut]);
        }
        let mut loops = bytes.clone();
        for i in 8..loops.len() {
            loops[i] ^= 0xA5;
            let _ = Exif::parse(&loops);
            loops[i] ^= 0xA5;
        }
        assert!(Exif::parse(b"not a tiff").is_none());
    }
}
