//! [`Orientation`] — how a stored image must be turned to display upright.
//!
//! Cameras write pixels in sensor order and record the way the device was held
//! as metadata rather than rotating the pixels: EXIF's `Orientation` tag in
//! JPEG, TIFF, PNG and WebP, and the `irot`/`imir` properties in HEIF/AVIF.
//! Decoders report it here; applying it is a pipeline decision (`auto_orient`,
//! SPEC §Safety and limits), so a decoder never rotates its own output.

/// One of the eight orientations the EXIF/TIFF `Orientation` tag (274) can
/// name.
///
/// Each variant describes the transform that makes the stored image upright.
/// All eight are a clockwise quarter-turn rotation followed, optionally, by a
/// horizontal mirror, which is the form [`Orientation::clockwise_turns`] and
/// [`Orientation::mirrored`] expose and [`Orientation::from_parts`] builds
/// from. HEIF's rotate-then-mirror properties reduce to the same form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Orientation {
    /// EXIF 1: already upright.
    #[default]
    Normal,
    /// EXIF 2: mirror left to right.
    FlipHorizontal,
    /// EXIF 3: rotate 180 degrees.
    Rotate180,
    /// EXIF 4: mirror top to bottom.
    FlipVertical,
    /// EXIF 5: mirror across the top-left to bottom-right diagonal.
    Transpose,
    /// EXIF 6: rotate 90 degrees clockwise.
    Rotate90,
    /// EXIF 7: mirror across the top-right to bottom-left diagonal.
    Transverse,
    /// EXIF 8: rotate 270 degrees clockwise, i.e. 90 anticlockwise.
    Rotate270,
}

impl Orientation {
    /// The orientation an EXIF `Orientation` value names, or `None` for a value
    /// outside 1–8.
    #[must_use]
    pub const fn from_exif(value: u16) -> Option<Self> {
        Some(match value {
            1 => Self::Normal,
            2 => Self::FlipHorizontal,
            3 => Self::Rotate180,
            4 => Self::FlipVertical,
            5 => Self::Transpose,
            6 => Self::Rotate90,
            7 => Self::Transverse,
            8 => Self::Rotate270,
            _ => return None,
        })
    }

    /// The EXIF `Orientation` value, 1–8.
    #[must_use]
    pub const fn exif(self) -> u8 {
        match self {
            Self::Normal => 1,
            Self::FlipHorizontal => 2,
            Self::Rotate180 => 3,
            Self::FlipVertical => 4,
            Self::Transpose => 5,
            Self::Rotate90 => 6,
            Self::Transverse => 7,
            Self::Rotate270 => 8,
        }
    }

    /// Rotate `turns` quarter turns clockwise (taken modulo 4), then mirror
    /// left to right if `mirrored`.
    #[must_use]
    pub const fn from_parts(turns: u8, mirrored: bool) -> Self {
        match (turns % 4, mirrored) {
            (0, false) => Self::Normal,
            (0, true) => Self::FlipHorizontal,
            (1, false) => Self::Rotate90,
            // Transpose sends (x, y) to (y, x): a clockwise quarter turn sends
            // it to (h - 1 - y, x), and the mirror then undoes the reflection.
            (1, true) => Self::Transpose,
            (2, false) => Self::Rotate180,
            // A half turn mirrors both axes; mirroring left to right again
            // leaves only the vertical one.
            (2, true) => Self::FlipVertical,
            (3, false) => Self::Rotate270,
            _ => Self::Transverse,
        }
    }

    /// Clockwise quarter turns to apply first, 0–3.
    #[must_use]
    pub const fn clockwise_turns(self) -> u8 {
        match self {
            Self::Normal | Self::FlipHorizontal => 0,
            Self::Rotate90 | Self::Transpose => 1,
            Self::Rotate180 | Self::FlipVertical => 2,
            Self::Rotate270 | Self::Transverse => 3,
        }
    }

    /// Whether a left-to-right mirror follows the rotation.
    #[must_use]
    pub const fn mirrored(self) -> bool {
        matches!(
            self,
            Self::FlipHorizontal | Self::FlipVertical | Self::Transpose | Self::Transverse
        )
    }

    /// Whether displaying upright exchanges width and height.
    #[must_use]
    pub const fn transposes(self) -> bool {
        self.clockwise_turns() % 2 == 1
    }

    /// Read the `Orientation` tag from an EXIF block.
    ///
    /// `exif` is the TIFF structure EXIF is made of, optionally behind the
    /// `Exif\0\0` identifier a JPEG APP1 segment carries. PNG's `eXIf` and
    /// WebP's `EXIF` chunk are specified without the identifier, but writers
    /// that copy it across from a JPEG are common, so both are accepted.
    ///
    /// A missing tag, an out-of-range value or a malformed block all yield
    /// `None` rather than an error: broken metadata is not a broken image,
    /// and refusing to decode a photograph over it would be the wrong trade.
    #[must_use]
    pub fn from_exif_block(exif: &[u8]) -> Option<Self> {
        let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);

        let big_endian = match tiff.get(..2)? {
            b"MM" => true,
            b"II" => false,
            _ => return None,
        };
        let short = |at: usize| -> Option<u16> {
            let bytes = [*tiff.get(at)?, *tiff.get(at.checked_add(1)?)?];
            Some(if big_endian {
                u16::from_be_bytes(bytes)
            } else {
                u16::from_le_bytes(bytes)
            })
        };
        let long = |at: usize| -> Option<u32> {
            let bytes = tiff.get(at..at.checked_add(4)?)?;
            let bytes = [
                *bytes.first()?,
                *bytes.get(1)?,
                *bytes.get(2)?,
                *bytes.get(3)?,
            ];
            Some(if big_endian {
                u32::from_be_bytes(bytes)
            } else {
                u32::from_le_bytes(bytes)
            })
        };

        if short(2)? != 42 {
            return None;
        }
        let ifd = usize::try_from(long(4)?).ok()?;
        let entries = short(ifd)?;
        for entry in 0..usize::from(entries) {
            let at = ifd.checked_add(2)?.checked_add(entry.checked_mul(12)?)?;
            // 0x0112 is Orientation; a SHORT, so its single value sits in the
            // first two bytes of the value field rather than at an offset.
            if short(at)? == 0x0112 {
                return Self::from_exif(short(at.checked_add(8)?)?);
            }
        }
        None
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests assert on known-good values"
)]
mod tests {
    use super::*;

    const ALL: [Orientation; 8] = [
        Orientation::Normal,
        Orientation::FlipHorizontal,
        Orientation::Rotate180,
        Orientation::FlipVertical,
        Orientation::Transpose,
        Orientation::Rotate90,
        Orientation::Transverse,
        Orientation::Rotate270,
    ];

    /// Apply `orientation` to a `w` x `h` grid of distinct values by the
    /// parts decomposition, one primitive at a time.
    fn by_parts(orientation: Orientation, w: usize, h: usize) -> (Vec<usize>, usize, usize) {
        let mut grid: Vec<usize> = (0..w * h).collect();
        let (mut w, mut h) = (w, h);
        for _ in 0..orientation.clockwise_turns() {
            // Clockwise: output (x, y) comes from input (y, h - 1 - x).
            let mut turned = vec![0; w * h];
            for y in 0..w {
                for x in 0..h {
                    turned[y * h + x] = grid[(h - 1 - x) * w + y];
                }
            }
            grid = turned;
            (w, h) = (h, w);
        }
        if orientation.mirrored() {
            for row in grid.chunks_mut(w) {
                row.reverse();
            }
        }
        (grid, w, h)
    }

    /// The same, from the EXIF definitions directly: where the stored image's
    /// 0th row and 0th column end up (TIFF 6.0 / EXIF 2.3 §4.6.4).
    fn by_definition(orientation: Orientation, w: usize, h: usize) -> (Vec<usize>, usize, usize) {
        let (ow, oh) = if orientation.transposes() {
            (h, w)
        } else {
            (w, h)
        };
        let mut grid = vec![0; w * h];
        for y in 0..oh {
            for x in 0..ow {
                let (sx, sy) = match orientation {
                    Orientation::Normal => (x, y),
                    Orientation::FlipHorizontal => (w - 1 - x, y),
                    Orientation::Rotate180 => (w - 1 - x, h - 1 - y),
                    Orientation::FlipVertical => (x, h - 1 - y),
                    Orientation::Transpose => (y, x),
                    Orientation::Rotate90 => (y, h - 1 - x),
                    Orientation::Transverse => (w - 1 - y, h - 1 - x),
                    Orientation::Rotate270 => (w - 1 - y, x),
                };
                grid[y * ow + x] = sy * w + sx;
            }
        }
        (grid, ow, oh)
    }

    #[test]
    fn the_parts_decomposition_matches_the_exif_definitions() {
        for orientation in ALL {
            assert_eq!(
                by_parts(orientation, 3, 2),
                by_definition(orientation, 3, 2),
                "{orientation:?}"
            );
        }
    }

    #[test]
    fn parts_and_exif_values_round_trip() {
        for orientation in ALL {
            assert_eq!(
                Orientation::from_parts(orientation.clockwise_turns(), orientation.mirrored()),
                orientation
            );
            assert_eq!(
                Orientation::from_exif(u16::from(orientation.exif())),
                Some(orientation)
            );
        }
        assert_eq!(Orientation::from_parts(5, false), Orientation::Rotate90);
        assert_eq!(Orientation::from_exif(0), None);
        assert_eq!(Orientation::from_exif(9), None);
    }

    #[test]
    fn exif_orientation_is_read_from_both_byte_orders() {
        // Little-endian: II, 42, IFD at 8, one entry, tag 0x0112, SHORT, 1, 6.
        let little = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0";
        assert_eq!(
            Orientation::from_exif_block(little),
            Some(Orientation::Rotate90)
        );

        let big = b"Exif\0\0MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x03\0\0";
        assert_eq!(
            Orientation::from_exif_block(big),
            Some(Orientation::Rotate180)
        );
    }

    #[test]
    fn the_exif_identifier_is_optional() {
        let bare = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x08\0\0\0";
        assert_eq!(
            Orientation::from_exif_block(bare),
            Some(Orientation::Rotate270)
        );
    }

    #[test]
    fn broken_exif_yields_no_orientation_rather_than_an_error() {
        assert_eq!(
            Orientation::from_exif_block(b"Exif\0\0XX*\0\x08\0\0\0"),
            None
        );
        assert_eq!(Orientation::from_exif_block(b"Exif\0\0II*\0"), None);
        assert_eq!(Orientation::from_exif_block(b"not exif at all"), None);
        assert_eq!(Orientation::from_exif_block(b""), None);
        // An IFD offset pointing far past the end.
        assert_eq!(Orientation::from_exif_block(b"II*\0\xff\xff\xff\xff"), None);
        // An entry count promising more entries than the block holds.
        assert_eq!(
            Orientation::from_exif_block(b"II*\0\x08\0\0\0\xff\xff"),
            None
        );
        // An out-of-range orientation is metadata we decline to trust.
        let bogus = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x09\0\0\0";
        assert_eq!(Orientation::from_exif_block(bogus), None);
    }

    #[test]
    fn a_block_without_the_tag_has_no_orientation() {
        // One entry, tag 0x0100 (ImageWidth).
        let other = b"II*\0\x08\0\0\0\x01\0\x00\x01\x03\0\x01\0\0\0\x06\0\0\0";
        assert_eq!(Orientation::from_exif_block(other), None);
    }
}
