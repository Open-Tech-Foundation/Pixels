//! JPEG marker segments: the structures a decoder parses before any pixel.
//!
//! JPEG is a sequence of marker segments, each `FF xx` followed (for most
//! markers) by a big-endian 16-bit length that *includes* the length field
//! itself. Everything here parses from a byte slice already known to be the
//! segment payload, so segment framing is the reader's job and validation is
//! this module's.

use otf_pixels_core::{PixelsError, Result};

/// Marker bytes, without the preceding `0xFF`.
pub mod marker {
    /// Start of image.
    pub const SOI: u8 = 0xD8;
    /// End of image.
    pub const EOI: u8 = 0xD9;
    /// Baseline DCT, Huffman coded — the only frame type we decode ourselves.
    pub const SOF0: u8 = 0xC0;
    /// Extended sequential DCT. Identical to baseline for our purposes: the
    /// difference is 12-bit precision support, which the frame header states.
    pub const SOF1: u8 = 0xC1;
    /// Progressive DCT.
    pub const SOF2: u8 = 0xC2;
    /// Define Huffman tables.
    pub const DHT: u8 = 0xC4;
    /// Define arithmetic coding conditioning.
    pub const DAC: u8 = 0xCC;
    /// Start of scan.
    pub const SOS: u8 = 0xDA;
    /// Define quantization tables.
    pub const DQT: u8 = 0xDB;
    /// Define restart interval.
    pub const DRI: u8 = 0xDD;
    /// First restart marker; there are eight, `RST0..=RST7`.
    pub const RST0: u8 = 0xD0;
    /// Last restart marker.
    pub const RST7: u8 = 0xD7;
    /// First application segment (`APP0`, JFIF).
    pub const APP0: u8 = 0xE0;
    /// EXIF lives here.
    pub const APP1: u8 = 0xE1;
    /// Adobe's colour transform flag lives here.
    pub const APP14: u8 = 0xEE;
    /// Last application segment.
    pub const APP15: u8 = 0xEF;
    /// Comment.
    pub const COM: u8 = 0xFE;
    /// Temporary marker, and the only `0xFF01` that is not a segment.
    pub const TEM: u8 = 0x01;

    /// Whether `code` is one of the eight restart markers.
    #[must_use]
    pub const fn is_restart(code: u8) -> bool {
        code >= RST0 && code <= RST7
    }

    /// Whether `code` is a start-of-frame marker of any kind.
    ///
    /// `DHT`, `DAC` and the restart markers sit inside the `0xC0..=0xCF`
    /// range without being frame headers, which is why this is not a range
    /// test.
    #[must_use]
    pub const fn is_frame(code: u8) -> bool {
        matches!(code, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
    }

    /// Whether `code` is a standalone marker carrying no length or payload.
    #[must_use]
    pub const fn is_standalone(code: u8) -> bool {
        is_restart(code) || matches!(code, SOI | EOI | TEM)
    }
}

/// The two-byte prefix every JPEG stream starts with, plus the marker that
/// must follow it.
///
/// Detection is by magic bytes only (SPEC §Formats). `FF D8` alone is a weak
/// signature — two bytes match by chance often — so the third byte, which
/// must begin the next marker, is part of what we check.
pub const SIGNATURE: [u8; 3] = [0xFF, 0xD8, 0xFF];

/// Natural (row-major) position of each coefficient in zigzag order.
///
/// Coefficients arrive zigzagged so that the low frequencies, which carry
/// nearly all the energy, come first and the long run of high-frequency zeros
/// lands at the end where the end-of-block code can collapse it.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, //
    17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, //
    27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, //
    29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, //
    53, 60, 61, 54, 47, 55, 62, 63,
];

/// One component of a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    /// The component identifier scans refer to it by.
    pub id: u8,
    /// Horizontal sampling factor, 1..=4.
    pub h: u8,
    /// Vertical sampling factor, 1..=4.
    pub v: u8,
    /// Which of the four quantization table slots this component uses.
    pub quant: u8,
}

/// A parsed start-of-frame header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// Sample precision in bits. Baseline is always 8.
    pub precision: u8,
    /// Image width in pixels.
    pub width: u16,
    /// Image height in pixels. Zero here means the height is deferred to a
    /// `DNL` marker after the first scan.
    pub height: u16,
    /// The components, in the order the frame declares them.
    pub components: Vec<Component>,
}

impl Frame {
    /// Parse a `SOF0`/`SOF1` payload (the bytes after the segment length).
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a truncated header, a zero
    /// width, an out-of-range sampling factor or a duplicate component id.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let (Some(&precision), Some(&count)) = (payload.first(), payload.get(5)) else {
            return Err(PixelsError::malformed("jpeg", "frame header is truncated"));
        };
        let height = be16(payload.get(1..3))?;
        let width = be16(payload.get(3..5))?;

        // Precision is the one field that tells baseline from 12-bit extended
        // sequential; the marker does not.
        if precision != 8 {
            return Err(PixelsError::unsupported(format!(
                "jpeg: {precision}-bit samples; baseline JPEG is 8-bit"
            )));
        }
        if width == 0 {
            return Err(PixelsError::malformed("jpeg", "frame declares zero width"));
        }
        // A zero height is legal only as a promise that DNL will supply it,
        // which no real encoder emits and which we do not support.
        if height == 0 {
            return Err(PixelsError::unsupported(
                "jpeg: height deferred to a DNL marker",
            ));
        }
        if !(1..=4).contains(&count) {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("frame declares {count} components; 1..=4 is the legal range"),
            ));
        }

        let mut components = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            let at = 6 + index * 3;
            let (Some(&id), Some(&sampling), Some(&quant)) =
                (payload.get(at), payload.get(at + 1), payload.get(at + 2))
            else {
                return Err(PixelsError::malformed(
                    "jpeg",
                    "frame header ends mid-component",
                ));
            };
            let (h, v) = (sampling >> 4, sampling & 0x0F);
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("component {id} has sampling factors {h}x{v}; 1..=4 each"),
                ));
            }
            if quant > 3 {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("component {id} names quantization table {quant}; only 0..=3 exist"),
                ));
            }
            if components.iter().any(|c: &Component| c.id == id) {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("component id {id} appears twice"),
                ));
            }
            components.push(Component { id, h, v, quant });
        }

        Ok(Self {
            precision,
            width,
            height,
            components,
        })
    }

    /// The largest horizontal sampling factor across components.
    #[must_use]
    pub fn h_max(&self) -> u8 {
        self.components.iter().map(|c| c.h).max().unwrap_or(1)
    }

    /// The largest vertical sampling factor across components.
    #[must_use]
    pub fn v_max(&self) -> u8 {
        self.components.iter().map(|c| c.v).max().unwrap_or(1)
    }
}

/// One component of a scan, naming the entropy tables it is coded with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScanComponent {
    /// Index into [`Frame::components`], resolved from the component id.
    pub index: usize,
    /// DC Huffman table slot, 0..=3.
    pub dc: u8,
    /// AC Huffman table slot, 0..=3.
    pub ac: u8,
}

/// A parsed start-of-scan header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scan {
    /// The components this scan codes, in the order they interleave.
    pub components: Vec<ScanComponent>,
    /// First coefficient in the spectral selection.
    pub spectral_start: u8,
    /// Last coefficient in the spectral selection.
    pub spectral_end: u8,
    /// Successive approximation high bit.
    pub approx_high: u8,
    /// Successive approximation low bit.
    pub approx_low: u8,
}

impl Scan {
    /// Parse an `SOS` payload against the frame it belongs to.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a truncated header, a component
    /// id the frame never declared, or a table slot above 3.
    pub fn parse(payload: &[u8], frame: &Frame) -> Result<Self> {
        let Some(&count) = payload.first() else {
            return Err(PixelsError::malformed("jpeg", "scan header is truncated"));
        };
        if count == 0 || count as usize > frame.components.len() {
            return Err(PixelsError::malformed(
                "jpeg",
                format!(
                    "scan declares {count} components; the frame has {}",
                    frame.components.len()
                ),
            ));
        }

        let mut components = Vec::with_capacity(count as usize);
        for slot in 0..count as usize {
            let at = 1 + slot * 2;
            let (Some(&id), Some(&tables)) = (payload.get(at), payload.get(at + 1)) else {
                return Err(PixelsError::malformed(
                    "jpeg",
                    "scan header ends mid-component",
                ));
            };
            let Some(index) = frame.components.iter().position(|c| c.id == id) else {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("scan names component {id}, which the frame does not declare"),
                ));
            };
            let (dc, ac) = (tables >> 4, tables & 0x0F);
            if dc > 3 || ac > 3 {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("component {id} names Huffman tables {dc}/{ac}; only 0..=3 exist"),
                ));
            }
            components.push(ScanComponent { index, dc, ac });
        }

        let tail = 1 + count as usize * 2;
        let (Some(&spectral_start), Some(&spectral_end), Some(&approx)) = (
            payload.get(tail),
            payload.get(tail + 1),
            payload.get(tail + 2),
        ) else {
            return Err(PixelsError::malformed(
                "jpeg",
                "scan header is missing its spectral selection",
            ));
        };

        Ok(Self {
            components,
            spectral_start,
            spectral_end,
            approx_high: approx >> 4,
            approx_low: approx & 0x0F,
        })
    }
}

/// The colour transform an Adobe `APP14` segment declares.
///
/// Without it, a three-component JPEG is assumed to be YCbCr — which is right
/// essentially always, the exception being files that label their components
/// `'R'`, `'G'`, `'B'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdobeTransform {
    /// No transform: the components are already RGB (or CMYK).
    None,
    /// YCbCr.
    YCbCr,
    /// YCCK — four components, the first three transformed.
    YCck,
}

/// Read the transform byte out of an `APP14` payload, if it is Adobe's.
#[must_use]
pub fn adobe_transform(payload: &[u8]) -> Option<AdobeTransform> {
    if payload.get(..5) != Some(b"Adobe") {
        return None;
    }
    // The transform is the last byte of a 12-byte payload (5 tag + 2 version
    // + 3 flags + 1 transform); short segments carry no opinion.
    match payload.get(11) {
        Some(0) => Some(AdobeTransform::None),
        Some(1) => Some(AdobeTransform::YCbCr),
        Some(2) => Some(AdobeTransform::YCck),
        _ => None,
    }
}

/// The EXIF orientation tag, 1..=8, if `payload` is an EXIF `APP1` segment
/// that carries one.
///
/// Failure at any step returns `None` rather than an error: a broken EXIF
/// block is not a broken image, and refusing to decode a photograph because
/// its metadata is malformed would be the wrong trade.
#[must_use]
pub fn exif_orientation(payload: &[u8]) -> Option<u8> {
    let tiff = payload.strip_prefix(b"Exif\0\0")?;

    let big_endian = match tiff.get(..2)? {
        b"MM" => true,
        b"II" => false,
        _ => return None,
    };
    let short = |at: usize| -> Option<u16> {
        let bytes = [*tiff.get(at)?, *tiff.get(at + 1)?];
        Some(if big_endian {
            u16::from_be_bytes(bytes)
        } else {
            u16::from_le_bytes(bytes)
        })
    };
    let long = |at: usize| -> Option<u32> {
        let bytes = [
            *tiff.get(at)?,
            *tiff.get(at + 1)?,
            *tiff.get(at + 2)?,
            *tiff.get(at + 3)?,
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
    let ifd = long(4)? as usize;
    let entries = short(ifd)?;
    for entry in 0..entries as usize {
        let at = ifd.checked_add(2)?.checked_add(entry.checked_mul(12)?)?;
        // 0x0112 is Orientation; a SHORT, so its single value sits in the
        // first two bytes of the value field rather than at an offset.
        if short(at)? == 0x0112 {
            let value = short(at + 8)?;
            return (1..=8).contains(&value).then_some(value as u8);
        }
    }
    None
}

/// Read a big-endian `u16` out of an exactly-two-byte slice.
fn be16(bytes: Option<&[u8]>) -> Result<u16> {
    match bytes {
        Some(&[hi, lo]) => Ok(u16::from_be_bytes([hi, lo])),
        _ => Err(PixelsError::malformed(
            "jpeg",
            "segment ends where a 16-bit field was expected",
        )),
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

    /// A 16x8 YCbCr frame header with 2x1 chroma subsampling.
    fn sof_payload() -> Vec<u8> {
        vec![
            8, // precision
            0, 8, // height
            0, 16, // width
            3,  // components
            1, 0x21, 0, // Y,  2x1, quant 0
            2, 0x11, 1, // Cb, 1x1, quant 1
            3, 0x11, 1, // Cr, 1x1, quant 1
        ]
    }

    #[test]
    fn frame_header_parses_components_and_sampling() {
        let frame = Frame::parse(&sof_payload()).unwrap();
        assert_eq!((frame.width, frame.height), (16, 8));
        assert_eq!(frame.components.len(), 3);
        assert_eq!(
            frame.components[0],
            Component {
                id: 1,
                h: 2,
                v: 1,
                quant: 0
            }
        );
        assert_eq!((frame.h_max(), frame.v_max()), (2, 1));
    }

    #[test]
    fn zigzag_is_a_permutation_of_the_block() {
        let mut seen = [false; 64];
        for &position in &ZIGZAG {
            assert!(!seen[position], "position {position} appears twice");
            seen[position] = true;
        }
        assert!(seen.iter().all(|&s| s));
        // The DC coefficient is first and the highest frequency is last.
        assert_eq!(ZIGZAG[0], 0);
        assert_eq!(ZIGZAG[63], 63);
    }

    #[test]
    fn twelve_bit_precision_is_unsupported_not_malformed() {
        let mut payload = sof_payload();
        payload[0] = 12;
        assert_eq!(
            Frame::parse(&payload).unwrap_err().code(),
            ErrorCode::Unsupported
        );
    }

    #[test]
    fn frame_header_rejects_degenerate_shapes() {
        // Zero width.
        let mut payload = sof_payload();
        payload[3] = 0;
        payload[4] = 0;
        assert_eq!(
            Frame::parse(&payload).unwrap_err().code(),
            ErrorCode::Malformed
        );

        // Sampling factor of zero would make an MCU zero blocks wide.
        let mut payload = sof_payload();
        payload[7] = 0x01;
        assert_eq!(
            Frame::parse(&payload).unwrap_err().code(),
            ErrorCode::Malformed
        );

        // Duplicate component ids leave scans unable to name one of them.
        let mut payload = sof_payload();
        payload[9] = 1;
        assert_eq!(
            Frame::parse(&payload).unwrap_err().code(),
            ErrorCode::Malformed
        );

        // Truncated mid-component.
        assert_eq!(
            Frame::parse(&sof_payload()[..10]).unwrap_err().code(),
            ErrorCode::Malformed
        );
    }

    #[test]
    fn scan_header_resolves_component_ids_to_frame_indices() {
        let frame = Frame::parse(&sof_payload()).unwrap();
        // Scan components in a different order from the frame.
        let payload = [3, 3, 0x11, 1, 0x00, 2, 0x11, 0, 63, 0];
        let scan = Scan::parse(&payload, &frame).unwrap();
        assert_eq!(scan.components.len(), 3);
        assert_eq!(
            scan.components[0],
            ScanComponent {
                index: 2,
                dc: 1,
                ac: 1
            }
        );
        assert_eq!(
            scan.components[1],
            ScanComponent {
                index: 0,
                dc: 0,
                ac: 0
            }
        );
        assert_eq!((scan.spectral_start, scan.spectral_end), (0, 63));
    }

    #[test]
    fn scan_naming_an_absent_component_is_malformed() {
        let frame = Frame::parse(&sof_payload()).unwrap();
        let payload = [1, 9, 0x00, 0, 63, 0];
        assert_eq!(
            Scan::parse(&payload, &frame).unwrap_err().code(),
            ErrorCode::Malformed
        );
    }

    #[test]
    fn adobe_transform_is_read_only_from_adobe_segments() {
        let mut payload = b"Adobe\0\x64\0\0\0\0\x01".to_vec();
        assert_eq!(adobe_transform(&payload), Some(AdobeTransform::YCbCr));
        payload[11] = 0;
        assert_eq!(adobe_transform(&payload), Some(AdobeTransform::None));
        assert_eq!(adobe_transform(b"JFIF\0\0\0\0\0\0\0\0"), None);
        assert_eq!(adobe_transform(b"Adobe"), None);
    }

    #[test]
    fn exif_orientation_is_read_from_both_byte_orders() {
        // Little-endian: II, 42, IFD at 8, one entry, tag 0x0112, SHORT, 1, 6.
        let little = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0";
        assert_eq!(exif_orientation(little), Some(6));

        let big = b"Exif\0\0MM\0*\0\0\0\x08\0\x01\x01\x12\0\x03\0\0\0\x01\0\x03\0\0";
        assert_eq!(exif_orientation(big), Some(3));
    }

    #[test]
    fn broken_exif_yields_no_orientation_rather_than_an_error() {
        assert_eq!(exif_orientation(b"Exif\0\0XX*\0\x08\0\0\0"), None);
        assert_eq!(exif_orientation(b"Exif\0\0II*\0"), None);
        assert_eq!(exif_orientation(b"not exif at all"), None);
        // An out-of-range orientation is metadata we decline to trust.
        let bogus = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x09\0\0\0";
        assert_eq!(exif_orientation(bogus), None);
    }

    #[test]
    fn marker_classification_excludes_the_impostors_in_the_c0_range() {
        assert!(marker::is_frame(marker::SOF0));
        assert!(marker::is_frame(marker::SOF2));
        assert!(!marker::is_frame(marker::DHT));
        assert!(!marker::is_frame(marker::DAC));
        assert!(!marker::is_frame(marker::RST0));
        assert!(marker::is_restart(marker::RST7));
        assert!(!marker::is_restart(marker::SOS));
        assert!(marker::is_standalone(marker::EOI));
        assert!(!marker::is_standalone(marker::DQT));
    }
}
