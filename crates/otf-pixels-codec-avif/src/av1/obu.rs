//! The Open Bitstream Unit layer (AV1 spec §5.3).
//!
//! An AV1 stream — whether the configuration OBUs in an `av1C` box or the coded
//! frame in an item's payload — is a sequence of OBUs, each a one- or two-byte
//! header followed by an optional `leb128` size and a payload. This module
//! splits a byte buffer into borrowed OBUs without interpreting their contents;
//! [`super::seq`] and [`super::frame`] interpret the ones that matter.
//!
//! AVIF mandates the *low-overhead* framing, in which every OBU carries its own
//! size field. This reader accepts a sizeless final OBU as well — it extends to
//! the end of the buffer — because that is the one unambiguous case and
//! rejecting it would turn a legal-enough stream into a spurious error.

use super::bits::BitReader;
use otf_pixels_core::{PixelsError, Result};

/// The kind of an OBU (`obu_type`, §6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObuType {
    /// `OBU_SEQUENCE_HEADER` (1).
    SequenceHeader,
    /// `OBU_TEMPORAL_DELIMITER` (2).
    TemporalDelimiter,
    /// `OBU_FRAME_HEADER` (3).
    FrameHeader,
    /// `OBU_TILE_GROUP` (4).
    TileGroup,
    /// `OBU_METADATA` (5).
    Metadata,
    /// `OBU_FRAME` (6) — a frame header immediately followed by tile data.
    Frame,
    /// `OBU_REDUNDANT_FRAME_HEADER` (7).
    RedundantFrameHeader,
    /// `OBU_TILE_LIST` (8).
    TileList,
    /// `OBU_PADDING` (15).
    Padding,
    /// Any reserved or unknown type, carried through so a stream that uses one
    /// is skipped rather than rejected.
    Reserved(u8),
}

impl ObuType {
    fn from_bits(value: u32) -> Self {
        match value {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other as u8),
        }
    }
}

/// A parsed OBU header (§5.3.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    /// The OBU type.
    pub kind: ObuType,
    /// `temporal_id` from the extension header, or 0 when absent.
    pub temporal_id: u8,
    /// `spatial_id` from the extension header, or 0 when absent.
    pub spatial_id: u8,
}

/// A single OBU: its header and a borrow of its payload bytes.
#[derive(Debug, Clone, Copy)]
pub struct Obu<'a> {
    /// The parsed header.
    pub header: ObuHeader,
    /// The payload, exactly `obu_size` bytes, ready for a fresh [`BitReader`].
    pub payload: &'a [u8],
    /// The byte offset of `payload` within the buffer `parse_stream` was given,
    /// so a caller can locate tile data absolutely without pointer arithmetic.
    pub payload_start: usize,
}

impl<'a> Obu<'a> {
    /// Split a byte buffer into its OBUs.
    ///
    /// Stops cleanly at the end of the buffer. Rejects a header whose forbidden
    /// bit is set and a size that would run past the buffer — the two ways OBU
    /// framing is used to point a decoder outside its input.
    pub fn parse_stream(data: &'a [u8]) -> Result<Vec<Obu<'a>>> {
        let mut obus = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            let Some(rest) = data.get(pos..) else {
                break;
            };
            let mut reader = BitReader::new(rest);

            if reader.f(1)? != 0 {
                return Err(PixelsError::malformed(
                    "avif",
                    "an AV1 OBU forbidden bit was set",
                ));
            }
            let kind = ObuType::from_bits(reader.f(4)?);
            let extension_flag = reader.flag()?;
            let has_size_field = reader.flag()?;
            let _obu_reserved_1bit = reader.f(1)?;

            let (temporal_id, spatial_id) = if extension_flag {
                let temporal_id = reader.f(3)? as u8;
                let spatial_id = reader.f(2)? as u8;
                let _extension_reserved_3bits = reader.f(3)?;
                (temporal_id, spatial_id)
            } else {
                (0, 0)
            };

            let payload_len = if has_size_field {
                usize::try_from(reader.leb128()?).map_err(|_| {
                    PixelsError::malformed("avif", "an AV1 OBU size exceeds this platform's usize")
                })?
            } else {
                // A sizeless OBU runs to the end of the buffer; only the last
                // OBU can legally be sizeless.
                rest.len().saturating_sub(reader.byte_position())
            };

            let payload_start = pos + reader.byte_position();
            let payload_end = payload_start.checked_add(payload_len).filter(|&e| e <= data.len());
            let Some(payload_end) = payload_end else {
                return Err(PixelsError::malformed(
                    "avif",
                    "an AV1 OBU declares more bytes than the stream holds",
                ));
            };
            let Some(payload) = data.get(payload_start..payload_end) else {
                return Err(PixelsError::malformed(
                    "avif",
                    "an AV1 OBU payload range is invalid",
                ));
            };

            obus.push(Obu {
                header: ObuHeader {
                    kind,
                    temporal_id,
                    spatial_id,
                },
                payload,
                payload_start,
            });
            pos = payload_end;
        }
        Ok(obus)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unusual_byte_groupings,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::ErrorCode;

    /// Build a low-overhead OBU: header byte with the size-field flag set,
    /// a single-byte leb128 length, then the payload.
    fn sized_obu(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // forbidden=0, type=kind, ext=0, has_size=1, reserved=0
        let header = (kind << 3) | 0b0000_0010;
        out.push(header);
        assert!(payload.len() < 128, "test payloads stay single-byte leb128");
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn splits_a_stream_into_sized_obus() {
        let mut stream = sized_obu(2, &[]); // temporal delimiter
        stream.extend(sized_obu(1, &[0xAA, 0xBB])); // sequence header
        stream.extend(sized_obu(6, &[0xCC])); // frame

        let obus = Obu::parse_stream(&stream).unwrap();
        assert_eq!(obus.len(), 3);
        assert_eq!(obus[0].header.kind, ObuType::TemporalDelimiter);
        assert_eq!(obus[0].payload, &[] as &[u8]);
        assert_eq!(obus[1].header.kind, ObuType::SequenceHeader);
        assert_eq!(obus[1].payload, &[0xAA, 0xBB]);
        assert_eq!(obus[2].header.kind, ObuType::Frame);
        assert_eq!(obus[2].payload, &[0xCC]);
    }

    #[test]
    fn reads_the_extension_header_ids() {
        // forbidden=0 type=1 ext=1 has_size=1 reserved=0 -> 0b0_0001_1_1_0
        let mut stream = vec![0b0000_1110];
        // extension byte: temporal_id=3 (011) spatial_id=2 (10) reserved=0(000)
        stream.push(0b011_10_000);
        stream.push(0x01); // leb128 size = 1
        stream.push(0x55); // payload
        let obus = Obu::parse_stream(&stream).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].header.temporal_id, 3);
        assert_eq!(obus[0].header.spatial_id, 2);
        assert_eq!(obus[0].payload, &[0x55]);
    }

    #[test]
    fn a_sizeless_final_obu_takes_the_rest_of_the_buffer() {
        // header: type=6 ext=0 has_size=0 -> 0b0_0110_0_0_0
        let mut stream = vec![0b0011_0000];
        stream.extend_from_slice(&[1, 2, 3, 4]);
        let obus = Obu::parse_stream(&stream).unwrap();
        assert_eq!(obus.len(), 1);
        assert_eq!(obus[0].header.kind, ObuType::Frame);
        assert_eq!(obus[0].payload, &[1, 2, 3, 4]);
    }

    #[test]
    fn a_size_past_the_buffer_is_rejected() {
        // has_size=1, size=200, but only a few bytes follow.
        let stream = vec![0b0011_0010, 200, 1, 2, 3];
        let err = Obu::parse_stream(&stream).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
    }

    #[test]
    fn a_set_forbidden_bit_is_rejected() {
        let stream = vec![0b1011_0010, 0x00];
        let err = Obu::parse_stream(&stream).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
    }

    #[test]
    fn an_unknown_type_is_carried_through_not_rejected() {
        let stream = sized_obu(9, &[0x01]);
        let obus = Obu::parse_stream(&stream).unwrap();
        assert_eq!(obus[0].header.kind, ObuType::Reserved(9));
    }
}
