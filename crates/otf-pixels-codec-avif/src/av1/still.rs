//! The still-picture front-end driver: OBUs to parsed headers.
//!
//! This ties the layers below together the way an AVIF still is actually laid
//! out. The sequence header arrives in the `av1C` box's configuration OBUs; the
//! coded frame arrives as the primary item's data, as an `OBU_FRAME` (a frame
//! header immediately followed by its tile group) or as a separate
//! `OBU_FRAME_HEADER` plus `OBU_TILE_GROUP`. Either way this returns the two
//! parsed headers and the byte range of the tile-group data that follows them,
//! which is where P2 stops and reconstruction picks up.

use super::bits::BitReader;
use super::frame::FrameHeader;
use super::obu::{Obu, ObuType};
use super::seq::SequenceHeader;
use otf_pixels_core::{PixelsError, Result};

/// The parsed headers of a still picture, plus a locator for its tile data.
#[derive(Debug, Clone)]
pub struct StillPicture {
    /// The sequence header governing the frame.
    pub sequence: SequenceHeader,
    /// The frame (uncompressed) header.
    pub frame: FrameHeader,
    /// The byte offset of the tile-group data within the coded frame buffer.
    ///
    /// For an `OBU_FRAME` this points inside that OBU's payload, past the
    /// byte-aligned frame header. For a separate `OBU_TILE_GROUP` it points at
    /// that OBU's payload. It is an offset into the `frame_data` passed to
    /// [`StillPicture::parse`].
    pub tile_data_offset: usize,
    /// The length in bytes of the tile-group data.
    pub tile_data_len: usize,
}

/// Parse the sequence header out of a run of configuration OBUs.
///
/// The `av1C` config OBUs must contain exactly the sequence header (and may
/// contain metadata or padding around it). The first sequence header found
/// wins.
pub fn sequence_header_from_config(config_obus: &[u8]) -> Result<SequenceHeader> {
    for obu in Obu::parse_stream(config_obus)? {
        if obu.header.kind == ObuType::SequenceHeader {
            let mut reader = BitReader::new(obu.payload);
            return SequenceHeader::parse(&mut reader);
        }
    }
    Err(PixelsError::malformed(
        "avif",
        "the av1C configuration OBUs contain no sequence header",
    ))
}

impl StillPicture {
    /// Parse the headers of a coded still picture.
    ///
    /// `config_obus` is the `av1C` configuration OBU run; `frame_data` is the
    /// coded frame. A sequence header repeated in `frame_data` overrides the
    /// configuration one, as the spec allows.
    pub fn parse(config_obus: &[u8], frame_data: &[u8]) -> Result<Self> {
        let mut sequence = sequence_header_from_config(config_obus).ok();

        let obus = Obu::parse_stream(frame_data)?;
        let mut frame = None;

        for obu in &obus {
            // Each OBU records where its payload begins in frame_data, so the
            // tile-data locator stays absolute.
            let payload_offset = obu.payload_start;

            match obu.header.kind {
                ObuType::SequenceHeader => {
                    let mut reader = BitReader::new(obu.payload);
                    sequence = Some(SequenceHeader::parse(&mut reader)?);
                }
                ObuType::Frame => {
                    let seq = sequence.as_ref().ok_or_else(missing_sequence)?;
                    let mut reader = BitReader::new(obu.payload);
                    let header = FrameHeader::parse(
                        &mut reader,
                        seq,
                        obu.header.temporal_id,
                        obu.header.spatial_id,
                    )?;
                    reader.byte_alignment()?;
                    let consumed = reader.byte_position();
                    frame = Some((
                        header,
                        payload_offset + consumed,
                        obu.payload.len().saturating_sub(consumed),
                    ));
                }
                ObuType::FrameHeader | ObuType::RedundantFrameHeader => {
                    let seq = sequence.as_ref().ok_or_else(missing_sequence)?;
                    let mut reader = BitReader::new(obu.payload);
                    let header = FrameHeader::parse(
                        &mut reader,
                        seq,
                        obu.header.temporal_id,
                        obu.header.spatial_id,
                    )?;
                    // Tile data arrives in a following OBU_TILE_GROUP; record
                    // the header now and fill the locator when it is seen.
                    frame = Some((header, 0, 0));
                }
                ObuType::TileGroup => {
                    if let Some((header, off, len)) = frame.take() {
                        // A separate tile group supersedes a placeholder locator.
                        let (off, len) = if len == 0 {
                            (payload_offset, obu.payload.len())
                        } else {
                            (off, len)
                        };
                        frame = Some((header, off, len));
                    }
                }
                _ => {}
            }
        }

        let sequence = sequence.ok_or_else(missing_sequence)?;
        let (frame, tile_data_offset, tile_data_len) = frame.ok_or_else(|| {
            PixelsError::malformed("avif", "the coded frame contains no frame header")
        })?;

        Ok(Self {
            sequence,
            frame,
            tile_data_offset,
            tile_data_len,
        })
    }
}

fn missing_sequence() -> PixelsError {
    PixelsError::malformed(
        "avif",
        "a frame header was reached before any sequence header",
    )
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// Bit-level builder shared in shape with the seq/frame tests.
    struct Bldr {
        bits: Vec<u8>,
    }
    impl Bldr {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn put(&mut self, value: u32, n: u32) -> &mut Self {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
            self
        }
        fn flag(&mut self, b: bool) -> &mut Self {
            self.put(u32::from(b), 1)
        }
        fn pack(&self) -> Vec<u8> {
            let mut out = vec![0_u8; self.bits.len().div_ceil(8)];
            for (i, &bit) in self.bits.iter().enumerate() {
                if bit != 0 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    /// Wrap payload bytes in a low-overhead OBU of the given type.
    fn obu(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![(kind << 3) | 0b0000_0010];
        assert!(payload.len() < 128);
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        out
    }

    /// The reduced-still-picture sequence header bytes for 8-bit 4:2:0.
    fn seq_bytes(width: u32, height: u32) -> Vec<u8> {
        let mut b = Bldr::new();
        b.put(0, 3).flag(true).flag(true).put(1, 5);
        b.put(15, 4).put(15, 4).put(width - 1, 16).put(height - 1, 16);
        b.flag(false).flag(false).flag(false);
        b.flag(false).flag(false).flag(false);
        b.flag(false).flag(false).flag(false);
        b.flag(false).put(0, 2).flag(false);
        b.flag(false);
        b.pack()
    }

    /// The uncompressed header bytes for the reduced-still key frame used above,
    /// followed by whatever trailing tile bytes the caller wants.
    fn frame_bytes(tile: &[u8]) -> Vec<u8> {
        let mut b = Bldr::new();
        b.flag(false); // disable_cdf_update
        b.flag(false); // allow_screen_content_tools
        b.flag(false); // render_and_frame_size_different
        b.flag(true); // uniform_tile_spacing
        b.put(100, 8); // base_q_idx
        b.flag(false).flag(false).flag(false).flag(false); // deltas + qmatrix
        b.flag(false); // segmentation_enabled
        b.flag(false); // delta_q_present
        b.put(0, 6).put(0, 6).put(0, 3).flag(false); // loop filter
        b.flag(false); // tx_mode_select -> Largest
        b.flag(false); // reduced_tx_set
        let mut bytes = b.pack();
        // The header is byte-aligned by construction here (its bit count is a
        // multiple of 8 after packing rounds up); append the tile payload.
        bytes.extend_from_slice(tile);
        bytes
    }

    #[test]
    fn parses_config_and_frame_into_headers_and_tile_locator() {
        let config = obu(1, &seq_bytes(64, 64)); // OBU_SEQUENCE_HEADER
        let tile = [0xDE, 0xAD, 0xBE, 0xEF];
        let frame_payload = frame_bytes(&tile);
        let frame_stream = obu(6, &frame_payload); // OBU_FRAME

        let still = StillPicture::parse(&config, &frame_stream).unwrap();
        assert_eq!(still.sequence.max_frame_width, 64);
        assert_eq!(still.frame.frame_width, 64);
        assert_eq!(still.frame.quantization.base_q_idx, 100);
        assert_eq!(still.frame.tile_info.count(), 1);
        // The tile locator points at the trailing bytes of the OBU_FRAME.
        let located = &frame_stream[still.tile_data_offset..][..still.tile_data_len];
        assert_eq!(located, &tile);
    }

    #[test]
    fn a_separate_frame_header_and_tile_group_are_joined() {
        let config = obu(1, &seq_bytes(32, 32));
        let header_only = frame_bytes(&[]);
        let mut stream = obu(3, &header_only); // OBU_FRAME_HEADER
        let tile = [1, 2, 3];
        stream.extend(obu(4, &tile)); // OBU_TILE_GROUP

        let still = StillPicture::parse(&config, &stream).unwrap();
        assert_eq!(still.frame.frame_width, 32);
        let located = &stream[still.tile_data_offset..][..still.tile_data_len];
        assert_eq!(located, &tile);
    }

    #[test]
    fn an_in_band_sequence_header_overrides_the_config_one() {
        // Config says 64x64; the frame stream carries a 40x40 sequence header
        // ahead of the frame, which must win.
        let config = obu(1, &seq_bytes(64, 64));
        let mut stream = obu(1, &seq_bytes(40, 40));
        stream.extend(obu(6, &frame_bytes(&[0x00])));
        let still = StillPicture::parse(&config, &stream).unwrap();
        assert_eq!(still.sequence.max_frame_width, 40);
        assert_eq!(still.frame.frame_width, 40);
    }

    #[test]
    fn a_frame_with_no_sequence_header_anywhere_is_rejected() {
        let stream = obu(6, &frame_bytes(&[0x00]));
        let err = StillPicture::parse(&[], &stream).unwrap_err();
        assert!(err.to_string().contains("sequence header"), "{err}");
    }

    #[test]
    fn config_without_a_sequence_header_is_reported() {
        let config = obu(2, &[]); // a temporal delimiter, no seq header
        let err = sequence_header_from_config(&config).unwrap_err();
        assert!(err.to_string().contains("no sequence header"), "{err}");
    }
}
