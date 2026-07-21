//! GIF structure: the header, the logical screen, blocks and extensions.
//!
//! GIF89a (and the near-identical GIF87a) is a stream of *blocks* introduced by
//! a one-byte label. A decoder walks them, accumulating state from extensions
//! until it reaches an image, which it then renders onto a persistent canvas.
//!
//! # Sub-blocks
//!
//! Almost every variable-length payload in GIF is a chain of *sub-blocks*: a
//! length byte, that many data bytes, repeating until a zero length. The chain
//! is how a decoder skips an extension it does not understand without knowing
//! anything about its contents, which is what makes unknown extensions
//! skippable rather than fatal.

use otf_pixels_core::{PixelsError, Result, Source};

/// The two signatures GIF has ever used.
pub const SIGNATURE_87A: [u8; 6] = *b"GIF87a";
/// The 89a signature, which adds extensions and transparency.
pub const SIGNATURE_89A: [u8; 6] = *b"GIF89a";

/// Block labels (§Appendix A).
pub mod label {
    /// Introduces an extension block.
    pub const EXTENSION: u8 = 0x21;
    /// Introduces an image descriptor.
    pub const IMAGE: u8 = 0x2C;
    /// Ends the stream.
    pub const TRAILER: u8 = 0x3B;

    /// Graphic control extension: delay, disposal, transparency.
    pub const GRAPHIC_CONTROL: u8 = 0xF9;
    // The remaining extension labels are listed for the reader rather than
    // matched on: the decoder skips every extension it does not understand by
    // walking its sub-block chain, which is exactly what §Appendix A requires
    // and needs no knowledge of the label.
    //
    // 0x01 plain text, 0xFE comment, 0xFF application (the Netscape loop
    // count lives here).
}

/// The logical screen descriptor: the canvas every frame is drawn onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Screen {
    /// Canvas width in pixels.
    pub width: u16,
    /// Canvas height in pixels.
    pub height: u16,
    /// Number of entries in the global colour table, or zero if absent.
    pub global_table_size: usize,
    /// Index into the global table used as the background, if any.
    pub background: u8,
}

impl Screen {
    /// Parse the seven bytes following the signature.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the canvas has no pixels.
    pub fn parse(bytes: &[u8; 7]) -> Result<Self> {
        let width = u16::from_le_bytes([bytes[0], bytes[1]]);
        let height = u16::from_le_bytes([bytes[2], bytes[3]]);
        let packed = bytes[4];
        let background = bytes[5];
        // bytes[6] is the pixel aspect ratio, which v1 ignores: it describes
        // display geometry, not pixels, and honouring it would resample.

        if width == 0 || height == 0 {
            return Err(PixelsError::malformed(
                "gif",
                format!("logical screen is {width}x{height}"),
            ));
        }

        // Bit 7 is the global colour table flag; bits 0..2 encode its size as
        // 2^(n+1) entries.
        let global_table_size = if packed & 0x80 != 0 {
            1_usize << ((packed & 0x07) + 1)
        } else {
            0
        };

        Ok(Self {
            width,
            height,
            global_table_size,
            background,
        })
    }
}

/// How the canvas is treated after a frame is displayed (§23.c.iv).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Disposal {
    /// Unspecified: treated as [`Disposal::Keep`], which is what viewers do.
    #[default]
    None,
    /// Leave the frame in place; the next one draws over it.
    Keep,
    /// Restore the frame's rectangle to the background colour.
    Background,
    /// Restore the frame's rectangle to what was there before it was drawn.
    Previous,
}

impl Disposal {
    /// The disposal for a packed graphic-control field.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            1 => Self::Keep,
            2 => Self::Background,
            3 => Self::Previous,
            // 4..=7 are reserved. Treating them as "none" is what viewers do,
            // and rejecting a file over a reserved disposal value would fail
            // images that display fine everywhere else.
            _ => Self::None,
        }
    }
}

/// State from a graphic control extension, applying to the next image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct GraphicControl {
    /// How to dispose of the frame after displaying it.
    pub disposal: Disposal,
    /// Delay before the next frame, in hundredths of a second.
    pub delay_centiseconds: u16,
    /// Palette index treated as transparent, if any.
    pub transparent: Option<u8>,
}

impl GraphicControl {
    /// Parse a four-byte graphic control payload.
    #[must_use]
    pub fn parse(payload: &[u8]) -> Self {
        let packed = payload.first().copied().unwrap_or(0);
        let delay = u16::from_le_bytes([
            payload.get(1).copied().unwrap_or(0),
            payload.get(2).copied().unwrap_or(0),
        ]);
        let index = payload.get(3).copied().unwrap_or(0);
        Self {
            disposal: Disposal::from_bits((packed >> 2) & 0x07),
            delay_centiseconds: delay,
            transparent: if packed & 0x01 != 0 {
                Some(index)
            } else {
                None
            },
        }
    }
}

/// An image descriptor: where a frame sits on the canvas and how it is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImageDescriptor {
    /// Left edge, in canvas coordinates.
    pub left: u16,
    /// Top edge, in canvas coordinates.
    pub top: u16,
    /// Frame width.
    pub width: u16,
    /// Frame height.
    pub height: u16,
    /// Entries in the local colour table, or zero if absent.
    pub local_table_size: usize,
    /// Whether the frame's pixels are stored interlaced.
    pub interlaced: bool,
}

impl ImageDescriptor {
    /// Parse the nine bytes of an image descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the frame has no pixels.
    pub fn parse(bytes: &[u8; 9]) -> Result<Self> {
        let left = u16::from_le_bytes([bytes[0], bytes[1]]);
        let top = u16::from_le_bytes([bytes[2], bytes[3]]);
        let width = u16::from_le_bytes([bytes[4], bytes[5]]);
        let height = u16::from_le_bytes([bytes[6], bytes[7]]);
        let packed = bytes[8];

        if width == 0 || height == 0 {
            return Err(PixelsError::malformed(
                "gif",
                format!("frame is {width}x{height}"),
            ));
        }

        let local_table_size = if packed & 0x80 != 0 {
            1_usize << ((packed & 0x07) + 1)
        } else {
            0
        };

        Ok(Self {
            left,
            top,
            width,
            height,
            local_table_size,
            interlaced: packed & 0x40 != 0,
        })
    }
}

/// GIF's four-pass interlace (§20.i), which is *not* Adam7.
///
/// Rows are emitted in four passes: every 8th from 0, every 8th from 4, every
/// 4th from 2, every 2nd from 1. Unlike PNG's Adam7 this interlaces rows only,
/// never columns, so a pass is whole scanlines and needs no scatter.
#[must_use]
pub fn interlaced_row(pass: usize, index: u32, height: u32) -> Option<u32> {
    let (start, step) = match pass {
        0 => (0, 8),
        1 => (4, 8),
        2 => (2, 4),
        3 => (1, 2),
        _ => return None,
    };
    let row = start + index * step;
    if row < height { Some(row) } else { None }
}

/// How many rows pass `pass` contains for an image of `height` rows.
#[must_use]
pub const fn interlaced_pass_rows(pass: usize, height: u32) -> u32 {
    let (start, step) = match pass {
        0 => (0_u32, 8_u32),
        1 => (4, 8),
        2 => (2, 4),
        3 => (1, 2),
        _ => return 0,
    };
    if height > start {
        (height - start).div_ceil(step)
    } else {
        0
    }
}

/// Read a chain of sub-blocks, concatenating their payloads.
///
/// `limit` bounds the total, because a chain is length-prefixed per block but
/// unbounded overall — an attacker can repeat 255-byte blocks indefinitely.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] if the stream ends mid-chain or the
/// total exceeds `limit`.
pub fn read_sub_blocks<S: Source>(source: &mut S, limit: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let mut length = [0_u8; 1];
        source.read_exact(&mut length)?;
        let length = length[0] as usize;
        if length == 0 {
            return Ok(out);
        }
        if out.len() + length > limit {
            return Err(PixelsError::malformed(
                "gif",
                format!("sub-block chain exceeds the {limit} byte limit"),
            ));
        }
        let at = out.len();
        out.resize(at + length, 0);
        let Some(slot) = out.get_mut(at..) else {
            return Err(PixelsError::malformed("gif", "sub-block buffer is short"));
        };
        source.read_exact(slot)?;
    }
}

/// Skip a chain of sub-blocks without retaining it.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] if the stream ends mid-chain.
pub fn skip_sub_blocks<S: Source>(source: &mut S) -> Result<()> {
    let mut scratch = [0_u8; 255];
    loop {
        let mut length = [0_u8; 1];
        source.read_exact(&mut length)?;
        let length = length[0] as usize;
        if length == 0 {
            return Ok(());
        }
        let Some(slot) = scratch.get_mut(..length) else {
            return Err(PixelsError::malformed("gif", "sub-block longer than 255"));
        };
        source.read_exact(slot)?;
    }
}

/// Write a payload as a chain of sub-blocks, terminated by a zero byte.
pub fn write_sub_blocks(out: &mut Vec<u8>, payload: &[u8]) {
    for chunk in payload.chunks(255) {
        out.push(chunk.len() as u8);
        out.extend_from_slice(chunk);
    }
    out.push(0);
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

    #[test]
    fn a_screen_descriptor_decodes_its_packed_fields() {
        // 4x3 canvas, global table present with 2^(3+1) = 16 entries.
        let bytes = [4, 0, 3, 0, 0x80 | 0x03, 7, 0];
        let screen = Screen::parse(&bytes).unwrap();
        assert_eq!((screen.width, screen.height), (4, 3));
        assert_eq!(screen.global_table_size, 16);
        assert_eq!(screen.background, 7);
    }

    #[test]
    fn a_screen_without_a_global_table_reports_none() {
        let bytes = [4, 0, 3, 0, 0x07, 0, 0];
        let screen = Screen::parse(&bytes).unwrap();
        assert_eq!(
            screen.global_table_size, 0,
            "the size bits must be ignored when the flag is clear"
        );
    }

    #[test]
    fn a_zero_sized_screen_is_malformed() {
        assert!(Screen::parse(&[0, 0, 3, 0, 0, 0, 0]).is_err());
        assert!(Screen::parse(&[4, 0, 0, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn an_image_descriptor_decodes_its_packed_fields() {
        // At (1,2), 5x6, local table of 2^(2+1) = 8, interlaced.
        let bytes = [1, 0, 2, 0, 5, 0, 6, 0, 0x80 | 0x40 | 0x02];
        let image = ImageDescriptor::parse(&bytes).unwrap();
        assert_eq!((image.left, image.top), (1, 2));
        assert_eq!((image.width, image.height), (5, 6));
        assert_eq!(image.local_table_size, 8);
        assert!(image.interlaced);
    }

    #[test]
    fn graphic_control_decodes_disposal_and_transparency() {
        // Disposal 2 (background), delay 0x0102, transparent index 5.
        let control = GraphicControl::parse(&[(2 << 2) | 0x01, 0x02, 0x01, 5]);
        assert_eq!(control.disposal, Disposal::Background);
        assert_eq!(control.delay_centiseconds, 0x0102);
        assert_eq!(control.transparent, Some(5));

        // Transparency flag clear means no transparent index, whatever the
        // index byte says.
        let opaque = GraphicControl::parse(&[0, 0, 0, 5]);
        assert_eq!(opaque.transparent, None);
    }

    #[test]
    fn reserved_disposal_values_are_treated_as_none() {
        // Rejecting a reserved value would fail files that display fine
        // everywhere else.
        for bits in 4..=7 {
            assert_eq!(Disposal::from_bits(bits), Disposal::None, "{bits}");
        }
    }

    #[test]
    fn the_four_interlace_passes_cover_every_row_exactly_once() {
        // The classic interlace bug is a row emitted twice or not at all, and
        // it is invisible on a height that happens to divide evenly.
        for height in 1..40_u32 {
            let mut seen = vec![0_u32; height as usize];
            let mut total = 0;
            for pass in 0..4 {
                let rows = interlaced_pass_rows(pass, height);
                for index in 0..rows {
                    let row = interlaced_row(pass, index, height)
                        .unwrap_or_else(|| panic!("pass {pass} row {index} of {height}"));
                    seen[row as usize] += 1;
                    total += 1;
                }
            }
            assert_eq!(total, height, "height {height} emitted {total} rows");
            assert!(
                seen.iter().all(|&n| n == 1),
                "height {height} covered rows {seen:?}"
            );
        }
    }

    #[test]
    fn interlaced_rows_past_a_pass_are_none() {
        assert_eq!(interlaced_row(0, 0, 1), Some(0));
        assert_eq!(interlaced_row(1, 0, 1), None, "pass 1 starts at row 4");
        assert_eq!(
            interlaced_row(9, 0, 100),
            None,
            "there are only four passes"
        );
    }

    #[test]
    fn sub_blocks_round_trip() {
        for length in [0_usize, 1, 254, 255, 256, 1000] {
            let payload: Vec<u8> = (0..length).map(|i| (i % 251) as u8).collect();
            let mut encoded = Vec::new();
            write_sub_blocks(&mut encoded, &payload);
            assert_eq!(
                encoded.last().copied(),
                Some(0),
                "a chain must end with a zero length"
            );
            let mut source = &encoded[..];
            let decoded = read_sub_blocks(&mut source, 1 << 20).unwrap();
            assert_eq!(decoded, payload, "length {length}");
        }
    }

    #[test]
    fn a_sub_block_chain_can_be_skipped_without_reading_it() {
        let mut encoded = Vec::new();
        write_sub_blocks(&mut encoded, &vec![7_u8; 600]);
        encoded.extend_from_slice(b"after");
        let mut source = &encoded[..];
        skip_sub_blocks(&mut source).unwrap();
        assert_eq!(source, b"after", "skipping stopped in the wrong place");
    }

    #[test]
    fn a_truncated_chain_is_an_error_not_a_hang() {
        let mut encoded = Vec::new();
        write_sub_blocks(&mut encoded, &vec![1_u8; 300]);
        for cut in 0..encoded.len() {
            let mut source = &encoded[..cut];
            let _ = read_sub_blocks(&mut source, 1 << 20);
        }
    }

    #[test]
    fn an_unbounded_chain_is_refused() {
        // A chain is length-prefixed per block but unbounded overall, so an
        // attacker can repeat 255-byte blocks indefinitely.
        let mut encoded = Vec::new();
        write_sub_blocks(&mut encoded, &vec![0_u8; 100_000]);
        let mut source = &encoded[..];
        let error = read_sub_blocks(&mut source, 4096).unwrap_err();
        assert!(error.to_string().contains("limit"), "{error}");
    }
}
