//! The codestream's headers (ISO/IEC 18181-1 §A, §C): the image header, a
//! frame header, and a frame's table of contents.
//!
//! Only the fields this encoder sets are written as values; every other
//! field takes its default through the bundles' `all_default` and
//! condition bits, exactly as a decoder reads them.

use crate::bits::{BitWriter, bits, val};

/// What the image header says about the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    /// Bits per sample, 8 or 16.
    pub bit_depth: u32,
    /// Grey rather than RGB.
    pub grey: bool,
    /// An alpha channel, as the one extra channel.
    pub alpha: bool,
}

/// `U32(1 + u(9), 1 + u(13), 1 + u(18), 1 + u(30))`, a dimension.
const DIMENSION: [crate::bits::Dist; 4] = [bits(1, 9), bits(1, 13), bits(1, 18), bits(1, 30)];

/// Write `BitDepth` for integer samples of `depth` bits.
fn bit_depth(w: &mut BitWriter, depth: u32) {
    w.bool(false); // integer samples
    w.u32(depth, [val(8), val(10), val(12), bits(1, 6)]);
}

/// Write the signature and the image header, padded to a byte, ready for
/// the first frame.
pub fn write_image_header(w: &mut BitWriter, info: &ImageInfo) {
    w.write(16, 0x0AFF);

    // SizeHeader, in its general form.
    w.bool(false); // not div8
    w.u32(info.height, DIMENSION);
    w.write(3, 0); // no ratio: the width follows
    w.u32(info.width, DIMENSION);

    // ImageMetadata.
    w.bool(false); // not all_default
    w.bool(false); // no extra fields: orientation 1, no preview, no animation
    bit_depth(w, info.bit_depth);
    // Every sample, and every residual, fits 16 bits until 14-bit samples.
    w.bool(info.bit_depth <= 12);
    if info.alpha {
        w.u32(1, [val(0), val(1), bits(2, 4), bits(1, 12)]);
        if info.bit_depth == 8 {
            w.bool(true); // the default: 8-bit unassociated alpha
        } else {
            w.bool(false);
            w.enumeration(0); // alpha
            bit_depth(w, info.bit_depth);
            w.u32(0, [val(0), val(3), val(4), bits(1, 3)]); // dim_shift
            w.u32(0, [val(0), bits(0, 4), bits(16, 5), bits(48, 10)]); // no name
            w.bool(false); // not premultiplied
        }
    } else {
        w.u32(0, [val(0), val(1), bits(2, 4), bits(1, 12)]);
    }
    w.bool(false); // not XYB: the samples are the image's own
    if info.grey {
        w.bool(false); // not all_default
        w.bool(false); // no ICC profile
        w.enumeration(1); // grey
        w.enumeration(1); // D65
        w.bool(false); // no gamma
        w.enumeration(13); // sRGB transfer
        w.enumeration(1); // relative rendering intent
    } else {
        w.bool(true); // sRGB
    }
    w.u64(0); // no extensions
    w.bool(true); // default transform data (opsin matrix, upsampling)
    w.zero_pad_to_byte();
}

/// Write a frame header for a lossless Modular frame, the last frame.
pub fn write_modular_frame_header(w: &mut BitWriter, info: &ImageInfo) {
    w.bool(false); // not all_default
    w.write(2, 0); // regular frame
    w.write(1, 1); // Modular
    w.u64(0); // no flags: no noise, patches or splines
    w.bool(false); // not YCbCr
    w.u32(1, [val(1), val(2), val(4), val(8)]); // no upsampling
    if info.alpha {
        w.u32(1, [val(1), val(2), val(4), val(8)]);
    }
    w.write(2, 1); // group_size_shift 1: 256x256 groups
    w.u32(1, [val(1), val(2), val(3), bits(4, 3)]); // one pass
    w.bool(false); // no crop
    w.u32(0, [val(0), val(1), val(2), bits(3, 2)]); // replace
    if info.alpha {
        w.u32(0, [val(0), val(1), val(2), bits(3, 2)]);
    }
    w.bool(true); // is_last
    w.u32(0, [val(0), bits(0, 4), bits(16, 5), bits(48, 10)]); // no name
    // Restoration filters off: they would change lossless pixels.
    w.bool(false); // not all_default
    w.bool(false); // no Gaborish
    w.write(2, 0); // no edge-preserving filter
    w.u64(0); // no restoration filter extensions
    w.u64(0); // no frame header extensions
}

/// Write the table of contents for sections of `sizes` bytes, in order.
pub fn write_toc(w: &mut BitWriter, sizes: &[usize]) {
    w.bool(false); // not permuted
    w.zero_pad_to_byte();
    for &size in sizes {
        w.u32(
            u32::try_from(size).unwrap_or(u32::MAX),
            [
                bits(0, 10),
                bits(1024, 14),
                bits(17_408, 22),
                bits(4_211_712, 30),
            ],
        );
    }
    w.zero_pad_to_byte();
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;

    #[test]
    fn the_image_header_starts_with_the_signature_and_is_byte_aligned() {
        let mut w = BitWriter::new();
        write_image_header(
            &mut w,
            &ImageInfo {
                width: 100,
                height: 50,
                bit_depth: 8,
                grey: false,
                alpha: false,
            },
        );
        assert_eq!(w.bit_len() % 8, 0);
        assert_eq!(w.finish()[..2], [0xFF, 0x0A]);
    }

    #[test]
    fn toc_entries_take_the_smallest_bucket() {
        let mut w = BitWriter::new();
        write_toc(&mut w, &[0, 1023]);
        // Permutation bit and padding, then two 12-bit entries, padded.
        assert_eq!(w.bit_len(), 8 + 24);
        let mut w = BitWriter::new();
        write_toc(&mut w, &[1024]);
        assert_eq!(w.bit_len(), 8 + 16);
    }
}
