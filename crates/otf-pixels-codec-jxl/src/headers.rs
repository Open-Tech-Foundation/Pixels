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
    /// Colour coded in XYB, as lossy VarDCT frames are: the decoder turns
    /// it back into the colour encoding the header names (sRGB).
    pub xyb: bool,
    /// For an animation, its loop count (0 forever); its ticks are
    /// milliseconds.
    pub animation: Option<u32>,
}

/// How one frame sits in the image (§C.2): the defaults are a still's one
/// frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSpec {
    /// The frame's rectangle on the canvas, `(x0, y0, width, height)`, when
    /// it covers less than all of it.
    pub crop: Option<(i32, i32, u32, u32)>,
    /// The reference slot the frame replaces pixels over, when cropped.
    pub source: u32,
    /// How long it shows, in ticks, for an animation's frame.
    pub duration: u32,
    pub is_last: bool,
    /// The slot the blended canvas is saved to, for a frame before the last.
    pub save_as_reference: u32,
}

impl FrameSpec {
    /// A still's frame: the whole canvas, the last.
    pub const STILL: Self = Self {
        crop: None,
        source: 0,
        duration: 0,
        is_last: true,
        save_as_reference: 0,
    };
}

/// A signed value as `UnpackSigned` reads it back.
const fn pack_signed(v: i32) -> u32 {
    if v >= 0 {
        (v as u32) << 1
    } else {
        ((-(v + 1)) as u32) << 1 | 1
    }
}

/// Write a frame header's crop, blending, timing and reference fields, from
/// `have_crop` to `save_before_ct`.
fn write_placement(w: &mut BitWriter, info: &ImageInfo, frame: &FrameSpec) {
    let position = [bits(0, 8), bits(256, 11), bits(2304, 14), bits(18688, 30)];
    w.bool(frame.crop.is_some());
    if let Some((x0, y0, width, height)) = frame.crop {
        w.u32(pack_signed(x0), position);
        w.u32(pack_signed(y0), position);
        w.u32(width, position);
        w.u32(height, position);
    }
    // Replace, for colour and each extra channel; a frame that does not
    // cover the canvas replaces pixels over a reference frame.
    for _ in 0..1 + usize::from(info.alpha) {
        w.u32(0, [val(0), val(1), val(2), bits(3, 2)]);
        if frame.crop.is_some() {
            w.write(2, u64::from(frame.source));
        }
    }
    if info.animation.is_some() {
        w.u32(frame.duration, [val(0), val(1), bits(0, 8), bits(0, 32)]);
    }
    w.bool(frame.is_last);
    if !frame.is_last {
        w.write(2, u64::from(frame.save_as_reference));
        // Saved after the colour transform, as displayed.
        if frame.crop.is_none() && (frame.duration == 0 || frame.save_as_reference != 0) {
            w.bool(false);
        }
    }
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
    if let Some(loops) = info.animation {
        w.bool(true); // extra fields
        w.write(3, 0); // orientation 1
        w.bool(false); // no intrinsic size
        w.bool(false); // no preview
        w.bool(true); // an animation
        w.u32(1000, [val(100), val(1000), bits(1, 10), bits(1, 30)]); // ticks per second
        w.u32(1, [val(1), val(1001), bits(1, 8), bits(1, 10)]); // per one
        w.u32(loops, [val(0), bits(0, 3), bits(0, 16), bits(0, 32)]);
        w.bool(false); // no timecodes
    } else {
        w.bool(false); // no extra fields: orientation 1, no preview, no animation
    }
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
    w.bool(info.xyb);
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
    if info.animation.is_some() {
        w.bool(true); // default tone mapping, read with extra fields
    }
    w.u64(0); // no extensions
    w.bool(true); // default transform data (opsin matrix, upsampling)
    w.zero_pad_to_byte();
}

/// Write a frame header for a lossless Modular frame placed by `frame`.
pub fn write_modular_frame_header(w: &mut BitWriter, info: &ImageInfo, frame: &FrameSpec) {
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
    write_placement(w, info, frame);
    w.u32(0, [val(0), bits(0, 4), bits(16, 5), bits(48, 10)]); // no name
    // Restoration filters off: they would change lossless pixels.
    w.bool(false); // not all_default
    w.bool(false); // no Gaborish
    w.write(2, 0); // no edge-preserving filter
    w.u64(0); // no restoration filter extensions
    w.u64(0); // no frame header extensions
}

/// Write a frame header for a lossy VarDCT frame, the last frame. Adaptive
/// LF smoothing, Gaborish and the edge-preserving filter are off.
pub fn write_vardct_frame_header(w: &mut BitWriter, info: &ImageInfo, frame: &FrameSpec) {
    w.bool(false); // not all_default
    w.write(2, 0); // regular frame
    w.write(1, 0); // VarDCT
    w.u64(0x80); // skip adaptive LF smoothing
    w.u32(1, [val(1), val(2), val(4), val(8)]); // no upsampling
    if info.alpha {
        w.u32(1, [val(1), val(2), val(4), val(8)]);
    }
    w.write(3, 3); // x_qm_scale, the default
    w.write(3, 2); // b_qm_scale, the default
    w.u32(1, [val(1), val(2), val(3), bits(4, 3)]); // one pass
    write_placement(w, info, frame);
    w.u32(0, [val(0), bits(0, 4), bits(16, 5), bits(48, 10)]); // no name
    w.bool(false); // restoration filter: not all_default
    w.bool(false); // no Gaborish
    w.write(2, 0); // no edge-preserving filter
    w.u64(0);
    w.u64(0);
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
                animation: None,
                width: 100,
                height: 50,
                bit_depth: 8,
                grey: false,
                alpha: false,
                xyb: false,
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
