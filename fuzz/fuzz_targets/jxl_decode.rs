//! Decode arbitrary bytes as JPEG XL. The property is that nothing panics.
//!
//! No assertion is made about the pixels: an arbitrary input has no correct
//! decoding. Correctness lives in the crate's `tests/decode.rs`, against
//! libjxl.
//!
//! JPEG XL's attack surface is mostly sizes it states for itself: frames
//! cropped beyond the image, tables of contents pointing past the frame,
//! entropy codes whose histograms must sum exactly, MA trees that must end,
//! transforms that rearrange channels by indices the stream chooses, and
//! frames that blend from reference frames earlier ones saved.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_jxl::JxlDecoder;
use otf_pixels_core::{Decoder, Limits};

fuzz_target!(|data: &[u8]| {
    let limits = Limits::default().with_max_pixels(4_000_000);
    let Ok(mut decoder) = JxlDecoder::new(data, limits) else {
        return;
    };
    let descriptor = decoder.descriptor();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    // Every displayed frame of an animation, as the pipeline walks them
    // (ADR-0020), up to a cap that keeps a run quick.
    for _ in 0..16 {
        for _ in 0..descriptor.height {
            if decoder.read_row(&mut row).is_err() {
                return;
            }
        }
        match decoder.advance_frame() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => return,
        }
    }
});
