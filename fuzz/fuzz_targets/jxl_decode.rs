//! Decode arbitrary bytes as JPEG XL. The property is that nothing panics.
//!
//! No assertion is made about the pixels: an arbitrary input has no correct
//! decoding. Correctness lives in the crate's `tests/decode.rs`, against
//! libjxl.
//!
//! JPEG XL's attack surface is mostly sizes it states for itself: frames
//! cropped beyond the image, tables of contents pointing past the frame,
//! entropy codes whose histograms must sum exactly, MA trees that must end,
//! and transforms that rearrange channels by indices the stream chooses.
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
    let mut rows = 0;
    while rows < descriptor.height {
        if decoder.read_row(&mut row).is_err() {
            break;
        }
        rows += 1;
    }
});
