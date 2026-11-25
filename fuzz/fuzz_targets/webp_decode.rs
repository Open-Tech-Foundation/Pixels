//! Decode arbitrary bytes as WebP, every frame of an animation
//! included. The property is that nothing panics.
//!
//! No assertion is made about the pixels: an arbitrary input has no correct
//! decoding. Correctness lives in the crate's reference tests.
//!
//! Animation is most of the surface: frame rectangles placed by the stream,
//! disposal and blending between frames, and frame counts the header does
//! not bound. Frames are walked through `Decoder::advance_frame`, as the
//! pipeline walks them (ADR-0020), up to a cap that keeps a run quick.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_webp::WebPDecoder;
use otf_pixels_core::{Decoder, Limits};

/// Frames walked per input.
const MAX_FRAMES: u32 = 64;

fuzz_target!(|data: &[u8]| {
    let limits = Limits::default().with_max_pixels(4_000_000);
    let Ok(mut decoder) = WebPDecoder::new(data, limits) else {
        return;
    };
    let descriptor = decoder.descriptor();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..MAX_FRAMES {
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
