//! Decode arbitrary bytes as PNG. The property is that nothing panics.
//!
//! No assertion is made about the pixels: an arbitrary input has no correct
//! decoding, so the only thing to check is that the decoder returns a value.
//! Correctness lives in `tests/pngsuite.rs`, against libpng.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_png::PngDecoder;
use otf_pixels_core::{Decoder, Limits};

fuzz_target!(|data: &[u8]| {
    // Bounded so a header claiming enormous dimensions is rejected rather
    // than turning every run into an out-of-memory report.
    let limits = Limits::default().with_max_pixels(4_000_000);
    let Ok(mut decoder) = PngDecoder::new(data, limits) else {
        return;
    };
    let mut rows = 0;
    while rows < decoder.descriptor().height {
        // `tRNS` can widen the format once the stream is parsed, so the row
        // length is re-read rather than cached.
        let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
        if decoder.read_row(&mut row).is_err() {
            break;
        }
        rows += 1;
    }
});
