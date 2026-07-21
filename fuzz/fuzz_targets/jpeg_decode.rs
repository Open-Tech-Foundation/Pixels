//! Decode arbitrary bytes as baseline JPEG. The property is that nothing panics.
//!
//! No assertion is made about the pixels: an arbitrary input has no correct
//! decoding, so the only thing to check is that the decoder returns a value.
//! Correctness lives in the crate's `tests/reference.rs`, against libjpeg-turbo.
//!
//! JPEG earns its own target rather than riding along with PNG's because its
//! attack surface is a different shape: Huffman tables that index a symbol
//! list, sampling factors that multiply into MCU geometry, quantization steps
//! that multiply into coefficients, and an entropy stream with no length.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_jpeg::JpegDecoder;
use otf_pixels_core::{Decoder, Limits};

fuzz_target!(|data: &[u8]| {
    // Bounded so a frame header claiming enormous dimensions is rejected
    // rather than turning every run into an out-of-memory report.
    let limits = Limits::default().with_max_pixels(4_000_000);
    let Ok(mut decoder) = JpegDecoder::new(data, limits) else {
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
