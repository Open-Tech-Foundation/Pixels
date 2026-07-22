//! Decode arbitrary bytes as AVIF. The property is that nothing panics.
//!
//! No assertion is made about the pixels: an arbitrary input has no correct
//! decoding, so the only thing to check is that the decoder returns a value.
//! Correctness lives in the crate's `tests/reference.rs`, against libavif.
//!
//! AVIF earns its own target because its attack surface is a container before
//! it is ever a bitstream: box sizes that can point past their parent, item
//! extents that can address outside the file, an `iloc` table whose offsets are
//! attacker-chosen, and property associations indexing a store — every one a
//! place a length can lie. `AvifDecoder::new` walks all of it before a single
//! pixel exists, so most of this target's value is in that parse.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_avif::AvifDecoder;
use otf_pixels_core::{Decoder, Limits};

fuzz_target!(|data: &[u8]| {
    // Bounded so a container claiming enormous dimensions is rejected rather
    // than turning every run into an out-of-memory report.
    let limits = Limits::default().with_max_pixels(4_000_000);
    let Ok(mut decoder) = AvifDecoder::new(data, limits) else {
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
