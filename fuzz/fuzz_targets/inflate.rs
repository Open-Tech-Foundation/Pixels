//! Inflate arbitrary bytes. The property is that nothing panics and that the
//! output bound is honoured.
//!
//! This is the component where a bad back-reference would be a memory-safety
//! bug in a language that allowed one; `unsafe_code = "forbid"` makes that
//! unrepresentable, and this checks the remaining failure mode — a panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_png::{inflate_to, zlib_decompress};

const LIMIT: usize = 1 << 22;

fuzz_target!(|data: &[u8]| {
    if let Ok(out) = inflate_to(data, LIMIT) {
        assert!(out.len() <= LIMIT, "inflate exceeded its own bound");
    }
    if let Ok(out) = zlib_decompress(data, LIMIT) {
        assert!(out.len() <= LIMIT, "zlib_decompress exceeded its own bound");
    }
});
