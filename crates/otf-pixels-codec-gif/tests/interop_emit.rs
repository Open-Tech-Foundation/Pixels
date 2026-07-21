//! Emits our own GIFs so libgif can verify them.
//!
//! Our decoder cannot validate our encoder: a shared misreading of the
//! specification would round-trip perfectly and still be wrong.
//! `tests/reference.rs` checks one direction by decoding what libgif
//! produced; this checks the other, by producing files for libgif to read.
//!
//! Only images whose colours fit the palette exactly are emitted, so the
//! comparison can be exact. Quantization error is real but is not what this
//! test is about — `reference.rs` and the encoder's own tests cover fidelity;
//! this covers whether the *container and LZW stream* are well-formed.
//!
//! Inert unless `OTF_EMIT_DIR` is set, so an ordinary `cargo test` is
//! unaffected. `scripts/check-gif-interop.sh` drives it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_gif::GifEncoder;
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

#[test]
fn emit_gifs_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }

    // Sizes chosen to catch the classic off-by-ones, including the ones that
    // only appear when a row is not a whole number of LZW codes.
    let sizes = [
        (1_u32, 1_u32),
        (17, 1),
        (1, 17),
        (23, 19),
        (64, 64),
        (255, 3),
    ];
    // Palette sizes spanning every code width GIF allows.
    let palettes = [2_usize, 3, 4, 7, 16, 64, 256];

    for &(width, height) in &sizes {
        for &colours in &palettes {
            let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();
            let raster = exact_palette_image(width, height, colours);

            let mut encoder = GifEncoder::new().with_colours(colours).unwrap();
            let mut gif: Vec<u8> = Vec::new();
            encoder.write_header(&descriptor, &mut gif).unwrap();
            for row in raster.chunks_exact(descriptor.row_bytes()) {
                encoder.write_row(row, &mut gif).unwrap();
            }
            encoder.finish(&mut gif).unwrap();

            let name = format!("{dir}/{width}x{height}_p{colours}");
            std::fs::write(format!("{name}.gif"), &gif).unwrap();
            std::fs::write(format!("{name}.rgb"), &raster).unwrap();
        }
    }
}

/// An image using exactly `colours` distinct values, so a palette of that size
/// represents it losslessly and the comparison can be exact.
fn exact_palette_image(width: u32, height: u32, colours: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height) as usize * 3);
    for i in 0..(width * height) as usize {
        let n = (i % colours) as u32;
        let spread = (n * 255 / colours.max(1) as u32) as u8;
        out.extend_from_slice(&[spread, 255 - spread, spread / 2]);
    }
    out
}
