//! Emits our own TIFFs so libtiff can verify them.
//!
//! Our decoder cannot validate our encoder: a shared misreading of the
//! specification would round-trip perfectly and still be wrong.
//! `tests/reference.rs` checks one direction by decoding what libtiff
//! produced; this checks the other.
//!
//! Inert unless `OTF_EMIT_DIR` is set. `scripts/check-tiff-interop.sh` drives it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_tiff::{TiffEncoder, TiffLayout};
use otf_pixels_compress::Level;
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

#[test]
fn emit_tiffs_for_external_verification() {
    let Ok(dir) = std::env::var("OTF_EMIT_DIR") else {
        return;
    };
    if dir.is_empty() {
        return;
    }

    // Sizes that divide the chunking evenly and sizes that do not, because
    // the last strip and the padded edge tile are where arithmetic breaks.
    let sizes = [
        (1_u32, 1_u32),
        (17, 1),
        (1, 17),
        (37, 29),
        (70, 50),
        (128, 96),
    ];
    let formats = [
        PixelFormat::Gray8,
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Gray16,
        PixelFormat::Rgb16,
    ];
    let layouts = [
        ("strip8", TiffLayout::Strips { rows: 8 }),
        ("strip1", TiffLayout::Strips { rows: 1 }),
        (
            "tile16",
            TiffLayout::Tiles {
                width: 16,
                height: 16,
            },
        ),
        (
            "tile32",
            TiffLayout::Tiles {
                width: 32,
                height: 32,
            },
        ),
    ];

    for &(width, height) in &sizes {
        for format in formats {
            for &(layout_name, layout) in &layouts {
                for (compression, level) in [("none", None), ("deflate", Some(Level::DEFAULT))] {
                    let descriptor = ImageDescriptor::new(width, height, format).unwrap();
                    let len = descriptor.byte_len().unwrap();
                    let raster: Vec<u8> = (0..len).map(|i| ((i * 37) % 251) as u8).collect();

                    let mut encoder = TiffEncoder::new().with_layout(layout).unwrap();
                    if let Some(level) = level {
                        encoder = encoder.with_deflate(level);
                    }
                    let mut tiff: Vec<u8> = Vec::new();
                    encoder.write_header(&descriptor, &mut tiff).unwrap();
                    for row in raster.chunks_exact(descriptor.row_bytes()) {
                        encoder.write_row(row, &mut tiff).unwrap();
                    }
                    encoder.finish(&mut tiff).unwrap();

                    let name =
                        format!("{dir}/{format}_{width}x{height}_{layout_name}_{compression}");
                    std::fs::write(format!("{name}.tif"), &tiff).unwrap();
                    std::fs::write(format!("{name}.raw"), &raster).unwrap();
                    std::fs::write(
                        format!("{name}.meta"),
                        format!("{width} {height} {format}\n"),
                    )
                    .unwrap();
                }
            }
        }
    }
}
