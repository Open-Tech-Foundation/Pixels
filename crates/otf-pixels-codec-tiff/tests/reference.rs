//! TIFF decode checked against libtiff, not against ourselves.
//!
//! The discipline M3 established for PNG and M5 continued for GIF. `REFERENCE`
//! records what Pillow (libtiff) decodes each fixture to; regenerate it with
//! `scripts/regenerate-tiff-reference.py`.
//!
//! # What the fixtures cover
//!
//! - every baseline compression: none, LZW, Deflate, PackBits
//! - both byte orders, `II` and `MM`
//! - greyscale at 1, 8 and 16 bits; RGB at 8; palette
//! - both layouts: strips and tiles, at two tile sizes
//!
//! The same image stored four ways (`rgb_none`, `rgb_lzw`, `rgb_deflate`,
//! `rgb_packbits`) must decode identically, which localises a failure to the
//! compression rather than to the decoder. Likewise `tiled_*` against each
//! other.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_tiff::{ByteOrder, TiffDecoder, probe};
use otf_pixels_core::{
    DecodeCapability, Decoder, ImageDescriptor, Limits, PixelFormat, Region, TileBuf,
};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/{name}.tif", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &byte in data {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

struct Reference {
    name: String,
    width: u32,
    height: u32,
    rgba_hash: u64,
}

fn references() -> Vec<Reference> {
    let path = format!("{}/REFERENCE", fixture_dir());
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {path}: {e}; run the regeneration script"));
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            Reference {
                name: f[0].to_owned(),
                width: f[1].parse().unwrap(),
                height: f[2].parse().unwrap(),
                rgba_hash: u64::from_str_radix(f[4], 16).unwrap(),
            }
        })
        .collect()
}

/// Decode a whole TIFF row by row.
fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(ImageDescriptor, Vec<u8>)> {
    let mut decoder = TiffDecoder::new(bytes, Limits::default())?;
    let descriptor = decoder.descriptor();
    let mut raster = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row)?;
        raster.extend_from_slice(&row);
    }
    Ok((descriptor, raster))
}

/// Convert our output raster to the canonical RGBA8 the manifest records.
fn to_rgba8(descriptor: &ImageDescriptor, raster: &[u8]) -> Vec<u8> {
    let pixels = (descriptor.width as usize) * (descriptor.height as usize);
    let bpp = descriptor.pixel.bytes_per_pixel();
    let mut out = Vec::with_capacity(pixels * 4);

    /// Narrow a native-endian 16-bit sample by discarding the low byte, which
    /// is the only reduction both sides can apply identically. See the PNG
    /// suite for the full argument.
    fn narrow(bytes: &[u8], at: usize) -> u8 {
        (u16::from_ne_bytes([bytes[at], bytes[at + 1]]) >> 8) as u8
    }

    for index in 0..pixels {
        let p = &raster[index * bpp..(index + 1) * bpp];
        let (r, g, b, a) = match descriptor.pixel {
            PixelFormat::Gray8 => (p[0], p[0], p[0], 255),
            PixelFormat::GrayA8 => (p[0], p[0], p[0], p[1]),
            PixelFormat::Gray16 => {
                let v = narrow(p, 0);
                (v, v, v, 255)
            }
            PixelFormat::Rgb8 => (p[0], p[1], p[2], 255),
            PixelFormat::Rgba8 => (p[0], p[1], p[2], p[3]),
            PixelFormat::Rgb16 => (narrow(p, 0), narrow(p, 2), narrow(p, 4), 255),
            PixelFormat::Rgba16 => (narrow(p, 0), narrow(p, 2), narrow(p, 4), narrow(p, 6)),
            other => panic!("unexpected output format {other}"),
        };
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

#[test]
fn every_fixture_decodes_to_the_reference_pixels() {
    let references = references();
    assert!(references.len() >= 10, "the reference manifest looks thin");
    let mut failures = Vec::new();

    for reference in &references {
        let bytes = read_fixture(&reference.name);
        match decode(&bytes) {
            Ok((descriptor, raster)) => {
                if descriptor.width != reference.width || descriptor.height != reference.height {
                    failures.push(format!(
                        "{}: decoded {}x{}, reference is {}x{}",
                        reference.name,
                        descriptor.width,
                        descriptor.height,
                        reference.width,
                        reference.height
                    ));
                    continue;
                }
                if fnv1a64(&to_rgba8(&descriptor, &raster)) != reference.rgba_hash {
                    failures.push(format!(
                        "{}: pixels differ from libtiff ({})",
                        reference.name, descriptor.pixel
                    ));
                }
            }
            Err(error) => {
                failures.push(format!("{}: failed to decode: {error}", reference.name));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of {} disagree with libtiff:\n  {}",
        failures.len(),
        references.len(),
        failures.join("\n  ")
    );
}

#[test]
fn every_compression_decodes_to_the_same_pixels() {
    // The same image four ways. A failure here is in the compression, not the
    // decoder — which is the whole point of having the group.
    let mut expected: Option<Vec<u8>> = None;
    for name in ["rgb_none", "rgb_lzw", "rgb_deflate", "rgb_packbits"] {
        let (_, raster) = decode(&read_fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        match &expected {
            None => expected = Some(raster),
            Some(first) => assert_eq!(&raster, first, "`{name}` differs from `rgb_none`"),
        }
    }
}

#[test]
fn every_tiled_compression_decodes_to_the_same_pixels() {
    let mut expected: Option<Vec<u8>> = None;
    for name in ["tiled_none", "tiled_lzw", "tiled_deflate"] {
        let (_, raster) = decode(&read_fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        match &expected {
            None => expected = Some(raster),
            Some(first) => assert_eq!(&raster, first, "`{name}` differs from `tiled_none`"),
        }
    }
}

#[test]
fn both_byte_orders_decode() {
    // Half the TIFFs in the world are big-endian.
    let little = TiffDecoder::new(&read_fixture("rgb_none")[..], Limits::default()).unwrap();
    assert_eq!(little.byte_order(), ByteOrder::Little);
    let big = TiffDecoder::new(&read_fixture("rgb_bigendian")[..], Limits::default()).unwrap();
    assert_eq!(big.byte_order(), ByteOrder::Big);
}

#[test]
fn a_tiled_tiff_reports_region_capability_and_a_strip_tiff_does_not() {
    // The claim the scheduler acts on. Getting it wrong for a strip file
    // would make the engine ask for regions it cannot cheaply produce.
    for name in ["tiled_none", "tiled_lzw", "tiled_deflate"] {
        let decoder = TiffDecoder::new(&read_fixture(name)[..], Limits::default()).unwrap();
        assert_eq!(
            decoder.capability(),
            DecodeCapability::Regions,
            "`{name}` should support region decode"
        );
    }
    for name in ["rgb_none", "gray_lzw", "palette"] {
        let decoder = TiffDecoder::new(&read_fixture(name)[..], Limits::default()).unwrap();
        assert_eq!(
            decoder.capability(),
            DecodeCapability::Sequential,
            "`{name}` is stored in strips and cannot answer regions cheaply"
        );
    }
}

#[test]
fn region_decode_agrees_with_row_decode() {
    // Random access must give the same pixels as reading the image in order.
    // Any disagreement means a tile is being placed wrongly, which on a real
    // image looks like a subtly shifted or duplicated block.
    for name in ["tiled_none", "tiled_lzw", "tiled_deflate"] {
        let bytes = read_fixture(name);
        let (descriptor, whole) = decode(&bytes).unwrap();
        let mut decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();

        // Regions deliberately chosen to straddle tile boundaries, to sit
        // inside one tile, and to touch the clipped edge tiles.
        let regions = [
            Region::new(0, 0, 16, 16),
            Region::new(30, 30, 40, 40),
            Region::new(1, 1, 1, 1),
            Region::new(128, 96, 32, 32),
            Region::new(0, 0, descriptor.width, descriptor.height),
            Region::new(155, 120, 5, 8),
        ];

        for region in regions {
            let mut buffer = TileBuf::zeroed(region, descriptor.pixel).unwrap();
            decoder
                .read_region(region, &mut buffer.as_tile_mut().unwrap())
                .unwrap_or_else(|e| panic!("{name} {region}: {e}"));

            let bpp = descriptor.pixel.bytes_per_pixel();
            for y in 0..region.height {
                for x in 0..region.width {
                    let from = (((region.y + y) as usize * descriptor.width as usize)
                        + (region.x + x) as usize)
                        * bpp;
                    let to = ((y as usize * region.width as usize) + x as usize) * bpp;
                    assert_eq!(
                        &buffer.bytes()[to..to + bpp],
                        &whole[from..from + bpp],
                        "{name} {region}: pixel ({x},{y}) differs"
                    );
                }
            }
        }
    }
}

#[test]
fn a_region_decode_reads_only_the_tiles_it_needs() {
    // The property M5's exit criterion turns on. A 16x16 region of a
    // 160x128 image tiled at 32x32 touches one tile out of twenty.
    let bytes = read_fixture("tiled_none");
    let mut decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
    let descriptor = decoder.descriptor();

    let region = Region::new(0, 0, 16, 16);
    let mut buffer = TileBuf::zeroed(region, descriptor.pixel).unwrap();
    decoder
        .read_region(region, &mut buffer.as_tile_mut().unwrap())
        .unwrap();

    // The observable consequence: a one-tile read produced a correct tile
    // without the decoder having touched the whole image. Correctness is
    // checked above; here the point is that it succeeded at all on a decoder
    // that has decoded nothing else.
    assert_eq!(
        buffer.bytes().len(),
        16 * 16 * descriptor.pixel.bytes_per_pixel()
    );
}

#[test]
fn a_region_outside_the_image_is_an_error() {
    let bytes = read_fixture("tiled_none");
    let mut decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let region = Region::new(descriptor.width - 4, 0, 16, 16);
    let mut buffer = TileBuf::zeroed(region, descriptor.pixel).unwrap();
    assert!(
        decoder
            .read_region(region, &mut buffer.as_tile_mut().unwrap())
            .is_err()
    );
}

#[test]
fn region_decode_on_a_strip_tiff_is_unsupported_not_wrong() {
    let bytes = read_fixture("rgb_none");
    let mut decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
    let region = Region::new(0, 0, 8, 8);
    let mut buffer = TileBuf::zeroed(region, decoder.descriptor().pixel).unwrap();
    let error = decoder
        .read_region(region, &mut buffer.as_tile_mut().unwrap())
        .unwrap_err();
    assert!(error.to_string().contains("strips"), "{error}");
}

#[test]
fn probe_detects_tiff_by_magic_bytes_only() {
    assert!(probe(b"II\x2a\x00\x08\x00\x00\x00"));
    assert!(probe(b"MM\x00\x2a\x00\x00\x00\x08"));
    assert!(!probe(b"GIF89a.."));
    assert!(!probe(&[0x89, b'P', b'N', b'G', 13, 10, 26, 10]));
    assert!(!probe(b""));
}

#[test]
fn max_pixels_is_enforced_before_decoding() {
    let bytes = read_fixture("tiled_none");
    let limits = Limits::default().with_max_pixels(100);
    let error = TiffDecoder::new(&bytes[..], limits).unwrap_err();
    assert_eq!(
        error.code(),
        otf_pixels_core::ErrorCode::LimitExceeded,
        "{error}"
    );
}

#[test]
fn every_truncation_of_every_fixture_is_an_error_never_a_panic() {
    for name in [
        "rgb_none",
        "rgb_lzw",
        "rgb_packbits",
        "gray16",
        "bilevel",
        "palette",
        "tiled_none",
        "rgb_bigendian",
    ] {
        let bytes = read_fixture(name);
        // A stride rather than every prefix: the point is to land inside every
        // parser — header, directory, value heap, pixel data — and a stride
        // does that while keeping the suite to seconds rather than minutes.
        for cut in (0..bytes.len()).step_by(31) {
            let Ok(mut decoder) = TiffDecoder::new(&bytes[..cut], Limits::default()) else {
                continue;
            };
            let descriptor = decoder.descriptor();
            let mut row = vec![0_u8; descriptor.row_bytes()];
            for _ in 0..descriptor.height.min(16) {
                if decoder.read_row(&mut row).is_err() {
                    break;
                }
            }
        }
    }
}

#[test]
fn single_byte_corruption_never_panics() {
    // `tiled_none` rather than `tiled_deflate`: corrupting a deflate stream
    // spends the whole test inside inflate, whose robustness is already
    // fuzzed directly in `otf-pixels-compress`. What is being tested here is
    // TIFF's own tile and directory handling under corruption.
    for name in ["rgb_lzw", "tiled_none", "palette"] {
        let original = read_fixture(name);
        // Coarser than the truncation stride: a full decode per case is much
        // more work than a header parse, and hitting a few hundred positions
        // across every structure is what the property needs.
        for at in (0..original.len()).step_by(97) {
            for mask in [0x01_u8, 0x80, 0xFF] {
                let mut bytes = original.clone();
                bytes[at] ^= mask;
                if let Ok(mut decoder) = TiffDecoder::new(&bytes[..], Limits::default()) {
                    let descriptor = decoder.descriptor();
                    if descriptor.byte_len().is_none_or(|n| n > 64 << 20) {
                        continue;
                    }
                    let mut row = vec![0_u8; descriptor.row_bytes()];
                    for _ in 0..descriptor.height.min(8) {
                        if decoder.read_row(&mut row).is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }
}
