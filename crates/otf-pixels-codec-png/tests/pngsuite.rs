//! The M3 exit criterion: decode the PngSuite correctly, reject its corrupt
//! files, and never panic on either.
//!
//! # What is checked, and against what
//!
//! Our decoding is compared to **libpng's**, not to itself. `REFERENCE`
//! records a hash of each file's RGBA decoding as produced by Pillow; a
//! self-consistent-but-wrong decoder therefore fails here, which a round-trip
//! test could never detect. Regenerate it with
//! `scripts/regenerate-pngsuite-reference.py`.
//!
//! On top of that, PngSuite contains equivalence classes — the same image at
//! different zlib levels, at different IDAT splits, interlaced and not. Those
//! must decode identically to each other, which localises a failure to the
//! axis that broke. The `f0*` filter files are *not* such a class: each
//! carries different pixels, so they are checked individually against libpng.
//!
//! # Coverage, and what is deliberately absent
//!
//! The vendored subset covers every axis the decoder implements:
//!
//! - colour types 0, 2, 3, 4, 6 at every legal bit depth (`basn*`)
//! - Adam7 interlacing of the same (`basi*`)
//! - all five filter types and mixed filtering (`f0*n*`, `f99n*`)
//! - odd sizes 1..9 and 32..40, interlaced and not (`s0*`, `s3*`, `s4*`)
//! - palettes, including sub-byte indices and `tRNS` alpha (`*3p*`, `tb*`)
//! - transparency for grey, RGB and palette (`tb*`, `tp*`)
//! - every zlib level (`z0*`) and split IDAT chunks (`oi*`)
//! - ancillary chunks that must be skipped (`cm*`, `cs*`, `ct*`)
//! - all 14 deliberately corrupt files (`x*`)
//!
//! Deliberately **excluded**, because v1 does not implement the feature and a
//! passing test would be meaningless: gamma correction (`g*`) and background
//! compositing (`bg*`). SPEC §Pixel formats is sRGB-assumed and ICC is v2, so
//! those chunks are ancillary data we correctly skip rather than honour.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_png::{PngDecoder, probe};
use otf_pixels_core::{Decoder, Limits, PixelFormat};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures/pngsuite", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/{name}.png", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// One reference decoding: dimensions and a hash of the RGBA raster.
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

/// Every corrupt file in the suite. All must be rejected.
const CORRUPT: [&str; 14] = [
    "xc1n0g08", // colour type 1, which does not exist
    "xc9n2c08", // colour type 9
    "xcrn0g04", // incorrect chunk CRC
    "xcsn0g01", // incorrect IDAT checksum
    "xd0n2c08", // bit depth 0
    "xd3n2c08", // bit depth 3
    "xd9n2c08", // bit depth 99
    "xdtn0g01", // missing IDAT
    "xhdn0g08", // incorrect IHDR CRC
    "xlfn0g04", // length field beyond the file
    "xs1n0g01", // signature byte 1 wrong
    "xs2n0g01", // signature byte 2 wrong
    "xs4n0g01", // signature truncated
    "xs7n0g01", // 7-byte signature
];

/// Decode a whole PNG to its output-format bytes.
fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(otf_pixels_core::ImageDescriptor, Vec<u8>)> {
    let mut decoder = PngDecoder::new(bytes, Limits::default())?;
    // The descriptor is not final until the stream is read, because tRNS
    // appears after IHDR; read one row first, then ask.
    let mut raster = Vec::new();
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    // Re-size the row buffer after the first read resolves the real format.
    let mut first = true;
    let mut height = decoder.descriptor().height;
    let mut y = 0;
    while y < height {
        if first {
            // Probe the true row length by decoding once with a correctly
            // sized buffer after the format settles.
            let mut probe_decoder = PngDecoder::new(bytes, Limits::default())?;
            let mut scratch = vec![0_u8; probe_decoder.descriptor().row_bytes()];
            let _ = probe_decoder.read_row(&mut scratch);
            let descriptor = probe_decoder.descriptor();
            row = vec![0_u8; descriptor.row_bytes()];
            height = descriptor.height;
            first = false;
        }
        decoder.read_row(&mut row)?;
        raster.extend_from_slice(&row);
        y += 1;
    }
    Ok((decoder.descriptor(), raster))
}

/// FNV-1a, 64-bit — matches the regeneration script.
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325_u64;
    for &byte in data {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Convert our output raster to RGBA8, matching what the reference produced.
fn to_rgba8(descriptor: &otf_pixels_core::ImageDescriptor, raster: &[u8]) -> Vec<u8> {
    let pixels = (descriptor.width as usize) * (descriptor.height as usize);
    let mut out = Vec::with_capacity(pixels * 4);
    let bpp = descriptor.pixel.bytes_per_pixel();
    /// Narrow a native-endian 16-bit sample to 8 bits by discarding the low
    /// byte, matching the canonical form the reference manifest records.
    ///
    /// This is deliberately *not* `round(v * 255 / 65535)`, which is the
    /// better reduction: Pillow narrows 16-bit colour during load and never
    /// exposes the full-precision samples, so the reference cannot use the
    /// better rule. Neither side's decoder narrows anything — this is only
    /// how the two are compared. See the regeneration script.
    fn narrow(bytes: &[u8], at: usize) -> u8 {
        let value = u16::from_ne_bytes([bytes[at], bytes[at + 1]]);
        (value >> 8) as u8
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

// ---------------------------------------------------------------------------
// Ground truth
// ---------------------------------------------------------------------------

#[test]
fn every_valid_file_decodes_to_the_reference_pixels() {
    let references = references();
    assert!(
        references.len() >= 80,
        "the reference manifest looks incomplete"
    );
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
                let actual = fnv1a64(&to_rgba8(&descriptor, &raster));
                if actual != reference.rgba_hash {
                    failures.push(format!(
                        "{}: pixels differ from libpng ({} {}x{})",
                        reference.name, descriptor.pixel, descriptor.width, descriptor.height
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
        "{} of {} files disagree with libpng:\n  {}",
        failures.len(),
        references.len(),
        failures.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// Equivalence classes: same image, different encoding
// ---------------------------------------------------------------------------

/// Assert every named file decodes to identical pixels.
fn assert_same_pixels(group: &str, names: &[&str]) {
    let mut expected: Option<Vec<u8>> = None;
    for name in names {
        let (descriptor, raster) = decode(&read_fixture(name))
            .unwrap_or_else(|e| panic!("{group}: `{name}` failed to decode: {e}"));
        let rgba = to_rgba8(&descriptor, &raster);
        match &expected {
            None => expected = Some(rgba),
            Some(first) => assert_eq!(
                &rgba, first,
                "{group}: `{name}` differs from `{}`",
                names[0]
            ),
        }
    }
}

#[test]
fn every_filter_type_decodes_to_the_reference_pixels() {
    // f00..f04 each exercise one filter type. They are *not* one image five
    // ways — verified against libpng, each carries different pixels — so the
    // check is per-file against ground truth, not against each other. The
    // filters' algebraic round-trip is covered as a unit test in `format`;
    // this is the check that our reading of §9.2 matches libpng's.
    let references = references();
    for name in ["f00n0g08", "f01n0g08", "f02n0g08", "f03n0g08", "f04n0g08"] {
        let reference = references
            .iter()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("`{name}` missing from the reference manifest"));
        let (descriptor, raster) =
            decode(&read_fixture(name)).unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert_eq!(
            fnv1a64(&to_rgba8(&descriptor, &raster)),
            reference.rgba_hash,
            "`{name}` disagrees with libpng"
        );
    }
}

#[test]
fn every_zlib_level_decodes_to_the_same_image() {
    // z00..z09 differ only in compression level, so this isolates inflate
    // from everything above it.
    assert_same_pixels(
        "zlib levels",
        &["z00n2c08", "z03n2c08", "z06n2c08", "z09n2c08"],
    );
}

#[test]
fn split_idat_chunks_decode_to_the_same_image() {
    // oi1/oi2/oi4 are the same image with the IDAT data split across
    // different numbers of chunks, which must be concatenated before inflate.
    assert_same_pixels("split IDAT", &["oi1n0g16", "oi2n0g16", "oi4n0g16"]);
}

#[test]
fn interlaced_and_non_interlaced_decode_identically() {
    // The strongest Adam7 check available: the same image, both ways.
    let pairs = [
        ("basn0g01", "basi0g01"),
        ("basn0g08", "basi0g08"),
        ("basn0g16", "basi0g16"),
        ("basn2c08", "basi2c08"),
        ("basn3p08", "basi3p08"),
        ("basn4a08", "basi4a08"),
        ("basn6a08", "basi6a08"),
        ("basn6a16", "basi6a16"),
    ];
    for (plain, interlaced) in pairs {
        assert_same_pixels("interlace", &[plain, interlaced]);
    }
}

#[test]
fn ancillary_chunks_do_not_change_the_pixels() {
    // cm*, ct* carry timestamps and text. They must be skipped, not honoured
    // and not rejected.
    //
    // `cten0g04` is deliberately absent: despite the `ct` prefix it is not the
    // same image as the rest of the group — it carries text in a different
    // encoding *and* different pixels. Verified against libpng before
    // excluding it, since "the reference disagrees" is normally our bug.
    assert_same_pixels(
        "ancillary",
        &["ct0n0g04", "ct1n0g04", "ctzn0g04", "cm0n0g04"],
    );
}

// ---------------------------------------------------------------------------
// Corrupt input
// ---------------------------------------------------------------------------

#[test]
fn every_corrupt_file_is_rejected() {
    for name in CORRUPT {
        let bytes = read_fixture(name);
        let result = decode(&bytes);
        assert!(
            result.is_err(),
            "`{name}` is deliberately corrupt but decoded successfully"
        );
        let error = result.err().unwrap();
        assert_eq!(
            error.code(),
            otf_pixels_core::ErrorCode::Malformed,
            "`{name}` should be malformed, got: {error}"
        );
    }
}

#[test]
fn every_truncation_of_every_fixture_is_an_error_never_a_panic() {
    // The failure model in one test: a truncated file is a value, at every
    // possible truncation point, for every file in the suite.
    for reference in references() {
        let full = read_fixture(&reference.name);
        // Every byte for small files; a stride for larger ones, to stay quick.
        let step = if full.len() > 512 { 7 } else { 1 };
        for len in (0..full.len()).step_by(step) {
            let _ = decode(&full[..len]);
        }
    }
}

#[test]
fn single_byte_corruption_never_panics() {
    // Every byte of a representative file, flipped three ways.
    for name in ["basn0g08", "basn6a08", "basn3p08", "basi2c08"] {
        let original = read_fixture(name);
        for index in 0..original.len() {
            for mask in [0x01_u8, 0x80, 0xFF] {
                let mut corrupted = original.clone();
                corrupted[index] ^= mask;
                let _ = decode(&corrupted);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Format sniffing and limits
// ---------------------------------------------------------------------------

#[test]
fn probe_detects_png_by_magic_bytes_only() {
    let bytes = read_fixture("basn0g08");
    assert!(probe(&bytes));
    assert!(probe(&bytes[..8]));
    assert!(!probe(&bytes[..7]), "a short prefix must not be claimed");
    assert!(!probe(&[]));
    assert!(!probe(b"GIF89a\0\0"));
    // A file named .png whose bytes are not PNG is not PNG.
    assert!(!probe(b"\x89PNGxxxx"));
    // Corrupt-signature fixtures must not be claimed either.
    for name in ["xs1n0g01", "xs2n0g01", "xs7n0g01"] {
        assert!(
            !probe(&read_fixture(name)),
            "`{name}` has a broken signature"
        );
    }
}

#[test]
fn max_pixels_is_enforced_before_decoding() {
    // SPEC §Safety: the check happens at header parse, so a large image is
    // refused without allocating for it.
    let bytes = read_fixture("basn0g08");
    let tight = Limits::default().with_max_pixels(100);
    let error = PngDecoder::new(bytes.as_slice(), tight).unwrap_err();
    assert_eq!(error.code(), otf_pixels_core::ErrorCode::LimitExceeded);
    // The same file is fine under the default limit.
    assert!(PngDecoder::new(bytes.as_slice(), Limits::default()).is_ok());
}

#[test]
fn construction_reads_only_the_header() {
    // SPEC §Guarantees 3: metadata costs the header, not the pixels.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct Counting {
        data: std::io::Cursor<Vec<u8>>,
        read: Arc<AtomicUsize>,
    }
    impl std::io::Read for Counting {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = std::io::Read::read(&mut self.data, buf)?;
            self.read.fetch_add(n, Ordering::Relaxed);
            Ok(n)
        }
    }

    let bytes = read_fixture("basn6a16");
    assert!(
        bytes.len() > 1000,
        "need a file big enough for the test to mean something"
    );
    let read = Arc::new(AtomicUsize::new(0));
    let source = Counting {
        data: std::io::Cursor::new(bytes.clone()),
        read: Arc::clone(&read),
    };
    let decoder = PngDecoder::new(source, Limits::default()).unwrap();
    assert_eq!(decoder.descriptor().width, 32);
    assert_eq!(
        read.load(Ordering::Relaxed),
        33,
        "construction should read exactly the signature and IHDR"
    );
}

#[test]
fn odd_sized_images_decode_correctly() {
    // s01..s09 are 1x1 through 9x9, where Adam7 passes go empty and sub-byte
    // rows have partial trailing bytes. These are the classic crash cases.
    for (name, size) in [
        ("s01n3p01", 1),
        ("s02n3p01", 2),
        ("s03n3p01", 3),
        ("s04n3p01", 4),
        ("s05n3p02", 5),
        ("s06n3p02", 6),
        ("s07n3p02", 7),
        ("s08n3p02", 8),
        ("s09n3p02", 9),
    ] {
        let (descriptor, raster) =
            decode(&read_fixture(name)).unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert_eq!(descriptor.width, size, "`{name}` width");
        assert_eq!(descriptor.height, size, "`{name}` height");
        assert_eq!(
            raster.len(),
            descriptor.byte_len().unwrap(),
            "`{name}` raster size"
        );
    }
    // The same sizes, interlaced.
    for (name, size) in [("s01i3p01", 1), ("s05i3p02", 5), ("s09i3p02", 9)] {
        let (descriptor, _) =
            decode(&read_fixture(name)).unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert_eq!(
            (descriptor.width, descriptor.height),
            (size, size),
            "`{name}`"
        );
    }
}

#[test]
fn transparency_produces_an_alpha_channel() {
    // tRNS on grey, RGB and palette images must all yield alpha.
    for name in ["tbbn3p08", "tbrn2c08", "tbwn0g16", "tp1n3p08", "tbwn3p08"] {
        let (descriptor, _) =
            decode(&read_fixture(name)).unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert!(
            descriptor.pixel.has_alpha(),
            "`{name}` has tRNS but decoded to {} without alpha",
            descriptor.pixel
        );
    }
    // `tp0n3p08` is "transparent, but no tRNS chunk" — the suite's control for
    // exactly this test. It must *not* gain alpha, and neither must a plain
    // RGB image.
    for name in ["tp0n3p08", "basn2c08"] {
        let (descriptor, _) =
            decode(&read_fixture(name)).unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert!(
            !descriptor.pixel.has_alpha(),
            "`{name}` has no tRNS but decoded to {} with alpha",
            descriptor.pixel
        );
    }
}

#[test]
fn a_non_interlaced_png_decodes_without_buffering_the_image() {
    // The point of the streaming path: the decoder produces a row without
    // having read the whole file. A buffered decoder drains its source before
    // producing anything, so this is the difference expressed as a test
    // rather than as a comment.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A source that reports how far it has been read.
    #[derive(Debug)]
    struct Metered {
        data: Vec<u8>,
        at: Arc<AtomicUsize>,
    }

    impl otf_pixels_core::Source for Metered {
        fn read(&mut self, buf: &mut [u8]) -> otf_pixels_core::Result<usize> {
            let from = self.at.load(Ordering::SeqCst);
            let take = (self.data.len() - from).min(buf.len());
            buf[..take].copy_from_slice(&self.data[from..from + take]);
            self.at.store(from + take, Ordering::SeqCst);
            Ok(take)
        }
    }

    // A tall image, so "one row" and "the whole file" are far apart.
    let bytes = build_tall_png(64, 4096);
    let at = Arc::new(AtomicUsize::new(0));
    let source = Metered {
        data: bytes.clone(),
        at: Arc::clone(&at),
    };

    let mut decoder = PngDecoder::new(source, Limits::default()).unwrap();
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    decoder.read_row(&mut row).unwrap();

    let consumed = at.load(Ordering::SeqCst);
    assert!(
        consumed < bytes.len(),
        "the decoder read all {} bytes before producing one row, so it is still buffering",
        bytes.len()
    );

    // Reading the rest must then consume the rest, so the check above is
    // measuring laziness rather than a decoder that simply stopped early.
    for _ in 1..decoder.descriptor().height {
        decoder.read_row(&mut row).unwrap();
    }
    assert_eq!(
        at.load(Ordering::SeqCst),
        bytes.len(),
        "the stream was not fully read"
    );
}

/// Build a tall non-interlaced greyscale PNG with our own encoder.
fn build_tall_png(width: u32, height: u32) -> Vec<u8> {
    use otf_pixels_codec_png::PngEncoder;
    use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

    let descriptor = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
    let mut encoder = PngEncoder::new();
    let mut out = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();
    // Noise, so the rows do not all compress into one tiny block.
    for y in 0..height {
        let row: Vec<u8> = (0..width)
            .map(|x| ((x * 31 + y * 17) % 251) as u8)
            .collect();
        encoder.write_row(&row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

#[test]
fn decoding_a_tall_png_row_by_row_matches_decoding_it_whole() {
    // Streaming must not change the pixels. Same image, same bytes, whether
    // the source hands them over in one piece or in 13-byte dribbles.
    let bytes = build_tall_png(37, 500);
    let (_, whole) = decode(&bytes).unwrap();

    struct Trickle<'a>(&'a [u8], usize);
    impl otf_pixels_core::Source for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> otf_pixels_core::Result<usize> {
            let take = (self.0.len() - self.1).min(buf.len()).min(13);
            buf[..take].copy_from_slice(&self.0[self.1..self.1 + take]);
            self.1 += take;
            Ok(take)
        }
    }
    impl std::fmt::Debug for Trickle<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Trickle")
        }
    }

    let mut decoder = PngDecoder::new(Trickle(&bytes, 0), Limits::default()).unwrap();
    let mut dribbled = Vec::new();
    let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
    for _ in 0..decoder.descriptor().height {
        decoder.read_row(&mut row).unwrap();
        dribbled.extend_from_slice(&row);
    }
    assert_eq!(dribbled, whole, "a trickled source decoded differently");
}
