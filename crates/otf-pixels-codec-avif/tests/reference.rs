//! AVIF decode checked against libaom/libavif, not against ourselves.
//!
//! The codec is owned end to end (ADR-0013), so the AV1 bitstream *is* on
//! trial. The reference rasters are produced by libavif's `avifdec`, and a
//! lossless fixture must match one to the byte. A fixture that exercised a tool
//! this decoder does not implement would report `Unsupported` and be skipped
//! rather than asserted; the whole lossless corpus currently decodes.
//!
//! Regenerate the fixtures and manifest with `scripts/regenerate-avif-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_avif::AvifDecoder;
use otf_pixels_core::{Decoder, Limits, PixelFormat};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str, extension: &str) -> Vec<u8> {
    let path = format!("{}/{name}.{extension}", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// One fixture as the manifest describes it:
/// `name width height channels tolerance bits`.
struct Reference {
    name: String,
    width: u32,
    height: u32,
    channels: usize,
    /// The largest per-channel difference from libavif's raster allowed, in
    /// steps of the output depth.
    tolerance: u16,
    /// Bits per output sample: 8, or 16 for a 10/12-bit file (the reference is
    /// then little-endian, ours native-endian).
    bits: u8,
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
                channels: f[3].parse().unwrap(),
                tolerance: f[4].parse().unwrap(),
                bits: f[5].parse().unwrap(),
            }
        })
        .collect()
}

fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(Vec<u8>, otf_pixels_core::ImageDescriptor)> {
    let mut decoder = AvifDecoder::new(bytes, Limits::default())?;
    let descriptor = decoder.descriptor();
    let mut pixels = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row)?;
        pixels.extend_from_slice(&row);
    }
    Ok((pixels, descriptor))
}

/// Every fixture decodes, and its raster matches libavif's within the manifest's
/// tolerance. For everything coded in 4:4:4 with the identity matrix the
/// tolerance is 0: the RGB raster *is* the decoded planes, so the whole AV1
/// decode must match to the byte. Flavours: `lossless` (4x4 WHT, no filters);
/// `nofilter` (genuinely lossy — DCT/ADST, larger transforms, chroma-from-luma —
/// but every in-loop filter off); `deblock` (lossy with the deblocking loop
/// filter §7.14 on, CDEF/restoration off); `cdef` (lossy with CDEF §7.15 on,
/// deblocking/restoration off, to isolate it); `restore` (loop restoration §7.17
/// on, deblocking/CDEF off); `restore_full` (all three in-loop filters on);
/// `superres` / `superres_full` (coded at a reduced width and upscaled, §7.16,
/// with the in-loop filters off / on); and `photo` — real YUV (4:2:0, 4:2:2,
/// BT.601/709/2020, full and studio range) converted to RGB, where the
/// tolerance is the gap to libavif's libyuv-based conversion (see the
/// regeneration script) and the decode underneath is pinned exactly by
/// `tests/planes.rs`. Files that must be refused live in `tests/unsupported.rs`,
/// so here a refusal is a failure.
#[test]
fn reference_fixtures_decode_exactly() {
    let mut compared = 0;
    let mut lossy_compared = 0;
    let mut deblock_compared = 0;
    let mut cdef_compared = 0;
    let mut restore_compared = 0;
    let mut superres_compared = 0;
    for reference in references() {
        let (ours, descriptor) = decode(&read_fixture(&reference.name, "avif"))
            .unwrap_or_else(|e| panic!("{}: {e}", reference.name));
        let theirs = read_fixture(&reference.name, "raw");

        assert_eq!(
            (descriptor.width, descriptor.height),
            (reference.width, reference.height),
            "{}: dimensions",
            reference.name
        );
        assert_eq!(
            descriptor.pixel,
            match (reference.channels, reference.bits) {
                (4, 16) => PixelFormat::Rgba16,
                (4, _) => PixelFormat::Rgba8,
                (_, 16) => PixelFormat::Rgb16,
                _ => PixelFormat::Rgb8,
            },
            "{}: pixel format",
            reference.name
        );
        assert_eq!(ours.len(), theirs.len(), "{}: raster size", reference.name);
        let worst = if reference.bits == 16 {
            ours.chunks_exact(2)
                .zip(theirs.chunks_exact(2))
                .map(|(a, b)| {
                    u16::from_ne_bytes([a[0], a[1]]).abs_diff(u16::from_le_bytes([b[0], b[1]]))
                })
                .max()
        } else {
            ours.iter()
                .zip(&theirs)
                .map(|(a, b)| u16::from(a.abs_diff(*b)))
                .max()
        }
        .unwrap_or(0);
        assert!(
            worst <= reference.tolerance,
            "{}: differs from libavif's by up to {worst}, tolerance {}",
            reference.name,
            reference.tolerance
        );
        compared += 1;
        if reference.name.contains("nofilter") {
            lossy_compared += 1;
        }
        if reference.name.contains("deblock") {
            deblock_compared += 1;
        }
        if reference.name.contains("cdef") {
            cdef_compared += 1;
        }
        if reference.name.contains("restore") {
            restore_compared += 1;
        }
        if reference.name.contains("superres") {
            superres_compared += 1;
        }
    }
    assert!(compared >= 2, "only {compared} fixtures decoded");
    assert!(
        lossy_compared >= 1,
        "no filters-off lossy fixture was compared — the lossy path regressed \
         to Unsupported or the manifest lost its nofilter entries"
    );
    assert!(
        deblock_compared >= 1,
        "no deblock fixture was compared — deblocking regressed to Unsupported \
         or the manifest lost its deblock entries"
    );
    assert!(
        cdef_compared >= 1,
        "no cdef fixture was compared — CDEF regressed to Unsupported or the \
         manifest lost its cdef entries"
    );
    assert!(
        restore_compared >= 1,
        "no restore fixture was compared — loop restoration regressed to \
         Unsupported or the manifest lost its restore entries"
    );
    assert!(
        superres_compared >= 1,
        "no superres fixture was compared — super-resolution regressed to \
         Unsupported or the manifest lost its superres entries"
    );
}
