//! Decoded sample planes checked against libaom, before colour conversion.
//!
//! `tests/reference.rs` compares final RGB rasters with libavif's, which works
//! only where YUV-to-RGB is the identity. Subsampled chroma never is (4:2:0 and
//! 4:2:2 cannot use the identity matrix), so the AV1 reconstruction of those
//! formats is verified here instead: each fixture in `tests/fixtures/planes` is
//! an avifenc encode, and its `.yuv` is what `aomdec --rawvideo` makes of the
//! same coded frame — Y, then U, then V, display-cropped, one byte per sample
//! at 8 bits and two little-endian bytes at 10 or 12. Our planes must
//! equal it sample for sample, which holds independently of how YUV later
//! becomes RGB.
//!
//! Regenerate with `scripts/regenerate-avif-reference.py`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_avif::{Meta, Reader, StillPicture, decode_still};

/// Every fixture, with the chroma subsampling it is expected to exercise, so a
/// regenerated fixture that silently changed format fails rather than passing
/// on easier ground.
const CASES: &[(&str, (u8, u8))] = &[
    ("gradient_420_nofilter", (1, 1)),
    ("gradient_odd_420_nofilter", (1, 1)),
    ("textured_420", (1, 1)),
    ("textured_odd_420", (1, 1)),
    ("mixed_420", (1, 1)),
    ("textured_422", (1, 0)),
    ("blocks_420_palette", (1, 1)),
    ("blocks_422_palette", (1, 0)),
    ("blocks_444_palette", (0, 0)),
    ("soft_420_sb128", (1, 1)),
    ("textured_odd_420_10bit", (1, 1)),
    ("textured_444_12bit", (0, 0)),
    ("blocks_420_10bit_palette", (1, 1)),
    // Monochrome signals 4:2:0 subsampling but codes no chroma planes.
    ("textured_odd_400", (1, 1)),
    ("blocks_400_palette", (1, 1)),
    // Per-superblock quantizer and loop-filter deltas.
    ("mixed_420_deltaq", (1, 1)),
    ("mixed_420_deltalf", (1, 1)),
    ("textured_444_10bit_deltalf", (0, 0)),
    ("textured_400_deltalf", (1, 1)),
];

/// Fixtures that exist to exercise a frame-header tool, with the
/// `(delta_q_present, delta_lf_present)` each must actually carry: an encoder
/// that stopped emitting the tool would otherwise leave the case passing on
/// easier ground.
const DELTA_TOOLS: &[(&str, (bool, bool))] = &[
    ("mixed_420_deltaq", (true, false)),
    ("mixed_420_deltalf", (true, true)),
    ("textured_444_10bit_deltalf", (true, true)),
    ("textured_400_deltalf", (true, true)),
];

fn fixture_dir() -> String {
    format!("{}/tests/fixtures/planes", env!("CARGO_MANIFEST_DIR"))
}

/// The primary item's coded AV1 frame and its `av1C` configuration OBUs.
fn primary_frame(file: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut reader = Reader::new(file);
    reader.next_box();
    let meta = Meta::parse(reader.find(b"meta").unwrap().expect("a meta box")).unwrap();
    let primary = meta.primary_item().expect("a primary item");
    let config = meta
        .properties
        .av1_config(primary.id)
        .expect("an av1C property");
    (
        meta.item_data(file, primary).unwrap().into_owned(),
        config.config_obus.clone(),
    )
}

#[test]
fn decoded_planes_match_libaom() {
    for &(name, (expect_sub_x, expect_sub_y)) in CASES {
        let file = std::fs::read(format!("{}/{name}.avif", fixture_dir())).unwrap();
        let theirs = std::fs::read(format!("{}/{name}.yuv", fixture_dir())).unwrap();
        let (frame_data, config_obus) = primary_frame(&file);
        let still = StillPicture::parse(&config_obus, &frame_data).unwrap();
        let color = &still.sequence.color;
        assert_eq!(
            (color.subsampling_x, color.subsampling_y),
            (expect_sub_x, expect_sub_y),
            "{name}: chroma subsampling"
        );
        let tile =
            &frame_data[still.tile_data_offset..still.tile_data_offset + still.tile_data_len];
        let decoded = decode_still(&still.sequence, &still.frame, tile)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        let width = still.frame.upscaled_width as usize;
        let height = still.frame.frame_height as usize;
        let (sub_x, sub_y) = (usize::from(expect_sub_x), usize::from(expect_sub_y));
        let wide = color.bit_depth > 8;
        let mut offset = 0;
        for (index, plane) in decoded.planes.iter().enumerate() {
            let (w, h) = if index == 0 {
                (width, height)
            } else {
                ((width + sub_x) >> sub_x, (height + sub_y) >> sub_y)
            };
            for y in 0..h {
                for x in 0..w {
                    let i = offset + y * w + x;
                    let expected = if wide {
                        u16::from_le_bytes([theirs[2 * i], theirs[2 * i + 1]])
                    } else {
                        u16::from(theirs[i])
                    };
                    let ours = plane.get(x, y).unwrap();
                    assert_eq!(
                        ours, expected,
                        "{name}: plane {index} differs from libaom at ({x}, {y})"
                    );
                }
            }
            offset += w * h;
        }
        let bytes = if wide { 2 } else { 1 };
        assert_eq!(
            offset * bytes,
            theirs.len(),
            "{name}: reference plane sizes"
        );
    }
}

#[test]
fn delta_fixtures_carry_the_tools_they_test() {
    for &(name, expected) in DELTA_TOOLS {
        let file = std::fs::read(format!("{}/{name}.avif", fixture_dir())).unwrap();
        let (frame_data, config_obus) = primary_frame(&file);
        let still = StillPicture::parse(&config_obus, &frame_data).unwrap();
        assert_eq!(
            (still.frame.delta_q_present, still.frame.delta_lf_present),
            expected,
            "{name}"
        );
    }
}

#[test]
fn every_plane_fixture_is_covered() {
    let mut files: Vec<String> = std::fs::read_dir(fixture_dir())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter_map(|name| name.strip_suffix(".avif").map(str::to_owned))
        .collect();
    files.sort();
    let mut cases: Vec<String> = CASES.iter().map(|(name, _)| (*name).to_owned()).collect();
    cases.sort();
    assert_eq!(files, cases);
}
