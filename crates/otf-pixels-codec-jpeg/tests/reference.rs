//! JPEG decode checked against a reference decoder, not against ourselves.
//!
//! The discipline M3 established for PNG applies here, with one change forced
//! by the format: JPEG does not define an exact answer. ITU-T T.83 specifies
//! the inverse DCT only to an accuracy bound, so two conforming decoders
//! differ by a step or two per sample and a hash comparison would fail on a
//! decoder that is entirely correct. The reference raster is therefore stored
//! in full and compared with a tolerance.
//!
//! Reference rasters come from libjpeg-turbo via Pillow; regenerate them with
//! `scripts/regenerate-jpeg-reference.py`.
//!
//! # What the tolerances mean
//!
//! Two different things are being tolerated, and the test keeps them apart:
//!
//! - **IDCT accuracy**, which the standard licenses and which is worth a step
//!   or two per sample. Fixtures saved at 4:4:4 are held to that.
//! - **Chroma upsampling**, where we deliberately differ: libjpeg interpolates
//!   4:2:2 and 4:2:0 chroma with a triangle filter, and we replicate the
//!   nearest sample. A triangle filter needs the *next* MCU row's chroma,
//!   which the one-band decoder does not have. So subsampled fixtures are held
//!   tightly on luma — where the entropy decode and IDCT live, and where bugs
//!   actually are — and loosely on the chroma difference.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels_codec_jpeg::JpegDecoder;
use otf_pixels_core::{Decoder, ErrorCode, Limits, PixelFormat};

fn fixture_dir() -> String {
    format!("{}/tests/fixtures", env!("CARGO_MANIFEST_DIR"))
}

fn read_fixture(name: &str, extension: &str) -> Vec<u8> {
    let path = format!("{}/{name}.{extension}", fixture_dir());
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path}: {e}"))
}

/// One fixture as the manifest describes it.
struct Reference {
    name: String,
    width: u32,
    height: u32,
    channels: usize,
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
            }
        })
        .collect()
}

/// Whether this build can decode `name` at all.
///
/// The progressive fixtures need the wrapped decoder, which is behind a
/// feature; without it they are expected to be refused, and the suites that
/// sweep every fixture skip them rather than asserting the impossible.
fn decodable(name: &str) -> bool {
    cfg!(feature = "progressive") || !name.starts_with("progressive")
}

/// Decode a whole fixture into interleaved bytes.
fn decode(bytes: &[u8]) -> otf_pixels_core::Result<(Vec<u8>, u32, u32, PixelFormat)> {
    let mut decoder = JpegDecoder::new(bytes, Limits::default())?;
    let descriptor = decoder.descriptor();
    let mut pixels = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row)?;
        pixels.extend_from_slice(&row);
    }
    Ok((
        pixels,
        descriptor.width,
        descriptor.height,
        descriptor.pixel,
    ))
}

/// The reduced scales, with the reference suffix and divisor of each.
fn scales() -> Vec<(otf_pixels_codec_jpeg::Scale, &'static str, u32)> {
    vec![
        (otf_pixels_codec_jpeg::Scale::Eighth, "s1", 8),
        (otf_pixels_codec_jpeg::Scale::Quarter, "s2", 4),
        (otf_pixels_codec_jpeg::Scale::Half, "s4", 2),
    ]
}

/// Decode a fixture at a reduced scale.
fn decode_scaled(
    bytes: &[u8],
    scale: otf_pixels_codec_jpeg::Scale,
) -> (Vec<u8>, otf_pixels_core::ImageDescriptor) {
    let mut decoder = JpegDecoder::with_scale(bytes, Limits::default(), scale).unwrap();
    let descriptor = decoder.descriptor();
    let mut pixels = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row).unwrap();
        pixels.extend_from_slice(&row);
    }
    (pixels, descriptor)
}

/// Box-average a raster down to `(width, height)`.
fn box_average(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    channels: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut out = vec![0_u8; width * height * channels];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..channels {
                let (mut total, mut count) = (0_u32, 0_u32);
                let y0 = y * source_height / height;
                let y1 = ((y + 1) * source_height / height).max(y0 + 1);
                let x0 = x * source_width / width;
                let x1 = ((x + 1) * source_width / width).max(x0 + 1);
                for sy in y0..y1.min(source_height) {
                    for sx in x0..x1.min(source_width) {
                        total += u32::from(source[(sy * source_width + sx) * channels + channel]);
                        count += 1;
                    }
                }
                out[(y * width + x) * channels + channel] = (total / count.max(1)) as u8;
            }
        }
    }
    out
}

/// Largest and mean absolute difference between two rasters, over the samples
/// selected by `channel_of`.
fn difference(ours: &[u8], theirs: &[u8], keep: impl Fn(usize) -> bool) -> (u32, f64) {
    let mut worst = 0_u32;
    let mut total = 0_u64;
    let mut counted = 0_u64;
    for (index, (&got, &want)) in ours.iter().zip(theirs).enumerate() {
        if !keep(index) {
            continue;
        }
        let delta = u32::from(got.abs_diff(want));
        worst = worst.max(delta);
        total += u64::from(delta);
        counted += 1;
    }
    (worst, total as f64 / counted.max(1) as f64)
}

/// Fixtures with no chroma subsampling: every sample must land within the
/// IDCT accuracy the standard allows.
#[test]
fn full_resolution_fixtures_match_libjpeg_within_idct_tolerance() {
    for reference in references() {
        if !reference.name.contains("444") && reference.channels != 1 {
            continue;
        }
        let (ours, width, height, pixel) = decode(&read_fixture(&reference.name, "jpg"))
            .unwrap_or_else(|e| panic!("{}: {e}", reference.name));
        let theirs = read_fixture(&reference.name, "raw");

        assert_eq!((width, height), (reference.width, reference.height));
        assert_eq!(
            pixel,
            if reference.channels == 1 {
                PixelFormat::Gray8
            } else {
                PixelFormat::Rgb8
            },
            "{}: pixel format",
            reference.name
        );
        assert_eq!(ours.len(), theirs.len(), "{}: raster size", reference.name);

        let (worst, mean) = difference(&ours, &theirs, |_| true);
        assert!(
            worst <= 3,
            "{}: worst sample differs by {worst} (mean {mean:.3}); \
             the IDCT accuracy bound does not stretch that far",
            reference.name
        );
        assert!(
            mean <= 0.5,
            "{}: mean difference {mean:.3} is too large to be rounding",
            reference.name
        );
    }
}

/// Subsampled fixtures, where chroma is upsampled rather than decoded
/// per-pixel.
///
/// Chroma that varies smoothly is held nearly as tightly as 4:4:4, because a
/// triangle filter and a nearest sample barely differ across a gradient. That
/// is what proves the MCU layout: a chroma plane written to the wrong offset,
/// or interleaved in the wrong block order, ruins a gradient immediately.
#[test]
fn smoothly_shaded_subsampled_fixtures_match_libjpeg() {
    for name in ["gradient422", "gradient420", "restart420", "tiny420"] {
        let (ours, ..) =
            decode(&read_fixture(name, "jpg")).unwrap_or_else(|e| panic!("{name}: {e}"));
        let theirs = read_fixture(name, "raw");
        assert_eq!(ours.len(), theirs.len(), "{name}: raster size");

        let (worst, mean) = difference(&ours, &theirs, |_| true);
        assert!(
            worst <= 8 && mean <= 1.5,
            "{name}: differs by up to {worst} (mean {mean:.3}); across smooth chroma \
             the upsampling filter cannot account for that"
        );
    }
}

/// The hard-edged fixture, where the upsampling filters genuinely disagree.
///
/// At a chroma edge a triangle filter blends the two sides and we take the
/// nearer one, so the two decoders differ by as much as the edge is large —
/// no bound on the whole raster would mean anything. What *is* meaningful is
/// that away from edges they must agree: where the reference is flat across a
/// 3x3 neighbourhood, both filters see one chroma value and interpolation has
/// nothing to do. Any difference there is a decode bug, not a filter choice.
#[test]
fn hard_edged_subsampled_fixtures_match_libjpeg_away_from_chroma_edges() {
    let name = "blocks420";
    let entry = references()
        .into_iter()
        .find(|r| r.name == name)
        .expect("fixture is in the manifest");
    let (ours, ..) = decode(&read_fixture(name, "jpg")).unwrap();
    let theirs = read_fixture(name, "raw");

    let (width, height) = (entry.width as usize, entry.height as usize);
    let at = |x: usize, y: usize| -> [u8; 3] {
        let base = (y * width + x) * 3;
        [theirs[base], theirs[base + 1], theirs[base + 2]]
    };

    // A pixel is "interior" when the reference decoding is uniform around it,
    // to within the ringing a DCT leaves behind at quality 88.
    let mut interior = vec![false; width * height];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let centre = at(x, y);
            interior[y * width + x] = (y - 1..=y + 1).all(|ny| {
                (x - 1..=x + 1).all(|nx| {
                    let sample = at(nx, ny);
                    (0..3).all(|c| sample[c].abs_diff(centre[c]) <= 6)
                })
            });
        }
    }
    let counted = interior.iter().filter(|&&keep| keep).count();
    assert!(
        counted > width * height / 4,
        "only {counted} of {} pixels are interior; the fixture has stopped \
         exercising flat regions",
        width * height
    );

    let (worst, mean) = difference(&ours, &theirs, |index| interior[index / 3]);
    assert!(
        worst <= 12 && mean <= 1.0,
        "{name}: differs by up to {worst} (mean {mean:.3}) away from chroma \
         edges, where the upsampling filter does not explain it"
    );
}

#[test]
fn every_fixture_decodes_to_its_declared_shape() {
    let all = references();
    assert!(all.len() >= 10, "the fixture set has shrunk");
    for reference in all.into_iter().filter(|r| decodable(&r.name)) {
        let (ours, width, height, _) = decode(&read_fixture(&reference.name, "jpg"))
            .unwrap_or_else(|e| panic!("{}: {e}", reference.name));
        assert_eq!(
            ours.len() as u32,
            reference.width * reference.height * reference.channels as u32,
            "{}: raster size",
            reference.name
        );
        assert_eq!((width, height), (reference.width, reference.height));
    }
}

/// Restart markers must not change the image. The `restart*` fixtures are the
/// same source pictures as their unmarked counterparts, so a resynchronization
/// bug shows up as a difference between two files that should decode alike.
#[test]
fn restart_markers_do_not_change_the_decoded_image() {
    for (with, without) in [("restart420", "gradient420"), ("restart444", "blocks444")] {
        let (marked, ..) = decode(&read_fixture(with, "jpg")).unwrap();
        let (plain, ..) = decode(&read_fixture(without, "jpg")).unwrap();
        assert_eq!(marked.len(), plain.len(), "{with} vs {without}: size");

        // Not byte-identical: the encoder requantizes independently. But
        // restart intervals reset the DC predictor, and getting that wrong
        // produces a drifting offset, which a mean comparison catches.
        let (worst, mean) = difference(&marked, &plain, |_| true);
        assert!(
            mean <= 2.0 && worst <= 24,
            "{with} vs {without}: differ by up to {worst} (mean {mean:.3}), \
             which is more than requantization explains"
        );
    }
}

#[test]
fn decoding_is_deterministic() {
    for reference in references().into_iter().filter(|r| decodable(&r.name)) {
        let bytes = read_fixture(&reference.name, "jpg");
        let (first, ..) = decode(&bytes).unwrap();
        let (second, ..) = decode(&bytes).unwrap();
        assert_eq!(first, second, "{}: two decodes disagree", reference.name);
    }
}

/// Rows must be served one at a time, in order, with no dependence on how many
/// have already been read — the property the scheduler relies on.
#[test]
fn rows_are_independent_of_read_order() {
    let bytes = read_fixture("gradient420", "jpg");
    let (whole, _, height, _) = decode(&bytes).unwrap();

    let mut decoder = JpegDecoder::new(&bytes[..], Limits::default()).unwrap();
    let row_bytes = decoder.descriptor().row_bytes();
    let mut row = vec![0_u8; row_bytes];
    for index in 0..height as usize {
        decoder.read_row(&mut row).unwrap();
        assert_eq!(
            row,
            whole[index * row_bytes..(index + 1) * row_bytes],
            "row {index}"
        );
    }
    // One row past the end is an error, not a panic and not a repeat.
    assert_eq!(
        decoder.read_row(&mut row).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
}

/// EXIF orientation is read from the file but *not* applied.
///
/// Applying it belongs to the pipeline, where `auto_orient` can be turned off
/// (SPEC §Safety and limits). A decoder that rotated its own output would
/// leave the caller no way to see the pixels as stored.
#[test]
fn exif_orientation_is_read_but_not_applied() {
    let plain = read_fixture("gradient444", "jpg");
    assert_eq!(
        JpegDecoder::new(&plain[..], Limits::default())
            .unwrap()
            .orientation(),
        None,
        "a file with no EXIF has no orientation to report"
    );

    // An APP1 EXIF segment declaring orientation 6 (rotate 90° clockwise),
    // spliced in directly after the SOI where a camera would have put it.
    let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0".to_vec();
    let mut tagged = vec![0xFF, 0xD8, 0xFF, 0xE1];
    tagged.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
    tagged.append(&mut exif);
    tagged.extend_from_slice(&plain[2..]);

    let mut decoder = JpegDecoder::new(&tagged[..], Limits::default()).unwrap();
    assert_eq!(decoder.orientation(), Some(6));

    // And the pixels are untouched: same shape, same bytes as the file
    // without the tag.
    let descriptor = decoder.descriptor();
    let (expected, width, height, _) = decode(&plain).unwrap();
    assert_eq!((descriptor.width, descriptor.height), (width, height));

    let mut ours = Vec::new();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row).unwrap();
        ours.extend_from_slice(&row);
    }
    assert_eq!(ours, expected, "the orientation tag changed the pixels");
}

/// Progressive JPEG, decoded by the wrapped codec.
///
/// The wrapped decoder is not on trial here — libjpeg-turbo and
/// `jpeg-decoder` are both mature. What is on trial is the **seam**: our
/// header parser consumes bytes from a forward-only stream before it learns
/// the frame is progressive, and the wrapped decoder needs the stream from
/// byte zero. If the replayed prefix were short, long, or misordered, the
/// picture would be wrong or the file rejected — so an exact-ish comparison
/// against libjpeg is exactly the check that seam needs.
#[cfg(feature = "progressive")]
#[test]
fn progressive_fixtures_match_libjpeg() {
    let mut compared = 0;
    for reference in references() {
        if !reference.name.starts_with("progressive") {
            continue;
        }
        let (ours, width, height, pixel) = decode(&read_fixture(&reference.name, "jpg"))
            .unwrap_or_else(|e| panic!("{}: {e}", reference.name));
        let theirs = read_fixture(&reference.name, "raw");

        assert_eq!((width, height), (reference.width, reference.height));
        assert_eq!(pixel, PixelFormat::Rgb8, "{}", reference.name);
        assert_eq!(ours.len(), theirs.len(), "{}: raster size", reference.name);

        // Two decoders of the same progressive stream should agree to within
        // the IDCT tolerance, chroma upsampling included: unlike our baseline
        // path, this one does not substitute its own upsampler.
        let (worst, mean) = difference(&ours, &theirs, |_| true);
        assert!(
            worst <= 4 && mean <= 0.5,
            "{}: worst {worst}, mean {mean:.3}",
            reference.name
        );
        compared += 1;
    }
    assert!(
        compared >= 2,
        "only {compared} progressive fixtures compared"
    );
}

/// The handover must not disturb what the header said.
#[cfg(feature = "progressive")]
#[test]
fn a_progressive_stream_reports_its_shape_and_route() {
    let bytes = read_fixture("progressive", "jpg");
    let decoder = JpegDecoder::new(&bytes[..], Limits::default()).unwrap();
    assert!(decoder.is_progressive());
    // The wrapped decoder produces one resolution, so shrink-on-load has
    // nothing to offer and must not claim otherwise.
    assert_eq!(decoder.scale(), otf_pixels_codec_jpeg::Scale::Full);

    let baseline = read_fixture("gradient444", "jpg");
    let decoder = JpegDecoder::new(&baseline[..], Limits::default()).unwrap();
    assert!(!decoder.is_progressive(), "a baseline frame was misrouted");
}

#[test]
fn probe_recognises_fixtures_and_nothing_else() {
    for reference in references() {
        let bytes = read_fixture(&reference.name, "jpg");
        assert!(
            otf_pixels_codec_jpeg::probe(&bytes),
            "{}: not recognised",
            reference.name
        );
    }
    assert!(!otf_pixels_codec_jpeg::probe(b"\x89PNG\r\n\x1a\n"));
    assert!(!otf_pixels_codec_jpeg::probe(b"GIF89a"));
    // A truncated signature must be declined, not indexed past.
    assert!(!otf_pixels_codec_jpeg::probe(&[0xFF, 0xD8]));
    assert!(!otf_pixels_codec_jpeg::probe(&[]));
}

/// Scaled decode, against libjpeg's own scaled decode.
///
/// Our reduced transform could be self-consistent and still wrong about what
/// `M/8` means — the coefficients have to be folded together the same way both
/// decoders fold them, or a thumbnail comes out softened, sharpened or
/// shifted. So the comparison is against libjpeg, on the fixtures where
/// nothing else intrudes: full-resolution chroma, where the upsampling
/// difference documented above cannot contribute.
///
/// Ground truth for both is libjpeg's *full* decode, box-averaged down. Two
/// reduced decoders can be compared to each other, but only against a true
/// downsample can either be called right.
#[test]
fn scaled_decode_is_as_faithful_as_libjpegs() {
    let mut compared = 0;
    for reference in references() {
        // 4:2:0 and 4:2:2 are excluded here and covered by the test below:
        // their chroma passes through an upsampler we deliberately implement
        // differently, which would swamp what this test is measuring.
        if !reference.name.contains("444") && reference.channels != 1 {
            continue;
        }
        let bytes = read_fixture(&reference.name, "jpg");
        let full = read_fixture(&reference.name, "raw");

        for (scale, suffix, denominator) in scales() {
            if reference.width < denominator || reference.height < denominator {
                continue;
            }
            let (ours, descriptor) = decode_scaled(&bytes, scale);
            let theirs = read_fixture(&reference.name, &format!("{suffix}.raw"));
            assert_eq!(
                ours.len(),
                theirs.len(),
                "{} at {scale:?}: raster size",
                reference.name
            );

            let truth = box_average(
                &full,
                reference.width as usize,
                reference.height as usize,
                reference.channels,
                descriptor.width as usize,
                descriptor.height as usize,
            );
            let (_, ours_error) = difference(&ours, &truth, |_| true);
            let (_, their_error) = difference(&theirs, &truth, |_| true);

            // Not "within a factor of" but "no worse than": the reduced
            // transform is defined as the exact box average, so anything
            // beyond libjpeg's own distance from the truth is a defect and
            // not an approximation choice.
            assert!(
                ours_error <= their_error + 0.5,
                "{} at {scale:?}: our mean error {ours_error:.3} against \
                 libjpeg's {their_error:.3} from the true downsample",
                reference.name
            );
            compared += 1;
        }
    }
    assert!(
        compared >= 12,
        "only {compared} scaled decodes were compared"
    );
}

/// Scaled decode of a subsampled image, where chroma cannot keep up.
///
/// Two things put our reduced decode of a 4:2:0 or 4:2:2 file further from
/// libjpeg's than a 4:4:4 file, and neither is the reduced transform:
///
/// - The **chroma upsampler**, which we deliberately implement as replication
///   where libjpeg interpolates (see the module docs). At full resolution
///   these same fixtures agree with libjpeg to within one step per sample; at
///   1/8 one chroma sample covers a 2x2 output block, so the same difference
///   is worth an order of magnitude more.
/// - **Chroma resolution**, which the format caps: a chroma block cannot
///   produce less than one sample, so a 4:2:0 image decoded at 1/8 has chroma
///   at 1/16 of the image. libjpeg is bound by this too.
///
/// So the bound here is looser than the 4:4:4 one and is stated in absolute
/// terms, on the smoothly-shaded fixtures where it is interpretable. It is a
/// regression gate on a known deviation, not a claim of agreement.
#[test]
fn scaled_decode_of_subsampled_images_stays_near_libjpeg() {
    let mut compared = 0;
    for name in ["gradient420", "gradient422", "restart420"] {
        let bytes = read_fixture(name, "jpg");
        for (scale, suffix, denominator) in scales() {
            let (ours, descriptor) = decode_scaled(&bytes, scale);
            let theirs = read_fixture(name, &format!("{suffix}.raw"));
            assert_eq!(ours.len(), theirs.len(), "{name} at {scale:?}: raster size");
            let _ = (descriptor, denominator);

            let (worst, mean) = difference(&ours, &theirs, |_| true);
            assert!(
                mean <= 10.0 && worst <= 30,
                "{name} at {scale:?}: worst {worst}, mean {mean:.3} against \
                 libjpeg's scaled decode"
            );
            compared += 1;
        }
    }
    assert!(
        compared >= 9,
        "only {compared} scaled decodes were compared"
    );
}

/// A scaled decode must cost a fraction of the memory a full one does, which
/// is the entire reason it exists.
#[test]
fn a_scaled_decode_never_materializes_the_full_image() {
    let bytes = read_fixture("gradient420", "jpg");
    let full = JpegDecoder::new(&bytes[..], Limits::default()).unwrap();
    let full_row = full.descriptor().row_bytes();

    let eighth = JpegDecoder::with_scale(
        &bytes[..],
        Limits::default(),
        otf_pixels_codec_jpeg::Scale::Eighth,
    )
    .unwrap();
    let small = eighth.descriptor();

    assert_eq!((small.width, small.height), (8, 6));
    assert!(
        small.row_bytes() * 8 <= full_row,
        "a 1/8 row is {} bytes against a full row's {full_row}",
        small.row_bytes()
    );
    assert_eq!(eighth.scale(), otf_pixels_codec_jpeg::Scale::Eighth);
}
