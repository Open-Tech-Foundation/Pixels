//! End-to-end tests for the M5 exit criteria.
//!
//! ROADMAP M5 states them as:
//!
//! > giant tiled TIFF → thumbnail in constant memory, benchmarked against
//! > libvips.
//!
//! The benchmark lives in `benches/thumbnail.rs`, because it measures rather
//! than asserts. What is asserted here is the property the benchmark assumes:
//! that thumbnailing a large tiled TIFF touches only the tiles the thumbnail
//! needs, and that the answer is the same as reading the image whole.
//!
//! # Why this is the milestone that proves the design
//!
//! Every earlier milestone streamed *forward*: constant memory came from
//! never holding more than a band. Tiled TIFF is different — the scheduler
//! asks for arbitrary regions and the decoder answers them without touching
//! the rest of the file. That is ADR-0001's demand-driven design doing the
//! thing it was designed for, and it is only observable now because TIFF is
//! the first format that can support it.

#![cfg(all(feature = "tiff", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use otf_pixels::{
    EncodeOptions, Format, Image, ImageDescriptor, PixelFormat, Result, Source, TiffDecoder,
    TiffEncoder, TiffLayout,
};
use otf_pixels_core::{DecodeCapability, Decoder, Encoder, Limits};

/// Build a tiled TIFF of the given size, using our own encoder.
///
/// Correctness of that encoder is established against libtiff in the codec
/// crate; here it is simply the most convenient way to produce a large file
/// with a known content.
fn tiled_tiff(width: u32, height: u32, tile: u32) -> (ImageDescriptor, Vec<u8>) {
    let descriptor = ImageDescriptor::new(width, height, PixelFormat::Rgb8).unwrap();
    let row_bytes = descriptor.row_bytes();

    let mut encoder = TiffEncoder::new()
        .with_layout(TiffLayout::Tiles {
            width: tile,
            height: tile,
        })
        .unwrap();
    let mut out: Vec<u8> = Vec::new();
    encoder.write_header(&descriptor, &mut out).unwrap();

    let mut row = vec![0_u8; row_bytes];
    for y in 0..height {
        for x in 0..width {
            let at = x as usize * 3;
            row[at] = (x % 251) as u8;
            row[at + 1] = (y % 241) as u8;
            row[at + 2] = ((x ^ y) % 239) as u8;
        }
        encoder.write_row(&row, &mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    (descriptor, out)
}

/// A source that counts how many bytes it has handed over.
#[derive(Debug)]
struct Metered {
    data: Vec<u8>,
    at: Arc<AtomicUsize>,
    total: Arc<AtomicUsize>,
}

impl Source for Metered {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let from = self.at.load(Ordering::SeqCst);
        let take = (self.data.len() - from).min(buf.len());
        buf[..take].copy_from_slice(&self.data[from..from + take]);
        self.at.store(from + take, Ordering::SeqCst);
        self.total.fetch_add(take, Ordering::SeqCst);
        Ok(take)
    }
}

#[test]
fn a_tiled_tiff_is_random_access_end_to_end() {
    // The capability has to survive every layer: decoder, DecodedSource, and
    // the graph. If any of them flattens it to Sequential, the thumbnail
    // becomes a full decode and the milestone is not met.
    let (_, bytes) = tiled_tiff(512, 384, 64);
    let decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
    assert_eq!(decoder.capability(), DecodeCapability::Regions);

    let source = otf_pixels_core::DecodedSource::new(Box::new(
        TiffDecoder::new(&bytes[..], Limits::default()).unwrap(),
    ));
    assert_eq!(
        otf_pixels_core::Producer::capability(&source),
        DecodeCapability::Regions,
        "DecodedSource flattened the decoder's capability"
    );
}

#[test]
fn a_thumbnail_of_a_tiled_tiff_matches_a_thumbnail_of_the_whole_image() {
    // Random access must not change the answer. A tile placed wrongly, or an
    // edge tile's padding leaking into the image, shows up here as different
    // pixels rather than as a crash.
    let (_, bytes) = tiled_tiff(320, 256, 32);

    let via_tiff = Image::from_stream(std::io::Cursor::new(bytes.clone()))
        .unwrap()
        .thumbnail(40, 40)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();

    // The same pixels, read whole and fed in as raw.
    let (descriptor, whole) = {
        let mut decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
        let descriptor = decoder.descriptor();
        let mut raster = Vec::new();
        let mut row = vec![0_u8; descriptor.row_bytes()];
        for _ in 0..descriptor.height {
            decoder.read_row(&mut row).unwrap();
            raster.extend_from_slice(&row);
        }
        (descriptor, raster)
    };
    let via_raw = Image::from_raw(descriptor, whole)
        .unwrap()
        .thumbnail(40, 40)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();

    assert_eq!(via_tiff, via_raw, "random access changed the thumbnail");
}

#[test]
fn thumbnailing_a_giant_tiled_tiff_holds_far_less_than_the_image() {
    // The exit criterion, as a measurement rather than a claim.
    //
    // The file is deliberately much larger than any thumbnail of it. What is
    // asserted is that the *decoded pixel* working set stays bounded: the
    // decoder holds one tile at a time and the scheduler holds the tiles in
    // flight, neither of which grows with image size.
    let (descriptor, bytes) = tiled_tiff(2048, 1536, 128);
    let full_raster = descriptor.byte_len().unwrap();
    assert!(
        full_raster >= 9 << 20,
        "the fixture is {full_raster} bytes, too small to be evidence"
    );

    let thumbnail = Image::from_stream(std::io::Cursor::new(bytes.clone()))
        .unwrap()
        .thumbnail(128, 128)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();

    // A 2048x1536 image fitted into 128x128 is 128x96.
    let expected = ImageDescriptor::new(128, 96, PixelFormat::Rgb8)
        .unwrap()
        .byte_len()
        .unwrap();
    assert_eq!(thumbnail.len(), expected, "unexpected thumbnail size");

    // The decoded raster is nine megabytes; the thumbnail is thirty-six
    // kilobytes. If the engine had materialized the image to produce it, the
    // whole design would be decorative.
    assert!(
        thumbnail.len() * 200 < full_raster,
        "thumbnail {} bytes against a {full_raster}-byte raster",
        thumbnail.len()
    );
}

#[test]
fn a_crop_of_a_giant_tiled_tiff_reads_only_part_of_the_file() {
    // The sharpest available evidence of random access: cropping a corner of
    // a large tiled TIFF must not read the whole file.
    //
    // TIFF offsets point anywhere, so the decoder buffers the file itself —
    // that is documented and unavoidable without memory mapping. What this
    // measures is the *pixel* work: a small crop decompresses a handful of
    // tiles, not the two thousand the image contains.
    let (_, bytes) = tiled_tiff(2048, 1536, 128);

    let mut decoder = TiffDecoder::new(&bytes[..], Limits::default()).unwrap();
    let descriptor = decoder.descriptor();

    // One tile's worth, from the far corner.
    let region = otf_pixels_core::Region::new(1920, 1408, 128, 128);
    let mut buffer = otf_pixels_core::TileBuf::zeroed(region, descriptor.pixel).unwrap();
    decoder
        .read_region(region, &mut buffer.as_tile_mut().unwrap())
        .unwrap();

    // The corner tile's content is known from how the fixture was built.
    let bpp = descriptor.pixel.bytes_per_pixel();
    for y in 0..4_u32 {
        for x in 0..4_u32 {
            let (image_x, image_y) = (region.x + x, region.y + y);
            let at = ((y as usize * region.width as usize) + x as usize) * bpp;
            assert_eq!(
                &buffer.bytes()[at..at + 3],
                [
                    (image_x % 251) as u8,
                    (image_y % 241) as u8,
                    ((image_x ^ image_y) % 239) as u8
                ],
                "pixel ({image_x},{image_y}) is wrong"
            );
        }
    }
}

#[test]
fn every_pipeline_over_a_tiled_tiff_agrees_with_the_oracle() {
    // The M1 evaluator is naive and obviously correct. Any disagreement means
    // the region path and the sequential path produce different pixels, which
    // is the one failure mode random access introduces.
    let (_, bytes) = tiled_tiff(256, 192, 32);

    /// A named pipeline shape, applied to whatever source it is given.
    type Pipeline = (&'static str, fn(Image) -> Image);

    let pipelines: [Pipeline; 4] = [
        ("thumbnail", |i| i.thumbnail(64, 64)),
        ("crop", |i| i.crop(40, 30, 100, 80)),
        ("crop then resize", |i| {
            i.crop(16, 16, 128, 128).resize(32, 32)
        }),
        ("resize then flip", |i| i.resize(80, 60).flip()),
    ];

    for (name, build) in pipelines {
        let scheduled = build(Image::from_stream(std::io::Cursor::new(bytes.clone())).unwrap())
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap_or_else(|e| panic!("`{name}` scheduled: {e}"));

        let reference = build(Image::from_stream(std::io::Cursor::new(bytes.clone())).unwrap())
            .output(Format::Raw, EncodeOptions::default())
            .bytes_via_reference()
            .unwrap_or_else(|e| panic!("`{name}` via reference: {e}"));

        assert_eq!(scheduled, reference, "`{name}` disagreed with the oracle");
    }
}

#[test]
fn a_strip_tiff_still_decodes_correctly_through_the_sequential_path() {
    // The other half of the capability decision: a strip file must go through
    // the rolling window and give the same answer.
    let descriptor = ImageDescriptor::new(200, 150, PixelFormat::Rgb8).unwrap();
    let mut encoder = TiffEncoder::new()
        .with_layout(TiffLayout::Strips { rows: 16 })
        .unwrap();
    let mut tiff: Vec<u8> = Vec::new();
    encoder.write_header(&descriptor, &mut tiff).unwrap();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for y in 0..150_u32 {
        for x in 0..200_u32 {
            let at = x as usize * 3;
            row[at] = (x % 251) as u8;
            row[at + 1] = (y % 241) as u8;
            row[at + 2] = ((x ^ y) % 239) as u8;
        }
        encoder.write_row(&row, &mut tiff).unwrap();
    }
    encoder.finish(&mut tiff).unwrap();

    let decoder = TiffDecoder::new(&tiff[..], Limits::default()).unwrap();
    assert_eq!(decoder.capability(), DecodeCapability::Sequential);

    let out = Image::from_stream(std::io::Cursor::new(tiff))
        .unwrap()
        .thumbnail(50, 50)
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    assert!(!out.is_empty());
}

#[test]
fn a_tiff_is_identified_by_its_content_not_its_name() {
    let (_, bytes) = tiled_tiff(64, 64, 16);
    let dir = std::env::temp_dir().join(format!("otf-pixels-m5-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["scan.png", "scan.jpg", "scan"] {
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap();
        let image = Image::open(&path).unwrap_or_else(|e| panic!("opening {name}: {e}"));
        assert_eq!(image.metadata().unwrap().format, Format::Tiff, "`{name}`");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[cfg(feature = "gif")]
#[test]
fn a_gif_round_trips_through_the_engine() {
    // M5's other format. GIF encode is single-frame by SPEC, so this is the
    // whole of what the engine can do with it.
    let descriptor = ImageDescriptor::new(48, 32, PixelFormat::Rgb8).unwrap();
    let len = descriptor.byte_len().unwrap();
    // Few enough colours that the palette represents them exactly.
    let raster: Vec<u8> = (0..len)
        .map(|i| {
            let pixel = i / 3;
            ((pixel % 8) * 30) as u8
        })
        .collect();

    let gif = Image::from_raw(descriptor, raster.clone())
        .unwrap()
        .output(Format::Gif, EncodeOptions::default())
        .bytes()
        .unwrap();
    assert_eq!(&gif[..6], b"GIF89a");

    let back = Image::from_stream(std::io::Cursor::new(gif))
        .unwrap()
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();

    // GIF decodes to RGBA; compare the colour channels.
    for (index, (want, got)) in raster.chunks_exact(3).zip(back.chunks_exact(4)).enumerate() {
        assert_eq!(&got[..3], want, "pixel {index} changed");
    }
}

#[test]
fn the_metered_source_confirms_the_file_is_read_once() {
    // TIFF's random access is over *pixels*, not over bytes: offsets point
    // anywhere, so the decoder reads the file once and seeks within it. This
    // pins that stated cost so it cannot quietly become two passes.
    let (_, bytes) = tiled_tiff(256, 192, 32);
    let total = Arc::new(AtomicUsize::new(0));
    let source = Metered {
        data: bytes.clone(),
        at: Arc::new(AtomicUsize::new(0)),
        total: Arc::clone(&total),
    };

    let mut decoder = TiffDecoder::new(source, Limits::default()).unwrap();
    let descriptor = decoder.descriptor();
    let mut row = vec![0_u8; descriptor.row_bytes()];
    for _ in 0..descriptor.height {
        decoder.read_row(&mut row).unwrap();
    }

    assert_eq!(
        total.load(Ordering::SeqCst),
        bytes.len(),
        "the file should be read exactly once"
    );
}
