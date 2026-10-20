//! JPEG XL input through the facade: sniffed, decoded, and carried through a
//! pipeline. Exactness against libjxl is the codec crate's `reference` test;
//! this checks the facade reaches the decoder and what it reports.

#![cfg(all(feature = "raw", feature = "jxl", feature = "png"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{EncodeOptions, Format, Image, ImageDescriptor, PixelFormat};

fn image(width: u32, height: u32, pixel: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
    let descriptor = ImageDescriptor::new(width, height, pixel).unwrap();
    let bytes = (0..descriptor.byte_len().unwrap())
        .map(|i| ((i / 3) % 200 + (i % 3) * 20) as u8)
        .collect();
    (descriptor, bytes)
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/../otf-pixels-codec-jxl/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn lossless_output_comes_back_exactly() {
    for pixel in [
        PixelFormat::Gray8,
        PixelFormat::Rgb8,
        PixelFormat::Rgba8,
        PixelFormat::Rgb16,
    ] {
        let (descriptor, bytes) = image(70, 45, pixel);
        let jxl = Image::from_raw(descriptor, bytes.clone())
            .unwrap()
            .output(Format::Jxl, EncodeOptions::default().with_lossless(true))
            .bytes()
            .unwrap();
        let decoded = Image::from_stream(std::io::Cursor::new(jxl)).unwrap();
        let metadata = decoded.metadata().unwrap();
        assert_eq!(metadata.format, Format::Jxl, "{pixel}: sniffed");
        assert_eq!(metadata.pixel, pixel);
        let raw = decoded
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert!(raw == bytes, "{pixel}: lossless round trip");
    }
}

#[test]
fn lossy_output_comes_back_close() {
    let (descriptor, bytes) = image(96, 64, PixelFormat::Rgb8);
    let jxl = Image::from_raw(descriptor, bytes.clone())
        .unwrap()
        .output(Format::Jxl, EncodeOptions::default())
        .bytes()
        .unwrap();
    let raw = Image::from_stream(std::io::Cursor::new(jxl))
        .unwrap()
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let mse = raw
        .iter()
        .zip(&bytes)
        .map(|(&a, &b)| f64::from(a.abs_diff(b)).powi(2))
        .sum::<f64>()
        / bytes.len() as f64;
    let psnr = 10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10();
    assert!(psnr > 35.0, "PSNR {psnr:.1} dB");
}

#[test]
fn a_libjxl_file_converts_to_png_through_a_pipeline() {
    let image = Image::from_stream(std::io::Cursor::new(fixture("photo_d1.jxl"))).unwrap();
    let metadata = image.metadata().unwrap();
    assert_eq!(
        (metadata.format, metadata.width, metadata.height),
        (Format::Jxl, 96, 72)
    );
    let png = image
        .resize(48, 36)
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    assert_eq!(&png[1..4], b"PNG");
    let back = Image::from_stream(std::io::Cursor::new(png))
        .unwrap()
        .metadata()
        .unwrap();
    assert_eq!((back.width, back.height), (48, 36));
}

#[test]
fn jpeg_recompressed_and_container_files_open() {
    for name in ["photo_jpeg.jxl", "anim.jxl", "text_d1_patches.jxl"] {
        let image = Image::from_stream(std::io::Cursor::new(fixture(name))).unwrap();
        assert_eq!(image.metadata().unwrap().format, Format::Jxl, "{name}");
        image
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
    }
}
