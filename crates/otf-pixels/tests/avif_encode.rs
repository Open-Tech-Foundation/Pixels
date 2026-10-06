//! AVIF encoding through the public facade: pixels in, an AVIF out, and that
//! AVIF back through the decoder. libaom's agreement with our decoder on the
//! coded frames is checked separately (`scripts/check-avif-interop.sh`); this
//! checks what a user goes through, end to end.

#![cfg(all(feature = "avif", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]

use otf_pixels::{EncodeOptions, ErrorCode, Format, Image, ImageDescriptor, PixelFormat};

fn photo() -> (u32, u32, Vec<u8>) {
    let path = format!(
        "{}/../otf-pixels-codec-avif/tests/fixtures/photo_420.raw",
        env!("CARGO_MANIFEST_DIR")
    );
    (96, 64, std::fs::read(path).unwrap())
}

fn encode(
    width: u32,
    height: u32,
    pixel: PixelFormat,
    bytes: Vec<u8>,
    options: EncodeOptions,
) -> Vec<u8> {
    let descriptor = ImageDescriptor::new(width, height, pixel).unwrap();
    Image::from_raw(descriptor, bytes)
        .unwrap()
        .output(Format::Avif, options)
        .bytes()
        .unwrap()
}

fn decode(avif: Vec<u8>) -> (otf_pixels::Metadata, Vec<u8>) {
    let image = Image::from_stream(std::io::Cursor::new(avif)).unwrap();
    let metadata = image.metadata().unwrap();
    (
        metadata,
        image
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap(),
    )
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x.abs_diff(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

#[test]
fn a_photo_round_trips_through_avif() {
    let (width, height, rgb) = photo();
    let avif = encode(
        width,
        height,
        PixelFormat::Rgb8,
        rgb.clone(),
        EncodeOptions::default(),
    );
    assert_eq!(&avif[4..12], b"ftypavif");
    let (metadata, back) = decode(avif);
    assert_eq!(metadata.format, Format::Avif);
    assert_eq!(
        (metadata.width, metadata.height, metadata.pixel),
        (96, 64, PixelFormat::Rgb8)
    );
    let p = psnr(&back, &rgb);
    assert!(p > 30.0, "{p:.1} dB");
}

#[test]
fn quality_is_honoured() {
    let (width, height, rgb) = photo();
    let low = encode(
        width,
        height,
        PixelFormat::Rgb8,
        rgb.clone(),
        EncodeOptions::with_quality(20).unwrap(),
    );
    let high = encode(
        width,
        height,
        PixelFormat::Rgb8,
        rgb.clone(),
        EncodeOptions::with_quality(95).unwrap(),
    );
    assert!(low.len() < high.len(), "{} vs {}", low.len(), high.len());
    assert!(psnr(&decode(high).1, &rgb) > psnr(&decode(low).1, &rgb));
}

#[test]
fn transparency_is_kept_and_opaque_alpha_is_dropped() {
    let (width, height, rgb) = photo();
    let mut rgba = Vec::new();
    for (i, p) in rgb.chunks(3).enumerate() {
        let x = i % width as usize;
        rgba.extend_from_slice(p);
        rgba.push((x * 255 / (width as usize - 1)) as u8);
    }
    let (metadata, back) = decode(encode(
        width,
        height,
        PixelFormat::Rgba8,
        rgba.clone(),
        EncodeOptions::default(),
    ));
    assert_eq!(metadata.pixel, PixelFormat::Rgba8);
    let alpha_in: Vec<u8> = rgba.iter().skip(3).step_by(4).copied().collect();
    let alpha_out: Vec<u8> = back.iter().skip(3).step_by(4).copied().collect();
    assert!(psnr(&alpha_out, &alpha_in) > 40.0);

    let opaque: Vec<u8> = rgb
        .chunks(3)
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    let (metadata, _) = decode(encode(
        width,
        height,
        PixelFormat::Rgba8,
        opaque,
        EncodeOptions::default(),
    ));
    assert_eq!(metadata.pixel, PixelFormat::Rgb8);
}

#[test]
fn grey_codes_as_monochrome() {
    let grey: Vec<u8> = (0..40 * 30).map(|i| ((i % 40) * 6) as u8).collect();
    let (metadata, back) = decode(encode(
        40,
        30,
        PixelFormat::Gray8,
        grey.clone(),
        EncodeOptions::default(),
    ));
    assert_eq!(metadata.pixel, PixelFormat::Gray8);
    assert!(psnr(&back, &grey) > 35.0);
}

#[test]
fn tiny_and_odd_sizes_encode() {
    for (w, h) in [(1, 1), (3, 5), (17, 9)] {
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 37 % 256) as u8).collect();
        let (metadata, back) = decode(encode(
            w,
            h,
            PixelFormat::Rgb8,
            rgb,
            EncodeOptions::default(),
        ));
        assert_eq!((metadata.width, metadata.height), (w, h));
        assert_eq!(back.len(), (w * h * 3) as usize);
    }
}

#[test]
fn lossless_and_sixteen_bit_are_refused() {
    let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Rgb8).unwrap();
    let error = Image::from_raw(descriptor, vec![0; 12])
        .unwrap()
        .output(Format::Avif, EncodeOptions::default().with_lossless(true))
        .bytes()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported);
    let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Rgb16).unwrap();
    let error = Image::from_raw(descriptor, vec![0; 24])
        .unwrap()
        .output(Format::Avif, EncodeOptions::default())
        .bytes()
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::Unsupported);
}
