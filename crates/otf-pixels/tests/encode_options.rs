//! sharp's size-changing output options through the facade: palette PNG
//! (`palette`, `colours`, `dither`), PNG `compressionLevel`, GIF `colours`
//! and `dither`, WebP `effort`, `alphaQuality` and `nearLossless`, AVIF
//! `effort`, and `chromaSubsampling` for JPEG and AVIF.
//!
//! The encoders check exact bitstreams; this checks the options reach them
//! from `output`, that the files decode, and that defaults are unchanged.

#![cfg(all(
    feature = "raw",
    feature = "png",
    feature = "gif",
    feature = "webp",
    feature = "avif",
    feature = "jpeg"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    ChromaSubsampling, EncodeOptions, ErrorCode, Format, Image, ImageDescriptor, PixelFormat,
};

/// A `width` x `height` image of photographic noise in `pixel`.
fn noisy(width: u32, height: u32, pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(width, height, pixel).unwrap();
    let mut seed = 7_u32;
    let bytes = (0..descriptor.byte_len().unwrap())
        .map(|_| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (seed >> 16) as u8
        })
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

fn encode(image: Image, format: Format, options: EncodeOptions) -> Vec<u8> {
    image.output(format, options).bytes().unwrap()
}

/// The distinct pixels of an encoded file, decoded.
fn distinct(bytes: Vec<u8>) -> usize {
    let image = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
    let bpp = image.descriptor().unwrap().pixel.bytes_per_pixel();
    let raw = encode(image, Format::Raw, EncodeOptions::default());
    let mut pixels: Vec<&[u8]> = raw.chunks_exact(bpp).collect();
    pixels.sort_unstable();
    pixels.dedup();
    pixels.len()
}

#[test]
fn a_palette_png_is_smaller_and_has_at_most_its_colours() {
    let image = || noisy(96, 64, PixelFormat::Rgba8);
    let full = encode(image(), Format::Png, EncodeOptions::default());
    let palette = encode(
        image(),
        Format::Png,
        EncodeOptions::default().with_colours(32).unwrap(),
    );
    assert!(
        palette.len() * 2 < full.len(),
        "{} against {}",
        palette.len(),
        full.len()
    );
    assert!(distinct(palette) <= 32);

    // `palette` alone gives 256 colours; `dither` alone implies a palette.
    let implied = encode(
        image(),
        Format::Png,
        EncodeOptions::default().with_palette(true),
    );
    assert!(distinct(implied) <= 256);
    let undithered = encode(
        image(),
        Format::Png,
        EncodeOptions::default().with_dither(0.0).unwrap(),
    );
    assert_eq!(undithered[25], 3, "colour type 3 is a palette");
}

#[test]
fn a_palette_png_keeps_a_wide_image_decodable() {
    let bytes = encode(
        noisy(20, 10, PixelFormat::Rgb16).resize(10, 5),
        Format::Png,
        EncodeOptions::default().with_palette(true),
    );
    let decoded = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
    assert_eq!(decoded.descriptor().unwrap().pixel, PixelFormat::Rgb8);
}

#[test]
fn the_compression_level_trades_size_for_time() {
    let image = || noisy(64, 64, PixelFormat::Gray8).blur(2.0);
    let level = |n| {
        encode(
            image(),
            Format::Png,
            EncodeOptions::default().with_compression_level(n).unwrap(),
        )
        .len()
    };
    assert!(level(9) < level(0));
}

#[test]
fn gif_takes_colours_and_dither() {
    let image = || noisy(64, 48, PixelFormat::Rgb8);
    let few = encode(
        image(),
        Format::Gif,
        EncodeOptions::default().with_colours(4).unwrap(),
    );
    assert!(distinct(few) <= 4);
    let flat = EncodeOptions::default()
        .with_colours(4)
        .unwrap()
        .with_dither(0.0)
        .unwrap();
    assert_ne!(
        encode(image(), Format::Gif, flat),
        encode(
            image(),
            Format::Gif,
            EncodeOptions::default().with_colours(4).unwrap()
        )
    );
}

#[test]
fn out_of_range_options_are_invalid_arguments() {
    for error in [
        EncodeOptions::default().with_colours(1).unwrap_err(),
        EncodeOptions::default().with_colours(257).unwrap_err(),
        EncodeOptions::default().with_dither(1.5).unwrap_err(),
        EncodeOptions::default()
            .with_compression_level(10)
            .unwrap_err(),
    ] {
        assert_eq!(error.code(), ErrorCode::InvalidArgument, "{error}");
    }
}

#[test]
fn webp_effort_trades_time_for_size_and_keeps_its_default() {
    let image = || noisy(96, 80, PixelFormat::Rgb8).blur(1.5);
    for lossless in [false, true] {
        let options = EncodeOptions::default().with_lossless(lossless);
        let size =
            |effort| encode(image(), Format::WebP, options.with_effort(effort).unwrap()).len();
        assert_eq!(
            encode(image(), Format::WebP, options),
            encode(image(), Format::WebP, options.with_effort(4).unwrap()),
            "the default effort is 4"
        );
        assert!(size(6) <= size(0), "lossless {lossless}");
    }
}

#[test]
fn webp_alpha_quality_and_near_lossless_shrink_the_file() {
    let translucent = || noisy(96, 80, PixelFormat::Rgba8).blur(2.0);
    let full = encode(translucent(), Format::WebP, EncodeOptions::default());
    let coarse = encode(
        translucent(),
        Format::WebP,
        EncodeOptions::default().with_alpha_quality(10).unwrap(),
    );
    assert!(
        coarse.len() < full.len(),
        "{} against {}",
        coarse.len(),
        full.len()
    );

    let photo = || noisy(128, 96, PixelFormat::Rgb8).blur(1.0);
    let lossless = encode(
        photo(),
        Format::WebP,
        EncodeOptions::default().with_lossless(true),
    );
    let near = encode(
        photo(),
        Format::WebP,
        EncodeOptions::with_quality(40)
            .unwrap()
            .with_near_lossless(true),
    );
    assert!(
        near.len() < lossless.len(),
        "{} against {}",
        near.len(),
        lossless.len()
    );
    // Close to the source, which plain lossy WebP is not held to.
    let decoded = Image::from_stream(std::io::Cursor::new(near)).unwrap();
    let got = encode(decoded, Format::Raw, EncodeOptions::default());
    let want = encode(photo(), Format::Raw, EncodeOptions::default());
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(worst <= 16, "near-lossless moved a sample by {worst}");
}

/// Thin red and blue stripes, which 4:2:0 averages to purple.
fn stripes() -> Image {
    let descriptor = ImageDescriptor::new(48, 32, PixelFormat::Rgb8).unwrap();
    let bytes = (0..48 * 32)
        .flat_map(|i| {
            if i % 2 == 0 {
                [230, 20, 20]
            } else {
                [20, 20, 230]
            }
        })
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

/// Mean absolute error of an encoded file against the stripes.
fn stripe_error(bytes: Vec<u8>) -> u64 {
    let decoded = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
    let got = encode(
        decoded.to_pixel_format(PixelFormat::Rgb8),
        Format::Raw,
        EncodeOptions::default(),
    );
    let want = encode(stripes(), Format::Raw, EncodeOptions::default());
    got.iter()
        .zip(&want)
        .map(|(a, b)| u64::from(a.abs_diff(*b)))
        .sum::<u64>()
        / want.len() as u64
}

#[test]
fn full_chroma_keeps_colour_edges_in_jpeg_and_avif() {
    for format in [Format::Jpeg, Format::Avif] {
        let options = EncodeOptions::with_quality(85).unwrap();
        let sub = stripe_error(encode(
            stripes(),
            format,
            options.with_chroma_subsampling(ChromaSubsampling::Yuv420),
        ));
        let full = stripe_error(encode(
            stripes(),
            format,
            options.with_chroma_subsampling(ChromaSubsampling::Yuv444),
        ));
        assert!(
            full * 2 < sub,
            "{format}: 4:4:4 error {full} against 4:2:0 {sub}"
        );
    }
    // Unset, JPEG still follows quality and AVIF stays 4:2:0, as before.
    let jpeg = EncodeOptions::with_quality(85).unwrap();
    assert_eq!(
        encode(stripes(), Format::Jpeg, jpeg),
        encode(
            stripes(),
            Format::Jpeg,
            jpeg.with_chroma_subsampling(ChromaSubsampling::Yuv420)
        )
    );
    assert_eq!(
        encode(stripes(), Format::Avif, EncodeOptions::default()),
        encode(
            stripes(),
            Format::Avif,
            EncodeOptions::default().with_chroma_subsampling(ChromaSubsampling::Yuv420)
        )
    );
}

#[test]
fn avif_effort_keeps_its_default_and_decodes_at_every_level() {
    let image = || noisy(40, 30, PixelFormat::Rgba8).blur(1.0);
    let default = encode(image(), Format::Avif, EncodeOptions::default());
    assert_eq!(
        default,
        encode(
            image(),
            Format::Avif,
            EncodeOptions::default().with_effort(4).unwrap()
        )
    );
    for effort in [0, 2, 9] {
        let bytes = encode(
            image(),
            Format::Avif,
            EncodeOptions::default().with_effort(effort).unwrap(),
        );
        let decoded = Image::from_stream(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(decoded.descriptor().unwrap().width, 40);
    }
}
