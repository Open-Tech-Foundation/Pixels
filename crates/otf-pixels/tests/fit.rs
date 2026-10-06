//! sharp's five fit modes through the facade, including the fast path that
//! matters most for them: a cover or contain thumbnail of a JPEG still
//! decodes at a reduced scale, and the scheduled result still matches the
//! reference evaluator byte for byte.

#![cfg(all(feature = "jpeg", feature = "raw"))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{EncodeOptions, Fit, Format, Image, ImageDescriptor, PixelFormat, ResizeOptions};

/// A 1024x768 JPEG, made here so the reduced decode has room to act.
fn photo() -> Vec<u8> {
    let (w, h) = (1024_u32, 768_u32);
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [(x / 4) as u8, (y / 3) as u8, ((x + y) / 7) as u8]
        })
        .collect();
    Image::from_raw(
        ImageDescriptor::new(w, h, PixelFormat::Rgb8).unwrap(),
        pixels,
    )
    .unwrap()
    .output(Format::Jpeg, EncodeOptions::default())
    .bytes()
    .unwrap()
}

#[test]
fn every_fit_mode_has_sharps_output_size() {
    let jpeg = photo();
    for (fit, expected) in [
        (Fit::Fill, (200, 200)),
        (Fit::Inside, (200, 150)),
        (Fit::Outside, (267, 200)),
        (Fit::Cover, (200, 200)),
        (Fit::Contain, (200, 200)),
    ] {
        let image = Image::from_stream(std::io::Cursor::new(jpeg.clone()))
            .unwrap()
            .resize_with(200, 200, ResizeOptions::default().with_fit(fit));
        let meta = image.metadata().unwrap();
        assert_eq!((meta.width, meta.height), expected, "{fit:?}");
    }
}

#[test]
fn cover_and_contain_keep_shrink_on_load_and_match_the_reference() {
    let jpeg = photo();
    for fit in [Fit::Cover, Fit::Contain] {
        let pipeline = || {
            Image::from_stream(std::io::Cursor::new(jpeg.clone()))
                .unwrap()
                .resize_with(
                    100,
                    100,
                    ResizeOptions::default()
                        .with_fit(fit)
                        .with_background([255, 0, 0, 255]),
                )
                .output(Format::Raw, EncodeOptions::default())
        };
        let mut scheduled = Vec::new();
        let stats = pipeline().write_with_stats(&mut scheduled).unwrap();
        assert!(
            stats.reduction.is_some(),
            "{fit:?} lost the reduced JPEG decode"
        );
        assert_eq!(
            scheduled,
            pipeline().bytes_via_reference().unwrap(),
            "{fit:?}"
        );
        if fit == Fit::Contain {
            // 4:3 into a square: red bars top and bottom.
            assert_eq!(&scheduled[..3], &[255, 0, 0]);
            assert_eq!(&scheduled[scheduled.len() - 3..], &[255, 0, 0]);
        }
    }
}
