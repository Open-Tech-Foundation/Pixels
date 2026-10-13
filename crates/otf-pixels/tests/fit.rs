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

use otf_pixels::{
    EncodeOptions, Fit, Format, Image, ImageDescriptor, PixelFormat, Position, ResizeOptions,
};

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

/// A 400x200 raw image: black on the left half, white on the right.
fn halves() -> Image {
    let (w, h) = (400_u32, 200_u32);
    let pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| if i % w < w / 2 { [0; 3] } else { [255; 3] })
        .collect();
    Image::from_raw(
        ImageDescriptor::new(w, h, PixelFormat::Rgb8).unwrap(),
        pixels,
    )
    .unwrap()
}

/// The mean of a raw RGB raster, 0 for black and 255 for white.
fn mean(raster: &[u8]) -> u32 {
    (raster.iter().map(|&v| u64::from(v)).sum::<u64>() / raster.len() as u64) as u32
}

#[test]
fn cover_keeps_the_part_its_position_names() {
    // 2:1 into a square keeps a 100x100 window of a 200x100 scaled image.
    let cover = |position| {
        halves()
            .resize_with(
                100,
                100,
                ResizeOptions::default()
                    .with_fit(Fit::Cover)
                    .with_position(position),
            )
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap()
    };
    assert!(
        mean(&cover(Position::West)) < 10,
        "west kept the black half"
    );
    assert!(
        mean(&cover(Position::East)) > 245,
        "east kept the white half"
    );
    let centre = mean(&cover(Position::Centre));
    assert!(
        (100..156).contains(&centre),
        "centre straddles both: {centre}"
    );
}

#[test]
fn contain_places_the_image_where_its_position_says() {
    // 2:1 into a square: 100x50 of image and 50 rows of red padding.
    let raster = halves()
        .resize_with(
            100,
            100,
            ResizeOptions::default()
                .with_fit(Fit::Contain)
                .with_position(Position::North)
                .with_background([255, 0, 0, 255]),
        )
        .output(Format::Raw, EncodeOptions::default())
        .bytes()
        .unwrap();
    let row = |y: usize| &raster[y * 300..(y + 1) * 300];
    assert_eq!(&row(0)[..3], &[0, 0, 0], "the image starts at the top");
    assert!(
        row(99).chunks(3).all(|p| p == [255, 0, 0]),
        "padding at the bottom"
    );
    assert!(
        row(50).chunks(3).all(|p| p == [255, 0, 0]),
        "the bottom half is padding"
    );
}

#[test]
fn every_position_keeps_shrink_on_load_and_matches_the_reference() {
    let jpeg = photo();
    for position in [Position::NorthWest, Position::South, Position::East] {
        for fit in [Fit::Cover, Fit::Contain] {
            let pipeline = || {
                Image::from_stream(std::io::Cursor::new(jpeg.clone()))
                    .unwrap()
                    .resize_with(
                        100,
                        100,
                        ResizeOptions::default()
                            .with_fit(fit)
                            .with_position(position),
                    )
                    .output(Format::Raw, EncodeOptions::default())
            };
            let mut scheduled = Vec::new();
            let stats = pipeline().write_with_stats(&mut scheduled).unwrap();
            assert!(stats.reduction.is_some(), "{fit:?} {position:?}");
            assert_eq!(
                scheduled,
                pipeline().bytes_via_reference().unwrap(),
                "{fit:?} {position:?}"
            );
        }
    }
}

#[test]
fn without_reduction_keeps_a_large_image_its_size() {
    let image = halves().resize_with(
        100,
        100,
        ResizeOptions::default()
            .with_fit(Fit::Inside)
            .without_reduction(true),
    );
    let meta = image.metadata().unwrap();
    assert_eq!((meta.width, meta.height), (400, 200));
}
