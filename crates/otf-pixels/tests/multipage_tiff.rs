//! Multi-page TIFF through the public facade: counting pages, choosing one,
//! and refusing a page that does not exist.
//!
//! The fixture is the TIFF codec's: three pages that differ in size and colour
//! type, written by libtiff (`scripts/generate-tiff-multipage-fixture.py`).
//! The codec's reference test checks each page's pixels against libtiff; this
//! checks that the facade asks for the right one.

#![cfg(all(feature = "tiff", feature = "png"))]
#![allow(clippy::unwrap_used, reason = "tests operate on known-good values")]

use otf_pixels::{EncodeOptions, ErrorCode, Format, Image, OpenOptions};

fn fixture() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../otf-pixels-codec-tiff/tests/fixtures/multipage.tif")
}

fn size(image: &Image) -> (u32, u32) {
    let metadata = image.metadata().unwrap();
    (metadata.width, metadata.height)
}

#[test]
fn opening_counts_the_pages_and_takes_the_first() {
    let image = Image::open(fixture()).unwrap();
    assert_eq!(image.pages(), 3);
    assert_eq!(size(&image), (48, 32));
}

#[test]
fn a_chosen_page_is_the_one_processed() {
    for (page, expected) in [(0, (48, 32)), (1, (24, 40)), (2, (16, 16))] {
        let image = Image::open_with(fixture(), OpenOptions::default().with_page(page)).unwrap();
        assert_eq!(size(&image), expected, "page {page}");
        // The count survives the pipeline, like `animation`.
        let resized = image.resize(8, 8);
        assert_eq!(resized.pages(), 3);
        let png = resized
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert_eq!(
            Image::from_stream(std::io::Cursor::new(png))
                .unwrap()
                .pages(),
            1
        );
    }
}

#[test]
fn a_page_that_does_not_exist_is_an_invalid_argument() {
    let past_the_end = Image::open_with(fixture(), OpenOptions::default().with_page(3));
    assert_eq!(past_the_end.unwrap_err().code(), ErrorCode::InvalidArgument);

    // Every other format is a document of one page.
    let png = Image::open(fixture())
        .unwrap()
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let page_one = Image::from_stream_with(
        std::io::Cursor::new(png),
        OpenOptions::default().with_page(1),
    );
    assert_eq!(page_one.unwrap_err().code(), ErrorCode::InvalidArgument);
}
