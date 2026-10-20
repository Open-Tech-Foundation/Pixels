//! sharp's `metadata()`, `stats()`, `keepXmp`/`withXmp`, `keepMetadata` and
//! `withMetadata({ density })`: what a file says about itself, read without
//! decoding, and carried into outputs that ask for it.

#![cfg(all(
    feature = "raw",
    feature = "png",
    feature = "jpeg",
    feature = "webp",
    feature = "avif",
    feature = "gif",
    feature = "tiff",
    feature = "jxl"
))]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::float_cmp,
    clippy::panic,
    reason = "tests operate on known-good values"
)]

use otf_pixels::{
    Background, Density, EncodeOptions, Format, Image, ImageDescriptor, Orientation, PixelFormat,
    ResolutionUnit,
};

const XMP: &[u8] = b"<?xpacket begin=\"\xef\xbb\xbf\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\
<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\
<rdf:Description xmlns:dc=\"http://purl.org/dc/elements/1.1/\" dc:creator=\"Pixels\"/></rdf:RDF></x:xmpmeta>\
<?xpacket end=\"w\"?>";

/// A little-endian EXIF block: IFD0 with Make "Pixels" and Orientation 1,
/// and an Exif IFD with ExposureTime 1/250.
fn exif_block() -> Vec<u8> {
    let mut b = b"II*\0\x08\0\0\0".to_vec();
    // IFD0 at 8: three entries, then the next-IFD link: 8 + 2 + 36 + 4 = 50.
    b.extend_from_slice(&3_u16.to_le_bytes());
    b.extend_from_slice(&[0x0F, 0x01, 2, 0]);
    b.extend_from_slice(&7_u32.to_le_bytes());
    b.extend_from_slice(&50_u32.to_le_bytes());
    b.extend_from_slice(&[0x12, 0x01, 3, 0]);
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&[1, 0, 0, 0]);
    b.extend_from_slice(&[0x69, 0x87, 4, 0]);
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&58_u32.to_le_bytes());
    b.extend_from_slice(&0_u32.to_le_bytes());
    b.extend_from_slice(b"Pixels\0\0"); // 50..58, padded
    // Exif IFD at 58: one RATIONAL at 58 + 2 + 12 + 4 = 76.
    b.extend_from_slice(&1_u16.to_le_bytes());
    b.extend_from_slice(&[0x9A, 0x82, 5, 0]);
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&76_u32.to_le_bytes());
    b.extend_from_slice(&0_u32.to_le_bytes());
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&250_u32.to_le_bytes());
    b
}

fn image(pixel: PixelFormat) -> Image {
    let descriptor = ImageDescriptor::new(40, 30, pixel).unwrap();
    let bytes = (0..descriptor.byte_len().unwrap())
        .map(|i| ((i / 3) * 7 % 200 + i % 3 * 20) as u8)
        .collect();
    Image::from_raw(descriptor, bytes).unwrap()
}

fn reopen(bytes: Vec<u8>) -> Image {
    Image::from_stream(std::io::Cursor::new(bytes)).unwrap()
}

/// The Make and ExposureTime an EXIF block holds, wherever its byte order
/// and layout put them.
fn make_and_exposure(exif: &[u8]) -> (Vec<u8>, Option<Vec<u8>>) {
    let tree = otf_pixels_core::exif::Exif::parse(exif)
        .unwrap()
        .with_byte_order(false);
    let make = tree.get(0x010F).unwrap().data.clone();
    let exposure = tree
        .get(0x8769)
        .and_then(|e| e.sub.as_ref())
        .and_then(|sub| sub.iter().find(|e| e.tag == 0x829A))
        .map(|e| e.data.clone());
    (make, exposure)
}

#[test]
fn xmp_exif_and_density_round_trip_through_every_format_that_holds_them() {
    let formats = [
        Format::Png,
        Format::Jpeg,
        Format::WebP,
        Format::Avif,
        Format::Tiff,
        Format::Jxl,
    ];
    for format in formats {
        let out = image(PixelFormat::Rgb8)
            .with_exif(Some(exif_block()))
            .with_xmp(Some(XMP.to_vec()))
            .with_density(Some(Density::dpi(300.0)))
            .output(format, EncodeOptions::default().with_keep_metadata(true))
            .bytes()
            .unwrap();
        let back = reopen(out);
        let meta = back.source_metadata().unwrap();
        assert_eq!(meta.format, format);
        assert_eq!(meta.xmp.as_deref(), Some(XMP), "{format}: XMP");
        assert_eq!(back.xmp(), Some(XMP), "{format}: carried");
        let exif = meta
            .exif
            .as_deref()
            .unwrap_or_else(|| panic!("{format}: no EXIF"));
        let (make, exposure) = make_and_exposure(exif);
        assert_eq!(make, b"Pixels\0", "{format}: Make");
        assert_eq!(
            exposure.unwrap(),
            [1, 0, 0, 0, 250, 0, 0, 0],
            "{format}: ExposureTime"
        );
        if matches!(format, Format::Png | Format::Jpeg | Format::Tiff) {
            let density = meta
                .density
                .unwrap_or_else(|| panic!("{format}: no density"));
            assert!(
                (density.x_dpi() - 300.0).abs() < 0.05,
                "{format}: {density:?}"
            );
            assert!(
                (density.y_dpi() - 300.0).abs() < 0.05,
                "{format}: {density:?}"
            );
        }
    }
}

#[test]
fn outputs_drop_xmp_and_exif_unless_kept() {
    for format in [
        Format::Png,
        Format::Jpeg,
        Format::WebP,
        Format::Avif,
        Format::Tiff,
        Format::Jxl,
    ] {
        let marked = image(PixelFormat::Rgb8)
            .with_exif(Some(exif_block()))
            .with_xmp(Some(XMP.to_vec()));
        let plain = reopen(
            marked
                .clone()
                .output(format, EncodeOptions::default())
                .bytes()
                .unwrap(),
        );
        let meta = plain.source_metadata().unwrap();
        assert!(
            meta.xmp.is_none() && meta.exif.is_none(),
            "{format}: stripped by default"
        );
        let xmp_only = reopen(
            marked
                .output(format, EncodeOptions::default().with_keep_xmp(true))
                .bytes()
                .unwrap(),
        );
        let meta = xmp_only.source_metadata().unwrap();
        assert_eq!(meta.xmp.as_deref(), Some(XMP), "{format}");
        assert!(meta.exif.is_none(), "{format}: EXIF not asked for");
    }
}

#[test]
fn the_source_density_is_kept_with_the_metadata_and_only_then() {
    let png = image(PixelFormat::Rgb8)
        .with_density(Some(Density {
            x: 72.0,
            y: 144.0,
            unit: ResolutionUnit::Inch,
        }))
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    let source = reopen(png);
    let d = source.source_metadata().unwrap().density.unwrap();
    assert_eq!(d.unit, ResolutionUnit::Centimetre);
    assert!(
        (d.x_dpi() - 72.0).abs() < 0.05 && (d.y_dpi() - 144.0).abs() < 0.05,
        "{d:?}"
    );
    let dropped = reopen(
        source
            .clone()
            .output(Format::Jpeg, EncodeOptions::default())
            .bytes()
            .unwrap(),
    );
    assert!(dropped.source_metadata().unwrap().density.is_none());
    let kept = reopen(
        source
            .output(
                Format::Jpeg,
                EncodeOptions::default().with_keep_metadata(true),
            )
            .bytes()
            .unwrap(),
    );
    // JFIF holds whole pixels per centimetre: 28 and 57.
    let d = kept.source_metadata().unwrap().density.unwrap();
    assert_eq!((d.x, d.y, d.unit), (28.0, 57.0, ResolutionUnit::Centimetre));
}

#[test]
fn each_format_reports_its_own_facts() {
    let rgb = image(PixelFormat::Rgb8);
    let facts = |format: Format, options: EncodeOptions| {
        let back = reopen(rgb.clone().output(format, options).bytes().unwrap());
        back.source_metadata().unwrap().clone()
    };
    let jpeg = facts(Format::Jpeg, EncodeOptions::default());
    assert_eq!(jpeg.chroma_subsampling, Some("4:2:0"));
    assert!(!jpeg.progressive);
    assert_eq!(jpeg.bits_per_sample, Some(8));
    let jpeg = facts(
        Format::Jpeg,
        EncodeOptions::default()
            .with_progressive(true)
            .with_chroma_subsampling(otf_pixels::ChromaSubsampling::Yuv444),
    );
    assert!(jpeg.progressive);
    assert_eq!(jpeg.chroma_subsampling, Some("4:4:4"));

    let png = facts(Format::Png, EncodeOptions::default().with_palette(true));
    assert!(png.palette);
    assert_eq!(
        (png.width, png.height, png.channels, png.has_alpha),
        (40, 30, 3, false)
    );
    assert!(!facts(Format::Png, EncodeOptions::default()).palette);

    let gif = facts(Format::Gif, EncodeOptions::default());
    assert!(gif.palette && !gif.progressive);
    assert!(matches!(gif.background, Some(Background::Rgb(..))));

    assert_eq!(
        facts(Format::Tiff, EncodeOptions::default()).compression,
        Some("deflate")
    );
    let avif = facts(Format::Avif, EncodeOptions::default());
    assert_eq!(
        (
            avif.compression,
            avif.bits_per_sample,
            avif.chroma_subsampling
        ),
        (Some("av1"), Some(8), Some("4:2:0"))
    );
    let webp = facts(Format::WebP, EncodeOptions::default().with_lossless(true));
    assert_eq!(webp.compression, Some("lossless"));
    let jxl = facts(Format::Jxl, EncodeOptions::default().with_lossless(true));
    assert_eq!(
        (jxl.compression, jxl.bits_per_sample),
        (Some("lossless"), Some(8))
    );

    let rgba = reopen(
        image(PixelFormat::Rgba8)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap(),
    );
    let meta = rgba.source_metadata().unwrap();
    assert!(meta.has_alpha && meta.channels == 4 && meta.pages == 1);
    assert!(meta.animation.is_none() && !meta.has_profile());
}

/// A PNG chunk: length, type, payload, CRC-32.
fn chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut crc = 0xFFFF_FFFF_u32;
    for &b in kind.iter().chain(payload) {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    let mut out = (payload.len() as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out.extend_from_slice(&(!crc).to_be_bytes());
    out
}

#[test]
fn text_chunks_and_comments_are_reported() {
    let mut png = image(PixelFormat::Rgb8)
        .output(Format::Png, EncodeOptions::default())
        .bytes()
        .unwrap();
    // After the signature and IHDR (8 + 25 bytes).
    let extra = [
        chunk(b"tEXt", b"Title\0Harbour at dawn"),
        chunk(b"iTXt", "Author\0\0\0en\0Auteur\0Zoë".as_bytes()),
        chunk(b"bKGD", &[0, 10, 0, 20, 0, 30]),
    ]
    .concat();
    png.splice(33..33, extra);
    let meta = reopen(png).source_metadata().unwrap().clone();
    let pairs: Vec<(&str, &str)> = meta
        .comments
        .iter()
        .map(|c| (c.keyword.as_str(), c.text.as_str()))
        .collect();
    assert_eq!(pairs, [("Title", "Harbour at dawn"), ("Author", "Zoë")]);
    assert_eq!(meta.background, Some(Background::Rgb(10, 20, 30)));

    let mut jpeg = image(PixelFormat::Rgb8)
        .output(Format::Jpeg, EncodeOptions::default())
        .bytes()
        .unwrap();
    let text = b"made by Pixels";
    let mut com = vec![0xFF, 0xFE];
    com.extend_from_slice(&((text.len() + 2) as u16).to_be_bytes());
    com.extend_from_slice(text);
    jpeg.splice(2..2, com);
    let meta = reopen(jpeg).source_metadata().unwrap().clone();
    assert_eq!(meta.comments.len(), 1);
    assert_eq!(meta.comments[0].text, "made by Pixels");
}

#[test]
fn a_file_opened_from_a_path_reports_its_size_and_orientation() {
    let path = format!(
        "{}/tests/fixtures/orientation/exif6.jpg",
        env!("CARGO_MANIFEST_DIR")
    );
    let len = std::fs::metadata(&path).unwrap().len();
    let image = Image::open_with(
        &path,
        otf_pixels::OpenOptions::default().with_auto_orient(false),
    )
    .unwrap();
    let meta = image.source_metadata().unwrap();
    assert_eq!(meta.size, Some(len));
    assert_eq!(meta.orientation, Orientation::from_exif(6).unwrap());
    assert_eq!(
        (meta.oriented_width(), meta.oriented_height()),
        (meta.height, meta.width)
    );
    assert!(meta.exif.is_some());

    let icc = format!(
        "{}/tests/fixtures/icc/display_p3.png",
        env!("CARGO_MANIFEST_DIR")
    );
    assert!(
        Image::open(&icc)
            .unwrap()
            .source_metadata()
            .unwrap()
            .has_profile()
    );
    // A built image has no source file.
    assert!(image_without_source().source_metadata().is_none());
}

fn image_without_source() -> Image {
    image(PixelFormat::Gray8)
}

#[test]
fn stats_measure_the_pipeline_result() {
    let descriptor = ImageDescriptor::new(4, 2, PixelFormat::Rgba8).unwrap();
    let mut bytes = Vec::new();
    for i in 0..8_u8 {
        bytes.extend_from_slice(&[i * 10, 200, 0, if i == 5 { 0 } else { 255 }]);
    }
    let image = Image::from_raw(descriptor, bytes).unwrap();
    let stats = image.stats().unwrap();
    let r = stats.channels[0];
    assert_eq!((r.min, r.max, r.mean), (0.0, 70.0, 35.0));
    assert_eq!((r.max_x, r.max_y), (3, 1));
    assert!(!stats.is_opaque);
    assert_eq!(stats.channels[3].min_x, 1);
    // After a crop to the opaque left half, it is opaque.
    let left = image.crop(0, 0, 1, 2).stats().unwrap();
    assert!(left.is_opaque);
    assert_eq!(left.channels[0].max, 40.0);
    assert_eq!(left.dominant[1], 200);
}

/// Files Pillow wrote (`scripts/regenerate-metadata-fixtures.py`): another
/// implementation's XMP, EXIF, density and text read back.
#[test]
fn pillows_metadata_reads_back() {
    let fixture = |ext: &str| {
        let path = format!(
            "{}/tests/fixtures/metadata/pillow.{ext}",
            env!("CARGO_MANIFEST_DIR")
        );
        Image::open_with(
            path,
            otf_pixels::OpenOptions::default().with_auto_orient(false),
        )
        .unwrap()
    };
    for ext in ["jpg", "png", "webp", "tif"] {
        let image = fixture(ext);
        let meta = image.source_metadata().unwrap();
        let xmp = meta
            .xmp
            .as_deref()
            .unwrap_or_else(|| panic!("{ext}: no XMP"));
        assert!(xmp.windows(17).any(|w| w == b"dc:creator=\"Pillo"), "{ext}");
        let exif = meta
            .exif
            .as_deref()
            .unwrap_or_else(|| panic!("{ext}: no EXIF"));
        let tree = otf_pixels_core::exif::Exif::parse(exif).unwrap();
        assert_eq!(tree.get(0x010F).unwrap().data, b"Pillow\0", "{ext}");
        assert_eq!(
            meta.orientation,
            Orientation::from_exif(3).unwrap(),
            "{ext}"
        );
        if ext != "webp" {
            let d = meta.density.unwrap_or_else(|| panic!("{ext}: no density"));
            assert!((d.x_dpi() - 240.0).abs() < 0.05, "{ext}: {d:?}");
        }
        if ext != "tif" {
            assert!(make_and_exposure(exif).1.is_some(), "{ext}: Exif IFD");
        }
    }
    assert!(fixture("jpg").source_metadata().unwrap().progressive);
    let png = fixture("png");
    let comments = &png.source_metadata().unwrap().comments;
    assert_eq!(comments[0].keyword, "Title");
    assert_eq!(comments[0].text, "Harbour");
    assert_eq!(comments[1].text, "zipped ".repeat(20), "zTXt");
    assert_eq!(
        fixture("tif").source_metadata().unwrap().compression,
        Some("lzw")
    );
}
