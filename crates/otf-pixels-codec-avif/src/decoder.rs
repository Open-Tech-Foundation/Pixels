//! The AVIF decoder: container parsing and the [`Decoder`] implementation.
//!
//! [`AvifDecoder::new`] parses only boxes, so `metadata()` stays a no-decode
//! operation (SPEC §Guarantees 3): it answers dimensions, bit depth, chroma
//! format, and alpha without touching the bitstream. [`AvifDecoder::read_row`]
//! reconstructs the whole primary frame on the first call and serves rows from
//! the cache — an AVIF still is one AV1 key frame with no prefix that yields a
//! partial raster, so the decode is inherently whole-image (SPEC §Memory).
//!
//! The reconstruction covers the lossless 4:4:4 intra subset; anything outside
//! it (lossy transforms, subsampled chroma, screen-content tools, grids) is
//! reported as [`PixelsError::Unsupported`] rather than decoded wrong.

use crate::boxes::{FourCc, Reader};
use crate::meta::Meta;
use crate::props::{Av1Config, Subsampling};
use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Format, ImageDescriptor, Limits, PixelFormat, PixelsError,
    Result, Source,
};

/// The most container bytes read before a file is called hostile.
///
/// The container addresses its payload by absolute file offset, so the whole
/// file must be resident before any of it can be interpreted and there is no
/// bound derivable from the image dimensions: a small `meta` can be followed
/// by an unbounded `mdat`. `max_pixels` bounds the output; this bounds the
/// input, exactly as the WebP decoder does for the same reason.
const MAX_CONTAINER: usize = 256 * 1024 * 1024;

/// What the container says about the primary image.
///
/// Everything here comes from boxes, so it is available without touching the
/// AV1 bitstream.
#[derive(Debug, Clone)]
pub struct AvifInfo {
    /// The primary item's dimensions, from its `ispe`.
    pub width: u32,
    /// See [`AvifInfo::width`].
    pub height: u32,
    /// The AV1 configuration of the primary item, or of its first tile when
    /// the primary item is a grid.
    pub config: Av1Config,
    /// Whether the file carries an alpha plane as an auxiliary item.
    pub has_alpha: bool,
    /// Whether the primary item is a derived grid rather than a single coded
    /// image.
    pub is_grid: bool,
}

/// Decodes an AVIF stream.
#[derive(Debug)]
pub struct AvifDecoder {
    descriptor: ImageDescriptor,
    info: AvifInfo,
    /// The primary item's coded AV1 bytes, retained for the lazy pixel decode.
    /// `None` for a grid, whose per-tile decode is not implemented.
    frame_data: Option<Vec<u8>>,
    /// The reconstructed interleaved raster, produced on the first row read.
    raster: Option<Vec<u8>>,
    /// Rows already served.
    row: u32,
}

impl AvifDecoder {
    /// Read the container and describe the primary image.
    ///
    /// Decodes no pixels: this parses boxes only, which is what makes
    /// `metadata()` free.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a stream that is not a
    /// structurally valid AVIF, [`PixelsError::Unsupported`] for a valid file
    /// using a feature this decoder does not implement, or
    /// [`PixelsError::LimitExceeded`] if the image exceeds `limits`.
    pub fn new<S: Source>(source: S, limits: Limits) -> Result<Self> {
        let bytes = read_all(source)?;
        let info = parse_container(&bytes)?;
        let frame_data = if info.is_grid {
            None
        } else {
            Some(locate_primary_frame(&bytes)?)
        };
        let pixel = pixel_format(&info);
        let descriptor = ImageDescriptor::with_limits(info.width, info.height, pixel, &limits)?;
        Ok(Self {
            descriptor,
            info,
            frame_data,
            raster: None,
            row: 0,
        })
    }

    /// What the container said about this image.
    #[must_use]
    pub const fn info(&self) -> &AvifInfo {
        &self.info
    }
}

/// Drain `source` into a bounded buffer.
fn read_all<S: Source>(mut source: S) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        if bytes.len() > MAX_CONTAINER {
            return Err(PixelsError::malformed(
                "avif",
                format!("stream exceeds {MAX_CONTAINER} bytes"),
            ));
        }
        match source.read(&mut chunk)? {
            0 => break,
            read => {
                let Some(filled) = chunk.get(..read) else {
                    break;
                };
                bytes.extend_from_slice(filled);
            }
        }
    }
    Ok(bytes)
}

/// Parse the container down to the facts about the primary image.
fn parse_container(bytes: &[u8]) -> Result<AvifInfo> {
    let mut reader = Reader::new(bytes);

    // The first box must be `ftyp`. Checking it here rather than trusting
    // sniffing means a decoder handed bytes directly still validates them.
    let first = reader
        .next_box()
        .ok_or_else(|| PixelsError::malformed("avif", "the file holds no boxes"))??;
    if first.kind != FourCc::new(b"ftyp") {
        return Err(PixelsError::malformed(
            "avif",
            format!("the file begins with '{}' rather than 'ftyp'", first.kind),
        ));
    }
    if !probe(bytes) {
        return Err(PixelsError::malformed(
            "avif",
            "the ftyp box declares no brand this decoder recognises",
        ));
    }

    let meta_reader = reader
        .find(b"meta")?
        .ok_or_else(|| PixelsError::malformed("avif", "the file has no meta box"))?;
    let meta = Meta::parse(meta_reader)?;

    let primary = meta
        .primary_item()
        .ok_or_else(|| PixelsError::malformed("avif", "the file names no primary image item"))?;

    // An essential property we cannot interpret was declared to change how the
    // pixels are to be read, so decoding anyway would produce a confidently
    // wrong image.
    if let Some(kind) = meta.properties.essential_unknown(primary.id) {
        return Err(PixelsError::unsupported(format!(
            "avif: the primary item requires the '{kind}' property, which this decoder does not implement"
        )));
    }

    if !primary.is_coded_image() && !primary.is_grid() {
        return Err(PixelsError::unsupported(format!(
            "avif: the primary item has type '{}', which is not an image this decoder produces",
            primary.kind
        )));
    }

    let extents = meta.properties.extents(primary.id).ok_or_else(|| {
        PixelsError::malformed(
            "avif",
            format!(
                "item {} has no ispe property, so its size is unknown",
                primary.id
            ),
        )
    })?;

    // A grid's own configuration lives on its tiles: the grid item has no
    // coded data of its own, only a dimg reference to the items that do.
    let config_item = if primary.is_grid() {
        *meta
            .referenced(primary.id, b"dimg")
            .first()
            .ok_or_else(|| {
                PixelsError::malformed(
                    "avif",
                    format!("grid item {} references no tiles", primary.id),
                )
            })?
    } else {
        primary.id
    };

    let config = meta
        .properties
        .av1_config(config_item)
        .ok_or_else(|| {
            PixelsError::malformed(
                "avif",
                format!("item {config_item} has no av1C property, so it is not a coded AV1 image"),
            )
        })?
        .clone();

    Ok(AvifInfo {
        width: extents.width,
        height: extents.height,
        config,
        has_alpha: meta.alpha_item(primary.id).is_some(),
        is_grid: primary.is_grid(),
    })
}

/// Locate and copy out the primary coded image item's AV1 bytes.
fn locate_primary_frame(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut reader = Reader::new(bytes);
    reader.next_box();
    let meta_reader = reader
        .find(b"meta")?
        .ok_or_else(|| PixelsError::malformed("avif", "the file has no meta box"))?;
    let meta = Meta::parse(meta_reader)?;
    let primary = meta
        .primary_item()
        .ok_or_else(|| PixelsError::malformed("avif", "the file names no primary image item"))?;
    Ok(meta.item_data(bytes, primary)?.into_owned())
}

/// Decode the primary AV1 still and convert it to the interleaved raster the
/// descriptor promises.
fn decode_raster(info: &AvifInfo, frame_data: &[u8]) -> Result<Vec<u8>> {
    use crate::av1::{StillPicture, decode_still};

    let still = StillPicture::parse(&info.config.config_obus, frame_data)?;
    let located = frame_data
        .get(still.tile_data_offset..still.tile_data_offset + still.tile_data_len)
        .ok_or_else(|| PixelsError::malformed("avif", "tile data runs past the coded frame"))?;
    let frame = decode_still(&still.sequence, &still.frame, located)?;

    let width = info.width as usize;
    let height = info.height as usize;
    let matrix = still.sequence.color.matrix_coefficients;
    if matrix != 0 {
        return Err(PixelsError::unsupported(
            "avif: only the identity colour matrix is implemented in the lossless path",
        ));
    }
    let plane = |i: usize| {
        frame
            .planes
            .get(i)
            .ok_or_else(|| PixelsError::malformed("avif", "a colour plane is missing"))
    };
    // Identity matrix (matrix_coefficients == 0): the planes are G, B, R.
    let (g, b, r) = (plane(0)?, plane(1)?, plane(2)?);
    let mut raster = vec![0_u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * 3;
            if let Some(px) = raster.get_mut(base..base + 3) {
                px.copy_from_slice(&[
                    r.get(x, y).unwrap_or(0) as u8,
                    g.get(x, y).unwrap_or(0) as u8,
                    b.get(x, y).unwrap_or(0) as u8,
                ]);
            }
        }
    }
    Ok(raster)
}

/// The pixel format this image decodes to.
///
/// AV1 codes YUV; the engine's formats are RGB and greyscale, so the mapping
/// happens here and the colour conversion happens at decode. Ten- and
/// twelve-bit samples widen to sixteen because SPEC §Pixel formats has no
/// narrower wide type — the sample values keep their original range and are
/// not rescaled by this choice.
fn pixel_format(info: &AvifInfo) -> PixelFormat {
    let wide = info.config.bit_depth > 8;
    let monochrome = info.config.subsampling == Subsampling::Monochrome;
    match (monochrome, info.has_alpha, wide) {
        (true, false, false) => PixelFormat::Gray8,
        (true, false, true) => PixelFormat::Gray16,
        (true, true, false) => PixelFormat::GrayA8,
        // SPEC §Pixel formats has no GrayA16, so wide greyscale with alpha
        // widens to RGBA rather than silently dropping either the alpha or
        // the low bits.
        (true, true, true) => PixelFormat::Rgba16,
        (false, false, false) => PixelFormat::Rgb8,
        (false, false, true) => PixelFormat::Rgb16,
        (false, true, false) => PixelFormat::Rgba8,
        (false, true, true) => PixelFormat::Rgba16,
    }
}

impl Decoder for AvifDecoder {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        // A grid AVIF stores independently coded tiles and could serve
        // regions, but this decoder does not yet implement `read_region`.
        // Claiming `Regions` before it does would be a lie the scheduler acts
        // on, which is the mistake `codec.rs` records having made once for
        // JPEG's scaled decode.
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        // Validate the call even though it cannot yet be served, so that the
        // contract is already enforced when the bitstream decoder lands.
        if self.row >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("all {} rows have already been read", self.descriptor.height),
            ));
        }
        let row_bytes = self.descriptor.row_bytes();
        if out.len() != row_bytes {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {row_bytes}", out.len()),
            ));
        }

        // Reconstruct the whole frame on the first row: an AVIF still is coded
        // as one AV1 frame with no prefix that yields a partial raster, so the
        // decode is inherently whole-image (SPEC §Memory). Later rows are served
        // from the cached raster.
        if self.raster.is_none() {
            let frame_data = self
                .frame_data
                .as_deref()
                .ok_or_else(|| PixelsError::unsupported("avif: grid images are not decoded yet"))?;
            self.raster = Some(decode_raster(&self.info, frame_data)?);
        }
        let raster = self.raster.as_deref().unwrap_or(&[]);
        let start = self.row as usize * row_bytes;
        let src = raster
            .get(start..start + row_bytes)
            .ok_or_else(|| PixelsError::malformed("avif", "the reconstructed raster is short"))?;
        out.copy_from_slice(src);
        self.row += 1;
        Ok(())
    }
}

/// Whether `prefix` starts with an ISOBMFF file declaring a brand this
/// decoder recognises.
///
/// Detection is by magic bytes only (SPEC §Formats). `ftyp` at offset 4 marks
/// the whole ISOBMFF family, which also holds MP4 and HEIC, so the brands are
/// what actually identify an AVIF. The major brand alone is not enough: many
/// encoders write `mif1` as the major brand and put `avif` in the compatible
/// list, so both are scanned.
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    if prefix.get(4..8) != Some(&crate::SIGNATURE_FTYP[..]) {
        return false;
    }
    let Some(brands) = prefix.get(8..) else {
        return false;
    };
    // Words from offset 8: index 0 is the major brand, index 1 is the minor
    // version — a number, not a brand — and the rest are compatible brands.
    brands
        .chunks_exact(4)
        .enumerate()
        .filter(|(index, _)| *index != 1)
        .any(|(_, brand)| is_known_brand(brand))
}

/// Whether these four bytes name a brand this decoder claims.
fn is_known_brand(brand: &[u8]) -> bool {
    crate::BRANDS_STILL
        .iter()
        .chain(core::iter::once(&crate::BRAND_SEQUENCE))
        .any(|known| known == brand)
}

/// The AVIF entry in a sniffing registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct AvifCodec;

impl Codec for AvifCodec {
    fn format(&self) -> Format {
        Format::Avif
    }

    fn magic_len(&self) -> usize {
        // Enough for `ftyp`, the major brand, the minor version and four
        // compatible brands, which covers every file this has been tried on.
        // `probe` reads whatever it is given, so a shorter prefix still works
        // when the brand appears early.
        32
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::ErrorCode;

    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let total = u32::try_from(8 + payload.len()).unwrap();
        out.extend_from_slice(&total.to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    fn ftyp(major: &[u8; 4], compatible: &[&[u8; 4]]) -> Vec<u8> {
        let mut payload = Vec::from(*major);
        payload.extend_from_slice(&0_u32.to_be_bytes());
        for brand in compatible {
            payload.extend_from_slice(*brand);
        }
        boxed(b"ftyp", &payload)
    }

    /// An `av1C` for 8-bit 4:2:0, profile 0.
    fn av1c() -> Vec<u8> {
        boxed(b"av1C", &[0x81, 0x00, 0x0C, 0x00])
    }

    fn ispe(width: u32, height: u32) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&width.to_be_bytes());
        payload.extend_from_slice(&height.to_be_bytes());
        boxed(b"ispe", &payload)
    }

    fn infe(id: u16, kind: &[u8; 4], hidden: bool) -> Vec<u8> {
        let mut payload = vec![2, 0, 0, u8::from(hidden)];
        payload.extend_from_slice(&id.to_be_bytes());
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(kind);
        payload.push(0);
        boxed(b"infe", &payload)
    }

    fn iinf(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&u16::try_from(entries.len()).unwrap().to_be_bytes());
        for entry in entries {
            payload.extend_from_slice(entry);
        }
        boxed(b"iinf", &payload)
    }

    fn iloc(rows: &[(u16, u32, u32)]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0, 0x44, 0x00];
        payload.extend_from_slice(&u16::try_from(rows.len()).unwrap().to_be_bytes());
        for (id, offset, length) in rows {
            payload.extend_from_slice(&id.to_be_bytes());
            payload.extend_from_slice(&[0, 0]);
            payload.extend_from_slice(&1_u16.to_be_bytes());
            payload.extend_from_slice(&offset.to_be_bytes());
            payload.extend_from_slice(&length.to_be_bytes());
        }
        boxed(b"iloc", &payload)
    }

    fn pitm(id: u16) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&id.to_be_bytes());
        boxed(b"pitm", &payload)
    }

    /// An `ipma` associating each `(item, [properties])` pair, all
    /// non-essential.
    fn ipma(rows: &[(u16, &[u8])]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&u32::try_from(rows.len()).unwrap().to_be_bytes());
        for (item, properties) in rows {
            payload.extend_from_slice(&item.to_be_bytes());
            payload.push(u8::try_from(properties.len()).unwrap());
            payload.extend_from_slice(properties);
        }
        boxed(b"ipma", &payload)
    }

    fn iprp(properties: &[Vec<u8>], associations: &[(u16, &[u8])]) -> Vec<u8> {
        let mut ipco = Vec::new();
        for property in properties {
            ipco.extend_from_slice(property);
        }
        let mut payload = boxed(b"ipco", &ipco);
        payload.extend_from_slice(&ipma(associations));
        boxed(b"iprp", &payload)
    }

    fn meta_box(children: &[Vec<u8>]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        for child in children {
            payload.extend_from_slice(child);
        }
        boxed(b"meta", &payload)
    }

    /// A minimal single-item AVIF: one `av01` item, `ispe` then `av1C`.
    fn minimal_avif(width: u32, height: u32) -> Vec<u8> {
        let mut file = ftyp(b"avif", &[b"mif1"]);
        file.extend_from_slice(&meta_box(&[
            pitm(1),
            iinf(&[infe(1, b"av01", false)]),
            iloc(&[(1, 0, 1)]),
            iprp(&[ispe(width, height), av1c()], &[(1, &[1, 2])]),
        ]));
        file.extend_from_slice(&boxed(b"mdat", &[0]));
        file
    }

    #[test]
    fn reads_dimensions_without_decoding_pixels() {
        let file = minimal_avif(320, 240);
        let decoder = AvifDecoder::new(&file[..], Limits::default()).unwrap();
        let descriptor = decoder.descriptor();
        assert_eq!((descriptor.width, descriptor.height), (320, 240));
        assert_eq!(descriptor.pixel, PixelFormat::Rgb8);
        assert_eq!(decoder.info().config.bit_depth, 8);
        assert_eq!(decoder.info().config.subsampling, Subsampling::Yuv420);
        assert!(!decoder.info().has_alpha);
        assert!(!decoder.info().is_grid);
    }

    #[test]
    fn an_alpha_auxiliary_item_widens_the_pixel_format() {
        let mut auxl = Vec::new();
        auxl.extend_from_slice(&2_u16.to_be_bytes()); // from the alpha item
        auxl.extend_from_slice(&1_u16.to_be_bytes());
        auxl.extend_from_slice(&1_u16.to_be_bytes()); // to the colour item
        let mut iref_payload = vec![0, 0, 0, 0];
        iref_payload.extend_from_slice(&boxed(b"auxl", &auxl));

        let mut auxc_payload = vec![0, 0, 0, 0];
        auxc_payload.extend_from_slice(crate::meta::URN_ALPHA.as_bytes());
        auxc_payload.push(0);

        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&meta_box(&[
            pitm(1),
            iinf(&[infe(1, b"av01", false), infe(2, b"av01", true)]),
            iloc(&[(1, 0, 1), (2, 0, 1)]),
            boxed(b"iref", &iref_payload),
            iprp(
                &[ispe(8, 8), av1c(), boxed(b"auxC", &auxc_payload)],
                &[(1, &[1, 2]), (2, &[1, 2, 3])],
            ),
        ]));

        let decoder = AvifDecoder::new(&file[..], Limits::default()).unwrap();
        assert!(decoder.info().has_alpha);
        assert_eq!(decoder.descriptor().pixel, PixelFormat::Rgba8);
    }

    #[test]
    fn a_grid_takes_its_configuration_from_its_first_tile() {
        let mut dimg = Vec::new();
        dimg.extend_from_slice(&1_u16.to_be_bytes()); // from the grid
        dimg.extend_from_slice(&2_u16.to_be_bytes()); // two tiles
        dimg.extend_from_slice(&2_u16.to_be_bytes());
        dimg.extend_from_slice(&3_u16.to_be_bytes());
        let mut iref_payload = vec![0, 0, 0, 0];
        iref_payload.extend_from_slice(&boxed(b"dimg", &dimg));

        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&meta_box(&[
            pitm(1),
            iinf(&[
                infe(1, b"grid", false),
                infe(2, b"av01", true),
                infe(3, b"av01", true),
            ]),
            iloc(&[(1, 0, 1), (2, 0, 1), (3, 0, 1)]),
            boxed(b"iref", &iref_payload),
            // The grid carries the full size; the tiles carry the av1C.
            iprp(
                &[ispe(128, 64), av1c(), ispe(64, 64)],
                &[(1, &[1]), (2, &[3, 2]), (3, &[3, 2])],
            ),
        ]));

        let decoder = AvifDecoder::new(&file[..], Limits::default()).unwrap();
        assert!(decoder.info().is_grid);
        assert_eq!(
            (decoder.descriptor().width, decoder.descriptor().height),
            (128, 64)
        );
        assert_eq!(decoder.info().config.bit_depth, 8);
    }

    #[test]
    fn monochrome_and_wide_samples_choose_the_right_pixel_format() {
        fn format_for(av1c_bytes: [u8; 4], has_alpha: bool) -> PixelFormat {
            let info = AvifInfo {
                width: 1,
                height: 1,
                config: Av1Config {
                    seq_profile: av1c_bytes[1] >> 5,
                    seq_level_idx0: 0,
                    seq_tier0: 0,
                    bit_depth: 8,
                    subsampling: Subsampling::Yuv420,
                    chroma_sample_position: 0,
                    config_obus: Vec::new(),
                },
                has_alpha,
                is_grid: false,
            };
            pixel_format(&info)
        }
        assert_eq!(format_for([0x81, 0, 0x0C, 0], false), PixelFormat::Rgb8);
        assert_eq!(format_for([0x81, 0, 0x0C, 0], true), PixelFormat::Rgba8);

        let mono = |depth: u8, alpha: bool| {
            pixel_format(&AvifInfo {
                width: 1,
                height: 1,
                config: Av1Config {
                    seq_profile: 0,
                    seq_level_idx0: 0,
                    seq_tier0: 0,
                    bit_depth: depth,
                    subsampling: Subsampling::Monochrome,
                    chroma_sample_position: 0,
                    config_obus: Vec::new(),
                },
                has_alpha: alpha,
                is_grid: false,
            })
        };
        assert_eq!(mono(8, false), PixelFormat::Gray8);
        assert_eq!(mono(8, true), PixelFormat::GrayA8);
        assert_eq!(mono(10, false), PixelFormat::Gray16);
        // No GrayA16 exists, so this widens rather than dropping a channel.
        assert_eq!(mono(12, true), PixelFormat::Rgba16);
    }

    #[test]
    fn the_row_buffer_size_contract_is_enforced_before_decoding() {
        let file = minimal_avif(4, 4);
        let mut decoder = AvifDecoder::new(&file[..], Limits::default()).unwrap();

        // A wrong-sized buffer is an argument error, checked before the decode
        // is reached.
        let mut wrong = [0_u8; 3];
        assert_eq!(
            decoder.read_row(&mut wrong).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );

        // The synthetic fixture carries no real coded frame, so the decode
        // itself fails cleanly rather than panicking. (Real decodes are proven
        // bit-exact against libavif in tests/reference.rs.)
        let mut row = vec![0_u8; decoder.descriptor().row_bytes()];
        assert!(decoder.read_row(&mut row).is_err());
    }

    #[test]
    fn dimensions_are_limited_before_anything_is_allocated() {
        let file = minimal_avif(100_000, 100_000);
        let error = AvifDecoder::new(&file[..], Limits::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::LimitExceeded);
    }

    #[test]
    fn an_essential_property_we_do_not_understand_refuses_the_file() {
        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&meta_box(&[
            pitm(1),
            iinf(&[infe(1, b"av01", false)]),
            iloc(&[(1, 0, 1)]),
            // Property 3 is unknown and marked essential (high bit set).
            iprp(
                &[ispe(8, 8), av1c(), boxed(b"zzzz", &[0; 4])],
                &[(1, &[1, 2, 0x83])],
            ),
        ]));

        let error = AvifDecoder::new(&file[..], Limits::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Unsupported);
        assert!(error.to_string().contains("zzzz"), "{error}");
    }

    #[test]
    fn a_file_without_the_boxes_an_image_needs_is_rejected() {
        // No meta at all.
        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&boxed(b"mdat", &[0; 4]));
        let error = AvifDecoder::new(&file[..], Limits::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("no meta box"), "{error}");

        // A meta with no items.
        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&meta_box(&[pitm(1)]));
        let error = AvifDecoder::new(&file[..], Limits::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(
            error.to_string().contains("no primary image item"),
            "{error}"
        );

        // An item with no ispe, so no dimensions.
        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&meta_box(&[
            pitm(1),
            iinf(&[infe(1, b"av01", false)]),
            iloc(&[(1, 0, 1)]),
            iprp(&[av1c()], &[(1, &[1])]),
        ]));
        let error = AvifDecoder::new(&file[..], Limits::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("no ispe"), "{error}");

        // An item with no av1C, so it is not a coded AV1 image.
        let mut file = ftyp(b"avif", &[]);
        file.extend_from_slice(&meta_box(&[
            pitm(1),
            iinf(&[infe(1, b"av01", false)]),
            iloc(&[(1, 0, 1)]),
            iprp(&[ispe(8, 8)], &[(1, &[1])]),
        ]));
        let error = AvifDecoder::new(&file[..], Limits::default()).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("no av1C"), "{error}");
    }

    #[test]
    fn a_stream_that_is_not_an_avif_is_rejected() {
        for bytes in [
            &b""[..],
            &b"not an avif at all"[..],
            &b"\x89PNG\r\n\x1a\n"[..],
        ] {
            let error = AvifDecoder::new(bytes, Limits::default()).unwrap_err();
            assert_eq!(error.code(), ErrorCode::Malformed, "for {bytes:?}");
        }
    }

    #[test]
    fn probe_needs_a_brand_not_just_an_isobmff_header() {
        assert!(probe(&ftyp(b"avif", &[])));
        // The common real-world shape: mif1 major, avif compatible.
        assert!(probe(&ftyp(b"mif1", &[b"avif", b"miaf"])));
        assert!(probe(&ftyp(b"MA1B", &[])));
        assert!(probe(&ftyp(b"avis", &[b"avif"])));

        // Other ISOBMFF families must not be claimed.
        assert!(!probe(&ftyp(b"isom", &[b"mp42"])));
        assert!(!probe(&ftyp(b"heic", &[b"heix"])));
        assert!(!probe(&ftyp(b"qt  ", &[])));

        // Short prefixes are declined, never indexed past.
        assert!(!probe(b""));
        assert!(!probe(b"\0\0\0\x20ftyp"));
        assert!(!probe(b"\x89PNG\r\n\x1a\n"));
    }

    /// The minor version sits between the major brand and the compatible
    /// brands, and is a number. A file whose minor version happened to spell
    /// a brand must not be claimed on that basis.
    #[test]
    fn the_minor_version_is_not_read_as_a_brand() {
        let mut payload = Vec::from(*b"isom");
        payload.extend_from_slice(b"avif"); // minor version, not a brand
        payload.extend_from_slice(b"mp42");
        assert!(!probe(&boxed(b"ftyp", &payload)));
    }

    #[test]
    fn the_codec_entry_reports_the_format() {
        assert_eq!(AvifCodec.format(), Format::Avif);
        assert!(AvifCodec.probe(&ftyp(b"avif", &[])));
        assert!(!AvifCodec.probe(b"short"));
    }
}
