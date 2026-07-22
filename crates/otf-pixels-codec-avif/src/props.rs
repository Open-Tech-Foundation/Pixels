//! Item properties: `iprp` holding `ipco` (the property store) and `ipma`
//! (the item-to-property associations).
//!
//! HEIF keeps an item's descriptive data out of line. `ipco` is an array of
//! property boxes shared by every item in the file, and `ipma` maps each item
//! to the 1-based indices of the properties that describe it. So an image
//! item's dimensions, its AV1 configuration and its colour information are
//! three separate boxes elsewhere in the file, joined only by index.
//!
//! # Essential properties
//!
//! Each association carries an `essential` bit. A property marked essential
//! that the decoder does not understand makes the item undecodable — the
//! specification is explicit that it must not be ignored, because essential
//! properties change how the pixels are to be interpreted. Honouring that bit
//! is why [`Properties::essential_unknown`] exists: silently skipping an
//! unknown essential property would produce a confidently wrong image.

use crate::boxes::{FourCc, Reader};
use otf_pixels_core::{PixelsError, Result};
use std::collections::HashMap;

/// How the chroma planes are sampled relative to luma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:0:0 — no chroma planes at all.
    Monochrome,
    /// 4:2:0 — chroma at half resolution on both axes.
    Yuv420,
    /// 4:2:2 — chroma at half resolution horizontally.
    Yuv422,
    /// 4:4:4 — chroma at full resolution.
    Yuv444,
}

impl Subsampling {
    /// How many luma samples share one chroma sample horizontally.
    #[must_use]
    pub const fn x_shift(self) -> u32 {
        match self {
            Self::Yuv420 | Self::Yuv422 => 1,
            Self::Monochrome | Self::Yuv444 => 0,
        }
    }

    /// How many luma samples share one chroma sample vertically.
    #[must_use]
    pub const fn y_shift(self) -> u32 {
        match self {
            Self::Yuv420 => 1,
            Self::Monochrome | Self::Yuv422 | Self::Yuv444 => 0,
        }
    }
}

/// The `av1C` AV1 codec configuration.
///
/// This is the bridge between the two specifications AVIF is made of: it
/// declares the AV1 profile and sample format, and carries the sequence header
/// OBU that the bitstream decoder needs before it can read a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1Config {
    /// AV1 sequence profile, 0–2.
    pub seq_profile: u8,
    /// AV1 level index.
    pub seq_level_idx0: u8,
    /// AV1 tier.
    pub seq_tier0: u8,
    /// Bits per sample: 8, 10 or 12.
    pub bit_depth: u8,
    /// How chroma is sampled.
    pub subsampling: Subsampling,
    /// Chroma sample position, 0–3.
    pub chroma_sample_position: u8,
    /// The OBUs carried in the configuration record itself.
    ///
    /// Normally the sequence header, so a decoder can be configured before it
    /// reaches the frame data in `mdat`.
    pub config_obus: Vec<u8>,
}

/// The `ispe` image spatial extents — an item's dimensions in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extents {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The `pixi` pixel information — bit depth per channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelInfo {
    /// Bits per channel, one entry per channel.
    pub bits_per_channel: Vec<u8>,
}

/// The `colr` colour information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Colour {
    /// CICP code points, as `nclx`.
    Nclx {
        /// CICP colour primaries.
        primaries: u16,
        /// CICP transfer characteristics.
        transfer: u16,
        /// CICP matrix coefficients, which choose the YUV↔RGB matrix.
        matrix: u16,
        /// Whether samples use the full range rather than the studio range.
        full_range: bool,
    },
    /// An embedded ICC profile, as `rICC` or `prof`.
    ///
    /// Carried through but not applied: SPEC §Pixel formats puts ICC
    /// transforms in v2, and v1 is sRGB-assumed.
    Icc(Vec<u8>),
}

/// One entry of the `ipco` property store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Property {
    /// `ispe` — the item's dimensions.
    Extents(Extents),
    /// `av1C` — the AV1 codec configuration.
    Av1Config(Av1Config),
    /// `pixi` — bit depth per channel.
    PixelInfo(PixelInfo),
    /// `colr` — colour information.
    Colour(Colour),
    /// `irot` — rotation, in counter-clockwise multiples of 90 degrees (0–3).
    Rotation(u8),
    /// `imir` — mirroring. `false` mirrors left-to-right about a vertical
    /// axis; `true` mirrors top-to-bottom about a horizontal axis.
    Mirror(bool),
    /// `auxC` — the auxiliary type URN, which is how an alpha plane announces
    /// itself.
    AuxiliaryType(String),
    /// `pasp` — the pixel aspect ratio, as horizontal and vertical spacing.
    PixelAspect(u32, u32),
    /// A property this decoder does not interpret.
    ///
    /// Retained rather than dropped so that [`Properties::essential_unknown`]
    /// can tell whether ignoring it is safe.
    Unknown(FourCc),
}

impl Property {
    /// The four-character type this property was parsed from.
    #[must_use]
    pub const fn kind(&self) -> FourCc {
        match self {
            Self::Extents(_) => FourCc::new(b"ispe"),
            Self::Av1Config(_) => FourCc::new(b"av1C"),
            Self::PixelInfo(_) => FourCc::new(b"pixi"),
            Self::Colour(_) => FourCc::new(b"colr"),
            Self::Rotation(_) => FourCc::new(b"irot"),
            Self::Mirror(_) => FourCc::new(b"imir"),
            Self::AuxiliaryType(_) => FourCc::new(b"auxC"),
            Self::PixelAspect(_, _) => FourCc::new(b"pasp"),
            Self::Unknown(kind) => *kind,
        }
    }
}

/// One item's association with a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Association {
    /// 1-based index into the `ipco` store. Zero means "no property".
    pub index: u16,
    /// Whether an unrecognised property here makes the item undecodable.
    pub essential: bool,
}

/// The parsed contents of `iprp`.
#[derive(Debug, Clone, Default)]
pub struct Properties {
    /// The `ipco` store, in file order. Association indices are 1-based.
    entries: Vec<Property>,
    /// The `ipma` map from item ID to that item's associations.
    associations: HashMap<u32, Vec<Association>>,
}

impl Properties {
    /// Parse an `iprp` box payload.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the box or any child is
    /// structurally invalid.
    pub fn parse(mut iprp: Reader<'_>) -> Result<Self> {
        let mut entries = Vec::new();
        let mut associations = HashMap::new();

        while let Some(header) = iprp.next_box() {
            let header = header?;
            let payload = iprp.payload(&header);
            match &header.kind.0 {
                b"ipco" => entries = parse_ipco(payload)?,
                // A file may carry more than one `ipma`; the associations
                // accumulate rather than the later box replacing the earlier.
                b"ipma" => parse_ipma(payload, &mut associations)?,
                _ => {}
            }
        }

        Ok(Self {
            entries,
            associations,
        })
    }

    /// Every property associated with `item_id`, in association order.
    #[must_use]
    pub fn for_item(&self, item_id: u32) -> Vec<&Property> {
        let Some(associations) = self.associations.get(&item_id) else {
            return Vec::new();
        };
        associations
            .iter()
            .filter_map(|association| {
                // Indices are 1-based, and zero means "no property".
                let index = usize::from(association.index).checked_sub(1)?;
                self.entries.get(index)
            })
            .collect()
    }

    /// The type of the first essential property of `item_id` that this
    /// decoder does not interpret, if there is one.
    ///
    /// An item with such a property must be reported [`PixelsError::Unsupported`]
    /// rather than decoded, because the property was declared to change the
    /// meaning of the pixels.
    #[must_use]
    pub fn essential_unknown(&self, item_id: u32) -> Option<FourCc> {
        let associations = self.associations.get(&item_id)?;
        associations.iter().find_map(|association| {
            if !association.essential {
                return None;
            }
            let index = usize::from(association.index).checked_sub(1)?;
            match self.entries.get(index) {
                Some(Property::Unknown(kind)) => Some(*kind),
                _ => None,
            }
        })
    }

    /// The item's dimensions, from its `ispe`.
    #[must_use]
    pub fn extents(&self, item_id: u32) -> Option<Extents> {
        self.for_item(item_id).into_iter().find_map(|property| {
            match property {
                Property::Extents(extents) => Some(*extents),
                _ => None,
            }
        })
    }

    /// The item's AV1 configuration, from its `av1C`.
    #[must_use]
    pub fn av1_config(&self, item_id: u32) -> Option<&Av1Config> {
        self.for_item(item_id).into_iter().find_map(|property| {
            match property {
                Property::Av1Config(config) => Some(config),
                _ => None,
            }
        })
    }

    /// The item's colour information, from its `colr`.
    #[must_use]
    pub fn colour(&self, item_id: u32) -> Option<&Colour> {
        self.for_item(item_id).into_iter().find_map(|property| {
            match property {
                Property::Colour(colour) => Some(colour),
                _ => None,
            }
        })
    }

    /// The item's auxiliary type URN, from its `auxC`.
    #[must_use]
    pub fn auxiliary_type(&self, item_id: u32) -> Option<&str> {
        self.for_item(item_id).into_iter().find_map(|property| {
            match property {
                Property::AuxiliaryType(urn) => Some(urn.as_str()),
                _ => None,
            }
        })
    }

    /// The item's rotation and mirroring, as `(rotation, mirror)`.
    ///
    /// Rotation is counter-clockwise quarter turns, 0–3.
    #[must_use]
    pub fn orientation(&self, item_id: u32) -> (u8, Option<bool>) {
        let mut rotation = 0;
        let mut mirror = None;
        for property in self.for_item(item_id) {
            match property {
                Property::Rotation(turns) => rotation = *turns,
                Property::Mirror(axis) => mirror = Some(*axis),
                _ => {}
            }
        }
        (rotation, mirror)
    }
}

/// Parse the `ipco` property store.
fn parse_ipco(mut ipco: Reader<'_>) -> Result<Vec<Property>> {
    let mut entries = Vec::new();
    while let Some(header) = ipco.next_box() {
        let header = header?;
        let payload = ipco.payload(&header);
        entries.push(parse_property(header.kind, payload)?);
    }
    Ok(entries)
}

/// Parse one property box.
fn parse_property(kind: FourCc, mut payload: Reader<'_>) -> Result<Property> {
    match &kind.0 {
        b"ispe" => {
            let (_version, _flags) = payload.full_box()?;
            let width = payload.u32()?;
            let height = payload.u32()?;
            Ok(Property::Extents(Extents { width, height }))
        }
        b"av1C" => parse_av1c(payload).map(Property::Av1Config),
        b"pixi" => {
            let (_version, _flags) = payload.full_box()?;
            let count = payload.u8()?;
            let mut bits_per_channel = Vec::with_capacity(usize::from(count));
            for _ in 0..count {
                bits_per_channel.push(payload.u8()?);
            }
            Ok(Property::PixelInfo(PixelInfo { bits_per_channel }))
        }
        b"colr" => parse_colr(payload).map(Property::Colour),
        b"irot" => {
            // Only the low two bits are the angle; the rest is reserved.
            let turns = payload.u8()? & 0x03;
            Ok(Property::Rotation(turns))
        }
        b"imir" => {
            // Low bit: 0 is a vertical axis (mirror left-to-right), 1 is a
            // horizontal axis (mirror top-to-bottom). An early HEIF draft
            // defined these the other way round, and files written against it
            // exist; we follow the published standard.
            let axis = payload.u8()? & 0x01;
            Ok(Property::Mirror(axis == 1))
        }
        b"auxC" => {
            let (_version, _flags) = payload.full_box()?;
            let urn = payload.cstring()?;
            Ok(Property::AuxiliaryType(urn.to_owned()))
        }
        b"pasp" => {
            let horizontal = payload.u32()?;
            let vertical = payload.u32()?;
            Ok(Property::PixelAspect(horizontal, vertical))
        }
        _ => Ok(Property::Unknown(kind)),
    }
}

/// Parse an `av1C` AV1 codec configuration record.
fn parse_av1c(mut payload: Reader<'_>) -> Result<Av1Config> {
    let first = payload.u8()?;
    // Top bit is a marker that must be 1, low seven bits the record version,
    // which is 1. Both are fixed by the specification, so a mismatch means
    // this is not an AV1 configuration record.
    if first >> 7 != 1 {
        return Err(PixelsError::malformed(
            "avif",
            "the av1C marker bit is not set",
        ));
    }
    let version = first & 0x7f;
    if version != 1 {
        return Err(PixelsError::malformed(
            "avif",
            format!("av1C record version {version} is not 1"),
        ));
    }

    let second = payload.u8()?;
    let seq_profile = second >> 5;
    let seq_level_idx0 = second & 0x1f;

    let third = payload.u8()?;
    let seq_tier0 = (third >> 7) & 1;
    let high_bitdepth = (third >> 6) & 1;
    let twelve_bit = (third >> 5) & 1;
    let monochrome = (third >> 4) & 1;
    let subsampling_x = (third >> 3) & 1;
    let subsampling_y = (third >> 2) & 1;
    let chroma_sample_position = third & 0x03;

    // Byte four is reserved bits plus the initial presentation delay, which
    // only matters for sequences.
    let _fourth = payload.u8()?;

    // Profile 2 is the only one that can carry twelve-bit samples.
    let bit_depth = if seq_profile == 2 && high_bitdepth == 1 {
        if twelve_bit == 1 { 12 } else { 10 }
    } else if high_bitdepth == 1 {
        10
    } else {
        8
    };

    let subsampling = match (monochrome, subsampling_x, subsampling_y) {
        (1, _, _) => Subsampling::Monochrome,
        (_, 1, 1) => Subsampling::Yuv420,
        (_, 1, 0) => Subsampling::Yuv422,
        (_, 0, 0) => Subsampling::Yuv444,
        // 4:4:0 is not a format AV1 defines.
        (_, x, y) => {
            return Err(PixelsError::malformed(
                "avif",
                format!("av1C declares chroma subsampling ({x}, {y}), which AV1 has no such format for"),
            ));
        }
    };

    Ok(Av1Config {
        seq_profile,
        seq_level_idx0,
        seq_tier0,
        bit_depth,
        subsampling,
        chroma_sample_position,
        config_obus: payload.rest().to_vec(),
    })
}

/// Parse a `colr` colour information box.
fn parse_colr(mut payload: Reader<'_>) -> Result<Colour> {
    let kind = payload.fourcc()?;
    match &kind.0 {
        b"nclx" => {
            let primaries = payload.u16()?;
            let transfer = payload.u16()?;
            let matrix = payload.u16()?;
            // Full-range is the top bit of the next byte; the rest is reserved.
            let full_range = payload.u8()? >> 7 == 1;
            Ok(Colour::Nclx {
                primaries,
                transfer,
                matrix,
                full_range,
            })
        }
        b"rICC" | b"prof" => Ok(Colour::Icc(payload.rest().to_vec())),
        other => Err(PixelsError::malformed(
            "avif",
            format!("colr declares colour type '{}', which is not one this format defines", FourCc(*other)),
        )),
    }
}

/// Parse an `ipma` association box into `out`.
fn parse_ipma(mut payload: Reader<'_>, out: &mut HashMap<u32, Vec<Association>>) -> Result<()> {
    let (version, flags) = payload.full_box()?;
    let entry_count = payload.u32()?;
    // Each entry is at least three bytes, so a count larger than the box could
    // hold is malformed. Checking up front stops a huge count from driving a
    // huge allocation before the truncation is noticed.
    let smallest_entry = if version < 1 { 3 } else { 5 };
    if u64::from(entry_count) * smallest_entry > payload.remaining() as u64 {
        return Err(PixelsError::malformed(
            "avif",
            format!(
                "ipma declares {entry_count} entries, more than its {} remaining bytes can hold",
                payload.remaining()
            ),
        ));
    }

    // Flag bit 0 selects 15-bit property indices over 7-bit ones.
    let wide_indices = flags & 1 == 1;

    for _ in 0..entry_count {
        let item_id = if version < 1 {
            u32::from(payload.u16()?)
        } else {
            payload.u32()?
        };
        let association_count = payload.u8()?;
        let mut associations = Vec::with_capacity(usize::from(association_count));
        for _ in 0..association_count {
            let (essential, index) = if wide_indices {
                let word = payload.u16()?;
                (word >> 15 == 1, word & 0x7fff)
            } else {
                let byte = payload.u8()?;
                (byte >> 7 == 1, u16::from(byte & 0x7f))
            };
            associations.push(Association { index, essential });
        }
        out.entry(item_id).or_default().extend(associations);
    }
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
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

    fn ispe(width: u32, height: u32) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&width.to_be_bytes());
        payload.extend_from_slice(&height.to_be_bytes());
        boxed(b"ispe", &payload)
    }

    /// An `av1C` for 8-bit 4:2:0, profile 0, with one trailing config byte.
    fn av1c_420_8bit() -> Vec<u8> {
        // marker+version, profile/level, tier/depth/mono/subsampling, delay.
        boxed(b"av1C", &[0x81, 0x00, 0x0C, 0x00, 0xAA])
    }

    fn parse_one(kind: &[u8; 4], payload: &[u8]) -> Result<Property> {
        let file = boxed(kind, payload);
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        parse_property(header.kind, reader.payload(&header))
    }

    #[test]
    fn av1c_decodes_profile_depth_and_subsampling() {
        let file = av1c_420_8bit();
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let Property::Av1Config(config) =
            parse_property(header.kind, reader.payload(&header)).unwrap()
        else {
            panic!("expected an av1C");
        };
        assert_eq!(config.seq_profile, 0);
        assert_eq!(config.bit_depth, 8);
        assert_eq!(config.subsampling, Subsampling::Yuv420);
        // Everything after the four-byte record is the config OBU run.
        assert_eq!(config.config_obus, vec![0xAA]);
    }

    #[test]
    fn av1c_twelve_bit_requires_profile_two() {
        // high_bitdepth=1, twelve_bit=1, but profile 0: reads as 10-bit,
        // because only profile 2 can carry twelve-bit samples.
        let file = boxed(b"av1C", &[0x81, 0x00, 0x6C, 0x00]);
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let Property::Av1Config(config) =
            parse_property(header.kind, reader.payload(&header)).unwrap()
        else {
            panic!("expected an av1C");
        };
        assert_eq!(config.bit_depth, 10);

        // The same bits under profile 2 do mean twelve.
        let file = boxed(b"av1C", &[0x81, 0x40, 0x6C, 0x00]);
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let Property::Av1Config(config) =
            parse_property(header.kind, reader.payload(&header)).unwrap()
        else {
            panic!("expected an av1C");
        };
        assert_eq!(config.seq_profile, 2);
        assert_eq!(config.bit_depth, 12);
    }

    #[test]
    fn av1c_rejects_a_bad_marker_or_version() {
        let error = parse_one(b"av1C", &[0x01, 0x00, 0x0C, 0x00]).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("marker"), "{error}");

        let error = parse_one(b"av1C", &[0x82, 0x00, 0x0C, 0x00]).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("version 2"), "{error}");
    }

    #[test]
    fn av1c_rejects_a_subsampling_av1_does_not_define() {
        // 4:4:0 — subsampling_x = 0, subsampling_y = 1.
        let error = parse_one(b"av1C", &[0x81, 0x00, 0x04, 0x00]).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("subsampling"), "{error}");
    }

    #[test]
    fn monochrome_wins_over_the_subsampling_bits() {
        // monochrome = 1 with both subsampling bits set.
        let file = boxed(b"av1C", &[0x81, 0x00, 0x1C, 0x00]);
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let Property::Av1Config(config) =
            parse_property(header.kind, reader.payload(&header)).unwrap()
        else {
            panic!("expected an av1C");
        };
        assert_eq!(config.subsampling, Subsampling::Monochrome);
        assert_eq!(config.subsampling.x_shift(), 0);
        assert_eq!(config.subsampling.y_shift(), 0);
    }

    #[test]
    fn colr_reads_nclx_code_points_and_the_range_flag() {
        let mut payload = Vec::from(*b"nclx");
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&13_u16.to_be_bytes());
        payload.extend_from_slice(&6_u16.to_be_bytes());
        payload.push(0x80);

        let Property::Colour(Colour::Nclx {
            primaries,
            transfer,
            matrix,
            full_range,
        }) = parse_one(b"colr", &payload).unwrap()
        else {
            panic!("expected an nclx colr");
        };
        assert_eq!((primaries, transfer, matrix), (1, 13, 6));
        assert!(full_range);
    }

    #[test]
    fn colr_carries_an_icc_profile_through() {
        let mut payload = Vec::from(*b"prof");
        payload.extend_from_slice(&[1, 2, 3, 4]);
        let Property::Colour(Colour::Icc(profile)) = parse_one(b"colr", &payload).unwrap() else {
            panic!("expected an ICC colr");
        };
        assert_eq!(profile, vec![1, 2, 3, 4]);
    }

    #[test]
    fn orientation_properties_are_masked_to_their_defined_bits() {
        // Reserved high bits set; only the low two are the angle.
        assert_eq!(parse_one(b"irot", &[0xFE]).unwrap(), Property::Rotation(2));
        assert_eq!(parse_one(b"imir", &[0xFE]).unwrap(), Property::Mirror(false));
        assert_eq!(parse_one(b"imir", &[0xFF]).unwrap(), Property::Mirror(true));
    }

    #[test]
    fn pixi_reads_one_depth_per_channel() {
        let Property::PixelInfo(info) = parse_one(b"pixi", &[0, 0, 0, 0, 3, 8, 8, 8]).unwrap()
        else {
            panic!("expected a pixi");
        };
        assert_eq!(info.bits_per_channel, vec![8, 8, 8]);
    }

    #[test]
    fn an_uninterpreted_property_is_retained_by_type() {
        assert_eq!(
            parse_one(b"clap", &[0; 32]).unwrap(),
            Property::Unknown(FourCc::new(b"clap"))
        );
    }

    /// Build an `iprp` with one `ispe` and one `av1C`, associated to item 1.
    fn sample_iprp() -> Vec<u8> {
        let mut ipco = ispe(64, 48);
        ipco.extend_from_slice(&av1c_420_8bit());

        // version 0, flags 0, one entry, item 1, two associations: property 1
        // non-essential, property 2 essential.
        let ipma_payload = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0x01, 0x82];

        let mut iprp = boxed(b"ipco", &ipco);
        iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));
        boxed(b"iprp", &iprp)
    }

    #[test]
    fn properties_resolve_through_the_association_map() {
        let file = sample_iprp();
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let properties = Properties::parse(reader.payload(&header)).unwrap();

        assert_eq!(
            properties.extents(1),
            Some(Extents {
                width: 64,
                height: 48
            })
        );
        assert_eq!(properties.av1_config(1).unwrap().bit_depth, 8);
        // An item with no associations resolves to nothing rather than failing.
        assert!(properties.extents(2).is_none());
        assert!(properties.for_item(2).is_empty());
    }

    #[test]
    fn an_out_of_range_association_index_is_skipped_not_indexed() {
        let mut ipco = ispe(8, 8);
        ipco.extend_from_slice(&av1c_420_8bit());
        // Associates property 99, which does not exist, and index 0, which
        // means "no property".
        let ipma_payload = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 99, 0];
        let mut iprp = boxed(b"ipco", &ipco);
        iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));
        let file = boxed(b"iprp", &iprp);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let properties = Properties::parse(reader.payload(&header)).unwrap();
        assert!(properties.for_item(1).is_empty());
    }

    #[test]
    fn an_essential_property_we_do_not_understand_is_reported() {
        let mut ipco = ispe(8, 8);
        // A property type this decoder does not interpret.
        ipco.extend_from_slice(&boxed(b"zzzz", &[0; 4]));
        // Item 1 associates property 2 as essential.
        let ipma_payload = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0x82];
        let mut iprp = boxed(b"ipco", &ipco);
        iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));
        let file = boxed(b"iprp", &iprp);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let properties = Properties::parse(reader.payload(&header)).unwrap();
        assert_eq!(
            properties.essential_unknown(1),
            Some(FourCc::new(b"zzzz")),
            "an essential unknown property must not be silently ignored"
        );

        // The same property marked non-essential is ignorable.
        let ipma_payload = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0x02];
        let mut iprp = boxed(b"ipco", &ipco);
        iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));
        let file = boxed(b"iprp", &iprp);
        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let properties = Properties::parse(reader.payload(&header)).unwrap();
        assert_eq!(properties.essential_unknown(1), None);
    }

    #[test]
    fn ipma_supports_wide_indices_and_wide_item_ids() {
        let mut ipco = ispe(16, 16);
        ipco.extend_from_slice(&av1c_420_8bit());
        // version 1 (32-bit item IDs), flags 1 (15-bit property indices).
        let mut ipma_payload = vec![1, 0, 0, 1];
        ipma_payload.extend_from_slice(&1_u32.to_be_bytes()); // entry count
        ipma_payload.extend_from_slice(&70_000_u32.to_be_bytes()); // item ID
        ipma_payload.push(1); // association count
        ipma_payload.extend_from_slice(&0x0001_u16.to_be_bytes()); // property 1

        let mut iprp = boxed(b"ipco", &ipco);
        iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));
        let file = boxed(b"iprp", &iprp);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let properties = Properties::parse(reader.payload(&header)).unwrap();
        assert_eq!(
            properties.extents(70_000),
            Some(Extents {
                width: 16,
                height: 16
            })
        );
    }

    /// A huge entry count must be rejected against the box's actual size
    /// rather than driving an allocation proportional to the claim.
    #[test]
    fn ipma_rejects_more_entries_than_the_box_can_hold() {
        let ipma_payload = vec![0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0, 1, 0];
        let mut iprp = boxed(b"ipco", &ispe(8, 8));
        iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));
        let file = boxed(b"iprp", &iprp);

        let mut reader = Reader::new(&file);
        let header = reader.next_box().unwrap().unwrap();
        let error = Properties::parse(reader.payload(&header)).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("more than its"), "{error}");
    }

    #[test]
    fn auxc_reads_the_alpha_urn() {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0");
        assert_eq!(
            parse_one(b"auxC", &payload).unwrap(),
            Property::AuxiliaryType("urn:mpeg:mpegB:cicp:systems:auxiliary:alpha".to_owned())
        );
    }
}
