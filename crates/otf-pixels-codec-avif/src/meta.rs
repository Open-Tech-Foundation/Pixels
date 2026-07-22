//! The `meta` box: what items a file holds, where their bytes are, and how
//! they relate.
//!
//! A HEIF file is not a picture with some metadata attached — it is a small
//! database of *items*, one of which is nominated primary. An ordinary AVIF
//! has one `av01` item and that is the image. A file with transparency has a
//! second `av01` item carrying the alpha plane, joined to the first by an
//! `auxl` reference. A large image may have a `grid` item whose pixels are a
//! `dimg` reference to a list of tile items.
//!
//! Four boxes do the work:
//!
//! - `pitm` names the primary item.
//! - `iinf` lists the items and their types.
//! - `iloc` says where each item's bytes are, as extents in the file or in
//!   `idat`.
//! - `iref` records the relationships between items.

use crate::boxes::{FourCc, Reader};
use crate::props::Properties;
use otf_pixels_core::{PixelsError, Result};
use std::borrow::Cow;

/// The auxiliary type URN that marks an item as an alpha plane.
pub const URN_ALPHA: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";

/// An older URN for the same thing, written by encoders that predate the
/// current registration and still found in the wild.
pub const URN_ALPHA_LEGACY: &str = "urn:mpeg:hevc:2015:auxid:1";

/// Where an item's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Construction {
    /// Offsets are absolute within the file. This is what almost every AVIF
    /// uses: the extents point into `mdat`.
    File,
    /// Offsets are relative to the start of the `idat` box, which is how small
    /// items are carried inside `meta` itself. A `grid` item's configuration
    /// is normally stored this way.
    Idat,
    /// Offsets are relative to another item's data.
    ///
    /// Legal but vanishingly rare, and it invites reference cycles that a
    /// resolver has to defend against. Reported [`PixelsError::Unsupported`]
    /// rather than implemented speculatively.
    Item,
}

/// One contiguous run of an item's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// Offset, interpreted according to the item's [`Construction`].
    pub offset: u64,
    /// Length in bytes.
    pub length: u64,
}

/// One entry of the file's item table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// The item's identifier, unique within the file.
    pub id: u32,
    /// The item type: `av01` for a coded image, `grid` for a derived tiling,
    /// `Exif` or `mime` for metadata.
    pub kind: FourCc,
    /// The item's name, which is informational.
    pub name: String,
    /// Whether the item is marked hidden and so must not be displayed on its
    /// own. An alpha plane is normally hidden.
    pub hidden: bool,
    /// How this item's extents are addressed.
    pub construction: Construction,
    /// The runs of bytes that make up the item, in order.
    pub extents: Vec<Extent>,
}

impl Item {
    /// Whether this item is a coded AV1 image.
    #[must_use]
    pub fn is_coded_image(&self) -> bool {
        self.kind == FourCc::new(b"av01")
    }

    /// Whether this item is a derived grid of tiles.
    #[must_use]
    pub fn is_grid(&self) -> bool {
        self.kind == FourCc::new(b"grid")
    }
}

/// One `iref` entry: a typed link from one item to others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    /// The reference type. `dimg` links a derived image to its inputs, `auxl`
    /// links an auxiliary item to the item it describes, `thmb` links a
    /// thumbnail to its full-size image.
    pub kind: FourCc,
    /// The item the reference is from.
    pub from: u32,
    /// The items it points to, in order — which for `dimg` is the tile order.
    pub to: Vec<u32>,
}

/// The parsed `meta` box.
#[derive(Debug, Clone, Default)]
pub struct Meta {
    /// The primary item, from `pitm`.
    pub primary: Option<u32>,
    /// Every item in the file, in `iinf` order.
    pub items: Vec<Item>,
    /// Every `iref` entry.
    pub references: Vec<Reference>,
    /// The item properties, from `iprp`.
    pub properties: Properties,
    /// Where the `idat` payload starts in the file, and how long it is.
    idat: Option<(usize, usize)>,
}

impl Meta {
    /// Parse a `meta` box payload.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if any child box is structurally
    /// invalid.
    pub fn parse(mut meta: Reader<'_>) -> Result<Self> {
        // `meta` is a full box: version and flags come before its children.
        let (_version, _flags) = meta.full_box()?;

        let mut out = Self::default();
        // `iloc` gives extents but not types and `iinf` the reverse, so both
        // are collected and joined once the whole box has been read — the
        // specification fixes no order between them.
        let mut locations = Vec::new();
        let mut infos = Vec::new();

        while let Some(header) = meta.next_box() {
            let header = header?;
            let payload = meta.payload(&header);
            match &header.kind.0 {
                b"pitm" => out.primary = Some(parse_pitm(payload)?),
                b"iinf" => infos = parse_iinf(payload)?,
                b"iloc" => locations = parse_iloc(payload)?,
                b"iref" => out.references = parse_iref(payload)?,
                b"iprp" => out.properties = Properties::parse(payload)?,
                b"idat" => out.idat = Some((header.payload_start, header.payload_len)),
                _ => {}
            }
        }

        out.items = join(infos, locations);
        Ok(out)
    }

    /// The item with this ID, if the file has one.
    #[must_use]
    pub fn item(&self, id: u32) -> Option<&Item> {
        self.items.iter().find(|item| item.id == id)
    }

    /// The primary item, resolved.
    ///
    /// Falls back to the first coded image or grid when `pitm` is absent.
    /// A file without `pitm` is malformed, but the intent is unambiguous when
    /// there is exactly one image, and rejecting it buys nothing.
    #[must_use]
    pub fn primary_item(&self) -> Option<&Item> {
        self.primary.and_then(|id| self.item(id)).or_else(|| {
            self.items
                .iter()
                .find(|item| item.is_coded_image() || item.is_grid())
        })
    }

    /// The items `from` points to with a reference of this type.
    #[must_use]
    pub fn referenced(&self, from: u32, kind: &[u8; 4]) -> &[u32] {
        let wanted = FourCc::new(kind);
        self.references
            .iter()
            .find(|reference| reference.from == from && reference.kind == wanted)
            .map_or(&[], |reference| reference.to.as_slice())
    }

    /// The alpha plane item for `id`, if the file carries one.
    ///
    /// An alpha item is an auxiliary item whose `auxl` reference points at
    /// `id` and whose `auxC` property carries the alpha URN. Both halves are
    /// checked: an `auxl` item with some other auxiliary type is a depth map
    /// or a gain map, not transparency, and treating it as alpha would produce
    /// a confidently wrong image.
    #[must_use]
    pub fn alpha_item(&self, id: u32) -> Option<&Item> {
        self.references
            .iter()
            .filter(|reference| {
                reference.kind == FourCc::new(b"auxl") && reference.to.contains(&id)
            })
            .find_map(|reference| {
                let urn = self.properties.auxiliary_type(reference.from)?;
                if urn == URN_ALPHA || urn == URN_ALPHA_LEGACY {
                    self.item(reference.from)
                } else {
                    None
                }
            })
    }

    /// The bytes of `item`, resolved from `file`.
    ///
    /// Borrows when the item is one contiguous extent, which is the common
    /// case, and concatenates otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] for item-relative construction,
    /// or [`PixelsError::Malformed`] if an extent falls outside the file or
    /// outside `idat`.
    pub fn item_data<'a>(&self, file: &'a [u8], item: &Item) -> Result<Cow<'a, [u8]>> {
        let (base, region) = match item.construction {
            Construction::File => (0_usize, file),
            Construction::Idat => {
                let (start, len) = self.idat.ok_or_else(|| {
                    PixelsError::malformed(
                        "avif",
                        format!(
                            "item {} is stored in idat, but the file has no idat box",
                            item.id
                        ),
                    )
                })?;
                let region = file.get(start..start.saturating_add(len)).ok_or_else(|| {
                    PixelsError::malformed("avif", "the idat box extends past the file")
                })?;
                (0_usize, region)
            }
            Construction::Item => {
                return Err(PixelsError::unsupported(format!(
                    "avif: item {} uses item-relative extent offsets, which this decoder does not resolve",
                    item.id
                )));
            }
        };
        let _ = base;

        // Extents may repeat, so their lengths can sum past the file size even
        // though each is individually in range. Bound the total before any of
        // it is copied.
        let total: u64 = item.extents.iter().map(|extent| extent.length).sum();
        if total > region.len() as u64 {
            return Err(PixelsError::malformed(
                "avif",
                format!(
                    "item {} declares {total} bytes across {} extents, more than the {} available",
                    item.id,
                    item.extents.len(),
                    region.len()
                ),
            ));
        }

        let slice_of = |extent: &Extent| -> Result<&'a [u8]> {
            let start = usize::try_from(extent.offset).map_err(|_| {
                PixelsError::malformed(
                    "avif",
                    format!(
                        "item {} starts at offset {}, beyond this platform's addressing",
                        item.id, extent.offset
                    ),
                )
            })?;
            let len = usize::try_from(extent.length).map_err(|_| {
                PixelsError::malformed(
                    "avif",
                    format!(
                        "item {} declares a {}-byte extent, beyond this platform's addressing",
                        item.id, extent.length
                    ),
                )
            })?;
            let end = start.checked_add(len).ok_or_else(|| {
                PixelsError::malformed("avif", format!("item {}'s extent overflows", item.id))
            })?;
            region.get(start..end).ok_or_else(|| {
                PixelsError::malformed(
                    "avif",
                    format!(
                        "item {} has an extent at {start}..{end}, outside the {} bytes available",
                        item.id,
                        region.len()
                    ),
                )
            })
        };

        match item.extents.as_slice() {
            [] => Err(PixelsError::malformed(
                "avif",
                format!("item {} has no extents, so it has no data", item.id),
            )),
            [single] => slice_of(single).map(Cow::Borrowed),
            many => {
                let mut joined = Vec::with_capacity(usize::try_from(total).unwrap_or(0));
                for extent in many {
                    joined.extend_from_slice(slice_of(extent)?);
                }
                Ok(Cow::Owned(joined))
            }
        }
    }
}

/// Join the type table from `iinf` with the extent table from `iloc`.
///
/// An item present in one but not the other is dropped: without a type it
/// cannot be interpreted, and without extents it has no bytes.
fn join(infos: Vec<(u32, FourCc, String, bool)>, locations: Vec<Located>) -> Vec<Item> {
    infos
        .into_iter()
        .filter_map(|(id, kind, name, hidden)| {
            let located = locations.iter().find(|located| located.id == id)?;
            Some(Item {
                id,
                kind,
                name,
                hidden,
                construction: located.construction,
                extents: located.extents.clone(),
            })
        })
        .collect()
}

/// An `iloc` row, before it is joined with its type.
#[derive(Debug, Clone)]
struct Located {
    id: u32,
    construction: Construction,
    extents: Vec<Extent>,
}

/// Parse `pitm`, the primary item declaration.
fn parse_pitm(mut payload: Reader<'_>) -> Result<u32> {
    let (version, _flags) = payload.full_box()?;
    if version == 0 {
        payload.u16().map(u32::from)
    } else {
        payload.u32()
    }
}

/// Parse `iinf` into `(id, type, name, hidden)` rows.
fn parse_iinf(mut payload: Reader<'_>) -> Result<Vec<(u32, FourCc, String, bool)>> {
    let (version, _flags) = payload.full_box()?;
    let count = if version == 0 {
        u32::from(payload.u16()?)
    } else {
        payload.u32()?
    };

    // Each `infe` is a box, so at least eight bytes. Bound the count against
    // the box's real size before reserving anything.
    if u64::from(count) * 8 > payload.remaining() as u64 {
        return Err(PixelsError::malformed(
            "avif",
            format!(
                "iinf declares {count} items, more than its {} remaining bytes can hold",
                payload.remaining()
            ),
        ));
    }

    let mut out = Vec::with_capacity(count as usize);
    while let Some(header) = payload.next_box() {
        let header = header?;
        if header.kind != FourCc::new(b"infe") {
            continue;
        }
        out.push(parse_infe(payload.payload(&header))?);
    }
    Ok(out)
}

/// Parse one `infe` item information entry.
fn parse_infe(mut payload: Reader<'_>) -> Result<(u32, FourCc, String, bool)> {
    let (version, flags) = payload.full_box()?;
    // Versions 0 and 1 predate typed items and cannot describe an `av01`
    // item at all; every AVIF uses version 2 or 3.
    if version < 2 {
        return Err(PixelsError::malformed(
            "avif",
            format!("infe version {version} cannot carry an item type"),
        ));
    }
    let id = if version == 2 {
        u32::from(payload.u16()?)
    } else {
        payload.u32()?
    };
    let _protection = payload.u16()?;
    let kind = payload.fourcc()?;
    let name = payload.cstring()?.to_owned();
    // Flag bit 0 marks the item hidden.
    let hidden = flags & 1 == 1;
    Ok((id, kind, name, hidden))
}

/// Parse `iloc`, the item location table.
fn parse_iloc(mut payload: Reader<'_>) -> Result<Vec<Located>> {
    let (version, _flags) = payload.full_box()?;

    let sizes = payload.u8()?;
    let offset_size = sizes >> 4;
    let length_size = sizes & 0x0f;
    let sizes = payload.u8()?;
    let base_offset_size = sizes >> 4;
    // The low nibble is the index size on versions 1 and 2, reserved on 0.
    let index_size = if version == 1 || version == 2 {
        sizes & 0x0f
    } else {
        0
    };

    let count = match version {
        0 | 1 => u32::from(payload.u16()?),
        2 => payload.u32()?,
        other => {
            return Err(PixelsError::malformed(
                "avif",
                format!("iloc version {other} is not one this format defines"),
            ));
        }
    };

    // Every row costs at least an ID, a data reference index and an extent
    // count: six bytes on version 0. Bound before reserving.
    if u64::from(count) * 6 > payload.remaining() as u64 {
        return Err(PixelsError::malformed(
            "avif",
            format!(
                "iloc declares {count} items, more than its {} remaining bytes can hold",
                payload.remaining()
            ),
        ));
    }

    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let id = if version < 2 {
            u32::from(payload.u16()?)
        } else {
            payload.u32()?
        };

        let construction = if version == 1 || version == 2 {
            // Twelve reserved bits then a four-bit construction method.
            let word = payload.u16()?;
            match word & 0x0f {
                0 => Construction::File,
                1 => Construction::Idat,
                2 => Construction::Item,
                other => {
                    return Err(PixelsError::malformed(
                        "avif",
                        format!("iloc construction method {other} is not one this format defines"),
                    ));
                }
            }
        } else {
            Construction::File
        };

        let _data_reference_index = payload.u16()?;
        let base_offset = payload.uint(base_offset_size)?;
        let extent_count = payload.u16()?;

        // An extent costs at least the offset and length widths declared in
        // the header, so a zero-width declaration would let a huge count cost
        // nothing to declare. Charge a minimum of one byte per extent.
        let per_extent = u64::from(offset_size)
            .saturating_add(u64::from(length_size))
            .saturating_add(u64::from(index_size))
            .max(1);
        if u64::from(extent_count) * per_extent > payload.remaining() as u64 {
            return Err(PixelsError::malformed(
                "avif",
                format!(
                    "item {id} declares {extent_count} extents, more than its {} remaining bytes can hold",
                    payload.remaining()
                ),
            ));
        }

        let mut extents = Vec::with_capacity(usize::from(extent_count));
        for _ in 0..extent_count {
            if index_size > 0 {
                let _extent_index = payload.uint(index_size)?;
            }
            let offset = payload.uint(offset_size)?;
            let length = payload.uint(length_size)?;
            // The base offset is added here so that everything downstream sees
            // one absolute number.
            let offset = base_offset.checked_add(offset).ok_or_else(|| {
                PixelsError::malformed(
                    "avif",
                    format!("item {id} has an extent offset that overflows its base"),
                )
            })?;
            extents.push(Extent { offset, length });
        }

        out.push(Located {
            id,
            construction,
            extents,
        });
    }
    Ok(out)
}

/// Parse `iref`, the item reference box.
fn parse_iref(mut payload: Reader<'_>) -> Result<Vec<Reference>> {
    let (version, _flags) = payload.full_box()?;
    let mut out = Vec::new();

    while let Some(header) = payload.next_box() {
        let header = header?;
        let mut reference = payload.payload(&header);
        // Version 0 uses 16-bit item IDs throughout, version 1 uses 32-bit.
        let from = if version == 0 {
            u32::from(reference.u16()?)
        } else {
            reference.u32()?
        };
        let count = reference.u16()?;
        let width = if version == 0 { 2_u64 } else { 4 };
        if u64::from(count) * width > reference.remaining() as u64 {
            return Err(PixelsError::malformed(
                "avif",
                format!(
                    "a '{}' reference declares {count} targets, more than its {} remaining bytes can hold",
                    header.kind,
                    reference.remaining()
                ),
            ));
        }
        let mut to = Vec::with_capacity(usize::from(count));
        for _ in 0..count {
            to.push(if version == 0 {
                u32::from(reference.u16()?)
            } else {
                reference.u32()?
            });
        }
        out.push(Reference {
            kind: header.kind,
            from,
            to,
        });
    }
    Ok(out)
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

    /// An `infe` version 2 entry.
    fn infe(id: u16, kind: &[u8; 4], name: &str, hidden: bool) -> Vec<u8> {
        let mut payload = vec![2, 0, 0, if hidden { 1 } else { 0 }];
        payload.extend_from_slice(&id.to_be_bytes());
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(kind);
        payload.extend_from_slice(name.as_bytes());
        payload.push(0);
        boxed(b"infe", &payload)
    }

    /// An `iinf` version 0 wrapping the given entries.
    fn iinf(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&u16::try_from(entries.len()).unwrap().to_be_bytes());
        for entry in entries {
            payload.extend_from_slice(entry);
        }
        boxed(b"iinf", &payload)
    }

    /// An `iloc` version 0 with 32-bit offsets and lengths, one extent each.
    fn iloc(rows: &[(u16, u32, u32)]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        payload.push(0x44); // offset_size 4, length_size 4
        payload.push(0x00); // base_offset_size 0, reserved
        payload.extend_from_slice(&u16::try_from(rows.len()).unwrap().to_be_bytes());
        for (id, offset, length) in rows {
            payload.extend_from_slice(&id.to_be_bytes());
            payload.extend_from_slice(&[0, 0]); // data reference index
            payload.extend_from_slice(&1_u16.to_be_bytes()); // extent count
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

    fn meta_box(children: &[Vec<u8>]) -> Vec<u8> {
        let mut payload = vec![0, 0, 0, 0];
        for child in children {
            payload.extend_from_slice(child);
        }
        boxed(b"meta", &payload)
    }

    fn parse(file: &[u8]) -> Result<Meta> {
        let mut reader = Reader::new(file);
        let header = reader.next_box().unwrap().unwrap();
        Meta::parse(reader.payload(&header))
    }

    #[test]
    fn joins_the_type_table_with_the_extent_table() {
        let file = meta_box(&[
            pitm(1),
            iinf(&[infe(1, b"av01", "color", false)]),
            iloc(&[(1, 100, 42)]),
        ]);
        let meta = parse(&file).unwrap();

        assert_eq!(meta.primary, Some(1));
        assert_eq!(meta.items.len(), 1);
        let item = &meta.items[0];
        assert_eq!(item.id, 1);
        assert!(item.is_coded_image());
        assert_eq!(item.name, "color");
        assert!(!item.hidden);
        assert_eq!(item.construction, Construction::File);
        assert_eq!(
            item.extents,
            vec![Extent {
                offset: 100,
                length: 42
            }]
        );
    }

    /// An item listed in `iinf` but absent from `iloc` has no bytes, and one
    /// in `iloc` but not `iinf` has no type. Neither is decodable, so neither
    /// survives the join.
    #[test]
    fn an_item_missing_from_either_table_is_dropped() {
        let file = meta_box(&[
            iinf(&[infe(1, b"av01", "a", false), infe(2, b"av01", "b", false)]),
            iloc(&[(1, 0, 4), (3, 0, 4)]),
        ]);
        let meta = parse(&file).unwrap();
        assert_eq!(meta.items.len(), 1);
        assert_eq!(meta.items[0].id, 1);
    }

    /// Build a file whose `meta` box is followed by `data`, with `build`
    /// given the absolute offset `data` will land at.
    ///
    /// The offset has to be known before the box is built, but the box's
    /// length depends only on its shape and not on the offset value, so
    /// building it twice settles the circularity.
    fn file_with_data(build: impl Fn(u32) -> Vec<u8>, data: &[u8]) -> Vec<u8> {
        let probe = build(0);
        let base = u32::try_from(probe.len()).unwrap();
        let mut file = build(base);
        assert_eq!(file.len(), probe.len(), "the offset changed the box length");
        file.extend_from_slice(data);
        file
    }

    #[test]
    fn resolves_item_data_by_borrowing_a_single_extent() {
        let file = file_with_data(
            |base| meta_box(&[iinf(&[infe(1, b"av01", "", false)]), iloc(&[(1, base, 3)])]),
            &[0xDE, 0xAD, 0xBE],
        );

        let meta = parse(&file).unwrap();
        let data = meta.item_data(&file, &meta.items[0]).unwrap();
        assert_eq!(&*data, &[0xDE, 0xAD, 0xBE]);
        assert!(matches!(data, Cow::Borrowed(_)));
    }

    #[test]
    fn concatenates_multiple_extents() {
        // Two extents, two bytes each, with two bytes of filler between them.
        let file = file_with_data(
            |base| {
                let mut payload = vec![0, 0, 0, 0, 0x44, 0x00];
                payload.extend_from_slice(&1_u16.to_be_bytes()); // one item
                payload.extend_from_slice(&1_u16.to_be_bytes()); // item ID 1
                payload.extend_from_slice(&[0, 0]);
                payload.extend_from_slice(&2_u16.to_be_bytes()); // two extents
                payload.extend_from_slice(&base.to_be_bytes());
                payload.extend_from_slice(&2_u32.to_be_bytes());
                payload.extend_from_slice(&(base + 4).to_be_bytes());
                payload.extend_from_slice(&2_u32.to_be_bytes());
                meta_box(&[
                    iinf(&[infe(1, b"av01", "", false)]),
                    boxed(b"iloc", &payload),
                ])
            },
            &[1, 2, 0xFF, 0xFF, 3, 4],
        );

        let meta = parse(&file).unwrap();
        let data = meta.item_data(&file, &meta.items[0]).unwrap();
        assert_eq!(&*data, &[1, 2, 3, 4]);
        assert!(matches!(data, Cow::Owned(_)));
    }

    #[test]
    fn an_extent_outside_the_file_is_rejected() {
        let file = meta_box(&[
            iinf(&[infe(1, b"av01", "", false)]),
            iloc(&[(1, 0, 100_000)]),
        ]);
        let meta = parse(&file).unwrap();
        let error = meta.item_data(&file, &meta.items[0]).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
    }

    /// Extents may repeat, so the sum of their lengths can exceed the file
    /// even when each is individually in range. Without the total check, a
    /// small file could name an arbitrarily large allocation.
    #[test]
    fn repeated_extents_cannot_sum_past_the_file() {
        let mut payload = vec![0, 0, 0, 0, 0x44, 0x00];
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&64_u16.to_be_bytes()); // 64 extents
        for _ in 0..64 {
            payload.extend_from_slice(&0_u32.to_be_bytes());
            payload.extend_from_slice(&40_u32.to_be_bytes());
        }
        let iloc = boxed(b"iloc", &payload);
        let file = meta_box(&[iinf(&[infe(1, b"av01", "", false)]), iloc]);

        let meta = parse(&file).unwrap();
        let error = meta.item_data(&file, &meta.items[0]).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("more than the"), "{error}");
    }

    #[test]
    fn idat_construction_resolves_against_the_idat_box() {
        // iloc version 1 carries a construction method.
        let mut payload = vec![1, 0, 0, 0, 0x44, 0x00];
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes()); // item ID
        payload.extend_from_slice(&1_u16.to_be_bytes()); // construction: idat
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&1_u16.to_be_bytes()); // one extent
        payload.extend_from_slice(&2_u32.to_be_bytes()); // offset within idat
        payload.extend_from_slice(&2_u32.to_be_bytes());
        let iloc = boxed(b"iloc", &payload);

        let idat = boxed(b"idat", &[9, 9, 0xC0, 0xDE]);
        let file = meta_box(&[iinf(&[infe(1, b"grid", "", false)]), iloc, idat]);

        let meta = parse(&file).unwrap();
        let data = meta.item_data(&file, &meta.items[0]).unwrap();
        assert_eq!(&*data, &[0xC0, 0xDE]);
    }

    #[test]
    fn item_relative_construction_is_unsupported_rather_than_guessed() {
        let mut payload = vec![1, 0, 0, 0, 0x44, 0x00];
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&2_u16.to_be_bytes()); // construction: item
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&0_u32.to_be_bytes());
        payload.extend_from_slice(&1_u32.to_be_bytes());
        let iloc = boxed(b"iloc", &payload);
        let file = meta_box(&[iinf(&[infe(1, b"av01", "", false)]), iloc]);

        let meta = parse(&file).unwrap();
        let error = meta.item_data(&file, &meta.items[0]).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Unsupported);
    }

    #[test]
    fn references_are_read_and_queryable() {
        let mut dimg = vec![];
        dimg.extend_from_slice(&1_u16.to_be_bytes()); // from item 1
        dimg.extend_from_slice(&2_u16.to_be_bytes()); // two targets
        dimg.extend_from_slice(&2_u16.to_be_bytes());
        dimg.extend_from_slice(&3_u16.to_be_bytes());

        let mut payload = vec![0, 0, 0, 0];
        payload.extend_from_slice(&boxed(b"dimg", &dimg));
        let iref = boxed(b"iref", &payload);

        let file = meta_box(&[
            iinf(&[infe(1, b"grid", "", false)]),
            iloc(&[(1, 0, 1)]),
            iref,
        ]);
        let meta = parse(&file).unwrap();
        assert_eq!(meta.referenced(1, b"dimg"), &[2, 3]);
        assert_eq!(meta.referenced(1, b"auxl"), &[] as &[u32]);
        assert_eq!(meta.referenced(9, b"dimg"), &[] as &[u32]);
    }

    /// An `auxl` item whose auxiliary type is not alpha is a depth or gain
    /// map. Treating it as transparency would silently wreck the image.
    #[test]
    fn only_an_alpha_urn_makes_an_auxiliary_item_the_alpha_plane() {
        fn build(urn: &str) -> Vec<u8> {
            let mut auxl = vec![];
            auxl.extend_from_slice(&2_u16.to_be_bytes()); // from item 2
            auxl.extend_from_slice(&1_u16.to_be_bytes());
            auxl.extend_from_slice(&1_u16.to_be_bytes()); // to item 1
            let mut iref_payload = vec![0, 0, 0, 0];
            iref_payload.extend_from_slice(&boxed(b"auxl", &auxl));

            let mut auxc_payload = vec![0, 0, 0, 0];
            auxc_payload.extend_from_slice(urn.as_bytes());
            auxc_payload.push(0);
            let ipco = boxed(b"auxC", &auxc_payload);
            // Item 2 associates property 1.
            let ipma_payload = vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 0x01];
            let mut iprp = boxed(b"ipco", &ipco);
            iprp.extend_from_slice(&boxed(b"ipma", &ipma_payload));

            meta_box(&[
                pitm(1),
                iinf(&[
                    infe(1, b"av01", "color", false),
                    infe(2, b"av01", "alpha", true),
                ]),
                iloc(&[(1, 0, 1), (2, 0, 1)]),
                boxed(b"iref", &iref_payload),
                boxed(b"iprp", &iprp),
            ])
        }

        let file = build(URN_ALPHA);
        let meta = parse(&file).unwrap();
        assert_eq!(meta.alpha_item(1).map(|item| item.id), Some(2));

        let file = build(URN_ALPHA_LEGACY);
        let meta = parse(&file).unwrap();
        assert_eq!(meta.alpha_item(1).map(|item| item.id), Some(2));

        let file = build("urn:mpeg:mpegB:cicp:systems:auxiliary:depth");
        let meta = parse(&file).unwrap();
        assert!(
            meta.alpha_item(1).is_none(),
            "a depth map must not be mistaken for an alpha plane"
        );
    }

    #[test]
    fn a_missing_pitm_falls_back_to_the_first_image_item() {
        let file = meta_box(&[iinf(&[infe(4, b"av01", "", false)]), iloc(&[(4, 0, 1)])]);
        let meta = parse(&file).unwrap();
        assert_eq!(meta.primary, None);
        assert_eq!(meta.primary_item().map(|item| item.id), Some(4));
    }

    #[test]
    fn a_huge_declared_item_count_is_rejected_against_the_box_size() {
        let mut payload = vec![0, 0, 0, 0, 0x44, 0x00];
        payload.extend_from_slice(&0xFFFF_u16.to_be_bytes());
        let file = meta_box(&[boxed(b"iloc", &payload)]);
        let error = parse(&file).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("more than its"), "{error}");
    }

    #[test]
    fn an_infe_too_old_to_carry_a_type_is_rejected() {
        let mut payload = vec![1, 0, 0, 0];
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&[0, 0]);
        let file = meta_box(&[iinf(&[boxed(b"infe", &payload)])]);
        let error = parse(&file).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("item type"), "{error}");
    }

    #[test]
    fn an_undefined_construction_method_is_rejected() {
        let mut payload = vec![1, 0, 0, 0, 0x44, 0x00];
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&7_u16.to_be_bytes()); // no such method
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&0_u16.to_be_bytes());
        let file = meta_box(&[boxed(b"iloc", &payload)]);
        let error = parse(&file).unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed);
        assert!(error.to_string().contains("construction method"), "{error}");
    }
}
