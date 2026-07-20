//! Tiles: strided views over a rectangular region of pixels.
//!
//! A tile is the unit of work that moves through the graph. In M1 the naive
//! evaluator uses one whole-image tile per node; in M2 the scheduler will
//! subdivide into negotiated strips and squares (ADR-0003). Nothing in this
//! module assumes either shape — a tile is defined purely by its [`Region`],
//! [`PixelFormat`] and row stride.
//!
//! Rows are addressed in **image coordinates**, not tile-relative ones: a tile
//! covering `Region::new(10, 20, 8, 8)` answers `row(20)` through `row(27)`.
//! Ops translate between input and output coordinates through the regions they
//! are handed, so absolute addressing removes a whole class of off-by-origin
//! bugs from kernels.

use crate::{ImageDescriptor, PixelFormat, PixelsError, Region, Result};

/// Validate that `data` can back `region` at `stride`, returning the row length.
///
/// A buffer must hold every full row plus the packed bytes of the final row;
/// trailing stride padding after the last row is not required.
fn validate(
    region: Region,
    pixel: PixelFormat,
    stride: usize,
    data_len: usize,
) -> Result<usize> {
    let row_bytes = (region.width as usize)
        .checked_mul(pixel.bytes_per_pixel())
        .ok_or_else(|| PixelsError::invalid_argument("region", "row byte length overflows"))?;
    if stride < row_bytes {
        return Err(PixelsError::invalid_argument(
            "stride",
            format!("stride {stride} is shorter than a {row_bytes}-byte row"),
        ));
    }
    if region.is_empty() {
        return Ok(row_bytes);
    }
    let needed = (region.height as usize - 1)
        .checked_mul(stride)
        .and_then(|full| full.checked_add(row_bytes))
        .ok_or_else(|| PixelsError::invalid_argument("region", "tile byte length overflows"))?;
    if data_len < needed {
        return Err(PixelsError::invalid_argument(
            "data",
            format!("buffer of {data_len} bytes is too small for {region} (needs {needed})"),
        ));
    }
    Ok(row_bytes)
}

/// Byte offset of the row at image coordinate `y`, if the tile covers it.
fn row_offset(region: Region, stride: usize, y: u32) -> Option<usize> {
    if y < region.y || u64::from(y) >= region.bottom() {
        return None;
    }
    ((y - region.y) as usize).checked_mul(stride)
}

/// An immutable strided view over a region of pixels.
#[derive(Debug, Clone, Copy)]
pub struct Tile<'a> {
    region: Region,
    pixel: PixelFormat,
    stride: usize,
    row_bytes: usize,
    data: &'a [u8],
}

impl<'a> Tile<'a> {
    /// Wrap `data` as a tile covering `region`.
    ///
    /// `stride` is the distance in bytes between the starts of consecutive
    /// rows, and must be at least `region.width * pixel.bytes_per_pixel()`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `stride` is shorter than one
    /// packed row, or if `data` is too small to cover `region` at `stride`.
    pub fn new(
        region: Region,
        pixel: PixelFormat,
        stride: usize,
        data: &'a [u8],
    ) -> Result<Self> {
        let row_bytes = validate(region, pixel, stride, data.len())?;
        Ok(Self { region, pixel, stride, row_bytes, data })
    }

    /// The region of the image this tile covers.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }

    /// The pixel format of the samples in this tile.
    #[must_use]
    pub const fn pixel(&self) -> PixelFormat {
        self.pixel
    }

    /// Distance in bytes between the starts of consecutive rows.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Packed length in bytes of one row of this tile.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The row at image coordinate `y`, or [`None`] if outside the region.
    #[must_use]
    pub fn row(&self, y: u32) -> Option<&'a [u8]> {
        let offset = row_offset(self.region, self.stride, y)?;
        self.data.get(offset..offset.checked_add(self.row_bytes)?)
    }

    /// The tile's rows, top to bottom.
    pub fn rows(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        let first = self.region.y;
        (0..self.region.height).filter_map(move |i| self.row(first + i))
    }
}

/// A mutable strided view over a region of pixels.
///
/// This is the output half of [`Op::compute`]: the evaluator hands the kernel a
/// `TileMut` sized to the region it must fill.
///
/// [`Op::compute`]: crate::Op::compute
#[derive(Debug)]
pub struct TileMut<'a> {
    region: Region,
    pixel: PixelFormat,
    stride: usize,
    row_bytes: usize,
    data: &'a mut [u8],
}

impl<'a> TileMut<'a> {
    /// Wrap `data` as a mutable tile covering `region`.
    ///
    /// # Errors
    ///
    /// As [`Tile::new`].
    pub fn new(
        region: Region,
        pixel: PixelFormat,
        stride: usize,
        data: &'a mut [u8],
    ) -> Result<Self> {
        let row_bytes = validate(region, pixel, stride, data.len())?;
        Ok(Self { region, pixel, stride, row_bytes, data })
    }

    /// The region of the image this tile covers.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }

    /// The pixel format of the samples in this tile.
    #[must_use]
    pub const fn pixel(&self) -> PixelFormat {
        self.pixel
    }

    /// Distance in bytes between the starts of consecutive rows.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Packed length in bytes of one row of this tile.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The row at image coordinate `y`, or [`None`] if outside the region.
    #[must_use]
    pub fn row_mut(&mut self, y: u32) -> Option<&mut [u8]> {
        let offset = row_offset(self.region, self.stride, y)?;
        let end = offset.checked_add(self.row_bytes)?;
        self.data.get_mut(offset..end)
    }

    /// The tile's rows, top to bottom, mutably.
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [u8]> {
        let (stride, row_bytes, height) = (self.stride, self.row_bytes, self.region.height as usize);
        self.data.chunks_mut(stride).take(height).filter_map(move |chunk| chunk.get_mut(..row_bytes))
    }

    /// Reborrow as an immutable tile.
    #[must_use]
    pub fn as_ref(&self) -> Tile<'_> {
        Tile {
            region: self.region,
            pixel: self.pixel,
            stride: self.stride,
            row_bytes: self.row_bytes,
            data: self.data,
        }
    }
}

/// An owned, densely packed buffer of pixels for one region.
///
/// This is what the M1 evaluator materializes per node. It is deliberately
/// simple: M2 replaces most uses with scheduler-owned tile memory, but the
/// buffer's view API ([`TileBuf::as_tile`], [`TileBuf::as_tile_mut`]) is the
/// same one kernels already program against, so kernels do not change.
#[derive(Debug, Clone)]
pub struct TileBuf {
    region: Region,
    pixel: PixelFormat,
    stride: usize,
    data: Vec<u8>,
}

impl TileBuf {
    /// Allocate a zeroed, densely packed buffer covering `region`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if the region's byte length
    /// overflows `usize` on this platform.
    pub fn zeroed(region: Region, pixel: PixelFormat) -> Result<Self> {
        let stride = (region.width as usize)
            .checked_mul(pixel.bytes_per_pixel())
            .ok_or_else(|| PixelsError::invalid_argument("region", "row length overflows"))?;
        let len = (region.height as usize)
            .checked_mul(stride)
            .ok_or_else(|| PixelsError::invalid_argument("region", "tile length overflows"))?;
        Ok(Self { region, pixel, stride, data: vec![0; len] })
    }

    /// Allocate a zeroed buffer covering the whole of `desc`.
    ///
    /// # Errors
    ///
    /// As [`TileBuf::zeroed`].
    pub fn for_image(desc: &ImageDescriptor) -> Result<Self> {
        Self::zeroed(desc.region(), desc.pixel)
    }

    /// Take ownership of `data` as the pixels of `region`, densely packed.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `data` is not exactly the
    /// packed byte length of `region`.
    pub fn from_vec(region: Region, pixel: PixelFormat, data: Vec<u8>) -> Result<Self> {
        let stride = (region.width as usize)
            .checked_mul(pixel.bytes_per_pixel())
            .ok_or_else(|| PixelsError::invalid_argument("region", "row length overflows"))?;
        let expected = (region.height as usize)
            .checked_mul(stride)
            .ok_or_else(|| PixelsError::invalid_argument("region", "tile length overflows"))?;
        if data.len() != expected {
            return Err(PixelsError::invalid_argument(
                "data",
                format!("expected exactly {expected} packed bytes, got {}", data.len()),
            ));
        }
        Ok(Self { region, pixel, stride, data })
    }

    /// The region this buffer covers.
    #[must_use]
    pub const fn region(&self) -> Region {
        self.region
    }

    /// The pixel format of the samples in this buffer.
    #[must_use]
    pub const fn pixel(&self) -> PixelFormat {
        self.pixel
    }

    /// The packed bytes of this buffer.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the buffer, returning its packed bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }

    /// Borrow the buffer as an immutable tile.
    ///
    /// # Errors
    ///
    /// Only if the buffer's own invariants were violated, which construction
    /// prevents.
    pub fn as_tile(&self) -> Result<Tile<'_>> {
        Tile::new(self.region, self.pixel, self.stride, &self.data)
    }

    /// Borrow the buffer as a mutable tile.
    ///
    /// # Errors
    ///
    /// As [`TileBuf::as_tile`].
    pub fn as_tile_mut(&mut self) -> Result<TileMut<'_>> {
        TileMut::new(self.region, self.pixel, self.stride, &mut self.data)
    }
}

/// Copy the pixels of `region` from `src` into `dst`.
///
/// Both tiles address rows in image coordinates, so this is the general
/// "move a rectangle between two views of the same image space" primitive:
/// it handles differing origins and strides without either side knowing the
/// other's layout.
///
/// # Errors
///
/// Returns [`PixelsError::InvalidArgument`] if the tiles disagree on pixel
/// format, or if `region` is not covered by both tiles.
pub fn copy_region(src: &Tile<'_>, dst: &mut TileMut<'_>, region: Region) -> Result<()> {
    if src.pixel() != dst.pixel() {
        return Err(PixelsError::invalid_argument(
            "pixel",
            format!("cannot copy {} pixels into a {} tile", src.pixel(), dst.pixel()),
        ));
    }
    if !src.region().contains(region) {
        return Err(PixelsError::invalid_argument(
            "region",
            format!("source tile {} does not cover {region}", src.region()),
        ));
    }
    if !dst.region().contains(region) {
        return Err(PixelsError::invalid_argument(
            "region",
            format!("destination tile {} does not cover {region}", dst.region()),
        ));
    }
    if region.is_empty() {
        return Ok(());
    }
    let bpp = src.pixel().bytes_per_pixel();
    let span = region.width as usize * bpp;
    let src_offset = (region.x - src.region().x) as usize * bpp;
    let dst_offset = (region.x - dst.region().x) as usize * bpp;
    for y in region.y..region.y.saturating_add(region.height) {
        let (Some(src_row), Some(dst_row)) = (src.row(y), dst.row_mut(y)) else {
            return Err(PixelsError::invalid_argument("region", format!("row {y} is out of range")));
        };
        let (Some(from), Some(into)) = (
            src_row.get(src_offset..src_offset + span),
            dst_row.get_mut(dst_offset..dst_offset + span),
        ) else {
            return Err(PixelsError::invalid_argument(
                "region",
                format!("row {y} is too short for {region}"),
            ));
        };
        into.copy_from_slice(from);
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests operate on known-good values and assert shapes directly")]
mod tests {
    use super::*;

    fn buf(width: u32, height: u32) -> TileBuf {
        TileBuf::zeroed(Region::from_size(width, height), PixelFormat::Rgb8).unwrap()
    }

    #[test]
    fn rows_are_addressed_in_image_coordinates() {
        let region = Region::new(10, 20, 2, 3);
        let data = vec![7_u8; 2 * 3 * 3];
        let tile = Tile::new(region, PixelFormat::Rgb8, 6, &data).unwrap();
        assert!(tile.row(19).is_none(), "above the tile");
        assert!(tile.row(20).is_some(), "first row");
        assert!(tile.row(22).is_some(), "last row");
        assert!(tile.row(23).is_none(), "below the tile");
        assert_eq!(tile.rows().count(), 3);
        assert_eq!(tile.row(20).unwrap().len(), 6);
    }

    #[test]
    fn stride_padding_is_skipped_by_row_views() {
        // 2px RGB8 rows (6 bytes) padded to a stride of 8.
        let data: Vec<u8> = (0..16).collect();
        let tile = Tile::new(Region::from_size(2, 2), PixelFormat::Rgb8, 8, &data).unwrap();
        assert_eq!(tile.row(0).unwrap(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(tile.row(1).unwrap(), &[8, 9, 10, 11, 12, 13]);
        assert_eq!(tile.stride(), 8);
        assert_eq!(tile.row_bytes(), 6);
    }

    #[test]
    fn a_buffer_one_byte_short_is_an_error_not_a_panic() {
        let data = vec![0_u8; 6 * 3 - 1];
        let err = Tile::new(Region::from_size(6, 3), PixelFormat::Gray8, 6, &data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidArgument);
    }

    #[test]
    fn stride_shorter_than_a_row_is_rejected() {
        let data = vec![0_u8; 64];
        let err = Tile::new(Region::from_size(4, 2), PixelFormat::Rgb8, 11, &data).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidArgument);
    }

    #[test]
    fn trailing_stride_padding_is_not_required() {
        // 3 rows at stride 8 but only 6 packed bytes needed in the last row.
        let data = vec![0_u8; 8 * 2 + 6];
        assert!(Tile::new(Region::from_size(2, 3), PixelFormat::Rgb8, 8, &data).is_ok());
    }

    #[test]
    fn mutable_rows_write_through_to_the_buffer() {
        let mut b = buf(2, 2);
        {
            let mut tile = b.as_tile_mut().unwrap();
            for (i, row) in tile.rows_mut().enumerate() {
                row.fill(i as u8 + 1);
            }
        }
        assert_eq!(b.bytes(), &[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);
    }

    #[test]
    fn row_mut_addresses_absolute_coordinates() {
        let mut b = TileBuf::zeroed(Region::new(0, 5, 1, 2), PixelFormat::Gray8).unwrap();
        let mut tile = b.as_tile_mut().unwrap();
        assert!(tile.row_mut(4).is_none());
        tile.row_mut(5).unwrap().fill(9);
        tile.row_mut(6).unwrap().fill(8);
        assert!(tile.row_mut(7).is_none());
        assert_eq!(b.bytes(), &[9, 8]);
    }

    #[test]
    fn from_vec_requires_an_exactly_packed_buffer() {
        let region = Region::from_size(2, 2);
        assert!(TileBuf::from_vec(region, PixelFormat::Gray8, vec![0; 4]).is_ok());
        assert!(TileBuf::from_vec(region, PixelFormat::Gray8, vec![0; 3]).is_err());
        assert!(TileBuf::from_vec(region, PixelFormat::Gray8, vec![0; 5]).is_err());
    }

    #[test]
    fn empty_regions_are_representable() {
        let tile = Tile::new(Region::EMPTY, PixelFormat::Rgb8, 0, &[]).unwrap();
        assert_eq!(tile.rows().count(), 0);
        assert!(tile.row(0).is_none());
        assert_eq!(TileBuf::zeroed(Region::EMPTY, PixelFormat::Rgb8).unwrap().bytes().len(), 0);
    }

    #[test]
    fn copy_region_moves_a_rectangle_between_differing_origins() {
        // Source: 4x4 gray, value = y*10 + x.
        let mut src_buf = TileBuf::zeroed(Region::from_size(4, 4), PixelFormat::Gray8).unwrap();
        {
            let mut tile = src_buf.as_tile_mut().unwrap();
            for y in 0..4_u32 {
                let row = tile.row_mut(y).unwrap();
                for (x, cell) in row.iter_mut().enumerate() {
                    *cell = (y as u8) * 10 + x as u8;
                }
            }
        }
        // Destination covers only the 2x2 rectangle at (1,1).
        let mut dst_buf = TileBuf::zeroed(Region::new(1, 1, 2, 2), PixelFormat::Gray8).unwrap();
        let src = src_buf.as_tile().unwrap();
        let mut dst = dst_buf.as_tile_mut().unwrap();
        copy_region(&src, &mut dst, Region::new(1, 1, 2, 2)).unwrap();
        assert_eq!(dst_buf.bytes(), &[11, 12, 21, 22]);
    }

    #[test]
    fn copy_region_rejects_uncovered_or_mismatched_requests() {
        let src_buf = TileBuf::zeroed(Region::from_size(4, 4), PixelFormat::Gray8).unwrap();
        let mut dst_buf = TileBuf::zeroed(Region::from_size(4, 4), PixelFormat::Gray8).unwrap();
        let src = src_buf.as_tile().unwrap();
        {
            let mut dst = dst_buf.as_tile_mut().unwrap();
            // Region extends past the source.
            let err = copy_region(&src, &mut dst, Region::new(2, 2, 4, 4)).unwrap_err();
            assert_eq!(err.code(), crate::ErrorCode::InvalidArgument);
            // Empty regions are a no-op, not an error.
            copy_region(&src, &mut dst, Region::EMPTY).unwrap();
        }
        // Mismatched pixel formats.
        let mut rgb = TileBuf::zeroed(Region::from_size(4, 4), PixelFormat::Rgb8).unwrap();
        let src2 = src_buf.as_tile().unwrap();
        let mut dst2 = rgb.as_tile_mut().unwrap();
        let err = copy_region(&src2, &mut dst2, Region::from_size(4, 4)).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::InvalidArgument);
    }

    #[test]
    fn copy_region_survives_stride_padding_on_both_sides() {
        let src_data: Vec<u8> = (0..16).collect();
        let src = Tile::new(Region::from_size(2, 4), PixelFormat::Gray8, 4, &src_data).unwrap();
        let mut dst_data = vec![0_u8; 16];
        let mut dst = TileMut::new(Region::from_size(2, 4), PixelFormat::Gray8, 4, &mut dst_data)
            .unwrap();
        copy_region(&src, &mut dst, Region::from_size(2, 4)).unwrap();
        // Packed pixels copied; stride padding left untouched.
        assert_eq!(&dst_data[0..2], &[0, 1]);
        assert_eq!(&dst_data[4..6], &[4, 5]);
        assert_eq!(&dst_data[2..4], &[0, 0], "padding is not written");
    }

    #[test]
    fn as_ref_preserves_the_view() {
        let mut b = buf(2, 2);
        let tile = b.as_tile_mut().unwrap();
        let view = tile.as_ref();
        assert_eq!(view.region(), Region::from_size(2, 2));
        assert_eq!(view.pixel(), PixelFormat::Rgb8);
        assert_eq!(view.stride(), 6);
    }
}
