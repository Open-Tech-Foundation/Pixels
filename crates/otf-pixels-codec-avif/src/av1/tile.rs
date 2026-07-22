//! The tile decode driver for the lossless intra path (spec §5.11).
//!
//! This is where the pieces meet: the arithmetic decoder walks the partition
//! tree, reads each block's intra mode, predicts every 4x4 transform block from
//! the already-reconstructed neighbours, decodes its coefficients, inverts the
//! transform, and writes the samples back. The neighbour-context arrays it
//! threads between blocks are what make the entropy contexts match the encoder.
//!
//! Scope is the lossless still-image subset: a single tile, `CodedLossless`
//! (every transform is 4x4 WHT, no post-filters), and YUV 4:4:4. All the intra
//! modes a natural image uses are handled — DC, Paeth, the smooth family, the
//! slanted directional modes (with their edge-filter and upsample machinery),
//! and recursive filter-intra. The screen-content tools (palette, intra block
//! copy) and chroma-from-luma are detected and reported as
//! [`PixelsError::unsupported`] rather than decoded wrong, so a stream that uses
//! them fails cleanly instead of desynchronising.

use super::cdf;
use super::coeff::{CoeffCdfs, decode_coeffs_4x4};
use super::direction::{
    ANGLE_STEP, Edge, mode_base_angle, predict_directional_4x4, predict_filter_intra_4x4,
};
use super::frame::FrameHeader;
use super::plane::Plane;
use super::predict::{IntraMode, Neighbours, predict_intra_4x4};
use super::seq::SequenceHeader;
use super::symbol::SymbolDecoder;
use super::transform::{add_residual_4x4, inverse_wht_4x4};
use otf_pixels_core::{PixelsError, Result};

/// `MI_SIZE` (§3): the side of the smallest coded block, in samples.
const MI_SIZE: usize = 4;
/// `DC_PRED` mode index.
const DC_PRED: usize = 0;
/// `UV_CFL_PRED`: the chroma-from-luma UV mode, one past the intra modes.
const UV_CFL_PRED: usize = 13;
/// `MAX_ANGLE_DELTA` (§3).
const MAX_ANGLE_DELTA: i32 = 3;

/// `Intra_Mode_Context` (§8.3.2): folds an intra mode into the small context
/// used to select the key-frame Y-mode CDF.
const INTRA_MODE_CONTEXT: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// The partition types (§6.10.4), in coded order.
const PARTITION_NONE: usize = 0;
const PARTITION_HORZ: usize = 1;
const PARTITION_VERT: usize = 2;
const PARTITION_SPLIT: usize = 3;
const PARTITION_HORZ_A: usize = 4;
const PARTITION_HORZ_B: usize = 5;
const PARTITION_VERT_A: usize = 6;
const PARTITION_VERT_B: usize = 7;
const PARTITION_HORZ_4: usize = 8;
const PARTITION_VERT_4: usize = 9;

/// A decoded frame's sample planes, in coded order (Y, U, V for 4:4:4).
pub struct DecodedFrame {
    /// The reconstructed planes.
    pub planes: Vec<Plane>,
}

/// Decode a single-tile lossless still frame into its sample planes.
///
/// # Errors
///
/// Returns [`PixelsError::unsupported`] for anything outside the lossless 4:4:4
/// intra subset, and [`PixelsError::malformed`] for a stream that ends early or
/// violates the syntax.
pub fn decode_still(
    seq: &SequenceHeader,
    frame: &FrameHeader,
    tile_data: &[u8],
) -> Result<DecodedFrame> {
    if !frame.coded_lossless {
        return Err(PixelsError::unsupported(
            "avif: only the lossless intra path is implemented",
        ));
    }
    if seq.color.subsampling_x != 0 || seq.color.subsampling_y != 0 {
        return Err(PixelsError::unsupported(
            "avif: only 4:4:4 is implemented in the lossless path",
        ));
    }
    if frame.tile_info.count() != 1 {
        return Err(PixelsError::unsupported(
            "avif: multi-tile decode is not implemented yet",
        ));
    }
    if frame.allow_screen_content_tools {
        // Palette and intra block copy would be coded, and neither is
        // implemented; decoding anyway would desynchronise the symbol stream.
        return Err(PixelsError::unsupported(
            "avif: screen-content tools (palette, intra block copy) are not implemented yet",
        ));
    }

    let mut state = TileState::new(seq, frame)?;
    state.decode(tile_data)?;
    Ok(DecodedFrame {
        planes: state.planes,
    })
}

/// Mutable CDFs for the frame, cloned from the defaults and adapted as symbols
/// are read. Only the tables the lossless intra path exercises are held.
struct FrameCdfs {
    partition_w8: [[u16; 5]; 4],
    partition_w16: [[u16; 11]; 4],
    partition_w32: [[u16; 11]; 4],
    partition_w64: [[u16; 11]; 4],
    partition_w128: [[u16; 9]; 4],
    skip: [[u16; 3]; 3],
    intra_frame_y_mode: [[[u16; 14]; 5]; 5],
    uv_cfl_allowed: [[u16; 15]; 13],
    uv_cfl_not_allowed: [[u16; 14]; 13],
    angle_delta: [[u16; 8]; 8],
    filter_intra: [[u16; 3]; 22],
    filter_intra_mode: [u16; 6],
    coeff: CoeffCdfs,
}

impl FrameCdfs {
    fn new(qctx: usize) -> Self {
        Self {
            partition_w8: cdf::DEFAULT_PARTITION_W8_CDF,
            partition_w16: cdf::DEFAULT_PARTITION_W16_CDF,
            partition_w32: cdf::DEFAULT_PARTITION_W32_CDF,
            partition_w64: cdf::DEFAULT_PARTITION_W64_CDF,
            partition_w128: cdf::DEFAULT_PARTITION_W128_CDF,
            skip: cdf::DEFAULT_SKIP_CDF,
            intra_frame_y_mode: cdf::DEFAULT_INTRA_FRAME_Y_MODE_CDF,
            uv_cfl_allowed: cdf::DEFAULT_UV_MODE_CFL_ALLOWED_CDF,
            uv_cfl_not_allowed: cdf::DEFAULT_UV_MODE_CFL_NOT_ALLOWED_CDF,
            angle_delta: cdf::DEFAULT_ANGLE_DELTA_CDF,
            filter_intra: cdf::DEFAULT_FILTER_INTRA_CDF,
            filter_intra_mode: cdf::DEFAULT_FILTER_INTRA_MODE_CDF,
            coeff: CoeffCdfs::new(qctx),
        }
    }
}

/// Per-plane neighbour level and DC-sign context arrays (`AboveLevelContext`
/// and friends), one entry per 4-sample column or row.
struct LevelContext {
    above_level: Vec<u8>,
    above_dc: Vec<u8>,
    left_level: Vec<u8>,
    left_dc: Vec<u8>,
}

/// The whole mutable state of a tile decode.
struct TileState {
    planes: Vec<Plane>,
    cdfs: FrameCdfs,
    bit_depth: u8,
    num_planes: usize,
    mi_cols: usize,
    mi_rows: usize,
    enable_filter_intra: bool,
    enable_edge_filter: bool,
    sb_size4: usize,
    /// `BlockDecoded[plane]`, one flat `(sb+2) x (sb+2)` grid per plane, reset
    /// per superblock; addressed with a one-unit border so index -1 is valid.
    block_decoded: Vec<Vec<u8>>,
    /// `YModes[r][c]` flattened row-major, one entry per 4x4 unit.
    y_modes: Vec<u8>,
    /// `UVModes[r][c]` flattened, for the intra filter-type decision.
    uv_modes: Vec<u8>,
    /// `Skips[r][c]` flattened.
    skips: Vec<u8>,
    /// `Mi_Width_Log2` of the block owning each 4x4 unit (for partition ctx).
    mi_wide_log2: Vec<u8>,
    /// `Mi_Height_Log2` of the block owning each 4x4 unit.
    mi_high_log2: Vec<u8>,
    /// Level contexts, one per plane.
    ctx: Vec<LevelContext>,
}

impl TileState {
    fn new(seq: &SequenceHeader, frame: &FrameHeader) -> Result<Self> {
        let mi_cols = frame.mi_cols as usize;
        let mi_rows = frame.mi_rows as usize;
        let num_planes = seq.color.num_planes as usize;
        let width = mi_cols * MI_SIZE;
        let height = mi_rows * MI_SIZE;
        let planes = (0..num_planes).map(|_| Plane::new(width, height)).collect();
        let ctx = (0..num_planes)
            .map(|_| LevelContext {
                above_level: vec![0; mi_cols],
                above_dc: vec![0; mi_cols],
                left_level: vec![0; mi_rows],
                left_dc: vec![0; mi_rows],
            })
            .collect();
        let sb_size4 = if seq.use_128x128_superblock { 32 } else { 16 };
        let bd_stride = sb_size4 + 2;
        let block_decoded = (0..num_planes)
            .map(|_| vec![0; bd_stride * bd_stride])
            .collect();
        // Lossless frames have base_q_idx == 0, so the quantiser context is 0.
        Ok(Self {
            planes,
            cdfs: FrameCdfs::new(0),
            bit_depth: seq.color.bit_depth,
            num_planes,
            mi_cols,
            mi_rows,
            enable_filter_intra: seq.enable_filter_intra,
            enable_edge_filter: seq.enable_intra_edge_filter,
            sb_size4,
            block_decoded,
            y_modes: vec![0; mi_cols * mi_rows],
            uv_modes: vec![0; mi_cols * mi_rows],
            skips: vec![0; mi_cols * mi_rows],
            mi_wide_log2: vec![0; mi_cols * mi_rows],
            mi_high_log2: vec![0; mi_cols * mi_rows],
            ctx,
        })
    }

    fn decode(&mut self, tile_data: &[u8]) -> Result<()> {
        let mut dec = SymbolDecoder::new(tile_data, false)?;
        let sb_size4 = self.sb_size4;
        // Superblocks are decoded in raster order; each seeds the partition
        // recursion. The left contexts reset at the start of each SB row.
        let mut sb_row = 0;
        while sb_row < self.mi_rows {
            self.reset_left_context();
            let mut sb_col = 0;
            while sb_col < self.mi_cols {
                self.clear_block_decoded(sb_row, sb_col);
                self.decode_partition(&mut dec, sb_row, sb_col, sb_size4)?;
                sb_col += sb_size4;
            }
            sb_row += sb_size4;
        }
        Ok(())
    }

    /// `clear_block_decoded_flags` (§5.11.3) for one superblock, 4:4:4.
    fn clear_block_decoded(&mut self, r: usize, c: usize) {
        let sb = self.sb_size4;
        let stride = sb + 2;
        let sb_width4 = self.mi_cols - c;
        let sb_height4 = self.mi_rows - r;
        for plane in 0..self.num_planes {
            let Some(grid) = self.block_decoded.get_mut(plane) else {
                continue;
            };
            for v in grid.iter_mut() {
                *v = 0;
            }
            // Row above (y == -1) valid where x < sbWidth4; column left (x == -1)
            // valid where y < sbHeight4. Indices carry a +1 border.
            for x in -1_isize..=sb as isize {
                if x < sb_width4 as isize {
                    if let Some(slot) = grid.get_mut(bd_index(stride, -1, x)) {
                        *slot = 1;
                    }
                }
            }
            for y in -1_isize..=sb as isize {
                if y < sb_height4 as isize {
                    if let Some(slot) = grid.get_mut(bd_index(stride, y, -1)) {
                        *slot = 1;
                    }
                }
            }
            if let Some(slot) = grid.get_mut(bd_index(stride, sb as isize, -1)) {
                *slot = 0;
            }
        }
    }

    fn block_decoded_at(&self, plane: usize, sub_row: isize, sub_col: isize) -> bool {
        let stride = self.sb_size4 + 2;
        self.block_decoded
            .get(plane)
            .and_then(|g| g.get(bd_index(stride, sub_row, sub_col)))
            .is_some_and(|&v| v != 0)
    }

    fn set_block_decoded(&mut self, plane: usize, sub_row: isize, sub_col: isize) {
        let stride = self.sb_size4 + 2;
        if let Some(slot) = self
            .block_decoded
            .get_mut(plane)
            .and_then(|g| g.get_mut(bd_index(stride, sub_row, sub_col)))
        {
            *slot = 1;
        }
    }

    fn reset_left_context(&mut self) {
        for c in &mut self.ctx {
            for v in &mut c.left_level {
                *v = 0;
            }
            for v in &mut c.left_dc {
                *v = 0;
            }
        }
    }

    /// `decode_partition` (§5.11.4), restricted to what the lossless subset
    /// produces. `bsize4` is the block side in 4-sample units (a power of two).
    fn decode_partition(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        bsize4: usize,
    ) -> Result<()> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        let avail_u = r > 0;
        let avail_l = c > 0;
        let half = bsize4 >> 1;
        let has_rows = r + half < self.mi_rows;
        let has_cols = c + half < self.mi_cols;

        let partition = if bsize4 < 2 {
            PARTITION_NONE
        } else if has_rows && has_cols {
            self.read_partition(dec, r, c, bsize4, avail_u, avail_l)?
        } else if has_cols {
            if self.read_split_or(dec, r, c, bsize4, avail_u, avail_l, true)? {
                PARTITION_SPLIT
            } else {
                PARTITION_HORZ
            }
        } else if has_rows {
            if self.read_split_or(dec, r, c, bsize4, avail_u, avail_l, false)? {
                PARTITION_SPLIT
            } else {
                PARTITION_VERT
            }
        } else {
            PARTITION_SPLIT
        };

        let quarter = bsize4 >> 2;
        match partition {
            PARTITION_NONE => self.decode_block(dec, r, c, bsize4, bsize4)?,
            PARTITION_HORZ => {
                self.decode_block(dec, r, c, bsize4, half)?;
                if has_rows {
                    self.decode_block(dec, r + half, c, bsize4, half)?;
                }
            }
            PARTITION_VERT => {
                self.decode_block(dec, r, c, half, bsize4)?;
                if has_cols {
                    self.decode_block(dec, r, c + half, half, bsize4)?;
                }
            }
            PARTITION_SPLIT => {
                self.decode_partition(dec, r, c, half)?;
                self.decode_partition(dec, r, c + half, half)?;
                self.decode_partition(dec, r + half, c, half)?;
                self.decode_partition(dec, r + half, c + half, half)?;
            }
            PARTITION_HORZ_A => {
                self.decode_block(dec, r, c, half, half)?;
                self.decode_block(dec, r, c + half, half, half)?;
                self.decode_block(dec, r + half, c, bsize4, half)?;
            }
            PARTITION_HORZ_B => {
                self.decode_block(dec, r, c, bsize4, half)?;
                self.decode_block(dec, r + half, c, half, half)?;
                self.decode_block(dec, r + half, c + half, half, half)?;
            }
            PARTITION_VERT_A => {
                self.decode_block(dec, r, c, half, half)?;
                self.decode_block(dec, r + half, c, half, half)?;
                self.decode_block(dec, r, c + half, half, bsize4)?;
            }
            PARTITION_VERT_B => {
                self.decode_block(dec, r, c, half, bsize4)?;
                self.decode_block(dec, r, c + half, half, half)?;
                self.decode_block(dec, r + half, c + half, half, half)?;
            }
            PARTITION_HORZ_4 => {
                for k in 0..4 {
                    let rr = r + quarter * k;
                    if k == 3 && rr >= self.mi_rows {
                        break;
                    }
                    self.decode_block(dec, rr, c, bsize4, quarter)?;
                }
            }
            PARTITION_VERT_4 => {
                for k in 0..4 {
                    let cc = c + quarter * k;
                    if k == 3 && cc >= self.mi_cols {
                        break;
                    }
                    self.decode_block(dec, r, cc, quarter, bsize4)?;
                }
            }
            _ => {
                return Err(PixelsError::malformed("avif", "invalid partition type"));
            }
        }
        Ok(())
    }

    /// Read the `partition` symbol and return the partition type.
    fn read_partition(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        bsize4: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<usize> {
        let ctx = self.partition_ctx(r, c, bsize4, avail_u, avail_l);
        let bsl = floor_log2_usize(bsize4);
        let cdf_row = self.partition_cdf(bsl, ctx)?;
        dec.read_symbol(cdf_row)
    }

    /// The context for the `partition` and `split_or_*` symbols (§8.3.2).
    fn partition_ctx(
        &self,
        r: usize,
        c: usize,
        bsize4: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> usize {
        let bsl = floor_log2_usize(bsize4) as u8;
        let above = avail_u
            && r.checked_sub(1)
                .and_then(|ru| self.mi_wide_log2.get(ru * self.mi_cols + c))
                .is_some_and(|&w| w < bsl);
        let left = avail_l
            && c.checked_sub(1)
                .and_then(|cl| self.mi_high_log2.get(r * self.mi_cols + cl))
                .is_some_and(|&h| h < bsl);
        usize::from(left) * 2 + usize::from(above)
    }

    /// The mutable `partition` CDF row for `bsl` and `ctx`.
    fn partition_cdf(&mut self, bsl: u32, ctx: usize) -> Result<&mut [u16]> {
        let row: &mut [u16] = match bsl {
            1 => get_mut(&mut self.cdfs.partition_w8, ctx)?,
            2 => get_mut(&mut self.cdfs.partition_w16, ctx)?,
            3 => get_mut(&mut self.cdfs.partition_w32, ctx)?,
            4 => get_mut(&mut self.cdfs.partition_w64, ctx)?,
            _ => get_mut(&mut self.cdfs.partition_w128, ctx)?,
        };
        Ok(row)
    }

    /// Read `split_or_horz` / `split_or_vert` (§8.3.2): a binary decision built
    /// from the full partition CDF. Returns whether the partition is a split.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors the split_or_* context inputs"
    )]
    fn read_split_or(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        bsize4: usize,
        avail_u: bool,
        avail_l: bool,
        horz: bool,
    ) -> Result<bool> {
        let ctx = self.partition_ctx(r, c, bsize4, avail_u, avail_l);
        let bsl = floor_log2_usize(bsize4);
        let is_128 = bsize4 == 32;
        // Copy the partition CDF so the derived binary read does not adapt it.
        let src = self.partition_cdf(bsl, ctx)?;
        let partition_cdf: Vec<u16> = src.to_vec();
        let prob = |k: usize| -> i32 {
            let hi = partition_cdf.get(k).copied().unwrap_or(0);
            let lo = k
                .checked_sub(1)
                .and_then(|i| partition_cdf.get(i))
                .copied()
                .unwrap_or(0);
            i32::from(hi) - i32::from(lo)
        };
        // split_or_horz cannot return VERT, split_or_vert cannot return HORZ:
        // the excluded direction's mass is folded into the split probability.
        let mut psum = if horz {
            prob(PARTITION_VERT) + prob(PARTITION_SPLIT) + prob(4) + prob(6) + prob(7)
        } else {
            prob(PARTITION_HORZ) + prob(PARTITION_SPLIT) + prob(4) + prob(5) + prob(6)
        };
        if !is_128 {
            psum += if horz { prob(9) } else { prob(8) };
        }
        let mut derived = [((1 << 15) - psum) as u16, 1 << 15, 0];
        Ok(dec.read_symbol(&mut derived)? != 0)
    }

    /// `decode_block` (§5.11.5) plus mode info and residual, for one block.
    fn decode_block(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<()> {
        let avail_u = r > 0;
        let avail_l = c > 0;
        let has_chroma = self.num_planes > 1;

        // --- intra_frame_mode_info (lossless key-frame subset) ---
        let skip = self.read_skip(dec, r, c, avail_u, avail_l)?;

        let y_mode = self.read_intra_frame_y_mode(dec, r, c, avail_u, avail_l)?;
        let y_delta = self.read_angle_delta(dec, y_mode, bw4, bh4)?;

        let (uv_mode, uv_delta) = if has_chroma {
            let uv = self.read_uv_mode(dec, y_mode, bw4, bh4)?;
            let d = self.read_angle_delta(dec, uv, bw4, bh4)?;
            (uv, d)
        } else {
            (DC_PRED, 0)
        };

        let filter_intra = self.read_filter_intra(dec, y_mode, bw4, bh4)?;

        // Record the block's mode and geometry across its 4x4 units.
        self.record_block(r, c, bw4, bh4, y_mode, uv_mode, skip);

        if skip {
            self.reset_block_context(r, c, bw4, bh4);
        }

        // --- residual: every plane, every 4x4 transform block ---
        let modes = BlockModes {
            r,
            c,
            avail_u,
            avail_l,
            y_mode,
            uv_mode,
            y_delta,
            uv_delta,
            filter_intra,
        };
        self.residual(dec, &modes, bw4, bh4, skip, has_chroma)?;
        Ok(())
    }

    fn read_skip(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<bool> {
        let mut ctx = 0;
        if avail_u {
            ctx += usize::from(self.skip_at(r.wrapping_sub(1), c));
        }
        if avail_l {
            ctx += usize::from(self.skip_at(r, c.wrapping_sub(1)));
        }
        let cdf_row = get_mut(&mut self.cdfs.skip, ctx)?;
        Ok(dec.read_symbol(cdf_row)? != 0)
    }

    fn read_intra_frame_y_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<usize> {
        let above = if avail_u {
            self.y_mode_at(r.wrapping_sub(1), c)
        } else {
            DC_PRED
        };
        let left = if avail_l {
            self.y_mode_at(r, c.wrapping_sub(1))
        } else {
            DC_PRED
        };
        let a = INTRA_MODE_CONTEXT.get(above).copied().unwrap_or(0);
        let l = INTRA_MODE_CONTEXT.get(left).copied().unwrap_or(0);
        let cdf_row = get_mut(get_mut(&mut self.cdfs.intra_frame_y_mode, a)?, l)?;
        dec.read_symbol(cdf_row)
    }

    fn read_uv_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        y_mode: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<usize> {
        // Lossless with a 4x4 chroma residual allows chroma-from-luma; here the
        // residual size equals the block size, so CFL is allowed only for 4x4.
        let cfl_allowed = bw4 == 1 && bh4 == 1;
        let uv = if cfl_allowed {
            let cdf_row = get_mut(&mut self.cdfs.uv_cfl_allowed, y_mode)?;
            dec.read_symbol(cdf_row)?
        } else {
            let cdf_row = get_mut(&mut self.cdfs.uv_cfl_not_allowed, y_mode)?;
            dec.read_symbol(cdf_row)?
        };
        if uv == UV_CFL_PRED {
            return Err(PixelsError::unsupported(
                "avif: chroma-from-luma prediction is not implemented yet",
            ));
        }
        Ok(uv)
    }

    /// Read `angle_delta` for a directional mode on a block of at least 8x8,
    /// returning the signed delta (`angle_delta - MAX_ANGLE_DELTA`). Zero for
    /// non-directional modes and small blocks, which read nothing.
    fn read_angle_delta(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        mode: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<i32> {
        let directional = (1..=8).contains(&mode);
        // "MiSize >= BLOCK_8X8": both dimensions at least two 4x4 units.
        if directional && bw4 >= 2 && bh4 >= 2 {
            let index = mode - 1;
            let cdf_row = get_mut(&mut self.cdfs.angle_delta, index)?;
            let symbol = dec.read_symbol(cdf_row)? as i32;
            return Ok(symbol - MAX_ANGLE_DELTA);
        }
        Ok(0)
    }

    /// `filter_intra_mode_info` (§5.11.10): whether luma uses recursive
    /// filter-intra, and if so which of the five kernels.
    fn read_filter_intra(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        y_mode: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<Option<usize>> {
        let max_dim = bw4.max(bh4) * MI_SIZE;
        if self.enable_filter_intra && y_mode == DC_PRED && max_dim <= 32 {
            let size = block_size_index(bw4, bh4);
            let cdf_row = get_mut(&mut self.cdfs.filter_intra, size)?;
            if dec.read_symbol(cdf_row)? != 0 {
                let mode = dec.read_symbol(&mut self.cdfs.filter_intra_mode)?;
                return Ok(Some(mode));
            }
        }
        Ok(None)
    }

    fn residual(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        modes: &BlockModes,
        bw4: usize,
        bh4: usize,
        skip: bool,
        has_chroma: bool,
    ) -> Result<()> {
        let planes = if has_chroma { self.num_planes } else { 1 };
        let base_x = modes.c * MI_SIZE;
        let base_y = modes.r * MI_SIZE;
        for plane in 0..planes {
            let mode = if plane == 0 {
                modes.y_mode
            } else {
                modes.uv_mode
            };
            let delta = if plane == 0 {
                modes.y_delta
            } else {
                modes.uv_delta
            };
            let intra = IntraMode::from_index(mode as u8)
                .ok_or_else(|| PixelsError::malformed("avif", "intra mode index out of range"))?;
            let filter_type = self.filter_type(modes, plane);
            for ty in 0..bh4 {
                for tx in 0..bw4 {
                    let x = base_x + tx * MI_SIZE;
                    let y = base_y + ty * MI_SIZE;
                    let have_left = modes.avail_l || tx > 0;
                    let have_above = modes.avail_u || ty > 0;
                    // Filter-intra is a luma-only tool.
                    let filter_intra = if plane == 0 { modes.filter_intra } else { None };
                    let tb = TxBlock {
                        plane,
                        x,
                        y,
                        mode: intra,
                        mode_index: mode,
                        angle_delta: delta,
                        have_left,
                        have_above,
                        filter_type,
                        filter_intra,
                        skip,
                        bw4,
                        bh4,
                    };
                    self.transform_block(dec, &tb)?;
                }
            }
        }
        Ok(())
    }

    /// `get_filter_type` (§7.11.2.8): whether the above or left neighbour block
    /// used a smooth mode, which softens the directional edge filter. 4:4:4.
    fn filter_type(&self, modes: &BlockModes, plane: usize) -> bool {
        let is_smooth = |mode: usize| (9..=11).contains(&mode);
        let above = modes.avail_u
            && modes.r.checked_sub(1).is_some_and(|ru| {
                let m = if plane == 0 {
                    self.y_mode_at(ru, modes.c)
                } else {
                    self.uv_mode_at(ru, modes.c)
                };
                is_smooth(m)
            });
        let left = modes.avail_l
            && modes.c.checked_sub(1).is_some_and(|cl| {
                let m = if plane == 0 {
                    self.y_mode_at(modes.r, cl)
                } else {
                    self.uv_mode_at(modes.r, cl)
                };
                is_smooth(m)
            });
        above || left
    }

    fn transform_block(&mut self, dec: &mut SymbolDecoder<'_>, tb: &TxBlock) -> Result<()> {
        let (plane, x, y, bw4, bh4, skip) = (tb.plane, tb.x, tb.y, tb.bw4, tb.bh4, tb.skip);

        // A transform block whose top-left lies outside the frame is not coded:
        // the block may extend past the right or bottom edge, but only the tx
        // blocks that start inside it read symbols (spec §5.11.35). For 4:4:4 the
        // luma and chroma edges coincide. Skipping this desynchronises every
        // symbol after the edge — for a last-region block that surfaces as wrong
        // chroma while the luma before it stays correct.
        let max_x = self.mi_cols * MI_SIZE;
        let max_y = self.mi_rows * MI_SIZE;
        if x >= max_x || y >= max_y {
            return Ok(());
        }

        // Predict from the reconstructed neighbours.
        let prediction = self.predict(tb)?;

        let x4 = x / MI_SIZE;
        let y4 = y / MI_SIZE;
        let final_block = if skip {
            prediction
        } else {
            let all_zero_ctx = self.all_zero_ctx(plane, x4, y4, bw4, bh4);
            let dc_sign_ctx = self.dc_sign_ctx(plane, x4, y4);
            let ptype = usize::from(plane > 0);
            let block =
                decode_coeffs_4x4(dec, &mut self.cdfs.coeff, ptype, all_zero_ctx, dc_sign_ctx)?;
            self.update_level_context(plane, x4, y4, block.cul_level, block.dc_category);
            if block.eob > 0 {
                let residual = inverse_wht_4x4(&block.quant);
                add_residual_4x4(&prediction, &residual, self.bit_depth)
            } else {
                prediction
            }
        };

        if let Some(p) = self.planes.get_mut(plane) {
            for (i, row) in final_block.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    p.set(x + j, y + i, value);
                }
            }
        }

        // Mark this 4x4 unit decoded for the neighbour-availability tests.
        let mask = (self.sb_size4 - 1) as isize;
        let sub_row = (y4 as isize) & mask;
        let sub_col = (x4 as isize) & mask;
        self.set_block_decoded(plane, sub_row, sub_col);
        Ok(())
    }

    /// Predict a 4x4 transform block: directional modes go through the edge
    /// machinery, the rest through the pure non-directional predictors.
    fn predict(&self, tb: &TxBlock) -> Result<[[u16; 4]; 4]> {
        if let Some(filter_mode) = tb.filter_intra {
            let (above, left) = self.gather_edges(tb);
            return Ok(predict_filter_intra_4x4(
                filter_mode,
                &above,
                &left,
                self.bit_depth,
            ));
        }
        if let Some(base_angle) = mode_base_angle(tb.mode_index) {
            let p_angle = base_angle + tb.angle_delta * ANGLE_STEP;
            if p_angle != 90 && p_angle != 180 {
                return Ok(self.predict_directional(tb, p_angle));
            }
        }
        let neighbours = self.gather_neighbours(tb.plane, tb.x, tb.y, tb.have_left, tb.have_above);
        predict_intra_4x4(tb.mode, &neighbours, self.bit_depth)
    }

    /// Build the extended `AboveRow`/`LeftCol` edge arrays for a 4x4 block
    /// (§7.11.2 general), with the `haveAboveRight`/`haveBelowLeft` extension
    /// from `BlockDecoded`. Shared by the directional and filter-intra paths.
    fn gather_edges(&self, tb: &TxBlock) -> (Edge, Edge) {
        let (plane, x, y) = (tb.plane, tb.x, tb.y);
        let mid = 1_i32 << (self.bit_depth - 1);
        let p = self.planes.get(plane);
        let at =
            |px: usize, py: usize| -> i32 { p.and_then(|pl| pl.get(px, py)).map_or(0, i32::from) };
        let (max_x, max_y) = p.map_or((0, 0), |pl| {
            (pl.width().saturating_sub(1), pl.height().saturating_sub(1))
        });

        let x4 = x / MI_SIZE;
        let y4 = y / MI_SIZE;
        let mask = (self.sb_size4 - 1) as isize;
        let sub_row = (y4 as isize) & mask;
        let sub_col = (x4 as isize) & mask;
        let have_above_right = self.block_decoded_at(plane, sub_row - 1, sub_col + 1);
        let have_below_left = self.block_decoded_at(plane, sub_row + 1, sub_col - 1);

        let mut above = Edge::new();
        let mut left = Edge::new();
        // AboveRow[0..w+h-1]; w = h = 4 so eight entries.
        if tb.have_above {
            let above_limit = (x + (if have_above_right { 8 } else { 4 }) - 1).min(max_x);
            for i in 0..8 {
                above.set(i, at((x + i as usize).min(above_limit), y.wrapping_sub(1)));
            }
        } else if tb.have_left {
            let v = at(x.wrapping_sub(1), y);
            for i in 0..8 {
                above.set(i, v);
            }
        } else {
            for i in 0..8 {
                above.set(i, mid - 1);
            }
        }
        if tb.have_left {
            let left_limit = (y + (if have_below_left { 8 } else { 4 }) - 1).min(max_y);
            for i in 0..8 {
                left.set(i, at(x.wrapping_sub(1), (y + i as usize).min(left_limit)));
            }
        } else if tb.have_above {
            let v = at(x, y.wrapping_sub(1));
            for i in 0..8 {
                left.set(i, v);
            }
        } else {
            for i in 0..8 {
                left.set(i, mid + 1);
            }
        }
        let corner = match (tb.have_above, tb.have_left) {
            (true, true) => at(x.wrapping_sub(1), y.wrapping_sub(1)),
            (true, false) => at(x, y.wrapping_sub(1)),
            (false, true) => at(x.wrapping_sub(1), y),
            (false, false) => mid,
        };
        above.set(-1, corner);
        left.set(-1, corner);
        (above, left)
    }

    /// Slanted directional prediction (§7.11.2.4).
    fn predict_directional(&self, tb: &TxBlock, p_angle: i32) -> [[u16; 4]; 4] {
        let (x, y) = (tb.x, tb.y);
        let (mut above, mut left) = self.gather_edges(tb);
        let (max_x, max_y) = self.planes.get(tb.plane).map_or((0, 0), |pl| {
            (pl.width().saturating_sub(1), pl.height().saturating_sub(1))
        });
        let avail_above_px = (max_x as i32) - (x as i32) + 1;
        let avail_left_px = (max_y as i32) - (y as i32) + 1;
        predict_directional_4x4(
            p_angle,
            &mut above,
            &mut left,
            tb.have_left,
            tb.have_above,
            tb.filter_type,
            self.enable_edge_filter,
            avail_above_px,
            avail_left_px,
            self.bit_depth,
        )
    }

    /// Build the `AboveRow`/`LeftCol` neighbour arrays for a 4x4 block
    /// (§7.11.2 general). `haveAboveRight`/`haveBelowLeft` are taken as false,
    /// which the non-directional modes do not observe.
    fn gather_neighbours(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
    ) -> Neighbours {
        let mid = 1_i32 << (self.bit_depth - 1);
        let p = self.planes.get(plane);
        let at =
            |px: usize, py: usize| -> i32 { p.and_then(|pl| pl.get(px, py)).map_or(0, i32::from) };
        let (max_x, max_y) = p.map_or((0, 0), |pl| {
            (pl.width().saturating_sub(1), pl.height().saturating_sub(1))
        });

        let mut above = [0_i32; 8];
        let mut left = [0_i32; 8];

        if have_above {
            let above_limit = (x + 4 - 1).min(max_x);
            for (i, slot) in above.iter_mut().enumerate() {
                *slot = at((x + i).min(above_limit), y - 1);
            }
        } else if have_left {
            above = [at(x - 1, y); 8];
        } else {
            above = [mid - 1; 8];
        }

        if have_left {
            let left_limit = (y + 4 - 1).min(max_y);
            for (i, slot) in left.iter_mut().enumerate() {
                *slot = at(x - 1, (y + i).min(left_limit));
            }
        } else if have_above {
            left = [at(x, y - 1); 8];
        } else {
            left = [mid + 1; 8];
        }

        let corner = match (have_above, have_left) {
            (true, true) => at(x - 1, y - 1),
            (true, false) => at(x, y - 1),
            (false, true) => at(x - 1, y),
            (false, false) => mid,
        };

        Neighbours {
            above,
            left,
            corner,
            have_above,
            have_left,
        }
    }

    /// `all_zero` context (§8.3.2). A block whose coding size equals the 4x4
    /// transform is context 0 for luma; otherwise the neighbour level contexts
    /// select it. `bw4`/`bh4` are the coding block's size in 4x4 units (equal to
    /// the chroma residual size in 4:4:4).
    fn all_zero_ctx(&self, plane: usize, x4: usize, y4: usize, bw4: usize, bh4: usize) -> usize {
        let Some(ctx) = self.ctx.get(plane) else {
            return 0;
        };
        if plane == 0 {
            // bw == w and bh == h exactly when the coding block is 4x4.
            if bw4 == 1 && bh4 == 1 {
                return 0;
            }
            // Both are u8, so already within the spec's Min(_, 255).
            let top = ctx.above_level.get(x4).copied().unwrap_or(0);
            let left = ctx.left_level.get(y4).copied().unwrap_or(0);
            if top == 0 && left == 0 {
                1
            } else if top == 0 || left == 0 {
                2 + usize::from(top.max(left) > 3)
            } else if top.max(left) <= 3 {
                4
            } else if top.min(left) <= 3 {
                5
            } else {
                6
            }
        } else {
            let above = ctx.above_level.get(x4).copied().unwrap_or(0)
                | ctx.above_dc.get(x4).copied().unwrap_or(0);
            let left = ctx.left_level.get(y4).copied().unwrap_or(0)
                | ctx.left_dc.get(y4).copied().unwrap_or(0);
            let mut c = 7 + usize::from(above != 0) + usize::from(left != 0);
            // bw * bh > w * h whenever the chroma block is bigger than 4x4.
            if bw4 * bh4 > 1 {
                c += 3;
            }
            c
        }
    }

    /// `dc_sign` context (§8.3.2).
    fn dc_sign_ctx(&self, plane: usize, x4: usize, y4: usize) -> usize {
        let Some(ctx) = self.ctx.get(plane) else {
            return 0;
        };
        let mut dc_sign = 0_i32;
        match ctx.above_dc.get(x4).copied().unwrap_or(0) {
            1 => dc_sign -= 1,
            2 => dc_sign += 1,
            _ => {}
        }
        match ctx.left_dc.get(y4).copied().unwrap_or(0) {
            1 => dc_sign -= 1,
            2 => dc_sign += 1,
            _ => {}
        }
        if dc_sign < 0 {
            1
        } else if dc_sign > 0 {
            2
        } else {
            0
        }
    }

    fn update_level_context(&mut self, plane: usize, x4: usize, y4: usize, cul: u8, dc: u8) {
        if let Some(ctx) = self.ctx.get_mut(plane) {
            if let Some(v) = ctx.above_level.get_mut(x4) {
                *v = cul;
            }
            if let Some(v) = ctx.above_dc.get_mut(x4) {
                *v = dc;
            }
            if let Some(v) = ctx.left_level.get_mut(y4) {
                *v = cul;
            }
            if let Some(v) = ctx.left_dc.get_mut(y4) {
                *v = dc;
            }
        }
    }

    fn reset_block_context(&mut self, r: usize, c: usize, bw4: usize, bh4: usize) {
        for ctx in &mut self.ctx {
            for i in c..c + bw4 {
                if let Some(v) = ctx.above_level.get_mut(i) {
                    *v = 0;
                }
                if let Some(v) = ctx.above_dc.get_mut(i) {
                    *v = 0;
                }
            }
            for i in r..r + bh4 {
                if let Some(v) = ctx.left_level.get_mut(i) {
                    *v = 0;
                }
                if let Some(v) = ctx.left_dc.get_mut(i) {
                    *v = 0;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments, reason = "records every per-block field")]
    fn record_block(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        y_mode: usize,
        uv_mode: usize,
        skip: bool,
    ) {
        let wide = floor_log2_usize(bw4) as u8;
        let high = floor_log2_usize(bh4) as u8;
        for y in r..(r + bh4).min(self.mi_rows) {
            for x in c..(c + bw4).min(self.mi_cols) {
                let idx = y * self.mi_cols + x;
                if let Some(v) = self.y_modes.get_mut(idx) {
                    *v = y_mode as u8;
                }
                if let Some(v) = self.uv_modes.get_mut(idx) {
                    *v = uv_mode as u8;
                }
                if let Some(v) = self.skips.get_mut(idx) {
                    *v = u8::from(skip);
                }
                if let Some(v) = self.mi_wide_log2.get_mut(idx) {
                    *v = wide;
                }
                if let Some(v) = self.mi_high_log2.get_mut(idx) {
                    *v = high;
                }
            }
        }
    }

    fn y_mode_at(&self, r: usize, c: usize) -> usize {
        self.y_modes
            .get(r * self.mi_cols + c)
            .map_or(DC_PRED, |&v| usize::from(v))
    }

    fn uv_mode_at(&self, r: usize, c: usize) -> usize {
        self.uv_modes
            .get(r * self.mi_cols + c)
            .map_or(DC_PRED, |&v| usize::from(v))
    }

    fn skip_at(&self, r: usize, c: usize) -> u8 {
        self.skips.get(r * self.mi_cols + c).copied().unwrap_or(0)
    }
}

/// A coding block's modes, threaded from `intra_frame_mode_info` into the
/// residual loop.
struct BlockModes {
    r: usize,
    c: usize,
    avail_u: bool,
    avail_l: bool,
    y_mode: usize,
    uv_mode: usize,
    y_delta: i32,
    uv_delta: i32,
    /// The luma filter-intra kernel, if this block uses recursive filter-intra.
    filter_intra: Option<usize>,
}

/// One 4x4 transform block's prediction inputs.
struct TxBlock {
    plane: usize,
    x: usize,
    y: usize,
    mode: IntraMode,
    mode_index: usize,
    angle_delta: i32,
    have_left: bool,
    have_above: bool,
    filter_type: bool,
    filter_intra: Option<usize>,
    skip: bool,
    bw4: usize,
    bh4: usize,
}

/// Flat index into a `BlockDecoded` grid with a one-unit border (origin at
/// `[1][1]`). Out-of-border coordinates fold to 0, harmless for a miss.
fn bd_index(stride: usize, row: isize, col: isize) -> usize {
    let r = usize::try_from(row + 1).unwrap_or(0);
    let c = usize::try_from(col + 1).unwrap_or(0);
    r.saturating_mul(stride).saturating_add(c)
}

/// `FloorLog2` for a `usize`.
fn floor_log2_usize(x: usize) -> u32 {
    (usize::BITS - 1) - x.max(1).leading_zeros()
}

/// The `BLOCK_SIZES` index for a block `bw4` by `bh4` 4x4 units.
fn block_size_index(bw4: usize, bh4: usize) -> usize {
    match (bw4, bh4) {
        (1, 1) => 0,
        (1, 2) => 1,
        (2, 1) => 2,
        (2, 2) => 3,
        (2, 4) => 4,
        (4, 2) => 5,
        (4, 4) => 6,
        (4, 8) => 7,
        (8, 4) => 8,
        (8, 8) => 9,
        (8, 16) => 10,
        (16, 8) => 11,
        (16, 16) => 12,
        (16, 32) => 13,
        (32, 16) => 14,
        (32, 32) => 15,
        (1, 4) => 16,
        (4, 1) => 17,
        (2, 8) => 18,
        (8, 2) => 19,
        (4, 16) => 20,
        (16, 4) => 21,
        _ => 15,
    }
}

/// `slice.get_mut(index)`, mapping a miss to a malformed-stream error.
fn get_mut<T>(slice: &mut [T], index: usize) -> Result<&mut T> {
    slice
        .get_mut(index)
        .ok_or_else(|| PixelsError::malformed("avif", "an AV1 tile CDF index ran out of range"))
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

    #[test]
    fn block_size_indices_match_the_spec_ordering() {
        assert_eq!(block_size_index(1, 1), 0);
        assert_eq!(block_size_index(2, 2), 3);
        assert_eq!(block_size_index(16, 16), 12);
        assert_eq!(block_size_index(16, 4), 21);
    }

    #[test]
    fn floor_log2_of_block_units() {
        assert_eq!(floor_log2_usize(1), 0);
        assert_eq!(floor_log2_usize(2), 1);
        assert_eq!(floor_log2_usize(16), 4);
    }
}
