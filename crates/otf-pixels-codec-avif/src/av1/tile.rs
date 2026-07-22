//! The tile decode driver for the lossless intra path (spec §5.11).
//!
//! This is where the pieces meet: the arithmetic decoder walks the partition
//! tree, reads each block's intra mode, predicts every 4x4 transform block from
//! the already-reconstructed neighbours, decodes its coefficients, inverts the
//! transform, and writes the samples back. The neighbour-context arrays it
//! threads between blocks are what make the entropy contexts match the encoder.
//!
//! Scope is the lossless still-image subset: a single tile, `CodedLossless`
//! (every transform is 4x4 WHT, no post-filters), and YUV 4:4:4. Every intra
//! prediction mode is handled — DC, Paeth, the smooth family, the slanted
//! directional modes (with their edge-filter and upsample machinery), recursive
//! filter-intra, palette, and chroma-from-luma. Intra block copy is detected and
//! reported as [`PixelsError::unsupported`] rather than decoded wrong, so a
//! stream that uses it fails cleanly instead of desynchronising.

use super::cdf;
use super::coeff::{CoeffCdfs, decode_coeffs_4x4};
use super::direction::{
    ANGLE_STEP, Edge, mode_base_angle, predict_directional_4x4, predict_filter_intra_4x4,
};
use super::frame::FrameHeader;
use super::palette::{PALETTE_COLORS, color_context, palette_cache};
use super::plane::Plane;
use super::predict::{IntraMode, Neighbours, predict_intra_4x4};
use super::seq::SequenceHeader;
use super::symbol::SymbolDecoder;
use super::transform::{
    TxSize, TxType, ac_q, add_residual_4x4, dc_q, dequantize, inverse_transform_2d,
};
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
    if frame.allow_intrabc {
        // Intra block copy is not implemented; decoding its blocks would
        // desynchronise the symbol stream.
        return Err(PixelsError::unsupported(
            "avif: intra block copy is not implemented yet",
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
    palette_y_mode: [[[u16; 3]; 3]; 7],
    palette_uv_mode: [[u16; 3]; 2],
    palette_y_size: [[u16; 8]; 7],
    palette_uv_size: [[u16; 8]; 7],
    palette_y_color: PaletteColorCdfs,
    palette_uv_color: PaletteColorCdfs,
    cfl_sign: [u16; 9],
    cfl_alpha: [[u16; 17]; 6],
    coeff: CoeffCdfs,
}

/// The seven palette colour-index CDFs, one per palette size 2..=8.
struct PaletteColorCdfs {
    size2: [[u16; 3]; 5],
    size3: [[u16; 4]; 5],
    size4: [[u16; 5]; 5],
    size5: [[u16; 6]; 5],
    size6: [[u16; 7]; 5],
    size7: [[u16; 8]; 5],
    size8: [[u16; 9]; 5],
}

impl PaletteColorCdfs {
    /// The colour-index CDF row for `palette_size` (2..=8) and context `ctx`.
    fn row(&mut self, palette_size: usize, ctx: usize) -> Result<&mut [u16]> {
        Ok(match palette_size {
            2 => get_mut(&mut self.size2, ctx)?,
            3 => get_mut(&mut self.size3, ctx)?,
            4 => get_mut(&mut self.size4, ctx)?,
            5 => get_mut(&mut self.size5, ctx)?,
            6 => get_mut(&mut self.size6, ctx)?,
            7 => get_mut(&mut self.size7, ctx)?,
            _ => get_mut(&mut self.size8, ctx)?,
        })
    }
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
            palette_y_mode: cdf::DEFAULT_PALETTE_Y_MODE_CDF,
            palette_uv_mode: cdf::DEFAULT_PALETTE_UV_MODE_CDF,
            palette_y_size: cdf::DEFAULT_PALETTE_Y_SIZE_CDF,
            palette_uv_size: cdf::DEFAULT_PALETTE_UV_SIZE_CDF,
            palette_y_color: PaletteColorCdfs {
                size2: cdf::DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF,
                size3: cdf::DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF,
                size4: cdf::DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF,
                size5: cdf::DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF,
                size6: cdf::DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF,
                size7: cdf::DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF,
                size8: cdf::DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF,
            },
            palette_uv_color: PaletteColorCdfs {
                size2: cdf::DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF,
                size3: cdf::DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF,
                size4: cdf::DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF,
                size5: cdf::DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF,
                size6: cdf::DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF,
                size7: cdf::DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF,
                size8: cdf::DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF,
            },
            cfl_sign: cdf::DEFAULT_CFL_SIGN_CDF,
            cfl_alpha: cdf::DEFAULT_CFL_ALPHA_CDF,
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
    allow_screen_content: bool,
    sb_size4: usize,
    /// `BlockDecoded[plane]`, one flat `(sb+2) x (sb+2)` grid per plane, reset
    /// per superblock; addressed with a one-unit border so index -1 is valid.
    block_decoded: Vec<Vec<u8>>,
    /// `YModes[r][c]` flattened row-major, one entry per 4x4 unit.
    y_modes: Vec<u8>,
    /// `UVModes[r][c]` flattened, for the intra filter-type decision.
    uv_modes: Vec<u8>,
    /// `PaletteSizes[plane][r][c]` flattened, for the neighbour palette cache
    /// and `has_palette` contexts.
    palette_sizes: [Vec<u8>; 2],
    /// `PaletteColors[plane][r][c][0..8]` flattened (8 colours per unit).
    palette_colors: [Vec<[u16; PALETTE_COLORS]>; 2],
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
            allow_screen_content: frame.allow_screen_content_tools,
            sb_size4,
            block_decoded,
            y_modes: vec![0; mi_cols * mi_rows],
            uv_modes: vec![0; mi_cols * mi_rows],
            palette_sizes: [vec![0; mi_cols * mi_rows], vec![0; mi_cols * mi_rows]],
            palette_colors: [
                vec![[0; PALETTE_COLORS]; mi_cols * mi_rows],
                vec![[0; PALETTE_COLORS]; mi_cols * mi_rows],
            ],
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

        let (uv_mode, uv_delta, cfl) = if has_chroma {
            let (uv, cfl) = self.read_uv_mode(dec, y_mode, bw4, bh4)?;
            let d = self.read_angle_delta(dec, uv, bw4, bh4)?;
            (uv, d, cfl)
        } else {
            (DC_PRED, 0, None)
        };

        // palette_mode_info (§5.11.46): only for DC blocks of 8x8..64x64 when
        // screen-content tools are enabled.
        let mut palette = Palette {
            block_w: bw4 * MI_SIZE,
            block_h: bh4 * MI_SIZE,
            ..Palette::default()
        };
        let palette_ok =
            self.allow_screen_content && bw4 >= 2 && bh4 >= 2 && bw4 <= 16 && bh4 <= 16;
        if palette_ok {
            self.read_palette_mode_info(
                dec,
                r,
                c,
                bw4,
                bh4,
                y_mode,
                uv_mode,
                has_chroma,
                &mut palette,
            )?;
        }

        // filter-intra is not coded for a palette-Y block.
        let filter_intra = if palette.size_y > 0 {
            None
        } else {
            self.read_filter_intra(dec, y_mode, bw4, bh4)?
        };

        // Record the block's mode, geometry, and palette across its 4x4 units.
        self.record_block(r, c, bw4, bh4, y_mode, uv_mode, skip, &palette);

        // palette_tokens (§5.11.49): the colour-index maps.
        if palette.size_y > 0 || palette.size_uv > 0 {
            self.read_palette_tokens(dec, &mut palette)?;
        }

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
            cfl,
            palette,
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

    /// Read `uv_mode` and, for a chroma-from-luma block, its alphas. Returns the
    /// UV mode and `Some((alphaU, alphaV))` when the block is CfL.
    fn read_uv_mode(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        y_mode: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<(usize, Option<(i32, i32)>)> {
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
        let cfl = if uv == UV_CFL_PRED {
            Some(self.read_cfl_alphas(dec)?)
        } else {
            None
        };
        Ok((uv, cfl))
    }

    /// `read_cfl_alphas` (§5.11.45): the signed U and V scaling factors.
    fn read_cfl_alphas(&mut self, dec: &mut SymbolDecoder<'_>) -> Result<(i32, i32)> {
        let signs = dec.read_symbol(&mut self.cdfs.cfl_sign)? as i32;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        // CFL_SIGN_ZERO = 0, CFL_SIGN_NEG = 1, CFL_SIGN_POS = 2.
        let alpha_u = if sign_u != 0 {
            let ctx = ((sign_u - 1) * 3 + sign_v) as usize;
            let mag = dec.read_symbol(get_mut(&mut self.cdfs.cfl_alpha, ctx)?)? as i32 + 1;
            if sign_u == 1 { -mag } else { mag }
        } else {
            0
        };
        let alpha_v = if sign_v != 0 {
            let ctx = ((sign_v - 1) * 3 + sign_u) as usize;
            let mag = dec.read_symbol(get_mut(&mut self.cdfs.cfl_alpha, ctx)?)? as i32 + 1;
            if sign_v == 1 { -mag } else { mag }
        } else {
            0
        };
        Ok((alpha_u, alpha_v))
    }

    /// `palette_mode_info` (§5.11.46): read the luma and chroma palettes into
    /// `palette` (their colours and sizes).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors palette_mode_info inputs"
    )]
    fn read_palette_mode_info(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        y_mode: usize,
        uv_mode: usize,
        has_chroma: bool,
        palette: &mut Palette,
    ) -> Result<()> {
        // bsizeCtx = Mi_Width_Log2 + Mi_Height_Log2 - 2 (spec §5.11.46); palette
        // is only reached for 8x8 and larger, so the subtraction never underflows.
        let bsize_ctx = (floor_log2_usize(bw4) + floor_log2_usize(bh4)).saturating_sub(2) as usize;
        let avail_u = r > 0;
        let avail_l = c > 0;
        if y_mode == DC_PRED {
            let ctx = usize::from(avail_u && self.palette_size_at(0, r.wrapping_sub(1), c) > 0)
                + usize::from(avail_l && self.palette_size_at(0, r, c.wrapping_sub(1)) > 0);
            let cdf_row = get_mut(get_mut(&mut self.cdfs.palette_y_mode, bsize_ctx)?, ctx)?;
            if dec.read_symbol(cdf_row)? != 0 {
                let size_cdf = get_mut(&mut self.cdfs.palette_y_size, bsize_ctx)?;
                let size = dec.read_symbol(size_cdf)? + 2;
                let cache = self.palette_cache_for(0, r, c);
                palette.size_y = size;
                palette.colors_y = self.read_palette_colors(dec, size, &cache, true)?;
            }
        }
        if has_chroma && uv_mode == DC_PRED {
            let ctx = usize::from(palette.size_y > 0);
            let cdf_row = get_mut(&mut self.cdfs.palette_uv_mode, ctx)?;
            if dec.read_symbol(cdf_row)? != 0 {
                let size_cdf = get_mut(&mut self.cdfs.palette_uv_size, bsize_ctx)?;
                let size = dec.read_symbol(size_cdf)? + 2;
                let cache = self.palette_cache_for(1, r, c);
                palette.size_uv = size;
                palette.colors_u = self.read_palette_colors(dec, size, &cache, false)?;
                palette.colors_v = self.read_palette_colors_v(dec, size)?;
            }
        }
        Ok(())
    }

    /// The neighbour palette cache for `plane` at `(r, c)` (`get_palette_cache`).
    fn palette_cache_for(&self, plane: usize, r: usize, c: usize) -> Vec<u16> {
        let above = if r > 0 && (r * MI_SIZE) % 64 != 0 {
            let n = self.palette_size_at(plane, r - 1, c) as usize;
            self.palette_colors_at(plane, r - 1, c, n)
        } else {
            Vec::new()
        };
        let left = if c > 0 {
            let n = self.palette_size_at(plane, r, c - 1) as usize;
            self.palette_colors_at(plane, r, c - 1, n)
        } else {
            Vec::new()
        };
        palette_cache(&above, &left)
    }

    /// Read a palette's colours (`palette_colors_y`/`_u`): cache hits first, then
    /// a base colour, then Clip1-accumulated deltas, sorted ascending.
    fn read_palette_colors(
        &self,
        dec: &mut SymbolDecoder<'_>,
        size: usize,
        cache: &[u16],
        is_luma: bool,
    ) -> Result<[u16; PALETTE_COLORS]> {
        let bd = u32::from(self.bit_depth);
        let max = (1_i32 << bd) - 1;
        let clip1 = |v: i32| v.clamp(0, max) as u16;
        let mut colors = [0_u16; PALETTE_COLORS];
        let mut idx = 0;
        for &cached in cache.iter() {
            if idx >= size {
                break;
            }
            if dec.read_literal(1)? != 0 {
                set_at(&mut colors, idx, cached);
                idx += 1;
            }
        }
        if idx < size {
            set_at(&mut colors, idx, dec.read_literal(bd)? as u16);
            idx += 1;
        }
        if idx < size {
            let min_bits = bd.saturating_sub(3);
            let mut palette_bits = min_bits + dec.read_literal(2)?;
            while idx < size {
                // The luma delta is coded one less than its value; the chroma
                // delta is coded directly (spec §5.11.47). The range that bounds
                // the next `paletteBits` likewise drops one only for luma.
                let delta = dec.read_literal(palette_bits)? + u32::from(is_luma);
                let prev = i32::from(at(&colors, idx - 1));
                let color = clip1(prev + delta as i32);
                set_at(&mut colors, idx, color);
                let range = (1_i32 << bd) - i32::from(color) - i32::from(is_luma);
                palette_bits = palette_bits.min(ceil_log2(range.max(0) as u32));
                idx += 1;
            }
        }
        let slice = colors.get_mut(..size).unwrap_or(&mut []);
        slice.sort_unstable();
        Ok(colors)
    }

    /// Read the V-plane palette colours (`palette_colors_v`), which are coded
    /// either as wrapping deltas or as raw literals.
    fn read_palette_colors_v(
        &self,
        dec: &mut SymbolDecoder<'_>,
        size: usize,
    ) -> Result<[u16; PALETTE_COLORS]> {
        let bd = u32::from(self.bit_depth);
        let max = (1_i32 << bd) - 1;
        let max_val = 1_i32 << bd;
        let mut colors = [0_u16; PALETTE_COLORS];
        if dec.read_literal(1)? != 0 {
            let mut palette_bits = bd.saturating_sub(4) + dec.read_literal(2)?;
            set_at(&mut colors, 0, dec.read_literal(bd)? as u16);
            for idx in 1..size {
                let mut delta = dec.read_literal(palette_bits)? as i32;
                if delta != 0 && dec.read_literal(1)? != 0 {
                    delta = -delta;
                }
                let mut val = i32::from(at(&colors, idx - 1)) + delta;
                if val < 0 {
                    val += max_val;
                }
                if val >= max_val {
                    val -= max_val;
                }
                set_at(&mut colors, idx, val.clamp(0, max) as u16);
                let _ = &mut palette_bits;
            }
        } else {
            for idx in 0..size {
                set_at(&mut colors, idx, dec.read_literal(bd)? as u16);
            }
        }
        Ok(colors)
    }

    /// `palette_tokens` (§5.11.49): decode the colour-index maps by the
    /// wavefront traversal.
    fn read_palette_tokens(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        palette: &mut Palette,
    ) -> Result<()> {
        let (bw, bh) = (palette.block_w, palette.block_h);
        if palette.size_y > 0 {
            palette.map_y = self.read_color_map(dec, palette.size_y, bw, bh, false)?;
        }
        if palette.size_uv > 0 {
            // 4:4:4: the chroma map is the same shape as luma.
            palette.map_uv = self.read_color_map(dec, palette.size_uv, bw, bh, true)?;
        }
        Ok(())
    }

    /// Decode one colour-index map (`ColorMapY`/`ColorMapUV`).
    fn read_color_map(
        &mut self,
        dec: &mut SymbolDecoder<'_>,
        size: usize,
        bw: usize,
        bh: usize,
        chroma: bool,
    ) -> Result<Vec<u8>> {
        // Whole block is on screen here (partial edges clamped by the caller's
        // block dimensions), so onscreen == block dimensions.
        let mut map = vec![0_u8; bw * bh];
        let first = dec.read_ns(size as u32)? as u8;
        if let Some(m) = map.first_mut() {
            *m = first;
        }
        let get = |map: &[u8], i: usize, j: usize| -> Option<u8> { map.get(i * bw + j).copied() };
        for i in 1..(bh + bw - 1) {
            let j_hi = i.min(bw - 1);
            let j_lo = i.saturating_sub(bh - 1);
            let mut j = j_hi as isize;
            while j >= j_lo as isize {
                let jj = j as usize;
                let row = i - jj;
                let left = if jj > 0 { get(&map, row, jj - 1) } else { None };
                let above_left = if row > 0 && jj > 0 {
                    get(&map, row - 1, jj - 1)
                } else {
                    None
                };
                let above = if row > 0 {
                    get(&map, row - 1, jj)
                } else {
                    None
                };
                let (order, ctx) = color_context(left, above_left, above, size);
                let cdf = if chroma {
                    self.cdfs.palette_uv_color.row(size, ctx)?
                } else {
                    self.cdfs.palette_y_color.row(size, ctx)?
                };
                let sym = dec.read_symbol(cdf)?;
                let color = order.get(sym).copied().unwrap_or(0);
                if let Some(slot) = map.get_mut(row * bw + jj) {
                    *slot = color;
                }
                j -= 1;
            }
        }
        Ok(map)
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
            // A CfL chroma block predicts from DC and then adds the scaled luma.
            let cfl_alpha = if plane == 1 {
                modes.cfl.map(|(u, _)| u)
            } else if plane == 2 {
                modes.cfl.map(|(_, v)| v)
            } else {
                None
            };
            let mode = if plane == 0 {
                modes.y_mode
            } else if modes.cfl.is_some() {
                DC_PRED
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
            // Palette-coded plane: luma uses ColorMapY + colors_y; chroma uses
            // ColorMapUV + colors_u (U) or colors_v (V).
            let palette = self.palette_view_for(&modes.palette, plane, base_x, base_y);
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
                        cfl_alpha,
                        palette,
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

    /// Build the palette view for `plane` if that plane is palette-coded.
    fn palette_view_for<'a>(
        &self,
        palette: &'a Palette,
        plane: usize,
        base_x: usize,
        base_y: usize,
    ) -> Option<PaletteView<'a>> {
        if plane == 0 && palette.size_y > 0 {
            Some(PaletteView {
                map: &palette.map_y,
                colors: &palette.colors_y,
                block_w: palette.block_w,
                base_x,
                base_y,
            })
        } else if plane == 1 && palette.size_uv > 0 {
            Some(PaletteView {
                map: &palette.map_uv,
                colors: &palette.colors_u,
                block_w: palette.block_w,
                base_x,
                base_y,
            })
        } else if plane == 2 && palette.size_uv > 0 {
            Some(PaletteView {
                map: &palette.map_uv,
                colors: &palette.colors_v,
                block_w: palette.block_w,
                base_x,
                base_y,
            })
        } else {
            None
        }
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
                // Lossless is qindex 0: dc_q == ac_q == 4 and dqDenom == 1, so
                // dequantise is a plain "times 4" and the WHT divides it back
                // out. Routing it through the general transform driver keeps the
                // path bit-exact while sharing the lossy machinery.
                let dc = dc_q(self.bit_depth, 0);
                let ac = ac_q(self.bit_depth, 0);
                let dequant = dequantize(&block.quant, TxSize::Tx4x4, dc, ac, self.bit_depth);
                let residual = inverse_transform_2d(
                    &dequant,
                    TxSize::Tx4x4,
                    TxType::DctDct,
                    true,
                    self.bit_depth,
                );
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
        if let Some(pv) = tb.palette {
            // predict_palette (§7.11.4): each sample is the palette colour its
            // index map selects. The map is block-relative.
            let mut pred = [[0_u16; 4]; 4];
            for (i, row) in pred.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let my = (tb.y + i).saturating_sub(pv.base_y);
                    let mx = (tb.x + j).saturating_sub(pv.base_x);
                    let idx = pv.map.get(my * pv.block_w + mx).copied().unwrap_or(0);
                    *cell = pv.colors.get(usize::from(idx)).copied().unwrap_or(0);
                }
            }
            return Ok(pred);
        }
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
        let mut pred = predict_intra_4x4(tb.mode, &neighbours, self.bit_depth)?;
        if let Some(alpha) = tb.cfl_alpha {
            self.apply_cfl(&mut pred, tb.x, tb.y, alpha);
        }
        Ok(pred)
    }

    /// `predict_chroma_from_luma` (§7.11.5) for a 4x4 4:4:4 block: add the
    /// alpha-scaled, DC-removed reconstructed luma to the DC chroma prediction.
    fn apply_cfl(&self, pred: &mut [[u16; 4]; 4], x: usize, y: usize, alpha: i32) {
        let max = (1_i32 << self.bit_depth) - 1;
        let luma = self.planes.first();
        // L holds the co-located luma with 3 fractional bits (no subsampling).
        let mut l = [[0_i32; 4]; 4];
        let mut sum = 0_i32;
        for (i, row) in l.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let v = i32::from(luma.and_then(|p| p.get(x + j, y + i)).unwrap_or(0)) << 3;
                *cell = v;
                sum += v;
            }
        }
        // lumaAvg = Round2(sum, log2W + log2H) = Round2(sum, 4) for 4x4.
        let luma_avg = (sum + 8) >> 4;
        for (i, row) in pred.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let ac = l.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0) - luma_avg;
                let scaled = round2_signed(alpha * ac, 6);
                *cell = (i32::from(*cell) + scaled).clamp(0, max) as u16;
            }
        }
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
        palette: &Palette,
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
                let [ps_y, ps_uv] = &mut self.palette_sizes;
                if let Some(v) = ps_y.get_mut(idx) {
                    *v = palette.size_y as u8;
                }
                if let Some(v) = ps_uv.get_mut(idx) {
                    *v = palette.size_uv as u8;
                }
                let [pc_y, pc_uv] = &mut self.palette_colors;
                if let Some(v) = pc_y.get_mut(idx) {
                    *v = palette.colors_y;
                }
                if let Some(v) = pc_uv.get_mut(idx) {
                    *v = palette.colors_u;
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

    fn palette_size_at(&self, plane: usize, r: usize, c: usize) -> u8 {
        self.palette_sizes
            .get(plane)
            .and_then(|p| p.get(r * self.mi_cols + c))
            .copied()
            .unwrap_or(0)
    }

    fn palette_colors_at(&self, plane: usize, r: usize, c: usize, n: usize) -> Vec<u16> {
        self.palette_colors
            .get(plane)
            .and_then(|p| p.get(r * self.mi_cols + c))
            .map(|colors| colors.iter().take(n).copied().collect())
            .unwrap_or_default()
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
    /// The chroma-from-luma alphas `(alphaU, alphaV)`, if this block is CfL.
    cfl: Option<(i32, i32)>,
    /// The block's palette state (sizes zero when unused).
    palette: Palette,
}

/// One block's palette: the colours and the per-sample colour-index maps.
#[derive(Default, Clone)]
struct Palette {
    /// `PaletteSizeY` (0 when the luma plane is not palette-coded).
    size_y: usize,
    /// `PaletteSizeUV`.
    size_uv: usize,
    /// `palette_colors_y`, ascending.
    colors_y: [u16; PALETTE_COLORS],
    /// `palette_colors_u`.
    colors_u: [u16; PALETTE_COLORS],
    /// `palette_colors_v`.
    colors_v: [u16; PALETTE_COLORS],
    /// `ColorMapY`, `block_h * block_w` row-major.
    map_y: Vec<u8>,
    /// `ColorMapUV`, same shape (4:4:4).
    map_uv: Vec<u8>,
    /// The luma block width and height in samples.
    block_w: usize,
    block_h: usize,
}

/// One 4x4 transform block's prediction inputs.
struct TxBlock<'a> {
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
    /// The chroma-from-luma alpha for this plane, if the block is CfL.
    cfl_alpha: Option<i32>,
    /// The plane's palette view when the block is palette-coded on this plane.
    palette: Option<PaletteView<'a>>,
    skip: bool,
    bw4: usize,
    bh4: usize,
}

/// A palette-coded plane's data for one transform block: the block-relative
/// colour-index map plus the colours it selects.
#[derive(Clone, Copy)]
struct PaletteView<'a> {
    map: &'a [u8],
    colors: &'a [u16; PALETTE_COLORS],
    block_w: usize,
    base_x: usize,
    base_y: usize,
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

/// `array[i]` for a palette colour array, 0 outside range.
fn at(colors: &[u16; PALETTE_COLORS], i: usize) -> u16 {
    colors.get(i).copied().unwrap_or(0)
}

/// Set `array[i]` for a palette colour array; out-of-range writes are dropped.
fn set_at(colors: &mut [u16; PALETTE_COLORS], i: usize, value: u16) {
    if let Some(slot) = colors.get_mut(i) {
        *slot = value;
    }
}

/// `Round2Signed(x, n)` (§4.7).
fn round2_signed(x: i32, n: u32) -> i32 {
    if x >= 0 {
        (x + (1 << (n - 1))) >> n
    } else {
        -((-x + (1 << (n - 1))) >> n)
    }
}

/// `CeilLog2(x)` (§4.7).
fn ceil_log2(x: u32) -> u32 {
    if x < 2 {
        0
    } else {
        u32::BITS - (x - 1).leading_zeros()
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
