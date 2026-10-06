//! The tile decode driver for the intra still-picture path (spec §5.11).
//!
//! This is where the pieces meet: the arithmetic decoder walks the partition
//! tree, reads each block's intra mode and transform size, predicts every
//! transform block from the already-reconstructed neighbours, decodes its
//! coefficients, inverts the transform, and writes the samples back. The
//! neighbour-context arrays it threads between blocks are what make the entropy
//! contexts match the encoder.
//!
//! Scope is the intra still image in YUV 4:4:4, 4:2:2 or 4:2:0, in any tiling.
//! Each tile decodes from fresh CDFs and cleared above contexts, and treats a
//! neighbour across its edge as unavailable (`is_inside`); the in-loop
//! filters then run over the whole frame.
//! Subsampled chroma is decoded in its own sample grid: a block one unit wide
//! or high shares its chroma with its neighbour, the odd one of the pair coding
//! it (`HasChroma`). Both `CodedLossless`
//! (every transform is 4x4 WHT) and lossy frames decode here — the lossy path
//! runs the full DCT/ADST/identity inverse transforms at every transform size,
//! followed by the in-loop filters: deblocking (§7.14), CDEF (§7.15) and loop
//! restoration (§7.17), with the super-resolution upscale (§7.16) between CDEF
//! and restoration. Film grain is not implemented, so a lossy frame is
//! reproduced exactly only when it codes grain off (`unimplemented_filters_off`);
//! any other lossy frame is refused. Every intra
//! prediction mode is handled —
//! DC, Paeth, the smooth family, the slanted directional modes (with their
//! edge-filter and upsample machinery), recursive filter-intra, palette, and
//! chroma-from-luma. Intra block copy is detected and reported as
//! [`PixelsError::unsupported`] rather than decoded wrong, so a stream that uses
//! it fails cleanly instead of desynchronising.

use super::bits::BitReader;
use super::cdef::CdefFilter;
use super::cdf;
use super::coder::{CoeffJob, Site, TileCoder};
use super::coeff::{CoeffCdfs, TxTypeCtx};
use super::deblock::Deblock;
use super::direction::{
    ANGLE_STEP, Edge, mode_base_angle, predict_directional, predict_filter_intra,
};
use super::frame::{Cdef, FrameHeader, LoopFilter, Segmentation, TileInfo, TxMode};
use super::palette::{PALETTE_COLORS, color_context, palette_cache};
use super::plane::Plane;
use super::predict::{IntraMode, PredBlock, predict_intra_block};
use super::restoration::{
    LoopRestore, PlaneLr, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
    SGRPROJ_XQD_MID, WIENER_TAPS_MID, count_units_in_frame, read_sgrproj_unit, read_wiener_unit,
};
use super::seq::SequenceHeader;
use super::superres::{SUPERRES_NUM, Superres};
use super::symbol::SymbolDecoder;
use super::transform::{
    TxSize, TxType, ac_q, add_residual, dc_q, dequantize_with_matrix, inverse_transform_2d,
    quantizer_matrix,
};
use super::transform_type::{IntraTxTypeCdfs, intra_dir, intra_tx_set};
use super::tx_size::{BLOCK_4X4, TxDepthCdfs, TxSizeParams, code_tx_size, max_tx_size_rect};
use otf_pixels_core::{PixelsError, Result};

/// `MI_SIZE` (§3): the side of the smallest coded block, in samples.
const MI_SIZE: usize = 4;
/// `FRAME_LF_COUNT` (§3): loop-filter levels — two luma directions, U, V.
pub const FRAME_LF_COUNT: usize = 4;
/// `DELTA_Q_SMALL` and `DELTA_LF_SMALL` (§3), which are equal: the symbol
/// value that escapes to an explicitly sized literal.
const DELTA_SMALL: i32 = 3;
/// `MAX_LOOP_FILTER` (§3).
const MAX_LOOP_FILTER: i32 = 63;
/// `MAX_SEGMENTS` (§3).
pub const MAX_SEGMENTS: usize = 8;
/// `SEG_LVL_ALT_Q` (§3): the per-segment quantizer offset.
const SEG_LVL_ALT_Q: usize = 0;
/// `SEG_LVL_ALT_LF_Y_V` (§3): the first of the four per-segment loop-filter
/// offsets, in `loop_filter_level` order.
const SEG_LVL_ALT_LF_Y_V: usize = 1;
/// `SEG_LVL_SKIP` (§3): every block of the segment is skipped.
const SEG_LVL_SKIP: usize = 6;
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

/// A decoded frame's sample planes, in coded order (Y, U, V). Each plane covers
/// whole superblocks (in its own subsampled grid); only the top-left
/// display-sized region is the picture.
pub struct DecodedFrame {
    /// The reconstructed planes.
    pub planes: Vec<Plane>,
}

/// Decode an intra still frame into its sample planes, from the payloads of its
/// tile groups in order. Handles both lossless and lossy frames, the latter
/// only when film grain is disabled (see `unimplemented_filters_off`).
///
/// # Errors
///
/// Returns [`PixelsError::unsupported`] for anything outside the intra subset
/// this decodes — intra block copy, or a lossy frame using film grain — and [`PixelsError::malformed`] for a stream
/// that ends early, violates the syntax, or does not code every tile exactly
/// once.
pub fn decode_still(
    seq: &SequenceHeader,
    frame: &FrameHeader,
    tile_groups: &[&[u8]],
) -> Result<DecodedFrame> {
    // We reconstruct the residual, apply every in-loop filter — deblocking
    // (§7.14), CDEF (§7.15) and loop restoration (§7.17) — and the
    // super-resolution upscale (§7.16), but not film grain. A lossy frame is
    // reproduced exactly only when grain is disabled; any other would decode to
    // a visibly wrong image, so it is refused. A coded-lossless frame turns the
    // in-loop filters off by definition.
    if !frame.coded_lossless && !unimplemented_filters_off(frame) {
        return Err(PixelsError::unsupported(
            "avif: lossy frames are decoded only with film grain disabled; film \
             grain synthesis is not implemented yet",
        ));
    }
    if frame.allow_intrabc {
        // Intra block copy is not implemented; decoding its blocks would
        // desynchronise the symbol stream.
        return Err(PixelsError::unsupported(
            "avif: intra block copy is not implemented yet",
        ));
    }

    let info = &frame.tile_info;
    let sb_shift = if seq.use_128x128_superblock { 5 } else { 4 };
    let mut state = TileState::new(seq, frame)?;
    let mut next_tile = 0;
    for group in tile_groups {
        for tile in split_tile_group(group, info)? {
            // Tiles must arrive in order, each exactly once; tg_start says
            // where a group begins, so a gap or a repeat is a broken stream.
            if tile.number != next_tile {
                return Err(PixelsError::malformed(
                    "avif",
                    format!(
                        "tile {} arrived where tile {next_tile} was due",
                        tile.number
                    ),
                ));
            }
            next_tile += 1;
            let bounds = tile_bounds(info, sb_shift, state.mi_rows, state.mi_cols, tile.number);
            state.decode_tile(tile.data, bounds)?;
        }
    }
    if next_tile != info.count() {
        return Err(PixelsError::malformed(
            "avif",
            format!("the frame codes {next_tile} of its {} tiles", info.count()),
        ));
    }
    state.post_filter();
    Ok(DecodedFrame {
        planes: state.planes,
    })
}

/// One tile's coded bytes and its index in raster order.
#[derive(Debug)]
struct Tile<'a> {
    number: u32,
    data: &'a [u8],
}

/// Split a tile group (`tile_group_obu`, §5.11.1) into its tiles.
///
/// A group with several tiles may name the range it carries; every tile but
/// its last is preceded by its size, `TileSizeBytes` little-endian bytes
/// holding the size minus one, and the last runs to the end of the group.
fn split_tile_group<'a>(group: &'a [u8], info: &TileInfo) -> Result<Vec<Tile<'a>>> {
    let count = info.count();
    let mut reader = BitReader::new(group);
    let (start, end) = if count > 1 && reader.flag()? {
        let bits = info.cols_log2 + info.rows_log2;
        (reader.f(bits)?, reader.f(bits)?)
    } else {
        (0, count.saturating_sub(1))
    };
    if start > end || end >= count {
        return Err(PixelsError::malformed(
            "avif",
            format!("tile group spans tiles {start}..={end} of {count}"),
        ));
    }
    reader.byte_alignment()?;
    let mut rest = group.get(reader.byte_position()..).unwrap_or(&[]);

    let mut tiles = Vec::new();
    for number in start..=end {
        let data = if number == end {
            core::mem::take(&mut rest)
        } else {
            let width = info.tile_size_bytes as usize;
            let size_bytes = rest.get(..width).ok_or_else(|| tile_overrun(number))?;
            let size = size_bytes
                .iter()
                .rev()
                .fold(0_usize, |acc, &b| (acc << 8) | usize::from(b))
                + 1;
            let body = rest.get(width..).ok_or_else(|| tile_overrun(number))?;
            let (data, after) = (body.get(..size), body.get(size..));
            rest = after.ok_or_else(|| tile_overrun(number))?;
            data.ok_or_else(|| tile_overrun(number))?
        };
        tiles.push(Tile { number, data });
    }
    Ok(tiles)
}

fn tile_overrun(number: u32) -> PixelsError {
    PixelsError::malformed(
        "avif",
        format!("tile {number} claims more bytes than its tile group holds"),
    )
}

/// Tile `number`'s extent in 4x4 units, from the frame's tile layout.
pub(crate) fn tile_bounds(
    info: &TileInfo,
    sb_shift: u32,
    mi_rows: usize,
    mi_cols: usize,
    number: u32,
) -> TileBounds {
    let (row, col) = (number / info.cols.max(1), number % info.cols.max(1));
    let start = |starts: &[u32], i: u32, limit: usize| {
        starts
            .get(i as usize)
            .map_or(limit, |&sb| ((sb as usize) << sb_shift).min(limit))
    };
    TileBounds {
        row_start: start(&info.row_starts_sb, row, mi_rows),
        row_end: start(&info.row_starts_sb, row + 1, mi_rows),
        col_start: start(&info.col_starts_sb, col, mi_cols),
        col_end: start(&info.col_starts_sb, col + 1, mi_cols),
    }
}

/// One tile's extent in 4x4 units (`MiRowStart`..`MiRowEnd`,
/// `MiColStart`..`MiColEnd`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TileBounds {
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) col_start: usize,
    pub(crate) col_end: usize,
}

/// Whether every post-filter this decoder does *not* implement is disabled, so
/// the reconstruct plus the implemented in-loop filters reproduces the frame
/// exactly. Only film grain remains: deblocking (§7.14), CDEF (§7.15),
/// super-resolution (§7.16) and loop restoration (§7.17) are all implemented.
fn unimplemented_filters_off(frame: &FrameHeader) -> bool {
    !frame.film_grain.apply_grain
}

/// `MiSize >= BLOCK_8X8` for a block of `bw4 x bh4` 4x4 units. The spec compares
/// block-size *enum* values, and 4x16/16x4 sort after 8x8, so this excludes only
/// 4x4, 4x8 and 8x4 — an area of at least four units — rather than requiring
/// both dimensions to reach 8. Gates `angle_delta` and `palette_mode_info`.
fn at_least_block_8x8(bw4: usize, bh4: usize) -> bool {
    bw4 * bh4 >= 4
}

/// Build the per-plane loop-restoration records (§7.17): the frame filter type,
/// the unit size, and the unit-grid dimensions (`count_units_in_frame` over the
/// plane's rounded dimensions). Per-unit data is filled later by `read_lr`.
fn build_plane_lr(seq: &SequenceHeader, frame: &FrameHeader) -> Vec<PlaneLr> {
    let num_planes = seq.color.num_planes as usize;
    let lr = &frame.loop_restoration;
    let upscaled_width = frame.upscaled_width as usize;
    let frame_height = frame.frame_height as usize;
    // Round2(x, n) for n in {0, 1}: the only subsampling shifts.
    let round2 = |x: usize, n: usize| if n == 0 { x } else { (x + 1) >> 1 };
    (0..num_planes)
        .map(|plane| {
            let frt = lr
                .frame_restoration_type
                .get(plane)
                .copied()
                .unwrap_or(RESTORE_NONE);
            let unit_size = lr.unit_size.get(plane).copied().unwrap_or(256) as usize;
            let (sub_x, sub_y) = if plane == 0 {
                (0, 0)
            } else {
                (
                    seq.color.subsampling_x as usize,
                    seq.color.subsampling_y as usize,
                )
            };
            let rows = count_units_in_frame(unit_size, round2(frame_height, sub_y));
            let cols = count_units_in_frame(unit_size, round2(upscaled_width, sub_x));
            PlaneLr::new(frt, unit_size, rows, cols)
        })
        .collect()
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
    intra_tx_type: IntraTxTypeCdfs,
    tx_depth: TxDepthCdfs,
    restoration_type: [u16; 4],
    use_wiener: [u16; 3],
    use_sgrproj: [u16; 3],
    delta_q: [u16; 5],
    delta_lf: [u16; 5],
    segment_id: [[u16; 9]; 3],
    /// `DeltaLFMultiCdf[i]`, one per loop-filter level when `delta_lf_multi`.
    delta_lf_multi: [[u16; 5]; FRAME_LF_COUNT],
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
            intra_tx_type: IntraTxTypeCdfs::new(),
            tx_depth: TxDepthCdfs::new(),
            restoration_type: cdf::DEFAULT_RESTORATION_TYPE_CDF,
            use_wiener: cdf::DEFAULT_USE_WIENER_CDF,
            use_sgrproj: cdf::DEFAULT_USE_SGRPROJ_CDF,
            delta_q: cdf::DEFAULT_DELTA_Q_CDF,
            delta_lf: cdf::DEFAULT_DELTA_LF_CDF,
            segment_id: cdf::DEFAULT_SEGMENT_ID_CDF,
            delta_lf_multi: [cdf::DEFAULT_DELTA_LF_CDF; FRAME_LF_COUNT],
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
pub(crate) struct TileState {
    pub(crate) planes: Vec<Plane>,
    cdfs: FrameCdfs,
    bit_depth: u8,
    num_planes: usize,
    mi_cols: usize,
    mi_rows: usize,
    enable_filter_intra: bool,
    enable_edge_filter: bool,
    allow_screen_content: bool,
    /// `reduced_tx_set` (frame header): shrinks the intra transform set.
    reduced_tx_set: bool,
    /// `CodedLossless`: every segment is lossless. Gates `cdef_idx`.
    coded_lossless: bool,
    /// The current block's `Lossless` (`LosslessArray[segment_id]`): a 4x4
    /// WHT, no transform-type choice, and no quantizer matrix.
    lossless: bool,
    /// `LosslessArray[segment]`.
    lossless_array: [bool; MAX_SEGMENTS],
    /// The frame's segmentation parameters; disabled means one segment, 0.
    segmentation: Segmentation,
    /// `SegIdPreSkip`: `segment_id` is read before `skip`.
    seg_pre_skip: bool,
    /// `LastActiveSegId`: the coded `segment_id` range is `0..=` this.
    last_active_segment: usize,
    /// The current block's `segment_id`.
    segment_id: usize,
    /// `SegmentIds[r][c]`, which predicts the next block's `segment_id` and
    /// selects the deblocking strength's segment offsets.
    segment_ids: Vec<u8>,
    /// `using_qmatrix` and `qm_y`/`qm_u`/`qm_v`, for `SegQMLevel`.
    using_qmatrix: bool,
    qm: [u8; 3],
    /// The frame transform mode (`ONLY_4X4` / `LARGEST` / `SELECT`).
    tx_mode: TxMode,
    /// Per-plane DC quantiser offset from the block's qindex (`DeltaQYDc`,
    /// `DeltaQUDc`, `DeltaQVDc`).
    q_dc: [i32; 3],
    /// Per-plane AC quantiser offset (zero for luma).
    q_ac: [i32; 3],
    /// `CurrentQIndex` (§5.11.12): `base_q_idx` at each tile's start, moved by
    /// each `delta_qindex` when `delta_q_present`. Never 0 once moved, so never
    /// lossless.
    current_qindex: i32,
    /// `base_q_idx`, which `CurrentQIndex` restarts from in every tile.
    base_q: i32,
    /// The coefficient-CDF quantiser context, for each tile's fresh CDFs.
    qctx: usize,
    /// The tile being decoded. Neighbours outside it are unavailable
    /// (`is_inside`, §5.11.51) even when they lie inside the frame.
    tile: TileBounds,
    /// `delta_q_present`, `delta_q_res` (§5.9.17).
    delta_q_present: bool,
    delta_q_res: u32,
    /// `delta_lf_present`, `delta_lf_res`, `delta_lf_multi` (§5.9.18).
    delta_lf_present: bool,
    delta_lf_res: u32,
    delta_lf_multi: bool,
    /// `ReadDeltas`: set at each superblock, cleared by its first block.
    read_deltas: bool,
    /// `DeltaLF[i]`: the running loop-filter deltas, reset per tile.
    delta_lf: [i8; FRAME_LF_COUNT],
    /// `DeltaLFs[r][c]`: each 4x4 unit's `DeltaLF` when its block was decoded,
    /// which the deblocking filter reads for the strength (§7.14.4).
    delta_lfs: Vec<[i8; FRAME_LF_COUNT]>,
    /// `InterTxSizes[r][c]`: the luma transform size (a `TxSize` index) chosen
    /// for each 4x4 unit, for the `tx_depth` neighbour context under `SELECT`.
    tx_sizes: Vec<u8>,
    /// `LoopfilterTxSizes[plane][r][c]`: the transform size actually applied at
    /// each 4x4 unit, per plane, filled as transform blocks reconstruct. The
    /// deblocking loop filter reads it to find transform edges (§7.14.2).
    lf_tx_sizes: [Vec<u8>; 3],
    /// Coded frame dimensions in luma samples (`FrameWidth`/`FrameHeight`), for
    /// the loop filter's on-screen test. With super-resolution `FrameWidth` is
    /// the reduced width every tile-level step runs at.
    frame_width: usize,
    frame_height: usize,
    /// `UpscaledWidth`: the display width, which loop restoration runs at.
    upscaled_width: usize,
    /// `SuperresDenom` when `use_superres`, else `None`.
    superres_denom: Option<usize>,
    /// The frame loop-filter parameters, for deblocking after reconstruct.
    loop_filter: LoopFilter,
    /// The frame CDEF parameters, for the CDEF pass after deblocking (§7.15).
    cdef: Cdef,
    /// Whether CDEF is enabled at all (`enable_cdef`); when false no `cdef_idx`
    /// is coded and the grid stays all -1.
    enable_cdef: bool,
    /// `cdef_idx[row][col]` (§5.11.56): the CDEF strength index per 64x64 block,
    /// -1 until read. Only the 64x64-aligned entries are meaningful.
    cdef_idx: Vec<i16>,
    /// Chroma subsampling (0 for 4:4:4), for the CDEF filter's plane geometry.
    subsampling_x: usize,
    subsampling_y: usize,
    /// Whether any plane uses loop restoration (`UsesLr`).
    uses_lr: bool,
    /// Per-plane loop-restoration parameters and per-unit filter data, filled by
    /// `read_lr` during tile decode. Empty when `uses_lr` is false.
    lr: Vec<PlaneLr>,
    /// `RefLrWiener[plane][pass][coeff]`: the running Wiener reference, reset per
    /// tile and updated by each coded unit.
    ref_lr_wiener: [[[i32; 3]; 2]; 3],
    /// `RefSgrXqd[plane][i]`: the running self-guided projection reference.
    ref_sgr_xqd: [[i32; 2]; 3],
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
    /// `MaxLumaW`/`MaxLumaH`: the right and bottom edge of the latest luma
    /// transform block, which bounds chroma-from-luma's luma reads.
    max_luma_w: usize,
    max_luma_h: usize,
}

impl TileState {
    fn new(seq: &SequenceHeader, frame: &FrameHeader) -> Result<Self> {
        let mi_cols = frame.mi_cols as usize;
        let mi_rows = frame.mi_rows as usize;
        let num_planes = seq.color.num_planes as usize;
        // `CurrFrame` must hold every sample a transform block writes, and a
        // block overhanging the right or bottom edge writes its whole prediction
        // and residual past `MiCols * MI_SIZE` — samples chroma-from-luma then
        // reads back. So the planes cover whole superblocks; the decoded area
        // (`frame_bounds`) is tracked separately. Each chroma plane is the luma
        // grid shifted by its subsampling (superblocks are even, so the shift
        // is exact).
        let sb_size4 = if seq.use_128x128_superblock { 32 } else { 16 };
        let padded_w = mi_cols.div_ceil(sb_size4) * sb_size4 * MI_SIZE;
        let padded_h = mi_rows.div_ceil(sb_size4) * sb_size4 * MI_SIZE;
        let (sub_x, sub_y) = (
            seq.color.subsampling_x as usize,
            seq.color.subsampling_y as usize,
        );
        let planes = (0..num_planes)
            .map(|p| {
                if p == 0 {
                    Plane::new(padded_w, padded_h)
                } else {
                    Plane::new(padded_w >> sub_x, padded_h >> sub_y)
                }
            })
            .collect();
        let ctx = (0..num_planes)
            .map(|_| LevelContext {
                above_level: vec![0; mi_cols],
                above_dc: vec![0; mi_cols],
                left_level: vec![0; mi_rows],
                left_dc: vec![0; mi_rows],
            })
            .collect();
        let bd_stride = sb_size4 + 2;
        let block_decoded = (0..num_planes)
            .map(|_| vec![0; bd_stride * bd_stride])
            .collect();
        // The coefficient-CDF quantiser context (`get_qctx`, §8.3.2): base_q_idx
        // <=20 -> 0, <=60 -> 1, <=120 -> 2, else 3. Lossless (0) is 0.
        let base_q = i32::from(frame.quantization.base_q_idx);
        let qctx = match base_q {
            0..=20 => 0,
            21..=60 => 1,
            61..=120 => 2,
            _ => 3,
        };
        let q = &frame.quantization;
        Ok(Self {
            planes,
            cdfs: FrameCdfs::new(qctx),
            bit_depth: seq.color.bit_depth,
            num_planes,
            mi_cols,
            mi_rows,
            enable_filter_intra: seq.enable_filter_intra,
            enable_edge_filter: seq.enable_intra_edge_filter,
            allow_screen_content: frame.allow_screen_content_tools,
            reduced_tx_set: frame.reduced_tx_set,
            coded_lossless: frame.coded_lossless,
            lossless: frame.coded_lossless,
            lossless_array: frame.lossless,
            segmentation: frame.segmentation.clone(),
            seg_pre_skip: frame.segmentation.pre_skip(),
            last_active_segment: frame.segmentation.last_active_segment(),
            segment_id: 0,
            segment_ids: vec![0; mi_cols * mi_rows],
            using_qmatrix: q.using_qmatrix,
            qm: [q.qm_y, q.qm_u, q.qm_v],
            tx_mode: frame.tx_mode,
            q_dc: [q.delta_q_y_dc, q.delta_q_u_dc, q.delta_q_v_dc],
            q_ac: [0, q.delta_q_u_ac, q.delta_q_v_ac],
            current_qindex: base_q,
            base_q,
            qctx,
            tile: TileBounds {
                row_start: 0,
                row_end: mi_rows,
                col_start: 0,
                col_end: mi_cols,
            },
            delta_q_present: frame.delta_q_present,
            delta_q_res: frame.delta_q_res,
            delta_lf_present: frame.delta_lf_present,
            delta_lf_res: frame.delta_lf_res,
            delta_lf_multi: frame.delta_lf_multi,
            read_deltas: false,
            delta_lf: [0; FRAME_LF_COUNT],
            delta_lfs: vec![[0; FRAME_LF_COUNT]; mi_cols * mi_rows],
            tx_sizes: vec![0; mi_cols * mi_rows],
            lf_tx_sizes: [
                vec![0; mi_cols * mi_rows],
                vec![0; mi_cols * mi_rows],
                vec![0; mi_cols * mi_rows],
            ],
            frame_width: frame.frame_width as usize,
            frame_height: frame.frame_height as usize,
            upscaled_width: frame.upscaled_width as usize,
            // `SuperresDenom` is SUPERRES_NUM exactly when use_superres is 0. (The
            // widths are no test: a 1-sample frame codes at its full width even
            // with super-resolution on.)
            superres_denom: (frame.superres_denom as usize != SUPERRES_NUM)
                .then_some(frame.superres_denom as usize),
            loop_filter: frame.loop_filter.clone(),
            cdef: frame.cdef.clone(),
            enable_cdef: seq.enable_cdef,
            cdef_idx: vec![-1; mi_cols * mi_rows],
            subsampling_x: seq.color.subsampling_x as usize,
            subsampling_y: seq.color.subsampling_y as usize,
            uses_lr: frame.loop_restoration.uses_lr,
            lr: build_plane_lr(seq, frame),
            ref_lr_wiener: [[WIENER_TAPS_MID; 2]; 3],
            ref_sgr_xqd: [SGRPROJ_XQD_MID; 3],
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
            max_luma_w: 0,
            max_luma_h: 0,
        })
    }

    /// `decode_tile` (§5.11.2): one tile, from fresh CDFs and cleared above
    /// contexts, its superblocks in raster order within its bounds.
    fn decode_tile(&mut self, tile_data: &[u8], tile: TileBounds) -> Result<()> {
        let mut dec = SymbolDecoder::new(tile_data, false)?;
        self.code_tile(&mut dec, tile)
    }

    /// `decode_tile` (§5.11.2) over any [`TileCoder`]: the decoder reading a
    /// stream, or the encoder deciding and writing one.
    pub(crate) fn code_tile(&mut self, dec: &mut impl TileCoder, tile: TileBounds) -> Result<()> {
        self.tile = tile;
        self.cdfs = FrameCdfs::new(self.qctx);
        self.current_qindex = self.base_q;
        self.delta_lf = [0; FRAME_LF_COUNT];
        self.ref_lr_wiener = [[WIENER_TAPS_MID; 2]; 3];
        self.ref_sgr_xqd = [SGRPROJ_XQD_MID; 3];
        for c in &mut self.ctx {
            c.above_level.fill(0);
            c.above_dc.fill(0);
        }
        let sb_size4 = self.sb_size4;
        // Superblocks are decoded in raster order; each seeds the partition
        // recursion. The left contexts reset at the start of each SB row.
        let mut sb_row = tile.row_start;
        while sb_row < tile.row_end {
            self.reset_left_context();
            let mut sb_col = tile.col_start;
            while sb_col < tile.col_end {
                self.read_deltas = self.delta_q_present;
                self.clear_block_decoded(sb_row, sb_col);
                self.read_lr(dec, sb_row, sb_col)?;
                self.decode_partition(dec, sb_row, sb_col, sb_size4)?;
                sb_col += sb_size4;
            }
            sb_row += sb_size4;
        }
        Ok(())
    }

    /// The in-loop filters and upscale, over the whole frame once every tile
    /// is reconstructed: they cross tile boundaries (§5.11.52).
    pub(crate) fn post_filter(&mut self) {
        self.deblock();
        // Loop restoration reads both the pre-CDEF (deblocked) and post-CDEF
        // frames, so when it runs, snapshot the deblocked frame before CDEF. Both
        // are then upscaled (§7.4 steps 3–4; a no-op without super-resolution)
        // and restoration filters the upscaled CDEF output in place.
        let curr = self.uses_lr.then(|| self.planes.clone());
        self.cdef();
        if let Some(superres) = self.superres() {
            self.planes = superres.upscale(&self.planes);
            if let Some(curr) = curr {
                let curr = superres.upscale(&curr);
                self.loop_restore(&curr);
            }
        } else if let Some(curr) = curr {
            self.loop_restore(&curr);
        }
    }

    /// The upscaling geometry when the frame uses super-resolution.
    fn superres(&self) -> Option<Superres> {
        self.superres_denom.map(|_| Superres {
            frame_width: self.frame_width,
            upscaled_width: self.upscaled_width,
            frame_height: self.frame_height,
            mi_cols: self.mi_cols,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            bit_depth: self.bit_depth,
        })
    }

    /// Apply the deblocking loop filter to the reconstructed planes (§7.14). A
    /// no-op when every filter level is zero, which is always so for a
    /// coded-lossless frame.
    fn deblock(&mut self) {
        Deblock {
            planes: &mut self.planes,
            loop_filter: &self.loop_filter,
            bit_depth: self.bit_depth,
            num_planes: self.num_planes,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            frame_width: self.frame_width,
            frame_height: self.frame_height,
            lf_tx_sizes: &self.lf_tx_sizes,
            delta_lfs: &self.delta_lfs,
            delta_lf_multi: self.delta_lf_multi,
            segment_ids: &self.segment_ids,
            segment_lf: core::array::from_fn(|segment| {
                core::array::from_fn(|i| {
                    self.segmentation
                        .feature_value(segment, SEG_LVL_ALT_LF_Y_V + i)
                })
            }),
        }
        .run();
    }

    /// Apply the constrained directional enhancement filter (§7.15) to the
    /// deblocked planes. A no-op when CDEF is disabled: the `cdef_idx` grid is
    /// then all -1, so every 8x8 block is left as it is.
    fn cdef(&mut self) {
        CdefFilter {
            planes: &mut self.planes,
            cdef: &self.cdef,
            cdef_idx: &self.cdef_idx,
            skips: &self.skips,
            bit_depth: self.bit_depth,
            num_planes: self.num_planes,
            mi_rows: self.mi_rows,
            mi_cols: self.mi_cols,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
        }
        .run();
    }

    /// Apply loop restoration (§7.17) to the CDEF output. `curr` is the pre-CDEF
    /// (deblocked) frame; the post-CDEF frame is the current `planes`, which also
    /// seed `LrFrame` — restoration overwrites only the blocks that need it,
    /// reading throughout from the two snapshots so it never sees its own output.
    fn loop_restore(&mut self, curr: &[Plane]) {
        let cdef = self.planes.clone();
        LoopRestore {
            planes: &mut self.planes,
            curr,
            cdef: &cdef,
            lr: &self.lr,
            bit_depth: self.bit_depth,
            num_planes: self.num_planes,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            upscaled_width: self.upscaled_width,
            frame_height: self.frame_height,
        }
        .run();
    }

    /// `clear_block_decoded_flags` (§5.11.3) for one superblock, each plane in
    /// its own (subsampled) 4x4 units.
    fn clear_block_decoded(&mut self, r: usize, c: usize) {
        let sb = self.sb_size4;
        let stride = sb + 2;
        for plane in 0..self.num_planes {
            let (sub_x, sub_y) = self.plane_subsampling(plane);
            let sb_width4 = ((self.tile.col_end - c) >> sub_x) as isize;
            let sb_height4 = ((self.tile.row_end - r) >> sub_y) as isize;
            let (sb_w, sb_h) = ((sb >> sub_x) as isize, (sb >> sub_y) as isize);
            let Some(grid) = self.block_decoded.get_mut(plane) else {
                continue;
            };
            for v in grid.iter_mut() {
                *v = 0;
            }
            // Row above (y == -1) valid where x < sbWidth4; column left (x == -1)
            // valid where y < sbHeight4. Indices carry a +1 border.
            for x in -1_isize..=sb_w {
                if x < sb_width4 {
                    if let Some(slot) = grid.get_mut(bd_index(stride, -1, x)) {
                        *slot = 1;
                    }
                }
            }
            for y in -1_isize..=sb_h {
                if y < sb_height4 {
                    if let Some(slot) = grid.get_mut(bd_index(stride, y, -1)) {
                        *slot = 1;
                    }
                }
            }
            if let Some(slot) = grid.get_mut(bd_index(stride, sb_h, -1)) {
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

    /// `AvailU`: `is_inside(r - 1, c)` for a block at `(r, c)` in this tile.
    const fn avail_u(&self, r: usize) -> bool {
        r > self.tile.row_start
    }

    /// `AvailL`: `is_inside(r, c - 1)`.
    const fn avail_l(&self, c: usize) -> bool {
        c > self.tile.col_start
    }

    fn reset_left_context(&mut self) {
        for c in &mut self.ctx {
            c.left_level.fill(0);
            c.left_dc.fill(0);
        }
    }

    /// `decode_partition` (§5.11.4), restricted to what the lossless subset
    /// produces. `bsize4` is the block side in 4-sample units (a power of two).
    fn decode_partition(
        &mut self,
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        bsize4: usize,
    ) -> Result<()> {
        if r >= self.mi_rows || c >= self.mi_cols {
            return Ok(());
        }
        let avail_u = self.avail_u(r);
        let avail_l = self.avail_l(c);
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
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        bsize4: usize,
        avail_u: bool,
        avail_l: bool,
    ) -> Result<usize> {
        let ctx = self.partition_ctx(r, c, bsize4, avail_u, avail_l);
        let bsl = floor_log2_usize(bsize4);
        let cdf_row = self.partition_cdf(bsl, ctx)?;
        dec.symbol(cdf_row, Site::Partition { r, c, bsize4 })
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
        dec: &mut impl TileCoder,
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
        Ok(dec.symbol(&mut derived, Site::SplitOr { r, c, bsize4 })? != 0)
    }

    /// `decode_block` (§5.11.5) plus mode info and residual, for one block.
    fn decode_block(
        &mut self,
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<()> {
        let avail_u = self.avail_u(r);
        let avail_l = self.avail_l(c);
        // HasChroma (§5.11.5): with subsampling, a block one unit wide (high)
        // at an even column (row) shares its chroma with the next block, which
        // codes it; only that odd-positioned block carries chroma.
        let (sub_x, sub_y) = (self.subsampling_x, self.subsampling_y);
        let shares_chroma =
            (bh4 == 1 && sub_y == 1 && r & 1 == 0) || (bw4 == 1 && sub_x == 1 && c & 1 == 0);
        let has_chroma = self.num_planes > 1 && !shares_chroma;
        // AvailUChroma/AvailLChroma: such a chroma-owning block's chroma extends
        // one unit further up (left), so its neighbour is two units away.
        let (avail_u_chroma, avail_l_chroma) = if has_chroma {
            (
                if sub_y == 1 && bh4 == 1 {
                    r >= self.tile.row_start + 2
                } else {
                    avail_u
                },
                if sub_x == 1 && bw4 == 1 {
                    c >= self.tile.col_start + 2
                } else {
                    avail_l
                },
            )
        } else {
            (false, false)
        };

        // --- intra_frame_mode_info (§5.11.7) ---
        // segment_id comes before skip when a segment forces skipping, and
        // after it (where a skipped block inherits its prediction) otherwise.
        dec.plan_block(self, r, c, bw4, bh4);
        self.segment_id = 0;
        if self.seg_pre_skip {
            self.read_segment_id(dec, r, c, avail_u, avail_l, false)?;
        }
        let skip = if self.seg_pre_skip
            && self
                .segmentation
                .feature_active(self.segment_id, SEG_LVL_SKIP)
        {
            true
        } else {
            self.read_skip(dec, r, c, avail_u, avail_l)?
        };
        if !self.seg_pre_skip {
            self.read_segment_id(dec, r, c, avail_u, avail_l, skip)?;
        }
        self.lossless = self
            .lossless_array
            .get(self.segment_id)
            .copied()
            .unwrap_or(false);
        let segment = self.segment_id as u8;
        for row in r..(r + bh4).min(self.mi_rows) {
            for col in c..(c + bw4).min(self.mi_cols) {
                if let Some(slot) = self.segment_ids.get_mut(row * self.mi_cols + col) {
                    *slot = segment;
                }
            }
        }

        // read_cdef (§5.11.56) sits right after the segment id, then the
        // superblock's quantizer and loop-filter deltas.
        self.read_cdef(dec, r, c, bw4, bh4, skip)?;
        self.read_delta_qindex(dec, bw4, bh4, skip)?;
        self.read_delta_lf(dec, bw4, bh4, skip)?;
        self.read_deltas = false;
        for row in r..(r + bh4).min(self.mi_rows) {
            for col in c..(c + bw4).min(self.mi_cols) {
                if let Some(slot) = self.delta_lfs.get_mut(row * self.mi_cols + col) {
                    *slot = self.delta_lf;
                }
            }
        }

        let y_mode = self.read_intra_frame_y_mode(dec, r, c, avail_u, avail_l)?;
        let y_delta = self.read_angle_delta(dec, y_mode, bw4, bh4, Site::AngleDeltaY)?;

        let (uv_mode, uv_delta, cfl) = if has_chroma {
            let (uv, cfl) = self.read_uv_mode(dec, y_mode, bw4, bh4)?;
            let d = self.read_angle_delta(dec, uv, bw4, bh4, Site::AngleDeltaUv)?;
            (uv, d, cfl)
        } else {
            (DC_PRED, 0, None)
        };

        // palette_mode_info (§5.11.46): only when screen-content tools are
        // enabled, for `MiSize >= BLOCK_8X8` up to 64x64.
        let mut palette = Palette {
            block_w: bw4 * MI_SIZE,
            block_h: bh4 * MI_SIZE,
            ..Palette::default()
        };
        let palette_ok =
            self.allow_screen_content && at_least_block_8x8(bw4, bh4) && bw4 <= 16 && bh4 <= 16;
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
        self.record_block(r, c, bw4, bh4, y_mode, uv_mode, has_chroma, skip, &palette);

        // palette_tokens (§5.11.49): the colour-index maps.
        if palette.size_y > 0 || palette.size_uv > 0 {
            self.read_palette_tokens(dec, r, c, &mut palette)?;
        }

        // read_block_tx_size (§5.11.16): the luma transform size for the block.
        // Under TX_MODE_SELECT this reads a `tx_depth` symbol, so it must run
        // before residual and after mode info.
        let luma_tx_size = self.read_block_tx_size(dec, r, c, bw4, bh4, skip)?;

        if skip {
            self.reset_block_context(r, c, bw4, bh4, has_chroma);
        }

        // --- residual: every plane, every transform block ---
        let modes = BlockModes {
            r,
            c,
            avail_u,
            avail_l,
            avail_u_chroma,
            avail_l_chroma,
            y_mode,
            uv_mode,
            y_delta,
            uv_delta,
            filter_intra,
            cfl,
            palette,
            luma_tx_size,
        };
        self.residual(dec, &modes, bw4, bh4, skip, has_chroma)?;
        Ok(())
    }

    fn read_skip(
        &mut self,
        dec: &mut impl TileCoder,
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
        Ok(dec.symbol(cdf_row, Site::Skip)? != 0)
    }

    /// `read_cdef` (§5.11.56): read the `cdef_idx` literal for the 64x64 block
    /// containing `(r, c)`, the first time that block is reached. A skip block,
    /// a coded-lossless frame, or CDEF being disabled reads nothing (`allow_intrabc`
    /// is always false in this subset). `cdef_bits` is often zero, in which case
    /// the literal is empty and the index is simply 0 (filtering with the single
    /// coded strength).
    fn read_cdef(
        &mut self,
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        skip: bool,
    ) -> Result<()> {
        if skip || self.coded_lossless || !self.enable_cdef {
            return Ok(());
        }
        // CDEF parameters are stored per 64x64 luma block (16 units).
        let cdef_size4 = 16;
        let mask = !(cdef_size4 - 1);
        let base_r = r & mask;
        let base_c = c & mask;
        if self.cdef_idx.get(base_r * self.mi_cols + base_c).copied() != Some(-1) {
            return Ok(());
        }
        let value = dec.literal(self.cdef.bits, Site::CdefIdx)? as i16;
        let mut i = base_r;
        while i < base_r + bh4 {
            let mut j = base_c;
            while j < base_c + bw4 {
                if i < self.mi_rows && j < self.mi_cols {
                    if let Some(slot) = self.cdef_idx.get_mut(i * self.mi_cols + j) {
                        *slot = value;
                    }
                }
                j += cdef_size4;
            }
            i += cdef_size4;
        }
        Ok(())
    }

    /// `intra_segment_id` / `read_segment_id` (§5.11.8, §5.11.9): predict the
    /// block's segment from its neighbours, then — unless it is skipped, when
    /// the prediction stands — read the difference from it.
    fn read_segment_id(
        &mut self,
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        avail_u: bool,
        avail_l: bool,
        skip: bool,
    ) -> Result<()> {
        if !self.segmentation.enabled {
            self.segment_id = 0;
            return Ok(());
        }
        let at = |row: usize, col: usize| -> i32 {
            self.segment_ids
                .get(row * self.mi_cols + col)
                .map_or(-1, |&s| i32::from(s))
        };
        let prev_ul = if avail_u && avail_l {
            at(r - 1, c - 1)
        } else {
            -1
        };
        let prev_u = if avail_u { at(r - 1, c) } else { -1 };
        let prev_l = if avail_l { at(r, c - 1) } else { -1 };
        let pred = if prev_u == -1 {
            prev_l.max(0)
        } else if prev_l == -1 || prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        let segment = if skip {
            pred
        } else {
            let ctx = if prev_ul < 0 {
                0
            } else if prev_ul == prev_u && prev_ul == prev_l {
                2
            } else if prev_ul == prev_u || prev_ul == prev_l || prev_u == prev_l {
                1
            } else {
                0
            };
            let diff = dec.symbol(get_mut(&mut self.cdfs.segment_id, ctx)?, Site::Other)? as i32;
            neg_deinterleave(diff, pred, self.last_active_segment as i32 + 1)
        };
        // A conformant stream stays within 0..=LastActiveSegId.
        self.segment_id = usize::try_from(segment)
            .ok()
            .filter(|&s| s <= self.last_active_segment)
            .ok_or_else(|| {
                PixelsError::malformed(
                    "avif",
                    format!(
                        "segment_id {segment} is outside 0..={}",
                        self.last_active_segment
                    ),
                )
            })?;
        Ok(())
    }

    /// `get_qindex(ignoreDeltaQ, segment_id)` (§7.12.2) for the current block.
    fn segment_qindex(&self, ignore_delta_q: bool) -> i32 {
        let base = if !ignore_delta_q && self.delta_q_present {
            self.current_qindex
        } else {
            self.base_q
        };
        if self
            .segmentation
            .feature_active(self.segment_id, SEG_LVL_ALT_Q)
        {
            (base
                + self
                    .segmentation
                    .feature_value(self.segment_id, SEG_LVL_ALT_Q))
            .clamp(0, 255)
        } else {
            base
        }
    }

    /// `read_delta_qindex` (§5.11.12): the first block of a superblock may move
    /// `CurrentQIndex`, unless it covers the whole superblock and is skipped
    /// (it then has no coefficients for a quantizer to matter to).
    fn read_delta_qindex(
        &mut self,
        dec: &mut impl TileCoder,
        bw4: usize,
        bh4: usize,
        skip: bool,
    ) -> Result<()> {
        if (bw4 == self.sb_size4 && bh4 == self.sb_size4 && skip) || !self.read_deltas {
            return Ok(());
        }
        let delta = read_delta(dec, &mut self.cdfs.delta_q)?;
        if delta != 0 {
            self.current_qindex = (self.current_qindex + (delta << self.delta_q_res)).clamp(1, 255);
        }
        Ok(())
    }

    /// `read_delta_lf` (§5.11.13): likewise for the loop-filter deltas — one,
    /// or with `delta_lf_multi` one per filter level (two luma directions and,
    /// with chroma, U and V).
    fn read_delta_lf(
        &mut self,
        dec: &mut impl TileCoder,
        bw4: usize,
        bh4: usize,
        skip: bool,
    ) -> Result<()> {
        if (bw4 == self.sb_size4 && bh4 == self.sb_size4 && skip)
            || !self.read_deltas
            || !self.delta_lf_present
        {
            return Ok(());
        }
        let count = if !self.delta_lf_multi {
            1
        } else if self.num_planes > 1 {
            FRAME_LF_COUNT
        } else {
            FRAME_LF_COUNT - 2
        };
        for i in 0..count {
            let cdf = if self.delta_lf_multi {
                get_mut(&mut self.cdfs.delta_lf_multi, i)?
            } else {
                &mut self.cdfs.delta_lf
            };
            let delta = read_delta(dec, cdf)?;
            if delta != 0 {
                let slot = get_mut(&mut self.delta_lf, i)?;
                let level = (i32::from(*slot) + (delta << self.delta_lf_res))
                    .clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
                *slot = level as i8;
            }
        }
        Ok(())
    }

    /// `read_lr` (§5.11.57): read the loop-restoration units this superblock
    /// covers, once per plane that uses restoration. `allow_intrabc` is always
    /// false in this subset. Units are laid out over the upscaled frame, so with
    /// super-resolution a superblock's coded columns are scaled by
    /// `SuperresDenom / SUPERRES_NUM` to find the units it covers.
    fn read_lr(&mut self, dec: &mut impl TileCoder, r: usize, c: usize) -> Result<()> {
        if !self.uses_lr {
            return Ok(());
        }
        let (w4, h4) = (self.sb_size4, self.sb_size4);
        for plane in 0..self.num_planes {
            let Some(info) = self.lr.get(plane) else {
                continue;
            };
            if info.frame_restoration_type == RESTORE_NONE {
                continue;
            }
            let (sub_x, sub_y) = if plane == 0 {
                (0, 0)
            } else {
                (self.subsampling_x, self.subsampling_y)
            };
            let unit_size = info.unit_size;
            let unit_rows = info.unit_rows;
            let unit_cols = info.unit_cols;
            let unit_row_start = (r * (MI_SIZE >> sub_y)).div_ceil(unit_size);
            let unit_row_end = unit_rows.min(((r + h4) * (MI_SIZE >> sub_y)).div_ceil(unit_size));
            let (numerator, denominator) = match self.superres_denom {
                Some(denom) => ((MI_SIZE >> sub_x) * denom, unit_size * SUPERRES_NUM),
                None => (MI_SIZE >> sub_x, unit_size),
            };
            let unit_col_start = (c * numerator).div_ceil(denominator);
            let unit_col_end = unit_cols.min(((c + w4) * numerator).div_ceil(denominator));
            for unit_row in unit_row_start..unit_row_end {
                for unit_col in unit_col_start..unit_col_end {
                    self.read_lr_unit(dec, plane, unit_row, unit_col)?;
                }
            }
        }
        Ok(())
    }

    /// `read_lr_unit` (§5.11.58): read one restoration unit's type and, for a
    /// Wiener or self-guided unit, its coefficients (which update the running
    /// per-plane reference used by the sub-exponential coding).
    fn read_lr_unit(
        &mut self,
        dec: &mut impl TileCoder,
        plane: usize,
        unit_row: usize,
        unit_col: usize,
    ) -> Result<()> {
        let frame_type = self
            .lr
            .get(plane)
            .map_or(RESTORE_NONE, |p| p.frame_restoration_type);
        let restoration_type = match frame_type {
            RESTORE_WIENER => {
                if dec.symbol(&mut self.cdfs.use_wiener, Site::Other)? != 0 {
                    RESTORE_WIENER
                } else {
                    RESTORE_NONE
                }
            }
            RESTORE_SGRPROJ => {
                if dec.symbol(&mut self.cdfs.use_sgrproj, Site::Other)? != 0 {
                    RESTORE_SGRPROJ
                } else {
                    RESTORE_NONE
                }
            }
            RESTORE_SWITCHABLE => dec.symbol(&mut self.cdfs.restoration_type, Site::Other)? as u8,
            _ => RESTORE_NONE,
        };

        let unit_cols = self.lr.get(plane).map_or(0, |p| p.unit_cols);
        let idx = unit_row * unit_cols + unit_col;
        if let Some(p) = self.lr.get_mut(plane) {
            if let Some(t) = p.lr_type.get_mut(idx) {
                *t = restoration_type;
            }
        }

        if restoration_type == RESTORE_WIENER {
            let coeffs = match self.ref_lr_wiener.get_mut(plane) {
                Some(reference) => read_wiener_unit(dec, reference, plane != 0)?,
                None => return Ok(()),
            };
            if let Some(p) = self.lr.get_mut(plane) {
                if let Some(slot) = p.wiener.get_mut(idx) {
                    *slot = coeffs;
                }
            }
        } else if restoration_type == RESTORE_SGRPROJ {
            let (set, xqd) = match self.ref_sgr_xqd.get_mut(plane) {
                Some(reference) => read_sgrproj_unit(dec, reference)?,
                None => return Ok(()),
            };
            if let Some(p) = self.lr.get_mut(plane) {
                if let Some(s) = p.sgr_set.get_mut(idx) {
                    *s = set;
                }
                if let Some(x) = p.sgr_xqd.get_mut(idx) {
                    *x = xqd;
                }
            }
        }
        Ok(())
    }

    fn read_intra_frame_y_mode(
        &mut self,
        dec: &mut impl TileCoder,
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
        dec.symbol(cdf_row, Site::YMode)
    }

    /// Read `uv_mode` and, for a chroma-from-luma block, its alphas. Returns the
    /// UV mode and `Some((alphaU, alphaV))` when the block is CfL.
    fn read_uv_mode(
        &mut self,
        dec: &mut impl TileCoder,
        y_mode: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<(usize, Option<(i32, i32)>)> {
        // `is_cfl_allowed` (§5.11.5). Lossless allows chroma-from-luma only when
        // the chroma residual is 4x4 (for 4:4:4 a 4x4 block; for 4:2:0 anything
        // up to 8x8). Otherwise CfL is allowed for any block up to
        // 32x32 (Block_Width/Height <= 32, i.e. <= 8 mode-info units). Getting
        // this wrong picks the other uv_mode CDF — one has the extra CfL symbol,
        // the other does not — which desynchronises the whole tile.
        let cfl_allowed = if self.lossless {
            let block = block_size_index(bw4, bh4);
            plane_residual_size(block, self.subsampling_x, self.subsampling_y) == BLOCK_4X4
        } else {
            bw4 <= 8 && bh4 <= 8
        };
        let uv = if cfl_allowed {
            let cdf_row = get_mut(&mut self.cdfs.uv_cfl_allowed, y_mode)?;
            dec.symbol(cdf_row, Site::UvMode)?
        } else {
            let cdf_row = get_mut(&mut self.cdfs.uv_cfl_not_allowed, y_mode)?;
            dec.symbol(cdf_row, Site::UvMode)?
        };
        let cfl = if uv == UV_CFL_PRED {
            Some(self.read_cfl_alphas(dec)?)
        } else {
            None
        };
        Ok((uv, cfl))
    }

    /// `read_cfl_alphas` (§5.11.45): the signed U and V scaling factors.
    fn read_cfl_alphas(&mut self, dec: &mut impl TileCoder) -> Result<(i32, i32)> {
        let signs = dec.symbol(&mut self.cdfs.cfl_sign, Site::CflSign)? as i32;
        let sign_u = (signs + 1) / 3;
        let sign_v = (signs + 1) % 3;
        // CFL_SIGN_ZERO = 0, CFL_SIGN_NEG = 1, CFL_SIGN_POS = 2.
        let alpha_u = if sign_u != 0 {
            let ctx = ((sign_u - 1) * 3 + sign_v) as usize;
            let mag =
                dec.symbol(get_mut(&mut self.cdfs.cfl_alpha, ctx)?, Site::CflAlpha)? as i32 + 1;
            if sign_u == 1 { -mag } else { mag }
        } else {
            0
        };
        let alpha_v = if sign_v != 0 {
            let ctx = ((sign_v - 1) * 3 + sign_u) as usize;
            let mag =
                dec.symbol(get_mut(&mut self.cdfs.cfl_alpha, ctx)?, Site::CflAlpha)? as i32 + 1;
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
        dec: &mut impl TileCoder,
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
        // is only reached for blocks of at least four 4x4 units (log2 sum >= 2),
        // so the subtraction never underflows.
        let bsize_ctx = (floor_log2_usize(bw4) + floor_log2_usize(bh4)).saturating_sub(2) as usize;
        let avail_u = self.avail_u(r);
        let avail_l = self.avail_l(c);
        if y_mode == DC_PRED {
            let ctx = usize::from(avail_u && self.palette_size_at(0, r.wrapping_sub(1), c) > 0)
                + usize::from(avail_l && self.palette_size_at(0, r, c.wrapping_sub(1)) > 0);
            let cdf_row = get_mut(get_mut(&mut self.cdfs.palette_y_mode, bsize_ctx)?, ctx)?;
            if dec.symbol(cdf_row, Site::Other)? != 0 {
                let size_cdf = get_mut(&mut self.cdfs.palette_y_size, bsize_ctx)?;
                let size = dec.symbol(size_cdf, Site::Other)? + 2;
                let cache = self.palette_cache_for(0, r, c);
                palette.size_y = size;
                palette.colors_y = self.read_palette_colors(dec, size, &cache, true)?;
            }
        }
        if has_chroma && uv_mode == DC_PRED {
            let ctx = usize::from(palette.size_y > 0);
            let cdf_row = get_mut(&mut self.cdfs.palette_uv_mode, ctx)?;
            if dec.symbol(cdf_row, Site::Other)? != 0 {
                let size_cdf = get_mut(&mut self.cdfs.palette_uv_size, bsize_ctx)?;
                let size = dec.symbol(size_cdf, Site::Other)? + 2;
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
        let above = if self.avail_u(r) && (r * MI_SIZE) % 64 != 0 {
            let n = self.palette_size_at(plane, r - 1, c) as usize;
            self.palette_colors_at(plane, r - 1, c, n)
        } else {
            Vec::new()
        };
        let left = if self.avail_l(c) {
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
        dec: &mut impl TileCoder,
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
            if dec.literal(1, Site::Other)? != 0 {
                set_at(&mut colors, idx, cached);
                idx += 1;
            }
        }
        if idx < size {
            set_at(&mut colors, idx, dec.literal(bd, Site::Other)? as u16);
            idx += 1;
        }
        if idx < size {
            let min_bits = bd.saturating_sub(3);
            let mut palette_bits = min_bits + dec.literal(2, Site::Other)?;
            while idx < size {
                // The luma delta is coded one less than its value; the chroma
                // delta is coded directly (spec §5.11.47). The range that bounds
                // the next `paletteBits` likewise drops one only for luma.
                let delta = dec.literal(palette_bits, Site::Other)? + u32::from(is_luma);
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
        dec: &mut impl TileCoder,
        size: usize,
    ) -> Result<[u16; PALETTE_COLORS]> {
        let bd = u32::from(self.bit_depth);
        let max = (1_i32 << bd) - 1;
        let max_val = 1_i32 << bd;
        let mut colors = [0_u16; PALETTE_COLORS];
        if dec.literal(1, Site::Other)? != 0 {
            let mut palette_bits = bd.saturating_sub(4) + dec.literal(2, Site::Other)?;
            set_at(&mut colors, 0, dec.literal(bd, Site::Other)? as u16);
            for idx in 1..size {
                let mut delta = dec.literal(palette_bits, Site::Other)? as i32;
                if delta != 0 && dec.literal(1, Site::Other)? != 0 {
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
                set_at(&mut colors, idx, dec.literal(bd, Site::Other)? as u16);
            }
        }
        Ok(colors)
    }

    /// `palette_tokens` (§5.11.49): decode the colour-index maps by the
    /// wavefront traversal. Only the on-screen part of a block that overhangs the
    /// frame edge is coded; the rest of the map replicates its last column/row.
    fn read_palette_tokens(
        &mut self,
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        palette: &mut Palette,
    ) -> Result<()> {
        let (bw, bh) = (palette.block_w, palette.block_h);
        let onscreen_w = bw.min((self.mi_cols - c) * MI_SIZE);
        let onscreen_h = bh.min((self.mi_rows - r) * MI_SIZE);
        if palette.size_y > 0 {
            let dims = MapDims {
                w: bw,
                h: bh,
                onscreen_w,
                onscreen_h,
            };
            palette.map_y = self.read_color_map(dec, palette.size_y, dims, false)?;
        }
        if palette.size_uv > 0 {
            // The chroma map is the subsampled block, widened by 2 where that
            // leaves it under 4 samples (a 4xN luma block in 4:2:0).
            let (sub_x, sub_y) = (self.subsampling_x, self.subsampling_y);
            let mut dims = MapDims {
                w: bw >> sub_x,
                h: bh >> sub_y,
                onscreen_w: onscreen_w >> sub_x,
                onscreen_h: onscreen_h >> sub_y,
            };
            if dims.w < 4 {
                dims.w += 2;
                dims.onscreen_w += 2;
            }
            if dims.h < 4 {
                dims.h += 2;
                dims.onscreen_h += 2;
            }
            palette.uv_w = dims.w;
            palette.map_uv = self.read_color_map(dec, palette.size_uv, dims, true)?;
        }
        Ok(())
    }

    /// Decode one colour-index map (`ColorMapY`/`ColorMapUV`) of `dims`.
    fn read_color_map(
        &mut self,
        dec: &mut impl TileCoder,
        size: usize,
        dims: MapDims,
        chroma: bool,
    ) -> Result<Vec<u8>> {
        let (bw, bh) = (dims.onscreen_w, dims.onscreen_h);
        let stride = dims.w;
        let mut map = vec![0_u8; dims.w * dims.h];
        let first = dec.ns(size as u32)? as u8;
        if let Some(m) = map.first_mut() {
            *m = first;
        }
        let get =
            |map: &[u8], i: usize, j: usize| -> Option<u8> { map.get(i * stride + j).copied() };
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
                let sym = dec.symbol(cdf, Site::Other)?;
                let color = order.get(sym).copied().unwrap_or(0);
                if let Some(slot) = map.get_mut(row * stride + jj) {
                    *slot = color;
                }
                j -= 1;
            }
        }
        // Replicate the last on-screen column rightward, then the last on-screen
        // row downward, over the part of the block past the frame edge.
        for i in 0..bh {
            let last = get(&map, i, bw - 1).unwrap_or(0);
            for j in bw..dims.w {
                if let Some(slot) = map.get_mut(i * stride + j) {
                    *slot = last;
                }
            }
        }
        for i in bh..dims.h {
            for j in 0..dims.w {
                let v = get(&map, bh - 1, j).unwrap_or(0);
                if let Some(slot) = map.get_mut(i * stride + j) {
                    *slot = v;
                }
            }
        }
        Ok(map)
    }

    /// Read `angle_delta` for a directional mode on a `MiSize >= BLOCK_8X8`
    /// block, returning the signed delta (`angle_delta - MAX_ANGLE_DELTA`). Zero
    /// for non-directional modes and small blocks, which read nothing.
    fn read_angle_delta(
        &mut self,
        dec: &mut impl TileCoder,
        mode: usize,
        bw4: usize,
        bh4: usize,
        site: Site,
    ) -> Result<i32> {
        let directional = (1..=8).contains(&mode);
        if directional && at_least_block_8x8(bw4, bh4) {
            let index = mode - 1;
            let cdf_row = get_mut(&mut self.cdfs.angle_delta, index)?;
            let symbol = dec.symbol(cdf_row, site)? as i32;
            return Ok(symbol - MAX_ANGLE_DELTA);
        }
        Ok(0)
    }

    /// `filter_intra_mode_info` (§5.11.10): whether luma uses recursive
    /// filter-intra, and if so which of the five kernels.
    fn read_filter_intra(
        &mut self,
        dec: &mut impl TileCoder,
        y_mode: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<Option<usize>> {
        let max_dim = bw4.max(bh4) * MI_SIZE;
        if self.enable_filter_intra && y_mode == DC_PRED && max_dim <= 32 {
            let size = block_size_index(bw4, bh4);
            let cdf_row = get_mut(&mut self.cdfs.filter_intra, size)?;
            if dec.symbol(cdf_row, Site::Other)? != 0 {
                let mode = dec.symbol(&mut self.cdfs.filter_intra_mode, Site::Other)?;
                return Ok(Some(mode));
            }
        }
        Ok(None)
    }

    /// `read_block_tx_size` (§5.11.16) for an intra block: resolve the luma
    /// transform size and record it across the block's 4x4 units for the
    /// `tx_depth` neighbour context. Intra frames have `is_inter == 0`, so the
    /// selection gate is always open.
    fn read_block_tx_size(
        &mut self,
        dec: &mut impl TileCoder,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        _skip: bool,
    ) -> Result<TxSize> {
        let above_w = if self.avail_u(r) {
            self.tx_width_at(r - 1, c)
        } else {
            0
        };
        let left_h = if self.avail_l(c) {
            self.tx_height_at(r, c - 1)
        } else {
            0
        };
        let params = TxSizeParams {
            block: block_size_index(bw4, bh4),
            tx_mode_select: self.tx_mode == TxMode::Select,
            lossless: self.lossless,
            allow_select: true,
            above_w,
            left_h,
        };
        let tx = code_tx_size(dec, &mut self.cdfs.tx_depth, &params)?;
        for y in r..(r + bh4).min(self.mi_rows) {
            for x in c..(c + bw4).min(self.mi_cols) {
                if let Some(v) = self.tx_sizes.get_mut(y * self.mi_cols + x) {
                    *v = tx as u8;
                }
            }
        }
        Ok(tx)
    }

    /// `Tx_Width[InterTxSizes[r][c]]`: the stored luma transform width at a unit.
    fn tx_width_at(&self, r: usize, c: usize) -> usize {
        self.tx_sizes
            .get(r * self.mi_cols + c)
            .map_or(0, |&i| TxSize::from_index(usize::from(i)).width())
    }

    /// `Tx_Height[InterTxSizes[r][c]]`: the stored luma transform height.
    fn tx_height_at(&self, r: usize, c: usize) -> usize {
        self.tx_sizes
            .get(r * self.mi_cols + c)
            .map_or(0, |&i| TxSize::from_index(usize::from(i)).height())
    }

    /// `residual` (§5.11.34): every transform block of every plane the block
    /// codes. A block wider or taller than 64 is walked in 64x64 chunks, each
    /// chunk doing all its planes before the next; each plane steps its own
    /// (subsampled) residual block in that plane's transform size.
    fn residual(
        &mut self,
        dec: &mut impl TileCoder,
        modes: &BlockModes,
        bw4: usize,
        bh4: usize,
        skip: bool,
        has_chroma: bool,
    ) -> Result<()> {
        let planes = if has_chroma { self.num_planes } else { 1 };
        let block = block_size_index(bw4, bh4);
        let width_chunks = (bw4 / 16).max(1);
        let height_chunks = (bh4 / 16).max(1);
        let chunk_size = if width_chunks > 1 || height_chunks > 1 {
            BLOCK_64X64
        } else {
            block
        };
        for chunk_y in 0..height_chunks {
            for chunk_x in 0..width_chunks {
                for plane in 0..planes {
                    self.residual_plane(
                        dec,
                        modes,
                        block,
                        chunk_size,
                        (chunk_x, chunk_y),
                        plane,
                        skip,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// One plane of one `residual` chunk (§5.11.34).
    #[allow(clippy::too_many_arguments, reason = "mirrors the residual loop state")]
    fn residual_plane(
        &mut self,
        dec: &mut impl TileCoder,
        modes: &BlockModes,
        block: usize,
        chunk_size: usize,
        (chunk_x, chunk_y): (usize, usize),
        plane: usize,
        skip: bool,
    ) -> Result<()> {
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        // A CfL chroma block predicts from DC and then adds the scaled luma.
        let cfl_alpha = match plane {
            1 => modes.cfl.map(|(u, _)| u),
            2 => modes.cfl.map(|(_, v)| v),
            _ => None,
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
        // The block's residual size on this plane (for the coefficient
        // contexts), and the chunk's (for the loop bounds).
        let plane_block = plane_residual_size(block, sub_x, sub_y);
        if plane_block == BLOCK_INVALID {
            return Err(PixelsError::malformed(
                "avif",
                "a block shape the chroma subsampling cannot represent",
            ));
        }
        let (plane_bw4, plane_bh4) = block_4x4_dims(plane_block);
        let (num_w, num_h) = block_4x4_dims(plane_residual_size(chunk_size, sub_x, sub_y));
        // The plane's transform size: lossless forces TX_4X4 for every plane (the
        // WHT is 4x4 only); otherwise luma is the read block size and chroma is
        // its residual block's largest rectangular transform (`get_tx_size`).
        let tx_size = if self.lossless {
            TxSize::Tx4x4
        } else if plane == 0 {
            modes.luma_tx_size
        } else {
            chroma_tx_size(plane_block)
        };
        let step_x = (tx_size.width() / MI_SIZE).max(1);
        let step_y = (tx_size.height() / MI_SIZE).max(1);
        // Everything below is in this plane's sample grid.
        let base_x = (modes.c >> sub_x) * MI_SIZE;
        let base_y = (modes.r >> sub_y) * MI_SIZE;
        let palette = self.palette_view_for(&modes.palette, plane, base_x, base_y);
        let (avail_l, avail_u) = if plane == 0 {
            (modes.avail_l, modes.avail_u)
        } else {
            (modes.avail_l_chroma, modes.avail_u_chroma)
        };
        let mut y = 0;
        while y < num_h {
            let mut x = 0;
            while x < num_w {
                // Offset of this transform block within the whole block, in the
                // plane's 4x4 units.
                let bx = x + ((chunk_x << 4) >> sub_x);
                let by = y + ((chunk_y << 4) >> sub_y);
                let tb = TxBlock {
                    plane,
                    x: base_x + bx * MI_SIZE,
                    y: base_y + by * MI_SIZE,
                    tx_size,
                    mode: intra,
                    mode_index: mode,
                    angle_delta: delta,
                    have_left: avail_l || bx > 0,
                    have_above: avail_u || by > 0,
                    filter_type,
                    // Filter-intra is a luma-only tool.
                    filter_intra: if plane == 0 { modes.filter_intra } else { None },
                    cfl_alpha,
                    palette,
                    skip,
                    plane_bw4,
                    plane_bh4,
                };
                self.transform_block(dec, &tb)?;
                x += step_x;
            }
            y += step_y;
        }
        Ok(())
    }

    /// `(maxX, maxY)` for intra edge fetches on `plane` (§7.11.2): the last
    /// sample column and row of the decoded (mode-info) area in that plane. The
    /// plane buffers extend further, to whole superblocks.
    fn frame_bounds(&self, plane: usize) -> (usize, usize) {
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        (
            ((self.mi_cols * MI_SIZE) >> sub_x) - 1,
            ((self.mi_rows * MI_SIZE) >> sub_y) - 1,
        )
    }

    /// `(sbMask >> subX, sbMask >> subY)`: the superblock masks in `plane`'s
    /// 4x4 units along each axis, which index `BlockDecoded[plane]`.
    fn plane_sb_mask_xy(&self, plane: usize) -> (isize, isize) {
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let mask = self.sb_size4 - 1;
        ((mask >> sub_x) as isize, (mask >> sub_y) as isize)
    }

    /// `(subsampling_x, subsampling_y)` for `plane`: zero for luma.
    fn plane_subsampling(&self, plane: usize) -> (usize, usize) {
        if plane == 0 {
            (0, 0)
        } else {
            (self.subsampling_x, self.subsampling_y)
        }
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
                block_w: palette.uv_w,
                base_x,
                base_y,
            })
        } else if plane == 2 && palette.size_uv > 0 {
            Some(PaletteView {
                map: &palette.map_uv,
                colors: &palette.colors_v,
                block_w: palette.uv_w,
                base_x,
                base_y,
            })
        } else {
            None
        }
    }

    /// `get_filter_type` (§7.11.2.8): whether the above or left neighbour block
    /// used a smooth mode, which softens the directional edge filter. On a
    /// subsampled chroma plane the neighbour is looked up at the unit that owns
    /// the co-located chroma (the odd column/row of each pair).
    /// What intra `mode` would predict for the whole of plane `plane` of the
    /// block at `(r, c)`, as the decoder will predict its first transform
    /// block — exact for an encoder that codes every block of 8x8 or more
    /// with one transform. `None` for a mode or shape this cannot preview.
    pub(crate) fn preview_prediction(
        &self,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        plane: usize,
        mode: usize,
    ) -> Option<Vec<u16>> {
        let (avail_u, avail_l) = (self.avail_u(r), self.avail_l(c));
        let modes = BlockModes {
            r,
            c,
            avail_u,
            avail_l,
            avail_u_chroma: avail_u,
            avail_l_chroma: avail_l,
            y_mode: mode,
            uv_mode: mode,
            y_delta: 0,
            uv_delta: 0,
            filter_intra: None,
            cfl: None,
            palette: Palette::default(),
            luma_tx_size: TxSize::Tx4x4,
        };
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let block = block_size_index(bw4, bh4);
        let plane_block = plane_residual_size(block, sub_x, sub_y);
        if plane_block == BLOCK_INVALID {
            return None;
        }
        let (plane_bw4, plane_bh4) = block_4x4_dims(plane_block);
        let tx_size = if plane == 0 {
            max_tx_size_rect(block)
        } else {
            chroma_tx_size(plane_block)
        };
        let tb = TxBlock {
            plane,
            x: (c >> sub_x) * MI_SIZE,
            y: (r >> sub_y) * MI_SIZE,
            tx_size,
            mode: IntraMode::from_index(mode as u8)?,
            mode_index: mode,
            angle_delta: 0,
            have_left: avail_l,
            have_above: avail_u,
            filter_type: self.filter_type(&modes, plane),
            filter_intra: None,
            cfl_alpha: None,
            palette: None,
            skip: false,
            plane_bw4,
            plane_bh4,
        };
        self.predict(&tb, tx_size.width(), tx_size.height()).ok()
    }

    /// The decoder state for `seq` and `frame`, ready to code tiles into.
    pub(crate) fn for_frame(seq: &SequenceHeader, frame: &FrameHeader) -> Result<Self> {
        Self::new(seq, frame)
    }

    /// `MiRows` and `MiCols`.
    pub(crate) const fn mi_dims(&self) -> (usize, usize) {
        (self.mi_rows, self.mi_cols)
    }

    fn filter_type(&self, modes: &BlockModes, plane: usize) -> bool {
        let is_smooth = |mode: usize| (9..=11).contains(&mode);
        let smooth_at = |r: usize, c: usize| {
            is_smooth(if plane == 0 {
                self.y_mode_at(r, c)
            } else {
                self.uv_mode_at(r, c)
            })
        };
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let (r, c) = (modes.r, modes.c);
        let (avail_u, avail_l) = if plane == 0 {
            (modes.avail_u, modes.avail_l)
        } else {
            (modes.avail_u_chroma, modes.avail_l_chroma)
        };
        let above = avail_u && {
            let row = r - 1 - usize::from(sub_y == 1 && r & 1 == 1);
            let col = c + usize::from(sub_x == 1 && c & 1 == 0);
            smooth_at(row, col)
        };
        let left = avail_l && {
            let row = r + usize::from(sub_y == 1 && r & 1 == 0);
            let col = c - 1 - usize::from(sub_x == 1 && c & 1 == 1);
            smooth_at(row, col)
        };
        above || left
    }

    fn transform_block(&mut self, dec: &mut impl TileCoder, tb: &TxBlock) -> Result<()> {
        let (plane, x, y, tx_size, skip) = (tb.plane, tb.x, tb.y, tb.tx_size, tb.skip);
        let w = tx_size.width();
        let h = tx_size.height();
        let w4 = (w / MI_SIZE).max(1);
        let h4 = (h / MI_SIZE).max(1);

        // `x`/`y` are in this plane's sample grid. A transform block whose
        // top-left lies outside the (plane's) frame is not coded: the block may
        // extend past the right or bottom edge, but only the tx blocks that
        // start inside it read symbols (spec §5.11.35). Skipping this
        // desynchronises every symbol after the edge — for a last-region block
        // that surfaces as wrong chroma while the luma before it stays correct.
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let max_x = (self.mi_cols * MI_SIZE) >> sub_x;
        let max_y = (self.mi_rows * MI_SIZE) >> sub_y;
        if x >= max_x || y >= max_y {
            return Ok(());
        }

        // Predict from the reconstructed neighbours.
        let prediction = self.predict(tb, w, h)?;
        if plane == 0 {
            // MaxLumaW/MaxLumaH: how far this block's luma reaches, which bounds
            // the luma a chroma-from-luma prediction may read.
            self.max_luma_w = x + w;
            self.max_luma_h = y + h;
        }

        let x4 = x / MI_SIZE;
        let y4 = y / MI_SIZE;
        let final_block = if skip {
            prediction
        } else {
            let all_zero_ctx =
                self.all_zero_ctx(plane, x4, y4, w4, h4, tx_size, tb.plane_bw4, tb.plane_bh4);
            let dc_sign_ctx = self.dc_sign_ctx(plane, x4, y4, w4, h4);
            let ptype = usize::from(plane > 0);
            // Resolve the block's PlaneTxType inside decode_coeffs, at the spec
            // position between all_zero and eob_pt. Luma reads the intra_tx_type
            // symbol against these contexts; chroma derives from its mode. When
            // qindex is 0 (lossless) no symbol is read and the type is DCT_DCT.
            let tx_set = intra_tx_set(tx_size, self.reduced_tx_set);
            let dir = if plane == 0 {
                intra_dir(tb.mode_index, tb.filter_intra)
            } else {
                0
            };
            // transform_type (§5.11.47) gates the luma symbol on the segment's
            // quantizer before any delta_q.
            let qindex_positive = self.segment_qindex(true) > 0;
            // The block's quantizers, which an encoder needs before it codes
            // the coefficients and the decoder after.
            let qindex = self.segment_qindex(false);
            let dc = dc_q(
                self.bit_depth,
                qindex + self.q_dc.get(plane).copied().unwrap_or(0),
            );
            let ac = ac_q(
                self.bit_depth,
                qindex + self.q_ac.get(plane).copied().unwrap_or(0),
            );
            let tx_ctx = TxTypeCtx {
                set: tx_set,
                intra_cdfs: &mut self.cdfs.intra_tx_type,
                intra_dir: dir,
                uv_mode: tb.mode_index,
                qindex_positive,
                lossless: self.lossless,
            };
            let job = CoeffJob {
                plane,
                x,
                y,
                prediction: &prediction,
                dc_q: dc,
                ac_q: ac,
            };
            let block = dec.coeffs(
                &mut self.cdfs.coeff,
                tx_size,
                tx_ctx,
                ptype,
                all_zero_ctx,
                dc_sign_ctx,
                &job,
            )?;
            self.update_level_context(plane, x4, y4, w4, h4, block.cul_level, block.dc_category);
            if block.eob > 0 {
                // §7.12.3 step 1b: a matrix weights only the 2D transforms
                // (types before IDTX), and level 15 means none.
                // SegQMLevel (§5.9.12): 15, meaning none, for a lossless segment.
                let level = if self.using_qmatrix && !self.lossless {
                    self.qm.get(plane).copied().unwrap_or(15)
                } else {
                    15
                };
                let matrix = ((block.tx_type as usize) < TxType::Idtx as usize)
                    .then(|| quantizer_matrix(level, plane > 0, tx_size))
                    .flatten();
                let dequant =
                    dequantize_with_matrix(&block.quant, tx_size, dc, ac, matrix, self.bit_depth);
                let residual = inverse_transform_2d(
                    &dequant,
                    tx_size,
                    block.tx_type,
                    self.lossless,
                    self.bit_depth,
                );
                add_residual(&prediction, &residual, block.tx_type, self.bit_depth)
            } else {
                prediction
            }
        };

        if let Some(p) = self.planes.get_mut(plane) {
            for i in 0..h {
                for j in 0..w {
                    let value = final_block.get(i * w + j).copied().unwrap_or(0);
                    p.set(x + j, y + i, value);
                }
            }
        }

        // Mark the tx block's 4x4 units decoded for the neighbour tests, and
        // record its transform size per unit for the deblocking loop filter —
        // both in the plane's 4x4 units.
        let (mask_x, mask_y) = self.plane_sb_mask_xy(plane);
        let tx_index = tx_size as u8;
        for dy in 0..h4 {
            for dx in 0..w4 {
                let sub_row = ((y4 + dy) as isize) & mask_y;
                let sub_col = ((x4 + dx) as isize) & mask_x;
                self.set_block_decoded(plane, sub_row, sub_col);
                if let Some(grid) = self.lf_tx_sizes.get_mut(plane) {
                    if let Some(cell) = grid.get_mut((y4 + dy) * self.mi_cols + (x4 + dx)) {
                        *cell = tx_index;
                    }
                }
            }
        }
        Ok(())
    }

    /// Predict a `w` by `h` transform block: directional modes go through the
    /// edge machinery, the rest through the pure non-directional predictors. The
    /// result is `w * h` samples in row-major order.
    fn predict(&self, tb: &TxBlock, w: usize, h: usize) -> Result<Vec<u16>> {
        if let Some(pv) = tb.palette {
            // predict_palette (§7.11.4): each sample is the palette colour its
            // index map selects. The map is block-relative.
            let mut pred = vec![0_u16; w * h];
            for i in 0..h {
                for j in 0..w {
                    let my = (tb.y + i).saturating_sub(pv.base_y);
                    let mx = (tb.x + j).saturating_sub(pv.base_x);
                    let idx = pv.map.get(my * pv.block_w + mx).copied().unwrap_or(0);
                    if let Some(cell) = pred.get_mut(i * w + j) {
                        *cell = pv.colors.get(usize::from(idx)).copied().unwrap_or(0);
                    }
                }
            }
            return Ok(pred);
        }
        if let Some(filter_mode) = tb.filter_intra {
            let (above, left) = self.gather_edges(tb, w, h);
            return Ok(predict_filter_intra(
                filter_mode,
                &above,
                &left,
                w,
                h,
                self.bit_depth,
            ));
        }
        if let Some(base_angle) = mode_base_angle(tb.mode_index) {
            let p_angle = base_angle + tb.angle_delta * ANGLE_STEP;
            if p_angle != 90 && p_angle != 180 {
                return Ok(self.predict_directional(tb, p_angle, w, h));
            }
        }
        let (above, left, corner, have_above, have_left) =
            self.gather_neighbours(tb.plane, tb.x, tb.y, tb.have_left, tb.have_above, w, h);
        let block = PredBlock {
            above: &above,
            left: &left,
            corner,
            have_above,
            have_left,
            w,
            h,
        };
        let mut pred = predict_intra_block(tb.mode, &block, self.bit_depth)?;
        if let Some(alpha) = tb.cfl_alpha {
            self.apply_cfl(&mut pred, tb.x, tb.y, w, h, alpha);
        }
        Ok(pred)
    }

    /// `predict_chroma_from_luma` (§7.11.5) for a `w` by `h` chroma block at
    /// chroma `(x, y)`: add the alpha-scaled, DC-removed reconstructed luma to
    /// the DC chroma prediction. With subsampling each chroma sample averages
    /// its 2 (4:2:2) or 4 (4:2:0) co-located luma samples, clamped to the luma
    /// the block actually reconstructed (`MaxLumaW`/`MaxLumaH`).
    fn apply_cfl(&self, pred: &mut [u16], x: usize, y: usize, w: usize, h: usize, alpha: i32) {
        let max = (1_i32 << self.bit_depth) - 1;
        let luma = self.planes.first();
        let (sub_x, sub_y) = (self.subsampling_x, self.subsampling_y);
        let luma_at =
            |lx: usize, ly: usize| i32::from(luma.and_then(|p| p.get(lx, ly)).unwrap_or(0));
        // L holds the (subsampled) co-located luma with 3 fractional bits.
        let mut l = vec![0_i32; w * h];
        let mut sum = 0_i32;
        for i in 0..h {
            let luma_y = ((y + i) << sub_y).min(self.max_luma_h.saturating_sub(1 << sub_y));
            for j in 0..w {
                let luma_x = ((x + j) << sub_x).min(self.max_luma_w.saturating_sub(1 << sub_x));
                let mut t = 0;
                for dy in 0..=sub_y {
                    for dx in 0..=sub_x {
                        t += luma_at(luma_x + dx, luma_y + dy);
                    }
                }
                let v = t << (3 - sub_x - sub_y);
                if let Some(cell) = l.get_mut(i * w + j) {
                    *cell = v;
                }
                sum += v;
            }
        }
        // lumaAvg = Round2(sum, log2W + log2H).
        let shift = w.trailing_zeros() + h.trailing_zeros();
        let luma_avg = (sum + (1 << (shift - 1))) >> shift;
        for i in 0..h {
            for j in 0..w {
                let ac = l.get(i * w + j).copied().unwrap_or(0) - luma_avg;
                let scaled = round2_signed(alpha * ac, 6);
                if let Some(cell) = pred.get_mut(i * w + j) {
                    *cell = (i32::from(*cell) + scaled).clamp(0, max) as u16;
                }
            }
        }
    }

    /// Build the extended `AboveRow`/`LeftCol` edge arrays for a `w` by `h`
    /// block (§7.11.2 general), with the `haveAboveRight`/`haveBelowLeft`
    /// extension from `BlockDecoded`. The same edge serves every prediction mode;
    /// the non-directional modes simply never read past index `w`/`h`.
    fn gather_edges(&self, tb: &TxBlock, w: usize, h: usize) -> (Edge, Edge) {
        let (plane, x, y) = (tb.plane, tb.x, tb.y);
        let mid = 1_i32 << (self.bit_depth - 1);
        let p = self.planes.get(plane);
        let at =
            |px: usize, py: usize| -> i32 { p.and_then(|pl| pl.get(px, py)).map_or(0, i32::from) };
        let (max_x, max_y) = self.frame_bounds(plane);

        let x4 = x / MI_SIZE;
        let y4 = y / MI_SIZE;
        let w4 = (w / MI_SIZE).max(1);
        let h4 = (h / MI_SIZE).max(1);
        let (mask_x, mask_y) = self.plane_sb_mask_xy(plane);
        let sub_row = (y4 as isize) & mask_y;
        let sub_col = (x4 as isize) & mask_x;
        let have_above_right = self.block_decoded_at(plane, sub_row - 1, sub_col + w4 as isize);
        let have_below_left = self.block_decoded_at(plane, sub_row + h4 as isize, sub_col - 1);

        let mut above = Edge::new();
        let mut left = Edge::new();
        let num = (w + h) as isize;
        // AboveRow[0..w+h-1].
        if tb.have_above {
            let extent = if have_above_right { 2 * w } else { w };
            let above_limit = (x + extent - 1).min(max_x);
            for i in 0..num {
                above.set(i, at((x + i as usize).min(above_limit), y.wrapping_sub(1)));
            }
        } else if tb.have_left {
            let v = at(x.wrapping_sub(1), y);
            for i in 0..num {
                above.set(i, v);
            }
        } else {
            for i in 0..num {
                above.set(i, mid - 1);
            }
        }
        // LeftCol[0..w+h-1].
        if tb.have_left {
            let extent = if have_below_left { 2 * h } else { h };
            let left_limit = (y + extent - 1).min(max_y);
            for i in 0..num {
                left.set(i, at(x.wrapping_sub(1), (y + i as usize).min(left_limit)));
            }
        } else if tb.have_above {
            let v = at(x, y.wrapping_sub(1));
            for i in 0..num {
                left.set(i, v);
            }
        } else {
            for i in 0..num {
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

    /// Slanted directional prediction (§7.11.2.4) at any size, returned as `w * h`
    /// row-major samples.
    fn predict_directional(&self, tb: &TxBlock, p_angle: i32, w: usize, h: usize) -> Vec<u16> {
        let (x, y) = (tb.x, tb.y);
        let (mut above, mut left) = self.gather_edges(tb, w, h);
        let (max_x, max_y) = self.frame_bounds(tb.plane);
        let avail_above_px = (max_x as i32) - (x as i32) + 1;
        let avail_left_px = (max_y as i32) - (y as i32) + 1;
        predict_directional(
            p_angle,
            &mut above,
            &mut left,
            w,
            h,
            tb.have_left,
            tb.have_above,
            tb.filter_type,
            self.enable_edge_filter,
            avail_above_px,
            avail_left_px,
            self.bit_depth,
        )
    }

    /// Extract the `AboveRow[0..w]`, `LeftCol[0..h]` and corner from the edge
    /// arrays as the non-directional predictors consume them.
    #[allow(clippy::too_many_arguments, reason = "mirrors the §7.11.2 edge inputs")]
    fn gather_neighbours(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        have_left: bool,
        have_above: bool,
        w: usize,
        h: usize,
    ) -> (Vec<i32>, Vec<i32>, i32, bool, bool) {
        let tb = TxBlock {
            plane,
            x,
            y,
            tx_size: TxSize::Tx4x4,
            mode: IntraMode::Dc,
            mode_index: 0,
            angle_delta: 0,
            have_left,
            have_above,
            filter_type: false,
            filter_intra: None,
            cfl_alpha: None,
            palette: None,
            skip: false,
            plane_bw4: 0,
            plane_bh4: 0,
        };
        let (above_edge, left_edge) = self.gather_edges(&tb, w, h);
        let above: Vec<i32> = (0..w as isize).map(|j| above_edge.get(j)).collect();
        let left: Vec<i32> = (0..h as isize).map(|i| left_edge.get(i)).collect();
        (above, left, above_edge.get(-1), have_above, have_left)
    }

    /// `all_zero` context (§8.3.2) for a `w4` by `h4` (4x4 units) transform block.
    /// A block whose coding size equals the transform is context 0 for luma;
    /// otherwise the neighbour level contexts over the block's span select it.
    /// `bw4`/`bh4` are the coding block's residual size on this plane, in the
    /// plane's 4x4 units.
    #[allow(clippy::too_many_arguments, reason = "mirrors the §8.3.2 ctx inputs")]
    fn all_zero_ctx(
        &self,
        plane: usize,
        x4: usize,
        y4: usize,
        w4: usize,
        h4: usize,
        tx_size: TxSize,
        bw4: usize,
        bh4: usize,
    ) -> usize {
        let Some(ctx) = self.ctx.get(plane) else {
            return 0;
        };
        // maxX4/maxY4: the frame's mode-info dimensions in this plane's units.
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let (max_x4, max_y4) = (self.mi_cols >> sub_x, self.mi_rows >> sub_y);
        let w = tx_size.width();
        let h = tx_size.height();
        if plane == 0 {
            let mut top = 0_u32;
            let mut left = 0_u32;
            for k in 0..w4 {
                if x4 + k < max_x4 {
                    top = top.max(u32::from(ctx.above_level.get(x4 + k).copied().unwrap_or(0)));
                }
            }
            for k in 0..h4 {
                if y4 + k < max_y4 {
                    left = left.max(u32::from(ctx.left_level.get(y4 + k).copied().unwrap_or(0)));
                }
            }
            let top = top.min(255);
            let left = left.min(255);
            if bw4 * MI_SIZE == w && bh4 * MI_SIZE == h {
                0
            } else if top == 0 && left == 0 {
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
            let mut above = 0_u8;
            let mut left = 0_u8;
            for i in 0..w4 {
                if x4 + i < max_x4 {
                    above |= ctx.above_level.get(x4 + i).copied().unwrap_or(0);
                    above |= ctx.above_dc.get(x4 + i).copied().unwrap_or(0);
                }
            }
            for i in 0..h4 {
                if y4 + i < max_y4 {
                    left |= ctx.left_level.get(y4 + i).copied().unwrap_or(0);
                    left |= ctx.left_dc.get(y4 + i).copied().unwrap_or(0);
                }
            }
            let mut c = 7 + usize::from(above != 0) + usize::from(left != 0);
            if bw4 * MI_SIZE * bh4 * MI_SIZE > w * h {
                c += 3;
            }
            c
        }
    }

    /// `dc_sign` context (§8.3.2) over a `w4` by `h4` transform block.
    fn dc_sign_ctx(&self, plane: usize, x4: usize, y4: usize, w4: usize, h4: usize) -> usize {
        let Some(ctx) = self.ctx.get(plane) else {
            return 0;
        };
        let (sub_x, sub_y) = self.plane_subsampling(plane);
        let (max_x4, max_y4) = (self.mi_cols >> sub_x, self.mi_rows >> sub_y);
        let mut dc_sign = 0_i32;
        for k in 0..w4 {
            if x4 + k < max_x4 {
                match ctx.above_dc.get(x4 + k).copied().unwrap_or(0) {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        for k in 0..h4 {
            if y4 + k < max_y4 {
                match ctx.left_dc.get(y4 + k).copied().unwrap_or(0) {
                    1 => dc_sign -= 1,
                    2 => dc_sign += 1,
                    _ => {}
                }
            }
        }
        if dc_sign < 0 {
            1
        } else if dc_sign > 0 {
            2
        } else {
            0
        }
    }

    /// Spread the block's `culLevel`/`dcCategory` across the `w4` above columns
    /// and `h4` left rows it covers (§7.12.3 / level-context update).
    #[allow(clippy::too_many_arguments, reason = "one level entry per plane axis")]
    fn update_level_context(
        &mut self,
        plane: usize,
        x4: usize,
        y4: usize,
        w4: usize,
        h4: usize,
        cul: u8,
        dc: u8,
    ) {
        if let Some(ctx) = self.ctx.get_mut(plane) {
            for i in 0..w4 {
                if let Some(v) = ctx.above_level.get_mut(x4 + i) {
                    *v = cul;
                }
                if let Some(v) = ctx.above_dc.get_mut(x4 + i) {
                    *v = dc;
                }
            }
            for i in 0..h4 {
                if let Some(v) = ctx.left_level.get_mut(y4 + i) {
                    *v = cul;
                }
                if let Some(v) = ctx.left_dc.get_mut(y4 + i) {
                    *v = dc;
                }
            }
        }
    }

    /// `reset_block_context` (§5.11.5) for a skip block: clear the level and DC
    /// contexts it covers, on each plane it codes, in that plane's 4x4 units.
    fn reset_block_context(
        &mut self,
        r: usize,
        c: usize,
        bw4: usize,
        bh4: usize,
        has_chroma: bool,
    ) {
        let planes = if has_chroma { self.num_planes } else { 1 };
        for plane in 0..planes {
            let (sub_x, sub_y) = self.plane_subsampling(plane);
            let Some(ctx) = self.ctx.get_mut(plane) else {
                continue;
            };
            for i in (c >> sub_x)..((c + bw4) >> sub_x) {
                if let Some(v) = ctx.above_level.get_mut(i) {
                    *v = 0;
                }
                if let Some(v) = ctx.above_dc.get_mut(i) {
                    *v = 0;
                }
            }
            for i in (r >> sub_y)..((r + bh4) >> sub_y) {
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
        has_chroma: bool,
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
                // UVModes keeps the previous value where this block codes no
                // chroma: a later chroma block's filter type reads it.
                if has_chroma {
                    if let Some(v) = self.uv_modes.get_mut(idx) {
                        *v = uv_mode as u8;
                    }
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
    /// `AvailUChroma`/`AvailLChroma`: neighbour availability on the chroma
    /// planes, which differs from luma for a subsampled block one unit wide or
    /// high (§5.11.5).
    avail_u_chroma: bool,
    avail_l_chroma: bool,
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
    /// The luma transform size (`read_block_tx_size`); chroma derives its own.
    luma_tx_size: TxSize,
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
    /// `ColorMapUV`, `uv_w` wide (the subsampled block, widened to at least 4).
    map_uv: Vec<u8>,
    /// The luma block width and height in samples.
    block_w: usize,
    block_h: usize,
    /// The chroma colour-index map's width.
    uv_w: usize,
}

/// A colour-index map's dimensions and the part of it on screen.
#[derive(Clone, Copy)]
struct MapDims {
    w: usize,
    h: usize,
    onscreen_w: usize,
    onscreen_h: usize,
}

/// One transform block's prediction inputs.
struct TxBlock<'a> {
    plane: usize,
    x: usize,
    y: usize,
    /// This transform block's size.
    tx_size: TxSize,
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
    /// The coding block's residual size on this plane, in the plane's 4x4 units
    /// (`get_plane_residual_size(MiSize, plane)`), for the `all_zero` context.
    plane_bw4: usize,
    plane_bh4: usize,
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

/// `get_tx_size` for a chroma plane (§5.11.37), given the block's residual size
/// on that plane: its largest rectangular transform, with any 64-sample side
/// reduced to 32 (chroma codes no 64-wide/high transform).
/// `neg_deinterleave` (§5.11.9): undo the coding of a `segment_id` as its
/// distance from the predicted one, alternating either side of it.
fn neg_deinterleave(diff: i32, reference: i32, max: i32) -> i32 {
    if reference == 0 {
        return diff;
    }
    if reference >= max - 1 {
        return max - diff - 1;
    }
    if 2 * reference < max {
        if diff <= 2 * reference {
            return if diff & 1 == 1 {
                reference + ((diff + 1) >> 1)
            } else {
                reference - (diff >> 1)
            };
        }
        return diff;
    }
    if diff <= 2 * (max - reference - 1) {
        if diff & 1 == 1 {
            reference + ((diff + 1) >> 1)
        } else {
            reference - (diff >> 1)
        }
    } else {
        max - (diff + 1)
    }
}

/// The magnitude-and-sign coding shared by `delta_qindex` and `delta_lf`: a
/// symbol up to `DELTA_Q_SMALL` (= `DELTA_LF_SMALL` = 3), escaping to a
/// literal-length literal, then a sign bit when nonzero.
fn read_delta(dec: &mut impl TileCoder, cdf: &mut [u16]) -> Result<i32> {
    let mut abs = dec.symbol(cdf, Site::Other)? as i32;
    if abs == DELTA_SMALL {
        let rem_bits = dec.literal(3, Site::Other)? + 1;
        let abs_bits = dec.literal(rem_bits, Site::Other)? as i32;
        abs = abs_bits + (1 << rem_bits) + 1;
    }
    if abs != 0 && dec.literal(1, Site::Other)? == 1 {
        abs = -abs;
    }
    Ok(abs)
}

fn chroma_tx_size(block: usize) -> TxSize {
    match max_tx_size_rect(block) {
        TxSize::Tx64x64 | TxSize::Tx32x64 | TxSize::Tx64x32 => TxSize::Tx32x32,
        TxSize::Tx16x64 => TxSize::Tx16x32,
        TxSize::Tx64x16 => TxSize::Tx32x16,
        other => other,
    }
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

/// `BLOCK_64X64` (§9.3): the chunk size `residual` walks larger blocks in.
const BLOCK_64X64: usize = 12;

/// `BLOCK_INVALID`: a block shape a subsampled plane cannot take.
const BLOCK_INVALID: usize = usize::MAX;

/// `Num_4x4_Blocks_Wide[BLOCK_SIZES]` / `Num_4x4_Blocks_High[BLOCK_SIZES]`
/// (§9.3): a block size's dimensions in 4x4 units, the inverse of
/// `block_size_index`.
const BLOCK_4X4_DIMS: [(usize, usize); 22] = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 4),
    (8, 8),
    (8, 16),
    (16, 8),
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (1, 4),
    (4, 1),
    (2, 8),
    (8, 2),
    (4, 16),
    (16, 4),
];

/// `Subsampled_Size[BLOCK_SIZES][subX][subY]` (§5.11.38): the residual block a
/// plane with the given subsampling takes for each luma block size.
const SUBSAMPLED_SIZE: [[[usize; 2]; 2]; 22] = {
    const I: usize = BLOCK_INVALID;
    [
        [[0, 0], [0, 0]],
        [[1, 0], [I, 0]],
        [[2, I], [0, 0]],
        [[3, 2], [1, 0]],
        [[4, 3], [I, 1]],
        [[5, I], [3, 2]],
        [[6, 5], [4, 3]],
        [[7, 6], [I, 4]],
        [[8, I], [6, 5]],
        [[9, 8], [7, 6]],
        [[10, 9], [I, 7]],
        [[11, I], [9, 8]],
        [[12, 11], [10, 9]],
        [[13, 12], [I, 10]],
        [[14, I], [12, 11]],
        [[15, 14], [13, 12]],
        [[16, 1], [I, 1]],
        [[17, I], [2, 2]],
        [[18, 4], [I, 16]],
        [[19, I], [5, 17]],
        [[20, 7], [I, 18]],
        [[21, I], [8, 19]],
    ]
};

/// `get_plane_residual_size(block, plane)` (§5.11.38) for a plane subsampled
/// by `(sub_x, sub_y)`; `BLOCK_INVALID` for a shape the stream may not use.
fn plane_residual_size(block: usize, sub_x: usize, sub_y: usize) -> usize {
    SUBSAMPLED_SIZE
        .get(block)
        .and_then(|row| row.get(sub_x))
        .and_then(|col| col.get(sub_y))
        .copied()
        .unwrap_or(BLOCK_INVALID)
}

/// A block size's dimensions in 4x4 units, `(0, 0)` for `BLOCK_INVALID`.
fn block_4x4_dims(block: usize) -> (usize, usize) {
    BLOCK_4X4_DIMS.get(block).copied().unwrap_or((0, 0))
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

    fn tiling(cols: u32, rows: u32, size_bytes: u32) -> TileInfo {
        TileInfo {
            cols_log2: cols.trailing_zeros(),
            rows_log2: rows.trailing_zeros(),
            cols,
            rows,
            col_starts_sb: (0..=cols).collect(),
            row_starts_sb: (0..=rows).collect(),
            context_update_tile_id: 0,
            tile_size_bytes: size_bytes,
        }
    }

    fn numbers_and_data<'a>(tiles: &[Tile<'a>]) -> Vec<(u32, &'a [u8])> {
        tiles.iter().map(|t| (t.number, t.data)).collect()
    }

    #[test]
    fn a_single_tile_group_has_no_header_and_no_sizes() {
        let group = [9, 8, 7];
        let tiles = split_tile_group(&group, &tiling(1, 1, 4)).unwrap();
        assert_eq!(numbers_and_data(&tiles), [(0, &group[..])]);
    }

    #[test]
    fn tiles_are_split_by_their_little_endian_sizes() {
        // Flag 0 (whole frame), aligned to a byte; then tile 0 with a two-byte
        // size of 2 (coded as 1), tile 1 likewise with 1, and tile 2 last.
        let group = [0x00, 0x01, 0x00, 0xA, 0xB, 0x00, 0x00, 0xC, 0xD, 0xE];
        let tiles = split_tile_group(&group, &tiling(3, 1, 2)).unwrap();
        assert_eq!(
            numbers_and_data(&tiles),
            [(0, &[0xA, 0xB][..]), (1, &[0xC][..]), (2, &[0xD, 0xE][..])]
        );
    }

    #[test]
    fn a_group_can_name_the_tiles_it_carries() {
        // 2x2 tiles: flag 1, tg_start = 2 and tg_end = 3 in two bits each
        // (1 10 11 + padding = 0b1101_1000), then one one-byte size.
        let group = [0b1101_1000, 0x00, 0xA, 0xB];
        let tiles = split_tile_group(&group, &tiling(2, 2, 1)).unwrap();
        assert_eq!(numbers_and_data(&tiles), [(2, &[0xA][..]), (3, &[0xB][..])]);
    }

    #[test]
    fn broken_tile_groups_are_malformed_not_panics() {
        let info = tiling(2, 1, 4);
        // A size larger than what follows it.
        let error = split_tile_group(&[0x00, 0xFF, 0, 0, 0, 1], &info).unwrap_err();
        assert_eq!(error.code(), otf_pixels_core::ErrorCode::Malformed);
        // Too short to hold the size field at all.
        let error = split_tile_group(&[0x00, 0x01], &info).unwrap_err();
        assert_eq!(error.code(), otf_pixels_core::ErrorCode::Malformed);
        // tg_start after tg_end (2x2: flag 1, start 3, end 0).
        let error = split_tile_group(&[0b1110_0000], &tiling(2, 2, 1)).unwrap_err();
        assert_eq!(error.code(), otf_pixels_core::ErrorCode::Malformed);
        // Empty input.
        assert!(split_tile_group(&[], &info).is_err());
    }

    #[test]
    fn neg_deinterleave_maps_each_difference_to_a_distinct_segment() {
        // For every prediction and segment count, the coded differences
        // 0..max reach every segment exactly once, nearest first.
        for max in 1..=8 {
            for reference in 0..max {
                let mut seen: Vec<i32> = (0..max)
                    .map(|diff| neg_deinterleave(diff, reference, max))
                    .collect();
                assert_eq!(seen[0], reference, "difference 0 is the prediction");
                seen.sort_unstable();
                assert_eq!(
                    seen,
                    (0..max).collect::<Vec<_>>(),
                    "max {max} ref {reference}"
                );
            }
        }
        // The alternation around a middle prediction: 3, 4, 2, 5, 1, ...
        let order: Vec<i32> = (0..8).map(|d| neg_deinterleave(d, 3, 8)).collect();
        assert_eq!(order, [3, 4, 2, 5, 1, 6, 0, 7]);
    }

    #[test]
    fn block_size_indices_match_the_spec_ordering() {
        assert_eq!(block_size_index(1, 1), 0);
        assert_eq!(block_size_index(2, 2), 3);
        assert_eq!(block_size_index(16, 16), 12);
        assert_eq!(block_size_index(16, 4), 21);
    }

    #[test]
    fn at_least_block_8x8_follows_the_enum_order_not_the_dimensions() {
        // Every block shape (in 4x4 units, up to 64x64): the gate must agree
        // with `MiSize >= BLOCK_8X8` on the spec's enum. 4x16 and 16x4 sort
        // after 8x8 though one side is 4 — missing that desynchronised
        // `angle_delta` and `palette_mode_info` reads on those shapes.
        let shapes = [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 4),
            (4, 2),
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 16),
            (16, 8),
            (16, 16),
            (1, 4),
            (4, 1),
            (2, 8),
            (8, 2),
            (4, 16),
            (16, 4),
        ];
        for (bw4, bh4) in shapes {
            assert_eq!(
                at_least_block_8x8(bw4, bh4),
                block_size_index(bw4, bh4) >= 3,
                "{bw4}x{bh4}"
            );
        }
        assert!(at_least_block_8x8(1, 4));
        assert!(at_least_block_8x8(4, 1));
        assert!(!at_least_block_8x8(1, 2));
    }

    #[test]
    fn subsampled_sizes_halve_each_subsampled_axis() {
        // Every valid Subsampled_Size entry is the luma block with each
        // subsampled axis halved (never below one 4x4 unit); 4:4:4 is the
        // identity. Catches a mistranscribed table cell.
        for block in 0..22 {
            let (bw4, bh4) = block_4x4_dims(block);
            assert_eq!(block_size_index(bw4, bh4), block);
            assert_eq!(plane_residual_size(block, 0, 0), block);
            for (sub_x, sub_y) in [(1, 0), (1, 1)] {
                let sub = plane_residual_size(block, sub_x, sub_y);
                if sub == BLOCK_INVALID {
                    continue;
                }
                assert_eq!(
                    block_4x4_dims(sub),
                    ((bw4 >> sub_x).max(1), (bh4 >> sub_y).max(1)),
                    "block {block} at {sub_x}x{sub_y}"
                );
            }
        }
        // 4:2:0 always has a chroma block; 4:2:2 has none for the tall shapes
        // whose halved width would change their aspect class.
        assert!((0..22).all(|b| plane_residual_size(b, 1, 1) != BLOCK_INVALID));
        assert_eq!(plane_residual_size(1, 1, 0), BLOCK_INVALID);
    }

    #[test]
    fn floor_log2_of_block_units() {
        assert_eq!(floor_log2_usize(1), 0);
        assert_eq!(floor_log2_usize(2), 1);
        assert_eq!(floor_log2_usize(16), 4);
    }
}
