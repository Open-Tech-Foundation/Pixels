//! The AV1 frame (uncompressed) header, restricted to intra frames (spec §5.9).
//!
//! An AVIF still is a `KEY_FRAME`, which is intra. That collapses the frame
//! header's branching enormously: no reference-frame selection, no motion
//! vectors, no global motion, no skip mode, no warped motion — every one of
//! those reads nothing here. What remains is the size, the quantizer,
//! segmentation, the loop-filter/CDEF/restoration parameters, the transform
//! mode, and the tile layout, all of which the reconstruction phases need.
//!
//! The parse still walks every field in order, because the syntax has no length
//! prefixes: the value of `CodedLossless` decides whether the loop filter reads
//! six bits or zero, so it must be computed here, from the quantizer, before
//! the filter parameters are reached.

use super::bits::BitReader;
use super::seq::SequenceHeader;
use otf_pixels_core::{PixelsError, Result};

const KEY_FRAME: u8 = 0;
const INTER_FRAME: u8 = 1;
const INTRA_ONLY_FRAME: u8 = 2;
const SWITCH_FRAME: u8 = 3;

const PRIMARY_REF_NONE: u8 = 7;
const NUM_REF_FRAMES: u32 = 8;
const SELECT: u8 = 2;

const SUPERRES_NUM: u32 = 8;
const SUPERRES_DENOM_MIN: u32 = 9;
const SUPERRES_DENOM_BITS: u32 = 3;

const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;
const MAX_TILE_COLS: u32 = 64;
const MAX_TILE_ROWS: u32 = 64;

const MAX_SEGMENTS: usize = 8;
const SEG_LVL_ALT_Q: usize = 0;
const SEG_LVL_MAX: usize = 8;
const MAX_LOOP_FILTER: i32 = 63;

const TOTAL_REFS_PER_FRAME: usize = 8;

const RESTORE_NONE: u8 = 0;
const RESTORATION_TILESIZE_MAX: u32 = 256;

const SEG_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, true, true, true, false, false, false];
const SEG_FEATURE_MAX: [i32; SEG_LVL_MAX] = [
    255,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    7,
    0,
    0,
];

/// `Remap_Lr_Type` (§5.9.20): coded value to restoration type.
const REMAP_LR_TYPE: [u8; 4] = [
    RESTORE_NONE,
    3, /*SWITCHABLE*/
    1, /*WIENER*/
    2, /*SGRPROJ*/
];

/// The tile layout (`tile_info`, §5.9.15).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    /// `log2(TileCols)`.
    pub cols_log2: u32,
    /// `log2(TileRows)`.
    pub rows_log2: u32,
    /// Number of tile columns.
    pub cols: u32,
    /// Number of tile rows.
    pub rows: u32,
    /// Superblock column index at which each tile starts, length `cols + 1`.
    pub col_starts_sb: Vec<u32>,
    /// Superblock row index at which each tile starts, length `rows + 1`.
    pub row_starts_sb: Vec<u32>,
    /// `context_update_tile_id` — which tile carries the CDF update.
    pub context_update_tile_id: u32,
    /// Bytes used to code each tile's size in the tile group.
    pub tile_size_bytes: u32,
}

impl TileInfo {
    /// Total number of tiles.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.cols * self.rows
    }
}

/// Dequant deltas from `quantization_params` (§5.9.12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quantization {
    /// Base quantizer index, 0..=255.
    pub base_q_idx: u8,
    /// Luma DC delta.
    pub delta_q_y_dc: i32,
    /// Chroma-U DC delta.
    pub delta_q_u_dc: i32,
    /// Chroma-U AC delta.
    pub delta_q_u_ac: i32,
    /// Chroma-V DC delta.
    pub delta_q_v_dc: i32,
    /// Chroma-V AC delta.
    pub delta_q_v_ac: i32,
    /// Whether quantizer matrices are used.
    pub using_qmatrix: bool,
    /// Luma qm index.
    pub qm_y: u8,
    /// Chroma-U qm index.
    pub qm_u: u8,
    /// Chroma-V qm index.
    pub qm_v: u8,
}

/// Segmentation parameters (`segmentation_params`, §5.9.14).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segmentation {
    /// Whether segmentation is on.
    pub enabled: bool,
    /// Per-segment, per-feature enable flags.
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// Per-segment, per-feature clipped values.
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl Segmentation {
    fn disabled() -> Self {
        Self {
            enabled: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }

    fn feature_active(&self, segment: usize, feature: usize) -> bool {
        self.enabled
            && self
                .feature_enabled
                .get(segment)
                .and_then(|f| f.get(feature))
                .copied()
                .unwrap_or(false)
    }
}

/// Loop-filter parameters (`loop_filter_params`, §5.9.11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopFilter {
    /// Filter level per component: `[y_vert, y_horz, u, v]`.
    pub level: [u8; 4],
    /// Sharpness, 0..=7.
    pub sharpness: u8,
    /// Whether ref/mode deltas are enabled.
    pub delta_enabled: bool,
    /// Per-reference-frame deltas.
    pub ref_deltas: [i32; TOTAL_REFS_PER_FRAME],
    /// Per-mode deltas.
    pub mode_deltas: [i32; 2],
}

/// CDEF parameters (`cdef_params`, §5.9.19).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cdef {
    /// `CdefDamping`.
    pub damping: u32,
    /// `cdef_bits` — log2 of the number of filter strengths.
    pub bits: u32,
    /// Luma primary strengths.
    pub y_pri_strength: Vec<u32>,
    /// Luma secondary strengths.
    pub y_sec_strength: Vec<u32>,
    /// Chroma primary strengths.
    pub uv_pri_strength: Vec<u32>,
    /// Chroma secondary strengths.
    pub uv_sec_strength: Vec<u32>,
}

/// Loop-restoration parameters (`lr_params`, §5.9.20).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopRestoration {
    /// Restoration type per plane.
    pub frame_restoration_type: [u8; 3],
    /// Restoration unit size per plane.
    pub unit_size: [u32; 3],
    /// Whether any plane uses restoration.
    pub uses_lr: bool,
}

/// Film-grain parameters (`film_grain_params`, §5.9.30), stored raw so
/// synthesis can apply them later without re-parsing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilmGrain {
    /// Whether grain is applied to this frame.
    pub apply_grain: bool,
    /// PRNG seed.
    pub grain_seed: u16,
}

/// A fully parsed intra frame header.
#[derive(Debug, Clone)]
pub struct FrameHeader {
    /// `frame_type` — `KEY_FRAME` or `INTRA_ONLY_FRAME` in this decoder.
    pub frame_type: u8,
    /// Whether the frame is shown.
    pub show_frame: bool,
    /// Whether the frame is later showable.
    pub showable_frame: bool,
    /// Whether error-resilient mode is set.
    pub error_resilient_mode: bool,
    /// Whether CDF updates are disabled for the frame.
    pub disable_cdf_update: bool,
    /// Whether screen-content tools are allowed.
    pub allow_screen_content_tools: bool,
    /// Coded frame width, after any super-resolution downscale.
    pub frame_width: u32,
    /// Coded frame height.
    pub frame_height: u32,
    /// Width before super-resolution upscale (the displayed coded width).
    pub upscaled_width: u32,
    /// Render (display) width.
    pub render_width: u32,
    /// Render (display) height.
    pub render_height: u32,
    /// `SuperresDenom`.
    pub superres_denom: u32,
    /// Frame width in 4x4 mode-info units.
    pub mi_cols: u32,
    /// Frame height in 4x4 mode-info units.
    pub mi_rows: u32,
    /// Whether intra block copy is allowed.
    pub allow_intrabc: bool,
    /// Whether the frame-end CDF update is disabled.
    pub disable_frame_end_update_cdf: bool,
    /// The tile layout.
    pub tile_info: TileInfo,
    /// Quantizer parameters.
    pub quantization: Quantization,
    /// Segmentation parameters.
    pub segmentation: Segmentation,
    /// Whether per-block delta-Q is present.
    pub delta_q_present: bool,
    /// Delta-Q resolution.
    pub delta_q_res: u32,
    /// Whether per-block delta-LF is present.
    pub delta_lf_present: bool,
    /// Delta-LF resolution.
    pub delta_lf_res: u32,
    /// Whether delta-LF is signalled per edge type.
    pub delta_lf_multi: bool,
    /// Whether every segment is coded losslessly.
    pub coded_lossless: bool,
    /// Whether the frame is lossless and unscaled.
    pub all_lossless: bool,
    /// Per-segment lossless flags.
    pub lossless: [bool; MAX_SEGMENTS],
    /// Loop-filter parameters.
    pub loop_filter: LoopFilter,
    /// CDEF parameters.
    pub cdef: Cdef,
    /// Loop-restoration parameters.
    pub loop_restoration: LoopRestoration,
    /// `TxMode`.
    pub tx_mode: TxMode,
    /// Whether the reduced transform set is used.
    pub reduced_tx_set: bool,
    /// Film-grain parameters.
    pub film_grain: FilmGrain,
}

/// The transform mode (`read_tx_mode`, §5.9.21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxMode {
    /// `ONLY_4X4` — lossless frames.
    Only4x4,
    /// `TX_MODE_LARGEST`.
    Largest,
    /// `TX_MODE_SELECT`.
    Select,
}

impl FrameHeader {
    /// Parse an intra frame header, given the governing sequence header and the
    /// OBU's temporal/spatial layer ids.
    pub fn parse(
        r: &mut BitReader<'_>,
        seq: &SequenceHeader,
        temporal_id: u8,
        spatial_id: u8,
    ) -> Result<Self> {
        let id_len = if seq.frame_id_numbers_present {
            seq.additional_frame_id_length + seq.delta_frame_id_length
        } else {
            0
        };
        let all_frames = (1_u32 << NUM_REF_FRAMES) - 1;

        let (frame_type, show_frame, showable_frame, error_resilient_mode);
        if seq.reduced_still_picture_header {
            frame_type = KEY_FRAME;
            show_frame = true;
            showable_frame = false;
            error_resilient_mode = false;
        } else {
            let show_existing_frame = r.flag()?;
            if show_existing_frame {
                return Err(PixelsError::unsupported(
                    "avif: show_existing_frame is an animation/reference feature outside the still-picture subset",
                ));
            }
            frame_type = r.f(2)? as u8;
            show_frame = r.flag()?;
            if show_frame && seq.decoder_model_info_present && !seq.equal_picture_interval {
                // temporal_point_info(): frame_presentation_time f(n).
                r.f(seq.frame_presentation_time_length)?;
            }
            showable_frame = if show_frame {
                frame_type != KEY_FRAME
            } else {
                r.flag()?
            };
            error_resilient_mode =
                if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show_frame) {
                    true
                } else {
                    r.flag()?
                };
        }

        let frame_is_intra = frame_type == KEY_FRAME || frame_type == INTRA_ONLY_FRAME;
        if !frame_is_intra {
            return Err(PixelsError::unsupported(
                "avif: inter frames are outside the still-picture subset",
            ));
        }

        let disable_cdf_update = r.flag()?;
        let allow_screen_content_tools = if seq.seq_force_screen_content_tools == SELECT {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        // force_integer_mv resolves to 1 for intra regardless; consume its bits.
        if allow_screen_content_tools && seq.seq_force_integer_mv == SELECT {
            r.f(1)?;
        }

        if seq.frame_id_numbers_present {
            r.f(id_len)?; // current_frame_id
        }

        let frame_size_override_flag = if frame_type == SWITCH_FRAME {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };

        // order_hint (OrderHintBits wide; 0 in the reduced header).
        r.f(seq.order_hint_bits)?;

        // primary_ref_frame is PRIMARY_REF_NONE for intra; no bits read.
        let _primary_ref_frame = PRIMARY_REF_NONE;

        if seq.decoder_model_info_present {
            let buffer_removal_time_present = r.flag()?;
            if buffer_removal_time_present {
                for op in &seq.operating_points {
                    // decoder_model_present_for_this_op was not retained per-op;
                    // the reduced/still path never sets it, so this loop only
                    // runs when a decoder model is genuinely present.
                    let idc = op.idc;
                    let in_temporal = (idc >> temporal_id) & 1;
                    let in_spatial = (idc >> (spatial_id + 8)) & 1;
                    if idc == 0 || (in_temporal != 0 && in_spatial != 0) {
                        r.f(seq.buffer_removal_time_length)?;
                    }
                }
            }
        }

        // refresh_frame_flags: a shown key frame refreshes all slots.
        let _refresh_frame_flags =
            if frame_type == SWITCH_FRAME || (frame_type == KEY_FRAME && show_frame) {
                all_frames
            } else {
                r.f(8)?
            };

        // Intra path: frame_size(), render_size(), then maybe allow_intrabc.
        let size = parse_frame_size(r, seq, frame_size_override_flag)?;
        let (render_width, render_height) =
            parse_render_size(r, size.upscaled_width, size.frame_height)?;

        let allow_intrabc = if allow_screen_content_tools && size.upscaled_width == size.frame_width
        {
            r.flag()?
        } else {
            false
        };

        let disable_frame_end_update_cdf = if seq.reduced_still_picture_header || disable_cdf_update
        {
            true
        } else {
            r.flag()?
        };

        let tile_info = parse_tile_info(r, seq, size.mi_cols, size.mi_rows)?;
        let quantization = parse_quantization(r, seq)?;
        let segmentation = parse_segmentation(r)?;

        // delta_q_params / delta_lf_params.
        let mut delta_q_present = false;
        let mut delta_q_res = 0;
        if quantization.base_q_idx > 0 {
            delta_q_present = r.flag()?;
        }
        if delta_q_present {
            delta_q_res = r.f(2)?;
        }
        let mut delta_lf_present = false;
        let mut delta_lf_res = 0;
        let mut delta_lf_multi = false;
        if delta_q_present {
            if !allow_intrabc {
                delta_lf_present = r.flag()?;
            }
            if delta_lf_present {
                delta_lf_res = r.f(2)?;
                delta_lf_multi = r.flag()?;
            }
        }

        // CodedLossless / AllLossless from the quantizer and segmentation.
        let mut lossless = [false; MAX_SEGMENTS];
        let mut coded_lossless = true;
        let seg_count = if segmentation.enabled {
            MAX_SEGMENTS
        } else {
            1
        };
        for (segment, slot) in lossless.iter_mut().enumerate().take(seg_count) {
            let qindex = get_qindex(&segmentation, &quantization, segment);
            let is_lossless = qindex == 0
                && quantization.delta_q_y_dc == 0
                && quantization.delta_q_u_ac == 0
                && quantization.delta_q_u_dc == 0
                && quantization.delta_q_v_ac == 0
                && quantization.delta_q_v_dc == 0;
            *slot = is_lossless;
            if !is_lossless {
                coded_lossless = false;
            }
        }
        // Segments beyond seg_count inherit segment 0's lossless flag when
        // segmentation is off; only the active range gates CodedLossless.
        let all_lossless = coded_lossless && size.frame_width == size.upscaled_width;

        let loop_filter = parse_loop_filter(r, seq, coded_lossless, allow_intrabc)?;
        let cdef = parse_cdef(r, seq, coded_lossless, allow_intrabc)?;
        let loop_restoration = parse_lr(r, seq, all_lossless, allow_intrabc)?;

        let tx_mode = if coded_lossless {
            TxMode::Only4x4
        } else if r.flag()? {
            TxMode::Select
        } else {
            TxMode::Largest
        };

        // frame_reference_mode: intra reads nothing (reference_select = 0).
        // skip_mode_params: intra reads nothing.
        // allow_warped_motion: intra reads nothing.
        let reduced_tx_set = r.flag()?;
        // global_motion_params: intra reads nothing.
        let film_grain = parse_film_grain(r, seq, frame_type, show_frame, showable_frame)?;

        Ok(Self {
            frame_type,
            show_frame,
            showable_frame,
            error_resilient_mode,
            disable_cdf_update,
            allow_screen_content_tools,
            frame_width: size.frame_width,
            frame_height: size.frame_height,
            upscaled_width: size.upscaled_width,
            render_width,
            render_height,
            superres_denom: size.superres_denom,
            mi_cols: size.mi_cols,
            mi_rows: size.mi_rows,
            allow_intrabc,
            disable_frame_end_update_cdf,
            tile_info,
            quantization,
            segmentation,
            delta_q_present,
            delta_q_res,
            delta_lf_present,
            delta_lf_res,
            delta_lf_multi,
            coded_lossless,
            all_lossless,
            lossless,
            loop_filter,
            cdef,
            loop_restoration,
            tx_mode,
            reduced_tx_set,
            film_grain,
        })
    }
}

struct FrameSize {
    frame_width: u32,
    frame_height: u32,
    upscaled_width: u32,
    superres_denom: u32,
    mi_cols: u32,
    mi_rows: u32,
}

fn parse_frame_size(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    override_flag: bool,
) -> Result<FrameSize> {
    let (mut frame_width, frame_height) = if override_flag {
        let w = r.f(seq.frame_width_bits)? + 1;
        let h = r.f(seq.frame_height_bits)? + 1;
        (w, h)
    } else {
        (seq.max_frame_width, seq.max_frame_height)
    };

    // superres_params.
    let use_superres = if seq.enable_superres {
        r.flag()?
    } else {
        false
    };
    let superres_denom = if use_superres {
        r.f(SUPERRES_DENOM_BITS)? + SUPERRES_DENOM_MIN
    } else {
        SUPERRES_NUM
    };
    let upscaled_width = frame_width;
    frame_width = (upscaled_width * SUPERRES_NUM + (superres_denom / 2)) / superres_denom;

    let mi_cols = 2 * ((frame_width + 7) >> 3);
    let mi_rows = 2 * ((frame_height + 7) >> 3);

    Ok(FrameSize {
        frame_width,
        frame_height,
        upscaled_width,
        superres_denom,
        mi_cols,
        mi_rows,
    })
}

fn parse_render_size(
    r: &mut BitReader<'_>,
    upscaled_width: u32,
    frame_height: u32,
) -> Result<(u32, u32)> {
    if r.flag()? {
        let w = r.f(16)? + 1;
        let h = r.f(16)? + 1;
        Ok((w, h))
    } else {
        Ok((upscaled_width, frame_height))
    }
}

/// `tile_log2(blkSize, target)` (§5.9.16): smallest `k` with
/// `blkSize << k >= target`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

fn parse_tile_info(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    mi_cols: u32,
    mi_rows: u32,
) -> Result<TileInfo> {
    let (sb_cols, sb_rows, sb_shift) = if seq.use_128x128_superblock {
        (((mi_cols + 31) >> 5), ((mi_rows + 31) >> 5), 5)
    } else {
        (((mi_cols + 15) >> 4), ((mi_rows + 15) >> 4), 4)
    };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let uniform_tile_spacing = r.flag()?;
    let mut col_starts_sb = Vec::new();
    let mut row_starts_sb = Vec::new();
    let cols_log2;
    let rows_log2;

    if uniform_tile_spacing {
        let mut c = min_log2_tile_cols;
        while c < max_log2_tile_cols {
            if r.flag()? {
                c += 1;
            } else {
                break;
            }
        }
        cols_log2 = c;
        let tile_width_sb = (sb_cols + (1 << cols_log2) - 1) >> cols_log2;
        let mut start = 0;
        while start < sb_cols {
            col_starts_sb.push(start);
            start += tile_width_sb;
        }
        col_starts_sb.push(sb_cols);

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(cols_log2);
        let mut rl = min_log2_tile_rows;
        while rl < max_log2_tile_rows {
            if r.flag()? {
                rl += 1;
            } else {
                break;
            }
        }
        rows_log2 = rl;
        let tile_height_sb = (sb_rows + (1 << rows_log2) - 1) >> rows_log2;
        let mut start = 0;
        while start < sb_rows {
            row_starts_sb.push(start);
            start += tile_height_sb;
        }
        row_starts_sb.push(sb_rows);
    } else {
        let mut widest_tile_sb = 0;
        let mut start = 0;
        while start < sb_cols {
            col_starts_sb.push(start);
            let max_width = (sb_cols - start).min(max_tile_width_sb);
            let size_sb = r.ns(max_width)? + 1;
            widest_tile_sb = widest_tile_sb.max(size_sb);
            start += size_sb;
        }
        col_starts_sb.push(sb_cols);
        let tile_cols = (col_starts_sb.len() as u32) - 1;
        cols_log2 = tile_log2(1, tile_cols);

        let max_tile_area_sb = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
        let mut start = 0;
        while start < sb_rows {
            row_starts_sb.push(start);
            let max_height = (sb_rows - start).min(max_tile_height_sb);
            let size_sb = r.ns(max_height)? + 1;
            start += size_sb;
        }
        row_starts_sb.push(sb_rows);
        let tile_rows = (row_starts_sb.len() as u32) - 1;
        rows_log2 = tile_log2(1, tile_rows);
    }

    let cols = (col_starts_sb.len() as u32) - 1;
    let rows = (row_starts_sb.len() as u32) - 1;

    let (context_update_tile_id, tile_size_bytes) = if cols_log2 > 0 || rows_log2 > 0 {
        let id = r.f(rows_log2 + cols_log2)?;
        let bytes = r.f(2)? + 1;
        (id, bytes)
    } else {
        (0, 1)
    };

    Ok(TileInfo {
        cols_log2,
        rows_log2,
        cols,
        rows,
        col_starts_sb,
        row_starts_sb,
        context_update_tile_id,
        tile_size_bytes,
    })
}

fn read_delta_q(r: &mut BitReader<'_>) -> Result<i32> {
    if r.flag()? { r.su(6) } else { Ok(0) }
}

fn parse_quantization(r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<Quantization> {
    let base_q_idx = r.f(8)? as u8;
    let delta_q_y_dc = read_delta_q(r)?;
    let (delta_q_u_dc, delta_q_u_ac, delta_q_v_dc, delta_q_v_ac);
    if seq.color.num_planes > 1 {
        let diff_uv_delta = if seq.color.separate_uv_delta_q {
            r.flag()?
        } else {
            false
        };
        let u_dc = read_delta_q(r)?;
        let u_ac = read_delta_q(r)?;
        if diff_uv_delta {
            delta_q_u_dc = u_dc;
            delta_q_u_ac = u_ac;
            delta_q_v_dc = read_delta_q(r)?;
            delta_q_v_ac = read_delta_q(r)?;
        } else {
            delta_q_u_dc = u_dc;
            delta_q_u_ac = u_ac;
            delta_q_v_dc = u_dc;
            delta_q_v_ac = u_ac;
        }
    } else {
        delta_q_u_dc = 0;
        delta_q_u_ac = 0;
        delta_q_v_dc = 0;
        delta_q_v_ac = 0;
    }

    let using_qmatrix = r.flag()?;
    let (qm_y, qm_u, qm_v) = if using_qmatrix {
        let y = r.f(4)? as u8;
        let u = r.f(4)? as u8;
        let v = if seq.color.separate_uv_delta_q {
            r.f(4)? as u8
        } else {
            u
        };
        (y, u, v)
    } else {
        (0, 0, 0)
    };

    Ok(Quantization {
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
        qm_y,
        qm_u,
        qm_v,
    })
}

fn parse_segmentation(r: &mut BitReader<'_>) -> Result<Segmentation> {
    let enabled = r.flag()?;
    if !enabled {
        return Ok(Segmentation::disabled());
    }
    // Intra frame: primary_ref_frame is PRIMARY_REF_NONE, so update_map and
    // update_data are both implied 1 and no flags are read for them.
    let mut seg = Segmentation::disabled();
    seg.enabled = true;
    for segment in 0..MAX_SEGMENTS {
        for feature in 0..SEG_LVL_MAX {
            let feature_enabled = r.flag()?;
            let mut clipped = 0;
            if feature_enabled {
                let bits = SEG_FEATURE_BITS.get(feature).copied().unwrap_or(0);
                let limit = SEG_FEATURE_MAX.get(feature).copied().unwrap_or(0);
                let signed = SEG_FEATURE_SIGNED.get(feature).copied().unwrap_or(false);
                if signed {
                    let value = r.su(bits)?;
                    clipped = value.clamp(-limit, limit);
                } else {
                    let value = r.f(bits)? as i32;
                    clipped = value.clamp(0, limit);
                }
            }
            if let (Some(en), Some(dat)) = (
                seg.feature_enabled.get_mut(segment),
                seg.feature_data.get_mut(segment),
            ) {
                if let (Some(e), Some(d)) = (en.get_mut(feature), dat.get_mut(feature)) {
                    *e = feature_enabled;
                    *d = clipped;
                }
            }
        }
    }
    Ok(seg)
}

fn get_qindex(seg: &Segmentation, quant: &Quantization, segment: usize) -> i32 {
    let base = i32::from(quant.base_q_idx);
    if seg.feature_active(segment, SEG_LVL_ALT_Q) {
        let data = seg
            .feature_data
            .get(segment)
            .and_then(|f| f.get(SEG_LVL_ALT_Q))
            .copied()
            .unwrap_or(0);
        (base + data).clamp(0, 255)
    } else {
        base
    }
}

fn parse_loop_filter(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopFilter> {
    // Default ref deltas per §5.9.11: intra +1, others as listed.
    let mut lf = LoopFilter {
        level: [0; 4],
        sharpness: 0,
        delta_enabled: false,
        ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
        mode_deltas: [0, 0],
    };
    if coded_lossless || allow_intrabc {
        return Ok(lf);
    }
    lf.level[0] = r.f(6)? as u8;
    lf.level[1] = r.f(6)? as u8;
    if seq.color.num_planes > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = r.f(6)? as u8;
        lf.level[3] = r.f(6)? as u8;
    }
    lf.sharpness = r.f(3)? as u8;
    lf.delta_enabled = r.flag()?;
    if lf.delta_enabled {
        let delta_update = r.flag()?;
        if delta_update {
            for slot in lf.ref_deltas.iter_mut() {
                if r.flag()? {
                    *slot = r.su(6)?;
                }
            }
            for slot in lf.mode_deltas.iter_mut() {
                if r.flag()? {
                    *slot = r.su(6)?;
                }
            }
        }
    }
    Ok(lf)
}

fn parse_cdef(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<Cdef> {
    if coded_lossless || allow_intrabc || !seq.enable_cdef {
        return Ok(Cdef {
            damping: 3,
            bits: 0,
            y_pri_strength: vec![0],
            y_sec_strength: vec![0],
            uv_pri_strength: vec![0],
            uv_sec_strength: vec![0],
        });
    }
    let damping = r.f(2)? + 3;
    let bits = r.f(2)?;
    let count = 1_usize << bits;
    let mut y_pri = Vec::with_capacity(count);
    let mut y_sec = Vec::with_capacity(count);
    let mut uv_pri = Vec::with_capacity(count);
    let mut uv_sec = Vec::with_capacity(count);
    for _ in 0..count {
        y_pri.push(r.f(4)?);
        let mut ys = r.f(2)?;
        if ys == 3 {
            ys += 1;
        }
        y_sec.push(ys);
        if seq.color.num_planes > 1 {
            uv_pri.push(r.f(4)?);
            let mut us = r.f(2)?;
            if us == 3 {
                us += 1;
            }
            uv_sec.push(us);
        } else {
            uv_pri.push(0);
            uv_sec.push(0);
        }
    }
    Ok(Cdef {
        damping,
        bits,
        y_pri_strength: y_pri,
        y_sec_strength: y_sec,
        uv_pri_strength: uv_pri,
        uv_sec_strength: uv_sec,
    })
}

fn parse_lr(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    all_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopRestoration> {
    let mut lr = LoopRestoration {
        frame_restoration_type: [RESTORE_NONE; 3],
        unit_size: [RESTORATION_TILESIZE_MAX; 3],
        uses_lr: false,
    };
    if all_lossless || allow_intrabc || !seq.enable_restoration {
        return Ok(lr);
    }
    let mut uses_lr = false;
    let mut uses_chroma_lr = false;
    for plane in 0..usize::from(seq.color.num_planes) {
        let lr_type = r.f(2)? as usize;
        let mapped = REMAP_LR_TYPE.get(lr_type).copied().unwrap_or(RESTORE_NONE);
        if let Some(slot) = lr.frame_restoration_type.get_mut(plane) {
            *slot = mapped;
        }
        if mapped != RESTORE_NONE {
            uses_lr = true;
            if plane > 0 {
                uses_chroma_lr = true;
            }
        }
    }
    lr.uses_lr = uses_lr;
    if uses_lr {
        let mut lr_unit_shift;
        if seq.use_128x128_superblock {
            lr_unit_shift = r.f(1)? + 1;
        } else {
            lr_unit_shift = r.f(1)?;
            if lr_unit_shift != 0 {
                lr_unit_shift += r.f(1)?;
            }
        }
        let size0 = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift);
        let lr_uv_shift =
            if seq.color.subsampling_x == 1 && seq.color.subsampling_y == 1 && uses_chroma_lr {
                r.f(1)?
            } else {
                0
            };
        lr.unit_size[0] = size0;
        if let Some(s) = lr.unit_size.get_mut(1) {
            *s = size0 >> lr_uv_shift;
        }
        if let Some(s) = lr.unit_size.get_mut(2) {
            *s = size0 >> lr_uv_shift;
        }
    }
    Ok(lr)
}

fn parse_film_grain(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    frame_type: u8,
    show_frame: bool,
    showable_frame: bool,
) -> Result<FilmGrain> {
    if !seq.film_grain_params_present || (!show_frame && !showable_frame) {
        return Ok(FilmGrain::default());
    }
    let apply_grain = r.flag()?;
    if !apply_grain {
        return Ok(FilmGrain::default());
    }
    let grain_seed = r.f(16)? as u16;
    let update_grain = if frame_type == INTER_FRAME {
        r.flag()?
    } else {
        true
    };
    if !update_grain {
        // film_grain_params_ref_idx f(3); the rest loads from that reference.
        r.f(3)?;
        return Ok(FilmGrain {
            apply_grain,
            grain_seed,
        });
    }

    let num_y_points = r.f(4)?;
    for _ in 0..num_y_points {
        r.f(8)?; // point_y_value
        r.f(8)?; // point_y_scaling
    }
    let chroma_scaling_from_luma = if seq.color.mono_chrome {
        false
    } else {
        r.flag()?
    };
    let (num_cb_points, num_cr_points);
    if seq.color.mono_chrome
        || chroma_scaling_from_luma
        || (seq.color.subsampling_x == 1 && seq.color.subsampling_y == 1 && num_y_points == 0)
    {
        num_cb_points = 0;
        num_cr_points = 0;
    } else {
        let cb = r.f(4)?;
        for _ in 0..cb {
            r.f(8)?;
            r.f(8)?;
        }
        let cr = r.f(4)?;
        for _ in 0..cr {
            r.f(8)?;
            r.f(8)?;
        }
        num_cb_points = cb;
        num_cr_points = cr;
    }
    r.f(2)?; // grain_scaling_minus_8
    let ar_coeff_lag = r.f(2)?;
    let num_pos_luma = 2 * ar_coeff_lag * (ar_coeff_lag + 1);
    let num_pos_chroma = if num_y_points > 0 {
        for _ in 0..num_pos_luma {
            r.f(8)?;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if chroma_scaling_from_luma || num_cb_points > 0 {
        for _ in 0..num_pos_chroma {
            r.f(8)?;
        }
    }
    if chroma_scaling_from_luma || num_cr_points > 0 {
        for _ in 0..num_pos_chroma {
            r.f(8)?;
        }
    }
    r.f(2)?; // ar_coeff_shift_minus_6
    r.f(2)?; // grain_scale_shift
    if num_cb_points > 0 {
        r.f(8)?; // cb_mult
        r.f(8)?; // cb_luma_mult
        r.f(9)?; // cb_offset
    }
    if num_cr_points > 0 {
        r.f(8)?;
        r.f(8)?;
        r.f(9)?;
    }
    r.f(1)?; // overlap_flag
    r.f(1)?; // clip_to_restricted_range

    Ok(FilmGrain {
        apply_grain,
        grain_seed,
    })
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

    /// Bit-level builder mirroring the syntax, shared shape with seq.rs tests.
    struct Bldr {
        bits: Vec<u8>,
    }
    impl Bldr {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn put(&mut self, value: u32, n: u32) -> &mut Self {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
            self
        }
        fn flag(&mut self, b: bool) -> &mut Self {
            self.put(u32::from(b), 1)
        }
        fn pack(&self) -> Vec<u8> {
            let mut out = vec![0_u8; self.bits.len().div_ceil(8)];
            for (i, &bit) in self.bits.iter().enumerate() {
                if bit != 0 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    /// A reduced-still-picture sequence header for an 8-bit 4:2:0 image, with
    /// CDEF and restoration disabled so the frame header stays compact.
    fn seq_reduced(width: u32, height: u32) -> SequenceHeader {
        let mut b = Bldr::new();
        b.put(0, 3).flag(true).flag(true).put(1, 5);
        b.put(15, 4)
            .put(15, 4)
            .put(width - 1, 16)
            .put(height - 1, 16);
        b.flag(false).flag(false).flag(false); // sb / filter-intra / edge
        b.flag(false).flag(false).flag(false); // superres / cdef / restoration
        b.flag(false).flag(false).flag(false); // hbd / mono / colordesc
        b.flag(false).put(0, 2).flag(false); // range / chroma pos / uv delta q
        b.flag(false); // film grain
        let bytes = b.pack();
        let mut r = BitReader::new(&bytes);
        SequenceHeader::parse(&mut r).unwrap()
    }

    #[test]
    fn parses_a_key_frame_header() {
        let seq = seq_reduced(64, 64);
        // Reduced header: frame_type/show/error implied. First read is
        // disable_cdf_update, then (force_screen_content_tools == SELECT) so
        // allow_screen_content_tools f(1), then force_integer_mv only if sct.
        let mut b = Bldr::new();
        b.flag(false); // disable_cdf_update
        b.flag(false); // allow_screen_content_tools = 0
        // frame_size_override = 0 (reduced), order_hint 0 bits, refresh all.
        // frame_size: enable_superres=0 so no bit. render_size:
        b.flag(false); // render_and_frame_size_different = 0
        // allow_intrabc gated off (sct=0). disable_frame_end_update_cdf implied.
        // tile_info: 64x64 -> 16x16 mi -> 1 superblock. uniform_tile_spacing:
        b.flag(true); // uniform_tile_spacing
        // min==max log2 cols/rows == 0, so no increment bits, no ids.
        // quantization: base_q_idx f(8) = 100.
        b.put(100, 8);
        b.flag(false); // delta_q_y_dc coded = 0
        b.flag(false); // diff not read (separate_uv=0) -> u_dc coded
        b.flag(false); // u_ac coded = 0
        b.flag(false); // using_qmatrix = 0
        // segmentation:
        b.flag(false); // segmentation_enabled = 0
        // delta_q: base>0 so delta_q_present f(1)
        b.flag(false); // delta_q_present = 0
        // delta_lf gated by delta_q_present=0 -> nothing.
        // loop_filter (not lossless, not intrabc): level[0] f(6), level[1] f(6)
        b.put(0, 6).put(0, 6); // both 0 -> chroma levels skipped
        b.put(0, 3); // sharpness
        b.flag(false); // delta_enabled = 0
        // cdef disabled in seq -> nothing. lr disabled -> nothing.
        // tx_mode: not lossless -> tx_mode_select f(1)
        b.flag(false); // tx_mode_select = 0 -> Largest
        // reference/skip/warp: nothing. reduced_tx_set f(1):
        b.flag(false);
        // film grain not present -> nothing.
        let bytes = b.pack();
        let mut r = BitReader::new(&bytes);
        let fh = FrameHeader::parse(&mut r, &seq, 0, 0).unwrap();
        assert_eq!(fh.frame_type, KEY_FRAME);
        assert!(fh.show_frame);
        assert_eq!(fh.frame_width, 64);
        assert_eq!(fh.frame_height, 64);
        assert_eq!(fh.mi_cols, 16);
        assert_eq!(fh.mi_rows, 16);
        assert_eq!(fh.quantization.base_q_idx, 100);
        assert!(!fh.coded_lossless);
        assert_eq!(fh.tile_info.count(), 1);
        assert_eq!(fh.tx_mode, TxMode::Largest);
    }

    #[test]
    fn a_zero_quantizer_is_coded_lossless_and_forces_only_4x4() {
        let seq = seq_reduced(32, 32);
        let mut b = Bldr::new();
        b.flag(false); // disable_cdf_update
        b.flag(false); // allow_screen_content_tools
        b.flag(false); // render size differ
        b.flag(true); // uniform tile spacing
        b.put(0, 8); // base_q_idx = 0 -> lossless
        b.flag(false); // delta_q_y_dc coded
        b.flag(false); // u_dc coded
        b.flag(false); // u_ac coded
        b.flag(false); // using_qmatrix
        b.flag(false); // segmentation_enabled
        // base_q_idx == 0 -> delta_q_present not read.
        // CodedLossless -> loop_filter reads nothing, cdef nothing, lr nothing.
        // tx_mode: CodedLossless -> Only4x4, no bit.
        b.flag(false); // reduced_tx_set
        let bytes = b.pack();
        let mut r = BitReader::new(&bytes);
        let fh = FrameHeader::parse(&mut r, &seq, 0, 0).unwrap();
        assert!(fh.coded_lossless);
        assert!(fh.all_lossless);
        assert_eq!(fh.tx_mode, TxMode::Only4x4);
        assert!(fh.lossless[0]);
    }

    #[test]
    fn tile_log2_is_the_smallest_covering_shift() {
        assert_eq!(tile_log2(1, 1), 0);
        assert_eq!(tile_log2(1, 2), 1);
        assert_eq!(tile_log2(1, 3), 2);
        assert_eq!(tile_log2(1, 4), 2);
        assert_eq!(tile_log2(4, 16), 2);
    }
}
