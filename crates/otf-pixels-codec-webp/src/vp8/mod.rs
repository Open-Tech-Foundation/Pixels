//! VP8 key-frame decoding (RFC 6386), the lossy half of WebP.
//!
//! A WebP lossy image is one VP8 key frame. Decoding it is: the frame header
//! and per-macroblock modes from the first partition; DCT tokens from one or
//! more token partitions; intra prediction plus inverse transforms to
//! reconstruct each macroblock; then the loop filter over the whole frame.
//! Intra prediction reads the reconstruction *before* loop filtering, so the
//! filter runs once every macroblock is in place.
//!
//! The output is the frame's Y, U and V planes, padded to whole macroblocks.
//! Where the reference decoder and libwebp read the specification
//! differently, this follows libwebp, the decoder every WebP is checked
//! against here.

#![allow(
    clippy::indexing_slicing,
    clippy::needless_range_loop,
    reason = "block loops index the coefficient, context and edge arrays in \
              parallel by the spec's block number; macroblock arithmetic over planes allocated to whole macroblocks, \
              fixed-size coefficient and context arrays indexed by the spec's \
              own small constants; stream-derived values are checked or masked \
              before they index anything"
)]

mod bool_decoder;
mod filter;
mod predict;
mod transform;

#[allow(missing_docs, reason = "generated tables carry their own doc lines")]
mod tables {
    include!("tables.rs");
}

use bool_decoder::BoolDecoder;
use filter::Strength;
use otf_pixels_core::{PixelsError, Result};
use predict::{B_PRED, DC_PRED, Edges, H_PRED, SubEdges, TM_PRED, V_PRED, b};
use tables::{AC_Q_LOOKUP, COEFF_UPDATE_PROBS, DC_Q_LOOKUP, DEFAULT_COEFF_PROBS, KF_B_MODE_PROBS};

fn malformed(detail: impl Into<String>) -> PixelsError {
    PixelsError::malformed("webp", detail.into())
}

/// A decoded frame: planes padded to whole macroblocks.
pub struct Frame {
    /// Visible width.
    pub width: usize,
    /// Visible height.
    pub height: usize,
    /// Luma, `y_stride` wide.
    pub y: Vec<u8>,
    /// Blue-difference chroma, `uv_stride` wide.
    pub u: Vec<u8>,
    /// Red-difference chroma.
    pub v: Vec<u8>,
    /// Luma row length: 16 per macroblock column.
    pub y_stride: usize,
    /// Chroma row length: 8 per macroblock column.
    pub uv_stride: usize,
}

/// `kf_y_mode_tree` (§11.2).
const KF_Y_MODE_TREE: [i8; 8] = [
    -(B_PRED as i8),
    2,
    4,
    6,
    -(DC_PRED as i8),
    -(V_PRED as i8),
    -(H_PRED as i8),
    -(TM_PRED as i8),
];
/// `uv_mode_tree` (§11.4).
const UV_MODE_TREE: [i8; 6] = [
    -(DC_PRED as i8),
    2,
    -(V_PRED as i8),
    4,
    -(H_PRED as i8),
    -(TM_PRED as i8),
];
/// `b_mode_tree` (§11.3).
const B_MODE_TREE: [i8; 18] = [
    -(b::DC as i8),
    2,
    -(b::TM as i8),
    4,
    -(b::VE as i8),
    6,
    8,
    12,
    -(b::HE as i8),
    10,
    -(b::RD as i8),
    -(b::VR as i8),
    -(b::LD as i8),
    14,
    -(b::VL as i8),
    16,
    -(b::HD as i8),
    -(b::HU as i8),
];
const KF_Y_MODE_PROBS: [u8; 4] = tables::KF_Y_MODE_PROBS;
const KF_UV_MODE_PROBS: [u8; 3] = tables::KF_UV_MODE_PROBS;

/// Coefficient positions in scan order (§13).
const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];
/// The probability band of each scan position.
const BANDS: [usize; 17] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0];
/// `DCT_VAL_CATEGORY1..6`: base value and extra-bit probabilities (§13.2).
const CATEGORIES: [(i32, &[u8]); 6] = [
    (5, &[159]),
    (7, &[165, 145]),
    (11, &[173, 148, 140]),
    (19, &[176, 155, 140, 135]),
    (35, &[180, 157, 141, 134, 130]),
    (67, &[254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129]),
];

/// Token probabilities: `[block type][band][context][node]`.
type CoeffProbs = [[[[u8; 11]; 3]; 8]; 4];

/// Dequantization factors for one segment: `[block kind][dc, ac]`, block
/// kinds Y (after Y2), Y2, and chroma.
#[derive(Clone, Copy, Default)]
struct Dequant {
    y: [i32; 2],
    y2: [i32; 2],
    uv: [i32; 2],
}

/// What the frame header says.
struct Header {
    segmentation: bool,
    update_map: bool,
    absolute_segments: bool,
    segment_q: [i32; 4],
    segment_lf: [i32; 4],
    segment_probs: [u8; 3],
    simple_filter: bool,
    filter_level: i32,
    sharpness: i32,
    lf_deltas: Option<(i32, i32)>,
    dequant: [Dequant; 4],
    coeff_probs: Box<CoeffProbs>,
    skip_prob: Option<u8>,
}

/// One macroblock's modes and filter facts.
#[derive(Clone, Copy)]
struct MacroblockInfo {
    y_mode: u8,
    segment: u8,
    /// Whether the inner edges are filtered: subblock prediction, or any
    /// non-zero coefficient.
    filter_inner: bool,
}

/// Decode a VP8 key frame.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for anything that is not a well-formed
/// key frame, including a stream cut short.
pub fn decode(data: &[u8]) -> Result<Frame> {
    let tag = data
        .get(..10)
        .ok_or_else(|| malformed("the VP8 frame header is cut short"))?;
    if tag[0] & 1 != 0 {
        return Err(malformed("the VP8 frame is not a key frame"));
    }
    let first_size =
        ((u32::from(tag[0]) | (u32::from(tag[1]) << 8) | (u32::from(tag[2]) << 16)) >> 5) as usize;
    if tag[3..6] != [0x9d, 0x01, 0x2a] {
        return Err(malformed("the VP8 frame lacks its start code"));
    }
    let width = usize::from(u16::from_le_bytes([tag[6], tag[7]]) & 0x3fff);
    let height = usize::from(u16::from_le_bytes([tag[8], tag[9]]) & 0x3fff);
    if width == 0 || height == 0 {
        return Err(malformed("the VP8 frame has no area"));
    }
    let rest = &data[10..];
    let first = rest
        .get(..first_size)
        .ok_or_else(|| malformed("the first VP8 partition runs past the frame"))?;
    let mut bits = BoolDecoder::new(first);
    // color_space and clamping_type: neither changes how a key frame
    // decodes, and libwebp ignores both.
    bits.literal(2);
    let header = parse_header(&mut bits)?;

    let partition_count = 1_usize << bits.literal(2);
    let after = &rest[first_size..];
    let sizes_len = 3 * (partition_count - 1);
    let sizes = after
        .get(..sizes_len)
        .ok_or_else(|| malformed("the token partition sizes are cut short"))?;
    let mut partitions = Vec::with_capacity(partition_count);
    let mut tokens = &after[sizes_len..];
    for i in 0..partition_count {
        let size = if i + 1 < partition_count {
            usize::from(sizes[3 * i])
                | (usize::from(sizes[3 * i + 1]) << 8)
                | (usize::from(sizes[3 * i + 2]) << 16)
        } else {
            tokens.len()
        };
        let part = tokens
            .get(..size)
            .ok_or_else(|| malformed("a token partition runs past the frame"))?;
        tokens = &tokens[size..];
        partitions.push(BoolDecoder::new(part));
    }

    let header = finish_header(&mut bits, header)?;
    let mut decoder = FrameDecoder::new(width, height, header);
    decoder.decode(&mut bits, &mut partitions)?;
    if bits.exhausted() {
        return Err(malformed("the first VP8 partition is cut short"));
    }
    decoder.filter();
    Ok(decoder.into_frame())
}

/// Segmentation and loop-filter headers (§9.3, §9.4).
fn parse_header(bits: &mut BoolDecoder<'_>) -> Result<Header> {
    let segmentation = bits.read(128);
    let (mut update_map, mut absolute_segments) = (false, false);
    let (mut segment_q, mut segment_lf, mut segment_probs) = ([0; 4], [0; 4], [255; 3]);
    if segmentation {
        update_map = bits.read(128);
        let update_data = bits.read(128);
        if update_data {
            absolute_segments = bits.read(128);
            for q in &mut segment_q {
                *q = bits.maybe_signed(7);
            }
            for lf in &mut segment_lf {
                *lf = bits.maybe_signed(6);
            }
        }
        if update_map {
            for p in &mut segment_probs {
                *p = if bits.read(128) {
                    bits.literal(8) as u8
                } else {
                    255
                };
            }
        }
    }
    let simple_filter = bits.read(128);
    let filter_level = bits.literal(6) as i32;
    let sharpness = bits.literal(3) as i32;
    let mut lf_deltas = None;
    if bits.read(128) {
        let (mut ref_delta, mut mode_delta) = ([0; 4], [0; 4]);
        if bits.read(128) {
            for d in &mut ref_delta {
                *d = bits.maybe_signed(6);
            }
            for d in &mut mode_delta {
                *d = bits.maybe_signed(6);
            }
        }
        // A key frame only uses the intra reference delta and the B_PRED
        // mode delta.
        lf_deltas = Some((ref_delta[0], mode_delta[0]));
    }
    Ok(Header {
        segmentation,
        update_map,
        absolute_segments,
        segment_q,
        segment_lf,
        segment_probs,
        simple_filter,
        filter_level,
        sharpness,
        lf_deltas,
        dequant: [Dequant::default(); 4],
        coeff_probs: Box::new(DEFAULT_COEFF_PROBS),
        skip_prob: None,
    })
}

/// Quantizer, refresh flag and probability updates (§9.6–§9.11), which come
/// after the partition count.
fn finish_header(bits: &mut BoolDecoder<'_>, mut header: Header) -> Result<Header> {
    let base_q = bits.literal(7) as i32;
    let deltas: Vec<i32> = (0..5).map(|_| bits.maybe_signed(4)).collect();
    let (y_dc, y2_dc, y2_ac, uv_dc, uv_ac) =
        (deltas[0], deltas[1], deltas[2], deltas[3], deltas[4]);
    let dc = |q: i32| i32::from(DC_Q_LOOKUP[q.clamp(0, 127) as usize]);
    let ac = |q: i32| i32::from(AC_Q_LOOKUP[q.clamp(0, 127) as usize]);
    for (segment, factors) in header.dequant.iter_mut().enumerate() {
        let q = if !header.segmentation {
            base_q
        } else if header.absolute_segments {
            header.segment_q[segment]
        } else {
            base_q + header.segment_q[segment]
        };
        *factors = Dequant {
            y: [dc(q + y_dc), ac(q)],
            y2: [dc(q + y2_dc) * 2, (ac(q + y2_ac) * 155 / 100).max(8)],
            uv: [dc(q + uv_dc).min(132), ac(q + uv_ac)],
        };
    }
    // refresh_entropy_probs: meaningless for a single frame.
    bits.read(128);
    for (t, bands) in header.coeff_probs.iter_mut().enumerate() {
        for (band, contexts) in bands.iter_mut().enumerate() {
            for (ctx, nodes) in contexts.iter_mut().enumerate() {
                for (node, p) in nodes.iter_mut().enumerate() {
                    if bits.read(COEFF_UPDATE_PROBS[t][band][ctx][node]) {
                        *p = bits.literal(8) as u8;
                    }
                }
            }
        }
    }
    header.skip_prob = bits.read(128).then(|| bits.literal(8) as u8);
    if bits.exhausted() {
        return Err(malformed("the VP8 frame header is cut short"));
    }
    Ok(header)
}

/// Per-macroblock decode state.
struct FrameDecoder {
    header: Header,
    width: usize,
    height: usize,
    mb_cols: usize,
    mb_rows: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    info: Vec<MacroblockInfo>,
    /// Subblock modes along the bottom of each macroblock column, for the
    /// next row's contexts.
    above_modes: Vec<[u8; 4]>,
    /// Non-zero flags above each macroblock column: 4 Y, 2 U, 2 V, Y2.
    above_nz: Vec<[bool; 9]>,
}

impl FrameDecoder {
    fn new(width: usize, height: usize, header: Header) -> Self {
        let mb_cols = width.div_ceil(16);
        let mb_rows = height.div_ceil(16);
        Self {
            header,
            width,
            height,
            mb_cols,
            mb_rows,
            y: vec![0; mb_cols * 16 * mb_rows * 16],
            u: vec![0; mb_cols * 8 * mb_rows * 8],
            v: vec![0; mb_cols * 8 * mb_rows * 8],
            info: Vec::with_capacity(mb_cols * mb_rows),
            above_modes: vec![[b::DC; 4]; mb_cols],
            above_nz: vec![[false; 9]; mb_cols],
        }
    }

    fn into_frame(self) -> Frame {
        Frame {
            width: self.width,
            height: self.height,
            y: self.y,
            u: self.u,
            v: self.v,
            y_stride: self.mb_cols * 16,
            uv_stride: self.mb_cols * 8,
        }
    }

    fn decode(
        &mut self,
        bits: &mut BoolDecoder<'_>,
        partitions: &mut [BoolDecoder<'_>],
    ) -> Result<()> {
        for my in 0..self.mb_rows {
            let mut left_modes = [b::DC; 4];
            let mut left_nz = [false; 9];
            let partition = &mut partitions[my % partitions.len()];
            for mx in 0..self.mb_cols {
                // Modes, from the first partition (§19.3).
                let segment = if self.header.update_map {
                    let p = self.header.segment_probs;
                    if bits.read(p[0]) {
                        2 + u8::from(bits.read(p[2]))
                    } else {
                        u8::from(bits.read(p[1]))
                    }
                } else {
                    0
                };
                let skip = self.header.skip_prob.is_some_and(|p| bits.read(p));
                let y_mode = bits.tree(&KF_Y_MODE_TREE, &KF_Y_MODE_PROBS);
                let mut modes = [0_u8; 16];
                if y_mode == B_PRED {
                    for i in 0..16 {
                        let above = if i < 4 {
                            self.above_modes[mx][i]
                        } else {
                            modes[i - 4]
                        };
                        let left = if i & 3 == 0 {
                            left_modes[i >> 2]
                        } else {
                            modes[i - 1]
                        };
                        modes[i] = bits.tree(
                            &B_MODE_TREE,
                            &KF_B_MODE_PROBS[usize::from(above)][usize::from(left)],
                        );
                    }
                } else {
                    modes = [match y_mode {
                        V_PRED => b::VE,
                        H_PRED => b::HE,
                        TM_PRED => b::TM,
                        _ => b::DC,
                    }; 16];
                }
                self.above_modes[mx] = [modes[12], modes[13], modes[14], modes[15]];
                left_modes = [modes[3], modes[7], modes[11], modes[15]];
                let uv_mode = bits.tree(&UV_MODE_TREE, &KF_UV_MODE_PROBS);

                // Tokens, from this row's partition (§13).
                let mut coeffs = [[0_i16; 16]; 25];
                let has_y2 = y_mode != B_PRED;
                if skip {
                    left_nz[..8].fill(false);
                    self.above_nz[mx][..8].fill(false);
                    if has_y2 {
                        left_nz[8] = false;
                        self.above_nz[mx][8] = false;
                    }
                } else {
                    let dq = self.header.dequant[usize::from(segment) & 3];
                    self.read_tokens(partition, mx, &mut left_nz, has_y2, &dq, &mut coeffs);
                    if partition.exhausted() {
                        return Err(malformed("a VP8 token partition is cut short"));
                    }
                }
                if has_y2 {
                    let dcs = transform::inverse_wht(&coeffs[24]);
                    for (block, dc) in coeffs.iter_mut().zip(dcs) {
                        block[0] = dc;
                    }
                }
                // libwebp's test for filtering inner edges: any non-zero
                // coefficient once the second-order DCs are in place.
                let any_nonzero = !skip
                    && coeffs[..24]
                        .iter()
                        .any(|block| block.iter().any(|&c| c != 0));
                self.reconstruct(mx, my, y_mode, &modes, uv_mode, &coeffs);
                self.info.push(MacroblockInfo {
                    y_mode,
                    segment,
                    filter_inner: y_mode == B_PRED || any_nonzero,
                });
            }
        }
        Ok(())
    }

    /// Read every block's tokens for one macroblock, dequantized, into
    /// `coeffs` (Y 0..16, U 16..20, V 20..24, Y2 24), updating the non-zero
    /// contexts.
    fn read_tokens(
        &mut self,
        bits: &mut BoolDecoder<'_>,
        mx: usize,
        left_nz: &mut [bool; 9],
        has_y2: bool,
        dq: &Dequant,
        coeffs: &mut [[i16; 16]; 25],
    ) {
        let probs = &self.header.coeff_probs;
        let above_nz = &mut self.above_nz[mx];
        let block = |bits: &mut BoolDecoder<'_>,
                     kind: usize,
                     first: usize,
                     ctx: usize,
                     factors: [i32; 2],
                     out: &mut [i16; 16]|
         -> bool {
            let end = read_block(bits, &probs[kind], first, ctx, factors, out);
            end > first
        };
        let first_y = if has_y2 {
            let ctx = usize::from(left_nz[8]) + usize::from(above_nz[8]);
            let nz = block(bits, 1, 0, ctx, dq.y2, &mut coeffs[24]);
            left_nz[8] = nz;
            above_nz[8] = nz;
            1
        } else {
            0
        };
        let y_kind = if has_y2 { 0 } else { 3 };
        for i in 0..16 {
            let (row, col) = (i >> 2, i & 3);
            let ctx = usize::from(left_nz[row]) + usize::from(above_nz[col]);
            let nz = block(bits, y_kind, first_y, ctx, dq.y, &mut coeffs[i]);
            left_nz[row] = nz;
            above_nz[col] = nz;
        }
        for i in 16..24 {
            // U is contexts 4 and 5, V 6 and 7, each 2x2.
            let base = if i < 20 { 4 } else { 6 };
            let k = i & 3;
            let (l, a) = (base + (k >> 1), base + (k & 1));
            let ctx = usize::from(left_nz[l]) + usize::from(above_nz[a]);
            let nz = block(bits, 2, 0, ctx, dq.uv, &mut coeffs[i]);
            left_nz[l] = nz;
            above_nz[a] = nz;
        }
    }

    /// Predict and add residuals for one macroblock (§12, §14).
    fn reconstruct(
        &mut self,
        mx: usize,
        my: usize,
        y_mode: u8,
        modes: &[u8; 16],
        uv_mode: u8,
        coeffs: &[[i16; 16]; 25],
    ) {
        let stride = self.mb_cols * 16;
        let (x0, y0) = (mx * 16, my * 16);
        if y_mode == B_PRED {
            for i in 0..16 {
                let (bx, by) = (i & 3, i >> 2);
                let edges = self.sub_edges(mx, my, bx, by);
                let mut block = predict::predict_subblock(modes[i], &edges);
                transform::idct_add(&coeffs[i], &mut block);
                for (r, row) in block.iter().enumerate() {
                    let at = (y0 + by * 4 + r) * stride + x0 + bx * 4;
                    self.y[at..at + 4].copy_from_slice(row);
                }
            }
        } else {
            let edges = block_edges::<16>(&self.y, stride, mx, my);
            let mut pred = predict::predict_block(y_mode, &edges);
            for i in 0..16 {
                let (bx, by) = (i & 3, i >> 2);
                let mut block = [[0_u8; 4]; 4];
                for (r, row) in block.iter_mut().enumerate() {
                    row.copy_from_slice(&pred[by * 4 + r][bx * 4..bx * 4 + 4]);
                }
                transform::idct_add(&coeffs[i], &mut block);
                for (r, row) in block.iter().enumerate() {
                    pred[by * 4 + r][bx * 4..bx * 4 + 4].copy_from_slice(row);
                }
            }
            for (r, row) in pred.iter().enumerate() {
                let at = (y0 + r) * stride + x0;
                self.y[at..at + 16].copy_from_slice(row);
            }
        }
        let uv_stride = self.mb_cols * 8;
        for (plane, first) in [(&mut self.u, 16), (&mut self.v, 20)] {
            let edges = block_edges::<8>(plane, uv_stride, mx, my);
            let mut pred = predict::predict_block(uv_mode, &edges);
            for k in 0..4 {
                let (bx, by) = (k & 1, k >> 1);
                let mut block = [[0_u8; 4]; 4];
                for (r, row) in block.iter_mut().enumerate() {
                    row.copy_from_slice(&pred[by * 4 + r][bx * 4..bx * 4 + 4]);
                }
                transform::idct_add(&coeffs[first + k], &mut block);
                for (r, row) in block.iter().enumerate() {
                    pred[by * 4 + r][bx * 4..bx * 4 + 4].copy_from_slice(row);
                }
            }
            for (r, row) in pred.iter().enumerate() {
                let at = (my * 8 + r) * uv_stride + mx * 8;
                plane[at..at + 8].copy_from_slice(row);
            }
        }
    }

    /// Edges for luma subblock `(bx, by)` of macroblock `(mx, my)`, with the
    /// frame-border rules of the reference decoder: 127 above the frame, 129
    /// left of it, the corner 127 on the first row and 129 down the left; and
    /// above-right samples taken from the macroblock above-right — repeated
    /// from the last sample of the row past the last column — for the whole
    /// right-hand column of subblocks, not just the top one.
    fn sub_edges(&self, mx: usize, my: usize, bx: usize, by: usize) -> SubEdges {
        let stride = self.mb_cols * 16;
        let (x0, y0) = (mx * 16 + bx * 4, my * 16 + by * 4);
        let px = |x: usize, y: usize| self.y[y * stride + x];
        let mut above = [127_u8; 8];
        if y0 > 0 {
            for (k, a) in above.iter_mut().take(4).enumerate() {
                *a = px(x0 + k, y0 - 1);
            }
        }
        if bx < 3 {
            if y0 > 0 {
                for k in 4..8 {
                    above[k] = px(x0 + k, y0 - 1);
                }
            }
        } else if my > 0 {
            let row = my * 16 - 1;
            for k in 4..8 {
                above[k] = if mx + 1 < self.mb_cols {
                    px(x0 + k, row)
                } else {
                    px(stride - 1, row)
                };
            }
        }
        let mut left = [129_u8; 4];
        if x0 > 0 {
            for (k, l) in left.iter_mut().enumerate() {
                *l = px(x0 - 1, y0 + k);
            }
        }
        let corner = if y0 == 0 {
            127
        } else if x0 == 0 {
            129
        } else {
            px(x0 - 1, y0 - 1)
        };
        SubEdges {
            above,
            left,
            corner,
        }
    }

    /// The loop filter over the reconstructed frame (§15), in macroblock
    /// raster order: left edge, inner vertical edges, top edge, inner
    /// horizontal edges.
    fn filter(&mut self) {
        let h = &self.header;
        if h.filter_level == 0 {
            return;
        }
        let y_stride = self.mb_cols * 16;
        let uv_stride = self.mb_cols * 8;
        for my in 0..self.mb_rows {
            for mx in 0..self.mb_cols {
                let info = self.info[my * self.mb_cols + mx];
                let Some(f) = strength(h, info) else { continue };
                if h.simple_filter {
                    let mb_limit = 2 * (f.level + 2) + f.interior;
                    let b_limit = 2 * f.level + f.interior;
                    let origin = my * 16 * y_stride + mx * 16;
                    if mx > 0 {
                        filter::edge_simple(&mut self.y, origin, 1, y_stride, mb_limit);
                    }
                    if info.filter_inner {
                        for k in [4, 8, 12] {
                            filter::edge_simple(&mut self.y, origin + k, 1, y_stride, b_limit);
                        }
                    }
                    if my > 0 {
                        filter::edge_simple(&mut self.y, origin, y_stride, 1, mb_limit);
                    }
                    if info.filter_inner {
                        for k in [4, 8, 12] {
                            filter::edge_simple(
                                &mut self.y,
                                origin + k * y_stride,
                                y_stride,
                                1,
                                b_limit,
                            );
                        }
                    }
                    continue;
                }
                let planes: [(&mut Vec<u8>, usize, usize); 3] = [
                    (&mut self.y, y_stride, 16),
                    (&mut self.u, uv_stride, 8),
                    (&mut self.v, uv_stride, 8),
                ];
                for (plane, stride, size) in planes {
                    let origin = my * size * stride + mx * size;
                    if mx > 0 {
                        filter::mb_edge_normal(plane, origin, 1, stride, size, f);
                    }
                    if info.filter_inner {
                        for k in (4..size).step_by(4) {
                            filter::subblock_edge_normal(plane, origin + k, 1, stride, size, f);
                        }
                    }
                    if my > 0 {
                        filter::mb_edge_normal(plane, origin, stride, 1, size, f);
                    }
                    if info.filter_inner {
                        for k in (4..size).step_by(4) {
                            filter::subblock_edge_normal(
                                plane,
                                origin + k * stride,
                                stride,
                                1,
                                size,
                                f,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Edges for an `N`x`N` macroblock-level prediction of macroblock
/// `(mx, my)` in a plane of `N`-sample macroblocks.
fn block_edges<const N: usize>(plane: &[u8], stride: usize, mx: usize, my: usize) -> Edges<N> {
    let (x0, y0) = (mx * N, my * N);
    let mut above = [127_u8; N];
    if my > 0 {
        above.copy_from_slice(&plane[(y0 - 1) * stride + x0..][..N]);
    }
    let mut left = [129_u8; N];
    if mx > 0 {
        for (k, l) in left.iter_mut().enumerate() {
            *l = plane[(y0 + k) * stride + x0 - 1];
        }
    }
    let corner = if my == 0 {
        127
    } else if mx == 0 {
        129
    } else {
        plane[(y0 - 1) * stride + x0 - 1]
    };
    Edges {
        above,
        left,
        corner,
        have_above: my > 0,
        have_left: mx > 0,
    }
}

/// One macroblock's filter strength, or `None` where its level is zero
/// (libwebp's `PrecomputeFilterStrengths`).
fn strength(h: &Header, info: MacroblockInfo) -> Option<Strength> {
    let mut level = h.filter_level;
    if h.segmentation {
        let segment = h.segment_lf[usize::from(info.segment) & 3];
        level = if h.absolute_segments {
            segment
        } else {
            level + segment
        };
    }
    if let Some((ref_delta, mode_delta)) = h.lf_deltas {
        level += ref_delta;
        if info.y_mode == B_PRED {
            level += mode_delta;
        }
    }
    let level = level.clamp(0, 63);
    if level == 0 {
        return None;
    }
    let mut interior = level;
    if h.sharpness > 0 {
        interior >>= if h.sharpness > 4 { 2 } else { 1 };
        interior = interior.min(9 - h.sharpness);
    }
    let interior = interior.max(1);
    let hev = if level >= 40 {
        2
    } else {
        i32::from(level >= 15)
    };
    Some(Strength {
        level,
        interior,
        hev,
    })
}

/// Read one block's tokens (§13.2) with the probabilities for its kind,
/// writing dequantized coefficients in raster order. Returns the scan index
/// after the last token, which is `first` for a block with none.
fn read_block(
    bits: &mut BoolDecoder<'_>,
    probs: &[[[u8; 11]; 3]; 8],
    first: usize,
    ctx: usize,
    factors: [i32; 2],
    out: &mut [i16; 16],
) -> usize {
    let mut n = first;
    let mut p = &probs[BANDS[n]][ctx];
    if !bits.read(p[0]) {
        return n; // EOB straight away.
    }
    while n < 16 {
        // A zero token: no EOB check before the next one.
        if !bits.read(p[1]) {
            n += 1;
            if n == 16 {
                return 16;
            }
            p = &probs[BANDS[n]][0];
            continue;
        }
        let (value, next_ctx) = if !bits.read(p[2]) {
            (1, 1)
        } else {
            let v = if !bits.read(p[3]) {
                if !bits.read(p[4]) {
                    2
                } else if !bits.read(p[5]) {
                    3
                } else {
                    4
                }
            } else {
                let category = if !bits.read(p[6]) {
                    usize::from(bits.read(p[7]))
                } else if !bits.read(p[8]) {
                    2 + usize::from(bits.read(p[9]))
                } else {
                    4 + usize::from(bits.read(p[10]))
                };
                let (base, extra) = CATEGORIES[category];
                base + extra
                    .iter()
                    .fold(0, |acc, &q| (acc << 1) | i32::from(bits.read(q)))
            };
            (v, 2)
        };
        let signed = if bits.read(128) { -value } else { value };
        let factor = factors[usize::from(n > 0)];
        out[ZIGZAG[n]] = (signed * factor) as i16;
        n += 1;
        if n == 16 {
            return 16;
        }
        p = &probs[BANDS[n]][next_ctx];
        if !bits.read(p[0]) {
            return n; // EOB
        }
    }
    n
}
