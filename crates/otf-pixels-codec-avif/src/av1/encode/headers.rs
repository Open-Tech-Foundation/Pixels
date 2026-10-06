//! The AV1 headers an AVIF still needs: a reduced still-picture sequence
//! header and a key-frame header, wrapped in OBUs.
//!
//! Each writer mirrors its parser in `seq.rs` / `frame.rs` field for field,
//! restricted to the subset this encoder uses: one 8-bit picture, 4:2:0 or
//! monochrome, 64x64 superblocks, no screen-content tools, no superres or
//! loop restoration, and the deblocking filter and CDEF signalled.

/// Writes bits most-significant first, as AV1 headers are packed.
#[derive(Default)]
pub struct BitWriter {
    out: Vec<u8>,
    bits: u32,
}

impl BitWriter {
    /// `f(n)`: the low `n` bits of `value`, most-significant first.
    pub fn f(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1;
            if self.bits % 8 == 0 {
                self.out.push(0);
            }
            if bit == 1 {
                if let Some(last) = self.out.last_mut() {
                    *last |= 0x80 >> (self.bits % 8);
                }
            }
            self.bits += 1;
        }
    }

    /// One flag bit.
    pub fn flag(&mut self, value: bool) {
        self.f(1, u32::from(value));
    }

    /// `trailing_bits()` (§5.3.4): a one, then zeros to a byte boundary.
    pub fn trailing_bits(&mut self) {
        self.f(1, 1);
        while self.bits % 8 != 0 {
            self.f(1, 0);
        }
    }

    /// `byte_alignment()`: zeros to a byte boundary.
    pub fn byte_alignment(&mut self) {
        while self.bits % 8 != 0 {
            self.f(1, 0);
        }
    }

    /// The bytes written.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.out
    }
}

/// `leb128()`.
pub fn leb128(mut value: usize, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// An OBU with `obu_has_size_field` set and no extension.
pub fn obu(kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![(kind << 3) | 0b10];
    leb128(payload.len(), &mut out);
    out.extend_from_slice(payload);
    out
}

/// `OBU_SEQUENCE_HEADER`.
pub const OBU_SEQUENCE_HEADER: u8 = 1;
/// `OBU_FRAME`: a frame header with its tile group.
pub const OBU_FRAME: u8 = 6;

/// What the sequence header declares.
#[derive(Debug, Clone, Copy)]
pub struct SequenceParams {
    /// Picture width in pixels.
    pub width: u32,
    /// Picture height in pixels.
    pub height: u32,
    /// One plane (an alpha image) rather than 4:2:0 colour.
    pub mono_chrome: bool,
    /// CICP colour primaries, transfer characteristics, matrix coefficients.
    pub cicp: (u8, u8, u8),
    /// Full-range samples rather than studio range.
    pub full_range: bool,
}

impl SequenceParams {
    /// `seq_level_idx`: the smallest level whose picture-size limits hold
    /// the picture, or 31 ("no level") past them all.
    #[must_use]
    pub fn level(&self) -> u8 {
        // (seq_level_idx, MaxPicSize, MaxHSize, MaxVSize) for levels 2.0 to
        // 6.0 (Annex A.3).
        const LEVELS: [(u8, u64, u32, u32); 5] = [
            (0, 147_456, 2048, 1152),
            (4, 665_856, 4096, 2176),
            (8, 2_359_296, 4096, 2176),
            (12, 8_912_896, 8192, 4352),
            (16, 35_651_584, 16384, 8704),
        ];
        let area = u64::from(self.width) * u64::from(self.height);
        LEVELS
            .iter()
            .find(|&&(_, size, w, h)| area <= size && self.width <= w && self.height <= h)
            .map_or(31, |&(idx, ..)| idx)
    }
}

fn bits_for(value: u32) -> u32 {
    32 - value.max(1).leading_zeros()
}

/// The payload of a reduced still-picture sequence header (§5.5).
#[must_use]
pub fn sequence_header(p: &SequenceParams) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.f(3, 0); // seq_profile 0
    w.flag(true); // still_picture
    w.flag(true); // reduced_still_picture_header
    w.f(5, u32::from(p.level()));
    let (wb, hb) = (bits_for(p.width - 1), bits_for(p.height - 1));
    w.f(4, wb - 1);
    w.f(4, hb - 1);
    w.f(wb, p.width - 1);
    w.f(hb, p.height - 1);
    w.flag(false); // use_128x128_superblock
    w.flag(false); // enable_filter_intra
    w.flag(true); // enable_intra_edge_filter
    w.flag(false); // enable_superres
    w.flag(true); // enable_cdef
    w.flag(false); // enable_restoration
    // color_config (§5.5.2).
    w.flag(false); // high_bitdepth
    w.flag(p.mono_chrome);
    w.flag(true); // color_description_present_flag
    w.f(8, u32::from(p.cicp.0));
    w.f(8, u32::from(p.cicp.1));
    w.f(8, u32::from(p.cicp.2));
    w.flag(p.full_range);
    if !p.mono_chrome {
        w.f(2, 0); // chroma_sample_position: unknown
        w.flag(false); // separate_uv_delta_q
    }
    w.flag(false); // film_grain_params_present
    w.trailing_bits();
    w.finish()
}

/// CDEF strengths for one `cdef_idx` value.
#[derive(Debug, Clone, Copy)]
pub struct CdefStrength {
    /// Luma primary strength, 0..=15.
    pub y_pri: u8,
    /// Luma secondary strength, 0..=3 (3 meaning 4).
    pub y_sec: u8,
    /// Chroma primary strength.
    pub uv_pri: u8,
    /// Chroma secondary strength.
    pub uv_sec: u8,
}

/// What the frame header declares.
#[derive(Debug, Clone)]
pub struct FrameParams {
    /// `base_q_idx`, 1..=255 (0 would be lossless).
    pub base_q_idx: u8,
    /// Deblocking levels: luma vertical, luma horizontal, U, V.
    pub loop_filter: [u8; 4],
    /// Deblocking sharpness, 0..=7.
    pub sharpness: u8,
    /// `cdef_damping_minus_3` and the strengths `cdef_idx` selects among
    /// (one, two, four or eight).
    pub cdef: (u8, Vec<CdefStrength>),
    /// `TX_MODE_SELECT` rather than `TX_MODE_LARGEST`.
    pub tx_mode_select: bool,
    /// `log2` of the tile columns and rows, each at least its minimum.
    pub tile_log2: (u32, u32),
}

/// Tile-layout bounds for a picture (§5.9.15): `(min_log2_tile_cols,
/// max_log2_tile_cols, min_log2_tiles, max_log2_tile_rows)` and the
/// superblock grid.
#[must_use]
pub fn tile_limits(width: u32, height: u32) -> (u32, u32, u32, u32, u32, u32) {
    let tile_log2 = |blk: u32, target: u32| {
        let mut k = 0;
        while (blk << k) < target {
            k += 1;
        }
        k
    };
    let mi_cols = 2 * width.div_ceil(8);
    let mi_rows = 2 * height.div_ceil(8);
    let sb_cols = mi_cols.div_ceil(16);
    let sb_rows = mi_rows.div_ceil(16);
    let max_tile_width_sb = 4096 >> 6;
    let max_tile_area_sb = (4096 * 2304) >> 12;
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
    (
        min_log2_tile_cols,
        max_log2_tile_cols,
        min_log2_tiles,
        max_log2_tile_rows,
        sb_cols,
        sb_rows,
    )
}

/// The uncompressed key-frame header (§5.9), byte-aligned, ready for the
/// tile group that follows it in an `OBU_FRAME`.
#[must_use]
pub fn frame_header(seq: &SequenceParams, p: &FrameParams) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.flag(false); // disable_cdf_update
    w.flag(false); // allow_screen_content_tools
    w.flag(false); // render_and_frame_size_different
    // tile_info: uniform spacing at the requested (or minimum) layout.
    let (min_cols, max_cols, min_tiles, max_rows, _, _) = tile_limits(seq.width, seq.height);
    w.flag(true); // uniform_tile_spacing_flag
    let cols = p.tile_log2.0.clamp(min_cols, max_cols);
    for _ in min_cols..cols {
        w.flag(true);
    }
    if cols < max_cols {
        w.flag(false);
    }
    let min_rows = min_tiles.saturating_sub(cols);
    let rows = p.tile_log2.1.clamp(min_rows, max_rows.max(min_rows));
    for _ in min_rows..rows {
        w.flag(true);
    }
    if rows < max_rows {
        w.flag(false);
    }
    if cols + rows > 0 {
        w.f(cols + rows, 0); // context_update_tile_id
        w.f(2, 3); // tile_size_bytes_minus_1: four-byte sizes
    }
    // quantization_params.
    w.f(8, u32::from(p.base_q_idx));
    w.flag(false); // DeltaQYDc
    if !seq.mono_chrome {
        w.flag(false); // DeltaQUDc
        w.flag(false); // DeltaQUAc
    }
    w.flag(false); // using_qmatrix
    w.flag(false); // segmentation_enabled
    w.flag(false); // delta_q_present
    // loop_filter_params.
    w.f(6, u32::from(p.loop_filter[0]));
    w.f(6, u32::from(p.loop_filter[1]));
    if !seq.mono_chrome && (p.loop_filter[0] != 0 || p.loop_filter[1] != 0) {
        w.f(6, u32::from(p.loop_filter[2]));
        w.f(6, u32::from(p.loop_filter[3]));
    }
    w.f(3, u32::from(p.sharpness));
    w.flag(false); // loop_filter_delta_enabled
    // cdef_params.
    let (damping, strengths) = &p.cdef;
    w.f(2, u32::from(*damping));
    let bits = strengths.len().max(1).trailing_zeros();
    w.f(2, bits);
    for s in strengths.iter().take(1 << bits) {
        w.f(4, u32::from(s.y_pri));
        w.f(2, u32::from(s.y_sec));
        if !seq.mono_chrome {
            w.f(4, u32::from(s.uv_pri));
            w.f(2, u32::from(s.uv_sec));
        }
    }
    // read_tx_mode, then reduced_tx_set.
    w.flag(p.tx_mode_select);
    w.flag(false);
    w.byte_alignment();
    w.finish()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::super::super::bits::BitReader;
    use super::super::super::frame::FrameHeader;
    use super::super::super::seq::SequenceHeader;
    use super::*;

    fn seq(width: u32, height: u32, mono: bool) -> SequenceParams {
        SequenceParams {
            width,
            height,
            mono_chrome: mono,
            cicp: (1, 13, 6),
            full_range: true,
        }
    }

    fn frame() -> FrameParams {
        FrameParams {
            base_q_idx: 120,
            loop_filter: [10, 12, 5, 6],
            sharpness: 2,
            cdef: (
                2,
                vec![
                    CdefStrength {
                        y_pri: 3,
                        y_sec: 1,
                        uv_pri: 2,
                        uv_sec: 0,
                    },
                    CdefStrength {
                        y_pri: 7,
                        y_sec: 2,
                        uv_pri: 4,
                        uv_sec: 1,
                    },
                ],
            ),
            tx_mode_select: true,
            tile_log2: (0, 0),
        }
    }

    #[test]
    fn headers_parse_back_to_what_was_written() {
        for &(w, h, mono) in &[
            (1, 1, false),
            (97, 61, false),
            (640, 480, true),
            (5000, 3000, false),
        ] {
            let sp = seq(w, h, mono);
            let bytes = sequence_header(&sp);
            let parsed = SequenceHeader::parse(&mut BitReader::new(&bytes)).unwrap();
            assert_eq!((parsed.max_frame_width, parsed.max_frame_height), (w, h));
            assert_eq!(parsed.color.mono_chrome, mono);
            assert_eq!(parsed.color.matrix_coefficients, 6);
            assert!(
                parsed.color.color_range && parsed.enable_cdef && !parsed.use_128x128_superblock
            );

            let fp = frame();
            let bytes = frame_header(&sp, &fp);
            let mut r = BitReader::new(&bytes);
            let fh = FrameHeader::parse(&mut r, &parsed, 0, 0).unwrap();
            r.byte_alignment().unwrap();
            assert_eq!(r.byte_position(), bytes.len(), "{w}x{h}: header length");
            assert_eq!(fh.quantization.base_q_idx, 120);
            assert_eq!(fh.loop_filter.sharpness, 2);
            assert_eq!(fh.loop_filter.level[0], 10);
            if !mono {
                assert_eq!(fh.loop_filter.level, [10, 12, 5, 6]);
            }
            assert_eq!(fh.cdef.bits, 1);
            assert_eq!(fh.cdef.y_pri_strength, vec![3, 7]);
            assert_eq!((fh.frame_width, fh.frame_height), (w, h));
            // 5000 wide needs two tile columns.
            assert_eq!(fh.tile_info.cols, if w > 4096 { 2 } else { 1 }, "{w}x{h}");
        }
    }

    #[test]
    fn levels_follow_the_picture_size() {
        assert_eq!(seq(640, 480, false).level(), 4);
        assert_eq!(seq(1920, 1080, false).level(), 8);
        assert_eq!(seq(4000, 2000, false).level(), 12);
        assert_eq!(seq(8000, 4000, false).level(), 16);
        assert_eq!(seq(20000, 100, false).level(), 31);
    }
}
