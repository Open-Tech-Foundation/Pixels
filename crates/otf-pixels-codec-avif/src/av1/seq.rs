//! The AV1 sequence header (spec §5.5).
//!
//! One sequence header governs every frame that follows it: the maximum frame
//! size, which coding tools are enabled, the superblock size, and the colour
//! configuration (bit depth, subsampling, range, matrix). In AVIF it arrives in
//! the `av1C` box's configuration OBUs, and the frame header cannot be read
//! without it.
//!
//! The whole header is parsed even though a still image leaves most of its
//! inter-prediction switches off: the fields sit in a fixed order with no
//! length prefix, so a field skipped is every field after it misread. What the
//! still-picture restriction buys is not a shorter parse but the guarantee that
//! the *values* land in their defaults — order hint off, force-integer-MV
//! selected — which the frame header then relies on.

use super::bits::BitReader;
use otf_pixels_core::{PixelsError, Result};

/// `SELECT_SCREEN_CONTENT_TOOLS` / `SELECT_INTEGER_MV` (§3): the sentinel that
/// defers the choice to each frame header.
const SELECT: u8 = 2;

// Colour code points that select the sRGB "identity matrix" fast path (§5.5.2).
const CP_BT_709: u8 = 1;
const TC_SRGB: u8 = 13;
const MC_IDENTITY: u8 = 0;
const CP_UNSPECIFIED: u8 = 2;
const TC_UNSPECIFIED: u8 = 2;
const MC_UNSPECIFIED: u8 = 2;

/// One operating point (§5.5.1). A still image has exactly one and decodes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatingPoint {
    /// `operating_point_idc` — the layer bitmask this point selects.
    pub idc: u16,
    /// `seq_level_idx` — the AV1 level.
    pub seq_level_idx: u8,
    /// `seq_tier` — Main (0) or High (1) tier.
    pub seq_tier: u8,
}

/// The colour configuration (`color_config`, §5.5.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    /// Sample bit depth: 8, 10 or 12.
    pub bit_depth: u8,
    /// Whether the stream carries only a luma plane.
    pub mono_chrome: bool,
    /// 1 for monochrome, 3 otherwise.
    pub num_planes: u8,
    /// CICP colour primaries.
    pub color_primaries: u8,
    /// CICP transfer characteristics.
    pub transfer_characteristics: u8,
    /// CICP matrix coefficients.
    pub matrix_coefficients: u8,
    /// Full-range (`true`) versus studio-range (`false`) samples.
    pub color_range: bool,
    /// Horizontal chroma subsampling (0 or 1).
    pub subsampling_x: u8,
    /// Vertical chroma subsampling (0 or 1).
    pub subsampling_y: u8,
    /// `chroma_sample_position` — siting of chroma against luma.
    pub chroma_sample_position: u8,
    /// Whether U and V carry independent delta-Q.
    pub separate_uv_delta_q: bool,
}

/// A fully parsed sequence header.
#[derive(Debug, Clone)]
pub struct SequenceHeader {
    /// `seq_profile` — 0, 1 or 2.
    pub seq_profile: u8,
    /// Whether the stream is flagged as a single still picture.
    pub still_picture: bool,
    /// Whether the compact still-picture header form was used.
    pub reduced_still_picture_header: bool,
    /// Every operating point in declaration order.
    pub operating_points: Vec<OperatingPoint>,
    /// `OperatingPointIdc` for the chosen point (point 0): the layer mask the
    /// frame header uses to decide whether to read temporal/spatial IDs.
    pub operating_point_idc: u16,
    /// Bits used to code a frame width, `frame_width_bits_minus_1 + 1`.
    pub frame_width_bits: u32,
    /// Bits used to code a frame height, `frame_height_bits_minus_1 + 1`.
    pub frame_height_bits: u32,
    /// Maximum frame width in samples.
    pub max_frame_width: u32,
    /// Maximum frame height in samples.
    pub max_frame_height: u32,
    /// Whether frames carry explicit frame-id numbers.
    pub frame_id_numbers_present: bool,
    /// `delta_frame_id_length_minus_2 + 2` when ids are present.
    pub delta_frame_id_length: u32,
    /// `additional_frame_id_length_minus_1 + 1` when ids are present.
    pub additional_frame_id_length: u32,
    /// Whether superblocks are 128x128 (`true`) or 64x64 (`false`).
    pub use_128x128_superblock: bool,
    /// Whether filter-intra prediction is enabled.
    pub enable_filter_intra: bool,
    /// Whether the intra edge filter is enabled.
    pub enable_intra_edge_filter: bool,
    /// Inter-only tool switches, retained so the frame header parse stays
    /// faithful even though a still image never exercises them.
    pub enable_interintra_compound: bool,
    /// Whether masked compound is enabled.
    pub enable_masked_compound: bool,
    /// Whether warped motion is enabled.
    pub enable_warped_motion: bool,
    /// Whether the dual interpolation filter is enabled.
    pub enable_dual_filter: bool,
    /// Whether order hints are coded.
    pub enable_order_hint: bool,
    /// Whether jnt_comp (distance-weighted compound) is enabled.
    pub enable_jnt_comp: bool,
    /// Whether reference-frame motion vectors are enabled.
    pub enable_ref_frame_mvs: bool,
    /// `seq_force_screen_content_tools`, possibly the `SELECT` sentinel.
    pub seq_force_screen_content_tools: u8,
    /// `seq_force_integer_mv`, possibly the `SELECT` sentinel.
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits` — bits used to code an order hint, 0 when disabled.
    pub order_hint_bits: u32,
    /// Whether super-resolution is enabled.
    pub enable_superres: bool,
    /// Whether CDEF is enabled.
    pub enable_cdef: bool,
    /// Whether loop restoration is enabled.
    pub enable_restoration: bool,
    /// The colour configuration.
    pub color: ColorConfig,
    /// Whether film-grain parameters may appear in frame headers.
    pub film_grain_params_present: bool,
    /// Whether a decoder model was signalled (affects the frame header).
    pub decoder_model_info_present: bool,
    /// `buffer_delay_length_minus_1 + 1` from the decoder model.
    pub buffer_delay_length: u32,
    /// `buffer_removal_time_length_minus_1 + 1` from the decoder model.
    pub buffer_removal_time_length: u32,
    /// `frame_presentation_time_length_minus_1 + 1` from the decoder model.
    pub frame_presentation_time_length: u32,
    /// Whether pictures are equally spaced in time.
    pub equal_picture_interval: bool,
    /// Whether timing information was present.
    pub timing_info_present: bool,
}

impl SequenceHeader {
    /// Parse a sequence header from the start of an OBU payload.
    pub fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        let seq_profile = r.f(3)? as u8;
        if seq_profile > 2 {
            return Err(PixelsError::malformed(
                "avif",
                "AV1 seq_profile above 2 is not a defined profile",
            ));
        }
        let still_picture = r.flag()?;
        let reduced_still_picture_header = r.flag()?;

        let mut timing_info_present = false;
        let mut decoder_model_info_present = false;
        let mut buffer_delay_length = 0;
        let mut buffer_removal_time_length = 0;
        let mut frame_presentation_time_length = 0;
        let mut equal_picture_interval = false;
        let mut operating_points = Vec::new();

        if reduced_still_picture_header {
            let seq_level_idx = r.f(5)? as u8;
            operating_points.push(OperatingPoint {
                idc: 0,
                seq_level_idx,
                seq_tier: 0,
            });
        } else {
            timing_info_present = r.flag()?;
            if timing_info_present {
                equal_picture_interval = parse_timing_info(r)?;
                decoder_model_info_present = r.flag()?;
                if decoder_model_info_present {
                    let model = parse_decoder_model_info(r)?;
                    buffer_delay_length = model.0;
                    buffer_removal_time_length = model.1;
                    frame_presentation_time_length = model.2;
                }
            }
            let initial_display_delay_present = r.flag()?;
            let operating_points_cnt = r.f(5)? + 1;
            for _ in 0..operating_points_cnt {
                let idc = r.f(12)? as u16;
                let seq_level_idx = r.f(5)? as u8;
                let seq_tier = if seq_level_idx > 7 { r.f(1)? as u8 } else { 0 };
                if decoder_model_info_present {
                    let present_for_op = r.flag()?;
                    if present_for_op {
                        // operating_parameters_info: two buffer delays + a flag.
                        r.f(buffer_delay_length)?;
                        r.f(buffer_delay_length)?;
                        r.f(1)?;
                    }
                }
                if initial_display_delay_present {
                    let present_for_op = r.flag()?;
                    if present_for_op {
                        r.f(4)?;
                    }
                }
                operating_points.push(OperatingPoint {
                    idc,
                    seq_level_idx,
                    seq_tier,
                });
            }
        }

        // choose_operating_point defaults to point 0.
        let operating_point_idc = operating_points.first().map_or(0, |op| op.idc);

        let frame_width_bits = r.f(4)? + 1;
        let frame_height_bits = r.f(4)? + 1;
        let max_frame_width = r.f(frame_width_bits)? + 1;
        let max_frame_height = r.f(frame_height_bits)? + 1;

        let frame_id_numbers_present = if reduced_still_picture_header {
            false
        } else {
            r.flag()?
        };
        let mut delta_frame_id_length = 0;
        let mut additional_frame_id_length = 0;
        if frame_id_numbers_present {
            delta_frame_id_length = r.f(4)? + 2;
            additional_frame_id_length = r.f(3)? + 1;
        }

        let use_128x128_superblock = r.flag()?;
        let enable_filter_intra = r.flag()?;
        let enable_intra_edge_filter = r.flag()?;

        let mut enable_interintra_compound = false;
        let mut enable_masked_compound = false;
        let mut enable_warped_motion = false;
        let mut enable_dual_filter = false;
        let mut enable_order_hint = false;
        let mut enable_jnt_comp = false;
        let mut enable_ref_frame_mvs = false;
        let mut seq_force_screen_content_tools = SELECT;
        let mut seq_force_integer_mv = SELECT;
        let mut order_hint_bits = 0;

        if !reduced_still_picture_header {
            enable_interintra_compound = r.flag()?;
            enable_masked_compound = r.flag()?;
            enable_warped_motion = r.flag()?;
            enable_dual_filter = r.flag()?;
            enable_order_hint = r.flag()?;
            if enable_order_hint {
                enable_jnt_comp = r.flag()?;
                enable_ref_frame_mvs = r.flag()?;
            }
            let seq_choose_screen_content_tools = r.flag()?;
            seq_force_screen_content_tools = if seq_choose_screen_content_tools {
                SELECT
            } else {
                r.f(1)? as u8
            };
            if seq_force_screen_content_tools > 0 {
                let seq_choose_integer_mv = r.flag()?;
                seq_force_integer_mv = if seq_choose_integer_mv {
                    SELECT
                } else {
                    r.f(1)? as u8
                };
            } else {
                seq_force_integer_mv = SELECT;
            }
            if enable_order_hint {
                order_hint_bits = r.f(3)? + 1;
            }
        }

        let enable_superres = r.flag()?;
        let enable_cdef = r.flag()?;
        let enable_restoration = r.flag()?;
        let color = parse_color_config(r, seq_profile)?;
        let film_grain_params_present = r.flag()?;

        Ok(Self {
            seq_profile,
            still_picture,
            reduced_still_picture_header,
            operating_points,
            operating_point_idc,
            frame_width_bits,
            frame_height_bits,
            max_frame_width,
            max_frame_height,
            frame_id_numbers_present,
            delta_frame_id_length,
            additional_frame_id_length,
            use_128x128_superblock,
            enable_filter_intra,
            enable_intra_edge_filter,
            enable_interintra_compound,
            enable_masked_compound,
            enable_warped_motion,
            enable_dual_filter,
            enable_order_hint,
            enable_jnt_comp,
            enable_ref_frame_mvs,
            seq_force_screen_content_tools,
            seq_force_integer_mv,
            order_hint_bits,
            enable_superres,
            enable_cdef,
            enable_restoration,
            color,
            film_grain_params_present,
            decoder_model_info_present,
            buffer_delay_length,
            buffer_removal_time_length,
            frame_presentation_time_length,
            equal_picture_interval,
            timing_info_present,
        })
    }
}

/// `timing_info` (§5.5.3). Returns `equal_picture_interval`.
fn parse_timing_info(r: &mut BitReader<'_>) -> Result<bool> {
    let _num_units_in_display_tick = r.f(32)?;
    let _time_scale = r.f(32)?;
    let equal_picture_interval = r.flag()?;
    if equal_picture_interval {
        let _num_ticks_per_picture_minus_1 = r.uvlc()?;
    }
    Ok(equal_picture_interval)
}

/// `decoder_model_info` (§5.5.4). Returns the three length fields the frame
/// header needs to size its own delay and presentation-time reads.
fn parse_decoder_model_info(r: &mut BitReader<'_>) -> Result<(u32, u32, u32)> {
    let buffer_delay_length = r.f(5)? + 1;
    let _num_units_in_decoding_tick = r.f(32)?;
    let buffer_removal_time_length = r.f(5)? + 1;
    let frame_presentation_time_length = r.f(5)? + 1;
    Ok((
        buffer_delay_length,
        buffer_removal_time_length,
        frame_presentation_time_length,
    ))
}

/// `color_config` (§5.5.2).
fn parse_color_config(r: &mut BitReader<'_>, seq_profile: u8) -> Result<ColorConfig> {
    let high_bitdepth = r.flag()?;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        if r.flag()? {
            12
        } else {
            10
        }
    } else if high_bitdepth {
        10
    } else {
        8
    };

    let mono_chrome = if seq_profile == 1 { false } else { r.flag()? };
    let num_planes = if mono_chrome { 1 } else { 3 };

    let color_description_present = r.flag()?;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present {
            (r.f(8)? as u8, r.f(8)? as u8, r.f(8)? as u8)
        } else {
            (CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED)
        };

    if mono_chrome {
        let color_range = r.flag()?;
        return Ok(ColorConfig {
            bit_depth,
            mono_chrome,
            num_planes,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
            subsampling_x: 1,
            subsampling_y: 1,
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        });
    }

    let (color_range, subsampling_x, subsampling_y);
    if color_primaries == CP_BT_709
        && transfer_characteristics == TC_SRGB
        && matrix_coefficients == MC_IDENTITY
    {
        // The sRGB fast path is implicitly full-range 4:4:4.
        color_range = true;
        subsampling_x = 0;
        subsampling_y = 0;
    } else {
        color_range = r.flag()?;
        match seq_profile {
            0 => {
                subsampling_x = 1;
                subsampling_y = 1;
            }
            1 => {
                subsampling_x = 0;
                subsampling_y = 0;
            }
            _ => {
                if bit_depth == 12 {
                    subsampling_x = r.f(1)? as u8;
                    subsampling_y = if subsampling_x != 0 { r.f(1)? as u8 } else { 0 };
                } else {
                    subsampling_x = 1;
                    subsampling_y = 0;
                }
            }
        }
    }

    let chroma_sample_position = if subsampling_x == 1 && subsampling_y == 1 {
        r.f(2)? as u8
    } else {
        0
    };
    let separate_uv_delta_q = r.flag()?;

    Ok(ColorConfig {
        bit_depth,
        mono_chrome,
        num_planes,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        color_range,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        separate_uv_delta_q,
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

    /// A minimal reduced still-picture sequence header, assembled bit by bit so
    /// the test reads as the syntax does. Returns the packed bytes.
    struct SeqBuilder {
        bits: Vec<u8>,
    }
    impl SeqBuilder {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        fn put(&mut self, value: u32, n: u32) -> &mut Self {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
            self
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

    /// Build the common reduced-still-picture header for an 8-bit 4:2:0 image
    /// of the given size, with no colour description.
    fn reduced_still(width: u32, height: u32) -> Vec<u8> {
        let mut b = SeqBuilder::new();
        b.put(0, 3); // seq_profile = 0
        b.put(1, 1); // still_picture = 1
        b.put(1, 1); // reduced_still_picture_header = 1
        b.put(1, 5); // seq_level_idx[0]
        b.put(15, 4); // frame_width_bits_minus_1 = 15 -> 16 bits
        b.put(15, 4); // frame_height_bits_minus_1 = 15 -> 16 bits
        b.put(width - 1, 16); // max_frame_width_minus_1
        b.put(height - 1, 16); // max_frame_height_minus_1
        b.put(0, 1); // use_128x128_superblock = 0
        b.put(0, 1); // enable_filter_intra = 0
        b.put(0, 1); // enable_intra_edge_filter = 0
        b.put(0, 1); // enable_superres = 0
        b.put(0, 1); // enable_cdef = 0
        b.put(0, 1); // enable_restoration = 0
        // color_config: high_bitdepth=0, mono_chrome=0,
        // color_description_present=0, then (not sRGB path) color_range=0,
        // profile 0 -> 4:2:0, chroma_sample_position(2), separate_uv_delta_q=0
        b.put(0, 1); // high_bitdepth = 0 -> 8-bit
        b.put(0, 1); // mono_chrome = 0
        b.put(0, 1); // color_description_present = 0
        b.put(0, 1); // color_range = 0
        b.put(0, 2); // chroma_sample_position
        b.put(0, 1); // separate_uv_delta_q = 0
        b.put(0, 1); // film_grain_params_present = 0
        b.pack()
    }

    #[test]
    fn parses_a_reduced_still_picture_header() {
        let bytes = reduced_still(320, 240);
        let mut r = BitReader::new(&bytes);
        let seq = SequenceHeader::parse(&mut r).unwrap();
        assert_eq!(seq.seq_profile, 0);
        assert!(seq.still_picture);
        assert!(seq.reduced_still_picture_header);
        assert_eq!(seq.max_frame_width, 320);
        assert_eq!(seq.max_frame_height, 240);
        assert_eq!(seq.color.bit_depth, 8);
        assert!(!seq.color.mono_chrome);
        assert_eq!(seq.color.subsampling_x, 1);
        assert_eq!(seq.color.subsampling_y, 1);
        assert_eq!(seq.operating_points.len(), 1);
        // Reduced header forces the inter tools to their off/select defaults.
        assert_eq!(seq.seq_force_screen_content_tools, SELECT);
        assert_eq!(seq.seq_force_integer_mv, SELECT);
        assert_eq!(seq.order_hint_bits, 0);
        assert!(!seq.enable_order_hint);
    }

    #[test]
    fn monochrome_forces_a_single_plane_and_subsampling() {
        let mut b = SeqBuilder::new();
        b.put(0, 3).put(1, 1).put(1, 1).put(1, 5);
        b.put(7, 4).put(7, 4); // 8-bit dimension fields
        b.put(99, 8).put(49, 8); // 100 x 50
        b.put(0, 1).put(0, 1).put(0, 1); // sb / filter-intra / edge
        b.put(0, 1).put(0, 1).put(0, 1); // superres / cdef / restoration
        b.put(0, 1); // high_bitdepth
        b.put(1, 1); // mono_chrome = 1
        b.put(0, 1); // color_description_present = 0
        b.put(0, 1); // color_range
        b.put(0, 1); // film_grain_params_present
        let bytes = b.pack();
        let mut r = BitReader::new(&bytes);
        let seq = SequenceHeader::parse(&mut r).unwrap();
        assert!(seq.color.mono_chrome);
        assert_eq!(seq.color.num_planes, 1);
        assert_eq!(seq.color.subsampling_x, 1);
        assert_eq!(seq.color.subsampling_y, 1);
        assert_eq!(seq.max_frame_width, 100);
        assert_eq!(seq.max_frame_height, 50);
    }

    #[test]
    fn the_srgb_identity_path_is_full_range_444() {
        let mut b = SeqBuilder::new();
        b.put(1, 3); // seq_profile = 1 (4:4:4 capable)
        b.put(1, 1).put(1, 1).put(1, 5);
        b.put(7, 4).put(7, 4).put(63, 8).put(63, 8); // 64 x 64
        b.put(0, 1).put(0, 1).put(0, 1);
        b.put(0, 1).put(0, 1).put(0, 1);
        b.put(0, 1); // high_bitdepth = 0
        // seq_profile == 1 -> mono_chrome not read
        b.put(1, 1); // color_description_present = 1
        b.put(CP_BT_709 as u32, 8);
        b.put(TC_SRGB as u32, 8);
        b.put(MC_IDENTITY as u32, 8);
        // sRGB identity path: color_range/subsampling not read;
        // subsampling is 0,0 so chroma_sample_position not read.
        b.put(0, 1); // separate_uv_delta_q
        b.put(0, 1); // film_grain_params_present
        let bytes = b.pack();
        let mut r = BitReader::new(&bytes);
        let seq = SequenceHeader::parse(&mut r).unwrap();
        assert_eq!(seq.color.subsampling_x, 0);
        assert_eq!(seq.color.subsampling_y, 0);
        assert!(seq.color.color_range);
        assert_eq!(seq.color.matrix_coefficients, MC_IDENTITY);
    }

    #[test]
    fn a_profile_above_two_is_rejected() {
        let bytes = [0xE0]; // seq_profile = 7
        let mut r = BitReader::new(&bytes);
        assert!(SequenceHeader::parse(&mut r).is_err());
    }
}
