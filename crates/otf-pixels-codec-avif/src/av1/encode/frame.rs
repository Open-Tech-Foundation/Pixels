//! A whole AV1 key frame: headers, tiles, OBUs.
//!
//! The headers are written, then parsed back with the decoder's own parsers,
//! and the decoder's tile state is built from what it read, so the encoder
//! codes exactly the frame a decoder will see.

use super::super::bits::BitReader;
use super::super::frame::FrameHeader;
use super::super::seq::SequenceHeader;
use super::super::tile::{TileState, tile_bounds};
use super::super::transform::ac_q;
use super::headers::{
    CdefStrength, FrameParams, OBU_FRAME, OBU_SEQUENCE_HEADER, SequenceParams, frame_header, obu,
    sequence_header,
};
use super::tile::{SourcePlane, TileEncoder, Tuning};
use otf_pixels_core::{PixelsError, Result};

/// An 8-bit picture to code: one luma plane, or luma and two 4:2:0 chroma
/// planes, each row-major at its own size.
pub(crate) struct Picture<'a> {
    pub width: u32,
    pub height: u32,
    /// Y, then U and V (half size, rounded up) unless monochrome.
    pub planes: &'a [&'a [u16]],
    /// `(colour_primaries, transfer, matrix)`.
    pub cicp: (u8, u8, u8),
    pub full_range: bool,
}

/// A coded still: the sequence header OBU (for `av1C`) and the full temporal
/// unit (for the item's data).
pub(crate) struct CodedStill {
    pub sequence_header_obu: Vec<u8>,
    pub data: Vec<u8>,
}

/// `base_q_idx` for a 0 (worst) to 100 (best) quality. Never 0, which would
/// make the frame lossless and change its whole coding.
pub(crate) fn qindex_for_quality(quality: u8) -> u8 {
    let q = (f64::from(100 - quality.min(100)) * 2.55).round();
    q.clamp(1.0, 255.0) as u8
}

/// The deblocking level libaom estimates for an 8-bit key frame at `qindex`.
fn loop_filter_level(qindex: u8) -> u8 {
    let q = ac_q(8, i32::from(qindex));
    let guess = ((q * 17_563 - 421_574 + (1 << 17)) >> 18) - 4;
    guess.clamp(0, 63) as u8
}

fn cdef_strength(qindex: u8) -> CdefStrength {
    let y_pri = (qindex / 32).min(8);
    let y_sec = u8::from(qindex >= 80);
    CdefStrength {
        y_pri,
        y_sec,
        uv_pri: y_pri / 2,
        uv_sec: y_sec,
    }
}

/// Encode `picture` at `qindex` as a reduced-still-picture AV1 stream.
pub(crate) fn encode_still(picture: &Picture<'_>, qindex: u8) -> Result<CodedStill> {
    let mono_chrome = picture.planes.len() == 1;
    if picture.width == 0
        || picture.height == 0
        || (picture.planes.len() != 1 && picture.planes.len() != 3)
    {
        return Err(PixelsError::invalid_argument(
            "picture",
            "an AV1 picture needs a size and one or three planes",
        ));
    }
    let seq_params = SequenceParams {
        width: picture.width,
        height: picture.height,
        mono_chrome,
        cicp: picture.cicp,
        full_range: picture.full_range,
    };
    let level = loop_filter_level(qindex);
    let frame_params = FrameParams {
        base_q_idx: qindex,
        loop_filter: [level, level, level, level],
        sharpness: 0,
        cdef: (1, vec![cdef_strength(qindex)]),
        tx_mode_select: false,
        tile_log2: (0, 0),
    };
    let seq_payload = sequence_header(&seq_params);
    let frame_payload = frame_header(&seq_params, &frame_params);
    let seq = SequenceHeader::parse(&mut BitReader::new(&seq_payload))?;
    let frame = FrameHeader::parse(&mut BitReader::new(&frame_payload), &seq, 0, 0)?;
    let mut state = TileState::for_frame(&seq, &frame)?;

    // The source, padded to the decoder's plane sizes.
    let mut source = Vec::with_capacity(picture.planes.len());
    for (index, samples) in picture.planes.iter().enumerate() {
        let sub = u32::from(index > 0);
        let (w, h) = (
            ((picture.width + sub) >> sub) as usize,
            ((picture.height + sub) >> sub) as usize,
        );
        if samples.len() < w * h {
            return Err(PixelsError::invalid_argument(
                "picture",
                "an AV1 plane is smaller than the picture",
            ));
        }
        let plane = state.planes.get(index).ok_or_else(|| {
            PixelsError::invalid_argument("picture", "the frame has fewer planes than given")
        })?;
        source.push(SourcePlane::padded(
            samples,
            w,
            h,
            plane.width(),
            plane.height(),
        ));
    }

    let tuning = Tuning {
        ac_q: ac_q(8, i32::from(qindex)),
        dc_bias: 0.5,
        ac_bias: 0.36,
    };
    let info = &frame.tile_info;
    let (mi_rows, mi_cols) = state.mi_dims();
    let mut tiles = Vec::with_capacity(info.count() as usize);
    for number in 0..info.count() {
        let bounds = tile_bounds(info, 4, mi_rows, mi_cols, number);
        let mut coder = TileEncoder::new(&source, tuning);
        state.code_tile(&mut coder, bounds)?;
        tiles.push(coder.finish());
    }

    // OBU_FRAME: the header, then the tile group (§5.11.1).
    let mut payload = frame_payload;
    if tiles.len() > 1 {
        payload.push(0); // tile_start_and_end_present_flag = 0, byte-aligned
    }
    let last = tiles.len().saturating_sub(1);
    for (i, tile) in tiles.iter().enumerate() {
        if i < last {
            let size = u32::try_from(tile.len() - 1).map_err(|_| {
                PixelsError::invalid_argument("picture", "a tile larger than 4 GiB")
            })?;
            payload.extend_from_slice(&size.to_le_bytes());
        }
        payload.extend_from_slice(tile);
    }
    let sequence_header_obu = obu(OBU_SEQUENCE_HEADER, &seq_payload);
    let mut data = sequence_header_obu.clone();
    data.extend_from_slice(&obu(OBU_FRAME, &payload));
    Ok(CodedStill {
        sequence_header_obu,
        data,
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::super::super::still::StillPicture;
    use super::super::super::tile::decode_still;
    use super::*;

    /// A picture with smooth gradients, edges and texture.
    fn scene(width: u32, height: u32) -> Vec<Vec<u16>> {
        let (w, h) = (width as usize, height as usize);
        let mut y = vec![0_u16; w * h];
        for j in 0..h {
            for i in 0..w {
                let ramp = (i * 160 / w.max(1) + j * 60 / h.max(1)) as i32;
                let disc = if (i as i32 - w as i32 / 2).pow(2) + (j as i32 - h as i32 / 3).pow(2)
                    < (w as i32 / 4).pow(2)
                {
                    50
                } else {
                    0
                };
                let stripes = if (i / 3 + j / 5) % 2 == 0 && j > h * 2 / 3 {
                    30
                } else {
                    0
                };
                y[j * w + i] = (ramp + disc + stripes).clamp(0, 255) as u16;
            }
        }
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let u = (0..cw * ch)
            .map(|k| (128 + (k % cw) * 40 / cw.max(1)) as u16)
            .collect();
        let v = (0..cw * ch)
            .map(|k| (100 + (k / cw) * 60 / ch.max(1)) as u16)
            .collect();
        vec![y, u, v]
    }

    fn psnr(a: &[u16], b: &[u16]) -> f64 {
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x as i32 - y as i32).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        }
    }

    fn round_trip(
        width: u32,
        height: u32,
        planes: &[Vec<u16>],
        qindex: u8,
    ) -> (Vec<Vec<u16>>, usize) {
        let refs: Vec<&[u16]> = planes.iter().map(Vec::as_slice).collect();
        let picture = Picture {
            width,
            height,
            planes: &refs,
            cicp: (1, 13, 6),
            full_range: true,
        };
        let coded = encode_still(&picture, qindex).unwrap();
        let still = StillPicture::parse(&coded.sequence_header_obu, &coded.data).unwrap();
        let groups: Vec<&[u8]> = still
            .tile_groups
            .iter()
            .map(|r| &coded.data[r.clone()])
            .collect();
        let decoded = decode_still(&still.sequence, &still.frame, &groups).unwrap();
        let out = decoded
            .planes
            .iter()
            .enumerate()
            .map(|(i, plane)| {
                let sub = u32::from(i > 0);
                let (w, h) = (
                    ((width + sub) >> sub) as usize,
                    ((height + sub) >> sub) as usize,
                );
                (0..w * h)
                    .map(|k| plane.get(k % w, k / w).unwrap())
                    .collect()
            })
            .collect();
        (out, coded.data.len())
    }

    /// With `OTF_EMIT_DIR` set, writes coded streams (`.obu`, Annex-less with
    /// a temporal delimiter) and our decode of them (`.yuv`, I420 or Y only)
    /// for `scripts/check-avif-interop.sh` to compare against libaom.
    #[test]
    fn emit_streams_for_interop() {
        let Some(dir) = std::env::var_os("OTF_EMIT_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        for (width, height, qindex, mono) in [
            (64, 64, 40, false),
            (100, 75, 120, false),
            (130, 33, 1, false),
            (1, 1, 255, false),
            (300, 200, 80, false),
            (48, 40, 60, true),
        ] {
            let mut planes = scene(width, height);
            if mono {
                planes.truncate(1);
            }
            let refs: Vec<&[u16]> = planes.iter().map(Vec::as_slice).collect();
            let picture = Picture {
                width,
                height,
                planes: &refs,
                cicp: (1, 13, 6),
                full_range: true,
            };
            let coded = encode_still(&picture, qindex).unwrap();
            let mut obus = vec![0x12, 0x00];
            obus.extend_from_slice(&coded.data);
            let name = format!(
                "{width}x{height}_q{qindex}{}",
                if mono { "_mono" } else { "" }
            );
            std::fs::write(dir.join(format!("{name}.obu")), obus).unwrap();
            let (out, _) = round_trip(width, height, &planes, qindex);
            let bytes: Vec<u8> = out.iter().flatten().map(|&v| v as u8).collect();
            std::fs::write(dir.join(format!("{name}.yuv")), bytes).unwrap();
        }
    }

    #[test]
    fn a_coded_picture_decodes_close_to_its_source() {
        for (width, height) in [(64, 64), (100, 75), (8, 8), (1, 1), (130, 33)] {
            let planes = scene(width, height);
            let (out, _) = round_trip(width, height, &planes, 40);
            for (plane, (a, b)) in out.iter().zip(&planes).enumerate() {
                let p = psnr(a, b);
                assert!(p > 36.0, "{width}x{height} plane {plane}: {p:.1} dB");
            }
        }
    }

    #[test]
    fn quality_trades_size_for_fidelity() {
        let planes = scene(128, 96);
        let (fine, fine_size) = round_trip(128, 96, &planes, 20);
        let (coarse, coarse_size) = round_trip(128, 96, &planes, 200);
        assert!(coarse_size < fine_size, "{coarse_size} >= {fine_size}");
        assert!(psnr(&fine[0], &planes[0]) > psnr(&coarse[0], &planes[0]));
        assert!(psnr(&coarse[0], &planes[0]) > 24.0);
    }

    #[test]
    fn a_monochrome_picture_codes_one_plane() {
        let planes = vec![scene(48, 40).swap_remove(0)];
        let (out, _) = round_trip(48, 40, &planes, 60);
        assert_eq!(out.len(), 1);
        assert!(psnr(&out[0], &planes[0]) > 34.0);
    }

    #[test]
    fn extreme_samples_survive_the_finest_quantizer() {
        // Full-swing checkerboard at qindex 1: large coefficients, Golomb tails.
        let (w, h) = (32_u32, 32_u32);
        let y: Vec<u16> = (0..w * h)
            .map(|k| if (k % w + k / w) % 2 == 0 { 0 } else { 255 })
            .collect();
        let c = vec![255_u16; 256];
        let planes = vec![y, c.clone(), vec![0; 256]];
        let (out, _) = round_trip(w, h, &planes, 1);
        assert!(
            psnr(&out[0], &planes[0]) > 40.0,
            "{}",
            psnr(&out[0], &planes[0])
        );
        assert_eq!(out[1], planes[1]);
    }

    #[test]
    fn quality_maps_onto_nonzero_quantizers() {
        assert_eq!(qindex_for_quality(100), 1);
        assert_eq!(qindex_for_quality(0), 255);
        assert!(qindex_for_quality(50) > qindex_for_quality(80));
        assert_eq!(loop_filter_level(1), 0);
        assert!(loop_filter_level(200) > loop_filter_level(60));
    }

    #[test]
    fn rejects_a_malformed_picture() {
        let y = vec![0_u16; 4];
        let refs: Vec<&[u16]> = vec![&y, &y];
        let picture = Picture {
            width: 2,
            height: 2,
            planes: &refs,
            cicp: (1, 13, 6),
            full_range: true,
        };
        assert!(encode_still(&picture, 50).is_err());
        let refs: Vec<&[u16]> = vec![&y];
        let picture = Picture {
            width: 3,
            height: 2,
            planes: &refs,
            cicp: (1, 13, 6),
            full_range: true,
        };
        assert!(encode_still(&picture, 50).is_err());
    }
}
