//! The `ALPH` chunk (RFC 9649 §2.7.1.2): a lossy image's alpha plane.
//!
//! One header byte, then the plane either raw or as a VP8L image stream of
//! implicit size whose green channel is the alpha; then an optional spatial
//! filter to undo.

#![allow(
    clippy::indexing_slicing,
    reason = "the filter reads the left neighbour only where x > 0 and the row \
              above only where y > 0, inside a plane of exactly width * height"
)]

use crate::vp8l::{BitReader, decode_image_stream};
use otf_pixels_core::{PixelsError, Result};

fn malformed(detail: impl Into<String>) -> PixelsError {
    PixelsError::malformed("webp", detail.into())
}

/// Decode an `ALPH` payload into `width * height` alpha samples.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for an unknown compression method or a
/// broken or short alpha stream.
pub fn decode(payload: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
    let (&header, data) = payload
        .split_first()
        .ok_or_else(|| malformed("the ALPH chunk is empty"))?;
    let count = width * height;
    let mut alpha = match header & 3 {
        0 => data
            .get(..count)
            .ok_or_else(|| malformed("the raw alpha plane is cut short"))?
            .to_vec(),
        1 => {
            let mut bits = BitReader::new(data);
            decode_image_stream(&mut bits, width, height, true)?
                .into_iter()
                .map(|argb| (argb >> 8) as u8)
                .collect()
        }
        method => {
            return Err(malformed(format!(
                "alpha compression method {method} is not defined"
            )));
        }
    };
    unfilter(&mut alpha, width, (header >> 2) & 3);
    Ok(alpha)
}

/// Undo the alpha filter in place: 1 horizontal, 2 vertical, 3 gradient.
fn unfilter(alpha: &mut [u8], width: usize, method: u8) {
    if method == 0 || width == 0 {
        return;
    }
    for i in 0..alpha.len() {
        let (x, y) = (i % width, i / width);
        let left = || alpha[i - 1];
        let above = || alpha[i - width];
        let prediction = match (x, y) {
            (0, 0) => 0,
            // The top row predicts from the left whatever the method, and
            // the left column from above.
            (_, 0) => left(),
            (0, _) => above(),
            _ => match method {
                1 => left(),
                2 => above(),
                _ => {
                    let gradient =
                        i32::from(left()) + i32::from(above()) - i32::from(alpha[i - width - 1]);
                    gradient.clamp(0, 255) as u8
                }
            },
        };
        alpha[i] = alpha[i].wrapping_add(prediction);
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    #[test]
    fn raw_alpha_with_each_filter_undoes_to_the_same_plane() {
        // A 3x2 plane: [10, 20, 30 / 40, 50, 60].
        let plane = [10_u8, 20, 30, 40, 50, 60];
        let encode = |method: u8| -> Vec<u8> {
            let mut out = vec![method << 2];
            for i in 0..6 {
                let (x, y) = (i % 3, i / 3);
                let p = match (x, y, method) {
                    (_, _, 0) | (0, 0, _) => 0,
                    (_, 0, _) => plane[i - 1],
                    (0, _, _) => plane[i - 3],
                    (_, _, 1) => plane[i - 1],
                    (_, _, 2) => plane[i - 3],
                    _ => (i32::from(plane[i - 1]) + i32::from(plane[i - 3])
                        - i32::from(plane[i - 4]))
                    .clamp(0, 255) as u8,
                };
                out.push(plane[i].wrapping_sub(p));
            }
            out
        };
        for method in 0..4 {
            assert_eq!(
                decode(&encode(method), 3, 2).unwrap(),
                plane,
                "method {method}"
            );
        }
    }

    #[test]
    fn a_short_or_unknown_alpha_chunk_is_malformed() {
        assert!(decode(&[], 2, 2).is_err());
        assert!(decode(&[0, 1, 2], 2, 2).is_err());
        assert!(decode(&[2, 0, 0, 0, 0], 2, 2).is_err());
    }
}
