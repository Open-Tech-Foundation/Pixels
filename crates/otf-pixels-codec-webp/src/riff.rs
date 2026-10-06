//! The WebP RIFF container (RFC 9649 §2).
//!
//! A WebP file is a RIFF form of type `WEBP` holding one of three layouts: a
//! lone `VP8 ` (lossy) or `VP8L` (lossless) chunk, or an extended file opened
//! by `VP8X` whose image may carry an `ALPH` alpha chunk, metadata, and — for
//! an animation — `ANMF` frames. Parsing reduces all three to one
//! [`Container`]: the canvas, the coded image to decode and where it sits on
//! that canvas, and the metadata the engine uses.

use otf_pixels_core::{PixelsError, Result};

/// How the primary image is coded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bitstream<'a> {
    /// A `VP8 ` key frame, with the `ALPH` chunk's payload if one precedes it.
    Lossy {
        /// The VP8 data.
        vp8: &'a [u8],
        /// The `ALPH` payload: its one header byte, then the alpha bitstream.
        alpha: Option<&'a [u8]>,
    },
    /// A `VP8L` image stream, alpha included.
    Lossless(&'a [u8]),
}

/// Where the decoded image lands on the canvas: the whole canvas for a still,
/// the first frame's rectangle for an animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Left edge on the canvas.
    pub x: u32,
    /// Top edge on the canvas.
    pub y: u32,
    /// Width of the coded image.
    pub width: u32,
    /// Height of the coded image.
    pub height: u32,
}

/// What the container says about the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Container<'a> {
    /// The canvas width: the `VP8X` canvas, or the bitstream's own size.
    pub width: u32,
    /// The canvas height.
    pub height: u32,
    /// Whether the image has alpha, as libwebp reports it: the `VP8X` flag in
    /// an extended file, otherwise the presence of `ALPH` or the `VP8L`
    /// header's `alpha_is_used` hint.
    pub has_alpha: bool,
    /// Whether the file is an animation, of which only the first frame is
    /// decoded (SPEC §Formats).
    pub animated: bool,
    /// The coded primary image.
    pub bitstream: Bitstream<'a>,
    /// Where that image sits on the canvas.
    pub frame: Frame,
    /// The first `EXIF` chunk's payload.
    pub exif: Option<&'a [u8]>,
    /// The `ICCP` chunk's payload: the ICC profile.
    pub icc: Option<&'a [u8]>,
}

/// One chunk: its FourCC and payload.
struct Chunk<'a> {
    kind: [u8; 4],
    payload: &'a [u8],
}

/// Walk the padded chunks of `data` in order.
fn chunks(mut data: &[u8]) -> impl Iterator<Item = Result<Chunk<'_>>> {
    core::iter::from_fn(move || {
        if data.is_empty() {
            return None;
        }
        let Some((head, rest)) = data.split_first_chunk::<8>() else {
            data = &[];
            return Some(Err(malformed("a chunk header is cut short")));
        };
        let kind = [head[0], head[1], head[2], head[3]];
        let size = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
        let Some(payload) = rest.get(..size) else {
            data = &[];
            return Some(Err(malformed(format!(
                "chunk '{}' claims {size} bytes, more than the file holds",
                String::from_utf8_lossy(&kind)
            ))));
        };
        // An odd payload is followed by one padding byte (§2.3); a file that
        // ends without it is accepted, as libwebp does.
        data = rest.get(size + (size & 1)..).unwrap_or(&[]);
        Some(Ok(Chunk { kind, payload }))
    })
}

fn malformed(detail: impl Into<String>) -> PixelsError {
    PixelsError::malformed("webp", detail.into())
}

fn u24(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .take(3)
        .rev()
        .fold(0, |acc, &b| (acc << 8) | u32::from(b))
}

/// The size a `VP8 ` key frame declares (RFC 6386 §9.1): a 3-byte frame tag,
/// the start code `9d 01 2a`, then 14-bit width and height.
fn vp8_size(data: &[u8]) -> Result<(u32, u32)> {
    let header = data
        .first_chunk::<10>()
        .ok_or_else(|| malformed("the VP8 frame header is cut short"))?;
    if header[0] & 1 != 0 {
        return Err(malformed("the VP8 frame is not a key frame"));
    }
    if [header[3], header[4], header[5]] != [0x9d, 0x01, 0x2a] {
        return Err(malformed("the VP8 frame lacks its start code"));
    }
    let width = u32::from(u16::from_le_bytes([header[6], header[7]]) & 0x3fff);
    let height = u32::from(u16::from_le_bytes([header[8], header[9]]) & 0x3fff);
    Ok((width, height))
}

/// The size and alpha hint a `VP8L` stream declares (RFC 9649 §3.4).
fn vp8l_size(data: &[u8]) -> Result<(u32, u32, bool)> {
    let header = data
        .first_chunk::<5>()
        .ok_or_else(|| malformed("the VP8L header is cut short"))?;
    if header[0] != 0x2f {
        return Err(malformed("the VP8L stream lacks its 0x2f signature"));
    }
    let bits = u32::from_le_bytes([header[1], header[2], header[3], header[4]]);
    let width = (bits & 0x3fff) + 1;
    let height = ((bits >> 14) & 0x3fff) + 1;
    let alpha = (bits >> 28) & 1 == 1;
    if bits >> 29 != 0 {
        return Err(malformed("the VP8L version is not 0"));
    }
    Ok((width, height, alpha))
}

/// Parse a whole WebP file.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for anything that is not a well-formed
/// WebP: a bad header, a chunk overrunning the file, a missing or misplaced
/// bitstream, or a frame that does not fit its canvas.
pub fn parse(file: &[u8]) -> Result<Container<'_>> {
    let (header, rest) = file
        .split_first_chunk::<12>()
        .ok_or_else(|| malformed("the file is shorter than a RIFF header"))?;
    if header[..4] != *b"RIFF" || header[8..] != *b"WEBP" {
        return Err(malformed("not a RIFF WEBP file"));
    }
    // The RIFF size counts from offset 8; trailing data past it is ignored
    // (§2.4), and a size claiming more than the file holds is truncation.
    let riff_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let body = rest
        .get(
            ..riff_size
                .checked_sub(4)
                .ok_or_else(|| malformed("RIFF size below 4"))?,
        )
        .ok_or_else(|| malformed("the RIFF size exceeds the file"))?;

    let mut chunks = chunks(body);
    let first = chunks
        .next()
        .ok_or_else(|| malformed("the file has no chunks"))??;
    match &first.kind {
        b"VP8 " => {
            let (width, height) = vp8_size(first.payload)?;
            Ok(simple(
                width,
                height,
                false,
                Bitstream::Lossy {
                    vp8: first.payload,
                    alpha: None,
                },
            ))
        }
        b"VP8L" => {
            let (width, height, alpha) = vp8l_size(first.payload)?;
            Ok(simple(
                width,
                height,
                alpha,
                Bitstream::Lossless(first.payload),
            ))
        }
        b"VP8X" => extended(first.payload, chunks),
        other => Err(malformed(format!(
            "the first chunk is '{}', not an image",
            String::from_utf8_lossy(other)
        ))),
    }
}

fn simple(width: u32, height: u32, has_alpha: bool, bitstream: Bitstream<'_>) -> Container<'_> {
    Container {
        width,
        height,
        has_alpha,
        animated: false,
        bitstream,
        frame: Frame {
            x: 0,
            y: 0,
            width,
            height,
        },
        exif: None,
        icc: None,
    }
}

/// An extended file (§2.7): `VP8X`, then the image or its first frame.
fn extended<'a>(
    vp8x: &'a [u8],
    chunks: impl Iterator<Item = Result<Chunk<'a>>>,
) -> Result<Container<'a>> {
    let header = vp8x
        .first_chunk::<10>()
        .ok_or_else(|| malformed("the VP8X chunk is cut short"))?;
    let flags = header[0];
    let has_alpha = flags & 0x10 != 0;
    let animated = flags & 0x02 != 0;
    let width = u24(&[header[4], header[5], header[6]]) + 1;
    let height = u24(&[header[7], header[8], header[9]]) + 1;
    if u64::from(width) * u64::from(height) > u64::from(u32::MAX) {
        return Err(malformed("the canvas exceeds 2^32 - 1 pixels"));
    }

    let mut image: Option<(Bitstream<'a>, Frame)> = None;
    let mut alpha: Option<&'a [u8]> = None;
    let mut exif: Option<&'a [u8]> = None;
    let mut icc: Option<&'a [u8]> = None;
    for chunk in chunks {
        let chunk = chunk?;
        match &chunk.kind {
            b"EXIF" => exif = exif.or(Some(chunk.payload)),
            b"ICCP" => icc = icc.or(Some(chunk.payload)),
            _ if image.is_some() => {}
            b"ALPH" if !animated => alpha = alpha.or(Some(chunk.payload)),
            b"VP8 " if !animated => {
                let (w, h) = vp8_size(chunk.payload)?;
                image = Some((
                    Bitstream::Lossy {
                        vp8: chunk.payload,
                        alpha,
                    },
                    whole(w, h),
                ));
            }
            b"VP8L" if !animated => {
                let (w, h, _) = vp8l_size(chunk.payload)?;
                image = Some((Bitstream::Lossless(chunk.payload), whole(w, h)));
            }
            b"ANMF" if animated => image = Some(first_frame(chunk.payload)?),
            _ => {}
        }
    }
    let (bitstream, frame) =
        image.ok_or_else(|| malformed("the extended file holds no image data"))?;
    let fits = u64::from(frame.x) + u64::from(frame.width) <= u64::from(width)
        && u64::from(frame.y) + u64::from(frame.height) <= u64::from(height);
    if !fits || (!animated && (frame.width, frame.height) != (width, height)) {
        return Err(malformed(format!(
            "a {}x{} image at ({}, {}) does not fit the {width}x{height} canvas",
            frame.width, frame.height, frame.x, frame.y
        )));
    }
    Ok(Container {
        width,
        height,
        has_alpha,
        animated,
        bitstream,
        frame,
        exif,
        icc,
    })
}

const fn whole(width: u32, height: u32) -> Frame {
    Frame {
        x: 0,
        y: 0,
        width,
        height,
    }
}

/// The image inside an `ANMF` chunk (§2.7.1.1) and its rectangle.
fn first_frame(anmf: &[u8]) -> Result<(Bitstream<'_>, Frame)> {
    let header = anmf
        .first_chunk::<12>()
        .ok_or_else(|| malformed("the ANMF chunk is cut short"))?;
    if anmf.len() < 16 {
        return Err(malformed("the ANMF chunk is cut short"));
    }
    let frame = Frame {
        x: u24(&[header[0], header[1], header[2]]) * 2,
        y: u24(&[header[3], header[4], header[5]]) * 2,
        width: u24(&[header[6], header[7], header[8]]) + 1,
        height: u24(&[header[9], header[10], header[11]]) + 1,
    };
    let mut alpha = None;
    for chunk in chunks(anmf.get(16..).unwrap_or(&[])) {
        let chunk = chunk?;
        let size = match &chunk.kind {
            b"ALPH" => {
                alpha = alpha.or(Some(chunk.payload));
                continue;
            }
            b"VP8 " => vp8_size(chunk.payload)?,
            b"VP8L" => {
                let (w, h, _) = vp8l_size(chunk.payload)?;
                (w, h)
            }
            _ => continue,
        };
        if size != (frame.width, frame.height) {
            return Err(malformed(
                "an animation frame's bitstream does not match its size",
            ));
        }
        let bitstream = if chunk.kind == *b"VP8L" {
            Bitstream::Lossless(chunk.payload)
        } else {
            Bitstream::Lossy {
                vp8: chunk.payload,
                alpha,
            }
        };
        return Ok((bitstream, frame));
    }
    Err(malformed("the first animation frame holds no bitstream"))
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

    fn chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = kind.to_vec();
        out.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn riff(body: &[u8]) -> Vec<u8> {
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&u32::try_from(body.len() + 4).unwrap().to_le_bytes());
        out.extend_from_slice(b"WEBP");
        out.extend_from_slice(body);
        out
    }

    /// A VP8L header for a `w` x `h` image, alpha hint `alpha`, plus a byte.
    fn vp8l(w: u32, h: u32, alpha: bool) -> Vec<u8> {
        let bits = (w - 1) | ((h - 1) << 14) | (u32::from(alpha) << 28);
        let mut out = vec![0x2f];
        out.extend_from_slice(&bits.to_le_bytes());
        out.push(0);
        out
    }

    fn vp8(w: u16, h: u16) -> Vec<u8> {
        let mut out = vec![0x10, 0x02, 0x00, 0x9d, 0x01, 0x2a];
        out.extend_from_slice(&w.to_le_bytes());
        out.extend_from_slice(&h.to_le_bytes());
        out
    }

    fn vp8x(flags: u8, w: u32, h: u32) -> Vec<u8> {
        let mut out = vec![flags, 0, 0, 0];
        out.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
        out.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
        out
    }

    #[test]
    fn simple_lossless_and_lossy_files_report_their_size() {
        let file = riff(&chunk(b"VP8L", &vp8l(7, 5, true)));
        let c = parse(&file).unwrap();
        assert_eq!((c.width, c.height, c.has_alpha), (7, 5, true));
        assert!(matches!(c.bitstream, Bitstream::Lossless(_)));

        let file = riff(&chunk(b"VP8 ", &vp8(33, 17)));
        let c = parse(&file).unwrap();
        assert_eq!((c.width, c.height, c.has_alpha), (33, 17, false));
        assert!(matches!(c.bitstream, Bitstream::Lossy { alpha: None, .. }));
    }

    #[test]
    fn an_extended_file_carries_alpha_and_exif() {
        let mut body = chunk(b"VP8X", &vp8x(0x18, 33, 17));
        body.extend(chunk(b"ALPH", &[0, 1, 2, 3]));
        body.extend(chunk(b"VP8 ", &vp8(33, 17)));
        body.extend(chunk(b"EXIF", b"II*\0"));
        let file = riff(&body);
        let c = parse(&file).unwrap();
        assert!(c.has_alpha);
        assert_eq!(c.exif, Some(&b"II*\0"[..]));
        let Bitstream::Lossy { alpha, .. } = c.bitstream else {
            panic!("expected lossy")
        };
        assert_eq!(alpha, Some(&[0, 1, 2, 3][..]));
    }

    #[test]
    fn an_animation_yields_its_first_frame_rectangle() {
        let mut anmf = Vec::new();
        anmf.extend_from_slice(&[2, 0, 0, 1, 0, 0]); // x = 4, y = 2
        anmf.extend_from_slice(&[6, 0, 0, 4, 0, 0]); // 7 x 5
        anmf.extend_from_slice(&[100, 0, 0, 0]); // duration, flags
        anmf.extend(chunk(b"VP8L", &vp8l(7, 5, false)));
        let mut body = chunk(b"VP8X", &vp8x(0x12, 20, 10));
        body.extend(chunk(b"ANIM", &[0; 6]));
        body.extend(chunk(b"ANMF", &anmf));
        body.extend(chunk(b"ANMF", &anmf));
        let file = riff(&body);
        let c = parse(&file).unwrap();
        assert!(c.animated && c.has_alpha);
        assert_eq!(
            c.frame,
            Frame {
                x: 4,
                y: 2,
                width: 7,
                height: 5
            }
        );
    }

    #[test]
    fn broken_containers_are_malformed_not_panics() {
        let cases: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"RIFF\0\0\0\0WAVE".to_vec(),
            // Chunk size past the end.
            riff(&[b'V', b'P', b'8', b'L', 0xff, 0, 0, 0, 0x2f]),
            // VP8X with no image.
            riff(&chunk(b"VP8X", &vp8x(0, 4, 4))),
            // VP8X canvas differing from its still image.
            {
                let mut b = chunk(b"VP8X", &vp8x(0, 8, 8));
                b.extend(chunk(b"VP8L", &vp8l(4, 4, false)));
                riff(&b)
            },
            // A VP8L stream with a bad signature or version.
            riff(&chunk(b"VP8L", &[0x2e, 0, 0, 0, 0])),
            riff(&chunk(b"VP8L", &[0x2f, 0, 0, 0, 0x20])),
            // A VP8 inter frame.
            riff(&chunk(b"VP8 ", &[1, 0, 0, 0x9d, 1, 0x2a, 1, 0, 1, 0])),
            // An unknown first chunk.
            riff(&chunk(b"JUNK", &[])),
        ];
        for (i, file) in cases.iter().enumerate() {
            let error = parse(file).unwrap_err();
            assert_eq!(
                error.code(),
                otf_pixels_core::ErrorCode::Malformed,
                "case {i}"
            );
        }
    }
}
