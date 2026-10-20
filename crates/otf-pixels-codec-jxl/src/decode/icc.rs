//! The embedded ICC profile (ISO/IEC 18181-1 §E.4): an entropy-coded byte
//! stream, then a small command language that rebuilds the profile from
//! predicted header bytes, common tags and shuffled numeric data. Ported
//! from jxl-oxide (MIT/Apache-2.0); see NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "offsets are checked against the output before use"
)]

use otf_pixels_core::Result;

use super::coding::Decoder;
use super::reader::{BitReader, malformed};

fn bad(detail: &str) -> otf_pixels_core::PixelsError {
    malformed(format!("ICC profile: {detail}"))
}

/// Read the encoded profile stream.
///
/// # Errors
///
/// For a malformed stream.
pub fn read(r: &mut BitReader<'_>) -> Result<Vec<u8>> {
    let size = r.u64()?;
    if size > 1 << 28 {
        return Err(bad("too large"));
    }
    let mut d = Decoder::parse(r, 41)?;
    d.begin(r)?;
    let mut out = Vec::with_capacity(size as usize);
    let (mut b1, mut b2) = (0_u8, 0_u8);
    for i in 0..size as usize {
        let v = d.read(r, context(i, b1, b2))?;
        let b = u8::try_from(v).map_err(|_| bad("a byte out of range"))?;
        out.push(b);
        b2 = b1;
        b1 = b;
    }
    d.finish()?;
    Ok(out)
}

fn context(i: usize, b1: u8, b2: u8) -> u32 {
    if i <= 128 {
        return 0;
    }
    let p1 = match b1 {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=1 => 2 + u32::from(b1),
        2..=15 => 4,
        241..=254 => 5,
        255 => 6,
        _ => 7,
    };
    let p2 = match b2 {
        b'a'..=b'z' | b'A'..=b'Z' => 0,
        b'0'..=b'9' | b'.' | b',' => 1,
        0..=15 => 2,
        241..=255 => 3,
        _ => 4,
    };
    1 + p1 + 8 * p2
}

/// A byte reader over a slice.
struct Bytes<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bytes<'_> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.data.get(self.at)?;
        self.at += 1;
        Some(b)
    }

    fn varint(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        let mut shift = 0;
        while shift < 63 {
            let b = self.byte().ok_or_else(|| bad("cut short"))?;
            value |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(value)
    }

    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.data.len()).ok_or_else(|| bad("cut short"))?;
        let out = &self.data[self.at..end];
        self.at = end;
        Ok(out)
    }
}

fn predict_header(i: usize, size: u32, header: &[u8]) -> u8 {
    let at = |j: usize| header.get(j).copied().unwrap_or(0);
    match i {
        0..=3 => size.to_be_bytes()[i],
        8 => 4,
        12..=23 => b"mntrRGB XYZ "[i - 12],
        36..=39 => b"acsp"[i - 36],
        41 | 42 if at(40) == b'A' => b'P',
        43 if at(40) == b'A' => b'L',
        41 if at(40) == b'M' => b'S',
        42 if at(40) == b'M' => b'F',
        43 if at(40) == b'M' => b'T',
        42 if at(40) == b'S' && at(41) == b'G' => b'I',
        43 if at(40) == b'S' && at(41) == b'G' => b' ',
        42 if at(40) == b'S' && at(41) == b'U' => b'N',
        43 if at(40) == b'S' && at(41) == b'U' => b'W',
        70 => 246,
        71 => 214,
        73 => 1,
        78 => 211,
        79 => 45,
        80..=83 => at(4 + i - 80),
        _ => 0,
    }
}

fn shuffle2(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let (height, odd) = (len / 2, len % 2);
    let mut out = Vec::with_capacity(len);
    for i in 0..height {
        out.push(bytes[i]);
        out.push(bytes[i + height + odd]);
    }
    if odd != 0 {
        out.push(bytes[height]);
    }
    out
}

fn shuffle4(bytes: &[u8]) -> Vec<u8> {
    let len = bytes.len();
    let (step, wide) = (len / 4, len % 4);
    let mut out = Vec::with_capacity(len);
    for i in 0..step {
        let mut base = i;
        for _ in 0..wide {
            out.push(bytes[base]);
            base += step + 1;
        }
        for _ in wide..4 {
            out.push(bytes[base]);
            base += step;
        }
    }
    for i in 1..=wide {
        out.push(bytes[(step + 1) * i - 1]);
    }
    out
}

/// Rebuild the profile from its encoded stream.
///
/// # Errors
///
/// For a malformed stream.
#[allow(clippy::too_many_lines, reason = "one command after another")]
pub fn decode(stream: &[u8]) -> Result<Vec<u8>> {
    const TAGS: [&[u8; 4]; 19] = [
        b"rTRC", b"rXYZ", b"cprt", b"wtpt", b"bkpt", b"rXYZ", b"gXYZ", b"bXYZ", b"kXYZ", b"rTRC", b"gTRC",
        b"bTRC", b"kTRC", b"chad", b"desc", b"chrm", b"dmnd", b"dmdd", b"lumi",
    ];
    const DATA: [&[u8; 4]; 8] = [b"XYZ ", b"desc", b"text", b"mluc", b"para", b"curv", b"sf32", b"gbd "];
    let mut head = Bytes { data: stream, at: 0 };
    let size = head.varint()?;
    let commands_size = head.varint()? as usize;
    if size > 1 << 28 {
        return Err(bad("too large"));
    }
    let rest = &stream[head.at..];
    if commands_size > rest.len() {
        return Err(bad("bad command size"));
    }
    let (commands, data) = rest.split_at(commands_size);
    let header_size = (size as usize).min(128);
    if data.len() < header_size {
        return Err(bad("bad size"));
    }
    let (header, data) = data.split_at(header_size);
    let mut data = Bytes { data, at: 0 };
    let mut cmds = Bytes { data: commands, at: 0 };
    let mut out = Vec::with_capacity(size as usize);
    for (i, &e) in header.iter().enumerate() {
        out.push(predict_header(i, size as u32, header).wrapping_add(e));
    }
    if size <= 128 {
        return Ok(out);
    }
    let v = cmds.varint()?;
    if let Some(num_tags) = v.checked_sub(1) {
        if (size - 128) / 12 < num_tags {
            return Err(bad("too many tags"));
        }
        let num_tags = num_tags as u32;
        out.extend_from_slice(&num_tags.to_be_bytes());
        let mut prev_start = num_tags * 12 + 128;
        let mut prev_size = 0_u32;
        loop {
            let Some(command) = cmds.byte() else {
                return Ok(out);
            };
            let code = command & 63;
            let tag: [u8; 4] = match code {
                0 => break,
                1 => data.take(4)?.try_into().map_err(|_| bad("tag"))?,
                2..=20 => *TAGS[(code - 2) as usize],
                _ => return Err(bad("bad tag code")),
            };
            let start = if command & 64 == 0 {
                prev_start + prev_size
            } else {
                cmds.varint()? as u32
            };
            let tag_size = if command & 128 != 0 {
                cmds.varint()? as u32
            } else if matches!(&tag, b"rXYZ" | b"gXYZ" | b"bXYZ" | b"kXYZ" | b"wtpt" | b"bkpt" | b"lumi") {
                20
            } else {
                prev_size
            };
            if u64::from(start) + u64::from(tag_size) > size {
                return Err(bad("a tag runs past the profile"));
            }
            prev_start = start;
            prev_size = tag_size;
            out.extend_from_slice(&tag);
            out.extend_from_slice(&start.to_be_bytes());
            out.extend_from_slice(&tag_size.to_be_bytes());
            if code == 2 {
                for t in [b"gTRC", b"bTRC"] {
                    out.extend_from_slice(t);
                    out.extend_from_slice(&start.to_be_bytes());
                    out.extend_from_slice(&tag_size.to_be_bytes());
                }
            } else if code == 3 {
                for (k, t) in [b"gXYZ", b"bXYZ"].into_iter().enumerate() {
                    out.extend_from_slice(t);
                    out.extend_from_slice(&(start + tag_size * (k as u32 + 1)).to_be_bytes());
                    out.extend_from_slice(&tag_size.to_be_bytes());
                }
            }
        }
    }
    while let Some(command) = cmds.byte() {
        match command {
            1 => {
                let n = cmds.varint()? as usize;
                out.extend_from_slice(data.take(n)?);
            }
            2 | 3 => {
                let n = cmds.varint()? as usize;
                let bytes = data.take(n)?;
                out.extend(if command == 2 { shuffle2(bytes) } else { shuffle4(bytes) });
            }
            4 => {
                let flags = cmds.byte().ok_or_else(|| bad("cut short"))?;
                let width = usize::from(flags & 3) + 1;
                let order = usize::from((flags >> 2) & 3);
                if width == 3 || order == 3 {
                    return Err(bad("bad predictor"));
                }
                let stride = if flags & 16 == 0 {
                    width
                } else {
                    let s = cmds.varint()? as usize;
                    if s < width {
                        return Err(bad("bad stride"));
                    }
                    s
                };
                if stride.saturating_mul(4) >= out.len() {
                    return Err(bad("stride past the output"));
                }
                let n = cmds.varint()? as usize;
                let raw = data.take(n)?;
                let bytes = match width {
                    2 => shuffle2(raw),
                    4 => shuffle4(raw),
                    _ => raw.to_vec(),
                };
                for i in (0..n).step_by(width) {
                    let mut prev = [0_u32; 3];
                    for (j, p) in prev.iter_mut().take(order + 1).enumerate() {
                        let offset = out.len() - stride * (j + 1);
                        let mut b = [0_u8; 4];
                        b[4 - width..].copy_from_slice(&out[offset..offset + width]);
                        *p = u32::from_be_bytes(b);
                    }
                    let p = match order {
                        0 => prev[0],
                        1 => prev[0].wrapping_mul(2).wrapping_sub(prev[1]),
                        _ => prev[0].wrapping_sub(prev[1]).wrapping_mul(3).wrapping_add(prev[2]),
                    };
                    for j in 0..width.min(n - i) {
                        out.push((u32::from(bytes[i + j]).wrapping_add(p >> (8 * (width - 1 - j)))) as u8);
                    }
                }
            }
            10 => {
                out.extend_from_slice(b"XYZ \0\0\0\0");
                out.extend_from_slice(data.take(12)?);
            }
            16..=23 => {
                out.extend_from_slice(DATA[usize::from(command - 16)]);
                out.extend_from_slice(&[0; 4]);
            }
            _ => return Err(bad("bad command")),
        }
    }
    if out.len() != size as usize {
        return Err(bad("the rebuilt profile is the wrong size"));
    }
    Ok(out)
}
