//! The JPEG XL file: a bare codestream (`FF 0A`), or an ISO-BMFF container
//! (ISO/IEC 18181-2) whose `jxlc` box, or `jxlp` boxes in order, hold it,
//! beside metadata boxes such as `Exif`.

#![allow(
    clippy::indexing_slicing,
    reason = "fixed-size headers and tables, indexed after their lengths are checked"
)]

use otf_pixels_core::Result;

use super::reader::malformed;

/// The container's signature box.
pub const CONTAINER_SIGNATURE: [u8; 12] =
    [0, 0, 0, 12, b'J', b'X', b'L', b' ', 0x0D, 0x0A, 0x87, 0x0A];
/// A bare codestream's first two bytes.
pub const CODESTREAM_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// What a file holds.
#[derive(Debug, Default)]
pub struct Contents {
    pub codestream: Vec<u8>,
    /// The `Exif` box's TIFF-structured block, without its offset.
    pub exif: Option<Vec<u8>>,
}

/// Whether `prefix` starts a JPEG XL file.
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    prefix.starts_with(&CODESTREAM_SIGNATURE) || prefix.starts_with(&CONTAINER_SIGNATURE)
}

/// Split `file` into its codestream and metadata.
///
/// # Errors
///
/// For a file that is neither form, or a malformed box.
pub fn parse(file: &[u8]) -> Result<Contents> {
    if file.starts_with(&CODESTREAM_SIGNATURE) {
        return Ok(Contents {
            codestream: file.to_vec(),
            exif: None,
        });
    }
    if !file.starts_with(&CONTAINER_SIGNATURE) {
        return Err(malformed("not a JPEG XL file"));
    }
    let mut out = Contents::default();
    let mut at = 0;
    while at < file.len() {
        let header = file
            .get(at..at + 8)
            .ok_or_else(|| malformed("a box header is cut short"))?;
        let mut size = u64::from(u32::from_be_bytes([
            header[0], header[1], header[2], header[3],
        ]));
        let kind = [header[4], header[5], header[6], header[7]];
        let mut body = at + 8;
        if size == 1 {
            let large = file
                .get(body..body + 8)
                .ok_or_else(|| malformed("a box size is cut short"))?;
            size = u64::from_be_bytes(large.try_into().map_err(|_| malformed("box size"))?);
            body += 8;
        }
        let end = if size == 0 {
            file.len()
        } else {
            usize::try_from(size)
                .ok()
                .and_then(|s| at.checked_add(s))
                .filter(|&e| e >= body && e <= file.len())
                .ok_or_else(|| malformed("a box runs past the file"))?
        };
        let payload = &file[body..end];
        match &kind {
            b"jxlc" => out.codestream.extend_from_slice(payload),
            b"jxlp" => {
                out.codestream
                    .extend_from_slice(payload.get(4..).unwrap_or(&[]));
            }
            b"Exif" if out.exif.is_none() => {
                // A 32-bit offset to the TIFF header, then the block.
                if let Some(offset) = payload.get(..4) {
                    let skip =
                        u32::from_be_bytes([offset[0], offset[1], offset[2], offset[3]]) as usize;
                    out.exif = payload.get(skip.saturating_add(4)..).map(<[u8]>::to_vec);
                }
            }
            _ => {}
        }
        at = end;
    }
    if out.codestream.is_empty() {
        return Err(malformed("the container holds no codestream"));
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    fn bx(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn a_container_yields_its_codestream_and_exif() {
        let mut file = CONTAINER_SIGNATURE.to_vec();
        file.extend(bx(b"ftyp", b"jxl \0\0\0\0jxl "));
        file.extend(bx(b"Exif", &[0, 0, 0, 0, b'I', b'I', 42, 0]));
        file.extend(bx(b"jxlp", &[0, 0, 0, 0, 0xFF, 0x0A]));
        file.extend(bx(b"jxlp", &[0x80, 0, 0, 1, 1, 2]));
        let contents = parse(&file).unwrap();
        assert_eq!(contents.codestream, [0xFF, 0x0A, 1, 2]);
        assert_eq!(contents.exif.unwrap(), b"II\x2a\0");
        assert!(probe(&file));
        assert!(parse(&file[..20]).is_err());
        assert!(parse(b"not jxl").is_err());
    }
}
