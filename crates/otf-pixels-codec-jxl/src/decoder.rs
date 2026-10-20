//! The JPEG XL decoder: a bare codestream or the ISOBMFF container, decoded
//! to its first displayed frame.

#![allow(
    clippy::indexing_slicing,
    reason = "planes are the canvas size, so every pixel index is in range"
)]

use otf_pixels_core::{
    Animation, Codec, Decoder, Format, ImageDescriptor, Limits, Orientation, PixelFormat,
    PixelsError, Result, Source, SourceInfo,
};

use crate::decode::{container, header::ExtraKind, render};

/// The most compressed bytes read before a file is called hostile.
const MAX_COMPRESSED: usize = 256 * 1024 * 1024;

/// Decodes a JPEG XL stream.
#[derive(Debug)]
pub struct JxlDecoder {
    descriptor: ImageDescriptor,
    /// The codestream, until the first row is asked for.
    codestream: Vec<u8>,
    /// The decoded image, interleaved in the output format.
    pixels: Option<Vec<u8>>,
    row: u32,
    orientation: Orientation,
    icc: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    info: SourceInfo,
    animation: Option<Animation>,
}

impl JxlDecoder {
    /// Read the file and parse its headers; the pixels decode when the first
    /// row is read.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for headers it rejects, or
    /// [`PixelsError::LimitExceeded`] if the image exceeds `limits`.
    pub fn new<S: Source>(mut source: S, limits: Limits) -> Result<Self> {
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 64 * 1024];
        loop {
            if bytes.len() > MAX_COMPRESSED {
                return Err(PixelsError::malformed(
                    "jxl",
                    format!("stream exceeds {MAX_COMPRESSED} bytes"),
                ));
            }
            match source.read(&mut chunk)? {
                0 => break,
                read => bytes.extend_from_slice(chunk.get(..read).unwrap_or(&[])),
            }
        }
        let contents = container::parse(&bytes)?;
        let exif = contents.exif;
        let xmp = contents.xmp;
        let codestream = contents.codestream;
        // Limits first: the size is the attacker's to state, and walking the
        // frame headers sizes their tables of contents by it.
        let header = crate::decode::header::ImageHeader::parse(
            &mut crate::decode::reader::BitReader::new(&codestream),
        )?;
        ImageDescriptor::with_limits(header.width, header.height, PixelFormat::Rgba8, &limits)?;
        let (header, icc, durations) = render::probe(&codestream)?;
        let (width, height) = (header.width, header.height);
        let alpha = header.alpha().is_some();
        let wide = header.bit_depth.bits > 8;
        let pixel = match (header.grey(), alpha, wide, header.bit_depth.float) {
            (_, false, _, true) => PixelFormat::RgbF32,
            (_, true, _, true) => PixelFormat::RgbaF32,
            (true, false, false, _) => PixelFormat::Gray8,
            (true, false, true, _) => PixelFormat::Gray16,
            (true, true, false, _) => PixelFormat::GrayA8,
            (false, false, false, _) => PixelFormat::Rgb8,
            (false, false, true, _) => PixelFormat::Rgb16,
            (false, true, false, _) => PixelFormat::Rgba8,
            (_, true, true, _) => PixelFormat::Rgba16,
        };
        let descriptor = ImageDescriptor::with_limits(width, height, pixel, &limits)?;
        let animation = header.animation.and_then(|a| {
            if durations.len() < 2 || a.tps_numerator == 0 {
                return None;
            }
            let ms = durations
                .iter()
                .map(|&t| {
                    (u64::from(t) * 1000 * u64::from(a.tps_denominator)
                        / u64::from(a.tps_numerator)) as u32
                })
                .collect();
            Animation::new(ms, a.num_loops)
        });
        // The codestream's own orientation; an EXIF block repeats it.
        let orientation = Orientation::from_exif(header.orientation as u16).unwrap_or_default();
        // XYB images are delivered as sRGB, so their profile does not apply.
        let icc = if header.xyb { None } else { icc };
        let mut info = SourceInfo::default();
        info.bits_per_sample = u8::try_from(header.bit_depth.bits).ok();
        // XYB is how libjxl codes lossy images; lossless keeps the colour
        // space it was given.
        info.compression = Some(if header.xyb { "lossy" } else { "lossless" });
        Ok(Self {
            descriptor,
            codestream,
            pixels: None,
            row: 0,
            orientation,
            icc,
            exif,
            xmp,
            info,
            animation,
        })
    }

    fn decode_pixels(&self) -> Result<Vec<u8>> {
        let image = render::decode(&self.codestream)?;
        let canvas = &image.canvas;
        let alpha = image
            .header
            .extra
            .iter()
            .position(|e| matches!(e.kind, ExtraKind::Alpha { .. }))
            .and_then(|i| canvas.extra.get(i));
        let n = canvas.width * canvas.height;
        let pixel = self.descriptor.pixel;
        let channels: Vec<&[f32]> = if pixel.channels() <= 2 {
            vec![&canvas.colour[0][..n]]
        } else {
            canvas.colour.iter().map(|p| &p[..n]).collect()
        };
        let mut planes = channels;
        if pixel.has_alpha() {
            planes.push(alpha.map_or(&[][..], |a| &a[..n]));
        }
        let mut out = Vec::with_capacity(self.descriptor.row_bytes() * canvas.height);
        for i in 0..n {
            for p in &planes {
                let v = p.get(i).copied().unwrap_or(1.0);
                match pixel {
                    PixelFormat::RgbF32 | PixelFormat::RgbaF32 => {
                        out.extend_from_slice(&v.to_ne_bytes())
                    }
                    PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                        out.extend_from_slice(
                            &((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_ne_bytes(),
                        );
                    }
                    _ => out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8),
                }
            }
        }
        Ok(out)
    }
}

impl Decoder for JxlDecoder {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn orientation(&self) -> Orientation {
        self.orientation
    }

    fn icc_profile(&self) -> Option<&[u8]> {
        self.icc.as_deref()
    }

    fn exif(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }

    /// The container's `xml ` box.
    fn xmp(&self) -> Option<&[u8]> {
        self.xmp.as_deref()
    }

    /// Bit depth, and lossy (XYB) or lossless.
    fn source_info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn animation(&self) -> Option<Animation> {
        self.animation.clone()
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        if self.row >= self.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("all {} rows have already been read", self.descriptor.height),
            ));
        }
        let row_bytes = self.descriptor.row_bytes();
        if out.len() != row_bytes {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {row_bytes}", out.len()),
            ));
        }
        if self.pixels.is_none() {
            self.pixels = Some(self.decode_pixels()?);
            self.codestream = Vec::new();
        }
        let start = self.row as usize * row_bytes;
        let row = self
            .pixels
            .as_deref()
            .and_then(|p| p.get(start..start + row_bytes))
            .ok_or_else(|| PixelsError::malformed("jxl", "decoded image is short"))?;
        out.copy_from_slice(row);
        self.row += 1;
        Ok(())
    }
}

/// Whether `prefix` starts with a JPEG XL signature, bare or contained.
#[must_use]
pub fn probe(prefix: &[u8]) -> bool {
    container::probe(prefix)
}

/// The JPEG XL entry in a sniffing registry.
#[derive(Debug, Clone, Copy, Default)]
pub struct JxlCodec;

impl Codec for JxlCodec {
    fn format(&self) -> Format {
        Format::Jxl
    }

    fn magic_len(&self) -> usize {
        12
    }

    fn probe(&self, prefix: &[u8]) -> bool {
        probe(prefix)
    }
}
