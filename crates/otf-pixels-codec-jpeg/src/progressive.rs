//! Progressive JPEG, wrapped rather than owned.
//!
//! ADR-0004 splits the format table into codecs we implement and codecs we
//! wrap, and puts progressive JPEG on the wrapped side. The marker differs
//! from baseline's by one byte, but the decoder differs by much more: the
//! coefficients arrive spread across many scans, each refining a band of
//! frequencies or a slice of bit depth, so nothing is final until the last
//! scan. That rules out the one-MCU-row streaming the baseline decoder is
//! built around — it is a different program, not a variation.
//!
//! # Memory
//!
//! Internally buffered, as SPEC §Formats says. The whole image is decoded up
//! front and rows are served from it, because a progressive file has no
//! prefix that yields a finished row. This is the documented exception to the
//! constant-memory guarantee, not a lapse in it.
//!
//! # What is wrapped
//!
//! `jpeg-decoder`, with default features off so it brings no transitive
//! dependencies of its own. It is pure Rust, which is what ADR-0004 asks for.
//! The trait boundary means replacing it later — with our own progressive
//! decoder, or another crate — changes nothing above this module.

use otf_pixels_core::{
    DecodeCapability, Decoder, ImageDescriptor, Limits, PixelFormat, PixelsError, Result, Source,
};

/// The most compressed bytes read before a file is called hostile.
///
/// A progressive decode has to hold the whole stream, so unlike the baseline
/// path there is no natural bound from the image dimensions: a small frame
/// header can be followed by unlimited scan data. `max_pixels` bounds the
/// output; this bounds the input.
const MAX_COMPRESSED: usize = 256 * 1024 * 1024;

/// A progressive JPEG, decoded whole and served by rows.
#[derive(Debug)]
pub struct Progressive {
    descriptor: ImageDescriptor,
    /// The decoded image, interleaved.
    pixels: Vec<u8>,
    /// Rows already served.
    row: u32,
    /// Read from the header before the handover, so a progressive photograph
    /// reports its orientation like a baseline one does.
    orientation: Option<u8>,
}

impl Progressive {
    /// Decode a progressive JPEG from `replay` followed by `source`.
    ///
    /// `replay` is the header the caller already consumed while discovering
    /// the file was progressive; the wrapped decoder needs the stream from
    /// byte zero.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] for a stream the wrapped decoder
    /// rejects, [`PixelsError::Unsupported`] for a JPEG feature it does not
    /// implement or a pixel format v1 has no equivalent for (CMYK), or
    /// [`PixelsError::LimitExceeded`] if the frame exceeds `limits` or the
    /// stream exceeds the compressed-size bound.
    pub fn new<S: Source>(
        replay: Vec<u8>,
        mut source: S,
        limits: Limits,
        orientation: Option<u8>,
    ) -> Result<Self> {
        let mut bytes = replay;
        let mut chunk = [0_u8; 64 * 1024];
        loop {
            if bytes.len() > MAX_COMPRESSED {
                return Err(PixelsError::malformed(
                    "jpeg",
                    format!("progressive stream exceeds {MAX_COMPRESSED} bytes"),
                ));
            }
            match source.read(&mut chunk)? {
                0 => break,
                read => {
                    let Some(filled) = chunk.get(..read) else {
                        break;
                    };
                    bytes.extend_from_slice(filled);
                }
            }
        }

        let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
        decoder.read_info().map_err(wrapped_error)?;
        let info = decoder
            .info()
            .ok_or_else(|| PixelsError::malformed("jpeg", "no frame header in the stream"))?;

        let pixel = match info.pixel_format {
            jpeg_decoder::PixelFormat::L8 => PixelFormat::Gray8,
            jpeg_decoder::PixelFormat::L16 => PixelFormat::Gray16,
            jpeg_decoder::PixelFormat::RGB24 => PixelFormat::Rgb8,
            // Same refusal the baseline path makes: v1 is sRGB-assumed, and
            // an ink model guessed without a profile is confidently wrong
            // colour (SPEC §Pixel formats).
            jpeg_decoder::PixelFormat::CMYK32 => {
                return Err(PixelsError::unsupported(
                    "jpeg: CMYK/YCCK images are not supported",
                ));
            }
        };
        // Enforced before the pixel buffer is allocated, as on the baseline
        // path (SPEC §Safety and limits).
        let descriptor = ImageDescriptor::with_limits(
            u32::from(info.width),
            u32::from(info.height),
            pixel,
            &limits,
        )?;

        let pixels = decoder.decode().map_err(wrapped_error)?;
        let expected = descriptor
            .byte_len()
            .ok_or_else(|| PixelsError::malformed("jpeg", "image size overflows"))?;
        if pixels.len() != expected {
            return Err(PixelsError::malformed(
                "jpeg",
                format!(
                    "decoded {} bytes for a {}x{} image expecting {expected}",
                    pixels.len(),
                    descriptor.width,
                    descriptor.height
                ),
            ));
        }

        Ok(Self {
            descriptor,
            pixels,
            row: 0,
            orientation,
        })
    }

    /// The EXIF orientation tag carried over from the header.
    #[must_use]
    pub const fn orientation(&self) -> Option<u8> {
        self.orientation
    }
}

/// Translate the wrapped decoder's failure into this crate's error type.
///
/// The mapping matters more than it looks: a caller routes on
/// [`PixelsError::Unsupported`] versus [`PixelsError::Malformed`], and
/// collapsing both into "broken image" would send someone hunting for a
/// corrupt file that is merely using a feature we do not decode.
fn wrapped_error(error: jpeg_decoder::Error) -> PixelsError {
    match error {
        jpeg_decoder::Error::Format(detail) => PixelsError::malformed("jpeg", detail),
        jpeg_decoder::Error::Unsupported(feature) => {
            PixelsError::unsupported(format!("jpeg: {feature:?}"))
        }
        jpeg_decoder::Error::Io(error) => PixelsError::io("decoding a progressive JPEG", error),
        other => PixelsError::malformed("jpeg", other.to_string()),
    }
}

impl Decoder for Progressive {
    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        // The image is already in memory, so any row is equally cheap — but
        // `Sequential` is what the row contract describes, and claiming
        // `Regions` would promise a `read_region` this does not implement.
        DecodeCapability::Sequential
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
        let start = self.row as usize * row_bytes;
        let row = self
            .pixels
            .get(start..)
            .and_then(|rest| rest.get(..row_bytes))
            .ok_or_else(|| PixelsError::malformed("jpeg", "decoded image is short"))?;
        out.copy_from_slice(row);
        self.row += 1;
        Ok(())
    }
}
