//! Raw (uncompressed) pixel codec for `otf-pixels`.
//!
//! Raw is the degenerate format: there is no container, no header and no magic
//! bytes, so the caller supplies width, height, pixel format and stride
//! (SPEC §Formats). That makes it the simplest possible exercise of the
//! [`Decoder`]/[`Encoder`] contracts, and the format M1's round-trip test uses
//! at both ends of the pipeline.
//!
//! Because there is no header, "malformed" raw input means exactly one thing:
//! the stream is shorter than the declared dimensions require. That is
//! reported as a malformed-input error, never a panic.
//!
//! # Streaming
//!
//! Both directions are strictly row-at-a-time. [`RawDecoder`] reads exactly one
//! row per [`Decoder::read_row`] call and never buffers the image;
//! [`RawEncoder`] writes each row straight through to the sink. Raw is
//! therefore a true constant-memory format in both directions (SPEC §Formats).
//!
//! ```
//! use otf_pixels_codec_raw::{RawDecoder, RawFormat};
//! use otf_pixels_core::{Decoder, ImageDescriptor, PixelFormat};
//!
//! # fn main() -> Result<(), otf_pixels_core::PixelsError> {
//! let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Gray8)?;
//! let pixels: &[u8] = &[1, 2, 3, 4];
//!
//! let mut decoder = RawDecoder::new(RawFormat::packed(descriptor), pixels)?;
//! let mut row = vec![0_u8; descriptor.row_bytes()];
//! decoder.read_row(&mut row)?;
//! assert_eq!(row, [1, 2]);
//! # Ok(())
//! # }
//! ```

use otf_pixels_core::{
    Codec, DecodeCapability, Decoder, Encoder, Format, ImageDescriptor, Limits, PixelFormat,
    PixelsError, Result, Sink, Source,
};

/// The layout of a raw pixel stream.
///
/// Raw carries no self-description, so this is the contract the caller must
/// supply on both the decode and encode side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RawFormat {
    /// Dimensions and pixel format of the stream.
    pub descriptor: ImageDescriptor,
    /// Bytes between the starts of consecutive rows in the *stream*.
    ///
    /// At least `descriptor.row_bytes()`. Any excess is row padding, which the
    /// decoder reads and discards and the encoder writes as zeroes — this is
    /// how raw dumps from graphics APIs with aligned rows are consumed.
    pub stride: usize,
}

impl RawFormat {
    /// A densely packed layout: stride equals one row of pixels.
    #[must_use]
    pub const fn packed(descriptor: ImageDescriptor) -> Self {
        Self { descriptor, stride: descriptor.row_bytes() }
    }

    /// A layout with explicit row padding.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `stride` is shorter than one
    /// packed row.
    pub fn with_stride(descriptor: ImageDescriptor, stride: usize) -> Result<Self> {
        let row_bytes = descriptor.row_bytes();
        if stride < row_bytes {
            return Err(PixelsError::invalid_argument(
                "stride",
                format!("stride {stride} is shorter than a {row_bytes}-byte row"),
            ));
        }
        Ok(Self { descriptor, stride })
    }

    /// Describe a raw stream from its dimensions, checked against `limits`.
    ///
    /// This is the raw equivalent of a header parse: dimensions are validated
    /// **before** any pixel buffer is allocated (SPEC §Safety), so a caller
    /// forwarding untrusted dimensions cannot provoke a huge allocation.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::LimitExceeded`] if the dimensions exceed
    /// `limits`, or [`PixelsError::InvalidArgument`] if either is zero.
    pub fn from_dimensions(
        width: u32,
        height: u32,
        pixel: PixelFormat,
        limits: &Limits,
    ) -> Result<Self> {
        Ok(Self::packed(ImageDescriptor::with_limits(width, height, pixel, limits)?))
    }

    /// Bytes of padding after each row.
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.stride - self.descriptor.row_bytes()
    }
}

/// Format sniffing for raw streams.
///
/// Raw has no magic bytes, so [`RawCodec::probe`] always returns `false`: raw
/// can never be *detected*, only requested explicitly. Sniffing an unknown
/// stream as raw would mean treating arbitrary bytes as pixels of arbitrary
/// dimensions, which is not a decision the engine can make for the caller.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawCodec;

impl Codec for RawCodec {
    fn format(&self) -> Format {
        Format::Raw
    }

    fn magic_len(&self) -> usize {
        0
    }

    fn probe(&self, _prefix: &[u8]) -> bool {
        false
    }
}

/// Decodes a raw pixel stream, one row per call.
#[derive(Debug)]
pub struct RawDecoder<S: Source> {
    layout: RawFormat,
    source: S,
    rows_read: u32,
    /// Scratch for reading and discarding row padding.
    padding: Vec<u8>,
}

impl<S: Source> RawDecoder<S> {
    /// Wrap `source` as a decoder for a stream laid out as `layout`.
    ///
    /// No pixel bytes are read here: construction is the (empty) header parse,
    /// so laziness holds and the descriptor is available immediately.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if the layout's byte length
    /// overflows `usize` on this platform.
    pub fn new(layout: RawFormat, source: S) -> Result<Self> {
        if layout.descriptor.byte_len().is_none() {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "image byte length overflows this platform's address space",
            ));
        }
        Ok(Self { layout, source, rows_read: 0, padding: vec![0; layout.padding()] })
    }

    /// The layout this decoder was constructed with.
    #[must_use]
    pub const fn layout(&self) -> RawFormat {
        self.layout
    }

    /// How many rows have been decoded so far.
    #[must_use]
    pub const fn rows_read(&self) -> u32 {
        self.rows_read
    }
}

impl<S: Source + std::fmt::Debug> Decoder for RawDecoder<S> {
    fn descriptor(&self) -> ImageDescriptor {
        self.layout.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        // Raw is a forward-only byte stream. It could support random access
        // over a seekable source, but ADR-0005 makes forward-only the contract,
        // so region decode stays unavailable and M2 pulls rows in order.
        DecodeCapability::Sequential
    }

    fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
        let expected = self.layout.descriptor.row_bytes();
        if out.len() != expected {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("row buffer is {} bytes, expected {expected}", out.len()),
            ));
        }
        if self.rows_read >= self.layout.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "out",
                format!("all {} rows have already been read", self.layout.descriptor.height),
            ));
        }
        // A stream that ends mid-image surfaces as Malformed from read_exact.
        self.source.read_exact(out)?;
        if !self.padding.is_empty() {
            self.source.read_exact(&mut self.padding)?;
        }
        self.rows_read += 1;
        Ok(())
    }
}

/// Encodes rows of pixels as a raw stream.
#[derive(Debug, Default)]
pub struct RawEncoder {
    layout: Option<RawFormat>,
    rows_written: u32,
    /// Zero padding written after each row, when the layout calls for it.
    padding: Vec<u8>,
}

impl RawEncoder {
    /// A raw encoder that writes densely packed rows.
    #[must_use]
    pub const fn new() -> Self {
        Self { layout: None, rows_written: 0, padding: Vec::new() }
    }

    /// A raw encoder that writes rows padded to `layout`'s stride.
    #[must_use]
    pub fn with_layout(layout: RawFormat) -> Self {
        Self { layout: Some(layout), rows_written: 0, padding: vec![0; layout.padding()] }
    }

    /// How many rows have been written so far.
    #[must_use]
    pub const fn rows_written(&self) -> u32 {
        self.rows_written
    }
}

impl Encoder for RawEncoder {
    fn write_header(&mut self, desc: &ImageDescriptor, _sink: &mut dyn Sink) -> Result<()> {
        if self.rows_written > 0 {
            return Err(PixelsError::invalid_argument(
                "descriptor",
                "write_header called after rows were already written",
            ));
        }
        match self.layout {
            // A layout fixed at construction must match what the pipeline
            // actually produced, or the stride padding would be wrong.
            Some(layout) if layout.descriptor != *desc => {
                return Err(PixelsError::invalid_argument(
                    "descriptor",
                    format!(
                        "encoder was built for {} but the pipeline produced {desc}",
                        layout.descriptor
                    ),
                ));
            }
            Some(_) => {}
            None => {
                let layout = RawFormat::packed(*desc);
                self.padding = vec![0; layout.padding()];
                self.layout = Some(layout);
            }
        }
        // Raw has no header bytes; this call exists to fix the layout.
        Ok(())
    }

    fn write_row(&mut self, row: &[u8], sink: &mut dyn Sink) -> Result<()> {
        let Some(layout) = self.layout else {
            return Err(PixelsError::invalid_argument(
                "row",
                "write_row called before write_header",
            ));
        };
        let expected = layout.descriptor.row_bytes();
        if row.len() != expected {
            return Err(PixelsError::invalid_argument(
                "row",
                format!("row is {} bytes, expected {expected}", row.len()),
            ));
        }
        if self.rows_written >= layout.descriptor.height {
            return Err(PixelsError::invalid_argument(
                "row",
                format!("all {} declared rows have already been written", layout.descriptor.height),
            ));
        }
        sink.write_all(row)?;
        if !self.padding.is_empty() {
            sink.write_all(&self.padding)?;
        }
        self.rows_written += 1;
        Ok(())
    }

    fn finish(&mut self, sink: &mut dyn Sink) -> Result<()> {
        let Some(layout) = self.layout else {
            return Err(PixelsError::malformed("raw", "finish called before write_header"));
        };
        let declared = layout.descriptor.height;
        if self.rows_written != declared {
            // Partial output is never silently accepted.
            return Err(PixelsError::malformed(
                "raw",
                format!("wrote {} of {declared} declared rows", self.rows_written),
            ));
        }
        sink.flush()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use otf_pixels_core::{ErrorCode, Limit};

    fn descriptor(width: u32, height: u32) -> ImageDescriptor {
        ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap()
    }

    fn decode_all(layout: RawFormat, bytes: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = RawDecoder::new(layout, bytes)?;
        let mut out = Vec::new();
        let mut row = vec![0_u8; layout.descriptor.row_bytes()];
        for _ in 0..layout.descriptor.height {
            decoder.read_row(&mut row)?;
            out.extend_from_slice(&row);
        }
        Ok(out)
    }

    #[test]
    fn packed_streams_round_trip() {
        let desc = descriptor(2, 2);
        let pixels = [1, 2, 3, 4];
        let decoded = decode_all(RawFormat::packed(desc), &pixels).unwrap();
        assert_eq!(decoded, pixels);

        let mut sink = Vec::new();
        let mut encoder = RawEncoder::new();
        encoder.write_header(&desc, &mut sink).unwrap();
        encoder.write_row(&[1, 2], &mut sink).unwrap();
        encoder.write_row(&[3, 4], &mut sink).unwrap();
        encoder.finish(&mut sink).unwrap();
        assert_eq!(sink, pixels);
    }

    #[test]
    fn stride_padding_is_skipped_on_decode_and_zeroed_on_encode() {
        let desc = descriptor(2, 2);
        let layout = RawFormat::with_stride(desc, 4).unwrap();
        assert_eq!(layout.padding(), 2);
        // Rows are `1,2` and `3,4`, each followed by two padding bytes.
        let stream = [1, 2, 9, 9, 3, 4, 9, 9];
        assert_eq!(decode_all(layout, &stream).unwrap(), [1, 2, 3, 4]);

        let mut sink = Vec::new();
        let mut encoder = RawEncoder::with_layout(layout);
        encoder.write_header(&desc, &mut sink).unwrap();
        encoder.write_row(&[1, 2], &mut sink).unwrap();
        encoder.write_row(&[3, 4], &mut sink).unwrap();
        encoder.finish(&mut sink).unwrap();
        assert_eq!(sink, [1, 2, 0, 0, 3, 4, 0, 0]);
    }

    #[test]
    fn a_truncated_stream_is_malformed_not_a_panic() {
        let layout = RawFormat::packed(descriptor(4, 4));
        // Every truncation length, including mid-row and empty.
        for len in 0..16 {
            let stream = vec![7_u8; len];
            let err = decode_all(layout, &stream).unwrap_err();
            assert_eq!(err.code(), ErrorCode::Malformed, "truncated at {len} bytes");
        }
        assert!(decode_all(layout, &[7_u8; 16]).is_ok());
    }

    #[test]
    fn a_stream_truncated_inside_padding_is_malformed() {
        let layout = RawFormat::with_stride(descriptor(2, 2), 4).unwrap();
        // Full first row, but the stream ends inside that row's padding.
        let err = decode_all(layout, &[1, 2, 9]).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
    }

    #[test]
    fn reading_past_the_end_is_an_error_not_a_panic() {
        let layout = RawFormat::packed(descriptor(2, 1));
        let stream: &[u8] = &[1, 2];
        let mut decoder = RawDecoder::new(layout, stream).unwrap();
        let mut row = vec![0_u8; 2];
        decoder.read_row(&mut row).unwrap();
        let err = decoder.read_row(&mut row).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
        assert_eq!(decoder.rows_read(), 1);
    }

    #[test]
    fn a_wrong_sized_row_buffer_is_rejected() {
        let layout = RawFormat::packed(descriptor(4, 1));
        let stream: &[u8] = &[1, 2, 3, 4];
        let mut decoder = RawDecoder::new(layout, stream).unwrap();
        assert_eq!(decoder.read_row(&mut [0; 3]).unwrap_err().code(), ErrorCode::InvalidArgument);
        assert_eq!(decoder.read_row(&mut [0; 5]).unwrap_err().code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn hostile_dimensions_are_rejected_before_allocation() {
        // A caller forwarding untrusted dimensions must not be able to provoke
        // a huge allocation: the limit check happens at layout construction.
        let err =
            RawFormat::from_dimensions(u32::MAX, u32::MAX, PixelFormat::Rgba8, &Limits::default())
                .unwrap_err();
        assert_eq!(err.code(), ErrorCode::LimitExceeded);
        match err {
            PixelsError::LimitExceeded { limit, .. } => assert_eq!(limit, Limit::MaxPixels),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn a_short_stride_is_rejected() {
        let err = RawFormat::with_stride(descriptor(4, 4), 3).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
        assert_eq!(RawFormat::with_stride(descriptor(4, 4), 4).unwrap().padding(), 0);
    }

    #[test]
    fn finishing_early_never_yields_partial_output() {
        let desc = descriptor(2, 3);
        let mut sink = Vec::new();
        let mut encoder = RawEncoder::new();
        encoder.write_header(&desc, &mut sink).unwrap();
        encoder.write_row(&[1, 2], &mut sink).unwrap();
        let err = encoder.finish(&mut sink).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
        assert!(err.to_string().contains("1 of 3"), "{err}");
    }

    #[test]
    fn writing_more_rows_than_declared_is_rejected() {
        let desc = descriptor(2, 1);
        let mut sink = Vec::new();
        let mut encoder = RawEncoder::new();
        encoder.write_header(&desc, &mut sink).unwrap();
        encoder.write_row(&[1, 2], &mut sink).unwrap();
        let err = encoder.write_row(&[3, 4], &mut sink).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn encoding_out_of_order_is_rejected() {
        let mut sink = Vec::new();
        let mut encoder = RawEncoder::new();
        // write_row before write_header.
        assert_eq!(
            encoder.write_row(&[1, 2], &mut sink).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        // finish before write_header.
        assert_eq!(encoder.finish(&mut sink).unwrap_err().code(), ErrorCode::Malformed);
    }

    #[test]
    fn a_fixed_layout_encoder_rejects_a_mismatched_pipeline() {
        let layout = RawFormat::with_stride(descriptor(2, 2), 4).unwrap();
        let mut encoder = RawEncoder::with_layout(layout);
        let mut sink = Vec::new();
        let err = encoder.write_header(&descriptor(3, 2), &mut sink).unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn every_v1_pixel_format_round_trips() {
        for &pixel in PixelFormat::ALL {
            let desc = ImageDescriptor::new(3, 2, pixel).unwrap();
            let bytes: Vec<u8> = (0..desc.byte_len().unwrap()).map(|i| (i % 251) as u8).collect();
            let decoded = decode_all(RawFormat::packed(desc), &bytes).unwrap();
            assert_eq!(decoded, bytes, "{pixel} did not round-trip");
        }
    }

    #[test]
    fn raw_never_claims_a_stream_by_sniffing() {
        let codec = RawCodec;
        assert_eq!(codec.format(), Format::Raw);
        assert_eq!(codec.magic_len(), 0);
        assert!(!codec.probe(&[]));
        assert!(!codec.probe(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn decoding_reads_exactly_one_row_at_a_time() {
        // Proves the decoder streams: after one read_row, only that row's bytes
        // have been consumed from the source.
        let desc = descriptor(2, 4);
        let stream: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8];
        let mut cursor = std::io::Cursor::new(stream);
        let mut decoder = RawDecoder::new(RawFormat::packed(desc), &mut cursor).unwrap();
        let mut row = vec![0_u8; 2];
        decoder.read_row(&mut row).unwrap();
        assert_eq!(row, [1, 2]);
        drop(decoder);
        assert_eq!(cursor.position(), 2, "only one row was consumed");
    }
}
