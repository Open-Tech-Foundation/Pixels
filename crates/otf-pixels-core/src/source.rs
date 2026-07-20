//! Producers that root a graph in actual pixels.
//!
//! [`BufferSource`] wraps pixels the caller already holds. [`DecodedSource`]
//! bridges a streaming [`Decoder`] into the graph, deferring all pixel work
//! until a terminal pulls.

use crate::{
    Decoder, ImageDescriptor, PixelsError, Producer, Region, Result, TileBuf, TileMut,
    copy_region,
};
use std::sync::{Arc, Mutex};

/// A producer over pixels already in memory.
///
/// The buffer is shared, not copied, so cloning is cheap and two graphs can
/// root at the same pixels.
#[derive(Debug, Clone)]
pub struct BufferSource {
    buffer: Arc<TileBuf>,
    descriptor: ImageDescriptor,
}

impl BufferSource {
    /// Wrap a whole-image buffer as a producer.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `buffer` does not start at
    /// the image origin or does not match `descriptor`.
    pub fn new(descriptor: ImageDescriptor, buffer: Arc<TileBuf>) -> Result<Self> {
        if buffer.region() != descriptor.region() {
            return Err(PixelsError::invalid_argument(
                "buffer",
                format!(
                    "buffer covers {} but the image is {}",
                    buffer.region(),
                    descriptor.region()
                ),
            ));
        }
        if buffer.pixel() != descriptor.pixel {
            return Err(PixelsError::invalid_argument(
                "buffer",
                format!("buffer is {} but the image is {}", buffer.pixel(), descriptor.pixel),
            ));
        }
        Ok(Self { buffer, descriptor })
    }

    /// The pixels backing this producer.
    #[must_use]
    pub fn buffer(&self) -> &Arc<TileBuf> {
        &self.buffer
    }
}

impl Producer for BufferSource {
    fn name(&self) -> &'static str {
        "buffer"
    }

    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn produce(&self, region: Region, output: &mut TileMut<'_>) -> Result<()> {
        let tile = self.buffer.as_tile()?;
        copy_region(&tile, output, region)
    }
}

/// The decode state of a [`DecodedSource`].
///
/// A streaming decoder is single-pass, so the first pull runs it to completion
/// and every later pull is served from the result.
#[derive(Debug)]
enum DecodeState {
    /// Header parsed, no pixels read yet.
    Pending(Box<dyn Decoder>),
    /// Fully decoded.
    Done(Arc<TileBuf>),
    /// Decoding failed; the error is not retryable because the source has
    /// already been consumed.
    Failed(String),
}

/// A producer that decodes a stream on first pull.
///
/// Construction parses the header only — no pixel bytes are read — so
/// [`Image::metadata`] stays free and laziness holds (SPEC §Guarantees 3).
///
/// # Memory
///
/// This M1 producer materializes the whole image on first pull, because the
/// naive evaluator it feeds is itself whole-image (ROADMAP M1). It is
/// deliberately *not* the constant-memory path: M2's scheduler pulls regions in
/// order and streams rows through, which is where the constant-memory exit
/// criterion is proved. The [`Producer`] contract does not change — only what
/// sits behind it.
///
/// [`Image::metadata`]: crate::Image::metadata
#[derive(Debug)]
pub struct DecodedSource {
    descriptor: ImageDescriptor,
    state: Mutex<DecodeState>,
}

impl DecodedSource {
    /// Wrap a decoder whose header has been parsed.
    ///
    /// No pixel rows are read here.
    #[must_use]
    pub fn new(decoder: Box<dyn Decoder>) -> Self {
        Self {
            descriptor: decoder.descriptor(),
            state: Mutex::new(DecodeState::Pending(decoder)),
        }
    }

    /// Decode every row into a buffer.
    fn decode_all(decoder: &mut dyn Decoder, descriptor: ImageDescriptor) -> Result<TileBuf> {
        let mut buffer = TileBuf::for_image(&descriptor)?;
        {
            let mut tile = buffer.as_tile_mut()?;
            for y in 0..descriptor.height {
                let row = tile.row_mut(y).ok_or_else(|| {
                    PixelsError::malformed("stream", format!("row {y} is outside the image"))
                })?;
                decoder.read_row(row)?;
            }
        }
        Ok(buffer)
    }

    /// Decode on first call; return the shared buffer on every call.
    fn decoded(&self) -> Result<Arc<TileBuf>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PixelsError::graph("decoder state was poisoned by a panicking thread"))?;
        match &*state {
            DecodeState::Done(buffer) => return Ok(Arc::clone(buffer)),
            DecodeState::Failed(detail) => {
                return Err(PixelsError::malformed("stream", detail.clone()));
            }
            DecodeState::Pending(_) => {}
        }
        let DecodeState::Pending(decoder) = &mut *state else {
            // Unreachable: the match above returned for every other variant.
            return Err(PixelsError::graph("decoder state changed unexpectedly"));
        };
        match Self::decode_all(decoder.as_mut(), self.descriptor) {
            Ok(buffer) => {
                let buffer = Arc::new(buffer);
                *state = DecodeState::Done(Arc::clone(&buffer));
                Ok(buffer)
            }
            Err(error) => {
                // The stream is consumed; remember the failure so a second pull
                // reports the same error rather than decoding a partial stream.
                *state = DecodeState::Failed(error.to_string());
                Err(error)
            }
        }
    }
}

impl Producer for DecodedSource {
    fn name(&self) -> &'static str {
        "decoded"
    }

    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn produce(&self, region: Region, output: &mut TileMut<'_>) -> Result<()> {
        let buffer = self.decoded()?;
        let tile = buffer.as_tile()?;
        copy_region(&tile, output, region)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests operate on known-good values and assert shapes directly")]
mod tests {
    use super::*;
    use crate::{ErrorCode, PixelFormat};

    /// A decoder that yields `height` rows of an increasing counter, then
    /// optionally fails at row `fail_at`.
    #[derive(Debug)]
    struct StubDecoder {
        descriptor: ImageDescriptor,
        row: u32,
        fail_at: Option<u32>,
        rows_read: Arc<std::sync::atomic::AtomicU32>,
    }

    impl Decoder for StubDecoder {
        fn descriptor(&self) -> ImageDescriptor {
            self.descriptor
        }
        fn read_row(&mut self, out: &mut [u8]) -> Result<()> {
            if Some(self.row) == self.fail_at {
                return Err(PixelsError::malformed("stub", "corrupt row"));
            }
            self.rows_read.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            out.fill(self.row as u8);
            self.row += 1;
            Ok(())
        }
    }

    fn stub(fail_at: Option<u32>) -> (DecodedSource, Arc<std::sync::atomic::AtomicU32>) {
        let rows_read = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let decoder = StubDecoder {
            descriptor: ImageDescriptor::new(2, 3, PixelFormat::Gray8).unwrap(),
            row: 0,
            fail_at,
            rows_read: Arc::clone(&rows_read),
        };
        (DecodedSource::new(Box::new(decoder)), rows_read)
    }

    #[test]
    fn construction_reads_no_pixel_rows() {
        let (source, rows_read) = stub(None);
        assert_eq!(source.descriptor().width, 2);
        assert_eq!(rows_read.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn the_stream_is_decoded_once_and_reused() {
        let (source, rows_read) = stub(None);
        let mut out = TileBuf::zeroed(Region::from_size(2, 3), PixelFormat::Gray8).unwrap();
        for _ in 0..3 {
            let mut tile = out.as_tile_mut().unwrap();
            source.produce(Region::from_size(2, 3), &mut tile).unwrap();
        }
        assert_eq!(rows_read.load(std::sync::atomic::Ordering::Relaxed), 3, "decoded exactly once");
        assert_eq!(out.bytes(), &[0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn a_partial_region_pull_returns_just_that_region() {
        let (source, _) = stub(None);
        let mut out = TileBuf::zeroed(Region::new(0, 1, 2, 1), PixelFormat::Gray8).unwrap();
        let mut tile = out.as_tile_mut().unwrap();
        source.produce(Region::new(0, 1, 2, 1), &mut tile).unwrap();
        assert_eq!(out.bytes(), &[1, 1]);
    }

    #[test]
    fn a_decode_failure_is_sticky_and_never_partial() {
        let (source, _) = stub(Some(1));
        let mut out = TileBuf::zeroed(Region::from_size(2, 3), PixelFormat::Gray8).unwrap();
        for attempt in 0..2 {
            let mut tile = out.as_tile_mut().unwrap();
            let err = source.produce(Region::from_size(2, 3), &mut tile).unwrap_err();
            assert_eq!(err.code(), ErrorCode::Malformed, "attempt {attempt}");
        }
    }

    #[test]
    fn buffer_source_validates_against_its_descriptor() {
        let desc = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let good = Arc::new(TileBuf::zeroed(desc.region(), PixelFormat::Gray8).unwrap());
        assert!(BufferSource::new(desc, good).is_ok());

        let wrong_size =
            Arc::new(TileBuf::zeroed(Region::from_size(3, 3), PixelFormat::Gray8).unwrap());
        assert_eq!(
            BufferSource::new(desc, wrong_size).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );

        let wrong_format = Arc::new(TileBuf::zeroed(desc.region(), PixelFormat::Rgb8).unwrap());
        assert_eq!(
            BufferSource::new(desc, wrong_format).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn buffer_source_serves_regions() {
        let desc = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let buffer = Arc::new(
            TileBuf::from_vec(desc.region(), PixelFormat::Gray8, vec![1, 2, 3, 4]).unwrap(),
        );
        let source = BufferSource::new(desc, buffer).unwrap();
        let mut out = TileBuf::zeroed(Region::new(1, 0, 1, 2), PixelFormat::Gray8).unwrap();
        let mut tile = out.as_tile_mut().unwrap();
        source.produce(Region::new(1, 0, 1, 2), &mut tile).unwrap();
        assert_eq!(out.bytes(), &[2, 4]);
    }
}
