//! Producers that root a graph in actual pixels.
//!
//! [`BufferSource`] wraps pixels the caller already holds. [`DecodedSource`]
//! bridges a streaming [`Decoder`] into the graph, deferring all pixel work
//! until a terminal pulls.

use crate::{
    DecodeCapability, Decoder, ImageDescriptor, PixelsError, Producer, Region, Result, TileBuf,
    TileMut, copy_region,
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
                format!(
                    "buffer is {} but the image is {}",
                    buffer.pixel(),
                    descriptor.pixel
                ),
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

    fn capability(&self) -> DecodeCapability {
        // The pixels are already in memory, so any region is equally cheap.
        // This is what lets `flip` over a memory buffer stream rather than
        // materialize (ADR-0009).
        DecodeCapability::Regions
    }

    fn produce(&self, region: Region, output: &mut TileMut<'_>) -> Result<()> {
        let tile = self.buffer.as_tile()?;
        copy_region(&tile, output, region)
    }
}

/// The decode state of a [`DecodedSource`].
#[derive(Debug)]
enum DecodeState {
    /// Header parsed; `cursor` rows have been consumed from the stream.
    Reading {
        /// The decoder, positioned at row `cursor`.
        decoder: Box<dyn Decoder>,
        /// The next row the decoder will emit.
        cursor: u32,
    },
    /// Decoding failed. The stream is consumed and cannot be retried, so the
    /// failure is remembered and replayed rather than producing partial pixels.
    Failed(String),
}

/// A producer that decodes a stream on demand, forward only.
///
/// Construction parses the header only — no pixel bytes are read — so
/// [`Image::metadata`] stays free and laziness holds (SPEC §Guarantees 3).
///
/// # Memory
///
/// Constant, and independent of image height: the decoder is advanced to the
/// requested band and exactly that band is retained as a rolling window. This
/// is what makes SPEC §Guarantees 1 true for streaming formats — the whole
/// image is never resident.
///
/// The window also absorbs *repeated* demand for the same band, which is what
/// lets two graph branches share one source without either re-reading a stream
/// that has already moved on.
///
/// # Forward only
///
/// A request that starts before the window is an error, not a rewind: the
/// bytes are gone (ADR-0005). The scheduler is responsible for never asking —
/// [`Plan`] detects non-forward demand ahead of time and materializes instead
/// (ADR-0009). Reaching this error therefore indicates a scheduling defect, and
/// it is reported rather than papered over.
///
/// [`Image::metadata`]: crate::Image::metadata
/// [`Plan`]: crate::Plan
#[derive(Debug)]
pub struct DecodedSource {
    descriptor: ImageDescriptor,
    capability: DecodeCapability,
    state: Mutex<DecodeState>,
    /// The most recently decoded band, reused for repeated demand.
    window: Mutex<Option<(Region, Arc<TileBuf>)>>,
}

impl DecodedSource {
    /// Wrap a decoder whose header has been parsed.
    ///
    /// No pixel rows are read here.
    #[must_use]
    pub fn new(decoder: Box<dyn Decoder>) -> Self {
        Self {
            descriptor: decoder.descriptor(),
            capability: decoder.capability(),
            state: Mutex::new(DecodeState::Reading { decoder, cursor: 0 }),
            window: Mutex::new(None),
        }
    }

    /// Decode rows `band.y .. band.bottom()` into a fresh buffer.
    ///
    /// Rows before `band.y` are decoded and discarded, since a forward-only
    /// stream has no way to skip them.
    fn decode_band(&self, band: Region) -> Result<Arc<TileBuf>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| PixelsError::graph("decoder state was poisoned by a panicking thread"))?;

        if let DecodeState::Failed(detail) = &*state {
            return Err(PixelsError::malformed("stream", detail.clone()));
        }
        let DecodeState::Reading { decoder, cursor } = &mut *state else {
            return Err(PixelsError::graph("decoder state changed unexpectedly"));
        };

        if band.y < *cursor {
            return Err(PixelsError::graph(format!(
                "source cannot rewind: row {} was requested but the stream is at row {cursor}; \
                 this pipeline needed a materialization point",
                band.y
            )));
        }

        let outcome = Self::fill_band(decoder.as_mut(), cursor, band, self.descriptor);
        match outcome {
            Ok(buffer) => Ok(Arc::new(buffer)),
            Err(error) => {
                *state = DecodeState::Failed(error.to_string());
                Err(error)
            }
        }
    }

    /// Advance `decoder` to `band` and read it, updating `cursor`.
    fn fill_band(
        decoder: &mut dyn Decoder,
        cursor: &mut u32,
        band: Region,
        descriptor: ImageDescriptor,
    ) -> Result<TileBuf> {
        let mut scratch = vec![0_u8; descriptor.row_bytes()];
        while *cursor < band.y {
            decoder.read_row(&mut scratch)?;
            *cursor += 1;
        }
        let mut buffer = TileBuf::zeroed(band, descriptor.pixel)?;
        {
            let mut tile = buffer.as_tile_mut()?;
            for y in band.y..band.y.saturating_add(band.height) {
                let row = tile.row_mut(y).ok_or_else(|| {
                    PixelsError::malformed("stream", format!("row {y} is outside the band"))
                })?;
                decoder.read_row(row)?;
                *cursor += 1;
            }
        }
        Ok(buffer)
    }

    /// The band covering `region`, from the window or freshly decoded.
    fn band_for(&self, region: Region) -> Result<Arc<TileBuf>> {
        {
            let window = self
                .window
                .lock()
                .map_err(|_| PixelsError::graph("source window was poisoned"))?;
            if let Some((covered, buffer)) = &*window {
                if covered.contains(region) {
                    return Ok(Arc::clone(buffer));
                }
            }
        }
        // Decode full-width bands: a forward-only stream produces whole rows
        // anyway, so narrowing would discard pixels a later request may want.
        let band = Region::new(0, region.y, self.descriptor.width, region.height);
        let buffer = self.decode_band(band)?;
        if let Ok(mut window) = self.window.lock() {
            *window = Some((band, Arc::clone(&buffer)));
        }
        Ok(buffer)
    }
}

impl Producer for DecodedSource {
    fn name(&self) -> &'static str {
        "decoded"
    }

    fn descriptor(&self) -> ImageDescriptor {
        self.descriptor
    }

    fn capability(&self) -> DecodeCapability {
        self.capability
    }

    fn produce(&self, region: Region, output: &mut TileMut<'_>) -> Result<()> {
        let buffer = self.band_for(region)?;
        let tile = buffer.as_tile()?;
        copy_region(&tile, output, region)
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
            self.rows_read
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    /// A 2x64 source, tall enough that a band is much smaller than the image.
    fn tall_stub() -> (DecodedSource, Arc<std::sync::atomic::AtomicU32>) {
        let rows_read = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let decoder = StubDecoder {
            descriptor: ImageDescriptor::new(2, 64, PixelFormat::Gray8).unwrap(),
            row: 0,
            fail_at: None,
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
        assert_eq!(
            rows_read.load(std::sync::atomic::Ordering::Relaxed),
            3,
            "decoded exactly once"
        );
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
            let err = source
                .produce(Region::from_size(2, 3), &mut tile)
                .unwrap_err();
            assert_eq!(err.code(), ErrorCode::Malformed, "attempt {attempt}");
        }
    }

    #[test]
    fn decoding_advances_only_as_far_as_demanded() {
        // The constant-memory property: asking for an early band must not
        // decode the rest of the image.
        let (source, rows_read) = tall_stub();
        let mut out = TileBuf::zeroed(Region::new(0, 0, 2, 4), PixelFormat::Gray8).unwrap();
        let mut tile = out.as_tile_mut().unwrap();
        source.produce(Region::new(0, 0, 2, 4), &mut tile).unwrap();
        assert_eq!(
            rows_read.load(std::sync::atomic::Ordering::Relaxed),
            4,
            "decoded past the requested band"
        );
    }

    #[test]
    fn successive_bands_stream_forward() {
        let (source, rows_read) = tall_stub();
        for start in (0..64).step_by(4) {
            let band = Region::new(0, start, 2, 4);
            let mut out = TileBuf::zeroed(band, PixelFormat::Gray8).unwrap();
            let mut tile = out.as_tile_mut().unwrap();
            source.produce(band, &mut tile).unwrap();
            // Each row of the ramp holds its own index.
            assert_eq!(out.bytes()[0], start as u8, "band at row {start} is wrong");
        }
        assert_eq!(
            rows_read.load(std::sync::atomic::Ordering::Relaxed),
            64,
            "rows re-read"
        );
    }

    #[test]
    fn repeated_demand_for_a_band_is_served_from_the_window() {
        // Two graph branches pulling the same band must not re-read a stream
        // that has already moved on.
        let (source, rows_read) = tall_stub();
        let band = Region::new(0, 0, 2, 4);
        for _ in 0..5 {
            let mut out = TileBuf::zeroed(band, PixelFormat::Gray8).unwrap();
            let mut tile = out.as_tile_mut().unwrap();
            source.produce(band, &mut tile).unwrap();
        }
        assert_eq!(
            rows_read.load(std::sync::atomic::Ordering::Relaxed),
            4,
            "band was re-decoded"
        );
    }

    #[test]
    fn a_sub_band_of_the_window_is_served_without_re_reading() {
        let (source, rows_read) = tall_stub();
        let band = Region::new(0, 0, 2, 8);
        let mut out = TileBuf::zeroed(band, PixelFormat::Gray8).unwrap();
        source
            .produce(band, &mut out.as_tile_mut().unwrap())
            .unwrap();

        let inner = Region::new(0, 2, 2, 2);
        let mut small = TileBuf::zeroed(inner, PixelFormat::Gray8).unwrap();
        source
            .produce(inner, &mut small.as_tile_mut().unwrap())
            .unwrap();
        assert_eq!(small.bytes(), &[2, 2, 3, 3]);
        assert_eq!(rows_read.load(std::sync::atomic::Ordering::Relaxed), 8);
    }

    #[test]
    fn rewinding_is_a_reported_error_not_silent_corruption() {
        // A forward-only stream cannot go back (ADR-0005). Reaching this means
        // the plan failed to insert a materialization point (ADR-0009), so it
        // is surfaced rather than papered over.
        let (source, _) = tall_stub();
        let later = Region::new(0, 16, 2, 4);
        let mut out = TileBuf::zeroed(later, PixelFormat::Gray8).unwrap();
        source
            .produce(later, &mut out.as_tile_mut().unwrap())
            .unwrap();

        let earlier = Region::new(0, 0, 2, 4);
        let mut back = TileBuf::zeroed(earlier, PixelFormat::Gray8).unwrap();
        let err = source
            .produce(earlier, &mut back.as_tile_mut().unwrap())
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::Graph);
        assert!(err.to_string().contains("rewind"), "{err}");
        assert!(err.to_string().contains("materialization"), "{err}");
    }

    #[test]
    fn skipped_rows_are_consumed_not_lost() {
        // Jumping forward past a band still has to read those bytes, because a
        // forward-only stream cannot seek.
        let (source, rows_read) = tall_stub();
        let band = Region::new(0, 32, 2, 4);
        let mut out = TileBuf::zeroed(band, PixelFormat::Gray8).unwrap();
        source
            .produce(band, &mut out.as_tile_mut().unwrap())
            .unwrap();
        assert_eq!(rows_read.load(std::sync::atomic::Ordering::Relaxed), 36);
        assert_eq!(out.bytes()[0], 32, "landed on the wrong row");
    }

    #[test]
    fn producers_report_their_capability() {
        // The upstream half of ADR-0009's analysis.
        let (source, _) = tall_stub();
        assert_eq!(source.capability(), DecodeCapability::Sequential);

        let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let buffer = Arc::new(TileBuf::for_image(&descriptor).unwrap());
        let buffered = BufferSource::new(descriptor, buffer).unwrap();
        assert_eq!(buffered.capability(), DecodeCapability::Regions);
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
