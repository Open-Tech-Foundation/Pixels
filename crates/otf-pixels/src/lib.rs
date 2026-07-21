//! A streaming, demand-driven image processing engine.
//!
//! Pixels is a libvips-class pipeline engine: images are lazy operation
//! graphs, pixels are pulled through the graph on demand, and memory stays
//! bounded regardless of image size. This crate is the facade — the chainable
//! [`Image`] API and the [`Image::output`] terminal — over
//! [`otf_pixels_core`]'s engine, [`otf_pixels_ops`]' kernels and the codec
//! crates.
//!
// The example writes raw bytes, so it only runs when that codec is compiled
// in. `raw` is a default feature, so docs.rs and an ordinary build both run it.
#![cfg_attr(feature = "raw", doc = "```")]
#![cfg_attr(not(feature = "raw"), doc = "```ignore")]
//! use otf_pixels::{Format, Image, ImageDescriptor, PixelFormat};
//!
//! # fn main() -> Result<(), otf_pixels::PixelsError> {
//! let descriptor = ImageDescriptor::new(4, 4, PixelFormat::Gray8)?;
//! let pixels: Vec<u8> = (0..16).collect();
//!
//! // Construction and chaining do no pixel work.
//! let bytes = Image::from_raw(descriptor, pixels)?
//!     .crop(1, 1, 2, 2)
//!     .flip()
//!     .output(Format::Raw, Default::default())
//!     .bytes()?;
//!
//! assert_eq!(bytes, [9, 10, 5, 6]);
//! # Ok(())
//! # }
//! ```
//!
//! # Errors are deferred, not swallowed
//!
//! Chaining methods take and return `Self` rather than [`Result`], so a
//! pipeline reads as one expression. An error raised mid-chain — a crop window
//! outside the image, say — is *captured* and carried to the terminal, where
//! it surfaces from [`Output::write`] or [`Output::bytes`]. Nothing is
//! silently ignored, and no operation runs after a failed one.
//!
//! # Evaluation
//!
//! Terminals run the pipeline on the demand-driven tile scheduler: output
//! tiles are evaluated in parallel and delivered to the sink in order, with
//! peak memory bounded by tiles in flight rather than image size. Tune it with
//! [`Output::threads`] or [`Output::scheduler_options`].
//!
//! [`Output::bytes_via_reference`] runs the same pipeline through the M1
//! whole-image evaluator instead. That path is slow and holds every
//! intermediate in full, but it is obviously correct, so it is the oracle the
//! scheduler is verified against.
//!
//! # Current scope
//!
//! Through ROADMAP M3. Notably **not** here yet:
//!
//! - `resize`, `rotate`, `modulate`, `convolve`, `composite` and the channel
//!   ops (M4).
//! - Every format but PNG and raw. [`Image::open`] reports an unsupported
//!   format for bytes it cannot identify rather than guessing.

use otf_pixels_core::{BufferSource, Op, Prefixed, Producer, Scheduler, TileBuf};
use std::sync::Arc;

pub use otf_pixels_core::{
    AccessPattern, ChannelLayout, Codec, ColorModel, Decoder, EncodeOptions, Encoder, ErrorCode,
    Format, ImageDescriptor, Limit, Limits, Metadata, PixelFormat, PixelsError, PlanOptions,
    Region, Result, RunStats, SchedulerOptions, Sink, Source, TileShape,
    evaluate as evaluate_reference,
};
pub use otf_pixels_ops::{Crop, Flip, Flop};

#[cfg(feature = "raw")]
pub use otf_pixels_codec_raw::{RawCodec, RawDecoder, RawEncoder, RawFormat};

#[cfg(feature = "png")]
pub use otf_pixels_codec_png::{PngCodec, PngDecoder, PngEncoder};

/// A lazily evaluated image pipeline.
///
/// Cheap to clone: clones share graph nodes rather than pixels. Chaining
/// builds graph structure and executes nothing (SPEC §Guarantees 3).
#[derive(Debug, Clone)]
pub struct Image {
    /// The graph so far, or the first error that occurred while building it.
    inner: std::result::Result<otf_pixels_core::Image, Arc<PixelsError>>,
}

impl Image {
    /// Build an image from raw pixels already in memory.
    ///
    /// `bytes` must be exactly the packed byte length of `descriptor`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if `bytes` is not exactly the
    /// packed length `descriptor` implies.
    pub fn from_raw(descriptor: ImageDescriptor, bytes: Vec<u8>) -> Result<Self> {
        let buffer = TileBuf::from_vec(descriptor.region(), descriptor.pixel, bytes)?;
        let source = BufferSource::new(descriptor, Arc::new(buffer))?;
        Ok(Self::from_producer(Arc::new(source), Format::Raw))
    }

    /// Build an image by decoding a raw pixel stream.
    ///
    /// The header parse is trivial for raw — the layout *is* the header — so
    /// this reads no bytes from `source`. Pixels are pulled at the terminal.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] if the layout is not
    /// representable on this platform.
    #[cfg(feature = "raw")]
    pub fn from_raw_stream(
        layout: RawFormat,
        source: impl Source + std::fmt::Debug + 'static,
    ) -> Result<Self> {
        let decoder = RawDecoder::new(layout, source)?;
        Ok(Self::from_decoder(Box::new(decoder), Format::Raw))
    }

    /// Open an image file, identifying its format from its contents.
    ///
    /// The path's extension is **ignored**. Detection is by magic bytes only
    /// (SPEC §Formats), because a name is an attacker-controlled hint while
    /// the bytes are a fact.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the file cannot be opened,
    /// [`PixelsError::Unsupported`] if no built-in codec recognises it, and
    /// [`PixelsError::Malformed`] if the header is invalid for the format its
    /// magic bytes claim.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| {
            // `PixelsError::io` takes a static context, so the path goes into
            // the wrapped error instead. Losing which file failed would make
            // the error useless in exactly the case it fires.
            let kind = e.kind();
            let detail = std::io::Error::new(kind, format!("{}: {e}", path.display()));
            PixelsError::io("opening image file", detail)
        })?;
        Self::from_stream(std::io::BufReader::new(file))
    }

    /// Build an image from a byte stream, identifying its format from the
    /// leading bytes.
    ///
    /// Sniffing reads only the longest magic prefix any known codec needs, and
    /// replays it to the decoder rather than seeking — a [`Source`] is
    /// forward-only (ADR-0005), so a pipe or socket works here exactly as a
    /// file does. A stream shorter than that prefix is not an error at this
    /// stage: it simply matches nothing.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] if no built-in codec recognises
    /// the stream, [`PixelsError::Io`] on read failure, or
    /// [`PixelsError::Malformed`] if the header is invalid for the format its
    /// magic bytes claim.
    pub fn from_stream(mut source: impl Source + std::fmt::Debug + 'static) -> Result<Self> {
        let codecs = sniffing_codecs();
        let longest = codecs.iter().map(|c| c.magic_len()).max().unwrap_or(0);

        // Short reads are normal, and a stream shorter than `longest` is a
        // legitimate no-match rather than a failure, so this loop stops at end
        // of input instead of demanding a full prefix.
        let mut prefix = Vec::with_capacity(longest);
        let mut buffer = vec![0_u8; longest];
        while prefix.len() < longest {
            let Some(rest) = buffer.get_mut(prefix.len()..) else {
                break;
            };
            match source.read(rest)? {
                0 => break,
                n => {
                    let Some(read) = rest.get(..n) else { break };
                    prefix.extend_from_slice(read);
                }
            }
        }

        let Some(codec) = codecs.iter().find(|codec| codec.probe(&prefix)) else {
            return Err(PixelsError::unsupported(format!(
                "no codec recognises this stream; its first {} bytes are {:02x?}",
                prefix.len().min(8),
                prefix.get(..prefix.len().min(8)).unwrap_or(&[])
            )));
        };
        let stream = Prefixed::new(prefix, source);
        // Unused when no decoding codec is compiled in, which is a legitimate
        // if degenerate build rather than a mistake.
        let _ = &stream;

        match codec.format() {
            #[cfg(feature = "png")]
            Format::Png => {
                let decoder = PngDecoder::new(stream, Limits::default())?;
                Ok(Self::from_decoder(Box::new(decoder), Format::Png))
            }
            other => Err(PixelsError::unsupported(format!(
                "{other} was detected but no decoder for it is compiled in"
            ))),
        }
    }

    /// Build an image from any decoder whose header has already been parsed.
    ///
    /// This is the extension point for codecs living outside this crate.
    #[must_use]
    pub fn from_decoder(decoder: Box<dyn Decoder>, format: Format) -> Self {
        let source = otf_pixels_core::DecodedSource::new(decoder);
        Self::from_producer(Arc::new(source), format)
    }

    /// Build an image from any pixel producer.
    #[must_use]
    pub fn from_producer(producer: Arc<dyn Producer>, format: Format) -> Self {
        Self {
            inner: Ok(otf_pixels_core::Image::from_producer(producer, format)),
        }
    }

    /// Header-only facts about this image: dimensions, format, pixel format.
    ///
    /// Free — descriptors are resolved as the graph is built, so this decodes
    /// nothing (SPEC §Guarantees 3).
    ///
    /// # Errors
    ///
    /// Returns the first error captured while building the pipeline, if any.
    pub fn metadata(&self) -> Result<Metadata> {
        self.graph()?.metadata()
    }

    /// The shape of this image at this point in the pipeline.
    ///
    /// # Errors
    ///
    /// Returns the first error captured while building the pipeline, if any.
    pub fn descriptor(&self) -> Result<ImageDescriptor> {
        Ok(self.graph()?.descriptor())
    }

    /// Extract the rectangular window at `(x, y)` of size `width` × `height`.
    ///
    /// A window outside the image is an error, surfaced at the terminal.
    #[must_use]
    pub fn crop(self, x: u32, y: u32, width: u32, height: u32) -> Self {
        match Crop::at(x, y, width, height) {
            Ok(op) => self.apply(Arc::new(op)),
            Err(error) => Self {
                inner: Err(Arc::new(error)),
            },
        }
    }

    /// Mirror vertically: the top row becomes the bottom row.
    #[must_use]
    pub fn flip(self) -> Self {
        self.apply(Arc::new(Flip))
    }

    /// Mirror horizontally: the left column becomes the right column.
    #[must_use]
    pub fn flop(self) -> Self {
        self.apply(Arc::new(Flop))
    }

    /// Chain an arbitrary op onto this pipeline.
    ///
    /// The escape hatch for ops defined outside this crate. Errors are
    /// deferred to the terminal, like every other chaining method.
    #[must_use]
    pub fn apply(self, op: Arc<dyn Op>) -> Self {
        Self {
            inner: match self.inner {
                Ok(image) => image.apply(op).map_err(Arc::new),
                // An earlier failure short-circuits: later ops never run.
                Err(error) => Err(error),
            },
        }
    }

    /// Choose the encoder and options for this pipeline's output.
    ///
    /// This is the single encode terminal, with format as data (ADR-0006):
    /// requesting a format that is not yet implemented is a catchable
    /// [`PixelsError::Unsupported`], not a compile error.
    #[must_use]
    pub fn output(self, format: Format, options: EncodeOptions) -> Output {
        Output {
            image: self,
            format,
            options,
            scheduler: SchedulerOptions::default(),
        }
    }

    /// The underlying graph, or the first captured error.
    fn graph(&self) -> Result<&otf_pixels_core::Image> {
        match &self.inner {
            Ok(image) => Ok(image),
            // The error is shared, so it is rebuilt rather than moved out.
            Err(error) => Err(rebuild(error)),
        }
    }
}

/// Reconstruct an owned error from a shared one.
///
/// [`PixelsError`] is not [`Clone`] — [`std::io::Error`] is not — so a captured
/// error is rebuilt preserving its code and message. The [`ErrorCode`], which
/// is the part under semver (SPEC §Guarantees 4), is exact.
fn rebuild(error: &Arc<PixelsError>) -> PixelsError {
    let detail = error.to_string();
    match error.code() {
        ErrorCode::Io => PixelsError::io("running the pipeline", std::io::Error::other(detail)),
        ErrorCode::Malformed => PixelsError::malformed("pipeline", detail),
        ErrorCode::Unsupported => PixelsError::unsupported(detail),
        ErrorCode::InvalidArgument => PixelsError::invalid_argument("pipeline", detail),
        ErrorCode::Graph => PixelsError::graph(detail),
        ErrorCode::LimitExceeded => match **error {
            PixelsError::LimitExceeded {
                limit,
                requested,
                allowed,
            } => PixelsError::limit_exceeded(limit, requested, allowed),
            _ => PixelsError::graph(detail),
        },
        // A code added in a later version still round-trips as an error.
        _ => PixelsError::graph(detail),
    }
}

/// A pipeline with its output format chosen, ready to be pulled.
///
/// Nothing has executed yet: the terminals on this type are what pull pixels
/// through the graph.
#[derive(Debug, Clone)]
pub struct Output {
    image: Image,
    format: Format,
    options: EncodeOptions,
    scheduler: SchedulerOptions,
}

impl Output {
    /// The format this output will be encoded as.
    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    /// The encoder options in effect.
    #[must_use]
    pub const fn options(&self) -> EncodeOptions {
        self.options
    }

    /// Run this pipeline on `threads` worker threads.
    ///
    /// Zero, the default, means one per available core. Setting it to one
    /// gives a fully deterministic serial run, which is what the differential
    /// tests against the reference evaluator use.
    #[must_use]
    pub const fn threads(mut self, threads: usize) -> Self {
        self.scheduler.threads = threads;
        self
    }

    /// Tune the scheduler directly.
    #[must_use]
    pub const fn scheduler_options(mut self, options: SchedulerOptions) -> Self {
        self.scheduler = options;
        self
    }

    /// Run the pipeline, streaming encoded bytes into `sink`.
    ///
    /// Rows are encoded and written in order as they are produced. A failure
    /// anywhere fails the whole call; partial output is never reported as
    /// success (ARCHITECTURE §Failure model), though bytes already handed to
    /// `sink` are of course already gone — a caller needing all-or-nothing
    /// should write to a buffer or a temporary and commit on success.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Unsupported`] if the format has no encoder in
    /// this build, and otherwise any error from the pipeline or the sink.
    pub fn write(self, mut sink: impl Sink) -> Result<()> {
        let image = self.image.graph()?.clone();
        let descriptor = image.descriptor();
        let mut encoder = encoder_for(self.format, self.options)?;
        encoder.write_header(&descriptor, &mut sink)?;

        // The scheduler delivers tiles; an encoder wants whole rows in order.
        let scheduler = Scheduler::new(self.scheduler)?;
        let mut rows = RowAssembler::new(descriptor);
        scheduler.run(&image, |region, tile| {
            rows.accept(region, tile, &mut |row| encoder.write_row(row, &mut sink))
        })?;
        rows.finish(&mut |row| encoder.write_row(row, &mut sink))?;
        encoder.finish(&mut sink)
    }

    /// Run the pipeline through the **reference** evaluator instead of the
    /// tile scheduler, collecting encoded bytes.
    ///
    /// The reference evaluator is single-threaded and whole-image: it holds
    /// every intermediate in full, so it is slow and its memory scales with
    /// the image. It exists because it is *obviously* correct, which makes it
    /// the oracle the scheduler is verified against — the two must produce
    /// byte-identical output for every pipeline (ROADMAP M2).
    ///
    /// Use it to verify, to debug a suspected scheduler bug, or where an image
    /// is small and determinism matters more than throughput. Prefer
    /// [`Output::bytes`] otherwise.
    ///
    /// # Errors
    ///
    /// As [`Output::bytes`].
    pub fn bytes_via_reference(self) -> Result<Vec<u8>> {
        let image = self.image.graph()?.clone();
        let descriptor = image.descriptor();
        let mut encoder = encoder_for(self.format, self.options)?;
        let mut sink = Vec::with_capacity(descriptor.byte_len().unwrap_or_default());
        encoder.write_header(&descriptor, &mut sink)?;
        otf_pixels_core::evaluate_rows(&image, |_, row| encoder.write_row(row, &mut sink))?;
        encoder.finish(&mut sink)?;
        Ok(sink)
    }

    /// Run the pipeline, collecting encoded bytes into a [`Vec`].
    ///
    /// # Errors
    ///
    /// As [`Output::write`].
    pub fn bytes(self) -> Result<Vec<u8>> {
        // Sizing the buffer up front avoids repeated growth for raw output,
        // where the encoded length is exactly the packed pixel length.
        let hint = self
            .image
            .descriptor()
            .ok()
            .and_then(|d| d.byte_len())
            .unwrap_or_default();
        let mut buffer = Vec::with_capacity(hint);
        self.write(&mut buffer)?;
        Ok(buffer)
    }
}

/// Reassembles scheduler tiles into whole rows for an encoder.
///
/// Sequential pipelines deliver full-width strips, so rows pass straight
/// through untouched — the common case costs nothing. Where a spatial op puts
/// the output on square tiles (ADR-0003), tiles arrive left-to-right within a
/// band, and a row is only complete once its band is. Those are buffered one
/// band at a time, so the cost is a band rather than an image.
#[derive(Debug)]
struct RowAssembler {
    descriptor: ImageDescriptor,
    /// The band being assembled, when tiles are narrower than the image.
    band: Option<otf_pixels_core::TileBuf>,
    /// Next row not yet emitted.
    next_row: u32,
}

impl RowAssembler {
    const fn new(descriptor: ImageDescriptor) -> Self {
        Self {
            descriptor,
            band: None,
            next_row: 0,
        }
    }

    /// Take one tile, emitting whatever rows it completes.
    fn accept(
        &mut self,
        region: Region,
        tile: &otf_pixels_core::Tile<'_>,
        emit: &mut impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        // Fast path: a full-width tile completes its own rows.
        if region.width == self.descriptor.width && self.band.is_none() {
            for y in region.y..region.y.saturating_add(region.height) {
                let row = tile
                    .row(y)
                    .ok_or_else(|| PixelsError::graph(format!("output tile is missing row {y}")))?;
                emit(row)?;
                self.next_row = y.saturating_add(1);
            }
            return Ok(());
        }

        // Narrow tile: accumulate into a full-width band, flushing the
        // previous one when a new band starts.
        let band_region = Region::new(0, region.y, self.descriptor.width, region.height);
        let starts_new_band = self
            .band
            .as_ref()
            .is_none_or(|band| band.region().y != region.y);
        if starts_new_band {
            self.flush(emit)?;
            self.band = Some(otf_pixels_core::TileBuf::zeroed(
                band_region,
                self.descriptor.pixel,
            )?);
        }
        let Some(band) = self.band.as_mut() else {
            return Err(PixelsError::graph("row band vanished"));
        };
        otf_pixels_core::copy_region(tile, &mut band.as_tile_mut()?, region)?;

        // A band is complete once its rightmost column has arrived.
        if region.right() >= u64::from(self.descriptor.width) {
            self.flush(emit)?;
        }
        Ok(())
    }

    /// Emit any buffered band's rows and drop it.
    fn flush(&mut self, emit: &mut impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let Some(band) = self.band.take() else {
            return Ok(());
        };
        let region = band.region();
        let view = band.as_tile()?;
        for y in region.y..region.y.saturating_add(region.height) {
            let row = view
                .row(y)
                .ok_or_else(|| PixelsError::graph(format!("row band is missing row {y}")))?;
            emit(row)?;
            self.next_row = y.saturating_add(1);
        }
        Ok(())
    }

    /// Emit anything still buffered at the end of a run.
    fn finish(&mut self, emit: &mut impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        self.flush(emit)
    }
}

/// Every codec that can be sniffed, in probe order.
///
/// Raw is deliberately absent: it has no magic bytes, so it can only be
/// requested, never detected. Adding it here would make it match everything.
fn sniffing_codecs() -> Vec<Box<dyn Codec>> {
    let codecs: Vec<Box<dyn Codec>> = vec![
        #[cfg(feature = "png")]
        Box::new(PngCodec),
    ];
    codecs
}

/// Build the encoder for `format`, or report that it is not available.
fn encoder_for(format: Format, options: EncodeOptions) -> Result<Box<dyn Encoder>> {
    // Only read by codecs that have something to tune; see the comment in
    // `from_stream` for why a build with none of them still has to compile.
    let _ = &options;
    match format {
        #[cfg(feature = "raw")]
        Format::Raw => Ok(Box::new(RawEncoder::new())),
        #[cfg(feature = "png")]
        Format::Png => Ok(Box::new(PngEncoder::from_options(&options))),
        #[cfg(not(feature = "raw"))]
        Format::Raw => Err(PixelsError::unsupported(
            "raw encoding requires the `raw` feature of otf-pixels",
        )),
        other => Err(PixelsError::unsupported(format!(
            "encoding {other} is not implemented yet; \
             see the ROADMAP for which milestone lands it"
        ))),
    }
}

// Gated on `raw` because almost every test here reaches pixels through the
// raw encoder, which is the only format that can express "these exact bytes".
// A build without it has nothing to compare against, so the suite compiles out
// rather than asserting less.
#[cfg(all(test, feature = "raw"))]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    fn ramp(width: u32, height: u32) -> Image {
        let descriptor = ImageDescriptor::new(width, height, PixelFormat::Gray8).unwrap();
        let len = descriptor.byte_len().unwrap();
        Image::from_raw(descriptor, (0..len).map(|i| i as u8).collect()).unwrap()
    }

    #[test]
    fn from_raw_requires_an_exactly_sized_buffer() {
        let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        assert!(Image::from_raw(descriptor, vec![0; 4]).is_ok());
        assert_eq!(
            Image::from_raw(descriptor, vec![0; 3]).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            Image::from_raw(descriptor, vec![0; 5]).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
    }

    #[test]
    fn a_chain_error_surfaces_at_the_terminal() {
        // A crop window outside the image, with more ops chained after it.
        let result = ramp(4, 4)
            .crop(3, 3, 4, 4)
            .flip()
            .flop()
            .output(Format::Raw, EncodeOptions::default())
            .bytes();
        let err = result.unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn an_error_short_circuits_later_ops() {
        // `crop` fails; `metadata` on the resulting pipeline reports it rather
        // than describing a shape that was never built.
        let broken = ramp(4, 4).crop(0, 0, 0, 0);
        assert!(broken.metadata().is_err());
        assert!(broken.clone().flip().descriptor().is_err());
    }

    #[test]
    fn metadata_is_free_and_reports_the_pipeline_shape() {
        let meta = ramp(8, 6).metadata().unwrap();
        assert_eq!((meta.width, meta.height), (8, 6));
        assert_eq!(meta.format, Format::Raw);
        assert_eq!(meta.pixel, PixelFormat::Gray8);
        // After a crop the shape reflects the crop, still without decoding.
        let cropped = ramp(8, 6).crop(1, 1, 3, 2).metadata().unwrap();
        assert_eq!((cropped.width, cropped.height), (3, 2));
    }

    #[test]
    fn unimplemented_formats_are_catchable_errors() {
        // Png is absent: it landed in M3 and is checked by the round-trip
        // tests instead. Every remaining format must still fail cleanly.
        for format in [
            Format::Jpeg,
            Format::Gif,
            Format::Tiff,
            Format::WebP,
            Format::Avif,
        ] {
            let err = ramp(2, 2)
                .output(format, EncodeOptions::default())
                .bytes()
                .unwrap_err();
            assert_eq!(err.code(), ErrorCode::Unsupported, "{format}");
            assert!(err.to_string().contains(format.as_str()), "{err}");
        }
    }

    #[test]
    fn write_streams_into_any_sink() {
        let mut sink = Vec::new();
        ramp(2, 2)
            .output(Format::Raw, EncodeOptions::default())
            .write(&mut sink)
            .unwrap();
        assert_eq!(sink, [0, 1, 2, 3]);
    }

    #[test]
    fn output_reports_its_settings() {
        let options = EncodeOptions::with_quality(55).unwrap();
        let output = ramp(2, 2).output(Format::Raw, options);
        assert_eq!(output.format(), Format::Raw);
        assert_eq!(output.options().quality, 55);
    }

    #[cfg(feature = "raw")]
    #[test]
    fn from_raw_stream_defers_decoding_to_the_terminal() {
        let descriptor = ImageDescriptor::new(2, 2, PixelFormat::Gray8).unwrap();
        let layout = RawFormat::packed(descriptor);
        let cursor = std::io::Cursor::new(vec![1_u8, 2, 3, 4]);
        let image = Image::from_raw_stream(layout, cursor).unwrap();
        // Header facts are available without reading pixels.
        assert_eq!(image.metadata().unwrap().width, 2);
        let bytes = image
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
    }

    #[cfg(feature = "raw")]
    #[test]
    fn a_truncated_stream_fails_the_terminal_without_panicking() {
        let descriptor = ImageDescriptor::new(4, 4, PixelFormat::Gray8).unwrap();
        let layout = RawFormat::packed(descriptor);
        let cursor = std::io::Cursor::new(vec![1_u8; 5]);
        let image = Image::from_raw_stream(layout, cursor).unwrap();
        let err = image
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
    }

    #[test]
    fn images_are_send_sync_and_cheap_to_clone() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Image>();
        assert_send_sync::<Output>();
        let image = ramp(4, 4);
        let clone = image.clone();
        assert_eq!(
            image.metadata().unwrap().width,
            clone.metadata().unwrap().width
        );
    }

    #[cfg(feature = "png")]
    #[test]
    fn a_png_round_trips_through_the_facade() {
        let png = ramp(37, 21)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap();
        let image = Image::from_stream(std::io::Cursor::new(png)).unwrap();
        assert_eq!(image.descriptor().unwrap().width, 37);
        assert_eq!(image.metadata().unwrap().format, Format::Png);

        let back = image
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        let expected = ramp(37, 21)
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert_eq!(back, expected, "pixels changed across a PNG round trip");
    }

    #[cfg(feature = "png")]
    #[test]
    fn a_pipeline_survives_a_png_round_trip() {
        // The point of sniffing is that a decoded image is an ordinary graph
        // source, so ops compose over it exactly as over raw pixels.
        let png = ramp(16, 16)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap();
        let cropped = Image::from_stream(std::io::Cursor::new(png))
            .unwrap()
            .crop(2, 3, 8, 5)
            .flip()
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        let direct = ramp(16, 16)
            .crop(2, 3, 8, 5)
            .flip()
            .output(Format::Raw, EncodeOptions::default())
            .bytes()
            .unwrap();
        assert_eq!(cropped, direct);
    }

    #[cfg(feature = "png")]
    #[test]
    fn open_ignores_the_extension_and_reads_the_bytes() {
        let dir = std::env::temp_dir().join(format!("otf-pixels-open-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("actually-a-png.jpg");
        let png = ramp(8, 8)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap();
        std::fs::write(&path, &png).unwrap();

        let image = Image::open(&path).unwrap();
        assert_eq!(
            image.metadata().unwrap().format,
            Format::Png,
            "extension won over content"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_names_the_file_it_could_not_read() {
        let error = Image::open("/nonexistent/otf-pixels/missing.png").unwrap_err();
        assert_eq!(error.code(), ErrorCode::Io);
        assert!(error.to_string().contains("missing.png"), "{error}");
    }

    #[test]
    fn an_unrecognised_stream_is_unsupported_not_a_guess() {
        for bytes in [&b""[..], &b"not an image"[..], &[0_u8; 64][..]] {
            let error = Image::from_stream(std::io::Cursor::new(bytes.to_vec())).unwrap_err();
            assert_eq!(error.code(), ErrorCode::Unsupported, "{bytes:02x?}");
        }
    }

    #[cfg(feature = "png")]
    #[test]
    fn a_truncated_png_is_malformed_not_a_panic() {
        let png = ramp(8, 8)
            .output(Format::Png, EncodeOptions::default())
            .bytes()
            .unwrap();
        // Past the signature, so sniffing succeeds and the header parse is
        // what has to fail cleanly.
        for cut in [9, 16, 24, 32, png.len() - 1] {
            let truncated = png[..cut].to_vec();
            let result = Image::from_stream(std::io::Cursor::new(truncated))
                .and_then(|i| i.output(Format::Raw, EncodeOptions::default()).bytes());
            assert!(result.is_err(), "truncating to {cut} bytes should fail");
        }
    }

    #[cfg(feature = "png")]
    #[test]
    fn sniffing_reads_only_the_magic_bytes_before_deciding() {
        // A stream that is exactly the signature and nothing else must be
        // recognised as PNG and then fail on its missing header, proving the
        // sniff does not need — or read — more than the magic.
        let error = Image::from_stream(std::io::Cursor::new(
            otf_pixels_codec_png::SIGNATURE.to_vec(),
        ))
        .unwrap_err();
        assert_eq!(error.code(), ErrorCode::Malformed, "{error}");
    }
}
