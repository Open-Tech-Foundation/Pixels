//! Encode an arbitrary animation as GIF, WebP or JPEG XL, then decode it
//! back frame by frame (ADR-0020). The input is a raster we chose, so
//! content can be asserted:
//!
//! - lossless WebP and JPEG XL give back every frame exactly, with its
//!   duration, once repeated frames are merged as the encoders merge them;
//! - GIF gives back the frame count, durations to the hundredth, and which
//!   pixels are transparent, exactly — colours go through a palette;
//! - lossy WebP decodes, with the frame count and alpha exact.
//!
//! Each frame after the first repaints a rectangle the input chooses, so the
//! encoders' changed-rectangle and blending paths are what is fuzzed.
#![no_main]

use libfuzzer_sys::fuzz_target;
use otf_pixels_codec_gif::{GifDecoder, GifEncoder};
use otf_pixels_codec_jxl::{JxlDecoder, JxlEncoder};
use otf_pixels_codec_webp::{WebPDecoder, WebPEncoder};
use otf_pixels_core::{Decoder, EncodeOptions, Encoder, ImageDescriptor, Limits, PixelFormat};

/// Bytes taken from the input in order, zero once it runs out.
struct Take<'a>(&'a [u8]);

impl Take<'_> {
    fn byte(&mut self) -> u8 {
        let Some((&b, rest)) = self.0.split_first() else {
            return 0;
        };
        self.0 = rest;
        b
    }
}

fuzz_target!(|data: &[u8]| {
    let mut take = Take(data);
    let width = u32::from(take.byte()) % 40 + 1;
    let height = u32::from(take.byte()) % 40 + 1;
    let count = usize::from(take.byte()) % 6 + 1;
    let alpha = take.byte() % 2 == 1;
    let codec = take.byte() % 4;
    let format = if alpha { PixelFormat::Rgba8 } else { PixelFormat::Rgb8 };
    let channels = format.channels();
    let Ok(descriptor) = ImageDescriptor::new(width, height, format) else {
        return;
    };
    let len = descriptor.byte_len().unwrap_or(0);

    let mut frames: Vec<Vec<u8>> = vec![(0..len).map(|_| take.byte()).collect()];
    let mut durations = vec![u32::from(take.byte()) * 10 + 10];
    for _ in 1..count {
        let mut frame = frames.last().cloned().unwrap_or_default();
        let x = u32::from(take.byte()) % width;
        let y = u32::from(take.byte()) % height;
        let w = u32::from(take.byte()) % (width - x) + 1;
        let h = u32::from(take.byte()) % (height - y) + 1;
        let fill: Vec<u8> = (0..channels).map(|_| take.byte()).collect();
        for yy in y..y + h {
            for xx in x..x + w {
                let at = (yy * width + xx) as usize * channels;
                if let Some(pixel) = frame.get_mut(at..at + channels) {
                    pixel.copy_from_slice(&fill);
                }
            }
        }
        frames.push(frame);
        durations.push(u32::from(take.byte()) * 10 + 10);
    }

    let lossless = EncodeOptions::default().with_lossless(true);
    let mut encoder: Box<dyn Encoder> = match codec {
        0 => Box::new(GifEncoder::from_options(&EncodeOptions::default())),
        1 => Box::new(WebPEncoder::from_options(&lossless)),
        2 => Box::new(WebPEncoder::from_options(&EncodeOptions::default())),
        _ => Box::new(JxlEncoder::from_options(&lossless)),
    };
    let mut file = Vec::new();
    encoder.set_animation(0).unwrap();
    encoder.write_header(&descriptor, &mut file).unwrap();
    for (frame, &duration) in frames.iter().zip(&durations) {
        encoder.begin_frame(duration, &mut file).unwrap();
        for row in frame.chunks_exact(descriptor.row_bytes()) {
            encoder.write_row(row, &mut file).unwrap();
        }
        encoder.end_frame(&mut file).unwrap();
    }
    encoder.finish(&mut file).unwrap();

    let mut decoder: Box<dyn Decoder> = match codec {
        0 => Box::new(GifDecoder::new(&file[..], Limits::default()).unwrap()),
        1 | 2 => Box::new(WebPDecoder::new(&file[..], Limits::default()).unwrap()),
        _ => Box::new(JxlDecoder::new(&file[..], Limits::default()).unwrap()),
    };
    let out = decoder.descriptor();
    let decoded = otf_pixels_core::testing::read_frames(decoder.as_mut()).unwrap();

    // Each frame as RGBA, what a viewer shows: GIF draws a pixel or not, at
    // alpha one half; transparent pixels are all alike.
    let gif = codec == 0;
    let shown = |pixels: &[u8], channels: usize| -> Vec<u8> {
        pixels
            .chunks_exact(channels)
            .flat_map(|p| {
                let a = if channels == 4 { p[3] } else { 255 };
                let a = if gif { if a < 128 { 0 } else { 255 } } else { a };
                if a == 0 { [0, 0, 0, 0] } else { [p[0], p[1], p[2], a] }
            })
            .collect()
    };
    // The frames as the encoders keep them: a repeat lengthens the one
    // before. GIF and WebP compare what is shown; lossless JPEG XL keeps the
    // colour under a transparent pixel, so it compares every byte.
    let jxl = codec == 3;
    let mut expected: Vec<(Vec<u8>, u32)> = Vec::new();
    for (frame, &duration) in frames.iter().zip(&durations) {
        let frame = if jxl { frame.clone() } else { shown(frame, channels) };
        match expected.last_mut() {
            Some((last, total)) if *last == frame => *total += duration,
            _ => expected.push((frame, duration)),
        }
    }
    assert_eq!(decoded.len(), expected.len(), "frame count");
    let mut elapsed = 0_u32;
    let mut written = 0_u32;
    for ((info, pixels), (want, duration)) in decoded.iter().zip(&expected) {
        let got = if jxl { pixels.clone() } else { shown(pixels, out.pixel.channels()) };
        elapsed += duration;
        if gif {
            let cs = (elapsed + 5) / 10;
            // A single frame is a still, with no timing to read back.
            if expected.len() > 1 {
                assert_eq!(info.duration_ms, (cs - written) * 10, "GIF delay");
            }
            written = cs;
            let transparent = |v: &[u8]| v.chunks_exact(4).map(|p| p[3] == 0).collect::<Vec<_>>();
            assert_eq!(transparent(&got), transparent(want), "GIF transparency");
            continue;
        }
        if expected.len() > 1 {
            assert_eq!(info.duration_ms, *duration, "duration");
        }
        if codec == 2 {
            let alpha = |v: &[u8]| v.chunks_exact(4).map(|p| p[3]).collect::<Vec<_>>();
            assert_eq!(alpha(&got), alpha(want), "lossy WebP alpha");
        } else {
            assert!(got == *want, "a lossless frame changed");
        }
    }
});
