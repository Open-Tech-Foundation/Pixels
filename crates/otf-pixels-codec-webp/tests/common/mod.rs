//! Animations for the WebP encoder's tests and the interop emitter: every
//! kind of change the encoder draws differently.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test scenes of known-good values"
)]

use otf_pixels_codec_webp::WebPEncoder;
use otf_pixels_core::{EncodeOptions, Encoder, ImageDescriptor, PixelFormat};

pub struct Scene {
    pub name: &'static str,
    pub descriptor: ImageDescriptor,
    pub frames: Vec<Vec<u8>>,
    pub durations: Vec<u32>,
    pub loop_count: u32,
}

fn rect(frame: &mut [u8], width: u32, x: u32, y: u32, w: u32, h: u32, rgba: [u8; 4]) {
    for yy in y..y + h {
        for xx in x..x + w {
            let at = ((yy * width + xx) * 4) as usize;
            frame[at..at + 4].copy_from_slice(&rgba);
        }
    }
}

/// A smooth, busy background: lossy-friendly, and every pixel different.
fn backdrop(width: u32, height: u32, alpha: u8) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&[(x * 6) as u8, (y * 9) as u8, ((x + y) * 3) as u8, alpha]);
        }
    }
    out
}

pub fn scenes() -> Vec<Scene> {
    let rgba = |w, h| ImageDescriptor::new(w, h, PixelFormat::Rgba8).unwrap();
    let mut out = Vec::new();

    // An opaque square crossing an opaque backdrop, at odd positions: the
    // blended path, offsets rounded to even.
    let base = backdrop(40, 30, 255);
    out.push(Scene {
        name: "moving",
        descriptor: rgba(40, 30),
        frames: (0..6)
            .map(|k| {
                let mut f = base.clone();
                rect(&mut f, 40, 3 + k * 5, 5 + k * 3, 7, 7, [250, 30, 30, 255]);
                f
            })
            .collect(),
        durations: vec![100, 40, 70, 33, 33, 34],
        loop_count: 0,
    });

    // Translucent changes and pixels turning transparent: the overwritten
    // path, which needs no disposal.
    let mut f0 = backdrop(32, 24, 200);
    rect(&mut f0, 32, 4, 4, 10, 10, [10, 200, 90, 255]);
    let mut f1 = f0.clone();
    rect(&mut f1, 32, 6, 6, 8, 8, [10, 200, 90, 90]);
    let mut f2 = f1.clone();
    rect(&mut f2, 32, 0, 0, 32, 12, [0, 0, 0, 0]);
    let f3 = backdrop(32, 24, 255);
    out.push(Scene {
        name: "translucent",
        descriptor: rgba(32, 24),
        frames: vec![f0, f1, f2, f3],
        durations: vec![60, 60, 60, 60],
        loop_count: 3,
    });

    // A first frame mostly transparent, then a single pixel at an odd
    // position, then nothing at all.
    let blank = vec![0_u8; 20 * 16 * 4];
    let mut g0 = blank.clone();
    rect(&mut g0, 20, 9, 7, 3, 2, [255, 255, 0, 255]);
    let mut g1 = g0.clone();
    rect(&mut g1, 20, 17, 13, 1, 1, [0, 0, 255, 255]);
    out.push(Scene {
        name: "sparse",
        descriptor: rgba(20, 16),
        frames: vec![g0, g1, blank],
        durations: vec![30, 30, 30],
        loop_count: 1,
    });
    out
}

pub fn encode(
    scene: &Scene,
    frames: &[Vec<u8>],
    durations: &[u32],
    options: &EncodeOptions,
) -> Vec<u8> {
    let mut encoder = WebPEncoder::from_options(options);
    let mut out = Vec::new();
    encoder.set_animation(scene.loop_count).unwrap();
    encoder.write_header(&scene.descriptor, &mut out).unwrap();
    for (frame, &duration) in frames.iter().zip(durations) {
        encoder.begin_frame(duration, &mut out).unwrap();
        for row in frame.chunks_exact(scene.descriptor.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.end_frame(&mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

pub fn lossless() -> EncodeOptions {
    EncodeOptions::default().with_lossless(true)
}
