//! Animations whose every frame fits a GIF palette exactly, so a decoder must
//! give back exactly what was encoded; shared by the encoder's tests and the
//! interop emitter.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test scenes of known-good values"
)]

use otf_pixels_codec_gif::GifEncoder;
use otf_pixels_core::{EncodeOptions, Encoder, ImageDescriptor, PixelFormat};

/// One test animation: its canvas, frames (RGBA), durations and loop count.
pub struct Scene {
    pub name: &'static str,
    pub descriptor: ImageDescriptor,
    pub frames: Vec<Vec<u8>>,
    pub durations: Vec<u32>,
    pub loop_count: u32,
}

fn canvas(width: u32, height: u32) -> (ImageDescriptor, Vec<u8>) {
    (
        ImageDescriptor::new(width, height, PixelFormat::Rgba8).unwrap(),
        vec![0; (width * height * 4) as usize],
    )
}

fn rect(frame: &mut [u8], width: u32, x: u32, y: u32, w: u32, h: u32, rgba: [u8; 4]) {
    for yy in y..y + h {
        for xx in x..x + w {
            let at = ((yy * width + xx) * 4) as usize;
            frame[at..at + 4].copy_from_slice(&rgba);
        }
    }
}

/// Every scene.
pub fn scenes() -> Vec<Scene> {
    let mut out = Vec::new();

    // A square crossing a transparent canvas past a still bar.
    let (descriptor, blank) = canvas(32, 24);
    let frames = (0..6)
        .map(|k| {
            let mut f = blank.clone();
            rect(&mut f, 32, 0, 20, 32, 3, [20, 40, 200, 255]);
            rect(&mut f, 32, 2 + k * 4, 3 + k * 2, 6, 6, [220, 30, 30, 255]);
            f
        })
        .collect();
    out.push(Scene {
        name: "moving",
        descriptor,
        frames,
        durations: vec![100, 40, 70, 33, 33, 34],
        loop_count: 0,
    });

    // Opaque, then partly and nearly wholly transparent, then opaque again:
    // only disposal can clear what an earlier frame drew.
    let (descriptor, blank) = canvas(20, 16);
    let mut f0 = blank.clone();
    rect(&mut f0, 20, 0, 0, 20, 16, [30, 160, 60, 255]);
    let mut f1 = f0.clone();
    rect(&mut f1, 20, 0, 0, 10, 16, [0, 0, 0, 0]);
    let mut f2 = blank.clone();
    rect(&mut f2, 20, 15, 3, 1, 1, [250, 250, 0, 255]);
    let f3 = f0.clone();
    out.push(Scene {
        name: "clearing",
        descriptor,
        frames: vec![f0, f1, f2, f3],
        durations: vec![80, 80, 80, 80],
        loop_count: 2,
    });

    // A frame needing colours the first palette lacks, then the first again.
    let (descriptor, blank) = canvas(24, 20);
    let few = |shift: u8| {
        let mut f = blank.clone();
        for i in 0..8_u32 {
            rect(
                &mut f,
                24,
                i * 3,
                0,
                3,
                20,
                [i as u8 * 30 + shift, 100, 50, 255],
            );
        }
        f
    };
    let mut many = blank.clone();
    for i in 0..(24 * 20) {
        let v = (i % 200) as u8;
        many[i * 4..i * 4 + 4].copy_from_slice(&[255 - v, v, (v / 3) + 60, 255]);
    }
    out.push(Scene {
        name: "palettes",
        descriptor,
        frames: vec![few(0), many, few(0), few(5)],
        durations: vec![50, 50, 50, 50],
        loop_count: 0,
    });
    out
}

/// Encode `frames` of `descriptor` as an animation with `options`.
pub fn encode(
    descriptor: &ImageDescriptor,
    frames: &[Vec<u8>],
    durations: &[u32],
    loop_count: u32,
    options: &EncodeOptions,
) -> Vec<u8> {
    let mut encoder = GifEncoder::from_options(options);
    let mut out = Vec::new();
    encoder.set_animation(loop_count).unwrap();
    encoder.write_header(descriptor, &mut out).unwrap();
    for (frame, &duration) in frames.iter().zip(durations) {
        encoder.begin_frame(duration, &mut out).unwrap();
        for row in frame.chunks_exact(descriptor.row_bytes()) {
            encoder.write_row(row, &mut out).unwrap();
        }
        encoder.end_frame(&mut out).unwrap();
    }
    encoder.finish(&mut out).unwrap();
    out
}

/// Options that keep every scene exact: a full palette, no dithering, and
/// no reuse of a palette that is merely close.
pub fn exact() -> EncodeOptions {
    EncodeOptions::default()
        .with_colours(256)
        .unwrap()
        .with_dither(0.0)
        .unwrap()
        .with_inter_palette_max_error(0.0)
        .unwrap()
}
