//! Animations for the JPEG XL encoder's tests and the interop emitter.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test scenes of known-good values"
)]

use otf_pixels_codec_jxl::JxlEncoder;
use otf_pixels_core::{Encoder, ImageDescriptor, PixelFormat};

pub struct Scene {
    pub name: &'static str,
    pub descriptor: ImageDescriptor,
    pub frames: Vec<Vec<u8>>,
    pub durations: Vec<u32>,
    pub loop_count: u32,
}

/// A smooth backdrop with a square of `side` at `(x, y)`, in `format`.
fn frame(
    format: PixelFormat,
    width: u32,
    height: u32,
    square: (u32, u32, u32),
    alpha: u8,
) -> Vec<u8> {
    let channels = format.channels();
    let wide = format.bytes_per_pixel() / channels == 2;
    let mut out = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let inside = (square.0..square.0 + square.2).contains(&x)
                && (square.1..square.1 + square.2).contains(&y);
            let rgba: [u16; 4] = if inside {
                [250, 40, 30, 255]
            } else {
                [
                    (x * 5 % 256) as u16,
                    (y * 7 % 256) as u16,
                    ((x + y) * 2 % 256) as u16,
                    u16::from(alpha),
                ]
            };
            let samples: Vec<u16> = match channels {
                1 => vec![rgba[1]],
                2 => vec![rgba[1], rgba[3]],
                3 => rgba[..3].to_vec(),
                _ => rgba.to_vec(),
            };
            for v in samples {
                if wide {
                    out.extend_from_slice(&(v * 257 + (x as u16 & 3)).to_ne_bytes());
                } else {
                    out.push(v as u8);
                }
            }
        }
    }
    out
}

pub fn scenes() -> Vec<Scene> {
    let moving = |format, alpha: u8| -> Vec<Vec<u8>> {
        (0..5)
            .map(|k| frame(format, 37, 29, (3 + k * 5, 5 + k * 3, 7), alpha))
            .collect()
    };
    let rgba = PixelFormat::Rgba8;
    let mut translucent = moving(rgba, 160);
    // A frame turning a band transparent, then the first again.
    let band: Vec<u8> = translucent[1]
        .chunks_exact(4)
        .enumerate()
        .flat_map(|(i, p)| {
            if i / 37 < 6 {
                [0, 0, 0, 0]
            } else {
                [p[0], p[1], p[2], p[3]]
            }
        })
        .collect();
    translucent.push(band);
    translucent.push(translucent[0].clone());
    vec![
        Scene {
            name: "rgba8",
            descriptor: ImageDescriptor::new(37, 29, rgba).unwrap(),
            durations: vec![100, 40, 70, 33, 33, 50, 50],
            frames: translucent,
            loop_count: 0,
        },
        Scene {
            name: "rgb8",
            descriptor: ImageDescriptor::new(37, 29, PixelFormat::Rgb8).unwrap(),
            frames: moving(PixelFormat::Rgb8, 255),
            durations: vec![60, 60, 60, 60, 60],
            loop_count: 2,
        },
        Scene {
            name: "gray16",
            descriptor: ImageDescriptor::new(37, 29, PixelFormat::Gray16).unwrap(),
            frames: moving(PixelFormat::Gray16, 255),
            durations: vec![10, 20, 30, 40, 1000],
            loop_count: 7,
        },
    ]
}

pub fn encode(
    scene: &Scene,
    frames: &[Vec<u8>],
    durations: &[u32],
    mut encoder: JxlEncoder,
) -> Vec<u8> {
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
