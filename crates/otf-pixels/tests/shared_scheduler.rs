//! One scheduler shared by many concurrent pipelines, as a server or a
//! runtime runs it: every result matches the reference evaluator, and runs
//! reuse the scheduler's threads instead of spawning their own.

#![cfg(all(feature = "png", feature = "raw"))]
#![allow(clippy::unwrap_used, reason = "tests operate on known-good values")]

use otf_pixels::{
    EncodeOptions, Fit, Format, Image, ImageDescriptor, PixelFormat, ResizeOptions, Scheduler,
    SchedulerOptions,
};
use std::sync::Arc;

fn pipeline(seed: u32) -> otf_pixels::Output {
    let (w, h) = (300 + seed * 7, 200 + seed * 3);
    let pixels: Vec<u8> = (0..w * h * 3).map(|i| ((i * (seed + 3)) % 251) as u8).collect();
    Image::from_raw(ImageDescriptor::new(w, h, PixelFormat::Rgb8).unwrap(), pixels)
        .unwrap()
        .resize_with(97, 61, ResizeOptions::default().with_fit(Fit::Cover))
        .blur(1.5)
        .output(Format::Png, EncodeOptions::default())
}

/// Threads in this process, where the platform says.
fn thread_count() -> Option<usize> {
    std::fs::read_dir("/proc/self/task").ok().map(Iterator::count)
}

#[test]
fn concurrent_pipelines_share_one_scheduler() {
    let scheduler = Arc::new(Scheduler::new(SchedulerOptions::default().with_threads(3)).unwrap());
    let before = thread_count();
    let handles: Vec<_> = (0..8)
        .map(|worker| {
            let scheduler = Arc::clone(&scheduler);
            std::thread::spawn(move || {
                for round in 0..3 {
                    let seed = worker * 3 + round;
                    let ours = pipeline(seed).with_scheduler(Arc::clone(&scheduler)).bytes().unwrap();
                    assert_eq!(ours, pipeline(seed).bytes_via_reference().unwrap(), "pipeline {seed}");
                    // Mid-run, the process holds the 8 callers and the 3
                    // shared workers: no run brought threads of its own.
                    if let (Some(before), Some(now)) = (before, thread_count()) {
                        assert!(now <= before + 8, "{now} threads, {before} before");
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
