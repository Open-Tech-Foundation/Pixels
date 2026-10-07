//! Outputs that name no scheduler share the process-wide one.
//!
//! This is the default a server or runtime gets without writing any code, so
//! it is checked the way such a host runs: many requests at once, each a plain
//! `output(...).bytes()`. It lives in its own test binary because it counts
//! the process's threads, which another test running alongside would disturb.

#![cfg(all(feature = "png", feature = "raw"))]
#![allow(clippy::unwrap_used, reason = "tests operate on known-good values")]

use otf_pixels::{
    EncodeOptions, Fit, Format, Image, ImageDescriptor, PixelFormat, ResizeOptions, Scheduler,
};

fn pipeline(seed: u32) -> otf_pixels::Output {
    let (w, h) = (300 + seed * 7, 200 + seed * 3);
    let pixels: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * (seed + 3)) % 251) as u8)
        .collect();
    Image::from_raw(
        ImageDescriptor::new(w, h, PixelFormat::Rgb8).unwrap(),
        pixels,
    )
    .unwrap()
    .resize_with(97, 61, ResizeOptions::default().with_fit(Fit::Cover))
    .blur(1.5)
    .output(Format::Png, EncodeOptions::default())
}

/// Threads in this process, where the platform says.
fn thread_count() -> Option<usize> {
    std::fs::read_dir("/proc/self/task")
        .ok()
        .map(Iterator::count)
}

#[test]
fn concurrent_default_outputs_share_the_global_scheduler() {
    // Built up front so its workers are already counted in `before`.
    let global = Scheduler::global().unwrap();
    let before = thread_count();
    let callers = 8;
    let handles: Vec<_> = (0..callers)
        .map(|caller| {
            std::thread::spawn(move || {
                for round in 0..3 {
                    let seed = caller * 3 + round;
                    let ours = pipeline(seed).bytes().unwrap();
                    assert_eq!(
                        ours,
                        pipeline(seed).bytes_via_reference().unwrap(),
                        "pipeline {seed}"
                    );
                    // Mid-run, the process holds the callers and the global
                    // workers: no run brought a pool of its own.
                    if let (Some(before), Some(now)) = (before, thread_count()) {
                        assert!(
                            now <= before + callers as usize,
                            "{now} threads, {before} before"
                        );
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    assert!(std::sync::Arc::ptr_eq(
        &global,
        &Scheduler::global().unwrap()
    ));

    // `threads` still asks for a pool of the run's own, after the count
    // above so its workers cannot disturb it, and gives the same pixels.
    assert_eq!(
        pipeline(5).threads(2).bytes().unwrap(),
        pipeline(5).bytes().unwrap()
    );
}
