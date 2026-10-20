//! libwebp's alpha-level reduction for sharp's `alphaQuality`: a lossy
//! image's alpha plane mapped to fewer distinct levels before it is coded,
//! lossy by design and much cheaper to code.

#![allow(
    clippy::indexing_slicing,
    reason = "indices are into 256-entry histograms by byte value, or into \
              `levels`-long centroid tables below their length"
)]

/// How many alpha levels libwebp keeps at `quality` below 100: 2 to 16 up
/// to 70, then up to 256.
pub fn alpha_levels(quality: u8) -> usize {
    let quality = usize::from(quality.min(100));
    if quality <= 70 {
        2 + quality / 5
    } else {
        16 + (quality - 70) * 8
    }
}

/// Reduce `data` to at most `levels` distinct values by libwebp's
/// `QuantizeLevels`: k-means over the histogram, from evenly spread
/// centroids, the lowest and highest values fixed.
pub fn quantize_levels(data: &mut [u8], levels: usize) {
    const ITERATIONS: usize = 6;
    if !(2..=256).contains(&levels) || data.is_empty() {
        return;
    }
    let mut freq = [0_u64; 256];
    for &v in data.iter() {
        freq[usize::from(v)] += 1;
    }
    if freq.iter().filter(|&&f| f > 0).count() <= levels {
        return;
    }
    let min = data.iter().copied().min().map_or(0, usize::from);
    let max = data.iter().copied().max().map_or(255, usize::from);

    let mut centre: Vec<f64> = (0..levels)
        .map(|i| min as f64 + (max - min) as f64 * i as f64 / (levels - 1) as f64)
        .collect();
    let mut level_of = [0_usize; 256];
    let threshold = 1e-4 * data.len() as f64;
    let mut last_error = f64::MAX;
    for _ in 0..ITERATIONS {
        let mut sum = vec![0.0_f64; levels];
        let mut count = vec![0.0_f64; levels];
        let mut slot = 0;
        for s in min..=max {
            // The nearest centroid, walking up as values rise.
            while slot < levels - 1 && 2.0 * s as f64 > centre[slot] + centre[slot + 1] {
                slot += 1;
            }
            if freq[s] > 0 {
                sum[slot] += (s as u64 * freq[s]) as f64;
                count[slot] += freq[s] as f64;
            }
            level_of[s] = slot;
        }
        // The two ends stay where they are.
        for slot in 1..levels - 1 {
            if count[slot] > 0.0 {
                centre[slot] = sum[slot] / count[slot];
            }
        }
        let error: f64 = (min..=max)
            .map(|s| {
                let e = s as f64 - centre[level_of[s]];
                freq[s] as f64 * e * e
            })
            .sum();
        if last_error - error < threshold {
            break;
        }
        last_error = error;
    }
    let map: Vec<u8> = (0..256)
        .map(|s| (centre[level_of[s]] + 0.5) as u8)
        .collect();
    for v in data.iter_mut() {
        *v = map[usize::from(*v)];
    }
}
