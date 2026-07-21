//! Palette quantization: true colour down to at most 256 entries.
//!
//! Median cut, with optional Floyd-Steinberg dithering. Both choices are
//! permanent: SPEC §Guarantees 2 promises byte-identical output across
//! versions, so changing the algorithm later would change pixels we said were
//! stable.
//!
//! # Why median cut
//!
//! It is deterministic with no seed, runs in one pass over a histogram, and is
//! what GIF encoders are expected to do — which matters, because a palette
//! that differs wildly from every other encoder's makes our output look wrong
//! rather than merely different.
//!
//! # Why dither by default
//!
//! Two hundred and fifty-six colours cannot represent a gradient. Without
//! dithering a photograph bands visibly, and every other encoder dithers, so
//! not doing it would make our output look worse for no reason. Flat-colour
//! art — the case GIF is still genuinely used for — is unaffected, because
//! when every colour is already in the palette the error is zero and there is
//! nothing to diffuse.

#![allow(
    clippy::indexing_slicing,
    reason = "every index here is into a three-element array or a \
              chunks_exact(3) slice, both of which the compiler already \
              knows the length of; get()/unwrap_or() would hide that rather \
              than prove anything"
)]

use std::collections::HashMap;

/// A colour table of at most 256 entries.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Palette {
    entries: Vec<[u8; 3]>,
}

impl Palette {
    /// Build a palette from explicit entries, truncated to 256.
    #[must_use]
    pub fn new(entries: Vec<[u8; 3]>) -> Self {
        let mut entries = entries;
        entries.truncate(256);
        Self { entries }
    }

    /// The entries, in index order.
    #[must_use]
    pub fn entries(&self) -> &[[u8; 3]] {
        &self.entries
    }

    /// How many entries the table holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the palette is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The bits per index a GIF colour table of this size needs.
    ///
    /// GIF stores the table size as `2^(n+1)`, so a table is always rounded up
    /// to a power of two with a minimum of two entries.
    #[must_use]
    pub fn code_bits(&self) -> u32 {
        let mut bits = 1_u32;
        while (1_usize << bits) < self.entries.len().max(2) {
            bits += 1;
        }
        bits.min(8)
    }

    /// The table padded to `2^code_bits` entries, as GIF requires.
    #[must_use]
    pub fn padded(&self) -> Vec<[u8; 3]> {
        let size = 1_usize << self.code_bits();
        let mut out = self.entries.clone();
        out.resize(size, [0, 0, 0]);
        out
    }

    /// The index of the entry closest to `colour`, by squared distance.
    ///
    /// Linear search over at most 256 entries. A k-d tree would be
    /// asymptotically better and is not worth it here: 256 is small enough
    /// that the linear scan vectorizes, and the tree's build cost lands on
    /// every image rather than only on large ones.
    #[must_use]
    pub fn nearest(&self, colour: [i32; 3]) -> u8 {
        let mut best = 0_usize;
        let mut best_distance = i32::MAX;
        for (index, entry) in self.entries.iter().enumerate() {
            let dr = colour[0] - i32::from(entry[0]);
            let dg = colour[1] - i32::from(entry[1]);
            let db = colour[2] - i32::from(entry[2]);
            let distance = dr * dr + dg * dg + db * db;
            if distance < best_distance {
                best_distance = distance;
                best = index;
                if distance == 0 {
                    break;
                }
            }
        }
        best as u8
    }
}

/// One box in the median-cut partition.
#[derive(Debug, Clone)]
struct Box3 {
    /// Distinct colours in this box, with their pixel counts.
    colours: Vec<([u8; 3], u32)>,
    /// Total pixels, which is what the split target is measured in.
    weight: u64,
}

impl Box3 {
    fn new(colours: Vec<([u8; 3], u32)>) -> Self {
        let weight = colours.iter().map(|&(_, count)| u64::from(count)).sum();
        Self { colours, weight }
    }

    /// The channel with the widest spread, and that spread.
    fn widest_channel(&self) -> (usize, u8) {
        let mut widest = 0;
        let mut spread = 0;
        for channel in 0..3 {
            let mut low = u8::MAX;
            let mut high = 0_u8;
            for &(colour, _) in &self.colours {
                let value = colour.get(channel).copied().unwrap_or(0);
                low = low.min(value);
                high = high.max(value);
            }
            let extent = high.saturating_sub(low);
            if extent > spread {
                spread = extent;
                widest = channel;
            }
        }
        (widest, spread)
    }

    /// Split at the weighted median of the widest channel.
    fn split(mut self) -> (Self, Self) {
        let (channel, _) = self.widest_channel();
        // Sorting by the channel, then by the whole colour, keeps the order
        // total — two colours equal on `channel` must not be ordered by
        // whichever the sort happened to see first, or the palette stops being
        // deterministic.
        self.colours
            .sort_by_key(|&(colour, _)| (colour.get(channel).copied().unwrap_or(0), colour));

        let half = self.weight / 2;
        let mut running = 0_u64;
        let mut at = 0_usize;
        for (index, &(_, count)) in self.colours.iter().enumerate() {
            running += u64::from(count);
            if running >= half {
                // Always leave at least one colour on each side, or the split
                // makes no progress and the loop that calls it never finishes.
                at = (index + 1).min(self.colours.len() - 1).max(1);
                break;
            }
        }
        let right = self.colours.split_off(at);
        (Self::new(self.colours), Self::new(right))
    }

    /// The pixel-count-weighted mean colour of this box.
    fn average(&self) -> [u8; 3] {
        let mut totals = [0_u64; 3];
        let mut count = 0_u64;
        for &(colour, weight) in &self.colours {
            for (slot, &value) in totals.iter_mut().zip(colour.iter()) {
                *slot += u64::from(value) * u64::from(weight);
            }
            count += u64::from(weight);
        }
        if count == 0 {
            return [0, 0, 0];
        }
        let mut out = [0_u8; 3];
        for (slot, &total) in out.iter_mut().zip(totals.iter()) {
            // Round rather than truncate: truncation biases every entry dark,
            // which on a large flat area is a visible shift.
            *slot = ((total + count / 2) / count).min(255) as u8;
        }
        out
    }
}

/// Build a palette of at most `size` entries for `pixels`.
///
/// `pixels` is RGB8, three bytes per pixel. Alpha is not considered: GIF
/// transparency is a palette index, so it is the caller's to reserve.
#[must_use]
pub fn build_palette(pixels: &[u8], size: usize) -> Palette {
    let size = size.clamp(2, 256);

    // Histogram first. Real images have far fewer distinct colours than
    // pixels, so this is what makes the rest cheap — and it is also what makes
    // the result depend on the image's colours rather than on its size.
    let mut histogram: HashMap<[u8; 3], u32> = HashMap::new();
    for pixel in pixels.chunks_exact(3) {
        let key = [pixel[0], pixel[1], pixel[2]];
        *histogram.entry(key).or_insert(0) += 1;
    }

    if histogram.is_empty() {
        return Palette::new(vec![[0, 0, 0]]);
    }

    let mut colours: Vec<([u8; 3], u32)> = histogram.into_iter().collect();
    // A HashMap iterates in an unspecified order, so sorting here is what
    // makes the palette deterministic rather than accidentally stable.
    colours.sort_unstable();

    if colours.len() <= size {
        // Fewer distinct colours than the palette holds: use them exactly, so
        // flat-colour art round-trips losslessly.
        return Palette::new(colours.into_iter().map(|(colour, _)| colour).collect());
    }

    let mut boxes = vec![Box3::new(colours)];
    while boxes.len() < size {
        // Split the heaviest box that still can be split. Choosing by weight
        // rather than by extent puts detail where the pixels are.
        let candidate = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colours.len() > 1)
            .max_by_key(|(_, b)| (b.weight, b.widest_channel().1))
            .map(|(index, _)| index);
        let Some(index) = candidate else {
            // Every box holds a single colour; nothing left to split.
            break;
        };
        let chosen = boxes.swap_remove(index);
        let (left, right) = chosen.split();
        boxes.push(left);
        boxes.push(right);
    }

    let mut entries: Vec<[u8; 3]> = boxes.iter().map(Box3::average).collect();
    // Sorting the final table keeps the output independent of the order boxes
    // happened to be split in, which `swap_remove` above does not preserve.
    entries.sort_unstable();
    entries.dedup();
    Palette::new(entries)
}

/// How colour error is handled when mapping pixels to a palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Dither {
    /// Diffuse the error to neighbouring pixels (Floyd-Steinberg).
    #[default]
    FloydSteinberg,
    /// Map each pixel independently to its nearest entry.
    None,
}

/// Map `pixels` onto `palette`, returning one index per pixel.
///
/// `pixels` is RGB8 and `width` is in pixels, which dithering needs in order
/// to find a pixel's neighbours.
#[must_use]
pub fn quantize(pixels: &[u8], width: usize, palette: &Palette, dither: Dither) -> Vec<u8> {
    let count = pixels.len() / 3;
    let mut out = vec![0_u8; count];
    if palette.is_empty() || width == 0 {
        return out;
    }

    match dither {
        Dither::None => {
            for (index, pixel) in pixels.chunks_exact(3).enumerate() {
                let colour = [
                    i32::from(pixel[0]),
                    i32::from(pixel[1]),
                    i32::from(pixel[2]),
                ];
                if let Some(slot) = out.get_mut(index) {
                    *slot = palette.nearest(colour);
                }
            }
        }
        Dither::FloydSteinberg => {
            // Two rows of running error: the current row and the next. Holding
            // the whole image's error would be no more accurate and would grow
            // with image size, which is the thing this engine does not do.
            let mut current = vec![0_i32; (width + 2) * 3];
            let mut next = vec![0_i32; (width + 2) * 3];
            let height = count / width;

            for y in 0..height {
                next.fill(0);
                for x in 0..width {
                    let index = y * width + x;
                    let Some(pixel) = pixels.get(index * 3..index * 3 + 3) else {
                        continue;
                    };
                    // The error buffers are offset by one so a pixel's left
                    // neighbour is always in range without a branch.
                    let at = (x + 1) * 3;
                    let mut wanted = [0_i32; 3];
                    for channel in 0..3 {
                        let carried = current.get(at + channel).copied().unwrap_or(0);
                        wanted[channel] = (i32::from(pixel[channel]) + carried).clamp(0, 255);
                    }

                    let chosen = palette.nearest(wanted);
                    if let Some(slot) = out.get_mut(index) {
                        *slot = chosen;
                    }
                    let entry = palette
                        .entries()
                        .get(chosen as usize)
                        .copied()
                        .unwrap_or([0, 0, 0]);

                    // Floyd-Steinberg weights: 7/16 right, 3/16 below-left,
                    // 5/16 below, 1/16 below-right.
                    for channel in 0..3 {
                        let error = wanted[channel] - i32::from(entry[channel]);
                        if error == 0 {
                            continue;
                        }
                        diffuse(&mut current, at + 3 + channel, error * 7 / 16);
                        diffuse(&mut next, at - 3 + channel, error * 3 / 16);
                        diffuse(&mut next, at + channel, error * 5 / 16);
                        diffuse(&mut next, at + 3 + channel, error / 16);
                    }
                }
                std::mem::swap(&mut current, &mut next);
            }
        }
    }
    out
}

/// Add `amount` to an error accumulator, ignoring out-of-range positions.
fn diffuse(row: &mut [i32], at: usize, amount: i32) {
    if let Some(slot) = row.get_mut(at) {
        *slot += amount;
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    /// An image with `n` distinct colours, one pixel each.
    fn distinct(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 3);
        for i in 0..n {
            out.extend_from_slice(&[(i % 256) as u8, ((i / 256) % 256) as u8, (i % 7) as u8]);
        }
        out
    }

    #[test]
    fn an_image_within_the_palette_size_is_represented_exactly() {
        // Flat-colour art is what GIF is still genuinely used for, and it must
        // round-trip losslessly rather than being approximated.
        let pixels = distinct(40);
        let palette = build_palette(&pixels, 256);
        let indices = quantize(&pixels, 40, &palette, Dither::None);
        for (index, pixel) in indices.iter().zip(pixels.chunks_exact(3)) {
            let entry = palette.entries()[*index as usize];
            assert_eq!(entry, [pixel[0], pixel[1], pixel[2]], "colour was altered");
        }
    }

    #[test]
    fn a_palette_never_exceeds_the_requested_size() {
        for size in [2_usize, 4, 16, 100, 256] {
            let pixels = distinct(1000);
            let palette = build_palette(&pixels, size);
            assert!(
                palette.len() <= size,
                "asked for {size}, got {}",
                palette.len()
            );
            assert!(!palette.is_empty());
        }
    }

    #[test]
    fn quantization_is_deterministic() {
        // SPEC §Guarantees 2. The histogram is a HashMap, whose iteration
        // order is unspecified, so this is the test that the sort actually
        // makes the result stable rather than accidentally reproducible.
        let pixels: Vec<u8> = (0..3000).map(|i| ((i * 37) % 251) as u8).collect();
        let first = build_palette(&pixels, 64);
        for _ in 0..8 {
            assert_eq!(build_palette(&pixels, 64), first, "palette is not stable");
        }
        let indices = quantize(&pixels, 100, &first, Dither::FloydSteinberg);
        for _ in 0..4 {
            assert_eq!(
                quantize(&pixels, 100, &first, Dither::FloydSteinberg),
                indices,
                "dithering is not stable"
            );
        }
    }

    #[test]
    fn a_gradient_quantizes_closely() {
        // The case 256 colours cannot represent exactly, and therefore the one
        // that shows whether the palette is placed sensibly.
        let mut pixels = Vec::new();
        for i in 0..256_u32 {
            for _ in 0..4 {
                let v = i as u8;
                pixels.extend_from_slice(&[v, v, v]);
            }
        }
        let palette = build_palette(&pixels, 64);
        let indices = quantize(&pixels, 32, &palette, Dither::None);

        let mut worst = 0_i32;
        for (index, pixel) in indices.iter().zip(pixels.chunks_exact(3)) {
            let entry = palette.entries()[*index as usize];
            worst = worst.max((i32::from(entry[0]) - i32::from(pixel[0])).abs());
        }
        assert!(
            worst <= 8,
            "worst error on a 64-entry grey ramp was {worst}"
        );
    }

    #[test]
    fn dithering_reduces_average_error_on_a_gradient() {
        // The reason dithering is on by default. Per-pixel error rises — that
        // is what diffusion does — but the average over a neighbourhood falls,
        // which is what the eye integrates.
        let width = 64_usize;
        let mut pixels = Vec::new();
        for _y in 0..64 {
            for x in 0..width {
                let v = (x * 255 / width) as u8;
                pixels.extend_from_slice(&[v, v, v]);
            }
        }
        let palette = build_palette(&pixels, 4);

        let mean_error = |dither: Dither| -> f64 {
            let indices = quantize(&pixels, width, &palette, dither);
            // Average over each row, which is what banding is visible against.
            let mut total = 0.0;
            for y in 0..64_usize {
                for block in 0..(width / 8) {
                    let mut want = 0.0;
                    let mut got = 0.0;
                    for x in block * 8..block * 8 + 8 {
                        let index = y * width + x;
                        want += f64::from(pixels[index * 3]);
                        got += f64::from(palette.entries()[indices[index] as usize][0]);
                    }
                    total += (want - got).abs() / 8.0;
                }
            }
            total
        };

        let dithered = mean_error(Dither::FloydSteinberg);
        let flat = mean_error(Dither::None);
        assert!(
            dithered < flat,
            "dithering did not reduce block-average error: {dithered} vs {flat}"
        );
    }

    #[test]
    fn a_flat_image_needs_no_dithering_and_gets_none() {
        // When every colour is already in the palette the error is zero, so
        // diffusion has nothing to spread and cannot introduce noise.
        let pixels = vec![77_u8; 300];
        let palette = build_palette(&pixels, 16);
        let indices = quantize(&pixels, 10, &palette, Dither::FloydSteinberg);
        assert!(
            indices.iter().all(|&i| i == indices[0]),
            "dithering introduced noise into a flat image"
        );
    }

    #[test]
    fn an_empty_image_is_not_a_panic() {
        let palette = build_palette(&[], 16);
        assert!(
            !palette.is_empty(),
            "a palette must have at least one entry"
        );
        assert!(quantize(&[], 0, &palette, Dither::FloydSteinberg).is_empty());
    }

    #[test]
    fn code_bits_round_up_to_a_power_of_two() {
        for (count, expected) in [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (256, 8)] {
            let palette = Palette::new(vec![[0, 0, 0]; count]);
            assert_eq!(palette.code_bits(), expected, "{count} entries");
            assert_eq!(palette.padded().len(), 1 << expected);
        }
    }

    #[test]
    fn nearest_finds_the_closest_entry() {
        let palette = Palette::new(vec![[0, 0, 0], [255, 255, 255], [255, 0, 0]]);
        assert_eq!(palette.nearest([10, 10, 10]), 0);
        assert_eq!(palette.nearest([250, 250, 250]), 1);
        assert_eq!(palette.nearest([200, 20, 20]), 2);
    }

    #[test]
    fn splitting_always_makes_progress() {
        // A box that splits into itself plus nothing would loop forever. The
        // pathological input is many pixels of one colour and one of another.
        let mut pixels = vec![0_u8; 3 * 1000];
        pixels.extend_from_slice(&[255, 255, 255]);
        let palette = build_palette(&pixels, 256);
        assert!(palette.len() >= 2, "the outlier colour was lost");
    }
}
