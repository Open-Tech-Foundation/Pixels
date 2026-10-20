//! [`Trim`] — find the box inside a border of near-background pixels
//! (sharp's `trim`, libvips' `find_trim`).
//!
//! Unlike every other op here, a trim's output shape depends on pixel
//! values, so it cannot be declared when the graph is built. This is
//! therefore not an [`Op`](otf_pixels_core::Op) but a search over pixels
//! already computed: the facade evaluates the pipeline so far, finds the box,
//! and crops to it.
//!
//! # What counts as background
//!
//! As in libvips: alpha is first flattened onto the background colour, a 3x3
//! median removes isolated noise (unless the image is line art, where a
//! one-pixel line is content), and a pixel is content when any colour channel
//! differs from the background by at least the threshold. sharp also searches
//! the alpha channel on its own and takes the union, so a transparent border
//! is trimmed even where the colour under it is not the background; so does
//! this.

use otf_pixels_core::{ChannelLayout, PixelsError, Region, Result, Tile};

use crate::convert::to_rgba;

/// Options for finding a trim box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trim {
    background: Option<[u8; 4]>,
    threshold: f32,
    line_art: bool,
}

impl Default for Trim {
    /// sharp's defaults: the top-left pixel is the background, the threshold
    /// is 10, and the image is not line art.
    fn default() -> Self {
        Self {
            background: None,
            threshold: 10.0,
            line_art: false,
        }
    }
}

impl Trim {
    /// Trim this RGBA colour rather than the top-left pixel's.
    #[must_use]
    pub const fn with_background(mut self, rgba: [u8; 4]) -> Self {
        self.background = Some(rgba);
        self
    }

    /// How far, in 8-bit units at any depth, a channel may differ from the
    /// background and still be background.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] unless `threshold` is finite
    /// and not negative.
    pub fn with_threshold(mut self, threshold: f32) -> Result<Self> {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(PixelsError::invalid_argument(
                "threshold",
                format!("must be finite and not negative, got {threshold}"),
            ));
        }
        self.threshold = threshold;
        Ok(self)
    }

    /// Skip the median, so a line one pixel wide counts as content (sharp's
    /// `lineArt`).
    #[must_use]
    pub const fn line_art(mut self, line_art: bool) -> Self {
        self.line_art = line_art;
        self
    }

    /// The box of `tile` that is not background, in the tile's coordinates,
    /// or `None` if every pixel is background (sharp then leaves the image
    /// alone).
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::InvalidArgument`] for an image narrower and
    /// shorter than 3 pixels, which sharp refuses too.
    pub fn find(&self, tile: &Tile<'_>) -> Result<Option<Region>> {
        let region = tile.region();
        if region.width < 3 && region.height < 3 {
            return Err(PixelsError::invalid_argument(
                "image",
                format!(
                    "an image to trim must be at least 3 pixels on a side, got {}x{}",
                    region.width, region.height
                ),
            ));
        }
        let format = tile.pixel();
        let bpp = format.bytes_per_pixel();
        let alpha = matches!(
            format.layout(),
            ChannelLayout::GrayAlpha | ChannelLayout::Rgba
        );

        // A row as background-flattened colour plus alpha, all 0..=1.
        let background = match self.background {
            Some(rgba) => rgba.map(|v| f32::from(v) / 255.0),
            None => {
                let first = tile
                    .row(region.y)
                    .and_then(|row| row.get(..bpp))
                    .ok_or_else(|| PixelsError::graph("trim was given an empty tile"))?;
                to_rgba(first, format)
            }
        };
        let flattened = |y: u32| -> Result<Vec<[f32; 4]>> {
            let row = tile
                .row(y)
                .ok_or_else(|| PixelsError::graph(format!("trim tile is missing row {y}")))?;
            Ok(row
                .chunks_exact(bpp)
                .map(|pixel| {
                    let [r, g, b, a] = to_rgba(pixel, format);
                    let over = |c: f32, under: f32| c * a + under * (1.0 - a);
                    [
                        over(r, background[0]),
                        over(g, background[1]),
                        over(b, background[2]),
                        a,
                    ]
                })
                .collect())
        };

        // A difference reaching the threshold is content. The small allowance
        // keeps an exact 8-bit difference from rounding below it.
        let level = self.threshold / 255.0 - 1e-6;
        let content = |pixel: [f32; 4]| {
            let differs = |c: f32, b: f32| (c - b).abs() >= level;
            let colour = pixel
                .iter()
                .zip(&background)
                .take(3)
                .any(|(&c, &b)| differs(c, b));
            colour || (alpha && differs(pixel[3], background[3]))
        };

        let (width, height) = (region.width as usize, region.height as usize);
        let mut found: Option<(usize, usize, usize, usize)> = None;
        let mut mark = |x: usize, y: usize| {
            found = Some(match found {
                None => (x, y, x, y),
                Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
            });
        };

        if self.line_art {
            for y in 0..height {
                for (x, &pixel) in flattened(region.y + y as u32)?.iter().enumerate() {
                    if content(pixel) {
                        mark(x, y);
                    }
                }
            }
        } else {
            // Three rows at a time, the edges repeated as libvips' median
            // does, so each row is flattened once.
            let last = height - 1;
            let mut window = [
                flattened(region.y)?,
                flattened(region.y)?,
                flattened(region.y + u32::from(last > 0))?,
            ];
            for y in 0..height {
                if y > 0 {
                    window.rotate_left(1);
                    window[2] = flattened(region.y + (y + 1).min(last) as u32)?;
                }
                for x in 0..width {
                    let columns = [x.saturating_sub(1), x, (x + 1).min(width - 1)];
                    let mut pixel = [0.0; 4];
                    for (channel, value) in pixel.iter_mut().enumerate() {
                        let mut values = [0.0_f32; 9];
                        for (slot, (row, column)) in values.iter_mut().zip(
                            window
                                .iter()
                                .flat_map(|row| columns.iter().map(move |&c| (row, c))),
                        ) {
                            *slot = row
                                .get(column)
                                .and_then(|p| p.get(channel))
                                .copied()
                                .unwrap_or(0.0);
                        }
                        values.sort_unstable_by(f32::total_cmp);
                        *value = values[4];
                    }
                    if content(pixel) {
                        mark(x, y);
                    }
                }
            }
        }

        Ok(found.map(|(x0, y0, x1, y1)| {
            Region::new(
                region.x + x0 as u32,
                region.y + y0 as u32,
                (x1 - x0 + 1) as u32,
                (y1 - y0 + 1) as u32,
            )
        }))
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
    use otf_pixels_core::{PixelFormat, TileBuf};

    /// A `width` x `height` image of `fill` with `paint` over the pixels it
    /// picks.
    fn image(
        width: u32,
        height: u32,
        format: PixelFormat,
        fill: &[u8],
        paint: impl Fn(u32, u32) -> Option<Vec<u8>>,
    ) -> TileBuf {
        let mut bytes = Vec::new();
        for y in 0..height {
            for x in 0..width {
                bytes.extend(paint(x, y).unwrap_or_else(|| fill.to_vec()));
            }
        }
        TileBuf::from_vec(Region::new(0, 0, width, height), format, bytes).unwrap()
    }

    fn find(trim: Trim, buf: &TileBuf) -> Option<Region> {
        trim.find(&buf.as_tile().unwrap()).unwrap()
    }

    #[test]
    fn a_block_on_a_plain_border_is_found() {
        let buf = image(12, 10, PixelFormat::Rgb8, &[250, 250, 250], |x, y| {
            ((3..7).contains(&x) && (2..6).contains(&y)).then(|| vec![20, 30, 40])
        });
        assert_eq!(find(Trim::default(), &buf), Some(Region::new(3, 2, 4, 4)));
    }

    #[test]
    fn a_lone_pixel_is_noise_unless_the_image_is_line_art() {
        let buf = image(9, 9, PixelFormat::Gray8, &[255], |x, y| {
            (x == 4 && y == 5).then(|| vec![0])
        });
        assert_eq!(find(Trim::default(), &buf), None);
        assert_eq!(
            find(Trim::default().line_art(true), &buf),
            Some(Region::new(4, 5, 1, 1))
        );
    }

    #[test]
    fn the_threshold_is_the_smallest_difference_that_counts() {
        let with = |value: u8| {
            image(5, 5, PixelFormat::Gray8, &[100], move |x, y| {
                (x >= 2 && y >= 2).then(|| vec![value])
            })
        };
        let trim = Trim::default().line_art(true);
        assert_eq!(find(trim, &with(109)), None, "9 is under the default 10");
        assert_eq!(find(trim, &with(110)), Some(Region::new(2, 2, 3, 3)));
        assert_eq!(find(trim, &with(91)), None, "differences are absolute");
        let loose = trim.with_threshold(30.0).unwrap();
        assert_eq!(find(loose, &with(110)), None);
        assert!(Trim::default().with_threshold(-1.0).is_err());
        assert!(Trim::default().with_threshold(f32::NAN).is_err());
    }

    #[test]
    fn an_explicit_background_replaces_the_top_left_pixel() {
        // The top-left pixel is content here, so only a given background
        // finds the black border.
        let buf = image(6, 6, PixelFormat::Rgb8, &[0, 0, 0], |x, y| {
            ((x == 0 && y == 0) || ((2..4).contains(&x) && (2..4).contains(&y)))
                .then(|| vec![255, 0, 0])
        });
        let trim = Trim::default().line_art(true);
        assert_eq!(find(trim, &buf), Some(Region::new(0, 0, 6, 6)));
        assert_eq!(
            find(trim.with_background([0, 0, 0, 255]), &buf),
            Some(Region::new(0, 0, 4, 4))
        );
    }

    #[test]
    fn a_transparent_border_is_trimmed_whatever_colour_it_hides() {
        let buf = image(8, 8, PixelFormat::Rgba8, &[0, 0, 0, 0], |x, y| {
            if (2..6).contains(&x) && (3..6).contains(&y) {
                Some(vec![0, 0, 0, 255])
            } else {
                // Invisible noise in the colour channels.
                Some(vec![(x * 40) as u8, (y * 30) as u8, 7, 0])
            }
        });
        assert_eq!(find(Trim::default(), &buf), Some(Region::new(2, 3, 4, 3)));
    }

    #[test]
    fn an_image_of_nothing_but_background_has_no_box() {
        let buf = image(
            4,
            4,
            PixelFormat::Rgba16,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            |_, _| None,
        );
        assert_eq!(find(Trim::default(), &buf), None);
    }

    #[test]
    fn wide_formats_measure_the_threshold_in_8_bit_units() {
        // 9.5 in 8-bit units is under the threshold; 16 is over it.
        let with = |value: u16| {
            image(
                4,
                4,
                PixelFormat::Gray16,
                &0_u16.to_ne_bytes(),
                move |x, _| (x == 3).then(|| value.to_ne_bytes().to_vec()),
            )
        };
        let trim = Trim::default().line_art(true);
        assert_eq!(find(trim, &with(9 * 257 + 128)), None);
        assert_eq!(find(trim, &with(16 * 257)), Some(Region::new(3, 0, 1, 4)));
    }

    #[test]
    fn a_tiny_image_is_refused_as_sharp_refuses_it() {
        let buf = image(2, 2, PixelFormat::Gray8, &[0], |_, _| None);
        assert!(Trim::default().find(&buf.as_tile().unwrap()).is_err());
        // One long side is enough.
        let buf = image(1, 5, PixelFormat::Gray8, &[0], |_, _| None);
        assert!(Trim::default().find(&buf.as_tile().unwrap()).is_ok());
    }
}
