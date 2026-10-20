//! [`Composite`] — draw one image over another (sharp's `composite`).
//!
//! This is the first op with two inputs, which makes it the first place
//! `Op::arity` and the two-element `input_regions` vector do real work.
//!
//! # Placement
//!
//! An overlay is drawn at an offset, at one of sharp's nine gravities, or
//! tiled across the whole base ([`Placement`]). Gravity and tiling depend on
//! both images' sizes, so like [`crate::Resize`] an instance binds them when
//! it is chained and belongs to that one graph edge.
//!
//! # Alpha
//!
//! SPEC §Formats: alpha is unassociated at API boundaries. Blending is
//! therefore computed in premultiplied form internally and converted back on
//! the way out, because straight-alpha compositing has no correct one-line
//! form — the naive `f*a + b*(1-a)` is only right when the backdrop is opaque.
//!
//! Getting this wrong shows up as a dark halo around soft edges, which is the
//! single most common compositing bug and looks like a bad matte rather than
//! like arithmetic.
//!
//! # Blend modes
//!
//! sharp's, which are libvips', which are Cairo's operators: the Porter-Duff
//! set and the separable PDF blend modes, with the formulas of
//! <https://www.cairographics.org/operators/>. A mode acts only where the
//! overlay lies: outside it the base is untouched, even for modes such as
//! [`Blend::Clear`] that would erase it if the overlay were extended with
//! transparency.

use std::sync::OnceLock;

use otf_pixels_core::{
    ChannelLayout, ImageDescriptor, Op, PixelFormat, PixelsError, Region, Result, SampleKind, Tile,
    TileMut,
};

use crate::convert::{from_rgba, to_rgba};
use crate::resize::Position;

/// How the overlay is combined with the base: sharp's `blend`.
///
/// In the formulas, `A` is the overlay and `B` the base.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Blend {
    /// Nothing: the area under the overlay becomes transparent.
    Clear,
    /// The overlay replaces the base, alpha included.
    Source,
    /// Porter-Duff source-over: the overlay is drawn on top of the base.
    #[default]
    Over,
    /// The overlay where the base is, transparent elsewhere.
    In,
    /// The overlay where the base is not.
    Out,
    /// The overlay over the base, kept to the base's alpha.
    Atop,
    /// The base alone: the overlay is ignored.
    Dest,
    /// The base over the overlay.
    DestOver,
    /// The base where the overlay is.
    DestIn,
    /// The base where the overlay is not: an overlay used as an eraser.
    DestOut,
    /// The base over the overlay, kept to the overlay's alpha.
    DestAtop,
    /// Each where the other is not.
    Xor,
    /// Colours and alphas summed.
    Add,
    /// The overlay added only as far as the base has room for it.
    Saturate,
    /// Colours multiplied: darkens.
    Multiply,
    /// Inverted colours multiplied: lightens.
    Screen,
    /// Multiply or screen, chosen by the base's colour.
    Overlay,
    /// The darker of the two colours.
    Darken,
    /// The lighter of the two colours.
    Lighten,
    /// The base brightened to reflect the overlay.
    ColourDodge,
    /// The base darkened to reflect the overlay.
    ColourBurn,
    /// Multiply or screen, chosen by the overlay's colour.
    HardLight,
    /// A softer [`Blend::HardLight`].
    SoftLight,
    /// The absolute difference of the colours.
    Difference,
    /// [`Blend::Difference`] with lower contrast.
    Exclusion,
}

impl Blend {
    /// Whether an opaque base stays opaque under this mode wherever an
    /// overlay with (`overlay_alpha`) or without alpha lands. When it does
    /// not, an opaque base format gains alpha.
    const fn keeps_opaque(self, overlay_alpha: bool) -> bool {
        match self {
            Self::Clear | Self::Out | Self::DestOut | Self::Xor => false,
            Self::Source | Self::In | Self::DestIn | Self::DestAtop => !overlay_alpha,
            _ => true,
        }
    }

    /// A PDF mode's mix of overlay colour `a` and base colour `b`, both
    /// straight, 0..=1.
    fn mix(self, a: f32, b: f32) -> f32 {
        let multiply = |x: f32, y: f32| x * y;
        let screen = |x: f32, y: f32| x + y - x * y;
        let hard_light = |source: f32, backdrop: f32| {
            if source <= 0.5 {
                multiply(backdrop, 2.0 * source)
            } else {
                screen(backdrop, 2.0 * source - 1.0)
            }
        };
        match self {
            Self::Multiply => multiply(a, b),
            Self::Screen => screen(a, b),
            Self::Overlay => hard_light(b, a),
            Self::Darken => a.min(b),
            Self::Lighten => a.max(b),
            Self::ColourDodge => {
                if b <= 0.0 {
                    0.0
                } else if a >= 1.0 {
                    1.0
                } else {
                    (b / (1.0 - a)).min(1.0)
                }
            }
            Self::ColourBurn => {
                if b >= 1.0 {
                    1.0
                } else if a <= 0.0 {
                    0.0
                } else {
                    1.0 - ((1.0 - b) / a).min(1.0)
                }
            }
            Self::HardLight => hard_light(a, b),
            Self::SoftLight => {
                if a <= 0.5 {
                    b - (1.0 - 2.0 * a) * b * (1.0 - b)
                } else {
                    let d = if b <= 0.25 {
                        ((16.0 * b - 12.0) * b + 4.0) * b
                    } else {
                        b.sqrt()
                    };
                    b + (2.0 * a - 1.0) * (d - b)
                }
            }
            Self::Difference => (a - b).abs(),
            Self::Exclusion => a + b - 2.0 * a * b,
            _ => a,
        }
    }

    /// Blend straight RGBA `a` (overlay) onto `b` (base), giving straight
    /// RGBA.
    fn apply(self, a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
        let (aa, ab) = (a[3], b[3]);
        // Each mode is its result alpha and, per channel, its premultiplied
        // colour from the straight colours `x` (overlay) and `y` (base).
        let alpha = match self {
            Self::Clear => 0.0,
            Self::Source | Self::DestAtop => aa,
            Self::In | Self::DestIn => aa * ab,
            Self::Out => aa * (1.0 - ab),
            Self::Atop | Self::Dest => ab,
            Self::DestOut => (1.0 - aa) * ab,
            Self::Xor => aa + ab - 2.0 * aa * ab,
            Self::Add | Self::Saturate => (aa + ab).min(1.0),
            // Over, DestOver and the PDF modes.
            _ => aa + ab * (1.0 - aa),
        };
        let colour = |x: f32, y: f32| -> f32 {
            match self {
                Self::Clear => 0.0,
                Self::Source => x * aa,
                Self::Over => x * aa + y * ab * (1.0 - aa),
                Self::In => x * aa * ab,
                Self::Out => x * aa * (1.0 - ab),
                Self::Atop => x * aa * ab + y * ab * (1.0 - aa),
                Self::Dest => y * ab,
                Self::DestOver => y * ab + x * aa * (1.0 - ab),
                Self::DestIn => y * ab * aa,
                Self::DestOut => y * ab * (1.0 - aa),
                Self::DestAtop => y * ab * aa + x * aa * (1.0 - ab),
                Self::Xor => x * aa * (1.0 - ab) + y * ab * (1.0 - aa),
                Self::Add => x * aa + y * ab,
                Self::Saturate => x * aa.min(1.0 - ab) + y * ab,
                _ => {
                    x * aa * (1.0 - ab)
                        + y * ab * (1.0 - aa)
                        + aa * ab * self.mix(x.clamp(0.0, 1.0), y.clamp(0.0, 1.0))
                }
            }
        };
        if alpha <= 0.0 {
            // Fully transparent: colour is undefined, so it is zeroed rather
            // than divided by zero.
            return [0.0; 4];
        }
        let straight = |x: f32, y: f32| (colour(x, y) / alpha).clamp(0.0, 1.0);
        [
            straight(a[0], b[0]),
            straight(a[1], b[1]),
            straight(a[2], b[2]),
            alpha.clamp(0.0, 1.0),
        ]
    }
}

/// Where the overlay goes: sharp's `left`/`top`, `gravity` and `tile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Placement {
    /// The overlay's top-left corner at `(x, y)` in base coordinates.
    /// Negative offsets clip rather than fail.
    At(i64, i64),
    /// Against an edge, a corner or the middle of the base, as sharp's
    /// `gravity` places it. An overlay larger than the base overhangs it
    /// evenly and is clipped.
    Gravity(Position),
    /// Repeated over the whole base, aligned as sharp aligns `tile` with a
    /// `gravity`: centred, for instance, puts one copy in the middle.
    Tiled(Position),
    /// Repeated over the whole base, the base's top-left pixel showing the
    /// overlay's pixel at `(x, y)` (taken modulo its size), as sharp's
    /// `tile` with `left` and `top` does.
    TiledFrom(i64, i64),
}

impl Default for Placement {
    /// Centred, sharp's default gravity.
    fn default() -> Self {
        Self::Gravity(Position::Centre)
    }
}

/// The sizes a composite was chained with, and what they resolve to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Bound {
    base: (u32, u32),
    overlay: (u32, u32),
    /// The overlay's offset for a single copy, or the overlay pixel shown at
    /// the base's origin for a tiled one.
    at: (i64, i64),
    tiled: bool,
}

/// The near edge, the middle or the far edge of a `spare` span, for gravity
/// along one axis: `near` and `far` say which edge the gravity names.
/// `round_up` follows sharp, which centres a tile crop rounding up and an
/// overlay rounding toward zero.
const fn along(spare: i64, near: bool, far: bool, round_up: bool) -> i64 {
    if near {
        0
    } else if far {
        spare
    } else if round_up {
        (spare + 1) / 2
    } else {
        spare / 2
    }
}

/// Which edges a gravity names: `(west, east, north, south)`.
const fn edges(position: Position) -> (bool, bool, bool, bool) {
    match position {
        Position::Centre => (false, false, false, false),
        Position::North => (false, false, true, false),
        Position::NorthEast => (false, true, true, false),
        Position::East => (false, true, false, false),
        Position::SouthEast => (false, true, false, true),
        Position::South => (false, false, false, true),
        Position::SouthWest => (true, false, false, true),
        Position::West => (true, false, false, false),
        Position::NorthWest => (true, false, true, false),
    }
}

/// The phase of a tiled overlay along one axis: which overlay pixel the
/// base's first pixel shows. sharp replicates the overlay enough times to
/// cover the base — an odd number when centred on this axis, so one copy
/// sits in the middle — then crops the result to the base by the gravity.
fn tile_phase(base: u32, overlay: u32, near: bool, far: bool) -> i64 {
    let (base, overlay) = (i64::from(base), i64::from(overlay));
    let copies = if overlay <= base {
        let copies = (base + overlay - 1) / overlay;
        if near || far { copies } else { copies | 1 }
    } else {
        1
    };
    along(copies * overlay - base, near, far, true).rem_euclid(overlay)
}

impl Bound {
    fn new(placement: Placement, base: &ImageDescriptor, overlay: &ImageDescriptor) -> Self {
        let (bw, bh) = (base.width, base.height);
        let (ow, oh) = (overlay.width, overlay.height);
        let (at, tiled) = match placement {
            Placement::At(x, y) => ((x, y), false),
            Placement::Gravity(position) => {
                let (west, east, north, south) = edges(position);
                let spare_x = i64::from(bw) - i64::from(ow);
                let spare_y = i64::from(bh) - i64::from(oh);
                (
                    (
                        along(spare_x, west, east, false),
                        along(spare_y, north, south, false),
                    ),
                    false,
                )
            }
            Placement::Tiled(position) => {
                let (west, east, north, south) = edges(position);
                (
                    (
                        tile_phase(bw, ow, west, east),
                        tile_phase(bh, oh, north, south),
                    ),
                    true,
                )
            }
            Placement::TiledFrom(x, y) => (
                (x.rem_euclid(i64::from(ow)), y.rem_euclid(i64::from(oh))),
                true,
            ),
        };
        Self {
            base: (bw, bh),
            overlay: (ow, oh),
            at,
            tiled,
        }
    }

    /// The overlay coordinate base coordinate `at` shows along one axis,
    /// or `None` where a single copy does not reach.
    fn source(&self, at: u32, offset: i64, size: u32) -> Option<u32> {
        let i = i64::from(at);
        let mapped = if self.tiled {
            (i + offset).rem_euclid(i64::from(size))
        } else {
            let j = i - offset;
            if j < 0 || j >= i64::from(size) {
                return None;
            }
            j
        };
        u32::try_from(mapped).ok()
    }

    /// The overlay span base span `from..to` reads along one axis.
    fn span(&self, from: u32, to: u32, offset: i64, size: u32) -> Option<(u32, u32)> {
        if !self.tiled {
            let start = (i64::from(from) - offset).max(0);
            let end = (i64::from(to) - offset).min(i64::from(size));
            return (start < end).then_some((start as u32, end as u32));
        }
        if to - from >= size {
            return Some((0, size));
        }
        // A span shorter than the overlay wraps at most once.
        let start = (i64::from(from) + offset).rem_euclid(i64::from(size)) as u32;
        let end = start + (to - from);
        Some(if end <= size { (start, end) } else { (0, size) })
    }

    /// The part of the overlay output `region` reads, in overlay
    /// coordinates, or `None` if the overlay misses it.
    fn overlay_region(&self, region: Region) -> Option<Region> {
        let (x0, x1) = self.span(region.x, region.x + region.width, self.at.0, self.overlay.0)?;
        let (y0, y1) = self.span(
            region.y,
            region.y + region.height,
            self.at.1,
            self.overlay.1,
        )?;
        Some(Region::new(x0, y0, x1 - x0, y1 - y0))
    }
}

/// Draw an overlay over a base.
///
/// Input order is `[base, overlay]`, matching the reading order of "composite
/// this over that". The two may differ in format: the result has the base's
/// depth, gains colour when the overlay has it and the base is grey, and
/// gains alpha when the blend can make an opaque base translucent.
#[derive(Debug)]
pub struct Composite {
    placement: Placement,
    blend: Blend,
    bound: OnceLock<Bound>,
}

impl Clone for Composite {
    fn clone(&self) -> Self {
        // Not the binding: a clone may be chained onto other images.
        Self::new(self.placement, self.blend)
    }
}

impl Composite {
    /// Draw the overlay where `placement` says, combined by `blend`.
    #[must_use]
    pub const fn new(placement: Placement, blend: Blend) -> Self {
        Self {
            placement,
            blend,
            bound: OnceLock::new(),
        }
    }

    /// Draw the overlay with its top-left corner at `(x, y)` in base
    /// coordinates.
    ///
    /// Negative offsets are allowed: the overlay is clipped to the base rather
    /// than rejected, which is what makes "centre a watermark" expressible
    /// without the caller doing the clipping arithmetic.
    #[must_use]
    pub const fn at(x: i64, y: i64, blend: Blend) -> Self {
        Self::new(Placement::At(x, y), blend)
    }

    /// Draw the overlay over the base at the origin with source-over.
    #[must_use]
    pub const fn over() -> Self {
        Self::at(0, 0, Blend::Over)
    }

    /// Where the overlay goes.
    #[must_use]
    pub const fn placement(&self) -> Placement {
        self.placement
    }

    /// The blend mode.
    #[must_use]
    pub const fn blend(&self) -> Blend {
        self.blend
    }

    /// Bind to the inputs' sizes, or check they match those already bound.
    fn bind(&self, base: &ImageDescriptor, overlay: &ImageDescriptor) -> Result<&Bound> {
        let candidate = Bound::new(self.placement, base, overlay);
        let bound = self.bound.get_or_init(|| candidate);
        if (bound.base, bound.overlay) != (candidate.base, candidate.overlay) {
            return Err(PixelsError::graph(format!(
                "`composite` is bound to a {}x{} base and a {}x{} overlay; \
                 build a new Composite per pair of images",
                bound.base.0, bound.base.1, bound.overlay.0, bound.overlay.1
            )));
        }
        Ok(bound)
    }

    /// The output format for a base in `base` and an overlay in `overlay`.
    fn output_format(&self, base: PixelFormat, overlay: PixelFormat) -> Result<PixelFormat> {
        let has_alpha = |layout| matches!(layout, ChannelLayout::GrayAlpha | ChannelLayout::Rgba);
        let colour = |layout| matches!(layout, ChannelLayout::Rgb | ChannelLayout::Rgba);
        let (b, o) = (base.layout(), overlay.layout());
        let alpha = has_alpha(b) || !self.blend.keeps_opaque(has_alpha(o));
        let layout = match (colour(b) || colour(o), alpha) {
            (false, false) => ChannelLayout::Gray,
            (false, true) => ChannelLayout::GrayAlpha,
            (true, false) => ChannelLayout::Rgb,
            (true, true) => ChannelLayout::Rgba,
        };
        PixelFormat::from_parts(layout, base.sample_kind()).ok_or_else(|| {
            PixelsError::unsupported(format!(
                "composite has no {layout:?} format at the depth of {base}"
            ))
        })
    }
}

impl Op for Composite {
    /// Never rescaled, for two independent reasons: the overlay position is in
    /// pixels, and the second input is a separate image that would not be
    /// reduced alongside the first.
    fn rescaled(&self) -> Option<std::sync::Arc<dyn Op>> {
        None
    }
    fn name(&self) -> &'static str {
        "composite"
    }

    fn arity(&self) -> usize {
        2
    }

    fn output_descriptor(&self, inputs: &[ImageDescriptor]) -> Result<ImageDescriptor> {
        let (base, overlay) = pair(inputs)?;
        self.bind(base, overlay)?;
        // The output is the base's shape: compositing draws *onto* something,
        // so the backdrop defines the canvas.
        let mut out = *base;
        out.pixel = self.output_format(base.pixel, overlay.pixel)?;
        Ok(out)
    }

    fn input_regions(&self, output: Region, inputs: &[ImageDescriptor]) -> Result<Vec<Region>> {
        let (base, overlay) = pair(inputs)?;
        let bound = self.bind(base, overlay)?;
        // The base is needed wherever the output is. The overlay is needed only
        // where it actually lands — a watermark in the corner must not make the
        // scheduler pull the whole overlay for every tile.
        let wanted = bound
            .overlay_region(output)
            // An empty region still has to be a valid one, so a miss is
            // reported as a 1x1 at the origin rather than a zero-sized region
            // the tile machinery would reject.
            .unwrap_or_else(|| Region::new(0, 0, 1, 1));
        Ok(vec![output, wanted])
    }

    fn compute(&self, inputs: &[Tile<'_>], output: &mut TileMut<'_>) -> Result<()> {
        let base = inputs
            .first()
            .ok_or_else(|| PixelsError::graph("`composite` needs a base tile"))?;
        let overlay = inputs
            .get(1)
            .ok_or_else(|| PixelsError::graph("`composite` needs an overlay tile"))?;
        let bound = self
            .bound
            .get()
            .ok_or_else(|| PixelsError::graph("`composite` computed before it was chained"))?;

        let region = output.region();
        let area = overlay.region();
        let (from, over, to) = (base.pixel(), overlay.pixel(), output.pixel());
        let (from_bytes, over_bytes, to_bytes) = (
            from.bytes_per_pixel(),
            over.bytes_per_pixel(),
            to.bytes_per_pixel(),
        );
        // Source-over between 8-bit images with the same colour channels,
        // either with or without alpha, runs in integers: that is the common
        // watermark, an RGBA PNG over an RGB or RGBA photo. Everything else
        // goes through RGBA floats.
        let colour = |format: PixelFormat| {
            matches!(format.layout(), ChannelLayout::Rgb | ChannelLayout::Rgba)
        };
        let integer = self.blend == Blend::Over
            && from == to
            && from.sample_kind() == SampleKind::U8
            && over.sample_kind() == SampleKind::U8
            && colour(from) == colour(over);
        let colours = if colour(from) { 3 } else { 1 };

        // Where each output column reads the overlay, as a byte offset into an
        // overlay row, or `None` where it misses. The same for every row.
        let reached = bound.overlay_region(region).is_some();
        let columns: Vec<Option<usize>> = (region.x..region.x + region.width)
            .map(|x| {
                bound
                    .source(x, bound.at.0, bound.overlay.0)
                    .filter(|_| reached)
                    .map(|sx| (sx - area.x) as usize * over_bytes)
            })
            .collect();

        for y in region.y..region.y + region.height {
            let source = base
                .row(y)
                .ok_or_else(|| PixelsError::graph(format!("composite base is missing row {y}")))?;
            let over_row = match bound
                .source(y, bound.at.1, bound.overlay.1)
                .filter(|_| reached)
            {
                Some(sy) => Some(overlay.row(sy).ok_or_else(|| {
                    PixelsError::graph(format!("composite overlay {area} is missing row {sy}"))
                })?),
                None => None,
            };
            let target = output.row_mut(y).ok_or_else(|| {
                PixelsError::graph(format!("composite output is missing row {y}"))
            })?;

            // The base shows through wherever the overlay is not, so in the
            // same format it is copied a row at a time and only the pixels
            // under the overlay are visited below.
            let same = from == to;
            if same {
                let len = target.len().min(source.len());
                if let (Some(to), Some(from)) = (target.get_mut(..len), source.get(..len)) {
                    to.copy_from_slice(from);
                }
                if over_row.is_none() {
                    continue;
                }
            }

            let pixels = source
                .chunks_exact(from_bytes)
                .zip(target.chunks_exact_mut(to_bytes))
                .zip(&columns);
            for ((backdrop, slot), column) in pixels {
                let drawn = over_row
                    .zip(*column)
                    .and_then(|(row, at)| row.get(at..at + over_bytes));
                match drawn {
                    None if same => {}
                    None => from_rgba(to_rgba(backdrop, from), slot, to),
                    Some(drawn) if integer => over_8_bit(drawn, backdrop, slot, colours),
                    // Replacement in the same format is a copy, which keeps
                    // even the colour under zero alpha.
                    Some(drawn) if self.blend == Blend::Source && over == to => {
                        slot.copy_from_slice(drawn);
                    }
                    Some(drawn) => {
                        let blended = self
                            .blend
                            .apply(to_rgba(drawn, over), to_rgba(backdrop, from));
                        from_rgba(blended, slot, to);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Source-over for one 8-bit pixel with `colours` colour channels, in
/// integers. The overlay and the backdrop (whose format `target` is in)
/// each have an alpha after their colours or not.
fn over_8_bit(over: &[u8], backdrop: &[u8], target: &mut [u8], colours: usize) {
    let sa = u32::from(over.get(colours).copied().unwrap_or(255));
    if sa == 255 {
        // An opaque pixel replaces whatever is under it.
        for (slot, &value) in target.iter_mut().zip(over).take(colours) {
            *slot = value;
        }
        if let Some(slot) = target.get_mut(colours) {
            *slot = 255;
        }
        return;
    }
    let alpha = colours;
    let ba = u32::from(backdrop.get(alpha).copied().unwrap_or(255));
    // Porter-Duff over, on unassociated alpha:
    //   out_a = sa + ba*(1-sa)
    //   out_c = (sc*sa + bc*ba*(1-sa)) / out_a
    // Computing it premultiplied and dividing back out is what avoids the
    // dark halo the naive form produces over a translucent backdrop.
    let inverse = 255 - sa;
    let out_a = sa * 255 + ba * inverse;
    if out_a == 0 {
        // Fully transparent result: colour is undefined, so leave it zeroed
        // rather than dividing by zero.
        target.fill(0);
        return;
    }
    for channel in 0..alpha {
        let sc = u32::from(over.get(channel).copied().unwrap_or(0));
        let bc = u32::from(backdrop.get(channel).copied().unwrap_or(0));
        let numerator = sc * sa * 255 + bc * ba * inverse;
        if let Some(slot) = target.get_mut(channel) {
            *slot = ((numerator + out_a / 2) / out_a).min(255) as u8;
        }
    }
    if let Some(slot) = target.get_mut(alpha) {
        *slot = ((out_a + 127) / 255).min(255) as u8;
    }
}

/// Fetch the two input descriptors a composite was given.
fn pair(inputs: &[ImageDescriptor]) -> Result<(&ImageDescriptor, &ImageDescriptor)> {
    match (inputs.first(), inputs.get(1)) {
        (Some(base), Some(overlay)) => Ok((base, overlay)),
        _ => Err(PixelsError::graph(format!(
            "`composite` takes two inputs, got {}",
            inputs.len()
        ))),
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
    use otf_pixels_core::TileBuf;

    fn solid(
        width: u32,
        height: u32,
        format: PixelFormat,
        pixel: &[u8],
    ) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let mut bytes = Vec::with_capacity(descriptor.byte_len().unwrap());
        for _ in 0..(width * height) {
            bytes.extend_from_slice(pixel);
        }
        (descriptor, bytes)
    }

    /// Composite two whole images.
    fn apply(
        op: &Composite,
        base: (&ImageDescriptor, &[u8]),
        overlay: (&ImageDescriptor, &[u8]),
    ) -> Result<(ImageDescriptor, Vec<u8>)> {
        let inputs = [*base.0, *overlay.0];
        let out_desc = op.output_descriptor(&inputs)?;
        let base_buf = TileBuf::from_vec(base.0.region(), base.0.pixel, base.1.to_vec())?;
        let over_buf = TileBuf::from_vec(overlay.0.region(), overlay.0.pixel, overlay.1.to_vec())?;
        let mut target = TileBuf::for_image(&out_desc)?;
        op.compute(
            &[base_buf.as_tile()?, over_buf.as_tile()?],
            &mut target.as_tile_mut()?,
        )?;
        Ok((out_desc, target.into_bytes()))
    }

    #[test]
    fn an_opaque_overlay_replaces_the_backdrop() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[10, 20, 30, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[200, 100, 50, 255]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel, [200, 100, 50, 255], "opaque overlay did not replace");
        }
    }

    #[test]
    fn a_fully_transparent_overlay_leaves_the_backdrop_alone() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[10, 20, 30, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[200, 100, 50, 0]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(
                pixel,
                [10, 20, 30, 255],
                "transparent overlay changed the base"
            );
        }
    }

    #[test]
    fn half_alpha_over_an_opaque_backdrop_is_the_midpoint() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[0, 0, 0, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel[3], 255, "an opaque backdrop must stay opaque");
            assert_eq!(pixel[0], 128, "expected the midpoint, got {pixel:?}");
        }
    }

    #[test]
    fn compositing_onto_a_translucent_backdrop_does_not_darken() {
        // The dark-halo bug: the naive `f*a + b*(1-a)` is only correct over an
        // opaque backdrop. Over a translucent one it under-weights the
        // backdrop's colour and the result creeps toward black.
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            // White over white is white at any alpha. The naive formula gives
            // 191 here instead.
            assert_eq!(
                &pixel[..3],
                [255, 255, 255],
                "white over white darkened to {pixel:?}"
            );
            assert!(pixel[3] > 128, "alpha should accumulate, got {}", pixel[3]);
        }
    }

    #[test]
    fn two_transparent_pixels_do_not_divide_by_zero() {
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgba8, &[9, 9, 9, 0]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[7, 7, 7, 0]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel[3], 0, "result should stay transparent");
        }
    }

    #[test]
    fn the_overlay_is_placed_at_its_offset() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[0, 0, 0, 255]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 0, 0, 255]);
        let op = Composite::at(1, 1, Blend::Over);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();

        let pixel_at = |x: usize, y: usize| -> &[u8] { &out[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4] };
        assert_eq!(pixel_at(0, 0), [0, 0, 0, 255], "outside the overlay");
        assert_eq!(pixel_at(1, 1), [255, 0, 0, 255], "inside the overlay");
        assert_eq!(pixel_at(2, 2), [255, 0, 0, 255], "inside the overlay");
        assert_eq!(pixel_at(3, 3), [0, 0, 0, 255], "outside the overlay");
    }

    #[test]
    fn a_negative_offset_clips_rather_than_failing() {
        // "Centre a watermark larger than the base" must work without the
        // caller doing the clipping arithmetic.
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[0, 0, 0, 255]);
        let (over_desc, over) = solid(4, 4, PixelFormat::Rgba8, &[255, 0, 0, 255]);
        let op = Composite::at(-2, -2, Blend::Over);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();

        let pixel_at = |x: usize, y: usize| -> &[u8] { &out[(y * 4 + x) * 4..(y * 4 + x) * 4 + 4] };
        assert_eq!(
            pixel_at(0, 0),
            [255, 0, 0, 255],
            "clipped overlay should cover here"
        );
        assert_eq!(pixel_at(3, 3), [0, 0, 0, 255], "beyond the clipped overlay");
    }

    #[test]
    fn an_overlay_entirely_outside_the_base_changes_nothing() {
        let (base_desc, base) = solid(4, 4, PixelFormat::Rgba8, &[1, 2, 3, 255]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 0, 0, 255]);
        let op = Composite::at(100, 100, Blend::Over);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();
        assert_eq!(out, base, "a missed overlay changed the base");
    }

    #[test]
    fn demand_asks_only_for_the_overlay_that_lands() {
        // A watermark in one corner must not make every tile pull the whole
        // overlay — that is the difference between a demand-driven engine and
        // a loop.
        let base = ImageDescriptor::new(1000, 1000, PixelFormat::Rgba8).unwrap();
        let overlay = ImageDescriptor::new(50, 50, PixelFormat::Rgba8).unwrap();
        let op = Composite::at(900, 900, Blend::Over);

        let far = op
            .input_regions(Region::new(0, 0, 100, 100), &[base, overlay])
            .unwrap();
        assert_eq!(
            far[0],
            Region::new(0, 0, 100, 100),
            "base demand is the output"
        );
        assert!(
            far[1].width <= 1 && far[1].height <= 1,
            "a tile the overlay misses should not demand it: {}",
            far[1]
        );

        let near = op
            .input_regions(Region::new(900, 900, 50, 50), &[base, overlay])
            .unwrap();
        assert_eq!(near[1], Region::new(0, 0, 50, 50), "overlapping tile");
    }

    #[test]
    fn the_source_blend_ignores_alpha() {
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgba8, &[9, 9, 9, 255]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[1, 2, 3, 0]);
        let op = Composite::at(0, 0, Blend::Source);
        let (_, out) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(4) {
            assert_eq!(pixel, [1, 2, 3, 0], "Source should copy verbatim");
        }
    }

    #[test]
    fn an_opaque_format_composites_as_replacement() {
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgb8, &[0, 0, 0]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgb8, &[5, 6, 7]);
        let (_, out) = apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        for pixel in out.chunks_exact(3) {
            assert_eq!(pixel, [5, 6, 7]);
        }
    }

    #[test]
    fn one_input_is_a_graph_error() {
        let base = ImageDescriptor::new(4, 4, PixelFormat::Rgba8).unwrap();
        assert!(Composite::over().output_descriptor(&[base]).is_err());
        assert_eq!(Composite::over().arity(), 2);
    }

    #[test]
    fn the_output_is_independent_of_how_the_image_is_tiled() {
        let (base_desc, base) = solid(16, 12, PixelFormat::Rgba8, &[40, 80, 120, 200]);
        let (over_desc, over) = solid(7, 5, PixelFormat::Rgba8, &[255, 0, 0, 128]);
        let op = Composite::at(3, 2, Blend::Over);
        let (out_desc, whole) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();

        let base_buf = TileBuf::from_vec(base_desc.region(), base_desc.pixel, base).unwrap();
        let over_buf = TileBuf::from_vec(over_desc.region(), over_desc.pixel, over).unwrap();

        for (tw, th) in [(4_u32, 4_u32), (1, 12), (16, 1), (5, 3)] {
            let mut target = TileBuf::for_image(&out_desc).unwrap();
            let mut y = 0;
            while y < out_desc.height {
                let h = th.min(out_desc.height - y);
                let mut x = 0;
                while x < out_desc.width {
                    let w = tw.min(out_desc.width - x);
                    let region = Region::new(x, y, w, h);
                    let demand = op.input_regions(region, &[base_desc, over_desc]).unwrap();

                    let mut base_cut = TileBuf::zeroed(demand[0], base_desc.pixel).unwrap();
                    otf_pixels_core::copy_region(
                        &base_buf.as_tile().unwrap(),
                        &mut base_cut.as_tile_mut().unwrap(),
                        demand[0],
                    )
                    .unwrap();
                    let mut over_cut = TileBuf::zeroed(demand[1], over_desc.pixel).unwrap();
                    otf_pixels_core::copy_region(
                        &over_buf.as_tile().unwrap(),
                        &mut over_cut.as_tile_mut().unwrap(),
                        demand[1],
                    )
                    .unwrap();

                    let mut sub = TileBuf::zeroed(region, out_desc.pixel).unwrap();
                    op.compute(
                        &[base_cut.as_tile().unwrap(), over_cut.as_tile().unwrap()],
                        &mut sub.as_tile_mut().unwrap(),
                    )
                    .unwrap();
                    otf_pixels_core::copy_region(
                        &sub.as_tile().unwrap(),
                        &mut target.as_tile_mut().unwrap(),
                        region,
                    )
                    .unwrap();
                    x += w;
                }
                y += h;
            }
            assert_eq!(
                target.bytes(),
                whole.as_slice(),
                "tiling at {tw}x{th} changed the pixels"
            );
        }
    }

    /// An image whose every pixel differs.
    fn varied(width: u32, height: u32, format: PixelFormat) -> (ImageDescriptor, Vec<u8>) {
        let descriptor = ImageDescriptor::new(width, height, format).unwrap();
        let bytes = (0..descriptor.byte_len().unwrap())
            .map(|i| (i * 37 % 251) as u8)
            .collect();
        (descriptor, bytes)
    }

    /// Run `op` tile by tile at `tw` x `th`, each tile given only the input
    /// regions it demands.
    fn tiled(
        op: &Composite,
        base: (&ImageDescriptor, &[u8]),
        overlay: (&ImageDescriptor, &[u8]),
        (tw, th): (u32, u32),
    ) -> Vec<u8> {
        let inputs = [*base.0, *overlay.0];
        let out = op.output_descriptor(&inputs).unwrap();
        let base_buf = TileBuf::from_vec(base.0.region(), base.0.pixel, base.1.to_vec()).unwrap();
        let over_buf =
            TileBuf::from_vec(overlay.0.region(), overlay.0.pixel, overlay.1.to_vec()).unwrap();
        let cut = |buf: &TileBuf, region: Region| {
            let mut part = TileBuf::zeroed(region, buf.pixel()).unwrap();
            otf_pixels_core::copy_region(
                &buf.as_tile().unwrap(),
                &mut part.as_tile_mut().unwrap(),
                region,
            )
            .unwrap();
            part
        };
        let mut target = TileBuf::for_image(&out).unwrap();
        for y in (0..out.height).step_by(th as usize) {
            for x in (0..out.width).step_by(tw as usize) {
                let region = Region::new(x, y, tw.min(out.width - x), th.min(out.height - y));
                let demand = op.input_regions(region, &inputs).unwrap();
                let (b, o) = (cut(&base_buf, demand[0]), cut(&over_buf, demand[1]));
                let mut sub = TileBuf::zeroed(region, out.pixel).unwrap();
                op.compute(
                    &[b.as_tile().unwrap(), o.as_tile().unwrap()],
                    &mut sub.as_tile_mut().unwrap(),
                )
                .unwrap();
                otf_pixels_core::copy_region(
                    &sub.as_tile().unwrap(),
                    &mut target.as_tile_mut().unwrap(),
                    region,
                )
                .unwrap();
            }
        }
        target.into_bytes()
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn the_pdf_modes_mix_as_the_spec_defines() {
        let a = [0.5, 0.25, 0.8, 1.0];
        let b = [0.4, 0.5, 0.1, 1.0];
        let cases = [
            (Blend::Multiply, [0.2, 0.125, 0.08, 1.0]),
            (Blend::Screen, [0.7, 0.625, 0.82, 1.0]),
            (Blend::Darken, [0.4, 0.25, 0.1, 1.0]),
            (Blend::Lighten, [0.5, 0.5, 0.8, 1.0]),
            (Blend::Difference, [0.1, 0.25, 0.7, 1.0]),
            (Blend::Exclusion, [0.5, 0.5, 0.74, 1.0]),
            // The base decides: dark bases multiply, light ones screen.
            (Blend::Overlay, [0.4, 0.25, 0.16, 1.0]),
            // The overlay decides.
            (Blend::HardLight, [0.4, 0.25, 0.64, 1.0]),
            (Blend::ColourDodge, [0.8, 0.5 / 0.75, 0.5, 1.0]),
            (Blend::ColourBurn, [0.0, 0.0, 1.0 - 0.9 / 0.8, 1.0]),
        ];
        for (blend, expected) in cases {
            let got = blend.apply(a, b);
            let expected = expected.map(|v: f32| v.clamp(0.0, 1.0));
            assert!(
                close(got, expected),
                "{blend:?}: {got:?}, expected {expected:?}"
            );
        }
        // Soft light at its two branches.
        let soft = Blend::SoftLight.apply([0.25, 0.75, 0.5, 1.0], [0.5, 0.16, 0.3, 1.0]);
        let d = ((16.0 * 0.16 - 12.0) * 0.16 + 4.0) * 0.16;
        assert!(close(
            soft,
            [0.5 - 0.5 * 0.5 * 0.5, 0.16 + 0.5 * (d - 0.16), 0.3, 1.0]
        ));
    }

    #[test]
    fn a_transparent_overlay_leaves_the_base_under_every_mixing_mode() {
        let b = [0.3, 0.6, 0.9, 0.7];
        for blend in [
            Blend::Over,
            Blend::Atop,
            Blend::Add,
            Blend::Saturate,
            Blend::Multiply,
            Blend::Screen,
            Blend::Overlay,
            Blend::Darken,
            Blend::Lighten,
            Blend::ColourDodge,
            Blend::ColourBurn,
            Blend::HardLight,
            Blend::SoftLight,
            Blend::Difference,
            Blend::Exclusion,
            Blend::DestOver,
            Blend::Dest,
        ] {
            let got = blend.apply([1.0, 0.0, 0.5, 0.0], b);
            assert!(close(got, b), "{blend:?} changed the base: {got:?}");
        }
    }

    #[test]
    fn the_porter_duff_modes_follow_cairo() {
        let a = [1.0, 0.0, 0.0, 0.5];
        let b = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(Blend::Clear.apply(a, b), [0.0; 4]);
        assert!(close(Blend::Source.apply(a, b), a));
        assert!(close(Blend::Dest.apply(a, b), b));
        assert!(
            close(Blend::In.apply(a, b), a),
            "in keeps the overlay's colour"
        );
        assert_eq!(
            Blend::Out.apply(a, b),
            [0.0; 4],
            "an opaque base leaves no outside"
        );
        assert!(close(Blend::Atop.apply(a, b), [0.5, 0.0, 0.5, 1.0]));
        assert!(close(Blend::DestIn.apply(a, b), [0.0, 0.0, 1.0, 0.5]));
        assert!(close(Blend::DestOut.apply(a, b), [0.0, 0.0, 1.0, 0.5]));
        assert!(close(Blend::DestAtop.apply(a, b), [0.0, 0.0, 1.0, 0.5]));
        assert!(close(Blend::Xor.apply(a, b), [0.0, 0.0, 1.0, 0.5]));
        assert!(close(Blend::DestOver.apply(a, b), b));
        // Sums, with alpha capped at one.
        assert!(close(Blend::Add.apply(a, b), [0.5, 0.0, 1.0, 1.0]));
        assert!(close(
            Blend::Saturate.apply([1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.75]),
            [0.25, 0.0, 0.75, 1.0]
        ));
    }

    #[test]
    fn the_float_and_integer_source_over_agree_to_a_level() {
        for (sa, ba) in [(0, 0), (1, 255), (128, 128), (200, 3), (255, 90)] {
            let (over, back) = ([200_u8, 17, 99, sa], [3_u8, 250, 128, ba]);
            let mut integer = [0; 4];
            over_8_bit(&over, &back, &mut integer, 3);
            let mut float = [0; 4];
            let blended = Blend::Over.apply(
                to_rgba(&over, PixelFormat::Rgba8),
                to_rgba(&back, PixelFormat::Rgba8),
            );
            from_rgba(blended, &mut float, PixelFormat::Rgba8);
            for (i, f) in integer.iter().zip(&float) {
                assert!(i.abs_diff(*f) <= 1, "{integer:?} against {float:?}");
            }
            // And onto an opaque backdrop with no alpha channel.
            let mut integer = [0; 3];
            over_8_bit(&over, &back[..3], &mut integer, 3);
            let mut float = [0; 3];
            let blended = Blend::Over.apply(
                to_rgba(&over, PixelFormat::Rgba8),
                to_rgba(&back[..3], PixelFormat::Rgb8),
            );
            from_rgba(blended, &mut float, PixelFormat::Rgb8);
            for (i, f) in integer.iter().zip(&float) {
                assert!(
                    i.abs_diff(*f) <= 1,
                    "{integer:?} against {float:?} with no alpha"
                );
            }
        }
    }

    #[test]
    fn a_translucent_overlay_draws_onto_an_opaque_base() {
        // The watermark case: an RGBA PNG over an RGB JPEG.
        let (base_desc, base) = solid(2, 2, PixelFormat::Rgb8, &[0, 0, 0]);
        let (over_desc, over) = solid(2, 2, PixelFormat::Rgba8, &[255, 255, 255, 128]);
        let (out, bytes) =
            apply(&Composite::over(), (&base_desc, &base), (&over_desc, &over)).unwrap();
        assert_eq!(out.pixel, PixelFormat::Rgb8, "source-over keeps it opaque");
        assert!(bytes.iter().all(|&v| v == 128), "{bytes:?}");
    }

    #[test]
    fn the_result_takes_colour_and_alpha_where_it_needs_them() {
        let format = |base, overlay, blend| {
            let base = ImageDescriptor::new(4, 4, base).unwrap();
            let overlay = ImageDescriptor::new(2, 2, overlay).unwrap();
            Composite::at(0, 0, blend)
                .output_descriptor(&[base, overlay])
                .unwrap()
                .pixel
        };
        use PixelFormat::{Gray8, GrayA8, Rgb8, Rgb16, Rgba8, Rgba16, RgbaF32};
        assert_eq!(format(Gray8, Rgb8, Blend::Over), Rgb8);
        assert_eq!(format(Gray8, GrayA8, Blend::Multiply), Gray8);
        assert_eq!(format(Rgb8, Rgb8, Blend::Clear), Rgba8);
        assert_eq!(format(Rgb8, Rgb8, Blend::Source), Rgb8);
        assert_eq!(format(Rgb8, Rgba8, Blend::Source), Rgba8);
        assert_eq!(format(Gray8, Gray8, Blend::DestOut), GrayA8);
        assert_eq!(format(Rgb16, Rgba8, Blend::Over), Rgb16, "the base's depth");
        assert_eq!(format(Rgba16, RgbaF32, Blend::Over), Rgba16);
    }

    #[test]
    fn clearing_affects_only_the_overlay_area() {
        let (base_desc, base) = solid(3, 1, PixelFormat::Rgb8, &[9, 9, 9]);
        let (over_desc, over) = solid(1, 1, PixelFormat::Rgb8, &[1, 1, 1]);
        let op = Composite::at(1, 0, Blend::Clear);
        let (_, bytes) = apply(&op, (&base_desc, &base), (&over_desc, &over)).unwrap();
        assert_eq!(bytes, [9, 9, 9, 255, 0, 0, 0, 0, 9, 9, 9, 255]);
    }

    #[test]
    fn wide_depths_composite() {
        let (base_desc, base) = solid(2, 1, PixelFormat::Rgba16, &[0, 0, 0, 0, 0, 0, 255, 255]);
        let (over_desc, over) = solid(1, 1, PixelFormat::Rgba8, &[255, 255, 255, 255]);
        let (_, bytes) = apply(
            &Composite::at(1, 0, Blend::Over),
            (&base_desc, &base),
            (&over_desc, &over),
        )
        .unwrap();
        assert_eq!(&bytes[..8], [0, 0, 0, 0, 0, 0, 255, 255], "untouched");
        assert_eq!(&bytes[8..], [255; 8], "white at 16 bits");
    }

    #[test]
    fn gravity_places_the_overlay_as_sharp_does() {
        let base = ImageDescriptor::new(10, 6, PixelFormat::Gray8).unwrap();
        let overlay = ImageDescriptor::new(3, 2, PixelFormat::Gray8).unwrap();
        let at = |position| Bound::new(Placement::Gravity(position), &base, &overlay).at;
        assert_eq!(at(Position::Centre), (3, 2));
        assert_eq!(at(Position::North), (3, 0));
        assert_eq!(at(Position::NorthEast), (7, 0));
        assert_eq!(at(Position::East), (7, 2));
        assert_eq!(at(Position::SouthEast), (7, 4));
        assert_eq!(at(Position::South), (3, 4));
        assert_eq!(at(Position::SouthWest), (0, 4));
        assert_eq!(at(Position::West), (0, 2));
        assert_eq!(at(Position::NorthWest), (0, 0));
        // A larger overlay overhangs evenly, rounding toward zero.
        let large = ImageDescriptor::new(13, 9, PixelFormat::Gray8).unwrap();
        let bound = Bound::new(Placement::Gravity(Position::Centre), &base, &large);
        assert_eq!(bound.at, (-1, -1));
    }

    #[test]
    fn tiles_align_to_the_gravity() {
        let tile = |placement| {
            let (base_desc, base) = solid(7, 1, PixelFormat::Gray8, &[0]);
            let over_desc = ImageDescriptor::new(3, 1, PixelFormat::Gray8).unwrap();
            let op = Composite::new(placement, Blend::Over);
            apply(&op, (&base_desc, &base), (&over_desc, &[1, 2, 3]))
                .unwrap()
                .1
        };
        assert_eq!(
            tile(Placement::Tiled(Position::West)),
            [1, 2, 3, 1, 2, 3, 1]
        );
        assert_eq!(
            tile(Placement::Tiled(Position::East)),
            [3, 1, 2, 3, 1, 2, 3]
        );
        // Centred: three copies cover seven, and the middle one is centred.
        assert_eq!(
            tile(Placement::Tiled(Position::Centre)),
            [2, 3, 1, 2, 3, 1, 2]
        );
        assert_eq!(tile(Placement::TiledFrom(1, 0)), [2, 3, 1, 2, 3, 1, 2]);
        assert_eq!(tile(Placement::TiledFrom(-1, 0)), [3, 1, 2, 3, 1, 2, 3]);
        // An even count is made odd when centred, so a copy sits mid-image.
        let base = ImageDescriptor::new(10, 1, PixelFormat::Gray8).unwrap();
        let overlay = ImageDescriptor::new(3, 1, PixelFormat::Gray8).unwrap();
        let bound = Bound::new(Placement::Tiled(Position::Centre), &base, &overlay);
        assert_eq!(bound.at.0, 0, "five copies of 3 cropped by 3 on the left");
    }

    #[test]
    fn a_tiled_tile_demands_only_the_overlay_it_shows() {
        let base = ImageDescriptor::new(100, 100, PixelFormat::Rgba8).unwrap();
        let overlay = ImageDescriptor::new(10, 10, PixelFormat::Rgba8).unwrap();
        let op = Composite::new(Placement::TiledFrom(0, 0), Blend::Over);
        op.output_descriptor(&[base, overlay]).unwrap();
        let demand = |region| op.input_regions(region, &[base, overlay]).unwrap()[1];
        assert_eq!(demand(Region::new(22, 31, 4, 3)), Region::new(2, 1, 4, 3));
        assert_eq!(
            demand(Region::new(28, 0, 4, 1)),
            Region::new(0, 0, 10, 1),
            "wraps"
        );
        assert_eq!(demand(Region::new(0, 0, 50, 50)), Region::new(0, 0, 10, 10));
    }

    #[test]
    fn every_placement_is_independent_of_tiling() {
        let (base_desc, base) = varied(23, 17, PixelFormat::Rgb8);
        let (over_desc, over) = varied(6, 5, PixelFormat::Rgba8);
        for (placement, blend) in [
            (Placement::Gravity(Position::SouthEast), Blend::Multiply),
            (Placement::At(-2, 13), Blend::Over),
            (Placement::Tiled(Position::Centre), Blend::Screen),
            (Placement::TiledFrom(4, -3), Blend::DestOut),
            (Placement::Tiled(Position::North), Blend::Xor),
        ] {
            let op = Composite::new(placement, blend);
            let whole = tiled(&op, (&base_desc, &base), (&over_desc, &over), (23, 17));
            for shape in [(4, 4), (1, 17), (23, 1), (5, 3)] {
                assert_eq!(
                    tiled(&op, (&base_desc, &base), (&over_desc, &over), shape),
                    whole,
                    "{placement:?} {blend:?} at {shape:?}"
                );
            }
        }
    }

    #[test]
    fn one_instance_on_two_pairs_of_shapes_is_a_graph_error() {
        let op = Composite::over();
        let a = ImageDescriptor::new(4, 4, PixelFormat::Rgb8).unwrap();
        let b = ImageDescriptor::new(5, 4, PixelFormat::Rgb8).unwrap();
        op.output_descriptor(&[a, a]).unwrap();
        assert!(op.output_descriptor(&[a, b]).is_err());
        assert!(op.clone().output_descriptor(&[a, b]).is_ok());
    }
}
