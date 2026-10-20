//! Modular image decoding (ISO/IEC 18181-1 §H): the MA tree, the fourteen
//! predictors, and the RCT, palette and squeeze transforms, over channels
//! that may be split across a frame's groups. Follows the procedure as
//! jxl-oxide (MIT/Apache-2.0) implements it; see NOTICE.

#![allow(
    clippy::indexing_slicing,
    reason = "sample indices stay within each plane; tree indices are built in range"
)]

use otf_pixels_core::Result;

use super::coding::Decoder;
use super::reader::{BitReader, malformed, unpack_signed};
use crate::bits::{bits, val};
use crate::predict::{WeightedState, WpParams};

/// One channel's samples.
#[derive(Debug, Clone, Default)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    /// Subsampling shifts; -1 for a palette's meta channel.
    pub hshift: i32,
    pub vshift: i32,
    pub data: Vec<i32>,
}

impl Plane {
    #[must_use]
    pub fn new(width: usize, height: usize, hshift: i32, vshift: i32) -> Self {
        Self {
            width,
            height,
            hshift,
            vshift,
            data: vec![0; width * height],
        }
    }

    fn shape(&self) -> (usize, usize, i32, i32) {
        (self.width, self.height, self.hshift, self.vshift)
    }

    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> i32 {
        self.data[y * self.width + x]
    }
}

fn parse_wp(r: &mut BitReader<'_>) -> Result<WpParams> {
    if r.bool()? {
        return Ok(WpParams::default());
    }
    let p1 = i64::from(r.read(5)?);
    let p2 = i64::from(r.read(5)?);
    let mut p3 = [0; 5];
    for v in &mut p3 {
        *v = i64::from(r.read(5)?);
    }
    let mut weights = [0; 4];
    for w in &mut weights {
        *w = r.read(4)?;
    }
    Ok(WpParams { p1, p2, p3, weights })
}

const BEGIN_C: [crate::bits::Dist; 4] = [bits(0, 3), bits(8, 6), bits(72, 10), bits(1096, 13)];

#[derive(Debug, Clone, Copy)]
struct SqueezeStep {
    horizontal: bool,
    in_place: bool,
    begin: usize,
    num: usize,
}

#[derive(Debug, Clone)]
enum Transform {
    Rct {
        begin: usize,
        kind: u32,
    },
    Palette {
        begin: usize,
        num: usize,
        colours: usize,
        deltas: i32,
        predictor: u32,
        wp: WpParams,
    },
    Squeeze(Vec<SqueezeStep>),
}

/// A stream's header: its transforms and how it predicts.
#[derive(Debug, Clone)]
struct Header {
    use_global_tree: bool,
    wp: WpParams,
    transforms: Vec<Transform>,
}

impl Header {
    fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        let use_global_tree = r.bool()?;
        let wp = parse_wp(r)?;
        let n = r.u32([val(0), val(1), bits(2, 4), bits(18, 8)])?;
        if n > 512 {
            return Err(malformed("too many Modular transforms"));
        }
        let mut transforms = Vec::with_capacity(n as usize);
        for _ in 0..n {
            transforms.push(match r.read(2)? {
                0 => Transform::Rct {
                    begin: r.u32(BEGIN_C)? as usize,
                    kind: r.u32([val(6), bits(0, 2), bits(2, 4), bits(10, 6)])?,
                },
                1 => {
                    let begin = r.u32(BEGIN_C)? as usize;
                    let num = r.u32([val(1), val(3), val(4), bits(1, 13)])? as usize;
                    let colours = r.u32([bits(0, 8), bits(256, 10), bits(1280, 12), bits(5376, 16)])? as usize;
                    let deltas = r.u32([val(0), bits(1, 8), bits(257, 10), bits(1281, 16)])? as i32;
                    let predictor = r.read(4)?;
                    if predictor > 13 {
                        return Err(malformed("unknown palette predictor"));
                    }
                    Transform::Palette {
                        begin,
                        num,
                        colours,
                        deltas,
                        predictor,
                        wp,
                    }
                }
                2 => {
                    let n = r.u32([val(0), bits(1, 4), bits(9, 6), bits(41, 8)])?;
                    let steps = (0..n)
                        .map(|_| -> Result<SqueezeStep> {
                            let horizontal = r.bool()?;
                            let in_place = r.bool()?;
                            let begin = r.u32(BEGIN_C)? as usize;
                            let num = r.u32([val(1), val(2), val(3), bits(4, 4)])? as usize;
                            Ok(SqueezeStep {
                                horizontal,
                                in_place,
                                begin,
                                num,
                            })
                        })
                        .collect::<Result<_>>()?;
                    Transform::Squeeze(steps)
                }
                _ => return Err(malformed("unknown Modular transform")),
            });
        }
        Ok(Self {
            use_global_tree,
            wp,
            transforms,
        })
    }
}

/// A channel's shape before decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub width: usize,
    pub height: usize,
    pub hshift: i32,
    pub vshift: i32,
}

/// Apply the transforms to channel shapes as the encoder did, filling in
/// default squeeze steps; returns the shapes and meta channel count.
fn forward_shapes(transforms: &mut [Transform], shapes: &mut Vec<Shape>) -> Result<usize> {
    let mut nb_meta = 0_usize;
    for t in transforms {
        match t {
            Transform::Rct { begin, .. } => {
                let b = *begin;
                if b + 3 > shapes.len() || shapes[b + 1] != shapes[b] || shapes[b + 2] != shapes[b] {
                    return Err(malformed("bad RCT channels"));
                }
            }
            Transform::Palette {
                begin, num, colours, ..
            } => {
                let (b, n) = (*begin, *num);
                let end = b + n;
                if n == 0 || end > shapes.len() {
                    return Err(malformed("bad palette channels"));
                }
                if b < nb_meta {
                    if end > nb_meta {
                        return Err(malformed("bad palette channels"));
                    }
                    nb_meta = nb_meta + 2 - n;
                } else {
                    nb_meta += 1;
                }
                if shapes[b + 1..end].iter().any(|s| s.width != shapes[b].width || s.height != shapes[b].height) {
                    return Err(malformed("bad palette channels"));
                }
                shapes.drain(b + 1..end);
                shapes.insert(
                    0,
                    Shape {
                        width: *colours,
                        height: n,
                        hshift: -1,
                        vshift: -1,
                    },
                );
            }
            Transform::Squeeze(steps) => {
                if steps.is_empty() {
                    *steps = default_squeeze(shapes, nb_meta);
                }
                for s in steps.iter() {
                    let end = s.begin + s.num;
                    if end > shapes.len() {
                        return Err(malformed("bad squeeze channels"));
                    }
                    if s.begin < nb_meta {
                        if !s.in_place || end > nb_meta {
                            return Err(malformed("bad squeeze channels"));
                        }
                        nb_meta += s.num;
                    }
                    let mut residuals = Vec::new();
                    for ch in &mut shapes[s.begin..end] {
                        if ch.width == 0 || ch.height == 0 || ch.hshift > 30 || ch.vshift > 30 {
                            return Err(malformed("bad squeeze channels"));
                        }
                        let mut res = *ch;
                        if s.horizontal {
                            res.width = ch.width / 2;
                            ch.width = ch.width.div_ceil(2);
                            if ch.hshift >= 0 {
                                ch.hshift += 1;
                                res.hshift += 1;
                            }
                        } else {
                            res.height = ch.height / 2;
                            ch.height = ch.height.div_ceil(2);
                            if ch.vshift >= 0 {
                                ch.vshift += 1;
                                res.vshift += 1;
                            }
                        }
                        residuals.push(res);
                    }
                    if s.in_place {
                        let tail: Vec<Shape> = shapes.drain(end..).collect();
                        residuals.extend(tail);
                    }
                    shapes.extend(residuals);
                }
            }
        }
        if shapes.len() > 1 << 16 {
            return Err(malformed("too many Modular channels"));
        }
    }
    Ok(nb_meta)
}

fn default_squeeze(shapes: &[Shape], first: usize) -> Vec<SqueezeStep> {
    let mut out = Vec::new();
    let Some(ch) = shapes.get(first) else {
        return out;
    };
    let (mut w, mut h) = (ch.width, ch.height);
    if shapes.len() - first >= 3 && shapes[first + 1].width == w && shapes[first + 1].height == h {
        for horizontal in [true, false] {
            out.push(SqueezeStep {
                horizontal,
                in_place: false,
                begin: first + 1,
                num: 2,
            });
        }
    }
    let step = |horizontal| SqueezeStep {
        horizontal,
        in_place: true,
        begin: first,
        num: shapes.len() - first,
    };
    if h >= w && h > 8 {
        out.push(step(false));
        h = h.div_ceil(2);
    }
    while w > 8 || h > 8 {
        if w > 8 {
            out.push(step(true));
            w = w.div_ceil(2);
        }
        if h > 8 {
            out.push(step(false));
            h = h.div_ceil(2);
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
enum Node {
    Decision { property: u32, value: i32, left: u32, right: u32 },
    Leaf { ctx: u32, predictor: u32, offset: i32, multiplier: u32 },
}

/// An MA tree with the entropy code its leaves' contexts use.
#[derive(Debug, Clone)]
pub struct Tree {
    nodes: Vec<Node>,
    decoder: Decoder,
}

impl Tree {
    /// Parse a tree and its code.
    ///
    /// # Errors
    ///
    /// For a malformed tree.
    pub fn parse(r: &mut BitReader<'_>, node_limit: usize) -> Result<Self> {
        let mut d = Decoder::parse(r, 6)?;
        d.begin(r)?;
        let mut nodes = Vec::new();
        let mut left = 1_usize;
        let mut next_child = 1_u32;
        let mut ctx = 0;
        while left > 0 {
            if nodes.len() > node_limit.min(1 << 22) {
                return Err(malformed("an MA tree is too large"));
            }
            left -= 1;
            let property = d.read(r, 1)?;
            if let Some(property) = property.checked_sub(1) {
                let value = unpack_signed(d.read(r, 0)?);
                nodes.push(Node::Decision {
                    property,
                    value,
                    left: next_child,
                    right: next_child + 1,
                });
                next_child += 2;
                left += 2;
            } else {
                let predictor = d.read(r, 2)?;
                if predictor > 13 {
                    return Err(malformed("unknown predictor"));
                }
                let offset = unpack_signed(d.read(r, 3)?);
                let mul_log = d.read(r, 4)?;
                if mul_log > 30 {
                    return Err(malformed("bad MA tree multiplier"));
                }
                let mul_bits = d.read(r, 5)?;
                if mul_bits > (1 << (31 - mul_log)) - 2 {
                    return Err(malformed("bad MA tree multiplier"));
                }
                nodes.push(Node::Leaf {
                    ctx,
                    predictor,
                    offset,
                    multiplier: (mul_bits + 1) << mul_log,
                });
                ctx += 1;
            }
        }
        d.finish()?;
        let decoder = Decoder::parse(r, ctx)?;
        Ok(Self { nodes, decoder })
    }

    /// The tree with decisions on channel and stream index resolved.
    fn specialise(&self, channel: u32, stream: u32) -> Vec<Node> {
        // Rebuild with the static decisions folded away, keeping indices.
        let resolve = |mut i: usize| -> usize {
            loop {
                match self.nodes[i] {
                    Node::Decision {
                        property: p @ (0 | 1),
                        value,
                        left,
                        right,
                    } => {
                        let v = if p == 0 { channel } else { stream };
                        i = if i64::from(v) > i64::from(value) { left } else { right } as usize;
                    }
                    _ => return i,
                }
            }
        };
        let mut out = Vec::new();
        let mut map = std::collections::HashMap::new();
        let mut stack = vec![resolve(0)];
        // Assign new indices in discovery order.
        while let Some(i) = stack.pop() {
            if map.contains_key(&i) {
                continue;
            }
            map.insert(i, out.len());
            out.push(self.nodes[i]);
            if let Node::Decision { left, right, .. } = self.nodes[i] {
                stack.push(resolve(right as usize));
                stack.push(resolve(left as usize));
            }
        }
        for n in &mut out {
            if let Node::Decision { left, right, .. } = n {
                *left = map[&resolve(*left as usize)] as u32;
                *right = map[&resolve(*right as usize)] as u32;
            }
        }
        out
    }
}

/// Neighbours of a sample, with the decoder's edge rules.
#[derive(Clone, Copy)]
struct Nb {
    w: i32,
    n: i32,
    nw: i32,
    ne: i32,
    nn: i32,
    ww: i32,
    nee: i32,
}

#[inline]
fn neighbours(p: &Plane, x: usize, y: usize) -> Nb {
    let width = p.width;
    let d = &p.data;
    let row = y * width;
    let w = if x > 0 {
        d[row + x - 1]
    } else if y > 0 {
        d[row - width]
    } else {
        0
    };
    let n = if y > 0 { d[row - width + x] } else { w };
    let nw = if x > 0 && y > 0 { d[row - width + x - 1] } else { w };
    let ne = if y > 0 && x + 1 < width { d[row - width + x + 1] } else { n };
    let nn = if y > 1 { d[row - 2 * width + x] } else { n };
    let ww = if x > 1 { d[row + x - 2] } else { w };
    let nee = if y > 0 && x + 2 < width { d[row - width + x + 2] } else { ne };
    Nb {
        w,
        n,
        nw,
        ne,
        nn,
        ww,
        nee,
    }
}

fn to_pnb(nb: &Nb) -> crate::predict::Neighbours {
    crate::predict::Neighbours {
        w: nb.w,
        n: nb.n,
        nw: nb.nw,
        ne: nb.ne,
        nn: nb.nn,
        ww: nb.ww,
    }
}

#[inline]
fn predict(predictor: u32, nb: &Nb, wp: i64) -> i64 {
    let (w, n, nw, ne) = (i64::from(nb.w), i64::from(nb.n), i64::from(nb.nw), i64::from(nb.ne));
    match predictor {
        0 => 0,
        1 => w,
        2 => n,
        3 => (w + n) / 2,
        4 => {
            if nb.n.abs_diff(nb.nw) < nb.w.abs_diff(nb.nw) {
                w
            } else {
                n
            }
        }
        5 => (n + w - nw).clamp(w.min(n), w.max(n)),
        6 => (wp + 3) >> 3,
        7 => ne,
        8 => nw,
        9 => i64::from(nb.ww),
        10 => (w + nw) / 2,
        11 => (n + nw) / 2,
        12 => (n + ne) / 2,
        _ => (6 * n - 2 * i64::from(nb.nn) + 7 * w + i64::from(nb.ww) + i64::from(nb.nee) + 3 * ne + 8) / 16,
    }
}

fn grad_clamped(n: i32, w: i32, nw: i32) -> i32 {
    let (n, w, nw) = (i64::from(n), i64::from(w), i64::from(nw));
    (n + w - nw).clamp(w.min(n), w.max(n)) as i32
}

/// Property `16 + k` from the previous channels (most recent first).
fn extra_property(prev: &[&Plane], k: usize, x: usize, y: usize) -> i32 {
    let Some(p) = prev.get(k / 4) else {
        return 0;
    };
    let c = p.at(x, y);
    match k % 4 {
        0 => c.wrapping_abs(),
        1 => c,
        i => {
            let g = match (x, y) {
                (0, 0) => 0,
                (0, y) => p.at(0, y - 1),
                (x, 0) => p.at(x - 1, 0),
                (x, y) => grad_clamped(p.at(x, y - 1), p.at(x - 1, y), p.at(x - 1, y - 1)),
            };
            if i == 2 { c.abs_diff(g) as i32 } else { c.wrapping_sub(g) }
        }
    }
}

/// Decode the planes of one stream in order.
fn decode_planes(
    r: &mut BitReader<'_>,
    tree: &Tree,
    wp_params: WpParams,
    planes: &mut [Plane],
    stream: u32,
) -> Result<()> {
    let mut d = tree.decoder.clone();
    d.begin(r)?;
    let dist_multiplier = planes.iter().map(|p| p.width).max().unwrap_or(0) as u32;
    for i in 0..planes.len() {
        let (done, rest) = planes.split_at_mut(i);
        let plane = &mut rest[0];
        if plane.width == 0 || plane.height == 0 {
            continue;
        }
        let nodes = tree.specialise(i as u32, stream);
        let mut uses_wp = false;
        let mut max_prev = 0_usize;
        let mut uses_props = false;
        for n in &nodes {
            match *n {
                Node::Decision { property, .. } => {
                    uses_props = true;
                    if property == 15 {
                        uses_wp = true;
                    }
                    if property >= 16 {
                        max_prev = max_prev.max((property as usize - 16) / 4 + 1);
                    }
                }
                Node::Leaf { predictor: 6, .. } => uses_wp = true,
                Node::Leaf { .. } => {}
            }
        }
        let shape = plane.shape();
        let prev: Vec<&Plane> = done
            .iter()
            .rev()
            .filter(|p| p.shape() == shape)
            .take(max_prev)
            .collect();
        decode_plane(r, &mut d, &nodes, plane, &prev, uses_wp.then_some(wp_params), uses_props, i as i32, dist_multiplier)?;
    }
    d.finish()
}

#[allow(clippy::too_many_arguments, reason = "the per-plane decode state")]
fn decode_plane(
    r: &mut BitReader<'_>,
    d: &mut Decoder,
    nodes: &[Node],
    plane: &mut Plane,
    prev: &[&Plane],
    wp_params: Option<WpParams>,
    uses_props: bool,
    channel: i32,
    dist_multiplier: u32,
) -> Result<()> {
    let (width, height) = (plane.width, plane.height);
    let mut wp = wp_params.map(|p| WeightedState::with_params(width, p));
    // A single leaf: no properties to compute.
    if let [Node::Leaf {
        ctx,
        predictor,
        offset,
        multiplier,
    }] = *nodes
    {
        if predictor == 0 && multiplier == 1 && offset == 0 {
            for v in &mut plane.data {
                *v = unpack_signed(d.read_with_multiplier(r, ctx, dist_multiplier)?);
            }
            return Ok(());
        }
        for y in 0..height {
            for x in 0..width {
                let nb = neighbours(plane, x, y);
                let wpred = wp.as_ref().map(|s| s.predict(&to_pnb(&nb)));
                let guess = predict(predictor, &nb, wpred.map_or(0, |p| p.eighths()));
                let diff = i64::from(unpack_signed(d.read_with_multiplier(r, ctx, dist_multiplier)?));
                let v = (guess + diff * i64::from(multiplier) + i64::from(offset)) as i32;
                plane.data[y * width + x] = v;
                if let (Some(s), Some(p)) = (wp.as_mut(), wpred) {
                    s.record(&p, v);
                }
            }
        }
        return Ok(());
    }
    let mut props = [0_i32; 16];
    props[0] = channel;
    for y in 0..height {
        let mut prev_grad = 0_i32;
        for x in 0..width {
            let nb = neighbours(plane, x, y);
            let wpred = wp.as_ref().map(|s| s.predict(&to_pnb(&nb)));
            if uses_props {
                let w_nw = nb.w.wrapping_sub(nb.nw);
                props[2] = y as i32;
                props[3] = x as i32;
                props[4] = nb.n.unsigned_abs() as i32;
                props[5] = nb.w.unsigned_abs() as i32;
                props[6] = nb.n;
                props[7] = nb.w;
                props[8] = nb.w.wrapping_sub(prev_grad);
                props[9] = w_nw.wrapping_add(nb.n);
                props[10] = w_nw;
                props[11] = nb.nw.wrapping_sub(nb.n);
                props[12] = nb.n.wrapping_sub(nb.ne);
                props[13] = nb.n.wrapping_sub(nb.nn);
                props[14] = nb.w.wrapping_sub(nb.ww);
                props[15] = wpred.map_or(0, |p| p.max_error);
                prev_grad = props[9];
            }
            let mut i = 0;
            let (ctx, predictor, offset, multiplier) = loop {
                match nodes[i] {
                    Node::Decision {
                        property,
                        value,
                        left,
                        right,
                    } => {
                        let p = if property < 16 {
                            props[property as usize]
                        } else {
                            extra_property(prev, property as usize - 16, x, y)
                        };
                        i = if p > value { left } else { right } as usize;
                    }
                    Node::Leaf {
                        ctx,
                        predictor,
                        offset,
                        multiplier,
                    } => break (ctx, predictor, offset, multiplier),
                }
            };
            let guess = predict(predictor, &nb, wpred.map_or(0, |p| p.eighths()));
            let diff = i64::from(unpack_signed(d.read_with_multiplier(r, ctx, dist_multiplier)?));
            let v = (guess + diff * i64::from(multiplier) + i64::from(offset)) as i32;
            plane.data[y * width + x] = v;
            if let (Some(s), Some(p)) = (wp.as_mut(), wpred) {
                s.record(&p, v);
            }
        }
    }
    Ok(())
}

const DELTA_PALETTE: [[i16; 3]; 72] = [
    [0, 0, 0], [4, 4, 4], [11, 0, 0], [0, 0, -13], [0, -12, 0], [-10, -10, -10],
    [-18, -18, -18], [-27, -27, -27], [-18, -18, 0], [0, 0, -32], [-32, 0, 0], [-37, -37, -37],
    [0, -32, -32], [24, 24, 45], [50, 50, 50], [-45, -24, -24], [-24, -45, -45], [0, -24, -24],
    [-34, -34, 0], [-24, 0, -24], [-45, -45, -24], [64, 64, 64], [-32, 0, -32], [0, -32, 0],
    [-32, 0, 32], [-24, -45, -24], [45, 24, 45], [24, -24, -45], [-45, -24, 24], [80, 80, 80],
    [64, 0, 0], [0, 0, -64], [0, -64, -64], [-24, -24, 45], [96, 96, 96], [64, 64, 0],
    [45, -24, -24], [34, -34, 0], [112, 112, 112], [24, -45, -45], [45, 45, -24], [0, -32, 32],
    [24, -24, 45], [0, 96, 96], [45, -24, 24], [24, -45, -24], [-24, -45, 24], [0, -64, 0],
    [96, 0, 0], [128, 128, 128], [64, 0, 64], [144, 144, 144], [96, 96, 0], [-36, -36, 36],
    [45, -24, -45], [45, -45, -24], [0, 0, -96], [0, 128, 128], [0, 96, 0], [45, 24, -45],
    [-128, 0, 0], [24, -45, 24], [-45, 24, -45], [64, 0, -64], [64, -64, -64], [96, 0, 96],
    [45, -45, 24], [24, 45, -45], [64, 64, -64], [128, 128, 0], [0, 0, -128], [-24, 45, -45],
];

fn inverse_rct(planes: &mut [Plane], begin: usize, kind: u32) -> Result<()> {
    if begin + 3 > planes.len() {
        return Err(malformed("bad RCT channels"));
    }
    let permutation = kind / 7;
    let ty = kind % 7;
    let n = planes[begin].data.len();
    let [a, b, c] = &mut planes[begin..begin + 3] else {
        return Err(malformed("bad RCT channels"));
    };
    for i in 0..n {
        let (va, vb, vc) = (a.data[i], b.data[i], c.data[i]);
        let (d, e, f);
        if ty == 6 {
            let tmp = va.wrapping_sub(vc >> 1);
            e = vc.wrapping_add(tmp);
            f = tmp.wrapping_sub(vb >> 1);
            d = f.wrapping_add(vb);
        } else {
            d = va;
            f = if ty & 1 != 0 { vc.wrapping_add(va) } else { vc };
            e = match ty >> 1 {
                1 => vb.wrapping_add(va),
                2 => vb.wrapping_add((va.wrapping_add(f)) >> 1),
                _ => vb,
            };
        }
        a.data[i] = d;
        b.data[i] = e;
        c.data[i] = f;
    }
    let (a, b, c) = (begin, begin + 1, begin + 2);
    match permutation {
        1 => {
            planes.swap(a, b);
            planes.swap(a, c);
        }
        2 => {
            planes.swap(a, b);
            planes.swap(b, c);
        }
        3 => planes.swap(b, c),
        4 => planes.swap(a, b),
        5 => planes.swap(a, c),
        _ => {}
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, reason = "the palette transform's parameters")]
fn inverse_palette(
    planes: &mut Vec<Plane>,
    begin: usize,
    num: usize,
    colours: usize,
    deltas: i32,
    predictor: u32,
    wp: WpParams,
    bit_depth: u32,
) -> Result<()> {
    if planes.is_empty() {
        return Err(malformed("bad palette channels"));
    }
    let palette = planes.remove(0);
    if begin >= planes.len() || palette.height != num {
        return Err(malformed("bad palette channels"));
    }
    let index = std::mem::take(&mut planes[begin]);
    let (width, height) = (index.width, index.height);
    let mut outs: Vec<Plane> = (0..num).map(|_| Plane::new(width, height, index.hshift, index.vshift)).collect();
    let nb_colours = colours as i32;
    let mut need_delta = Vec::new();
    for (i, &idx) in index.data.iter().enumerate() {
        if idx < deltas {
            need_delta.push(i);
        }
        for (c, out) in outs.iter_mut().enumerate() {
            out.data[i] = if (0..nb_colours).contains(&idx) {
                palette.data[c * palette.width + idx as usize]
            } else if idx >= nb_colours {
                let idx = idx - nb_colours;
                if idx < 64 {
                    ((idx >> (2 * c)) % 4) * ((1 << bit_depth) - 1) / 4 + (1 << bit_depth.saturating_sub(3))
                } else {
                    let mut idx = idx - 64;
                    for _ in 0..c {
                        idx /= 5;
                    }
                    (idx % 5) * ((1 << bit_depth) - 1) / 4
                }
            } else if c >= 3 {
                0
            } else {
                let idx = (-(i64::from(idx) + 1) % 143) as usize;
                let mut v = i32::from(DELTA_PALETTE[(idx + 1) >> 1][c]);
                if idx & 1 == 0 {
                    v = -v;
                }
                if bit_depth > 8 {
                    v <<= bit_depth.min(24) - 8;
                }
                v
            };
        }
    }
    if !need_delta.is_empty() {
        for out in &mut outs {
            let mut state = (predictor == 6).then(|| WeightedState::with_params(width, wp));
            let mut next = 0;
            for i in 0..width * height {
                let (x, y) = (i % width, i / width);
                let nb = neighbours(out, x, y);
                let wpred = state.as_ref().map(|s| s.predict(&to_pnb(&nb)));
                let mut v = out.data[i];
                if need_delta.get(next) == Some(&i) {
                    v = (i64::from(v) + predict(predictor, &nb, wpred.map_or(0, |p| p.eighths()))) as i32;
                    out.data[i] = v;
                    next += 1;
                    if next >= need_delta.len() {
                        break;
                    }
                }
                if let (Some(s), Some(p)) = (state.as_mut(), wpred) {
                    s.record(&p, v);
                }
            }
        }
    }
    planes.splice(begin..=begin, outs);
    Ok(())
}

fn tendency(a: i32, b: i32, c: i32) -> i32 {
    let (a, b, c) = (i64::from(a), i64::from(b), i64::from(c));
    let x = if a >= b && b >= c {
        let mut x = (4 * a - 3 * c - b + 6) / 12;
        if x - (x & 1) > 2 * (a - b) {
            x = 2 * (a - b) + 1;
        }
        if x + (x & 1) > 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else if a <= b && b <= c {
        let mut x = (4 * a - 3 * c - b - 6) / 12;
        if x + (x & 1) < 2 * (a - b) {
            x = 2 * (a - b) - 1;
        }
        if x - (x & 1) < 2 * (b - c) {
            x = 2 * (b - c);
        }
        x
    } else {
        0
    };
    x as i32
}

/// Undo one squeeze along a line: `avg` (ceil(n/2)) and `res` (floor(n/2))
/// into `out` (n).
fn unsqueeze(avg: &[i32], res: &[i32], out: &mut [i32]) {
    let mut a = avg[0];
    let mut left = a;
    for (x, &r) in res.iter().enumerate() {
        let next = avg.get(x + 1).copied().unwrap_or(a);
        let diff = r.wrapping_add(tendency(left, a, next));
        let first = a.wrapping_add(diff / 2);
        let second = first.wrapping_sub(diff);
        out[2 * x] = first;
        out[2 * x + 1] = second;
        a = next;
        left = second;
    }
    if avg.len() > res.len() {
        out[out.len() - 1] = avg[avg.len() - 1];
    }
}

#[allow(clippy::needless_range_loop, reason = "columns gathered across rows")]
fn merge(avg: &Plane, res: &Plane, horizontal: bool) -> Result<Plane> {
    if horizontal {
        if avg.height != res.height || res.width > avg.width || avg.width > res.width + 1 {
            return Err(malformed("bad squeeze residual"));
        }
        let width = avg.width + res.width;
        let mut out = Plane::new(width, avg.height, avg.hshift - i32::from(avg.hshift > 0), avg.vshift);
        for y in 0..avg.height {
            unsqueeze(
                &avg.data[y * avg.width..(y + 1) * avg.width],
                &res.data[y * res.width..(y + 1) * res.width],
                &mut out.data[y * width..(y + 1) * width],
            );
        }
        Ok(out)
    } else {
        if avg.width != res.width || res.height > avg.height || avg.height > res.height + 1 {
            return Err(malformed("bad squeeze residual"));
        }
        let height = avg.height + res.height;
        let width = avg.width;
        let mut out = Plane::new(width, height, avg.hshift, avg.vshift - i32::from(avg.vshift > 0));
        if width == 0 {
            return Ok(out);
        }
        let (mut ca, mut cr, mut co) = (vec![0; avg.height], vec![0; res.height], vec![0; height]);
        for x in 0..width {
            for y in 0..avg.height {
                ca[y] = avg.data[y * width + x];
            }
            for y in 0..res.height {
                cr[y] = res.data[y * width + x];
            }
            unsqueeze(&ca, &cr, &mut co);
            for y in 0..height {
                out.data[y * width + x] = co[y];
            }
        }
        Ok(out)
    }
}

fn inverse_squeeze(planes: &mut Vec<Plane>, steps: &[SqueezeStep]) -> Result<()> {
    for s in steps.iter().rev() {
        let end = s.begin + s.num;
        let residuals: Vec<Plane> = if s.in_place {
            if end + s.num > planes.len() {
                return Err(malformed("bad squeeze channels"));
            }
            planes.drain(end..end + s.num).collect()
        } else {
            if s.num > planes.len() {
                return Err(malformed("bad squeeze channels"));
            }
            let at = planes.len() - s.num;
            planes.drain(at..).collect()
        };
        if end > planes.len() {
            return Err(malformed("bad squeeze channels"));
        }
        for (i, res) in residuals.iter().enumerate() {
            let avg = &planes[s.begin + i];
            planes[s.begin + i] = if avg.width == 0 || avg.height == 0 {
                Plane::new(avg.width + if s.horizontal { res.width } else { 0 }, avg.height + if s.horizontal { 0 } else { res.height }, avg.hshift, avg.vshift)
            } else {
                merge(avg, res, s.horizontal)?
            };
        }
    }
    Ok(())
}

fn inverse(transforms: &[Transform], planes: &mut Vec<Plane>, bit_depth: u32) -> Result<()> {
    for t in transforms.iter().rev() {
        match t {
            Transform::Rct { begin, kind } => inverse_rct(planes, *begin, *kind)?,
            Transform::Palette {
                begin,
                num,
                colours,
                deltas,
                predictor,
                wp,
            } => inverse_palette(planes, *begin, *num, *colours, *deltas, *predictor, *wp, bit_depth)?,
            Transform::Squeeze(steps) => inverse_squeeze(planes, steps)?,
        }
    }
    Ok(())
}

fn local_tree<'a>(
    r: &mut BitReader<'_>,
    header: &Header,
    global: Option<&'a Tree>,
    shapes: &[Shape],
    owned: &'a mut Option<Tree>,
) -> Result<&'a Tree> {
    if header.use_global_tree {
        return global.ok_or_else(|| malformed("no global MA tree"));
    }
    let samples: usize = shapes.iter().map(|s| s.width * s.height).sum();
    Ok(owned.insert(Tree::parse(r, 1024 + samples)?))
}

fn planes_for(shapes: &[Shape]) -> Vec<Plane> {
    shapes.iter().map(|s| Plane::new(s.width, s.height, s.hshift, s.vshift)).collect()
}

/// Decode a whole, self-contained Modular stream for channels of `shapes`.
///
/// # Errors
///
/// For malformed data.
pub fn decode_stream(
    r: &mut BitReader<'_>,
    shapes: &[Shape],
    stream: u32,
    global: Option<&Tree>,
    bit_depth: u32,
) -> Result<Vec<Plane>> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    let mut header = Header::parse(r)?;
    let mut tshapes = shapes.to_vec();
    forward_shapes(&mut header.transforms, &mut tshapes)?;
    let mut owned = None;
    let tree = local_tree(r, &header, global, &tshapes, &mut owned)?;
    let mut planes = planes_for(&tshapes);
    decode_planes(r, tree, header.wp, &mut planes, stream)?;
    inverse(&header.transforms, &mut planes, bit_depth)?;
    Ok(planes)
}

/// A frame's global Modular image, filled in by its groups.
#[derive(Debug)]
pub struct GlobalImage {
    header: Header,
    pub planes: Vec<Plane>,
    /// Planes from here on are decoded by groups.
    pub group_from: usize,
    bit_depth: u32,
}

impl GlobalImage {
    /// Parse the global stream and decode the planes it holds itself.
    ///
    /// # Errors
    ///
    /// For malformed data.
    pub fn parse(
        r: &mut BitReader<'_>,
        shapes: &[Shape],
        global: Option<&Tree>,
        group_dim: usize,
        bit_depth: u32,
    ) -> Result<Self> {
        if shapes.is_empty() {
            return Ok(Self {
                header: Header {
                    use_global_tree: false,
                    wp: WpParams::default(),
                    transforms: Vec::new(),
                },
                planes: Vec::new(),
                group_from: 0,
                bit_depth,
            });
        }
        let mut header = Header::parse(r)?;
        let mut tshapes = shapes.to_vec();
        let nb_meta = forward_shapes(&mut header.transforms, &mut tshapes)?;
        let mut owned = None;
        let tree = local_tree(r, &header, global, &tshapes, &mut owned)?;
        let mut planes = planes_for(&tshapes);
        let group_from = tshapes
            .iter()
            .enumerate()
            .position(|(i, s)| i >= nb_meta && (s.width > group_dim || s.height > group_dim))
            .unwrap_or(tshapes.len());
        decode_planes(r, tree, header.wp, &mut planes[..group_from], 0)?;
        Ok(Self {
            header,
            planes,
            group_from,
            bit_depth,
        })
    }

    /// Decode one group's share of the remaining planes. `lf` picks the LF
    /// groups' planes (shifts of 3 or more), else those of pass shifts
    /// `[min, max)`; `(gx, gy)` is the group's position on its grid.
    ///
    /// # Errors
    ///
    /// For malformed data.
    #[allow(clippy::too_many_arguments, reason = "a group's position and kind")]
    pub fn decode_group(
        &mut self,
        r: &mut BitReader<'_>,
        global: Option<&Tree>,
        stream: u32,
        group_dim: usize,
        lf: bool,
        shifts: (i32, i32),
        (gx, gy): (usize, usize),
    ) -> Result<()> {
        let mut members = Vec::new();
        let mut shapes = Vec::new();
        for (i, p) in self.planes.iter().enumerate().skip(self.group_from) {
            if p.hshift < 0 || p.vshift < 0 {
                return Err(malformed("a group channel has no shift"));
            }
            let min = p.hshift.min(p.vshift);
            let (gw, gh) = if lf {
                if p.hshift < 3 || p.vshift < 3 {
                    continue;
                }
                (group_dim >> (p.hshift - 3), group_dim >> (p.vshift - 3))
            } else {
                if (p.hshift >= 3 && p.vshift >= 3) || !(shifts.0..shifts.1).contains(&min) {
                    continue;
                }
                (group_dim >> p.hshift, group_dim >> p.vshift)
            };
            if gw == 0 || gh == 0 {
                return Err(malformed("a channel shift is too large"));
            }
            let (x0, y0) = (gx * gw, gy * gh);
            if x0 >= p.width || y0 >= p.height {
                continue;
            }
            let w = gw.min(p.width - x0);
            let h = gh.min(p.height - y0);
            members.push((i, x0, y0));
            shapes.push(Shape {
                width: w,
                height: h,
                hshift: p.hshift,
                vshift: p.vshift,
            });
        }
        if shapes.is_empty() {
            return Ok(());
        }
        let planes = decode_stream(r, &shapes, stream, global, self.bit_depth)?;
        for ((i, x0, y0), local) in members.into_iter().zip(planes) {
            let dest = &mut self.planes[i];
            for y in 0..local.height {
                let at = (y0 + y) * dest.width + x0;
                dest.data[at..at + local.width].copy_from_slice(&local.data[y * local.width..(y + 1) * local.width]);
            }
        }
        Ok(())
    }

    /// Undo the global transforms, returning the frame's channels.
    ///
    /// # Errors
    ///
    /// For inconsistent transforms.
    pub fn finish(mut self) -> Result<Vec<Plane>> {
        inverse(&self.header.transforms, &mut self.planes, self.bit_depth)?;
        Ok(self.planes)
    }
}
