//! Modular image coding (ISO/IEC 18181-1 §H), lossless.
//!
//! A Modular image is a set of integer channels. Each sample is predicted
//! from its decoded neighbours and the residual coded in a context chosen by
//! a decision tree over "properties" of the sample's position. This encoder
//! uses the simplest useful shape of that machinery:
//!
//! - colour goes through the reversible YCoCg transform (RCT type 6);
//! - every channel is predicted by the clamped gradient (predictor 5);
//! - the tree splits on the channel index alone, so each channel has its
//!   own context and code.

#![allow(
    clippy::indexing_slicing,
    reason = "sample indices are within each channel's width x height, and \
              context indices within the tree's leaves"
)]

use crate::bits::BitWriter;
use crate::entropy::Token;
use crate::predict::{Predictor, Region, WeightedState, gradient, properties};

/// One channel of samples, row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    pub width: usize,
    pub height: usize,
    pub samples: Vec<i32>,
}

impl Channel {
    /// A channel of zeros.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            samples: vec![0; width * height],
        }
    }
}

/// Forward YCoCg-R on the first three channels, in place: they become Y, Co
/// and Cg, which the decoder's RCT type 6 turns back into R, G and B.
pub fn forward_ycocg(channels: &mut [Channel]) {
    let [r, g, b, ..] = channels else {
        return;
    };
    for ((r, g), b) in r.samples.iter_mut().zip(&mut g.samples).zip(&mut b.samples) {
        let co = *r - *b;
        let tmp = *b + (co >> 1);
        let cg = *g - tmp;
        let y = tmp + (cg >> 1);
        (*r, *g, *b) = (y, co, cg);
    }
}

/// A tree node: a decision on a property, or a leaf naming a predictor and
/// a context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node {
    /// `property > value` goes to node `first`, otherwise to `first + 1`.
    Split {
        property: u32,
        value: i32,
        first: usize,
    },
    /// The predictor and the residuals' context.
    Leaf { predictor: Predictor, ctx: u32 },
}

/// A decision tree, nodes in the breadth-first order it is written in.
#[derive(Debug, Clone)]
pub struct Tree {
    nodes: Vec<Node>,
    leaves: usize,
}

/// The contexts a tree stream codes its fields in.
mod tree_ctx {
    pub const VALUE: u32 = 0;
    pub const PROPERTY: u32 = 1;
    pub const PREDICTOR: u32 = 2;
    pub const OFFSET: u32 = 3;
    pub const MUL_LOG: u32 = 4;
    pub const MUL_BITS: u32 = 5;
}

impl Tree {
    /// Split ranges breadth first: `split(lo, hi)` gives the decision for a
    /// range, or `None` for a leaf, and the two sub-ranges.
    fn build(
        root: (usize, usize),
        predictor: Predictor,
        split: impl Fn(usize, usize) -> Option<(u32, i32, (usize, usize), (usize, usize))>,
    ) -> Self {
        let mut nodes = vec![Node::Leaf { predictor, ctx: 0 }];
        let mut queue = std::collections::VecDeque::from([(root, 0)]);
        while let Some(((lo, hi), pos)) = queue.pop_front() {
            let Some((property, value, first, second)) = split(lo, hi) else {
                continue;
            };
            nodes[pos] = Node::Split {
                property,
                value,
                first: nodes.len(),
            };
            queue.push_back((first, nodes.len()));
            nodes.push(Node::Leaf { predictor, ctx: 0 });
            queue.push_back((second, nodes.len()));
            nodes.push(Node::Leaf { predictor, ctx: 0 });
        }
        // Leaves take contexts in the order they are written.
        let mut leaves = 0;
        for node in &mut nodes {
            if let Node::Leaf { ctx, .. } = node {
                *ctx = leaves;
                leaves += 1;
            }
        }
        Self {
            nodes,
            leaves: leaves as usize,
        }
    }

    /// One leaf per channel, all with `predictor`.
    #[must_use]
    #[cfg(test)]
    pub fn per_channel(channels: usize, predictor: Predictor) -> Self {
        Self::build((0, channels.saturating_sub(1)), predictor, |lo, hi| {
            (lo < hi).then(|| {
                let mid = (lo + hi) / 2;
                (0, mid as i32, (mid + 1, hi), (lo, mid))
            })
        })
    }

    /// libjxl's fixed tree (`MakeFixedTree`): a balanced split of
    /// `property` at `cutoffs`, shallower for small images, its cutoffs
    /// scaled up for deep samples.
    #[must_use]
    pub fn fixed(
        property: u32,
        cutoffs: &[i32],
        predictor: Predictor,
        pixels: usize,
        bit_depth: u32,
    ) -> Self {
        let log_px = usize::BITS - pixels.max(1).saturating_sub(1).leading_zeros();
        let min_gap = if log_px < 14 {
            8 * (14 - log_px as usize)
        } else {
            0
        };
        let mul = 1
            << if bit_depth > 11 {
                (bit_depth - 11).min(4)
            } else {
                0
            };
        Self::build((0, cutoffs.len()), predictor, |begin, end| {
            (begin + min_gap < end).then(|| {
                let split = (begin + end) / 2;
                (
                    property,
                    cutoffs[split] * mul,
                    (split + 1, end),
                    (begin, split),
                )
            })
        })
    }

    /// How many leaves, and so residual contexts, the tree has.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.leaves.max(1)
    }

    /// Whether any leaf predicts with, or any decision reads, the weighted
    /// predictor, which then has to run.
    #[must_use]
    pub fn uses_weighted(&self) -> bool {
        self.nodes.iter().any(|node| match node {
            Node::Split { property, .. } => *property == 15,
            Node::Leaf { predictor, .. } => *predictor == Predictor::Weighted,
        })
    }

    /// The leaf `properties` reach.
    #[must_use]
    pub fn leaf(&self, properties: &[i32; 16]) -> (Predictor, u32) {
        let mut at = 0;
        loop {
            match self.nodes.get(at) {
                Some(&Node::Split {
                    property,
                    value,
                    first,
                }) => {
                    at = if properties[property as usize] > value {
                        first
                    } else {
                        first + 1
                    };
                }
                Some(&Node::Leaf { predictor, ctx }) => return (predictor, ctx),
                None => return (Predictor::Gradient, 0),
            }
        }
    }

    /// The tree's own stream, in the contexts of `tree_ctx`.
    #[must_use]
    pub fn tokens(&self) -> Vec<Token> {
        let mut tokens = Vec::new();
        for node in &self.nodes {
            match *node {
                Node::Split {
                    property, value, ..
                } => {
                    tokens.push(Token::new(tree_ctx::PROPERTY, property + 1));
                    tokens.push(Token::signed(tree_ctx::VALUE, value));
                }
                Node::Leaf { predictor, .. } => {
                    tokens.push(Token::new(tree_ctx::PROPERTY, 0));
                    tokens.push(Token::new(tree_ctx::PREDICTOR, predictor.id()));
                    tokens.push(Token::signed(tree_ctx::OFFSET, 0));
                    tokens.push(Token::new(tree_ctx::MUL_LOG, 0));
                    tokens.push(Token::new(tree_ctx::MUL_BITS, 0));
                }
            }
        }
        tokens
    }
}

/// The residual tokens of `region` of channel number `index`, under `tree`.
/// The region is predicted from itself alone: a group decodes as an image
/// of its own.
pub fn residuals(region: &Region<'_>, index: u32, tree: &Tree, out: &mut Vec<Token>) {
    let mut weighted = tree
        .uses_weighted()
        .then(|| WeightedState::new(region.width));
    for y in 0..region.height {
        let mut prev_grad = 0;
        for x in 0..region.width {
            let nb = region.neighbours(x, y);
            let wp = weighted.as_ref().map(|state| state.predict(&nb));
            let props = properties(index, x, y, &nb, prev_grad, wp.map_or(0, |p| p.max_error));
            let (predictor, ctx) = tree.leaf(&props);
            let prediction = match (predictor, wp) {
                (Predictor::Weighted, Some(p)) => p.value(),
                _ => gradient(&nb),
            };
            let sample = region.at(x, y);
            out.push(Token::signed(ctx, sample.wrapping_sub(prediction)));
            if let (Some(state), Some(p)) = (weighted.as_mut(), wp) {
                state.record(&p, sample);
            }
            prev_grad = props[9];
        }
    }
}

/// Write a Modular sub-image header: the global tree, default weighted
/// predictor parameters, and the YCoCg transform if `rct`.
pub fn write_header(w: &mut BitWriter, rct: bool) {
    use crate::bits::{bits, val};
    w.bool(true); // use the global tree
    w.bool(true); // default weighted-predictor parameters
    w.u32(u32::from(rct), [val(0), val(1), bits(2, 4), bits(18, 8)]);
    if rct {
        w.write(2, 0); // RCT
        w.u32(0, [bits(0, 3), bits(8, 6), bits(72, 10), bits(1096, 13)]); // from channel 0
        w.u32(6, [val(6), bits(0, 2), bits(2, 4), bits(10, 6)]); // YCoCg
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values"
)]
mod tests {
    use super::*;

    /// The decoder's inverse of RCT type 6.
    fn inverse(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
        let tmp = y - (cg >> 1);
        let g = cg + tmp;
        let b = tmp - (co >> 1);
        let r = b + co;
        (r, g, b)
    }

    #[test]
    fn ycocg_inverts_exactly() {
        for (r, g, b) in [
            (0, 0, 0),
            (255, 0, 128),
            (17, 200, 255),
            (65_535, 1, 40_000),
        ] {
            let mut channels = vec![Channel::new(1, 1), Channel::new(1, 1), Channel::new(1, 1)];
            channels[0].samples[0] = r;
            channels[1].samples[0] = g;
            channels[2].samples[0] = b;
            forward_ycocg(&mut channels);
            let back = inverse(
                channels[0].samples[0],
                channels[1].samples[0],
                channels[2].samples[0],
            );
            assert_eq!(back, (r, g, b));
        }
    }

    #[test]
    fn a_flat_channel_has_zero_residuals_after_the_first() {
        let mut c = Channel::new(4, 3);
        c.samples.fill(9);
        let region = Region {
            channel: &c,
            x0: 0,
            y0: 0,
            width: 4,
            height: 3,
        };
        for predictor in [Predictor::Gradient, Predictor::Weighted] {
            let mut out = Vec::new();
            residuals(&region, 0, &Tree::per_channel(1, predictor), &mut out);
            assert_eq!(out[0], Token::signed(0, 9), "{predictor:?}");
            assert!(out[1..].iter().all(|t| t.value == 0), "{predictor:?}");
        }
    }

    #[test]
    fn the_per_channel_tree_gives_every_channel_its_own_leaf() {
        for n in 1..=5_u32 {
            let tree = Tree::per_channel(n as usize, Predictor::Gradient);
            assert_eq!(tree.leaves(), n as usize);
            let mut leaves: Vec<u32> = (0..n)
                .map(|c| {
                    let mut props = [0; 16];
                    props[0] = c as i32;
                    tree.leaf(&props).1
                })
                .collect();
            leaves.sort_unstable();
            assert_eq!(leaves, (0..n).collect::<Vec<_>>());
        }
        // Four channels: leaves in breadth-first order are channels 3, 2, 1, 0.
        let tree = Tree::per_channel(4, Predictor::Gradient);
        let ctx = |c: i32| {
            let mut props = [0; 16];
            props[0] = c;
            tree.leaf(&props).1
        };
        assert_eq!([ctx(0), ctx(1), ctx(2), ctx(3)], [3, 2, 1, 0]);
    }

    #[test]
    fn the_fixed_tree_follows_libjxl() {
        let cutoffs = [-3, -1, 0, 1, 3];
        // Large images get every cutoff: six leaves.
        let tree = Tree::fixed(15, &cutoffs, Predictor::Weighted, 1 << 20, 8);
        assert_eq!(tree.leaves(), 6);
        assert!(tree.uses_weighted());
        let ctx = |v: i32| {
            let mut props = [0; 16];
            props[15] = v;
            tree.leaf(&props).1
        };
        // Contexts rise as the property falls: greater-than goes first.
        let all: Vec<u32> = [10, 2, 1, 0, -1, -10].iter().map(|&v| ctx(v)).collect();
        let mut sorted = all.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 6, "{all:?}");
        // Small images, fewer: 64 pixels leave a single leaf.
        assert_eq!(
            Tree::fixed(15, &cutoffs, Predictor::Weighted, 64, 8).leaves(),
            1
        );
        // Sixteen-bit cutoffs are sixteen times larger.
        let deep = Tree::fixed(15, &cutoffs, Predictor::Weighted, 1 << 20, 16);
        assert!(deep.tokens().contains(&Token::signed(0, 16)));
    }
}
