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

/// The clamped gradient predictor: `W + N - NW`, kept between `W` and `N`.
const fn gradient(w: i32, n: i32, nw: i32) -> i32 {
    let (lo, hi) = if w < n { (w, n) } else { (n, w) };
    let g = w + n - nw;
    if g < lo {
        lo
    } else if g > hi {
        hi
    } else {
        g
    }
}

/// The residual tokens of the `width` x `height` region of `channel` at
/// `(x0, y0)`, predicted from that region alone (a group decodes as an
/// image of its own), in context `ctx`.
pub fn residuals(
    channel: &Channel,
    (x0, y0, width, height): (usize, usize, usize, usize),
    ctx: u32,
    out: &mut Vec<Token>,
) {
    let at = |x: usize, y: usize| channel.samples[(y0 + y) * channel.width + x0 + x];
    for y in 0..height {
        for x in 0..width {
            // Edges as the specification defines them: a missing neighbour
            // is the nearest one that exists, and nothing at all is 0.
            let w = if x > 0 {
                at(x - 1, y)
            } else if y > 0 {
                at(x, y - 1)
            } else {
                0
            };
            let n = if y > 0 { at(x, y - 1) } else { w };
            let nw = if x > 0 && y > 0 { at(x - 1, y - 1) } else { w };
            out.push(Token::signed(ctx, at(x, y) - gradient(w, n, nw)));
        }
    }
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

/// A tree with one leaf per channel, all with the gradient predictor.
#[derive(Debug, Clone)]
pub struct Tree {
    /// The tree's own stream, in the contexts of [`tree_ctx`].
    pub tokens: Vec<Token>,
    /// Each channel's leaf, which is its residual context.
    pub leaf_of: Vec<u32>,
}

impl Tree {
    /// The tree for `channels` channels.
    #[must_use]
    pub fn per_channel(channels: usize) -> Self {
        /// A node covering channels `lo..=hi`.
        #[derive(Clone, Copy)]
        struct Node {
            lo: usize,
            hi: usize,
        }
        let mut tokens = Vec::new();
        let mut leaf_of = vec![0; channels];
        let mut leaves = 0;
        // Nodes are written breadth first: a decision's two children follow
        // the nodes already queued.
        let mut queue = std::collections::VecDeque::from([Node {
            lo: 0,
            hi: channels.saturating_sub(1),
        }]);
        while let Some(Node { lo, hi }) = queue.pop_front() {
            if lo >= hi {
                tokens.push(Token::new(tree_ctx::PROPERTY, 0)); // a leaf
                tokens.push(Token::new(tree_ctx::PREDICTOR, 5)); // gradient
                tokens.push(Token::signed(tree_ctx::OFFSET, 0));
                tokens.push(Token::new(tree_ctx::MUL_LOG, 0));
                tokens.push(Token::new(tree_ctx::MUL_BITS, 0));
                if let Some(slot) = leaf_of.get_mut(lo) {
                    *slot = leaves;
                }
                leaves += 1;
                continue;
            }
            // "channel > mid" goes to the first child, the rest to the
            // second.
            let mid = (lo + hi) / 2;
            tokens.push(Token::new(tree_ctx::PROPERTY, 1)); // property 0 + 1
            tokens.push(Token::signed(tree_ctx::VALUE, mid as i32));
            queue.push_back(Node { lo: mid + 1, hi });
            queue.push_back(Node { lo, hi: mid });
        }
        Self { tokens, leaf_of }
    }

    /// How many leaves, and so residual contexts, the tree has.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.leaf_of.len().max(1)
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
        let mut out = Vec::new();
        residuals(&c, (0, 0, 4, 3), 0, &mut out);
        assert_eq!(out[0], Token::signed(0, 9));
        assert!(out[1..].iter().all(|t| t.value == 0));
    }

    #[test]
    fn the_tree_gives_every_channel_its_own_leaf() {
        for n in 1..=5 {
            let tree = Tree::per_channel(n);
            let mut leaves = tree.leaf_of.clone();
            leaves.sort_unstable();
            assert_eq!(leaves, (0..n as u32).collect::<Vec<_>>());
            let decisions = tree
                .tokens
                .iter()
                .filter(|t| t.ctx == 1 && t.value == 1)
                .count();
            assert_eq!(decisions, n - 1);
        }
        // Four channels: root "c > 1", then "c > 2" and "c > 0"; leaves in
        // breadth-first order are channels 3, 2, 1, 0.
        assert_eq!(Tree::per_channel(4).leaf_of, [3, 2, 1, 0]);
    }
}
