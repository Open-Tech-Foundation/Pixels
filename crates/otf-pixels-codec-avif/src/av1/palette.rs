//! Palette prediction (spec §5.11.46–§5.11.50, §7.11.4).
//!
//! A palette block codes a small set of colours (up to eight) and, per sample,
//! an index into that set. It is a screen-content tool: flat UI regions with few
//! distinct colours compress far better as an index map than as a transform
//! residual. This module owns the pure parts — merging the neighbour palettes
//! into the prediction cache, and the colour-index context model — while the
//! tile driver owns the reads, because those depend on its symbol decoder and
//! neighbour state.
//!
//! Only the still-image, non-subsampled case is exercised; the maps are stored
//! at full block resolution.

/// `PALETTE_COLORS` (§3): the largest palette.
pub const PALETTE_COLORS: usize = 8;
/// `PALETTE_NUM_NEIGHBORS` (§3).
const PALETTE_NUM_NEIGHBORS: usize = 3;
/// `Palette_Color_Hash_Multipliers` (§9.3).
const HASH_MULTIPLIERS: [u32; PALETTE_NUM_NEIGHBORS] = [1, 2, 2];
/// `Palette_Color_Context` (§9.3): maps a colour-context hash to a CDF context.
/// The `-1` entries are never reached; stored as a sentinel.
const PALETTE_COLOR_CONTEXT: [i8; 9] = [-1, -1, 0, -1, -1, 4, 3, 2, 1];

/// Merge the above and left neighbour palettes into the ascending, de-duplicated
/// prediction cache (`get_palette_cache`, §5.11.50).
///
/// `above` and `left` are the neighbour palettes (already ascending), each
/// truncated to its real length by the caller.
#[must_use]
pub fn palette_cache(above: &[u16], left: &[u16]) -> Vec<u16> {
    let mut cache = Vec::with_capacity(above.len() + left.len());
    let mut ai = 0;
    let mut li = 0;
    let push = |cache: &mut Vec<u16>, v: u16| {
        if cache.last() != Some(&v) {
            cache.push(v);
        }
    };
    while ai < above.len() && li < left.len() {
        let a = above.get(ai).copied().unwrap_or(0);
        let l = left.get(li).copied().unwrap_or(0);
        if l < a {
            push(&mut cache, l);
            li += 1;
        } else {
            push(&mut cache, a);
            ai += 1;
            if l == a {
                li += 1;
            }
        }
    }
    while ai < above.len() {
        if let Some(&v) = above.get(ai) {
            push(&mut cache, v);
        }
        ai += 1;
    }
    while li < left.len() {
        if let Some(&v) = left.get(li) {
            push(&mut cache, v);
        }
        li += 1;
    }
    cache
}

/// The colour ordering and CDF context for one colour-index sample
/// (`get_palette_color_context`, §5.11.50). `neighbours` are the already-decoded
/// left, above-left, and above indices `(has, index)`; the score weights are 2,
/// 1, 2 respectively.
#[must_use]
pub fn color_context(
    left: Option<u8>,
    above_left: Option<u8>,
    above: Option<u8>,
    n: usize,
) -> ([u8; PALETTE_COLORS], usize) {
    let mut scores = [0_u32; PALETTE_COLORS];
    let mut order = [0_u8; PALETTE_COLORS];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i as u8;
    }
    let mut add = |idx: Option<u8>, weight: u32| {
        if let Some(v) = idx {
            if let Some(s) = scores.get_mut(usize::from(v)) {
                *s += weight;
            }
        }
    };
    add(left, 2);
    add(above_left, 1);
    add(above, 2);

    // Partial selection sort of the top PALETTE_NUM_NEIGHBORS by score,
    // carrying the colour order alongside.
    for i in 0..PALETTE_NUM_NEIGHBORS.min(n) {
        let mut max_idx = i;
        let mut max_score = scores.get(i).copied().unwrap_or(0);
        for j in (i + 1)..n {
            let sj = scores.get(j).copied().unwrap_or(0);
            if sj > max_score {
                max_score = sj;
                max_idx = j;
            }
        }
        if max_idx != i {
            let saved_order = order.get(max_idx).copied().unwrap_or(0);
            let mut k = max_idx;
            while k > i {
                let prev_s = scores.get(k - 1).copied().unwrap_or(0);
                let prev_o = order.get(k - 1).copied().unwrap_or(0);
                if let Some(s) = scores.get_mut(k) {
                    *s = prev_s;
                }
                if let Some(o) = order.get_mut(k) {
                    *o = prev_o;
                }
                k -= 1;
            }
            if let Some(s) = scores.get_mut(i) {
                *s = max_score;
            }
            if let Some(o) = order.get_mut(i) {
                *o = saved_order;
            }
        }
    }

    let mut hash = 0_u32;
    for i in 0..PALETTE_NUM_NEIGHBORS {
        hash += scores.get(i).copied().unwrap_or(0) * HASH_MULTIPLIERS.get(i).copied().unwrap_or(0);
    }
    let ctx = PALETTE_COLOR_CONTEXT
        .get(hash as usize)
        .copied()
        .filter(|&c| c >= 0)
        .unwrap_or(0) as usize;
    (order, ctx)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;

    #[test]
    fn cache_merges_and_dedups_ascending() {
        assert_eq!(
            palette_cache(&[10, 20, 30], &[15, 20, 25]),
            vec![10, 15, 20, 25, 30]
        );
        assert_eq!(palette_cache(&[], &[5, 9]), vec![5, 9]);
        assert_eq!(palette_cache(&[1, 2], &[]), vec![1, 2]);
    }

    #[test]
    fn color_context_of_no_neighbours_orders_identity() {
        let (order, ctx) = color_context(None, None, None, 4);
        assert_eq!(order, [0, 1, 2, 3, 4, 5, 6, 7]);
        // All scores zero -> hash 0 -> Palette_Color_Context[0] is a sentinel,
        // clamped to 0.
        assert_eq!(ctx, 0);
    }

    #[test]
    fn a_dominant_neighbour_sorts_to_the_front() {
        // Left and above both index 2 -> score 4; it should lead the order.
        let (order, _ctx) = color_context(Some(2), None, Some(2), 4);
        assert_eq!(order[0], 2);
    }
}
