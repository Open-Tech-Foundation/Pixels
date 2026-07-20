//! The byte-budgeted tile cache.
//!
//! Intermediates are keyed by `(node, region)` (ARCHITECTURE §Layer 4) so that
//! a tile computed for one consumer can be reused by another: graph branches
//! sharing a prefix, and spatial ops whose input regions overlap. Without it,
//! a diamond-shaped graph recomputes its shared prefix once per branch.
//!
//! # Eviction bounds retention, not liveness
//!
//! Entries are `Arc<TileBuf>`. Evicting one drops *the cache's* reference; a
//! caller already holding the `Arc` keeps its tile alive and valid. That makes
//! eviction free of correctness hazards — there is nothing to pin, and a task
//! can never have its input yanked mid-computation.
//!
//! The consequence is that [`TileCache::budget`] bounds what the cache
//! **retains**, not total live tile memory. Callers holding many tiles at once
//! exceed it by construction; bounding *that* is the scheduler's job, by
//! limiting how much work is in flight.

use crate::{NodeId, Region, TileBuf};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// Identifies one cached intermediate: a region of one node's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    /// The graph node that produced the tile.
    pub node: NodeId,
    /// The region of that node's output the tile covers.
    pub region: Region,
}

impl TileKey {
    /// A key for `region` of `node`'s output.
    #[must_use]
    pub const fn new(node: NodeId, region: Region) -> Self {
        Self { node, region }
    }
}

/// Counters describing cache behaviour, for tests and diagnostics.
///
/// These are observations, not API guarantees: eviction policy details are
/// explicitly internal (ARCHITECTURE §Layer 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct CacheStats {
    /// Lookups that found a retained tile.
    pub hits: u64,
    /// Lookups that found nothing.
    pub misses: u64,
    /// Tiles dropped to stay within budget.
    pub evictions: u64,
    /// Tiles offered to the cache.
    pub insertions: u64,
    /// Tiles not retained because one alone exceeded the whole budget.
    pub rejections: u64,
}

impl CacheStats {
    /// Hit rate over all lookups, or [`None`] if there were none.
    #[must_use]
    pub fn hit_rate(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        (total > 0).then(|| self.hits as f64 / total as f64)
    }
}

/// One retained tile and its recency tick.
#[derive(Debug)]
struct Entry {
    tile: std::sync::Arc<TileBuf>,
    bytes: usize,
    tick: u64,
}

/// The cache's mutable state, behind one lock.
#[derive(Debug)]
struct Inner {
    entries: HashMap<TileKey, Entry>,
    /// Recency index: tick → key, ordered oldest-first. Kept in step with
    /// `entries`, so the least-recently-used key is always the first entry.
    recency: BTreeMap<u64, TileKey>,
    bytes: usize,
    next_tick: u64,
    stats: CacheStats,
}

/// A byte-budgeted LRU cache of graph intermediates.
///
/// Shared across scheduler threads; all methods take `&self`.
#[derive(Debug)]
pub struct TileCache {
    inner: Mutex<Inner>,
    budget: usize,
}

impl TileCache {
    /// The default retention budget: 64 MiB.
    pub const DEFAULT_BUDGET: usize = 64 * 1024 * 1024;

    /// A cache retaining at most `budget` bytes of tiles.
    ///
    /// A budget of zero disables retention entirely: every lookup misses,
    /// which is a useful way to test that a pipeline is correct without any
    /// caching at all.
    #[must_use]
    pub fn new(budget: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                recency: BTreeMap::new(),
                bytes: 0,
                next_tick: 0,
                stats: CacheStats::default(),
            }),
            budget,
        }
    }

    /// The retention budget in bytes.
    #[must_use]
    pub const fn budget(&self) -> usize {
        self.budget
    }

    /// Look up a tile, marking it most-recently-used on a hit.
    ///
    /// The returned [`Arc`] keeps the tile alive regardless of later eviction.
    ///
    /// [`Arc`]: std::sync::Arc
    #[must_use]
    pub fn get(&self, key: &TileKey) -> Option<std::sync::Arc<TileBuf>> {
        let mut inner = self.lock();
        let tick = inner.next_tick;
        let Some(entry) = inner.entries.get_mut(key) else {
            inner.stats.misses += 1;
            return None;
        };
        let previous = entry.tick;
        entry.tick = tick;
        let tile = std::sync::Arc::clone(&entry.tile);
        inner.next_tick += 1;
        inner.recency.remove(&previous);
        inner.recency.insert(tick, *key);
        inner.stats.hits += 1;
        Some(tile)
    }

    /// Offer a tile to the cache, evicting as needed to stay within budget.
    ///
    /// Returns the tile, so a caller can insert and use it in one step. A tile
    /// larger than the entire budget is returned without being retained rather
    /// than evicting everything to make room for something that cannot fit.
    pub fn insert(&self, key: TileKey, tile: std::sync::Arc<TileBuf>) -> std::sync::Arc<TileBuf> {
        let bytes = tile.bytes().len();
        let mut inner = self.lock();
        inner.stats.insertions += 1;

        if bytes > self.budget {
            inner.stats.rejections += 1;
            return tile;
        }
        // Replacing an existing entry releases its bytes first.
        if let Some(previous) = inner.entries.remove(&key) {
            inner.bytes -= previous.bytes;
            inner.recency.remove(&previous.tick);
        }
        let tick = inner.next_tick;
        inner.next_tick += 1;
        inner.bytes += bytes;
        inner.entries.insert(
            key,
            Entry {
                tile: std::sync::Arc::clone(&tile),
                bytes,
                tick,
            },
        );
        inner.recency.insert(tick, key);
        inner.evict_to_fit(self.budget);
        tile
    }

    /// Bytes currently retained.
    #[must_use]
    pub fn bytes_used(&self) -> usize {
        self.lock().bytes
    }

    /// Number of tiles currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether the cache retains nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A snapshot of the cache counters.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.lock().stats
    }

    /// Drop every retained tile, keeping the accumulated statistics.
    pub fn clear(&self) {
        let mut inner = self.lock();
        inner.entries.clear();
        inner.recency.clear();
        inner.bytes = 0;
    }

    /// Lock the inner state, recovering from a poisoned mutex.
    ///
    /// A cache is pure derived data: if a thread panicked holding the lock the
    /// worst case is a torn *statistic*, never a torn pixel, since tiles are
    /// immutable `Arc`s. Propagating poisoning here would turn an unrelated
    /// panic into a permanently unusable engine, so the guard is taken either
    /// way.
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl Default for TileCache {
    fn default() -> Self {
        Self::new(Self::DEFAULT_BUDGET)
    }
}

impl Inner {
    /// Drop least-recently-used entries until within `budget`.
    fn evict_to_fit(&mut self, budget: usize) {
        while self.bytes > budget {
            // `recency` is ordered by tick, so the first key is the LRU one.
            let Some((&tick, &key)) = self.recency.iter().next() else {
                // No entries left to evict; nothing more can be done.
                break;
            };
            self.recency.remove(&tick);
            if let Some(entry) = self.entries.remove(&key) {
                self.bytes -= entry.bytes;
                self.stats.evictions += 1;
            }
        }
    }
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
    use crate::PixelFormat;
    use std::sync::Arc;

    /// A `size` × 1 Gray8 tile, i.e. exactly `size` bytes.
    fn tile(size: u32) -> Arc<TileBuf> {
        Arc::new(TileBuf::zeroed(Region::from_size(size, 1), PixelFormat::Gray8).unwrap())
    }

    /// A key for `region` on a freshly minted node, i.e. distinct from every
    /// other key this helper has returned.
    fn key(region: Region) -> TileKey {
        TileKey::new(fresh_node_id(), region)
    }

    /// Mint a distinct `NodeId` without building a whole graph.
    fn fresh_node_id() -> NodeId {
        use crate::testing::CountingProducer;
        use crate::{Format, Image, ImageDescriptor};
        let descriptor = ImageDescriptor::new(1, 1, PixelFormat::Gray8).unwrap();
        let image = Image::from_producer(Arc::new(CountingProducer::new(descriptor)), Format::Raw);
        image.node().id()
    }

    #[test]
    fn a_tile_round_trips_through_the_cache() {
        let cache = TileCache::new(1024);
        let k = key(Region::from_size(4, 1));
        assert!(cache.get(&k).is_none(), "empty cache misses");
        cache.insert(k, tile(4));
        let found = cache.get(&k).unwrap();
        assert_eq!(found.bytes().len(), 4);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes_used(), 4);
    }

    #[test]
    fn the_budget_is_never_exceeded_by_retained_bytes() {
        let cache = TileCache::new(100);
        for _ in 0..50 {
            cache.insert(key(Region::from_size(10, 1)), tile(10));
            assert!(
                cache.bytes_used() <= 100,
                "retained {} bytes over a 100 byte budget",
                cache.bytes_used()
            );
        }
        assert!(cache.stats().evictions > 0, "nothing was ever evicted");
    }

    #[test]
    fn eviction_removes_the_least_recently_used_entry() {
        let cache = TileCache::new(30);
        let (a, b, c) = (
            key(Region::from_size(1, 1)),
            key(Region::from_size(2, 1)),
            key(Region::from_size(3, 1)),
        );
        cache.insert(a, tile(10));
        cache.insert(b, tile(10));
        // Touch `a` so `b` becomes the least recently used.
        assert!(cache.get(&a).is_some());
        cache.insert(c, tile(10));
        // Budget is exactly 30, so nothing has been evicted yet.
        assert_eq!(cache.len(), 3);

        // One more forces eviction of `b`, the LRU entry.
        let d = key(Region::from_size(4, 1));
        cache.insert(d, tile(10));
        assert!(cache.get(&b).is_none(), "LRU entry survived");
        assert!(cache.get(&a).is_some(), "recently used entry was evicted");
        assert!(cache.get(&c).is_some());
        assert!(cache.get(&d).is_some());
    }

    #[test]
    fn an_evicted_tile_stays_valid_for_whoever_holds_it() {
        // The property that makes pinning unnecessary: eviction drops the
        // cache's reference, never the caller's.
        let cache = TileCache::new(10);
        let k = key(Region::from_size(10, 1));
        let held = cache.insert(k, tile(10));
        // Force the entry out.
        cache.insert(key(Region::from_size(9, 1)), tile(10));
        assert!(cache.get(&k).is_none(), "expected eviction");
        // The held Arc is still perfectly usable.
        assert_eq!(held.bytes().len(), 10);
        assert!(held.as_tile().is_ok());
    }

    #[test]
    fn a_tile_larger_than_the_budget_is_returned_but_not_retained() {
        let cache = TileCache::new(10);
        let k = key(Region::from_size(50, 1));
        let returned = cache.insert(k, tile(50));
        assert_eq!(returned.bytes().len(), 50, "the tile is still usable");
        assert!(cache.get(&k).is_none());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().rejections, 1);
        // Crucially it did not evict everything else trying to fit.
        assert_eq!(cache.stats().evictions, 0);
    }

    #[test]
    fn a_zero_budget_disables_retention() {
        let cache = TileCache::new(0);
        let k = key(Region::from_size(4, 1));
        let returned = cache.insert(k, tile(4));
        assert_eq!(returned.bytes().len(), 4, "the tile is still returned");
        assert!(cache.get(&k).is_none());
        assert_eq!(cache.bytes_used(), 0);
    }

    #[test]
    fn reinserting_a_key_replaces_it_without_double_counting() {
        let cache = TileCache::new(1000);
        let k = key(Region::from_size(4, 1));
        cache.insert(k, tile(10));
        cache.insert(k, tile(20));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes_used(), 20, "old bytes were not released");
        assert_eq!(cache.get(&k).unwrap().bytes().len(), 20);
    }

    #[test]
    fn keys_distinguish_node_and_region() {
        let cache = TileCache::new(1000);
        let node = fresh_node_id();
        let a = TileKey::new(node, Region::from_size(4, 1));
        let b = TileKey::new(node, Region::new(4, 0, 4, 1));
        cache.insert(a, tile(4));
        assert!(
            cache.get(&b).is_none(),
            "different regions must not collide"
        );

        let other = TileKey::new(fresh_node_id(), Region::from_size(4, 1));
        assert!(
            cache.get(&other).is_none(),
            "different nodes must not collide"
        );
    }

    #[test]
    fn statistics_track_lookups_and_evictions() {
        let cache = TileCache::new(10);
        let k = key(Region::from_size(4, 1));
        assert!(cache.get(&k).is_none());
        cache.insert(k, tile(4));
        assert!(cache.get(&k).is_some());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.insertions, 1);
        assert_eq!(stats.hit_rate(), Some(0.5));
        assert_eq!(TileCache::new(1).stats().hit_rate(), None, "no lookups yet");
    }

    #[test]
    fn clear_drops_entries_but_keeps_statistics() {
        let cache = TileCache::new(1000);
        cache.insert(key(Region::from_size(4, 1)), tile(4));
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.bytes_used(), 0);
        assert_eq!(cache.stats().insertions, 1, "statistics are cumulative");
    }

    #[test]
    fn the_cache_is_usable_from_many_threads() {
        let cache = Arc::new(TileCache::new(4096));
        let keys: Vec<TileKey> = (0..8).map(|i| key(Region::from_size(i + 1, 1))).collect();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let cache = Arc::clone(&cache);
                let keys = keys.clone();
                scope.spawn(move || {
                    for _ in 0..200 {
                        for (i, k) in keys.iter().enumerate() {
                            cache.insert(*k, tile(i as u32 + 1));
                            let _ = cache.get(k);
                        }
                    }
                });
            }
        });
        // The invariant that matters under concurrency: accounting stayed
        // consistent with what is actually retained.
        let inner = cache.lock();
        let actual: usize = inner.entries.values().map(|e| e.bytes).sum();
        assert_eq!(inner.bytes, actual, "byte accounting drifted");
        assert_eq!(
            inner.recency.len(),
            inner.entries.len(),
            "recency index drifted"
        );
        assert!(inner.bytes <= 4096);
    }

    #[test]
    fn a_poisoned_lock_does_not_disable_the_cache() {
        // An unrelated panic must not permanently break the engine.
        let cache = Arc::new(TileCache::new(1000));
        let k = key(Region::from_size(4, 1));
        cache.insert(k, tile(4));
        let poisoner = Arc::clone(&cache);
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.lock();
            panic!("poison the mutex");
        });
        assert!(handle.join().is_err(), "the thread was supposed to panic");
        // The cache still works.
        assert!(cache.get(&k).is_some());
        cache.insert(key(Region::from_size(8, 1)), tile(8));
    }
}
