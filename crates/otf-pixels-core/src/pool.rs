//! The work-stealing thread pool.
//!
//! Per ADR-0008 the lock-free deque comes from `crossbeam-deque`; the pool,
//! the parking policy and every scheduling decision are ours.
//!
//! # Shape
//!
//! Each worker owns a LIFO deque and pushes new work onto it. LIFO on the
//! owner's end is deliberate: the most recently produced tile is the one most
//! likely still in cache, so depth-first execution on each worker keeps the
//! working set small. Idle workers steal from the *other* end of a victim's
//! deque, taking the oldest and coldest work — which is also the work least
//! likely to be stolen back immediately.
//!
//! # Panics are contained, not propagated
//!
//! A panicking task must not poison the pool or abort the process
//! (ARCHITECTURE §Failure model). Tasks are caught, and the panic is reported
//! to the submitter as a [`PixelsError`] rather than resumed. Ops are written
//! to return errors, so a panic here means a defect — but a defect in one tile
//! still must not take the host process down with it.

use crate::{PixelsError, Result};
use crossbeam_deque::{Injector, Stealer, Worker};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// A unit of work for the pool.
type Task = Box<dyn FnOnce() + Send>;

/// Shared state every worker sees.
struct Shared {
    /// Global queue: where non-worker threads submit work.
    injector: Injector<Task>,
    /// One stealer per worker, for cross-worker theft.
    stealers: Vec<Stealer<Task>>,
    /// Set once to tell workers to wind down.
    shutdown: AtomicBool,
    /// Tasks queued but not yet finished; drives idle parking.
    pending: AtomicUsize,
    /// Parking lot for idle workers.
    idle: Mutex<()>,
    wake: Condvar,
}

impl Shared {
    /// Wake one parked worker, if any.
    fn signal_one(&self) {
        // The lock is taken so a worker cannot check `pending`, decide to
        // park, and miss this notification in between.
        let _guard = self
            .idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.wake.notify_one();
    }

    /// Wake every parked worker.
    fn signal_all(&self) {
        let _guard = self
            .idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.wake.notify_all();
    }

    /// Take one task, preferring local work, then global, then theft.
    fn find_task(&self, local: &Worker<Task>) -> Option<Task> {
        // Local LIFO first: hottest tile, no synchronisation.
        if let Some(task) = local.pop() {
            return Some(task);
        }
        loop {
            // Global queue next, in batches so a submitter burst spreads out.
            match self.injector.steal_batch_and_pop(local) {
                crossbeam_deque::Steal::Success(task) => return Some(task),
                crossbeam_deque::Steal::Retry => continue,
                crossbeam_deque::Steal::Empty => break,
            }
        }
        // Finally steal from a peer's cold end.
        for stealer in &self.stealers {
            loop {
                match stealer.steal_batch_and_pop(local) {
                    crossbeam_deque::Steal::Success(task) => return Some(task),
                    crossbeam_deque::Steal::Retry => continue,
                    crossbeam_deque::Steal::Empty => break,
                }
            }
        }
        None
    }
}

/// A work-stealing pool of worker threads.
///
/// Dropping the pool signals shutdown and joins every worker, so no task
/// outlives it.
#[derive(Debug)]
pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<std::thread::JoinHandle<()>>,
    threads: usize,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("workers", &self.stealers.len())
            .field("pending", &self.pending.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl ThreadPool {
    /// Build a pool with `threads` workers.
    ///
    /// `threads` is clamped to at least one. Use [`ThreadPool::default_threads`]
    /// for the machine's parallelism.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the operating system refuses to spawn a
    /// worker thread.
    pub fn new(threads: usize) -> Result<Self> {
        let threads = threads.max(1);
        let mut locals = Vec::with_capacity(threads);
        let mut stealers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let worker = Worker::new_lifo();
            stealers.push(worker.stealer());
            locals.push(worker);
        }
        let shared = Arc::new(Shared {
            injector: Injector::new(),
            stealers,
            shutdown: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
            idle: Mutex::new(()),
            wake: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(threads);
        for (index, local) in locals.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let handle = std::thread::Builder::new()
                .name(format!("otf-pixels-worker-{index}"))
                .spawn(move || worker_loop(&shared, &local))
                .map_err(|e| PixelsError::io("spawning a scheduler worker thread", e))?;
            workers.push(handle);
        }
        Ok(Self {
            shared,
            workers,
            threads,
        })
    }

    /// The parallelism to use when the caller has no preference.
    ///
    /// Falls back to one thread where the platform cannot report it, which
    /// yields a correct if serial engine rather than a failure.
    #[must_use]
    pub fn default_threads() -> usize {
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    }

    /// A pool sized to [`ThreadPool::default_threads`].
    ///
    /// # Errors
    ///
    /// As [`ThreadPool::new`].
    pub fn with_default_threads() -> Result<Self> {
        Self::new(Self::default_threads())
    }

    /// How many workers this pool runs.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
    }

    /// Queue `task` for execution.
    ///
    /// Returns immediately. Panics inside `task` are contained by the worker
    /// and do not propagate here.
    pub fn spawn(&self, task: impl FnOnce() + Send + 'static) {
        self.shared.pending.fetch_add(1, Ordering::SeqCst);
        self.shared.injector.push(Box::new(task));
        self.shared.signal_one();
    }

    /// Run `tasks` on the pool and return once every one has finished.
    ///
    /// # Why `'static`
    ///
    /// Tasks must be `'static` because they run on long-lived worker threads
    /// that outlive this call. Erasing a shorter lifetime is what `rayon` uses
    /// `unsafe` for, and ADR-0008 keeps `unsafe_code = "forbid"` in our
    /// crates. This costs the scheduler nothing: tiles are already
    /// `Arc<TileBuf>` and graph nodes `Arc<Node>`, so a task moves cheap
    /// handles rather than borrowing.
    ///
    /// The calling thread blocks. Calling `run_all` from *inside* a pool task
    /// is not supported and can deadlock; the scheduler submits one flat batch
    /// per wave instead of nesting.
    ///
    /// # Errors
    ///
    /// Returns the failing task's error, choosing the **lowest-indexed**
    /// failure when several fail. Which worker happens to fail first is a race;
    /// which task index is lowest is not, so the reported error is
    /// deterministic (SPEC §Guarantees 2).
    pub fn run_all<F>(&self, tasks: Vec<F>) -> Result<()>
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        if tasks.is_empty() {
            return Ok(());
        }
        let batch = Arc::new(Batch::new(tasks.len()));
        for (index, task) in tasks.into_iter().enumerate() {
            let batch = Arc::clone(&batch);
            self.spawn(move || {
                let outcome = catch(task);
                batch.finish(index, outcome);
            });
        }
        batch.wait();
        batch.first_error()
    }
}

/// Completion tracking for one [`ThreadPool::run_all`] batch.
#[derive(Debug)]
struct Batch {
    /// One slot per task, so failures can be ranked by task index rather than
    /// by which worker got there first.
    slots: Vec<Mutex<Option<PixelsError>>>,
    remaining: AtomicUsize,
    finished: Mutex<bool>,
    complete: Condvar,
}

impl Batch {
    fn new(count: usize) -> Self {
        Self {
            slots: (0..count).map(|_| Mutex::new(None)).collect(),
            remaining: AtomicUsize::new(count),
            finished: Mutex::new(false),
            complete: Condvar::new(),
        }
    }

    /// Record one task's outcome and wake the waiter if it was the last.
    fn finish(&self, index: usize, outcome: Result<()>) {
        if let Err(error) = outcome
            && let Some(slot) = self.slots.get(index)
        {
            *slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(error);
        }
        if self.remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
            let mut finished = self
                .finished
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *finished = true;
            self.complete.notify_all();
        }
    }

    /// Block until every task in the batch has finished.
    fn wait(&self) {
        let mut finished = self
            .finished
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while !*finished {
            finished = self
                .complete
                .wait(finished)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The lowest-indexed failure, if any task failed.
    fn first_error(&self) -> Result<()> {
        for slot in &self.slots {
            let mut slot = slot
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(error) = slot.take() {
                return Err(error);
            }
        }
        Ok(())
    }
}

/// Run `task`, converting a panic into an error.
fn catch<F: FnOnce() -> Result<()>>(task: F) -> Result<()> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)) {
        Ok(result) => result,
        Err(payload) => {
            let detail = panic_message(payload.as_ref());
            Err(PixelsError::graph(format!(
                "a scheduler task panicked: {detail}"
            )))
        }
    }
}

/// Best-effort text of a panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        return (*text).to_owned();
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    "non-string panic payload".to_owned()
}

/// The body of each worker thread.
fn worker_loop(shared: &Arc<Shared>, local: &Worker<Task>) {
    loop {
        if shared.shutdown.load(Ordering::SeqCst) && shared.pending.load(Ordering::SeqCst) == 0 {
            return;
        }
        if let Some(task) = shared.find_task(local) {
            // A panicking task must not unwind out of the worker thread.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
            shared.pending.fetch_sub(1, Ordering::SeqCst);
            continue;
        }
        // Nothing to do: park until woken, with a timeout so a missed
        // notification costs latency rather than a hang.
        let guard = shared
            .idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let _unused = shared
            .wake
            .wait_timeout(guard, std::time::Duration::from_millis(1))
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.signal_all();
        for handle in self.workers.drain(..) {
            // A worker that panicked has already been contained; joining it
            // is still correct and must not panic the dropping thread.
            let _ = handle.join();
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

    /// A shared counter, the `'static` shape every pool task uses.
    fn counter() -> Arc<AtomicUsize> {
        Arc::new(AtomicUsize::new(0))
    }

    #[test]
    fn every_task_runs_exactly_once() {
        let pool = ThreadPool::new(4).unwrap();
        let count = counter();
        let tasks: Vec<_> = (0..1000)
            .map(|_| {
                let count = Arc::clone(&count);
                move || {
                    count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
            })
            .collect();
        pool.run_all(tasks).unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn tasks_share_state_through_arcs() {
        // The shape the scheduler uses: tiles are Arc<TileBuf>, so a task
        // moves handles rather than borrowing from the caller's frame.
        let pool = ThreadPool::new(4).unwrap();
        let data: Arc<Vec<usize>> = Arc::new((0..100).collect());
        let total = counter();
        let tasks: Vec<_> = (0..10)
            .map(|chunk| {
                let (data, total) = (Arc::clone(&data), Arc::clone(&total));
                move || {
                    let sum: usize = data[chunk * 10..(chunk + 1) * 10].iter().sum();
                    total.fetch_add(sum, Ordering::Relaxed);
                    Ok(())
                }
            })
            .collect();
        pool.run_all(tasks).unwrap();
        assert_eq!(total.load(Ordering::Relaxed), (0..100).sum::<usize>());
    }

    #[test]
    fn the_lowest_indexed_failure_is_reported() {
        // Determinism (SPEC §Guarantees 2): whichever worker fails first, the
        // reported error is always the lowest-indexed failing task.
        let pool = ThreadPool::new(8).unwrap();
        for attempt in 0..25 {
            let tasks: Vec<_> = (0..64)
                .map(|i| {
                    move || {
                        if i == 5 || i == 40 {
                            return Err(PixelsError::malformed("test", format!("task {i}")));
                        }
                        Ok(())
                    }
                })
                .collect();
            let err = pool.run_all(tasks).unwrap_err();
            assert!(
                err.to_string().contains("task 5"),
                "attempt {attempt}: {err}"
            );
        }
    }

    #[test]
    fn a_panicking_task_becomes_an_error_not_an_abort() {
        let pool = ThreadPool::new(4).unwrap();
        let tasks: Vec<_> = (0..8)
            .map(|i| {
                move || {
                    assert!(i != 3, "kernel defect");
                    Ok(())
                }
            })
            .collect();
        let err = pool.run_all(tasks).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Graph);
        assert!(err.to_string().contains("panicked"), "got: {err}");
        assert!(err.to_string().contains("kernel defect"), "got: {err}");

        // The pool is still usable afterwards: one bad tile does not kill it.
        let count = counter();
        let c = Arc::clone(&count);
        pool.run_all(vec![move || {
            c.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }])
        .unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn a_single_threaded_pool_still_completes() {
        // The caller blocks, so a one-worker pool must not deadlock.
        let pool = ThreadPool::new(1).unwrap();
        let count = counter();
        let tasks: Vec<_> = (0..100)
            .map(|_| {
                let count = Arc::clone(&count);
                move || {
                    count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
            })
            .collect();
        pool.run_all(tasks).unwrap();
        assert_eq!(count.load(Ordering::Relaxed), 100);
        assert_eq!(pool.threads(), 1);
    }

    #[test]
    fn zero_threads_is_clamped_to_one() {
        assert_eq!(ThreadPool::new(0).unwrap().threads(), 1);
    }

    #[test]
    fn an_empty_batch_is_a_no_op() {
        let pool = ThreadPool::new(2).unwrap();
        let tasks: Vec<fn() -> Result<()>> = Vec::new();
        pool.run_all(tasks).unwrap();
    }

    #[test]
    fn repeated_batches_reuse_the_same_workers() {
        // Worker threads are long-lived; a pipeline runs many waves.
        let pool = ThreadPool::new(4).unwrap();
        let count = counter();
        for _ in 0..50 {
            let tasks: Vec<_> = (0..20)
                .map(|_| {
                    let count = Arc::clone(&count);
                    move || {
                        count.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    }
                })
                .collect();
            pool.run_all(tasks).unwrap();
        }
        assert_eq!(count.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn outstanding_spawned_work_completes_before_drop() {
        let done = counter();
        {
            let pool = ThreadPool::new(4).unwrap();
            for _ in 0..200 {
                let done = Arc::clone(&done);
                pool.spawn(move || {
                    done.fetch_add(1, Ordering::Relaxed);
                });
            }
            // Dropping drains outstanding work, then joins.
        }
        assert_eq!(done.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn default_threads_is_at_least_one() {
        assert!(ThreadPool::default_threads() >= 1);
        assert!(ThreadPool::with_default_threads().unwrap().threads() >= 1);
    }

    #[test]
    fn work_is_actually_distributed_across_workers() {
        // Not merely "it completes": prove more than one thread ran tasks.
        // The M2 scaling benchmark is meaningless if this does not hold.
        let pool = ThreadPool::new(4).unwrap();
        let seen: Arc<Mutex<std::collections::HashSet<std::thread::ThreadId>>> =
            Arc::new(Mutex::new(std::collections::HashSet::new()));
        let tasks: Vec<_> = (0..2000)
            .map(|_| {
                let seen = Arc::clone(&seen);
                move || {
                    // Enough work that tasks overlap rather than draining
                    // faster than they are queued.
                    std::hint::black_box((0..500_u64).sum::<u64>());
                    seen.lock().unwrap().insert(std::thread::current().id());
                    Ok(())
                }
            })
            .collect();
        pool.run_all(tasks).unwrap();
        let count = seen.lock().unwrap().len();
        assert!(
            count > 1,
            "all work ran on one thread; stealing is not happening"
        );
    }

    #[test]
    fn nested_arcs_keep_results_alive_across_batches() {
        // Results produced in one wave feed the next, as tiles do.
        let pool = ThreadPool::new(4).unwrap();
        let stage1: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(vec![0; 16]));
        let tasks: Vec<_> = (0..16_u64)
            .map(|i| {
                let out = Arc::clone(&stage1);
                move || {
                    out.lock().unwrap()[i as usize] = i * 2;
                    Ok(())
                }
            })
            .collect();
        pool.run_all(tasks).unwrap();

        let total = Arc::new(AtomicUsize::new(0));
        let tasks: Vec<_> = (0..16_usize)
            .map(|i| {
                let (input, total) = (Arc::clone(&stage1), Arc::clone(&total));
                move || {
                    let v = input.lock().unwrap()[i];
                    total.fetch_add(v as usize, Ordering::Relaxed);
                    Ok(())
                }
            })
            .collect();
        pool.run_all(tasks).unwrap();
        assert_eq!(
            total.load(Ordering::Relaxed),
            (0..16).map(|i| i * 2).sum::<usize>()
        );
    }
}
