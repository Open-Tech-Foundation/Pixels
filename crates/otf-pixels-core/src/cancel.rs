//! [`Cancel`]: stopping a run that has started (ADR-0021).
//!
//! A run cannot be stopped from outside — it blocks its caller or works on
//! the pool's threads — so it stops itself: the scheduler, the evaluators,
//! the decoded source and every codec's long loops call [`Cancel::check`],
//! and a cancelled run fails there with [`ErrorCode::Cancelled`].
//!
//! # Inside a codec
//!
//! A codec's long loops are deep inside its own modules. Rather than pass the
//! flag through every function, the codec runs its heavy entry points —
//! the first `read_row` that decodes a whole image, an encoder's `finish` —
//! inside [`Cancel::scope`], and its loops call [`Cancel::checkpoint`], which
//! checks the flag scoped to the current thread. No codec here spawns
//! threads, so the scope covers all of its work.
//!
//! [`ErrorCode::Cancelled`]: crate::ErrorCode::Cancelled

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::{PixelsError, Result};

thread_local! {
    /// The flag [`Cancel::scope`] made current on this thread.
    static CURRENT: RefCell<Cancel> = const { RefCell::new(Cancel::never()) };
}

#[derive(Debug)]
struct Inner {
    cancelled: AtomicBool,
    deadline: Option<Instant>,
}

/// A flag that stops every run carrying it: cancelled by [`Cancel::cancel`]
/// from any thread, or by a deadline.
///
/// Cheap to clone; every clone is the same flag. The default never cancels,
/// and costs nothing to check.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Option<Arc<Inner>>);

impl Cancel {
    /// A flag that stops its runs when [`Cancel::cancel`] is called.
    #[must_use]
    pub fn new() -> Self {
        Self(Some(Arc::new(Inner {
            cancelled: AtomicBool::new(false),
            deadline: None,
        })))
    }

    /// A flag that stops its runs once `timeout` has passed from now, or
    /// sooner if [`Cancel::cancel`] is called (sharp's `timeout`).
    #[must_use]
    pub fn after(timeout: Duration) -> Self {
        Self(Some(Arc::new(Inner {
            cancelled: AtomicBool::new(false),
            deadline: Instant::now().checked_add(timeout),
        })))
    }

    /// A flag that never cancels: the default.
    #[must_use]
    pub const fn never() -> Self {
        Self(None)
    }

    /// Whether this is [`Cancel::never`]: a flag nothing can cancel.
    #[must_use]
    pub const fn is_never(&self) -> bool {
        self.0.is_none()
    }

    /// Stop every run carrying this flag, at its next check.
    pub fn cancel(&self) {
        if let Some(inner) = &self.0 {
            inner.cancelled.store(true, Ordering::Relaxed);
        }
    }

    /// Whether [`Cancel::cancel`] was called or the deadline has passed.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.state().is_some()
    }

    /// `Err(Cancelled)` once cancelled: the check every long loop makes.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Cancelled`] once cancelled or past the
    /// deadline.
    #[inline]
    pub fn check(&self) -> Result<()> {
        match self.state() {
            None => Ok(()),
            Some(deadline) => Err(PixelsError::cancelled(deadline)),
        }
    }

    /// Run `work` with this flag as the current thread's, for the
    /// [`Cancel::checkpoint`]s inside it; the previous one is restored after,
    /// even if `work` panics.
    pub fn scope<R>(&self, work: impl FnOnce() -> R) -> R {
        struct Restore(Option<Cancel>);
        impl Drop for Restore {
            fn drop(&mut self) {
                if let Some(previous) = self.0.take() {
                    CURRENT.with(|current| *current.borrow_mut() = previous);
                }
            }
        }
        let previous = CURRENT.with(|current| current.replace(self.clone()));
        let _restore = Restore(Some(previous));
        work()
    }

    /// Check the flag the current thread is running under, from
    /// [`Cancel::scope`]; outside any scope, never cancelled.
    ///
    /// # Errors
    ///
    /// As [`Cancel::check`].
    #[inline]
    pub fn checkpoint() -> Result<()> {
        CURRENT.with(|current| current.borrow().check())
    }

    /// `None` while live; `Some(true)` past the deadline, `Some(false)`
    /// cancelled.
    fn state(&self) -> Option<bool> {
        let inner = self.0.as_ref()?;
        if inner.cancelled.load(Ordering::Relaxed) {
            return Some(false);
        }
        inner
            .deadline
            .filter(|&deadline| Instant::now() >= deadline)
            .map(|_| true)
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests operate on known-good values, and one panics on purpose"
)]
mod tests {
    use super::*;
    use crate::ErrorCode;

    #[test]
    fn a_flag_stops_when_cancelled_from_another_thread() {
        let cancel = Cancel::new();
        assert!(cancel.check().is_ok());
        let other = cancel.clone();
        std::thread::spawn(move || other.cancel()).join().unwrap();
        let error = cancel.check().unwrap_err();
        assert_eq!(error.code(), ErrorCode::Cancelled);
        assert_eq!(error.to_string(), "cancelled");
    }

    #[test]
    fn a_deadline_stops_once_passed() {
        let soon = Cancel::after(Duration::from_millis(20));
        assert!(!soon.is_cancelled());
        std::thread::sleep(Duration::from_millis(30));
        let error = soon.check().unwrap_err();
        assert_eq!(error.code(), ErrorCode::Cancelled);
        assert!(error.to_string().contains("deadline"));
        assert!(Cancel::after(Duration::ZERO).is_cancelled());
    }

    #[test]
    fn a_scope_makes_the_flag_the_threads_until_it_ends() {
        let cancel = Cancel::new();
        cancel.cancel();
        assert!(Cancel::checkpoint().is_ok(), "no scope, nothing to stop");
        let inside = cancel.scope(Cancel::checkpoint);
        assert_eq!(inside.unwrap_err().code(), ErrorCode::Cancelled);
        assert!(Cancel::checkpoint().is_ok(), "restored after");
        // Nested scopes restore the outer one; other threads are untouched.
        let live = Cancel::new();
        live.scope(|| {
            cancel.scope(|| assert!(Cancel::checkpoint().is_err()));
            assert!(Cancel::checkpoint().is_ok());
            std::thread::spawn(|| assert!(Cancel::checkpoint().is_ok()))
                .join()
                .unwrap();
        });
        // A panic inside still restores.
        let _ = std::panic::catch_unwind(|| cancel.scope(|| panic!("inside")));
        assert!(Cancel::checkpoint().is_ok());
    }

    #[test]
    fn the_default_never_cancels() {
        let never = Cancel::default();
        never.cancel();
        assert!(never.check().is_ok());
        assert!(Cancel::never().check().is_ok());
        assert!(
            Cancel::after(Duration::MAX).check().is_ok(),
            "an overflowing deadline is never"
        );
    }
}
