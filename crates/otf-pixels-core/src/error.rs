//! The engine's error type.
//!
//! Errors are values end to end (ARCHITECTURE §Failure model). Malformed input
//! is a [`PixelsError::Malformed`], never a panic — codec parsers must return
//! this variant for every byte sequence a hostile source can produce.
//!
//! [`ErrorCode`] is the stable, machine-readable projection of an error and is
//! part of the public API under semver (SPEC §Guarantees 4). The
//! [`PixelsError`] variants carry human-readable detail and are
//! `#[non_exhaustive]`; match on [`PixelsError::code`] when you need
//! exhaustive, forward-compatible handling.

use core::fmt;

/// The engine's result alias.
pub type Result<T, E = PixelsError> = core::result::Result<T, E>;

/// A stable, machine-readable error classification.
///
/// Codes are part of the public API and follow semver: a code is never
/// removed or repurposed, and host bindings may expose them directly. New
/// codes may be added in minor releases, hence `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ErrorCode {
    /// Reading from a source or writing to a sink failed.
    Io,
    /// The input bytes are not valid for the format they claim to be.
    Malformed,
    /// The format or feature is recognized but not implemented.
    Unsupported,
    /// A configured safety limit was exceeded (SPEC §Safety).
    LimitExceeded,
    /// A caller-supplied argument is invalid (e.g. a zero-width crop).
    InvalidArgument,
    /// The op graph is not evaluable as constructed.
    Graph,
}

impl ErrorCode {
    /// A short, stable, lowercase identifier for this code.
    ///
    /// Suitable for logs and for host bindings that surface string codes.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Io => "io",
            Self::Malformed => "malformed",
            Self::Unsupported => "unsupported",
            Self::LimitExceeded => "limit_exceeded",
            Self::InvalidArgument => "invalid_argument",
            Self::Graph => "graph",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The safety limit that a [`PixelsError::LimitExceeded`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Limit {
    /// Total pixel count (width × height) exceeded [`Limits::max_pixels`].
    ///
    /// [`Limits::max_pixels`]: crate::Limits::max_pixels
    MaxPixels,
    /// A single dimension exceeded the representable maximum.
    Dimension,
}

impl Limit {
    /// A short, stable identifier for this limit.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxPixels => "max_pixels",
            Self::Dimension => "dimension",
        }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The error type returned by every fallible engine operation.
///
/// Construct these with the associated helpers ([`PixelsError::malformed`],
/// [`PixelsError::unsupported`], …) rather than the variants directly, so that
/// added fields stay non-breaking.
#[derive(Debug)]
#[non_exhaustive]
pub enum PixelsError {
    /// A source read or sink write failed.
    Io {
        /// What the engine was doing when the I/O failed.
        context: &'static str,
        /// The underlying operating system error.
        source: std::io::Error,
    },
    /// Input bytes are invalid for their format.
    ///
    /// Every codec parser returns this instead of panicking, for any input.
    Malformed {
        /// The format being parsed, e.g. `"raw"`, `"png"`.
        format: &'static str,
        /// What was wrong, in human-readable terms.
        detail: String,
    },
    /// A recognized format or feature that is not implemented.
    Unsupported {
        /// What is unsupported, in human-readable terms.
        detail: String,
    },
    /// A safety limit was exceeded before any pixel allocation.
    LimitExceeded {
        /// Which limit was hit.
        limit: Limit,
        /// The value the input asked for.
        requested: u64,
        /// The configured maximum.
        allowed: u64,
    },
    /// A caller-supplied argument is invalid.
    InvalidArgument {
        /// The parameter at fault, e.g. `"width"`.
        parameter: &'static str,
        /// Why it is invalid.
        detail: String,
    },
    /// The op graph cannot be evaluated as constructed.
    Graph {
        /// Why the graph is invalid.
        detail: String,
    },
}

impl PixelsError {
    /// The stable classification of this error.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::Io { .. } => ErrorCode::Io,
            Self::Malformed { .. } => ErrorCode::Malformed,
            Self::Unsupported { .. } => ErrorCode::Unsupported,
            Self::LimitExceeded { .. } => ErrorCode::LimitExceeded,
            Self::InvalidArgument { .. } => ErrorCode::InvalidArgument,
            Self::Graph { .. } => ErrorCode::Graph,
        }
    }

    /// Wrap an I/O error with the engine context it happened in.
    #[must_use]
    pub fn io(context: &'static str, source: std::io::Error) -> Self {
        Self::Io { context, source }
    }

    /// Report input bytes that are invalid for `format`.
    #[must_use]
    pub fn malformed(format: &'static str, detail: impl Into<String>) -> Self {
        Self::Malformed {
            format,
            detail: detail.into(),
        }
    }

    /// Report a recognized but unimplemented format or feature.
    #[must_use]
    pub fn unsupported(detail: impl Into<String>) -> Self {
        Self::Unsupported {
            detail: detail.into(),
        }
    }

    /// Report an invalid caller-supplied argument.
    #[must_use]
    pub fn invalid_argument(parameter: &'static str, detail: impl Into<String>) -> Self {
        Self::InvalidArgument {
            parameter,
            detail: detail.into(),
        }
    }

    /// Report a graph that cannot be evaluated.
    #[must_use]
    pub fn graph(detail: impl Into<String>) -> Self {
        Self::Graph {
            detail: detail.into(),
        }
    }

    /// Report an exceeded safety limit.
    #[must_use]
    pub const fn limit_exceeded(limit: Limit, requested: u64, allowed: u64) -> Self {
        Self::LimitExceeded {
            limit,
            requested,
            allowed,
        }
    }
}

impl fmt::Display for PixelsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "i/o error while {context}: {source}"),
            Self::Malformed { format, detail } => {
                write!(f, "malformed {format} input: {detail}")
            }
            Self::Unsupported { detail } => write!(f, "unsupported: {detail}"),
            Self::LimitExceeded {
                limit,
                requested,
                allowed,
            } => write!(
                f,
                "limit `{limit}` exceeded: requested {requested}, allowed {allowed}"
            ),
            Self::InvalidArgument { parameter, detail } => {
                write!(f, "invalid argument `{parameter}`: {detail}")
            }
            Self::Graph { detail } => write!(f, "invalid graph: {detail}"),
        }
    }
}

impl std::error::Error for PixelsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
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

    #[test]
    fn code_is_stable_per_variant() {
        assert_eq!(
            PixelsError::malformed("raw", "x").code(),
            ErrorCode::Malformed
        );
        assert_eq!(PixelsError::unsupported("x").code(), ErrorCode::Unsupported);
        assert_eq!(PixelsError::graph("x").code(), ErrorCode::Graph);
        assert_eq!(
            PixelsError::invalid_argument("w", "x").code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            PixelsError::limit_exceeded(Limit::MaxPixels, 5, 4).code(),
            ErrorCode::LimitExceeded
        );
        let io = PixelsError::io("reading", std::io::Error::other("boom"));
        assert_eq!(io.code(), ErrorCode::Io);
    }

    #[test]
    fn code_strings_are_stable() {
        assert_eq!(ErrorCode::Io.as_str(), "io");
        assert_eq!(ErrorCode::Malformed.as_str(), "malformed");
        assert_eq!(ErrorCode::Unsupported.as_str(), "unsupported");
        assert_eq!(ErrorCode::LimitExceeded.as_str(), "limit_exceeded");
        assert_eq!(ErrorCode::InvalidArgument.as_str(), "invalid_argument");
        assert_eq!(ErrorCode::Graph.as_str(), "graph");
        assert_eq!(Limit::MaxPixels.as_str(), "max_pixels");
        assert_eq!(Limit::Dimension.as_str(), "dimension");
    }

    #[test]
    fn io_errors_expose_their_source() {
        use std::error::Error as _;
        let err = PixelsError::io("reading header", std::io::Error::other("boom"));
        assert!(err.source().is_some());
        assert!(PixelsError::graph("x").source().is_none());
        assert!(err.to_string().contains("reading header"));
    }

    #[test]
    fn limit_display_names_the_limit() {
        let err = PixelsError::limit_exceeded(Limit::MaxPixels, 300, 268);
        let text = err.to_string();
        assert!(text.contains("max_pixels"), "{text}");
        assert!(text.contains("300"), "{text}");
        assert!(text.contains("268"), "{text}");
    }
}
