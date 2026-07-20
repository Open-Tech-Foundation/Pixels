//! Streaming byte input and output.
//!
//! Per ADR-0005, streaming is the **only** external I/O contract: a [`Source`]
//! is forward-only and a [`Sink`] is write-only. No seek is ever required of a
//! caller, so a socket, a pipe and a file are all equally valid inputs. Where a
//! format cannot decode incrementally, its codec buffers internally — the
//! external contract stays streaming.
//!
//! Both traits have blanket implementations over [`std::io::Read`] and
//! [`std::io::Write`], so `File`, `&[u8]`, `Vec<u8>`, `TcpStream` and
//! `Stdin`/`Stdout` work as sources and sinks with no adapter.

use crate::{PixelsError, Result};

/// A forward-only source of bytes.
///
/// Implementors need not support seeking, and the engine never asks them to.
/// Sources perform no decoding and allocate nothing proportional to image size.
pub trait Source: Send {
    /// Read into `buf`, returning the number of bytes read.
    ///
    /// Returning `Ok(0)` means end of input. A short read is not an error: the
    /// engine calls again until it has what it needs or the source ends.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the underlying reader fails.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    /// Fill `buf` completely.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] on reader failure, or
    /// [`PixelsError::Malformed`] if the source ends before `buf` is full —
    /// a truncated stream is malformed input, not an I/O fault.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let wanted = buf.len();
        let mut filled = 0;
        while filled < wanted {
            let Some(rest) = buf.get_mut(filled..) else { break };
            match self.read(rest)? {
                0 => {
                    return Err(PixelsError::malformed(
                        "stream",
                        format!("stream ended after {filled} of {wanted} expected bytes"),
                    ));
                }
                n => filled += n,
            }
        }
        Ok(())
    }
}

/// A sink for encoded bytes.
pub trait Sink: Send {
    /// Write all of `buf`.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the underlying writer fails.
    fn write_all(&mut self, buf: &[u8]) -> Result<()>;

    /// Flush any buffered bytes to the destination.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Io`] if the underlying writer fails.
    fn flush(&mut self) -> Result<()>;
}

impl<R: std::io::Read + Send> Source for R {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        loop {
            return match std::io::Read::read(self, buf) {
                Ok(n) => Ok(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => Err(PixelsError::io("reading from source", e)),
            };
        }
    }
}

impl<W: std::io::Write + Send> Sink for W {
    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        std::io::Write::write_all(self, buf).map_err(|e| PixelsError::io("writing to sink", e))
    }

    fn flush(&mut self) -> Result<()> {
        std::io::Write::flush(self).map_err(|e| PixelsError::io("flushing sink", e))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests operate on known-good values and assert shapes directly")]
mod tests {
    use super::*;
    use crate::ErrorCode;

    /// A source that yields at most `chunk` bytes per call, to exercise the
    /// short-read path that real sockets and pipes produce.
    struct Trickle {
        data: Vec<u8>,
        pos: usize,
        chunk: usize,
    }

    impl Source for Trickle {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
            let remaining = self.data.len() - self.pos;
            let n = remaining.min(buf.len()).min(self.chunk);
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn read_exact_reassembles_short_reads() {
        let mut src = Trickle { data: (0..10).collect(), pos: 0, chunk: 3 };
        let mut buf = [0_u8; 10];
        src.read_exact(&mut buf).unwrap();
        assert_eq!(buf, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn a_truncated_stream_is_malformed_not_an_io_error() {
        let mut src = Trickle { data: vec![1, 2, 3], pos: 0, chunk: 2 };
        let mut buf = [0_u8; 8];
        let err = src.read_exact(&mut buf).unwrap_err();
        assert_eq!(err.code(), ErrorCode::Malformed);
        assert!(err.to_string().contains("3 of 8"), "{err}");
    }

    #[test]
    fn slices_and_vecs_work_without_adapters() {
        let mut src: &[u8] = b"hello";
        let mut buf = [0_u8; 5];
        src.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");

        let mut sink: Vec<u8> = Vec::new();
        sink.write_all(b"out").unwrap();
        sink.flush().unwrap();
        assert_eq!(sink, b"out");
    }

    #[test]
    fn read_exact_of_nothing_succeeds() {
        let mut src: &[u8] = b"";
        src.read_exact(&mut []).unwrap();
    }

    #[test]
    fn sink_errors_are_reported_as_io() {
        struct Broken;
        impl std::io::Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("disk full"))
            }
        }
        let err = Sink::write_all(&mut Broken, b"x").unwrap_err();
        assert_eq!(err.code(), ErrorCode::Io);
        assert_eq!(Sink::flush(&mut Broken).unwrap_err().code(), ErrorCode::Io);
    }
}
