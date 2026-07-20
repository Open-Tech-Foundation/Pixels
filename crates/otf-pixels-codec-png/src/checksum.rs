//! CRC-32 and Adler-32, the two checksums PNG needs.
//!
//! PNG puts a CRC-32 on every chunk (PNG spec §5.5) and zlib puts an Adler-32
//! on the decompressed stream (RFC 1950 §2.2). Both are computed
//! incrementally, so a streaming decoder verifies as bytes arrive rather than
//! buffering to check afterwards.

/// The CRC-32 lookup table, built at compile time.
///
/// The polynomial is the reflected `0xEDB8_8320`, as PNG and zlib both use.
#[allow(
    clippy::indexing_slicing,
    reason = "a const block cannot use get_mut; n < 256 is a loop invariant"
)]
const CRC_TABLE: [u32; 256] = {
    let mut table = [0_u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// An incremental CRC-32, as specified for PNG chunks.
#[derive(Debug, Clone, Copy)]
pub struct Crc32 {
    state: u32,
}

impl Crc32 {
    /// A fresh checksum.
    #[must_use]
    pub const fn new() -> Self {
        Self { state: 0xFFFF_FFFF }
    }

    /// Fold `bytes` into the checksum.
    pub fn update(&mut self, bytes: &[u8]) {
        let mut state = self.state;
        for &byte in bytes {
            let index = ((state ^ u32::from(byte)) & 0xFF) as usize;
            // The index is masked to 8 bits, so the table lookup is total; the
            // fallback keeps the code panic-free without an unreachable claim.
            let entry = CRC_TABLE.get(index).copied().unwrap_or(0);
            state = entry ^ (state >> 8);
        }
        self.state = state;
    }

    /// The checksum of everything folded in so far.
    #[must_use]
    pub const fn finish(self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }

    /// The checksum of `bytes` in one call.
    #[must_use]
    pub fn of(bytes: &[u8]) -> u32 {
        let mut crc = Self::new();
        crc.update(bytes);
        crc.finish()
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// An incremental Adler-32, as specified for zlib streams.
#[derive(Debug, Clone, Copy)]
pub struct Adler32 {
    a: u32,
    b: u32,
}

impl Adler32 {
    /// The largest run of bytes that cannot overflow the accumulators.
    const CHUNK: usize = 5552;
    /// The modulus: the largest prime below 2^16.
    const BASE: u32 = 65521;

    /// A fresh checksum.
    #[must_use]
    pub const fn new() -> Self {
        Self { a: 1, b: 0 }
    }

    /// Fold `bytes` into the checksum.
    pub fn update(&mut self, bytes: &[u8]) {
        // Deferring the modulo to chunk boundaries is the standard trick: 5552
        // bytes of 0xFF cannot overflow a u32, so the reduction is exact.
        for chunk in bytes.chunks(Self::CHUNK) {
            let (mut a, mut b) = (self.a, self.b);
            for &byte in chunk {
                a += u32::from(byte);
                b += a;
            }
            self.a = a % Self::BASE;
            self.b = b % Self::BASE;
        }
    }

    /// The checksum of everything folded in so far.
    #[must_use]
    pub const fn finish(self) -> u32 {
        (self.b << 16) | self.a
    }

    /// The checksum of `bytes` in one call.
    #[must_use]
    pub fn of(bytes: &[u8]) -> u32 {
        let mut adler = Self::new();
        adler.update(bytes);
        adler.finish()
    }
}

impl Default for Adler32 {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn crc32_matches_known_vectors() {
        // Vectors from the zlib and PNG test suites.
        assert_eq!(Crc32::of(b""), 0x0000_0000);
        assert_eq!(Crc32::of(b"a"), 0xE8B7_BE43);
        assert_eq!(Crc32::of(b"abc"), 0x3524_41C2);
        assert_eq!(Crc32::of(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            Crc32::of(b"IEND"),
            0xAE42_6082,
            "the IEND chunk CRC in every PNG"
        );
    }

    #[test]
    fn crc32_is_incremental() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let once = Crc32::of(data);
        for split in 0..data.len() {
            let mut crc = Crc32::new();
            crc.update(&data[..split]);
            crc.update(&data[split..]);
            assert_eq!(crc.finish(), once, "split at {split}");
        }
    }

    #[test]
    fn adler32_matches_known_vectors() {
        // Vectors from RFC 1950 and zlib's test suite.
        assert_eq!(Adler32::of(b""), 0x0000_0001);
        assert_eq!(Adler32::of(b"a"), 0x0062_0062);
        assert_eq!(Adler32::of(b"abc"), 0x024D_0127);
        assert_eq!(Adler32::of(b"123456789"), 0x091E_01DE);
        assert_eq!(Adler32::of(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn adler32_is_incremental() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 251) as u8).collect();
        let once = Adler32::of(&data);
        for split in [0, 1, 7, 499, 999, 1000] {
            let mut adler = Adler32::new();
            adler.update(&data[..split]);
            adler.update(&data[split..]);
            assert_eq!(adler.finish(), once, "split at {split}");
        }
    }

    #[test]
    fn adler32_survives_runs_longer_than_the_chunk_window() {
        // The deferred-modulo trick is only valid within 5552 bytes; this
        // checks the chunking that keeps it valid for longer inputs.
        let data = vec![0xFF_u8; 5552 * 3 + 17];
        let once = Adler32::of(&data);
        let mut adler = Adler32::new();
        for piece in data.chunks(1000) {
            adler.update(piece);
        }
        assert_eq!(adler.finish(), once);
        // And it stays within the modulus.
        assert!(once & 0xFFFF < Adler32::BASE);
        assert!(once >> 16 < Adler32::BASE);
    }

    #[test]
    fn empty_updates_change_nothing() {
        let mut crc = Crc32::new();
        crc.update(b"abc");
        crc.update(b"");
        assert_eq!(crc.finish(), Crc32::of(b"abc"));

        let mut adler = Adler32::new();
        adler.update(b"abc");
        adler.update(b"");
        assert_eq!(adler.finish(), Adler32::of(b"abc"));
    }

    #[test]
    fn defaults_are_fresh_checksums() {
        assert_eq!(Crc32::default().finish(), Crc32::new().finish());
        assert_eq!(Adler32::default().finish(), Adler32::new().finish());
    }
}
