//! The example tables from ITU-T T.81 Annex K, which encoders use as defaults.
//!
//! The standard calls these "example" tables and requires no encoder to use
//! them. In practice essentially every baseline encoder does, because they
//! were derived from perceptual experiments and because a decoder that has
//! seen them a billion times has nothing to gain from a bespoke set.
//!
//! Deriving optimal Huffman tables from the actual coefficient statistics
//! would save a few percent, at the cost of buffering the whole image to
//! count symbols before the first byte can be written. That trade runs
//! directly against ADR-0005's streaming contract, so it is not made here.

/// Luminance quantization steps, in natural (row-major) order — Table K.1.
///
/// The values rise towards the bottom right because that is where the high
/// frequencies live, and the eye does not miss them.
pub const LUMA_QUANT: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, //
    12, 12, 14, 19, 26, 58, 60, 55, //
    14, 13, 16, 24, 40, 57, 69, 56, //
    14, 17, 22, 29, 51, 87, 80, 62, //
    18, 22, 37, 56, 68, 109, 103, 77, //
    24, 35, 55, 64, 81, 104, 113, 92, //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Chrominance quantization steps, in natural order — Table K.2.
///
/// Flat at 99 across most of the block: colour detail is discarded far more
/// aggressively than brightness detail, which is the same perceptual fact
/// that makes chroma subsampling acceptable.
pub const CHROMA_QUANT: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Luminance DC code lengths and symbols — Table K.3.
pub const LUMA_DC_COUNTS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
/// Symbols for [`LUMA_DC_COUNTS`]: the twelve magnitude categories.
pub const LUMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Chrominance DC code lengths and symbols — Table K.4.
pub const CHROMA_DC_COUNTS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// Symbols for [`CHROMA_DC_COUNTS`].
pub const CHROMA_DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Luminance AC code lengths — Table K.5.
pub const LUMA_AC_COUNTS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];

/// Luminance AC symbols, each a run of zeros in the high nibble and a
/// magnitude category in the low one — Table K.5.
pub const LUMA_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, //
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, //
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, //
    0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, //
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, //
    0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, //
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, //
    0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, //
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, //
    0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, //
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, //
    0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, //
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, //
    0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, //
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, //
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, //
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, //
    0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, //
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, //
    0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, //
    0xF9, 0xFA,
];

/// Chrominance AC code lengths — Table K.6.
pub const CHROMA_AC_COUNTS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];

/// Chrominance AC symbols — Table K.6.
pub const CHROMA_AC_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, //
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, //
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, //
    0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, //
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, //
    0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, //
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, //
    0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, //
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, //
    0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, //
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, //
    0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, //
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, //
    0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, //
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, //
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, //
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, //
    0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, //
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, //
    0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, //
    0xF9, 0xFA,
];

/// Scale a quantization table for a quality setting, 1..=100.
///
/// The mapping is the IJG one, and is worth keeping bit-compatible: quality
/// numbers are compared across tools constantly, and "quality 80" meaning
/// something different here than in every other encoder would be a trap.
/// Quality 50 leaves the tables untouched; below it they scale up steeply,
/// above it they flatten towards 1.
#[must_use]
pub fn scale_quant(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let quality = u32::from(quality.clamp(1, 100));
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - quality * 2
    };
    std::array::from_fn(|index| {
        let value = u32::from(*base.get(index).unwrap_or(&1));
        // Steps are clamped to 1..=255 so the table fits an 8-bit DQT, and
        // so a step of zero — which would divide by zero — cannot arise.
        ((value * scale + 50) / 100).clamp(1, 255) as u16
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests operate on known-good values and assert shapes directly"
)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanTable;

    #[test]
    fn the_standard_tables_are_valid_prefix_codes() {
        // Building a *decoding* table from each set proves the counts and
        // symbol lists agree and that no set is over-subscribed. A
        // transcription slip in 162 hex values would otherwise sit here
        // silently until a decoder somewhere rejected our output.
        for (counts, values) in [
            (&LUMA_DC_COUNTS, &LUMA_DC_VALUES[..]),
            (&CHROMA_DC_COUNTS, &CHROMA_DC_VALUES[..]),
            (&LUMA_AC_COUNTS, &LUMA_AC_VALUES[..]),
            (&CHROMA_AC_COUNTS, &CHROMA_AC_VALUES[..]),
        ] {
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            assert_eq!(total, values.len(), "counts and symbols disagree");
            HuffmanTable::new(counts, values.to_vec()).unwrap();
        }
    }

    #[test]
    fn ac_symbol_sets_are_complete_and_unique() {
        // Every (run, size) pair an encoder can emit must have a code. The
        // legal set is size 1..=10 for runs 0..=15, plus ZRL and EOB.
        for values in [&LUMA_AC_VALUES[..], &CHROMA_AC_VALUES[..]] {
            let mut seen = std::collections::HashSet::new();
            for &symbol in values {
                assert!(seen.insert(symbol), "duplicate AC symbol {symbol:#04x}");
            }
            assert!(seen.contains(&0x00), "no end-of-block code");
            assert!(seen.contains(&0xF0), "no zero-run-length code");
            for run in 0..16_u8 {
                for size in 1..=10_u8 {
                    let symbol = (run << 4) | size;
                    assert!(seen.contains(&symbol), "no code for {symbol:#04x}");
                }
            }
        }
    }

    #[test]
    fn quality_fifty_leaves_the_base_tables_alone() {
        // The definition of the IJG scale: 50 is the identity point.
        assert_eq!(scale_quant(&LUMA_QUANT, 50), LUMA_QUANT);
        assert_eq!(scale_quant(&CHROMA_QUANT, 50), CHROMA_QUANT);
    }

    #[test]
    fn quality_moves_the_steps_the_right_way() {
        let low = scale_quant(&LUMA_QUANT, 10);
        let high = scale_quant(&LUMA_QUANT, 95);
        // Coarser steps at low quality, finer at high, everywhere.
        for index in 0..64 {
            assert!(low[index] >= LUMA_QUANT[index].min(255), "index {index}");
            assert!(high[index] <= LUMA_QUANT[index], "index {index}");
        }
        // Quality 100 is the finest the mapping allows: every step is 1.
        assert_eq!(scale_quant(&LUMA_QUANT, 100), [1_u16; 64]);
    }

    #[test]
    fn every_step_stays_in_the_eight_bit_dqt_range() {
        // A step of zero would divide by zero; a step above 255 would not fit
        // the 8-bit table we write.
        for quality in 1..=100_u8 {
            for base in [&LUMA_QUANT, &CHROMA_QUANT] {
                for step in scale_quant(base, quality) {
                    assert!((1..=255).contains(&step), "quality {quality}: {step}");
                }
            }
        }
    }
}
