//! Progressive JPEG encoding: the scans, written from a coefficient buffer.
//!
//! The mirror of [`crate::progressive`]. Every block is transformed and
//! quantized as rows arrive, as for baseline, but kept rather than coded;
//! once the image is complete it is written as a series of scans, each a
//! band of frequencies (*spectral selection*) or one more bit of precision
//! (*successive approximation*), so a viewer can draw a coarse picture early
//! and sharpen it.
//!
//! # What is written
//!
//! libjpeg's default progression (`jpeg_simple_progression`), which every
//! decoder sees daily: DC at half precision, a few low luma frequencies,
//! chroma, the rest of luma, then the last bit of each. Each scan gets
//! Huffman tables built from its own symbol counts (ITU-T T.81 Annex K.2),
//! which is most of why a progressive file is smaller than the baseline
//! one; and runs of empty bands are coded as end-of-band runs, with
//! refinement scans' correction bits buffered across them, as libjpeg's
//! `jcphuff.c` does.
//!
//! # Memory
//!
//! Two bytes per coefficient until the image is complete, as the
//! progressive decoder holds: 36 MB for a 12-megapixel 4:2:0 photograph.

#![allow(
    clippy::indexing_slicing,
    reason = "block indices come from the component's own block grid, \
              coefficient indices from 0..64, table slots from 0..2"
)]

use crate::encoder::{BitWriter, magnitude, write_segment};
use crate::format::{ZIGZAG, marker};
use crate::huffman::HuffmanEncoder;
use otf_pixels_core::{Result, Sink};

/// One component's quantized blocks, natural (row-major) order within each.
#[derive(Debug)]
struct Component {
    /// Sampling factors.
    h: usize,
    v: usize,
    /// The block grid MCUs cover, which is what is stored.
    blocks_w: usize,
    /// The blocks a scan of this component alone visits: those holding
    /// samples of the component, without the MCU padding (T.81 A.2.2).
    scan_w: usize,
    scan_h: usize,
    /// Which DC table its DC scans code with: 0 for luma, 1 for chroma.
    table: usize,
    blocks: Vec<[i16; 64]>,
}

/// Every quantized block of an image, until its scans are written.
#[derive(Debug)]
pub(crate) struct Coefficients {
    components: Vec<Component>,
    mcus_w: usize,
    mcus_h: usize,
}

/// One scan of the script: its components, band and bit positions.
#[derive(Debug, Clone, Copy)]
struct ScanSpec {
    components: &'static [usize],
    ss: usize,
    se: usize,
    ah: u32,
    al: u32,
}

const fn scan(components: &'static [usize], ss: usize, se: usize, ah: u32, al: u32) -> ScanSpec {
    ScanSpec {
        components,
        ss,
        se,
        ah,
        al,
    }
}

/// libjpeg's `jpeg_simple_progression` for YCbCr.
const COLOUR_SCRIPT: [ScanSpec; 10] = [
    scan(&[0, 1, 2], 0, 0, 0, 1),
    scan(&[0], 1, 5, 0, 2),
    scan(&[2], 1, 63, 0, 1),
    scan(&[1], 1, 63, 0, 1),
    scan(&[0], 6, 63, 0, 2),
    scan(&[0], 1, 63, 2, 1),
    scan(&[0, 1, 2], 0, 0, 1, 0),
    scan(&[2], 1, 63, 1, 0),
    scan(&[1], 1, 63, 1, 0),
    scan(&[0], 1, 63, 1, 0),
];

/// The same for a single (grey) component.
const GREY_SCRIPT: [ScanSpec; 6] = [
    scan(&[0], 0, 0, 0, 1),
    scan(&[0], 1, 5, 0, 2),
    scan(&[0], 6, 63, 0, 2),
    scan(&[0], 1, 63, 2, 1),
    scan(&[0], 0, 0, 1, 0),
    scan(&[0], 1, 63, 1, 0),
];

impl Coefficients {
    /// Room for every block of a `width` x `height` image with luma
    /// sampling `factors` (chroma 1x1), one component if `grayscale`.
    pub(crate) fn new(width: u32, height: u32, factors: (u32, u32), grayscale: bool) -> Self {
        let (h, v) = (factors.0 as usize, factors.1 as usize);
        let (width, height) = (width as usize, height as usize);
        let mcus_w = width.div_ceil(h * 8);
        let mcus_h = height.div_ceil(v * 8);
        let component = |ch: usize, cv: usize, table: usize| {
            // Samples of a component: the image scaled by its share of the
            // largest sampling factor, rounded up (T.81 A.1.1).
            let (cw, chh) = ((width * ch).div_ceil(h), (height * cv).div_ceil(v));
            let (blocks_w, blocks_h) = (mcus_w * ch, mcus_h * cv);
            Component {
                h: ch,
                v: cv,
                blocks_w,
                scan_w: cw.div_ceil(8),
                scan_h: chh.div_ceil(8),
                table,
                blocks: vec![[0; 64]; blocks_w * blocks_h],
            }
        };
        let mut components = vec![component(h, v, 0)];
        if !grayscale {
            components.push(component(1, 1, 1));
            components.push(component(1, 1, 1));
        }
        Self {
            components,
            mcus_w,
            mcus_h,
        }
    }

    /// Keep `block` (natural order) as component `c`'s block at `(x, y)`.
    pub(crate) fn store(&mut self, c: usize, x: usize, y: usize, block: &[i32; 64]) {
        let Some(component) = self.components.get_mut(c) else {
            return;
        };
        if let Some(slot) = component.blocks.get_mut(y * component.blocks_w + x) {
            for (out, &value) in slot.iter_mut().zip(block) {
                // An 8-bit image's quantized coefficients fit 12 bits.
                *out = value.clamp(-32_768, 32_767) as i16;
            }
        }
    }

    /// Write every scan of the script, each with its own Huffman tables.
    pub(crate) fn write_scans(&self, sink: &mut dyn Sink) -> Result<()> {
        let script: &[ScanSpec] = if self.components.len() == 1 {
            &GREY_SCRIPT
        } else {
            &COLOUR_SCRIPT
        };
        for spec in script {
            self.write_scan(spec, sink)?;
        }
        Ok(())
    }

    fn write_scan(&self, spec: &ScanSpec, sink: &mut dyn Sink) -> Result<()> {
        let dc = spec.ss == 0;
        let refine_dc = dc && spec.ah > 0;
        // DC scans code each component with its own table; an AC scan has
        // one component, coded with table 0, redefined for every scan.
        let slot_of = |c: usize| if dc { self.components[c].table } else { 0 };

        let mut tables: [Option<HuffmanEncoder>; 2] = [None, None];
        if !refine_dc {
            let mut counter = Counter::default();
            self.code(spec, &slot_of, &mut counter);
            let mut payload = Vec::new();
            for (slot, freq) in counter.freq.iter().enumerate() {
                if !counter.used[slot] {
                    continue;
                }
                let (counts, values) = optimal_table(freq);
                payload.push((u8::from(!dc) << 4) | slot as u8);
                payload.extend_from_slice(&counts);
                payload.extend_from_slice(&values);
                tables[slot] = Some(HuffmanEncoder::new(&counts, &values)?);
            }
            write_segment(marker::DHT, &payload, sink)?;
        }

        let mut payload = vec![spec.components.len() as u8];
        for &c in spec.components {
            let slot = slot_of(c) as u8;
            payload.push(c as u8 + 1);
            payload.push(if dc { slot << 4 } else { slot });
        }
        payload.extend_from_slice(&[
            spec.ss as u8,
            spec.se as u8,
            ((spec.ah as u8) << 4) | spec.al as u8,
        ]);
        write_segment(marker::SOS, &payload, sink)?;

        let mut writer = Writer {
            bits: BitWriter::default(),
            tables,
        };
        self.code(spec, &slot_of, &mut writer);
        writer.bits.flush();
        sink.write_all(&writer.bits.bytes)
    }

    /// Code one scan's blocks into `out`, in the scan's block order.
    fn code(&self, spec: &ScanSpec, slot_of: &dyn Fn(usize) -> usize, out: &mut dyn Emit) {
        let mut state = AcState::default();
        let mut predictors = [0_i32; 3];
        let mut block = |c: usize, x: usize, y: usize, out: &mut dyn Emit| {
            let component = &self.components[c];
            let Some(coefficients) = component.blocks.get(y * component.blocks_w + x) else {
                return;
            };
            let slot = slot_of(c);
            match (spec.ss == 0, spec.ah > 0) {
                (true, false) => {
                    let value = i32::from(coefficients[0]) >> spec.al;
                    let (size, bits) = magnitude(value - predictors[c]);
                    predictors[c] = value;
                    out.symbol(slot, size as u8);
                    out.bits(bits, size);
                }
                (true, true) => out.bits((i32::from(coefficients[0]) >> spec.al) as u32 & 1, 1),
                (false, false) => state.first(coefficients, spec, slot, out),
                (false, true) => state.refine(coefficients, spec, slot, out),
            }
        };

        if let [c] = spec.components {
            let component = &self.components[*c];
            for y in 0..component.scan_h {
                for x in 0..component.scan_w {
                    block(*c, x, y, out);
                }
            }
        } else {
            for my in 0..self.mcus_h {
                for mx in 0..self.mcus_w {
                    for &c in spec.components {
                        let component = &self.components[c];
                        for by in 0..component.v {
                            for bx in 0..component.h {
                                block(c, mx * component.h + bx, my * component.v + by, out);
                            }
                        }
                    }
                }
            }
        }
        state.flush_eobrun(slot_of(spec.components[0]), out);
    }
}

/// Where a scan's symbols and bits go: counted, to build its tables, or
/// written with them.
trait Emit {
    fn symbol(&mut self, slot: usize, symbol: u8);
    fn bits(&mut self, value: u32, count: u32);
}

/// Symbol frequencies per table slot.
#[derive(Debug)]
struct Counter {
    freq: [[u32; 256]; 2],
    used: [bool; 2],
}

impl Default for Counter {
    fn default() -> Self {
        Self {
            freq: [[0; 256]; 2],
            used: [false; 2],
        }
    }
}

impl Emit for Counter {
    fn symbol(&mut self, slot: usize, symbol: u8) {
        self.freq[slot][usize::from(symbol)] += 1;
        self.used[slot] = true;
    }
    fn bits(&mut self, _: u32, _: u32) {}
}

/// Writes symbols with the tables the counting pass built.
struct Writer {
    bits: BitWriter,
    tables: [Option<HuffmanEncoder>; 2],
}

impl Emit for Writer {
    fn symbol(&mut self, slot: usize, symbol: u8) {
        if let Some((code, length)) = self.tables[slot].as_ref().and_then(|t| t.code(symbol)) {
            self.bits.write(code, length);
        }
    }
    fn bits(&mut self, value: u32, count: u32) {
        self.bits.write(value, count);
    }
}

/// An AC scan's state across blocks: the pending end-of-band run and, for
/// a refinement scan, the correction bits owed to the blocks it covers.
#[derive(Debug, Default)]
struct AcState {
    eobrun: u32,
    corrections: Vec<u8>,
}

/// The longest end-of-band run the format codes (`EOB14`).
const MAX_EOBRUN: u32 = 0x7FFF;

impl AcState {
    /// Emit the pending end-of-band run, then the correction bits it owes.
    fn flush_eobrun(&mut self, slot: usize, out: &mut dyn Emit) {
        if self.eobrun == 0 {
            return;
        }
        let bits = 31 - self.eobrun.leading_zeros();
        out.symbol(slot, (bits << 4) as u8);
        if bits > 0 {
            out.bits(self.eobrun, bits);
        }
        self.eobrun = 0;
        for &bit in &self.corrections {
            out.bits(u32::from(bit), 1);
        }
        self.corrections.clear();
    }

    /// An AC first scan's band of one block (T.81 G.1.2.2).
    fn first(&mut self, block: &[i16; 64], spec: &ScanSpec, slot: usize, out: &mut dyn Emit) {
        let mut run = 0_u32;
        for k in spec.ss..=spec.se {
            let value = i32::from(block[ZIGZAG[k]]);
            // The point transform divides, rounding towards zero.
            let magnitude_value = value.unsigned_abs() >> spec.al;
            if magnitude_value == 0 {
                run += 1;
                continue;
            }
            self.flush_eobrun(slot, out);
            while run > 15 {
                out.symbol(slot, 0xF0);
                run -= 16;
            }
            let size = 32 - magnitude_value.leading_zeros();
            let bits = if value < 0 {
                !magnitude_value
            } else {
                magnitude_value
            };
            out.symbol(slot, ((run << 4) | size) as u8);
            out.bits(bits, size);
            run = 0;
        }
        if run > 0 {
            self.eobrun += 1;
            if self.eobrun == MAX_EOBRUN {
                self.flush_eobrun(slot, out);
            }
        }
    }

    /// An AC refinement scan's band of one block (T.81 G.1.2.3), after
    /// libjpeg's `encode_mcu_AC_refine`.
    fn refine(&mut self, block: &[i16; 64], spec: &ScanSpec, slot: usize, out: &mut dyn Emit) {
        let mut absolute = [0_u32; 64];
        // The last coefficient that becomes nonzero in this scan.
        let mut eob = 0;
        for k in spec.ss..=spec.se {
            absolute[k] = i32::from(block[ZIGZAG[k]]).unsigned_abs() >> spec.al;
            if absolute[k] == 1 {
                eob = k;
            }
        }

        let mut run = 0_u32;
        // Correction bits for this block, sent with its next symbol.
        let mut pending: Vec<u8> = Vec::new();
        for k in spec.ss..=spec.se {
            let value = absolute[k];
            if value == 0 {
                run += 1;
                continue;
            }
            // A ZRL is needed only before a newly nonzero coefficient;
            // after the last one the zeros fold into the end of band.
            while run > 15 && k <= eob {
                self.flush_eobrun(slot, out);
                out.symbol(slot, 0xF0);
                run -= 16;
                for &bit in &pending {
                    out.bits(u32::from(bit), 1);
                }
                pending.clear();
            }
            if value > 1 {
                // Known already: only its next bit.
                pending.push((value & 1) as u8);
                continue;
            }
            self.flush_eobrun(slot, out);
            out.symbol(slot, ((run << 4) | 1) as u8);
            out.bits(u32::from(block[ZIGZAG[k]] >= 0), 1);
            for &bit in &pending {
                out.bits(u32::from(bit), 1);
            }
            pending.clear();
            run = 0;
        }
        if run > 0 || !pending.is_empty() {
            self.eobrun += 1;
            self.corrections.extend_from_slice(&pending);
            if self.eobrun == MAX_EOBRUN {
                self.flush_eobrun(slot, out);
            }
        }
    }
}

/// The optimal Huffman table for symbol frequencies `freq`, limited to 16-bit
/// codes, as `DHT` counts and symbols (T.81 Annex K.2, as libjpeg's
/// `jpeg_gen_optimal_table`).
fn optimal_table(freq: &[u32; 256]) -> ([u8; 16], Vec<u8>) {
    // One extra symbol, 256, reserves the all-ones code, which T.81 forbids.
    let mut weight = [0_u64; 257];
    for (w, &f) in weight.iter_mut().zip(freq) {
        *w = u64::from(f);
    }
    if weight.iter().take(256).all(|&w| w == 0) {
        // A table must define something; a scan never emits from an empty
        // one, so any single symbol will do.
        weight[0] = 1;
    }
    weight[256] = 1;
    let mut size = [0_usize; 257];
    let mut next = [usize::MAX; 257];
    loop {
        // The two least frequent, ties to the higher symbol, as libjpeg.
        let smallest = |skip: usize| {
            let mut best: Option<usize> = None;
            for i in 0..257 {
                if weight[i] > 0 && i != skip && best.is_none_or(|b| weight[i] <= weight[b]) {
                    best = Some(i);
                }
            }
            best
        };
        let Some(c1) = smallest(usize::MAX) else {
            break;
        };
        let Some(c2) = smallest(c1) else {
            break;
        };
        weight[c1] += weight[c2];
        weight[c2] = 0;
        let mut c = c1;
        size[c] += 1;
        while next[c] != usize::MAX {
            c = next[c];
            size[c] += 1;
        }
        next[c] = c2;
        let mut c = c2;
        size[c] += 1;
        while next[c] != usize::MAX {
            c = next[c];
            size[c] += 1;
        }
    }

    let mut bits = [0_u32; 33];
    for &s in &size {
        if s > 0 {
            bits[s.min(32)] += 1;
        }
    }
    // Lengths past 16 are folded back up the tree (K.3, Figure K.3).
    for i in (17..=32).rev() {
        while bits[i] > 0 {
            let mut j = i - 2;
            while bits[j] == 0 {
                j -= 1;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
    }
    // Drop the reserved symbol from the longest length.
    let mut i = 16;
    while i > 0 && bits[i] == 0 {
        i -= 1;
    }
    bits[i] = bits[i].saturating_sub(1);

    let mut counts = [0_u8; 16];
    for (count, &b) in counts.iter_mut().zip(&bits[1..=16]) {
        *count = b as u8;
    }
    let mut values = Vec::new();
    for length in 1..=32 {
        for (symbol, &s) in size.iter().take(256).enumerate() {
            if s == length {
                values.push(symbol as u8);
            }
        }
    }
    (counts, values)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "tests operate on known-good values")]
mod tests {
    use super::*;

    /// The code lengths a table assigns, per symbol.
    fn lengths(counts: &[u8; 16], values: &[u8]) -> Vec<(u8, usize)> {
        let mut out = Vec::new();
        let mut at = 0;
        for (length, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                out.push((values[at], length + 1));
                at += 1;
            }
        }
        out
    }

    #[test]
    fn frequent_symbols_get_short_codes_and_the_table_is_valid() {
        let mut freq = [0_u32; 256];
        freq[0] = 1000;
        freq[1] = 100;
        freq[2] = 10;
        freq[3] = 1;
        let (counts, values) = optimal_table(&freq);
        HuffmanEncoder::new(&counts, &values).unwrap();
        let lengths = lengths(&counts, &values);
        assert_eq!(lengths.len(), 4);
        let of = |s| lengths.iter().find(|(v, _)| *v == s).unwrap().1;
        assert!(of(0) < of(1) && of(1) <= of(2) && of(2) <= of(3));
    }

    #[test]
    fn codes_are_limited_to_sixteen_bits() {
        // Fibonacci frequencies make the deepest possible tree.
        let mut freq = [0_u32; 256];
        let (mut a, mut b) = (1_u32, 1_u32);
        for slot in freq.iter_mut().take(30) {
            *slot = a;
            (a, b) = (b, a.saturating_add(b));
        }
        let (counts, values) = optimal_table(&freq);
        assert_eq!(values.len(), 30);
        assert_eq!(counts.iter().map(|&c| usize::from(c)).sum::<usize>(), 30);
        HuffmanEncoder::new(&counts, &values).unwrap();
    }

    #[test]
    fn a_single_symbol_still_makes_a_table() {
        let mut freq = [0_u32; 256];
        freq[0x15] = 7;
        let (counts, values) = optimal_table(&freq);
        assert_eq!(values, [0x15]);
        let encoder = HuffmanEncoder::new(&counts, &values).unwrap();
        assert!(encoder.code(0x15).is_some());
        let (counts, values) = optimal_table(&[0; 256]);
        HuffmanEncoder::new(&counts, &values).unwrap();
    }

    #[test]
    fn the_scan_scripts_cover_every_bit_of_every_coefficient_once() {
        for (script, components) in [(&COLOUR_SCRIPT[..], 3), (&GREY_SCRIPT[..], 1)] {
            // For each component and coefficient, the bits sent so far, as
            // the lowest bit position reached.
            let mut lowest = vec![[None::<u32>; 64]; components];
            for spec in script {
                for &c in spec.components {
                    for slot in &mut lowest[c][spec.ss..=spec.se] {
                        match *slot {
                            None => assert_eq!(spec.ah, 0, "a refinement before a first scan"),
                            Some(at) => assert_eq!(spec.ah, at, "a gap in precision"),
                        }
                        *slot = Some(spec.al);
                    }
                }
            }
            for component in &lowest {
                assert!(component.iter().all(|&at| at == Some(0)), "{component:?}");
            }
        }
    }
}
