//! Progressive JPEG: the scans, decoded into a coefficient buffer.
//!
//! A progressive frame (`SOF2`) sends its coefficients in many scans, each
//! carrying a band of frequencies (*spectral selection*) or one more bit of
//! precision for coefficients already sent (*successive approximation*). No
//! coefficient is final until the last scan, so unlike the baseline path,
//! which turns each MCU row into pixels as it arrives, every scan is decoded
//! into a whole-image buffer of quantized coefficients first. The decoder
//! then reads blocks from that buffer through the same dequantize, inverse
//! DCT, upsample and colour path as baseline, a band at a time — which is
//! also what gives progressive files shrink-on-load: a reduced IDCT of the
//! finished coefficients.
//!
//! # The four kinds of scan (ITU-T T.81 §G.1.2)
//!
//! | Scan | Selects | Carries |
//! |---|---|---|
//! | DC first | coefficient 0 | its value above bit `Al`, as a difference from the previous block |
//! | DC refine | coefficient 0 | bit `Al`, one raw bit per block |
//! | AC first | `Ss..=Se` of one component | values above bit `Al`, with end-of-band runs across blocks |
//! | AC refine | `Ss..=Se` of one component | bit `Al` of known coefficients, and new ones of magnitude `2^Al` |
//!
//! # Memory
//!
//! Two bytes per coefficient: 36 MB for a 12-megapixel 4:2:0 photograph,
//! about what libjpeg holds for the same file. This is the documented
//! exception to streaming (SPEC §Formats); `max_pixels` bounds it.

use crate::entropy::Reader;
use crate::format::{Frame, Scan};
use crate::huffman::HuffmanTable;
use otf_pixels_core::{PixelsError, Result, Source};

/// The most scans one frame may have.
///
/// Real progressive files have about ten. Each scan walks every block of its
/// components, so without a bound a small file of empty scans could ask for
/// unbounded work; 500 is libjpeg-turbo's recommended limit.
pub const MAX_SCANS: usize = 500;

/// One component's quantized coefficients, in zigzag order per block.
#[derive(Debug)]
struct Plane {
    /// Blocks per row, padded to whole MCUs as the interleaved layout needs.
    blocks_wide: usize,
    /// Blocks that hold image data across and down, for non-interleaved scans,
    /// which code only those.
    data_wide: usize,
    data_high: usize,
    blocks: Vec<[i16; 64]>,
}

/// Every component's coefficients for the whole frame.
#[derive(Debug)]
pub struct Coefficients {
    planes: Vec<Plane>,
    mcus_per_line: usize,
    mcu_rows: usize,
}

impl Coefficients {
    /// Zeroed coefficients for `frame`, laid out as `mcus_per_line` by
    /// `mcu_rows` MCUs.
    ///
    /// # Errors
    ///
    /// Returns [`PixelsError::Malformed`] if the block count overflows.
    pub fn new(frame: &Frame, mcus_per_line: u32, mcu_rows: u32) -> Result<Self> {
        let (h_max, v_max) = (u32::from(frame.h_max()), u32::from(frame.v_max()));
        let (width, height) = (u32::from(frame.width), u32::from(frame.height));
        let mut planes = Vec::with_capacity(frame.components.len());
        for component in &frame.components {
            let (h, v) = (u32::from(component.h), u32::from(component.v));
            let blocks_wide = (mcus_per_line * h) as usize;
            let blocks_high = (mcu_rows * v) as usize;
            let count = blocks_wide
                .checked_mul(blocks_high)
                .ok_or_else(|| PixelsError::malformed("jpeg", "coefficient buffer overflows"))?;
            // T.81 §A.2.2: a component spans ceil(X * h / h_max) samples.
            let samples_wide = (width * h).div_ceil(h_max);
            let samples_high = (height * v).div_ceil(v_max);
            planes.push(Plane {
                blocks_wide,
                data_wide: samples_wide.div_ceil(8) as usize,
                data_high: samples_high.div_ceil(8) as usize,
                blocks: vec![[0; 64]; count],
            });
        }
        Ok(Self {
            planes,
            mcus_per_line: mcus_per_line as usize,
            mcu_rows: mcu_rows as usize,
        })
    }

    /// The block at `(x, y)` of `component`, if it exists.
    pub fn block(&self, component: usize, x: usize, y: usize) -> Option<&[i16; 64]> {
        let plane = self.planes.get(component)?;
        plane.blocks.get(y * plane.blocks_wide + x)
    }

    fn block_mut(&mut self, component: usize, x: usize, y: usize) -> Option<&mut [i16; 64]> {
        let plane = self.planes.get_mut(component)?;
        plane.blocks.get_mut(y * plane.blocks_wide + x)
    }
}

/// The tables a scan decodes with, which may change between scans.
pub struct Tables<'a> {
    pub dc: &'a [Option<HuffmanTable>; 4],
    pub ac: &'a [Option<HuffmanTable>; 4],
    /// MCUs between restart markers; zero means there are none.
    pub restart_interval: u16,
}

/// What kind of scan a header describes, checked against T.81 §G.1.1.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    DcFirst,
    DcRefine,
    AcFirst,
    AcRefine,
}

fn kind(scan: &Scan) -> Result<Kind> {
    let (start, end) = (scan.spectral_start, scan.spectral_end);
    let (high, low) = (scan.approx_high, scan.approx_low);
    // 13 is the largest shift a 16-bit coefficient can take and still mean
    // anything; the standard's own bound for 8-bit samples.
    if high > 13 || low > 13 {
        return Err(PixelsError::malformed(
            "jpeg",
            format!("successive approximation {high}/{low} is out of range"),
        ));
    }
    if start == 0 {
        if end != 0 {
            return Err(PixelsError::malformed(
                "jpeg",
                format!("a DC scan selects coefficients 0..={end}; it must stop at 0"),
            ));
        }
        return Ok(if high == 0 {
            Kind::DcFirst
        } else {
            Kind::DcRefine
        });
    }
    if end < start || end > 63 {
        return Err(PixelsError::malformed(
            "jpeg",
            format!("AC scan selects coefficients {start}..={end}"),
        ));
    }
    if scan.components.len() != 1 {
        return Err(PixelsError::malformed(
            "jpeg",
            "an AC scan codes more than one component; progressive AC scans are single-component",
        ));
    }
    Ok(if high == 0 {
        Kind::AcFirst
    } else {
        Kind::AcRefine
    })
}

/// Per-scan decoding state carried from block to block.
struct State {
    kind: Kind,
    start: usize,
    end: usize,
    low: u32,
    /// DC predictor per scan component, reset at each restart.
    predictors: Vec<i32>,
    /// Blocks still to skip in an end-of-band run.
    eob_run: u32,
}

/// Decode one scan's entropy-coded data into `coefficients`.
///
/// # Errors
///
/// Returns [`PixelsError::Malformed`] for a scan header outside T.81's rules,
/// a scan that names an undefined Huffman table, entropy data that decodes to
/// nothing, or a missing restart marker.
pub fn decode_scan<S: Source>(
    reader: &mut Reader<S>,
    frame: &Frame,
    scan: &Scan,
    tables: &Tables<'_>,
    coefficients: &mut Coefficients,
) -> Result<()> {
    let kind = kind(scan)?;
    // Only the tables this kind of scan reads have to exist.
    for component in &scan.components {
        let missing = match kind {
            Kind::DcFirst => tables
                .dc
                .get(component.dc as usize)
                .is_none_or(Option::is_none),
            Kind::AcFirst | Kind::AcRefine => tables
                .ac
                .get(component.ac as usize)
                .is_none_or(Option::is_none),
            Kind::DcRefine => false,
        };
        if missing {
            return Err(PixelsError::malformed(
                "jpeg",
                "scan names a Huffman table that was never defined",
            ));
        }
    }

    let mut state = State {
        kind,
        start: usize::from(scan.spectral_start),
        end: usize::from(scan.spectral_end),
        low: u32::from(scan.approx_low),
        predictors: vec![0; scan.components.len()],
        eob_run: 0,
    };

    let interval = u32::from(tables.restart_interval);
    let mut restarts_left = interval;
    if let [only] = scan.components.as_slice() {
        // A scan of one component codes that component's blocks in raster
        // order, one block per MCU and only blocks that hold image data.
        let (wide, high) = coefficients
            .planes
            .get(only.index)
            .map(|plane| (plane.data_wide, plane.data_high))
            .ok_or_else(|| PixelsError::malformed("jpeg", "scan names an unknown component"))?;
        for y in 0..high {
            for x in 0..wide {
                next_unit(reader, &mut state, interval, &mut restarts_left)?;
                decode_block(reader, tables, scan, 0, x, y, &mut state, coefficients)?;
            }
        }
    } else {
        // A scan of several — which progressive allows for DC only — walks
        // whole MCUs, each component's blocks in turn.
        for mcu_y in 0..coefficients.mcu_rows {
            for mcu_x in 0..coefficients.mcus_per_line {
                next_unit(reader, &mut state, interval, &mut restarts_left)?;
                for (slot, scanned) in scan.components.iter().enumerate() {
                    let Some(component) = frame.components.get(scanned.index) else {
                        continue;
                    };
                    let (h, v) = (usize::from(component.h), usize::from(component.v));
                    for block_y in 0..v {
                        for block_x in 0..h {
                            let (x, y) = (mcu_x * h + block_x, mcu_y * v + block_y);
                            decode_block(
                                reader,
                                tables,
                                scan,
                                slot,
                                x,
                                y,
                                &mut state,
                                coefficients,
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Account for one MCU, consuming a restart marker when an interval ends.
fn next_unit<S: Source>(
    reader: &mut Reader<S>,
    state: &mut State,
    interval: u32,
    restarts_left: &mut u32,
) -> Result<()> {
    if interval == 0 {
        return Ok(());
    }
    if *restarts_left == 0 {
        if !reader.restart()? {
            return Err(PixelsError::malformed(
                "jpeg",
                "expected a restart marker between MCU intervals",
            ));
        }
        // Both are differential within an interval only.
        state.predictors.iter_mut().for_each(|p| *p = 0);
        state.eob_run = 0;
        *restarts_left = interval;
    }
    *restarts_left -= 1;
    Ok(())
}

/// Decode scan component `slot`'s contribution to block `(x, y)`.
#[allow(
    clippy::too_many_arguments,
    reason = "one block of one scan needs all of it"
)]
fn decode_block<S: Source>(
    reader: &mut Reader<S>,
    tables: &Tables<'_>,
    scan: &Scan,
    slot: usize,
    x: usize,
    y: usize,
    state: &mut State,
    coefficients: &mut Coefficients,
) -> Result<()> {
    let Some(scanned) = scan.components.get(slot) else {
        return Ok(());
    };
    let Some(block) = coefficients.block_mut(scanned.index, x, y) else {
        return Err(PixelsError::malformed(
            "jpeg",
            "scan addresses a block outside its component",
        ));
    };
    match state.kind {
        Kind::DcFirst => dc_first(reader, table(tables.dc, scanned.dc)?, state, slot, block),
        Kind::DcRefine => dc_refine(reader, state, block),
        Kind::AcFirst => ac_first(reader, table(tables.ac, scanned.ac)?, state, block),
        Kind::AcRefine => ac_refine(reader, table(tables.ac, scanned.ac)?, state, block),
    }
}

fn table(tables: &[Option<HuffmanTable>; 4], slot: u8) -> Result<&HuffmanTable> {
    tables
        .get(slot as usize)
        .and_then(Option::as_ref)
        .ok_or_else(|| PixelsError::malformed("jpeg", "scan names an undefined Huffman table"))
}

/// Store `value << shift` in 16 bits, wrapping rather than panicking: a
/// crafted stream gets nonsense pixels, which is the right answer to it.
fn shifted(value: i32, shift: u32) -> i16 {
    value.wrapping_shl(shift) as i16
}

fn dc_first<S: Source>(
    reader: &mut Reader<S>,
    table: &HuffmanTable,
    state: &mut State,
    slot: usize,
    block: &mut [i16; 64],
) -> Result<()> {
    let magnitude = reader.decode(table)?;
    if magnitude > 15 {
        return Err(PixelsError::malformed(
            "jpeg",
            format!("DC coefficient claims {magnitude} bits; 15 is the maximum"),
        ));
    }
    let difference = reader.receive_extend(u32::from(magnitude))?;
    let Some(predictor) = state.predictors.get_mut(slot) else {
        return Ok(());
    };
    *predictor = predictor.wrapping_add(difference);
    block[0] = shifted(*predictor, state.low);
    Ok(())
}

fn dc_refine<S: Source>(
    reader: &mut Reader<S>,
    state: &State,
    block: &mut [i16; 64],
) -> Result<()> {
    if reader.get_bits(1)? == 1 {
        block[0] |= shifted(1, state.low);
    }
    Ok(())
}

fn ac_first<S: Source>(
    reader: &mut Reader<S>,
    table: &HuffmanTable,
    state: &mut State,
    block: &mut [i16; 64],
) -> Result<()> {
    if state.eob_run > 0 {
        state.eob_run -= 1;
        return Ok(());
    }
    let mut k = state.start;
    while k <= state.end {
        let symbol = reader.decode(table)?;
        let (run, size) = (usize::from(symbol >> 4), u32::from(symbol & 0x0F));
        if size == 0 {
            if run < 15 {
                // EOBn: this block and the next 2^run + extra - 1 end here.
                state.eob_run = (1_u32 << run) - 1;
                if run > 0 {
                    state.eob_run += reader.get_bits(run as u32)?;
                }
                break;
            }
            // ZRL: sixteen zeros.
            k += 16;
            continue;
        }
        k += run;
        let Some(slot) = block.get_mut(k) else {
            return Err(PixelsError::malformed(
                "jpeg",
                "coefficient run passes the end of the band",
            ));
        };
        *slot = shifted(reader.receive_extend(size)?, state.low);
        k += 1;
    }
    Ok(())
}

/// Refine one block's band by one bit: T.81 §G.1.2.3, as libjpeg implements
/// it.
///
/// Coefficients already nonzero each get one correction bit, read as they
/// are passed. Between them, the stream places new coefficients of magnitude
/// `2^Al`, each preceded by a run that counts only coefficients still zero.
fn ac_refine<S: Source>(
    reader: &mut Reader<S>,
    table: &HuffmanTable,
    state: &mut State,
    block: &mut [i16; 64],
) -> Result<()> {
    let positive = shifted(1, state.low);
    let negative = shifted(-1, state.low);
    let mut k = state.start;

    if state.eob_run == 0 {
        while k <= state.end {
            let symbol = reader.decode(table)?;
            let (mut run, size) = (i32::from(symbol >> 4), symbol & 0x0F);
            let mut value = 0_i16;
            if size != 0 {
                // A new coefficient is always of magnitude one at this bit;
                // libjpeg tolerates a wrong size here, and so do we.
                value = if reader.get_bits(1)? == 1 {
                    positive
                } else {
                    negative
                };
            } else if run != 15 {
                // EOBn: refine the rest of this band, then skip ahead.
                state.eob_run = 1_u32 << run;
                if run > 0 {
                    state.eob_run += reader.get_bits(run as u32)?;
                }
                break;
            }
            // Walk forward: correct coefficients already set, and count the
            // zeros the run skips; the new value lands on the zero after them.
            while k <= state.end {
                let Some(coefficient) = block.get_mut(k) else {
                    break;
                };
                if *coefficient != 0 {
                    refine(reader, coefficient, positive, negative)?;
                } else {
                    if run == 0 {
                        break;
                    }
                    run -= 1;
                }
                k += 1;
            }
            if value != 0 {
                if let Some(slot) = block.get_mut(k) {
                    *slot = value;
                }
            }
            k += 1;
        }
    }

    if state.eob_run > 0 {
        // Inside an end-of-band run only the correction bits remain.
        while k <= state.end {
            if let Some(coefficient) = block.get_mut(k) {
                if *coefficient != 0 {
                    refine(reader, coefficient, positive, negative)?;
                }
            }
            k += 1;
        }
        state.eob_run -= 1;
    }
    Ok(())
}

/// Apply one correction bit to a coefficient already known to be nonzero.
fn refine<S: Source>(
    reader: &mut Reader<S>,
    coefficient: &mut i16,
    positive: i16,
    negative: i16,
) -> Result<()> {
    if reader.get_bits(1)? == 1 && *coefficient & positive == 0 {
        *coefficient = coefficient.wrapping_add(if *coefficient >= 0 {
            positive
        } else {
            negative
        });
    }
    Ok(())
}
