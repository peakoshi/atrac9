//! A small model encoder for one mono or stereo block per frame at 48 kHz.
//!
//! Each frame names its band layout, gradient, stereo, extension and scale-factor
//! coding; the encoder derives precisions and spectrum codebooks the way
//! `LibAtrac9` does, writes a stream the C decoder accepts, and records which
//! codebooks and selection rules the stream exercised.

use super::fixtures::{BAND_UNITS, Bits, COEFF_END, c_array, c_table, emit_huffman, spectrum};
use std::collections::BTreeSet;

const MAX_BAND: usize = 18;
const MAX_EXTENSION_BAND: usize = 18;
const BASE_CURVE: [i32; 48] = [
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 26, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];
const BEX_BAND_COUNT: [usize; 8] = [0, 1, 2, 3, 3, 4, 4, 5];
const BEX_DATA_LENGTHS: [[[usize; 4]; 6]; 5] = [
    [
        [0; 4],
        [0; 4],
        [0; 4],
        [5, 4, 3, 3],
        [4, 4, 3, 4],
        [4, 5, 0, 0],
    ],
    [[0; 4]; 6],
    [
        [0; 4],
        [0; 4],
        [0; 4],
        [6, 6, 0, 0],
        [6, 6, 0, 0],
        [6, 0, 0, 0],
    ],
    [
        [0; 4],
        [0; 4],
        [0; 4],
        [4, 4, 0, 0],
        [4, 4, 0, 0],
        [4, 4, 0, 0],
    ],
    [
        [3, 0, 0, 0],
        [3, 0, 0, 0],
        [3, 0, 0, 0],
        [0; 4],
        [0; 4],
        [0; 4],
    ],
];
const BEX_VALUE_COUNTS: [[usize; 6]; 5] = [
    [0, 0, 0, 4, 4, 2],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 1],
    [0, 0, 0, 2, 2, 2],
    [1, 1, 1, 0, 0, 0],
];

#[derive(Clone, Copy, Debug)]
pub struct Bands {
    pub band: usize,
    pub stereo_band: usize,
    pub extension_band: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub enum Gradient {
    /// Mode 0: explicit endpoints.
    Linear {
        start_unit: usize,
        end_unit: usize,
        start_value: i32,
        end_value: i32,
    },
    /// Modes 1 to 3: a curve towards unit 31 and value 31, with scaled masks.
    Curve {
        mode: u32,
        start_unit: usize,
        start_value: i32,
    },
}

impl Gradient {
    pub const fn flat(value: i32) -> Self {
        Self::Linear {
            start_unit: 0,
            end_unit: 1,
            start_value: value,
            end_value: value,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Extension {
    Absent,
    /// Extension data in a block without band extension, skipped by the decoder.
    Skipped {
        mode: u32,
        length: usize,
    },
    /// Band-extension values. `None` writes a zero data length so each channel
    /// keeps its previous values; `extra` pads the declared data length, up to
    /// the largest length the field holds.
    Bex {
        modes: [u32; 2],
        values: Option<[[u32; 4]; 2]>,
        extra: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum Coding {
    /// Primary mode 1: fixed-width values over a base.
    Raw { bits: usize, base: i32 },
    /// Mode 0: unsigned Huffman deltas over a weight table and base.
    Delta {
        weights: usize,
        base: i32,
        bits: usize,
    },
    /// Signed Huffman distance from the previous frame (primary mode 2,
    /// secondary mode 3) or from the primary channel (secondary mode 1).
    Distance { bits: usize, previous: bool },
    /// Unsigned Huffman deltas over a baseline: the previous frame for the
    /// primary channel (mode 3), the primary channel for the secondary (mode 2).
    DeltaBaseline { base: i32, bits: usize },
}

#[derive(Clone, Debug)]
pub struct ScaleFactors {
    pub coding: Coding,
    /// Explicit values; otherwise the encoder picks representable ones.
    pub values: Option<Vec<i32>>,
}

impl ScaleFactors {
    pub const fn coded(coding: Coding) -> Self {
        Self {
            coding,
            values: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FramePlan {
    /// `None` reuses the previous band parameters.
    pub bands: Option<Bands>,
    pub gradient: Gradient,
    pub boundary: usize,
    pub primary: usize,
    pub signs: bool,
    pub extension: Extension,
    pub channels: [ScaleFactors; 2],
}

#[derive(Debug)]
pub struct Encoder {
    stereo: bool,
    frames_per_superframe: usize,
    frame: usize,
    bands: Option<Bands>,
    previous: [[i32; 31]; 2],
    units_previous: usize,
    bex: [(usize, [u32; 4]); 2],
    /// Codebooks and decision rules the written stream exercised.
    pub used: BTreeSet<String>,
}

impl Encoder {
    pub fn new(stereo: bool, frames_per_superframe: usize) -> Self {
        Self {
            stereo,
            frames_per_superframe,
            frame: 0,
            bands: None,
            previous: [[0; 31]; 2],
            units_previous: 0,
            bex: [(0, [0; 4]); 2],
            used: BTreeSet::new(),
        }
    }

    pub fn channels(&self) -> usize {
        if self.stereo { 2 } else { 1 }
    }

    #[allow(clippy::too_many_lines)] // One linear pass in bitstream order.
    pub fn frame(&mut self, plan: &FramePlan) -> Vec<u8> {
        let first = self.frame.is_multiple_of(self.frames_per_superframe);
        let mut b = Bits::default();
        b.write(u32::from(!first), 1);
        b.write(u32::from(plan.bands.is_none()), 1);
        assert!(
            !(first && plan.bands.is_none()),
            "reuse at superframe start"
        );
        if let Some(bands) = plan.bands {
            assert!(bands.band <= MAX_BAND && bands.stereo_band <= bands.band);
            b.write((bands.band - 3) as u32, 4);
            if self.stereo {
                b.write((bands.stereo_band - 3) as u32, 4);
            }
            b.write(u32::from(bands.extension_band.is_some()), 1);
            if let Some(extension) = bands.extension_band {
                assert!(extension >= bands.band && extension <= MAX_EXTENSION_BAND);
                b.write((extension - 3) as u32, 4);
            }
            self.bands = Some(bands);
        }
        let bands = self.bands.expect("band parameters");
        let units = BAND_UNITS[bands.band];
        let stereo_units = if self.stereo {
            BAND_UNITS[bands.stereo_band]
        } else {
            units
        };
        let extension_units = bands.extension_band.map_or(units, |band| BAND_UNITS[band]);

        let gradient = self.gradient(&mut b, plan, units);

        if self.stereo {
            b.write(plan.primary as u32, 1);
            b.write(u32::from(plan.signs), 1);
            if plan.signs {
                for unit in stereo_units..units {
                    b.write(u32::from(unit % 3 == 0), 1);
                }
            }
        }

        let bex_band = bands
            .extension_band
            .map(|_| BEX_BAND_COUNT[units.checked_sub(13).expect("extension units")]);
        self.extension(&mut b, &plan.extension, bex_band);

        let mut current = [[0; 31]; 2];
        for channel in 0..self.channels() {
            let values = self.scale_factors(
                &mut b,
                channel,
                &plan.channels[channel],
                &current[0],
                extension_units,
                first,
            );
            current[channel][..extension_units].copy_from_slice(&values);
            let coded = if !self.stereo || channel == plan.primary {
                units
            } else {
                stereo_units
            };
            let (precisions, fine) = precisions(&current[channel], &gradient, plan, units);
            let sets = self.codebook_sets(&current[channel], coded);
            for unit in 0..coded {
                let bits = precisions[unit] as usize + 1;
                if bits <= 7 {
                    let family = if sets[unit] { 'B' } else { 'A' };
                    let name = format!("Spectrum{family}{bits}{}", codebook_index(unit) + 1);
                    let lengths = c_array(&format!("{name}Bits"));
                    let valid: Vec<usize> = lengths
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &length)| (length > 0).then_some(index))
                        .collect();
                    let values = if bits == 2 && unit >= 8 {
                        4
                    } else if bits <= 4 {
                        2
                    } else {
                        1
                    };
                    for group in 0..(COEFF_END[unit + 1] - COEFF_END[unit]) / values {
                        let pick = self.frame * 7 + channel * 5 + unit * 3 + group;
                        emit_huffman(&mut b, &name, valid[pick % valid.len()]);
                    }
                    self.used.insert(name);
                } else {
                    for coefficient in COEFF_END[unit]..COEFF_END[unit + 1] {
                        b.write(spectrum(53, self.frame, 0, coefficient, channel), bits);
                    }
                    self.used.insert(format!("fixed{bits}"));
                }
            }
            for unit in 0..coded {
                if fine[unit] > 0 {
                    for coefficient in COEFF_END[unit]..COEFF_END[unit + 1] {
                        b.write(
                            spectrum(59, self.frame, 1, coefficient, channel),
                            fine[unit] as usize + 1,
                        );
                    }
                    self.used.insert("fine".into());
                }
            }
        }
        self.previous = current;
        self.units_previous = extension_units;
        self.frame += 1;
        b.finish()
    }

    fn gradient(&mut self, b: &mut Bits, plan: &FramePlan, units: usize) -> [i32; 48] {
        let (mode, start_unit, end_unit, start_value, end_value) = match plan.gradient {
            Gradient::Linear {
                start_unit,
                end_unit,
                start_value,
                end_value,
            } => {
                b.write(0, 2);
                b.write(start_unit as u32, 6);
                b.write((end_unit - 1) as u32, 6);
                b.write(start_value as u32, 5);
                b.write(end_value as u32, 5);
                (0, start_unit, end_unit, start_value, end_value)
            }
            Gradient::Curve {
                mode,
                start_unit,
                start_value,
            } => {
                assert!((1..4).contains(&mode));
                b.write(mode, 2);
                b.write(start_unit as u32, 5);
                b.write(start_value as u32, 5);
                (mode, start_unit, 31, start_value, 31)
            }
        };
        b.write(plan.boundary as u32, 4);
        // LibAtrac9 stores 31 gradient entries; later endpoints overrun its block.
        assert!(plan.boundary <= units && start_unit <= end_unit && end_unit <= 31);
        assert!((0..32).contains(&start_value) && (0..32).contains(&end_value));
        let mut gradient = [0; 48];
        gradient[..end_unit].fill(start_value);
        if end_unit <= units {
            gradient[end_unit..=units].fill(end_value);
        }
        let count = end_unit - start_unit;
        let difference = end_value - start_value;
        if count > 0 && difference != 0 {
            let scale = f64::from(difference.abs() - 1) / 31.0;
            for (offset, value) in gradient[start_unit..end_unit].iter_mut().enumerate() {
                let point = BASE_CURVE[offset * 48 / count];
                let step = (f64::from(point) * scale) as i32;
                *value = if difference > 0 {
                    start_value + 1 + step
                } else {
                    start_value - 1 - step
                };
            }
            self.used.insert(format!(
                "gradient mode {mode} {}",
                if difference > 0 {
                    "ascending"
                } else {
                    "descending"
                }
            ));
        }
        if plan.boundary > 0 {
            self.used.insert("gradient boundary".into());
        }
        gradient
    }

    fn extension(&mut self, b: &mut Bits, extension: &Extension, bex_band: Option<usize>) {
        if let Some(band) = bex_band {
            let header = match extension {
                Extension::Bex { modes, .. } => modes[1],
                _ => 0,
            };
            if self.stereo {
                b.write(header, 2);
                self.bex[1].0 = if band > 2 { header as usize } else { 4 };
            } else {
                b.write(0, 1);
            }
        }
        match (extension, bex_band) {
            (Extension::Absent, _) => b.write(0, 1),
            (Extension::Skipped { mode, length }, None) => {
                b.write(1, 1);
                b.write(*mode, 2);
                b.write(*length as u32, 5);
                for bit in 0..*length {
                    b.write((bit % 3 == 1).into(), 1);
                }
                self.used.insert("skipped extension".into());
            }
            (
                Extension::Bex {
                    modes,
                    values,
                    extra,
                },
                Some(band),
            ) => {
                b.write(1, 1);
                b.write(modes[0], 2);
                self.bex[0].0 = if band > 2 { modes[0] as usize } else { 4 };
                let Some(values) = values else {
                    b.write(0, 5);
                    self.used.insert("retained band extension".into());
                    self.validate_bex(band);
                    return;
                };
                let mut length = *extra;
                for &(mode, _) in &self.bex[..self.channels()] {
                    length += BEX_DATA_LENGTHS[mode][band][..BEX_VALUE_COUNTS[mode][band]]
                        .iter()
                        .sum::<usize>();
                }
                // The five-bit length field caps the padding.
                b.write(length.min(31) as u32, 5);
                let channels = self.channels();
                for ((mode, retained), values) in self.bex[..channels].iter_mut().zip(values) {
                    let mode = *mode;
                    for index in 0..BEX_VALUE_COUNTS[mode][band] {
                        let width = BEX_DATA_LENGTHS[mode][band][index];
                        let value = values[index] & ((1 << width) - 1);
                        b.write(value, width);
                        retained[index] = value;
                    }
                    self.used
                        .insert(format!("band extension group {band} mode {mode}"));
                }
                self.validate_bex(band);
            }
            other => panic!("extension {other:?} does not fit the band layout"),
        }
    }

    fn validate_bex(&self, band: usize) {
        for &(mode, values) in &self.bex[..self.channels()] {
            let limits: &[u32] = match (mode, band) {
                (0, 3) => &[32, 16, 8, 8],
                (0, 4) => &[16, 16, 8, 16],
                (0, 5) => &[16, 32],
                (1, _) => &[],
                (2, _) => &[64, 64],
                (3, _) => &[16, 16],
                (4, _) => &[8],
                _ => panic!("band extension mode {mode} in group {band}"),
            };
            assert!(
                values
                    .iter()
                    .zip(limits)
                    .all(|(value, limit)| value < limit)
            );
        }
    }

    #[allow(clippy::too_many_lines)] // One arm per coding, in bitstream order.
    fn scale_factors(
        &mut self,
        b: &mut Bits,
        channel: usize,
        plan: &ScaleFactors,
        primary: &[i32; 31],
        count: usize,
        first: bool,
    ) -> Vec<i32> {
        let seed = self.frame * 11 + channel * 7;
        // Prefer mid-range values so later frames can code relative to them.
        let pick = |unit: usize, low: i32, high: i32| -> i32 {
            assert!(
                low <= high,
                "no representable scale factor at unit {unit}: {plan:?}"
            );
            let (low, high) = if low.max(6) <= high.min(25) {
                (low.max(6), high.min(25))
            } else {
                (low, high)
            };
            let span = (high - low + 1) as usize;
            low + ((unit * 13 + seed * 5 + unit * unit) % span) as i32
        };
        let previous = self.previous[channel];
        let (mode, name) = match (channel, plan.coding) {
            (_, Coding::Delta { .. }) => (0, "delta"),
            (0, Coding::Raw { .. }) => (1, "raw"),
            (0, Coding::Distance { previous: true, .. }) => (2, "distance to previous"),
            (0, Coding::DeltaBaseline { .. }) => (3, "delta to previous"),
            (
                1,
                Coding::Distance {
                    previous: false, ..
                },
            ) => (1, "distance to primary"),
            (1, Coding::DeltaBaseline { .. }) => (2, "delta to primary"),
            (1, Coding::Distance { previous: true, .. }) => (3, "distance to previous"),
            other => panic!("unsupported scale factor coding {other:?}"),
        };
        b.write(mode, 2);
        let uses_previous = matches!((channel, mode), (0, 2 | 3) | (1, 3));
        assert!(
            !(first && uses_previous),
            "previous scale factors at superframe start"
        );
        let (baseline, baseline_count) = if uses_previous {
            (previous, self.units_previous)
        } else {
            (*primary, count)
        };
        let shared = count.min(baseline_count);
        let explicit = plan.values.as_ref();
        if let Some(values) = explicit {
            assert_eq!(values.len(), count);
        }
        let choose = |unit: usize, low: i32, high: i32| {
            explicit.map_or_else(
                || pick(unit, low.max(0), high.min(31)),
                |values| values[unit],
            )
        };
        let mut values = vec![0; count];
        match plan.coding {
            Coding::Raw { bits, base } => {
                b.write((bits - 2) as u32, 2);
                if bits < 5 {
                    b.write(base as u32, 5);
                }
                let base = if bits < 5 { base } else { 0 };
                for (unit, value) in values.iter_mut().enumerate() {
                    *value = choose(unit, base, base + (1 << bits) - 1);
                    let raw = *value - base;
                    assert!((0..1 << bits).contains(&raw));
                    b.write(raw as u32, bits);
                }
                self.used.insert(format!("{name} {bits} bits base {base}"));
            }
            Coding::Delta {
                weights,
                base,
                bits,
            } => {
                let table = &c_table("tables.c", "ScaleFactorWeights")[weights * 32..];
                b.write(weights as u32, 3);
                b.write(base as u32, 5);
                b.write((bits - 3) as u32, 2);
                let mut running = Vec::with_capacity(count);
                for (unit, value) in values.iter_mut().enumerate() {
                    let weight = table[unit] as i32;
                    *value = choose(unit, base - weight, base - weight + (1 << bits) - 1);
                    let raw = *value - base + weight;
                    assert!((0..1 << bits).contains(&raw));
                    running.push(raw);
                }
                b.write(running[0] as u32, bits);
                for unit in 1..count {
                    let delta = (running[unit] - running[unit - 1]) & ((1 << bits) - 1);
                    emit_huffman(b, &format!("ScaleFactorsA{bits}"), delta as usize);
                }
                self.used
                    .insert(format!("{name} A{bits} weights {weights}"));
            }
            Coding::Distance { bits, .. } => {
                b.write((bits - 2) as u32, 2);
                let limit = (1 << (bits - 1)) - 1;
                for (unit, value) in values.iter_mut().enumerate() {
                    if unit < shared {
                        *value = choose(unit, baseline[unit] - limit, baseline[unit] + limit);
                        let distance = (*value - baseline[unit] + 16).rem_euclid(32) - 16;
                        assert!(distance.abs() <= limit);
                        let index = (distance & ((1 << bits) - 1)) as usize;
                        emit_huffman(b, &format!("ScaleFactorsB{bits}"), index);
                    }
                }
                for (unit, value) in values.iter_mut().enumerate().skip(shared) {
                    *value = choose(unit, 0, 31);
                    b.write(*value as u32, 5);
                }
                self.used.insert(format!("{name} B{bits}"));
                if shared < count {
                    self.used.insert(format!("{name} raw tail"));
                }
            }
            Coding::DeltaBaseline { base, bits } => {
                b.write((base + 16) as u32, 5);
                b.write((bits - 1) as u32, 2);
                let mut running = Vec::with_capacity(shared);
                for (unit, value) in values.iter_mut().enumerate().take(shared) {
                    let low = base + baseline[unit];
                    *value = choose(unit, low, low + (1 << bits) - 1);
                    let raw = *value - low;
                    assert!((0..1 << bits).contains(&raw));
                    running.push(raw);
                }
                if shared > 0 {
                    b.write(running[0] as u32, bits);
                }
                for unit in 1..shared {
                    let delta = (running[unit] - running[unit - 1]) & ((1 << bits) - 1);
                    emit_huffman(b, &format!("ScaleFactorsA{bits}"), delta as usize);
                }
                for (unit, value) in values.iter_mut().enumerate().skip(shared) {
                    *value = choose(unit, 0, 31);
                    b.write(*value as u32, 5);
                }
                self.used.insert(format!("{name} A{bits} base {base}"));
                if shared < count {
                    self.used.insert(format!("{name} raw tail"));
                }
            }
        }
        assert!(values.iter().all(|value| (0..32).contains(value)));
        values
    }
}

fn precisions(
    sf: &[i32; 31],
    gradient: &[i32; 48],
    plan: &FramePlan,
    units: usize,
) -> ([i32; 30], [i32; 30]) {
    let mut mask = [0; 30];
    for unit in 1..units {
        let delta = sf[unit] - sf[unit - 1];
        if delta > 1 {
            mask[unit] += (delta - 1).min(5);
        } else if delta < -1 {
            mask[unit - 1] += (-delta - 1).min(5);
        }
    }
    let mode = match plan.gradient {
        Gradient::Linear { .. } => 0,
        Gradient::Curve { mode, .. } => mode,
    };
    let mut precisions = [0; 30];
    let mut fine = [0; 30];
    for unit in 0..units {
        let mut precision = sf[unit] - gradient[unit];
        if mode != 0 {
            precision += mask[unit];
            if precision > 0 {
                precision = match mode {
                    1 => precision / 2,
                    2 => 3 * precision / 8,
                    _ => precision / 4,
                };
            }
        }
        precision = precision.max(1) + i32::from(unit < plan.boundary);
        fine[unit] = (precision - 15).max(0);
        assert!(
            fine[unit] <= 15,
            "fine precision {} at unit {unit}",
            fine[unit]
        );
        precisions[unit] = precision.min(15);
    }
    (precisions, fine)
}

impl Encoder {
    fn codebook_sets(&mut self, scale_factors: &[i32; 31], count: usize) -> [bool; 30] {
        let mut sets = [false; 30];
        if count <= 1 {
            return sets;
        }
        let mut sf = *scale_factors;
        sf[count] = sf[count - 1];
        let average = if count > 12 {
            (sf[..12].iter().sum::<i32>() + 6) / 12
        } else {
            0
        };
        for unit in 8..count {
            let (previous, next) = (sf[unit - 1], sf[unit + 1]);
            if sf[unit] - previous.min(next) >= 3 || sf[unit] - previous + sf[unit] - next >= 3 {
                sets[unit] = true;
            }
        }
        for unit in 12..count {
            if !sets[unit] && sf[unit] - sf[unit - 1].min(sf[unit + 1]) >= 2 {
                let wide = COEFF_END[unit + 1] - COEFF_END[unit] == 16;
                if sf[unit] >= average - i32::from(wide) {
                    sets[unit] = true;
                    self.used.insert(format!(
                        "second codebook pass {}",
                        if wide && sf[unit] == average - 1 {
                            "wide unit below average"
                        } else {
                            "at or above average"
                        }
                    ));
                }
            }
        }
        sets
    }
}

const fn codebook_index(unit: usize) -> usize {
    match unit {
        0..8 => 0,
        8..12 => 1,
        12..20 => 2,
        _ => 3,
    }
}
