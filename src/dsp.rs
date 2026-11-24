//! Allocation-free synthesis ported from Alex Barney's MIT-licensed `LibAtrac9`.

use crate::Atrac9Config;
use crate::state::{Block, BlockType, Channel, Rng};
use crate::tables::{
    QUANT_UNIT_TO_COEFF_INDEX, QUANTIZER_FINE_STEP_SIZE, QUANTIZER_STEP_SIZE, SPECTRUM_SCALE,
};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug)]
struct Trig {
    sin: f64,
    cos: f64,
}

/// Reusable transform workspace shared by channels decoded sequentially.
#[derive(Debug)]
pub(crate) struct Scratch {
    output: [f64; 256],
    temp: [f64; 256],
}

impl Scratch {
    /// Initializes transform storage once. Each transform overwrites its active prefix.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            output: [0.0; 256],
            temp: [0.0; 256],
        }
    }
}

/// Shared transform tables for 64, 128 and 256 samples.
static SHARED: [OnceLock<DspTables>; 3] = [const { OnceLock::new() }; 3];
const MIN_BITS: usize = 6;

/// Frozen transform coefficients, built once per process and shared by every
/// decoder.
#[derive(Debug)]
pub(crate) struct DspTables {
    bits: usize,
    size: usize,
    trig: Box<[Trig]>,
    // Indices below 256 select from the fixed-size workspace without bounds checks.
    shuffle: Box<[u8]>,
    window: Box<[f64]>,
    bex_rates: [f64; 16],
}

impl DspTables {
    /// Generates reference transform tables once, before any frames are decoded.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::manual_midpoint)] // Sizes <=256; retain C window operation order.
    pub(crate) fn new(bits: usize) -> Self {
        let size = 1 << bits;
        // The first transform stage reads only the lower half of the top row.
        let mut trig = Vec::with_capacity(size + size / 2 - 1);
        for row in 0..=bits {
            let row_size = 1 << row;
            let used = if row == bits { row_size / 2 } else { row_size };
            for i in 0..used {
                // Preserve the C reference's multiplication/division order.
                let value = std::f64::consts::PI * f64::from(4 * i + 1) / f64::from(4 * row_size);
                trig.push(Trig {
                    sin: value.sin(),
                    cos: value.cos(),
                });
            }
        }
        let shuffle = (0..size).map(|i| shuffle_index(i, bits)).collect();
        let mdct_window = (0..size)
            .map(|i| (((i as f64 + 0.5) / size as f64 - 0.5) * std::f64::consts::PI).sin())
            .map(|value| (value + 1.0) * 0.5)
            .collect::<Vec<_>>();
        let window = mdct_window
            .iter()
            .zip(mdct_window.iter().rev())
            .map(|(&front, &back)| front / (back * back + front * front))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            bits,
            size,
            trig: trig.into_boxed_slice(),
            shuffle,
            window,
            bex_rates: BEX_MODE3_RATE.map(|rate| 2.0_f64.powf(rate)),
        }
    }

    /// Builds the tables for every transform size up to `2^bits` on first use
    /// and returns those for `2^bits`. Only decoder construction calls this,
    /// so decoding never waits on initialization.
    pub(crate) fn shared_up_to(bits: usize) -> &'static Self {
        for smaller in MIN_BITS..bits {
            SHARED[smaller - MIN_BITS].get_or_init(|| Self::new(smaller));
        }
        SHARED[bits - MIN_BITS].get_or_init(|| Self::new(bits))
    }

    /// Tables already built by [`Self::shared_up_to`]; a lock-free lookup.
    pub(crate) fn built(bits: usize) -> Option<&'static Self> {
        SHARED.get(bits.checked_sub(MIN_BITS)?)?.get()
    }

    fn trig_row(&self, bits: usize) -> &[Trig] {
        let length = 1 << bits;
        &self.trig[length - 1..(length * 2 - 1).min(self.trig.len())]
    }
}

#[allow(clippy::cast_possible_truncation)] // Bit-reversed indices of at most eight bits.
fn shuffle_index(i: usize, bits: usize) -> u8 {
    ((i ^ (i / 2)).reverse_bits() >> (usize::BITS as usize - bits)) as u8
}

/// Synthesizes one completely validated and unpacked block.
pub(crate) fn decode_block(
    _config: &Atrac9Config,
    block: &mut Block,
    tables: &DspTables,
    scratch: &mut Scratch,
) {
    dequantize(block, tables.size);
    intensity_stereo(block);
    for channel in &mut block.channels[..block.channel_count] {
        scale_quant_units(
            &mut channel.spectra,
            &channel.scale_factors,
            block.quantization_unit_count,
        );
        if block.band_extension_enabled && block.has_extension_data {
            band_extension(channel, block.quantization_unit_count, tables);
        }
        imdct(channel, tables, scratch);
    }
}

/// Writes every spectrum coefficient the transform reads: coded units from
/// the quantized values and zeros up to `size`. Quantized values are read only
/// where unpacking wrote them; units without fine precision skip the fine term,
/// whose zero product leaves the coarse value unchanged.
#[allow(clippy::cast_sign_loss)] // Unpack validates all precision indices in 0..16.
fn dequantize(block: &mut Block, size: usize) {
    for channel in &mut block.channels[..block.channel_count] {
        let units = channel.coded_quant_units;
        for unit in 0..units {
            let start = QUANT_UNIT_TO_COEFF_INDEX[unit];
            let end = QUANT_UNIT_TO_COEFF_INDEX[unit + 1];
            let coarse_step = QUANTIZER_STEP_SIZE[channel.precisions[unit] as usize];
            let output = &mut channel.spectra[start..end];
            let coarse = &channel.quantized_spectra[start..end];
            match channel.precisions_fine[unit] {
                0 => {
                    for (output, &coarse) in output.iter_mut().zip(coarse) {
                        *output = f64::from(coarse) * coarse_step;
                    }
                }
                fine_precision => {
                    let fine_step = QUANTIZER_FINE_STEP_SIZE[fine_precision as usize];
                    for ((output, &coarse), &fine) in output
                        .iter_mut()
                        .zip(coarse)
                        .zip(&channel.quantized_spectra_fine[start..end])
                    {
                        *output = f64::from(coarse) * coarse_step + f64::from(fine) * fine_step;
                    }
                }
            }
        }
        channel.spectra[QUANT_UNIT_TO_COEFF_INDEX[units]..size].fill(0.0);
    }
}

fn intensity_stereo(block: &mut Block) {
    if block.block_type != BlockType::Stereo {
        return;
    }
    let (first, second) = block.channels.split_at_mut(1);
    let (source, dest) = if block.primary_channel_index == 0 {
        (&first[0].spectra, &mut second[0].spectra)
    } else {
        (&second[0].spectra, &mut first[0].spectra)
    };
    for unit in block.stereo_quantization_unit..block.quantization_unit_count {
        let start = QUANT_UNIT_TO_COEFF_INDEX[unit];
        let end = QUANT_UNIT_TO_COEFF_INDEX[unit + 1];
        if block.joint_stereo_signs[unit] {
            for (output, &input) in dest[start..end].iter_mut().zip(&source[start..end]) {
                *output = -input;
            }
        } else {
            dest[start..end].copy_from_slice(&source[start..end]);
        }
    }
}

#[allow(clippy::cast_sign_loss)] // Unpack validates all scale-factor indices in 0..32.
fn scale_quant_units(spectra: &mut [f64; 256], scale_factors: &[i32; 31], units: usize) {
    for (unit, &factor) in scale_factors[..units].iter().enumerate() {
        let scale = SPECTRUM_SCALE[factor as usize];
        scale_slice(
            &mut spectra[QUANT_UNIT_TO_COEFF_INDEX[unit]..QUANT_UNIT_TO_COEFF_INDEX[unit + 1]],
            scale,
        );
    }
}

fn scale_slice(spectra: &mut [f64], scale: f64) {
    for value in spectra {
        *value *= scale;
    }
}

#[allow(clippy::cast_sign_loss)] // Unpack validates all scale-factor indices in 0..32.
fn band_extension(channel: &mut Channel, group_a_unit: usize, tables: &DspTables) {
    let group = BEX_GROUP_INFO[group_a_unit - 13];
    let total_units = group.group_c_unit.max(22);
    let bex_units = total_units - group_a_unit;
    let a = QUANT_UNIT_TO_COEFF_INDEX[group_a_unit];
    let b = QUANT_UNIT_TO_COEFF_INDEX[group.group_b_unit];
    let c = QUANT_UNIT_TO_COEFF_INDEX[group.group_c_unit];
    let end = QUANT_UNIT_TO_COEFF_INDEX[total_units];
    mirror_spectra(&mut channel.spectra, a, b);
    mirror_spectra(&mut channel.spectra, b, c);
    mirror_spectra(&mut channel.spectra, c, end);
    let values = channel.bex_values;
    let mut scales = [0.0; 6];
    match channel.bex_mode {
        0 => {
            match group.band_count {
                3 => {
                    scales[0] = BEX_MODE0_BANDS3[0][values[0]];
                    scales[1] = BEX_MODE0_BANDS3[1][values[0]];
                    scales[2] = BEX_MODE0_BANDS3[2][values[1]];
                    scales[3] = BEX_MODE0_BANDS3[3][values[2]];
                    scales[4] = BEX_MODE0_BANDS3[4][values[3]];
                }
                4 => {
                    scales[0] = BEX_MODE0_BANDS4[0][values[0]];
                    scales[1] = BEX_MODE0_BANDS4[1][values[0]];
                    scales[2] = BEX_MODE0_BANDS4[2][values[1]];
                    scales[3] = BEX_MODE0_BANDS4[3][values[2]];
                    scales[4] = BEX_MODE0_BANDS4[4][values[3]];
                }
                5 => {
                    scales[0] = BEX_MODE0_BANDS5[0][values[0]];
                    scales[1] = BEX_MODE0_BANDS5[1][values[1]];
                    scales[2] = BEX_MODE0_BANDS5[2][values[1]];
                }
                _ => {}
            }
            scales[bex_units - 1] = SPECTRUM_SCALE[channel.scale_factors[group_a_unit] as usize];
            add_noise(channel, QUANT_UNIT_TO_COEFF_INDEX[total_units - 1], end);
        }
        1 => {
            for (scale, &factor) in scales[..bex_units]
                .iter_mut()
                .zip(&channel.scale_factors[group_a_unit..total_units])
            {
                *scale = SPECTRUM_SCALE[factor as usize];
            }
            add_noise(channel, a, end);
        }
        2 => {
            scale_slice(&mut channel.spectra[a..b], BEX_MODE2_SCALE[values[0]]);
            scale_slice(&mut channel.spectra[b..c], BEX_MODE2_SCALE[values[1]]);
            return;
        }
        3 => {
            let rate = tables.bex_rates[values[1]];
            let mut scale = BEX_MODE3_INITIAL[values[0]];
            for coefficient in &mut channel.spectra[a..end] {
                scale *= rate;
                *coefficient *= scale;
            }
            return;
        }
        4 => {
            let mult = BEX_MODE4_MULTIPLIER[values[0]];
            scale_slice(&mut channel.spectra[a..b], 0.707_946_8 * mult);
            scale_slice(&mut channel.spectra[b..c], 0.501_190_2 * mult);
            scale_slice(&mut channel.spectra[c..end], 0.354_827_9 * mult);
            return;
        }
        _ => return,
    }
    for (unit, &scale) in (group_a_unit..total_units).zip(&scales) {
        scale_slice(
            &mut channel.spectra
                [QUANT_UNIT_TO_COEFF_INDEX[unit]..QUANT_UNIT_TO_COEFF_INDEX[unit + 1]],
            scale,
        );
    }
}

fn mirror_spectra(spectra: &mut [f64; 256], start: usize, end: usize) {
    let (source, dest) = spectra.split_at_mut(start);
    for (output, &input) in dest[..end - start].iter_mut().zip(source.iter().rev()) {
        *output = input;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // C deliberately reduces the nonnegative seed modulo 2^16.
fn add_noise(channel: &mut Channel, start: usize, end: usize) {
    if !channel.rng.initialized {
        let sf = &channel.scale_factors;
        let seed = (543 * (sf[8] + sf[12] + sf[15] + 1)) as u16;
        rng_init(&mut channel.rng, seed);
    }
    for value in &mut channel.spectra[start..end] {
        *value = f64::from(rng_next(&mut channel.rng)) / 65_535.0 * 2.0 - 1.0;
    }
}

fn rng_init(rng: &mut Rng, seed: u16) {
    let start = 0x4D93_u16.wrapping_mul(seed ^ (seed >> 14));
    rng.state_a = 3_u16.wrapping_sub(start);
    rng.state_b = 2_u16.wrapping_sub(start);
    rng.state_c = 1_u16.wrapping_sub(start);
    rng.state_d = 0_u16.wrapping_sub(start);
    rng.initialized = true;
}

fn rng_next(rng: &mut Rng) -> u16 {
    let t = rng.state_d ^ (rng.state_d << 5);
    rng.state_d = rng.state_c;
    rng.state_c = rng.state_b;
    rng.state_b = rng.state_a;
    rng.state_a = t ^ rng.state_a ^ ((t ^ (rng.state_a >> 5)) >> 4);
    rng.state_a
}

fn imdct(channel: &mut Channel, tables: &DspTables, scratch: &mut Scratch) {
    dct4(
        &channel.spectra[..tables.size],
        &mut scratch.output[..tables.size],
        tables,
        &mut scratch.temp,
    );
    let half = tables.size / 2;
    let (front, back) = scratch.output[..tables.size].split_at(half);
    let (pcm_front, pcm_back) = channel.pcm[..tables.size].split_at_mut(half);
    let (prev_front, prev_back) = channel.imdct_previous[..tables.size].split_at_mut(half);
    let (window_front, window_back) = tables.window[..tables.size].split_at(half);
    for (((pcm, &window), &dct), &previous) in pcm_front
        .iter_mut()
        .zip(window_front)
        .zip(back)
        .zip(prev_front.iter())
    {
        *pcm = window * dct + previous;
    }
    for (((pcm, &window), &dct), &previous) in pcm_back
        .iter_mut()
        .zip(window_back)
        .zip(back.iter().rev())
        .zip(prev_back.iter())
    {
        *pcm = window * -dct - previous;
    }
    for ((previous, &window), &dct) in prev_front
        .iter_mut()
        .zip(window_back.iter().rev())
        .zip(front.iter().rev())
    {
        *previous = window * -dct;
    }
    for ((previous, &window), &dct) in prev_back
        .iter_mut()
        .zip(window_front.iter().rev())
        .zip(front)
    {
        *previous = window * dct;
    }
}

fn dct4(input: &[f64], output: &mut [f64], tables: &DspTables, temp: &mut [f64; 256]) {
    // Equal-size pair iterators and the construction-time trig row cover every
    // active coefficient before any butterfly reads the workspace.
    for (((pair, front), back), trig) in temp[..tables.size]
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(input.as_chunks::<2>().0)
        .zip(input.as_chunks::<2>().0.iter().rev())
        .zip(tables.trig_row(tables.bits))
    {
        let a = front[0];
        let b = back[1];
        pair[0] = a * trig.cos + b * trig.sin;
        pair[1] = a * trig.sin - b * trig.cos;
    }
    for bits in (2..tables.bits).rev() {
        let block_size = 1 << bits;
        let trig = tables.trig_row(bits - 1);
        for block in temp[..tables.size].chunks_exact_mut(block_size * 2) {
            let (front, back) = block.split_at_mut(block_size);
            for ((front, back), trig) in front
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(back.as_chunks_mut::<2>().0)
                .zip(trig)
            {
                let a = front[0] - back[0];
                let b = front[1] - back[1];
                front[0] += back[0];
                front[1] += back[1];
                back[0] = a * trig.cos + b * trig.sin;
                back[1] = a * trig.sin - b * trig.cos;
            }
        }
    }
    // The last stage pairs neighbours with the single first-row coefficient.
    let trig = tables.trig_row(0)[0];
    for block in temp[..tables.size].as_chunks_mut::<4>().0 {
        let a = block[0] - block[2];
        let b = block[1] - block[3];
        block[0] += block[2];
        block[1] += block[3];
        block[2] = a * trig.cos + b * trig.sin;
        block[3] = a * trig.sin - b * trig.cos;
    }
    for (output, &index) in output.iter_mut().zip(&tables.shuffle) {
        *output = temp[usize::from(index)];
    }
}

/// Band-extension group layout from `LibAtrac9` `band_extension.c`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BexGroup {
    pub(crate) group_b_unit: usize,
    pub(crate) group_c_unit: usize,
    pub(crate) band_count: usize,
}

/// Band-extension groups for coded quantization-unit counts 13 through 20.
pub(crate) const BEX_GROUP_INFO: [BexGroup; 8] = [
    BexGroup {
        group_b_unit: 16,
        group_c_unit: 21,
        band_count: 0,
    },
    BexGroup {
        group_b_unit: 18,
        group_c_unit: 22,
        band_count: 1,
    },
    BexGroup {
        group_b_unit: 20,
        group_c_unit: 22,
        band_count: 2,
    },
    BexGroup {
        group_b_unit: 21,
        group_c_unit: 22,
        band_count: 3,
    },
    BexGroup {
        group_b_unit: 21,
        group_c_unit: 22,
        band_count: 3,
    },
    BexGroup {
        group_b_unit: 23,
        group_c_unit: 24,
        band_count: 4,
    },
    BexGroup {
        group_b_unit: 23,
        group_c_unit: 24,
        band_count: 4,
    },
    BexGroup {
        group_b_unit: 24,
        group_c_unit: 24,
        band_count: 5,
    },
];

/// Number of encoded extension values for each mode and group.
pub(crate) const BEX_ENCODED_VALUE_COUNTS: [[usize; 6]; 5] = [
    [0, 0, 0, 4, 4, 2],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 1],
    [0, 0, 0, 2, 2, 2],
    [1, 1, 1, 0, 0, 0],
];

/// Bit lengths of the encoded extension values for each mode and group.
pub(crate) const BEX_DATA_LENGTHS: [[[usize; 4]; 6]; 5] = [
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
const BEX_MODE0_BANDS3: [[f64; 32]; 5] = [
    [
        0.000_000e+0,
        1.988_220e-1,
        2.514_343e-1,
        2.960_510e-1,
        3.263_550e-1,
        3.771_362e-1,
        3.786_926e-1,
        4.540_405e-1,
        4.877_625e-1,
        5.262_451e-1,
        5.447_083e-1,
        5.737_000e-1,
        6.212_158e-1,
        6.222_839e-1,
        6.560_974e-1,
        6.896_667e-1,
        7.555_542e-1,
        7.677_917e-1,
        7.918_091e-1,
        7.971_497e-1,
        8.188_171e-1,
        8.446_045e-1,
        9.790_649e-1,
        9.822_083e-1,
        9.846_191e-1,
        9.859_314e-1,
        9.863_586e-1,
        9.863_892e-1,
        9.873_352e-1,
        9.881_287e-1,
        9.898_682e-1,
        9.913_330e-1,
    ],
    [
        0.000_000e+0,
        9.982_910e-1,
        7.592_773e-2,
        7.179_565e-1,
        9.851_379e-1,
        5.340_271e-1,
        9.013_672e-1,
        6.349_182e-1,
        7.226_257e-1,
        1.948_547e-1,
        7.628_174e-1,
        9.873_657e-1,
        8.112_183e-1,
        2.715_454e-1,
        9.734_192e-1,
        1.443_787e-1,
        4.640_198e-1,
        3.249_207e-1,
        3.790_894e-1,
        8.276_367e-2,
        5.954_590e-1,
        2.864_380e-1,
        9.806_824e-1,
        7.929_077e-1,
        6.292_114e-1,
        4.887_085e-1,
        2.905_273e-1,
        1.301_880e-1,
        3.140_869e-1,
        5.482_483e-1,
        4.210_815e-1,
        1.182_861e-1,
    ],
    [
        0.000_000e+0,
        3.155_518e-2,
        8.581_543e-2,
        1.364_746e-1,
        1.858_826e-1,
        2.368_469e-1,
        2.888_184e-1,
        3.432_617e-1,
        4.012_451e-1,
        4.623_108e-1,
        5.271_301e-1,
        5.954_895e-1,
        6.681_213e-1,
        7.448_425e-1,
        8.245_239e-1,
        9.097_290e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.000_000e+0,
        4.418_945e-2,
        1.303_711e-1,
        2.273_560e-1,
        3.395_996e-1,
        4.735_718e-1,
        6.267_090e-1,
        8.003_845e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.000_000e+0,
        2.804_565e-2,
        9.683_228e-2,
        1.849_976e-1,
        3.005_981e-1,
        4.470_520e-1,
        6.168_518e-1,
        8.007_813e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
];
const BEX_MODE0_BANDS4: [[f64; 16]; 5] = [
    [
        0.000_000e+0,
        2.708_740e-1,
        3.479_614e-1,
        3.578_186e-1,
        5.083_618e-1,
        5.299_072e-1,
        5.819_092e-1,
        6.381_836e-1,
        7.276_917e-1,
        7.595_520e-1,
        7.878_723e-1,
        9.707_336e-1,
        9.713_135e-1,
        9.736_023e-1,
        9.759_827e-1,
        9.832_458e-1,
    ],
    [
        0.000_000e+0,
        2.330_627e-1,
        5.891_418e-1,
        7.170_410e-1,
        2.036_438e-1,
        1.613_464e-1,
        6.668_701e-1,
        9.481_201e-1,
        9.769_897e-1,
        5.111_694e-1,
        3.522_644e-1,
        8.209_534e-1,
        2.933_960e-1,
        9.757_690e-1,
        5.289_917e-1,
        4.372_253e-1,
    ],
    [
        0.000_000e+0,
        4.360_962e-2,
        1.056_519e-1,
        1.590_576e-1,
        2.078_857e-1,
        2.572_937e-1,
        3.082_581e-1,
        3.616_028e-1,
        4.191_589e-1,
        4.792_175e-1,
        5.438_538e-1,
        6.125_183e-1,
        6.841_125e-1,
        7.589_417e-1,
        8.365_173e-1,
        9.148_254e-1,
    ],
    [
        0.000_000e+0,
        4.074_097e-2,
        1.164_551e-1,
        2.077_026e-1,
        3.184_509e-1,
        4.532_166e-1,
        6.124_268e-1,
        7.932_129e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.000_000e+0,
        8.880_615e-3,
        2.932_739e-2,
        5.593_872e-2,
        8.825_684e-2,
        1.259_155e-1,
        1.721_497e-1,
        2.270_813e-1,
        2.901_611e-1,
        3.579_712e-1,
        4.334_106e-1,
        5.147_095e-1,
        6.023_254e-1,
        6.956_177e-1,
        7.952_881e-1,
        8.977_356e-1,
    ],
];
const BEX_MODE0_BANDS5: [[f64; 32]; 3] = [
    [
        0.000_000e+0,
        7.379_150e-2,
        1.806_335e-1,
        2.687_073e-1,
        3.407_898e-1,
        4.047_546e-1,
        4.621_887e-1,
        5.168_762e-1,
        5.703_125e-1,
        6.237_488e-1,
        6.763_611e-1,
        7.288_208e-1,
        7.808_533e-1,
        8.337_708e-1,
        8.874_512e-1,
        9.418_030e-1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    [
        0.000_000e+0,
        7.980_347e-2,
        1.615_295e-1,
        1.665_649e-1,
        1.822_205e-1,
        2.185_669e-1,
        2.292_175e-1,
        2.456_665e-1,
        2.666_321e-1,
        3.306_580e-1,
        3.330_688e-1,
        3.765_259e-1,
        4.085_083e-1,
        4.400_024e-1,
        4.407_654e-1,
        4.817_505e-1,
        4.924_011e-1,
        5.320_740e-1,
        5.893_860e-1,
        6.131_287e-1,
        6.212_463e-1,
        6.278_076e-1,
        6.308_899e-1,
        7.660_828e-1,
        7.850_647e-1,
        7.910_461e-1,
        7.929_382e-1,
        8.038_330e-1,
        9.834_900e-1,
        9.846_191e-1,
        9.852_295e-1,
        9.862_671e-1,
    ],
    [
        0.000_000e+0,
        6.084_290e-1,
        3.672_791e-1,
        3.151_855e-1,
        1.488_953e-1,
        2.571_716e-1,
        5.103_455e-1,
        3.311_157e-1,
        5.426_025e-2,
        4.254_456e-1,
        7.998_352e-1,
        7.873_230e-1,
        5.418_701e-1,
        2.925_110e-1,
        8.468_628e-2,
        1.410_522e-1,
        9.819_641e-1,
        9.609_070e-1,
        3.530_884e-2,
        9.729_004e-2,
        5.758_362e-1,
        9.941_711e-1,
        7.215_576e-1,
        7.183_228e-1,
        2.028_809e-1,
        9.588_623e-2,
        2.032_166e-1,
        1.338_806e-1,
        5.003_357e-1,
        1.874_390e-1,
        9.804_993e-1,
        1.107_788e-1,
    ],
];
const BEX_MODE2_SCALE: [f64; 64] = [
    4.272_461e-4,
    1.312_256e-3,
    2.441_406e-3,
    3.692_627e-3,
    4.913_330e-3,
    6.134_033e-3,
    7.507_324e-3,
    8.972_168e-3,
    1.049_805e-2,
    1.223_755e-2,
    1.406_860e-2,
    1.599_121e-2,
    1.800_537e-2,
    2.026_367e-2,
    2.264_404e-2,
    2.517_700e-2,
    2.792_358e-2,
    3.073_120e-2,
    3.344_727e-2,
    3.631_592e-2,
    3.952_026e-2,
    4.275_513e-2,
    4.608_154e-2,
    4.968_262e-2,
    5.355_835e-2,
    5.783_081e-2,
    6.195_068e-2,
    6.677_246e-2,
    7.196_045e-2,
    7.745_361e-2,
    8.319_092e-2,
    8.993_530e-2,
    9.759_521e-2,
    1.056_213e-1,
    1.138_916e-1,
    1.236_267e-1,
    1.348_267e-1,
    1.470_337e-1,
    1.603_394e-1,
    1.755_676e-1,
    1.905_823e-1,
    2.071_228e-1,
    2.245_178e-1,
    2.444_153e-1,
    2.658_997e-1,
    2.897_644e-1,
    3.146_057e-1,
    3.450_012e-1,
    3.766_174e-1,
    4.122_620e-1,
    4.505_615e-1,
    4.893_799e-1,
    5.305_481e-1,
    5.731_201e-1,
    6.157_837e-1,
    6.580_811e-1,
    6.985_168e-1,
    7.435_303e-1,
    7.865_906e-1,
    8.302_612e-1,
    8.718_567e-1,
    9.125_671e-1,
    9.575_806e-1,
    9.996_643e-1,
];
const BEX_MODE3_INITIAL: [f64; 16] = [
    3.491_211e-1,
    5.371_094e-1,
    6.782_227e-1,
    7.910_156e-1,
    9.057_617e-1,
    1.024_902e+0,
    1.156_250e+0,
    1.290_527e+0,
    1.458_984e+0,
    1.664_551e+0,
    1.929_688e+0,
    2.278_320e+0,
    2.831_543e+0,
    3.659_180e+0,
    5.257_813e+0,
    8.373_047e+0,
];
const BEX_MODE3_RATE: [f64; 16] = [
    -2.913_818e-1,
    -2.541_504e-1,
    -1.664_429e-1,
    -1.476_440e-1,
    -1.342_163e-1,
    -1.220_703e-1,
    -1.117_554e-1,
    -1.026_611e-1,
    -9.436_035e-2,
    -8.483_887e-2,
    -7.476_807e-2,
    -6.304_932e-2,
    -4.492_188e-2,
    -2.447_510e-2,
    1.831_055e-4,
    4.174_805e-2,
];
const BEX_MODE4_MULTIPLIER: [f64; 8] = [
    3.610_229e-2,
    1.260_681e-1,
    2.227_478e-1,
    3.338_318e-1,
    4.662_170e-1,
    6.221_313e-1,
    7.989_197e-1,
    9.939_575e-1,
];

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)] // Exact sign/copy/zero invariants; synthetic sample indices <=256.
mod tests {
    use super::*;

    #[test]
    fn rng_zero_seed_sequence_and_wrap() {
        let mut rng = Rng::default();
        rng_init(&mut rng, 0);
        let expected = [
            3, 32, 102, 3, 102, 1028, 2154, 2059, 1123, 34989, 36241, 33834,
        ];
        for value in expected {
            assert_eq!(rng_next(&mut rng), value);
        }
        rng_init(&mut rng, u16::MAX);
        assert!(rng.initialized);
        assert_eq!(rng.state_a.wrapping_sub(rng.state_d), 3);
    }

    #[test]
    fn dct_matches_direct_cosine_definition_for_all_frame_sizes() {
        for bits in 6..=8 {
            let tables = DspTables::new(bits);
            let mut input = [0.0; 256];
            for (i, sample) in input[..tables.size].iter_mut().enumerate() {
                *sample = (i as f64 * 0.314).sin() + (i as f64 * 0.101).cos();
            }
            let mut output = [0.0; 256];
            let mut scratch = Scratch::new();
            dct4(
                &input[..tables.size],
                &mut output[..tables.size],
                &tables,
                &mut scratch.temp,
            );
            for (k, &actual) in output[..tables.size].iter().enumerate() {
                let expected = input[..tables.size]
                    .iter()
                    .enumerate()
                    .map(|(n, &sample)| {
                        sample
                            * (std::f64::consts::PI / tables.size as f64
                                * (n as f64 + 0.5)
                                * (k as f64 + 0.5))
                                .cos()
                    })
                    .sum::<f64>();
                assert!(
                    (actual - expected).abs() < 1.0e-10,
                    "size {} bin {k}: {actual} != {expected}",
                    tables.size
                );
            }
        }
    }

    #[test]
    fn shared_tables_are_built_once_and_match_fresh_tables_bit_exactly() {
        assert!(DspTables::built(5).is_none());
        assert!(DspTables::built(9).is_none());
        for bits in [6, 8, 7] {
            let shared = DspTables::shared_up_to(bits);
            assert!(std::ptr::eq(shared, DspTables::shared_up_to(bits)));
            for smaller in MIN_BITS..=bits {
                assert!(DspTables::built(smaller).is_some());
            }
            let expected = DspTables::new(bits);
            assert_eq!((shared.bits, shared.size), (bits, 1 << bits));
            assert_eq!(shared.shuffle, expected.shuffle);
            assert_eq!(shared.trig.len(), (1 << bits) * 3 / 2 - 1);
            assert!(
                shared
                    .window
                    .iter()
                    .zip(&expected.window)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(shared.trig.iter().zip(&expected.trig).all(|(a, b)| {
                a.sin.to_bits() == b.sin.to_bits() && a.cos.to_bits() == b.cos.to_bits()
            }));
        }
    }

    #[test]
    fn intensity_stereo_copies_primary_channel_with_per_unit_sign() {
        let mut block = Block::new(BlockType::Stereo);
        block.primary_channel_index = 1;
        block.stereo_quantization_unit = 1;
        block.quantization_unit_count = 3;
        block.joint_stereo_signs[2] = true;
        block.channels[1].spectra[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        intensity_stereo(&mut block);
        assert_eq!(
            &block.channels[0].spectra[..6],
            &[0.0, 0.0, 3.0, 4.0, -5.0, -6.0]
        );
        assert_eq!(
            &block.channels[1].spectra[..6],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn imdct_overlap_survives_one_silent_frame() {
        let tables = DspTables::new(6);
        let mut scratch = Scratch::new();
        let mut channel = Channel::new();
        channel.spectra[0] = 1.0;
        imdct(&mut channel, &tables, &mut scratch);
        let previous = channel.imdct_previous;
        assert!(previous[..tables.size].iter().any(|&value| value != 0.0));
        channel.spectra.fill(0.0);
        imdct(&mut channel, &tables, &mut scratch);
        let half = tables.size / 2;
        assert_eq!(&channel.pcm[..half], &previous[..half]);
        for (&sample, &previous) in channel.pcm[half..tables.size]
            .iter()
            .zip(&previous[half..tables.size])
        {
            assert_eq!(sample, -previous);
        }
        assert!(
            channel.imdct_previous[..tables.size]
                .iter()
                .all(|&value| value == 0.0)
        );
    }

    #[test]
    fn reused_scratch_overwrites_every_active_coefficient_before_reading() {
        let mut scratch = Scratch::new();
        for bits in [8, 6, 7, 8, 6] {
            let tables = DspTables::new(bits);
            scratch.output.fill(f64::NAN);
            scratch.temp.fill(f64::NAN);
            let mut channel = Channel::new();
            imdct(&mut channel, &tables, &mut scratch);
            assert!(channel.pcm[..tables.size].iter().all(|&value| value == 0.0));
            assert!(
                channel.imdct_previous[..tables.size]
                    .iter()
                    .all(|&value| value == 0.0)
            );
            assert!(
                scratch.output[..tables.size]
                    .iter()
                    .all(|&value| value == 0.0)
            );
            assert!(
                scratch.temp[..tables.size]
                    .iter()
                    .all(|&value| value == 0.0)
            );
        }
    }

    #[test]
    fn band_extension_supported_groups_and_modes_are_finite_and_bounded() {
        let tables = DspTables::new(8);
        for (index, group) in BEX_GROUP_INFO.iter().enumerate() {
            let units = index + 13;
            let first_mode = if group.band_count > 2 { 0 } else { 4 };
            let last_mode = if group.band_count > 2 { 3 } else { 4 };
            for (mode, lengths) in BEX_DATA_LENGTHS
                .iter()
                .enumerate()
                .take(last_mode + 1)
                .skip(first_mode)
            {
                let mut channel = Channel::new();
                channel.scale_factors.fill(15);
                channel.spectra.fill(123.0);
                channel.bex_mode = mode;
                for (value, &width) in channel
                    .bex_values
                    .iter_mut()
                    .zip(&lengths[group.band_count])
                {
                    *value = (1 << width) - 1;
                }
                band_extension(&mut channel, units, &tables);
                let start = QUANT_UNIT_TO_COEFF_INDEX[units];
                let end = QUANT_UNIT_TO_COEFF_INDEX[group.group_c_unit.max(22)];
                assert!(
                    channel.spectra[start..end]
                        .iter()
                        .all(|value| value.is_finite())
                );
                assert!(channel.spectra[..start].iter().all(|&value| value == 123.0));
                assert!(channel.spectra[end..].iter().all(|&value| value == 123.0));
            }
        }
    }
}
