//! Checked ATRAC9 frame unpacking, scale factors and precision allocation.
// Field widths and arithmetic below are bounded by the format (at most 6 bits).
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use crate::bits::BitReader;
use crate::dsp::{BEX_DATA_LENGTHS, BEX_ENCODED_VALUE_COUNTS, BEX_GROUP_INFO};
use crate::huffman::{Codebook, scale_factor_signed, scale_factor_unsigned, spectrum};
use crate::state::{Block, BlockType, Channel};
use crate::tables::{
    BAND_TO_QUANT_UNIT_COUNT, MAX_BAND_COUNT, MAX_EXTENSION_BAND, MAX_HUFF_PRECISION,
    MIN_BAND_COUNT, QUANT_UNIT_TO_CODEBOOK_INDEX, QUANT_UNIT_TO_COEFF_COUNT,
    QUANT_UNIT_TO_COEFF_INDEX, SCALE_FACTOR_WEIGHTS,
};
use crate::{Atrac9Config, Error};

pub(crate) fn unpack_frame(
    config: &Atrac9Config,
    blocks: &mut [Block],
    index: usize,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    for block in blocks {
        block.first_in_superframe = br.read(1)? == 0;
        block.reuse_band_params = br.read(1)? != 0;
        if block.first_in_superframe && index != 0 {
            return invalid("first-superframe flag in later frame");
        }
        if block.first_in_superframe
            && block.reuse_band_params
            && block.block_type != BlockType::Lfe
        {
            return invalid("band parameters reused at superframe start");
        }
        if block.block_type == BlockType::Lfe {
            unpack_lfe(block, br)?;
        } else {
            unpack_standard(config, block, br)?;
        }
        br.align()?;
    }
    Ok(())
}

#[cold]
fn invalid<T>(message: &'static str) -> Result<T, Error> {
    Err(Error::InvalidData(message))
}

fn unpack_standard(
    config: &Atrac9Config,
    block: &mut Block,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    if !block.reuse_band_params {
        read_band_params(config, block, br)?;
    }
    // Reuse before any band declaration cannot describe a valid standard block.
    if block.quantization_unit_count == 0 {
        return invalid("band parameters have not been initialized");
    }
    read_gradient_params(block, br)?;
    create_gradient(block);
    read_stereo_params(block, br)?;
    read_extension_params(block, br)?;
    for index in 0..block.channel_count {
        let context = ScaleFactorContext {
            count: block.extension_unit,
            first_in_superframe: block.first_in_superframe,
            previous_count: block.quantization_units_prev,
        };
        let (primary, secondary) = block.channels.split_at_mut(1);
        let channel = if index == 0 {
            read_scale_factors(&mut primary[0], 0, &[0; 31], context, br)?;
            &mut primary[0]
        } else {
            read_scale_factors(&mut secondary[0], 1, &primary[0].scale_factors, context, br)?;
            &mut secondary[0]
        };
        channel.coded_quant_units = if index == block.primary_channel_index {
            block.quantization_unit_count
        } else {
            block.stereo_quantization_unit
        };
        calculate_precisions(
            channel,
            &block.gradient,
            block.gradient_mode,
            block.gradient_boundary,
            block.quantization_unit_count,
        )?;
        calculate_codebook_set(config, channel);
        read_spectra(config, channel, br)?;
    }
    block.quantization_units_prev = if block.band_extension_enabled {
        block.extension_unit
    } else {
        block.quantization_unit_count
    };
    Ok(())
}

fn read_band_params(
    config: &Atrac9Config,
    block: &mut Block,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    let rate = usize::from(config.high_sample_rate);
    let minimum = MIN_BAND_COUNT[rate];
    block.band_count = br.read(4)? as usize + minimum;
    if block.band_count > MAX_BAND_COUNT[config.sample_rate_index] {
        return invalid("band count exceeds sample-rate limit");
    }
    block.quantization_unit_count = BAND_TO_QUANT_UNIT_COUNT[block.band_count];
    block.stereo_band = if block.block_type == BlockType::Stereo {
        br.read(4)? as usize + minimum
    } else {
        block.band_count
    };
    if block.stereo_band > block.band_count {
        return invalid("stereo band exceeds coded band");
    }
    block.stereo_quantization_unit = BAND_TO_QUANT_UNIT_COUNT[block.stereo_band];
    block.band_extension_enabled = br.read(1)? != 0;
    block.extension_band = if block.band_extension_enabled {
        br.read(4)? as usize + minimum
    } else {
        block.band_count
    };
    if block.band_extension_enabled
        && (block.extension_band < block.band_count
            || block.extension_band > MAX_EXTENSION_BAND[rate])
    {
        return invalid("invalid extension band");
    }
    block.extension_unit = BAND_TO_QUANT_UNIT_COUNT[block.extension_band];
    Ok(())
}

fn read_gradient_params(block: &mut Block, br: &mut BitReader<'_>) -> Result<(), Error> {
    block.gradient_mode = br.read(2)? as usize;
    if block.gradient_mode > 0 {
        block.gradient_end_unit = 31;
        block.gradient_end_value = 31;
        block.gradient_start_unit = br.read(5)? as usize;
        block.gradient_start_value = br.read(5)? as i32;
    } else {
        block.gradient_start_unit = br.read(6)? as usize;
        block.gradient_end_unit = br.read(6)? as usize + 1;
        block.gradient_start_value = br.read(5)? as i32;
        block.gradient_end_value = br.read(5)? as i32;
    }
    block.gradient_boundary = br.read(4)? as usize;
    if block.gradient_boundary > block.quantization_unit_count {
        return invalid("gradient boundary exceeds coded units");
    }
    if block.gradient_start_unit >= 48
        || block.gradient_end_unit >= 48
        || block.gradient_start_unit > block.gradient_end_unit
    {
        return invalid("invalid gradient unit range");
    }
    Ok(())
}

const BASE_CURVE: [u8; 48] = [
    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 26, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
];

const fn gradient_curves() -> [[u8; 48]; 48] {
    let mut curves = [[0; 48]; 48];
    let mut length = 1;
    while length <= 48 {
        let mut index = 0;
        while index < length {
            curves[length - 1][index] = BASE_CURVE[index * 48 / length];
            index += 1;
        }
        length += 1;
    }
    curves
}

const GRADIENT_CURVES: [[u8; 48]; 48] = gradient_curves();

fn create_gradient(block: &mut Block) {
    block.gradient[..block.gradient_end_unit].fill(block.gradient_start_value);
    if block.gradient_end_unit <= block.quantization_unit_count {
        block.gradient[block.gradient_end_unit..=block.quantization_unit_count]
            .fill(block.gradient_end_value);
    }
    let count = block.gradient_end_unit - block.gradient_start_unit;
    let difference = block.gradient_end_value - block.gradient_start_value;
    if count == 0 || difference == 0 {
        return;
    }
    let curve = &GRADIENT_CURVES[count - 1];
    // Match LibAtrac9's double multiply followed by truncation toward zero.
    let scale = f64::from(difference.abs() - 1) / 31.0;
    let direction = difference.signum();
    let base = block.gradient_start_value + direction;
    for (value, &point) in block.gradient[block.gradient_start_unit..block.gradient_end_unit]
        .iter_mut()
        .zip(curve)
    {
        *value = base + direction * (f64::from(point) * scale) as i32;
    }
}

fn read_stereo_params(block: &mut Block, br: &mut BitReader<'_>) -> Result<(), Error> {
    if block.block_type != BlockType::Stereo {
        return Ok(());
    }
    block.primary_channel_index = br.read(1)? as usize;
    block.has_joint_stereo_signs = br.read(1)? != 0;
    if block.has_joint_stereo_signs {
        for sign in &mut block.joint_stereo_signs
            [block.stereo_quantization_unit..block.quantization_unit_count]
        {
            *sign = br.read(1)? != 0;
        }
    } else {
        block.joint_stereo_signs.fill(false);
    }
    Ok(())
}

fn read_bex_header(
    channel: &mut Channel,
    band: usize,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    let mode = br.read(2)? as usize;
    channel.bex_mode = if band > 2 { mode } else { 4 };
    channel.bex_value_count = BEX_ENCODED_VALUE_COUNTS[channel.bex_mode][band];
    Ok(())
}

fn read_bex_values(
    channel: &mut Channel,
    band: usize,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    for (value, &bits) in channel.bex_values[..channel.bex_value_count]
        .iter_mut()
        .zip(&BEX_DATA_LENGTHS[channel.bex_mode][band])
    {
        *value = br.read(bits)? as usize;
    }
    Ok(())
}

fn read_extension_params(block: &mut Block, br: &mut BitReader<'_>) -> Result<(), Error> {
    let band = if block.band_extension_enabled {
        let index = block
            .quantization_unit_count
            .checked_sub(13)
            .ok_or(Error::InvalidData("invalid band extension unit count"))?;
        let group = BEX_GROUP_INFO
            .get(index)
            .ok_or(Error::InvalidData("invalid band extension unit count"))?;
        if block.block_type == BlockType::Stereo {
            read_bex_header(&mut block.channels[1], group.band_count, br)?;
        } else {
            br.skip(1)?;
        }
        group.band_count
    } else {
        0
    };
    block.has_extension_data = br.read(1)? != 0;
    if !block.has_extension_data {
        return Ok(());
    }
    if !block.band_extension_enabled {
        block.bex_mode = br.read(2)? as usize;
        block.bex_data_length = br.read(5)? as usize;
        br.skip(block.bex_data_length)?;
        return Ok(());
    }
    read_bex_header(&mut block.channels[0], band, br)?;
    block.bex_data_length = br.read(5)? as usize;
    if block.bex_data_length == 0 {
        return validate_bex_values(block, band);
    }
    let start = br.position();
    read_bex_values(&mut block.channels[0], band, br)?;
    if block.block_type == BlockType::Stereo {
        read_bex_values(&mut block.channels[1], band, br)?;
    }
    if br.position() - start > block.bex_data_length {
        return invalid("extension values exceed declared data length");
    }
    validate_bex_values(block, band)
}

fn validate_bex_values(block: &Block, band: usize) -> Result<(), Error> {
    // A zero-length extension retains old values after changing modes. Validate
    // every table index consumed by DSP, including mode 2's retained second value.
    for channel in &block.channels[..block.channel_count] {
        let limits: &[usize] = match channel.bex_mode {
            0 => match band {
                3 => &[32, 16, 8, 8],
                4 => &[16, 16, 8, 16],
                5 => &[16, 32],
                _ => return invalid("invalid band extension mode"),
            },
            1 => &[],
            2 => &[64, 64],
            3 => &[16, 16],
            4 => &[8],
            _ => return invalid("invalid band extension mode"),
        };
        if channel
            .bex_values
            .iter()
            .zip(limits)
            .any(|(&value, &limit)| value >= limit)
        {
            return invalid("band extension value exceeds table size");
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ScaleFactorContext {
    count: usize,
    first_in_superframe: bool,
    previous_count: usize,
}

/// `baseline` holds the primary channel's current scale factors when reading
/// the secondary channel.
fn read_scale_factors(
    channel: &mut Channel,
    index: usize,
    baseline: &[i32; 31],
    context: ScaleFactorContext,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    channel.scale_factors.fill(0);
    let mode = br.read(2)? as usize;
    channel.scale_factor_coding_mode = mode;
    let count = context.count;
    match (index, mode) {
        (_, 0) => read_sf_delta(channel, count, br)?,
        (0, 1) => read_sf_raw(channel, count, br)?,
        (0, 2) | (1, 3) => {
            if context.first_in_superframe {
                return invalid("previous-frame scale factors used at superframe start");
            }
            read_sf_distance(
                &mut channel.scale_factors,
                count,
                &channel.scale_factors_prev,
                context.previous_count,
                br,
            )?;
        }
        (0, 3) => {
            if context.first_in_superframe {
                return invalid("previous-frame scale factors used at superframe start");
            }
            read_sf_delta_baseline(
                &mut channel.scale_factors,
                count,
                &channel.scale_factors_prev,
                context.previous_count,
                br,
            )?;
        }
        (1, 1) => read_sf_distance(&mut channel.scale_factors, count, baseline, count, br)?,
        (1, 2) => read_sf_delta_baseline(&mut channel.scale_factors, count, baseline, count, br)?,
        _ => return invalid("invalid scale factor mode"),
    }
    if channel.scale_factors[..count]
        .iter()
        .any(|&value| !(0..=31).contains(&value))
    {
        return invalid("scale factor out of range");
    }
    channel.scale_factors_prev = channel.scale_factors;
    Ok(())
}

fn read_sf_raw(channel: &mut Channel, count: usize, br: &mut BitReader<'_>) -> Result<(), Error> {
    let bits = br.read(2)? as usize + 2;
    let base = if bits < 5 { br.read(5)? as i32 } else { 0 };
    for value in &mut channel.scale_factors[..count] {
        *value = br.read(bits)? as i32 + base;
    }
    Ok(())
}

fn read_sf_delta(channel: &mut Channel, count: usize, br: &mut BitReader<'_>) -> Result<(), Error> {
    let weights = &SCALE_FACTOR_WEIGHTS[br.read(3)? as usize];
    let base = br.read(5)? as i32;
    let bits = br.read(2)? as usize + 3;
    let codebook = scale_factor_unsigned(bits);
    channel.scale_factors[0] = br.read(bits)? as i32;
    for index in 1..count {
        let delta = codebook.read_value(br, false)?;
        channel.scale_factors[index] =
            (channel.scale_factors[index - 1] + delta) & (codebook.value_max - 1);
    }
    for (value, &weight) in channel.scale_factors[..count].iter_mut().zip(weights) {
        *value += base - weight;
    }
    Ok(())
}

fn read_sf_distance(
    sf: &mut [i32; 31],
    count: usize,
    baseline: &[i32; 31],
    baseline_count: usize,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    let bits = br.read(2)? as usize + 2;
    let codebook = scale_factor_signed(bits);
    let shared = count.min(baseline_count);
    for (value, &base) in sf[..shared].iter_mut().zip(baseline) {
        *value = (base + codebook.read_value(br, true)?) & 31;
    }
    for value in &mut sf[shared..count] {
        *value = br.read(5)? as i32;
    }
    Ok(())
}

fn read_sf_delta_baseline(
    sf: &mut [i32; 31],
    count: usize,
    baseline: &[i32; 31],
    baseline_count: usize,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    let base = br.read_offset_binary(5)?;
    let bits = br.read(2)? as usize + 1;
    let codebook = scale_factor_unsigned(bits);
    let shared = count.min(baseline_count);
    sf[0] = br.read(bits)? as i32;
    for index in 1..shared {
        sf[index] = (sf[index - 1] + codebook.read_value(br, false)?) & (codebook.value_max - 1);
    }
    for (value, &previous) in sf[..shared].iter_mut().zip(baseline) {
        *value += base + previous;
    }
    for value in &mut sf[shared..count] {
        *value = br.read(5)? as i32;
    }
    Ok(())
}

fn calculate_precisions(
    channel: &mut Channel,
    gradient: &[i32; 48],
    mode: usize,
    boundary: usize,
    count: usize,
) -> Result<(), Error> {
    channel.precision_mask.fill(0);
    for index in 1..count {
        let difference = channel.scale_factors[index] - channel.scale_factors[index - 1];
        if difference > 1 {
            channel.precision_mask[index] += (difference - 1).min(5);
        } else if difference < -1 {
            channel.precision_mask[index - 1] += (-difference - 1).min(5);
        }
    }
    for (index, &gradient_value) in gradient[..count].iter().enumerate() {
        let mut precision = channel.scale_factors[index] - gradient_value;
        if mode != 0 {
            precision += channel.precision_mask[index];
            if precision > 0 {
                precision = match mode {
                    1 => precision / 2,
                    2 => 3 * precision / 8,
                    _ => precision / 4,
                };
            }
        }
        precision = precision.max(1) + i32::from(index < boundary);
        let fine = (precision - 15).max(0);
        if fine > 15 {
            return invalid("fine precision exceeds quantizer table");
        }
        channel.precisions[index] = precision.min(15);
        channel.precisions_fine[index] = fine;
    }
    Ok(())
}

fn calculate_codebook_set(config: &Atrac9Config, channel: &mut Channel) {
    channel.codebook_set.fill(0);
    let count = channel.coded_quant_units;
    if count <= 1 || config.high_sample_rate {
        return;
    }
    let sf = &mut channel.scale_factors;
    let saved = sf[count];
    sf[count] = sf[count - 1];
    let average = if count > 12 {
        (sf[..12].iter().sum::<i32>() + 6) / 12
    } else {
        0
    };
    for index in 8..count {
        let previous = sf[index - 1];
        let next = sf[index + 1];
        if sf[index] - previous.min(next) >= 3 || sf[index] - previous + sf[index] - next >= 3 {
            channel.codebook_set[index] = 1;
        }
    }
    for index in 12..count {
        if channel.codebook_set[index] == 0 {
            let minimum = sf[index - 1].min(sf[index + 1]);
            if sf[index] - minimum >= 2
                && sf[index] >= average - i32::from(QUANT_UNIT_TO_COEFF_COUNT[index] == 16)
            {
                channel.codebook_set[index] = 1;
            }
        }
    }
    sf[count] = saved;
}

// Precisions have been validated as positive, with fine precision in 0..=15.
// Every coefficient of a coded unit is written, and fine values for each unit
// with fine precision; dequantization reads nothing else.
#[allow(clippy::cast_sign_loss)]
fn read_spectra(
    config: &Atrac9Config,
    channel: &mut Channel,
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    let maximum = MAX_HUFF_PRECISION[usize::from(config.high_sample_rate)];
    for unit in 0..channel.coded_quant_units {
        let bits = channel.precisions[unit] as usize + 1;
        let start = QUANT_UNIT_TO_COEFF_INDEX[unit];
        let end = QUANT_UNIT_TO_COEFF_INDEX[unit + 1];
        let output = &mut channel.quantized_spectra[start..end];
        if bits <= maximum {
            let codebook = spectrum(
                channel.codebook_set[unit],
                bits,
                QUANT_UNIT_TO_CODEBOOK_INDEX[unit],
            );
            match (codebook.value_count_power, codebook.value_bits) {
                (2, 2) => read_packed_spectra::<4, 2>(codebook, output, br)?,
                (1, 2) => read_packed_spectra::<2, 2>(codebook, output, br)?,
                (1, 3) => read_packed_spectra::<2, 3>(codebook, output, br)?,
                (1, 4) => read_packed_spectra::<2, 4>(codebook, output, br)?,
                (0, 5) => read_packed_spectra::<1, 5>(codebook, output, br)?,
                (0, 6) => read_packed_spectra::<1, 6>(codebook, output, br)?,
                (0, 7) => read_packed_spectra::<1, 7>(codebook, output, br)?,
                _ => return invalid("invalid spectrum codebook layout"),
            }
        } else {
            br.read_signed_into(bits, output)?;
        }
    }
    for unit in 0..channel.coded_quant_units {
        let precision = channel.precisions_fine[unit];
        if precision > 0 {
            let start = QUANT_UNIT_TO_COEFF_INDEX[unit];
            let end = QUANT_UNIT_TO_COEFF_INDEX[unit + 1];
            br.read_signed_into(
                precision as usize + 1,
                &mut channel.quantized_spectra_fine[start..end],
            )?;
        }
    }
    Ok(())
}

fn read_packed_spectra<const COUNT: usize, const BITS: usize>(
    codebook: &Codebook,
    output: &mut [i32],
    br: &mut BitReader<'_>,
) -> Result<(), Error> {
    // Quant-unit sizes are divisible by every layout selected by the frozen
    // spectrum tables. Constant group sizes and sign shifts permit unrolling.
    let (groups, _) = output.as_chunks_mut::<COUNT>();
    for group in groups {
        let mut packed = codebook.read_value(br, false)?;
        for value in group {
            *value = ((packed & ((1 << BITS) - 1)) << (32 - BITS)) >> (32 - BITS);
            packed >>= BITS;
        }
    }
    Ok(())
}

fn unpack_lfe(block: &mut Block, br: &mut BitReader<'_>) -> Result<(), Error> {
    block.quantization_unit_count = 2;
    let channel = &mut block.channels[0];
    channel.coded_quant_units = 2;
    channel.scale_factors.fill(0);
    let precision = if block.reuse_band_params { 8 } else { 4 };
    for unit in 0..2 {
        channel.scale_factors[unit] = br.read(5)? as i32;
        channel.precisions[unit] = precision;
        channel.precisions_fine[unit] = 0;
    }
    br.read_signed_into(
        if block.reuse_band_params { 9 } else { 5 },
        &mut channel.quantized_spectra[..QUANT_UNIT_TO_COEFF_INDEX[2]],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use proptest::prelude::*;

    fn pack(fields: &[(u32, usize)]) -> Vec<u8> {
        let bit_count: usize = fields.iter().map(|field| field.1).sum();
        let mut output = vec![0; bit_count.div_ceil(8)];
        let mut position = 0;
        for &(value, width) in fields {
            for bit in (0..width).rev() {
                output[position / 8] |= (((value >> bit) & 1) as u8) << (7 - position % 8);
                position += 1;
            }
        }
        output
    }

    #[test]
    fn gradient_known_answers_and_wide_endpoints() {
        let mut block = Block::new(BlockType::Mono);
        block.quantization_unit_count = 16;
        block.gradient_start_unit = 2;
        block.gradient_end_unit = 6;
        block.gradient_start_value = 3;
        block.gradient_end_value = 10;
        create_gradient(&mut block);
        assert_eq!(&block.gradient[..8], &[3, 3, 4, 4, 7, 9, 10, 10]);
        block.gradient_start_value = 10;
        block.gradient_end_value = 3;
        create_gradient(&mut block);
        assert_eq!(&block.gradient[..8], &[10, 10, 9, 9, 6, 4, 3, 3]);
        block.gradient_start_unit = 40;
        block.gradient_end_unit = 47;
        create_gradient(&mut block);
        assert_eq!(block.gradient[39], 10);
        assert!((3..=10).contains(&block.gradient[46]));
    }

    #[test]
    fn rejects_fine_precision_outside_reference_table() {
        let mut channel = Channel::new();
        channel.scale_factors[0] = 31;
        assert!(calculate_precisions(&mut channel, &[0; 48], 0, 0, 1).is_err());
        channel.scale_factors[0] = 30;
        assert!(calculate_precisions(&mut channel, &[0; 48], 0, 1, 1).is_err());
        assert!(calculate_precisions(&mut channel, &[0; 48], 0, 0, 1).is_ok());
        assert_eq!(channel.precisions[0], 15);
        assert_eq!(channel.precisions_fine[0], 15);
    }

    #[test]
    fn extension_mode_change_validates_retained_values() {
        let mut block = Block::new(BlockType::Mono);
        block.band_extension_enabled = true;
        block.quantization_unit_count = 16;
        block.channels[0].bex_values[0] = 63;
        // Reserved mono bit, extension present, mode0, zero-length retained data.
        let bytes = pack(&[(0, 1), (1, 1), (0, 2), (0, 5)]);
        let mut br = BitReader::new(&bytes);
        assert!(read_extension_params(&mut block, &mut br).is_err());
        block.channels[0].bex_values[0] = 31;
        assert!(read_extension_params(&mut block, &mut BitReader::new(&bytes)).is_ok());
        // Mode2 has one encoded value in band5, but DSP consumes a retained second.
        block.channels[0].bex_mode = 2;
        block.channels[0].bex_values[1] = 64;
        assert!(validate_bex_values(&block, 5).is_err());
    }

    #[test]
    fn previous_frame_scale_factor_modes_require_history() {
        let context = ScaleFactorContext {
            count: 13,
            first_in_superframe: true,
            previous_count: 0,
        };
        for (index, mode) in [(0, 2), (0, 3), (1, 3)] {
            let bytes = pack(&[(mode, 2)]);
            let mut channel = Channel::new();
            assert!(
                read_scale_factors(
                    &mut channel,
                    index,
                    &[0; 31],
                    context,
                    &mut BitReader::new(&bytes)
                )
                .is_err()
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]
        #[test]
        fn arbitrary_frames_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..2049),
            sample_rate in 0u32..16, layout in 0u32..6, frame_index in 0usize..8) {
            let word = 0xfe00_0000 | (sample_rate << 20) | (layout << 17) | (2047 << 5) | (3 << 3);
            let config = Atrac9Config::parse(word.to_be_bytes()).unwrap();
            let mut blocks: Vec<_> = config.block_types().iter().copied().map(Block::new).collect();
            let _ = unpack_frame(&config, &mut blocks, frame_index, &mut BitReader::new(&bytes));
        }

        #[test]
        fn raw_scale_factors_round_trip(values in prop::collection::vec(0u32..32, 1..31)) {
            let mut fields = vec![(3, 2)]; // Five-bit CLC, no base offset.
            fields.extend(values.iter().map(|&value| (value, 5)));
            let bytes = pack(&fields);
            let mut channel = Channel::new();
            read_sf_raw(&mut channel, values.len(), &mut BitReader::new(&bytes)).unwrap();
            prop_assert_eq!(&channel.scale_factors[..values.len()],
                &values.iter().map(|&value| value as i32).collect::<Vec<_>>());
        }

        #[test]
        fn all_scale_factor_modes_handle_arbitrary_payloads(
            bytes in prop::collection::vec(any::<u8>(), 0..80),
            baseline in prop::collection::vec(0i32..32, 31),
            count in 1usize..31, previous_count in 0usize..31,
            first in any::<bool>(), channel_index in 0usize..2) {
            let context = ScaleFactorContext { count, first_in_superframe: first, previous_count };
            let mut channel = Channel::new();
            channel.scale_factors_prev.copy_from_slice(&baseline);
            let mut current_baseline = [0; 31];
            current_baseline.copy_from_slice(&baseline);
            if read_scale_factors(&mut channel, channel_index, &current_baseline, context,
                &mut BitReader::new(&bytes)).is_ok() {
                prop_assert!(channel.scale_factors[..count]
                    .iter().all(|value| (0..=31).contains(value)));
            }
        }

        #[test]
        fn band_extension_handles_arbitrary_payloads_and_retained_values(
            bytes in prop::collection::vec(any::<u8>(), 0..12),
            values in prop::collection::vec(0usize..256, 4),
            quantization_units in 13usize..21, stereo in any::<bool>()) {
            let mut block = Block::new(if stereo { BlockType::Stereo } else { BlockType::Mono });
            block.band_extension_enabled = true;
            block.quantization_unit_count = quantization_units;
            block.channels[0].bex_values.copy_from_slice(&values);
            block.channels[1].bex_values.copy_from_slice(&values);
            if read_extension_params(&mut block, &mut BitReader::new(&bytes)).is_ok()
                && block.has_extension_data {
                let band = BEX_GROUP_INFO[quantization_units - 13].band_count;
                prop_assert!(validate_bex_values(&block, band).is_ok());
            }
        }

        #[test]
        fn gradients_remain_in_endpoint_range(start in 0usize..48, end in 0usize..48,
            start_value in 0i32..32, end_value in 0i32..32) {
            let mut block = Block::new(BlockType::Mono);
            block.quantization_unit_count = 30;
            block.gradient_start_unit = start.min(end);
            block.gradient_end_unit = start.max(end);
            block.gradient_start_value = start_value;
            block.gradient_end_value = end_value;
            create_gradient(&mut block);
            for &value in &block.gradient[..=30] {
                prop_assert!(value >= start_value.min(end_value) && value <= start_value.max(end_value));
            }
        }
    }
}
