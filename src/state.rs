//! Fixed decoder state; all memory is reserved before decoding.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockType {
    Mono,
    Stereo,
    Lfe,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rng {
    pub(crate) initialized: bool,
    pub(crate) state_a: u16,
    pub(crate) state_b: u16,
    pub(crate) state_c: u16,
    pub(crate) state_d: u16,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Channel {
    pub(crate) pcm: [f64; 256],
    pub(crate) spectra: [f64; 256],
    pub(crate) imdct_previous: [f64; 256],
    pub(crate) coded_quant_units: usize,
    pub(crate) scale_factor_coding_mode: usize,
    pub(crate) scale_factors: [i32; 31],
    pub(crate) scale_factors_prev: [i32; 31],
    pub(crate) precisions: [i32; 30],
    pub(crate) precisions_fine: [i32; 30],
    pub(crate) precision_mask: [i32; 30],
    pub(crate) codebook_set: [usize; 30],
    pub(crate) quantized_spectra: [i32; 256],
    pub(crate) quantized_spectra_fine: [i32; 256],
    pub(crate) bex_mode: usize,
    pub(crate) bex_value_count: usize,
    pub(crate) bex_values: [usize; 4],
    pub(crate) rng: Rng,
}

impl Channel {
    /// Restores the state of [`Self::new`] in place. The exhaustive pattern
    /// makes a new field a compile error until it is reset here too.
    fn reset(&mut self) {
        let Self {
            pcm,
            spectra,
            imdct_previous,
            coded_quant_units,
            scale_factor_coding_mode,
            scale_factors,
            scale_factors_prev,
            precisions,
            precisions_fine,
            precision_mask,
            codebook_set,
            quantized_spectra,
            quantized_spectra_fine,
            bex_mode,
            bex_value_count,
            bex_values,
            rng,
        } = self;
        pcm.fill(0.0);
        spectra.fill(0.0);
        imdct_previous.fill(0.0);
        *coded_quant_units = 0;
        *scale_factor_coding_mode = 0;
        scale_factors.fill(0);
        scale_factors_prev.fill(0);
        precisions.fill(0);
        precisions_fine.fill(0);
        precision_mask.fill(0);
        codebook_set.fill(0);
        quantized_spectra.fill(0);
        quantized_spectra_fine.fill(0);
        *bex_mode = 0;
        *bex_value_count = 0;
        bex_values.fill(0);
        *rng = Rng::default();
    }

    pub(crate) const fn new() -> Self {
        Self {
            pcm: [0.0; 256],
            spectra: [0.0; 256],
            imdct_previous: [0.0; 256],
            coded_quant_units: 0,
            scale_factor_coding_mode: 0,
            scale_factors: [0; 31],
            scale_factors_prev: [0; 31],
            precisions: [0; 30],
            precisions_fine: [0; 30],
            precision_mask: [0; 30],
            codebook_set: [0; 30],
            quantized_spectra: [0; 256],
            quantized_spectra_fine: [0; 256],
            bex_mode: 0,
            bex_value_count: 0,
            bex_values: [0; 4],
            rng: Rng {
                initialized: false,
                state_a: 0,
                state_b: 0,
                state_c: 0,
                state_d: 0,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
// Flags are independent bitstream fields, rather than interchangeable states.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Block {
    #[allow(clippy::struct_field_names)]
    pub(crate) block_type: BlockType,
    pub(crate) channels: [Channel; 2],
    pub(crate) channel_count: usize,
    pub(crate) first_in_superframe: bool,
    pub(crate) reuse_band_params: bool,
    pub(crate) band_count: usize,
    pub(crate) stereo_band: usize,
    pub(crate) extension_band: usize,
    pub(crate) quantization_unit_count: usize,
    pub(crate) stereo_quantization_unit: usize,
    pub(crate) extension_unit: usize,
    pub(crate) quantization_units_prev: usize,
    // The bitstream permits gradient endpoints up to 47, beyond the 30 coded units.
    pub(crate) gradient: [i32; 48],
    pub(crate) gradient_mode: usize,
    pub(crate) gradient_start_unit: usize,
    pub(crate) gradient_start_value: i32,
    pub(crate) gradient_end_unit: usize,
    pub(crate) gradient_end_value: i32,
    pub(crate) gradient_boundary: usize,
    pub(crate) primary_channel_index: usize,
    pub(crate) has_joint_stereo_signs: bool,
    pub(crate) joint_stereo_signs: [bool; 30],
    pub(crate) band_extension_enabled: bool,
    pub(crate) has_extension_data: bool,
    pub(crate) bex_data_length: usize,
    pub(crate) bex_mode: usize,
}

impl Block {
    /// Builds a block once, at decoder construction; [`Self::reset`] reuses it.
    #[allow(clippy::large_stack_arrays)]
    pub(crate) const fn new(block_type: BlockType) -> Self {
        Self {
            block_type,
            channels: [Channel::new(); 2],
            channel_count: if matches!(block_type, BlockType::Stereo) {
                2
            } else {
                1
            },
            first_in_superframe: false,
            reuse_band_params: false,
            band_count: 0,
            stereo_band: 0,
            extension_band: 0,
            quantization_unit_count: 0,
            stereo_quantization_unit: 0,
            extension_unit: 0,
            quantization_units_prev: 0,
            gradient: [0; 48],
            gradient_mode: 0,
            gradient_start_unit: 0,
            gradient_start_value: 0,
            gradient_end_unit: 0,
            gradient_end_value: 0,
            gradient_boundary: 0,
            primary_channel_index: 0,
            has_joint_stereo_signs: false,
            joint_stereo_signs: [false; 30],
            band_extension_enabled: false,
            has_extension_data: false,
            bex_data_length: 0,
            bex_mode: 0,
        }
    }

    /// Restores the state of [`Self::new`] in place, without building an
    /// 18 KiB block on the stack. The exhaustive pattern makes a new field a
    /// compile error until it is reset here too.
    pub(crate) fn reset(&mut self, kind: BlockType) {
        let Self {
            block_type,
            channels,
            channel_count,
            first_in_superframe,
            reuse_band_params,
            band_count,
            stereo_band,
            extension_band,
            quantization_unit_count,
            stereo_quantization_unit,
            extension_unit,
            quantization_units_prev,
            gradient,
            gradient_mode,
            gradient_start_unit,
            gradient_start_value,
            gradient_end_unit,
            gradient_end_value,
            gradient_boundary,
            primary_channel_index,
            has_joint_stereo_signs,
            joint_stereo_signs,
            band_extension_enabled,
            has_extension_data,
            bex_data_length,
            bex_mode,
        } = self;
        *block_type = kind;
        for channel in channels {
            channel.reset();
        }
        *channel_count = if matches!(kind, BlockType::Stereo) {
            2
        } else {
            1
        };
        *first_in_superframe = false;
        *reuse_band_params = false;
        *band_count = 0;
        *stereo_band = 0;
        *extension_band = 0;
        *quantization_unit_count = 0;
        *stereo_quantization_unit = 0;
        *extension_unit = 0;
        *quantization_units_prev = 0;
        gradient.fill(0);
        *gradient_mode = 0;
        *gradient_start_unit = 0;
        *gradient_start_value = 0;
        *gradient_end_unit = 0;
        *gradient_end_value = 0;
        *gradient_boundary = 0;
        *primary_channel_index = 0;
        *has_joint_stereo_signs = false;
        joint_stereo_signs.fill(false);
        *band_extension_enabled = false;
        *has_extension_data = false;
        *bex_data_length = 0;
        *bex_mode = 0;
    }
}
