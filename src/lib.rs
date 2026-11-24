//! A safe, allocation-free ATRAC9 decoder.
//!
//! ATRAC9 is the audio codec used by PlayStation Vita and PlayStation 4 software. A
//! stream is described by a four-byte configuration word and stored as a sequence of
//! fixed-size superframes, each holding one, two, four or eight frames.
//!
//! This crate is a Rust port of Alex Barney's
//! [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9). Its output matches the C
//! library bit for bit, as both 16-bit and floating-point PCM.
//!
//! ```
//! use atrac9::{Atrac9Config, Decoder};
//!
//! fn decode(config_word: [u8; 4], audio: &[u8]) -> Result<Vec<i16>, atrac9::Error> {
//!     let config = Atrac9Config::parse(config_word)?;
//!     let mut decoder = Decoder::new(config);
//!     let mut buffer = vec![0; config.superframe_pcm_values()];
//!     let mut pcm = Vec::new();
//!     for superframe in audio.chunks_exact(config.superframe_bytes()) {
//!         let info = decoder.decode_superframe_i16(superframe, &mut buffer)?;
//!         pcm.extend_from_slice(&buffer[..info.pcm_values_written]);
//!     }
//!     Ok(pcm)
//! }
//! ```
//!
//! [`Decoder::new`] reserves all state up front. Decoding and [`Decoder::reset`] never
//! allocate or take a lock, so a decoder can live on an audio thread.
#![forbid(unsafe_code)]

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

mod bits;
mod config;
mod dsp;
mod huffman;
mod state;
mod tables;
mod unpack;

pub use config::Atrac9Config;
use dsp::{DspTables, Scratch};
use state::Block;

/// Invalid configuration, encoded data, or caller buffer/state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Configuration header or channel layout is unsupported.
    InvalidConfig,
    /// Encoded input ends before a complete frame or superframe.
    Truncated,
    /// Encoded fields violate a format bound.
    InvalidData(&'static str),
    /// Caller PCM buffer is too small; no state or output was changed.
    OutputTooSmall {
        /// Required number of interleaved values.
        required: usize,
        /// Available number of interleaved values.
        available: usize,
    },
    /// A prior encoded-data failure requires a reset before continuing.
    NeedsReset,
    /// Superframe decoding must begin at a superframe boundary.
    NotSuperframeBoundary,
    /// A configuration needs more storage than this decoder reserved.
    InsufficientCapacity,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid ATRAC9 configuration"),
            Self::Truncated => f.write_str("truncated ATRAC9 input"),
            Self::InvalidData(reason) => write!(f, "invalid ATRAC9 data: {reason}"),
            Self::OutputTooSmall {
                required,
                available,
            } => write!(f, "PCM buffer requires {required} values, got {available}"),
            Self::NeedsReset => f.write_str("decoder requires reset after an encoded-data error"),
            Self::NotSuperframeBoundary => f.write_str("decoder is partway through a superframe"),
            Self::InsufficientCapacity => {
                f.write_str("ATRAC9 configuration exceeds decoder capacity; use Decoder::new_max")
            }
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    #[cold]
    pub(crate) const fn invalid(reason: &'static str) -> Self {
        Self::InvalidData(reason)
    }
}

/// What one decode call consumed and produced. Sample counts are per channel;
/// PCM buffer sizes count individual interleaved values.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeInfo {
    /// Encoded bytes consumed. Frame calls exclude end-of-superframe padding.
    pub bytes_consumed: usize,
    /// Declared padding to skip after a final frame call; it need not be present
    /// in that call's input. Superframe calls include it in `bytes_consumed`.
    pub padding_bytes: usize,
    /// Complete codec frames decoded.
    pub frames_consumed: usize,
    /// Samples per channel actually written after gapless trimming.
    pub samples_written: usize,
    /// Individual values written into the interleaved PCM buffer.
    pub pcm_values_written: usize,
    /// Samples per channel omitted for encoder delay or the configured tail.
    pub skipped_samples: usize,
    /// Remaining encoder delay, supplied by the caller rather than inferred
    /// from the four-byte configuration.
    pub encoder_delay_remaining: usize,
}

/// Stateful, safe Rust ATRAC9 decoder. Construction allocates fixed state and
/// builds the process-wide transform tables on first use; decode, reset and
/// reconfigure never allocate or acquire a lock.
///
/// Frames occupy variable byte lengths inside a fixed-size superframe. Frame
/// callers advance by `bytes_consumed`, adding `padding_bytes` on the final
/// frame. Superframe callers advance by `bytes_consumed` alone.
///
/// A frame unpacking error poisons the stream until [`Self::reset`]. A superframe
/// call given fewer than [`Atrac9Config::superframe_bytes`] bytes returns
/// [`Error::Truncated`] without decoding or poisoning anything. An error in a
/// later frame of a superframe poisons the decoder and restores
/// [`Self::gapless_state`] to its value before the call; the output buffer can
/// hold an already-decoded prefix, which callers must discard.
#[derive(Debug)]
pub struct Decoder {
    config: Atrac9Config,
    blocks: Box<[Block]>,
    tables: &'static DspTables,
    transform_capacity: usize,
    scratch: Box<Scratch>,
    index: usize,
    superframe_remaining: usize,
    poisoned: bool,
    initial_delay: usize,
    initial_length: Option<usize>,
    delay: usize,
    length: Option<usize>,
}

impl Decoder {
    /// Reserve all state for this configuration. There are no per-frame heap
    /// allocations, including on errors.
    #[must_use]
    pub fn new(config: Atrac9Config) -> Self {
        Self {
            config,
            blocks: config
                .block_types()
                .iter()
                .copied()
                .map(Block::new)
                .collect(),
            tables: DspTables::shared_up_to(config.frame_samples_power),
            transform_capacity: config.frame_samples_power,
            scratch: Box::new(Scratch::new()),
            index: 0,
            superframe_remaining: config.superframe_bytes(),
            poisoned: false,
            initial_delay: 0,
            initial_length: None,
            delay: 0,
            length: None,
        }
    }

    /// Reserve the largest supported channel and transform geometry. Use this
    /// when configuration arrives after instance creation: [`Self::reconfigure`]
    /// then accepts every validated configuration without allocating.
    #[must_use]
    pub fn new_max(config: Atrac9Config) -> Self {
        let mut decoder = Self {
            config,
            blocks: (0..5).map(|_| Block::new(state::BlockType::Mono)).collect(),
            tables: DspTables::shared_up_to(8),
            transform_capacity: 8,
            scratch: Box::new(Scratch::new()),
            index: 0,
            superframe_remaining: config.superframe_bytes(),
            poisoned: false,
            initial_delay: 0,
            initial_length: None,
            delay: 0,
            length: None,
        };
        decoder.tables = DspTables::shared_up_to(config.frame_samples_power);
        decoder.configure_blocks();
        decoder
    }

    /// Start a new stream using reserved storage, clearing codec history and
    /// all gapless metadata. This never allocates. Decoders constructed with
    /// [`Self::new_max`] support every valid configuration; a smaller decoder
    /// returns [`Error::InsufficientCapacity`] without changing any state when
    /// its reserved block count or transform size would be exceeded.
    pub fn reconfigure(&mut self, config: Atrac9Config) -> Result<(), Error> {
        let tables = match DspTables::built(config.frame_samples_power) {
            Some(tables)
                if config.block_types().len() <= self.blocks.len()
                    && config.frame_samples_power <= self.transform_capacity =>
            {
                tables
            }
            _ => return Err(Error::InsufficientCapacity),
        };
        self.tables = tables;
        self.config = config;
        self.configure_blocks();
        self.index = 0;
        self.superframe_remaining = config.superframe_bytes();
        self.poisoned = false;
        self.set_gapless(0, None);
        Ok(())
    }

    fn configure_blocks(&mut self) {
        for (block, &kind) in self.blocks.iter_mut().zip(self.config.block_types()) {
            block.reset(kind);
        }
    }
    /// Stream format used by this instance.
    #[must_use]
    pub const fn config(&self) -> Atrac9Config {
        self.config
    }
    /// Frame index at which the next decode starts within its superframe.
    #[must_use]
    pub const fn frame_index(&self) -> usize {
        self.index
    }
    /// Maximum input extent of the next frame, up to the superframe boundary.
    #[must_use]
    pub const fn superframe_bytes_remaining(&self) -> usize {
        self.superframe_remaining
    }
    /// Whether an encoded-data error has stopped decoding. Every decode call
    /// returns [`Error::NeedsReset`] until [`Self::reset`] or
    /// [`Self::reconfigure`].
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Reset history, band-extension RNG and frame index in place. Restores the
    /// delay and sample length most recently supplied to [`Self::set_gapless`].
    pub fn reset(&mut self) {
        for block in &mut self.blocks[..self.config.block_types().len()] {
            block.reset(block.block_type);
        }
        self.index = 0;
        self.superframe_remaining = self.config.superframe_bytes();
        self.poisoned = false;
        self.delay = self.initial_delay;
        self.length = self.initial_length;
    }

    /// Set gapless encoder delay and an optional number of samples per channel
    /// to emit after the delay (for example from a RIFF `fact` chunk).
    /// This changes trimming only; it does not reset codec history.
    pub fn set_gapless(&mut self, encoder_delay: usize, total_samples: Option<usize>) {
        self.initial_delay = encoder_delay;
        self.initial_length = total_samples;
        self.delay = encoder_delay;
        self.length = total_samples;
    }

    /// Update remaining trimming counts without changing the values restored
    /// by [`Self::reset`], or resetting codec history or superframe progress.
    pub fn set_gapless_state(&mut self, current_delay: usize, current_length: Option<usize>) {
        self.delay = current_delay;
        self.length = current_length;
    }

    /// Remaining encoder delay and playable samples per channel. The length is
    /// `None` when no tail limit is active.
    #[must_use]
    pub const fn gapless_state(&self) -> (usize, Option<usize>) {
        (self.delay, self.length)
    }

    /// Decode one codec frame into interleaved signed 16-bit PCM, rounding as
    /// `LibAtrac9` does (ties toward positive infinity), with saturation.
    pub fn decode_frame_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
    ) -> Result<DecodeInfo, Error> {
        if self.config.channel_count() == 1 {
            self.decode_frame(input, output, pcm_i16_mono)
        } else {
            self.decode_frame(input, output, pcm_i16)
        }
    }
    /// Decode one frame into normalized interleaved floating-point PCM. Unlike
    /// conversion through i16 this preserves fractional DSP precision.
    pub fn decode_frame_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<DecodeInfo, Error> {
        self.decode_frame(input, output, pcm_f32)
    }
    /// Decode a complete fixed-size superframe into interleaved i16 PCM.
    /// Ignores bytes beyond the first superframe and consumes its end padding.
    ///
    /// Input shorter than one superframe returns [`Error::Truncated`] and
    /// changes nothing, so the call can be retried with more data. An error
    /// inside the superframe poisons the decoder and leaves
    /// [`Self::gapless_state`] as it was before the call.
    pub fn decode_superframe_i16(
        &mut self,
        input: &[u8],
        output: &mut [i16],
    ) -> Result<DecodeInfo, Error> {
        if self.config.channel_count() == 1 {
            self.decode_superframe(input, output, pcm_i16_mono)
        } else {
            self.decode_superframe(input, output, pcm_i16)
        }
    }
    /// Decode a complete fixed-size superframe into normalized interleaved f32 PCM,
    /// with the same input and error handling as [`Self::decode_superframe_i16`].
    pub fn decode_superframe_f32(
        &mut self,
        input: &[u8],
        output: &mut [f32],
    ) -> Result<DecodeInfo, Error> {
        self.decode_superframe(input, output, pcm_f32)
    }

    fn written_samples(&self, samples: usize) -> usize {
        let after_delay = samples - self.delay.min(samples);
        self.length
            .map_or(after_delay, |length| length.min(after_delay))
    }

    fn check_output(&self, samples: usize, available: usize) -> Result<(), Error> {
        let required = self.written_samples(samples) * self.config.channel_count();
        if available < required {
            return Err(Self::output_error(required, available));
        }
        Ok(())
    }
    #[cold]
    const fn output_error(required: usize, available: usize) -> Error {
        Error::OutputTooSmall {
            required,
            available,
        }
    }

    fn decode_frame<T: Copy, F: Copy + Fn(f64) -> T>(
        &mut self,
        input: &[u8],
        output: &mut [T],
        convert: F,
    ) -> Result<DecodeInfo, Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }
        self.check_output(self.config.frame_samples, output.len())?;
        let input = &input[..input.len().min(self.superframe_remaining)];
        let mut br = bits::BitReader::new(input);
        let block_count = self.config.block_types().len();
        if let Err(error) = unpack::unpack_frame(
            &self.config,
            &mut self.blocks[..block_count],
            self.index,
            &mut br,
        ) {
            self.poisoned = true;
            return Err(error);
        }
        let bytes_consumed = br.position() / 8;
        self.superframe_remaining -= bytes_consumed;
        self.index += 1;
        let padding_bytes = if self.index == self.config.frames_per_superframe() {
            self.index = 0;
            let padding = self.superframe_remaining;
            self.superframe_remaining = self.config.superframe_bytes();
            padding
        } else {
            0
        };
        for block in &mut self.blocks[..block_count] {
            dsp::decode_block(&self.config, block, self.tables, &mut self.scratch);
        }
        let skip = self.delay.min(self.config.frame_samples);
        let samples_written = self.written_samples(self.config.frame_samples);
        let channels = self.config.channel_count();
        interleave(
            &self.blocks[..block_count],
            skip..skip + samples_written,
            &mut output[..samples_written * channels],
            convert,
        );
        self.delay -= skip;
        if let Some(length) = &mut self.length {
            *length -= samples_written;
        }
        Ok(DecodeInfo {
            bytes_consumed,
            padding_bytes,
            frames_consumed: 1,
            samples_written,
            pcm_values_written: samples_written * channels,
            skipped_samples: self.config.frame_samples - samples_written,
            encoder_delay_remaining: self.delay,
        })
    }

    fn decode_superframe<T: Copy, F: Copy + Fn(f64) -> T>(
        &mut self,
        input: &[u8],
        output: &mut [T],
        convert: F,
    ) -> Result<DecodeInfo, Error> {
        if self.poisoned {
            return Err(Error::NeedsReset);
        }
        if self.index != 0 {
            return Err(Error::NotSuperframeBoundary);
        }
        let input = input
            .get(..self.config.superframe_bytes())
            .ok_or(Error::Truncated)?;
        self.check_output(self.config.superframe_samples(), output.len())?;
        let gapless = (self.delay, self.length);
        let mut result = DecodeInfo::default();
        for _ in 0..self.config.frames_per_superframe() {
            let info = match self.decode_frame(
                &input[result.bytes_consumed..],
                &mut output[result.pcm_values_written..],
                convert,
            ) {
                Ok(info) => info,
                Err(error) => {
                    (self.delay, self.length) = gapless;
                    return Err(error);
                }
            };
            result.bytes_consumed += info.bytes_consumed;
            result.padding_bytes += info.padding_bytes;
            result.frames_consumed += 1;
            result.samples_written += info.samples_written;
            result.pcm_values_written += info.pcm_values_written;
            result.skipped_samples += info.skipped_samples;
            result.encoder_delay_remaining = info.encoder_delay_remaining;
        }
        result.bytes_consumed += result.padding_bytes;
        Ok(result)
    }
}

/// Writes the selected samples of every channel to interleaved `output`,
/// whose length is the sample count times the channel count.
fn interleave<'a, T: Copy, F: Fn(f64) -> T>(
    blocks: &'a [Block],
    samples: core::ops::Range<usize>,
    output: &mut [T],
    convert: F,
) {
    if samples.is_empty() {
        return;
    }
    let pcm = |block: &'a Block, channel: usize| &block.channels[channel].pcm[samples.clone()];
    let (left, right) = match blocks {
        [block] if block.channel_count == 1 => {
            for (output, &sample) in output.iter_mut().zip(pcm(block, 0)) {
                *output = convert(sample);
            }
            return;
        }
        [block] => (pcm(block, 0), pcm(block, 1)),
        [first, second] if first.channel_count + second.channel_count == 2 => {
            (pcm(first, 0), pcm(second, 0))
        }
        _ => {
            let channels = blocks.iter().map(|block| block.channel_count).sum();
            let mut offset = 0;
            for block in blocks {
                for channel in 0..block.channel_count {
                    for (output, &sample) in output[offset..]
                        .iter_mut()
                        .step_by(channels)
                        .zip(pcm(block, channel))
                    {
                        *output = convert(sample);
                    }
                    offset += 1;
                }
            }
            return;
        }
    };
    for (frame, (&left, &right)) in output
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(left.iter().zip(right))
    {
        *frame = [convert(left), convert(right)];
    }
}

#[allow(clippy::cast_possible_truncation)] // Explicit saturated PCM quantization.
fn pcm_i16(sample: f64) -> i16 {
    (sample + 0.5).floor().clamp(-32768.0, 32767.0) as i16
}

// The contiguous mono store loop favors this truncation/correction form;
// interleaved output favors the floor form above. Both round and clip identically.
#[allow(clippy::cast_possible_truncation)]
fn pcm_i16_mono(sample: f64) -> i16 {
    let shifted = (sample + 0.5).clamp(-32768.0, 32767.0);
    let integer = shifted as i32;
    (integer - i32::from(shifted < f64::from(integer))) as i16
}

#[allow(clippy::cast_possible_truncation)] // Requested f64 DSP to f32 PCM conversion.
fn pcm_f32(sample: f64) -> f32 {
    (sample / 32768.0).clamp(-1.0, 1.0) as f32
}

#[cfg(test)]
#[allow(
    dead_code,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unwrap_used
)]
mod test_fixtures;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::test_fixtures as fixtures;
    use proptest::prelude::*;

    #[test]
    fn config_known_answer_and_extremes() {
        let config = Atrac9Config::parse([0xfe, 0x74, 0x1b, 0xf0]).unwrap();
        assert_eq!(config.sample_rate(), 48000);
        assert_eq!(config.channel_count(), 2);
        assert_eq!(config.frame_samples(), 256);
        assert_eq!(config.frame_bytes(), 224);
        assert_eq!(config.frames_per_superframe(), 4);
        assert_eq!(
            Atrac9Config::parse([0xfe, 0x7c, 0, 0]),
            Err(Error::InvalidConfig)
        );
        assert_eq!(
            Atrac9Config::parse([0xfe, 0x75, 0, 0]),
            Err(Error::InvalidConfig)
        );
        assert_eq!(
            Atrac9Config::parse([0xff, 0x74, 0, 0]),
            Err(Error::InvalidConfig)
        );
    }

    #[test]
    fn rounding_and_clipping_match_reference() {
        for (input, expected) in [
            (-32769.0, -32768),
            (-1.5, -1),
            (-0.5, 0),
            (0.5, 1),
            (1.5, 2),
            (32768.0, 32767),
        ] {
            assert_eq!(pcm_i16(input), expected);
            assert_eq!(pcm_i16_mono(input), expected);
        }
        assert_eq!(pcm_f32(-32768.0), -1.0);
        assert_eq!(pcm_f32(32768.0), 1.0);
        assert_eq!(pcm_f32(0.25), 0.25 / 32768.0);
    }

    #[test]
    fn output_error_is_retryable_and_bad_data_requires_reset() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 0, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 0, 20, 17);
        let mut decoder = Decoder::new(config);
        let mut out = vec![123; 512];
        assert!(matches!(
            decoder.decode_frame_i16(&frames[0], &mut out[..1]),
            Err(Error::OutputTooSmall { .. })
        ));
        assert_eq!(decoder.frame_index(), 0);
        assert!(out.iter().all(|&x| x == 123));
        decoder.decode_frame_i16(&frames[0], &mut out).unwrap();
        let first = out.clone();
        decoder.reset();
        assert_eq!(
            decoder.decode_frame_i16(&[], &mut out),
            Err(Error::Truncated)
        );
        assert_eq!(
            decoder.decode_frame_i16(&frames[0], &mut out),
            Err(Error::NeedsReset)
        );
        decoder.reset();
        decoder.decode_frame_i16(&frames[0], &mut out).unwrap();
        assert_eq!(out, first);
    }

    #[test]
    fn superframe_padding_gapless_and_temporal_history() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 2, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 2, 20, 17);
        let mut input = frames[..4].concat();
        let payload_bytes = input.len();
        input.resize(config.superframe_bytes(), 0xcd);
        let mut decoder = Decoder::new(config);
        let mut complete = vec![0; 2048];
        let info = decoder
            .decode_superframe_i16(&input, &mut complete)
            .unwrap();
        assert_eq!(info.bytes_consumed, 8192);
        assert_eq!(info.padding_bytes, 8192 - payload_bytes);
        assert_eq!(info.samples_written, 1024);
        decoder.reset();
        decoder.set_gapless(300, Some(400));
        let mut trimmed = vec![99; 800];
        let info = decoder.decode_superframe_i16(&input, &mut trimmed).unwrap();
        assert_eq!(info.frames_consumed, 4);
        assert_eq!(info.samples_written, 400);
        assert_eq!(info.skipped_samples, 624);
        assert_eq!(info.encoder_delay_remaining, 0);
        assert_eq!(trimmed, complete[600..1400]);
        decoder.reset();
        assert_eq!(decoder.delay, 300);
        assert_eq!(decoder.length, Some(400));
    }

    #[test]
    fn maximum_geometry_reconfigures_all_layouts_without_replacing_storage() {
        let initial = Atrac9Config::parse(fixtures::config(0, 0, 0, 2048)).unwrap();
        let mut decoder = Decoder::new_max(initial);
        let blocks = decoder.blocks.as_ptr();
        let scratch = std::ptr::from_ref(&*decoder.scratch);
        for rate in 0..16 {
            for layout in (0..6).rev() {
                for sf in 0..4 {
                    let config =
                        Atrac9Config::parse(fixtures::config(rate, layout, sf, 2048)).unwrap();
                    decoder.reconfigure(config).unwrap();
                    assert_eq!(decoder.config(), config);
                    assert_eq!(decoder.blocks.as_ptr(), blocks);
                    assert_eq!(std::ptr::from_ref(&*decoder.scratch), scratch);
                    let mut ordinary = Decoder::new(config);
                    let mut expected = vec![0; config.frame_samples() * config.channel_count()];
                    let mut actual = expected.clone();
                    for frame in fixtures::frames(rate, layout, sf, 20, 17) {
                        let expected_info =
                            ordinary.decode_frame_i16(&frame, &mut expected).unwrap();
                        let actual_info = decoder.decode_frame_i16(&frame, &mut actual).unwrap();
                        assert_eq!(actual_info, expected_info, "{config:?}");
                        assert_eq!(actual, expected, "{config:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn insufficient_reconfigure_capacity_preserves_stream_progress() {
        let initial = Atrac9Config::parse(fixtures::config(0, 0, 2, 2048)).unwrap();
        let mut decoder = Decoder::new(initial);
        decoder.set_gapless(10, Some(400));
        let frame = &fixtures::frames(0, 0, 2, 20, 17)[0];
        let mut output = vec![0; initial.frame_samples()];
        decoder.decode_frame_i16(frame, &mut output).unwrap();
        let remaining = decoder.superframe_bytes_remaining();
        let trimming = decoder.gapless_state();
        for config in [
            Atrac9Config::parse(fixtures::config(0, 4, 0, 2048)).unwrap(),
            Atrac9Config::parse(fixtures::config(7, 0, 0, 2048)).unwrap(),
        ] {
            assert_eq!(
                decoder.reconfigure(config),
                Err(Error::InsufficientCapacity)
            );
            assert_eq!(decoder.config(), initial);
            assert_eq!(decoder.frame_index(), 1);
            assert_eq!(decoder.superframe_bytes_remaining(), remaining);
            assert_eq!(decoder.gapless_state(), trimming);
        }
    }

    #[test]
    fn gapless_state_update_preserves_history_progress_and_reset_metadata() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 2, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 2, 20, 17);
        let mut decoder = Decoder::new_max(config);
        let mut expected = Decoder::new(config);
        decoder.set_gapless(10, Some(600));
        expected.set_gapless(10, Some(600));
        let mut pcm = [0; 512];
        let mut reference_pcm = [0; 512];
        decoder.decode_frame_i16(&frames[0], &mut pcm).unwrap();
        expected
            .decode_frame_i16(&frames[0], &mut reference_pcm)
            .unwrap();
        let remaining = decoder.superframe_bytes_remaining();
        decoder.set_gapless_state(7, Some(30));
        expected.set_gapless(7, Some(30));
        assert_eq!(decoder.frame_index(), 1);
        assert_eq!(decoder.superframe_bytes_remaining(), remaining);
        assert_eq!(decoder.gapless_state(), (7, Some(30)));
        let info = decoder.decode_frame_i16(&frames[1], &mut pcm).unwrap();
        let reference = expected
            .decode_frame_i16(&frames[1], &mut reference_pcm)
            .unwrap();
        assert_eq!(info, reference);
        assert_eq!(
            &pcm[..info.pcm_values_written],
            &reference_pcm[..info.pcm_values_written]
        );
        assert_eq!(decoder.gapless_state(), (0, Some(0)));
        decoder.reset();
        assert_eq!(decoder.gapless_state(), (10, Some(600)));
        assert_eq!(decoder.frame_index(), 0);
        assert_eq!(
            decoder.decode_frame_i16(&[], &mut pcm),
            Err(Error::Truncated)
        );
        let saved = decoder.gapless_state();
        decoder.reset();
        decoder.set_gapless_state(saved.0, saved.1);
        decoder.decode_frame_i16(&frames[0], &mut pcm).unwrap();
        assert_eq!(decoder.gapless_state(), (0, Some(354)));
        decoder.reconfigure(config).unwrap();
        assert_eq!(decoder.gapless_state(), (0, None));
    }

    const _: () = {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Decoder>();
        assert_send_sync::<Atrac9Config>();
        assert_send_sync::<Error>();
        assert_send_sync::<DecodeInfo>();
    };

    fn superframe(config: Atrac9Config, frames: &[Vec<u8>]) -> Vec<u8> {
        let mut input = frames.concat();
        input.resize(config.superframe_bytes(), 0xcd);
        input
    }

    #[test]
    fn errors_display_their_cause() {
        assert_eq!(Error::Truncated.to_string(), "truncated ATRAC9 input");
        assert_eq!(
            Error::InvalidData("band count").to_string(),
            "invalid ATRAC9 data: band count"
        );
        assert_eq!(
            Error::OutputTooSmall {
                required: 4,
                available: 1
            }
            .to_string(),
            "PCM buffer requires 4 values, got 1"
        );
        let boxed: Box<dyn std::error::Error> = Box::new(Error::NeedsReset);
        assert!(boxed.source().is_none());
    }

    #[test]
    fn superframe_calls_require_a_boundary_without_poisoning() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 2, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 2, 20, 17);
        let input = superframe(config, &frames[..4]);
        let mut decoder = Decoder::new(config);
        let mut out = vec![0; config.superframe_pcm_values()];
        let first = decoder.decode_frame_i16(&input, &mut out).unwrap();
        let remaining = decoder.superframe_bytes_remaining();
        assert_eq!(
            decoder.decode_superframe_i16(&input, &mut out),
            Err(Error::NotSuperframeBoundary)
        );
        assert_eq!(
            decoder.decode_superframe_f32(&input, &mut [0.0; 4096]),
            Err(Error::NotSuperframeBoundary)
        );
        assert!(!decoder.is_poisoned());
        assert_eq!(decoder.frame_index(), 1);
        assert_eq!(decoder.superframe_bytes_remaining(), remaining);
        let mut offset = first.bytes_consumed;
        for _ in 1..4 {
            let info = decoder
                .decode_frame_i16(&input[offset..], &mut out)
                .unwrap();
            offset += info.bytes_consumed + info.padding_bytes;
        }
        assert_eq!(offset, config.superframe_bytes());
        decoder.decode_superframe_i16(&input, &mut out).unwrap();
    }

    #[test]
    fn short_superframe_input_is_retryable_and_does_not_poison() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 2, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 2, 20, 17);
        let input = superframe(config, &frames[..4]);
        let mut expected = vec![0; config.superframe_pcm_values()];
        Decoder::new(config)
            .decode_superframe_i16(&input, &mut expected)
            .unwrap();
        let mut decoder = Decoder::new(config);
        decoder.set_gapless(5, Some(900));
        let mut out = vec![7; config.superframe_pcm_values()];
        for length in [0, 1, config.superframe_bytes() - 1] {
            assert_eq!(
                decoder.decode_superframe_i16(&input[..length], &mut out),
                Err(Error::Truncated)
            );
            assert_eq!(
                decoder.decode_superframe_f32(&input[..length], &mut [0.0; 4096]),
                Err(Error::Truncated)
            );
            assert!(!decoder.is_poisoned());
            assert_eq!(decoder.frame_index(), 0);
            assert_eq!(decoder.gapless_state(), (5, Some(900)));
            assert!(out.iter().all(|&value| value == 7));
        }
        decoder.set_gapless(0, None);
        let info = decoder.decode_superframe_i16(&input, &mut out).unwrap();
        assert_eq!(info.bytes_consumed, config.superframe_bytes());
        assert_eq!(out, expected);
    }

    #[test]
    fn failure_inside_superframe_poisons_and_restores_gapless_state() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 2, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 2, 20, 17);
        let good = superframe(config, &frames[..4]);
        // A later frame with its first-in-superframe flag set is invalid.
        let mut bad = superframe(config, &frames[..1]);
        bad[frames[0].len()..].fill(0);
        let mut decoder = Decoder::new(config);
        decoder.set_gapless(300, Some(2000));
        let mut out = vec![0; config.superframe_pcm_values()];
        decoder.decode_superframe_i16(&good, &mut out).unwrap();
        assert_eq!(decoder.gapless_state(), (0, Some(1276)));
        assert!(!decoder.is_poisoned());
        assert!(matches!(
            decoder.decode_superframe_i16(&bad, &mut out),
            Err(Error::InvalidData(_))
        ));
        assert!(decoder.is_poisoned());
        assert_eq!(decoder.gapless_state(), (0, Some(1276)));
        assert_eq!(
            decoder.decode_superframe_i16(&good, &mut out),
            Err(Error::NeedsReset)
        );
        assert_eq!(
            decoder.decode_frame_f32(&frames[0], &mut [0.0; 512]),
            Err(Error::NeedsReset)
        );
        assert_eq!(decoder.gapless_state(), (0, Some(1276)));

        let saved = decoder.gapless_state();
        decoder.reset();
        assert!(!decoder.is_poisoned());
        assert_eq!(decoder.gapless_state(), (300, Some(2000)));
        decoder.set_gapless_state(saved.0, saved.1);
        let info = decoder.decode_superframe_i16(&good, &mut out).unwrap();
        assert_eq!(info.samples_written, 1024);
        assert_eq!(decoder.gapless_state(), (0, Some(252)));

        let mut float_decoder = Decoder::new(config);
        float_decoder.set_gapless(10, Some(50));
        assert!(
            float_decoder
                .decode_superframe_f32(&bad, &mut [0.0; 2048])
                .is_err()
        );
        assert!(float_decoder.is_poisoned());
        assert_eq!(float_decoder.gapless_state(), (10, Some(50)));
    }

    #[test]
    fn float_output_is_trimmed_like_integer_output() {
        let config = Atrac9Config::parse(fixtures::config(7, 2, 2, 2048)).unwrap();
        let frames = fixtures::frames(7, 2, 2, 30, 17);
        let input = superframe(config, &frames[..4]);
        let mut complete = vec![0.0; config.superframe_pcm_values()];
        Decoder::new(config)
            .decode_superframe_f32(&input, &mut complete)
            .unwrap();
        assert!(complete.iter().any(|&value| value != 0.0));
        let mut decoder = Decoder::new(config);
        decoder.set_gapless(300, Some(400));
        let mut trimmed = vec![f32::NAN; 800];
        let info = decoder.decode_superframe_f32(&input, &mut trimmed).unwrap();
        assert_eq!(info.samples_written, 400);
        assert_eq!(info.pcm_values_written, 800);
        assert_eq!(info.skipped_samples, 624);
        assert!(
            trimmed
                .iter()
                .zip(&complete[600..1400])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        decoder.reset();
        let mut frame = vec![f32::NAN; config.frame_pcm_values()];
        let mut written = Vec::new();
        for encoded in &frames[..4] {
            let info = decoder.decode_frame_f32(encoded, &mut frame).unwrap();
            written.extend_from_slice(&frame[..info.pcm_values_written]);
        }
        assert_eq!(written.len(), 800);
        assert!(
            written
                .iter()
                .zip(&complete[600..1400])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn delay_spanning_frames_with_empty_length_needs_no_output() {
        for layout in 0..6 {
            let config = Atrac9Config::parse(fixtures::config(7, layout, 2, 2048)).unwrap();
            let frames = fixtures::frames(7, layout, 2, 20, 17);
            let mut decoder = Decoder::new(config);
            decoder.set_gapless(600, Some(0));
            for (frame, delay) in frames[..4].iter().zip([344, 88, 0, 0]) {
                let info = decoder.decode_frame_i16(frame, &mut []).unwrap();
                assert_eq!(info.samples_written, 0);
                assert_eq!(info.pcm_values_written, 0);
                assert_eq!(info.skipped_samples, 256);
                assert_eq!(info.encoder_delay_remaining, delay);
            }
            assert_eq!(decoder.gapless_state(), (0, Some(0)));
            let info = decoder
                .decode_superframe_f32(&superframe(config, &frames[4..8]), &mut [])
                .unwrap();
            assert_eq!(info.frames_consumed, 4);
            assert_eq!(info.pcm_values_written, 0);
            assert_eq!(info.skipped_samples, 1024);
            decoder.reset();
            assert_eq!(decoder.gapless_state(), (600, Some(0)));
            let info = decoder
                .decode_superframe_i16(&superframe(config, &frames[..4]), &mut [])
                .unwrap();
            assert_eq!(info.encoder_delay_remaining, 0);
            assert_eq!(info.samples_written, 0);

            // Also verify multi-frame delay followed by real audio and a trimmed tail.
            let input = superframe(config, &frames[..4]);
            let mut complete = vec![0; config.superframe_pcm_values()];
            decoder.set_gapless(0, None);
            decoder.reset();
            decoder
                .decode_superframe_i16(&input, &mut complete)
                .unwrap();
            decoder.set_gapless(300, Some(400));
            decoder.reset();
            let mut trimmed = vec![99; 400 * config.channel_count()];
            let info = decoder.decode_superframe_i16(&input, &mut trimmed).unwrap();
            assert_eq!(info.samples_written, 400);
            assert_eq!(info.pcm_values_written, trimmed.len());
            let start = 300 * config.channel_count();
            assert_eq!(trimmed, complete[start..start + trimmed.len()]);
        }
    }

    #[test]
    fn reset_restores_constructed_blocks_in_place() {
        for layout in 0..6 {
            let config = Atrac9Config::parse(fixtures::config(7, layout, 2, 2048)).unwrap();
            let mut decoder = Decoder::new_max(config);
            let mut out = vec![0; config.frame_pcm_values()];
            for frame in &fixtures::frames(7, layout, 2, 30, 17)[..3] {
                decoder.decode_frame_i16(frame, &mut out).unwrap();
            }
            for block in &mut decoder.blocks {
                for kind in [
                    state::BlockType::Stereo,
                    state::BlockType::Lfe,
                    block.block_type,
                ] {
                    let fresh = format!("{:?}", Block::new(kind));
                    assert_ne!(format!("{block:?}"), fresh);
                    block.reset(kind);
                    assert_eq!(format!("{block:?}"), fresh);
                    block.channels[1].pcm[7] = 1.0;
                    block.gradient[3] = 2;
                }
            }
        }
    }

    #[test]
    fn config_reports_interleaved_buffer_sizes() {
        for layout in 0..6 {
            let config = Atrac9Config::parse(fixtures::config(7, layout, 3, 2048)).unwrap();
            assert_eq!(
                config.frame_pcm_values(),
                config.frame_samples() * config.channel_count()
            );
            assert_eq!(
                config.superframe_pcm_values(),
                config.superframe_samples() * config.channel_count()
            );
        }
        let config = Atrac9Config::parse(fixtures::config(7, 3, 2, 2048)).unwrap();
        assert_eq!(config.channel_config_index(), 3);
        assert_eq!(config.frame_pcm_values(), 256 * 6);
        assert_eq!(config.superframe_pcm_values(), 1024 * 6);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]
        #[test]
        fn mono_pcm_conversion_matches_floor(sample in any::<f64>()) {
            prop_assert_eq!(pcm_i16_mono(sample), pcm_i16(sample));
        }
        #[test]
        fn config_round_trip(rate in 0u8..16,layout in 0u8..6,sf in 0u8..4,frame_bytes in 1usize..=2048) {
            let data=fixtures::config(rate,layout,sf,frame_bytes);
            let parsed=Atrac9Config::parse(data).unwrap();
            prop_assert_eq!(parsed.config_data(),data);
            prop_assert_eq!(parsed.frame_bytes(),frame_bytes);
            prop_assert_eq!(parsed.frames_per_superframe(),1<<sf);
            prop_assert_eq!(parsed.superframe_bytes(),frame_bytes<<sf);
        }
        #[test]
        fn arbitrary_configuration_and_encoded_bytes_never_panic(data in any::<[u8;4]>(),input in prop::collection::vec(any::<u8>(),0..8193)) {
            if let Ok(config)=Atrac9Config::parse(data) {
                let mut decoder=Decoder::new(config);
                let mut out=vec![0;config.superframe_samples()*config.channel_count()];
                let _=decoder.decode_frame_i16(&input,&mut out);
                decoder.reset();
                let _=decoder.decode_superframe_i16(&input,&mut out);
            }
        }
        #[test]
        fn arbitrary_frames_in_valid_config_never_panic(rate in 0u8..16,layout in 0u8..6,sf in 0u8..4,input in prop::collection::vec(any::<u8>(),0..8193)) {
            let config=Atrac9Config::parse(fixtures::config(rate,layout,sf,2048)).unwrap();
            let mut decoder=Decoder::new(config);
            let mut out=vec![0.0;config.superframe_samples()*config.channel_count()];
            let _=decoder.decode_frame_f32(&input,&mut out);
            decoder.reset();
            let _=decoder.decode_superframe_f32(&input,&mut out);
        }
        #[test]
        fn mutations_after_valid_history_never_panic(rate in 0u8..16,layout in 0u8..6,input in prop::collection::vec(any::<u8>(),0..2049)) {
            let config=Atrac9Config::parse(fixtures::config(rate,layout,2,2048)).unwrap();
            let frames=fixtures::frames(rate,layout,2,20,17);
            let mut decoder=Decoder::new(config);
            let mut out=vec![0;config.frame_samples()*config.channel_count()];
            decoder.decode_frame_i16(&frames[0],&mut out).unwrap();
            let _=decoder.decode_frame_i16(&input,&mut out);
        }
    }
}
