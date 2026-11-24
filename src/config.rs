//! Validated ATRAC9 configuration word.
use crate::{
    Error,
    state::BlockType,
    tables::{SAMPLE_RATES, SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER},
};

const MONO: &[BlockType] = &[BlockType::Mono];
const DUAL: &[BlockType] = &[BlockType::Mono, BlockType::Mono];
const STEREO: &[BlockType] = &[BlockType::Stereo];
const SURROUND: &[BlockType] = &[
    BlockType::Stereo,
    BlockType::Mono,
    BlockType::Lfe,
    BlockType::Stereo,
];
const SURROUND_71: &[BlockType] = &[
    BlockType::Stereo,
    BlockType::Mono,
    BlockType::Lfe,
    BlockType::Stereo,
    BlockType::Stereo,
];
const QUAD: &[BlockType] = &[BlockType::Stereo, BlockType::Stereo];
const BLOCKS: [&[BlockType]; 6] = [MONO, DUAL, STEREO, SURROUND, SURROUND_71, QUAD];
const CHANNELS: [usize; 6] = [1, 2, 2, 6, 8, 4];

/// Immutable, checked ATRAC9 stream geometry. Frame byte size is an average:
/// encoded frames are packed into a fixed-size superframe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Atrac9Config {
    config_data: [u8; 4],
    pub(crate) sample_rate_index: usize,
    channel_config_index: usize,
    pub(crate) high_sample_rate: bool,
    pub(crate) frame_samples_power: usize,
    pub(crate) frame_samples: usize,
    frame_bytes: usize,
    frames_per_superframe: usize,
}

impl Atrac9Config {
    /// Parse the big-endian configuration word. Reserved low three bits are
    /// ignored, as in `LibAtrac9`; unsupported channel layouts are rejected.
    pub fn parse(config_data: [u8; 4]) -> Result<Self, Error> {
        let word = u32::from_be_bytes(config_data);
        let sample_rate_index = ((word >> 20) & 15) as usize;
        let channel_config_index = ((word >> 17) & 7) as usize;
        if word >> 24 != 0xfe || word & (1 << 16) != 0 || channel_config_index >= BLOCKS.len() {
            return Err(Error::InvalidConfig);
        }
        let frame_samples_power = SAMPLING_RATE_INDEX_TO_FRAME_SAMPLES_POWER[sample_rate_index];
        Ok(Self {
            config_data,
            sample_rate_index,
            channel_config_index,
            high_sample_rate: sample_rate_index > 7,
            frame_samples_power,
            frame_samples: 1 << frame_samples_power,
            frame_bytes: (((word >> 5) & 2047) + 1) as usize,
            frames_per_superframe: 1 << ((word >> 3) & 3),
        })
    }
    /// Original configuration bytes.
    #[must_use]
    pub const fn config_data(self) -> [u8; 4] {
        self.config_data
    }
    /// Sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(self) -> usize {
        SAMPLE_RATES[self.sample_rate_index]
    }
    /// Number of interleaved channels.
    #[must_use]
    pub const fn channel_count(self) -> usize {
        CHANNELS[self.channel_config_index]
    }
    /// Channel configuration index from the configuration word, which selects
    /// the interleaved channel layout:
    ///
    /// | Index | Layout | Channels |
    /// |---|---|---|
    /// | 0 | mono | 1 |
    /// | 1 | dual mono | 2 |
    /// | 2 | stereo | 2 |
    /// | 3 | 5.1 surround | 6 |
    /// | 4 | 7.1 surround | 8 |
    /// | 5 | quad | 4 |
    #[must_use]
    pub const fn channel_config_index(self) -> usize {
        self.channel_config_index
    }
    /// PCM samples per channel in one frame.
    #[must_use]
    pub const fn frame_samples(self) -> usize {
        self.frame_samples
    }
    /// Average encoded bytes per frame (not a frame delimiter).
    #[must_use]
    pub const fn frame_bytes(self) -> usize {
        self.frame_bytes
    }
    /// Number of frames packed into each superframe.
    #[must_use]
    pub const fn frames_per_superframe(self) -> usize {
        self.frames_per_superframe
    }
    /// Fixed encoded superframe size in bytes, including end padding.
    #[must_use]
    pub const fn superframe_bytes(self) -> usize {
        self.frame_bytes * self.frames_per_superframe
    }
    /// PCM samples per channel produced by a superframe.
    #[must_use]
    pub const fn superframe_samples(self) -> usize {
        self.frame_samples * self.frames_per_superframe
    }
    /// Interleaved PCM values produced by one frame: samples times channels.
    #[must_use]
    pub const fn frame_pcm_values(self) -> usize {
        self.frame_samples * self.channel_count()
    }
    /// Interleaved PCM values produced by one superframe; the buffer size that
    /// the superframe decode calls need without gapless trimming.
    #[must_use]
    pub const fn superframe_pcm_values(self) -> usize {
        self.superframe_samples() * self.channel_count()
    }
    pub(crate) const fn block_types(self) -> &'static [BlockType] {
        BLOCKS[self.channel_config_index]
    }
}
