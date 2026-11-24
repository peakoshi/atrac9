# atrac9

[![crates.io](https://img.shields.io/crates/v/atrac9.svg)](https://crates.io/crates/atrac9)
[![docs.rs](https://img.shields.io/docsrs/atrac9)](https://docs.rs/atrac9)
[![CI](https://github.com/peakoshi/atrac9/actions/workflows/ci.yml/badge.svg)](https://github.com/peakoshi/atrac9/actions/workflows/ci.yml)

A safe, allocation-free ATRAC9 decoder in pure Rust.

ATRAC9 is the audio codec used in PlayStation Vita and PlayStation 4 games, in
`.at9` files and inside sound banks. This crate decodes it to PCM, and its output
matches the reference C decoder, [LibAtrac9], bit for bit.

- Safe Rust only (`#![forbid(unsafe_code)]`), no C, zero dependencies.
- No allocation or locking while decoding, so it can run on an audio thread.
- Every sample rate, channel layout from mono to 7.1, and superframe size.
- 16-bit or floating-point output, with gapless trimming.
- Malformed input returns an error, never a panic.

## Usage

```toml
[dependencies]
atrac9 = "0.1"
```

A stream is described by a four-byte configuration word and stored as a run of
fixed-size superframes. In an `.at9` file the configuration word sits in the
`fmt ` chunk and the superframes fill the `data` chunk.

```rust
use atrac9::{Atrac9Config, Decoder};

fn decode(config_word: [u8; 4], audio: &[u8]) -> Result<Vec<i16>, atrac9::Error> {
    let config = Atrac9Config::parse(config_word)?;
    let mut decoder = Decoder::new(config);

    let mut buffer = vec![0; config.superframe_pcm_values()];
    let mut pcm = Vec::new();
    for superframe in audio.chunks_exact(config.superframe_bytes()) {
        let info = decoder.decode_superframe_i16(superframe, &mut buffer)?;
        pcm.extend_from_slice(&buffer[..info.pcm_values_written]);
    }
    Ok(pcm)
}
```

Samples are interleaved. `decode_superframe_f32` returns samples in [-1, 1] taken
straight from the decoder's double-precision output, so nothing is lost to 16-bit
rounding.

### Gapless playback

Encoders add a short delay at the start of a stream, and the last superframe is
usually only partly filled. The container records both numbers (the `fact` chunk
in an `.at9` file). Hand them to the decoder with
`decoder.set_gapless(encoder_delay, Some(total_samples))` and it trims the output
for you. `DecodeInfo::samples_written` says how much of the buffer is real audio.

### Frame by frame

Emulators and streaming players often feed one frame at a time. Frames vary in
size inside a superframe, so the decoder reports how far to move:

```rust
use atrac9::Decoder;

fn decode_frames(
    decoder: &mut Decoder,
    superframe: &[u8],
    pcm: &mut [i16],
) -> Result<(), atrac9::Error> {
    let mut offset = 0;
    for _ in 0..decoder.config().frames_per_superframe() {
        let info = decoder.decode_frame_i16(&superframe[offset..], pcm)?;
        offset += info.bytes_consumed + info.padding_bytes;
        // pcm[..info.pcm_values_written] now holds this frame.
    }
    Ok(())
}
```

`padding_bytes` is only non-zero after the last frame of a superframe. Frame and
superframe calls can be mixed at superframe boundaries.

### Errors

After bad data the decoder refuses to continue until you call `reset()`, which
keeps your gapless settings; `is_poisoned()` tells you when that is needed. When a
superframe call fails partway through, `gapless_state()` still holds the counts
from before that call. To skip a damaged superframe and keep playing, save
`gapless_state()`, reset, restore it with `set_gapless_state`, and carry on at the
next superframe.

Some errors change nothing, so you can simply retry: `OutputTooSmall` with a
larger buffer, and `Truncated` from a superframe call with the rest of the
superframe. A frame call that runs out of input is bad data like any other.

### One decoder, many streams

`Decoder::new_max` reserves room for the largest layout. `reconfigure` then
switches it to any other stream format without allocating, which suits a fixed
pool of decoders in a mixer or an emulator.

## Testing

The tests generate synthetic streams for every sample rate, channel layout and
superframe size. Between them they exercise fixed-length and Huffman-coded
spectra, every band-extension mode, intensity stereo, LFE channels, and all the
scale-factor and precision modes. Each stream is decoded by this crate and by the
original C LibAtrac9, and the results must be identical: the same `i16` samples,
the same `f32` bit patterns and the same byte counts. Property tests throw random
configurations and random bytes at the decoder to check that it never panics. CI
runs everything on Linux, macOS and Windows.

```sh
cargo test
```

The comparison tests build the C reference in `tests/reference` with your C
compiler: `cc` by default, or `cl` from a Visual Studio developer prompt on
Windows. [tests/reference/README.md](tests/reference/README.md) has the details.

## Performance

Decoding is cheap. Pinned to one efficiency core of a recent desktop processor,
the benchmark streams decode at roughly 2,000 times real time for 48 kHz mono,
800 times for stereo and 300 times for 5.1, within about 20% of the C library on
the same core and faster than it for mono. `cargo bench` runs the same streams
through this crate and through the C library, so you can compare the two on your
own machine.

## Credits

This crate is a Rust port of [LibAtrac9] by Alex Barney. The bitstream parsing,
tables and signal processing follow his C code, which is also the reference the
tests compare against.

ATRAC is a trademark of Sony Group Corporation. This project is not affiliated
with or endorsed by Sony.

## License

MIT, see [LICENSE](LICENSE).

[LibAtrac9]: https://github.com/Thealexbarney/LibAtrac9
