# Changelog

## 0.1.0

First release.

- Frame and superframe decoding to interleaved `i16` or `f32` PCM.
- Every sample rate, channel layout (mono to 7.1) and superframe size.
- Gapless trimming of encoder delay and stream length.
- `Decoder::new_max` and `reconfigure` reuse one decoder across streams without allocating.
