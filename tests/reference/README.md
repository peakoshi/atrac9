# C reference

Every `.c` and `.h` file here except `oracle.c` is an unmodified copy of the C
sources of [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) by Alex Barney,
under the MIT license in [`LICENSE`](LICENSE). They are used only by the tests and
benchmarks. The library itself never compiles or links them.

`oracle.c` wraps LibAtrac9 in a small executable that the tests talk to over
stdin and stdout, so the comparison needs no FFI and no `unsafe`.

## Building

The tests build the reference on first use, once per test binary, under Cargo's
target directory:

| Compiler | Flags |
|---|---|
| `cc`, `gcc`, `clang` (default outside Windows) | `-O2 -ffp-contract=off` |
| `cl`, `clang-cl` (default on Windows) | `/O2 /fp:precise` |

Set `CC` to choose the compiler. On Windows, run Cargo from a Visual Studio
developer prompt so `cl` is on `PATH`. To use a reference you built yourself, set
`ATRAC9_REFERENCE` to its path.

Floating-point contraction stays off because Rust never fuses multiplies and adds.
On FMA hardware a fused C build would round differently, and the comparison is
bit exact.

## Protocol

Integers are little endian. A request is a four-byte magic, the four
configuration bytes, an input count, a repeat count, and then each input as a
`u32` length followed by its bytes.

- `AT9O` with repeat count `0` decodes each input as one frame. With `u32::MAX` it
  decodes each input as one complete superframe. The response is LibAtrac9's
  initialization status, then for every frame its status, bytes consumed, PCM
  value count, the `i16` PCM, and an `f32` view of the decoder's internal `f64`
  PCM, divided by 32768 and clamped to [-1, 1]. LibAtrac9 has no float API of its
  own; this view is what the Rust `f32` output is checked against.
- `AT9O` with any other repeat count decodes the inputs that many times and
  returns the elapsed nanoseconds.
- `AT9B` starts a persistent timing worker for the benchmarks. After the
  initialization status it reads repeat counts until `0` or end of input. For
  each count it resets the decoder, decodes the inputs that many times, and
  returns the elapsed nanoseconds. Process start, IO and initialization are
  outside the timer.

Inputs get 16 zero guard bytes, because LibAtrac9's bit reader reads past the
last code. Only well-formed synthetic streams are sent to the C decoder. The
malformed-input tests run against the Rust decoder alone, since the C code does
not bounds-check.
