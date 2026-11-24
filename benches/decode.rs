#![forbid(unsafe_code)]
#![allow(clippy::expect_used)]

#[path = "../tests/support/mod.rs"]
mod support;

use atrac9::{Atrac9Config, Decoder};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Instant;

fn decode_benchmarks(c: &mut Criterion) {
    // Paired in-process loops; C times itself, excluding compilation, process start and IO.
    // Both retain overlap history and consume identical three-superframe batches.
    support::oracle::executable();
    let mut fixtures: Vec<_> = [
        ("mono_48k", 7, 0, 2),
        ("stereo_48k", 7, 2, 2),
        ("surround_48k", 7, 3, 2),
        ("stereo_96k", 12, 2, 2),
    ]
    .into_iter()
    .map(|(name, rate, layout, superframe)| {
        (
            name,
            support::fixtures::config(rate, layout, superframe, 2048),
            support::fixtures::frames(rate, layout, superframe, 20, 19),
        )
    })
    .collect();
    fixtures.push((
        "stereo_huffman_48k",
        support::fixtures::config(7, 2, 3, 2048),
        support::fixtures::advanced_frames(true, 4, None),
    ));
    fixtures.push((
        "stereo_bex_48k",
        support::fixtures::config(7, 2, 3, 2048),
        support::fixtures::advanced_frames(true, 10, Some(0)),
    ));
    for (name, config_data, frames) in fixtures {
        let config = Atrac9Config::parse(config_data).expect("synthetic config");
        let mut group = c.benchmark_group(name);
        group.throughput(Throughput::Elements(frames.len() as u64));
        group.bench_with_input(BenchmarkId::new("rust", "i16"), &frames, |b, frames| {
            b.iter_custom(|iterations| {
                let mut decoder = Decoder::new(config);
                let mut pcm = vec![0; config.frame_samples() * config.channel_count()];
                let start = Instant::now();
                for _ in 0..iterations {
                    for frame in frames {
                        decoder
                            .decode_frame_i16(black_box(frame), &mut pcm)
                            .expect("synthetic frame");
                        black_box(&pcm);
                    }
                }
                start.elapsed()
            });
        });
        group.bench_with_input(BenchmarkId::new("c", "i16"), &frames, |b, frames| {
            let mut timer =
                support::oracle::Timer::new(config_data, frames).expect("C timing worker");
            b.iter_custom(|iterations| {
                timer
                    .timed(u32::try_from(iterations).expect("benchmark iteration count"))
                    .expect("C timed batch")
            });
        });
        group.finish();
    }
}

criterion_group!(benches, decode_benchmarks);
criterion_main!(benches);
