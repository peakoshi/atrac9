#![forbid(unsafe_code)]
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

mod support;

use atrac9::{Atrac9Config, Decoder};

fn compare(config_data: [u8; 4], frames: &[Vec<u8>]) {
    let reference = support::oracle::decode(config_data, frames).expect("C oracle response");
    let config = Atrac9Config::parse(config_data).expect("synthetic config");
    let mut decoder = Decoder::new(config);
    let mut float_decoder = Decoder::new(config);
    let mut pcm = vec![0; config.frame_samples() * config.channel_count()];
    let mut float_pcm = vec![0.0; pcm.len()];
    for (index, (frame, expected)) in frames.iter().zip(&reference).enumerate() {
        assert_eq!(expected.status, 0, "C frame {index}");
        let info = decoder
            .decode_frame_i16(frame, &mut pcm)
            .unwrap_or_else(|error| panic!("Rust decode of frame {index}: {error:?}"));
        assert_eq!(info.bytes_consumed, expected.bytes_used, "frame {index}");
        assert_eq!(info.frames_consumed, 1);
        assert_eq!(info.samples_written, config.frame_samples());
        assert_eq!(info.pcm_values_written, pcm.len());
        assert_eq!(pcm, expected.samples, "frame {index}, config {config:?}");
        let float_info = float_decoder
            .decode_frame_f32(frame, &mut float_pcm)
            .expect("Rust f32 decode");
        assert_eq!(float_info, info);
        assert!(
            float_pcm
                .iter()
                .zip(&expected.float_samples)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "f32 frame {index}, config {config:?}"
        );
    }
    // Verify bulk superframe API consumes the fixed geometry including padding.
    let count = config.frames_per_superframe();
    let mut decoder = Decoder::new(config);
    let mut float_decoder = Decoder::new(config);
    let mut pcm = vec![0; config.frame_samples() * config.channel_count() * count];
    let mut float_pcm = vec![0.0; pcm.len()];
    for (group_index, group) in frames.chunks_exact(count).enumerate() {
        let mut input: Vec<u8> = group.iter().flatten().copied().collect();
        let encoded = input.len();
        input.resize(config.frame_bytes() * count, 0xa5);
        let info = decoder
            .decode_superframe_i16(&input, &mut pcm)
            .expect("Rust superframe decode");
        assert_eq!(info.bytes_consumed, input.len());
        assert_eq!(info.padding_bytes, input.len() - encoded);
        assert_eq!(info.frames_consumed, count);
        let expected: Vec<i16> = reference[group_index * count..(group_index + 1) * count]
            .iter()
            .flat_map(|f| f.samples.iter().copied())
            .collect();
        assert_eq!(pcm, expected, "bulk config {config:?} group {group_index}");
        let float_info = float_decoder
            .decode_superframe_f32(&input, &mut float_pcm)
            .expect("Rust f32 bulk decode");
        assert_eq!(float_info, info);
        let expected: Vec<f32> = reference[group_index * count..(group_index + 1) * count]
            .iter()
            .flat_map(|f| f.float_samples.iter().copied())
            .collect();
        assert!(
            float_pcm
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "f32 bulk config {config:?} group {group_index}"
        );
    }
}

#[test]
fn original_c_pcm_matches_all_configurations_and_history() {
    for rate in 0..16 {
        for layout in 0..6 {
            for superframe in 0..4 {
                for scale in [12, 20, 30] {
                    compare(
                        support::fixtures::config(rate, layout, superframe, 2048),
                        &support::fixtures::frames(rate, layout, superframe, scale, 17),
                    );
                }
            }
        }
    }
}

#[test]
fn huffman_temporal_scalefactors_and_band_extension_match_c() {
    for stereo in [false, true] {
        for qbits in 2..=16 {
            let frames = support::fixtures::advanced_frames(stereo, qbits, None);
            compare(
                support::fixtures::config(7, if stereo { 2 } else { 0 }, 3, 2048),
                &frames,
            );
        }
        for mode in 0..5 {
            let frames = support::fixtures::advanced_frames(stereo, 10, Some(mode));
            compare(
                support::fixtures::config(7, if stereo { 2 } else { 0 }, 3, 2048),
                &frames,
            );
        }
    }
}

#[test]
fn precision_masks_gradients_and_secondary_codebooks_match_c() {
    for gradient in 1..4 {
        for stereo in [false, true] {
            for qbits in 2..=6 {
                let frames =
                    support::fixtures::advanced_frames_with_gradient(stereo, qbits, None, gradient);
                compare(
                    support::fixtures::config(7, if stereo { 2 } else { 0 }, 3, 2048),
                    &frames,
                );
            }
        }
    }
    for gradient in 0..4 {
        compare(
            support::fixtures::config(7, 0, 3, 2048),
            &support::fixtures::spiked_frames(gradient),
        );
    }
}

use std::collections::BTreeSet;
use support::planned::{Bands, Coding, Encoder, Extension, FramePlan, Gradient, ScaleFactors};

const PRIMARY_RAW: Coding = Coding::Raw { bits: 5, base: 0 };
const SECONDARY_COPY: Coding = Coding::DeltaBaseline { base: 0, bits: 1 };

fn plan(bands: Option<Bands>, gradient: Gradient, codings: [Coding; 2]) -> FramePlan {
    FramePlan {
        bands,
        gradient,
        boundary: 0,
        primary: 0,
        signs: false,
        extension: Extension::Absent,
        channels: codings.map(ScaleFactors::coded),
    }
}

/// Encode 16 frames (two eight-frame superframes) and compare them with C.
fn compare_planned(stereo: bool, mut plan_for: impl FnMut(usize) -> FramePlan) -> BTreeSet<String> {
    let mut encoder = Encoder::new(stereo, 8);
    let frames: Vec<_> = (0..16)
        .map(|frame| encoder.frame(&plan_for(frame)))
        .collect();
    compare(
        support::fixtures::config(7, if stereo { 2 } else { 0 }, 3, 2048),
        &frames,
    );
    encoder.used
}

fn assert_used(used: &BTreeSet<String>, expected: &[String]) {
    for name in expected {
        assert!(
            used.contains(name),
            "stream never exercised {name}: {used:?}"
        );
    }
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|&name| name.to_owned()).collect()
}

#[test]
fn every_band_extension_group_and_retained_values_match_c() {
    let mut used = BTreeSet::new();
    for stereo in [false, true] {
        for (band, extension_band, modes) in [
            (5, 12, 4_u32..5),
            (6, 12, 4..5),
            (7, 12, 4..5),
            (8, 12, 0..4),
            (9, 14, 0..4),
            (10, 14, 0..4),
        ] {
            for mode in modes {
                used.extend(compare_planned(stereo, |frame| {
                    let first = frame % 8 == 0;
                    let bands = (first || frame % 8 == 4).then_some(Bands {
                        band,
                        stereo_band: band - 2,
                        extension_band: Some(extension_band),
                    });
                    let codings = if first {
                        [PRIMARY_RAW, SECONDARY_COPY]
                    } else {
                        [
                            Coding::Distance {
                                bits: 5,
                                previous: true,
                            },
                            Coding::Distance {
                                bits: 4,
                                previous: false,
                            },
                        ]
                    };
                    let mut plan = plan(bands, Gradient::flat(12 + (frame % 3) as i32), codings);
                    plan.primary = frame % 2;
                    let value =
                        |index: usize, channel: usize| ((index + channel + frame) % 8) as u32;
                    plan.extension = match frame % 8 {
                        // Retained values, read under another mode where the group has one.
                        3 | 7 => Extension::Bex {
                            modes: [(mode + 1) % 4, (mode + 2) % 4],
                            values: None,
                            extra: 0,
                        },
                        6 => Extension::Absent,
                        _ => Extension::Bex {
                            modes: [mode; 2],
                            values: Some(
                                [0, 1]
                                    .map(|channel| [0, 1, 2, 3].map(|index| value(index, channel))),
                            ),
                            extra: frame % 3,
                        },
                    };
                    plan
                }));
            }
        }
    }
    let mut expected = names(&["retained band extension"]);
    for band in 0..3 {
        expected.push(format!("band extension group {band} mode 4"));
    }
    for band in 3..6 {
        for mode in 0..4 {
            expected.push(format!("band extension group {band} mode {mode}"));
        }
    }
    assert_used(&used, &expected);
}

#[test]
fn extension_data_without_band_extension_is_skipped_like_c() {
    for stereo in [false, true] {
        let used = compare_planned(stereo, |frame| {
            let bands = (frame % 8 == 0).then_some(Bands {
                band: 8,
                stereo_band: 6,
                extension_band: None,
            });
            let mut plan = plan(bands, Gradient::flat(10), [PRIMARY_RAW, SECONDARY_COPY]);
            plan.extension = if frame % 4 == 1 {
                Extension::Absent
            } else {
                Extension::Skipped {
                    mode: (frame % 4) as u32,
                    length: [0, 5, 17, 31][frame % 4],
                }
            };
            plan
        });
        assert_used(&used, &names(&["skipped extension"]));
    }
}

#[test]
fn gradient_curves_and_boundaries_match_c() {
    let gradients = [
        Gradient::Linear {
            start_unit: 2,
            end_unit: 12,
            start_value: 3,
            end_value: 14,
        },
        Gradient::Linear {
            start_unit: 4,
            end_unit: 20,
            start_value: 18,
            end_value: 2,
        },
        Gradient::Linear {
            start_unit: 10,
            end_unit: 31,
            start_value: 5,
            end_value: 25,
        },
        Gradient::Linear {
            start_unit: 0,
            end_unit: 31,
            start_value: 31,
            end_value: 2,
        },
        Gradient::Curve {
            mode: 1,
            start_unit: 4,
            start_value: 2,
        },
        Gradient::Curve {
            mode: 2,
            start_unit: 9,
            start_value: 10,
        },
        Gradient::Curve {
            mode: 3,
            start_unit: 0,
            start_value: 0,
        },
        Gradient::Curve {
            mode: 2,
            start_unit: 20,
            start_value: 4,
        },
    ];
    let mut used = BTreeSet::new();
    for stereo in [false, true] {
        for band in [8, 12] {
            used.extend(compare_planned(stereo, |frame| {
                let bands = (frame % 8 == 0).then_some(Bands {
                    band,
                    stereo_band: band - 1,
                    extension_band: None,
                });
                let codings = if frame % 8 == 0 {
                    [PRIMARY_RAW, SECONDARY_COPY]
                } else {
                    [
                        Coding::Distance {
                            bits: 5,
                            previous: true,
                        },
                        Coding::Distance {
                            bits: 5,
                            previous: false,
                        },
                    ]
                };
                let mut plan = plan(bands, gradients[(frame + band) % gradients.len()], codings);
                plan.boundary = [0, 3, 9, 15][frame % 4];
                plan.primary = (frame / 2) % 2;
                plan.signs = frame % 3 == 0;
                plan
            }));
        }
    }
    assert_used(
        &used,
        &names(&[
            "gradient mode 0 ascending",
            "gradient mode 0 descending",
            "gradient mode 1 ascending",
            "gradient mode 2 ascending",
            "gradient mode 3 ascending",
            "gradient boundary",
            "fine",
        ]),
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Two literal tables of codings.
fn every_scale_factor_coding_matches_c() {
    // Band sizes change inside each superframe, so previous-frame codings also
    // read a raw five-bit tail beyond the previous frame's units.
    let bands = [6, 9, 10, 6, 7, 10, 5, 9];
    let primary = [
        Coding::Raw { bits: 2, base: 13 },
        Coding::Distance {
            bits: 3,
            previous: true,
        },
        Coding::DeltaBaseline { base: -3, bits: 2 },
        Coding::Distance {
            bits: 4,
            previous: true,
        },
        Coding::DeltaBaseline { base: 5, bits: 3 },
        Coding::Distance {
            bits: 5,
            previous: true,
        },
        Coding::Delta {
            weights: 4,
            base: 17,
            bits: 3,
        },
        Coding::DeltaBaseline { base: -1, bits: 4 },
        Coding::Raw { bits: 3, base: 7 },
        Coding::Delta {
            weights: 6,
            base: 12,
            bits: 4,
        },
        Coding::Distance {
            bits: 2,
            previous: true,
        },
        Coding::Delta {
            weights: 2,
            base: 9,
            bits: 5,
        },
        Coding::Raw { bits: 4, base: 2 },
        Coding::DeltaBaseline { base: -16, bits: 4 },
        Coding::Raw { bits: 5, base: 0 },
        Coding::Delta {
            weights: 7,
            base: 20,
            bits: 6,
        },
    ];
    let secondary = [
        Coding::Delta {
            weights: 0,
            base: 14,
            bits: 3,
        },
        Coding::Distance {
            bits: 2,
            previous: false,
        },
        Coding::DeltaBaseline { base: 2, bits: 2 },
        Coding::Distance {
            bits: 3,
            previous: true,
        },
        Coding::Delta {
            weights: 3,
            base: 11,
            bits: 4,
        },
        Coding::Distance {
            bits: 4,
            previous: false,
        },
        Coding::DeltaBaseline { base: -2, bits: 3 },
        Coding::Distance {
            bits: 5,
            previous: true,
        },
        Coding::Delta {
            weights: 5,
            base: 8,
            bits: 5,
        },
        Coding::DeltaBaseline { base: -4, bits: 4 },
        Coding::Distance {
            bits: 5,
            previous: false,
        },
        Coding::DeltaBaseline { base: 0, bits: 1 },
        Coding::Delta {
            weights: 1,
            base: 15,
            bits: 6,
        },
        Coding::Distance {
            bits: 4,
            previous: true,
        },
        Coding::Distance {
            bits: 3,
            previous: false,
        },
        Coding::DeltaBaseline { base: 3, bits: 3 },
    ];
    let mut used = BTreeSet::new();
    for stereo in [false, true] {
        for rotation in 0..4 {
            used.extend(compare_planned(stereo, |frame| {
                let first = frame % 8 == 0;
                let pick = |codings: &[Coding; 16]| codings[(frame * 3 + rotation * 5) % 16];
                let mut codings = [pick(&primary), pick(&secondary)];
                // Superframe starts have no previous frame to refer to.
                if first && !matches!(codings[0], Coding::Raw { .. } | Coding::Delta { .. }) {
                    codings[0] = PRIMARY_RAW;
                }
                if first && matches!(codings[1], Coding::Distance { previous: true, .. }) {
                    codings[1] = SECONDARY_COPY;
                }
                let bands = Bands {
                    band: bands[frame % 8],
                    stereo_band: 4,
                    extension_band: None,
                };
                let mut plan = plan(Some(bands), Gradient::flat(4 + (frame % 5) as i32), codings);
                plan.primary = frame % 2;
                plan
            }));
        }
    }
    assert_used(
        &used,
        &names(&[
            "raw 2 bits base 13",
            "raw 3 bits base 7",
            "raw 4 bits base 2",
            "delta A3 weights 0",
            "delta A3 weights 4",
            "delta A4 weights 6",
            "delta A5 weights 2",
            "delta A6 weights 7",
            "distance to previous B2",
            "distance to previous B3",
            "distance to previous B4",
            "distance to previous B5",
            "distance to previous raw tail",
            "delta to previous A2 base -3",
            "delta to previous A3 base 5",
            "delta to previous A4 base -1",
            "delta to previous raw tail",
            "distance to primary B2",
            "distance to primary B4",
            "delta to primary A2 base 2",
            "delta to primary A4 base -4",
        ]),
    );
}

#[test]
fn second_codebook_pass_and_secondary_families_match_c() {
    // Units 0..12 average 10. Unit 13 is two above its lower neighbour and at
    // the average; unit 16 is two above but below it; unit 21 has sixteen
    // coefficients and sits one below the average. Units 9 and 23 are
    // first-pass spikes.
    let mut values = vec![10; 24];
    for (unit, value) in [
        (9, 14),
        (12, 8),
        (15, 7),
        (16, 9),
        (17, 9),
        (18, 7),
        (19, 7),
        (20, 7),
        (21, 9),
        (22, 9),
        (23, 13),
    ] {
        values[unit] = value;
    }
    let gradients = [13, 8, 4, 7, 9, 3, 12, 6];
    let mut used = BTreeSet::new();
    for stereo in [false, true] {
        used.extend(compare_planned(stereo, |frame| {
            let bands = (frame % 8 == 0).then_some(Bands {
                band: 14,
                stereo_band: 13,
                extension_band: None,
            });
            let mut plan = plan(
                bands,
                Gradient::flat(gradients[frame % 8]),
                [PRIMARY_RAW, SECONDARY_COPY],
            );
            plan.channels[0].values = Some(values.clone());
            plan.primary = frame % 2;
            plan
        }));
    }
    assert_used(
        &used,
        &names(&[
            "second codebook pass at or above average",
            "second codebook pass wide unit below average",
            "SpectrumB22",
            "SpectrumB23",
            "SpectrumB24",
            "SpectrumB72",
            "SpectrumB73",
            "SpectrumB74",
        ]),
    );
}
