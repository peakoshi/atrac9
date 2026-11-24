//! Huffman, scale-factor, gradient and band-extension streams; the generic
//! writer lives in `src/test_fixtures.rs` so the unit tests can use it too.

#[path = "../../src/test_fixtures.rs"]
mod synthetic;
#[allow(unused_imports)]
pub use synthetic::{BAND_UNITS, Bits, COEFF_END, config, frames, spectrum};

/// Stereo/mono Huffman spectra, all scale factor modes, and BEX modes 0..4.
pub fn advanced_frames(stereo: bool, qbits: usize, bex_mode: Option<usize>) -> Vec<Vec<u8>> {
    advanced_frames_with_gradient(stereo, qbits, bex_mode, 0)
}

#[allow(clippy::too_many_lines)] // Linear synthetic bitstream layout aids oracle review.
pub fn advanced_frames_with_gradient(
    stereo: bool,
    qbits: usize,
    bex_mode: Option<usize>,
    gradient_mode: u32,
) -> Vec<Vec<u8>> {
    let band = if bex_mode == Some(4) { 5 } else { 8 };
    let units = BAND_UNITS[band];
    let extension_units = if bex_mode.is_some() { 22 } else { units };
    let scale = 20;
    let precision = qbits as u32 - 1;
    let gradient = scale
        - match gradient_mode {
            0 => precision,
            1 => precision * 2,
            2 => (precision * 8).div_ceil(3),
            3 => precision * 4,
            _ => unreachable!(),
        };
    let channels = if stereo { 2 } else { 1 };
    let mut result = Vec::new();
    for frame in 0..24 {
        let first = frame % 8 == 0;
        let mut b = Bits::default();
        b.write(u32::from(!first), 1);
        b.write(u32::from(!first), 1);
        if first {
            b.write((band - 3) as u32, 4);
            if stereo {
                b.write((band - 3) as u32, 4);
            }
            b.write(u32::from(bex_mode.is_some()), 1);
            if bex_mode.is_some() {
                b.write(9, 4);
            }
        }
        b.write(gradient_mode, 2);
        if gradient_mode == 0 {
            b.write(0, 6);
            b.write(0, 6);
            b.write(gradient, 5);
            b.write(gradient, 5);
        } else {
            b.write(31, 5); // all coded units precede gradient transition
            b.write(gradient, 5);
        }
        b.write(0, 4);
        if stereo {
            b.write((frame % 2) as u32, 1);
            b.write(0, 1);
        }
        if let Some(mode) = bex_mode {
            b.write(
                if mode == 4 { 0 } else { mode as u32 },
                if stereo { 2 } else { 1 },
            );
            b.write(1, 1);
            b.write(if mode == 4 { 0 } else { mode as u32 }, 2);
            let lengths: &[usize] = match mode {
                0 => &[5, 4, 3, 3],
                1 => &[],
                2 => &[6, 6],
                3 => &[4, 4],
                4 => &[3],
                _ => unreachable!(),
            };
            let length = lengths.iter().sum::<usize>() * channels;
            b.write(length as u32, 5);
            for channel in 0..channels {
                for (index, &length) in lengths.iter().enumerate() {
                    b.write((index + channel + frame) as u32, length);
                }
            }
        } else {
            b.write(0, 1);
        }
        for channel in 0..channels {
            let mode = if first {
                usize::from(channel != 0)
            } else if channel == 0 {
                2 + frame % 2
            } else {
                1 + frame % 3
            };
            b.write(mode as u32, 2);
            match (channel, mode) {
                (0, 0) => {
                    let weights = [
                        3, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
                    ];
                    b.write(1, 3);
                    b.write(scale, 5);
                    b.write(3, 2);
                    b.write(weights[0], 6);
                    for unit in 1..extension_units {
                        emit_huffman(
                            &mut b,
                            "ScaleFactorsA6",
                            weights[unit].wrapping_sub(weights[unit - 1]) as usize & 63,
                        );
                    }
                }
                (0, 2) | (1, 1 | 3) => {
                    b.write(0, 2);
                    for _ in 0..extension_units {
                        emit_huffman(&mut b, "ScaleFactorsB2", 0);
                    }
                }
                (0, 3) | (1, 2) => {
                    b.write(16, 5);
                    b.write(0, 2);
                    b.write(0, 1);
                    for _ in 1..extension_units {
                        b.write(0, 1);
                    }
                }
                _ => unreachable!(),
            }
            for unit in 0..units {
                let count = COEFF_END[unit + 1] - COEFF_END[unit];
                if qbits <= 7 {
                    let book = if unit < 8 {
                        1
                    } else if unit < 12 {
                        2
                    } else if unit < 20 {
                        3
                    } else {
                        4
                    };
                    let name = format!("SpectrumA{qbits}{book}");
                    let bits = c_array(&format!("{name}Bits"));
                    let valid: Vec<usize> = bits
                        .iter()
                        .enumerate()
                        .filter_map(|(index, &length)| (length > 0).then_some(index))
                        .collect();
                    let values = if qbits == 2 && unit >= 8 {
                        4
                    } else if qbits <= 4 {
                        2
                    } else {
                        1
                    };
                    for group in 0..count / values {
                        emit_huffman(
                            &mut b,
                            &name,
                            valid[(frame * 7 + channel * 5 + unit * 3 + group) % valid.len()],
                        );
                    }
                } else {
                    for coefficient in COEFF_END[unit]..COEFF_END[unit + 1] {
                        b.write(spectrum(41, frame, 0, coefficient, channel), qbits);
                    }
                }
            }
        }
        result.push(b.finish());
    }
    result
}

/// Alternating scale factors trigger the second spectrum codebook family.
pub fn spiked_frames(gradient_mode: u32) -> Vec<Vec<u8>> {
    let mut result = Vec::new();
    for frame in 0..24 {
        let first = frame % 8 == 0;
        let mut b = Bits::default();
        b.write(u32::from(!first), 1);
        b.write(u32::from(!first), 1);
        if first {
            b.write(9, 4);
            b.write(0, 1);
        }
        b.write(gradient_mode, 2);
        if gradient_mode == 0 {
            b.write(0, 6);
            b.write(0, 6);
            b.write(18, 5);
            b.write(18, 5);
        } else {
            b.write(31, 5);
            b.write(18, 5);
        }
        b.write(0, 4);
        b.write(0, 1);
        b.write(1, 2);
        b.write(3, 2);
        for unit in 0..22 {
            b.write(if unit >= 8 && unit % 2 == 0 { 23 } else { 20 }, 5);
        }
        for unit in 0..22 {
            let spike = unit >= 8 && unit % 2 == 0;
            // Each high unit gets mask 2 from both adjacent scale deltas.
            let base = if spike { 9usize } else { 2 };
            let precision = if gradient_mode == 0 {
                if spike { 5 } else { 2 }
            } else {
                match gradient_mode {
                    1 => base / 2,
                    2 => base * 3 / 8,
                    3 => base / 4,
                    _ => unreachable!(),
                }
                .max(1)
            };
            let qbits = precision + 1;
            let book = if unit < 8 {
                1
            } else if unit < 12 {
                2
            } else if unit < 20 {
                3
            } else {
                4
            };
            let family = if spike { 'B' } else { 'A' };
            let name = format!("Spectrum{family}{qbits}{book}");
            let lengths = c_array(&format!("{name}Bits"));
            let valid: Vec<usize> = lengths
                .iter()
                .enumerate()
                .filter_map(|(index, &length)| (length > 0).then_some(index))
                .collect();
            let values = if qbits == 2 && unit >= 8 {
                4
            } else if qbits <= 4 {
                2
            } else {
                1
            };
            let group_count = (COEFF_END[unit + 1] - COEFF_END[unit]) / values;
            for group in 0..group_count {
                emit_huffman(
                    &mut b,
                    &name,
                    valid[(frame * 7 + unit * 3 + group) % valid.len()],
                );
            }
        }
        result.push(b.finish());
    }
    result
}

/// A one-dimensional array from the C Huffman tables, parsed once per name.
pub fn c_array(name: &str) -> Vec<u32> {
    thread_local! {
        static CACHE: std::cell::RefCell<std::collections::HashMap<String, Vec<u32>>> =
            std::cell::RefCell::default();
    }
    CACHE.with(|cache| {
        cache
            .borrow_mut()
            .entry(name.to_owned())
            .or_insert_with(|| {
                let source = include_str!("../reference/huffCodes.c");
                let tail = source.split_once(&format!("{name}[")).unwrap().1;
                parse_numbers(tail.split_once('{').unwrap().1.split_once('}').unwrap().0)
            })
            .clone()
    })
}

/// A possibly multi-dimensional C table, flattened in row order.
pub fn c_table(file: &str, name: &str) -> Vec<u32> {
    let source = match file {
        "tables.c" => include_str!("../reference/tables.c"),
        "band_extension.c" => include_str!("../reference/band_extension.c"),
        _ => panic!("unknown reference file {file}"),
    };
    let tail = source.split_once(&format!("{name}[")).unwrap().1;
    let body = tail.split_once('{').unwrap().1.split_once("};").unwrap().0;
    parse_numbers(&body.replace(['{', '}'], ","))
}

fn parse_numbers(content: &str) -> Vec<u32> {
    content
        .split(',')
        .filter_map(|value| {
            let value = value.trim();
            if value.is_empty() {
                return None;
            }
            Some(if let Some(hex) = value.strip_prefix("0x") {
                u32::from_str_radix(hex, 16).unwrap()
            } else {
                value.parse().unwrap()
            })
        })
        .collect()
}

pub fn emit_huffman(bits: &mut Bits, name: &str, value: usize) {
    let codes = c_array(&format!("{name}Codes"));
    let lengths = c_array(&format!("{name}Bits"));
    assert!(lengths[value] > 0, "{name} has no code for {value}");
    bits.write(codes[value], lengths[value] as usize);
}
