//! Synthetic bitstream writer shared by unit tests, integration tests and
//! benchmarks; constants follow `LibAtrac9` tables and unpacking.

pub const LAYOUTS: &[&[u8]] = &[
    &[0],
    &[0, 0],
    &[1],
    &[1, 0, 2, 1],
    &[1, 0, 2, 1, 1],
    &[1, 1],
];
pub const BAND_UNITS: [usize; 19] = [
    0, 4, 8, 10, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30,
];
pub const COEFF_END: [usize; 31] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144,
    160, 176, 192, 208, 224, 240, 256,
];

#[derive(Default)]
pub struct Bits {
    bytes: Vec<u8>,
    position: usize,
}

impl Bits {
    pub fn write(&mut self, value: u32, count: usize) {
        for bit in (0..count).rev() {
            if self.position.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let index = self.position / 8;
            self.bytes[index] |= (((value >> bit) & 1) as u8) << (7 - self.position % 8);
            self.position += 1;
        }
    }

    pub fn align(&mut self) {
        while !self.position.is_multiple_of(8) {
            self.write(0, 1);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.align();
        self.bytes
    }
}

pub fn config(rate: u8, layout: u8, superframe: u8, frame_bytes: usize) -> [u8; 4] {
    let mut b = Bits::default();
    b.write(0xfe, 8);
    b.write(u32::from(rate), 4);
    b.write(u32::from(layout), 3);
    b.write(0, 1);
    b.write((frame_bytes - 1) as u32, 11);
    b.write(u32::from(superframe), 2);
    b.finish().try_into().unwrap()
}

/// Nonzero direct spectra, fine spectra, intensity stereo, band reuse and LFE.
pub fn frames(rate: u8, layout: u8, superframe: u8, scale: u32, seed: u32) -> Vec<Vec<u8>> {
    let count = 1usize << superframe;
    let minimum_band = if rate > 7 { 1 } else { 3 };
    let band = minimum_band + 2;
    let units = BAND_UNITS[band];
    let stereo_units = BAND_UNITS[minimum_band];
    let precision = scale.min(15);
    let fine = scale.saturating_sub(15);
    let mut result = Vec::new();
    for frame in 0..count * 3 {
        let mut bits = Bits::default();
        let first = frame % count == 0;
        for (block_index, &kind) in LAYOUTS[usize::from(layout)].iter().enumerate() {
            bits.write(u32::from(!first), 1);
            bits.write(u32::from(!first), 1);
            if kind == 2 {
                bits.write(scale, 5);
                bits.write(scale, 5);
                for coefficient in 0..4 {
                    bits.write(
                        spectrum(seed, frame, block_index, coefficient, 0),
                        if first { 5 } else { 9 },
                    );
                }
            } else {
                if first {
                    bits.write(2, 4);
                    if kind == 1 {
                        bits.write(0, 4);
                    }
                    bits.write(0, 1); // band extension disabled
                }
                bits.write(0, 2); // gradient mode zero
                bits.write(0, 6);
                bits.write(0, 6); // end unit one
                bits.write(0, 5);
                bits.write(0, 5);
                bits.write(0, 4);
                let primary = frame % 2;
                if kind == 1 {
                    bits.write(primary as u32, 1);
                    bits.write(1, 1);
                    for unit in stereo_units..units {
                        bits.write((unit % 2) as u32, 1);
                    }
                }
                bits.write(0, 1); // extension data absent
                for channel in 0..if kind == 1 { 2 } else { 1 } {
                    if channel == 0 {
                        bits.write(1, 2); // raw 5-bit scale factors
                        bits.write(3, 2);
                        for _ in 0..units {
                            bits.write(scale, 5);
                        }
                    } else {
                        bits.write(2, 2); // delta from primary scale factors
                        bits.write(16, 5); // offset-binary zero
                        bits.write(0, 2); // one-bit unsigned Huffman
                        bits.write(0, 1);
                        for _ in 1..units {
                            bits.write(0, 1);
                        }
                    }
                    let coded_units = if channel == primary || kind == 0 {
                        units
                    } else {
                        stereo_units
                    };
                    let coefficients = COEFF_END[coded_units];
                    for coefficient in 0..coefficients {
                        bits.write(
                            spectrum(seed, frame, block_index, coefficient, channel),
                            precision as usize + 1,
                        );
                    }
                    if fine > 0 {
                        for coefficient in 0..coefficients {
                            bits.write(
                                spectrum(
                                    seed ^ 0x00ab_cdef,
                                    frame,
                                    block_index,
                                    coefficient,
                                    channel,
                                ),
                                fine as usize + 1,
                            );
                        }
                    }
                }
            }
            bits.align();
        }
        result.push(bits.finish());
    }
    result
}

pub fn spectrum(seed: u32, frame: usize, block: usize, coefficient: usize, channel: usize) -> u32 {
    seed.wrapping_mul(747_796_405)
        .wrapping_add(
            frame as u32 * 127 + block as u32 * 31 + coefficient as u32 * 73 + channel as u32 * 83,
        )
        .rotate_left(coefficient as u32 % 31)
}
