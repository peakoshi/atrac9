//! Bounded, MSB-first ATRAC9 bit reader.
use crate::Error;

/// Reads from a 64-bit reservoir refilled several bytes at a time. Bits past
/// the end of the input read as zero, so prefix lookahead near the end of a
/// frame needs no padding, but consuming them is always an error.
#[derive(Debug)]
pub(crate) struct BitReader<'a> {
    input: &'a [u8],
    /// Index of the next input byte not yet counted in `available`.
    next: usize,
    /// Unread bits, MSB first. Bits below `available` are either zero or the
    /// input bits that follow, so lookahead never sees stale data.
    cache: u64,
    available: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) const fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            next: 0,
            cache: 0,
            available: 0,
        }
    }

    pub(crate) const fn position(&self) -> usize {
        self.next * 8 - self.available
    }

    /// Tops the reservoir up to at least 56 bits, or to the end of the input.
    /// Callers only refill below 32 available bits.
    #[inline]
    fn refill(&mut self) {
        debug_assert!(self.available < 32);
        if let Some(bytes) = self
            .input
            .get(self.next..)
            .and_then(<[u8]>::first_chunk::<8>)
        {
            self.cache |= u64::from_be_bytes(*bytes) >> self.available;
            let bytes = (63 - self.available) / 8;
            self.next += bytes;
            self.available += bytes * 8;
        } else {
            self.refill_tail();
        }
    }

    #[cold]
    fn refill_tail(&mut self) {
        while self.available <= 56 {
            let Some(&byte) = self.input.get(self.next) else {
                break;
            };
            self.cache |= u64::from(byte) << (56 - self.available);
            self.next += 1;
            self.available += 8;
        }
    }

    /// The next `bits` (at most 32) without consuming them, zero padded past
    /// the end of the input.
    #[inline]
    pub(crate) fn peek(&mut self, bits: usize) -> u32 {
        debug_assert!(bits <= 32);
        if self.available < bits {
            self.refill();
        }
        #[allow(clippy::cast_possible_truncation)] // At most 32 bits remain.
        let value = (self.cache >> 32 >> (32 - bits)) as u32;
        value
    }

    // Prefix lookahead permits zero padding, but consuming the selected code
    // still checks its actual length. A short final code needs no extra byte.
    #[inline]
    pub(crate) fn peek_padded(&mut self, bits: usize) -> Result<u32, Error> {
        if bits > 24 {
            return Err(Error::invalid("bit width"));
        }
        Ok(self.peek(bits))
    }

    #[inline]
    pub(crate) fn skip(&mut self, bits: usize) -> Result<(), Error> {
        if bits > self.available {
            if bits >= 32 {
                return self.seek(bits);
            }
            self.refill();
            if bits > self.available {
                return Err(Error::Truncated);
            }
        }
        // `bits` is at most `self.available` (at most 64) here.
        #[allow(clippy::cast_possible_truncation)]
        let shift = bits as u32;
        self.cache = self.cache.checked_shl(shift).unwrap_or(0);
        self.available -= bits;
        Ok(())
    }

    #[cold]
    fn seek(&mut self, bits: usize) -> Result<(), Error> {
        let end = self.position().checked_add(bits).ok_or(Error::Truncated)?;
        if end > self.input.len().saturating_mul(8) {
            return Err(Error::Truncated);
        }
        self.next = end / 8;
        self.cache = 0;
        self.available = 0;
        self.refill();
        self.skip(end % 8)
    }

    #[inline]
    pub(crate) fn read(&mut self, bits: usize) -> Result<u32, Error> {
        let value = self.peek_padded(bits)?;
        self.skip(bits)?;
        Ok(value)
    }

    #[cfg(test)]
    pub(crate) fn read_signed(&mut self, bits: usize) -> Result<i32, Error> {
        let value = self.read(bits)?;
        Ok(sign_extend(value, bits))
    }

    /// Fills `output` with consecutive signed `bits`-wide values (1 to 32).
    #[inline]
    pub(crate) fn read_signed_into(
        &mut self,
        bits: usize,
        output: &mut [i32],
    ) -> Result<(), Error> {
        debug_assert!((1..=32).contains(&bits));
        for value in output {
            if self.available < bits {
                self.refill();
                if self.available < bits {
                    return Err(Error::Truncated);
                }
            }
            // An arithmetic shift of the top bits sign-extends them.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            {
                *value = ((self.cache as i64) >> (64 - bits)) as i32;
            }
            self.cache <<= bits;
            self.available -= bits;
        }
        Ok(())
    }

    pub(crate) fn read_offset_binary(&mut self, bits: usize) -> Result<i32, Error> {
        if bits == 0 {
            return Err(Error::invalid("offset binary width"));
        }
        let value = self.read_signed_unsigned(bits)?;
        Ok(value - (1 << (bits - 1)))
    }

    fn read_signed_unsigned(&mut self, bits: usize) -> Result<i32, Error> {
        i32::try_from(self.read(bits)?).map_err(|_| Error::invalid("integer width"))
    }

    pub(crate) fn align(&mut self) -> Result<(), Error> {
        self.skip(self.available % 8)
    }
}

#[allow(clippy::cast_possible_wrap)] // Deliberate two's-complement sign extension.
pub(crate) fn sign_extend(value: u32, bits: usize) -> i32 {
    if bits == 0 {
        return 0;
    }
    ((value << (32 - bits)) as i32) >> (32 - bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    #[test]
    fn boundaries() {
        let mut br = BitReader::new(&[0b1011_0001]);
        assert_eq!(br.read(3), Ok(5));
        assert_eq!(br.read_signed(3), Ok(-4));
        assert_eq!(br.read(2), Ok(1));
        assert_eq!(br.read(1), Err(Error::Truncated));
        assert_eq!(br.position(), 8);
    }

    #[test]
    fn long_skips_and_alignment_keep_the_bit_position() {
        let bytes: Vec<u8> = (0..40).collect();
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.read(3), Ok(0));
        br.align().unwrap();
        assert_eq!(br.position(), 8);
        br.skip(31).unwrap();
        assert_eq!(br.position(), 39);
        br.skip(100).unwrap();
        assert_eq!(br.position(), 139);
        // Bits 139..147 span bytes 17 (0x11) and 18 (0x12).
        assert_eq!(br.read(8), Ok(0b1000_1000));
        br.align().unwrap();
        assert_eq!(br.position(), 152);
        assert_eq!(br.skip(320 - 152 + 1), Err(Error::Truncated));
        assert_eq!(br.position(), 152);
        br.skip(320 - 152).unwrap();
        assert_eq!(br.peek_padded(24), Ok(0));
        assert_eq!(br.skip(1), Err(Error::Truncated));
        assert_eq!(br.skip(usize::MAX), Err(Error::Truncated));
    }

    proptest! {
        #[test]
        fn agrees_with_bitwise_reference(bytes in prop::collection::vec(any::<u8>(), 0..100), widths in prop::collection::vec(0usize..25, 0..100)) {
            let mut reader = BitReader::new(&bytes);
            let mut pos = 0;
            for width in widths {
                if pos + width > bytes.len() * 8 {
                    prop_assert_eq!(reader.read(width), Err(Error::Truncated));
                    break;
                }
                let expected = (pos..pos+width).fold(0u32, |a,b| (a<<1) | u32::from((bytes[b/8]>>(7-b%8))&1));
                prop_assert_eq!(reader.read(width), Ok(expected));
                pos += width;
            }
        }

        #[test]
        fn signed_runs_match_single_reads(bytes in prop::collection::vec(any::<u8>(), 0..64),
            lead in 0usize..8, bits in 1usize..17, count in 0usize..40) {
            let mut single = BitReader::new(&bytes);
            let mut run = BitReader::new(&bytes);
            prop_assert_eq!(single.skip(lead), run.skip(lead));
            let mut expected = vec![0; count];
            let mut single_result = Ok(());
            for value in &mut expected {
                match single.read_signed(bits) {
                    Ok(read) => *value = read,
                    Err(error) => { single_result = Err(error); break; }
                }
            }
            let mut actual = vec![0; count];
            prop_assert_eq!(run.read_signed_into(bits, &mut actual), single_result);
            if single_result.is_ok() {
                prop_assert_eq!(actual, expected);
                prop_assert_eq!(run.position(), single.position());
            }
        }

        #[test]
        fn padded_lookahead_and_skips_agree_with_bitwise_reference(
            bytes in prop::collection::vec(any::<u8>(), 0..64),
            steps in prop::collection::vec((0usize..25, 0usize..80, any::<bool>()), 0..64)) {
            let bit = |index: usize| bytes.get(index / 8).map_or(0, |byte| u32::from(byte >> (7 - index % 8)) & 1);
            let mut reader = BitReader::new(&bytes);
            let mut pos = 0;
            for (peek, skip, align) in steps {
                let expected = (pos..pos + peek).fold(0, |a, b| (a << 1) | bit(b));
                prop_assert_eq!(reader.peek_padded(peek), Ok(expected));
                let step = if align { (8 - pos % 8) % 8 } else { skip };
                let result = if align { reader.align() } else { reader.skip(skip) };
                if pos + step > bytes.len() * 8 {
                    prop_assert_eq!(result, Err(Error::Truncated));
                } else {
                    prop_assert_eq!(result, Ok(()));
                    pos += step;
                }
                prop_assert_eq!(reader.position(), pos);
            }
        }
    }
}
