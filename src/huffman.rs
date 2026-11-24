//! Frozen prefix tables: no initialization, allocation or synchronization.
use crate::{
    Error,
    bits::{BitReader, sign_extend},
};
mod data;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Codebook {
    lookup: &'static [u16],
    max_bits: usize,
    pub(crate) value_count_power: usize,
    pub(crate) value_bits: usize,
    pub(crate) value_max: i32,
}

impl Codebook {
    #[inline]
    pub(crate) fn read_value(&self, br: &mut BitReader<'_>, signed: bool) -> Result<i32, Error> {
        // Frozen tables are at most ten bits deep.
        let prefix = br.peek(self.max_bits) as usize;
        let entry = *self
            .lookup
            .get(prefix)
            .ok_or_else(|| Error::invalid("Huffman codebook"))?;
        let bits = usize::from(entry >> 8);
        if bits == 0 {
            return Err(Error::invalid("Huffman prefix"));
        }
        br.skip(bits)?;
        let value = u32::from(entry & 255);
        Ok(if signed {
            sign_extend(value, self.value_bits)
        } else {
            i32::from(entry & 255)
        })
    }
}

pub(crate) fn scale_factor_unsigned(bits: usize) -> &'static Codebook {
    &data::UNSIGNED[bits]
}
pub(crate) fn scale_factor_signed(bits: usize) -> &'static Codebook {
    &data::SIGNED[bits]
}
pub(crate) fn spectrum(set: usize, precision: usize, index: usize) -> &'static Codebook {
    &data::SPECTRUM[set][precision][index]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_prefix_consumes_only_its_code() {
        for book in data::UNSIGNED
            .iter()
            .chain(data::SIGNED.iter())
            .chain(data::SPECTRUM.iter().flatten().flatten())
        {
            for (prefix, &entry) in book.lookup.iter().enumerate() {
                if entry == 0 {
                    continue;
                }
                let word = u32::try_from(prefix).expect("bounded prefix") << (32 - book.max_bits);
                let bytes = word.to_be_bytes();
                let mut reader = BitReader::new(&bytes);
                assert_eq!(
                    book.read_value(&mut reader, false),
                    Ok(i32::from(entry & 255))
                );
                assert_eq!(reader.position(), usize::from(entry >> 8));
            }
        }
    }
}
