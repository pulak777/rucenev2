// Copyright 2019 Zhizhesihai (Beijing) Technology Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// See the License for the specific language governing permissions and
// limitations under the License.

use core::codec::postings::BLOCK_SIZE;
use core::search::NO_MORE_DOCS;
use core::store::io::IndexInput;
use core::util::packed::{SIMD128Packer, SIMDPacker};
use error::Result;
use std::arch::aarch64 as simd;

mod neon {
    use std::arch::aarch64 as simd;

    pub unsafe fn movemask_epi32_neon(input: simd::int32x4_t) -> i32 {
        // Shift the MSB of each 32-bit element to the least significant bit position
        let mask = simd::vshrq_n_s32(input, 31); // Right shift each element by 31 to isolate the MSB

        // Now, the mask contains -1 (0xFFFFFFFF) where the MSB was 1, and 0 where the MSB was 0
        // We need to extract these bits and combine them into a 4-bit mask

        let bit0 = simd::vgetq_lane_s32(mask, 0) & 0x1;
        let bit1 = simd::vgetq_lane_s32(mask, 1) & 0x1;
        let bit2 = simd::vgetq_lane_s32(mask, 2) & 0x1;
        let bit3 = simd::vgetq_lane_s32(mask, 3) & 0x1;

        // Pack the bits into a single integer
        (bit0 << 0) | (bit1 << 1) | (bit2 << 2) | (bit3 << 3)
    }
}

#[repr(align(128))]
struct AlignedBuffer([u32; (BLOCK_SIZE + 4) as usize]);

pub struct SIMDBlockDecoder {
    data: AlignedBuffer,
    encoded: [u8; BLOCK_SIZE as usize * 4],
    packer: SIMD128Packer,
    next_index: usize,
    base_value: i32,
}

impl SIMDBlockDecoder {
    pub fn new() -> Self {
        assert_eq!(BLOCK_SIZE, 128);
        Self {
            data: AlignedBuffer([NO_MORE_DOCS as u32; (BLOCK_SIZE + 4) as usize]),
            encoded: [0u8; BLOCK_SIZE as usize * 4],
            packer: SIMD128Packer::new(BLOCK_SIZE as usize),
            next_index: 0,
            base_value: 0,
        }
    }

    pub fn reset_delta_base(&mut self, base: i32) {
        self.base_value = base;
    }

    #[inline(always)]
    pub fn set_single(&mut self, value: i32) {
        self.next_index = 0;
        let mut v = self.base_value;
        self.data.0[..BLOCK_SIZE as usize].iter_mut().for_each(|e| {
            v += value;
            *e = v as u32;
        });
    }

    pub fn parse_from(
        &mut self,
        input: &mut dyn IndexInput,
        num: usize,
        bits_num: usize,
    ) -> Result<()> {
        input.read_exact(&mut self.encoded[0..num])?;
        self.next_index = 0;
        self.packer.delta_unpack(
            &self.encoded[0..num],
            &mut self.data.0,
            self.base_value as u32,
            bits_num as u8,
        );
        Ok(())
    }

    pub fn parse_from_no_copy(
        &mut self,
        input: &mut dyn IndexInput,
        num: usize,
        bits_num: usize,
    ) -> Result<()> {
        let encoded = unsafe { input.get_and_advance(num) };
        self.next_index = 0;
        self.packer.delta_unpack(
            encoded,
            &mut self.data.0,
            self.base_value as u32,
            bits_num as u8,
        );
        Ok(())
    }

    #[inline(always)]
    pub fn next(&mut self) -> i32 {
        let pos = self.next_index;
        self.next_index += 1;
        self.data.0[pos] as i32
    }

    #[inline(always)]
    pub fn advance(&mut self, target: i32) -> (i32, usize) {
        unsafe {
            let input = self.data.0.as_ptr() as *const simd::int32x4_t;
            let target = simd::vdupq_n_s32(target);
            let mut count = simd::vdupq_n_s32(0);
            unroll! {
                for i in 0..8 {
                    let r1 = simd::vreinterpretq_s32_u32(simd::vcltq_s32(simd::vld1q_s32(input.add(i * 4) as *const i32), target));
                    let r2 = simd::vreinterpretq_s32_u32(simd::vcltq_s32(simd::vld1q_s32(input.add(i * 4 + 1) as *const i32), target));
                    let r3 = simd::vreinterpretq_s32_u32(simd::vcltq_s32(simd::vld1q_s32(input.add(i * 4 + 2) as *const i32), target));
                    let r4 = simd::vreinterpretq_s32_u32(simd::vcltq_s32(simd::vld1q_s32(input.add(i * 4 + 3) as *const i32), target));
                    let sum = simd::vaddq_s32(
                        simd::vaddq_s32(r1, r2),
                        simd::vaddq_s32(r3, r4)
                    );
                    count = simd::vsubq_s32(count, sum);
                }
            };
            let count = simd::vaddq_s32(count, simd::vextq_s32(count, count, 2));
            let count = simd::vaddq_s32(count, simd::vextq_s32(count, count, 1));
            let count = simd::vgetq_lane_s32(count, 0) as usize;
            self.next_index = count + 1;
            (self.data.0[count] as i32, count)
        }
    }

    #[inline(always)]
    pub fn advance_by_partial(&mut self, target: i32) -> (i32, usize) {
        let mut index = self.next_index & 0xFCusize;
        let mut input = self.data.0[index..].as_ptr() as *const simd::int32x4_t;
        unsafe {
            let target = simd::vdupq_n_s32(target);
            while index < 128 {
                let res = simd::vreinterpretq_s32_u32(simd::vcltq_s32(
                    simd::vld1q_s32(input as *const i32),
                    target,
                ));
                let res = neon::movemask_epi32_neon(res);
                if res != 0xFFFF {
                    index += ((32 - res.leading_zeros()) >> 2) as usize;
                    break;
                } else {
                    index += 4;
                    input = input.add(1);
                }
            }
            self.next_index = index + 1;
            (self.data.0[index] as i32, index)
        }
    }

    #[inline(always)]
    pub fn advance_by_binary_search(&mut self, target: i32) -> (i32, usize) {
        match self.data.0[self.next_index..BLOCK_SIZE as usize].binary_search(&(target as u32)) {
            Ok(p) | Err(p) => {
                let pos = self.next_index + p;
                self.next_index = pos + 1;
                (self.data.0[pos] as i32, pos)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::codec::postings::SIMDBlockDecoder;

    #[test]
    fn test_simd_advance() {
        let mut decoder = SIMDBlockDecoder::new();
        let mut i = 0;
        decoder.data.0.iter_mut().for_each(|e| {
            i += 128;
            *e = i;
        });
        assert_eq!(decoder.advance(1).0, 128);
        assert_eq!(decoder.advance(129).0, 256);
        assert_eq!(decoder.advance(130).0, 256);
        assert_eq!(decoder.advance(255).0, 256);
        assert_eq!(decoder.advance(256).0, 256);
        assert_eq!(decoder.advance(257).0, 384);
        assert_eq!(decoder.next(), 512);
        assert_eq!(decoder.advance(16283).0, 16384);
    }
    #[test]
    fn test_binary_search() {
        let mut decoder = SIMDBlockDecoder::new();
        let mut i = 0;
        decoder.data.0.iter_mut().for_each(|e| {
            i += 128;
            *e = i;
        });
        assert_eq!(decoder.advance_by_binary_search(1), (128, 0));
        assert_eq!(decoder.advance_by_binary_search(129), (256, 1));
        assert_eq!(decoder.advance_by_binary_search(512), (512, 3));
    }
}
