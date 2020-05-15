#![feature(test)]

use cfg_if::cfg_if;

#[inline]
pub fn sum_scalar(x: &[i32]) -> i32 {
    debug_assert_eq!(x.len() % 8, 0);
    x.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn sum_avx2(x: &[i32]) -> i32 {
    use packed_simd::i32x8;
    debug_assert!(is_x86_feature_detected!("avx2"));
    debug_assert_eq!(x.len() % 8, 0);
    x.chunks_exact(i32x8::lanes())
        .map(i32x8::from_slice_unaligned)
        .sum::<i32x8>()
        .wrapping_sum()
}

#[inline]
pub fn sum(x: &[i32], simd_preferred: bool) -> i32 {
    #[allow(clippy::collapsible_if)]
    if simd_preferred {
        cfg_if! {
            if #[cfg(target_arch = "x86_64")] {
                if is_x86_feature_detected!("avx2") {
                    // println!("sum_simd_avx2");
                    return sum_avx2(x)
                }
            }
        }
    }

    // println!("sum_scalar");
    sum_scalar(x)
}

#[inline]
pub fn bit_count_u8_scalar(u: u8) -> u32 {
    u.count_ones()
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn bit_count_u8_avx2(u: u8) -> u32 {
    use packed_simd::u8x2;
    debug_assert!(is_x86_feature_detected!("avx2"));
    let s = u8x2::new(u, 0);
    unsafe { s.count_ones().extract_unchecked(0) as u32 }
}

#[inline]
pub fn bit_count_u8_slice_scalar(u: &[u8]) -> u32 {
    u.iter().fold(0, |count, &u| count + u.count_ones())
}

static BIT_COUNT_TABLE: [u8; 256] = [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
];

#[inline]
pub fn bit_count_u8_slice_table(u: &[u8]) -> u32 {
    u.iter()
        .fold(0, |count, &u| count + BIT_COUNT_TABLE[u as usize] as u32)
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn bit_count_u8_slice_avx2(u: &[u8]) -> u32 {
    use packed_simd::{u16x16, u8x16, u8x32};
    use std::mem::MaybeUninit;
    debug_assert!(is_x86_feature_detected!("avx2"));

    if u.len() < 256 {
        let chunks = u.chunks_exact(u8x32::lanes());
        let mut sum = bit_count_u8_slice_table(chunks.remainder());
        sum += chunks
            .map(|s| unsafe { u8x32::from_slice_unaligned_unchecked(s).count_ones() })
            .sum::<u8x32>()
            .wrapping_sum() as u32;
        sum
    } else if u.len() < 1024 * 8 * 16 {
        let mut buf = unsafe { MaybeUninit::<[u16; 16]>::uninit().assume_init() };
        let chunks = u.chunks_exact(u8x16::lanes());
        let mut sum = bit_count_u8_slice_table(chunks.remainder());
        let result = chunks
            .map(|s| unsafe { u8x16::from_slice_unaligned_unchecked(s).count_ones() })
            .map(Into::<u16x16>::into)
            .sum::<u16x16>();
        unsafe {
            result.write_to_slice_unaligned_unchecked(&mut buf);
        }

        sum += buf.iter().fold(0, |count, &u| count + u as u32);
        sum
    } else {
        // Sum may be overflowed when length of `u` >= 128K, so panic!
        panic!("u.len()[{}] >= 1024 * 8 * 16", u.len())
    }
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn test_sum_scalar() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(sum_scalar(&x), 28);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_sum_avx2() {
        if std::is_x86_feature_detected!("avx2") {
            let x = [0, 1, 2, 3, 4, 5, 6, 7];
            assert_eq!(sum_avx2(&x), 28);
        }
    }

    #[test]
    fn test_sum() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(sum(&x, true), 28);
        assert_eq!(sum(&x, false), 28);
    }

    #[test]
    fn test_bit_count_u8_scalar() {
        let slow_bit_count = |n: u8| -> u32 {
            let mut count = 0;
            let mut i = n;
            while i != 0 {
                count += (i & 1) as u32;
                i = i >> 1;
            }
            count
        };

        for i in 0..=255u8 {
            assert_eq!(bit_count_u8_scalar(i), slow_bit_count(i));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_bit_count_u8_avx2() {
        for i in 0..=255u8 {
            assert_eq!(bit_count_u8_avx2(i), bit_count_u8_scalar(i));
        }
    }

    #[test]
    fn test_bit_count_u8_slice_scalar() {
        let slow_bit_count =
            |n: &[u8]| -> u32 { n.iter().fold(0, |count, &u| count + bit_count_u8_scalar(u)) };

        #[rustfmt::skip]
        let s = [
            0u8, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        ];

        assert_eq!(bit_count_u8_slice_scalar(&s), slow_bit_count(&s));
    }

    #[test]
    fn test_bit_count_u8_slice_table() {
        let s = build_u8_vector(256);
        assert_eq!(bit_count_u8_slice_table(&s), bit_count_u8_slice_scalar(&s));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_bit_count_u8_slice_avx2() {
        for n in (1u32..128) {
            let s = build_u8_vector(n);
            println!("{}: avx2: {}, scalar: {}", n, bit_count_u8_slice_avx2(&s), bit_count_u8_slice_scalar(&s));
            assert_eq!(bit_count_u8_slice_avx2(&s), bit_count_u8_slice_scalar(&s));
        }
    }

    fn build_i32_vector(len: u32) -> Vec<i32> {
        debug_assert_eq!(len % 8, 0);
        (0..len as i32).collect()
    }

    fn build_u8_vector(len: u32) -> Vec<u8> {
        vec![255; len as usize]
    }

    #[bench]
    fn bench_sum_scalar(b: &mut Bencher) {
        let x = build_i32_vector(6400);
        b.iter(|| {
            test::black_box(sum_scalar(test::black_box(&x)));
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[bench]
    fn bench_sum_avx2(b: &mut Bencher) {
        let x = build_i32_vector(6400);
        b.iter(|| {
            test::black_box(sum_avx2(test::black_box(&x)));
        });
    }

    #[bench]
    fn bench_sum_true(b: &mut Bencher) {
        let x = build_i32_vector(6400);
        b.iter(|| {
            test::black_box(sum(test::black_box(&x), true));
        });
    }

    #[bench]
    fn bench_sum_false(b: &mut Bencher) {
        let x = build_i32_vector(6400);
        b.iter(|| {
            test::black_box(sum(test::black_box(&x), false));
        });
    }

    #[bench]
    fn bench_bit_count_u8_scalar(b: &mut Bencher) {
        let u = 255u8;
        b.iter(|| {
            test::black_box(bit_count_u8_scalar(test::black_box(u)));
        })
    }

    #[cfg(target_arch = "x86_64")]
    #[bench]
    fn bench_bit_count_u8_avx2(b: &mut Bencher) {
        let u = 255u8;
        b.iter(|| {
            test::black_box(bit_count_u8_avx2(test::black_box(u)));
        })
    }

    #[bench]
    fn bench_bit_count_u8_slice_scalar(b: &mut Bencher) {
        let x = build_u8_vector(256);
        b.iter(|| {
            test::black_box(bit_count_u8_slice_scalar(test::black_box(&x)));
        });
    }

    #[bench]
    fn bench_bit_count_u8_slice_table(b: &mut Bencher) {
        let x = build_u8_vector(256);
        b.iter(|| {
            test::black_box(bit_count_u8_slice_table(test::black_box(&x)));
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[bench]
    fn bench_bit_count_u8_slice_avx2(b: &mut Bencher) {
        let x = build_u8_vector(256);
        b.iter(|| {
            test::black_box(bit_count_u8_slice_avx2(test::black_box(&x)));
        });
    }
}
