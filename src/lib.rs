#![feature(test)]

use packed_simd::i32x8;

#[inline]
pub fn sum_scalar(x: &[i32]) -> i32 {
    debug_assert_eq!(x.len() % 8, 0);
    x.iter().sum()
}

#[inline]
pub fn sum_simd(x: &[i32]) -> i32 {
    debug_assert_eq!(x.len() % 8, 0);
    x.chunks_exact(i32x8::lanes())
        .map(i32x8::from_slice_unaligned)
        .sum::<i32x8>()
        .wrapping_sum()
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn scalar() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(sum_scalar(&x), 28);
    }

    #[test]
    fn simd() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(sum_simd(&x), 28);
    }

    fn build_vector(len: u32) -> Vec<i32> {
        debug_assert_eq!(len % 8, 0);
        (0..len as i32).collect()
    }

    #[bench]
    fn bench_scalar(b: &mut Bencher) {
        let x = build_vector(6400);
        b.iter(|| {
            test::black_box(sum_scalar(test::black_box(&x)));
        });
    }

    #[bench]
    fn bench_simd(b: &mut Bencher) {
        let x = build_vector(6400);
        b.iter(|| {
            test::black_box(sum_simd(test::black_box(&x)));
        });
    }
}
