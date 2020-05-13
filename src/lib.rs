#![feature(test)]

use cfg_if::cfg_if;
use packed_simd::i32x8;

#[inline]
pub fn sum_scalar(x: &[i32]) -> i32 {
    debug_assert_eq!(x.len() % 8, 0);
    x.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn sum_simd_avx2(x: &[i32]) -> i32 {
    debug_assert!(std::is_x86_feature_detected!("avx2"));
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
                if std::is_x86_feature_detected!("avx2") {
                    // println!("sum_simd_avx2");
                    return sum_simd_avx2(x)
                }
            }
        }
    }

    // println!("sum_scalar");
    sum_scalar(x)
}

#[cfg(test)]
mod tests {
    extern crate test;

    use super::*;
    use test::Bencher;

    #[test]
    fn test_scalar() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(sum_scalar(&x), 28);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_avx2() {
        if std::is_x86_feature_detected!("avx2") {
            let x = [0, 1, 2, 3, 4, 5, 6, 7];
            assert_eq!(sum_simd_avx2(&x), 28);
        }
    }

    #[test]
    fn test_sum() {
        let x = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(sum(&x, true), 28);
        assert_eq!(sum(&x, false), 28);
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
    fn bench_avx2(b: &mut Bencher) {
        let x = build_vector(6400);
        b.iter(|| {
            test::black_box(sum_simd_avx2(test::black_box(&x)));
        });
    }

    #[bench]
    fn bench_sum_true(b: &mut Bencher) {
        let x = build_vector(6400);
        b.iter(|| {
            test::black_box(sum(test::black_box(&x), true));
        });
    }

    #[bench]
    fn bench_sum_false(b: &mut Bencher) {
        let x = build_vector(6400);
        b.iter(|| {
            test::black_box(sum(test::black_box(&x), false));
        });
    }
}
