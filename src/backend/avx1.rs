//! Hand-written AVX (AVX1) backend for `x86_64` (8-wide f32, 4-wide f64), `f32`/`f64` only.
//! No FMA at this tier: [`fma`](Backend::fma) is an unfused multiply then add.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::backend::Backend;

/// AVX (AVX1) execution token. Zero-sized; only construct it when the CPU supports `avx`
/// (see [`Avx1::detect`]).
#[derive(Clone, Copy, Debug)]
pub struct Avx1(());

impl Avx1 {
    /// Returns an [`Avx1`] token iff the current CPU supports AVX.
    // Dead when the build baseline statically guarantees avx; dispatch then uses `new_unchecked`.
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    #[inline]
    pub fn detect() -> Option<Self> {
        if is_x86_feature_detected!("avx") {
            Some(Self(()))
        } else {
            None
        }
    }

    /// # Safety
    /// The caller guarantees the running CPU supports `avx`. Calling any [`Backend`] method on a
    /// token built this way on an unsupported CPU is UB.
    // Dead only on a std build with no static avx guarantee, which goes through runtime `detect`.
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Backend<f32> for Avx1 {
    type Vector = __m256;
    type Mask = __m256;

    type IVector = [u32; 8];
    #[inline(always)]
    fn iload(self, s: &[u32]) -> [u32; 8] {
        let mut v = [0u32; 8];
        v.copy_from_slice(s);
        v
    }
    #[inline(always)]
    fn istore(self, v: [u32; 8], out: &mut [u32]) {
        out.copy_from_slice(&v);
    }

    #[inline(always)]
    fn lanes(self) -> usize {
        8
    }
    #[inline(always)]
    fn splat(self, v: f32) -> __m256 {
        unsafe { f32_splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f32]) -> __m256 {
        debug_assert_eq!(s.len(), 8);
        unsafe { f32_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m256, s: &mut [f32]) {
        debug_assert_eq!(s.len(), 8);
        unsafe { f32_store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_add(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m256) -> __m256 {
        unsafe { f32_neg(a) }
    }
    #[inline(always)]
    fn abs(self, a: __m256) -> __m256 {
        unsafe { f32_abs(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m256, b: __m256, c: __m256) -> __m256 {
        unsafe { f32_fma(a, b, c) }
    }
    #[inline(always)]
    fn madd(self, a: __m256, b: __m256, acc: __m256) -> __m256 {
        <Self as Backend<f32>>::fma(self, a, b, acc)
    }
    #[inline(always)]
    fn sqrt(self, a: __m256) -> __m256 {
        unsafe { f32_sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_cmp::<_CMP_LE_OQ>(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_cmp::<_CMP_LT_OQ>(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_cmp::<_CMP_GE_OQ>(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_cmp::<_CMP_GT_OQ>(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_and(a, b) }
    }
    #[inline(always)]
    fn mask_or(self, a: __m256, b: __m256) -> __m256 {
        unsafe { f32_or(a, b) }
    }
    #[inline(always)]
    fn mask_not(self, a: __m256) -> __m256 {
        unsafe { f32_not(a) }
    }
    #[inline(always)]
    fn select(self, m: __m256, a: __m256, b: __m256) -> __m256 {
        // blendv picks `a`(2nd) where mask high bit set, else `b`(1st): m ? a : b
        unsafe { f32_blend(b, a, m) }
    }
    #[inline(always)]
    fn any(self, m: __m256) -> bool {
        unsafe { f32_movemask(m) != 0 }
    }
    #[inline(always)]
    fn all(self, m: __m256) -> bool {
        unsafe { f32_movemask(m) == 0xFF }
    }
    #[inline(always)]
    fn mask_bitmask(self, m: __m256) -> u32 {
        unsafe { f32_movemask(m) as u32 }
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m256) -> f32 {
        unsafe { f32_reduce::<0>(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: __m256) -> f32 {
        unsafe { f32_reduce::<1>(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m256) -> f32 {
        unsafe { f32_reduce::<2>(v) }
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_splat(v: f32) -> __m256 {
    _mm256_set1_ps(v)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_load(p: *const f32) -> __m256 {
    _mm256_loadu_ps(p)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_store(p: *mut f32, v: __m256) {
    _mm256_storeu_ps(p, v)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_add(a: __m256, b: __m256) -> __m256 {
    _mm256_add_ps(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_sub(a: __m256, b: __m256) -> __m256 {
    _mm256_sub_ps(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_mul(a: __m256, b: __m256) -> __m256 {
    _mm256_mul_ps(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_div(a: __m256, b: __m256) -> __m256 {
    _mm256_div_ps(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_neg(a: __m256) -> __m256 {
    _mm256_xor_ps(a, _mm256_set1_ps(-0.0))
}
/// Clear the sign bit: one `andps`, cheaper than `max(a, -a)`.
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_abs(a: __m256) -> __m256 {
    _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF)))
}
/// No FMA at this tier: multiply then add (two roundings), matching the scalar oracle.
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_fma(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_add_ps(_mm256_mul_ps(a, b), c)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_sqrt(a: __m256) -> __m256 {
    _mm256_sqrt_ps(a)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_min(a: __m256, b: __m256) -> __m256 {
    // IEEE minimumNumber: `vminps` yields `b` when `a` is NaN; the blend patches the b-is-NaN case.
    let m = _mm256_min_ps(a, b);
    _mm256_blendv_ps(m, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_max(a: __m256, b: __m256) -> __m256 {
    let m = _mm256_max_ps(a, b);
    _mm256_blendv_ps(m, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_cmp<const OP: i32>(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<OP>(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_and(a: __m256, b: __m256) -> __m256 {
    _mm256_and_ps(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_or(a: __m256, b: __m256) -> __m256 {
    _mm256_or_ps(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_not(a: __m256) -> __m256 {
    let ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    _mm256_xor_ps(a, ones)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_blend(f: __m256, t: __m256, m: __m256) -> __m256 {
    _mm256_blendv_ps(f, t, m)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_movemask(m: __m256) -> i32 {
    _mm256_movemask_ps(m)
}
/// `OP`: 0 = sum, 1 = min, 2 = max. Horizontal reduce of 8 f32 lanes.
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f32_reduce<const OP: i32>(v: __m256) -> f32 {
    #[inline(always)]
    unsafe fn combine<const OP: i32>(a: __m128, b: __m128) -> __m128 {
        match OP {
            1 => {
                let m = _mm_min_ps(a, b);
                _mm_blendv_ps(m, a, _mm_cmpunord_ps(b, b))
            }
            2 => {
                let m = _mm_max_ps(a, b);
                _mm_blendv_ps(m, a, _mm_cmpunord_ps(b, b))
            }
            _ => _mm_add_ps(a, b),
        }
    }
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let q = combine::<OP>(lo, hi); // 4 lanes
    let shuf = _mm_movehdup_ps(q); // [1,1,3,3]
    let d = combine::<OP>(q, shuf); // lanes 0,2 carry pair results
    let shuf2 = _mm_movehl_ps(shuf, d);
    let r = combine::<OP>(d, shuf2);
    _mm_cvtss_f32(r)
}

impl Backend<f64> for Avx1 {
    type Vector = __m256d;
    type Mask = __m256d;

    type IVector = [u32; 4];
    #[inline(always)]
    fn iload(self, s: &[u32]) -> [u32; 4] {
        let mut v = [0u32; 4];
        v.copy_from_slice(s);
        v
    }
    #[inline(always)]
    fn istore(self, v: [u32; 4], out: &mut [u32]) {
        out.copy_from_slice(&v);
    }

    #[inline(always)]
    fn lanes(self) -> usize {
        4
    }
    #[inline(always)]
    fn splat(self, v: f64) -> __m256d {
        unsafe { f64_splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f64]) -> __m256d {
        debug_assert_eq!(s.len(), 4);
        unsafe { f64_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m256d, s: &mut [f64]) {
        debug_assert_eq!(s.len(), 4);
        unsafe { f64_store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_add(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m256d) -> __m256d {
        unsafe { f64_neg(a) }
    }
    #[inline(always)]
    fn abs(self, a: __m256d) -> __m256d {
        unsafe { f64_abs(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m256d, b: __m256d, c: __m256d) -> __m256d {
        unsafe { f64_fma(a, b, c) }
    }
    #[inline(always)]
    fn madd(self, a: __m256d, b: __m256d, acc: __m256d) -> __m256d {
        <Self as Backend<f64>>::fma(self, a, b, acc)
    }
    #[inline(always)]
    fn sqrt(self, a: __m256d) -> __m256d {
        unsafe { f64_sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_cmp::<_CMP_LE_OQ>(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_cmp::<_CMP_LT_OQ>(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_cmp::<_CMP_GE_OQ>(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_cmp::<_CMP_GT_OQ>(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_and(a, b) }
    }
    #[inline(always)]
    fn mask_or(self, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_or(a, b) }
    }
    #[inline(always)]
    fn mask_not(self, a: __m256d) -> __m256d {
        unsafe { f64_not(a) }
    }
    #[inline(always)]
    fn select(self, m: __m256d, a: __m256d, b: __m256d) -> __m256d {
        unsafe { f64_blend(b, a, m) }
    }
    #[inline(always)]
    fn any(self, m: __m256d) -> bool {
        unsafe { f64_movemask(m) != 0 }
    }
    #[inline(always)]
    fn all(self, m: __m256d) -> bool {
        unsafe { f64_movemask(m) == 0xF }
    }
    #[inline(always)]
    fn mask_bitmask(self, m: __m256d) -> u32 {
        unsafe { f64_movemask(m) as u32 }
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m256d) -> f64 {
        unsafe { f64_reduce::<0>(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: __m256d) -> f64 {
        unsafe { f64_reduce::<1>(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m256d) -> f64 {
        unsafe { f64_reduce::<2>(v) }
    }
}

#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_splat(v: f64) -> __m256d {
    _mm256_set1_pd(v)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_load(p: *const f64) -> __m256d {
    _mm256_loadu_pd(p)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_store(p: *mut f64, v: __m256d) {
    _mm256_storeu_pd(p, v)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_add(a: __m256d, b: __m256d) -> __m256d {
    _mm256_add_pd(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_sub(a: __m256d, b: __m256d) -> __m256d {
    _mm256_sub_pd(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_mul(a: __m256d, b: __m256d) -> __m256d {
    _mm256_mul_pd(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_div(a: __m256d, b: __m256d) -> __m256d {
    _mm256_div_pd(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_neg(a: __m256d) -> __m256d {
    _mm256_xor_pd(a, _mm256_set1_pd(-0.0))
}
/// Clear the sign bit: one `andpd`, cheaper than `max(a, -a)`.
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_abs(a: __m256d) -> __m256d {
    _mm256_and_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF)))
}
/// No FMA at this tier: multiply then add, matching the scalar oracle.
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_fma(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_add_pd(_mm256_mul_pd(a, b), c)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_sqrt(a: __m256d) -> __m256d {
    _mm256_sqrt_pd(a)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_min(a: __m256d, b: __m256d) -> __m256d {
    let m = _mm256_min_pd(a, b);
    _mm256_blendv_pd(m, a, _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b))
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_max(a: __m256d, b: __m256d) -> __m256d {
    let m = _mm256_max_pd(a, b);
    _mm256_blendv_pd(m, a, _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b))
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_cmp<const OP: i32>(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<OP>(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_and(a: __m256d, b: __m256d) -> __m256d {
    _mm256_and_pd(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_or(a: __m256d, b: __m256d) -> __m256d {
    _mm256_or_pd(a, b)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_not(a: __m256d) -> __m256d {
    let ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
    _mm256_xor_pd(a, ones)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_blend(f: __m256d, t: __m256d, m: __m256d) -> __m256d {
    _mm256_blendv_pd(f, t, m)
}
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_movemask(m: __m256d) -> i32 {
    _mm256_movemask_pd(m)
}
/// `OP`: 0 = sum, 1 = min, 2 = max. Horizontal reduce of 4 f64 lanes.
#[target_feature(enable = "avx")]
#[inline]
unsafe fn f64_reduce<const OP: i32>(v: __m256d) -> f64 {
    #[inline(always)]
    unsafe fn combine<const OP: i32>(a: __m128d, b: __m128d) -> __m128d {
        match OP {
            1 => {
                let m = _mm_min_pd(a, b);
                _mm_blendv_pd(m, a, _mm_cmpunord_pd(b, b))
            }
            2 => {
                let m = _mm_max_pd(a, b);
                _mm_blendv_pd(m, a, _mm_cmpunord_pd(b, b))
            }
            _ => _mm_add_pd(a, b),
        }
    }
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd::<1>(v);
    let p = combine::<OP>(lo, hi); // 2 lanes
    let sh = _mm_unpackhi_pd(p, p);
    let r = combine::<OP>(p, sh);
    _mm_cvtsd_f64(r)
}

// Correctness-only emulation so the token satisfies `BackendAll`; the dispatch ladders for
// these elements never pick Avx1.
crate::backend::emulated_float_element!(Avx1, half::f16, 8);
crate::backend::emulated_float_element!(Avx1, half::bf16, 8);
crate::backend::emulated_int_element!(Avx1, u32, 8);
crate::backend::emulated_int_element!(Avx1, i32, 8);
