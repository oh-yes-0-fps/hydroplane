//! Hand-written AVX2 (+FMA) backend for `x86_64`.
//!
//! `Backend<f32>` uses `__m256` (8 lanes); `Backend<f64>` uses `__m256d` (4 lanes). Every
//! op is a `#[target_feature(enable = "avx2,fma")]` body, so the unsafety is confined here
//! and justified by the single invariant: an [`Avx2`] token only exists when the running
//! CPU has AVX2 + FMA (enforced by [`Avx2::detect`] / [`Avx2::new_unchecked`]).
//!
//! `f16` rides the F16C widen path (`__m256` f32 compute, hardware `vcvtph2ps`/`vcvtps2ph` at the
//! load/store boundary); `bf16` widens to `__m256` in software. The wider native f16 arithmetic is
//! the separate AVX-512-FP16 backend ([`super::avx512fp16`]).
//!
//! Every free fn below is an `unsafe fn` whose body is wholly composed of `#[target_feature]`
//! intrinsics; we opt out of the edition-2024 `unsafe_op_in_unsafe_fn` requirement for the
//! whole module rather than wrapping each one-line body in its own `unsafe {}`.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::backend::Backend;

/// AVX2 + FMA execution token. Zero-sized; only construct it when the CPU supports
/// `avx2` and `fma` (see [`Avx2::detect`]).
#[derive(Clone, Copy, Debug)]
pub struct Avx2(());

impl Avx2 {
    /// Returns an [`Avx2`] token iff the current CPU supports AVX2 + FMA.
    // Unused once the build's baseline statically guarantees avx2 (x86-64-v3+): dispatch then
    // takes the backend branchlessly via `new_unchecked` instead of detecting it.
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    #[inline]
    pub fn detect() -> Option<Self> {
        let ok = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        // The `half` widen path also uses F16C; require it so an `Avx2` token always
        // implies a sound `Backend<f16>` as well. (F16C is present on every AVX2 CPU.)
        let ok = ok && is_x86_feature_detected!("f16c");
        if ok { Some(Self(())) } else { None }
    }

    /// # Safety
    /// The caller guarantees the running CPU supports `avx2` and `fma`. Calling any
    /// [`Backend`] method on a token built this way on an unsupported CPU is UB.
    // Used by the no-std path and by any std build whose baseline already guarantees avx2
    // (the branchless floor); unused only on a std build with no avx2 guarantee, where the
    // backend is reached through runtime `detect`.
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

// ───────────────────────────── f32 × __m256 (8 lanes) ─────────────────────────────

impl Backend<f32> for Avx2 {
    type Vector = __m256;
    type Mask = __m256;

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

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_splat(v: f32) -> __m256 {
    _mm256_set1_ps(v)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_load(p: *const f32) -> __m256 {
    _mm256_loadu_ps(p)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_store(p: *mut f32, v: __m256) {
    _mm256_storeu_ps(p, v)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_add(a: __m256, b: __m256) -> __m256 {
    _mm256_add_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_sub(a: __m256, b: __m256) -> __m256 {
    _mm256_sub_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_mul(a: __m256, b: __m256) -> __m256 {
    _mm256_mul_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_div(a: __m256, b: __m256) -> __m256 {
    _mm256_div_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_neg(a: __m256) -> __m256 {
    _mm256_xor_ps(a, _mm256_set1_ps(-0.0))
}
/// Clear the sign bit — a single `andps`, cheaper than `max(a, -a)`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_abs(a: __m256) -> __m256 {
    _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF)))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_fma(a: __m256, b: __m256, c: __m256) -> __m256 {
    _mm256_fmadd_ps(a, b, c)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_sqrt(a: __m256) -> __m256 {
    _mm256_sqrt_ps(a)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_min(a: __m256, b: __m256) -> __m256 {
    _mm256_min_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_max(a: __m256, b: __m256) -> __m256 {
    _mm256_max_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_cmp<const OP: i32>(a: __m256, b: __m256) -> __m256 {
    _mm256_cmp_ps::<OP>(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_and(a: __m256, b: __m256) -> __m256 {
    _mm256_and_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_or(a: __m256, b: __m256) -> __m256 {
    _mm256_or_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_not(a: __m256) -> __m256 {
    let ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
    _mm256_xor_ps(a, ones)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_blend(f: __m256, t: __m256, m: __m256) -> __m256 {
    _mm256_blendv_ps(f, t, m)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_movemask(m: __m256) -> i32 {
    _mm256_movemask_ps(m)
}
/// `OP`: 0 = sum, 1 = min, 2 = max. Horizontal reduce of 8 f32 lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_reduce<const OP: i32>(v: __m256) -> f32 {
    #[inline(always)]
    unsafe fn combine<const OP: i32>(a: __m128, b: __m128) -> __m128 {
        match OP {
            1 => _mm_min_ps(a, b),
            2 => _mm_max_ps(a, b),
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

// ───────────────────────────── f64 × __m256d (4 lanes) ────────────────────────────

impl Backend<f64> for Avx2 {
    type Vector = __m256d;
    type Mask = __m256d;

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

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_splat(v: f64) -> __m256d {
    _mm256_set1_pd(v)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_load(p: *const f64) -> __m256d {
    _mm256_loadu_pd(p)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_store(p: *mut f64, v: __m256d) {
    _mm256_storeu_pd(p, v)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_add(a: __m256d, b: __m256d) -> __m256d {
    _mm256_add_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_sub(a: __m256d, b: __m256d) -> __m256d {
    _mm256_sub_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_mul(a: __m256d, b: __m256d) -> __m256d {
    _mm256_mul_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_div(a: __m256d, b: __m256d) -> __m256d {
    _mm256_div_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_neg(a: __m256d) -> __m256d {
    _mm256_xor_pd(a, _mm256_set1_pd(-0.0))
}
/// Clear the sign bit — a single `andpd`, cheaper than `max(a, -a)`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_abs(a: __m256d) -> __m256d {
    _mm256_and_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF)))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_fma(a: __m256d, b: __m256d, c: __m256d) -> __m256d {
    _mm256_fmadd_pd(a, b, c)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_sqrt(a: __m256d) -> __m256d {
    _mm256_sqrt_pd(a)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_min(a: __m256d, b: __m256d) -> __m256d {
    _mm256_min_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_max(a: __m256d, b: __m256d) -> __m256d {
    _mm256_max_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_cmp<const OP: i32>(a: __m256d, b: __m256d) -> __m256d {
    _mm256_cmp_pd::<OP>(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_and(a: __m256d, b: __m256d) -> __m256d {
    _mm256_and_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_or(a: __m256d, b: __m256d) -> __m256d {
    _mm256_or_pd(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_not(a: __m256d) -> __m256d {
    let ones = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
    _mm256_xor_pd(a, ones)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_blend(f: __m256d, t: __m256d, m: __m256d) -> __m256d {
    _mm256_blendv_pd(f, t, m)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_movemask(m: __m256d) -> i32 {
    _mm256_movemask_pd(m)
}
/// `OP`: 0 = sum, 1 = min, 2 = max. Horizontal reduce of 4 f64 lanes.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_reduce<const OP: i32>(v: __m256d) -> f64 {
    #[inline(always)]
    unsafe fn combine<const OP: i32>(a: __m128d, b: __m128d) -> __m128d {
        match OP {
            1 => _mm_min_pd(a, b),
            2 => _mm_max_pd(a, b),
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

// ───────────── f16 × __m256 (8 lanes, F16C widen-to-f32 compute path) ─────────────
//
// Storage is 16-bit `half::f16` (half the memory/bandwidth); compute is f32x8. `load`
// widens 8 f16 → f32x8 via F16C, every arithmetic op reuses the f32 helpers above, and
// `store` narrows back. The Scalar oracle for `f16` widens identically, so single ops
// match exactly; multi-op kernels are *more* accurate here (no intermediate narrowing).

mod f16_impl {
    use super::*;
    use crate::backend::Backend;
    use half::f16;

    impl Backend<f16> for Avx2 {
        type Vector = __m256; // 8 × f32 (widened)
        type Mask = __m256;

        #[inline(always)]
        fn lanes(self) -> usize {
            8
        }
        #[inline(always)]
        fn splat(self, v: f16) -> __m256 {
            unsafe { f32_splat(v.to_f32()) }
        }
        #[inline(always)]
        fn load(self, s: &[f16]) -> __m256 {
            debug_assert_eq!(s.len(), 8);
            unsafe { h_load(s.as_ptr()) }
        }
        #[inline(always)]
        fn store(self, v: __m256, s: &mut [f16]) {
            debug_assert_eq!(s.len(), 8);
            unsafe { h_store(s.as_mut_ptr(), v) }
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
        fn reduce_sum(self, v: __m256) -> f16 {
            f16::from_f32(unsafe { f32_reduce::<0>(v) })
        }
        #[inline(always)]
        fn reduce_min(self, v: __m256) -> f16 {
            f16::from_f32(unsafe { f32_reduce::<1>(v) })
        }
        #[inline(always)]
        fn reduce_max(self, v: __m256) -> f16 {
            f16::from_f32(unsafe { f32_reduce::<2>(v) })
        }
    }

    /// Widen 8 packed `f16` (16 bytes) to `f32x8`.
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn h_load(p: *const f16) -> __m256 {
        _mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i))
    }
    /// Narrow `f32x8` to 8 packed `f16` (round to nearest even).
    #[target_feature(enable = "avx2,fma,f16c")]
    #[inline]
    unsafe fn h_store(p: *mut f16, v: __m256) {
        let packed = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
        _mm_storeu_si128(p as *mut __m128i, packed);
    }
}

// `bf16` on AVX2: same `f32x8` widen-compute-narrow as `f16`, but bf16 has no F16C path, so the
// boundary conversions are scalar (cheap — bf16 is just the high 16 bits of an f32). This is the
// element-wise substrate the AVX2/AVX-512 bf16 matmul and the AMX/AVX512-VNNI fast paths build on.
mod bf16_impl {
    use super::*;
    use crate::backend::Backend;
    use half::bf16;

    #[inline(always)]
    unsafe fn b_load(s: &[bf16]) -> __m256 {
        let t = [
            s[0].to_f32(), s[1].to_f32(), s[2].to_f32(), s[3].to_f32(),
            s[4].to_f32(), s[5].to_f32(), s[6].to_f32(), s[7].to_f32(),
        ];
        f32_load(t.as_ptr())
    }
    #[inline(always)]
    unsafe fn b_store(v: __m256, s: &mut [bf16]) {
        let mut t = [0f32; 8];
        f32_store(t.as_mut_ptr(), v);
        for (d, x) in s.iter_mut().zip(t) {
            *d = bf16::from_f32(x);
        }
    }

    impl Backend<bf16> for Avx2 {
        type Vector = __m256;
        type Mask = __m256;

        #[inline(always)]
        fn lanes(self) -> usize {
            8
        }
        #[inline(always)]
        fn splat(self, v: bf16) -> __m256 {
            unsafe { f32_splat(v.to_f32()) }
        }
        #[inline(always)]
        fn load(self, s: &[bf16]) -> __m256 {
            debug_assert_eq!(s.len(), 8);
            unsafe { b_load(s) }
        }
        #[inline(always)]
        fn store(self, v: __m256, s: &mut [bf16]) {
            debug_assert_eq!(s.len(), 8);
            unsafe { b_store(v, s) }
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
        fn reduce_sum(self, v: __m256) -> bf16 {
            bf16::from_f32(unsafe { f32_reduce::<0>(v) })
        }
        #[inline(always)]
        fn reduce_min(self, v: __m256) -> bf16 {
            bf16::from_f32(unsafe { f32_reduce::<1>(v) })
        }
        #[inline(always)]
        fn reduce_max(self, v: __m256) -> bf16 {
            bf16::from_f32(unsafe { f32_reduce::<2>(v) })
        }
    }
}
