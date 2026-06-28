//! Hand-written SSE4.1 backend for `x86_64` — the near-universal 128-bit baseline.
//!
//! `Backend<f32>` uses `__m128` (4 lanes); `Backend<f64>` uses `__m128d` (2 lanes). SSE4.1
//! has no FMA, so [`Backend::fma`] is `a*b + c` (two rounds). As with [`super::avx2`], the
//! unsafety is confined here and justified by the [`Sse4`] token only existing on a CPU
//! with SSE4.1 (see [`Sse4::detect`]).
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::backend::Backend;

/// SSE4.1 execution token. Construct only when the CPU supports `sse4.1`.
#[derive(Clone, Copy, Debug)]
pub struct Sse4(());

impl Sse4 {
    // Unused once the build's baseline statically guarantees sse4.1 (x86-64-v2+) or a wider
    // floor: dispatch then takes a backend branchlessly via `new_unchecked`.
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    #[inline]
    pub fn detect() -> Option<Self> {
        if is_x86_feature_detected!("sse4.1") {
            Some(Self(()))
        } else {
            None
        }
    }
    /// # Safety
    /// The CPU must support `sse4.1`.
    // Used only by the statically-pinned and no-std dispatch paths; on a std build the
    // backend is always reached through runtime `detect`, leaving this constructor unused.
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Backend<f32> for Sse4 {
    type Vector = __m128;
    type Mask = __m128;

    #[inline(always)]
    fn lanes(self) -> usize {
        4
    }
    #[inline(always)]
    fn splat(self, v: f32) -> __m128 {
        unsafe { s_splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f32]) -> __m128 {
        debug_assert_eq!(s.len(), 4);
        unsafe { s_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m128, s: &mut [f32]) {
        debug_assert_eq!(s.len(), 4);
        unsafe { s_store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_add(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m128) -> __m128 {
        unsafe { s_xor(a, s_splat(-0.0)) }
    }
    #[inline(always)]
    fn fma(self, a: __m128, b: __m128, c: __m128) -> __m128 {
        unsafe { s_add(s_mul(a, b), c) }
    }
    #[inline(always)]
    fn sqrt(self, a: __m128) -> __m128 {
        unsafe { s_sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_cmple(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_cmplt(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_cmpge(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_cmpgt(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_and(a, b) }
    }
    #[inline(always)]
    fn mask_or(self, a: __m128, b: __m128) -> __m128 {
        unsafe { s_or(a, b) }
    }
    #[inline(always)]
    fn mask_not(self, a: __m128) -> __m128 {
        unsafe { s_xor(a, _mm_castsi128_ps(_mm_set1_epi32(-1))) }
    }
    #[inline(always)]
    fn select(self, m: __m128, a: __m128, b: __m128) -> __m128 {
        unsafe { _mm_blendv_ps(b, a, m) }
    }
    #[inline(always)]
    fn any(self, m: __m128) -> bool {
        unsafe { _mm_movemask_ps(m) != 0 }
    }
    #[inline(always)]
    fn all(self, m: __m128) -> bool {
        unsafe { _mm_movemask_ps(m) == 0xF }
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m128) -> f32 {
        unsafe { s_reduce::<0>(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: __m128) -> f32 {
        unsafe { s_reduce::<1>(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m128) -> f32 {
        unsafe { s_reduce::<2>(v) }
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_splat(v: f32) -> __m128 {
    _mm_set1_ps(v)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_load(p: *const f32) -> __m128 {
    _mm_loadu_ps(p)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_store(p: *mut f32, v: __m128) {
    _mm_storeu_ps(p, v)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_add(a: __m128, b: __m128) -> __m128 {
    _mm_add_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_sub(a: __m128, b: __m128) -> __m128 {
    _mm_sub_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_mul(a: __m128, b: __m128) -> __m128 {
    _mm_mul_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_div(a: __m128, b: __m128) -> __m128 {
    _mm_div_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_sqrt(a: __m128) -> __m128 {
    _mm_sqrt_ps(a)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_min(a: __m128, b: __m128) -> __m128 {
    _mm_min_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_max(a: __m128, b: __m128) -> __m128 {
    _mm_max_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_cmple(a: __m128, b: __m128) -> __m128 {
    _mm_cmple_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_cmplt(a: __m128, b: __m128) -> __m128 {
    _mm_cmplt_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_cmpge(a: __m128, b: __m128) -> __m128 {
    _mm_cmpge_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_cmpgt(a: __m128, b: __m128) -> __m128 {
    _mm_cmpgt_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_and(a: __m128, b: __m128) -> __m128 {
    _mm_and_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_or(a: __m128, b: __m128) -> __m128 {
    _mm_or_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_xor(a: __m128, b: __m128) -> __m128 {
    _mm_xor_ps(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_reduce<const OP: i32>(v: __m128) -> f32 {
    #[inline(always)]
    unsafe fn combine<const OP: i32>(a: __m128, b: __m128) -> __m128 {
        match OP {
            1 => _mm_min_ps(a, b),
            2 => _mm_max_ps(a, b),
            _ => _mm_add_ps(a, b),
        }
    }
    let shuf = _mm_movehdup_ps(v); // [1,1,3,3]
    let d = combine::<OP>(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, d);
    let r = combine::<OP>(d, shuf2);
    _mm_cvtss_f32(r)
}

impl Backend<f64> for Sse4 {
    type Vector = __m128d;
    type Mask = __m128d;

    #[inline(always)]
    fn lanes(self) -> usize {
        2
    }
    #[inline(always)]
    fn splat(self, v: f64) -> __m128d {
        unsafe { d_splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f64]) -> __m128d {
        debug_assert_eq!(s.len(), 2);
        unsafe { d_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m128d, s: &mut [f64]) {
        debug_assert_eq!(s.len(), 2);
        unsafe { d_store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_add_pd_(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m128d) -> __m128d {
        unsafe { d_xor(a, d_splat(-0.0)) }
    }
    #[inline(always)]
    fn fma(self, a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        unsafe { _mm_add_pd_(d_mul(a, b), c) }
    }
    #[inline(always)]
    fn sqrt(self, a: __m128d) -> __m128d {
        unsafe { d_sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_cmple(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_cmplt(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_cmpge(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_cmpgt(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_and(a, b) }
    }
    #[inline(always)]
    fn mask_or(self, a: __m128d, b: __m128d) -> __m128d {
        unsafe { d_or(a, b) }
    }
    #[inline(always)]
    fn mask_not(self, a: __m128d) -> __m128d {
        unsafe { d_xor(a, _mm_castsi128_pd(_mm_set1_epi32(-1))) }
    }
    #[inline(always)]
    fn select(self, m: __m128d, a: __m128d, b: __m128d) -> __m128d {
        unsafe { _mm_blendv_pd(b, a, m) }
    }
    #[inline(always)]
    fn any(self, m: __m128d) -> bool {
        unsafe { _mm_movemask_pd(m) != 0 }
    }
    #[inline(always)]
    fn all(self, m: __m128d) -> bool {
        unsafe { _mm_movemask_pd(m) == 0x3 }
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m128d) -> f64 {
        unsafe { d_reduce::<0>(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: __m128d) -> f64 {
        unsafe { d_reduce::<1>(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m128d) -> f64 {
        unsafe { d_reduce::<2>(v) }
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_splat(v: f64) -> __m128d {
    _mm_set1_pd(v)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_load(p: *const f64) -> __m128d {
    _mm_loadu_pd(p)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_store(p: *mut f64, v: __m128d) {
    _mm_storeu_pd(p, v)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn _mm_add_pd_(a: __m128d, b: __m128d) -> __m128d {
    _mm_add_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_sub(a: __m128d, b: __m128d) -> __m128d {
    _mm_sub_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_mul(a: __m128d, b: __m128d) -> __m128d {
    _mm_mul_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_div(a: __m128d, b: __m128d) -> __m128d {
    _mm_div_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_sqrt(a: __m128d) -> __m128d {
    _mm_sqrt_pd(a)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_min(a: __m128d, b: __m128d) -> __m128d {
    _mm_min_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_max(a: __m128d, b: __m128d) -> __m128d {
    _mm_max_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_cmple(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmple_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_cmplt(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmplt_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_cmpge(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmpge_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_cmpgt(a: __m128d, b: __m128d) -> __m128d {
    _mm_cmpgt_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_and(a: __m128d, b: __m128d) -> __m128d {
    _mm_and_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_or(a: __m128d, b: __m128d) -> __m128d {
    _mm_or_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_xor(a: __m128d, b: __m128d) -> __m128d {
    _mm_xor_pd(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_reduce<const OP: i32>(v: __m128d) -> f64 {
    let sh = _mm_unpackhi_pd(v, v);
    let r = match OP {
        1 => _mm_min_pd(v, sh),
        2 => _mm_max_pd(v, sh),
        _ => _mm_add_pd(v, sh),
    };
    _mm_cvtsd_f64(r)
}
