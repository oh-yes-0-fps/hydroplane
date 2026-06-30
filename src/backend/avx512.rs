//! Hand-written AVX-512F backend for `x86_64` (16-wide f32, 8-wide f64).
//!
//! Unlike the 128/256-bit backends, comparisons here produce `k`-mask registers
//! (`__mmask16`/`__mmask8`, which are `u16`/`u8`), so mask ops are plain integer bit ops
//! and `select` is `_mm512_mask_blend`. Horizontal reductions use the hardware
//! `_mm512_reduce_*` sequences. AVX-512F includes FMA. The [`Avx512`] token must only
//! exist on a CPU with `avx512f` (see [`Avx512::detect`]).
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::backend::Backend;

/// AVX-512F execution token. Construct only when the CPU supports `avx512f`.
#[derive(Clone, Copy, Debug)]
pub struct Avx512(());

impl Avx512 {
    // Unused once the build's baseline statically guarantees avx512f (x86-64-v4 / native):
    // dispatch then pins the backend branchlessly via `new_unchecked` with no runtime check.
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    #[inline]
    pub fn detect() -> Option<Self> {
        if is_x86_feature_detected!("avx512f") {
            Some(Self(()))
        } else {
            None
        }
    }
    /// # Safety
    /// The CPU must support `avx512f`.
    // Used only by the statically-pinned and no-std dispatch paths; on a std build the
    // backend is always reached through runtime `detect`, leaving this constructor unused.
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Backend<f32> for Avx512 {
    type Vector = __m512;
    type Mask = __mmask16;

    #[inline(always)]
    fn lanes(self) -> usize {
        16
    }
    #[inline(always)]
    fn splat(self, v: f32) -> __m512 {
        unsafe { z_splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f32]) -> __m512 {
        debug_assert_eq!(s.len(), 16);
        unsafe { z_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m512, s: &mut [f32]) {
        debug_assert_eq!(s.len(), 16);
        unsafe { z_store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        unsafe { z_add(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        unsafe { z_sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        unsafe { z_mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m512, b: __m512) -> __m512 {
        unsafe { z_div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m512) -> __m512 {
        unsafe { z_sub(z_zero(), a) }
    }
    #[inline(always)]
    fn abs(self, a: __m512) -> __m512 {
        unsafe { z_abs(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        unsafe { z_fma(a, b, c) }
    }
    #[inline(always)]
    fn sqrt(self, a: __m512) -> __m512 {
        unsafe { z_sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m512, b: __m512) -> __m512 {
        unsafe { z_min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        unsafe { z_max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m512, b: __m512) -> __mmask16 {
        unsafe { z_cmp::<_CMP_LE_OQ>(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m512, b: __m512) -> __mmask16 {
        unsafe { z_cmp::<_CMP_LT_OQ>(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m512, b: __m512) -> __mmask16 {
        unsafe { z_cmp::<_CMP_GE_OQ>(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m512, b: __m512) -> __mmask16 {
        unsafe { z_cmp::<_CMP_GT_OQ>(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: __mmask16, b: __mmask16) -> __mmask16 {
        a & b
    }
    #[inline(always)]
    fn mask_or(self, a: __mmask16, b: __mmask16) -> __mmask16 {
        a | b
    }
    #[inline(always)]
    fn mask_not(self, a: __mmask16) -> __mmask16 {
        !a
    }
    #[inline(always)]
    fn select(self, m: __mmask16, a: __m512, b: __m512) -> __m512 {
        // mask_blend(k, a, b) = k ? b : a, so pass (k, b, a) for k ? a : b
        unsafe { z_blend(m, b, a) }
    }
    #[inline(always)]
    fn any(self, m: __mmask16) -> bool {
        m != 0
    }
    #[inline(always)]
    fn all(self, m: __mmask16) -> bool {
        m == 0xFFFF
    }
    #[inline(always)]
    fn mask_bitmask(self, m: __mmask16) -> u32 {
        m as u32
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m512) -> f32 {
        unsafe { _mm512_reduce_add_ps(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: __m512) -> f32 {
        unsafe { _mm512_reduce_min_ps(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m512) -> f32 {
        unsafe { _mm512_reduce_max_ps(v) }
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_zero() -> __m512 {
    _mm512_setzero_ps()
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_splat(v: f32) -> __m512 {
    _mm512_set1_ps(v)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_load(p: *const f32) -> __m512 {
    _mm512_loadu_ps(p)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_store(p: *mut f32, v: __m512) {
    _mm512_storeu_ps(p, v)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_add(a: __m512, b: __m512) -> __m512 {
    _mm512_add_ps(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_sub(a: __m512, b: __m512) -> __m512 {
    _mm512_sub_ps(a, b)
}
/// Native `vandps`-class abs (`_mm512_abs_ps`), one op vs `max(a, -a)`.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_abs(a: __m512) -> __m512 {
    _mm512_abs_ps(a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_mul(a: __m512, b: __m512) -> __m512 {
    _mm512_mul_ps(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_div(a: __m512, b: __m512) -> __m512 {
    _mm512_div_ps(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_fma(a: __m512, b: __m512, c: __m512) -> __m512 {
    _mm512_fmadd_ps(a, b, c)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_sqrt(a: __m512) -> __m512 {
    _mm512_sqrt_ps(a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_min(a: __m512, b: __m512) -> __m512 {
    _mm512_min_ps(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_max(a: __m512, b: __m512) -> __m512 {
    _mm512_max_ps(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_cmp<const OP: i32>(a: __m512, b: __m512) -> __mmask16 {
    _mm512_cmp_ps_mask::<OP>(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_blend(m: __mmask16, a: __m512, b: __m512) -> __m512 {
    _mm512_mask_blend_ps(m, a, b)
}

impl Backend<f64> for Avx512 {
    type Vector = __m512d;
    type Mask = __mmask8;

    #[inline(always)]
    fn lanes(self) -> usize {
        8
    }
    #[inline(always)]
    fn splat(self, v: f64) -> __m512d {
        unsafe { zd_splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f64]) -> __m512d {
        debug_assert_eq!(s.len(), 8);
        unsafe { zd_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m512d, s: &mut [f64]) {
        debug_assert_eq!(s.len(), 8);
        unsafe { zd_store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_add(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m512d) -> __m512d {
        unsafe { zd_sub(zd_zero(), a) }
    }
    #[inline(always)]
    fn abs(self, a: __m512d) -> __m512d {
        unsafe { zd_abs(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        unsafe { zd_fma(a, b, c) }
    }
    #[inline(always)]
    fn sqrt(self, a: __m512d) -> __m512d {
        unsafe { zd_sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m512d, b: __m512d) -> __mmask8 {
        unsafe { zd_cmp::<_CMP_LE_OQ>(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m512d, b: __m512d) -> __mmask8 {
        unsafe { zd_cmp::<_CMP_LT_OQ>(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m512d, b: __m512d) -> __mmask8 {
        unsafe { zd_cmp::<_CMP_GE_OQ>(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m512d, b: __m512d) -> __mmask8 {
        unsafe { zd_cmp::<_CMP_GT_OQ>(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: __mmask8, b: __mmask8) -> __mmask8 {
        a & b
    }
    #[inline(always)]
    fn mask_or(self, a: __mmask8, b: __mmask8) -> __mmask8 {
        a | b
    }
    #[inline(always)]
    fn mask_not(self, a: __mmask8) -> __mmask8 {
        !a
    }
    #[inline(always)]
    fn select(self, m: __mmask8, a: __m512d, b: __m512d) -> __m512d {
        unsafe { zd_blend(m, b, a) }
    }
    #[inline(always)]
    fn any(self, m: __mmask8) -> bool {
        m != 0
    }
    #[inline(always)]
    fn all(self, m: __mmask8) -> bool {
        m == 0xFF
    }
    #[inline(always)]
    fn mask_bitmask(self, m: __mmask8) -> u32 {
        m as u32
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m512d) -> f64 {
        unsafe { _mm512_reduce_add_pd(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: __m512d) -> f64 {
        unsafe { _mm512_reduce_min_pd(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m512d) -> f64 {
        unsafe { _mm512_reduce_max_pd(v) }
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_zero() -> __m512d {
    _mm512_setzero_pd()
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_splat(v: f64) -> __m512d {
    _mm512_set1_pd(v)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_load(p: *const f64) -> __m512d {
    _mm512_loadu_pd(p)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_store(p: *mut f64, v: __m512d) {
    _mm512_storeu_pd(p, v)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_add(a: __m512d, b: __m512d) -> __m512d {
    _mm512_add_pd(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_sub(a: __m512d, b: __m512d) -> __m512d {
    _mm512_sub_pd(a, b)
}
/// Native `_mm512_abs_pd`, one op vs `max(a, -a)`.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_abs(a: __m512d) -> __m512d {
    _mm512_abs_pd(a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_mul(a: __m512d, b: __m512d) -> __m512d {
    _mm512_mul_pd(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_div(a: __m512d, b: __m512d) -> __m512d {
    _mm512_div_pd(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_fma(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
    _mm512_fmadd_pd(a, b, c)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_sqrt(a: __m512d) -> __m512d {
    _mm512_sqrt_pd(a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_min(a: __m512d, b: __m512d) -> __m512d {
    _mm512_min_pd(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_max(a: __m512d, b: __m512d) -> __m512d {
    _mm512_max_pd(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_cmp<const OP: i32>(a: __m512d, b: __m512d) -> __mmask8 {
    _mm512_cmp_pd_mask::<OP>(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_blend(m: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_blend_pd(m, a, b)
}

// `bf16` on AVX-512: `f32x16` widen-compute-narrow (scalar boundary conversions; bf16 has no
// native ALU). `Mask = __mmask16` (k-register), as on the f32 path. This is the element-wise
// substrate the AVX512-VNNI (`vdpbf16ps`) and AMX matmul fast paths build on.
mod bf16_impl {
    use super::*;
    use crate::backend::Backend;
    use half::bf16;

    #[inline(always)]
    unsafe fn b_load(s: &[bf16]) -> __m512 {
        let mut t = [0f32; 16];
        for (d, x) in t.iter_mut().zip(s) {
            *d = x.to_f32();
        }
        z_load(t.as_ptr())
    }
    #[inline(always)]
    unsafe fn b_store(v: __m512, s: &mut [bf16]) {
        let mut t = [0f32; 16];
        z_store(t.as_mut_ptr(), v);
        for (d, x) in s.iter_mut().zip(t) {
            *d = bf16::from_f32(x);
        }
    }

    impl Backend<bf16> for Avx512 {
        type Vector = __m512;
        type Mask = __mmask16;

        #[inline(always)]
        fn lanes(self) -> usize {
            16
        }
        #[inline(always)]
        fn splat(self, v: bf16) -> __m512 {
            unsafe { z_splat(v.to_f32()) }
        }
        #[inline(always)]
        fn load(self, s: &[bf16]) -> __m512 {
            debug_assert_eq!(s.len(), 16);
            unsafe { b_load(s) }
        }
        #[inline(always)]
        fn store(self, v: __m512, s: &mut [bf16]) {
            debug_assert_eq!(s.len(), 16);
            unsafe { b_store(v, s) }
        }
        #[inline(always)]
        fn add(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_add(a, b) }
        }
        #[inline(always)]
        fn sub(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_sub(a, b) }
        }
        #[inline(always)]
        fn mul(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_mul(a, b) }
        }
        #[inline(always)]
        fn div(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_div(a, b) }
        }
        #[inline(always)]
        fn neg(self, a: __m512) -> __m512 {
            unsafe { z_sub(z_zero(), a) }
        }
        #[inline(always)]
        fn abs(self, a: __m512) -> __m512 {
            unsafe { z_abs(a) }
        }
        #[inline(always)]
        fn fma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
            unsafe { z_fma(a, b, c) }
        }
        #[inline(always)]
        fn sqrt(self, a: __m512) -> __m512 {
            unsafe { z_sqrt(a) }
        }
        #[inline(always)]
        fn min(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_min(a, b) }
        }
        #[inline(always)]
        fn max(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_max(a, b) }
        }
        #[inline(always)]
        fn le(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_LE_OQ>(a, b) }
        }
        #[inline(always)]
        fn lt(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_LT_OQ>(a, b) }
        }
        #[inline(always)]
        fn ge(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_GE_OQ>(a, b) }
        }
        #[inline(always)]
        fn gt(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_GT_OQ>(a, b) }
        }
        #[inline(always)]
        fn mask_and(self, a: __mmask16, b: __mmask16) -> __mmask16 {
            a & b
        }
        #[inline(always)]
        fn mask_or(self, a: __mmask16, b: __mmask16) -> __mmask16 {
            a | b
        }
        #[inline(always)]
        fn mask_not(self, a: __mmask16) -> __mmask16 {
            !a
        }
        #[inline(always)]
        fn select(self, m: __mmask16, a: __m512, b: __m512) -> __m512 {
            unsafe { z_blend(m, b, a) }
        }
        #[inline(always)]
        fn any(self, m: __mmask16) -> bool {
            m != 0
        }
        #[inline(always)]
        fn all(self, m: __mmask16) -> bool {
            m == 0xFFFF
        }
        #[inline(always)]
        fn mask_bitmask(self, m: __mmask16) -> u32 {
            m as u32
        }
        #[inline(always)]
        fn reduce_sum(self, v: __m512) -> bf16 {
            bf16::from_f32(unsafe { _mm512_reduce_add_ps(v) })
        }
        #[inline(always)]
        fn reduce_min(self, v: __m512) -> bf16 {
            bf16::from_f32(unsafe { _mm512_reduce_min_ps(v) })
        }
        #[inline(always)]
        fn reduce_max(self, v: __m512) -> bf16 {
            bf16::from_f32(unsafe { _mm512_reduce_max_ps(v) })
        }
    }
}

// `f16` on AVX-512 (without AVX-512-FP16): `f32x16` widen-compute-narrow, but with hardware
// `vcvtph2ps`/`vcvtps2ph` boundary conversion (zmm forms are AVX-512F, no F16C needed) instead of
// bf16's scalar loop. 16 lanes — double the 8-wide AVX2 F16C path that AVX-512-without-FP16 hosts
// (Cascade Lake / Ice Lake / Zen 4) otherwise fall back to. `Mask = __mmask16` as on f32.
mod f16_impl {
    use super::*;
    use crate::backend::Backend;
    use half::f16;

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn h_load(s: &[f16]) -> __m512 {
        _mm512_cvtph_ps(_mm256_loadu_si256(s.as_ptr().cast()))
    }
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn h_store(v: __m512, s: &mut [f16]) {
        let packed = _mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
        _mm256_storeu_si256(s.as_mut_ptr().cast(), packed);
    }

    impl Backend<f16> for Avx512 {
        type Vector = __m512;
        type Mask = __mmask16;

        #[inline(always)]
        fn lanes(self) -> usize {
            16
        }
        #[inline(always)]
        fn splat(self, v: f16) -> __m512 {
            unsafe { z_splat(v.to_f32()) }
        }
        #[inline(always)]
        fn load(self, s: &[f16]) -> __m512 {
            debug_assert_eq!(s.len(), 16);
            unsafe { h_load(s) }
        }
        #[inline(always)]
        fn store(self, v: __m512, s: &mut [f16]) {
            debug_assert_eq!(s.len(), 16);
            unsafe { h_store(v, s) }
        }
        #[inline(always)]
        fn add(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_add(a, b) }
        }
        #[inline(always)]
        fn sub(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_sub(a, b) }
        }
        #[inline(always)]
        fn mul(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_mul(a, b) }
        }
        #[inline(always)]
        fn div(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_div(a, b) }
        }
        #[inline(always)]
        fn neg(self, a: __m512) -> __m512 {
            unsafe { z_sub(z_zero(), a) }
        }
        #[inline(always)]
        fn abs(self, a: __m512) -> __m512 {
            unsafe { z_abs(a) }
        }
        #[inline(always)]
        fn fma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
            unsafe { z_fma(a, b, c) }
        }
        #[inline(always)]
        fn sqrt(self, a: __m512) -> __m512 {
            unsafe { z_sqrt(a) }
        }
        #[inline(always)]
        fn min(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_min(a, b) }
        }
        #[inline(always)]
        fn max(self, a: __m512, b: __m512) -> __m512 {
            unsafe { z_max(a, b) }
        }
        #[inline(always)]
        fn le(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_LE_OQ>(a, b) }
        }
        #[inline(always)]
        fn lt(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_LT_OQ>(a, b) }
        }
        #[inline(always)]
        fn ge(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_GE_OQ>(a, b) }
        }
        #[inline(always)]
        fn gt(self, a: __m512, b: __m512) -> __mmask16 {
            unsafe { z_cmp::<_CMP_GT_OQ>(a, b) }
        }
        #[inline(always)]
        fn mask_and(self, a: __mmask16, b: __mmask16) -> __mmask16 {
            a & b
        }
        #[inline(always)]
        fn mask_or(self, a: __mmask16, b: __mmask16) -> __mmask16 {
            a | b
        }
        #[inline(always)]
        fn mask_not(self, a: __mmask16) -> __mmask16 {
            !a
        }
        #[inline(always)]
        fn select(self, m: __mmask16, a: __m512, b: __m512) -> __m512 {
            unsafe { z_blend(m, b, a) }
        }
        #[inline(always)]
        fn any(self, m: __mmask16) -> bool {
            m != 0
        }
        #[inline(always)]
        fn all(self, m: __mmask16) -> bool {
            m == 0xFFFF
        }
        #[inline(always)]
        fn mask_bitmask(self, m: __mmask16) -> u32 {
            m as u32
        }
        #[inline(always)]
        fn reduce_sum(self, v: __m512) -> f16 {
            f16::from_f32(unsafe { _mm512_reduce_add_ps(v) })
        }
        #[inline(always)]
        fn reduce_min(self, v: __m512) -> f16 {
            f16::from_f32(unsafe { _mm512_reduce_min_ps(v) })
        }
        #[inline(always)]
        fn reduce_max(self, v: __m512) -> f16 {
            f16::from_f32(unsafe { _mm512_reduce_max_ps(v) })
        }
    }
}
