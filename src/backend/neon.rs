//! Hand-written AArch64 NEON backend (4-wide f32, 2-wide f64).
//!
//! NEON is baseline on AArch64, so the [`Neon`] token is always constructible there.
//! Masks are integer vectors (`uint32x4_t`/`uint64x2_t`); `select` is `vbslq` (bit-select);
//! cross-lane ops use the `v*vq` horizontal reductions. FMA is hardware (`vfmaq`).
#![allow(unsafe_op_in_unsafe_fn)]
#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

use crate::backend::Backend;

/// AArch64 NEON execution token (NEON is always available on AArch64).
#[derive(Clone, Copy, Debug, Default)]
pub struct Neon;

impl Neon {
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl Backend<f32> for Neon {
    type Vector = float32x4_t;
    type Mask = uint32x4_t;

    #[inline(always)]
    fn lanes(self) -> usize {
        4
    }
    #[inline(always)]
    fn splat(self, v: f32) -> float32x4_t {
        unsafe { vdupq_n_f32(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f32]) -> float32x4_t {
        debug_assert_eq!(s.len(), 4);
        unsafe { vld1q_f32(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: float32x4_t, s: &mut [f32]) {
        debug_assert_eq!(s.len(), 4);
        unsafe { vst1q_f32(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vsubq_f32(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vmulq_f32(a, b) }
    }
    #[inline(always)]
    fn div(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vdivq_f32(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: float32x4_t) -> float32x4_t {
        unsafe { vnegq_f32(a) }
    }
    #[inline(always)]
    fn fma(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        // vfmaq_f32(acc, x, y) = acc + x*y  ⇒  a*b + c
        unsafe { vfmaq_f32(c, a, b) }
    }
    #[inline(always)]
    fn sqrt(self, a: float32x4_t) -> float32x4_t {
        unsafe { vsqrtq_f32(a) }
    }
    #[inline(always)]
    fn min(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vminq_f32(a, b) }
    }
    #[inline(always)]
    fn max(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { vmaxq_f32(a, b) }
    }
    #[inline(always)]
    fn le(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        unsafe { vcleq_f32(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        unsafe { vcltq_f32(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        unsafe { vcgeq_f32(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        unsafe { vcgtq_f32(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vandq_u32(a, b) }
    }
    #[inline(always)]
    fn mask_or(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        unsafe { vorrq_u32(a, b) }
    }
    #[inline(always)]
    fn mask_not(self, a: uint32x4_t) -> uint32x4_t {
        unsafe { vmvnq_u32(a) }
    }
    #[inline(always)]
    fn select(self, m: uint32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        // vbslq_f32(mask, a, b) = mask ? a : b
        unsafe { vbslq_f32(m, a, b) }
    }
    #[inline(always)]
    fn any(self, m: uint32x4_t) -> bool {
        unsafe { vmaxvq_u32(m) != 0 }
    }
    #[inline(always)]
    fn all(self, m: uint32x4_t) -> bool {
        unsafe { vminvq_u32(m) == u32::MAX }
    }
    #[inline(always)]
    fn reduce_sum(self, v: float32x4_t) -> f32 {
        unsafe { vaddvq_f32(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: float32x4_t) -> f32 {
        unsafe { vminvq_f32(v) }
    }
    #[inline(always)]
    fn reduce_max(self, v: float32x4_t) -> f32 {
        unsafe { vmaxvq_f32(v) }
    }
}

impl Backend<f64> for Neon {
    type Vector = float64x2_t;
    type Mask = uint64x2_t;

    #[inline(always)]
    fn lanes(self) -> usize {
        2
    }
    #[inline(always)]
    fn splat(self, v: f64) -> float64x2_t {
        unsafe { vdupq_n_f64(v) }
    }
    #[inline(always)]
    fn load(self, s: &[f64]) -> float64x2_t {
        debug_assert_eq!(s.len(), 2);
        unsafe { vld1q_f64(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: float64x2_t, s: &mut [f64]) {
        debug_assert_eq!(s.len(), 2);
        unsafe { vst1q_f64(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vaddq_f64(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vsubq_f64(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vmulq_f64(a, b) }
    }
    #[inline(always)]
    fn div(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vdivq_f64(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: float64x2_t) -> float64x2_t {
        unsafe { vnegq_f64(a) }
    }
    #[inline(always)]
    fn fma(self, a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
        unsafe { vfmaq_f64(c, a, b) }
    }
    #[inline(always)]
    fn sqrt(self, a: float64x2_t) -> float64x2_t {
        unsafe { vsqrtq_f64(a) }
    }
    #[inline(always)]
    fn min(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vminq_f64(a, b) }
    }
    #[inline(always)]
    fn max(self, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vmaxq_f64(a, b) }
    }
    #[inline(always)]
    fn le(self, a: float64x2_t, b: float64x2_t) -> uint64x2_t {
        unsafe { vcleq_f64(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: float64x2_t, b: float64x2_t) -> uint64x2_t {
        unsafe { vcltq_f64(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: float64x2_t, b: float64x2_t) -> uint64x2_t {
        unsafe { vcgeq_f64(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: float64x2_t, b: float64x2_t) -> uint64x2_t {
        unsafe { vcgtq_f64(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vandq_u64(a, b) }
    }
    #[inline(always)]
    fn mask_or(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        unsafe { vorrq_u64(a, b) }
    }
    #[inline(always)]
    fn mask_not(self, a: uint64x2_t) -> uint64x2_t {
        unsafe { veorq_u64(a, vdupq_n_u64(u64::MAX)) }
    }
    #[inline(always)]
    fn select(self, m: uint64x2_t, a: float64x2_t, b: float64x2_t) -> float64x2_t {
        unsafe { vbslq_f64(m, a, b) }
    }
    #[inline(always)]
    fn any(self, m: uint64x2_t) -> bool {
        unsafe { (vgetq_lane_u64::<0>(m) | vgetq_lane_u64::<1>(m)) != 0 }
    }
    #[inline(always)]
    fn all(self, m: uint64x2_t) -> bool {
        unsafe { (vgetq_lane_u64::<0>(m) & vgetq_lane_u64::<1>(m)) == u64::MAX }
    }
    #[inline(always)]
    fn reduce_sum(self, v: float64x2_t) -> f64 {
        unsafe { vaddvq_f64(v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: float64x2_t) -> f64 {
        unsafe { vgetq_lane_f64::<0>(v).min(vgetq_lane_f64::<1>(v)) }
    }
    #[inline(always)]
    fn reduce_max(self, v: float64x2_t) -> f64 {
        unsafe { vgetq_lane_f64::<0>(v).max(vgetq_lane_f64::<1>(v)) }
    }
}

// `bf16` on NEON: storage is 16-bit, compute is `f32x4` (NEON has no native bf16 ALU). Widen on
// load/splat, narrow on store/reduce — conversions at the memory boundary only, so all arithmetic
// is native `f32` SIMD. `Vector = float32x4_t` (as on the AVX2 F16C f16 path). This is also the
// element-wise substrate the bf16 matmul (and the AMX/AVX512-VNNI fast paths) build on.
mod bf16_impl {
    use super::*;
    use half::bf16;

    #[inline(always)]
    unsafe fn load4(s: &[bf16]) -> float32x4_t {
        let t = [s[0].to_f32(), s[1].to_f32(), s[2].to_f32(), s[3].to_f32()];
        vld1q_f32(t.as_ptr())
    }
    #[inline(always)]
    unsafe fn store4(v: float32x4_t, s: &mut [bf16]) {
        let mut t = [0f32; 4];
        vst1q_f32(t.as_mut_ptr(), v);
        for (d, x) in s.iter_mut().zip(t) {
            *d = bf16::from_f32(x);
        }
    }

    impl Backend<bf16> for Neon {
        type Vector = float32x4_t;
        type Mask = uint32x4_t;

        #[inline(always)]
        fn lanes(self) -> usize {
            4
        }
        #[inline(always)]
        fn splat(self, v: bf16) -> float32x4_t {
            unsafe { vdupq_n_f32(v.to_f32()) }
        }
        #[inline(always)]
        fn load(self, s: &[bf16]) -> float32x4_t {
            debug_assert_eq!(s.len(), 4);
            unsafe { load4(s) }
        }
        #[inline(always)]
        fn store(self, v: float32x4_t, s: &mut [bf16]) {
            debug_assert_eq!(s.len(), 4);
            unsafe { store4(v, s) }
        }
        #[inline(always)]
        fn add(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vaddq_f32(a, b) }
        }
        #[inline(always)]
        fn sub(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vsubq_f32(a, b) }
        }
        #[inline(always)]
        fn mul(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vmulq_f32(a, b) }
        }
        #[inline(always)]
        fn div(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vdivq_f32(a, b) }
        }
        #[inline(always)]
        fn neg(self, a: float32x4_t) -> float32x4_t {
            unsafe { vnegq_f32(a) }
        }
        #[inline(always)]
        fn fma(self, a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
            unsafe { vfmaq_f32(c, a, b) }
        }
        #[inline(always)]
        fn sqrt(self, a: float32x4_t) -> float32x4_t {
            unsafe { vsqrtq_f32(a) }
        }
        #[inline(always)]
        fn min(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vminq_f32(a, b) }
        }
        #[inline(always)]
        fn max(self, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vmaxq_f32(a, b) }
        }
        #[inline(always)]
        fn le(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            unsafe { vcleq_f32(a, b) }
        }
        #[inline(always)]
        fn lt(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            unsafe { vcltq_f32(a, b) }
        }
        #[inline(always)]
        fn ge(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            unsafe { vcgeq_f32(a, b) }
        }
        #[inline(always)]
        fn gt(self, a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            unsafe { vcgtq_f32(a, b) }
        }
        #[inline(always)]
        fn mask_and(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
            unsafe { vandq_u32(a, b) }
        }
        #[inline(always)]
        fn mask_or(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
            unsafe { vorrq_u32(a, b) }
        }
        #[inline(always)]
        fn mask_not(self, a: uint32x4_t) -> uint32x4_t {
            unsafe { vmvnq_u32(a) }
        }
        #[inline(always)]
        fn select(self, m: uint32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
            unsafe { vbslq_f32(m, a, b) }
        }
        #[inline(always)]
        fn any(self, m: uint32x4_t) -> bool {
            unsafe { vmaxvq_u32(m) != 0 }
        }
        #[inline(always)]
        fn all(self, m: uint32x4_t) -> bool {
            unsafe { vminvq_u32(m) == u32::MAX }
        }
        #[inline(always)]
        fn reduce_sum(self, v: float32x4_t) -> bf16 {
            bf16::from_f32(unsafe { vaddvq_f32(v) })
        }
        #[inline(always)]
        fn reduce_min(self, v: float32x4_t) -> bf16 {
            bf16::from_f32(unsafe { vminvq_f32(v) })
        }
        #[inline(always)]
        fn reduce_max(self, v: float32x4_t) -> bf16 {
            bf16::from_f32(unsafe { vmaxvq_f32(v) })
        }
    }
}
