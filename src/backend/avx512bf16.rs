//! AVX-512-BF16 element-wise backend for `x86_64`.
//!
//! `bf16` has no native ALU, so this still computes in `f32x16` (16 lanes) exactly like the plain
//! [`Avx512`](super::avx512::Avx512) `bf16` path — but the load/store boundary uses the hardware
//! `vcvtneps2bf16`/widen conversions ([`crate::arch::avx512bf16`]) instead of a scalar
//! `bf16`↔`f32` loop. Every arithmetic/compare/reduce op delegates to `Avx512`'s `f32` backend, so
//! there is exactly one implementation of the math. Reached only through runtime `avx512bf16`
//! detection; without it, `bf16` falls back to the software-converting `Avx512`/`Avx2` paths.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::arch::avx512bf16 as p;
use crate::backend::Backend;
use crate::backend::avx512::Avx512;
use half::bf16;

/// AVX-512-BF16 execution token. Construct only when the CPU supports `avx512bf16`.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Bf16(());

impl Avx512Bf16 {
    #[cfg(feature = "std")]
    #[inline]
    // Unused when the build statically pins `avx512bf16` (dispatch takes the `new_unchecked` branch).
    #[allow(dead_code)]
    pub fn detect() -> Option<Self> {
        if is_x86_feature_detected!("avx512bf16")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
        {
            Some(Self(()))
        } else {
            None
        }
    }
    /// # Safety
    /// The CPU must support `avx512bf16` (and `avx512f`/`avx512bw`).
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }

    /// The shared `f32x16` math: `avx512bf16` implies `avx512f`, so this token is always valid here.
    /// Returned as `impl Backend<f32, …>` so op resolution picks `Avx512`'s `f32` impl (it also
    /// implements `Backend<f64>`/`Backend<bf16>`, which would otherwise be ambiguous).
    #[inline(always)]
    fn f32(self) -> impl Backend<f32, Vector = __m512, Mask = __mmask16> {
        unsafe { Avx512::new_unchecked() }
    }
}

impl Backend<bf16> for Avx512Bf16 {
    type Vector = __m512; // f32x16 (compute precision)
    type Mask = __mmask16;

    #[inline(always)]
    fn lanes(self) -> usize {
        16
    }
    #[inline(always)]
    fn splat(self, v: bf16) -> __m512 {
        self.f32().splat(v.to_f32())
    }
    #[inline(always)]
    fn load(self, s: &[bf16]) -> __m512 {
        debug_assert_eq!(s.len(), 16);
        unsafe { p::widen(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m512, s: &mut [bf16]) {
        debug_assert_eq!(s.len(), 16);
        unsafe { p::narrow(v, s.as_mut_ptr()) }
    }
    #[inline(always)]
    fn add(self, a: __m512, b: __m512) -> __m512 {
        self.f32().add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: __m512, b: __m512) -> __m512 {
        self.f32().sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: __m512, b: __m512) -> __m512 {
        self.f32().mul(a, b)
    }
    #[inline(always)]
    fn div(self, a: __m512, b: __m512) -> __m512 {
        self.f32().div(a, b)
    }
    #[inline(always)]
    fn neg(self, a: __m512) -> __m512 {
        self.f32().neg(a)
    }
    #[inline(always)]
    fn fma(self, a: __m512, b: __m512, c: __m512) -> __m512 {
        self.f32().fma(a, b, c)
    }
    #[inline(always)]
    fn sqrt(self, a: __m512) -> __m512 {
        self.f32().sqrt(a)
    }
    #[inline(always)]
    fn min(self, a: __m512, b: __m512) -> __m512 {
        self.f32().min(a, b)
    }
    #[inline(always)]
    fn max(self, a: __m512, b: __m512) -> __m512 {
        self.f32().max(a, b)
    }
    #[inline(always)]
    fn le(self, a: __m512, b: __m512) -> __mmask16 {
        self.f32().le(a, b)
    }
    #[inline(always)]
    fn lt(self, a: __m512, b: __m512) -> __mmask16 {
        self.f32().lt(a, b)
    }
    #[inline(always)]
    fn ge(self, a: __m512, b: __m512) -> __mmask16 {
        self.f32().ge(a, b)
    }
    #[inline(always)]
    fn gt(self, a: __m512, b: __m512) -> __mmask16 {
        self.f32().gt(a, b)
    }
    #[inline(always)]
    fn mask_and(self, a: __mmask16, b: __mmask16) -> __mmask16 {
        self.f32().mask_and(a, b)
    }
    #[inline(always)]
    fn mask_or(self, a: __mmask16, b: __mmask16) -> __mmask16 {
        self.f32().mask_or(a, b)
    }
    #[inline(always)]
    fn mask_not(self, a: __mmask16) -> __mmask16 {
        self.f32().mask_not(a)
    }
    #[inline(always)]
    fn select(self, m: __mmask16, a: __m512, b: __m512) -> __m512 {
        self.f32().select(m, a, b)
    }
    #[inline(always)]
    fn any(self, m: __mmask16) -> bool {
        self.f32().any(m)
    }
    #[inline(always)]
    fn all(self, m: __mmask16) -> bool {
        self.f32().all(m)
    }
    #[inline(always)]
    fn mask_bitmask(self, m: __mmask16) -> u32 {
        m as u32
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m512) -> bf16 {
        bf16::from_f32(self.f32().reduce_sum(v))
    }
    #[inline(always)]
    fn reduce_min(self, v: __m512) -> bf16 {
        bf16::from_f32(self.f32().reduce_min(v))
    }
    #[inline(always)]
    fn reduce_max(self, v: __m512) -> bf16 {
        bf16::from_f32(self.f32().reduce_max(v))
    }
}
