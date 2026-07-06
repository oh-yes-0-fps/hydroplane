//! Native AVX-512-FP16 backend for `x86_64`: 32-wide hardware `f16` arithmetic.
//! Works on stable: ops are raw `asm!` ([`crate::arch::avx512fp16`]) with `half::f16` elements
//! in a plain [`__m512i`] carrier.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::arch::avx512fp16 as p;
use crate::backend::Backend;
use half::f16 as Half;

/// AVX-512-FP16 execution token. Construct only when the CPU supports `avx512fp16`.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Fp16(());

impl Avx512Fp16 {
    #[cfg(feature = "std")]
    #[inline]
    // Dead when the build statically pins `avx512fp16`; dispatch then uses `new_unchecked`.
    #[allow(dead_code)]
    pub fn detect() -> Option<Self> {
        // `vpblendmw`/`vcmpph`+`kmovd` ride on AVX-512-BW; every `avx512fp16` part has BW, but
        // the check keeps the primitives' `target_feature` set honest.
        if is_x86_feature_detected!("avx512fp16")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
        {
            Some(Self(()))
        } else {
            None
        }
    }
    /// # Safety
    /// The CPU must support `avx512fp16` (and `avx512f`/`avx512bw`).
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Backend<Half> for Avx512Fp16 {
    type Vector = __m512i; // 32 × f16 (raw bits)
    type Mask = u32;

    type IVector = [u32; 32];
    #[inline(always)]
    fn iload(self, s: &[u32]) -> [u32; 32] {
        let mut v = [0u32; 32];
        v.copy_from_slice(s);
        v
    }
    #[inline(always)]
    fn istore(self, v: [u32; 32], out: &mut [u32]) {
        out.copy_from_slice(&v);
    }

    #[inline(always)]
    fn lanes(self) -> usize {
        32
    }
    #[inline(always)]
    fn splat(self, v: Half) -> __m512i {
        unsafe { p::splat(v) }
    }
    #[inline(always)]
    fn load(self, s: &[Half]) -> __m512i {
        debug_assert_eq!(s.len(), 32);
        unsafe { p::load(s.as_ptr()) }
    }
    #[inline(always)]
    fn store(self, v: __m512i, s: &mut [Half]) {
        debug_assert_eq!(s.len(), 32);
        unsafe { p::store(s.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn add(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::add(a, b) }
    }
    #[inline(always)]
    fn sub(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::sub(a, b) }
    }
    #[inline(always)]
    fn mul(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::mul(a, b) }
    }
    #[inline(always)]
    fn div(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::div(a, b) }
    }
    #[inline(always)]
    fn neg(self, a: __m512i) -> __m512i {
        unsafe { p::neg(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
        unsafe { p::fma(a, b, c) }
    }
    #[inline(always)]
    fn madd(self, a: __m512i, b: __m512i, acc: __m512i) -> __m512i {
        <Self as Backend<Half>>::fma(self, a, b, acc)
    }
    #[inline(always)]
    fn sqrt(self, a: __m512i) -> __m512i {
        unsafe { p::sqrt(a) }
    }
    #[inline(always)]
    fn min(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::min(a, b) }
    }
    #[inline(always)]
    fn max(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::max(a, b) }
    }
    #[inline(always)]
    fn le(self, a: __m512i, b: __m512i) -> u32 {
        unsafe { p::cmp::<{ p::CMP_LE_OQ }>(a, b) }
    }
    #[inline(always)]
    fn lt(self, a: __m512i, b: __m512i) -> u32 {
        unsafe { p::cmp::<{ p::CMP_LT_OQ }>(a, b) }
    }
    #[inline(always)]
    fn ge(self, a: __m512i, b: __m512i) -> u32 {
        unsafe { p::cmp::<{ p::CMP_GE_OQ }>(a, b) }
    }
    #[inline(always)]
    fn gt(self, a: __m512i, b: __m512i) -> u32 {
        unsafe { p::cmp::<{ p::CMP_GT_OQ }>(a, b) }
    }
    #[inline(always)]
    fn mask_and(self, a: u32, b: u32) -> u32 {
        a & b
    }
    #[inline(always)]
    fn mask_or(self, a: u32, b: u32) -> u32 {
        a | b
    }
    #[inline(always)]
    fn mask_not(self, a: u32) -> u32 {
        !a
    }
    #[inline(always)]
    fn select(self, m: u32, a: __m512i, b: __m512i) -> __m512i {
        unsafe { p::select(m, a, b) }
    }
    #[inline(always)]
    fn any(self, m: u32) -> bool {
        m != 0
    }
    #[inline(always)]
    fn all(self, m: u32) -> bool {
        m == u32::MAX
    }
    #[inline(always)]
    fn mask_bitmask(self, m: u32) -> u32 {
        m
    }
    #[inline(always)]
    fn reduce_sum(self, v: __m512i) -> Half {
        Half::from_f32(self.fold(v, |acc, x| acc + x, 0.0))
    }
    #[inline(always)]
    fn reduce_min(self, v: __m512i) -> Half {
        Half::from_f32(self.fold(v, f32::min, f32::INFINITY))
    }
    #[inline(always)]
    fn reduce_max(self, v: __m512i) -> Half {
        Half::from_f32(self.fold(v, f32::max, f32::NEG_INFINITY))
    }
}

impl Avx512Fp16 {
    /// Horizontal fold via a 32-element spill; reductions are rare enough for this to be fine.
    #[inline(always)]
    fn fold(self, v: __m512i, f: impl Fn(f32, f32) -> f32, init: f32) -> f32 {
        let mut tmp = [Half::ZERO; 32];
        self.store(v, &mut tmp);
        tmp.iter().fold(init, |acc, x| f(acc, x.to_f32()))
    }
}

// `avx512fp16` implies `avx512f`, so every non-f16 element rides the plain AVX-512 impls.
crate::backend::delegate_float_element!(Avx512Fp16, f32, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_float_element!(Avx512Fp16, f64, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_float_element!(Avx512Fp16, half::bf16, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_int_element!(Avx512Fp16, u32, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_int_element!(Avx512Fp16, i32, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
