//! AVX-512-BF16 element-wise backend for `x86_64`: compute stays `f32x16`, with hardware
//! bf16 conversions at the load/store boundary.
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
    // Dead when the build statically pins `avx512bf16`; dispatch then uses `new_unchecked`.
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

    /// The shared `f32x16` math: `avx512bf16` implies `avx512f`, so the token is always valid.
    /// Returned as `impl Backend<f32, …>` so op resolution pins `Avx512`'s `f32` impl rather
    /// than its ambiguous `Backend<f64>`/`Backend<bf16>` impls.
    #[inline(always)]
    fn f32(self) -> impl Backend<f32, Vector = __m512, Mask = __mmask16> {
        unsafe { Avx512::new_unchecked() }
    }
}

impl Backend<bf16> for Avx512Bf16 {
    type Vector = __m512; // f32x16 (compute precision)
    type Mask = __mmask16;

    type IVector = [u32; 16];
    #[inline(always)]
    fn iload(self, s: &[u32]) -> [u32; 16] {
        let mut v = [0u32; 16];
        v.copy_from_slice(s);
        v
    }
    #[inline(always)]
    fn istore(self, v: [u32; 16], out: &mut [u32]) {
        out.copy_from_slice(&v);
    }

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
    fn madd(self, a: __m512, b: __m512, acc: __m512) -> __m512 {
        <Self as Backend<bf16>>::fma(self, a, b, acc)
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

// `avx512bf16` implies `avx512f`, so every non-bf16 element rides the plain AVX-512 impls.
crate::backend::delegate_float_element!(Avx512Bf16, f32, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_float_element!(Avx512Bf16, f64, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_float_element!(Avx512Bf16, half::f16, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_int_element!(Avx512Bf16, u32, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
crate::backend::delegate_int_element!(Avx512Bf16, i32, crate::backend::avx512::Avx512, unsafe {
    crate::backend::avx512::Avx512::new_unchecked()
});
