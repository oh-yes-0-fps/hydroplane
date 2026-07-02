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

    type IVector = __m512i;
    #[inline(always)]
    fn iload(self, s: &[u32]) -> __m512i {
        debug_assert_eq!(s.len(), 16);
        unsafe { zi_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn istore(self, v: __m512i, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 16);
        unsafe { zi_store(out.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn isplat(self, v: u32) -> __m512i {
        unsafe { zi_splat(v) }
    }
    #[inline(always)]
    fn iadd(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_add(a, b) }
    }
    #[inline(always)]
    fn isub(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_sub(a, b) }
    }
    #[inline(always)]
    fn imul(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_mul(a, b) }
    }
    #[inline(always)]
    fn iand(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_and(a, b) }
    }
    #[inline(always)]
    fn ior(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_or(a, b) }
    }
    #[inline(always)]
    fn ixor(self, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_xor(a, b) }
    }
    #[inline(always)]
    fn inot(self, a: __m512i) -> __m512i {
        unsafe { zi_xor(a, zi_splat(u32::MAX)) }
    }
    #[inline(always)]
    fn ishl(self, a: __m512i, k: u32) -> __m512i {
        debug_assert!(k < 32);
        unsafe { zi_shl(a, k) }
    }
    #[inline(always)]
    fn ishr(self, a: __m512i, k: u32) -> __m512i {
        debug_assert!(k < 32);
        unsafe { zi_shr(a, k) }
    }
    #[inline(always)]
    fn ishr_arith(self, a: __m512i, k: u32) -> __m512i {
        debug_assert!(k < 32);
        unsafe { zi_sra(a, k) }
    }
    #[inline(always)]
    fn ieq(self, a: __m512i, b: __m512i) -> __mmask16 {
        unsafe { zi_eq(a, b) }
    }
    #[inline(always)]
    fn ilt_u(self, a: __m512i, b: __m512i) -> __mmask16 {
        unsafe { zi_lt_u(a, b) }
    }
    #[inline(always)]
    fn ilt_s(self, a: __m512i, b: __m512i) -> __mmask16 {
        unsafe { zi_lt_s(a, b) }
    }
    #[inline(always)]
    fn iselect(self, m: __mmask16, a: __m512i, b: __m512i) -> __m512i {
        unsafe { zi_select(m, a, b) }
    }
    #[inline(always)]
    fn to_bits(self, v: __m512) -> __m512i {
        unsafe { zi_from_ps(v) }
    }
    #[inline(always)]
    fn from_bits(self, v: __m512i) -> __m512 {
        unsafe { zi_to_ps(v) }
    }

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
    fn madd(self, a: __m512, b: __m512, acc: __m512) -> __m512 {
        <Self as Backend<f32>>::fma(self, a, b, acc)
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
        // minimumNumber fold: NaN lanes must not poison (or win) the reduction, so quiet them to
        // the identity first; an all-NaN register still reduces to NaN.
        unsafe {
            let nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(v, v);
            if nan == 0xffff {
                return f32::NAN;
            }
            _mm512_reduce_min_ps(_mm512_mask_mov_ps(v, nan, _mm512_set1_ps(f32::INFINITY)))
        }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m512) -> f32 {
        unsafe {
            let nan = _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(v, v);
            if nan == 0xffff {
                return f32::NAN;
            }
            _mm512_reduce_max_ps(_mm512_mask_mov_ps(v, nan, _mm512_set1_ps(f32::NEG_INFINITY)))
        }
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
    // IEEE minimumNumber: `vminps` already yields `b` when `a` is NaN; the masked move patches
    // the b-is-NaN case (bare `vminps` would return the NaN).
    let m = _mm512_min_ps(a, b);
    _mm512_mask_mov_ps(m, _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(b, b), a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn z_max(a: __m512, b: __m512) -> __m512 {
    let m = _mm512_max_ps(a, b);
    _mm512_mask_mov_ps(m, _mm512_cmp_ps_mask::<_CMP_UNORD_Q>(b, b), a)
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

macro_rules! zi_binop {
    ($name:ident, $intr:ident) => {
        #[target_feature(enable = "avx512f")]
        #[inline]
        unsafe fn $name(a: __m512i, b: __m512i) -> __m512i {
            $intr(a, b)
        }
    };
}

zi_binop!(zi_add, _mm512_add_epi32);
zi_binop!(zi_sub, _mm512_sub_epi32);
zi_binop!(zi_mul, _mm512_mullo_epi32);
zi_binop!(zi_and, _mm512_and_si512);
zi_binop!(zi_or, _mm512_or_si512);
zi_binop!(zi_xor, _mm512_xor_si512);

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_load(p: *const u32) -> __m512i {
    _mm512_loadu_si512(p as *const __m512i)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_store(p: *mut u32, v: __m512i) {
    _mm512_storeu_si512(p as *mut __m512i, v)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_splat(v: u32) -> __m512i {
    _mm512_set1_epi32(v as i32)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_shl(a: __m512i, k: u32) -> __m512i {
    _mm512_sll_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_shr(a: __m512i, k: u32) -> __m512i {
    _mm512_srl_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_sra(a: __m512i, k: u32) -> __m512i {
    _mm512_sra_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_eq(a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpeq_epi32_mask(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_lt_s(a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmplt_epi32_mask(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_lt_u(a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmplt_epu32_mask(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_select(m: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    _mm512_mask_mov_epi32(b, m, a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_from_ps(v: __m512) -> __m512i {
    _mm512_castps_si512(v)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_to_ps(v: __m512i) -> __m512 {
    _mm512_castsi512_ps(v)
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_neg(a: __m512i) -> __m512i {
    _mm512_sub_epi32(_mm512_setzero_si512(), a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_abs(a: __m512i) -> __m512i {
    _mm512_abs_epi32(a)
}

/// The 32-bit integer element backends: 16-lane `__m512i` with the `f32` impl's `__mmask16`
/// convention. Arithmetic is wrapping; compares are native unsigned/signed k-mask ops.
macro_rules! avx512_int_backend {
    ($t:ty, $le:ident, $lt:ident, $min:ident, $max:ident, $abs:expr, $shr:ident,
     $rmin:ident, $rmax:ident) => {
        impl Backend<$t> for Avx512 {
            type Vector = __m512i;
            type Mask = __mmask16;

            #[inline(always)]
            fn lanes(self) -> usize {
                16
            }
            #[inline(always)]
            fn splat(self, v: $t) -> __m512i {
                unsafe { zi_splat(v as u32) }
            }
            #[inline(always)]
            fn load(self, s: &[$t]) -> __m512i {
                debug_assert_eq!(s.len(), 16);
                unsafe { zi_load(s.as_ptr() as *const u32) }
            }
            #[inline(always)]
            fn store(self, v: __m512i, s: &mut [$t]) {
                debug_assert_eq!(s.len(), 16);
                unsafe { zi_store(s.as_mut_ptr() as *mut u32, v) }
            }
            #[inline(always)]
            fn add(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_add(a, b) }
            }
            #[inline(always)]
            fn sub(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_sub(a, b) }
            }
            #[inline(always)]
            fn mul(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_mul(a, b) }
            }
            #[inline(always)]
            fn neg(self, a: __m512i) -> __m512i {
                unsafe { zi_neg(a) }
            }
            #[inline(always)]
            fn abs(self, a: __m512i) -> __m512i {
                unsafe { ($abs)(a) }
            }
            #[inline(always)]
            fn min(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { $min(a, b) }
            }
            #[inline(always)]
            fn max(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { $max(a, b) }
            }
            #[inline(always)]
            fn le(self, a: __m512i, b: __m512i) -> __mmask16 {
                unsafe { $le(a, b) }
            }
            #[inline(always)]
            fn lt(self, a: __m512i, b: __m512i) -> __mmask16 {
                unsafe { $lt(a, b) }
            }
            #[inline(always)]
            fn ge(self, a: __m512i, b: __m512i) -> __mmask16 {
                unsafe { $le(b, a) }
            }
            #[inline(always)]
            fn gt(self, a: __m512i, b: __m512i) -> __mmask16 {
                unsafe { $lt(b, a) }
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
            fn select(self, m: __mmask16, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_select(m, a, b) }
            }
            #[inline(always)]
            fn any(self, m: __mmask16) -> bool {
                m != 0
            }
            #[inline(always)]
            fn all(self, m: __mmask16) -> bool {
                m == 0xffff
            }
            #[inline(always)]
            fn mask_bitmask(self, m: __mmask16) -> u32 {
                m as u32
            }
            #[inline(always)]
            fn reduce_sum(self, v: __m512i) -> $t {
                let mut b = [0u32; 16];
                unsafe { zi_store(b.as_mut_ptr(), v) };
                b.iter().fold(0 as $t, |acc, &x| acc.wrapping_add(x as $t))
            }
            #[inline(always)]
            fn reduce_min(self, v: __m512i) -> $t {
                unsafe { $rmin(v) as $t }
            }
            #[inline(always)]
            fn reduce_max(self, v: __m512i) -> $t {
                unsafe { $rmax(v) as $t }
            }
            #[inline(always)]
            fn shl(self, a: __m512i, k: u32) -> __m512i {
                debug_assert!(k < 32);
                unsafe { zi_shl(a, k) }
            }
            #[inline(always)]
            fn shr(self, a: __m512i, k: u32) -> __m512i {
                debug_assert!(k < 32);
                unsafe { $shr(a, k) }
            }
            #[inline(always)]
            fn bit_and(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_and(a, b) }
            }
            #[inline(always)]
            fn bit_or(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_or(a, b) }
            }
            #[inline(always)]
            fn bit_xor(self, a: __m512i, b: __m512i) -> __m512i {
                unsafe { zi_xor(a, b) }
            }
            #[inline(always)]
            fn bit_not(self, a: __m512i) -> __m512i {
                unsafe { zi_xor(a, zi_splat(u32::MAX)) }
            }

            type IVector = __m512i;
            #[inline(always)]
            fn iload(self, s: &[u32]) -> __m512i {
                debug_assert_eq!(s.len(), 16);
                unsafe { zi_load(s.as_ptr()) }
            }
            #[inline(always)]
            fn istore(self, v: __m512i, out: &mut [u32]) {
                debug_assert_eq!(out.len(), 16);
                unsafe { zi_store(out.as_mut_ptr(), v) }
            }
            #[inline(always)]
            fn to_bits(self, v: __m512i) -> __m512i {
                v
            }
            #[inline(always)]
            fn from_bits(self, v: __m512i) -> __m512i {
                v
            }
        }
    };
}

avx512_int_backend!(
    u32, _mm512_cmple_epu32_mask, _mm512_cmplt_epu32_mask, zi_min_u, zi_max_u, |a| a,
    zi_shr, _mm512_reduce_min_epu32, _mm512_reduce_max_epu32
);
avx512_int_backend!(
    i32, _mm512_cmple_epi32_mask, _mm512_cmplt_epi32_mask, zi_min_s, zi_max_s, |a| zi_abs(a),
    zi_sra, _mm512_reduce_min_epi32, _mm512_reduce_max_epi32
);

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_min_u(a: __m512i, b: __m512i) -> __m512i {
    _mm512_min_epu32(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_max_u(a: __m512i, b: __m512i) -> __m512i {
    _mm512_max_epu32(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_min_s(a: __m512i, b: __m512i) -> __m512i {
    _mm512_min_epi32(a, b)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zi_max_s(a: __m512i, b: __m512i) -> __m512i {
    _mm512_max_epi32(a, b)
}

impl Backend<f64> for Avx512 {
    type Vector = __m512d;
    type Mask = __mmask8;

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
    fn madd(self, a: __m512d, b: __m512d, acc: __m512d) -> __m512d {
        <Self as Backend<f64>>::fma(self, a, b, acc)
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
        unsafe {
            let nan = _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(v, v);
            if nan == 0xff {
                return f64::NAN;
            }
            _mm512_reduce_min_pd(_mm512_mask_mov_pd(v, nan, _mm512_set1_pd(f64::INFINITY)))
        }
    }
    #[inline(always)]
    fn reduce_max(self, v: __m512d) -> f64 {
        unsafe {
            let nan = _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(v, v);
            if nan == 0xff {
                return f64::NAN;
            }
            _mm512_reduce_max_pd(_mm512_mask_mov_pd(v, nan, _mm512_set1_pd(f64::NEG_INFINITY)))
        }
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
    let m = _mm512_min_pd(a, b);
    _mm512_mask_mov_pd(m, _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(b, b), a)
}
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn zd_max(a: __m512d, b: __m512d) -> __m512d {
    let m = _mm512_max_pd(a, b);
    _mm512_mask_mov_pd(m, _mm512_cmp_pd_mask::<_CMP_UNORD_Q>(b, b), a)
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
        fn madd(self, a: __m512, b: __m512, acc: __m512) -> __m512 {
            <Self as Backend<bf16>>::fma(self, a, b, acc)
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
        fn madd(self, a: __m512, b: __m512, acc: __m512) -> __m512 {
            <Self as Backend<f16>>::fma(self, a, b, acc)
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
