//! Hand-written SSE4.1 backend for `x86_64` (4-wide f32, 2-wide f64).
//! SSE4.1 has no FMA, so [`Backend::fma`] is `a*b + c` (two roundings).
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
    // Dead when the build baseline statically guarantees sse4.1 (x86-64-v2+) or wider; dispatch
    // then uses `new_unchecked`.
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
    // Used only by the statically-pinned and no-std dispatch paths; std builds go through
    // runtime `detect`.
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Backend<f32> for Sse4 {
    type Vector = __m128;
    type Mask = __m128;

    type IVector = __m128i;
    #[inline(always)]
    fn iload(self, s: &[u32]) -> __m128i {
        debug_assert_eq!(s.len(), 4);
        unsafe { si_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn istore(self, v: __m128i, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 4);
        unsafe { si_store(out.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn isplat(self, v: u32) -> __m128i {
        unsafe { si_splat(v) }
    }
    #[inline(always)]
    fn iadd(self, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_add(a, b) }
    }
    #[inline(always)]
    fn isub(self, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_sub(a, b) }
    }
    #[inline(always)]
    fn imul(self, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_mul(a, b) }
    }
    #[inline(always)]
    fn iand(self, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_and(a, b) }
    }
    #[inline(always)]
    fn ior(self, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_or(a, b) }
    }
    #[inline(always)]
    fn ixor(self, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_xor(a, b) }
    }
    #[inline(always)]
    fn inot(self, a: __m128i) -> __m128i {
        unsafe { si_xor(a, si_splat(u32::MAX)) }
    }
    #[inline(always)]
    fn ishl(self, a: __m128i, k: u32) -> __m128i {
        debug_assert!(k < 32);
        unsafe { si_shl(a, k) }
    }
    #[inline(always)]
    fn ishr(self, a: __m128i, k: u32) -> __m128i {
        debug_assert!(k < 32);
        unsafe { si_shr(a, k) }
    }
    #[inline(always)]
    fn ishr_arith(self, a: __m128i, k: u32) -> __m128i {
        debug_assert!(k < 32);
        unsafe { si_sra(a, k) }
    }
    #[inline(always)]
    fn ieq(self, a: __m128i, b: __m128i) -> __m128 {
        unsafe { si_eq(a, b) }
    }
    #[inline(always)]
    fn ilt_u(self, a: __m128i, b: __m128i) -> __m128 {
        unsafe { si_lt_u(a, b) }
    }
    #[inline(always)]
    fn ilt_s(self, a: __m128i, b: __m128i) -> __m128 {
        unsafe { si_lt_s(a, b) }
    }
    #[inline(always)]
    fn iselect(self, m: __m128, a: __m128i, b: __m128i) -> __m128i {
        unsafe { si_select(m, a, b) }
    }
    #[inline(always)]
    fn to_bits(self, v: __m128) -> __m128i {
        unsafe { si_from_ps(v) }
    }
    #[inline(always)]
    fn from_bits(self, v: __m128i) -> __m128 {
        unsafe { si_to_ps(v) }
    }

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
    fn abs(self, a: __m128) -> __m128 {
        unsafe { s_abs(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m128, b: __m128, c: __m128) -> __m128 {
        unsafe { s_add(s_mul(a, b), c) }
    }
    #[inline(always)]
    fn madd(self, a: __m128, b: __m128, acc: __m128) -> __m128 {
        <Self as Backend<f32>>::fma(self, a, b, acc)
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
    fn mask_bitmask(self, m: __m128) -> u32 {
        unsafe { _mm_movemask_ps(m) as u32 }
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
    // IEEE minimumNumber: `minps` yields `b` when `a` is NaN; the blend patches the b-is-NaN case.
    let m = _mm_min_ps(a, b);
    _mm_blendv_ps(m, a, _mm_cmpunord_ps(b, b))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_max(a: __m128, b: __m128) -> __m128 {
    let m = _mm_max_ps(a, b);
    _mm_blendv_ps(m, a, _mm_cmpunord_ps(b, b))
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
/// Clear the sign bit: one `andps`, cheaper than `max(a, -a)`.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_abs(a: __m128) -> __m128 {
    _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF)))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn s_reduce<const OP: i32>(v: __m128) -> f32 {
    #[inline(always)]
    unsafe fn combine<const OP: i32>(a: __m128, b: __m128) -> __m128 {
        match OP {
            1 => s_min(a, b),
            2 => s_max(a, b),
            _ => _mm_add_ps(a, b),
        }
    }
    let shuf = _mm_movehdup_ps(v); // [1,1,3,3]
    let d = combine::<OP>(v, shuf);
    let shuf2 = _mm_movehl_ps(shuf, d);
    let r = combine::<OP>(d, shuf2);
    _mm_cvtss_f32(r)
}

macro_rules! si_binop {
    ($name:ident, $intr:ident) => {
        #[target_feature(enable = "sse4.1")]
        #[inline]
        unsafe fn $name(a: __m128i, b: __m128i) -> __m128i {
            $intr(a, b)
        }
    };
}

si_binop!(si_add, _mm_add_epi32);
si_binop!(si_sub, _mm_sub_epi32);
si_binop!(si_mul, _mm_mullo_epi32);
si_binop!(si_and, _mm_and_si128);
si_binop!(si_or, _mm_or_si128);
si_binop!(si_xor, _mm_xor_si128);

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_load(p: *const u32) -> __m128i {
    _mm_loadu_si128(p as *const __m128i)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_store(p: *mut u32, v: __m128i) {
    _mm_storeu_si128(p as *mut __m128i, v)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_splat(v: u32) -> __m128i {
    _mm_set1_epi32(v as i32)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_shl(a: __m128i, k: u32) -> __m128i {
    _mm_sll_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_shr(a: __m128i, k: u32) -> __m128i {
    _mm_srl_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_sra(a: __m128i, k: u32) -> __m128i {
    _mm_sra_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_eq(a: __m128i, b: __m128i) -> __m128 {
    _mm_castsi128_ps(_mm_cmpeq_epi32(a, b))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_lt_s(a: __m128i, b: __m128i) -> __m128 {
    _mm_castsi128_ps(_mm_cmplt_epi32(a, b))
}
/// Unsigned `<` via the sign-flip trick: biasing both operands by `1 << 31` maps unsigned order
/// onto signed order, which is the only integer compare pre-AVX-512 hardware has.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_lt_u(a: __m128i, b: __m128i) -> __m128 {
    let bias = _mm_set1_epi32(i32::MIN);
    _mm_castsi128_ps(_mm_cmplt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_select(m: __m128, a: __m128i, b: __m128i) -> __m128i {
    _mm_blendv_epi8(b, a, _mm_castps_si128(m))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_from_ps(v: __m128) -> __m128i {
    _mm_castps_si128(v)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_to_ps(v: __m128i) -> __m128 {
    _mm_castsi128_ps(v)
}

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_neg(a: __m128i) -> __m128i {
    _mm_sub_epi32(_mm_setzero_si128(), a)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_abs(a: __m128i) -> __m128i {
    _mm_abs_epi32(a)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_min_u(a: __m128i, b: __m128i) -> __m128i {
    _mm_min_epu32(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_max_u(a: __m128i, b: __m128i) -> __m128i {
    _mm_max_epu32(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_min_s(a: __m128i, b: __m128i) -> __m128i {
    _mm_min_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_max_s(a: __m128i, b: __m128i) -> __m128i {
    _mm_max_epi32(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_or_ps(a: __m128, b: __m128) -> __m128 {
    _mm_or_ps(a, b)
}

/// The 32-bit integer element backends: 4-lane `__m128i` with the `f32` impl's `__m128` mask
/// convention, so integer and float compares compose. Arithmetic is wrapping.
macro_rules! sse4_int_backend {
    ($t:ty, $lt:ident, $min:ident, $max:ident, $abs:expr, $shr:ident) => {
        impl Backend<$t> for Sse4 {
            type Vector = __m128i;
            type Mask = __m128;

            #[inline(always)]
            fn lanes(self) -> usize {
                4
            }
            #[inline(always)]
            fn splat(self, v: $t) -> __m128i {
                unsafe { si_splat(v as u32) }
            }
            #[inline(always)]
            fn load(self, s: &[$t]) -> __m128i {
                debug_assert_eq!(s.len(), 4);
                unsafe { si_load(s.as_ptr() as *const u32) }
            }
            #[inline(always)]
            fn store(self, v: __m128i, s: &mut [$t]) {
                debug_assert_eq!(s.len(), 4);
                unsafe { si_store(s.as_mut_ptr() as *mut u32, v) }
            }
            #[inline(always)]
            fn add(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_add(a, b) }
            }
            #[inline(always)]
            fn sub(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_sub(a, b) }
            }
            #[inline(always)]
            fn mul(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_mul(a, b) }
            }
            #[inline(always)]
            fn neg(self, a: __m128i) -> __m128i {
                unsafe { si_neg(a) }
            }
            #[inline(always)]
            #[allow(unused_unsafe)] // identity for the u32 arm, intrinsic for i32
            fn abs(self, a: __m128i) -> __m128i {
                unsafe { ($abs)(a) }
            }
            #[inline(always)]
            fn min(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { $min(a, b) }
            }
            #[inline(always)]
            fn max(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { $max(a, b) }
            }
            #[inline(always)]
            fn le(self, a: __m128i, b: __m128i) -> __m128 {
                unsafe { si_or_ps($lt(a, b), si_eq(a, b)) }
            }
            #[inline(always)]
            fn lt(self, a: __m128i, b: __m128i) -> __m128 {
                unsafe { $lt(a, b) }
            }
            #[inline(always)]
            fn ge(self, a: __m128i, b: __m128i) -> __m128 {
                unsafe { si_or_ps($lt(b, a), si_eq(a, b)) }
            }
            #[inline(always)]
            fn gt(self, a: __m128i, b: __m128i) -> __m128 {
                unsafe { $lt(b, a) }
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
            fn select(self, m: __m128, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_select(m, a, b) }
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
            fn mask_bitmask(self, m: __m128) -> u32 {
                unsafe { _mm_movemask_ps(m) as u32 }
            }
            #[inline(always)]
            fn reduce_sum(self, v: __m128i) -> $t {
                let mut b = [0u32; 4];
                unsafe { si_store(b.as_mut_ptr(), v) };
                b.iter().fold(0 as $t, |acc, &x| acc.wrapping_add(x as $t))
            }
            #[inline(always)]
            fn reduce_min(self, v: __m128i) -> $t {
                let mut b = [0u32; 4];
                unsafe { si_store(b.as_mut_ptr(), v) };
                b.iter().map(|&x| x as $t).min().unwrap()
            }
            #[inline(always)]
            fn reduce_max(self, v: __m128i) -> $t {
                let mut b = [0u32; 4];
                unsafe { si_store(b.as_mut_ptr(), v) };
                b.iter().map(|&x| x as $t).max().unwrap()
            }
            #[inline(always)]
            fn shl(self, a: __m128i, k: u32) -> __m128i {
                debug_assert!(k < 32);
                unsafe { si_shl(a, k) }
            }
            #[inline(always)]
            fn shr(self, a: __m128i, k: u32) -> __m128i {
                debug_assert!(k < 32);
                unsafe { $shr(a, k) }
            }
            #[inline(always)]
            fn bit_and(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_and(a, b) }
            }
            #[inline(always)]
            fn bit_or(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_or(a, b) }
            }
            #[inline(always)]
            fn bit_xor(self, a: __m128i, b: __m128i) -> __m128i {
                unsafe { si_xor(a, b) }
            }
            #[inline(always)]
            fn bit_not(self, a: __m128i) -> __m128i {
                unsafe { si_xor(a, si_splat(u32::MAX)) }
            }

            type IVector = __m128i;
            #[inline(always)]
            fn iload(self, s: &[u32]) -> __m128i {
                debug_assert_eq!(s.len(), 4);
                unsafe { si_load(s.as_ptr()) }
            }
            #[inline(always)]
            fn istore(self, v: __m128i, out: &mut [u32]) {
                debug_assert_eq!(out.len(), 4);
                unsafe { si_store(out.as_mut_ptr(), v) }
            }
            #[inline(always)]
            fn to_bits(self, v: __m128i) -> __m128i {
                v
            }
            #[inline(always)]
            fn from_bits(self, v: __m128i) -> __m128i {
                v
            }
        }
    };
}

sse4_int_backend!(u32, si_lt_u_i, si_min_u, si_max_u, |a| a, si_shr);
sse4_int_backend!(i32, si_lt_s_i, si_min_s, si_max_s, |a| si_abs(a), si_sra);

#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_lt_u_i(a: __m128i, b: __m128i) -> __m128 {
    si_lt_u(a, b)
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn si_lt_s_i(a: __m128i, b: __m128i) -> __m128 {
    si_lt_s(a, b)
}

impl Backend<f64> for Sse4 {
    type Vector = __m128d;
    type Mask = __m128d;

    type IVector = [u32; 2];
    #[inline(always)]
    fn iload(self, s: &[u32]) -> [u32; 2] {
        let mut v = [0u32; 2];
        v.copy_from_slice(s);
        v
    }
    #[inline(always)]
    fn istore(self, v: [u32; 2], out: &mut [u32]) {
        out.copy_from_slice(&v);
    }

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
    fn abs(self, a: __m128d) -> __m128d {
        unsafe { d_abs(a) }
    }
    #[inline(always)]
    fn fma(self, a: __m128d, b: __m128d, c: __m128d) -> __m128d {
        unsafe { _mm_add_pd_(d_mul(a, b), c) }
    }
    #[inline(always)]
    fn madd(self, a: __m128d, b: __m128d, acc: __m128d) -> __m128d {
        <Self as Backend<f64>>::fma(self, a, b, acc)
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
    fn mask_bitmask(self, m: __m128d) -> u32 {
        unsafe { _mm_movemask_pd(m) as u32 }
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
    let m = _mm_min_pd(a, b);
    _mm_blendv_pd(m, a, _mm_cmpunord_pd(b, b))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_max(a: __m128d, b: __m128d) -> __m128d {
    let m = _mm_max_pd(a, b);
    _mm_blendv_pd(m, a, _mm_cmpunord_pd(b, b))
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
/// Clear the sign bit: one `andpd`, cheaper than `max(a, -a)`.
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_abs(a: __m128d) -> __m128d {
    _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFF_FFFF_FFFF_FFFF)))
}
#[target_feature(enable = "sse4.1")]
#[inline]
unsafe fn d_reduce<const OP: i32>(v: __m128d) -> f64 {
    let sh = _mm_unpackhi_pd(v, v);
    let r = match OP {
        1 => d_min(v, sh),
        2 => d_max(v, sh),
        _ => _mm_add_pd(v, sh),
    };
    _mm_cvtsd_f64(r)
}

// f16/bf16 on SSE4: f32x4 widen-compute-narrow with scalar boundary conversions (SSE4 can't
// assume F16C, and bf16 has no native ALU). Every compute op delegates to the shared f32 path;
// only the load/store boundary and the reduction return type differ, so one macro generates both.
macro_rules! sse4_widen_half {
    ($t:ident, $modname:ident) => {
        mod $modname {
            use super::*;
            use half::$t;

            #[target_feature(enable = "sse4.1")]
            unsafe fn h_load(s: &[$t]) -> __m128 {
                let t = [s[0].to_f32(), s[1].to_f32(), s[2].to_f32(), s[3].to_f32()];
                s_load(t.as_ptr())
            }
            #[target_feature(enable = "sse4.1")]
            unsafe fn h_store(v: __m128, s: &mut [$t]) {
                let mut t = [0f32; 4];
                s_store(t.as_mut_ptr(), v);
                for (d, x) in s.iter_mut().zip(t) {
                    *d = $t::from_f32(x);
                }
            }

            impl Backend<$t> for Sse4 {
                type Vector = __m128;
                type Mask = __m128;

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
                fn splat(self, v: $t) -> __m128 {
                    unsafe { s_splat(v.to_f32()) }
                }
                #[inline(always)]
                fn load(self, s: &[$t]) -> __m128 {
                    debug_assert_eq!(s.len(), 4);
                    unsafe { h_load(s) }
                }
                #[inline(always)]
                fn store(self, v: __m128, s: &mut [$t]) {
                    debug_assert_eq!(s.len(), 4);
                    unsafe { h_store(v, s) }
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
                #[allow(unused_unsafe)]
            fn abs(self, a: __m128) -> __m128 {
                    unsafe { s_abs(a) }
                }
                #[inline(always)]
                fn fma(self, a: __m128, b: __m128, c: __m128) -> __m128 {
                    unsafe { s_add(s_mul(a, b), c) }
                }
                            #[inline(always)]
                fn madd(self, a: __m128, b: __m128, acc: __m128) -> __m128 {
                    <Self as Backend<$t>>::fma(self, a, b, acc)
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
                fn mask_bitmask(self, m: __m128) -> u32 {
                    unsafe { _mm_movemask_ps(m) as u32 }
                }
                #[inline(always)]
                fn reduce_sum(self, v: __m128) -> $t {
                    $t::from_f32(unsafe { s_reduce::<0>(v) })
                }
                #[inline(always)]
                fn reduce_min(self, v: __m128) -> $t {
                    $t::from_f32(unsafe { s_reduce::<1>(v) })
                }
                #[inline(always)]
                fn reduce_max(self, v: __m128) -> $t {
                    $t::from_f32(unsafe { s_reduce::<2>(v) })
                }
            }
        }
    };
}
sse4_widen_half!(f16, f16_impl);
sse4_widen_half!(bf16, bf16_impl);
