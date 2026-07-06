//! Hand-written AVX2 (+FMA) backend for `x86_64` (8-wide f32, 4-wide f64).
//! `f16` uses the F16C widen path (f32 compute, hardware convert at load/store); `bf16` widens
//! to f32 in software.
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
    // Dead when the build baseline statically guarantees avx2 (x86-64-v3+); dispatch then uses
    // `new_unchecked`.
    #[cfg(feature = "std")]
    #[allow(dead_code)]
    #[inline]
    pub fn detect() -> Option<Self> {
        let ok = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        // The f16 widen path uses F16C; require it so an `Avx2` token also implies a sound
        // `Backend<f16>`. F16C is present on every AVX2 CPU.
        let ok = ok && is_x86_feature_detected!("f16c");
        if ok { Some(Self(())) } else { None }
    }

    /// # Safety
    /// The caller guarantees the running CPU supports `avx2` and `fma`. Calling any
    /// [`Backend`] method on a token built this way on an unsupported CPU is UB.
    // Dead only on a std build with no static avx2 guarantee, which goes through runtime `detect`.
    #[allow(dead_code)]
    #[inline]
    pub const unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

impl Backend<f32> for Avx2 {
    type Vector = __m256;
    type Mask = __m256;

    type IVector = __m256i;
    #[inline(always)]
    fn iload(self, s: &[u32]) -> __m256i {
        debug_assert_eq!(s.len(), 8);
        unsafe { yi_load(s.as_ptr()) }
    }
    #[inline(always)]
    fn istore(self, v: __m256i, out: &mut [u32]) {
        debug_assert_eq!(out.len(), 8);
        unsafe { yi_store(out.as_mut_ptr(), v) }
    }
    #[inline(always)]
    fn isplat(self, v: u32) -> __m256i {
        unsafe { yi_splat(v) }
    }
    #[inline(always)]
    fn iadd(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_add(a, b) }
    }
    #[inline(always)]
    fn isub(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_sub(a, b) }
    }
    #[inline(always)]
    fn imul(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_mul(a, b) }
    }
    #[inline(always)]
    fn iand(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_and(a, b) }
    }
    #[inline(always)]
    fn ior(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_or(a, b) }
    }
    #[inline(always)]
    fn ixor(self, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_xor(a, b) }
    }
    #[inline(always)]
    fn inot(self, a: __m256i) -> __m256i {
        unsafe { yi_xor(a, yi_splat(u32::MAX)) }
    }
    #[inline(always)]
    fn ishl(self, a: __m256i, k: u32) -> __m256i {
        debug_assert!(k < 32);
        unsafe { yi_shl(a, k) }
    }
    #[inline(always)]
    fn ishr(self, a: __m256i, k: u32) -> __m256i {
        debug_assert!(k < 32);
        unsafe { yi_shr(a, k) }
    }
    #[inline(always)]
    fn ishr_arith(self, a: __m256i, k: u32) -> __m256i {
        debug_assert!(k < 32);
        unsafe { yi_sra(a, k) }
    }
    #[inline(always)]
    fn ieq(self, a: __m256i, b: __m256i) -> __m256 {
        unsafe { yi_eq(a, b) }
    }
    #[inline(always)]
    fn ilt_u(self, a: __m256i, b: __m256i) -> __m256 {
        unsafe { yi_lt_u(a, b) }
    }
    #[inline(always)]
    fn ilt_s(self, a: __m256i, b: __m256i) -> __m256 {
        unsafe { yi_lt_s(a, b) }
    }
    #[inline(always)]
    fn iselect(self, m: __m256, a: __m256i, b: __m256i) -> __m256i {
        unsafe { yi_select(m, a, b) }
    }
    #[inline(always)]
    fn to_bits(self, v: __m256) -> __m256i {
        unsafe { yi_from_ps(v) }
    }
    #[inline(always)]
    fn from_bits(self, v: __m256i) -> __m256 {
        unsafe { yi_to_ps(v) }
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
/// Clear the sign bit: one `andps`, cheaper than `max(a, -a)`.
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
    // IEEE minimumNumber: `vminps` yields `b` when `a` is NaN; the blend patches the b-is-NaN case.
    let m = _mm256_min_ps(a, b);
    _mm256_blendv_ps(m, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f32_max(a: __m256, b: __m256) -> __m256 {
    let m = _mm256_max_ps(a, b);
    _mm256_blendv_ps(m, a, _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b))
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

macro_rules! yi_binop {
    ($name:ident, $intr:ident) => {
        #[target_feature(enable = "avx2,fma")]
        #[inline]
        unsafe fn $name(a: __m256i, b: __m256i) -> __m256i {
            $intr(a, b)
        }
    };
}

yi_binop!(yi_add, _mm256_add_epi32);
yi_binop!(yi_sub, _mm256_sub_epi32);
yi_binop!(yi_mul, _mm256_mullo_epi32);
yi_binop!(yi_and, _mm256_and_si256);
yi_binop!(yi_or, _mm256_or_si256);
yi_binop!(yi_xor, _mm256_xor_si256);

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_load(p: *const u32) -> __m256i {
    _mm256_loadu_si256(p as *const __m256i)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_store(p: *mut u32, v: __m256i) {
    _mm256_storeu_si256(p as *mut __m256i, v)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_splat(v: u32) -> __m256i {
    _mm256_set1_epi32(v as i32)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_shl(a: __m256i, k: u32) -> __m256i {
    _mm256_sll_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_shr(a: __m256i, k: u32) -> __m256i {
    _mm256_srl_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_sra(a: __m256i, k: u32) -> __m256i {
    _mm256_sra_epi32(a, _mm_cvtsi32_si128(k as i32))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_eq(a: __m256i, b: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_cmpeq_epi32(a, b))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_lt_s(a: __m256i, b: __m256i) -> __m256 {
    _mm256_castsi256_ps(_mm256_cmpgt_epi32(b, a))
}
/// Unsigned `<` via the sign-flip trick (see the SSE4 backend).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_lt_u(a: __m256i, b: __m256i) -> __m256 {
    let bias = _mm256_set1_epi32(i32::MIN);
    _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_select(m: __m256, a: __m256i, b: __m256i) -> __m256i {
    _mm256_blendv_epi8(b, a, _mm256_castps_si256(m))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_from_ps(v: __m256) -> __m256i {
    _mm256_castps_si256(v)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_to_ps(v: __m256i) -> __m256 {
    _mm256_castsi256_ps(v)
}

#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_neg(a: __m256i) -> __m256i {
    _mm256_sub_epi32(_mm256_setzero_si256(), a)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_abs(a: __m256i) -> __m256i {
    _mm256_abs_epi32(a)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_min_u(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epu32(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_max_u(a: __m256i, b: __m256i) -> __m256i {
    _mm256_max_epu32(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_min_s(a: __m256i, b: __m256i) -> __m256i {
    _mm256_min_epi32(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_max_s(a: __m256i, b: __m256i) -> __m256i {
    _mm256_max_epi32(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_or_ps(a: __m256, b: __m256) -> __m256 {
    _mm256_or_ps(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_lt_u_i(a: __m256i, b: __m256i) -> __m256 {
    yi_lt_u(a, b)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn yi_lt_s_i(a: __m256i, b: __m256i) -> __m256 {
    yi_lt_s(a, b)
}

/// The 32-bit integer element backends: 8-lane `__m256i` with the `f32` impl's `__m256` mask
/// convention, so integer and float compares compose. Arithmetic is wrapping.
macro_rules! avx2_int_backend {
    ($t:ty, $lt:ident, $min:ident, $max:ident, $abs:expr, $shr:ident) => {
        impl Backend<$t> for Avx2 {
            type Vector = __m256i;
            type Mask = __m256;

            #[inline(always)]
            fn lanes(self) -> usize {
                8
            }
            #[inline(always)]
            fn splat(self, v: $t) -> __m256i {
                unsafe { yi_splat(v as u32) }
            }
            #[inline(always)]
            fn load(self, s: &[$t]) -> __m256i {
                debug_assert_eq!(s.len(), 8);
                unsafe { yi_load(s.as_ptr() as *const u32) }
            }
            #[inline(always)]
            fn store(self, v: __m256i, s: &mut [$t]) {
                debug_assert_eq!(s.len(), 8);
                unsafe { yi_store(s.as_mut_ptr() as *mut u32, v) }
            }
            #[inline(always)]
            fn add(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_add(a, b) }
            }
            #[inline(always)]
            fn sub(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_sub(a, b) }
            }
            #[inline(always)]
            fn mul(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_mul(a, b) }
            }
            #[inline(always)]
            fn neg(self, a: __m256i) -> __m256i {
                unsafe { yi_neg(a) }
            }
            #[inline(always)]
            #[allow(unused_unsafe)] // identity for the u32 arm, intrinsic for i32
            fn abs(self, a: __m256i) -> __m256i {
                unsafe { ($abs)(a) }
            }
            #[inline(always)]
            fn min(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { $min(a, b) }
            }
            #[inline(always)]
            fn max(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { $max(a, b) }
            }
            #[inline(always)]
            fn le(self, a: __m256i, b: __m256i) -> __m256 {
                unsafe { yi_or_ps($lt(a, b), yi_eq(a, b)) }
            }
            #[inline(always)]
            fn lt(self, a: __m256i, b: __m256i) -> __m256 {
                unsafe { $lt(a, b) }
            }
            #[inline(always)]
            fn ge(self, a: __m256i, b: __m256i) -> __m256 {
                unsafe { yi_or_ps($lt(b, a), yi_eq(a, b)) }
            }
            #[inline(always)]
            fn gt(self, a: __m256i, b: __m256i) -> __m256 {
                unsafe { $lt(b, a) }
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
            fn select(self, m: __m256, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_select(m, a, b) }
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
            fn reduce_sum(self, v: __m256i) -> $t {
                let mut b = [0u32; 8];
                unsafe { yi_store(b.as_mut_ptr(), v) };
                b.iter().fold(0 as $t, |acc, &x| acc.wrapping_add(x as $t))
            }
            #[inline(always)]
            fn reduce_min(self, v: __m256i) -> $t {
                let mut b = [0u32; 8];
                unsafe { yi_store(b.as_mut_ptr(), v) };
                b.iter().map(|&x| x as $t).min().unwrap()
            }
            #[inline(always)]
            fn reduce_max(self, v: __m256i) -> $t {
                let mut b = [0u32; 8];
                unsafe { yi_store(b.as_mut_ptr(), v) };
                b.iter().map(|&x| x as $t).max().unwrap()
            }
            #[inline(always)]
            fn shl(self, a: __m256i, k: u32) -> __m256i {
                debug_assert!(k < 32);
                unsafe { yi_shl(a, k) }
            }
            #[inline(always)]
            fn shr(self, a: __m256i, k: u32) -> __m256i {
                debug_assert!(k < 32);
                unsafe { $shr(a, k) }
            }
            #[inline(always)]
            fn bit_and(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_and(a, b) }
            }
            #[inline(always)]
            fn bit_or(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_or(a, b) }
            }
            #[inline(always)]
            fn bit_xor(self, a: __m256i, b: __m256i) -> __m256i {
                unsafe { yi_xor(a, b) }
            }
            #[inline(always)]
            fn bit_not(self, a: __m256i) -> __m256i {
                unsafe { yi_xor(a, yi_splat(u32::MAX)) }
            }

            type IVector = __m256i;
            #[inline(always)]
            fn iload(self, s: &[u32]) -> __m256i {
                debug_assert_eq!(s.len(), 8);
                unsafe { yi_load(s.as_ptr()) }
            }
            #[inline(always)]
            fn istore(self, v: __m256i, out: &mut [u32]) {
                debug_assert_eq!(out.len(), 8);
                unsafe { yi_store(out.as_mut_ptr(), v) }
            }
            #[inline(always)]
            fn to_bits(self, v: __m256i) -> __m256i {
                v
            }
            #[inline(always)]
            fn from_bits(self, v: __m256i) -> __m256i {
                v
            }
        }
    };
}

avx2_int_backend!(u32, yi_lt_u_i, yi_min_u, yi_max_u, |a| a, yi_shr);
avx2_int_backend!(i32, yi_lt_s_i, yi_min_s, yi_max_s, |a| yi_abs(a), yi_sra);

impl Backend<f64> for Avx2 {
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
/// Clear the sign bit: one `andpd`, cheaper than `max(a, -a)`.
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
    let m = _mm256_min_pd(a, b);
    _mm256_blendv_pd(m, a, _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b))
}
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn f64_max(a: __m256d, b: __m256d) -> __m256d {
    let m = _mm256_max_pd(a, b);
    _mm256_blendv_pd(m, a, _mm256_cmp_pd::<_CMP_UNORD_Q>(b, b))
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

// f16 storage, f32x8 compute: `load` widens via F16C, `store` narrows back. The scalar oracle
// widens identically, so single ops match exactly; multi-op kernels skip the intermediate
// narrowing and come out more accurate.
mod f16_impl {
    use super::*;
    use crate::backend::Backend;
    use half::f16;

    impl Backend<f16> for Avx2 {
        type Vector = __m256; // 8 × f32 (widened)
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
        fn madd(self, a: __m256, b: __m256, acc: __m256) -> __m256 {
            <Self as Backend<f16>>::fma(self, a, b, acc)
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

// bf16: same widen-compute-narrow as f16, but with no F16C equivalent the boundary conversions
// are scalar (cheap: bf16 is the high 16 bits of an f32).
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
        fn madd(self, a: __m256, b: __m256, acc: __m256) -> __m256 {
            <Self as Backend<bf16>>::fma(self, a, b, acc)
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
