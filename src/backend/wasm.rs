//! Hand-written WebAssembly SIMD backends (4-wide f32, 2-wide f64).
//!
//! WASM SIMD is fixed-width 128-bit (`v128`) — the [`Simd128`] and [`RelaxedSimd`] tokens are the
//! WASM analogue of NEON, never a scalable vector. Both back `f32`/`f64`/`bf16` (bf16 via the
//! widen path: 16-bit storage, `f32x4` compute). The single 128-bit register type means
//! `Vector = Mask = v128` for every scalar; masks are all-ones/all-zero lanes, `select` is
//! `v128_bitselect`, and cross-lane reductions are unrolled lane extracts (WASM has no horizontal
//! reduce intrinsics).
//!
//! The two tokens differ only in [`fma`](Backend::fma): base `simd128` has no fused multiply-add,
//! so it lowers to a separate `mul` then `add`; [`RelaxedSimd`] uses the relaxed-SIMD
//! `f32x4_relaxed_madd`/`f64x2_relaxed_madd` (fused where the engine supports it, with the
//! non-deterministic rounding the relaxed proposal permits). WASM has no runtime feature
//! detection, so which token a build gets is decided at compile time from `target_feature`.
#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::wasm32::*;

use half::bf16;

use crate::backend::Backend;

#[inline(always)]
fn fma_mul_add_f32(a: v128, b: v128, c: v128) -> v128 {
    f32x4_add(f32x4_mul(a, b), c)
}
#[inline(always)]
fn fma_mul_add_f64(a: v128, b: v128, c: v128) -> v128 {
    f64x2_add(f64x2_mul(a, b), c)
}

#[inline(always)]
unsafe fn load4_bf16(s: &[bf16]) -> v128 {
    let t = [s[0].to_f32(), s[1].to_f32(), s[2].to_f32(), s[3].to_f32()];
    v128_load(t.as_ptr() as *const v128)
}
#[inline(always)]
unsafe fn store4_bf16(v: v128, s: &mut [bf16]) {
    let mut t = [0f32; 4];
    v128_store(t.as_mut_ptr() as *mut v128, v);
    for (d, x) in s.iter_mut().zip(t) {
        *d = bf16::from_f32(x);
    }
}

/// Emits `Backend<f32>`, `Backend<f64>` and the bf16 widen-path `Backend<bf16>` for a WASM token.
/// `$fma32`/`$fma64` are the only point of variation between [`Simd128`] and [`RelaxedSimd`]:
/// `fn(v128, v128, v128) -> v128` computing `a*b + c`.
macro_rules! wasm_backend {
    ($token:ty, $fma32:path, $fma64:path) => {
        impl Backend<f32> for $token {
            type Vector = v128;
            type Mask = v128;

            #[inline(always)]
            fn lanes(self) -> usize {
                4
            }
            #[inline(always)]
            fn splat(self, v: f32) -> v128 {
                f32x4_splat(v)
            }
            #[inline(always)]
            fn load(self, s: &[f32]) -> v128 {
                debug_assert_eq!(s.len(), 4);
                unsafe { v128_load(s.as_ptr() as *const v128) }
            }
            #[inline(always)]
            fn store(self, v: v128, s: &mut [f32]) {
                debug_assert_eq!(s.len(), 4);
                unsafe { v128_store(s.as_mut_ptr() as *mut v128, v) }
            }
            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {
                f32x4_add(a, b)
            }
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {
                f32x4_sub(a, b)
            }
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {
                f32x4_mul(a, b)
            }
            #[inline(always)]
            fn div(self, a: v128, b: v128) -> v128 {
                f32x4_div(a, b)
            }
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {
                f32x4_neg(a)
            }
            #[inline(always)]
            fn fma(self, a: v128, b: v128, c: v128) -> v128 {
                $fma32(a, b, c)
            }
            #[inline(always)]
            fn sqrt(self, a: v128) -> v128 {
                f32x4_sqrt(a)
            }
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {
                f32x4_min(a, b)
            }
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {
                f32x4_max(a, b)
            }
            #[inline(always)]
            fn le(self, a: v128, b: v128) -> v128 {
                f32x4_le(a, b)
            }
            #[inline(always)]
            fn lt(self, a: v128, b: v128) -> v128 {
                f32x4_lt(a, b)
            }
            #[inline(always)]
            fn ge(self, a: v128, b: v128) -> v128 {
                f32x4_ge(a, b)
            }
            #[inline(always)]
            fn gt(self, a: v128, b: v128) -> v128 {
                f32x4_gt(a, b)
            }
            #[inline(always)]
            fn mask_and(self, a: v128, b: v128) -> v128 {
                v128_and(a, b)
            }
            #[inline(always)]
            fn mask_or(self, a: v128, b: v128) -> v128 {
                v128_or(a, b)
            }
            #[inline(always)]
            fn mask_not(self, a: v128) -> v128 {
                v128_not(a)
            }
            #[inline(always)]
            fn select(self, m: v128, a: v128, b: v128) -> v128 {
                v128_bitselect(a, b, m)
            }
            #[inline(always)]
            fn any(self, m: v128) -> bool {
                v128_any_true(m)
            }
            #[inline(always)]
            fn all(self, m: v128) -> bool {
                i32x4_all_true(m)
            }
            #[inline(always)]
            fn reduce_sum(self, v: v128) -> f32 {
                f32x4_extract_lane::<0>(v)
                    + f32x4_extract_lane::<1>(v)
                    + f32x4_extract_lane::<2>(v)
                    + f32x4_extract_lane::<3>(v)
            }
            #[inline(always)]
            fn reduce_min(self, v: v128) -> f32 {
                f32x4_extract_lane::<0>(v)
                    .min(f32x4_extract_lane::<1>(v))
                    .min(f32x4_extract_lane::<2>(v))
                    .min(f32x4_extract_lane::<3>(v))
            }
            #[inline(always)]
            fn reduce_max(self, v: v128) -> f32 {
                f32x4_extract_lane::<0>(v)
                    .max(f32x4_extract_lane::<1>(v))
                    .max(f32x4_extract_lane::<2>(v))
                    .max(f32x4_extract_lane::<3>(v))
            }
        }

        impl Backend<f64> for $token {
            type Vector = v128;
            type Mask = v128;

            #[inline(always)]
            fn lanes(self) -> usize {
                2
            }
            #[inline(always)]
            fn splat(self, v: f64) -> v128 {
                f64x2_splat(v)
            }
            #[inline(always)]
            fn load(self, s: &[f64]) -> v128 {
                debug_assert_eq!(s.len(), 2);
                unsafe { v128_load(s.as_ptr() as *const v128) }
            }
            #[inline(always)]
            fn store(self, v: v128, s: &mut [f64]) {
                debug_assert_eq!(s.len(), 2);
                unsafe { v128_store(s.as_mut_ptr() as *mut v128, v) }
            }
            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {
                f64x2_add(a, b)
            }
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {
                f64x2_sub(a, b)
            }
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {
                f64x2_mul(a, b)
            }
            #[inline(always)]
            fn div(self, a: v128, b: v128) -> v128 {
                f64x2_div(a, b)
            }
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {
                f64x2_neg(a)
            }
            #[inline(always)]
            fn fma(self, a: v128, b: v128, c: v128) -> v128 {
                $fma64(a, b, c)
            }
            #[inline(always)]
            fn sqrt(self, a: v128) -> v128 {
                f64x2_sqrt(a)
            }
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {
                f64x2_min(a, b)
            }
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {
                f64x2_max(a, b)
            }
            #[inline(always)]
            fn le(self, a: v128, b: v128) -> v128 {
                f64x2_le(a, b)
            }
            #[inline(always)]
            fn lt(self, a: v128, b: v128) -> v128 {
                f64x2_lt(a, b)
            }
            #[inline(always)]
            fn ge(self, a: v128, b: v128) -> v128 {
                f64x2_ge(a, b)
            }
            #[inline(always)]
            fn gt(self, a: v128, b: v128) -> v128 {
                f64x2_gt(a, b)
            }
            #[inline(always)]
            fn mask_and(self, a: v128, b: v128) -> v128 {
                v128_and(a, b)
            }
            #[inline(always)]
            fn mask_or(self, a: v128, b: v128) -> v128 {
                v128_or(a, b)
            }
            #[inline(always)]
            fn mask_not(self, a: v128) -> v128 {
                v128_not(a)
            }
            #[inline(always)]
            fn select(self, m: v128, a: v128, b: v128) -> v128 {
                v128_bitselect(a, b, m)
            }
            #[inline(always)]
            fn any(self, m: v128) -> bool {
                v128_any_true(m)
            }
            #[inline(always)]
            fn all(self, m: v128) -> bool {
                i64x2_all_true(m)
            }
            #[inline(always)]
            fn reduce_sum(self, v: v128) -> f64 {
                f64x2_extract_lane::<0>(v) + f64x2_extract_lane::<1>(v)
            }
            #[inline(always)]
            fn reduce_min(self, v: v128) -> f64 {
                f64x2_extract_lane::<0>(v).min(f64x2_extract_lane::<1>(v))
            }
            #[inline(always)]
            fn reduce_max(self, v: v128) -> f64 {
                f64x2_extract_lane::<0>(v).max(f64x2_extract_lane::<1>(v))
            }
        }

        // bf16: 16-bit storage, `f32x4` compute (WASM has no bf16 ALU). Widen on load/splat, narrow
        // on store/reduce — conversions at the memory boundary only, all arithmetic native f32 SIMD.
        impl Backend<bf16> for $token {
            type Vector = v128;
            type Mask = v128;

            #[inline(always)]
            fn lanes(self) -> usize {
                4
            }
            #[inline(always)]
            fn splat(self, v: bf16) -> v128 {
                f32x4_splat(v.to_f32())
            }
            #[inline(always)]
            fn load(self, s: &[bf16]) -> v128 {
                debug_assert_eq!(s.len(), 4);
                unsafe { load4_bf16(s) }
            }
            #[inline(always)]
            fn store(self, v: v128, s: &mut [bf16]) {
                debug_assert_eq!(s.len(), 4);
                unsafe { store4_bf16(v, s) }
            }
            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {
                f32x4_add(a, b)
            }
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {
                f32x4_sub(a, b)
            }
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {
                f32x4_mul(a, b)
            }
            #[inline(always)]
            fn div(self, a: v128, b: v128) -> v128 {
                f32x4_div(a, b)
            }
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {
                f32x4_neg(a)
            }
            #[inline(always)]
            fn fma(self, a: v128, b: v128, c: v128) -> v128 {
                $fma32(a, b, c)
            }
            #[inline(always)]
            fn sqrt(self, a: v128) -> v128 {
                f32x4_sqrt(a)
            }
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {
                f32x4_min(a, b)
            }
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {
                f32x4_max(a, b)
            }
            #[inline(always)]
            fn le(self, a: v128, b: v128) -> v128 {
                f32x4_le(a, b)
            }
            #[inline(always)]
            fn lt(self, a: v128, b: v128) -> v128 {
                f32x4_lt(a, b)
            }
            #[inline(always)]
            fn ge(self, a: v128, b: v128) -> v128 {
                f32x4_ge(a, b)
            }
            #[inline(always)]
            fn gt(self, a: v128, b: v128) -> v128 {
                f32x4_gt(a, b)
            }
            #[inline(always)]
            fn mask_and(self, a: v128, b: v128) -> v128 {
                v128_and(a, b)
            }
            #[inline(always)]
            fn mask_or(self, a: v128, b: v128) -> v128 {
                v128_or(a, b)
            }
            #[inline(always)]
            fn mask_not(self, a: v128) -> v128 {
                v128_not(a)
            }
            #[inline(always)]
            fn select(self, m: v128, a: v128, b: v128) -> v128 {
                v128_bitselect(a, b, m)
            }
            #[inline(always)]
            fn any(self, m: v128) -> bool {
                v128_any_true(m)
            }
            #[inline(always)]
            fn all(self, m: v128) -> bool {
                i32x4_all_true(m)
            }
            #[inline(always)]
            fn reduce_sum(self, v: v128) -> bf16 {
                bf16::from_f32(
                    f32x4_extract_lane::<0>(v)
                        + f32x4_extract_lane::<1>(v)
                        + f32x4_extract_lane::<2>(v)
                        + f32x4_extract_lane::<3>(v),
                )
            }
            #[inline(always)]
            fn reduce_min(self, v: v128) -> bf16 {
                bf16::from_f32(
                    f32x4_extract_lane::<0>(v)
                        .min(f32x4_extract_lane::<1>(v))
                        .min(f32x4_extract_lane::<2>(v))
                        .min(f32x4_extract_lane::<3>(v)),
                )
            }
            #[inline(always)]
            fn reduce_max(self, v: v128) -> bf16 {
                bf16::from_f32(
                    f32x4_extract_lane::<0>(v)
                        .max(f32x4_extract_lane::<1>(v))
                        .max(f32x4_extract_lane::<2>(v))
                        .max(f32x4_extract_lane::<3>(v)),
                )
            }
        }
    };
}

/// Base WASM `simd128` execution token (fixed 128-bit). No fused multiply-add, so [`fma`] lowers
/// to `mul` then `add`. Available whenever the build enables `simd128`.
///
/// [`fma`]: Backend::fma
// A build that also enables `relaxed-simd` dispatches to `RelaxedSimd` instead, leaving this token
// reached only by the in-crate differential tests — hence the allow.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Simd128;

impl Simd128 {
    #[allow(dead_code)]
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

wasm_backend!(Simd128, fma_mul_add_f32, fma_mul_add_f64);

/// WASM relaxed-SIMD execution token. Identical to [`Simd128`] except [`fma`] uses the relaxed
/// `*_relaxed_madd` instructions — fused where the engine supports it, with the non-deterministic
/// rounding the relaxed-SIMD proposal permits. Available only when the build enables
/// `relaxed-simd`.
///
/// [`fma`]: Backend::fma
#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Copy, Debug, Default)]
pub struct RelaxedSimd;

#[cfg(target_feature = "relaxed-simd")]
impl RelaxedSimd {
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(target_feature = "relaxed-simd")]
wasm_backend!(RelaxedSimd, f32x4_relaxed_madd, f64x2_relaxed_madd);
