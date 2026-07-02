//! Scalable RVV backend tokens, one per vector byte-width `C` (`Rvv<16>` = 128-bit, `Rvv<32>` =
//! 256-bit, …). RVV registers are scalable, so the [`Backend::Vector`] is the memory image
//! [`RvvVec<C>`] and every op is a [`crate::arch::rvv`] primitive — the same design as the SVE
//! backend ([`crate::backend::sve`]). The byte-width is a const generic because the trait's `Vector`
//! must be `Sized`; dispatch monomorphizes the kernel for the detected `VLENB` and runs the matching
//! token. Base "V" has no FP16/bf16 vector ALU (those are the Zvfh/Zvfbfmin extensions), so only
//! `f32`/`f64` are implemented; `f16`/`bf16` take the scalar path.
#![cfg(target_arch = "riscv64")]
// Constructed only where the "V" extension is present (the dispatch enforces that); on a host
// without it the token is never built, so its constructors read as dead code there.
#![allow(dead_code)]

use crate::arch::rvv::{self, RvvVec};
use crate::backend::Backend;

/// RVV execution token at a `C`-byte vector width. Zero-sized; constructing one asserts (via
/// [`detect`](Rvv::detect) / [`new_unchecked`](Rvv::new_unchecked)) that the "V" extension is present
/// and `VLENB` is at least `C` bytes.
#[derive(Clone, Copy, Debug)]
pub struct Rvv<const C: usize>;

impl<const C: usize> Rvv<C> {
    /// # Safety
    /// The CPU must implement the RVV "V" extension and its `VLENB` must be ≥ `C` bytes.
    #[inline]
    pub unsafe fn new_unchecked() -> Self {
        Rvv
    }

    /// `Some(Rvv<C>)` if the "V" extension is present and `VLENB` covers `C` bytes, else `None`.
    #[cfg(feature = "std")]
    #[inline]
    pub fn detect() -> Option<Self> {
        if rvv::is_supported() && rvv::vlenb() >= C {
            Some(Rvv)
        } else {
            None
        }
    }
}

// Every op delegates to the matching `rvv` primitive. `and_mask`/`or_mask`/`not_mask`/`any_mask`/
// `all_mask` are element-agnostic (byte-granular), so they are shared across both scalar impls.
macro_rules! impl_rvv_backend {
    ($t:ty, $div:expr,
     $splat:ident, $load:ident, $store:ident,
     $add:ident, $sub:ident, $mul:ident, $divop:ident, $neg:ident, $fma:ident, $sqrt:ident,
     $min:ident, $max:ident,
     $le:ident, $lt:ident, $ge:ident, $gt:ident, $sel:ident,
     $rsum:ident, $rmin:ident, $rmax:ident) => {
        impl<const C: usize> Backend<$t> for Rvv<C> {
            type Vector = RvvVec<C>;
            type Mask = RvvVec<C>;

            // Scalable/emulated targets ride the fixed max-width array; ops take the trait's
            // portable defaults.
            type IVector = [u32; crate::MAX_LANES];
            #[inline(always)]
            fn iload(self, s: &[u32]) -> [u32; crate::MAX_LANES] {
                let mut v = [0u32; crate::MAX_LANES];
                v[..s.len()].copy_from_slice(s);
                v
            }
            #[inline(always)]
            fn istore(self, v: [u32; crate::MAX_LANES], out: &mut [u32]) {
                let n = out.len();
                out.copy_from_slice(&v[..n]);
            }


            #[inline(always)]
            fn lanes(self) -> usize {
                C / $div
            }
            #[inline(always)]
            fn splat(self, v: $t) -> RvvVec<C> {
                unsafe { rvv::$splat::<C>(v) }
            }
            #[inline(always)]
            fn load(self, s: &[$t]) -> RvvVec<C> {
                debug_assert_eq!(s.len(), C / $div);
                unsafe { rvv::$load::<C>(s.as_ptr()) }
            }
            #[inline(always)]
            fn store(self, v: RvvVec<C>, s: &mut [$t]) {
                debug_assert_eq!(s.len(), C / $div);
                unsafe { rvv::$store::<C>(&v, s.as_mut_ptr()) }
            }
            #[inline(always)]
            fn add(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$add::<C>(&a, &b) }
            }
            #[inline(always)]
            fn sub(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$sub::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mul(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$mul::<C>(&a, &b) }
            }
            #[inline(always)]
            fn div(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$divop::<C>(&a, &b) }
            }
            #[inline(always)]
            fn neg(self, a: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$neg::<C>(&a) }
            }
            #[inline(always)]
            fn fma(self, a: RvvVec<C>, b: RvvVec<C>, c: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$fma::<C>(&a, &b, &c) }
            }
            #[inline(always)]
            fn sqrt(self, a: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$sqrt::<C>(&a) }
            }
            #[inline(always)]
            fn min(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$min::<C>(&a, &b) }
            }
            #[inline(always)]
            fn max(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$max::<C>(&a, &b) }
            }
            #[inline(always)]
            fn le(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$le::<C>(&a, &b) }
            }
            #[inline(always)]
            fn lt(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$lt::<C>(&a, &b) }
            }
            #[inline(always)]
            fn ge(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$ge::<C>(&a, &b) }
            }
            #[inline(always)]
            fn gt(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$gt::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mask_and(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::and_mask::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mask_or(self, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::or_mask::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mask_not(self, a: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::not_mask::<C>(&a) }
            }
            #[inline(always)]
            fn select(self, m: RvvVec<C>, a: RvvVec<C>, b: RvvVec<C>) -> RvvVec<C> {
                unsafe { rvv::$sel::<C>(&m, &a, &b) }
            }
            #[inline(always)]
            fn any(self, m: RvvVec<C>) -> bool {
                unsafe { rvv::any_mask::<C>(&m) }
            }
            #[inline(always)]
            fn all(self, m: RvvVec<C>) -> bool {
                unsafe { rvv::all_mask::<C>(&m) }
            }
            #[inline(always)]
            fn reduce_sum(self, v: RvvVec<C>) -> $t {
                unsafe { rvv::$rsum::<C>(&v) }
            }
            #[inline(always)]
            fn reduce_min(self, v: RvvVec<C>) -> $t {
                unsafe { rvv::$rmin::<C>(&v) }
            }
            #[inline(always)]
            fn reduce_max(self, v: RvvVec<C>) -> $t {
                unsafe { rvv::$rmax::<C>(&v) }
            }
        }
    };
}

impl_rvv_backend!(
    f32, 4, splat_f32, load_f32, store_f32, add_f32, sub_f32, mul_f32, div_f32, neg_f32, fma_f32,
    sqrt_f32, min_f32, max_f32, le_f32, lt_f32, ge_f32, gt_f32, select_f32, reduce_sum_f32,
    reduce_min_f32, reduce_max_f32
);
impl_rvv_backend!(
    f64, 8, splat_f64, load_f64, store_f64, add_f64, sub_f64, mul_f64, div_f64, neg_f64, fma_f64,
    sqrt_f64, min_f64, max_f64, le_f64, lt_f64, ge_f64, gt_f64, select_f64, reduce_sum_f64,
    reduce_min_f64, reduce_max_f64
);
