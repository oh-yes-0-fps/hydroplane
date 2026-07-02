//! Scalable SVE backend tokens, one per vector byte-width `C` (`Sve<16>` = 128-bit, `Sve<32>` =
//! 256-bit, …). SVE registers are *sizeless* (see `SVE.md`), so the [`Backend::Vector`] is the
//! memory image [`SveVec<C>`] and every op is a [`crate::arch::sve1`] primitive. The byte-width is a
//! const generic because the trait's `Vector` must be `Sized`; dispatch monomorphizes the kernel for
//! the detected VL and runs the matching token. Base SVE doesn't exist on Apple (streaming-only via
//! SME), so these are constructed only on non-Apple hosts — the dispatch enforces that.
#![cfg(target_arch = "aarch64")]
// On Apple the dispatch never constructs an `Sve` token (NEON-only policy), so it reads as dead
// code there; it is live on non-Apple aarch64.
#![allow(dead_code)]

use crate::arch::sve1::{self, SveVec};
use crate::backend::Backend;
use half::{bf16, f16};

/// SVE execution token at a `C`-byte vector width. Zero-sized; constructing one asserts (via
/// [`detect`](Sve::detect) / [`new_unchecked`](Sve::new_unchecked)) that base SVE is present and the
/// hardware VL is at least `C` bytes.
#[derive(Clone, Copy, Debug)]
pub struct Sve<const C: usize>;

impl<const C: usize> Sve<C> {
    /// # Safety
    /// The CPU must implement base (non-streaming) SVE and its vector length must be ≥ `C` bytes.
    #[inline]
    pub unsafe fn new_unchecked() -> Self {
        Sve
    }

    /// `Some(Sve<C>)` if base SVE is present and the hardware VL covers `C` bytes, else `None`.
    #[cfg(feature = "std")]
    #[inline]
    pub fn detect() -> Option<Self> {
        if std::arch::is_aarch64_feature_detected!("sve") && crate::arch::sve2::vl_bytes() >= C {
            Some(Sve)
        } else {
            None
        }
    }
}

// Every op delegates to the matching `sve1` primitive. Those carry `#[target_feature(enable =
// "sve")]`, so the call is `unsafe`; the safety invariant is upheld by `Sve<C>` only existing where
// SVE is present (its constructors). `and_mask`/`or_mask`/`not_mask`/`any_mask`/`all_mask` are
// element-agnostic (byte-granular), so they are shared across all four scalar impls.
macro_rules! impl_sve_backend {
    ($t:ty, $div:expr,
     $splat:ident, $load:ident, $store:ident,
     $add:ident, $sub:ident, $mul:ident, $divop:ident, $neg:ident, $fma:ident, $sqrt:ident,
     $min:ident, $max:ident,
     $le:ident, $lt:ident, $ge:ident, $gt:ident, $sel:ident,
     $rsum:ident, $rmin:ident, $rmax:ident) => {
        impl<const C: usize> Backend<$t> for Sve<C> {
            type Vector = SveVec<C>;
            type Mask = SveVec<C>;

            #[inline(always)]
            fn lanes(self) -> usize {
                C / $div
            }
            #[inline(always)]
            fn splat(self, v: $t) -> SveVec<C> {
                unsafe { sve1::$splat::<C>(v) }
            }
            #[inline(always)]
            fn load(self, s: &[$t]) -> SveVec<C> {
                debug_assert_eq!(s.len(), C / $div);
                unsafe { sve1::$load::<C>(s.as_ptr()) }
            }
            #[inline(always)]
            fn store(self, v: SveVec<C>, s: &mut [$t]) {
                debug_assert_eq!(s.len(), C / $div);
                unsafe { sve1::$store::<C>(&v, s.as_mut_ptr()) }
            }
            #[inline(always)]
            fn add(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$add::<C>(&a, &b) }
            }
            #[inline(always)]
            fn sub(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$sub::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mul(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$mul::<C>(&a, &b) }
            }
            #[inline(always)]
            fn div(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$divop::<C>(&a, &b) }
            }
            #[inline(always)]
            fn neg(self, a: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$neg::<C>(&a) }
            }
            #[inline(always)]
            fn fma(self, a: SveVec<C>, b: SveVec<C>, c: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$fma::<C>(&a, &b, &c) }
            }
                    #[inline(always)]
            fn madd(self, a: SveVec<C>, b: SveVec<C>, acc: SveVec<C>) -> SveVec<C> {
                <Self as Backend<$t>>::fma(self, a, b, acc)
            }
    #[inline(always)]
            fn sqrt(self, a: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$sqrt::<C>(&a) }
            }
            #[inline(always)]
            fn min(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$min::<C>(&a, &b) }
            }
            #[inline(always)]
            fn max(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$max::<C>(&a, &b) }
            }
            #[inline(always)]
            fn le(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$le::<C>(&a, &b) }
            }
            #[inline(always)]
            fn lt(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$lt::<C>(&a, &b) }
            }
            #[inline(always)]
            fn ge(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$ge::<C>(&a, &b) }
            }
            #[inline(always)]
            fn gt(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$gt::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mask_and(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::and_mask::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mask_or(self, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::or_mask::<C>(&a, &b) }
            }
            #[inline(always)]
            fn mask_not(self, a: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::not_mask::<C>(&a) }
            }
            #[inline(always)]
            fn select(self, m: SveVec<C>, a: SveVec<C>, b: SveVec<C>) -> SveVec<C> {
                unsafe { sve1::$sel::<C>(&m, &a, &b) }
            }
            #[inline(always)]
            fn any(self, m: SveVec<C>) -> bool {
                unsafe { sve1::any_mask::<C>(&m) }
            }
            #[inline(always)]
            fn all(self, m: SveVec<C>) -> bool {
                unsafe { sve1::all_mask::<C>(&m) }
            }
            #[inline(always)]
            fn reduce_sum(self, v: SveVec<C>) -> $t {
                unsafe { sve1::$rsum::<C>(&v) }
            }
            #[inline(always)]
            fn reduce_min(self, v: SveVec<C>) -> $t {
                unsafe { sve1::$rmin::<C>(&v) }
            }
            #[inline(always)]
            fn reduce_max(self, v: SveVec<C>) -> $t {
                unsafe { sve1::$rmax::<C>(&v) }
            }

            // Scalable vector length, so the integer companion rides a fixed max-width array;
            // every op takes the trait's portable default.
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
        }
    };
}

impl_sve_backend!(
    f32, 4, splat_f32, load_f32, store_f32, add_f32, sub_f32, mul_f32, div_f32, neg_f32, fma_f32,
    sqrt_f32, min_f32, max_f32, le_f32, lt_f32, ge_f32, gt_f32, select_f32, reduce_sum_f32,
    reduce_min_f32, reduce_max_f32
);
impl_sve_backend!(
    f64, 8, splat_f64, load_f64, store_f64, add_f64, sub_f64, mul_f64, div_f64, neg_f64, fma_f64,
    sqrt_f64, min_f64, max_f64, le_f64, lt_f64, ge_f64, gt_f64, select_f64, reduce_sum_f64,
    reduce_min_f64, reduce_max_f64
);
// f16: native 16-bit lanes (C/2), full `.h` ALU.
impl_sve_backend!(
    f16, 2, splat_f16, load_f16, store_f16, add_f16, sub_f16, mul_f16, div_f16, neg_f16, fma_f16,
    sqrt_f16, min_f16, max_f16, le_f16, lt_f16, ge_f16, gt_f16, select_f16, reduce_sum_f16,
    reduce_min_f16, reduce_max_f16
);
// bf16: 16-bit storage, f32 compute image (C/4 lanes) — arithmetic/compare/select are the `*_f32`
// ops over the image; only the memory boundary (`*_bf16`) and reductions differ.
impl_sve_backend!(
    bf16, 4, splat_bf16, load_bf16, store_bf16, add_f32, sub_f32, mul_f32, div_f32, neg_f32, fma_f32,
    sqrt_f32, min_f32, max_f32, le_f32, lt_f32, ge_f32, gt_f32, select_f32, reduce_sum_bf16,
    reduce_min_bf16, reduce_max_bf16
);
