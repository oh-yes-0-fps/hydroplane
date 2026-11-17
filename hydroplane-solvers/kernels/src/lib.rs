#![cfg_attr(target_arch = "spirv", no_std)]
// rust-gpu compiles index loops but not slice or array iterators, and each storage buffer is its
// own entry-point argument.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! rust-gpu compute kernels behind `hydroplane-solvers`' `vulkan` feature: sparse Cholesky
//! ([`cholesky`]) and preconditioned conjugate gradient ([`pcg`]). The build script compiles this
//! crate three times, once per working precision [`Real`]: `f32`; native `f64` (feature `f64`,
//! for devices with `shaderFloat64`); and [`Ef64`], `f64` emulated in `f32` pairs (feature
//! `ef64`, for devices without). The kernels are written once against [`Scalar`], whose
//! implementations are where the precisions differ: fused multiply-add, inverse square root and
//! atomic accumulation are each done the way that is fast and exact enough for that type.

/// `for c in (t / 32 ..).step_by(8)` below `cols`, for invocation `t`'s row `r = t % 32` if it is
/// below `rows`: the entries of a `rows × cols` (≤ 32 × 32) column-major block over 256
/// invocations, a warp per column so loads coalesce, without dividing by the block's height.
macro_rules! each_entry {
    ($t:expr, $rows:expr, $cols:expr, |$r:ident, $c:ident| $body:block) => {{
        let $r = $t % crate::cholesky::TILE;
        if $r < $rows {
            let mut $c = $t / crate::cholesky::TILE;
            while $c < $cols {
                $body
                $c += 8;
            }
        }
    }};
}

/// `$body` written out 32 times with `$q` the constants 0 ‥ 31: rust-gpu has no unroll attribute,
/// and only constant indices let a local array live in registers.
macro_rules! unroll32 {
    ($q:ident, $body:block) => {
        unroll32!(@ $q, $body, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
    };
    (@ $q:ident, $body:block, $($n:literal),*) => {
        $({
            let $q: usize = $n;
            $body
        })*
    };
}

/// `$body` written out 4 times with `$q` the constants 0 ‥ 3.
macro_rules! unroll4 {
    ($q:ident, $body:block) => {
        unroll32!(@ $q, $body, 0, 1, 2, 3)
    };
}

pub mod amg;
pub mod bicgstab;
pub mod cholesky;
mod ef64;
pub mod pcg;

pub use ef64::Ef64;

#[cfg(feature = "f64")]
pub type Real = f64;
#[cfg(feature = "ef64")]
pub type Real = Ef64;
#[cfg(not(any(feature = "f64", feature = "ef64")))]
pub type Real = f32;

/// A kernel's working precision.
pub trait Scalar: Copy {
    const ZERO: Self;
    const ONE: Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    /// `self · b + c`.
    fn mul_add(self, b: Self, c: Self) -> Self;
    fn abs(self) -> Self;
    /// `1 / √self` for a positive `self`.
    fn rsqrt(self) -> Self;
    /// `v[i] −= d`, atomically across the device.
    fn atomic_sub(v: &mut [Self], i: usize, d: Self);
}

impl Scalar for Ef64 {
    const ZERO: Self = Ef64::ZERO;
    const ONE: Self = Ef64::ONE;
    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        Ef64::from_f32(v)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.hi
    }
    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        Ef64::mul_add(self, b, c)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        Ef64::abs(self)
    }
    #[inline(always)]
    fn rsqrt(self) -> Self {
        Ef64::rsqrt(self)
    }
    /// Two `f32` atomics: one subtracts `hi`, and since it returns the value it replaced, the
    /// rounding error of that subtraction is known exactly and goes, with `lo`, into the low
    /// word. Concurrent updates interleave freely; each one's error is still accounted for, and
    /// the low word only grows by a few of its own units over many updates.
    #[inline(always)]
    fn atomic_sub(v: &mut [Self], i: usize, d: Self) {
        #[cfg(target_arch = "spirv")]
        unsafe {
            use spirv_std::arch::{IndexUnchecked, atomic_f_add};
            const DEVICE: u32 = spirv_std::memory::Scope::Device as u32;
            let slot = v.index_unchecked_mut(i);
            let old = atomic_f_add::<f32, DEVICE, 0>(&mut slot.hi, -d.hi);
            let (_, e) = ef64::two_sum(old, -d.hi);
            atomic_f_add::<f32, DEVICE, 0>(&mut slot.lo, e - d.lo);
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            v[i] -= d;
        }
    }
}

#[cfg(not(feature = "ef64"))]
mod native {
    #[cfg(target_arch = "spirv")]
    use spirv_std::num_traits::Float;

    #[cfg(target_arch = "spirv")]
    #[inline(always)]
    fn fma<T: Float>(a: T, b: T, c: T) -> T {
        a.mul_add(b, c)
    }

    #[cfg(target_arch = "spirv")]
    #[inline(always)]
    fn abs<T: Float>(a: T) -> T {
        a.abs()
    }

    macro_rules! native {
        ($t:ty) => {
            impl super::Scalar for $t {
                const ZERO: Self = 0.0;
                const ONE: Self = 1.0;
                #[inline(always)]
                fn from_f32(v: f32) -> Self {
                    v as $t
                }
                #[inline(always)]
                fn to_f32(self) -> f32 {
                    self as f32
                }
                #[inline(always)]
                fn mul_add(self, b: Self, c: Self) -> Self {
                    #[cfg(target_arch = "spirv")]
                    {
                        fma(self, b, c)
                    }
                    #[cfg(not(target_arch = "spirv"))]
                    {
                        <$t>::mul_add(self, b, c)
                    }
                }
                #[inline(always)]
                fn abs(self) -> Self {
                    #[cfg(target_arch = "spirv")]
                    {
                        abs(self)
                    }
                    #[cfg(not(target_arch = "spirv"))]
                    {
                        <$t>::abs(self)
                    }
                }
                #[inline(always)]
                fn rsqrt(self) -> Self {
                    rsqrt(self)
                }
                #[inline(always)]
                fn atomic_sub(v: &mut [Self], i: usize, d: Self) {
                    #[cfg(target_arch = "spirv")]
                    unsafe {
                        spirv_std::arch::atomic_f_add::<$t, { spirv_std::memory::Scope::Device as u32 }, 0>(
                            spirv_std::arch::IndexUnchecked::index_unchecked_mut(v, i),
                            -d,
                        );
                    }
                    #[cfg(not(target_arch = "spirv"))]
                    {
                        v[i] -= d;
                    }
                }
            }
        };
    }

    #[cfg(not(any(feature = "f64", feature = "ef64")))]
    native!(f32);
    #[cfg(feature = "f64")]
    native!(f64);

    #[cfg(not(any(feature = "f64", feature = "ef64")))]
    #[inline(always)]
    fn rsqrt(v: f32) -> f32 {
        1.0 / v.sqrt()
    }

    /// From the `f32` hardware estimate and two Newton steps (each squares the relative error:
    /// 1e-7 → 1e-14 → below `f64` precision), where `v` is within `f32`'s range; `f64` square
    /// root and division are long software sequences on most GPUs.
    #[cfg(feature = "f64")]
    #[inline(always)]
    fn rsqrt(v: f64) -> f64 {
        if v > 1e-30 && v < 1e30 {
            let mut y = (1.0 / (v as f32).sqrt()) as f64;
            y = y * (1.5 - 0.5 * v * y * y);
            y * (1.5 - 0.5 * v * y * y)
        } else {
            1.0 / v.sqrt()
        }
    }
}

#[inline(always)]
pub(crate) fn ld<T: Copy>(s: &[T], i: usize) -> T {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { *spirv_std::arch::IndexUnchecked::index_unchecked(s, i) }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        s[i]
    }
}

#[inline(always)]
pub(crate) fn st<T: Copy>(s: &mut [T], i: usize, v: T) {
    #[cfg(target_arch = "spirv")]
    {
        unsafe { *spirv_std::arch::IndexUnchecked::index_unchecked_mut(s, i) = v }
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        s[i] = v
    }
}

/// A workgroup array whose indexing is unchecked in SPIR-V: rust-gpu lowers a checked index to a
/// compare and a branch out of the enclosing loop, which on every shared-memory access splits hot
/// loops into a dozen blocks.
#[repr(transparent)]
pub struct Shared<const N: usize, T: Copy = Real>(pub [T; N]);

impl<const N: usize, T: Copy> core::ops::Index<usize> for Shared<N, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        #[cfg(target_arch = "spirv")]
        {
            unsafe { spirv_std::arch::IndexUnchecked::index_unchecked(&self.0, i) }
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            &self.0[i]
        }
    }
}

impl<const N: usize, T: Copy> core::ops::IndexMut<usize> for Shared<N, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut T {
        #[cfg(target_arch = "spirv")]
        {
            unsafe { spirv_std::arch::IndexUnchecked::index_unchecked_mut(&mut self.0, i) }
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            &mut self.0[i]
        }
    }
}

#[inline(always)]
pub(crate) fn barrier() {
    #[cfg(target_arch = "spirv")]
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
}
