#![cfg_attr(target_arch = "spirv", no_std)]
// rust-gpu compiles index loops but not slice or array iterators, and each storage buffer is its
// own entry-point argument.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
//! rust-gpu compute kernels behind `hydroplane-solvers`' `vulkan` feature. The build script
//! compiles this crate twice: in `f32`, and with the `f64` feature in `f64` (for devices with
//! `shaderFloat64`).

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

pub mod cholesky;

#[cfg(feature = "f64")]
pub type Real = f64;
#[cfg(not(feature = "f64"))]
pub type Real = f32;

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
pub struct Shared<const N: usize>(pub [Real; N]);

impl<const N: usize> core::ops::Index<usize> for Shared<N> {
    type Output = Real;
    #[inline(always)]
    fn index(&self, i: usize) -> &Real {
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

impl<const N: usize> core::ops::IndexMut<usize> for Shared<N> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Real {
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
