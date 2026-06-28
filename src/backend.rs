//! The execution-context (ISA token) abstraction.
//!
//! A [`Backend<T>`] is a zero-sized token identifying an instruction set (scalar, AVX2,
//! NEON, …, or a GPU subgroup) *for a specific scalar `T`*. The trait is keyed per scalar
//! — rather than carrying a `Vector<T>` GAT — so a hand-written backend can pick the exact
//! intrinsic for the type (`_mm256_add_ps` vs `_mm256_add_pd`): each `(ISA, scalar)` pair
//! is its own concrete impl with concrete [`Backend::Vector`]/[`Backend::Mask`] types. A
//! kernel written against `S: Backend<T>` therefore runs for any `T` on any ISA — the
//! float-agnosticism and the portability come from the same place. The lane count is a
//! `fn` (not a `const`) because the GPU subgroup backend only learns it at runtime.

use crate::scalar::Scalar;

/// An instruction-set execution context for scalar `T`. Implemented by [`ScalarBackend`]
/// (every `T`) and, per `(ISA, scalar)`, by the hand-rolled `core::arch` backends.
pub trait Backend<T: Scalar>: Copy {
    /// The varying register holding [`Backend::lanes`] elements of `T`.
    type Vector: Copy;
    /// The boolean mask companion to [`Backend::Vector`].
    type Mask: Copy;

    /// Number of `T` lanes in one register under this backend.
    fn lanes(self) -> usize;

    fn splat(self, v: T) -> Self::Vector;
    /// Load exactly one register. `s.len()` must equal [`Backend::lanes`].
    fn load(self, s: &[T]) -> Self::Vector;
    /// Store exactly one register. `s.len()` must equal [`Backend::lanes`].
    fn store(self, v: Self::Vector, s: &mut [T]);

    fn add(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn sub(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn mul(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn div(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn neg(self, a: Self::Vector) -> Self::Vector;
    fn fma(self, a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
    fn sqrt(self, a: Self::Vector) -> Self::Vector;
    fn min(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn max(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;

    fn le(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;
    fn lt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;
    fn ge(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;
    fn gt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;

    fn mask_and(self, a: Self::Mask, b: Self::Mask) -> Self::Mask;
    fn mask_or(self, a: Self::Mask, b: Self::Mask) -> Self::Mask;
    fn mask_not(self, a: Self::Mask) -> Self::Mask;

    fn select(self, m: Self::Mask, a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Cross-lane: true if any active lane of the mask is set.
    fn any(self, m: Self::Mask) -> bool;
    /// Cross-lane: true if every lane of the mask is set.
    fn all(self, m: Self::Mask) -> bool;

    fn reduce_sum(self, v: Self::Vector) -> T;
    fn reduce_min(self, v: Self::Vector) -> T;
    fn reduce_max(self, v: Self::Vector) -> T;
}

/// The always-available 1-lane backend.
///
/// `Vector = T`, `Mask = bool`, for every `T: Scalar`. It is both the correctness oracle
/// for the SIMD backends (math routes through [`Scalar::Compute`] identically) and the
/// natural rust-gpu/SPIR-V lowering target (no data-movement intrinsics, everything
/// scalar).
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarBackend;

impl<T: Scalar> Backend<T> for ScalarBackend {
    type Vector = T;
    type Mask = bool;

    #[inline(always)]
    fn lanes(self) -> usize {
        1
    }
    #[inline(always)]
    fn splat(self, v: T) -> T {
        v
    }
    #[inline(always)]
    fn load(self, s: &[T]) -> T {
        s[0]
    }
    #[inline(always)]
    fn store(self, v: T, s: &mut [T]) {
        s[0] = v;
    }
    #[inline(always)]
    fn add(self, a: T, b: T) -> T {
        a.add(b)
    }
    #[inline(always)]
    fn sub(self, a: T, b: T) -> T {
        a.sub(b)
    }
    #[inline(always)]
    fn mul(self, a: T, b: T) -> T {
        a.mul(b)
    }
    #[inline(always)]
    fn div(self, a: T, b: T) -> T {
        a.div(b)
    }
    #[inline(always)]
    fn neg(self, a: T) -> T {
        a.neg()
    }
    #[inline(always)]
    fn fma(self, a: T, b: T, c: T) -> T {
        a.fma(b, c)
    }
    #[inline(always)]
    fn sqrt(self, a: T) -> T {
        a.sqrt()
    }
    #[inline(always)]
    fn min(self, a: T, b: T) -> T {
        a.min(b)
    }
    #[inline(always)]
    fn max(self, a: T, b: T) -> T {
        a.max(b)
    }
    #[inline(always)]
    fn le(self, a: T, b: T) -> bool {
        a.le(b)
    }
    #[inline(always)]
    fn lt(self, a: T, b: T) -> bool {
        a.lt(b)
    }
    #[inline(always)]
    fn ge(self, a: T, b: T) -> bool {
        a.ge(b)
    }
    #[inline(always)]
    fn gt(self, a: T, b: T) -> bool {
        a.gt(b)
    }
    #[inline(always)]
    fn mask_and(self, a: bool, b: bool) -> bool {
        a & b
    }
    #[inline(always)]
    fn mask_or(self, a: bool, b: bool) -> bool {
        a | b
    }
    #[inline(always)]
    fn mask_not(self, a: bool) -> bool {
        !a
    }
    #[inline(always)]
    fn select(self, m: bool, a: T, b: T) -> T {
        if m { a } else { b }
    }
    #[inline(always)]
    fn any(self, m: bool) -> bool {
        m
    }
    #[inline(always)]
    fn all(self, m: bool) -> bool {
        m
    }
    #[inline(always)]
    fn reduce_sum(self, v: T) -> T {
        v
    }
    #[inline(always)]
    fn reduce_min(self, v: T) -> T {
        v
    }
    #[inline(always)]
    fn reduce_max(self, v: T) -> T {
        v
    }
}

// The hand-rolled SIMD tokens are crate-internal: application code never names a backend,
// it goes through `dispatch`, which picks one by runtime CPU detection. They stay reachable
// for the in-crate differential tests (`diff_tests`) that verify each against the oracle.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx2;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx512;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx512bf16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx512fp16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod sse4;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
pub(crate) mod sve;
#[cfg(target_arch = "arm")]
pub(crate) mod neon_a32;
#[cfg(target_arch = "riscv64")]
pub(crate) mod rvv;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;

/// The GPU subgroup backend (SPIR-V) and its portable sequential-vs-subgroup scheduling
/// policy. Public: the `choose` policy compiles and is tested on the CPU; the `Subgroup`
/// backend itself compiles only under `target_arch = "spirv"`, reading the warp width from
/// the hardware `SubgroupSize` builtin.
pub mod subgroup;

#[cfg(test)]
mod diff_tests;
