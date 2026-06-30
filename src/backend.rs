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

    /// Independent accumulator chains the multi-accumulator reductions (`Gang::reduce`,
    /// `Gang::zip_reduce`, `Gang::count_n`) run to saturate this backend's FP pipes — the ILP
    /// unroll factor, baked in at the dispatch that picks the backend rather than measured per call.
    /// A compile-time constant, so the reduction loops unroll to exactly this many chains with no
    /// runtime `K` lookup. Must not exceed [`MAX_UNROLL`](crate::MAX_UNROLL). The default suits x86's
    /// 2–3 vector pipes; wide cores (Apple NEON, SVE) raise it, the scalar floor drops it to 1.
    const UNROLL: usize = 4;

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
    /// Absolute value. The default is `max(a, -a)` (two ops); a backend overrides it with a single
    /// dedicated instruction (NEON `fabs`, wasm `f*.abs`, AVX-512 `vabs`) or a sign-bit clear
    /// (x86 `andps` with the `0x7FFF…` mask).
    #[inline]
    fn abs(self, a: Self::Vector) -> Self::Vector {
        self.max(a, self.neg(a))
    }
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

    /// Pack the mask into the low [`lanes`](Backend::lanes) bits of a `u32`: bit `i` set iff lane `i`
    /// is set; bits at and above `lanes()` are zero. Lets a caller `popcount` the set lanes or walk
    /// them by `trailing_zeros` instead of a per-lane scalar scan. The default materializes the mask
    /// through `select`+`store` and packs scalar (no faster than the scan it replaces); the
    /// fixed-width backends override it with a native movemask (x86 `movemask_ps`, AVX-512 k-regs,
    /// NEON shift-and-add, wasm `bitmask`). `lanes()` never exceeds [`MAX_LANES`](crate::MAX_LANES)
    /// (32), so a `u32` always has room.
    #[inline]
    fn mask_bitmask(self, m: Self::Mask) -> u32 {
        let n = self.lanes();
        let ones = self.select(m, self.splat(T::ONE), self.splat(T::ZERO));
        let mut buf = [T::ZERO; crate::MAX_LANES];
        self.store(ones, &mut buf[..n]);
        let mut bits = 0u32;
        let mut i = 0;
        while i < n {
            if buf[i] != T::ZERO {
                bits |= 1 << i;
            }
            i += 1;
        }
        bits
    }

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

    /// One lane, one chain: a scalar add chain has nothing to interleave.
    const UNROLL: usize = 1;

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
    fn mask_bitmask(self, m: bool) -> u32 {
        m as u32
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

/// A backend `B` re-stamped with a compile-time unroll factor `K`. Every op delegates to `B`
/// (so codegen is identical after inlining); the only thing it changes is [`UNROLL`](Backend::UNROLL),
/// which becomes the const generic `K`. The dispatch adapter resolves `K` once by runtime detection
/// and wraps the chosen ISA backend in this, so each reduction sees `K` as a constant — no per-call
/// `K` lookup — without `K` having to thread through [`Gang`](crate::Gang) or [`Kernel`](crate::Kernel).
#[cfg(not(any(no_ilp, target_arch = "spirv")))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Unroll<B, const K: usize>(pub(crate) B);

#[cfg(not(any(no_ilp, target_arch = "spirv")))]
impl<T: Scalar, B: Backend<T>, const K: usize> Backend<T> for Unroll<B, K> {
    type Vector = B::Vector;
    type Mask = B::Mask;

    const UNROLL: usize = K;

    #[inline(always)]
    fn lanes(self) -> usize {
        self.0.lanes()
    }
    #[inline(always)]
    fn splat(self, v: T) -> Self::Vector {
        self.0.splat(v)
    }
    #[inline(always)]
    fn load(self, s: &[T]) -> Self::Vector {
        self.0.load(s)
    }
    #[inline(always)]
    fn store(self, v: Self::Vector, s: &mut [T]) {
        self.0.store(v, s)
    }
    #[inline(always)]
    fn add(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.mul(a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.div(a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Vector) -> Self::Vector {
        self.0.neg(a)
    }
    #[inline(always)]
    fn fma(self, a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        self.0.fma(a, b, c)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Vector) -> Self::Vector {
        self.0.sqrt(a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Vector) -> Self::Vector {
        self.0.abs(a)
    }
    #[inline(always)]
    fn min(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.min(a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.max(a, b)
    }
    #[inline(always)]
    fn le(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.le(a, b)
    }
    #[inline(always)]
    fn lt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.lt(a, b)
    }
    #[inline(always)]
    fn ge(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.ge(a, b)
    }
    #[inline(always)]
    fn gt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.gt(a, b)
    }
    #[inline(always)]
    fn mask_and(self, a: Self::Mask, b: Self::Mask) -> Self::Mask {
        self.0.mask_and(a, b)
    }
    #[inline(always)]
    fn mask_or(self, a: Self::Mask, b: Self::Mask) -> Self::Mask {
        self.0.mask_or(a, b)
    }
    #[inline(always)]
    fn mask_not(self, a: Self::Mask) -> Self::Mask {
        self.0.mask_not(a)
    }
    #[inline(always)]
    fn select(self, m: Self::Mask, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.select(m, a, b)
    }
    #[inline(always)]
    fn any(self, m: Self::Mask) -> bool {
        self.0.any(m)
    }
    #[inline(always)]
    fn all(self, m: Self::Mask) -> bool {
        self.0.all(m)
    }
    #[inline(always)]
    fn mask_bitmask(self, m: Self::Mask) -> u32 {
        self.0.mask_bitmask(m)
    }
    #[inline(always)]
    fn reduce_sum(self, v: Self::Vector) -> T {
        self.0.reduce_sum(v)
    }
    #[inline(always)]
    fn reduce_min(self, v: Self::Vector) -> T {
        self.0.reduce_min(v)
    }
    #[inline(always)]
    fn reduce_max(self, v: Self::Vector) -> T {
        self.0.reduce_max(v)
    }
}

// The hand-rolled SIMD tokens are crate-internal: application code never names a backend,
// it goes through `dispatch`, which picks one by runtime CPU detection. They stay reachable
// for the in-crate differential tests (`diff_tests`) that verify each against the oracle.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx1;
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
