//! `hydroplane`: float-agnostic, ISPC-style SPMD/SIMD infrastructure.
//!
//! Write one kernel generic over the scalar element ([`Scalar`]: `f32`, `f64`, `f16`, `bf16`);
//! [`dispatch()`] runs it on the widest backend runtime CPU detection finds, else the portable
//! [`ScalarBackend`] (also the rust-gpu/SPIR-V lowering target). Kernels never name a backend.
//! Stable Rust, `no_std`-compatible, no SIMD-crate dependency; `f16`/`bf16` come from the
//! `half` crate, re-exported as [`struct@f16`]/[`bf16`].
#![cfg_attr(not(feature = "std"), no_std)]
// The subgroup backend reaches SPIR-V through inline assembly (`asm_experimental_arch`), and the
// cooperative-matrix tiles hold `f16` as a real 16-bit float (`f16`). These are the crate's only
// nightly requirements and apply solely to the rust-gpu/SPIR-V build; every CPU target compiles
// on stable.
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch, f16))]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Padding granularity: the widest lane count we target (AVX-512-FP16 is 32-wide f16). Every
/// backend's lane count (1, 2, 4, 8, 16, 32) divides this, so a full-register loop never has
/// a remainder, and a single inline `[T; MAX_LANES]` buffer holds any one register.
pub const MAX_LANES: usize = 32;

/// Upper bound on a backend's [`UNROLL`](Backend::UNROLL): the most independent accumulator chains
/// any reduction unrolls to. Sizes the fixed `[init; MAX_UNROLL]` chain array, so a backend's
/// `UNROLL` must not exceed it.
pub const MAX_UNROLL: usize = 16;

pub mod backend;
#[doc(hidden)]
pub mod isa;
pub mod arch;
pub mod dispatch;
// The runtime unroll cache; gone when `build.rs` resolved the factor at compile time.
#[cfg(not(hp_resolved_unroll))]
pub(crate) mod ilp;
pub mod dense;
#[cfg(feature = "alloc")]
mod scratch;
pub mod matrix;
pub mod scalar;
pub mod varying;

#[cfg(feature = "alloc")]
pub mod cols;
#[cfg(feature = "alloc")]
pub mod soa;

/// Opt-in `glam`-aware wide-vector helpers ([`Vec3Wide`](glam_ext::Vec3Wide) etc.). Behind the
/// `glam` feature so the core stays geometry-free.
#[cfg(feature = "glam")]
pub mod glam_ext;

pub use backend::{Backend, BackendAll, Enter, ScalarBackend};
pub use dispatch::{Kernel, SimdDispatch, dispatch, run_scalar};

/// The on-device entry point (rust-gpu / SPIR-V target): mirrors [`dispatch`], but branches on
/// work size (subgroup-distributed vs. a single sequential invocation) instead of CPU ISA.
#[cfg(target_arch = "spirv")]
pub use backend::subgroup::{dispatch_matrix_subgroup, dispatch_subgroup};
pub use dense::{Diag, Mat, Side, Trans, Uplo, col_sums, fro_norm, gemv, row_sums};
#[cfg(feature = "alloc")]
pub use dense::{MatMut, gemm, potrf, syrk, trsm};
pub use matrix::{
    Accumulator, Layout, MatrixA, MatrixB, MatrixBackend, MatrixDispatch, MatrixKernel, Role, Tile,
    Tiles, dispatch_matrix, run_matrix_scalar,
};
pub use scalar::{FloatScalar, IntScalar, Scalar};
pub use varying::{ChunksExact, Varying, VaryingI32, VaryingU32, Mask, Gang};

/// The `#[kernel]` attribute: write a [`Kernel`]/[`MatrixKernel`] as a plain generic function.
/// Contexts, tuning flags (`tiny`, `noalias`, `unroll = N`), the generated `<name>_on` companion
/// for calling one kernel from another without re-dispatching, and the `matrix` form are all
/// documented on [the attribute itself](macro@kernel).
pub use hydroplane_macros::kernel;

/// `f16`/`bf16` element types (from the `half` crate), usable anywhere a [`Scalar`] is expected.
pub use half::{bf16, f16};

/// `$body` once for each `$i` in `0..$n`, with `$i` a literal; `$n` is a constant of at most 8
/// (a const generic works).
///
/// Use it for register blocking, so an array of tiles indexed by `$i` resolves to fixed
/// elements:
///
/// ```ignore
/// let mut acc = [[tl.zero_acc::<16, 16>(); BN]; BM];
/// hydroplane::unroll!(BM, i, {
///     let a = tl.load_a::<16, 16>(&a_mem[(row0 + i * 16) * k_dim + k..], k_dim, Layout::RowMajor);
///     hydroplane::unroll!(BN, j, {
///         acc[i][j] = tl.mma(a, b[j], acc[i][j]);
///     });
/// });
/// ```
///
/// On the GPU an array indexed by a loop variable lives in memory, which made a register-blocked
/// cooperative-matrix GEMM about 4× slower; there is no loop-unroll attribute to ask for this.
/// On the CPU it is an ordinary unrolled loop.
#[macro_export]
macro_rules! unroll {
    (0, $i:ident, $body:block) => {{}};
    (1, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0) };
    (2, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1) };
    (3, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1 2) };
    (4, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1 2 3) };
    (5, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1 2 3 4) };
    (6, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1 2 3 4 5) };
    (7, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1 2 3 4 5 6) };
    (8, $i:ident, $body:block) => { $crate::unroll!(@ $i, $body, 0 1 2 3 4 5 6 7) };
    (@ $i:ident, $body:block, $($k:literal)*) => {{
        $({
            let $i: usize = $k;
            $body
        })*
    }};
    // A count only known after monomorphization (a const generic) needs every possible copy,
    // each behind a test the compiler folds away.
    ($n:expr, $i:ident, $body:block) => {{
        const { assert!($n <= 8, "hydroplane::unroll! takes at most 8 iterations") };
        if $n > 0 {
            let $i: usize = 0;
            $body
        }
        if $n > 1 {
            let $i: usize = 1;
            $body
        }
        if $n > 2 {
            let $i: usize = 2;
            $body
        }
        if $n > 3 {
            let $i: usize = 3;
            $body
        }
        if $n > 4 {
            let $i: usize = 4;
            $body
        }
        if $n > 5 {
            let $i: usize = 5;
            $body
        }
        if $n > 6 {
            let $i: usize = 6;
            $body
        }
        if $n > 7 {
            let $i: usize = 7;
            $body
        }
    }};
}

/// The `#[kernel]` multiply-add rewrite: `a * b + c` becomes
/// `(&Fuse::new(a, b, c)).__hp_fuse()`, which method resolution sends to [`ViaFma`] (a fused
/// multiply-add) for varying operands and to [`ViaOps`] (the original operators) for anything
/// else.
#[doc(hidden)]
pub mod __fuse {
    use core::cell::Cell;
    use core::ops::{Add, Mul};

    use crate::backend::Backend;
    use crate::scalar::FloatScalar;
    use crate::varying::Varying;

    pub struct Fuse<A, B, C>(Cell<Option<(A, B, C)>>);

    impl<A, B, C> Fuse<A, B, C> {
        #[inline(always)]
        pub fn new(a: A, b: B, c: C) -> Self {
            Fuse(Cell::new(Some((a, b, c))))
        }

        #[inline(always)]
        fn take(&self) -> (A, B, C) {
            match self.0.take() {
                Some(v) => v,
                None => unreachable!(),
            }
        }
    }

    pub trait ViaFma {
        type Out;
        fn __hp_fuse(&self) -> Self::Out;
    }

    impl<T: FloatScalar, S: Backend<T>> ViaFma for Fuse<Varying<T, S>, Varying<T, S>, Varying<T, S>> {
        type Out = Varying<T, S>;
        #[inline(always)]
        fn __hp_fuse(&self) -> Varying<T, S> {
            let (a, b, c) = self.take();
            a.fma(b, c)
        }
    }

    impl<T: FloatScalar, S: Backend<T>> ViaFma for Fuse<Varying<T, S>, T, Varying<T, S>> {
        type Out = Varying<T, S>;
        #[inline(always)]
        fn __hp_fuse(&self) -> Varying<T, S> {
            let (a, b, c) = self.take();
            a.fma(c.splat_like(b), c)
        }
    }

    pub trait ViaOps {
        type Out;
        fn __hp_fuse(&self) -> Self::Out;
    }

    impl<A: Mul<B>, B, C> ViaOps for &Fuse<A, B, C>
    where
        A::Output: Add<C>,
    {
        type Out = <A::Output as Add<C>>::Output;
        #[inline(always)]
        fn __hp_fuse(&self) -> Self::Out {
            let (a, b, c) = self.take();
            a * b + c
        }
    }
}

/// Re-export of the crate supplying [`Scalar`]'s numeric supertrait
/// ([`FloatCore`](num_traits::float::FloatCore)), so generic kernels can name its traits without
/// depending on `num-traits` themselves.
pub use num_traits;

#[cfg(feature = "alloc")]
pub use cols::Cols;
#[cfg(feature = "alloc")]
pub use soa::Soa;

#[cfg(feature = "glam")]
pub use glam_ext::{GangGlamExt, Mat3Wide, MatWide, Vec3Wide};

/// The combo-dispatch tier tokens and codes, reachable by `#[kernel]`-generated wrappers.
/// Implementation detail: application code never names a backend, the generated match does.
#[doc(hidden)]
pub mod towers {
    pub use crate::dispatch::tier::*;
    pub use crate::ScalarBackend;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub use crate::backend::{
        avx2::Avx2, avx512::Avx512, avx512bf16::Avx512Bf16, avx512fp16::Avx512Fp16, sse4::Sse4,
    };
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "avx1"))]
    pub use crate::backend::avx1::Avx1;
    /// Without the `avx1` feature the AVX tier is never selected; its arm in a `#[kernel]` match
    /// reuses the SSE4 instantiation rather than compiling another copy of the kernel.
    #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), not(feature = "avx1")))]
    pub use crate::backend::sse4::Sse4 as Avx1;
    #[cfg(target_arch = "aarch64")]
    pub use crate::backend::neon::Neon;
    #[cfg(target_arch = "aarch64")]
    pub use crate::backend::sve::Sve;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub use crate::backend::wasm::Simd128;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128", target_feature = "relaxed-simd"))]
    pub use crate::backend::wasm::RelaxedSimd;
}

/// Test hook: the unroll factor in effect — `0` until the first dispatch resolves it via the runtime
/// sweep, or the `build.rs`-baked constant when it was resolved at compile time. Not part of the
/// stable surface.
#[doc(hidden)]
pub fn ilp_detected_for_test() -> u8 {
    #[cfg(not(hp_resolved_unroll))]
    {
        ilp::cached()
    }
    #[cfg(hp_resolved_unroll)]
    {
        varying::STATIC_UNROLL as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny generic kernel: are any of `xs` within `r` of the origin along one axis?
    /// Written once, runs on any `Backend<T>`.
    fn any_within<T: Scalar, S: Backend<T>>(s: S, xs: &[T], r: T) -> bool {
        let lanes = s.lanes();
        let rv = s.splat(r);
        let mut i = 0;
        while i + lanes <= xs.len() {
            let x = s.load(&xs[i..i + lanes]);
            let ax = s.max(x, s.neg(x)); // |x|
            if s.any(s.le(ax, rv)) {
                return true;
            }
            i += lanes;
        }
        // scalar tail
        while i < xs.len() {
            if xs[i] <= r && xs[i].neg() <= r {
                return true;
            }
            i += 1;
        }
        false
    }

    #[test]
    fn scalar_backend_smoke() {
        let xs = [3.0f32, -2.0, 5.0, 0.5];
        assert!(any_within(ScalarBackend, &xs, 1.0));
        assert!(!any_within(ScalarBackend, &xs, 0.4));

        let xd = [3.0f64, -2.0, 5.0, 0.5];
        assert!(any_within(ScalarBackend, &xd, 1.0));
        assert!(!any_within(ScalarBackend, &xd, 0.4));
    }

    /// A `Kernel` wrapping `any_within`, run through `dispatch` (whatever backend is best).
    struct AnyWithin<'a, T: Scalar> {
        xs: &'a [T],
        r: T,
    }
    impl<T: Scalar> Kernel<T> for AnyWithin<'_, T> {
        type Output = bool;
        fn run<S: backend::BackendAll + Backend<T>>(self, simd: Gang<S>) -> bool {
            any_within(simd.backend(), self.xs, self.r)
        }
    }

    #[test]
    fn dispatch_matches_scalar_oracle() {
        // The dispatched backend (AVX2 when present) must agree with the scalar path.
        let xs: Vec<f32> = (0..1000).map(|i| (i as f32 % 13.0) - 6.0).collect();
        for &r in &[0.1f32, 0.5, 1.0, 3.0] {
            let dispatched = dispatch(AnyWithin { xs: &xs, r });
            let oracle = any_within(ScalarBackend, &xs, r);
            assert_eq!(dispatched, oracle, "r={r}");
        }
    }
}
