//! `hydroplane` — float-agnostic, ISPC-style SPMD/SIMD infrastructure.
//!
//! Write one kernel generic over the scalar element ([`Scalar`]: `f32`, `f64`, `f16`, `bf16`);
//! [`dispatch()`] runs it on the backend it selects by
//! runtime CPU detection (the widest of AVX-512/AVX2/SSE4/NEON the host supports, else the
//! portable [`ScalarBackend`], which is also the rust-gpu/SPIR-V lowering target). The
//! concrete SIMD backends are an implementation detail — you never name one.
//!
//! The whole crate is **stable Rust** with no SIMD-crate dependency. `f16`/`bf16` come from the
//! `half` crate, re-exported here as [`f16`]/[`bf16`].
#![cfg_attr(not(feature = "std"), no_std)]
// The SubgroupSize-builtin reader in the subgroup backend uses inline SPIR-V assembly, which
// is behind the still-unstable `asm_experimental_arch` gate on the rust-gpu target. This is the
// crate's only nightly requirement, and it applies solely to the rust-gpu/SPIR-V build — every
// CPU target (including native AVX-512-FP16 f16) compiles on stable.
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]

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
pub mod arch;
pub mod dispatch;
// The runtime unroll cache; gone when `build.rs` resolved the factor at compile time.
#[cfg(not(hp_resolved_unroll))]
pub(crate) mod ilp;
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

pub use backend::{Backend, BackendAll, ScalarBackend};
pub use dispatch::{Kernel, SimdDispatch, dispatch, run_scalar};

/// The on-device entry point (rust-gpu / SPIR-V target): mirrors [`dispatch`], but branches
/// on work size — subgroup-distributed vs. a single sequential invocation — instead of CPU ISA.
#[cfg(target_arch = "spirv")]
pub use backend::subgroup::dispatch_subgroup;
pub use matrix::{
    Accumulator, Layout, MatrixA, MatrixB, MatrixBackend, MatrixDispatch, MatrixKernel, Role, Tile,
    Tiles, dispatch_matrix, run_matrix_scalar,
};
pub use scalar::{FloatScalar, IntScalar, Scalar};
pub use varying::{ChunksExact, Varying, VaryingI32, VaryingU32, Mask, Gang};

/// The `#[kernel]` attribute: write a [`Kernel`]/[`MatrixKernel`] as a plain generic function.
/// The full shape — contexts, tuning flags (`tiny`, `noalias`, `unroll = N`), the generated
/// `<name>_on` companion for calling one kernel from another without re-dispatching, and the
/// `matrix` form — is documented on [the attribute itself](macro@kernel) (the `hydroplane_macros`
/// crate docs).
pub use hydroplane_macros::kernel;

/// `f16`/`bf16` element types (from the `half` crate), usable anywhere a [`Scalar`] is expected.
pub use half::{bf16, f16};

/// Re-export of the crate supplying [`Scalar`]'s numeric supertrait
/// ([`FloatCore`](num_traits::float::FloatCore)), so generic kernels can name its traits without
/// depending on `num-traits` themselves.
pub use num_traits;

#[cfg(feature = "alloc")]
pub use cols::Cols;
#[cfg(feature = "alloc")]
pub use soa::Soa;

#[cfg(feature = "glam")]
pub use glam_ext::{GangGlamExt, Mat3Wide, Vec3Wide};

/// The combo-dispatch tier tokens and codes, reachable by `#[kernel]`-generated wrappers.
/// Implementation detail: application code never names a backend — the generated match does.
#[doc(hidden)]
pub mod towers {
    pub use crate::dispatch::tier::*;
    pub use crate::ScalarBackend;
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub use crate::backend::{
        avx1::Avx1, avx2::Avx2, avx512::Avx512, avx512bf16::Avx512Bf16, avx512fp16::Avx512Fp16,
        sse4::Sse4,
    };
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
