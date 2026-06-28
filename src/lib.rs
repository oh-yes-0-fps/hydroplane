//! `spmd` — float-agnostic, ISPC-style SPMD/SIMD infrastructure.
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

pub mod backend;
pub mod arch;
pub mod dispatch;
pub mod kernel_macro;
pub mod matrix;
pub mod scalar;
pub mod varying;

#[cfg(feature = "alloc")]
pub mod soa;

pub use backend::{Backend, ScalarBackend};
pub use dispatch::{Kernel, SimdDispatch, dispatch, run_scalar};

/// The on-device entry point (rust-gpu / SPIR-V target): mirrors [`dispatch`], but branches
/// on work size — subgroup-distributed vs. a single sequential invocation — instead of CPU ISA.
#[cfg(target_arch = "spirv")]
pub use backend::subgroup::dispatch_subgroup;
pub use matrix::{
    Accumulator, Layout, MatrixA, MatrixB, MatrixBackend, MatrixDispatch, MatrixKernel, Role, Tile,
    Tiles, dispatch_matrix, run_matrix_scalar,
};
pub use scalar::Scalar;
pub use varying::{Chunks, Varying, Mask, Simd};

/// `f16`/`bf16` element types (from the `half` crate), usable anywhere a [`Scalar`] is expected.
pub use half::{bf16, f16};

#[cfg(feature = "alloc")]
pub use soa::Soa;

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
            if xs[i].le(r) && xs[i].neg().le(r) {
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
        fn run<S: Backend<T>>(self, simd: Simd<T, S>) -> bool {
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
