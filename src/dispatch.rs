//! Entry points that pick a [`Backend`] and run a generic [`Kernel`].
//!
//! A kernel is written once against the [`Simd`] context and is monomorphized for whichever
//! backend is chosen. The backend is never named by the caller — it is chosen per scalar
//! type via [`SimdDispatch`]:
//!
//! * **runtime** (default, std): `is_x86_feature_detected!` picks the widest implemented
//!   backend at the dispatch boundary. The SIMD ops are `#[target_feature]` bodies, so a
//!   wider ISA is used whenever the running CPU has it — regardless of how the crate was
//!   built.
//! * **compile-time fast path**: if the build already guarantees the widest implemented ISA
//!   (e.g. `-C target-cpu=native` on an AVX-512 host sets `target_feature = "avx512f"`),
//!   that backend is taken unconditionally with no runtime branch.
//! * **no-std**: with no runtime detection available, the widest ISA the build guarantees is
//!   taken from `target_feature`.
//!
//! Either way [`ScalarBackend`] is the fallback, so every scalar type always has a path.

use crate::backend::{Backend, ScalarBackend};
use crate::scalar::Scalar;
use crate::varying::Simd;

/// The aarch64 tail of a dispatch, shared by [`SimdDispatch`] and [`MatrixDispatch`] (element-wise
/// and matrix). Policy: **non-Apple** aarch64 with base SVE takes the widest [`Sve`](crate::backend)
/// token the hardware VL covers — the kernel monomorphizes per width and the matching branch runs;
/// everything else (Apple, `--cfg no_sve`, `--cfg neon_over_sve`, no SVE) takes `$fallback`. On
/// Apple that fallback is the *only* aarch64 path, so the Apple-NEON / Apple-scalar policy is what
/// this expands to. `$fallback` is the per-scalar floor: `Neon` where it implements the scalar,
/// `ScalarBackend` for `f16` (no NEON f16). Expands to nothing off aarch64.
macro_rules! aarch64_dispatch_tail {
    ($kernel:expr, $fallback:expr) => {{
        #[cfg(all(
            target_arch = "aarch64",
            not(target_vendor = "apple"),
            feature = "std",
            not(no_sve),
            not(neon_over_sve)
        ))]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                let vl = crate::arch::sve2::vl_bytes();
                if vl >= 64 {
                    return $kernel.run(crate::varying::Simd::new(unsafe {
                        crate::backend::sve::Sve::<64>::new_unchecked()
                    }));
                }
                if vl >= 32 {
                    return $kernel.run(crate::varying::Simd::new(unsafe {
                        crate::backend::sve::Sve::<32>::new_unchecked()
                    }));
                }
                return $kernel.run(crate::varying::Simd::new(unsafe {
                    crate::backend::sve::Sve::<16>::new_unchecked()
                }));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return $kernel.run(crate::varying::Simd::new($fallback));
        }
    }};
}
pub(crate) use aarch64_dispatch_tail;

/// The wasm32 tail, shared by every dispatch. WASM has no runtime feature detection, so the token
/// is chosen at compile time from `target_feature`: `relaxed-simd` takes [`RelaxedSimd`](crate::backend),
/// plain `simd128` takes [`Simd128`](crate::backend), and a build with neither falls through to the
/// caller's scalar floor. Expands to nothing off wasm32.
macro_rules! wasm_dispatch_tail {
    ($kernel:expr) => {{
        #[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
        {
            return $kernel.run(crate::varying::Simd::new(
                crate::backend::wasm::RelaxedSimd::new(),
            ));
        }
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            not(target_feature = "relaxed-simd")
        ))]
        {
            return $kernel.run(crate::varying::Simd::new(crate::backend::wasm::Simd128::new()));
        }
    }};
}
pub(crate) use wasm_dispatch_tail;

/// The riscv64 tail, shared by every dispatch. RVV ("V" extension) is *scalable*, so — like the SVE
/// tail — it picks the widest [`Rvv`](crate::backend) token the detected `VLENB` covers and
/// monomorphizes the kernel for it. Unlike aarch64 (which always has the NEON floor), RISC-V has no
/// other SIMD backend here, so this returns *only* when "V" is present; otherwise it expands to
/// nothing and the caller falls through to its scalar floor. Expands to nothing off riscv64.
macro_rules! riscv_dispatch_tail {
    ($kernel:expr) => {{
        // Compile-time: a build that statically guarantees "V" skips detection (only `VLENB`,
        // which is runtime even then). Works in no-std.
        #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
        {
            let vl = crate::arch::rvv::vlenb();
            // SAFETY: the build guarantees the "V" extension; `VLENB` ≥ 16 (mandated by "V").
            if vl >= 64 {
                return $kernel.run(crate::varying::Simd::new(unsafe {
                    crate::backend::rvv::Rvv::<64>::new_unchecked()
                }));
            }
            if vl >= 32 {
                return $kernel.run(crate::varying::Simd::new(unsafe {
                    crate::backend::rvv::Rvv::<32>::new_unchecked()
                }));
            }
            return $kernel.run(crate::varying::Simd::new(unsafe {
                crate::backend::rvv::Rvv::<16>::new_unchecked()
            }));
        }
        #[cfg(all(target_arch = "riscv64", feature = "std", not(target_feature = "v")))]
        {
            if crate::arch::rvv::is_supported() {
                let vl = crate::arch::rvv::vlenb();
                if vl >= 64 {
                    return $kernel.run(crate::varying::Simd::new(unsafe {
                        crate::backend::rvv::Rvv::<64>::new_unchecked()
                    }));
                }
                if vl >= 32 {
                    return $kernel.run(crate::varying::Simd::new(unsafe {
                        crate::backend::rvv::Rvv::<32>::new_unchecked()
                    }));
                }
                return $kernel.run(crate::varying::Simd::new(unsafe {
                    crate::backend::rvv::Rvv::<16>::new_unchecked()
                }));
            }
        }
    }};
}
pub(crate) use riscv_dispatch_tail;

/// The 32-bit-ARM (armv7) tail. NEON there is **f32-only** (no `f64`/`f16` vector unit), so this is
/// invoked only from the `f32` dispatch — `f64`/`f16`/`bf16` fall through to scalar. Compile-time:
/// a build that guarantees NEON skips detection; std runtime: `Neon::detect()` (HWCAP). Returns only
/// when NEON is present, else expands to nothing (no other ARM-32 SIMD floor). Nothing off arm.
macro_rules! arm_dispatch_tail {
    ($kernel:expr) => {{
        #[cfg(all(target_arch = "arm", target_feature = "neon"))]
        {
            // SAFETY: the build guarantees NEON.
            return $kernel.run(crate::varying::Simd::new(unsafe {
                crate::backend::neon_a32::Neon::new_unchecked()
            }));
        }
        #[cfg(all(target_arch = "arm", feature = "std", not(target_feature = "neon")))]
        {
            if let Some(b) = crate::backend::neon_a32::Neon::detect() {
                return $kernel.run(crate::varying::Simd::new(b));
            }
        }
    }};
}
pub(crate) use arm_dispatch_tail;

/// A unit of work generic over the execution backend. Implement this once; `hydroplane` runs it on
/// the backend it selects, handing your [`run`](Kernel::run) a [`Simd`] context to build
/// varying values through (`splat`, `load`, …). Reach the raw [`Backend`] token, if you need
/// it, via [`Simd::backend`].
pub trait Kernel<T: Scalar> {
    type Output;
    fn run<S: Backend<T>>(self, simd: Simd<T, S>) -> Self::Output;
}

/// Per-scalar dispatch policy. `f32`/`f64` try a SIMD backend then fall back to scalar;
/// other scalars (e.g. `f16` before the native-FP16 milestone) use the scalar path.
pub trait SimdDispatch: Scalar {
    fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output;
}

/// Run `kernel` on the best available backend for `T`, chosen by runtime CPU detection.
#[inline]
pub fn dispatch<T: SimdDispatch, K: Kernel<T>>(kernel: K) -> K::Output {
    T::dispatch(kernel)
}

/// Run `kernel` on the always-available scalar (1-lane) backend. Useful as a correctness
/// oracle or baseline; normal code should use [`dispatch`].
#[inline]
pub fn run_scalar<T: Scalar, K: Kernel<T>>(kernel: K) -> K::Output {
    kernel.run(Simd::new(ScalarBackend))
}

macro_rules! impl_simd_dispatch_x86 {
    ($ty:ty $(, $arm_tail:ident)?) => {
        impl SimdDispatch for $ty {
            #[inline]
            // When the build statically pins the widest ISA, that branch `return`s and the
            // rest is unreachable — the intended compile-time fast path.
            #[allow(unreachable_code)]
            fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
                #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "avx512f"))]
                {
                    // SAFETY: target compiled with avx512f.
                    let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                    return kernel.run(Simd::new(b));
                }
                // std, not statically avx512: avx512 is the one genuinely unknown ISA, so it
                // is the only runtime check. The floor beneath it (avx2 then sse4) is taken
                // *statically* via `new_unchecked` whenever the build already guarantees it —
                // so an avx2-baseline build (x86-64-v3) multiversions on a single avx512 check,
                // and tiers below a static floor are `cfg`'d out rather than left as dead
                // detects.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    feature = "std",
                    not(target_feature = "avx512f")
                ))]
                {
                    if let Some(b) = crate::backend::avx512::Avx512::detect() {
                        return kernel.run(Simd::new(b));
                    }
                    #[cfg(all(target_feature = "avx2", target_feature = "fma"))]
                    {
                        // SAFETY: build guarantees avx2+fma.
                        let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                        return kernel.run(Simd::new(b));
                    }
                    #[cfg(not(all(target_feature = "avx2", target_feature = "fma")))]
                    {
                        if let Some(b) = crate::backend::avx2::Avx2::detect() {
                            return kernel.run(Simd::new(b));
                        }
                        #[cfg(target_feature = "sse4.1")]
                        {
                            // SAFETY: build guarantees sse4.1.
                            let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                            return kernel.run(Simd::new(b));
                        }
                        #[cfg(not(target_feature = "sse4.1"))]
                        {
                            if let Some(b) = crate::backend::sse4::Sse4::detect() {
                                return kernel.run(Simd::new(b));
                            }
                        }
                    }
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    not(feature = "std"),
                    not(target_feature = "avx512f"),
                    target_feature = "avx2",
                    target_feature = "fma"
                ))]
                {
                    // SAFETY: target compiled with avx2+fma.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Simd::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    not(feature = "std"),
                    not(all(target_feature = "avx2", target_feature = "fma")),
                    target_feature = "sse4.1"
                ))]
                {
                    // SAFETY: target compiled with sse4.1.
                    let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                    return kernel.run(Simd::new(b));
                }
                // aarch64: non-Apple SVE (by VL) else NEON — NEON is the only SIMD backend on Apple.
                aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
                // riscv64: RVV by VLENB when the "V" extension is present, else scalar below.
                riscv_dispatch_tail!(kernel);
                // arm (armv7): NEON when present — only emitted for f32 (NEON there is f32-only).
                $( $arm_tail!(kernel); )?
                // wasm32: relaxed-simd else simd128 (compile-time, no runtime detection).
                wasm_dispatch_tail!(kernel);
                kernel.run(Simd::new(ScalarBackend))
            }
        }
    };
}

impl_simd_dispatch_x86!(f32, arm_dispatch_tail);
impl_simd_dispatch_x86!(f64);

// Scalars without a hand-rolled SIMD backend yet (f16/bf16) always take the scalar path.
mod half_dispatch {
    use super::{Kernel, ScalarBackend, Simd, SimdDispatch};
    use half::{bf16, f16};

    impl SimdDispatch for f16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
            // Native AVX-512-FP16 (32-wide), statically guaranteed by the build.
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "avx512fp16"))]
            {
                // SAFETY: target compiled with avx512fp16.
                let b = unsafe { crate::backend::avx512fp16::Avx512Fp16::new_unchecked() };
                return kernel.run(Simd::new(b));
            }
            // Runtime detection: native FP16 if present, else the AVX2 F16C widen path.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(target_feature = "avx512fp16")
            ))]
            {
                if let Some(b) = crate::backend::avx512fp16::Avx512Fp16::detect() {
                    return kernel.run(Simd::new(b));
                }
                // The F16C widen path is the floor: branchless when the build guarantees it.
                #[cfg(all(
                    target_feature = "avx2",
                    target_feature = "fma",
                    target_feature = "f16c"
                ))]
                {
                    // SAFETY: build guarantees avx2+fma+f16c.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Simd::new(b));
                }
                #[cfg(not(all(
                    target_feature = "avx2",
                    target_feature = "fma",
                    target_feature = "f16c"
                )))]
                {
                    if let Some(b) = crate::backend::avx2::Avx2::detect() {
                        return kernel.run(Simd::new(b));
                    }
                }
            }
            // no-std: the AVX2 F16C widen path if the build guarantees it.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                not(feature = "std"),
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "f16c"
            ))]
            {
                // SAFETY: target compiled with avx2+fma+f16c.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Simd::new(b));
            }
            // aarch64: non-Apple SVE has native f16; NEON has no f16, so the fallback is scalar.
            super::aarch64_dispatch_tail!(kernel, ScalarBackend);
            kernel.run(Simd::new(ScalarBackend))
        }
    }
    impl SimdDispatch for bf16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
            // bf16 widen path: f32x16/f32x8 on x86, f32x4 on NEON (16-bit storage, f32 compute).
            // Native AVX-512-BF16 (hardware bf16↔f32 at load/store), statically guaranteed by the build.
            #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), target_feature = "avx512bf16"))]
            {
                // SAFETY: target compiled with avx512bf16.
                let b = unsafe { crate::backend::avx512bf16::Avx512Bf16::new_unchecked() };
                return kernel.run(Simd::new(b));
            }
            // Runtime detection: native AVX-512-BF16 first, then the AVX-512 / AVX2 f32 widen paths.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(target_feature = "avx512bf16")
            ))]
            {
                if let Some(b) = crate::backend::avx512bf16::Avx512Bf16::detect() {
                    return kernel.run(Simd::new(b));
                }
                if let Some(b) = crate::backend::avx512::Avx512::detect() {
                    return kernel.run(Simd::new(b));
                }
                if let Some(b) = crate::backend::avx2::Avx2::detect() {
                    return kernel.run(Simd::new(b));
                }
            }
            // aarch64: non-Apple SVE (bf16 widen path) else NEON.
            super::aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
            // wasm32: bf16 widen path on relaxed-simd else simd128.
            super::wasm_dispatch_tail!(kernel);
            kernel.run(Simd::new(ScalarBackend))
        }
    }
}
