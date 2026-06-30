//! Entry points that pick a [`Backend`] and run a generic [`Kernel`].
//!
//! A kernel is written once against the [`Gang`] context and is monomorphized for whichever
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
//!
//! Three build cfgs trim the x86 backend set, for squeezing a native build:
//!
//! * `--cfg static_dispatch` removes the runtime detection ladder *and* its cached atomic: the
//!   backend is taken from `target_feature` at compile time only, exactly as the no-std path does.
//!   Pair it with `-C target-cpu=native` (or explicit `-C target-feature`) to fold the whole
//!   dispatch into a single unconditional `run` with no branch. Also static-pins the SVE/RVV/NEON
//!   tails (the scalable-vector *width* is still read at runtime — that is intrinsic, not a branch
//!   over backends).
//! * `--cfg no_avx512` drops the AVX-512 tiers (`Avx512`/`Avx512Fp16`/`Avx512Bf16`), so x86 floors
//!   at AVX2 — runtime detection never probes for them and a statically-`avx512f` build won't take them.
//! * `--cfg no_avx` drops the whole AVX family (implies `no_avx512`), flooring x86 at SSE4.

use crate::backend::{Backend, ScalarBackend};
use crate::scalar::Scalar;
use crate::varying::Gang;

/// The aarch64 tail of a dispatch, shared by [`SimdDispatch`] and [`MatrixDispatch`] (element-wise
/// and matrix). Policy: **non-Apple** aarch64 with base SVE takes the widest [`Sve`](crate::backend)
/// token the hardware VL covers — the kernel monomorphizes per width and the matching branch runs;
/// everything else (Apple, `--cfg no_sve`, `--cfg neon_over_sve`, no SVE) takes `$fallback`. On
/// Apple that fallback is the *only* aarch64 path, so the Apple-NEON / Apple-scalar policy is what
/// this expands to. `$fallback` is the per-scalar floor: `Neon` where it implements the scalar,
/// `ScalarBackend` for `f16` (no NEON f16). Expands to nothing off aarch64.
macro_rules! aarch64_dispatch_tail {
    ($kernel:expr, $fallback:expr) => {{
        // Compile-time SVE: the build already guarantees the extension (`-C target-feature=+sve` or
        // native), so no detection is needed — only the scalable vector length is read at runtime,
        // which is intrinsic to SVE, not a backend branch. Taken on the no-std and `static_dispatch`
        // paths, and as a fast path whenever the build pins SVE.
        #[cfg(all(
            target_arch = "aarch64",
            not(target_vendor = "apple"),
            target_feature = "sve",
            not(no_sve),
            not(neon_over_sve)
        ))]
        {
            let vl = crate::arch::sve2::vl_bytes();
            if vl >= 64 {
                return $kernel.run(crate::varying::Gang::new(unsafe {
                    crate::backend::sve::Sve::<64>::new_unchecked()
                }));
            }
            if vl >= 32 {
                return $kernel.run(crate::varying::Gang::new(unsafe {
                    crate::backend::sve::Sve::<32>::new_unchecked()
                }));
            }
            return $kernel.run(crate::varying::Gang::new(unsafe {
                crate::backend::sve::Sve::<16>::new_unchecked()
            }));
        }
        #[cfg(all(
            target_arch = "aarch64",
            not(target_vendor = "apple"),
            feature = "std",
            not(static_dispatch),
            not(target_feature = "sve"),
            not(no_sve),
            not(neon_over_sve)
        ))]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                let vl = crate::arch::sve2::vl_bytes();
                if vl >= 64 {
                    return $kernel.run(crate::varying::Gang::new(unsafe {
                        crate::backend::sve::Sve::<64>::new_unchecked()
                    }));
                }
                if vl >= 32 {
                    return $kernel.run(crate::varying::Gang::new(unsafe {
                        crate::backend::sve::Sve::<32>::new_unchecked()
                    }));
                }
                return $kernel.run(crate::varying::Gang::new(unsafe {
                    crate::backend::sve::Sve::<16>::new_unchecked()
                }));
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return $kernel.run(crate::varying::Gang::new($fallback));
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
            return $kernel.run(crate::varying::Gang::new(
                crate::backend::wasm::RelaxedSimd::new(),
            ));
        }
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            not(target_feature = "relaxed-simd")
        ))]
        {
            return $kernel.run(crate::varying::Gang::new(crate::backend::wasm::Simd128::new()));
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
                return $kernel.run(crate::varying::Gang::new(unsafe {
                    crate::backend::rvv::Rvv::<64>::new_unchecked()
                }));
            }
            if vl >= 32 {
                return $kernel.run(crate::varying::Gang::new(unsafe {
                    crate::backend::rvv::Rvv::<32>::new_unchecked()
                }));
            }
            return $kernel.run(crate::varying::Gang::new(unsafe {
                crate::backend::rvv::Rvv::<16>::new_unchecked()
            }));
        }
        #[cfg(all(
            target_arch = "riscv64",
            feature = "std",
            not(static_dispatch),
            not(target_feature = "v")
        ))]
        {
            if crate::arch::rvv::is_supported() {
                let vl = crate::arch::rvv::vlenb();
                if vl >= 64 {
                    return $kernel.run(crate::varying::Gang::new(unsafe {
                        crate::backend::rvv::Rvv::<64>::new_unchecked()
                    }));
                }
                if vl >= 32 {
                    return $kernel.run(crate::varying::Gang::new(unsafe {
                        crate::backend::rvv::Rvv::<32>::new_unchecked()
                    }));
                }
                return $kernel.run(crate::varying::Gang::new(unsafe {
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
            return $kernel.run(crate::varying::Gang::new(unsafe {
                crate::backend::neon_a32::Neon::new_unchecked()
            }));
        }
        #[cfg(all(
            target_arch = "arm",
            feature = "std",
            not(static_dispatch),
            not(target_feature = "neon")
        ))]
        {
            if let Some(b) = crate::backend::neon_a32::Neon::detect() {
                return $kernel.run(crate::varying::Gang::new(b));
            }
        }
    }};
}
pub(crate) use arm_dispatch_tail;

/// A unit of work generic over the execution backend. Implement this once; `hydroplane` runs it on
/// the backend it selects, handing your [`run`](Kernel::run) a [`Gang`] context to build
/// varying values through (`splat`, `load`, …). Reach the raw [`Backend`] token, if you need
/// it, via [`Gang::backend`].
pub trait Kernel<T: Scalar> {
    type Output;
    fn run<S: Backend<T>>(self, simd: Gang<T, S>) -> Self::Output;
}

/// Per-scalar dispatch policy. `f32`/`f64` try a SIMD backend then fall back to scalar;
/// other scalars (e.g. `f16` before the native-FP16 milestone) use the scalar path.
pub trait SimdDispatch: Scalar {
    fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output;
}

/// Run `kernel` on the best available backend for `T`, chosen by runtime CPU detection.
///
/// The kernel is wrapped in `UnrollSelect` first: once the ISA backend is picked, the unroll factor
/// `K` is resolved for this core and baked into an `Unroll<S, K>` backend, so every reduction inside
/// the kernel sees `K` as a compile-time constant.
#[inline]
pub fn dispatch<T: SimdDispatch, K: Kernel<T>>(kernel: K) -> K::Output {
    T::dispatch(UnrollSelect(kernel))
}

/// Resolves the unroll factor on the dispatched backend, then re-runs the wrapped kernel on
/// [`Unroll<S, K>`](crate::backend::Unroll) so `K` is a constant inside it — the const-generic-at-
/// dispatch step that keeps `K` off [`Gang`] and [`Kernel`]. The ISA ladder runs this once; the
/// match on the detected factor then picks the const-`K` wrapper. Each arm monomorphizes the real
/// kernel for that `K`, which is the cost of `K` being a compile-time constant chosen at runtime.
struct UnrollSelect<K>(K);

impl<T: Scalar, K: Kernel<T>> Kernel<T> for UnrollSelect<K> {
    type Output = K::Output;

    #[inline]
    #[cfg(all(not(no_ilp), not(target_arch = "spirv"), not(hp_resolved_unroll)))]
    fn run<S: Backend<T>>(self, g: Gang<T, S>) -> Self::Output {
        use crate::backend::Unroll;
        let b = g.backend();
        match g.unroll() {
            2 => self.0.run(Gang::new(Unroll::<S, 2>(b))),
            4 => self.0.run(Gang::new(Unroll::<S, 4>(b))),
            8 => self.0.run(Gang::new(Unroll::<S, 8>(b))),
            12 => self.0.run(Gang::new(Unroll::<S, 12>(b))),
            16 => self.0.run(Gang::new(Unroll::<S, 16>(b))),
            _ => self.0.run(Gang::new(Unroll::<S, 1>(b))),
        }
    }

    /// Build-resolved (`static_dispatch` + pinned cpu): `K` is the constant `build.rs` baked in, so
    /// there is no detection sweep, no per-dispatch `match`, and the real kernel monomorphizes for
    /// exactly one `Unroll<S, K>` — the fully-static counterpart to the runtime path above.
    #[inline]
    #[cfg(hp_resolved_unroll)]
    fn run<S: Backend<T>>(self, g: Gang<T, S>) -> Self::Output {
        use crate::backend::Unroll;
        self.0
            .run(Gang::new(Unroll::<S, { crate::varying::STATIC_UNROLL }>(g.backend())))
    }

    /// ILP compiled out: no factor to resolve, no wrapper — run the kernel on the raw backend
    /// (whose reductions take the single-chain fold).
    #[inline]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    fn run<S: Backend<T>>(self, g: Gang<T, S>) -> Self::Output {
        self.0.run(g)
    }
}

/// Run `kernel` on the always-available scalar (1-lane) backend. Useful as a correctness
/// oracle or baseline; normal code should use [`dispatch`].
#[inline]
pub fn run_scalar<T: Scalar, K: Kernel<T>>(kernel: K) -> K::Output {
    kernel.run(Gang::new(ScalarBackend))
}

/// Resolve-once cache for the x86 runtime backend tier. The detected tier is immutable for the life
/// of the process, so each scalar's `dispatch` keeps it in a single relaxed atomic (a `static` in the
/// function body): the warm path is one load + a `match`, not a fresh `is_x86_feature_detected!`
/// ladder per call. `0` means unresolved; `resolve` returns the tier code and never `0`. Resolution
/// is idempotent, so a racing thread recomputing the same value and storing it again is harmless.
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std", not(static_dispatch)))]
#[inline]
fn cached_tier(slot: &core::sync::atomic::AtomicU8, resolve: impl FnOnce() -> u8) -> u8 {
    use core::sync::atomic::Ordering;
    match slot.load(Ordering::Relaxed) {
        0 => {
            let t = resolve();
            slot.store(t, Ordering::Relaxed);
            t
        }
        t => t,
    }
}

macro_rules! impl_simd_dispatch_x86 {
    ($ty:ty $(, $arm_tail:ident)?) => {
        impl SimdDispatch for $ty {
            #[inline]
            // When the build statically pins the widest ISA, that branch `return`s and the
            // rest is unreachable — the intended compile-time fast path.
            #[allow(unreachable_code)]
            fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    target_feature = "avx512f",
                    not(any(no_avx, no_avx512))
                ))]
                {
                    // SAFETY: target compiled with avx512f.
                    let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                // std, not statically avx512 (and not `static_dispatch`): the tier
                // (avx512 → avx2 → sse4 → scalar, minus any `no_avx*`-disabled tier) is resolved once
                // by runtime detection and cached in a process-global atomic, so each call is a load +
                // `match` rather than a fresh feature probe.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    feature = "std",
                    not(static_dispatch),
                    not(all(target_feature = "avx512f", not(any(no_avx, no_avx512))))
                ))]
                {
                    use crate::backend::sse4::Sse4;
                    #[cfg(not(any(no_avx, no_avx512)))]
                    use crate::backend::avx512::Avx512;
                    #[cfg(not(no_avx))]
                    use crate::backend::{avx1::Avx1, avx2::Avx2};
                    static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                    let t = cached_tier(&TIER, || {
                        #[cfg(not(any(no_avx, no_avx512)))]
                        if Avx512::detect().is_some() {
                            return 1;
                        }
                        #[cfg(not(no_avx))]
                        if Avx2::detect().is_some() {
                            return 2;
                        }
                        #[cfg(not(no_avx))]
                        if Avx1::detect().is_some() {
                            return 3;
                        }
                        if Sse4::detect().is_some() { 4 } else { u8::MAX }
                    });
                    // SAFETY: each token is built only for the tier `cached_tier` resolved via the
                    // matching `detect()`, which confirmed the CPU has those features this run.
                    return match t {
                        #[cfg(not(any(no_avx, no_avx512)))]
                        1 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                        #[cfg(not(no_avx))]
                        2 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                        #[cfg(not(no_avx))]
                        3 => kernel.run(Gang::new(unsafe { Avx1::new_unchecked() })),
                        4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                        _ => kernel.run(Gang::new(ScalarBackend)),
                    };
                }
                // Compile-time selection — the no-std path, and `static_dispatch` on std. The widest
                // `target_feature`-guaranteed tier that survives the `no_avx*` cfgs, with no branch.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), static_dispatch),
                    not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                    target_feature = "avx2",
                    target_feature = "fma",
                    not(no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx2+fma.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                // AVX (no AVX2/FMA): 256-bit floats with an unfused `fma`.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), static_dispatch),
                    not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                    not(all(target_feature = "avx2", target_feature = "fma", not(no_avx))),
                    target_feature = "avx",
                    not(no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx.
                    let b = unsafe { crate::backend::avx1::Avx1::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), static_dispatch),
                    not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                    not(all(target_feature = "avx2", target_feature = "fma", not(no_avx))),
                    not(all(target_feature = "avx", not(no_avx))),
                    target_feature = "sse4.1"
                ))]
                {
                    // SAFETY: target compiled with sse4.1.
                    let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                // aarch64: non-Apple SVE (by VL) else NEON — NEON is the only SIMD backend on Apple.
                aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
                // riscv64: RVV by VLENB when the "V" extension is present, else scalar below.
                riscv_dispatch_tail!(kernel);
                // arm (armv7): NEON when present — only emitted for f32 (NEON there is f32-only).
                $( $arm_tail!(kernel); )?
                // wasm32: relaxed-simd else simd128 (compile-time, no runtime detection).
                wasm_dispatch_tail!(kernel);
                kernel.run(Gang::new(ScalarBackend))
            }
        }
    };
}

impl_simd_dispatch_x86!(f32, arm_dispatch_tail);
impl_simd_dispatch_x86!(f64);

// Scalars without a hand-rolled SIMD backend yet (f16/bf16) always take the scalar path.
mod half_dispatch {
    use super::{Kernel, ScalarBackend, Gang, SimdDispatch};
    use half::{bf16, f16};

    impl SimdDispatch for f16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
            // Native AVX-512-FP16 (32-wide), statically guaranteed by the build.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                target_feature = "avx512fp16",
                not(any(no_avx, no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512fp16.
                let b = unsafe { crate::backend::avx512fp16::Avx512Fp16::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Runtime detection (cached): native FP16 if present, else the AVX2 F16C widen path.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(no_avx, no_avx512))))
            ))]
            {
                #[cfg(not(no_avx))]
                use crate::backend::avx2::Avx2;
                use crate::backend::sse4::Sse4;
                #[cfg(not(any(no_avx, no_avx512)))]
                use crate::backend::{avx512::Avx512, avx512fp16::Avx512Fp16};
                static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                let t = super::cached_tier(&TIER, || {
                    #[cfg(not(any(no_avx, no_avx512)))]
                    if Avx512Fp16::detect().is_some() {
                        return 1;
                    }
                    // Plain AVX-512 (no FP16): 16-wide `f32x16` widen via hardware vcvtph2ps — twice
                    // the 8-wide AVX2 F16C path on Cascade Lake / Ice Lake / Zen 4.
                    #[cfg(not(any(no_avx, no_avx512)))]
                    if Avx512::detect().is_some() {
                        return 2;
                    }
                    #[cfg(not(no_avx))]
                    if Avx2::detect().is_some() {
                        return 3;
                    }
                    // Pre-AVX2 x86: 4-wide SSE4 scalar-widen, still 4× the scalar floor.
                    if Sse4::detect().is_some() {
                        return 4;
                    }
                    u8::MAX
                });
                // SAFETY: each token is built only for the tier resolved via the matching `detect()`.
                return match t {
                    #[cfg(not(any(no_avx, no_avx512)))]
                    1 => kernel.run(Gang::new(unsafe { Avx512Fp16::new_unchecked() })),
                    #[cfg(not(any(no_avx, no_avx512)))]
                    2 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                    #[cfg(not(no_avx))]
                    3 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                    4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                    _ => kernel.run(Gang::new(ScalarBackend)),
                };
            }
            // Compile-time AVX-512 (no FP16) f16 widen — no-std or `static_dispatch` with an avx512f
            // baseline: 16-wide via hardware vcvtph2ps, preferred over the 8-wide AVX2 path below.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(no_avx, no_avx512)))),
                target_feature = "avx512f",
                not(any(no_avx, no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512f.
                let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Compile-time AVX2 F16C widen path — no-std, or `static_dispatch` on std.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(no_avx, no_avx512)))),
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "f16c",
                not(no_avx)
            ))]
            {
                // SAFETY: target compiled with avx2+fma+f16c.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Compile-time SSE4 scalar-widen f16 — no-std or `static_dispatch` on a pre-AVX2 baseline.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(no_avx, no_avx512)))),
                not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                not(all(target_feature = "avx2", target_feature = "fma", target_feature = "f16c", not(no_avx))),
                target_feature = "sse4.1"
            ))]
            {
                // SAFETY: target compiled with sse4.1.
                let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // aarch64: native NEON FEAT_FP16 is 8-wide `f16` — the Apple-silicon path (no SVE there).
            // Compile-time when the build guarantees `fp16` (Apple baseline, or `-C target-feature=+fp16`).
            #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
            {
                return kernel.run(Gang::new(crate::backend::neon::Neon::new()));
            }
            // Runtime FEAT_FP16 detection (std) → native NEON `f16`; else fall through to SVE / scalar.
            #[cfg(all(
                target_arch = "aarch64",
                feature = "std",
                not(static_dispatch),
                not(target_feature = "fp16")
            ))]
            {
                if std::arch::is_aarch64_feature_detected!("fp16") {
                    return kernel.run(Gang::new(crate::backend::neon::Neon::new()));
                }
            }
            // No NEON `f16` (pre-ARMv8.2): non-Apple SVE has native `f16`, else scalar.
            super::aarch64_dispatch_tail!(kernel, ScalarBackend);
            // wasm32: f16 widen path on relaxed-simd else simd128.
            super::wasm_dispatch_tail!(kernel);
            kernel.run(Gang::new(ScalarBackend))
        }
    }
    impl SimdDispatch for bf16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
            // bf16 widen path: f32x16/f32x8 on x86, f32x4 on NEON (16-bit storage, f32 compute).
            // Native AVX-512-BF16 (hardware bf16↔f32 at load/store), statically guaranteed by the build.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                target_feature = "avx512bf16",
                not(any(no_avx, no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512bf16.
                let b = unsafe { crate::backend::avx512bf16::Avx512Bf16::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Runtime detection (cached): native AVX-512-BF16 first, then the AVX-512 / AVX2 widen paths.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(no_avx, no_avx512))))
            ))]
            {
                #[cfg(not(no_avx))]
                use crate::backend::avx2::Avx2;
                use crate::backend::sse4::Sse4;
                #[cfg(not(any(no_avx, no_avx512)))]
                use crate::backend::{avx512::Avx512, avx512bf16::Avx512Bf16};
                static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                let t = super::cached_tier(&TIER, || {
                    #[cfg(not(any(no_avx, no_avx512)))]
                    if Avx512Bf16::detect().is_some() {
                        return 1;
                    }
                    #[cfg(not(any(no_avx, no_avx512)))]
                    if Avx512::detect().is_some() {
                        return 2;
                    }
                    #[cfg(not(no_avx))]
                    if Avx2::detect().is_some() {
                        return 3;
                    }
                    // Pre-AVX2 x86: 4-wide SSE4 scalar-widen, still 4× the scalar floor.
                    if Sse4::detect().is_some() {
                        return 4;
                    }
                    u8::MAX
                });
                // SAFETY: each token is built only for the tier resolved via the matching `detect()`.
                return match t {
                    #[cfg(not(any(no_avx, no_avx512)))]
                    1 => kernel.run(Gang::new(unsafe { Avx512Bf16::new_unchecked() })),
                    #[cfg(not(any(no_avx, no_avx512)))]
                    2 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                    #[cfg(not(no_avx))]
                    3 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                    4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                    _ => kernel.run(Gang::new(ScalarBackend)),
                };
            }
            // Compile-time bf16 widen — no-std, or `static_dispatch` on std. AVX-512 widen if the build
            // guarantees `avx512f` (and AVX-512 is enabled), else the AVX2 widen path.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(no_avx, no_avx512)))),
                target_feature = "avx512f",
                not(any(no_avx, no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512f.
                let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(no_avx, no_avx512)))),
                not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                target_feature = "avx2",
                target_feature = "fma",
                not(no_avx)
            ))]
            {
                // SAFETY: target compiled with avx2+fma.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Compile-time SSE4 scalar-widen bf16 — no-std or `static_dispatch` on a pre-AVX2 baseline.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(no_avx, no_avx512)))),
                not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                not(all(target_feature = "avx2", target_feature = "fma", not(no_avx))),
                target_feature = "sse4.1"
            ))]
            {
                // SAFETY: target compiled with sse4.1.
                let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // aarch64: non-Apple SVE (bf16 widen path) else NEON.
            super::aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
            // wasm32: bf16 widen path on relaxed-simd else simd128.
            super::wasm_dispatch_tail!(kernel);
            kernel.run(Gang::new(ScalarBackend))
        }
    }
}
