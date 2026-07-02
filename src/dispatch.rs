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

use crate::backend::{Backend, BackendAll, ScalarBackend};
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

    /// Ceiling on the ILP unroll factor for *this* kernel, `min`'d against the factor the runtime
    /// sweep resolves for the core. The default is [`MAX_UNROLL`](crate::MAX_UNROLL) — no cap. The
    /// `#[kernel]` macro overrides it from build-time MIR analysis (register-footprint estimate) so a
    /// register-heavy kernel is not unrolled past the point where it spills; hand-written kernels keep
    /// the default. Only the runtime dispatch path reads it — the build-resolved static path bakes `K`
    /// as a const and cannot be per-kernel-capped on stable.
    const K_CAP: usize = crate::MAX_UNROLL;

    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) -> Self::Output;
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
    fn run<S: BackendAll + Backend<T>>(self, g: Gang<S>) -> Self::Output {
        use crate::backend::Unroll;
        let b = g.backend();
        match g.unroll().min(<K as Kernel<T>>::K_CAP) {
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
    fn run<S: BackendAll + Backend<T>>(self, g: Gang<S>) -> Self::Output {
        use crate::backend::Unroll;
        self.0
            .run(Gang::new(Unroll::<S, { crate::varying::STATIC_UNROLL }>(g.backend())))
    }

    /// ILP compiled out: no factor to resolve, no wrapper — run the kernel on the raw backend
    /// (whose reductions take the single-chain fold).
    #[inline]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    fn run<S: BackendAll + Backend<T>>(self, g: Gang<S>) -> Self::Output {
        self.0.run(g)
    }
}

/// Run `kernel` on the always-available scalar (1-lane) backend. Useful as a correctness
/// oracle or baseline; normal code should use [`dispatch`].
#[inline]
pub fn run_scalar<T: Scalar, K: Kernel<T>>(kernel: K) -> K::Output {
    kernel.run(Gang::new(ScalarBackend))
}

/// Element bits a kernel's type-combo bitmask is built from (`Scalar::TYPE_BITS`).
pub(crate) const F16_BIT: u8 = 4;
pub(crate) const BF16_BIT: u8 = 8;
pub(crate) const INT_BITS: u8 = 48;

/// The ISA tiers combo dispatch can select. Each tier is a `BackendAll` token whose per-element
/// impls encode the best unit *that tier guarantees* — the fp16/bf16 tiers serve f32/f64/ints
/// through plain AVX-512, so a kernel whose combo lacks halves canonicalizes down to
/// [`TIER_AVX512`] and never monomorphizes for them.
#[doc(hidden)]
pub mod tier {
    pub const SCALAR: u8 = 1;
    pub const SSE4: u8 = 2;
    pub const AVX1: u8 = 3;
    pub const AVX2: u8 = 4;
    pub const AVX512: u8 = 5;
    pub const AVX512FP16: u8 = 6;
    pub const AVX512BF16: u8 = 7;
    pub const NEON: u8 = 8;
    pub const SVE16: u8 = 9;
    pub const SVE32: u8 = 10;
    pub const SVE64: u8 = 11;
    pub const SIMD128: u8 = 12;
    pub const RELAXED: u8 = 13;
}

/// What the host guarantees, as seen by [`canonical_tier`]. Pure data so the policy is
/// unit-testable off-target.
#[doc(hidden)]
#[derive(Clone, Copy, Default)]
pub struct Caps {
    pub sse4: bool,
    pub avx1: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fp16: bool,
    pub bf16: bool,
    pub neon: bool,
    /// SVE vector length in bytes; `0` when absent (or policy-disabled).
    pub sve_vl: u16,
    pub simd128: bool,
    pub relaxed: bool,
}

/// The canonical tier for a `(host, type-combo)` pair — the policy that lets kernels
/// monomorphize only for tiers *distinct on the elements they use*:
/// - the fp16/bf16 towers are chosen only when the combo contains that half type (their other
///   elements are identical to plain AVX-512);
/// - AVX1 is chosen only for pure `f32`/`f64` combos (its integer/half lanes are emulated,
///   while SSE4's are native);
/// - SVE is chosen only for integer-free combos (its integer lanes are emulated; NEON's are
///   native).
#[doc(hidden)]
pub fn canonical_tier(caps: &Caps, combo: u8) -> u8 {
    if caps.avx512 {
        if combo & F16_BIT != 0 && caps.fp16 {
            return tier::AVX512FP16;
        }
        if combo & BF16_BIT != 0 && caps.bf16 {
            return tier::AVX512BF16;
        }
        return tier::AVX512;
    }
    if caps.avx2 {
        return tier::AVX2;
    }
    if caps.avx1 && combo & (INT_BITS | F16_BIT | BF16_BIT) == 0 {
        return tier::AVX1;
    }
    if caps.sse4 {
        return tier::SSE4;
    }
    if caps.sve_vl > 0 && combo & INT_BITS == 0 {
        return if caps.sve_vl >= 64 {
            tier::SVE64
        } else if caps.sve_vl >= 32 {
            tier::SVE32
        } else {
            tier::SVE16
        };
    }
    if caps.neon {
        return tier::NEON;
    }
    if caps.relaxed {
        return tier::RELAXED;
    }
    if caps.simd128 {
        return tier::SIMD128;
    }
    tier::SCALAR
}

fn host_caps() -> Caps {
    #[allow(unused_mut)]
    let mut caps = Caps::default();
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(all(feature = "std", not(static_dispatch)))]
        {
            caps.sse4 = std::arch::is_x86_feature_detected!("sse4.1");
            #[cfg(not(no_avx))]
            {
                caps.avx1 = std::arch::is_x86_feature_detected!("avx");
                caps.avx2 = std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma");
            }
            #[cfg(not(any(no_avx, no_avx512)))]
            {
                caps.avx512 = std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw");
                caps.fp16 = caps.avx512 && std::arch::is_x86_feature_detected!("avx512fp16");
                caps.bf16 = caps.avx512 && std::arch::is_x86_feature_detected!("avx512bf16");
            }
        }
        #[cfg(any(not(feature = "std"), static_dispatch))]
        {
            caps.sse4 = cfg!(target_feature = "sse4.1");
            caps.avx1 = cfg!(all(target_feature = "avx", not(no_avx)));
            caps.avx2 = cfg!(all(target_feature = "avx2", target_feature = "fma", not(no_avx)));
            caps.avx512 = cfg!(all(target_feature = "avx512f", not(any(no_avx, no_avx512))));
            caps.fp16 = caps.avx512 && cfg!(target_feature = "avx512fp16");
            caps.bf16 = caps.avx512 && cfg!(target_feature = "avx512bf16");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        caps.neon = true;
        #[cfg(all(not(target_vendor = "apple"), not(no_sve), not(neon_over_sve)))]
        {
            let have_sve = cfg!(target_feature = "sve")
                || {
                    #[cfg(all(feature = "std", not(static_dispatch)))]
                    {
                        std::arch::is_aarch64_feature_detected!("sve")
                    }
                    #[cfg(any(not(feature = "std"), static_dispatch))]
                    {
                        false
                    }
                };
            if have_sve {
                caps.sve_vl = crate::arch::sve2::vl_bytes() as u16;
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        caps.simd128 = cfg!(target_feature = "simd128");
        caps.relaxed = cfg!(target_feature = "relaxed-simd");
    }
    caps
}

/// The dispatch tier for a kernel's type-combo, resolved once per `(process, combo)` and cached
/// in a 64-slot atomic table (6 element bits → one `AtomicU8` per subset). The hot path is a
/// relaxed load + the caller's `match`; the first call per combo pays one feature-detection
/// pass. Generated `#[kernel]` wrappers pass their compile-time combo and match on the result
/// with arms pruned to the tiers this function can actually return for that combo — which is
/// what keeps a pure-`f32` kernel from monomorphizing for the fp16/bf16 towers.
#[doc(hidden)]
#[inline]
pub fn combo_tier(combo: u8) -> u8 {
    let combo = combo & 63;
    #[cfg(all(feature = "std", not(static_dispatch)))]
    {
        use core::sync::atomic::{AtomicU8, Ordering};
        static COMBO_TIERS: [AtomicU8; 64] = [const { AtomicU8::new(0) }; 64];
        let slot = &COMBO_TIERS[combo as usize];
        match slot.load(Ordering::Relaxed) {
            0 => {
                let t = canonical_tier(&host_caps(), combo);
                slot.store(t, Ordering::Relaxed);
                t
            }
            t => t,
        }
    }
    #[cfg(not(all(feature = "std", not(static_dispatch))))]
    {
        // Compile-time capabilities: folds to a constant per combo after inlining.
        canonical_tier(&host_caps(), combo)
    }
}

/// Run a kernel on an explicit backend with the standard unroll-factor resolution — the arm body
/// the `#[kernel]` combo-dispatch match uses.
#[doc(hidden)]
#[inline]
pub fn run_kernel_on<T: Scalar, K: Kernel<T>, S: BackendAll + Backend<T>>(
    kernel: K,
    backend: S,
) -> K::Output {
    UnrollSelect(kernel).run(Gang::new(backend))
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

/// Integer-element dispatch: the same runtime x86 tier cache and compile-time selection as the
/// float ladder, minus the tiers with no 32-bit integer backend (AVX1 — 256-bit integer ops are
/// AVX2 — and the scalable SVE/RVV tokens). aarch64 goes straight to NEON.
macro_rules! impl_simd_dispatch_int {
    ($ty:ty) => {
        impl SimdDispatch for $ty {
            #[inline]
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
                    use crate::backend::avx2::Avx2;
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
                        if Sse4::detect().is_some() { 4 } else { u8::MAX }
                    });
                    // SAFETY: each token is built only for the tier the matching `detect()`
                    // confirmed this run.
                    return match t {
                        #[cfg(not(any(no_avx, no_avx512)))]
                        1 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                        #[cfg(not(no_avx))]
                        2 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                        4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                        _ => kernel.run(Gang::new(ScalarBackend)),
                    };
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), static_dispatch),
                    not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                    target_feature = "avx2",
                    not(no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx2.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), static_dispatch),
                    not(all(target_feature = "avx512f", not(any(no_avx, no_avx512)))),
                    not(all(target_feature = "avx2", not(no_avx))),
                    target_feature = "sse4.1"
                ))]
                {
                    // SAFETY: target compiled with sse4.1.
                    let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(target_arch = "aarch64")]
                {
                    return kernel.run(Gang::new(crate::backend::neon::Neon::new()));
                }
                wasm_dispatch_tail!(kernel);
                kernel.run(Gang::new(ScalarBackend))
            }
        }
    };
}

impl_simd_dispatch_int!(u32);
impl_simd_dispatch_int!(i32);

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
