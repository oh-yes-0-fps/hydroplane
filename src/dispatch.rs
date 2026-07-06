//! Entry points that pick a [`Backend`] and run a generic [`Kernel`]: runtime CPU detection
//! chooses the widest implemented backend per scalar (build-time `target_feature`s on no-std or
//! under `--cfg hp_static_dispatch`), with [`ScalarBackend`] as the universal fallback.

use crate::backend::{Backend, BackendAll, ScalarBackend};
use crate::scalar::Scalar;
use crate::varying::Gang;

/// The aarch64 tail of a dispatch, shared by [`SimdDispatch`] and [`MatrixDispatch`]. Non-Apple
/// aarch64 with SVE takes the widest [`Sve`](crate::backend) token the hardware VL covers;
/// everything else (Apple, `--cfg hp_no_sve`, `--cfg hp_neon_over_sve`, no SVE) takes `$fallback`, the
/// per-scalar floor (`Neon` where it implements the scalar, `ScalarBackend` for `f16`). Expands
/// to nothing off aarch64.
macro_rules! aarch64_dispatch_tail {
    ($kernel:expr, $fallback:expr) => {{
        // Build guarantees SVE, so no detection; only the scalable vector length is read at runtime.
        #[cfg(all(
            target_arch = "aarch64",
            not(target_vendor = "apple"),
            target_feature = "sve",
            not(hp_no_sve),
            not(hp_neon_over_sve)
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
            not(hp_static_dispatch),
            not(target_feature = "sve"),
            not(hp_no_sve),
            not(hp_neon_over_sve)
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
/// comes from `target_feature`: `relaxed-simd` takes [`RelaxedSimd`](crate::backend), plain
/// `simd128` takes [`Simd128`](crate::backend), neither falls through to the caller's scalar
/// floor. Expands to nothing off wasm32.
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

/// The riscv64 tail, shared by every dispatch. RVV is scalable, so like the SVE tail it picks the
/// widest [`Rvv`](crate::backend) token the detected `VLENB` covers. RISC-V has no other SIMD
/// backend here: returns only when "V" is present, else the caller falls through to its scalar
/// floor. Expands to nothing off riscv64.
macro_rules! riscv_dispatch_tail {
    ($kernel:expr) => {{
        // Build guarantees "V", so no detection; only `VLENB` is read at runtime. Works in no-std.
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
            not(hp_static_dispatch),
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

/// The 32-bit-ARM (armv7) tail. NEON there is f32-only, so only the `f32` dispatch invokes this;
/// other scalars fall through to scalar. Returns only when NEON is present (compile-time
/// `target_feature`, or `Neon::detect()` via HWCAP on std). Expands to nothing off arm.
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
            not(hp_static_dispatch),
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

/// A unit of work generic over the execution backend. Implement this once; `hydroplane` runs it
/// on the backend it selects, handing [`run`](Kernel::run) a [`Gang`] context to build varying
/// values through (`splat`, `load`, …). The raw [`Backend`] token is reachable via
/// [`Gang::backend`].
pub trait Kernel<T: Scalar> {
    type Output;

    /// Ceiling on the ILP unroll factor for this kernel, `min`'d against the factor the runtime
    /// sweep resolves for the core. Defaults to [`MAX_UNROLL`](crate::MAX_UNROLL) (no cap). The
    /// `#[kernel]` macro overrides it from build-time MIR analysis so a register-heavy kernel is
    /// not unrolled past the point where it spills. Only the runtime dispatch path reads it; the
    /// build-resolved static path bakes `K` as a const and cannot be per-kernel-capped on stable.
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
/// The kernel is wrapped in `UnrollSelect`: once the ISA backend is picked, the unroll factor `K`
/// is resolved for this core and baked into an `Unroll<S, K>` backend, so every reduction inside
/// the kernel sees `K` as a compile-time constant.
#[inline]
pub fn dispatch<T: SimdDispatch, K: Kernel<T>>(kernel: K) -> K::Output {
    T::dispatch(UnrollSelect(kernel))
}

/// Resolves the unroll factor on the dispatched backend, then re-runs the wrapped kernel on
/// [`Unroll<S, K>`](crate::backend::Unroll) so `K` is a constant inside it without threading `K`
/// through [`Gang`] or [`Kernel`]. Each match arm monomorphizes the real kernel for that `K`.
struct UnrollSelect<K>(K);

impl<T: Scalar, K: Kernel<T>> Kernel<T> for UnrollSelect<K> {
    type Output = K::Output;

    #[inline]
    #[cfg(all(not(hp_no_ilp), not(target_arch = "spirv"), not(hp_resolved_unroll)))]
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

    /// Build-resolved (`hp_static_dispatch` + pinned cpu): `K` is the constant `build.rs` baked in,
    /// so there is no detection sweep and the kernel monomorphizes for exactly one `Unroll<S, K>`.
    #[inline]
    #[cfg(hp_resolved_unroll)]
    fn run<S: BackendAll + Backend<T>>(self, g: Gang<S>) -> Self::Output {
        use crate::backend::Unroll;
        self.0
            .run(Gang::new(Unroll::<S, { crate::varying::STATIC_UNROLL }>(g.backend())))
    }

    /// ILP compiled out: no factor to resolve, run the kernel on the raw backend.
    #[inline]
    #[cfg(any(hp_no_ilp, target_arch = "spirv"))]
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
/// impls encode the best unit that tier guarantees. The fp16/bf16 tiers serve f32/f64/ints
/// through plain AVX-512, so a combo without halves canonicalizes down to `AVX512` and never
/// monomorphizes for them.
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

/// The canonical tier for a `(host, type-combo)` pair. Kernels monomorphize only for tiers
/// distinct on the elements they use:
/// - fp16/bf16 towers only when the combo contains that half type (their other elements are
///   identical to plain AVX-512);
/// - AVX1 only for pure `f32`/`f64` combos (its integer/half lanes are emulated, SSE4's are native);
/// - SVE only for integer-free combos (its integer lanes are emulated, NEON's are native).
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
        #[cfg(all(feature = "std", not(hp_static_dispatch)))]
        {
            caps.sse4 = std::arch::is_x86_feature_detected!("sse4.1");
            #[cfg(not(hp_no_avx))]
            {
                caps.avx1 = std::arch::is_x86_feature_detected!("avx");
                caps.avx2 = std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma");
            }
            #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
            {
                caps.avx512 = std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512bw");
                caps.fp16 = caps.avx512 && std::arch::is_x86_feature_detected!("avx512fp16");
                caps.bf16 = caps.avx512 && std::arch::is_x86_feature_detected!("avx512bf16");
            }
        }
        #[cfg(any(not(feature = "std"), hp_static_dispatch))]
        {
            caps.sse4 = cfg!(target_feature = "sse4.1");
            caps.avx1 = cfg!(all(target_feature = "avx", not(hp_no_avx)));
            caps.avx2 = cfg!(all(target_feature = "avx2", target_feature = "fma", not(hp_no_avx)));
            caps.avx512 = cfg!(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512))));
            caps.fp16 = caps.avx512 && cfg!(target_feature = "avx512fp16");
            caps.bf16 = caps.avx512 && cfg!(target_feature = "avx512bf16");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        caps.neon = true;
        #[cfg(all(not(target_vendor = "apple"), not(hp_no_sve), not(hp_neon_over_sve)))]
        {
            let have_sve = cfg!(target_feature = "sve")
                || {
                    #[cfg(all(feature = "std", not(hp_static_dispatch)))]
                    {
                        std::arch::is_aarch64_feature_detected!("sve")
                    }
                    #[cfg(any(not(feature = "std"), hp_static_dispatch))]
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
/// in a 64-slot atomic table. The hot path is a relaxed load plus the caller's `match`; the first
/// call per combo pays one feature-detection pass. `#[kernel]` wrappers match on the result with
/// arms pruned to the tiers reachable for their combo, which keeps a pure-`f32` kernel from
/// monomorphizing for the fp16/bf16 towers.
#[doc(hidden)]
#[inline]
pub fn combo_tier(combo: u8) -> u8 {
    let combo = combo & 63;
    #[cfg(all(feature = "std", not(hp_static_dispatch)))]
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
    #[cfg(not(all(feature = "std", not(hp_static_dispatch))))]
    {
        // Compile-time capabilities: folds to a constant per combo after inlining.
        canonical_tier(&host_caps(), combo)
    }
}

/// Run a kernel on an explicit backend with the standard unroll-factor resolution; the arm body
/// the `#[kernel]` combo-dispatch match uses.
#[doc(hidden)]
#[inline]
pub fn run_kernel_on<T: Scalar, K: Kernel<T>, S: BackendAll + Backend<T>>(
    kernel: K,
    backend: S,
) -> K::Output {
    UnrollSelect(kernel).run(Gang::new(backend))
}

/// Resolve-once cache for the x86 runtime backend tier: the warm path is one relaxed load plus a
/// `match`, not a fresh `is_x86_feature_detected!` ladder per call. `0` means unresolved;
/// `resolve` never returns `0`. Resolution is idempotent, so racing threads storing the same
/// value is harmless.
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "std", not(hp_static_dispatch)))]
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
            // When the build statically pins the widest ISA, that branch returns and the rest is dead.
            #[allow(unreachable_code)]
            fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    target_feature = "avx512f",
                    not(any(hp_no_avx, hp_no_avx512))
                ))]
                {
                    // SAFETY: target compiled with avx512f.
                    let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                // Runtime path: the tier is resolved once and cached in a process-global atomic,
                // so each call is a load + match rather than a fresh feature probe.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    feature = "std",
                    not(hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512))))
                ))]
                {
                    use crate::backend::sse4::Sse4;
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    use crate::backend::avx512::Avx512;
                    #[cfg(not(hp_no_avx))]
                    use crate::backend::{avx1::Avx1, avx2::Avx2};
                    static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                    let t = cached_tier(&TIER, || {
                        #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                        if Avx512::detect().is_some() {
                            return 1;
                        }
                        #[cfg(not(hp_no_avx))]
                        if Avx2::detect().is_some() {
                            return 2;
                        }
                        #[cfg(not(hp_no_avx))]
                        if Avx1::detect().is_some() {
                            return 3;
                        }
                        if Sse4::detect().is_some() { 4 } else { u8::MAX }
                    });
                    // SAFETY: each token is built only for the tier the matching `detect()` confirmed this run.
                    return match t {
                        #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                        1 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                        #[cfg(not(hp_no_avx))]
                        2 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                        #[cfg(not(hp_no_avx))]
                        3 => kernel.run(Gang::new(unsafe { Avx1::new_unchecked() })),
                        4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                        _ => kernel.run(Gang::new(ScalarBackend)),
                    };
                }
                // Compile-time selection: no-std, or `hp_static_dispatch` on std. The widest
                // `target_feature`-guaranteed tier that survives the `hp_no_avx*` cfgs.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    target_feature = "avx2",
                    target_feature = "fma",
                    not(hp_no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx2+fma.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                // AVX (no AVX2/FMA): 256-bit floats with an unfused `fma`.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    not(all(target_feature = "avx2", target_feature = "fma", not(hp_no_avx))),
                    target_feature = "avx",
                    not(hp_no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx.
                    let b = unsafe { crate::backend::avx1::Avx1::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    not(all(target_feature = "avx2", target_feature = "fma", not(hp_no_avx))),
                    not(all(target_feature = "avx", not(hp_no_avx))),
                    target_feature = "sse4.1"
                ))]
                {
                    // SAFETY: target compiled with sse4.1.
                    let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
                riscv_dispatch_tail!(kernel);
                $( $arm_tail!(kernel); )?
                wasm_dispatch_tail!(kernel);
                kernel.run(Gang::new(ScalarBackend))
            }
        }
    };
}

impl_simd_dispatch_x86!(f32, arm_dispatch_tail);
impl_simd_dispatch_x86!(f64);

/// Integer-element dispatch: the same x86 ladder as floats, minus the tiers with no 32-bit
/// integer backend (AVX1, whose 256-bit integer ops are AVX2, and the scalable SVE/RVV tokens).
/// aarch64 goes straight to NEON.
macro_rules! impl_simd_dispatch_int {
    ($ty:ty) => {
        impl SimdDispatch for $ty {
            #[inline]
            #[allow(unreachable_code)]
            fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    target_feature = "avx512f",
                    not(any(hp_no_avx, hp_no_avx512))
                ))]
                {
                    // SAFETY: target compiled with avx512f.
                    let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    feature = "std",
                    not(hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512))))
                ))]
                {
                    use crate::backend::sse4::Sse4;
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    use crate::backend::avx512::Avx512;
                    #[cfg(not(hp_no_avx))]
                    use crate::backend::avx2::Avx2;
                    static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                    let t = cached_tier(&TIER, || {
                        #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                        if Avx512::detect().is_some() {
                            return 1;
                        }
                        #[cfg(not(hp_no_avx))]
                        if Avx2::detect().is_some() {
                            return 2;
                        }
                        if Sse4::detect().is_some() { 4 } else { u8::MAX }
                    });
                    // SAFETY: each token is built only for the tier the matching `detect()` confirmed this run.
                    return match t {
                        #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                        1 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                        #[cfg(not(hp_no_avx))]
                        2 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                        4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                        _ => kernel.run(Gang::new(ScalarBackend)),
                    };
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    target_feature = "avx2",
                    not(hp_no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx2.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    not(all(target_feature = "avx2", not(hp_no_avx))),
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

// f16/bf16 dispatch: native half tiers where the hardware has them, widen paths elsewhere.
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
                not(any(hp_no_avx, hp_no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512fp16.
                let b = unsafe { crate::backend::avx512fp16::Avx512Fp16::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Runtime detection (cached): native FP16 if present, else the widen paths.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(hp_static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(hp_no_avx, hp_no_avx512))))
            ))]
            {
                #[cfg(not(hp_no_avx))]
                use crate::backend::avx2::Avx2;
                use crate::backend::sse4::Sse4;
                #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                use crate::backend::{avx512::Avx512, avx512fp16::Avx512Fp16};
                static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                let t = super::cached_tier(&TIER, || {
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    if Avx512Fp16::detect().is_some() {
                        return 1;
                    }
                    // Plain AVX-512 (no FP16): 16-wide f32 widen via vcvtph2ps.
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    if Avx512::detect().is_some() {
                        return 2;
                    }
                    #[cfg(not(hp_no_avx))]
                    if Avx2::detect().is_some() {
                        return 3;
                    }
                    if Sse4::detect().is_some() {
                        return 4;
                    }
                    u8::MAX
                });
                // SAFETY: each token is built only for the tier resolved via the matching `detect()`.
                return match t {
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    1 => kernel.run(Gang::new(unsafe { Avx512Fp16::new_unchecked() })),
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    2 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                    #[cfg(not(hp_no_avx))]
                    3 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                    4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                    _ => kernel.run(Gang::new(ScalarBackend)),
                };
            }
            // Compile-time AVX-512 (no FP16) f16 widen, preferred over the 8-wide AVX2 path below.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(hp_no_avx, hp_no_avx512)))),
                target_feature = "avx512f",
                not(any(hp_no_avx, hp_no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512f.
                let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(hp_no_avx, hp_no_avx512)))),
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "f16c",
                not(hp_no_avx)
            ))]
            {
                // SAFETY: target compiled with avx2+fma+f16c.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512fp16", not(any(hp_no_avx, hp_no_avx512)))),
                not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                not(all(target_feature = "avx2", target_feature = "fma", target_feature = "f16c", not(hp_no_avx))),
                target_feature = "sse4.1"
            ))]
            {
                // SAFETY: target compiled with sse4.1.
                let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // aarch64: native NEON FEAT_FP16 is 8-wide f16, the Apple-silicon path.
            #[cfg(all(target_arch = "aarch64", target_feature = "fp16"))]
            {
                return kernel.run(Gang::new(crate::backend::neon::Neon::new()));
            }
            #[cfg(all(
                target_arch = "aarch64",
                feature = "std",
                not(hp_static_dispatch),
                not(target_feature = "fp16")
            ))]
            {
                if std::arch::is_aarch64_feature_detected!("fp16") {
                    return kernel.run(Gang::new(crate::backend::neon::Neon::new()));
                }
            }
            // No NEON f16 (pre-ARMv8.2): non-Apple SVE has native f16, else scalar.
            super::aarch64_dispatch_tail!(kernel, ScalarBackend);
            super::wasm_dispatch_tail!(kernel);
            kernel.run(Gang::new(ScalarBackend))
        }
    }
    impl SimdDispatch for bf16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch<K: Kernel<Self>>(kernel: K) -> K::Output {
            // bf16 is 16-bit storage with f32 compute everywhere except native AVX-512-BF16.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                target_feature = "avx512bf16",
                not(any(hp_no_avx, hp_no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512bf16.
                let b = unsafe { crate::backend::avx512bf16::Avx512Bf16::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            // Runtime detection (cached): native AVX-512-BF16 first, then the widen paths.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(hp_static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(hp_no_avx, hp_no_avx512))))
            ))]
            {
                #[cfg(not(hp_no_avx))]
                use crate::backend::avx2::Avx2;
                use crate::backend::sse4::Sse4;
                #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                use crate::backend::{avx512::Avx512, avx512bf16::Avx512Bf16};
                static TIER: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
                let t = super::cached_tier(&TIER, || {
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    if Avx512Bf16::detect().is_some() {
                        return 1;
                    }
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    if Avx512::detect().is_some() {
                        return 2;
                    }
                    #[cfg(not(hp_no_avx))]
                    if Avx2::detect().is_some() {
                        return 3;
                    }
                    if Sse4::detect().is_some() {
                        return 4;
                    }
                    u8::MAX
                });
                // SAFETY: each token is built only for the tier resolved via the matching `detect()`.
                return match t {
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    1 => kernel.run(Gang::new(unsafe { Avx512Bf16::new_unchecked() })),
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    2 => kernel.run(Gang::new(unsafe { Avx512::new_unchecked() })),
                    #[cfg(not(hp_no_avx))]
                    3 => kernel.run(Gang::new(unsafe { Avx2::new_unchecked() })),
                    4 => kernel.run(Gang::new(unsafe { Sse4::new_unchecked() })),
                    _ => kernel.run(Gang::new(ScalarBackend)),
                };
            }
            // Compile-time bf16 widen: AVX-512 if the build guarantees avx512f, else AVX2 below.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(hp_no_avx, hp_no_avx512)))),
                target_feature = "avx512f",
                not(any(hp_no_avx, hp_no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512f.
                let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(hp_no_avx, hp_no_avx512)))),
                not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                target_feature = "avx2",
                target_feature = "fma",
                not(hp_no_avx)
            ))]
            {
                // SAFETY: target compiled with avx2+fma.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512bf16", not(any(hp_no_avx, hp_no_avx512)))),
                not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                not(all(target_feature = "avx2", target_feature = "fma", not(hp_no_avx))),
                target_feature = "sse4.1"
            ))]
            {
                // SAFETY: target compiled with sse4.1.
                let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            super::aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
            super::wasm_dispatch_tail!(kernel);
            kernel.run(Gang::new(ScalarBackend))
        }
    }
}
