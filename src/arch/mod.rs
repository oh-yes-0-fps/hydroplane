//! Low-level SIMD primitives (raw `asm!`) and the ARM backend-selection policy.
//!
//! On x86_64, [`avx512fp16`] emits the AVX-512-FP16 `v*ph` instructions directly, so 32-wide `f16`
//! arithmetic builds on **stable** without the unstable `f16` primitive / `stdarch_x86_avx512_f16`
//! intrinsics — the same raw-`asm!` route the ARM modules below take.
//!
//! SVE is *scalable* (runtime vector length), and a scalable register can't live in a Rust struct
//! (see `SVE.md`), so each op works on a fixed-size **memory image** (`sve1::SveVec<C>`). One byte
//! width `C` = one vector length, so each SVE size is its own backend; `sve1`/`sve2` are the two
//! ISA versions, and SME (`sme1`/`sme2`) runs streaming-SVE + the ZA matrix engine on top of one.
//!
//! The low-level asm compiles on any aarch64 target but *runs* only where the feature exists —
//! notably base SVE does not exist on Apple silicon (SVE there is streaming-only, via SME).

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx512bf16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx512fp16;

#[cfg(target_arch = "aarch64")]
pub mod sve1;
#[cfg(target_arch = "aarch64")]
pub mod sve2;
// SME emits its instructions through raw `asm!` + an in-block `.arch_extension sme` directive, so
// it builds on **stable** (no `sme` target_feature / nightly), exactly like the SVE modules.
#[cfg(target_arch = "aarch64")]
pub mod sme1;
#[cfg(target_arch = "aarch64")]
pub mod sme2;

// RVV (RISC-V Vector) is *scalable* like SVE, so it takes the same memory-image route: one byte
// width `C` = one vector length. The asm enables the "V" extension per-block via `.option arch, +v`,
// so it builds on **stable** (no `v` target_feature). Base "V" covers f32/f64; the FP16/bf16 vector
// forms are separate Zvfh/Zvfbfmin extensions and are not handled here.
#[cfg(target_arch = "riscv64")]
pub mod rvv;

// Intel AMX: the x86 tile matrix engine (the counterpart to ARM's SME ZA engine). A bf16 tile GEMM
// fast path for `mma`; the tile mnemonics assemble on **stable** with no `amx-*` target_feature.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub mod amx;

/// Read a boolean `hw.optional.*` CPU feature flag via `sysctlbyname` (Apple's stable feature probe;
/// the std `is_aarch64_feature_detected!` macro and the Linux `HWCAP` bits don't exist here). Returns
/// `false` if the flag is absent or the query fails.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple", feature = "std"))]
pub(crate) fn apple_sysctl_flag(name: &core::ffi::CStr) -> bool {
    use core::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut val: i32 = 0;
    let mut len = core::mem::size_of::<i32>();
    let rc = unsafe {
        sysctlbyname(name.as_ptr(), (&mut val as *mut i32).cast(), &mut len, core::ptr::null_mut(), 0)
    };
    rc == 0 && val != 0
}

/// Which scalable backend (if any) the ARM dispatch should prefer, after applying the build-time
/// opt-outs and the Apple "NEON + Accelerate over SVE/SME" rule. Resolved against runtime feature
/// detection in [`select`]. Unlike x86 (a short AVX-512→AVX2→SSE4 cascade), ARM has many backends —
/// one per `(vector length, ISA version)` — so selection keys on the detected SVE/SME width.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArmPath {
    /// Fixed-width NEON (128-bit). Always available on aarch64; the floor and the Apple default.
    Neon,
    /// Streaming-SVE2 + the SME ZA matrix engine, at the streaming vector length (bytes).
    Sme { svl_bytes: usize, v2: bool },
    /// Base (non-streaming) SVE at the detected vector length (bytes); `v2` = SVE2.
    Sve { vl_bytes: usize, v2: bool },
}

/// Pick the ARM execution path from runtime detection and the build-time policy.
///
/// Policy (in order): never use SVE/SME on Apple (prefer NEON + Accelerate); `--cfg no_sve`
/// disables SVE *and* SME; `--cfg no_sme` disables SME only; `--cfg neon_over_sve` prefers NEON
/// even when SVE is present. Otherwise prefer the SME matrix engine, then base SVE, then NEON.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unreachable_code)]
pub fn select() -> ArmPath {
    // Apple silicon: base SVE doesn't exist (streaming-only), and NEON + Accelerate is the
    // sanctioned fast path — so never take the SVE/SME route here.
    #[cfg(target_vendor = "apple")]
    {
        return ArmPath::Neon;
    }

    // `no_sve` turns off the entire scalable stack (SVE and SME); `neon_over_sve` keeps the
    // scalable feature compiled but biases selection to NEON.
    #[cfg(any(no_sve, neon_over_sve))]
    {
        return ArmPath::Neon;
    }

    #[cfg(all(not(target_vendor = "apple"), not(no_sve), not(neon_over_sve), feature = "std"))]
    {
        // SME first (matrix engine + streaming SVE2), unless disabled. The std
        // `is_aarch64_feature_detected!` macro doesn't cover SME yet, so `sme1::is_supported`
        // reads `HWCAP2_SME` from the Linux aux vector (false on other OSes — see its docs).
        #[cfg(not(no_sme))]
        if sme1::is_supported() {
            return ArmPath::Sme {
                svl_bytes: sme1::streaming_vl_bytes(),
                v2: sme2::is_supported(),
            };
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            return ArmPath::Sve {
                vl_bytes: sve2::vl_bytes(),
                v2: std::arch::is_aarch64_feature_detected!("sve2"),
            };
        }
    }

    ArmPath::Neon
}
