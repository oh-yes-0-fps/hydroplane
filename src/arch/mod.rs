//! Low-level SIMD primitives (raw `asm!`) and the ARM backend-selection policy.
//! Scalable (SVE/RVV) registers can't live in Rust structs, so ops work on fixed-size memory
//! images: one byte width = one vector length = one backend.

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx512bf16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub mod avx512fp16;

#[cfg(target_arch = "aarch64")]
pub mod sve1;
#[cfg(target_arch = "aarch64")]
pub mod sve2;
// SME goes through raw `asm!` + an in-block `.arch_extension sme`, so no `sme` target_feature
// (still nightly-only) is needed and it builds on stable.
#[cfg(target_arch = "aarch64")]
pub mod sme1;
#[cfg(target_arch = "aarch64")]
pub mod sme2;

// RVV is scalable like SVE and takes the same memory-image route. `.option arch, +v` per block
// keeps it on stable (no `v` target_feature). Base "V" covers f32/f64 only; the Zvfh/Zvfbfmin
// half-precision forms are not handled here.
#[cfg(target_arch = "riscv64")]
pub mod rvv;

// Intel AMX tile matrix engine: bf16/f16 tile GEMM fast path for `mma`. Assembles on stable with
// no `amx-*` target_feature.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
pub mod amx;

/// Read a boolean `hw.optional.*` CPU feature flag via `sysctlbyname` (the only stable probe on
/// Apple; no `HWCAP`, and `is_aarch64_feature_detected!` doesn't cover these features). Returns
/// `false` if the flag is absent or the query fails.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
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

/// Which scalable backend (if any) the ARM dispatch should prefer, after the build-time opt-outs
/// and the Apple "NEON + Accelerate over SVE/SME" rule. Resolved by [`select`]. ARM has one
/// backend per `(vector length, ISA version)`, so selection keys on the detected SVE/SME width.
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
/// Policy (in order): never use SVE/SME on Apple (prefer NEON + Accelerate); `--cfg hp_no_sve`
/// disables SVE *and* SME; `--cfg hp_no_sme` disables SME only; `--cfg hp_neon_over_sve` prefers NEON
/// even when SVE is present. Otherwise prefer the SME matrix engine, then base SVE, then NEON.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unreachable_code)]
pub fn select() -> ArmPath {
    // Apple silicon has no base SVE (streaming-only) and NEON + Accelerate is the fast path.
    #[cfg(target_vendor = "apple")]
    {
        return ArmPath::Neon;
    }

    #[cfg(any(hp_no_sve, hp_neon_over_sve))]
    {
        return ArmPath::Neon;
    }

    #[cfg(all(not(target_vendor = "apple"), not(hp_no_sve), not(hp_neon_over_sve), feature = "std"))]
    {
        // `is_aarch64_feature_detected!` doesn't cover SME yet; `sme1::is_supported` reads
        // `HWCAP2_SME` from the Linux aux vector (false on other OSes).
        #[cfg(not(hp_no_sme))]
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
