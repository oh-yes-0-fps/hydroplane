//! Execution test for the SME ZA matrix engine (`hydroplane::arch::sme1`) against a scalar GEMM
//! reference. SME instructions are raw `asm!`, so they compile on any aarch64 target but only run
//! where SME exists; detected at runtime (macOS `sysctl`, Linux `HWCAP2`), skipped otherwise.
#![cfg(target_arch = "aarch64")]

use half::{bf16, f16};

/// Runtime SME probe. The crate's `sme1::is_supported` is Linux-only; Apple silicon exposes SME
/// via `sysctl hw.optional.arm.FEAT_SME`.
fn sme_present() -> bool {
    #[cfg(target_os = "macos")]
    {
        let mut v: i32 = 0;
        let mut len = core::mem::size_of::<i32>();
        let name = c"hw.optional.arm.FEAT_SME";
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const core::ffi::c_char,
                oldp: *mut core::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut core::ffi::c_void,
                newlen: usize,
            ) -> core::ffi::c_int;
        }
        unsafe {
            sysctlbyname(
                name.as_ptr(),
                &mut v as *mut i32 as *mut _,
                &mut len,
                core::ptr::null_mut(),
                0,
            ) == 0
                && v == 1
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        hydroplane::arch::sme1::is_supported()
    }
}

/// SME2 presence: macOS `sysctl hw.optional.arm.FEAT_SME2`, else the crate's probe.
fn sme2_present() -> bool {
    #[cfg(target_os = "macos")]
    {
        let mut v: i32 = 0;
        let mut len = core::mem::size_of::<i32>();
        let name = c"hw.optional.arm.FEAT_SME2";
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const core::ffi::c_char,
                oldp: *mut core::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut core::ffi::c_void,
                newlen: usize,
            ) -> core::ffi::c_int;
        }
        unsafe {
            sysctlbyname(name.as_ptr(), &mut v as *mut i32 as *mut _, &mut len, core::ptr::null_mut(), 0)
                == 0
                && v == 1
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        hydroplane::arch::sme2::is_supported()
    }
}

fn ref_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += s;
        }
    }
}

fn maxabs_err(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).fold(0.0f32, |m, (&g, &w)| m.max((g - w).abs()))
}

#[test]
fn sme_mma_f32() {
    if !sme_present() {
        eprintln!("SME absent — skipping sme_mma_f32");
        return;
    }
    // svl/4 ≥ 4 (128-bit floor); use a non-square tile that fits the smallest plausible ZA.
    const M: usize = 5;
    const N: usize = 6;
    const K: usize = 7;
    let a: Vec<f32> = (0..M * K).map(|i| i as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..K * N).map(|i| i as f32 * -0.25 + 1.0).collect();
    let mut c: Vec<f32> = (0..M * N).map(|i| i as f32 * 0.1).collect();
    let mut want = c.clone();
    ref_gemm(M, N, K, &a, &b, &mut want);
    unsafe {
        hydroplane::arch::sme1::mma_f32::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, c.as_mut_ptr(), N);
    }
    assert!(maxabs_err(&c, &want) < 1e-3, "f32: {c:?} vs {want:?}");
}

/// Drives `mma_f32_wide` for one (M,N,K) against the scalar reference.
fn check_wide<const M: usize, const N: usize, const K: usize>() {
    let a: Vec<f32> = (0..M * K).map(|i| (i as f32 % 11.0) * 0.5 - 2.0).collect();
    let b: Vec<f32> = (0..K * N).map(|i| (i as f32 % 7.0) * -0.25 + 1.0).collect();
    let mut c: Vec<f32> = (0..M * N).map(|i| (i as f32 % 5.0) * 0.1).collect();
    let mut want = c.clone();
    ref_gemm(M, N, K, &a, &b, &mut want);
    unsafe {
        hydroplane::arch::sme2::mma_f32_wide::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, c.as_mut_ptr(), N);
    }
    assert!(maxabs_err(&c, &want) < 1e-3, "wide {M}x{N}x{K}: {c:?} vs {want:?}");
}

#[test]
fn sme_mma_f32_wide() {
    if !sme2_present() {
        eprintln!("SME2 absent — skipping sme_mma_f32_wide");
        return;
    }
    // svl/4 = 16 on the M5: shapes straddle the tile boundary so all four za tiles are exercised,
    // plus a degenerate single-tile case (M,N ≤ 16).
    check_wide::<24, 20, 10>();
    check_wide::<32, 32, 8>();
    check_wide::<17, 31, 13>();
    check_wide::<10, 8, 6>();
}

fn check_wide_f64<const M: usize, const N: usize, const K: usize>() {
    let a: Vec<f64> = (0..M * K).map(|i| (i as f64 % 11.0) * 0.5 - 2.0).collect();
    let b: Vec<f64> = (0..K * N).map(|i| (i as f64 % 7.0) * -0.25 + 1.0).collect();
    let mut c: Vec<f64> = (0..M * N).map(|i| (i as f64 % 5.0) * 0.1).collect();
    let mut want = c.clone();
    for i in 0..M {
        for j in 0..N {
            let mut s = 0.0f64;
            for p in 0..K {
                s += a[i * K + p] * b[p * N + j];
            }
            want[i * N + j] += s;
        }
    }
    unsafe {
        hydroplane::arch::sme2::mma_f64_wide::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, c.as_mut_ptr(), N);
    }
    let err = c.iter().zip(&want).fold(0.0f64, |m, (&g, &w)| m.max((g - w).abs()));
    assert!(err < 1e-9, "wide f64 {M}x{N}x{K} err={err}");
}

fn check_wide_f16<const M: usize, const N: usize, const K: usize>() {
    let af: Vec<f32> = (0..M * K).map(|i| (i as f32 % 11.0) * 0.1 - 0.5).collect();
    let bf: Vec<f32> = (0..K * N).map(|i| (i as f32 % 7.0) * 0.1 - 0.3).collect();
    let a16: Vec<f16> = af.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = bf.iter().map(|&x| f16::from_f32(x)).collect();
    let mut c = vec![f16::ZERO; M * N];
    let mut want = vec![0.0f32; M * N];
    ref_gemm(
        M,
        N,
        K,
        &a16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &b16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &mut want,
    );
    unsafe {
        hydroplane::arch::sme2::mma_f16_wide::<M, N, K>(a16.as_ptr(), K, b16.as_ptr(), N, c.as_mut_ptr(), N);
    }
    let got: Vec<f32> = c.iter().map(|x| x.to_f32()).collect();
    assert!(maxabs_err(&got, &want) < 5e-2, "wide f16 {M}x{N}x{K}: {got:?} vs {want:?}");
}

#[test]
fn sme_mma_f16_wide() {
    if !sme2_present() {
        eprintln!("SME2 absent — skipping sme_mma_f16_wide");
        return;
    }
    // f16 1×2 grid: one .h tile is svl/2 = 32 (M ≤ 32), the N-split spans up to svl = 64. Straddle 32.
    check_wide_f16::<20, 48, 10>();
    check_wide_f16::<32, 64, 8>();
    check_wide_f16::<12, 33, 9>();
    check_wide_f16::<8, 20, 6>();
}

fn check_wide_bf16<const M: usize, const N: usize, const K: usize>() {
    let af: Vec<f32> = (0..M * K).map(|i| (i as f32 % 11.0) * 0.25 - 1.0).collect();
    let bf: Vec<f32> = (0..K * N).map(|i| (i as f32 % 7.0) * 0.5 - 1.5).collect();
    let a: Vec<bf16> = af.iter().map(|&x| bf16::from_f32(x)).collect();
    let b: Vec<bf16> = bf.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut c = vec![0.0f32; M * N];
    let mut want = vec![0.0f32; M * N];
    ref_gemm(
        M,
        N,
        K,
        &a.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &b.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &mut want,
    );
    unsafe {
        hydroplane::arch::sme2::mma_bf16_wide::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, c.as_mut_ptr(), N);
    }
    // BFMOPA accumulates in f32, so the error is just bf16 input rounding.
    assert!(maxabs_err(&c, &want) < 1e-1, "wide bf16 {M}x{N}x{K}: {c:?} vs {want:?}");
}

#[test]
fn sme_mma_bf16_wide() {
    if !sme2_present() {
        eprintln!("SME2 absent — skipping sme_mma_bf16_wide");
        return;
    }
    // bf16 → 2×2 .s grid (q=svl/4=16), so up to svl/2=32; odd K exercises the pair zero-pad.
    check_wide_bf16::<24, 20, 9>();
    check_wide_bf16::<32, 32, 8>();
    check_wide_bf16::<17, 31, 13>();
    check_wide_bf16::<10, 8, 5>();
}

#[test]
fn sme_mma_f64_wide() {
    if !sme2_present() {
        eprintln!("SME2 absent — skipping sme_mma_f64_wide");
        return;
    }
    // f64 ZA tile is svl/8 = 8 on the M5; the 2×2 grid spans up to svl/4 = 16. Straddle 8.
    check_wide_f64::<12, 10, 9>();
    check_wide_f64::<16, 16, 7>();
    check_wide_f64::<5, 13, 11>();
    check_wide_f64::<6, 4, 5>();
}

#[test]
fn sme_mma_f64() {
    if !sme_present() {
        eprintln!("SME absent — skipping sme_mma_f64");
        return;
    }
    // f64 ZA tile is svl/8 wide (needs FEAT_SME_F64F64); keep M,N ≤ 4 for the 256-bit-svl floor.
    const M: usize = 4;
    const N: usize = 3;
    const K: usize = 5;
    let a: Vec<f64> = (0..M * K).map(|i| i as f64 * 0.5 - 3.0).collect();
    let b: Vec<f64> = (0..K * N).map(|i| i as f64 * -0.25 + 1.0).collect();
    let mut c: Vec<f64> = (0..M * N).map(|i| i as f64 * 0.1).collect();
    let mut want = c.clone();
    for i in 0..M {
        for j in 0..N {
            let mut s = 0.0f64;
            for p in 0..K {
                s += a[i * K + p] * b[p * N + j];
            }
            want[i * N + j] += s;
        }
    }
    unsafe {
        hydroplane::arch::sme1::mma_f64::<M, N, K>(a.as_ptr(), K, b.as_ptr(), N, c.as_mut_ptr(), N);
    }
    let err = c.iter().zip(&want).fold(0.0f64, |m, (&g, &w)| m.max((g - w).abs()));
    assert!(err < 1e-9, "f64 err={err}: {c:?} vs {want:?}");
}

#[test]
fn sme_mma_half() {
    if !sme_present() {
        eprintln!("SME absent — skipping sme_mma_half");
        return;
    }
    const M: usize = 4;
    const N: usize = 4;
    const K: usize = 5; // odd → exercises the BFMOPA last-pair zero-padding
    let af: Vec<f32> = (0..M * K).map(|i| i as f32 * 0.25 - 1.5).collect();
    let bf: Vec<f32> = (0..K * N).map(|i| i as f32 * 0.5 - 2.0).collect();

    let a16: Vec<f16> = af.iter().map(|&x| f16::from_f32(x)).collect();
    let b16: Vec<f16> = bf.iter().map(|&x| f16::from_f32(x)).collect();
    // Native f16 accumulate (za.h): compare widened, with f16-precision tolerance.
    let mut c16 = vec![f16::ZERO; M * N];
    let mut want16 = vec![0.0f32; M * N];
    ref_gemm(
        M,
        N,
        K,
        &a16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &b16.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &mut want16,
    );
    unsafe {
        hydroplane::arch::sme1::mma_f16::<M, N, K>(a16.as_ptr(), K, b16.as_ptr(), N, c16.as_mut_ptr(), N);
    }
    let got16: Vec<f32> = c16.iter().map(|x| x.to_f32()).collect();
    assert!(maxabs_err(&got16, &want16) < 5e-2, "f16: {got16:?} vs {want16:?}");

    let ab: Vec<bf16> = af.iter().map(|&x| bf16::from_f32(x)).collect();
    let bb: Vec<bf16> = bf.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut cb = vec![0.0f32; M * N];
    let mut wantb = vec![0.0f32; M * N];
    ref_gemm(
        M,
        N,
        K,
        &ab.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &bb.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &mut wantb,
    );
    unsafe {
        hydroplane::arch::sme1::mma_bf16::<M, N, K>(ab.as_ptr(), K, bb.as_ptr(), N, cb.as_mut_ptr(), N);
    }
    assert!(maxabs_err(&cb, &wantb) < 1e-1, "bf16: {cb:?} vs {wantb:?}");
}
