//! SME (v1): the ZA matrix engine (`FMOPA` outer-product GEMM) via raw `asm!`. Each block toggles
//! `.arch_extension sme`, so it builds on stable with no `sme` target_feature (still unstable),
//! and every op runs inside its own `SMSTART`/`SMSTOP` streaming-mode session.
#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::needless_range_loop
)]

use core::arch::asm;
use half::{bf16, f16};

/// Read the SME streaming vector length in bytes via `RDSVL`, which is legal outside streaming
/// mode.
///
/// # Safety
/// The CPU must implement SME — guard with [`is_supported`].
pub unsafe fn streaming_vl_bytes_raw() -> usize {
    let r: usize;
    asm!(
        ".arch_extension sme",
        "rdsvl {r}, #1",
        ".arch_extension nosme",
        r = out(reg) r,
        options(pure, nomem, nostack),
    );
    r
}

/// SME streaming vector length in bytes. Only valid where SME is present (the caller detects it).
#[inline]
pub fn streaming_vl_bytes() -> usize {
    unsafe { streaming_vl_bytes_raw() }
}

/// Whether the running CPU implements SME. `is_aarch64_feature_detected!` doesn't cover SME on
/// stable, so this probes the OS: `HWCAP2_SME` from the ELF aux vector on Linux, the
/// `hw.optional.arm.FEAT_SME` sysctl on Apple; other OSes return `false`. Capability only:
/// [`super::select`] still routes Apple to NEON + Accelerate.
#[cfg(feature = "std")]
pub fn is_supported() -> bool {
    // Cached: the Apple probe is a syscall and this runs on every matrix `mma`. The race is
    // benign (every thread computes the same constant), so Relaxed is enough.
    static CACHE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
    match CACHE.load(core::sync::atomic::Ordering::Relaxed) {
        0 => {
            let v = is_supported_uncached();
            CACHE.store(1 + v as u8, core::sync::atomic::Ordering::Relaxed);
            v
        }
        c => c == 2,
    }
}

fn is_supported_uncached() -> bool {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn getauxval(ty: core::ffi::c_ulong) -> core::ffi::c_ulong;
        }
        const AT_HWCAP2: core::ffi::c_ulong = 26;
        const HWCAP2_SME: core::ffi::c_ulong = 1 << 23;
        unsafe { getauxval(AT_HWCAP2) & HWCAP2_SME != 0 }
    }
    #[cfg(target_vendor = "apple")]
    {
        super::apple_sysctl_flag(c"hw.optional.arm.FEAT_SME")
    }
    #[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
    {
        false
    }
}

/// FMOPA GEMM core for one element type. `at` is `K×M` column-major (row `k` of `at` is column `k`
/// of A) so each `k`-step loads an A-column with one contiguous `ld1`; `b` is `K×N` row-major
/// (row stride `ldb`); `c` is the `M×N` in/out accumulator (row stride `ldc`). ZA lanes outside
/// `M×N` are never read, so no `zero za` is needed.
macro_rules! fmopa_core {
    ($core:ident, $t:ty, $sz:expr, $e:literal, $ld:literal, $st:literal, $open:literal) => {
        #[inline]
        unsafe fn $core<const M: usize, const N: usize, const K: usize>(
            at: *const $t,
            b: *const $t,
            ldb: usize,
            c: *mut $t,
            ldc: usize,
        ) {
            asm!(
                $open,                                       // enable SME (+ sme-f64f64 for .d)
                "smstart",                                   // enter streaming mode + enable ZA
                concat!("whilelt p0.", $e, ", xzr, {m}"),    // p0 = first M lanes (rows / Zn)
                concat!("whilelt p1.", $e, ", xzr, {n}"),    // p1 = first N lanes (cols / Zm)
                // preload C into ZA, one horizontal slice (= row) per i
                "mov {i}, xzr",
                "20:",
                "cmp {i}, {m}",
                "b.hs 21f",
                "madd {addr}, {i}, {ldc_b}, {c}",
                concat!($ld, " {{z0.", $e, "}}, p1/z, [{addr}]"),
                "mov w12, {i:w}",
                concat!("mova za0h.", $e, "[w12, 0], p1/m, z0.", $e),
                "add {i}, {i}, #1",
                "b 20b",
                "21:",
                // accumulate K rank-1 updates: ZA += At[k] (col) outer B[k,:] (row)
                "mov {k}, xzr",
                "22:",
                "cmp {k}, {kk}",
                "b.hs 23f",
                "madd {addr}, {k}, {m_b}, {at}",             // column k of A (contiguous M)
                concat!($ld, " {{z0.", $e, "}}, p0/z, [{addr}]"),
                "madd {addr}, {k}, {ldb_b}, {b}",            // row k of B (contiguous N)
                concat!($ld, " {{z1.", $e, "}}, p1/z, [{addr}]"),
                concat!("fmopa za0.", $e, ", p0/m, p1/m, z0.", $e, ", z1.", $e),
                "add {k}, {k}, #1",
                "b 22b",
                "23:",
                // store ZA back to C, one horizontal slice per i
                "mov {i}, xzr",
                "24:",
                "cmp {i}, {m}",
                "b.hs 25f",
                "mov w12, {i:w}",
                concat!("mova z0.", $e, ", p1/m, za0h.", $e, "[w12, 0]"),
                "madd {addr}, {i}, {ldc_b}, {c}",
                concat!($st, " {{z0.", $e, "}}, p1, [{addr}]"),
                "add {i}, {i}, #1",
                "b 24b",
                "25:",
                "smstop",
                ".arch_extension nosme",
                m = in(reg) M,
                n = in(reg) N,
                kk = in(reg) K,
                at = in(reg) at,
                b = in(reg) b,
                c = in(reg) c,
                ldc_b = in(reg) ldc * $sz,
                ldb_b = in(reg) ldb * $sz,
                m_b = in(reg) M * $sz,
                i = out(reg) _,
                k = out(reg) _,
                addr = out(reg) _,
                out("x12") _,
                // SMSTART/SMSTOP zero the entire SVE register file (Z0-Z31, P0-P15), so every
                // SVE/NEON register the compiler might hold live across this block is clobbered.
                out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
                out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
                out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
                out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
                out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
                out("z30") _, out("z31") _,
                out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
                out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
                out("p12") _, out("p13") _, out("p14") _, out("p15") _,
                options(nostack),
            );
        }
    };
}

fmopa_core!(fmopa_f32, f32, 4, "s", "ld1w", "st1w", ".arch_extension sme");
fmopa_core!(fmopa_f64, f64, 8, "d", "ld1d", "st1d", ".arch_extension sme\n.arch_extension sme-f64f64");
fmopa_core!(fmopa_f16, f16, 2, "h", "ld1h", "st1h", ".arch_extension sme\n.arch_extension sme-f16f16");

/// `D = C + A·B` for `f32` tiles via the ZA `FMOPA` engine. `a` is `M×K` (row stride `lda`), `b` is
/// `K×N` (row stride `ldb`), `c` is the `M×N` in/out accumulator (row stride `ldc`). `M, N ≤ svl/4`.
#[inline]
pub unsafe fn mma_f32<const M: usize, const N: usize, const K: usize>(
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    // Pack A column-major so each FMOPA step reads its A-column with one contiguous load.
    let mut at = [[0.0f32; M]; K];
    for m in 0..M {
        let row = a.add(m * lda);
        for k in 0..K {
            at[k][m] = *row.add(k);
        }
    }
    fmopa_f32::<M, N, K>(at.as_ptr().cast(), b, ldb, c, ldc);
}

/// `D = C + A·B` for `f64` tiles (needs `FEAT_SME_F64F64`). `M, N ≤ svl/8`.
#[inline]
pub unsafe fn mma_f64<const M: usize, const N: usize, const K: usize>(
    a: *const f64,
    lda: usize,
    b: *const f64,
    ldb: usize,
    c: *mut f64,
    ldc: usize,
) {
    let mut at = [[0.0f64; M]; K];
    for m in 0..M {
        let row = a.add(m * lda);
        for k in 0..K {
            at[k][m] = *row.add(k);
        }
    }
    fmopa_f64::<M, N, K>(at.as_ptr().cast(), b, ldb, c, ldc);
}

/// `D = C + A·B` for `f16` tiles with native f16 accumulation (FEAT_SME_F16F16) via the `za.h`
/// engine. Intermediates round at f16 precision, so results do not match the f32-accumulate
/// backends. `M, N ≤ svl/2`. For an f32 accumulator, widen and call [`mma_f32`].
#[inline]
pub unsafe fn mma_f16<const M: usize, const N: usize, const K: usize>(
    a: *const f16,
    lda: usize,
    b: *const f16,
    ldb: usize,
    c: *mut f16,
    ldc: usize,
) {
    let mut at = [[f16::ZERO; M]; K];
    for m in 0..M {
        let row = a.add(m * lda);
        for k in 0..K {
            at[k][m] = *row.add(k);
        }
    }
    fmopa_f16::<M, N, K>(at.as_ptr().cast(), b, ldb, c, ldc);
}

/// `D = C + A·B` for `bf16` tiles with an `f32` accumulator, via the native `BFMOPA` widening
/// outer product. Each instruction folds a pair of `k`-steps
/// (`ZA[i,j] += Zn[2i]·Zm[2j] + Zn[2i+1]·Zm[2j+1]` in f32), so `⌈K/2⌉` matrix ops. A/B are packed
/// into the `BFMOPA` pair layout (row `i`'s two `k`-neighbours adjacent; odd `K` zero-pads the
/// last pair). `M, N ≤ svl/4`.
#[inline]
pub unsafe fn mma_bf16<const M: usize, const N: usize, const K: usize>(
    a: *const bf16,
    lda: usize,
    b: *const bf16,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) {
    let pairs = K.div_ceil(2);
    // apack[p][i] = [A[i][2p], A[i][2p+1]]: row i's k-pair adjacent = Zn layout.
    // bpack[p][j] = [B[2p][j], B[2p+1][j]]: col j's k-pair adjacent = Zm layout.
    let mut apack = [[[bf16::ZERO; 2]; M]; K];
    let mut bpack = [[[bf16::ZERO; 2]; N]; K];
    for p in 0..pairs {
        let (k0, k1) = (2 * p, 2 * p + 1);
        for i in 0..M {
            let row = a.add(i * lda);
            apack[p][i] = [*row.add(k0), if k1 < K { *row.add(k1) } else { bf16::ZERO }];
        }
        for j in 0..N {
            let r0 = b.add(k0 * ldb);
            bpack[p][j] = [*r0.add(j), if k1 < K { *b.add(k1 * ldb).add(j) } else { bf16::ZERO }];
        }
    }
    asm!(
        ".arch_extension sme",
        "smstart",
        "whilelt p0.s, xzr, {n}",            // N cols (.s), preload/store predicate
        "whilelt p1.h, xzr, {m2}",           // 2M (.h), BFMOPA row predicate
        "whilelt p2.h, xzr, {n2}",           // 2N (.h), BFMOPA col predicate
        // preload C into ZA horizontal slices
        "mov {i}, xzr",
        "20:",
        "cmp {i}, {m}",
        "b.hs 21f",
        "madd {addr}, {i}, {ldc_b}, {c}",
        "ld1w {{z0.s}}, p0/z, [{addr}]",
        "mov w12, {i:w}",
        "mova za0h.s[w12, 0], p0/m, z0.s",
        "add {i}, {i}, #1",
        "b 20b",
        "21:",
        // accumulate ⌈K/2⌉ widening rank-2 updates
        "mov {p}, xzr",
        "22:",
        "cmp {p}, {pairs}",
        "b.hs 23f",
        "madd {addr}, {p}, {ap_b}, {apack}",
        "ld1h {{z0.h}}, p1/z, [{addr}]",
        "madd {addr}, {p}, {bp_b}, {bpack}",
        "ld1h {{z1.h}}, p2/z, [{addr}]",
        "bfmopa za0.s, p1/m, p2/m, z0.h, z1.h",
        "add {p}, {p}, #1",
        "b 22b",
        "23:",
        // store ZA back to C
        "mov {i}, xzr",
        "24:",
        "cmp {i}, {m}",
        "b.hs 25f",
        "mov w12, {i:w}",
        "mova z0.s, p0/m, za0h.s[w12, 0]",
        "madd {addr}, {i}, {ldc_b}, {c}",
        "st1w {{z0.s}}, p0, [{addr}]",
        "add {i}, {i}, #1",
        "b 24b",
        "25:",
        "smstop",
        ".arch_extension nosme",
        n = in(reg) N,
        m = in(reg) M,
        m2 = in(reg) 2 * M,
        n2 = in(reg) 2 * N,
        pairs = in(reg) pairs,
        apack = in(reg) apack.as_ptr(),
        bpack = in(reg) bpack.as_ptr(),
        ap_b = in(reg) 4 * M,
        bp_b = in(reg) 4 * N,
        ldc_b = in(reg) ldc * 4,
        c = in(reg) c,
        i = out(reg) _,
        p = out(reg) _,
        addr = out(reg) _,
        out("x12") _,
        out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _,
        out("z6") _, out("z7") _, out("z8") _, out("z9") _, out("z10") _, out("z11") _,
        out("z12") _, out("z13") _, out("z14") _, out("z15") _, out("z16") _, out("z17") _,
        out("z18") _, out("z19") _, out("z20") _, out("z21") _, out("z22") _, out("z23") _,
        out("z24") _, out("z25") _, out("z26") _, out("z27") _, out("z28") _, out("z29") _,
        out("z30") _, out("z31") _,
        out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _, out("p5") _,
        out("p6") _, out("p7") _, out("p8") _, out("p9") _, out("p10") _, out("p11") _,
        out("p12") _, out("p13") _, out("p14") _, out("p15") _,
        options(nostack),
    );
}
