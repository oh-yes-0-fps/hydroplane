//! Intel AMX (Advanced Matrix Extensions): the x86 tile matrix engine, via raw `asm!`.
//!
//! AMX is the x86 counterpart to ARM's SME ZA engine (see [`super::sme1`]): a grid of eight 2-D
//! **tile** registers (`tmm0`–`tmm7`, up to 16 rows × 64 bytes) plus tile-multiply instructions that
//! fuse a whole `D = C + A·B` block. The headline float op is [`mma_bf16`]: one `tdpbf16ps` computes
//! a `M×N` f32 tile from `bf16` `A`/`B` (`M, N ≤ 16`, `K ≤ 32`), the direct analogue of SME's
//! `BFMOPA`. [`mma_f16`] is the IEEE-half twin via `tdpfp16ps` (AMX-FP16, Granite Rapids and newer);
//! the two kernels share the VNNI word-pair operand layout and an `f32` accumulator, differing only
//! in the element type and the dot-product mnemonic.
//!
//! The tile mnemonics assemble on **stable** with no `amx-*` `target_feature` (the x86 integrated
//! assembler accepts them unconditionally — unlike ARM, no `.arch_extension` is needed); they only
//! *execute* where the CPU implements the matching extension (AMX-BF16 / AMX-FP16) and the OS has
//! granted tile-data permission, which [`is_supported`] / [`is_supported_f16`] check. AMX is detected
//! with raw `CPUID` because the `is_x86_feature_detected!` AMX strings are still unstable.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::needless_range_loop)]

use core::arch::asm;
use half::{bf16, f16};

/// The 64-byte AMX tile configuration (`palette 1`): per-tile row count and bytes-per-row. Loaded
/// with `ldtilecfg`. Layout is fixed by the ISA — `colsb[i]` at offset `16 + 2i`, `rows[i]` at
/// `48 + i`.
#[repr(C)]
struct TileCfg {
    palette: u8,
    start_row: u8,
    reserved: [u8; 14],
    colsb: [u16; 16],
    rows: [u8; 16],
}

impl TileCfg {
    #[inline]
    fn new() -> Self {
        Self {
            palette: 1,
            start_row: 0,
            reserved: [0; 14],
            colsb: [0; 16],
            rows: [0; 16],
        }
    }
    #[inline]
    fn set(&mut self, tile: usize, rows: u8, colsb: u16) {
        self.rows[tile] = rows;
        self.colsb[tile] = colsb;
    }
}

/// Emits a `D = C + A·B` tile kernel for a 2-byte float element type. The AMX `tdp*ps` float
/// family shares the VNNI word-pair operand layout and an `f32` accumulator, so the body is
/// identical apart from the element type (`$ty`) and the dot-product mnemonic (`$dot`).
///
/// `tdp*ps tmm_c, tmm_a, tmm_b` reads `A` as `M` rows of `K` elements (the natural row-major
/// layout) and `B` in the VNNI **pair** layout: tile row `p` holds, for each column `n`, the two
/// `k`-neighbours `B[2p][n], B[2p+1][n]`. So `A` is copied with the odd-`K` tail zero-padded to an
/// even width, and `B` is packed into `⌈K/2⌉` pair-rows.
macro_rules! tile_mma {
    ($(#[$doc:meta])* $name:ident, $ty:ty, $dot:literal) => {
        $(#[$doc])*
        #[inline]
        pub unsafe fn $name<const M: usize, const N: usize, const K: usize>(
            a: *const $ty,
            lda: usize,
            b: *const $ty,
            ldb: usize,
            c: *mut f32,
            ldc: usize,
        ) {
            let keven = K.next_multiple_of(2);
            let pairs = keven / 2;

            // A: M rows × keven elems (tsrc1 is row-major; pad the odd-K column to an even width).
            let mut apack = [[<$ty>::ZERO; 32]; 16];
            for i in 0..M {
                let row = a.add(i * lda);
                for k in 0..K {
                    apack[i][k] = *row.add(k);
                }
            }
            // B: ⌈K/2⌉ pair-rows × 2N elems — bpack[p][2n] = B[2p][n], bpack[p][2n+1] = B[2p+1][n] (or 0).
            let mut bpack = [[<$ty>::ZERO; 32]; 16];
            for p in 0..pairs {
                let (k0, k1) = (2 * p, 2 * p + 1);
                for n in 0..N {
                    bpack[p][2 * n] = *b.add(k0 * ldb + n);
                    bpack[p][2 * n + 1] = if k1 < K { *b.add(k1 * ldb + n) } else { <$ty>::ZERO };
                }
            }

            let mut cfg = TileCfg::new();
            cfg.set(0, M as u8, (N * 4) as u16); // tmm0 = C  (M × N f32)
            cfg.set(1, M as u8, (keven * 2) as u16); // tmm1 = A  (M × keven elems)
            cfg.set(2, pairs as u8, (N * 4) as u16); // tmm2 = B  (pairs × 2N elems)

            asm!(
                "ldtilecfg [{cfg}]",
                "tileloadd tmm0, [{c} + {ldc_b} * 1]",        // preload C
                "tileloadd tmm1, [{a} + {lda_b} * 1]",        // A
                "tileloadd tmm2, [{b} + {ldb_b} * 1]",        // B (VNNI pairs)
                concat!($dot, " tmm0, tmm1, tmm2"),           // C += A·B
                "tilestored [{c} + {ldc_b} * 1], tmm0",       // store C
                "tilerelease",
                cfg = in(reg) &cfg as *const TileCfg,
                c = in(reg) c,
                ldc_b = in(reg) ldc * 4,
                a = in(reg) apack.as_ptr(),
                lda_b = in(reg) keven * 2,
                b = in(reg) bpack.as_ptr(),
                ldb_b = in(reg) N * 4,
                options(nostack),
            );
        }
    };
}

tile_mma! {
    /// `D = C + A·B` for `bf16` tiles with an `f32` accumulator, via one `tdpbf16ps` on the AMX
    /// engine. `a` is `M×K` (row stride `lda`), `b` is `K×N` (row stride `ldb`), `c` is the `M×N`
    /// in/out accumulator (row stride `ldc`). Requires `M ≤ 16`, `N ≤ 16`, `K ≤ 32` (one tile
    /// block); the caller gates on those bounds and on [`is_supported`].
    mma_bf16, bf16, "tdpbf16ps"
}

tile_mma! {
    /// `D = C + A·B` for IEEE `f16` tiles with an `f32` accumulator, via one `tdpfp16ps` on the AMX
    /// engine — the half-precision twin of [`mma_bf16`] with the same operand bounds and layout.
    /// Requires `M ≤ 16`, `N ≤ 16`, `K ≤ 32` (one tile block); the caller gates on those bounds and
    /// on [`is_supported_f16`] (AMX-FP16 is a distinct CPUID bit from AMX-BF16).
    mma_f16, f16, "tdpfp16ps"
}

/// Whether the running CPU implements AMX-BF16 **and** the OS has granted AMX tile-data permission.
///
/// AMX is gated behind two things: the `AMX-TILE`/`AMX-BF16` CPUID bits, and (on Linux) a one-time
/// `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` that opts the process into the large tile
/// register state — without it, executing a tile instruction faults. The result is cached. On
/// non-Linux x86 this returns `false` (no portable stable way to request the permission — those
/// hosts fall back to the AVX-512-BF16 / SIMD matmul paths), mirroring [`super::sme1::is_supported`].
#[cfg(feature = "std")]
pub fn is_supported() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| cpuid_amx_bf16() && request_xtiledata_perm())
}

/// Whether the running CPU implements AMX-FP16 (Granite Rapids and newer) **and** the OS has granted
/// AMX tile-data permission.
///
/// AMX-FP16 — the `tdpfp16ps` IEEE-half tile multiply — is reported by a CPUID bit separate from
/// AMX-BF16, and a host can ship one without the other (Sapphire/Emerald Rapids have AMX-BF16 but no
/// AMX-FP16), so the `f16` path checks this independently. Same caching, tile-data permission, and
/// non-Linux fallback story as [`is_supported`]; where it returns `false`, `f16` matmul drops to the
/// AVX-512-FP16 / SIMD paths.
#[cfg(feature = "std")]
pub fn is_supported_f16() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| cpuid_amx_fp16() && request_xtiledata_perm())
}

/// `CPUID.(EAX=7,ECX=0).EDX` bit 24 (AMX-TILE) and bit 22 (AMX-BF16). Raw `CPUID` because the
/// `is_x86_feature_detected!("amx-*")` strings are still unstable.
fn cpuid_amx_bf16() -> bool {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::__cpuid_count;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::__cpuid_count;
    let leaf7 = __cpuid_count(7, 0);
    let amx_tile = leaf7.edx & (1 << 24) != 0;
    let amx_bf16 = leaf7.edx & (1 << 22) != 0;
    amx_tile && amx_bf16
}

/// `CPUID.(EAX=7,ECX=0).EDX` bit 24 (AMX-TILE) and `CPUID.(EAX=7,ECX=1).EAX` bit 21 (AMX-FP16).
/// AMX-FP16 lives in leaf-7 **subleaf 1**, not alongside AMX-BF16 in subleaf 0.
fn cpuid_amx_fp16() -> bool {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::__cpuid_count;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::__cpuid_count;
    let amx_tile = __cpuid_count(7, 0).edx & (1 << 24) != 0;
    let amx_fp16 = __cpuid_count(7, 1).eax & (1 << 21) != 0;
    amx_tile && amx_fp16
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn request_xtiledata_perm() -> bool {
    use core::ffi::c_long;
    const SYS_ARCH_PRCTL: c_long = 158;
    const ARCH_REQ_XCOMP_PERM: c_long = 0x1023;
    const XFEATURE_XTILEDATA: c_long = 18;
    unsafe extern "C" {
        fn syscall(num: c_long, ...) -> c_long;
    }
    unsafe { syscall(SYS_ARCH_PRCTL, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0 }
}

#[cfg(all(feature = "std", not(target_os = "linux")))]
fn request_xtiledata_perm() -> bool {
    false
}
