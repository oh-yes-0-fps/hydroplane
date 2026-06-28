//! AVX-512-FP16 (x86_64): true 32-wide half-precision arithmetic via raw `asm!`.
//!
//! The AVX-512-FP16 *intrinsics* (`stdarch_x86_avx512_f16`) and the `f16` primitive type are still
//! unstable, so this module emits the `v*ph` instructions directly with `core::arch::asm!` — the
//! same stable-on-x86 path the SME module ([`super::sme1`]) uses on aarch64. Inline asm and the
//! `zmm_reg`/`kreg` register classes are stable on x86; only the *intrinsic wrappers* needed nightly.
//!
//! Data rides in [`__m512i`] (a stable 512-bit carrier holding the 32×`f16` bit pattern of
//! `half::f16`), so nothing here needs nightly. The mnemonics only *execute* where the CPU
//! implements `avx512fp16` (Sapphire Rapids+, Zen 5); dispatch guards that with runtime detection.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::asm;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use half::f16;

/// Extended compare predicates (ordered, quiet) for `vcmpph`, matching the `_CMP_*_OQ` imms.
pub const CMP_LT_OQ: i32 = 0x11;
pub const CMP_LE_OQ: i32 = 0x12;
pub const CMP_GE_OQ: i32 = 0x1d;
pub const CMP_GT_OQ: i32 = 0x1e;

/// Load 32 contiguous `f16` (unaligned) into a register.
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn load(p: *const f16) -> __m512i {
    core::ptr::read_unaligned(p as *const __m512i)
}

/// Store a register to 32 contiguous `f16` (unaligned).
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn store(p: *mut f16, v: __m512i) {
    core::ptr::write_unaligned(p as *mut __m512i, v)
}

/// Broadcast one `f16` to all 32 lanes.
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn splat(v: f16) -> __m512i {
    _mm512_set1_epi16(v.to_bits() as i16)
}

/// Negate every lane by flipping its sign bit (exact, including `±0`/NaN — unlike `0 - x`).
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn neg(a: __m512i) -> __m512i {
    _mm512_xor_si512(a, _mm512_set1_epi16(0x8000u16 as i16))
}

macro_rules! binop {
    ($name:ident, $mnem:literal) => {
        #[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
        #[inline]
        pub unsafe fn $name(a: __m512i, b: __m512i) -> __m512i {
            let r;
            asm!(
                concat!($mnem, " {r}, {a}, {b}"),
                r = lateout(zmm_reg) r, a = in(zmm_reg) a, b = in(zmm_reg) b,
                options(pure, nomem, nostack, preserves_flags),
            );
            r
        }
    };
}

binop!(add, "vaddph");
binop!(sub, "vsubph");
binop!(mul, "vmulph");
binop!(div, "vdivph");
binop!(min, "vminph");
binop!(max, "vmaxph");

#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn sqrt(a: __m512i) -> __m512i {
    let r;
    asm!(
        "vsqrtph {r}, {a}",
        r = lateout(zmm_reg) r, a = in(zmm_reg) a,
        options(pure, nomem, nostack, preserves_flags),
    );
    r
}

/// Fused multiply-add: `a * b + c`, lane-wise, single rounding.
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn fma(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    // vfmadd213ph dst, s2, s3 => dst = s2*dst + s3; with dst=a, s2=b, s3=c => a*b + c.
    let mut dst = a;
    asm!(
        "vfmadd213ph {dst}, {b}, {c}",
        dst = inout(zmm_reg) dst, b = in(zmm_reg) b, c = in(zmm_reg) c,
        options(pure, nomem, nostack, preserves_flags),
    );
    dst
}

/// Lane-wise compare to a 32-bit mask (one bit per lane), with an ordered-quiet predicate.
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn cmp<const IMM: i32>(a: __m512i, b: __m512i) -> u32 {
    let m: u32;
    asm!(
        "vcmpph k1, {a}, {b}, {imm}",
        "kmovd {m:e}, k1",
        a = in(zmm_reg) a, b = in(zmm_reg) b, m = lateout(reg) m, imm = const IMM,
        out("k1") _,
        options(pure, nomem, nostack, preserves_flags),
    );
    m
}

/// Blend by mask, lane-wise: `mask_bit ? a : b` (16-bit granularity via `vpblendmw`).
#[target_feature(enable = "avx512fp16,avx512f,avx512bw")]
#[inline]
pub unsafe fn select(mask: u32, a: __m512i, b: __m512i) -> __m512i {
    // vpblendmw dst{k}, s1, s2 => dst = k ? s2 : s1; with s1=b, s2=a => k ? a : b.
    let r;
    asm!(
        "kmovd k1, {m:e}",
        "vpblendmw {r}{{k1}}, {b}, {a}",
        m = in(reg) mask, a = in(zmm_reg) a, b = in(zmm_reg) b, r = lateout(zmm_reg) r,
        out("k1") _,
        options(pure, nomem, nostack, preserves_flags),
    );
    r
}
