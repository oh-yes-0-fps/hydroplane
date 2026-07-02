//! Hand-written A32 NEON backend for 32-bit ARM (`target_arch = "arm"`): 4-wide `f32` via raw `asm!`.
//!
//! Unlike aarch64 (whose `core::arch::aarch64` NEON intrinsics are stable, see
//! [`crate::backend::neon`]), the 32-bit `core::arch::arm` NEON intrinsics are still unstable, so
//! this emits A32 NEON instructions directly — each `asm!` opens with `.fpu neon` so the assembler
//! accepts them on **stable**, the same raw-asm route SME/RVV take on their arches. A NEON register
//! can't be named in Rust without the intrinsic carrier types, so the [`Backend::Vector`] is the
//! 16-byte memory image [`F32x4`] and every op round-trips through it.
//!
//! **`f32` only.** armv7 NEON has no double-precision vector unit (`f64` is VFP-scalar) and no
//! native `f16`/`bf16`, so those scalars take the [`ScalarBackend`](crate::ScalarBackend) path. NEON
//! also lacks a vector divide and sqrt, so `div`/`sqrt` are the per-lane VFP-scalar ops; `fma` is
//! `vmla.f32` (multiply-accumulate, not IEEE-fused — within the crate's `fma` tolerance). All are
//! VFPv3-level, so this runs on every NEON armv7 core (Cortex-A8/A9+), not just VFPv4 ones.
#![cfg(target_arch = "arm")]
// Constructed only where NEON is present (dispatch enforces it via `detect`/`target_feature`); the
// constructors read as dead code on a NEON-less arm build.
#![allow(dead_code, unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::asm;

use crate::backend::Backend;

/// The 16-byte memory image of one NEON `q` register: 4 `f32` lanes. Reused as the mask type, where
/// each lane is `-1` (all-ones) or `0` — the NEON vector-mask convention.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct F32x4(pub [f32; 4]);

impl F32x4 {
    #[inline(always)]
    fn zeroed() -> Self {
        Self([0.0; 4])
    }
}

/// A32 NEON execution token. Zero-sized; construct only on a CPU with NEON
/// ([`detect`](Neon::detect) / [`new_unchecked`](Neon::new_unchecked)).
#[derive(Clone, Copy, Debug)]
pub struct Neon;

impl Neon {
    /// # Safety
    /// The CPU must implement NEON (Advanced SIMD).
    #[inline]
    pub unsafe fn new_unchecked() -> Self {
        Neon
    }

    /// `Some(Neon)` if the CPU implements NEON, else `None`. Reads `HWCAP_NEON` (bit 12) from the ELF
    /// aux vector on Linux (the stable `is_arm_feature_detected!` macro doesn't exist); `None` on
    /// other OSes — mirrors [`crate::arch::sme1::is_supported`].
    #[cfg(feature = "std")]
    #[inline]
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            unsafe extern "C" {
                fn getauxval(ty: core::ffi::c_ulong) -> core::ffi::c_ulong;
            }
            const AT_HWCAP: core::ffi::c_ulong = 16;
            const HWCAP_NEON: core::ffi::c_ulong = 1 << 12;
            if unsafe { getauxval(AT_HWCAP) & HWCAP_NEON != 0 } {
                return Some(Neon);
            }
            None
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }
}

/// Binary `q`-register op: `vld1` both images, apply `$op`, `vst1` the result.
macro_rules! binop {
    ($name:ident, $op:literal) => {
        #[inline]
        unsafe fn $name(a: &F32x4, b: &F32x4) -> F32x4 {
            let mut o = F32x4::zeroed();
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{a}]",
                "vld1.32 {{q1}}, [{b}]",
                $op,
                "vst1.32 {{q0}}, [{o}]",
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("q0") _, out("q1") _,
                options(nostack),
            );
            o
        }
    };
}

binop!(add, "vadd.f32 q0, q0, q1");
binop!(sub, "vsub.f32 q0, q0, q1");
binop!(mul, "vmul.f32 q0, q0, q1");

/// IEEE minimumNumber / maximumNumber. ARMv7 NEON's `vmin`/`vmax` propagate NaN (and `VMINNM` is
/// ARMv8-only), so patch: a-is-NaN lanes take `b`, b-is-NaN lanes take `a` (both-NaN stays NaN).
/// `vceq x, x` is the ordered mask; `vbsl` selects with its destination as the mask.
macro_rules! minmaxnm {
    ($name:ident, $op:literal) => {
        #[inline]
        unsafe fn $name(a: &F32x4, b: &F32x4) -> F32x4 {
            let mut o = F32x4::zeroed();
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{a}]",
                "vld1.32 {{q1}}, [{b}]",
                $op,
                "vceq.f32 q3, q0, q0",
                "vbsl q3, q2, q1",
                "vceq.f32 q2, q1, q1",
                "vbsl q2, q3, q0",
                "vst1.32 {{q2}}, [{o}]",
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("q0") _, out("q1") _, out("q2") _, out("q3") _,
                options(nostack),
            );
            o
        }
    };
}

minmaxnm!(min, "vmin.f32 q2, q0, q1");
minmaxnm!(max, "vmax.f32 q2, q0, q1");

/// Comparison → a `-1`/`0` lane mask.
macro_rules! cmpop {
    ($name:ident, $op:literal) => {
        #[inline]
        unsafe fn $name(a: &F32x4, b: &F32x4) -> F32x4 {
            let mut o = F32x4::zeroed();
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{a}]",
                "vld1.32 {{q1}}, [{b}]",
                $op,                        // q2 = (q0 ? q1) ? -1 : 0
                "vst1.32 {{q2}}, [{o}]",
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("q0") _, out("q1") _, out("q2") _,
                options(nostack),
            );
            o
        }
    };
}

cmpop!(le, "vcle.f32 q2, q0, q1");
cmpop!(lt, "vclt.f32 q2, q0, q1");
cmpop!(ge, "vcge.f32 q2, q0, q1");
cmpop!(gt, "vcgt.f32 q2, q0, q1");

/// Bitwise mask op over the byte image.
macro_rules! maskbin {
    ($name:ident, $op:literal) => {
        #[inline]
        unsafe fn $name(a: &F32x4, b: &F32x4) -> F32x4 {
            let mut o = F32x4::zeroed();
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{a}]",
                "vld1.32 {{q1}}, [{b}]",
                $op,
                "vst1.32 {{q0}}, [{o}]",
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("q0") _, out("q1") _,
                options(nostack),
            );
            o
        }
    };
}

maskbin!(and_mask, "vand q0, q0, q1");
maskbin!(or_mask, "vorr q0, q0, q1");

#[inline]
unsafe fn not_mask(a: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q0}}, [{a}]",
        "vmvn q0, q0",
        "vst1.32 {{q0}}, [{o}]",
        a = in(reg) a.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
        out("q0") _,
        options(nostack),
    );
    o
}

#[inline]
unsafe fn neg(a: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q0}}, [{a}]",
        "vneg.f32 q0, q0",
        "vst1.32 {{q0}}, [{o}]",
        a = in(reg) a.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
        out("q0") _,
        options(nostack),
    );
    o
}

/// Absolute value via `vabs.f32` — a single NEON op, cheaper than `max(a, -a)`.
#[inline]
unsafe fn abs(a: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q0}}, [{a}]",
        "vabs.f32 q0, q0",
        "vst1.32 {{q0}}, [{o}]",
        a = in(reg) a.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
        out("q0") _,
        options(nostack),
    );
    o
}

/// `a * b + c`, via `vmla.f32` (NEON multiply-accumulate; two roundings, not IEEE-fused).
#[inline]
unsafe fn fma(a: &F32x4, b: &F32x4, c: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q0}}, [{c}]", // accumulator
        "vld1.32 {{q1}}, [{a}]",
        "vld1.32 {{q2}}, [{b}]",
        "vmla.f32 q0, q1, q2",   // q0 += q1*q2
        "vst1.32 {{q0}}, [{o}]",
        a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), c = in(reg) c.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("q0") _, out("q1") _, out("q2") _,
        options(nostack),
    );
    o
}

/// Vector divide — NEON has none, so per-lane VFP-scalar `vdiv.f32` (`s0..3 / s4..7`).
#[inline]
unsafe fn div(a: &F32x4, b: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q0}}, [{a}]",
        "vld1.32 {{q1}}, [{b}]",
        "vdiv.f32 s0, s0, s4",
        "vdiv.f32 s1, s1, s5",
        "vdiv.f32 s2, s2, s6",
        "vdiv.f32 s3, s3, s7",
        "vst1.32 {{q0}}, [{o}]",
        a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
        out("q0") _, out("q1") _,
        options(nostack),
    );
    o
}

/// Vector sqrt — NEON has only the imprecise estimate, so per-lane VFP-scalar `vsqrt.f32`.
#[inline]
unsafe fn sqrt(a: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q0}}, [{a}]",
        "vsqrt.f32 s0, s0",
        "vsqrt.f32 s1, s1",
        "vsqrt.f32 s2, s2",
        "vsqrt.f32 s3, s3",
        "vst1.32 {{q0}}, [{o}]",
        a = in(reg) a.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
        out("q0") _,
        options(nostack),
    );
    o
}

#[inline]
unsafe fn select(m: &F32x4, a: &F32x4, b: &F32x4) -> F32x4 {
    let mut o = F32x4::zeroed();
    asm!(
        ".fpu neon",
        "vld1.32 {{q2}}, [{m}]",
        "vld1.32 {{q0}}, [{a}]",
        "vld1.32 {{q1}}, [{b}]",
        "vbsl q2, q0, q1",       // q2 = (q2 & a) | (~q2 & b)
        "vst1.32 {{q2}}, [{o}]",
        m = in(reg) m.0.as_ptr(), a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("q0") _, out("q1") _, out("q2") _,
        options(nostack),
    );
    o
}

/// Cross-lane unsigned reduction of the mask to lane 0, read out as a GPR. `$op` is `vpmax`
/// (→ `any`: non-zero if any lane set) or `vpmin` (→ `all`: all-ones only if every lane set).
macro_rules! mask_reduce {
    ($name:ident, $op:literal) => {
        #[inline]
        unsafe fn $name(m: &F32x4) -> u32 {
            let r: u32;
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{m}]",
                concat!($op, " d0, d0, d1"),
                concat!($op, " d0, d0, d0"),
                "vmov {r}, s0",
                m = in(reg) m.0.as_ptr(), r = out(reg) r,
                out("q0") _,
                options(nostack),
            );
            r
        }
    };
}

mask_reduce!(any_bits, "vpmax.u32");
mask_reduce!(all_bits, "vpmin.u32");

/// Horizontal float reduction to lane 0. `$pair` folds the two halves then within the half.
macro_rules! freduce {
    ($name:ident, $fold:literal) => {
        #[inline]
        unsafe fn $name(v: &F32x4) -> f32 {
            let bits: u32;
            asm!(
                ".fpu neon",
                "vld1.32 {{q0}}, [{a}]",
                concat!($fold, " d0, d0, d1"),
                concat!($fold, " d0, d0, d0"),
                "vmov {r}, s0",
                a = in(reg) v.0.as_ptr(), r = out(reg) bits,
                out("q0") _,
                options(nostack),
            );
            f32::from_bits(bits)
        }
    };
}

freduce!(reduce_sum, "vpadd.f32");

impl Backend<f32> for Neon {
    type Vector = F32x4;
    type Mask = F32x4;

    type IVector = [u32; 4];
    #[inline(always)]
    fn iload(self, s: &[u32]) -> [u32; 4] {
        let mut v = [0u32; 4];
        v.copy_from_slice(s);
        v
    }
    #[inline(always)]
    fn istore(self, v: [u32; 4], out: &mut [u32]) {
        out.copy_from_slice(&v);
    }


    #[inline(always)]
    fn lanes(self) -> usize {
        4
    }
    #[inline(always)]
    fn splat(self, v: f32) -> F32x4 {
        F32x4([v; 4])
    }
    #[inline(always)]
    fn load(self, s: &[f32]) -> F32x4 {
        debug_assert_eq!(s.len(), 4);
        F32x4([s[0], s[1], s[2], s[3]])
    }
    #[inline(always)]
    fn store(self, v: F32x4, s: &mut [f32]) {
        debug_assert_eq!(s.len(), 4);
        s[..4].copy_from_slice(&v.0);
    }
    #[inline(always)]
    fn add(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { add(&a, &b) }
    }
    #[inline(always)]
    fn sub(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { sub(&a, &b) }
    }
    #[inline(always)]
    fn mul(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { mul(&a, &b) }
    }
    #[inline(always)]
    fn div(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { div(&a, &b) }
    }
    #[inline(always)]
    fn neg(self, a: F32x4) -> F32x4 {
        unsafe { neg(&a) }
    }
    #[inline(always)]
    fn abs(self, a: F32x4) -> F32x4 {
        unsafe { abs(&a) }
    }
    #[inline(always)]
    fn fma(self, a: F32x4, b: F32x4, c: F32x4) -> F32x4 {
        unsafe { fma(&a, &b, &c) }
    }
    #[inline(always)]
    fn sqrt(self, a: F32x4) -> F32x4 {
        unsafe { sqrt(&a) }
    }
    #[inline(always)]
    fn min(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { min(&a, &b) }
    }
    #[inline(always)]
    fn max(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { max(&a, &b) }
    }
    #[inline(always)]
    fn le(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { le(&a, &b) }
    }
    #[inline(always)]
    fn lt(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { lt(&a, &b) }
    }
    #[inline(always)]
    fn ge(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { ge(&a, &b) }
    }
    #[inline(always)]
    fn gt(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { gt(&a, &b) }
    }
    #[inline(always)]
    fn mask_and(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { and_mask(&a, &b) }
    }
    #[inline(always)]
    fn mask_or(self, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { or_mask(&a, &b) }
    }
    #[inline(always)]
    fn mask_not(self, a: F32x4) -> F32x4 {
        unsafe { not_mask(&a) }
    }
    #[inline(always)]
    fn select(self, m: F32x4, a: F32x4, b: F32x4) -> F32x4 {
        unsafe { select(&m, &a, &b) }
    }
    #[inline(always)]
    fn any(self, m: F32x4) -> bool {
        unsafe { any_bits(&m) != 0 }
    }
    #[inline(always)]
    fn all(self, m: F32x4) -> bool {
        unsafe { all_bits(&m) == u32::MAX }
    }
    #[inline(always)]
    fn reduce_sum(self, v: F32x4) -> f32 {
        unsafe { reduce_sum(&v) }
    }
    #[inline(always)]
    fn reduce_min(self, v: F32x4) -> f32 {
        // Scalar minimumNumber fold — the pairwise `vpmin.f32` would propagate NaN.
        v.0.iter().copied().fold(f32::NAN, f32::min)
    }
    #[inline(always)]
    fn reduce_max(self, v: F32x4) -> f32 {
        v.0.iter().copied().fold(f32::NAN, f32::max)
    }
}
