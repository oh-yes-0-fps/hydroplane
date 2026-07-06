//! Low-level RVV (RISC-V "V" extension v1.0) primitives via raw `asm!`. RVV registers are
//! scalable like SVE, so a vector is its fixed-size memory image: [`RvvVec<C>`] = `C` bytes = one
//! vector length.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::asm;

/// The memory image of one RVV vector: `C` bytes (`C/4` `f32` lanes, `C/8` `f64` lanes). `C` is the
/// chosen vector byte-width and never exceeds `VLENB` (so a single `m1` group holds every lane).
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct RvvVec<const C: usize>(pub [u8; C]);

impl<const C: usize> RvvVec<C> {
    #[inline(always)]
    pub const fn zeroed() -> Self {
        Self([0u8; C])
    }
}

/// Read `VLENB` (vector register length in bytes) from its CSR. Only valid where the "V" extension
/// is present (the caller detects it first).
///
/// # Safety
/// The CPU must implement the "V" extension — guard with [`is_supported`].
pub unsafe fn vlenb_raw() -> usize {
    let r: usize;
    asm!(
        ".option push",
        ".option arch, +v",
        "csrr {r}, vlenb",
        ".option pop",
        r = out(reg) r,
        options(pure, nomem, nostack),
    );
    r
}

/// `VLENB` in bytes. Only valid where the "V" extension is present (the caller detects it).
#[inline]
pub fn vlenb() -> usize {
    unsafe { vlenb_raw() }
}

/// Whether the running CPU implements the RVV "V" extension. `is_riscv_feature_detected!` is
/// unstable, so on Linux this reads the `'V'` bit (bit 21) of `AT_HWCAP` from the ELF aux vector;
/// other OSes return `false` (no portable stable probe).
#[cfg(feature = "std")]
pub fn is_supported() -> bool {
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn getauxval(ty: core::ffi::c_ulong) -> core::ffi::c_ulong;
        }
        const AT_HWCAP: core::ffi::c_ulong = 16;
        const HWCAP_ISA_V: core::ffi::c_ulong = 1 << (b'V' - b'A'); // bit 21
        unsafe { getauxval(AT_HWCAP) & HWCAP_ISA_V != 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

// `$e` is the SEW selector, `$div` the element byte size, `$vle`/`$vse` the element load/store.
// Masks are vectors of `0`/`-1`, matching the NEON/SVE vector-mask convention.

macro_rules! splat {
    ($name:ident, $t:ty, $e:literal, $div:expr, $vse:literal) => {
        #[doc = concat!("Broadcast `v` to all `C/", stringify!($div), "` lanes.")]
        pub unsafe fn $name<const C: usize>(v: $t) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                "vmv.v.x v1, {bits}",
                concat!($vse, " v1, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                bits = in(reg) v.to_bits(), o = in(reg) o.0.as_mut_ptr(),
                out("v1") _,
                options(nostack),
            );
            o
        }
    };
}

macro_rules! loadstore {
    ($load:ident, $store:ident, $t:ty, $e:literal, $div:expr, $vle:literal, $vse:literal) => {
        pub unsafe fn $load<const C: usize>(p: *const $t) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({a})"),
                concat!($vse, " v1, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) p, o = in(reg) o.0.as_mut_ptr(),
                out("v1") _,
                options(nostack),
            );
            o
        }
        pub unsafe fn $store<const C: usize>(v: &RvvVec<C>, p: *mut $t) {
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({a})"),
                concat!($vse, " v1, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) v.0.as_ptr(), o = in(reg) p,
                out("v1") _,
                options(nostack),
            );
        }
    };
}

macro_rules! binop {
    ($name:ident, $e:literal, $div:expr, $vle:literal, $vse:literal, $op:literal) => {
        pub unsafe fn $name<const C: usize>(a: &RvvVec<C>, b: &RvvVec<C>) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({a})"),
                concat!($vle, " v2, ({b})"),
                $op,
                concat!($vse, " v1, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("v1") _, out("v2") _,
                options(nostack),
            );
            o
        }
    };
}

macro_rules! unop {
    ($name:ident, $e:literal, $div:expr, $vle:literal, $vse:literal, $op:literal) => {
        pub unsafe fn $name<const C: usize>(a: &RvvVec<C>) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({a})"),
                $op,
                concat!($vse, " v1, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) a.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("v1") _,
                options(nostack),
            );
            o
        }
    };
}

/// `a * b + c`, fused (`vfmacc` accumulates into the loaded `c`).
macro_rules! fma {
    ($name:ident, $e:literal, $div:expr, $vle:literal, $vse:literal) => {
        pub unsafe fn $name<const C: usize>(
            a: &RvvVec<C>,
            b: &RvvVec<C>,
            c: &RvvVec<C>,
        ) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({c})"),
                concat!($vle, " v2, ({a})"),
                concat!($vle, " v3, ({b})"),
                "vfmacc.vv v1, v2, v3",
                concat!($vse, " v1, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), c = in(reg) c.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("v1") _, out("v2") _, out("v3") _,
                options(nostack),
            );
            o
        }
    };
}

/// Comparison to a vector mask (`-1` where true, `0` where false). RVV float compares write a
/// packed mask register; `vmerge.vim` materialises it back to the `-1`/`0` vector image the rest
/// of the crate expects. There are no `vmfgt`/`vmfge` `.vv` forms, so `ge`/`gt` swap the operands.
macro_rules! cmp {
    ($name:ident, $e:literal, $div:expr, $vle:literal, $vse:literal, $cmp:literal) => {
        pub unsafe fn $name<const C: usize>(a: &RvvVec<C>, b: &RvvVec<C>) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({a})"),
                concat!($vle, " v2, ({b})"),
                $cmp,
                "vmv.v.i v3, 0",
                "vmerge.vim v3, v3, -1, v0", // v3 = mask ? -1 : 0
                concat!($vse, " v3, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                options(nostack),
            );
            o
        }
    };
}

/// `mask ? a : b`, lane-wise (mask is the `-1`/`0` vector image; `vmsne` recovers the mask register).
macro_rules! select {
    ($name:ident, $e:literal, $div:expr, $vle:literal, $vse:literal) => {
        pub unsafe fn $name<const C: usize>(
            mask: &RvvVec<C>,
            a: &RvvVec<C>,
            b: &RvvVec<C>,
        ) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                concat!($vle, " v1, ({m})"),
                "vmsne.vi v0, v1, 0",
                concat!($vle, " v2, ({a})"),
                concat!($vle, " v3, ({b})"),
                "vmerge.vvm v4, v3, v2, v0", // v4 = mask ? a : b
                concat!($vse, " v4, ({o})"),
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                m = in(reg) mask.0.as_ptr(), a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                options(nostack),
            );
            o
        }
    };
}

/// Horizontal reduction over the active lanes. The scalar seed in `v2[0]` is the op's identity
/// (`0` for sum, `±inf` for min/max). The result lane is read out as raw bits with `vmv.x.s`,
/// so no FP register is needed.
macro_rules! reduce {
    ($name:ident, $t:ty, $bits:ty, $e:literal, $div:expr, $vle:literal, $seed:literal, $op:literal) => {
        pub unsafe fn $name<const C: usize>(a: &RvvVec<C>) -> $t {
            let bits: $bits;
            asm!(
                ".option push", ".option arch, +v",
                concat!("vsetvli {vl}, {n}, ", $e, ", m1, ta, ma"),
                $seed,
                concat!($vle, " v1, ({a})"),
                $op,                        // v0[0] = reduce(v1, seed)
                "vmv.x.s {r}, v0",
                ".option pop",
                vl = out(reg) _, n = in(reg) C / $div,
                a = in(reg) a.0.as_ptr(), r = out(reg) bits, t = out(reg) _,
                out("v0") _, out("v1") _, out("v2") _,
                options(nostack),
            );
            <$t>::from_bits(bits)
        }
    };
}

splat!(splat_f32, f32, "e32", 4, "vse32.v");
loadstore!(load_f32, store_f32, f32, "e32", 4, "vle32.v", "vse32.v");
binop!(add_f32, "e32", 4, "vle32.v", "vse32.v", "vfadd.vv v1, v1, v2");
binop!(sub_f32, "e32", 4, "vle32.v", "vse32.v", "vfsub.vv v1, v1, v2");
binop!(mul_f32, "e32", 4, "vle32.v", "vse32.v", "vfmul.vv v1, v1, v2");
binop!(div_f32, "e32", 4, "vle32.v", "vse32.v", "vfdiv.vv v1, v1, v2");
// `vfmin`/`vfmax` are IEEE 754-2019 minimumNumber/maximumNumber by spec, matching the crate-wide
// `Backend::min`/`max` contract with no fixup (unlike x86/wasm).
binop!(min_f32, "e32", 4, "vle32.v", "vse32.v", "vfmin.vv v1, v1, v2");
binop!(max_f32, "e32", 4, "vle32.v", "vse32.v", "vfmax.vv v1, v1, v2");
unop!(neg_f32, "e32", 4, "vle32.v", "vse32.v", "vfneg.v v1, v1");
unop!(sqrt_f32, "e32", 4, "vle32.v", "vse32.v", "vfsqrt.v v1, v1");
fma!(fma_f32, "e32", 4, "vle32.v", "vse32.v");
cmp!(le_f32, "e32", 4, "vle32.v", "vse32.v", "vmfle.vv v0, v1, v2");
cmp!(lt_f32, "e32", 4, "vle32.v", "vse32.v", "vmflt.vv v0, v1, v2");
cmp!(ge_f32, "e32", 4, "vle32.v", "vse32.v", "vmfle.vv v0, v2, v1");
cmp!(gt_f32, "e32", 4, "vle32.v", "vse32.v", "vmflt.vv v0, v2, v1");
select!(select_f32, "e32", 4, "vle32.v", "vse32.v");
reduce!(reduce_sum_f32, f32, u32, "e32", 4, "vle32.v", "li {t}, 0\nvmv.s.x v2, {t}", "vfredusum.vs v0, v1, v2");
reduce!(reduce_min_f32, f32, u32, "e32", 4, "vle32.v", "li {t}, 0x7F800000\nvmv.s.x v2, {t}", "vfredmin.vs v0, v1, v2");
reduce!(reduce_max_f32, f32, u32, "e32", 4, "vle32.v", "li {t}, 0xFF800000\nvmv.s.x v2, {t}", "vfredmax.vs v0, v1, v2");

splat!(splat_f64, f64, "e64", 8, "vse64.v");
loadstore!(load_f64, store_f64, f64, "e64", 8, "vle64.v", "vse64.v");
binop!(add_f64, "e64", 8, "vle64.v", "vse64.v", "vfadd.vv v1, v1, v2");
binop!(sub_f64, "e64", 8, "vle64.v", "vse64.v", "vfsub.vv v1, v1, v2");
binop!(mul_f64, "e64", 8, "vle64.v", "vse64.v", "vfmul.vv v1, v1, v2");
binop!(div_f64, "e64", 8, "vle64.v", "vse64.v", "vfdiv.vv v1, v1, v2");
binop!(min_f64, "e64", 8, "vle64.v", "vse64.v", "vfmin.vv v1, v1, v2");
binop!(max_f64, "e64", 8, "vle64.v", "vse64.v", "vfmax.vv v1, v1, v2");
unop!(neg_f64, "e64", 8, "vle64.v", "vse64.v", "vfneg.v v1, v1");
unop!(sqrt_f64, "e64", 8, "vle64.v", "vse64.v", "vfsqrt.v v1, v1");
fma!(fma_f64, "e64", 8, "vle64.v", "vse64.v");
cmp!(le_f64, "e64", 8, "vle64.v", "vse64.v", "vmfle.vv v0, v1, v2");
cmp!(lt_f64, "e64", 8, "vle64.v", "vse64.v", "vmflt.vv v0, v1, v2");
cmp!(ge_f64, "e64", 8, "vle64.v", "vse64.v", "vmfle.vv v0, v2, v1");
cmp!(gt_f64, "e64", 8, "vle64.v", "vse64.v", "vmflt.vv v0, v2, v1");
select!(select_f64, "e64", 8, "vle64.v", "vse64.v");
reduce!(reduce_sum_f64, f64, u64, "e64", 8, "vle64.v", "li {t}, 0\nvmv.s.x v2, {t}", "vfredusum.vs v0, v1, v2");
reduce!(reduce_min_f64, f64, u64, "e64", 8, "vle64.v", "li {t}, 0x7FF0000000000000\nvmv.s.x v2, {t}", "vfredmin.vs v0, v1, v2");
reduce!(reduce_max_f64, f64, u64, "e64", 8, "vle64.v", "li {t}, 0xFFF0000000000000\nvmv.s.x v2, {t}", "vfredmax.vs v0, v1, v2");

// Masks are vectors of `0`/`-1`, so the logical ops are plain bitwise ops over the `C`-byte image
// and `any`/`all` are byte-granular, correct for every element width (as in `super::sve1`).

macro_rules! maskbin {
    ($name:ident, $op:literal) => {
        pub unsafe fn $name<const C: usize>(a: &RvvVec<C>, b: &RvvVec<C>) -> RvvVec<C> {
            let mut o = RvvVec::<C>::zeroed();
            asm!(
                ".option push", ".option arch, +v",
                "vsetvli {vl}, {n}, e8, m1, ta, ma",
                "vle8.v v1, ({a})",
                "vle8.v v2, ({b})",
                $op,
                "vse8.v v1, ({o})",
                ".option pop",
                vl = out(reg) _, n = in(reg) C,
                a = in(reg) a.0.as_ptr(), b = in(reg) b.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
                out("v1") _, out("v2") _,
                options(nostack),
            );
            o
        }
    };
}

maskbin!(and_mask, "vand.vv v1, v1, v2");
maskbin!(or_mask, "vor.vv v1, v1, v2");

/// `!mask` (bitwise NOT over the byte image).
pub unsafe fn not_mask<const C: usize>(a: &RvvVec<C>) -> RvvVec<C> {
    let mut o = RvvVec::<C>::zeroed();
    asm!(
        ".option push", ".option arch, +v",
        "vsetvli {vl}, {n}, e8, m1, ta, ma",
        "vle8.v v1, ({a})",
        "vnot.v v1, v1",
        "vse8.v v1, ({o})",
        ".option pop",
        vl = out(reg) _, n = in(reg) C,
        a = in(reg) a.0.as_ptr(), o = in(reg) o.0.as_mut_ptr(),
        out("v1") _,
        options(nostack),
    );
    o
}

/// True if any mask byte is set (some active lane true).
pub unsafe fn any_mask<const C: usize>(mask: &RvvVec<C>) -> bool {
    let r: usize;
    asm!(
        ".option push", ".option arch, +v",
        "vsetvli {vl}, {n}, e8, m1, ta, ma",
        "vle8.v v1, ({m})",
        "vmsne.vi v0, v1, 0",
        "vcpop.m {r}, v0",
        ".option pop",
        vl = out(reg) _, n = in(reg) C, m = in(reg) mask.0.as_ptr(), r = out(reg) r,
        out("v0") _, out("v1") _,
        options(nostack),
    );
    r != 0
}

/// True if every mask byte is set (every active lane true).
pub unsafe fn all_mask<const C: usize>(mask: &RvvVec<C>) -> bool {
    let r: usize;
    asm!(
        ".option push", ".option arch, +v",
        "vsetvli {vl}, {n}, e8, m1, ta, ma",
        "vle8.v v1, ({m})",
        "vmseq.vi v0, v1, 0", // count bytes that are zero (some lane false)
        "vcpop.m {r}, v0",
        ".option pop",
        vl = out(reg) _, n = in(reg) C, m = in(reg) mask.0.as_ptr(), r = out(reg) r,
        out("v0") _, out("v1") _,
        options(nostack),
    );
    r == 0
}
