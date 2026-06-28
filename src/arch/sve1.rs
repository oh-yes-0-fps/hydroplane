//! Low-level SVE (v1) primitives via raw `asm!`.
//!
//! An SVE register is *sizeless* and can't live in a Rust struct (see `SVE.md`), so a vector is
//! represented as its **memory image**: [`SveVec<C>`] = `C` bytes. Every op loads the image into a
//! `z` register at a fixed active-lane count (`whilelt` over `C / elem_size`), computes, and stores
//! back — so the fixed width is `C`, and one `SveVec<C>` byte size = one vector length. The
//! per-VL/per-version backends ([`crate::arch`]) wrap these.
//!
//! These emit SVE instructions under `#[target_feature(enable = "sve")]`; they **compile**
//! everywhere but **run only where base (non-streaming) SVE exists** — notably *not* on Apple
//! silicon, which has SVE only inside SME streaming mode (see [`crate::arch::sme1`]).
#![allow(dead_code, unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::asm;

/// The memory image of one SVE vector: `C` bytes (`C/4` `f32` lanes, `C/8` `f64` lanes).
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct SveVec<const C: usize>(pub [u8; C]);

impl<const C: usize> SveVec<C> {
    #[inline(always)]
    pub const fn zeroed() -> Self {
        Self([0u8; C])
    }
}

// ─────────────────────────────── f32 (C/4 lanes, `.s`) ───────────────────────────────

/// Broadcast `v` to all `C/4` lanes.
#[target_feature(enable = "sve")]
pub unsafe fn splat_f32<const C: usize>(v: f32) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.s, xzr, {n}",
        "mov z0.s, {bits:w}",
        "st1w {{z0.s}}, p0, [{o}]",
        n = in(reg) C / 4,
        bits = in(reg) v.to_bits(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Load `C/4` `f32` lanes from `p` (must point to ≥ `C` bytes).
#[target_feature(enable = "sve")]
pub unsafe fn load_f32<const C: usize>(p: *const f32) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.s, xzr, {n}",
        "ld1w {{z0.s}}, p0/z, [{a}]",
        "st1w {{z0.s}}, p0, [{o}]",
        n = in(reg) C / 4,
        a = in(reg) p,
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Store `C/4` `f32` lanes to `p`.
#[target_feature(enable = "sve")]
pub unsafe fn store_f32<const C: usize>(v: &SveVec<C>, p: *mut f32) {
    asm!(
        "whilelt p0.s, xzr, {n}",
        "ld1w {{z0.s}}, p0/z, [{a}]",
        "st1w {{z0.s}}, p0, [{o}]",
        n = in(reg) C / 4,
        a = in(reg) v.0.as_ptr(),
        o = in(reg) p,
        out("z0") _, out("p0") _,
    );
}

macro_rules! sve1_binop_f32 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.s, xzr, {n}",
                "ld1w {{z0.s}}, p0/z, [{a}]",
                "ld1w {{z1.s}}, p0/z, [{b}]",
                $op,
                "st1w {{z0.s}}, p0, [{o}]",
                n = in(reg) C / 4,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("p0") _,
            );
            o
        }
    };
}

sve1_binop_f32!(add_f32, "fadd z0.s, z0.s, z1.s");
sve1_binop_f32!(sub_f32, "fsub z0.s, z0.s, z1.s");
sve1_binop_f32!(mul_f32, "fmul z0.s, z0.s, z1.s");
sve1_binop_f32!(div_f32, "fdiv z0.s, p0/m, z0.s, z1.s");
sve1_binop_f32!(min_f32, "fmin z0.s, p0/m, z0.s, z1.s");
sve1_binop_f32!(max_f32, "fmax z0.s, p0/m, z0.s, z1.s");

/// `a * b + c`, fused.
#[target_feature(enable = "sve")]
pub unsafe fn fma_f32<const C: usize>(a: &SveVec<C>, b: &SveVec<C>, c: &SveVec<C>) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.s, xzr, {n}",
        "ld1w {{z0.s}}, p0/z, [{c}]", // accumulator
        "ld1w {{z1.s}}, p0/z, [{a}]",
        "ld1w {{z2.s}}, p0/z, [{b}]",
        "fmla z0.s, p0/m, z1.s, z2.s", // z0 += z1*z2
        "st1w {{z0.s}}, p0, [{o}]",
        n = in(reg) C / 4,
        a = in(reg) a.0.as_ptr(),
        b = in(reg) b.0.as_ptr(),
        c = in(reg) c.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("z1") _, out("z2") _, out("p0") _,
    );
    o
}

macro_rules! sve1_unop_f32 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.s, xzr, {n}",
                "ld1w {{z0.s}}, p0/z, [{a}]",
                $op,
                "st1w {{z0.s}}, p0, [{o}]",
                n = in(reg) C / 4,
                a = in(reg) a.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("p0") _,
            );
            o
        }
    };
}

sve1_unop_f32!(neg_f32, "fneg z0.s, p0/m, z0.s");
sve1_unop_f32!(sqrt_f32, "fsqrt z0.s, p0/m, z0.s");

/// Comparison → a vector mask (`-1` lanes where true, `0` where false), so the mask is itself an
/// `SveVec<C>` (matching the NEON vector-mask convention rather than a sizeless predicate).
macro_rules! sve1_cmp_f32 {
    ($name:ident, $cmp:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.s, xzr, {n}",
                "ld1w {{z0.s}}, p0/z, [{a}]",
                "ld1w {{z1.s}}, p0/z, [{b}]",
                $cmp,                       // p1 = (z0 ? z1)
                "cpy z2.s, p1/z, #-1",      // -1 in true lanes, 0 elsewhere
                "st1w {{z2.s}}, p0, [{o}]",
                n = in(reg) C / 4,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
            );
            o
        }
    };
}

sve1_cmp_f32!(le_f32, "fcmle p1.s, p0/z, z0.s, z1.s");
sve1_cmp_f32!(lt_f32, "fcmlt p1.s, p0/z, z0.s, z1.s");
sve1_cmp_f32!(ge_f32, "fcmge p1.s, p0/z, z0.s, z1.s");
sve1_cmp_f32!(gt_f32, "fcmgt p1.s, p0/z, z0.s, z1.s");

/// Bitwise mask ops (the masks are vectors of `0`/`-1`).
macro_rules! sve1_maskbin {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.b, xzr, {n}",
                "ld1b {{z0.b}}, p0/z, [{a}]",
                "ld1b {{z1.b}}, p0/z, [{b}]",
                $op,
                "st1b {{z0.b}}, p0, [{o}]",
                n = in(reg) C,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("p0") _,
            );
            o
        }
    };
}

sve1_maskbin!(and_mask, "and z0.d, z0.d, z1.d");
sve1_maskbin!(or_mask, "orr z0.d, z0.d, z1.d");

/// `!mask` (bitwise NOT over the byte image).
#[target_feature(enable = "sve")]
pub unsafe fn not_mask<const C: usize>(a: &SveVec<C>) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.b, xzr, {n}",
        "ld1b {{z0.b}}, p0/z, [{a}]",
        "not z0.b, p0/m, z0.b",
        "st1b {{z0.b}}, p0, [{o}]",
        n = in(reg) C,
        a = in(reg) a.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// `mask ? a : b`, lane-wise (mask is `-1`/`0`).
#[target_feature(enable = "sve")]
pub unsafe fn select_f32<const C: usize>(
    mask: &SveVec<C>,
    a: &SveVec<C>,
    b: &SveVec<C>,
) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.s, xzr, {n}",
        "ld1w {{z0.s}}, p0/z, [{m}]",
        "cmpne p1.s, p0/z, z0.s, #0", // p1 = mask lane set
        "ld1w {{z1.s}}, p0/z, [{a}]",
        "ld1w {{z2.s}}, p0/z, [{b}]",
        "sel z1.s, p1, z1.s, z2.s",   // p1 ? a : b
        "st1w {{z1.s}}, p0, [{o}]",
        n = in(reg) C / 4,
        m = in(reg) mask.0.as_ptr(),
        a = in(reg) a.0.as_ptr(),
        b = in(reg) b.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
    );
    o
}

/// True if any mask lane is set.
#[target_feature(enable = "sve")]
pub unsafe fn any_mask<const C: usize>(mask: &SveVec<C>) -> bool {
    let r: u64;
    asm!(
        "whilelt p0.b, xzr, {n}",
        "ld1b {{z0.b}}, p0/z, [{m}]",
        "cmpne p1.b, p0/z, z0.b, #0",
        "cset {r}, ne",               // Z clear (ANY) → some active lane true
        n = in(reg) C,
        m = in(reg) mask.0.as_ptr(),
        r = out(reg) r,
        out("z0") _, out("p0") _, out("p1") _,
    );
    r != 0
}

/// True if every mask lane is set.
/// True if every mask lane is set. Byte-granular (a set lane is all-`0xFF`, a clear lane all-`0x00`),
/// so it is correct for every element width — `f32`/`f64`/`f16`/`bf16` masks alike.
#[target_feature(enable = "sve")]
pub unsafe fn all_mask<const C: usize>(mask: &SveVec<C>) -> bool {
    let r: u64;
    asm!(
        "whilelt p0.b, xzr, {n}",
        "ld1b {{z0.b}}, p0/z, [{m}]",
        "cmpeq p1.b, p0/z, z0.b, #0", // p1 = byte is *zero* (some lane false)
        "cset {r}, eq",               // Z set (NONE zero) → every lane true
        n = in(reg) C,
        m = in(reg) mask.0.as_ptr(),
        r = out(reg) r,
        out("z0") _, out("p0") _, out("p1") _,
    );
    r != 0
}

/// Horizontal reductions over the active `f32` lanes.
macro_rules! sve1_reduce_f32 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> f32 {
            let bits: u32;
            asm!(
                "whilelt p0.s, xzr, {n}",
                "ld1w {{z0.s}}, p0/z, [{a}]",
                $op,                    // s0 = reduce(z0) — s0 aliases z0's low lane
                "fmov {r:w}, s0",
                n = in(reg) C / 4,
                a = in(reg) a.0.as_ptr(),
                r = out(reg) bits,
                out("z0") _, out("p0") _,
            );
            f32::from_bits(bits)
        }
    };
}

sve1_reduce_f32!(reduce_sum_f32, "faddv s0, p0, z0.s");
sve1_reduce_f32!(reduce_min_f32, "fminv s0, p0, z0.s");
sve1_reduce_f32!(reduce_max_f32, "fmaxv s0, p0, z0.s");

// ─────────────────────────────── f64 (C/8 lanes, `.d`) ───────────────────────────────

/// Broadcast `v` to all `C/8` lanes.
#[target_feature(enable = "sve")]
pub unsafe fn splat_f64<const C: usize>(v: f64) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.d, xzr, {n}",
        "mov z0.d, {bits}",
        "st1d {{z0.d}}, p0, [{o}]",
        n = in(reg) C / 8,
        bits = in(reg) v.to_bits(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Load `C/8` `f64` lanes from `p`.
#[target_feature(enable = "sve")]
pub unsafe fn load_f64<const C: usize>(p: *const f64) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.d, xzr, {n}",
        "ld1d {{z0.d}}, p0/z, [{a}]",
        "st1d {{z0.d}}, p0, [{o}]",
        n = in(reg) C / 8,
        a = in(reg) p,
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Store `C/8` `f64` lanes to `p`.
#[target_feature(enable = "sve")]
pub unsafe fn store_f64<const C: usize>(v: &SveVec<C>, p: *mut f64) {
    asm!(
        "whilelt p0.d, xzr, {n}",
        "ld1d {{z0.d}}, p0/z, [{a}]",
        "st1d {{z0.d}}, p0, [{o}]",
        n = in(reg) C / 8,
        a = in(reg) v.0.as_ptr(),
        o = in(reg) p,
        out("z0") _, out("p0") _,
    );
}

macro_rules! sve1_binop_f64 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.d, xzr, {n}",
                "ld1d {{z0.d}}, p0/z, [{a}]",
                "ld1d {{z1.d}}, p0/z, [{b}]",
                $op,
                "st1d {{z0.d}}, p0, [{o}]",
                n = in(reg) C / 8,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("p0") _,
            );
            o
        }
    };
}

sve1_binop_f64!(add_f64, "fadd z0.d, z0.d, z1.d");
sve1_binop_f64!(sub_f64, "fsub z0.d, z0.d, z1.d");
sve1_binop_f64!(mul_f64, "fmul z0.d, z0.d, z1.d");
sve1_binop_f64!(div_f64, "fdiv z0.d, p0/m, z0.d, z1.d");
sve1_binop_f64!(min_f64, "fmin z0.d, p0/m, z0.d, z1.d");
sve1_binop_f64!(max_f64, "fmax z0.d, p0/m, z0.d, z1.d");

/// `a * b + c`, fused.
#[target_feature(enable = "sve")]
pub unsafe fn fma_f64<const C: usize>(a: &SveVec<C>, b: &SveVec<C>, c: &SveVec<C>) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.d, xzr, {n}",
        "ld1d {{z0.d}}, p0/z, [{c}]",
        "ld1d {{z1.d}}, p0/z, [{a}]",
        "ld1d {{z2.d}}, p0/z, [{b}]",
        "fmla z0.d, p0/m, z1.d, z2.d",
        "st1d {{z0.d}}, p0, [{o}]",
        n = in(reg) C / 8,
        a = in(reg) a.0.as_ptr(),
        b = in(reg) b.0.as_ptr(),
        c = in(reg) c.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("z1") _, out("z2") _, out("p0") _,
    );
    o
}

macro_rules! sve1_unop_f64 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.d, xzr, {n}",
                "ld1d {{z0.d}}, p0/z, [{a}]",
                $op,
                "st1d {{z0.d}}, p0, [{o}]",
                n = in(reg) C / 8,
                a = in(reg) a.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("p0") _,
            );
            o
        }
    };
}

sve1_unop_f64!(neg_f64, "fneg z0.d, p0/m, z0.d");
sve1_unop_f64!(sqrt_f64, "fsqrt z0.d, p0/m, z0.d");

macro_rules! sve1_cmp_f64 {
    ($name:ident, $cmp:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.d, xzr, {n}",
                "ld1d {{z0.d}}, p0/z, [{a}]",
                "ld1d {{z1.d}}, p0/z, [{b}]",
                $cmp,
                "cpy z2.d, p1/z, #-1",
                "st1d {{z2.d}}, p0, [{o}]",
                n = in(reg) C / 8,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
            );
            o
        }
    };
}

sve1_cmp_f64!(le_f64, "fcmle p1.d, p0/z, z0.d, z1.d");
sve1_cmp_f64!(lt_f64, "fcmlt p1.d, p0/z, z0.d, z1.d");
sve1_cmp_f64!(ge_f64, "fcmge p1.d, p0/z, z0.d, z1.d");
sve1_cmp_f64!(gt_f64, "fcmgt p1.d, p0/z, z0.d, z1.d");

/// `mask ? a : b`, lane-wise (mask is `-1`/`0` per 64-bit lane).
#[target_feature(enable = "sve")]
pub unsafe fn select_f64<const C: usize>(
    mask: &SveVec<C>,
    a: &SveVec<C>,
    b: &SveVec<C>,
) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.d, xzr, {n}",
        "ld1d {{z0.d}}, p0/z, [{m}]",
        "cmpne p1.d, p0/z, z0.d, #0",
        "ld1d {{z1.d}}, p0/z, [{a}]",
        "ld1d {{z2.d}}, p0/z, [{b}]",
        "sel z1.d, p1, z1.d, z2.d",
        "st1d {{z1.d}}, p0, [{o}]",
        n = in(reg) C / 8,
        m = in(reg) mask.0.as_ptr(),
        a = in(reg) a.0.as_ptr(),
        b = in(reg) b.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
    );
    o
}

macro_rules! sve1_reduce_f64 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> f64 {
            let bits: u64;
            asm!(
                "whilelt p0.d, xzr, {n}",
                "ld1d {{z0.d}}, p0/z, [{a}]",
                $op,                    // d0 = reduce(z0) — d0 aliases z0's low lane
                "fmov {r}, d0",
                n = in(reg) C / 8,
                a = in(reg) a.0.as_ptr(),
                r = out(reg) bits,
                out("z0") _, out("p0") _,
            );
            f64::from_bits(bits)
        }
    };
}

sve1_reduce_f64!(reduce_sum_f64, "faddv d0, p0, z0.d");
sve1_reduce_f64!(reduce_min_f64, "fminv d0, p0, z0.d");
sve1_reduce_f64!(reduce_max_f64, "fmaxv d0, p0, z0.d");

use half::{bf16, f16};

// ──────────────── f16 native (C/2 lanes, `.h`) ────────────────
//
// Speed over parity: SVE mandates FEAT_FP16, so f16 computes in **native 16-bit lanes** — `C/2` of
// them, twice the f32 lane count — with no widen to f32. `SveVec<C>` for f16 is the raw f16 image,
// load/store are bare `ld1h`/`st1h`, and the ALU is the `.h` SVE ops. Intermediates round at f16
// precision, so results do NOT bit-match the f32-accumulate backends (the deliberate tradeoff).

/// Broadcast `v` to all `C/2` `f16` lanes.
#[target_feature(enable = "sve")]
pub unsafe fn splat_f16<const C: usize>(v: f16) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.h, xzr, {n}",
        "mov z0.h, {bits:w}",
        "st1h {{z0.h}}, p0, [{o}]",
        n = in(reg) C / 2,
        bits = in(reg) v.to_bits() as u32,
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Load `C/2` `f16` lanes from `p` (no conversion).
#[target_feature(enable = "sve")]
pub unsafe fn load_f16<const C: usize>(p: *const f16) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.h, xzr, {n}",
        "ld1h {{z0.h}}, p0/z, [{a}]",
        "st1h {{z0.h}}, p0, [{o}]",
        n = in(reg) C / 2,
        a = in(reg) p,
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Store `C/2` `f16` lanes to `p` (no conversion).
#[target_feature(enable = "sve")]
pub unsafe fn store_f16<const C: usize>(v: &SveVec<C>, p: *mut f16) {
    asm!(
        "whilelt p0.h, xzr, {n}",
        "ld1h {{z0.h}}, p0/z, [{v}]",
        "st1h {{z0.h}}, p0, [{s}]",
        n = in(reg) C / 2,
        v = in(reg) v.0.as_ptr(),
        s = in(reg) p,
        out("z0") _, out("p0") _,
    );
}

macro_rules! sve1_binop_f16 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.h, xzr, {n}",
                "ld1h {{z0.h}}, p0/z, [{a}]",
                "ld1h {{z1.h}}, p0/z, [{b}]",
                $op,
                "st1h {{z0.h}}, p0, [{o}]",
                n = in(reg) C / 2,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("p0") _,
            );
            o
        }
    };
}

sve1_binop_f16!(add_f16, "fadd z0.h, z0.h, z1.h");
sve1_binop_f16!(sub_f16, "fsub z0.h, z0.h, z1.h");
sve1_binop_f16!(mul_f16, "fmul z0.h, z0.h, z1.h");
sve1_binop_f16!(div_f16, "fdiv z0.h, p0/m, z0.h, z1.h");
sve1_binop_f16!(min_f16, "fmin z0.h, p0/m, z0.h, z1.h");
sve1_binop_f16!(max_f16, "fmax z0.h, p0/m, z0.h, z1.h");

/// `a * b + c`, fused, in native f16.
#[target_feature(enable = "sve")]
pub unsafe fn fma_f16<const C: usize>(a: &SveVec<C>, b: &SveVec<C>, c: &SveVec<C>) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.h, xzr, {n}",
        "ld1h {{z0.h}}, p0/z, [{c}]",
        "ld1h {{z1.h}}, p0/z, [{a}]",
        "ld1h {{z2.h}}, p0/z, [{b}]",
        "fmla z0.h, p0/m, z1.h, z2.h",
        "st1h {{z0.h}}, p0, [{o}]",
        n = in(reg) C / 2,
        a = in(reg) a.0.as_ptr(),
        b = in(reg) b.0.as_ptr(),
        c = in(reg) c.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("z1") _, out("z2") _, out("p0") _,
    );
    o
}

macro_rules! sve1_unop_f16 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.h, xzr, {n}",
                "ld1h {{z0.h}}, p0/z, [{a}]",
                $op,
                "st1h {{z0.h}}, p0, [{o}]",
                n = in(reg) C / 2,
                a = in(reg) a.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("p0") _,
            );
            o
        }
    };
}

sve1_unop_f16!(neg_f16, "fneg z0.h, p0/m, z0.h");
sve1_unop_f16!(sqrt_f16, "fsqrt z0.h, p0/m, z0.h");

macro_rules! sve1_cmp_f16 {
    ($name:ident, $cmp:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>, b: &SveVec<C>) -> SveVec<C> {
            let mut o = SveVec::<C>::zeroed();
            asm!(
                "whilelt p0.h, xzr, {n}",
                "ld1h {{z0.h}}, p0/z, [{a}]",
                "ld1h {{z1.h}}, p0/z, [{b}]",
                $cmp,
                "cpy z2.h, p1/z, #-1",
                "st1h {{z2.h}}, p0, [{o}]",
                n = in(reg) C / 2,
                a = in(reg) a.0.as_ptr(),
                b = in(reg) b.0.as_ptr(),
                o = in(reg) o.0.as_mut_ptr(),
                out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
            );
            o
        }
    };
}

sve1_cmp_f16!(le_f16, "fcmle p1.h, p0/z, z0.h, z1.h");
sve1_cmp_f16!(lt_f16, "fcmlt p1.h, p0/z, z0.h, z1.h");
sve1_cmp_f16!(ge_f16, "fcmge p1.h, p0/z, z0.h, z1.h");
sve1_cmp_f16!(gt_f16, "fcmgt p1.h, p0/z, z0.h, z1.h");

/// `mask ? a : b`, lane-wise (mask is `-1`/`0` per 16-bit lane).
#[target_feature(enable = "sve")]
pub unsafe fn select_f16<const C: usize>(
    mask: &SveVec<C>,
    a: &SveVec<C>,
    b: &SveVec<C>,
) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.h, xzr, {n}",
        "ld1h {{z0.h}}, p0/z, [{m}]",
        "cmpne p1.h, p0/z, z0.h, #0",
        "ld1h {{z1.h}}, p0/z, [{a}]",
        "ld1h {{z2.h}}, p0/z, [{b}]",
        "sel z1.h, p1, z1.h, z2.h",
        "st1h {{z1.h}}, p0, [{o}]",
        n = in(reg) C / 2,
        m = in(reg) mask.0.as_ptr(),
        a = in(reg) a.0.as_ptr(),
        b = in(reg) b.0.as_ptr(),
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
    );
    o
}

macro_rules! sve1_reduce_f16 {
    ($name:ident, $op:literal) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> f16 {
            let bits: u32;
            asm!(
                "whilelt p0.h, xzr, {n}",
                "ld1h {{z0.h}}, p0/z, [{a}]",
                $op,                    // h0 = reduce(z0) — h0 aliases z0's low lane
                "fmov {r:w}, h0",
                n = in(reg) C / 2,
                a = in(reg) a.0.as_ptr(),
                r = out(reg) bits,
                out("z0") _, out("p0") _,
            );
            f16::from_bits(bits as u16)
        }
    };
}

sve1_reduce_f16!(reduce_sum_f16, "faddv h0, p0, z0.h");
sve1_reduce_f16!(reduce_min_f16, "fminv h0, p0, z0.h");
sve1_reduce_f16!(reduce_max_f16, "fmaxv h0, p0, z0.h");

// ──────────────── bf16 (storage 16-bit, compute f32 — no native bf16 ALU) ────────────────
//
// Unlike f16, SVE has no native bf16 element-wise ALU (only the fused `BFDOT`/`BFMMLA`/`BFCVT`
// family), so bf16 keeps the widen-compute-narrow model: `SveVec<C>` holds the `f32` image (`C/4`
// lanes) and the arithmetic is the `*_f32` SVE asm above. The widen/narrow is vectorized — load is
// `ld1h {z.s}` + `lsl #16` (a bf16 is the top half of its f32); store is `bfcvt` (round-to-nearest
// -even, matching `bf16::from_f32`). Reuse `add_f32`/`mul_f32`/`fma_f32`/… on the image directly.

/// Widen `C/4` `bf16` lanes from `p` into the `f32` compute image (vectorized).
#[target_feature(enable = "sve")]
pub unsafe fn load_bf16<const C: usize>(p: *const bf16) -> SveVec<C> {
    let mut o = SveVec::<C>::zeroed();
    asm!(
        "whilelt p0.s, xzr, {n}",
        "ld1h {{z0.s}}, p0/z, [{s}]",   // C/4 × 16-bit → low half of each .s lane
        "lsl z0.s, z0.s, #16",          // bf16 (top 16 of f32) → f32
        "st1w {{z0.s}}, p0, [{o}]",
        n = in(reg) C / 4,
        s = in(reg) p,
        o = in(reg) o.0.as_mut_ptr(),
        out("z0") _, out("p0") _,
    );
    o
}

/// Narrow the `f32` compute image back to `C/4` `bf16` lanes in `p` (vectorized, round-to-nearest).
#[target_feature(enable = "sve")]
pub unsafe fn store_bf16<const C: usize>(v: &SveVec<C>, p: *mut bf16) {
    asm!(
        "whilelt p0.s, xzr, {n}",
        "ld1w {{z0.s}}, p0/z, [{v}]",
        ".arch_extension bf16",
        "bfcvt z0.h, p0/m, z0.s",       // f32 → bf16, round-to-nearest-even
        ".arch_extension nobf16",
        "st1h {{z0.s}}, p0, [{s}]",
        n = in(reg) C / 4,
        v = in(reg) v.0.as_ptr(),
        s = in(reg) p,
        out("z0") _, out("p0") _,
    );
}

/// Broadcast a `bf16` scalar (widened) across the `f32` image.
#[target_feature(enable = "sve")]
pub unsafe fn splat_bf16<const C: usize>(v: bf16) -> SveVec<C> {
    splat_f32::<C>(v.to_f32())
}

macro_rules! sve1_reduce_bf16 {
    ($name:ident, $f32op:ident) => {
        #[target_feature(enable = "sve")]
        pub unsafe fn $name<const C: usize>(a: &SveVec<C>) -> bf16 {
            bf16::from_f32($f32op::<C>(a))
        }
    };
}

sve1_reduce_bf16!(reduce_sum_bf16, reduce_sum_f32);
sve1_reduce_bf16!(reduce_min_bf16, reduce_min_f32);
sve1_reduce_bf16!(reduce_max_bf16, reduce_max_f32);
