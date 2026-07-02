//! The execution-context (ISA token) abstraction.
//!
//! A [`Backend<T>`] is a zero-sized token identifying an instruction set (scalar, AVX2,
//! NEON, …, or a GPU subgroup) *for a specific scalar `T`*. The trait is keyed per scalar
//! — rather than carrying a `Vector<T>` GAT — so a hand-written backend can pick the exact
//! intrinsic for the type (`_mm256_add_ps` vs `_mm256_add_pd`): each `(ISA, scalar)` pair
//! is its own concrete impl with concrete [`Backend::Vector`]/[`Backend::Mask`] types. A
//! kernel written against `S: Backend<T>` therefore runs for any `T` on any ISA — the
//! float-agnosticism and the portability come from the same place. The lane count is a
//! `fn` (not a `const`) because the GPU subgroup backend only learns it at runtime.

use crate::scalar::{FloatScalar, IntScalar, Scalar};

/// An instruction-set execution context for scalar `T`. Implemented by [`ScalarBackend`]
/// (every `T`) and, per `(ISA, scalar)`, by the hand-rolled `core::arch` backends.
pub trait Backend<T: Scalar>: Copy {
    /// The varying register holding [`Backend::lanes`] elements of `T`.
    type Vector: Copy;
    /// The boolean mask companion to [`Backend::Vector`].
    type Mask: Copy;

    /// Independent accumulator chains the multi-accumulator reductions (`Gang::reduce`,
    /// `Gang::zip_reduce`, `Gang::count_n`) run to saturate this backend's FP pipes — the ILP
    /// unroll factor, baked in at the dispatch that picks the backend rather than measured per call.
    /// A compile-time constant, so the reduction loops unroll to exactly this many chains with no
    /// runtime `K` lookup. Must not exceed [`MAX_UNROLL`](crate::MAX_UNROLL). The default suits x86's
    /// 2–3 vector pipes; wide cores (Apple NEON, SVE) raise it, the scalar floor drops it to 1.
    const UNROLL: usize = 4;

    /// Number of `T` lanes in one register under this backend.
    fn lanes(self) -> usize;

    fn splat(self, v: T) -> Self::Vector;
    /// Load exactly one register. `s.len()` must equal [`Backend::lanes`].
    fn load(self, s: &[T]) -> Self::Vector;
    /// Store exactly one register. `s.len()` must equal [`Backend::lanes`].
    fn store(self, v: Self::Vector, s: &mut [T]);

    fn add(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn sub(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    fn mul(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    /// Negation: IEEE sign flip for the float elements, wrapping for the integer elements.
    fn neg(self, a: Self::Vector) -> Self::Vector;

    /// Float-family only (`where T: FloatScalar` makes calls with an integer element a compile
    /// error, so the defaults below are statically unreachable — float backends override them,
    /// integer impls never mention them).
    #[inline]
    fn div(self, _a: Self::Vector, _b: Self::Vector) -> Self::Vector
    where
        T: FloatScalar,
    {
        unreachable!("`div` is implemented by every float backend")
    }
    #[inline]
    fn fma(self, _a: Self::Vector, _b: Self::Vector, _c: Self::Vector) -> Self::Vector
    where
        T: FloatScalar,
    {
        unreachable!("`fma` is implemented by every float backend")
    }
    #[inline]
    fn sqrt(self, _a: Self::Vector) -> Self::Vector
    where
        T: FloatScalar,
    {
        unreachable!("`sqrt` is implemented by every float backend")
    }

    /// `a*b + acc` with the element family's natural fusion: the float backends override this
    /// with their fused `fma`, integer elements and the portable default use the two-op
    /// multiply-add (wrapping for ints). Family-neutral — it exists so shared machinery (the
    /// runtime unroll sweep, generic reduction steps) can probe/feed the multiply-add pipes
    /// without a `FloatScalar` bound.
    #[inline]
    fn madd(self, a: Self::Vector, b: Self::Vector, acc: Self::Vector) -> Self::Vector {
        self.add(self.mul(a, b), acc)
    }

    /// Integer-family only: lane-wise shift by a uniform count (`k < 32`). `shr` is
    /// element-appropriate — logical (zero-filling) for `u32`, arithmetic (sign-filling) for
    /// `i32`. Statically unreachable defaults, same scheme as `div`.
    #[inline]
    fn shl(self, _a: Self::Vector, _k: u32) -> Self::Vector
    where
        T: IntScalar,
    {
        unreachable!("`shl` is implemented by every integer backend")
    }
    #[inline]
    fn shr(self, _a: Self::Vector, _k: u32) -> Self::Vector
    where
        T: IntScalar,
    {
        unreachable!("`shr` is implemented by every integer backend")
    }
    /// Integer-family lane-wise bitwise ops.
    #[inline]
    fn bit_and(self, _a: Self::Vector, _b: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        unreachable!("`bit_and` is implemented by every integer backend")
    }
    #[inline]
    fn bit_or(self, _a: Self::Vector, _b: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        unreachable!("`bit_or` is implemented by every integer backend")
    }
    #[inline]
    fn bit_xor(self, _a: Self::Vector, _b: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        unreachable!("`bit_xor` is implemented by every integer backend")
    }
    #[inline]
    fn bit_not(self, _a: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        unreachable!("`bit_not` is implemented by every integer backend")
    }
    /// Absolute value. The default is `max(a, -a)` (two ops); a backend overrides it with a single
    /// dedicated instruction (NEON `fabs`, wasm `f*.abs`, AVX-512 `vabs`) or a sign-bit clear
    /// (x86 `andps` with the `0x7FFF…` mask).
    #[inline]
    fn abs(self, a: Self::Vector) -> Self::Vector {
        self.max(a, self.neg(a))
    }
    /// Lane-wise IEEE 754-2019 minimumNumber: if exactly one operand of a lane is NaN, that lane
    /// takes the *other* operand; NaN comes out only when both are NaN. Which zero wins a
    /// `-0.0`/`+0.0` tie is backend-specific. Every backend implements this contract — natively
    /// where the ISA has it (aarch64 `FMINNM`, SVE `FMINNM`, RVV `vfmin`), with a NaN-patching
    /// fixup where it doesn't (x86 `min` + unord-blend, wasm `pmin` + bitselect).
    fn min(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;
    /// Lane-wise IEEE 754-2019 maximumNumber; see [`min`](Backend::min) for the NaN contract.
    fn max(self, a: Self::Vector, b: Self::Vector) -> Self::Vector;

    fn le(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;
    fn lt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;
    fn ge(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;
    fn gt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask;

    fn mask_and(self, a: Self::Mask, b: Self::Mask) -> Self::Mask;
    fn mask_or(self, a: Self::Mask, b: Self::Mask) -> Self::Mask;
    fn mask_not(self, a: Self::Mask) -> Self::Mask;

    fn select(self, m: Self::Mask, a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Cross-lane: true if any active lane of the mask is set.
    fn any(self, m: Self::Mask) -> bool;
    /// Cross-lane: true if every lane of the mask is set.
    fn all(self, m: Self::Mask) -> bool;

    /// Pack the mask into the low [`lanes`](Backend::lanes) bits of a `u32`: bit `i` set iff lane `i`
    /// is set; bits at and above `lanes()` are zero. Lets a caller `popcount` the set lanes or walk
    /// them by `trailing_zeros` instead of a per-lane scalar scan. The default materializes the mask
    /// through `select`+`store` and packs scalar (no faster than the scan it replaces); the
    /// fixed-width backends override it with a native movemask (x86 `movemask_ps`, AVX-512 k-regs,
    /// NEON shift-and-add, wasm `bitmask`). `lanes()` never exceeds [`MAX_LANES`](crate::MAX_LANES)
    /// (32), so a `u32` always has room.
    #[inline]
    fn mask_bitmask(self, m: Self::Mask) -> u32 {
        let n = self.lanes();
        let ones = self.select(m, self.splat(T::ONE), self.splat(T::ZERO));
        let mut buf = [T::ZERO; crate::MAX_LANES];
        self.store(ones, &mut buf[..n]);
        let mut bits = 0u32;
        let mut i = 0;
        while i < n {
            if buf[i] != T::ZERO {
                bits |= 1 << i;
            }
            i += 1;
        }
        bits
    }

    fn reduce_sum(self, v: Self::Vector) -> T;
    /// Horizontal minimum with [`min`](Backend::min)'s minimumNumber semantics folded pairwise:
    /// NaN lanes are ignored, and the result is NaN only if *every* lane is NaN.
    fn reduce_min(self, v: Self::Vector) -> T;
    /// Horizontal maximum; see [`reduce_min`](Backend::reduce_min).
    fn reduce_max(self, v: Self::Vector) -> T;

    /// The 32-bit integer companion register: `lanes()` lanes of `u32` (reinterpretable as
    /// `i32`) riding alongside the float lanes — lane indices, counters, and bit manipulation of
    /// float lane patterns. Every default below is a correct-but-slow store/compute/reload
    /// round-trip (the [`mask_bitmask`](Backend::mask_bitmask) precedent); backends with native
    /// integer lanes override the ones that matter.
    type IVector: Copy;

    /// Load exactly [`lanes()`](Backend::lanes) integers.
    fn iload(self, s: &[u32]) -> Self::IVector;
    /// Store exactly [`lanes()`](Backend::lanes) integers.
    fn istore(self, v: Self::IVector, out: &mut [u32]);

    #[doc(hidden)]
    #[inline]
    fn i_map(self, a: Self::IVector, f: impl Fn(u32) -> u32) -> Self::IVector {
        let n = self.lanes();
        let mut x = [0u32; crate::MAX_LANES];
        self.istore(a, &mut x[..n]);
        let mut i = 0;
        while i < n {
            x[i] = f(x[i]);
            i += 1;
        }
        self.iload(&x[..n])
    }

    #[doc(hidden)]
    #[inline]
    fn i_zip(self, a: Self::IVector, b: Self::IVector, f: impl Fn(u32, u32) -> u32) -> Self::IVector {
        let n = self.lanes();
        let (mut x, mut y) = ([0u32; crate::MAX_LANES], [0u32; crate::MAX_LANES]);
        self.istore(a, &mut x[..n]);
        self.istore(b, &mut y[..n]);
        let mut i = 0;
        while i < n {
            x[i] = f(x[i], y[i]);
            i += 1;
        }
        self.iload(&x[..n])
    }

    /// Build a [`Mask`](Backend::Mask) from a per-lane integer predicate. The portable default
    /// routes through a `1.0`/`0.0` float image and a `gt` compare; native backends compare the
    /// integer lanes directly (the mask layouts coincide on every fixed-width ISA).
    #[doc(hidden)]
    #[inline]
    fn i_cmp(self, a: Self::IVector, b: Self::IVector, f: impl Fn(u32, u32) -> bool) -> Self::Mask {
        let n = self.lanes();
        let (mut x, mut y) = ([0u32; crate::MAX_LANES], [0u32; crate::MAX_LANES]);
        self.istore(a, &mut x[..n]);
        self.istore(b, &mut y[..n]);
        let mut sel = [T::ZERO; crate::MAX_LANES];
        let mut i = 0;
        while i < n {
            if f(x[i], y[i]) {
                sel[i] = T::ONE;
            }
            i += 1;
        }
        self.gt(self.load(&sel[..n]), self.splat(T::ZERO))
    }

    #[inline]
    fn isplat(self, v: u32) -> Self::IVector {
        let buf = [v; crate::MAX_LANES];
        self.iload(&buf[..self.lanes()])
    }

    /// Lane indices `0, 1, …, lanes()-1`.
    #[inline]
    fn iramp(self) -> Self::IVector {
        let mut buf = [0u32; crate::MAX_LANES];
        let mut i = 0;
        while i < self.lanes() {
            buf[i] = i as u32;
            i += 1;
        }
        self.iload(&buf[..self.lanes()])
    }

    /// Wrapping lane-wise arithmetic, matching SIMD integer instruction semantics.
    #[inline]
    fn iadd(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.i_zip(a, b, u32::wrapping_add)
    }
    #[inline]
    fn isub(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.i_zip(a, b, u32::wrapping_sub)
    }
    /// Low 32 bits of the lane-wise product (`vmul`/`pmulld` semantics — identical for `u32`
    /// and `i32`).
    #[inline]
    fn imul(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.i_zip(a, b, u32::wrapping_mul)
    }
    #[inline]
    fn iand(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.i_zip(a, b, |x, y| x & y)
    }
    #[inline]
    fn ior(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.i_zip(a, b, |x, y| x | y)
    }
    #[inline]
    fn ixor(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.i_zip(a, b, |x, y| x ^ y)
    }
    #[inline]
    fn inot(self, a: Self::IVector) -> Self::IVector {
        self.i_map(a, |x| !x)
    }
    /// Lane-wise shifts by a uniform count; `k` must be `< 32`.
    #[inline]
    fn ishl(self, a: Self::IVector, k: u32) -> Self::IVector {
        debug_assert!(k < 32);
        self.i_map(a, |x| x << k)
    }
    /// Logical (zero-filling) right shift; `k` must be `< 32`.
    #[inline]
    fn ishr(self, a: Self::IVector, k: u32) -> Self::IVector {
        debug_assert!(k < 32);
        self.i_map(a, |x| x >> k)
    }
    /// Arithmetic (sign-filling) right shift, for the `i32` view; `k` must be `< 32`.
    #[inline]
    fn ishr_arith(self, a: Self::IVector, k: u32) -> Self::IVector {
        debug_assert!(k < 32);
        self.i_map(a, |x| ((x as i32) >> k) as u32)
    }

    #[inline]
    fn ieq(self, a: Self::IVector, b: Self::IVector) -> Self::Mask {
        self.i_cmp(a, b, |x, y| x == y)
    }
    /// Unsigned lane-wise `<`.
    #[inline]
    fn ilt_u(self, a: Self::IVector, b: Self::IVector) -> Self::Mask {
        self.i_cmp(a, b, |x, y| x < y)
    }
    /// Signed lane-wise `<` (the `i32` view).
    #[inline]
    fn ilt_s(self, a: Self::IVector, b: Self::IVector) -> Self::Mask {
        self.i_cmp(a, b, |x, y| (x as i32) < (y as i32))
    }

    /// `m ? a : b` on integer lanes, with the same [`Mask`](Backend::Mask) the float compares
    /// produce.
    #[inline]
    fn iselect(self, m: Self::Mask, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        let n = self.lanes();
        let (mut x, mut y) = ([0u32; crate::MAX_LANES], [0u32; crate::MAX_LANES]);
        self.istore(a, &mut x[..n]);
        self.istore(b, &mut y[..n]);
        let bits = self.mask_bitmask(m);
        let mut i = 0;
        while i < n {
            if bits & (1 << i) == 0 {
                x[i] = y[i];
            }
            i += 1;
        }
        self.iload(&x[..n])
    }

    /// Reinterpret each float lane's bit pattern as a `u32` lane — exact for 32-bit `T`, and
    /// free on backends whose integer companion shares the register file. 16-bit `T`
    /// zero-extends; `f64` truncates to the low half (see [`Scalar::to_bits32`]).
    #[inline]
    fn to_bits(self, v: Self::Vector) -> Self::IVector {
        let n = self.lanes();
        let mut f = [T::ZERO; crate::MAX_LANES];
        self.store(v, &mut f[..n]);
        let mut u = [0u32; crate::MAX_LANES];
        let mut i = 0;
        while i < n {
            u[i] = f[i].to_bits32();
            i += 1;
        }
        self.iload(&u[..n])
    }
    /// Inverse of [`to_bits`](Backend::to_bits).
    #[inline]
    #[allow(clippy::wrong_self_convention)] // `self` is the execution token, not the value
    fn from_bits(self, v: Self::IVector) -> Self::Vector {
        let n = self.lanes();
        let mut u = [0u32; crate::MAX_LANES];
        self.istore(v, &mut u[..n]);
        let mut f = [T::ZERO; crate::MAX_LANES];
        let mut i = 0;
        while i < n {
            f[i] = T::from_bits32(u[i]);
            i += 1;
        }
        self.load(&f[..n])
    }
}

/// The always-available 1-lane backend.
///
/// `Vector = T`, `Mask = bool`, for every `T: Scalar`. It is both the correctness oracle
/// for the SIMD backends (math routes through [`Scalar::Compute`] identically) and the
/// natural rust-gpu/SPIR-V lowering target (no data-movement intrinsics, everything
/// scalar).
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarBackend;

impl<T: Scalar> Backend<T> for ScalarBackend {
    type Vector = T;
    type Mask = bool;

    /// One lane, one chain: a scalar add chain has nothing to interleave.
    const UNROLL: usize = 1;

    #[inline(always)]
    fn lanes(self) -> usize {
        1
    }
    #[inline(always)]
    fn splat(self, v: T) -> T {
        v
    }
    #[inline(always)]
    fn load(self, s: &[T]) -> T {
        s[0]
    }
    #[inline(always)]
    fn store(self, v: T, s: &mut [T]) {
        s[0] = v;
    }
    #[inline(always)]
    fn add(self, a: T, b: T) -> T {
        a.wadd(b)
    }
    #[inline(always)]
    fn sub(self, a: T, b: T) -> T {
        a.wsub(b)
    }
    #[inline(always)]
    fn mul(self, a: T, b: T) -> T {
        a.wmul(b)
    }
    #[inline(always)]
    fn div(self, a: T, b: T) -> T
    where
        T: FloatScalar,
    {
        a / b
    }
    #[inline(always)]
    fn neg(self, a: T) -> T {
        a.neg()
    }
    #[inline(always)]
    fn abs(self, a: T) -> T {
        a.abs()
    }
    #[inline(always)]
    fn fma(self, a: T, b: T, c: T) -> T
    where
        T: FloatScalar,
    {
        a.fma(b, c)
    }
    #[inline(always)]
    fn sqrt(self, a: T) -> T
    where
        T: FloatScalar,
    {
        a.sqrt()
    }
    #[inline(always)]
    fn shl(self, a: T, k: u32) -> T
    where
        T: IntScalar,
    {
        a.unsigned_shl(k)
    }
    #[inline(always)]
    fn shr(self, a: T, k: u32) -> T
    where
        T: IntScalar,
    {
        a >> (k as usize)
    }
    #[inline(always)]
    fn bit_and(self, a: T, b: T) -> T
    where
        T: IntScalar,
    {
        a & b
    }
    #[inline(always)]
    fn bit_or(self, a: T, b: T) -> T
    where
        T: IntScalar,
    {
        a | b
    }
    #[inline(always)]
    fn bit_xor(self, a: T, b: T) -> T
    where
        T: IntScalar,
    {
        a ^ b
    }
    #[inline(always)]
    fn bit_not(self, a: T) -> T
    where
        T: IntScalar,
    {
        !a
    }
    #[inline(always)]
    fn min(self, a: T, b: T) -> T {
        // `Scalar::min` is the family-correct oracle: minimumNumber for floats, `Ord` for ints.
        a.min(b)
    }
    #[inline(always)]
    fn max(self, a: T, b: T) -> T {
        a.max(b)
    }
    #[inline(always)]
    fn le(self, a: T, b: T) -> bool {
        a <= b
    }
    #[inline(always)]
    fn lt(self, a: T, b: T) -> bool {
        a < b
    }
    #[inline(always)]
    fn ge(self, a: T, b: T) -> bool {
        a >= b
    }
    #[inline(always)]
    fn gt(self, a: T, b: T) -> bool {
        a > b
    }
    #[inline(always)]
    fn mask_and(self, a: bool, b: bool) -> bool {
        a & b
    }
    #[inline(always)]
    fn mask_or(self, a: bool, b: bool) -> bool {
        a | b
    }
    #[inline(always)]
    fn mask_not(self, a: bool) -> bool {
        !a
    }
    #[inline(always)]
    fn select(self, m: bool, a: T, b: T) -> T {
        if m { a } else { b }
    }
    #[inline(always)]
    fn any(self, m: bool) -> bool {
        m
    }
    #[inline(always)]
    fn all(self, m: bool) -> bool {
        m
    }
    #[inline(always)]
    fn mask_bitmask(self, m: bool) -> u32 {
        m as u32
    }
    #[inline(always)]
    fn reduce_sum(self, v: T) -> T {
        v
    }
    #[inline(always)]
    fn reduce_min(self, v: T) -> T {
        v
    }
    #[inline(always)]
    fn reduce_max(self, v: T) -> T {
        v
    }

    type IVector = u32;
    #[inline(always)]
    fn iload(self, s: &[u32]) -> u32 {
        s[0]
    }
    #[inline(always)]
    fn istore(self, v: u32, out: &mut [u32]) {
        out[0] = v;
    }
}

/// A backend `B` re-stamped with a compile-time unroll factor `K`. Every op delegates to `B`
/// (so codegen is identical after inlining); the only thing it changes is [`UNROLL`](Backend::UNROLL),
/// which becomes the const generic `K`. The dispatch adapter resolves `K` once by runtime detection
/// and wraps the chosen ISA backend in this, so each reduction sees `K` as a constant — no per-call
/// `K` lookup — without `K` having to thread through [`Gang`](crate::Gang) or [`Kernel`](crate::Kernel).
#[cfg(not(any(no_ilp, target_arch = "spirv")))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Unroll<B, const K: usize>(pub(crate) B);

#[cfg(not(any(no_ilp, target_arch = "spirv")))]
impl<T: Scalar, B: Backend<T>, const K: usize> Backend<T> for Unroll<B, K> {
    type Vector = B::Vector;
    type Mask = B::Mask;

    const UNROLL: usize = K;

    #[inline(always)]
    fn lanes(self) -> usize {
        self.0.lanes()
    }
    #[inline(always)]
    fn splat(self, v: T) -> Self::Vector {
        self.0.splat(v)
    }
    #[inline(always)]
    fn load(self, s: &[T]) -> Self::Vector {
        self.0.load(s)
    }
    #[inline(always)]
    fn store(self, v: Self::Vector, s: &mut [T]) {
        self.0.store(v, s)
    }
    #[inline(always)]
    fn add(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.mul(a, b)
    }
    #[inline(always)]
    fn div(self, a: Self::Vector, b: Self::Vector) -> Self::Vector
    where
        T: FloatScalar,
    {
        self.0.div(a, b)
    }
    #[inline(always)]
    fn neg(self, a: Self::Vector) -> Self::Vector {
        self.0.neg(a)
    }
    #[inline(always)]
    fn fma(self, a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector
    where
        T: FloatScalar,
    {
        self.0.fma(a, b, c)
    }
    #[inline(always)]
    fn sqrt(self, a: Self::Vector) -> Self::Vector
    where
        T: FloatScalar,
    {
        self.0.sqrt(a)
    }
    #[inline(always)]
    fn madd(self, a: Self::Vector, b: Self::Vector, acc: Self::Vector) -> Self::Vector {
        self.0.madd(a, b, acc)
    }
    #[inline(always)]
    fn shl(self, a: Self::Vector, k: u32) -> Self::Vector
    where
        T: IntScalar,
    {
        self.0.shl(a, k)
    }
    #[inline(always)]
    fn shr(self, a: Self::Vector, k: u32) -> Self::Vector
    where
        T: IntScalar,
    {
        self.0.shr(a, k)
    }
    #[inline(always)]
    fn bit_and(self, a: Self::Vector, b: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        self.0.bit_and(a, b)
    }
    #[inline(always)]
    fn bit_or(self, a: Self::Vector, b: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        self.0.bit_or(a, b)
    }
    #[inline(always)]
    fn bit_xor(self, a: Self::Vector, b: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        self.0.bit_xor(a, b)
    }
    #[inline(always)]
    fn bit_not(self, a: Self::Vector) -> Self::Vector
    where
        T: IntScalar,
    {
        self.0.bit_not(a)
    }
    #[inline(always)]
    fn abs(self, a: Self::Vector) -> Self::Vector {
        self.0.abs(a)
    }
    #[inline(always)]
    fn min(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.min(a, b)
    }
    #[inline(always)]
    fn max(self, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.max(a, b)
    }
    #[inline(always)]
    fn le(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.le(a, b)
    }
    #[inline(always)]
    fn lt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.lt(a, b)
    }
    #[inline(always)]
    fn ge(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.ge(a, b)
    }
    #[inline(always)]
    fn gt(self, a: Self::Vector, b: Self::Vector) -> Self::Mask {
        self.0.gt(a, b)
    }
    #[inline(always)]
    fn mask_and(self, a: Self::Mask, b: Self::Mask) -> Self::Mask {
        self.0.mask_and(a, b)
    }
    #[inline(always)]
    fn mask_or(self, a: Self::Mask, b: Self::Mask) -> Self::Mask {
        self.0.mask_or(a, b)
    }
    #[inline(always)]
    fn mask_not(self, a: Self::Mask) -> Self::Mask {
        self.0.mask_not(a)
    }
    #[inline(always)]
    fn select(self, m: Self::Mask, a: Self::Vector, b: Self::Vector) -> Self::Vector {
        self.0.select(m, a, b)
    }
    #[inline(always)]
    fn any(self, m: Self::Mask) -> bool {
        self.0.any(m)
    }
    #[inline(always)]
    fn all(self, m: Self::Mask) -> bool {
        self.0.all(m)
    }
    #[inline(always)]
    fn mask_bitmask(self, m: Self::Mask) -> u32 {
        self.0.mask_bitmask(m)
    }
    #[inline(always)]
    fn reduce_sum(self, v: Self::Vector) -> T {
        self.0.reduce_sum(v)
    }
    #[inline(always)]
    fn reduce_min(self, v: Self::Vector) -> T {
        self.0.reduce_min(v)
    }
    #[inline(always)]
    fn reduce_max(self, v: Self::Vector) -> T {
        self.0.reduce_max(v)
    }

    type IVector = B::IVector;
    #[inline(always)]
    fn iload(self, s: &[u32]) -> Self::IVector {
        self.0.iload(s)
    }
    #[inline(always)]
    fn istore(self, v: Self::IVector, out: &mut [u32]) {
        self.0.istore(v, out)
    }
    #[inline(always)]
    fn isplat(self, v: u32) -> Self::IVector {
        self.0.isplat(v)
    }
    #[inline(always)]
    fn iramp(self) -> Self::IVector {
        self.0.iramp()
    }
    #[inline(always)]
    fn iadd(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.iadd(a, b)
    }
    #[inline(always)]
    fn isub(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.isub(a, b)
    }
    #[inline(always)]
    fn imul(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.imul(a, b)
    }
    #[inline(always)]
    fn iand(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.iand(a, b)
    }
    #[inline(always)]
    fn ior(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.ior(a, b)
    }
    #[inline(always)]
    fn ixor(self, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.ixor(a, b)
    }
    #[inline(always)]
    fn inot(self, a: Self::IVector) -> Self::IVector {
        self.0.inot(a)
    }
    #[inline(always)]
    fn ishl(self, a: Self::IVector, k: u32) -> Self::IVector {
        self.0.ishl(a, k)
    }
    #[inline(always)]
    fn ishr(self, a: Self::IVector, k: u32) -> Self::IVector {
        self.0.ishr(a, k)
    }
    #[inline(always)]
    fn ishr_arith(self, a: Self::IVector, k: u32) -> Self::IVector {
        self.0.ishr_arith(a, k)
    }
    #[inline(always)]
    fn ieq(self, a: Self::IVector, b: Self::IVector) -> Self::Mask {
        self.0.ieq(a, b)
    }
    #[inline(always)]
    fn ilt_u(self, a: Self::IVector, b: Self::IVector) -> Self::Mask {
        self.0.ilt_u(a, b)
    }
    #[inline(always)]
    fn ilt_s(self, a: Self::IVector, b: Self::IVector) -> Self::Mask {
        self.0.ilt_s(a, b)
    }
    #[inline(always)]
    fn iselect(self, m: Self::Mask, a: Self::IVector, b: Self::IVector) -> Self::IVector {
        self.0.iselect(m, a, b)
    }
    #[inline(always)]
    fn to_bits(self, v: Self::Vector) -> Self::IVector {
        self.0.to_bits(v)
    }
    #[inline(always)]
    fn from_bits(self, v: Self::IVector) -> Self::Vector {
        self.0.from_bits(v)
    }
}

// The hand-rolled SIMD tokens are crate-internal: application code never names a backend,
// it goes through `dispatch`, which picks one by runtime CPU detection. They stay reachable
// for the in-crate differential tests (`diff_tests`) that verify each against the oracle.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx1;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx2;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx512;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx512bf16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod avx512fp16;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(crate) mod sse4;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "aarch64")]
pub(crate) mod sve;
#[cfg(target_arch = "arm")]
pub(crate) mod neon_a32;
#[cfg(target_arch = "riscv64")]
pub(crate) mod rvv;
#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm;

/// The GPU subgroup backend (SPIR-V) and its portable sequential-vs-subgroup scheduling
/// policy. Public: the `choose` policy compiles and is tested on the CPU; the `Subgroup`
/// backend itself compiles only under `target_arch = "spirv"`, reading the warp width from
/// the hardware `SubgroupSize` builtin.
pub mod subgroup;

#[cfg(test)]
mod diff_tests;
