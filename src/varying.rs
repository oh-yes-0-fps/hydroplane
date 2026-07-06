//! The ergonomic "varying" surface: [`Gang`] is the load/splat context, [`Varying`]/[`Mask`]
//! wrap a whole backend register (the ISPC "varying") with operator overloads, so kernels read
//! like ordinary scalar Rust. Everything is `Copy` and monomorphizes per `(Backend, Scalar)`.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use crate::backend::Backend;
use crate::scalar::{FloatScalar, IntScalar, Scalar};

/// The unroll factor `build.rs` resolved for a `hp_static_dispatch` + pinned-cpu build
/// (`hp_resolved_unroll`). `HP_STATIC_UNROLL` is the decimal `K`; falls back to 4 if unset.
#[cfg(hp_resolved_unroll)]
pub(crate) const STATIC_UNROLL: usize = {
    const fn parse(s: &str) -> usize {
        let b = s.as_bytes();
        let (mut v, mut i) = (0usize, 0usize);
        while i < b.len() {
            v = v * 10 + (b[i] - b'0') as usize;
            i += 1;
        }
        v
    }
    match option_env!("HP_STATIC_UNROLL") {
        Some(s) => parse(s),
        None => 4,
    }
};

/// Chain count for the element-wise loops (`map`/`zip_map`/`map_n`): the backend's
/// [`UNROLL`](Backend::UNROLL) constant, or `1` under `hp_no_ilp`/SPIR-V. Folds to a literal so the
/// inner `while j < k` unrolls to exactly `k` chains.
#[cfg(not(any(hp_no_ilp, target_arch = "spirv")))]
#[inline(always)]
fn unroll_k<T: Scalar, S: Backend<T>>() -> usize {
    <S as Backend<T>>::UNROLL
}
#[cfg(any(hp_no_ilp, target_arch = "spirv"))]
#[inline(always)]
#[allow(clippy::extra_unused_type_parameters)] // `S` keeps the call sites uniform with the ILP variant.
fn unroll_k<T: Scalar, S: Backend<T>>() -> usize {
    1
}

/// `s[..len]` without the bounds check; the checked form's panic branches are measurable in the
/// multi-column maps at small `len`.
///
/// # Safety
/// `len <= s.len()`.
#[inline(always)]
unsafe fn head_unchecked<E>(s: &[E], len: usize) -> &[E] {
    debug_assert!(len <= s.len());
    // SAFETY: caller guarantees `len <= s.len()`.
    unsafe { s.get_unchecked(..len) }
}

/// Mutable [`head_unchecked`].
///
/// # Safety
/// `len <= s.len()`.
#[inline(always)]
unsafe fn head_mut_unchecked<E>(s: &mut [E], len: usize) -> &mut [E] {
    debug_assert!(len <= s.len());
    // SAFETY: caller guarantees `len <= s.len()`.
    unsafe { s.get_unchecked_mut(..len) }
}

/// Chain-count cap for the multi-column maps: roughly `k * live` vector values are in flight, and
/// past the ~32-register file the interleave spills. Cap `k` to the largest power of two keeping
/// the product within budget.
#[inline(always)]
fn chain_cap(live: usize) -> usize {
    let budget = 32 / live.max(1);
    if budget == 0 {
        1
    } else {
        1 << (usize::BITS - 1 - budget.leading_zeros())
    }
}

/// The execution context on backend `S`: the "gang" of lanes that step through the kernel in
/// lockstep (the ISPC term). It is the gateway, not a value; the varying value type is
/// [`Varying`]. You never construct one — it is handed to your
/// [`Kernel::run`](crate::Kernel::run) by `dispatch`, which picks the backend from runtime CPU
/// detection.
///
/// The gang carries no element type: every method is generic over the element it touches, so one
/// kernel can mix `f32` compute, `u32` connectivity, and `f64` accumulation through one context.
/// Value methods infer the element from their arguments; geometry methods
/// ([`lanes`](Self::lanes), [`chunks_exact`](Self::chunks_exact), …) are per-element (lane counts
/// differ by width) and take it explicitly: `ctx.lanes::<f32>()`.
#[derive(Clone, Copy)]
pub struct Gang<S> {
    backend: S,
}

impl<S: Copy> Gang<S> {
    #[inline(always)]
    pub(crate) fn new(backend: S) -> Self {
        Self { backend }
    }

    /// Lanes per register for element `T` on this backend.
    #[inline(always)]
    pub fn lanes<T: Scalar>(self) -> usize
    where
        S: Backend<T>,
    {
        <S as Backend<T>>::lanes(self.backend)
    }

    /// Broadcast a scalar to all lanes.
    #[inline(always)]
    pub fn splat<T: Scalar>(self, v: T) -> Varying<T, S>
    where
        S: Backend<T>,
    {
        Varying::wrap(self.backend, self.backend.splat(v))
    }

    /// Load exactly one register; `s.len()` must equal [`Gang::lanes`] or this panics.
    ///
    /// The length check guards an unchecked SIMD load and folds away at the usual call shapes
    /// (`&a[off..off + n]` under a loop guard). For tails, use [`Gang::load_partial`].
    #[inline(always)]
    pub fn load<T: Scalar>(self, s: &[T]) -> Varying<T, S>
    where
        S: Backend<T>,
    {
        assert!(
            s.len() == self.backend.lanes(),
            "Gang::load: slice length must equal lanes()",
        );
        Varying::wrap(self.backend, self.backend.load(s))
    }

    /// Run `f(offset, count)` over `len` elements in full-register chunks; `count == lanes()`
    /// for every call except a final short tail. Pair with [`Gang::load_partial`] to run a
    /// kernel directly over unpadded, borrowed slices without a padded [`Soa`](crate::Soa).
    ///
    /// This is a two-phase loop, not an iterator with a runtime `count`: `f` inlines once into a
    /// branch-free full-register loop (constant `count`, so bounds checks and the full-vs-tail
    /// branch fold away) and once for the single tail call. `f` cannot early-exit; for
    /// short-circuiting predicates use [`any`](Self::any)/[`zip_any`](Self::zip_any) or a manual
    /// [`chunks_exact`](Self::chunks_exact) + [`remainder`](Self::remainder) loop.
    #[inline]
    pub fn for_each_chunk<T: Scalar>(self, len: usize, mut f: impl FnMut(usize, usize))
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let mut off = 0;
        while off + n <= len {
            f(off, n);
            off += n;
        }
        if off < len {
            f(off, len - off);
        }
    }

    /// Iterate only the full-register prefix of `len`: yields each `offset` (stepping by
    /// [`lanes()`](Self::lanes)) whose chunk is exactly `lanes()` wide. The loop body is
    /// branch-free (see [`ChunksExact`]); handle the short tail once, after the loop, with
    /// [`remainder`](Self::remainder) + [`Gang::load_partial`] (or [`Gang::active_mask`] when a
    /// `min`/`max` would scrub a NaN sentinel):
    ///
    /// ```ignore
    /// let n = ctx.lanes();
    /// for off in ctx.chunks_exact(a.len()) {
    ///     acc = acc + ctx.load(&a[off..off + n]) * ctx.load(&b[off..off + n]);
    /// }
    /// if let Some((off, cnt)) = ctx.remainder(a.len()) {
    ///     let x = ctx.load_partial(&a[off..off + cnt], 0.0);
    ///     let y = ctx.load_partial(&b[off..off + cnt], 0.0);
    ///     acc = acc + x * y;
    /// }
    /// ```
    #[inline]
    pub fn chunks_exact<T: Scalar>(self, len: usize) -> ChunksExact
    where
        S: Backend<T>,
    {
        ChunksExact {
            lanes: self.backend.lanes(),
            pos: 0,
            len,
        }
    }

    /// The tail [`chunks_exact`](Self::chunks_exact) leaves: `Some((offset, count))` with
    /// `0 < count < lanes()`, or `None` when `len` is a multiple of `lanes()`. Also available on
    /// the iterator itself as [`ChunksExact::remainder`].
    #[inline]
    pub fn remainder<T: Scalar>(self, len: usize) -> Option<(usize, usize)>
    where
        S: Backend<T>,
    {
        let cnt = len % self.backend.lanes();
        (cnt != 0).then_some((len - cnt, cnt))
    }

    /// Broadcast one `u32` to every lane of the integer companion register.
    #[inline(always)]
    pub fn splat_u32<T: Scalar>(self, v: u32) -> VaryingU32<T, S>
    where
        S: Backend<T>,
    {
        VaryingU32::wrap(self.backend, self.backend.isplat(v))
    }

    /// Load exactly one integer companion register; `s.len()` must equal `lanes()` or this
    /// panics.
    #[inline(always)]
    pub fn load_u32<T: Scalar>(self, s: &[u32]) -> VaryingU32<T, S>
    where
        S: Backend<T>,
    {
        assert!(
            s.len() == self.backend.lanes(),
            "Gang::load_u32: slice length must equal lanes()",
        );
        VaryingU32::wrap(self.backend, self.backend.iload(s))
    }

    /// The lane indices `0, 1, …, lanes()-1` as integer lanes:
    /// `ctx.ramp_u32() + ctx.splat_u32(off as u32)` is each lane's global element index inside a
    /// chunk loop.
    #[inline(always)]
    pub fn ramp_u32<T: Scalar>(self) -> VaryingU32<T, S>
    where
        S: Backend<T>,
    {
        VaryingU32::wrap(self.backend, self.backend.iramp())
    }

    /// Broadcast one `i32` to every lane of the integer companion (signed view).
    #[inline(always)]
    pub fn splat_i32<T: Scalar>(self, v: i32) -> VaryingI32<T, S>
    where
        S: Backend<T>,
    {
        self.splat_u32(v as u32).as_i32()
    }

    /// Load exactly one signed integer companion register; `s.len()` must equal `lanes()` or
    /// this panics.
    #[inline]
    pub fn load_i32<T: Scalar>(self, s: &[i32]) -> VaryingI32<T, S>
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        assert!(s.len() == n, "Gang::load_i32: slice length must equal lanes()");
        let mut buf = [0u32; crate::MAX_LANES];
        for (b, &x) in buf[..n].iter_mut().zip(s) {
            *b = x as u32;
        }
        VaryingU32::wrap(self.backend, self.backend.iload(&buf[..n])).as_i32()
    }

    /// Reinterpret integer lanes as float lanes; same as [`VaryingU32::to_float_bits`], the
    /// inverse of [`Varying::to_bits`].
    #[inline(always)]
    pub fn from_bits<T: Scalar>(self, v: VaryingU32<T, S>) -> Varying<T, S>
    where
        S: Backend<T>,
    {
        v.to_float_bits()
    }

    /// Load up to [`lanes()`](Gang::lanes) elements from `s` (`s.len()` must not exceed it),
    /// filling the inactive tail lanes with `fill` (e.g. `fill = NaN` keeps the tail out of
    /// distance comparisons and reductions).
    #[inline]
    pub fn load_partial<T: Scalar>(self, s: &[T], fill: T) -> Varying<T, S>
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        debug_assert!(s.len() <= n, "Gang::load_partial: slice longer than lanes()");
        // Tail staging is out-of-line and cold so it never spills registers out of a caller's
        // inner loop.
        if s.len() == n {
            self.load(s)
        } else {
            self.load_tail(s, fill)
        }
    }

    #[cold]
    #[inline(never)]
    fn load_tail<T: Scalar>(self, s: &[T], fill: T) -> Varying<T, S>
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = s.len();
        // Bounded `0..n` loop instead of `copy_from_slice` so DSE can shrink the
        // `[fill; MAX_LANES]` init to the lanes actually loaded once `n` is a constant.
        let mut buf = [fill; crate::MAX_LANES];
        for i in 0..n {
            if i < len {
                // SAFETY: `i < len == s.len()`, and `i < n <= MAX_LANES == buf.len()`.
                unsafe {
                    *buf.get_unchecked_mut(i) = *s.get_unchecked(i);
                }
            }
        }
        self.load(&buf[..n])
    }

    /// The underlying backend token.
    #[inline(always)]
    pub fn backend(self) -> S {
        self.backend
    }

    /// A mask with the first `cnt` lanes active (`true`) and the rest inactive: the general,
    /// always-correct tail handler for a short final chunk or a fixed-`N` batch. Combine it with
    /// [`select`](Varying::select) or the mask algebra (`&`/`|`/`!`) to drop padding lanes from a
    /// result.
    ///
    /// Prefer this to a NaN-filled [`load_partial`](Self::load_partial) whenever a
    /// [`min`](Varying::min)/[`max`](Varying::max) sits between the fill and the compare: those
    /// ops are non-NaN-propagating, so they scrub the poison and let padding contaminate the
    /// reduction.
    ///
    /// `cnt` must not exceed [`lanes()`](Self::lanes). For a concrete backend the ramp folds to a
    /// vector constant and a single compare.
    #[inline]
    pub fn active_mask<T: Scalar>(self, cnt: usize) -> Mask<T, S>
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        debug_assert!(cnt <= n, "Gang::active_mask: cnt must not exceed lanes()");
        let mut ramp = [T::ZERO; crate::MAX_LANES];
        for (i, slot) in ramp[..n].iter_mut().enumerate() {
            *slot = T::from_f64(i as f64);
        }
        self.load(&ramp[..n]).lt(self.splat(T::from_f64(cnt as f64)))
    }

    /// Fold a kernel over one column without writing the loop: full registers at a fixed stride
    /// (bounds-check- and tail-branch-free body), then `f` once more on a tail filled with
    /// `fill`.
    #[inline]
    pub fn fold<T: Scalar, A>(
        self,
        a: &[T],
        fill: T,
        init: A,
        mut f: impl FnMut(A, Varying<T, S>) -> A,
    ) -> A
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len();
        let mut acc = init;
        // `while off + n <= len` keeps the bound live at each load so LLVM drops the bounds checks.
        let mut off = 0;
        while off + n <= len {
            acc = f(acc, self.load(&a[off..off + n]));
            off += n;
        }
        if off < len {
            acc = f(acc, self.load_partial(&a[off..len], fill));
        }
        acc
    }

    /// Two-column [`fold`](Self::fold): `a` and `b` walked in lockstep, the full-register pass bounded
    /// by the shorter (so both loads are provably in bounds), each tail filled with its own sentinel.
    #[inline]
    pub fn zip_fold<T: Scalar, A>(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        init: A,
        mut f: impl FnMut(A, Varying<T, S>, Varying<T, S>) -> A,
    ) -> A
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len());
        let mut acc = init;
        let mut off = 0;
        while off + n <= len {
            let va = self.load(&a[off..off + n]);
            let vb = self.load(&b[off..off + n]);
            acc = f(acc, va, vb);
            off += n;
        }
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            acc = f(acc, va, vb);
        }
        acc
    }

    /// Three-column [`fold`](Self::fold): `a`, `b`, `c` walked in lockstep, the full-register pass
    /// bounded by the shortest, each tail filled with its own sentinel. The natural shape for a
    /// kernel reading three position columns (`x`, `y`, `z`) in one pass.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn zip3_fold<T: Scalar, A>(
        self,
        a: &[T],
        b: &[T],
        c: &[T],
        fill_a: T,
        fill_b: T,
        fill_c: T,
        init: A,
        mut f: impl FnMut(A, Varying<T, S>, Varying<T, S>, Varying<T, S>) -> A,
    ) -> A
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len()).min(c.len());
        let mut acc = init;
        let mut off = 0;
        while off + n <= len {
            let va = self.load(&a[off..off + n]);
            let vb = self.load(&b[off..off + n]);
            let vc = self.load(&c[off..off + n]);
            acc = f(acc, va, vb, vc);
            off += n;
        }
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            let vc = self.load_partial(&c[off..len], fill_c);
            acc = f(acc, va, vb, vc);
        }
        acc
    }

    /// Map a kernel over one column straight into `out` without writing the loop; the elementwise
    /// counterpart of [`fold`](Self::fold), with the same full-register stride and single tail.
    ///
    /// The pass is bounded by the shorter of `a` and `out`. In the tail `f` runs over the
    /// inactive input lanes too; they are computed and then dropped by
    /// [`store_partial`](Varying::store_partial), so `fill` only matters when `f` could fault or
    /// saturate on it (a divide whose padding would be `0`, say).
    #[inline]
    pub fn map<T: Scalar>(self, a: &[T], out: &mut [T], fill: T, mut f: impl FnMut(Varying<T, S>) -> Varying<T, S>)
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(out.len());
        let k = unroll_k::<T, S>();
        let mut off = 0;
        // `k` independent load-f-store groups per step over disjoint memory, so several stay in flight.
        while off + k * n <= len {
            let mut j = 0;
            while j < k {
                let o = off + j * n;
                f(self.load(&a[o..o + n])).store(&mut out[o..o + n]);
                j += 1;
            }
            off += k * n;
        }
        while off + n <= len {
            f(self.load(&a[off..off + n])).store(&mut out[off..off + n]);
            off += n;
        }
        if off < len {
            f(self.load_partial(&a[off..len], fill)).store_partial(&mut out[off..len]);
        }
    }

    /// Two-column [`map`](Self::map): `a` and `b` walked in lockstep into `out`, the pass bounded
    /// by the shortest of the three, each input tail filled with its own sentinel (computed then
    /// dropped by the partial store, as in [`map`](Self::map)).
    #[inline]
    pub fn zip_map<T: Scalar>(
        self,
        a: &[T],
        b: &[T],
        out: &mut [T],
        fill_a: T,
        fill_b: T,
        mut f: impl FnMut(Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    )
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len()).min(out.len());
        let k = unroll_k::<T, S>();
        let mut off = 0;
        while off + k * n <= len {
            let mut j = 0;
            while j < k {
                let o = off + j * n;
                let va = self.load(&a[o..o + n]);
                let vb = self.load(&b[o..o + n]);
                f(va, vb).store(&mut out[o..o + n]);
                j += 1;
            }
            off += k * n;
        }
        while off + n <= len {
            let va = self.load(&a[off..off + n]);
            let vb = self.load(&b[off..off + n]);
            f(va, vb).store(&mut out[off..off + n]);
            off += n;
        }
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            f(va, vb).store_partial(&mut out[off..len]);
        }
    }

    /// In-place two-column map: read a register from `a` (read-only) and from `b`, and write
    /// `f(a_i, b_i)` back to `b`. The in-place sibling of [`zip_map`](Self::zip_map) for updates
    /// whose output aliases an input (`y += a·x`, `y = max(y, x)`), which the borrow checker won't
    /// let you spell as a separate `out: &mut` alongside `b: &`. Same stride, tail handling, and
    /// `K`-chain ILP as [`map`](Self::map).
    #[inline]
    pub fn zip_map_inplace<T: Scalar>(
        self,
        a: &[T],
        b: &mut [T],
        fill_a: T,
        fill_b: T,
        mut f: impl FnMut(Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    )
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len());
        let k = unroll_k::<T, S>();
        let mut off = 0;
        while off + k * n <= len {
            let mut j = 0;
            while j < k {
                let o = off + j * n;
                let va = self.load(&a[o..o + n]);
                let vb = self.load(&b[o..o + n]);
                f(va, vb).store(&mut b[o..o + n]);
                j += 1;
            }
            off += k * n;
        }
        while off + n <= len {
            let va = self.load(&a[off..off + n]);
            let vb = self.load(&b[off..off + n]);
            f(va, vb).store(&mut b[off..off + n]);
            off += n;
        }
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            f(va, vb).store_partial(&mut b[off..len]);
        }
    }

    /// Stream `out[i] = f(a[i])`: a plain bounds-check-free scalar loop, vectorization left to
    /// LLVM. For memory-bandwidth-bound elementwise work (`a·x + b`, clamps, format conversions)
    /// this matches or beats explicit SIMD while skipping its overhead; reach for the
    /// [`map`](Self::map) family when the body is compute-bound or a reduction. Backend-independent:
    /// `self` only carries the element type.
    #[inline]
    pub fn stream_map<T: Scalar>(self, a: &[T], out: &mut [T], mut f: impl FnMut(T) -> T)
    where
        S: Backend<T>,
    {
        for (o, &x) in out.iter_mut().zip(a) {
            *o = f(x);
        }
    }

    /// Stream `out[i] = f(a[i], b[i])`; the [`stream_map`](Self::stream_map) counterpart of
    /// [`zip_map`](Self::zip_map).
    #[inline]
    pub fn stream_zip<T: Scalar>(self, a: &[T], b: &[T], out: &mut [T], mut f: impl FnMut(T, T) -> T)
    where
        S: Backend<T>,
    {
        for (o, (&x, &y)) in out.iter_mut().zip(a.iter().zip(b)) {
            *o = f(x, y);
        }
    }

    /// Stream `b[i] = f(a[i], b[i])` in place; the [`stream_map`](Self::stream_map) counterpart
    /// of [`zip_map_inplace`](Self::zip_map_inplace), for updates like `y += a·x`.
    #[inline]
    pub fn stream_zip_inplace<T: Scalar>(self, a: &[T], b: &mut [T], mut f: impl FnMut(T, T) -> T)
    where
        S: Backend<T>,
    {
        for (bi, &x) in b.iter_mut().zip(a) {
            *bi = f(x, *bi);
        }
    }

    /// In-place `N`-column element-wise transform: load one register from each column, hand the
    /// `[Varying; N]` lane-tuple to `f`, and write its `[Varying; N]` result back to the same
    /// columns. The multi-channel companion to [`map`](Self::map) for kernels where every output
    /// channel depends on every input channel, e.g. an SoA point transform. All columns must be
    /// the same length; each channel's inactive tail is filled with `fill` for the load and
    /// dropped by the partial store. Same `K`-chain ILP unrolling as [`map`](Self::map).
    #[inline]
    pub fn map_n<T: Scalar, const N: usize>(
        self,
        cols: [&mut [T]; N],
        fill: T,
        mut f: impl FnMut([Varying<T, S>; N]) -> [Varying<T, S>; N],
    )
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = cols.iter().map(|c| c.len()).min().unwrap_or(0);
        // Same exact-`len` re-slice as `map_cols` so the unrolled body's bounds checks fold away.
        // SAFETY: `len` is the minimum over these same columns' lengths.
        let cols: [&mut [T]; N] = cols.map(|c| unsafe { head_mut_unchecked(c, len) });
        let k = unroll_k::<T, S>().min(chain_cap(2 * N));
        let mut off = 0;
        while off + k * n <= len {
            let mut j = 0;
            while j < k {
                let o = off + j * n;
                let rs = f(core::array::from_fn(|c| self.load(&cols[c][o..o + n])));
                for c in 0..N {
                    rs[c].store(&mut cols[c][o..o + n]);
                }
                j += 1;
            }
            off += k * n;
        }
        while off + n <= len {
            let rs = f(core::array::from_fn(|c| self.load(&cols[c][off..off + n])));
            for c in 0..N {
                rs[c].store(&mut cols[c][off..off + n]);
            }
            off += n;
        }
        if off < len {
            let rs = f(core::array::from_fn(|c| self.load_partial(&cols[c][off..len], fill)));
            for c in 0..N {
                rs[c].store_partial(&mut cols[c][off..len]);
            }
        }
    }

    /// Asymmetric element-wise map: load one register from each of `IN` input columns, hand the
    /// `[Varying; IN]` lane-tuple to `f`, and write its `[Varying; OUT]` result to `OUT` distinct
    /// output columns. The general form of [`map`](Self::map)/[`zip_map`](Self::zip_map) for
    /// kernels whose output arity differs from the input: a batched `M·v` (nine matrix + three
    /// vector columns to three), a complex multiply (four to two). The pass is bounded by the
    /// shortest column, the tail is a single masked step, and it carries the same `K`-chain ILP
    /// as [`map`](Self::map). Inputs and outputs are distinct slices; for the in-place
    /// same-columns case use [`map_n`](Self::map_n).
    #[inline]
    pub fn map_cols<T: Scalar, const IN: usize, const OUT: usize>(
        self,
        inp: [&[T]; IN],
        out: [&mut [T]; OUT],
        fill: T,
        mut f: impl FnMut([Varying<T, S>; IN]) -> [Varying<T, S>; OUT],
    )
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = inp
            .iter()
            .map(|c| c.len())
            .chain(out.iter().map(|c| c.len()))
            .min()
            .unwrap_or(0);
        // Re-slice every column to exactly `len` so each access checks the loop guard's own bound;
        // otherwise LLVM keeps IN+OUT bounds checks per step.
        // SAFETY: `len` is the minimum over these same columns' lengths.
        let inp: [&[T]; IN] = inp.map(|c| unsafe { head_unchecked(c, len) });
        let out: [&mut [T]; OUT] = out.map(|c| unsafe { head_mut_unchecked(c, len) });
        let k = unroll_k::<T, S>().min(chain_cap(IN + OUT));
        let mut off = 0;
        while off + k * n <= len {
            let mut j = 0;
            while j < k {
                let o = off + j * n;
                let rs = f(core::array::from_fn(|c| self.load(&inp[c][o..o + n])));
                for c in 0..OUT {
                    rs[c].store(&mut out[c][o..o + n]);
                }
                j += 1;
            }
            off += k * n;
        }
        while off + n <= len {
            let rs = f(core::array::from_fn(|c| self.load(&inp[c][off..off + n])));
            for c in 0..OUT {
                rs[c].store(&mut out[c][off..off + n]);
            }
            off += n;
        }
        if off < len {
            let rs = f(core::array::from_fn(|c| self.load_partial(&inp[c][off..len], fill)));
            for c in 0..OUT {
                rs[c].store_partial(&mut out[c][off..len]);
            }
        }
    }

    /// Short-circuiting `any`: `true` as soon as some lane in some register satisfies `pred`.
    /// Tail via [`load_partial`](Self::load_partial).
    ///
    /// `fill` must be a value `pred` rejects, so the padding in the final partial register can
    /// never spuriously trip the result (e.g. `f32::NEG_INFINITY` for an `x > y` test). The
    /// opposite of [`all`](Self::all), whose fill must be accepted.
    #[inline]
    pub fn any<T: Scalar>(self, a: &[T], fill: T, mut pred: impl FnMut(Varying<T, S>) -> Mask<T, S>) -> bool
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len();
        let mut off = 0;
        while off + n <= len {
            if pred(self.load(&a[off..off + n])).any() {
                return true;
            }
            off += n;
        }
        off < len && pred(self.load_partial(&a[off..len], fill)).any()
    }

    /// Short-circuiting `all`: `false` as soon as some lane fails `pred`, else `true` (vacuously
    /// so for an empty slice). Tail via [`load_partial`](Self::load_partial).
    ///
    /// `fill` must be a value `pred` accepts, the mirror of [`any`](Self::any), so the padding of
    /// the final partial register cannot spuriously fail the check (for an `x <= hi` test, fill
    /// the `x` tail with `hi` or below).
    #[inline]
    pub fn all<T: Scalar>(self, a: &[T], fill: T, mut pred: impl FnMut(Varying<T, S>) -> Mask<T, S>) -> bool
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len();
        let mut off = 0;
        while off + n <= len {
            if !pred(self.load(&a[off..off + n])).all() {
                return false;
            }
            off += n;
        }
        off >= len || pred(self.load_partial(&a[off..len], fill)).all()
    }

    /// Two-column [`any`](Self::any): `true` as soon as a register pair satisfies `pred`. Pass
    /// bounded by the shorter column; each tail filled with a sentinel `pred` rejects (see
    /// [`any`](Self::any)).
    #[inline]
    pub fn zip_any<T: Scalar>(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        mut pred: impl FnMut(Varying<T, S>, Varying<T, S>) -> Mask<T, S>,
    ) -> bool
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len());
        let mut off = 0;
        while off + n <= len {
            if pred(self.load(&a[off..off + n]), self.load(&b[off..off + n])).any() {
                return true;
            }
            off += n;
        }
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            return pred(va, vb).any();
        }
        false
    }

    /// Two-column [`all`](Self::all): `false` as soon as a register pair fails `pred`, else
    /// `true`. Pass bounded by the shorter column; each tail filled with a sentinel `pred`
    /// accepts (see [`all`](Self::all)).
    #[inline]
    pub fn zip_all<T: Scalar>(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        mut pred: impl FnMut(Varying<T, S>, Varying<T, S>) -> Mask<T, S>,
    ) -> bool
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len());
        let mut off = 0;
        while off + n <= len {
            if !pred(self.load(&a[off..off + n]), self.load(&b[off..off + n])).all() {
                return false;
            }
            off += n;
        }
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            return pred(va, vb).all();
        }
        true
    }

    /// `N`-column [`any`](Self::any): `true` as soon as some lane satisfies `pred`, over `N`
    /// columns loaded in lockstep. Unlike [`any`](Self::any)/[`zip_any`](Self::zip_any), no
    /// sentinel fill is needed: the tail loads with `T::ZERO` and
    /// [`active_mask`](Self::active_mask) drops the inactive lanes, so it stays correct even when
    /// no value exists that `pred` rejects (e.g. a plane test whose normal may point either way).
    /// All columns are assumed the same length (`cols[0].len()`).
    #[inline]
    pub fn any_n<T: Scalar, const N: usize>(
        self,
        cols: [&[T]; N],
        mut pred: impl FnMut([Varying<T, S>; N]) -> Mask<T, S>,
    ) -> bool
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = if N == 0 { 0 } else { cols[0].len() };
        let mut off = 0;
        while off + n <= len {
            let vs = core::array::from_fn(|j| self.load(&cols[j][off..off + n]));
            if pred(vs).any() {
                return true;
            }
            off += n;
        }
        if off < len {
            let cnt = len - off;
            let vs = core::array::from_fn(|j| self.load_partial(&cols[j][off..len], T::ZERO));
            return (pred(vs) & self.active_mask(cnt)).any();
        }
        false
    }

    /// `N`-column [`all`](Self::all): the mirror of [`any_n`](Self::any_n). Inactive tail lanes
    /// are forced true (via `!active_mask`), so no sentinel fill is needed. All columns are
    /// assumed the same length (`cols[0].len()`).
    #[inline]
    pub fn all_n<T: Scalar, const N: usize>(
        self,
        cols: [&[T]; N],
        mut pred: impl FnMut([Varying<T, S>; N]) -> Mask<T, S>,
    ) -> bool
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = if N == 0 { 0 } else { cols[0].len() };
        let mut off = 0;
        while off + n <= len {
            let vs = core::array::from_fn(|j| self.load(&cols[j][off..off + n]));
            if !pred(vs).all() {
                return false;
            }
            off += n;
        }
        if off < len {
            let cnt = len - off;
            let vs = core::array::from_fn(|j| self.load_partial(&cols[j][off..len], T::ZERO));
            return (pred(vs) | !self.active_mask(cnt)).all();
        }
        true
    }

    /// `N`-column count: how many lanes across the whole column set satisfy `pred`. The tallying
    /// sibling of [`any_n`](Self::any_n)/[`all_n`](Self::all_n); inactive tail lanes are dropped
    /// via [`active_mask`](Self::active_mask), so no sentinel fill is needed. All columns are
    /// assumed the same length (`cols[0].len()`).
    ///
    /// A full count is a loop-carried add chain, so this uses the same `K`-independent-chain ILP
    /// as [`reduce`](Self::reduce); small inputs and `--cfg hp_no_ilp` builds stay single-chain.
    #[inline]
    #[cfg(not(any(hp_no_ilp, target_arch = "spirv")))]
    pub fn count_n<T: Scalar, const N: usize>(
        self,
        cols: [&[T]; N],
        mut pred: impl FnMut([Varying<T, S>; N]) -> Mask<T, S>,
    ) -> usize
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = if N == 0 { 0 } else { cols[0].len() };
        // `S::UNROLL` is a compile-time constant, so the `while j < k` loops unroll to exactly
        // `k` chains. `K == 1` and too-small inputs take the single-chain fold.
        let k = <S as Backend<T>>::UNROLL;
        if k == 1 || len / n < 8 {
            return self.count_n_fold(cols, pred).reduce_sum().into_f64() as usize;
        }
        let one = self.splat(T::ONE);
        let zero = self.splat(T::ZERO);
        let mut acc = [zero; crate::MAX_UNROLL];
        let mut off = 0;
        while off + k * n <= len {
            // Reborrow a `k*n`-wide window per column so each `[o..o + n]` is in bounds by
            // constants alone; drops the per-chain bounds checks.
            let w: [&[T]; N] = core::array::from_fn(|c| &cols[c][off..off + k * n]);
            let mut j = 0;
            while j < k {
                let o = j * n;
                let vs = core::array::from_fn(|c| self.load(&w[c][o..o + n]));
                acc[j] = acc[j] + one.select(pred(vs), zero);
                j += 1;
            }
            off += k * n;
        }
        // Leftover full chunks go to distinct chains, not all into `acc[0]`, to keep the latency
        // chain broken.
        let mut j = 0;
        while off + n <= len {
            let vs = core::array::from_fn(|c| self.load(&cols[c][off..off + n]));
            acc[j] = acc[j] + one.select(pred(vs), zero);
            off += n;
            j += 1;
        }
        let mut width = k;
        while width > 1 {
            let half = width / 2;
            let mut j = 0;
            while j < half {
                acc[j] = acc[j] + acc[width - half + j];
                j += 1;
            }
            width -= half;
        }
        let mut result = acc[0];
        if off < len {
            let cnt = len - off;
            let vs = core::array::from_fn(|c| self.load_partial(&cols[c][off..len], T::ZERO));
            let mask = pred(vs) & self.active_mask(cnt);
            result = result + one.select(mask, zero);
        }
        result.reduce_sum().into_f64() as usize
    }

    /// ILP compiled out: single-accumulator chain only.
    #[inline]
    #[cfg(any(hp_no_ilp, target_arch = "spirv"))]
    pub fn count_n<T: Scalar, const N: usize>(
        self,
        cols: [&[T]; N],
        pred: impl FnMut([Varying<T, S>; N]) -> Mask<T, S>,
    ) -> usize
    where
        S: Backend<T>,
    {
        self.count_n_fold(cols, pred).reduce_sum().into_f64() as usize
    }

    /// Single-chain count accumulator (the shared tail/small-input path). Returns per-lane
    /// partial sums for the caller to reduce once; a per-chunk horizontal `reduce_sum` would put
    /// a cross-lane add in the hot loop.
    #[inline]
    fn count_n_fold<T: Scalar, const N: usize>(
        self,
        cols: [&[T]; N],
        mut pred: impl FnMut([Varying<T, S>; N]) -> Mask<T, S>,
    ) -> Varying<T, S>
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let len = if N == 0 { 0 } else { cols[0].len() };
        let one = self.splat(T::ONE);
        let zero = self.splat(T::ZERO);
        let mut acc = zero;
        let mut off = 0;
        while off + n <= len {
            let vs = core::array::from_fn(|j| self.load(&cols[j][off..off + n]));
            acc = acc + one.select(pred(vs), zero);
            off += n;
        }
        if off < len {
            let cnt = len - off;
            let vs = core::array::from_fn(|j| self.load_partial(&cols[j][off..len], T::ZERO));
            let mask = pred(vs) & self.active_mask(cnt);
            acc = acc + one.select(mask, zero);
        }
        acc
    }

    /// `N`-column hit visitor: like [`any_n`](Self::any_n), but instead of short-circuiting it
    /// calls `on_hit(index)` for every lane (`index` in `0..cols[0].len()`) where `pred` holds,
    /// in order. Returns whether any lane matched. Inactive tail lanes are masked out via
    /// [`active_mask`](Self::active_mask), so no sentinel fill is needed. All columns are assumed
    /// the same length.
    #[inline]
    pub fn for_each_hit_n<T: Scalar, const N: usize>(
        self,
        cols: [&[T]; N],
        mut pred: impl FnMut([Varying<T, S>; N]) -> Mask<T, S>,
        mut on_hit: impl FnMut(usize),
    ) -> bool
    where
        S: Backend<T>,
    {
        let len = if N == 0 { 0 } else { cols[0].len() };
        let mut any = false;
        for (off, cnt, active) in self.masked_chunks(len) {
            let vs = core::array::from_fn(|j| self.load_partial(&cols[j][off..off + cnt], T::ZERO));
            // `trailing_zeros` + clear walks only the set lanes: one step per hit, no full scan.
            let mut bits = (pred(vs) & active).to_bitmask();
            any |= bits != 0;
            while bits != 0 {
                on_hit(off + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        any
    }

    /// Per-chunk [`active_mask`](Self::active_mask) alongside the full-register walk: yields
    /// `(offset, count, active)` per step, `count == lanes()` (all-active mask) for every chunk
    /// except a final short tail. For kernels that need the tail mask across several reductions
    /// while keeping their own control flow, which a single-predicate [`any_n`](Self::any_n)
    /// can't express.
    #[inline]
    pub fn masked_chunks<T: Scalar>(self, len: usize) -> impl Iterator<Item = (usize, usize, Mask<T, S>)>
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        self.chunks_exact(len)
            .map(move |off| (off, n, self.active_mask(n)))
            .chain(
                self.remainder(len)
                    .map(|(off, cnt)| (off, cnt, self.active_mask(cnt))),
            )
    }

    /// Splat each element of a fixed-size array to its own [`Varying`]; the multi-channel
    /// companion to [`splat`](Self::splat): `let [cx, cy, cz] = ctx.splat_n([q[0], q[1], q[2]]);`.
    #[inline]
    pub fn splat_n<T: Scalar, const N: usize>(self, vals: [T; N]) -> [Varying<T, S>; N]
    where
        S: Backend<T>,
    {
        core::array::from_fn(|i| self.splat(vals[i]))
    }

    /// [`load`](Self::load) one full register from each of `N` columns. Every slice must be
    /// exactly [`lanes()`](Self::lanes) long.
    #[inline]
    pub fn load_n<T: Scalar, const N: usize>(self, cols: [&[T]; N]) -> [Varying<T, S>; N]
    where
        S: Backend<T>,
    {
        core::array::from_fn(|i| self.load(cols[i]))
    }

    /// [`load_partial`](Self::load_partial) up to one register from each of `N` columns, filling
    /// the inactive tail of every channel with `fill`:
    /// `let [x, y, z] = ctx.load_partial_n([&xs[r], &ys[r], &zs[r]], 0.0);`.
    #[inline]
    pub fn load_partial_n<T: Scalar, const N: usize>(self, cols: [&[T]; N], fill: T) -> [Varying<T, S>; N]
    where
        S: Backend<T>,
    {
        core::array::from_fn(|i| self.load_partial(cols[i], fill))
    }

    /// Gather up to one register's worth of rows from an array-of-structures slice into `N`
    /// column [`Varying`]s, via a caller-supplied row extractor. `items.len()` must not exceed
    /// [`lanes()`](Self::lanes); `extract` maps each element to its `N` field values, and column
    /// `c`'s inactive tail lanes are filled with `fills[c]`.
    ///
    /// Per-column fills let a kernel pick a sentinel that makes inactive lanes self-reject (e.g.
    /// a radius of `NaN`, so a distance compare on the tail is always false), avoiding an
    /// [`active_mask`](Self::active_mask) `&` after the predicate.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn gather_n<T: Scalar, E, const N: usize>(
        self,
        items: &[E],
        fills: [T; N],
        mut extract: impl FnMut(&E) -> [T; N],
    ) -> [Varying<T, S>; N]
    where
        S: Backend<T>,
    {
        let n = self.backend.lanes();
        let cnt = items.len();
        debug_assert!(cnt <= n, "Gang::gather_n: more rows than lanes()");
        // Stage one register per column and fill inactive lanes in place: one transpose pass plus
        // one full-width load, no second staging copy through `load_partial`.
        let mut scratch = [[MaybeUninit::<T>::uninit(); crate::MAX_LANES]; N];
        for (i, item) in items.iter().enumerate() {
            let row = extract(item);
            for c in 0..N {
                scratch[c][i] = MaybeUninit::new(row[c]);
            }
        }
        if cnt < n {
            for c in 0..N {
                for slot in &mut scratch[c][cnt..n] {
                    *slot = MaybeUninit::new(fills[c]);
                }
            }
        }
        core::array::from_fn(|c| {
            // SAFETY: lanes `0..cnt` were written from `items` and `cnt..n` from `fills` above.
            let lane = unsafe { core::slice::from_raw_parts(scratch[c].as_ptr().cast::<T>(), n) };
            self.load(lane)
        })
    }

    /// Two-column multi-accumulator reduction across `K` independent chains, so a wide
    /// out-of-order core keeps one FMA in flight per pipe instead of stalling on a single
    /// latency-bound chain. `K` is `S::UNROLL`, the per-core saturation factor the dispatch
    /// adapter resolved; a balanced tree folds the chains at the end. `K == 1` and inputs too
    /// small to amortize the tree take the single-chain [`zip_fold`](Self::zip_fold). `step` is
    /// the per-chain combinator (use [`Varying::fma`] for a dot/AXPY-style update); `combine`
    /// folds two chains.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(any(hp_no_ilp, target_arch = "spirv")))]
    pub fn zip_reduce<T: Scalar, A: Copy, FS, FC>(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        init: A,
        step: FS,
        combine: FC,
    ) -> A
    where
        S: Backend<T>,
        FS: Fn(A, Varying<T, S>, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len());
        let k = <S as Backend<T>>::UNROLL;
        if k == 1 || len / n < 8 {
            return self.zip_fold(a, b, fill_a, fill_b, init, step);
        }
        let mut acc = [init; crate::MAX_UNROLL];
        let mut off = 0;
        // The guard keeps the window load in bounds, and the `k*n`-wide reborrow makes each
        // per-chain `[o..o + n]` in bounds by constants alone, so no bounds checks survive.
        while off + k * n <= len {
            let aw = &a[off..off + k * n];
            let bw = &b[off..off + k * n];
            let mut j = 0;
            while j < k {
                let o = j * n;
                acc[j] = step(acc[j], self.load(&aw[o..o + n]), self.load(&bw[o..o + n]));
                j += 1;
            }
            off += k * n;
        }
        // Leftover full registers go to distinct chains, not all into `acc[0]`, to keep the
        // latency chain broken.
        let mut j = 0;
        while off + n <= len {
            acc[j] = step(acc[j], self.load(&a[off..off + n]), self.load(&b[off..off + n]));
            off += n;
            j += 1;
        }
        let mut width = k;
        while width > 1 {
            let half = width / 2;
            let mut j = 0;
            while j < half {
                acc[j] = combine(acc[j], acc[width - half + j]);
                j += 1;
            }
            width -= half;
        }
        let mut result = acc[0];
        if off < len {
            let va = self.load_partial(&a[off..len], fill_a);
            let vb = self.load_partial(&b[off..len], fill_b);
            result = step(result, va, vb);
        }
        result
    }

    /// ILP compiled out: straight to the single-chain [`zip_fold`](Self::zip_fold); `combine` is
    /// inert.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(any(hp_no_ilp, target_arch = "spirv"))]
    pub fn zip_reduce<T: Scalar, A: Copy, FS, FC>(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        init: A,
        step: FS,
        combine: FC,
    ) -> A
    where
        S: Backend<T>,
        FS: Fn(A, Varying<T, S>, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let _ = combine;
        self.zip_fold(a, b, fill_a, fill_b, init, step)
    }

    /// Single-column counterpart of [`zip_reduce`](Self::zip_reduce): `K` independent chains over
    /// one slice (sum, norm, max-style kernels). Same loop discipline and tail handling; `K == 1`
    /// and tiny inputs take the single-chain [`fold`](Self::fold).
    #[inline]
    #[cfg(not(any(hp_no_ilp, target_arch = "spirv")))]
    pub fn reduce<T: Scalar, A: Copy, FS, FC>(self, a: &[T], fill: T, init: A, step: FS, combine: FC) -> A
    where
        S: Backend<T>,
        FS: Fn(A, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let n = self.backend.lanes();
        let len = a.len();
        let k = <S as Backend<T>>::UNROLL;
        if k == 1 || len / n < 8 {
            return self.fold(a, fill, init, step);
        }
        let mut acc = [init; crate::MAX_UNROLL];
        let mut off = 0;
        while off + k * n <= len {
            let aw = &a[off..off + k * n];
            let mut j = 0;
            while j < k {
                let o = j * n;
                acc[j] = step(acc[j], self.load(&aw[o..o + n]));
                j += 1;
            }
            off += k * n;
        }
        let mut j = 0;
        while off + n <= len {
            acc[j] = step(acc[j], self.load(&a[off..off + n]));
            off += n;
            j += 1;
        }
        let mut width = k;
        while width > 1 {
            let half = width / 2;
            let mut j = 0;
            while j < half {
                acc[j] = combine(acc[j], acc[width - half + j]);
                j += 1;
            }
            width -= half;
        }
        let mut result = acc[0];
        if off < len {
            result = step(result, self.load_partial(&a[off..len], fill));
        }
        result
    }

    /// ILP compiled out: straight to the single-chain [`fold`](Self::fold); `combine` is inert.
    #[inline]
    #[cfg(any(hp_no_ilp, target_arch = "spirv"))]
    pub fn reduce<T: Scalar, A: Copy, FS, FC>(self, a: &[T], fill: T, init: A, step: FS, combine: FC) -> A
    where
        S: Backend<T>,
        FS: Fn(A, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let _ = combine;
        self.fold(a, fill, init, step)
    }

    /// Sum `step` over two columns, fully reduced to a scalar. `0` serves as the accumulator
    /// seed, the tail fill for both columns, and the chain-combine identity, and the `K`-chain
    /// ILP of [`zip_reduce`](Self::zip_reduce) is applied automatically:
    ///
    /// ```ignore
    /// fn dot(ctx: Gang<f32>, a: &[f32], b: &[f32]) -> f32 {
    ///     ctx.zip_sum(a, b, |acc, x, y| x.fma(y, acc))
    /// }
    /// ```
    ///
    /// `step` is the per-register update (use [`Varying::fma`]). For a non-sum reduction
    /// (max/min/any) reach for [`zip_reduce`](Self::zip_reduce) with an explicit identity and
    /// combine.
    #[inline]
    pub fn zip_sum<T: Scalar, F>(self, a: &[T], b: &[T], step: F) -> T
    where
        S: Backend<T>,
        F: Fn(Varying<T, S>, Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    {
        self.zip_reduce(a, b, T::ZERO, T::ZERO, self.splat(T::ZERO), step, |p, q| p + q)
            .reduce_sum()
    }

    /// Single-column [`zip_sum`](Self::zip_sum): sum `step` over one column to a scalar.
    /// `ctx.sum(a, |acc, x| x.fma(x, acc))` is `‖a‖²`.
    #[inline]
    pub fn sum<T: Scalar, F>(self, a: &[T], step: F) -> T
    where
        S: Backend<T>,
        F: Fn(Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    {
        self.reduce(a, T::ZERO, self.splat(T::ZERO), step, |p, q| p + q)
            .reduce_sum()
    }

    /// Plain sum `Σ a[i]`: [`sum`](Self::sum) with a lane-wise add. Named to sidestep the
    /// closure-taking [`sum`](Self::sum).
    #[inline]
    pub fn total<T: Scalar>(self, a: &[T]) -> T
    where
        S: Backend<T>,
    {
        self.sum(a, |acc, x| acc + x)
    }

    /// Dot product `Σ a[i]·b[i]`: the [`zip_sum`](Self::zip_sum) FMA collapsed to one call,
    /// bounded by the shorter column.
    #[inline]
    pub fn dot<T: FloatScalar>(self, a: &[T], b: &[T]) -> T
    where
        S: Backend<T>,
    {
        self.zip_sum(a, b, |acc, x, y| x.fma(y, acc))
    }

    /// Squared L2 norm `Σ a[i]²`. Prefer it to [`norm`](Self::norm) when the squared magnitude is
    /// enough (a distance comparison), to skip the `sqrt`.
    #[inline]
    pub fn norm_sq<T: FloatScalar>(self, a: &[T]) -> T
    where
        S: Backend<T>,
    {
        self.sum(a, |acc, x| x.fma(x, acc))
    }

    /// L2 norm `√(Σ a[i]²)`: [`norm_sq`](Self::norm_sq) and a single scalar `sqrt`.
    #[inline]
    pub fn norm<T: FloatScalar>(self, a: &[T]) -> T
    where
        S: Backend<T>,
    {
        self.norm_sq(a).sqrt()
    }

    /// The cached unroll factor for this core, resolved on first use. The scalar backend
    /// (`lanes() == 1`) opts out: multiple chains buy it nothing. Also read by the matrix
    /// micro-kernel to size its register block.
    #[inline]
    #[cfg(all(not(hp_no_ilp), not(target_arch = "spirv"), not(hp_resolved_unroll)))]
    pub(crate) fn unroll(self) -> usize
    where
        S: Backend<f32>,
    {
        if <S as Backend<f32>>::lanes(self.backend) == 1 {
            return 1;
        }
        match crate::ilp::cached() {
            0 => self.detect_unroll(),
            k => k as usize,
        }
    }

    /// The cached factor as seen for element `T`; the matrix micro-kernel's block-sizing view.
    /// Falls back to the backend's static [`UNROLL`](Backend::UNROLL) before the sweep has run.
    #[inline]
    #[cfg(all(not(hp_no_ilp), not(target_arch = "spirv"), not(hp_resolved_unroll)))]
    pub(crate) fn unroll_for<T: Scalar>(self) -> usize
    where
        S: Backend<T>,
    {
        if <S as Backend<T>>::lanes(self.backend) == 1 {
            return 1;
        }
        match crate::ilp::cached() {
            0 => <S as Backend<T>>::UNROLL,
            k => k as usize,
        }
    }

    #[inline(always)]
    #[cfg(hp_resolved_unroll)]
    pub(crate) fn unroll_for<T: Scalar>(self) -> usize
    where
        S: Backend<T>,
    {
        if <S as Backend<T>>::lanes(self.backend) == 1 {
            return 1;
        }
        STATIC_UNROLL
    }

    #[inline(always)]
    #[cfg(any(hp_no_ilp, target_arch = "spirv"))]
    pub(crate) fn unroll_for<T: Scalar>(self) -> usize
    where
        S: Backend<T>,
    {
        1
    }

    /// Build-resolved (`hp_static_dispatch` + pinned cpu): the constant `build.rs` baked into
    /// [`STATIC_UNROLL`], no atomic and no startup sweep. The scalar backend still opts out.
    #[inline(always)]
    #[cfg(hp_resolved_unroll)]
    pub(crate) fn unroll(self) -> usize
    where
        S: Backend<f32>,
    {
        if <S as Backend<f32>>::lanes(self.backend) == 1 {
            return 1;
        }
        STATIC_UNROLL
    }

    /// ILP compiled out (`--cfg hp_no_ilp` / SPIR-V): one chain, no atomic and no startup sweep.
    #[inline(always)]
    #[cfg(any(hp_no_ilp, target_arch = "spirv"))]
    pub(crate) fn unroll(self) -> usize
    where
        S: Backend<f32>,
    {
        1
    }

    /// Resolve the unroll factor once and cache it: time a fixed-buffer dot at each candidate
    /// factor `{1,2,4,8,12,16}` and pick the fastest. Cold and out-of-line; runs at most once per
    /// process.
    #[cfg(all(feature = "std", not(hp_no_ilp), not(target_arch = "spirv"), not(hp_resolved_unroll)))]
    #[cold]
    #[inline(never)]
    fn detect_unroll(self) -> usize
    where
        S: Backend<f32>,
    {
        use std::hint::black_box;
        use std::time::Instant;

        use crate::backend::Unroll;

        let probe: std::vec::Vec<f32> = (0..4096)
            .map(|i| (i % 17) as f32 * 0.5 - 4.0)
            .collect();
        let a = probe.as_slice();
        let zero = 0.0f32;
        let b = self.backend();

        // Wrap the backend in `Unroll<S, $k>` so `zip_reduce` takes its `$k`-chain path, the same
        // code dispatch would run at that factor. Closures are rebuilt per `$k` because their
        // `Varying` is over the wrapped backend type.
        macro_rules! time_k {
            ($k:literal, $iters:expr) => {{
                let g = Gang::new(Unroll::<S, $k>(b));
                let init = g.splat(zero);
                let mut best = u64::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    let mut sink = 0.0f64;
                    for _ in 0..$iters {
                        let r = g.zip_reduce(
                            black_box(a),
                            black_box(a),
                            zero,
                            zero,
                            init,
                            |acc, x, y| x.madd(y, acc),
                            |p, q| p + q,
                        );
                        sink += r.reduce_sum().into_f64();
                    }
                    black_box(sink);
                    let e = t.elapsed().as_nanos() as u64;
                    if e < best {
                        best = e;
                    }
                }
                best
            }};
        }

        let one_ns = time_k!(1, 1).max(1);
        // ~0.5 ms per timed run amortizes `Instant` overhead; bound the count both ways.
        let iters = (500_000u64 / one_ns).clamp(1, 100_000) as u32;

        let cands = [
            (1u8, time_k!(1, iters)),
            (2u8, time_k!(2, iters)),
            (4u8, time_k!(4, iters)),
            (8u8, time_k!(8, iters)),
            (12u8, time_k!(12, iters)),
            (16u8, time_k!(16, iters)),
        ];
        let mut best = cands[0];
        for &c in &cands[1..] {
            if c.1 < best.1 {
                best = c;
            }
        }
        crate::ilp::store(best.0);
        best.0 as usize
    }

    /// No-std build: no timer for a sweep, so fall back to a per-target default near each
    /// family's `latency × pipes` saturation point (wide NEON FP wants more chains than x86's
    /// 2-3 vector pipes).
    #[cfg(all(not(feature = "std"), not(hp_no_ilp), not(target_arch = "spirv"), not(hp_resolved_unroll)))]
    #[cold]
    #[inline(never)]
    fn detect_unroll(self) -> usize
    where
        S: Backend<f32>,
    {
        let _ = self;
        let k: u8 = if cfg!(target_arch = "aarch64") {
            8
        } else if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
            4
        } else {
            1
        };
        crate::ilp::store(k);
        k as usize
    }
}

/// Full-register-only chunk iterator produced by [`Gang::chunks_exact`]. Yields each `offset`
/// whose chunk is exactly `lanes()` wide, stepping by `lanes()`, and stops before any short
/// tail; pick that up once, after the loop, via [`remainder`](ChunksExact::remainder) (or
/// [`Gang::remainder`]).
///
/// `next` tests `offset + lanes <= len`, the same guard a hand-written full-register `while`
/// loop carries, so after inlining the body's slice bounds checks fold away.
#[derive(Clone, Copy, Debug)]
pub struct ChunksExact {
    lanes: usize,
    pos: usize,
    len: usize,
}

impl ChunksExact {
    /// The tail the full-register pass leaves: `Some((offset, count))` with
    /// `0 < count < lanes()`, or `None` when `len` divides evenly. Independent of iteration
    /// progress, so it can be read before, during, or after the loop.
    #[inline]
    pub fn remainder(self) -> Option<(usize, usize)> {
        let cnt = self.len % self.lanes;
        (cnt != 0).then_some((self.len - cnt, cnt))
    }
}

impl Iterator for ChunksExact {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.pos + self.lanes <= self.len {
            let off = self.pos;
            self.pos += self.lanes;
            Some(off)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos.min(self.len)) / self.lanes;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ChunksExact {}

/// A varying value: `lanes()` elements of `T`, on backend `S`.
#[derive(Clone, Copy)]
pub struct Varying<T: Scalar, S: Backend<T>> {
    backend: S,
    _t: PhantomData<T>,
    v: S::Vector,
}

impl<T: Scalar, S: Backend<T>> Varying<T, S> {
    #[inline(always)]
    fn wrap(backend: S, v: S::Vector) -> Self {
        Self {
            backend,
            v,
            _t: PhantomData,
        }
    }

    /// The raw backend register.
    #[inline(always)]
    pub fn raw(self) -> S::Vector {
        self.v
    }

    /// Store this register; `out.len()` must equal `lanes()` or this panics.
    ///
    /// The length check guards an unchecked SIMD store and folds away at the usual
    /// provable-length call shapes (see [`Gang::load`]). For tails, use
    /// [`Varying::store_partial`].
    #[inline(always)]
    pub fn store(self, out: &mut [T]) {
        assert!(
            out.len() == self.backend.lanes(),
            "Varying::store: slice length must equal lanes()",
        );
        self.backend.store(self.v, out)
    }

    /// Store the first `out.len()` lanes (must not exceed `lanes()`) into `out`. The companion
    /// to [`Gang::load_partial`] for writing results back into a borrowed, unpadded column.
    #[inline]
    pub fn store_partial(self, out: &mut [T]) {
        let n = self.backend.lanes();
        debug_assert!(out.len() <= n, "Varying::store_partial: slice longer than lanes()");
        if out.len() == n {
            self.backend.store(self.v, out);
            return;
        }
        let mut buf = [T::ZERO; crate::MAX_LANES];
        self.backend.store(self.v, &mut buf[..n]);
        out.copy_from_slice(&buf[..out.len()]);
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self
    where
        T: FloatScalar,
    {
        Self::wrap(self.backend, self.backend.sqrt(self.v))
    }
    /// Lane-wise reciprocal `1/self`, full-precision (an IEEE divide, not a fast `rcp` estimate).
    #[inline(always)]
    pub fn recip(self) -> Self
    where
        T: FloatScalar,
    {
        let one = self.backend.splat(T::ONE);
        Self::wrap(self.backend, self.backend.div(one, self.v))
    }
    /// Absolute value. Backends with a dedicated abs instruction or a sign-bit clear use it; the
    /// rest fall back to `max(self, -self)`. `abs(NaN)` is NaN on every backend, so `abs` alone
    /// never breaks a NaN-poisoned tail; only an intervening [`min`](Self::min)/[`max`](Self::max)
    /// against a non-NaN does.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::wrap(self.backend, self.backend.abs(self.v))
    }
    /// Lane-wise IEEE 754-2019 minimumNumber, identically on every backend: a lane with exactly
    /// one NaN operand takes the other operand; NaN comes out only when both operands are NaN.
    /// Which zero wins a `-0.0`/`+0.0` tie is backend-specific; don't build logic on it.
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self::wrap(self.backend, self.backend.min(self.v, o.v))
    }
    /// Lane-wise IEEE 754-2019 maximumNumber, with the same NaN rule as [`min`](Self::min).
    /// Sharp edge: a NaN-poisoned tail (a `load_partial` NaN fill) is always scrubbed when a
    /// `min`/`max` against a non-NaN operand sits between the fill and the compare, letting
    /// padding leak into the reduction. Use [`Gang::active_mask`](crate::Gang::active_mask)
    /// whenever such an op intervenes.
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self::wrap(self.backend, self.backend.max(self.v, o.v))
    }
    /// `self * b + c`, fused where the backend supports it.
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self
    where
        T: FloatScalar,
    {
        Self::wrap(self.backend, self.backend.fma(self.v, b.v, c.v))
    }

    /// `self * b + acc` for any element family: fused on the float backends (identical to
    /// [`fma`](Self::fma) there), wrapping two-op multiply-add on the integer elements.
    #[inline(always)]
    pub fn madd(self, b: Self, acc: Self) -> Self {
        Self::wrap(self.backend, self.backend.madd(self.v, b.v, acc.v))
    }

    /// Each lane's bit pattern as an integer-companion lane; free on backends whose integer
    /// lanes share the register file. Exact for 32-bit `T`; see [`Scalar::to_bits32`] for the
    /// 16/64-bit story. Inverse: [`VaryingU32::to_float_bits`] / [`Gang::from_bits`].
    #[inline(always)]
    pub fn to_bits(self) -> VaryingU32<T, S> {
        VaryingU32::wrap(self.backend, self.backend.to_bits(self.v))
    }

    #[inline(always)]
    pub fn le(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.le(self.v, o.v))
    }
    #[inline(always)]
    pub fn lt(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.lt(self.v, o.v))
    }
    #[inline(always)]
    pub fn ge(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.ge(self.v, o.v))
    }
    #[inline(always)]
    pub fn gt(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.gt(self.v, o.v))
    }

    /// `mask ? self : other`, lane-wise.
    #[inline(always)]
    pub fn select(self, mask: Mask<T, S>, other: Self) -> Self {
        Self::wrap(self.backend, self.backend.select(mask.m, self.v, other.v))
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> T {
        self.backend.reduce_sum(self.v)
    }
    /// Horizontal minimum with [`min`](Self::min)'s minimumNumber rule folded across the lanes:
    /// NaN lanes are ignored; the result is NaN only if every lane is NaN.
    #[inline(always)]
    pub fn reduce_min(self) -> T {
        self.backend.reduce_min(self.v)
    }
    /// Horizontal maximum; see [`reduce_min`](Self::reduce_min).
    #[inline(always)]
    pub fn reduce_max(self) -> T {
        self.backend.reduce_max(self.v)
    }
}

macro_rules! lane_binop {
    ($trait:ident, $method:ident, $bk:ident) => {
        impl<T: Scalar, S: Backend<T>> $trait for Varying<T, S> {
            type Output = Varying<T, S>;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Varying::wrap(self.backend, self.backend.$bk(self.v, rhs.v))
            }
        }
    };
}
lane_binop!(Add, add, add);
lane_binop!(Sub, sub, sub);
lane_binop!(Mul, mul, mul);
impl<T: FloatScalar, S: Backend<T>> Div for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Varying::wrap(self.backend, self.backend.div(self.v, rhs.v))
    }
}

impl<T: Scalar, S: Backend<T>> Neg for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn neg(self) -> Self {
        Varying::wrap(self.backend, self.backend.neg(self.v))
    }
}

/// Scalar on the right-hand side: `v * 2.0`, `v + bias` splat the scalar and apply the op. Only
/// this direction works; the orphan rule forbids `impl Mul<Varying> for f32`.
macro_rules! varying_scalar_binop {
    ($trait:ident, $method:ident, $bk:ident) => {
        impl<T: Scalar, S: Backend<T>> $trait<T> for Varying<T, S> {
            type Output = Varying<T, S>;
            #[inline(always)]
            fn $method(self, rhs: T) -> Self {
                let r = self.backend.splat(rhs);
                Varying::wrap(self.backend, self.backend.$bk(self.v, r))
            }
        }
    };
}
varying_scalar_binop!(Add, add, add);
varying_scalar_binop!(Sub, sub, sub);
varying_scalar_binop!(Mul, mul, mul);
impl<T: FloatScalar, S: Backend<T>> Div<T> for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn div(self, rhs: T) -> Self {
        let r = self.backend.splat(rhs);
        Varying::wrap(self.backend, self.backend.div(self.v, r))
    }
}

/// Integer-element lane-wise shift by a uniform count (`k < 32`); `>>` is logical for `u32`,
/// arithmetic for `i32`.
impl<T: IntScalar, S: Backend<T>> core::ops::Shl<u32> for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn shl(self, k: u32) -> Self {
        Varying::wrap(self.backend, self.backend.shl(self.v, k))
    }
}
impl<T: IntScalar, S: Backend<T>> core::ops::Shr<u32> for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn shr(self, k: u32) -> Self {
        Varying::wrap(self.backend, self.backend.shr(self.v, k))
    }
}
impl<T: IntScalar, S: Backend<T>> BitAnd for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Varying::wrap(self.backend, self.backend.bit_and(self.v, rhs.v))
    }
}
impl<T: IntScalar, S: Backend<T>> BitOr for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Varying::wrap(self.backend, self.backend.bit_or(self.v, rhs.v))
    }
}
impl<T: IntScalar, S: Backend<T>> BitXor for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Varying::wrap(self.backend, self.backend.bit_xor(self.v, rhs.v))
    }
}
impl<T: IntScalar, S: Backend<T>> Not for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn not(self) -> Self {
        Varying::wrap(self.backend, self.backend.bit_not(self.v))
    }
}

/// A varying boolean mask companion to [`Varying`].
#[derive(Clone, Copy)]
pub struct Mask<T: Scalar, S: Backend<T>> {
    backend: S,
    m: S::Mask,
    _t: PhantomData<T>,
}

impl<T: Scalar, S: Backend<T>> Mask<T, S> {
    #[inline(always)]
    fn wrap(backend: S, m: S::Mask) -> Self {
        Self {
            backend,
            m,
            _t: PhantomData,
        }
    }
    /// The raw backend mask.
    #[inline(always)]
    pub fn raw(self) -> S::Mask {
        self.m
    }
    /// True if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        self.backend.any(self.m)
    }
    /// True if every lane is set.
    #[inline(always)]
    pub fn all(self) -> bool {
        self.backend.all(self.m)
    }

    /// The set lanes packed into the low [`lanes`](Gang::lanes) bits of a `u32`: bit `i` set iff
    /// lane `i` is set; bits at and above `lanes()` are zero. `count_ones()` gives an exact
    /// set-lane count and `trailing_zeros()` the first set lane.
    #[inline(always)]
    pub fn to_bitmask(self) -> u32 {
        self.backend.mask_bitmask(self.m)
    }
}

impl<T: Scalar, S: Backend<T>> BitAnd for Mask<T, S> {
    type Output = Mask<T, S>;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Mask::wrap(self.backend, self.backend.mask_and(self.m, rhs.m))
    }
}
impl<T: Scalar, S: Backend<T>> BitOr for Mask<T, S> {
    type Output = Mask<T, S>;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Mask::wrap(self.backend, self.backend.mask_or(self.m, rhs.m))
    }
}
impl<T: Scalar, S: Backend<T>> Not for Mask<T, S> {
    type Output = Mask<T, S>;
    #[inline(always)]
    fn not(self) -> Self {
        Mask::wrap(self.backend, self.backend.mask_not(self.m))
    }
}

/// The 32-bit unsigned integer companion register: [`lanes()`](Gang::lanes) lanes of `u32`
/// riding alongside a gang's float lanes, for lane indices ([`Gang::ramp_u32`]), counters, and
/// the integer half of float bit tricks ([`Varying::to_bits`]). Arithmetic is wrapping, matching
/// SIMD integer instructions. Compares produce the same [`Mask`] the float compares do, so
/// float- and integer-derived conditions compose freely (`&`, `|`, [`select`](Self::select)).
#[derive(Clone, Copy)]
pub struct VaryingU32<T: Scalar, S: Backend<T>> {
    backend: S,
    v: S::IVector,
    _t: PhantomData<T>,
}

/// The signed view of the integer companion: same register, arithmetic (sign-filling) right
/// shift and signed compares. Convert freely with [`VaryingU32::as_i32`]/[`VaryingI32::as_u32`]
/// (bit-identical, free).
#[derive(Clone, Copy)]
pub struct VaryingI32<T: Scalar, S: Backend<T>> {
    backend: S,
    v: S::IVector,
    _t: PhantomData<T>,
}

impl<T: Scalar, S: Backend<T>> VaryingU32<T, S> {
    #[inline(always)]
    fn wrap(backend: S, v: S::IVector) -> Self {
        Self { backend, v, _t: PhantomData }
    }
    /// The raw backend integer register.
    #[inline(always)]
    pub fn raw(self) -> S::IVector {
        self.v
    }
    /// Store this register; `out.len()` must equal `lanes()` or this panics.
    #[inline(always)]
    pub fn store(self, out: &mut [u32]) {
        assert!(
            out.len() == self.backend.lanes(),
            "VaryingU32::store: slice length must equal lanes()",
        );
        self.backend.istore(self.v, out)
    }
    /// Reinterpret as the signed view (free).
    #[inline(always)]
    pub fn as_i32(self) -> VaryingI32<T, S> {
        VaryingI32 { backend: self.backend, v: self.v, _t: PhantomData }
    }
    /// Reinterpret each lane's bits as the gang's float element; the inverse of
    /// [`Varying::to_bits`]. Exact for 32-bit `T`; see [`Scalar::from_bits32`] for the
    /// 16/64-bit story.
    #[inline(always)]
    pub fn to_float_bits(self) -> Varying<T, S> {
        Varying::wrap(self.backend, self.backend.from_bits(self.v))
    }
    #[inline(always)]
    pub fn eq(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.ieq(self.v, o.v))
    }
    #[inline(always)]
    pub fn ne(self, o: Self) -> Mask<T, S> {
        !self.eq(o)
    }
    /// Unsigned lane-wise `<`.
    #[inline(always)]
    pub fn lt(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.ilt_u(self.v, o.v))
    }
    #[inline(always)]
    pub fn gt(self, o: Self) -> Mask<T, S> {
        o.lt(self)
    }
    #[inline(always)]
    pub fn le(self, o: Self) -> Mask<T, S> {
        !o.lt(self)
    }
    #[inline(always)]
    pub fn ge(self, o: Self) -> Mask<T, S> {
        !self.lt(o)
    }
    /// `mask ? self : other`, lane-wise, driven by the same [`Mask`] the float compares produce.
    #[inline(always)]
    pub fn select(self, mask: Mask<T, S>, other: Self) -> Self {
        Self::wrap(self.backend, self.backend.iselect(mask.m, self.v, other.v))
    }
}

impl<T: Scalar, S: Backend<T>> VaryingI32<T, S> {
    /// The raw backend integer register.
    #[inline(always)]
    pub fn raw(self) -> S::IVector {
        self.v
    }
    /// Store this register; `out.len()` must equal `lanes()` or this panics.
    #[inline(always)]
    pub fn store(self, out: &mut [i32]) {
        let n = self.backend.lanes();
        assert!(out.len() == n, "VaryingI32::store: slice length must equal lanes()");
        let mut buf = [0u32; crate::MAX_LANES];
        self.backend.istore(self.v, &mut buf[..n]);
        for (o, &b) in out.iter_mut().zip(&buf[..n]) {
            *o = b as i32;
        }
    }
    /// Reinterpret as the unsigned view (free).
    #[inline(always)]
    pub fn as_u32(self) -> VaryingU32<T, S> {
        VaryingU32 { backend: self.backend, v: self.v, _t: PhantomData }
    }
    #[inline(always)]
    pub fn eq(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.ieq(self.v, o.v))
    }
    #[inline(always)]
    pub fn ne(self, o: Self) -> Mask<T, S> {
        !self.eq(o)
    }
    /// Signed lane-wise `<`.
    #[inline(always)]
    pub fn lt(self, o: Self) -> Mask<T, S> {
        Mask::wrap(self.backend, self.backend.ilt_s(self.v, o.v))
    }
    #[inline(always)]
    pub fn gt(self, o: Self) -> Mask<T, S> {
        o.lt(self)
    }
    #[inline(always)]
    pub fn le(self, o: Self) -> Mask<T, S> {
        !o.lt(self)
    }
    #[inline(always)]
    pub fn ge(self, o: Self) -> Mask<T, S> {
        !self.lt(o)
    }
    /// `mask ? self : other`, lane-wise.
    #[inline(always)]
    pub fn select(self, mask: Mask<T, S>, other: Self) -> Self {
        Self { backend: self.backend, v: self.backend.iselect(mask.m, self.v, other.v), _t: PhantomData }
    }
}

macro_rules! int_binop {
    ($ty:ident, $trait:ident, $method:ident, $bk:ident) => {
        impl<T: Scalar, S: Backend<T>> $trait for $ty<T, S> {
            type Output = $ty<T, S>;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self { backend: self.backend, v: self.backend.$bk(self.v, rhs.v), _t: PhantomData }
            }
        }
    };
}

int_binop!(VaryingU32, Add, add, iadd);
int_binop!(VaryingU32, Sub, sub, isub);
int_binop!(VaryingU32, Mul, mul, imul);
int_binop!(VaryingU32, BitAnd, bitand, iand);
int_binop!(VaryingU32, BitOr, bitor, ior);
int_binop!(VaryingU32, BitXor, bitxor, ixor);
int_binop!(VaryingI32, Add, add, iadd);
int_binop!(VaryingI32, Sub, sub, isub);
int_binop!(VaryingI32, Mul, mul, imul);
int_binop!(VaryingI32, BitAnd, bitand, iand);
int_binop!(VaryingI32, BitOr, bitor, ior);
int_binop!(VaryingI32, BitXor, bitxor, ixor);

impl<T: Scalar, S: Backend<T>> Not for VaryingU32<T, S> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self { backend: self.backend, v: self.backend.inot(self.v), _t: PhantomData }
    }
}
impl<T: Scalar, S: Backend<T>> Not for VaryingI32<T, S> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self { backend: self.backend, v: self.backend.inot(self.v), _t: PhantomData }
    }
}
/// Lane-wise shift by a uniform count (`k < 32`).
impl<T: Scalar, S: Backend<T>> core::ops::Shl<u32> for VaryingU32<T, S> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, k: u32) -> Self {
        Self { backend: self.backend, v: self.backend.ishl(self.v, k), _t: PhantomData }
    }
}
/// Logical (zero-filling) right shift by a uniform count (`k < 32`).
impl<T: Scalar, S: Backend<T>> core::ops::Shr<u32> for VaryingU32<T, S> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, k: u32) -> Self {
        Self { backend: self.backend, v: self.backend.ishr(self.v, k), _t: PhantomData }
    }
}
/// Lane-wise shift by a uniform count (`k < 32`).
impl<T: Scalar, S: Backend<T>> core::ops::Shl<u32> for VaryingI32<T, S> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, k: u32) -> Self {
        Self { backend: self.backend, v: self.backend.ishl(self.v, k), _t: PhantomData }
    }
}
/// Arithmetic (sign-filling) right shift by a uniform count (`k < 32`).
impl<T: Scalar, S: Backend<T>> core::ops::Shr<u32> for VaryingI32<T, S> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, k: u32) -> Self {
        Self { backend: self.backend, v: self.backend.ishr_arith(self.v, k), _t: PhantomData }
    }
}
