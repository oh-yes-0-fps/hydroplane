//! The ergonomic "varying" surface — Layer 2.
//!
//! Backends speak in raw method calls (`backend.add(a, b)`). This module wraps a backend
//! register in a [`Varying`] — a whole register of `lanes()` elements (the ISPC "varying" to a
//! scalar's "uniform") — that carries the backend token, so kernels read like ordinary scalar Rust:
//!
//! ```ignore
//! let d  = ctx.load(xs) - ctx.splat(cx);   // operator-overloaded
//! let d2 = d * d + dy * dy + dz * dz;       // looks scalar, is SIMD
//! if (d2.le(r * r)).any() { return true }   // cross-lane reduce
//! ```
//!
//! [`Gang`] is the *context* you load/splat through (it produces `Varying`s); [`Varying`] and
//! [`Mask`] are the varying values. Everything is `Copy`, zero-sized except the register payload,
//! and monomorphizes per `(Backend, Scalar)` — the ergonomics cost nothing at runtime.

use core::marker::PhantomData;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Sub};

use crate::backend::Backend;
use crate::scalar::Scalar;

/// The execution *context* for scalar `T` on backend `S` — the "gang" of lanes that step through
/// the kernel together (the ISPC term for the group of program instances running in lockstep). It is
/// the gateway, not a value: the varying value type is [`Varying`]. You never construct one — it is
/// handed to your [`Kernel::run`](crate::Kernel::run) by `dispatch`, which picks the backend from
/// runtime CPU detection. Build varying values (`splat`, `load`) through it.
#[derive(Clone, Copy)]
pub struct Gang<T: Scalar, S: Backend<T>> {
    backend: S,
    _t: PhantomData<T>,
}

impl<T: Scalar, S: Backend<T>> Gang<T, S> {
    #[inline(always)]
    pub(crate) fn new(backend: S) -> Self {
        Self {
            backend,
            _t: PhantomData,
        }
    }

    /// Lanes per register for this `(backend, scalar)`.
    #[inline(always)]
    pub fn lanes(self) -> usize {
        self.backend.lanes()
    }

    /// Broadcast a scalar to all lanes.
    #[inline(always)]
    pub fn splat(self, v: T) -> Varying<T, S> {
        Varying::wrap(self.backend, self.backend.splat(v))
    }

    /// Load exactly one register; `s.len()` must equal [`Gang::lanes`].
    ///
    /// The backend reads exactly `lanes()` elements from `s` with an unchecked SIMD load: passing a
    /// slice of any other length is undefined behaviour. The length match is the caller's contract,
    /// checked only under `debug_assertions`. For tails, use [`Gang::load_partial`] or the
    /// [`Gang::chunks`] iterator.
    #[inline(always)]
    pub fn load(self, s: &[T]) -> Varying<T, S> {
        debug_assert!(
            s.len() == self.backend.lanes(),
            "Gang::load: slice length must equal lanes()",
        );
        Varying::wrap(self.backend, self.backend.load(s))
    }

    /// Iterate `len` elements in full-register chunks, yielding `(offset, count)` per step.
    /// `count == lanes()` for every chunk except possibly the last. Pair with
    /// [`Gang::load_partial`] to run a kernel directly over unpadded, borrowed slices (e.g.
    /// the field slices of a `soa-rs` struct) — no copy and no padded [`Soa`](crate::Soa).
    #[inline]
    pub fn chunks(self, len: usize) -> Chunks {
        Chunks {
            lanes: self.backend.lanes(),
            pos: 0,
            len,
        }
    }

    /// Load up to [`lanes()`](Gang::lanes) elements from `s` (`s.len()` must not exceed it),
    /// filling the inactive tail lanes with `fill`. A full chunk is a plain [`load`](Gang::load);
    /// a short tail is staged through a stack buffer so the inactive lanes carry the sentinel
    /// (so e.g. `fill = NaN` keeps the tail out of distance comparisons and reductions).
    #[inline]
    pub fn load_partial(self, s: &[T], fill: T) -> Varying<T, S> {
        let n = self.backend.lanes();
        debug_assert!(s.len() <= n, "Gang::load_partial: slice longer than lanes()");
        // The full-register case is every chunk but the last, so it's the hot path: a plain `load`,
        // no staging. The short tail goes through an out-of-line cold helper, so the buffer fill and
        // copy never sit in — and never spill registers out of — a caller's inner loop.
        if s.len() == n {
            self.load(s)
        } else {
            self.load_tail(s, fill)
        }
    }

    #[cold]
    #[inline(never)]
    fn load_tail(self, s: &[T], fill: T) -> Varying<T, S> {
        let n = self.backend.lanes();
        let len = s.len();
        // Stage exactly `n` lanes: the active prefix from `s`, the rest `fill`. The bounded `0..n`
        // loop (no runtime-length `copy_from_slice`/`memcpy`) lets dead-store elimination drop the
        // `[fill; MAX_LANES]` init down to the lanes actually loaded once `n` is a constant.
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

    /// Fold a kernel over one column without writing the loop. The library walks full registers at a
    /// fixed stride — the loads are `&a[off..off + lanes()]` with a constant width, so the body is
    /// bounds-check- and tail-branch-free — then runs `f` once more on a masked tail filled with
    /// `fill`. Writing the same thing as `chunks` + `load_partial` keeps a runtime chunk count in the
    /// loop, which defeats both; routing through `fold` gives the hand-tuned shape from a naive call.
    #[inline]
    pub fn fold<A>(
        self,
        a: &[T],
        fill: T,
        init: A,
        mut f: impl FnMut(A, Varying<T, S>) -> A,
    ) -> A {
        let n = self.backend.lanes();
        let len = a.len();
        let mut acc = init;
        // `while off + n <= len` (rather than a counted `0..len/n`) keeps `off + n <= len` as a live
        // fact at each load, so the optimizer drops the per-iteration bounds check instead of having
        // to reason back through the integer division.
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
    pub fn zip_fold<A>(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        init: A,
        mut f: impl FnMut(A, Varying<T, S>, Varying<T, S>) -> A,
    ) -> A {
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
    /// bounded by the shortest so all three loads are provably in bounds, each tail filled with its own
    /// sentinel. The natural shape for a kernel reading three position columns (`x`, `y`, `z`) in a
    /// single pass instead of three.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn zip3_fold<A>(
        self,
        a: &[T],
        b: &[T],
        c: &[T],
        fill_a: T,
        fill_b: T,
        fill_c: T,
        init: A,
        mut f: impl FnMut(A, Varying<T, S>, Varying<T, S>, Varying<T, S>) -> A,
    ) -> A {
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

    /// Map a kernel over one column straight into `out` without writing the loop — the elementwise
    /// counterpart of [`fold`](Self::fold). Same full-register stride and single masked tail: the body
    /// is `f(load(..)).store(..)` at a constant width, so it stays bounds-check- and tail-branch-free,
    /// and the optimizer sees the same shape a hand-rolled `chunks` + `load_partial`/`store_partial`
    /// loop would produce.
    ///
    /// The pass is bounded by the shorter of `a` and `out`, so every load and store is provably in
    /// bounds. In the tail `f` runs over the inactive input lanes too — they are computed and then
    /// dropped by [`store_partial`](Varying::store_partial), so `fill` only matters when `f` could
    /// fault or saturate on it (a divide whose padding would be `0`, say); otherwise any value works.
    #[inline]
    pub fn map(self, a: &[T], out: &mut [T], fill: T, mut f: impl FnMut(Varying<T, S>) -> Varying<T, S>) {
        let n = self.backend.lanes();
        let len = a.len().min(out.len());
        let mut off = 0;
        while off + n <= len {
            f(self.load(&a[off..off + n])).store(&mut out[off..off + n]);
            off += n;
        }
        if off < len {
            f(self.load_partial(&a[off..len], fill)).store_partial(&mut out[off..len]);
        }
    }

    /// Two-column [`map`](Self::map): `a` and `b` walked in lockstep into `out`, the full-register pass
    /// bounded by the shortest of the three so every load and store is in bounds, each input tail
    /// filled with its own sentinel (computed then dropped by the partial store, as in [`map`](Self::map)).
    #[inline]
    pub fn zip_map(
        self,
        a: &[T],
        b: &[T],
        out: &mut [T],
        fill_a: T,
        fill_b: T,
        mut f: impl FnMut(Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    ) {
        let n = self.backend.lanes();
        let len = a.len().min(b.len()).min(out.len());
        let mut off = 0;
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

    /// Short-circuiting `any`: `true` as soon as some lane in some register satisfies `pred`. The
    /// full-register pass returns at the first register whose `pred(..).any()` holds — [`fold`](Self::fold)
    /// cannot do this, it must visit every element to build its accumulator. Tail via
    /// [`load_partial`](Self::load_partial).
    ///
    /// `fill` must be a value `pred` *rejects*, so the inactive padding in the final partial register
    /// can never spuriously trip the result (e.g. `f32::NEG_INFINITY` for an `x > y` test — `-inf` is
    /// never greater). The opposite of [`all`](Self::all), whose fill must be *accepted*.
    #[inline]
    pub fn any(self, a: &[T], fill: T, mut pred: impl FnMut(Varying<T, S>) -> Mask<T, S>) -> bool {
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

    /// Short-circuiting `all`: `false` as soon as some register has a lane that fails `pred`, else
    /// `true` (vacuously so for an empty slice). Returns at the first register whose `pred(..).all()`
    /// is false. Tail via [`load_partial`](Self::load_partial).
    ///
    /// `fill` must be a value `pred` *accepts* — the mirror of [`any`](Self::any) — so the inactive
    /// padding of the final partial register cannot spuriously fail the check (for an `x <= hi` test,
    /// fill the `x` tail with `hi` or below).
    #[inline]
    pub fn all(self, a: &[T], fill: T, mut pred: impl FnMut(Varying<T, S>) -> Mask<T, S>) -> bool {
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

    /// Two-column [`any`](Self::any): `true` as soon as a register pair satisfies `pred`. Pass bounded
    /// by the shorter column; each tail filled with a sentinel `pred` *rejects* (see [`any`](Self::any)).
    /// Directly replaces the hand-rolled `chunks` + early-`return` predicate loops.
    #[inline]
    pub fn zip_any(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        mut pred: impl FnMut(Varying<T, S>, Varying<T, S>) -> Mask<T, S>,
    ) -> bool {
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

    /// Two-column [`all`](Self::all): `false` as soon as a register pair fails `pred`, else `true`. Pass
    /// bounded by the shorter column; each tail filled with a sentinel `pred` *accepts* (see
    /// [`all`](Self::all)).
    #[inline]
    pub fn zip_all(
        self,
        a: &[T],
        b: &[T],
        fill_a: T,
        fill_b: T,
        mut pred: impl FnMut(Varying<T, S>, Varying<T, S>) -> Mask<T, S>,
    ) -> bool {
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

    /// Two-column reduction across `K` independent accumulator chains — the ILP/superscalar shape
    /// from `future/ILP_SUPERSCALAR.md`. The body advances `K` chains with no data dependency
    /// between them per iteration (stride `K * lanes()`), so a wide out-of-order core can keep one
    /// FMA in flight per pipe instead of stalling on a single latency-bound chain. `step` runs once
    /// per chain per loaded register pair; `combine` folds the `K` chains as a balanced tree (depth
    /// `~log2(K)`, correct for non-power-of-two `K`). A single masked tail handles the remainder.
    ///
    /// Prefer [`zip_reduce`](Self::zip_reduce), which picks `K` from the cached runtime saturation
    /// point; reach for this directly to pin a specific factor (benchmarking the knee, or a target
    /// whose `K` you already know).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(any(no_ilp, target_arch = "spirv")))]
    pub fn zip_reduce_k<const K: usize, A: Copy, FS, FC>(
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
        FS: Fn(A, Varying<T, S>, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let n = self.backend.lanes();
        let len = a.len().min(b.len());
        let mut acc = [init; K];
        let mut off = 0;
        // `while off + K*n <= len` (not a counted `0..len/(K*n)`) keeps the window load in bounds from
        // the guard alone. The per-iteration `K*n`-wide reborrow is the trick that drops *every*
        // per-chain bounds check: within a window of constant length `K*n`, each `[o..o + n]` with
        // `o = j*n < K*n` is in bounds by constants only, where indexing the original slices at
        // `off + j*n` leaves the optimizer unable to prove `off + j*n + n <= len` for `j < K-1`.
        while off + K * n <= len {
            let aw = &a[off..off + K * n];
            let bw = &b[off..off + K * n];
            let mut j = 0;
            while j < K {
                let o = j * n;
                acc[j] = step(acc[j], self.load(&aw[o..o + n]), self.load(&bw[o..o + n]));
                j += 1;
            }
            off += K * n;
        }
        // The `< K` leftover registers go to *distinct* chains, not all into `acc[0]` — dumping them
        // serially would rebuild a long dependency chain and undo the ILP for any `len` that isn't a
        // multiple of `K * lanes()`.
        let mut j = 0;
        while off + n <= len {
            acc[j] = step(acc[j], self.load(&a[off..off + n]), self.load(&b[off..off + n]));
            off += n;
            j += 1;
        }
        let mut width = K;
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

    /// ILP compiled out (`--cfg no_ilp` or the SPIR-V target): the multi-accumulator loop, the
    /// `[init; K]` chain array and the tree-combine all collapse to the single-chain [`zip_fold`](Self::zip_fold),
    /// so even an explicit `zip_reduce_k::<K>` emits no superscalar scaffolding. `K` and `combine` are
    /// inert.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    pub fn zip_reduce_k<const K: usize, A: Copy, FS, FC>(
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
        FS: Fn(A, Varying<T, S>, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let _ = combine;
        self.zip_fold(a, b, fill_a, fill_b, init, step)
    }

    /// Two-column multi-accumulator reduction with `K` chosen from the cached runtime saturation
    /// point ([`unroll`](Self::unroll)). The warm path is a single relaxed atomic load plus the
    /// `match` below, then the `K`-unrolled [`zip_reduce_k`](Self::zip_reduce_k) loop — the per-call
    /// dispatch cost stays negligible in a hot loop. `step` is the per-chain combinator (use
    /// [`Varying::fma`] for a dot/AXPY-style update — one rounding, the throughput-bound op);
    /// `combine` folds two chains.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(any(no_ilp, target_arch = "spirv")))]
    pub fn zip_reduce<A: Copy, FS, FC>(
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
        FS: Fn(A, Varying<T, S>, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        // Too few registers for ILP's tree-combine to amortize: take the single-chain `zip_fold`
        // directly — same fast loop as a hand-written reduction, and no atomic load or detection
        // probe, so a tiny kernel is never slower than `fold`/`opt`.
        let avail = a.len().min(b.len()) / self.backend.lanes();
        if avail < 8 {
            return self.zip_fold(a, b, fill_a, fill_b, init, step);
        }
        // Engage the cached chain count once a full K-stride window fits; otherwise one chain. The
        // `K == 1` arm is `zip_fold` (not `zip_reduce_k::<1>`) so the no-ILP case carries none of the
        // multi-accumulator scaffolding.
        let k = {
            let c = self.unroll();
            if avail >= c { c } else { 1 }
        };
        match k {
            2 => self.zip_reduce_k::<2, A, _, _>(a, b, fill_a, fill_b, init, &step, &combine),
            4 => self.zip_reduce_k::<4, A, _, _>(a, b, fill_a, fill_b, init, &step, &combine),
            8 => self.zip_reduce_k::<8, A, _, _>(a, b, fill_a, fill_b, init, &step, &combine),
            12 => self.zip_reduce_k::<12, A, _, _>(a, b, fill_a, fill_b, init, &step, &combine),
            16 => self.zip_reduce_k::<16, A, _, _>(a, b, fill_a, fill_b, init, &step, &combine),
            _ => self.zip_fold(a, b, fill_a, fill_b, init, step),
        }
    }

    /// ILP compiled out: no cached-`K` lookup, no dispatch `match` — straight to the single-chain
    /// [`zip_fold`](Self::zip_fold). `combine` is inert.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    pub fn zip_reduce<A: Copy, FS, FC>(
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
        FS: Fn(A, Varying<T, S>, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let _ = combine;
        self.zip_fold(a, b, fill_a, fill_b, init, step)
    }

    /// Single-column counterpart of [`zip_reduce_k`](Self::zip_reduce_k) — `K` independent chains
    /// over one slice (sum, norm, max-style kernels). Same loop discipline and tail handling.
    #[inline]
    #[cfg(not(any(no_ilp, target_arch = "spirv")))]
    pub fn reduce_k<const K: usize, A: Copy, FS, FC>(
        self,
        a: &[T],
        fill: T,
        init: A,
        step: FS,
        combine: FC,
    ) -> A
    where
        FS: Fn(A, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let n = self.backend.lanes();
        let len = a.len();
        let mut acc = [init; K];
        let mut off = 0;
        while off + K * n <= len {
            let aw = &a[off..off + K * n];
            let mut j = 0;
            while j < K {
                let o = j * n;
                acc[j] = step(acc[j], self.load(&aw[o..o + n]));
                j += 1;
            }
            off += K * n;
        }
        let mut j = 0;
        while off + n <= len {
            acc[j] = step(acc[j], self.load(&a[off..off + n]));
            off += n;
            j += 1;
        }
        let mut width = K;
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

    /// ILP compiled out: the `K` chains collapse to the single-chain [`fold`](Self::fold), so an
    /// explicit `reduce_k::<K>` emits no superscalar scaffolding. `K` and `combine` are inert.
    #[inline]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    pub fn reduce_k<const K: usize, A: Copy, FS, FC>(
        self,
        a: &[T],
        fill: T,
        init: A,
        step: FS,
        combine: FC,
    ) -> A
    where
        FS: Fn(A, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let _ = combine;
        self.fold(a, fill, init, step)
    }

    /// Single-column multi-accumulator reduction with `K` from the cached saturation point — the
    /// [`zip_reduce`](Self::zip_reduce) dispatcher for one slice.
    #[inline]
    #[cfg(not(any(no_ilp, target_arch = "spirv")))]
    pub fn reduce<A: Copy, FS, FC>(self, a: &[T], fill: T, init: A, step: FS, combine: FC) -> A
    where
        FS: Fn(A, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let avail = a.len() / self.backend.lanes();
        if avail < 8 {
            return self.fold(a, fill, init, step);
        }
        let k = {
            let c = self.unroll();
            if avail >= c { c } else { 1 }
        };
        match k {
            2 => self.reduce_k::<2, A, _, _>(a, fill, init, &step, &combine),
            4 => self.reduce_k::<4, A, _, _>(a, fill, init, &step, &combine),
            8 => self.reduce_k::<8, A, _, _>(a, fill, init, &step, &combine),
            12 => self.reduce_k::<12, A, _, _>(a, fill, init, &step, &combine),
            16 => self.reduce_k::<16, A, _, _>(a, fill, init, &step, &combine),
            _ => self.fold(a, fill, init, step),
        }
    }

    /// ILP compiled out: no cached-`K` lookup, no dispatch `match` — straight to the single-chain
    /// [`fold`](Self::fold). `combine` is inert.
    #[inline]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    pub fn reduce<A: Copy, FS, FC>(self, a: &[T], fill: T, init: A, step: FS, combine: FC) -> A
    where
        FS: Fn(A, Varying<T, S>) -> A,
        FC: Fn(A, A) -> A,
    {
        let _ = combine;
        self.fold(a, fill, init, step)
    }

    /// Sum `step` over two columns, fully reduced to a scalar, with ILP you never have to ask for.
    /// `0` is the sum identity, so it is the accumulator seed, the masked-tail fill for both columns,
    /// and the chain combine all at once — none of which the caller writes — and the number of
    /// independent accumulator chains is the per-core saturation point, resolved and cached like the
    /// backend itself. The obvious dot already feeds every FP pipe:
    ///
    /// ```ignore
    /// fn dot(ctx: Gang<f32>, a: &[f32], b: &[f32]) -> f32 {
    ///     ctx.zip_sum(a, b, |acc, x, y| x.fma(y, acc))
    /// }
    /// ```
    ///
    /// `step` is the per-register update (use [`Varying::fma`] — one rounding, the throughput-bound
    /// op). For a non-sum reduction (max/min/any) reach for [`zip_reduce`](Self::zip_reduce) with an
    /// explicit identity and combine.
    #[inline]
    pub fn zip_sum<F>(self, a: &[T], b: &[T], step: F) -> T
    where
        F: Fn(Varying<T, S>, Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    {
        self.zip_reduce(a, b, T::ZERO, T::ZERO, self.splat(T::ZERO), step, |p, q| p + q)
            .reduce_sum()
    }

    /// Single-column [`zip_sum`](Self::zip_sum): sum `step` over one column to a scalar with full,
    /// invisible ILP — `sum`, `norm`²-style kernels. `ctx.sum(a, |acc, x| x.fma(x, acc))` is `‖a‖²`.
    #[inline]
    pub fn sum<F>(self, a: &[T], step: F) -> T
    where
        F: Fn(Varying<T, S>, Varying<T, S>) -> Varying<T, S>,
    {
        self.reduce(a, T::ZERO, self.splat(T::ZERO), step, |p, q| p + q)
            .reduce_sum()
    }

    /// Plain sum `Σ a[i]` — [`sum`](Self::sum) with a lane-wise add, the same invisible per-core ILP.
    /// Named to sidestep the closure-taking [`sum`](Self::sum); reach for that when the per-register
    /// update is anything other than a bare add.
    #[inline]
    pub fn total(self, a: &[T]) -> T {
        self.sum(a, |acc, x| acc + x)
    }

    /// Dot product `Σ a[i]·b[i]` — the [`zip_sum`](Self::zip_sum) FMA collapsed to one call. The
    /// per-core ILP unroll, the `0` identity, both masked-tail fills and the chain combine all come for
    /// free; the walk is bounded by the shorter column.
    #[inline]
    pub fn dot(self, a: &[T], b: &[T]) -> T {
        self.zip_sum(a, b, |acc, x, y| x.fma(y, acc))
    }

    /// Squared L2 norm `Σ a[i]²` — [`sum`](Self::sum) with a self-FMA, the same invisible ILP. Prefer
    /// it to [`norm`](Self::norm) when the squared magnitude is enough (a distance comparison), to skip
    /// the `sqrt`.
    #[inline]
    pub fn norm_sq(self, a: &[T]) -> T {
        self.sum(a, |acc, x| x.fma(x, acc))
    }

    /// L2 norm `√(Σ a[i]²)` — [`norm_sq`](Self::norm_sq) and a single scalar `sqrt`.
    #[inline]
    pub fn norm(self, a: &[T]) -> T {
        self.norm_sq(a).sqrt()
    }

    /// The cached unroll factor for this core, resolving it on first use. The `lanes() == 1` guard
    /// const-folds to `return 1` for a concrete SIMD backend (lanes is a constant), and drops the
    /// scalar backend out of the atomic path entirely — a non-pipelined core gains nothing from
    /// multiple chains but the reduction tail, so it opts out. Also read by the matrix micro-kernel
    /// to size its register block (the same saturation count, taken to 2-D).
    #[inline]
    #[cfg(not(any(no_ilp, target_arch = "spirv")))]
    pub(crate) fn unroll(self) -> usize {
        if self.backend.lanes() == 1 {
            return 1;
        }
        match crate::ilp::cached() {
            0 => self.detect_unroll(),
            k => k as usize,
        }
    }

    /// ILP compiled out (`--cfg no_ilp` / SPIR-V): one chain, no atomic and no startup sweep. The
    /// matrix micro-kernel reads this too, so its register block degrades to single-width in step.
    #[inline(always)]
    #[cfg(any(no_ilp, target_arch = "spirv"))]
    pub(crate) fn unroll(self) -> usize {
        1
    }

    /// Resolve the unroll factor once and cache it. A sweep of the candidate factors `{1,2,4,8,12,16}`
    /// timing a fixed-buffer dot, picking the fastest — the saturation point of the actual core,
    /// which folds in register-spill effects for free. Out-of-line and `#[cold]`: it runs at most
    /// once per process, never on the warm path.
    #[cfg(all(feature = "std", not(any(no_ilp, target_arch = "spirv"))))]
    #[cold]
    #[inline(never)]
    fn detect_unroll(self) -> usize {
        use std::hint::black_box;
        use std::time::Instant;

        let probe: std::vec::Vec<T> = (0..4096)
            .map(|i| T::from_f64((i % 17) as f64 * 0.5 - 4.0))
            .collect();
        let a = probe.as_slice();
        let zero = T::ZERO;
        let init = self.splat(zero);
        let step =
            |acc: Varying<T, S>, x: Varying<T, S>, y: Varying<T, S>| -> Varying<T, S> { x.fma(y, acc) };
        let combine = |p: Varying<T, S>, q: Varying<T, S>| -> Varying<T, S> { p + q };

        let warm = Instant::now();
        let r = self.zip_reduce_k::<1, _, _, _>(
            black_box(a),
            black_box(a),
            zero,
            zero,
            init,
            &step,
            &combine,
        );
        black_box(r.reduce_sum());
        let one_ns = warm.elapsed().as_nanos().max(1) as u64;
        // Aim ~0.5 ms per timed run so `Instant` overhead is amortized; bound the count both ways.
        let iters = (500_000u64 / one_ns).clamp(1, 100_000) as u32;

        macro_rules! time_k {
            ($k:literal) => {{
                let mut best = u64::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    let mut sink = 0.0f64;
                    for _ in 0..iters {
                        let r = self.zip_reduce_k::<$k, _, _, _>(
                            black_box(a),
                            black_box(a),
                            zero,
                            zero,
                            init,
                            &step,
                            &combine,
                        );
                        sink += r.reduce_sum().to_f64();
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

        let cands = [
            (1u8, time_k!(1)),
            (2u8, time_k!(2)),
            (4u8, time_k!(4)),
            (8u8, time_k!(8)),
            (12u8, time_k!(12)),
            (16u8, time_k!(16)),
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

    /// No-std build: no timer/allocator for a sweep, so fall back to a per-target default that lands
    /// near each family's `latency × pipes` saturation point (Apple's wide NEON FP wants more chains
    /// than x86's 2–3 vector pipes).
    #[cfg(all(not(feature = "std"), not(any(no_ilp, target_arch = "spirv"))))]
    #[cold]
    #[inline(never)]
    fn detect_unroll(self) -> usize {
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

/// Full-register chunk iterator produced by [`Gang::chunks`]. Yields `(offset, count)` where
/// `count` is the number of valid elements in the chunk — `lanes()` for all but a final
/// short tail.
#[derive(Clone, Copy, Debug)]
pub struct Chunks {
    lanes: usize,
    pos: usize,
    len: usize,
}

impl Iterator for Chunks {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.pos >= self.len {
            return None;
        }
        let count = core::cmp::min(self.lanes, self.len - self.pos);
        let off = self.pos;
        self.pos += count;
        Some((off, count))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.len - self.pos).div_ceil(self.lanes);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Chunks {}

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

    /// Store this register; `out.len()` must equal `lanes()`.
    ///
    /// The backend writes exactly `lanes()` elements with an unchecked SIMD store: passing a slice
    /// of any other length is undefined behaviour. The length match is the caller's contract,
    /// checked only under `debug_assertions`. For tails, use [`Varying::store_partial`].
    #[inline(always)]
    pub fn store(self, out: &mut [T]) {
        debug_assert!(
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
    pub fn sqrt(self) -> Self {
        Self::wrap(self.backend, self.backend.sqrt(self.v))
    }
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self::wrap(self.backend, self.backend.min(self.v, o.v))
    }
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self::wrap(self.backend, self.backend.max(self.v, o.v))
    }
    /// `self * b + c`, fused where the backend supports it.
    #[inline(always)]
    pub fn fma(self, b: Self, c: Self) -> Self {
        Self::wrap(self.backend, self.backend.fma(self.v, b.v, c.v))
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
    #[inline(always)]
    pub fn reduce_min(self) -> T {
        self.backend.reduce_min(self.v)
    }
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
lane_binop!(Div, div, div);

impl<T: Scalar, S: Backend<T>> Neg for Varying<T, S> {
    type Output = Varying<T, S>;
    #[inline(always)]
    fn neg(self) -> Self {
        Varying::wrap(self.backend, self.backend.neg(self.v))
    }
}

/// Scalar on the right-hand side: `v * 2.0`, `v + bias`, … splat the scalar and apply the op, so a
/// uniform constant needs no explicit `ctx.splat`. (Only this direction — `2.0 * v` — is possible;
/// the orphan rule forbids `impl Mul<Varying> for f32`.)
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
varying_scalar_binop!(Div, div, div);

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
