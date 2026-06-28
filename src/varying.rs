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
//! [`Simd`] is the *context* you load/splat through (it produces `Varying`s); [`Varying`] and
//! [`Mask`] are the varying values. Everything is `Copy`, zero-sized except the register payload,
//! and monomorphizes per `(Backend, Scalar)` — the ergonomics cost nothing at runtime.

use core::marker::PhantomData;
use core::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Not, Sub};

use crate::backend::Backend;
use crate::scalar::Scalar;

/// A SIMD execution *context* for scalar `T` on backend `S`. You never construct one: it is
/// handed to your [`Kernel::run`](crate::Kernel::run) by `dispatch`, which picks the backend
/// from runtime CPU detection. Construct varying values (`splat`, `load`) through it.
#[derive(Clone, Copy)]
pub struct Simd<T: Scalar, S: Backend<T>> {
    backend: S,
    _t: PhantomData<T>,
}

impl<T: Scalar, S: Backend<T>> Simd<T, S> {
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

    /// Load exactly one register; `s.len()` must equal [`Simd::lanes`].
    ///
    /// The backend reads exactly `lanes()` elements from `s` with an unchecked SIMD load: passing a
    /// slice of any other length is undefined behaviour. The length match is the caller's contract,
    /// checked only under `debug_assertions`. For tails, use [`Simd::load_partial`] or the
    /// [`Simd::chunks`] iterator.
    #[inline(always)]
    pub fn load(self, s: &[T]) -> Varying<T, S> {
        debug_assert!(
            s.len() == self.backend.lanes(),
            "Simd::load: slice length must equal lanes()",
        );
        Varying::wrap(self.backend, self.backend.load(s))
    }

    /// Iterate `len` elements in full-register chunks, yielding `(offset, count)` per step.
    /// `count == lanes()` for every chunk except possibly the last. Pair with
    /// [`Simd::load_partial`] to run a kernel directly over unpadded, borrowed slices (e.g.
    /// the field slices of a `soa-rs` struct) — no copy and no padded [`Soa`](crate::Soa).
    #[inline]
    pub fn chunks(self, len: usize) -> Chunks {
        Chunks {
            lanes: self.backend.lanes(),
            pos: 0,
            len,
        }
    }

    /// Load up to [`lanes()`](Simd::lanes) elements from `s` (`s.len()` must not exceed it),
    /// filling the inactive tail lanes with `fill`. A full chunk is a plain [`load`](Simd::load);
    /// a short tail is staged through a stack buffer so the inactive lanes carry the sentinel
    /// (so e.g. `fill = NaN` keeps the tail out of distance comparisons and reductions).
    #[inline]
    pub fn load_partial(self, s: &[T], fill: T) -> Varying<T, S> {
        let n = self.backend.lanes();
        debug_assert!(s.len() <= n, "Simd::load_partial: slice longer than lanes()");
        if s.len() == n {
            return self.load(s);
        }
        let mut buf = [fill; crate::MAX_LANES];
        buf[..s.len()].copy_from_slice(s);
        self.load(&buf[..n])
    }

    /// The underlying backend token.
    #[inline(always)]
    pub fn backend(self) -> S {
        self.backend
    }
}

/// Full-register chunk iterator produced by [`Simd::chunks`]. Yields `(offset, count)` where
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
    /// to [`Simd::load_partial`] for writing results back into a borrowed, unpadded column.
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
