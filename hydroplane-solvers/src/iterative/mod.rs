//! Krylov methods for operators the caller supplies: matrix-free FEM operators, assembled
//! matrices, a factorization's solve as a preconditioner.
//!
//! - [`Cg`]: symmetric positive definite; standard, or flexible for a preconditioner that changes
//!   between applications (a multigrid cycle).
//! - [`Minres`]: symmetric indefinite or singular (saddle points, tangents past a limit point),
//!   with a symmetric positive definite preconditioner.
//! - [`Gmres`] (restarted, right-preconditioned; flexible as FGMRES) and [`BiCgStab`]: any operator.
//! - [`Lsqr`]: least squares `min ‖A x − b‖` for a rectangular `A`, with optional damping.
//!
//! All run over any [`Vector`]: slices of `f64`, `f32`, fixed arrays, `glam` vectors (feature
//! `glam`), or a caller's own type. Dot products are always accumulated in `f64` and reduced in
//! fixed blocks, so results are bit for bit the same for any number of threads.
//!
//! [`Amg`] (smoothed-aggregation algebraic multigrid) preconditions CG on assembled elliptic
//! problems with iteration counts that stay nearly flat under refinement.
//!
//! The preconditioners here compose with [`Additive`]: [`Jacobi`], [`BlockJacobi`], [`Chebyshev`]
//! (a polynomial smoother, its interval from [`estimate_spectrum`]), the incomplete
//! factorizations [`Ilu0`] and [`Ic0`] of an assembled matrix,
//! [`crate::dense::CoarseCorrection`], and any closure `FnMut(&V, &mut V)`.

use rayon::prelude::*;

pub(crate) mod amg;
mod bicgstab;
mod chebyshev;
mod gmres;
mod incomplete;
mod lsqr;
mod minres;

pub use amg::{Amg, AmgOptions};
pub use bicgstab::BiCgStab;
pub use chebyshev::{Chebyshev, Spectrum, estimate_spectrum};
pub use gmres::Gmres;
pub use incomplete::{Ic0, Ilu0};
pub use lsqr::{Lsqr, LsqrOptions, LsqrReport, LsqrStop};
pub use minres::Minres;

/// Elements per block of a reduction: the parallel split, and the unit whose partial sums are
/// added in a fixed order.
const BLOCK: usize = 2048;

/// `Σ f(i)` for `i < n` in blocks of 2048, the blocks' sums added in order: the same
/// result for any thread count.
pub fn sum_blocks(n: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    if n <= BLOCK {
        return (0..n).map(f).sum();
    }
    let partial: Vec<f64> = (0..n.div_ceil(BLOCK)).into_par_iter().map(|b| (b * BLOCK..((b + 1) * BLOCK).min(n)).map(&f).sum::<f64>()).collect();
    partial.iter().sum()
}

/// A vector entry: a scalar, or a node's few components ([`Self::LANES`]).
pub trait Element: Copy + Send + Sync + 'static {
    const LANES: usize;
    const ZERO: Self;
    fn lane(&self, k: usize) -> f64;
    fn set_lane(&mut self, k: usize, v: f64);
    /// `a · b`, in `f64`.
    fn dot(a: &Self, b: &Self) -> f64;
    /// `y += a x`.
    fn axpy(y: &mut Self, a: f64, x: &Self);
    /// `y = x + a y`.
    fn xpay(y: &mut Self, x: &Self, a: f64);
}

macro_rules! scalar {
    ($t:ty) => {
        impl Element for $t {
            const LANES: usize = 1;
            const ZERO: Self = 0.0;
            #[inline]
            fn lane(&self, _: usize) -> f64 {
                *self as f64
            }
            #[inline]
            fn set_lane(&mut self, _: usize, v: f64) {
                *self = v as $t;
            }
            #[inline]
            fn dot(a: &Self, b: &Self) -> f64 {
                *a as f64 * *b as f64
            }
            #[inline]
            fn axpy(y: &mut Self, a: f64, x: &Self) {
                *y = (*y as f64 + a * *x as f64) as $t;
            }
            #[inline]
            fn xpay(y: &mut Self, x: &Self, a: f64) {
                *y = (*x as f64 + a * *y as f64) as $t;
            }
        }

        impl<const N: usize> Element for [$t; N] {
            const LANES: usize = N;
            const ZERO: Self = [0.0; N];
            #[inline]
            fn lane(&self, k: usize) -> f64 {
                self[k] as f64
            }
            #[inline]
            fn set_lane(&mut self, k: usize, v: f64) {
                self[k] = v as $t;
            }
            #[inline]
            fn dot(a: &Self, b: &Self) -> f64 {
                (0..N).map(|k| a[k] as f64 * b[k] as f64).sum()
            }
            #[inline]
            fn axpy(y: &mut Self, a: f64, x: &Self) {
                for k in 0..N {
                    y[k] = (y[k] as f64 + a * x[k] as f64) as $t;
                }
            }
            #[inline]
            fn xpay(y: &mut Self, x: &Self, a: f64) {
                for k in 0..N {
                    y[k] = (x[k] as f64 + a * y[k] as f64) as $t;
                }
            }
        }
    };
}

scalar!(f64);
scalar!(f32);

#[cfg(feature = "glam")]
macro_rules! glam3 {
    ($t:ty, $s:ty) => {
        impl Element for $t {
            const LANES: usize = 3;
            const ZERO: Self = <$t>::ZERO;
            #[inline]
            fn lane(&self, k: usize) -> f64 {
                self[k] as f64
            }
            #[inline]
            fn set_lane(&mut self, k: usize, v: f64) {
                self[k] = v as $s;
            }
            #[inline]
            fn dot(a: &Self, b: &Self) -> f64 {
                a.x as f64 * b.x as f64 + a.y as f64 * b.y as f64 + a.z as f64 * b.z as f64
            }
            #[inline]
            fn axpy(y: &mut Self, a: f64, x: &Self) {
                for k in 0..3 {
                    y[k] = (y[k] as f64 + a * x[k] as f64) as $s;
                }
            }
            #[inline]
            fn xpay(y: &mut Self, x: &Self, a: f64) {
                for k in 0..3 {
                    y[k] = (x[k] as f64 + a * y[k] as f64) as $s;
                }
            }
        }
    };
}

#[cfg(feature = "glam")]
glam3!(glam::DVec3, f64);
#[cfg(feature = "glam")]
glam3!(glam::Vec3, f32);

/// What CG needs of a vector. Implemented for `[E]` with any [`Element`]; a caller's own layout
/// (say, three `f32` planes) implements it directly.
pub trait Vector: Send + Sync {
    type Owned: core::ops::DerefMut<Target = Self> + Send + Sync;
    fn zeros_like(&self) -> Self::Owned;
    fn copy_from(&mut self, other: &Self);
    fn fill_zero(&mut self);
    /// In `f64`, reproducible for any thread count.
    fn dot(&self, other: &Self) -> f64;
    /// `self += a x`.
    fn axpy(&mut self, a: f64, x: &Self);
    /// `self = x + a self`.
    fn xpay(&mut self, x: &Self, a: f64);
    /// `self = a self`.
    fn scale(&mut self, a: f64);
    /// Whether every entry is finite.
    fn all_finite(&self) -> bool;
    /// `out[j] = self · others[j]`: one pass over `self` for all of them where the layout allows
    /// (Gram–Schmidt against a whole basis), as reproducible as [`Self::dot`].
    fn dots(&self, others: &[&Self], out: &mut [f64]) {
        for (o, x) in out.iter_mut().zip(others) {
            *o = self.dot(x);
        }
    }
    /// `self += Σ a[j] xs[j]`, in one pass where the layout allows.
    fn axpys(&mut self, a: &[f64], xs: &[&Self]) {
        for (&a, x) in a.iter().zip(xs) {
            self.axpy(a, x);
        }
    }
}

impl<E: Element> Vector for [E] {
    type Owned = Vec<E>;

    fn zeros_like(&self) -> Vec<E> {
        vec![E::ZERO; self.len()]
    }

    fn copy_from(&mut self, other: &Self) {
        self.copy_from_slice(other);
    }

    fn fill_zero(&mut self) {
        self.par_iter_mut().with_min_len(BLOCK).for_each(|v| *v = E::ZERO);
    }

    fn dot(&self, other: &Self) -> f64 {
        sum_blocks(self.len(), |i| E::dot(&self[i], &other[i]))
    }

    fn axpy(&mut self, a: f64, x: &Self) {
        self.par_iter_mut().zip(x).with_min_len(BLOCK).for_each(|(y, x)| E::axpy(y, a, x));
    }

    fn xpay(&mut self, x: &Self, a: f64) {
        self.par_iter_mut().zip(x).with_min_len(BLOCK).for_each(|(y, x)| E::xpay(y, x, a));
    }

    fn scale(&mut self, a: f64) {
        self.par_iter_mut().with_min_len(BLOCK).for_each(|y| E::xpay(y, &E::ZERO, a));
    }

    fn all_finite(&self) -> bool {
        self.par_iter().with_min_len(BLOCK).all(|v| (0..E::LANES).all(|k| v.lane(k).is_finite()))
    }

    fn dots(&self, others: &[&Self], out: &mut [f64]) {
        let n = self.len();
        let block = |b: usize| -> Vec<f64> {
            let range = b * BLOCK..((b + 1) * BLOCK).min(n);
            let mine = &self[range.clone()];
            others.iter().map(|o| mine.iter().zip(&o[range.clone()]).map(|(a, b)| E::dot(a, b)).sum::<f64>()).collect()
        };
        let partial: Vec<Vec<f64>> = (0..n.div_ceil(BLOCK)).into_par_iter().map(block).collect();
        out[..others.len()].fill(0.0);
        for p in &partial {
            for (o, s) in out.iter_mut().zip(p) {
                *o += s;
            }
        }
    }

    fn axpys(&mut self, a: &[f64], xs: &[&Self]) {
        self.par_chunks_mut(BLOCK).enumerate().for_each(|(b, chunk)| {
            let range = b * BLOCK..b * BLOCK + chunk.len();
            for (&a, x) in a.iter().zip(xs) {
                for (y, x) in chunk.iter_mut().zip(&x[range.clone()]) {
                    E::axpy(y, a, x);
                }
            }
        });
    }
}

/// `y = A x`: symmetric positive definite for [`Cg`], symmetric for [`Minres`], anything for
/// [`Gmres`] and [`BiCgStab`].
pub trait LinearOperator<V: ?Sized> {
    fn apply(&mut self, x: &V, y: &mut V);
}

impl<V: ?Sized, F: FnMut(&V, &mut V)> LinearOperator<V> for F {
    fn apply(&mut self, x: &V, y: &mut V) {
        self(x, y)
    }
}

/// `z ≈ A⁻¹ r`: a symmetric positive definite approximation of `A⁻¹` for [`Cg`] and [`Minres`],
/// or any approximation at all for flexible CG and GMRES ([`Options::flexible`]).
pub trait Preconditioner<V: ?Sized> {
    fn apply(&mut self, r: &V, z: &mut V);
}

impl<V: ?Sized, F: FnMut(&V, &mut V)> Preconditioner<V> for F {
    fn apply(&mut self, r: &V, z: &mut V) {
        self(r, z)
    }
}

/// `z = r`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

impl<V: Vector + ?Sized> Preconditioner<V> for Identity {
    fn apply(&mut self, r: &V, z: &mut V) {
        z.copy_from(r);
    }
}

/// `z = P r + Q r`: a smoother plus a coarse correction, say.
pub struct Additive<V: Vector + ?Sized, P, Q> {
    pub first: P,
    pub second: Q,
    scratch: Option<V::Owned>,
}

impl<V: Vector + ?Sized, P, Q> Additive<V, P, Q> {
    pub fn new(first: P, second: Q) -> Self {
        Self { first, second, scratch: None }
    }
}

impl<V: Vector + ?Sized, P: Preconditioner<V>, Q: Preconditioner<V>> Preconditioner<V> for Additive<V, P, Q> {
    fn apply(&mut self, r: &V, z: &mut V) {
        self.first.apply(r, z);
        let tmp = self.scratch.get_or_insert_with(|| r.zeros_like());
        self.second.apply(r, tmp);
        z.axpy(1.0, tmp);
    }
}

/// `z = D⁻¹ r` with `D` the diagonal, by lanes (every component of every entry).
#[derive(Clone, Debug)]
pub struct Jacobi {
    inverse: Vec<f64>,
}

impl Jacobi {
    /// From the diagonal, lane by lane; a zero or non-finite entry is left alone (`z = r`
    /// there), as for a held degree of freedom.
    pub fn new(diagonal: &[f64]) -> Self {
        Self { inverse: diagonal.iter().map(|&d| if d != 0.0 && d.is_finite() { 1.0 / d } else { 1.0 }).collect() }
    }
}

impl<E: Element> Preconditioner<[E]> for Jacobi {
    fn apply(&mut self, r: &[E], z: &mut [E]) {
        assert_eq!(self.inverse.len(), r.len() * E::LANES);
        let inv = &self.inverse;
        z.par_iter_mut().zip(r).enumerate().with_min_len(BLOCK).for_each(|(i, (z, r))| {
            for k in 0..E::LANES {
                z.set_lane(k, inv[i * E::LANES + k] * r.lane(k));
            }
        });
    }
}

/// `z = Dᵢ⁻¹ rᵢ` for `B × B` diagonal blocks `Dᵢ` (a node's components), over the vector's lanes
/// in order: entries of `B` lanes, or scalars `B` at a time.
#[derive(Clone, Debug)]
pub struct BlockJacobi<const B: usize> {
    inverse: Vec<[[f64; B]; B]>,
}

impl<const B: usize> BlockJacobi<B> {
    /// From each node's diagonal block.
    pub fn new(blocks: &[[[f64; B]; B]]) -> Self {
        Self::build(blocks, None)
    }

    /// As [`Self::new`], with the lanes marked in `held` (one flag per lane) decoupled: their row
    /// and column of the block become the identity's.
    pub fn with_held(blocks: &[[[f64; B]; B]], held: &[bool]) -> Self {
        Self::build(blocks, Some(held))
    }

    fn build(blocks: &[[[f64; B]; B]], held: Option<&[bool]>) -> Self {
        let inverse = blocks
            .par_iter()
            .enumerate()
            .map(|(i, block)| {
                let mut a = *block;
                if let Some(held) = held {
                    for k in 0..B {
                        if held[i * B + k] {
                            a[k] = [0.0; B];
                            for row in a.iter_mut() {
                                row[k] = 0.0;
                            }
                            a[k][k] = 1.0;
                        }
                    }
                }
                invert(a)
            })
            .collect();
        Self { inverse }
    }
}

/// Gauss–Jordan with partial pivoting; a singular block gives the identity (the node is left
/// unpreconditioned rather than poisoned).
fn invert<const B: usize>(mut a: [[f64; B]; B]) -> [[f64; B]; B] {
    let mut inv = [[0.0; B]; B];
    for (k, row) in inv.iter_mut().enumerate() {
        row[k] = 1.0;
    }
    let identity = inv;
    for c in 0..B {
        let p = (c..B).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).expect("a row");
        if a[p][c] == 0.0 || !a[p][c].is_finite() {
            return identity;
        }
        a.swap(c, p);
        inv.swap(c, p);
        let d = 1.0 / a[c][c];
        for j in 0..B {
            a[c][j] *= d;
            inv[c][j] *= d;
        }
        for i in 0..B {
            if i != c {
                let f = a[i][c];
                for j in 0..B {
                    a[i][j] -= f * a[c][j];
                    inv[i][j] -= f * inv[c][j];
                }
            }
        }
    }
    inv
}

impl<E: Element, const B: usize> Preconditioner<[E]> for BlockJacobi<B> {
    fn apply(&mut self, r: &[E], z: &mut [E]) {
        assert_eq!(self.inverse.len() * B, r.len() * E::LANES, "one block per {B} lanes");
        let lane = |v: &[E], k: usize| v[k / E::LANES].lane(k % E::LANES);
        let out: Vec<[f64; B]> = self
            .inverse
            .par_iter()
            .enumerate()
            .with_min_len(BLOCK / B)
            .map(|(i, inv)| {
                let x: [f64; B] = core::array::from_fn(|k| lane(r, i * B + k));
                core::array::from_fn(|row| (0..B).map(|k| inv[row][k] * x[k]).sum())
            })
            .collect();
        for (i, y) in out.iter().enumerate() {
            for (k, &v) in y.iter().enumerate() {
                let at = i * B + k;
                z[at / E::LANES].set_lane(at % E::LANES, v);
            }
        }
    }
}

/// What the tolerance is relative to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RelativeTo {
    /// `‖r₀‖ = ‖b − A x₀‖`: a warm start must still be improved by the full factor.
    #[default]
    InitialResidual,
    /// `‖b‖`: a warm start already within tolerance needs no iterations.
    RightHandSide,
}

/// The settings every Krylov solver here takes (each solver's `options` field).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Options {
    /// Stop once `‖r‖ ≤ tolerance · ‖reference‖` (see [`Self::relative_to`]).
    pub tolerance: f64,
    pub relative_to: RelativeTo,
    /// Iterations ([`Gmres`]: operator applications).
    pub max_iterations: usize,
    /// Start from the `x` passed in (else from zero, sparing one operator application).
    pub warm_start: bool,
    /// For a preconditioner that changes between applications (a multigrid cycle, an inner
    /// iteration): flexible CG (`β = −(z·q)/(p·q)`) or FGMRES. [`Minres`] and [`BiCgStab`] need a
    /// fixed one.
    pub flexible: bool,
    /// [`Gmres`]'s basis size before a restart.
    pub restart: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { tolerance: 1e-8, relative_to: RelativeTo::InitialResidual, max_iterations: 1000, warm_start: false, flexible: false, restart: 30 }
    }
}

/// Why the iteration ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Converged,
    MaxIterations,
    /// `pᵀAp ≤ 0`: the operator is not positive definite along `p` (geometric stiffness, a
    /// softening material). `x` is the last iterate, a descent direction for Newton.
    NegativeCurvature,
    /// A non-finite value appeared.
    NotFinite,
    /// A quantity the method divides by vanished before convergence (BiCGStab's `r̂·r`, an
    /// indefinite preconditioner under MINRES): restarting, or another method, may still succeed.
    Breakdown,
}

/// How an iteration ended.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Report {
    pub iterations: usize,
    /// `‖r‖` relative to the tolerance's reference, as the iteration tracked it (in the norm the
    /// method minimizes; see each solver).
    pub relative_residual: f64,
    pub stop: Stop,
}

impl Report {
    pub fn converged(&self) -> bool {
        self.stop == Stop::Converged
    }
}

/// How far a factorization's `solve_refined` takes a solution: GMRES on the true matrix,
/// preconditioned by the factor, until `‖b − A x‖ ≤ tolerance · ‖b‖`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refinement {
    pub tolerance: f64,
    pub max_steps: usize,
}

impl Default for Refinement {
    fn default() -> Self {
        Self { tolerance: 1e-12, max_steps: 20 }
    }
}

impl Refinement {
    /// `x = A⁻¹ b` from `x = b`, by GMRES on `a` preconditioned by `solve`.
    pub(crate) fn run(&self, mut a: impl FnMut(&[f64], &mut [f64]), mut solve: impl FnMut(&mut [f64]), x: &mut [f64]) -> Report {
        let b = x.to_vec();
        let steps = self.max_steps.max(1);
        let options = Options { tolerance: self.tolerance, relative_to: RelativeTo::RightHandSide, max_iterations: steps, restart: steps, ..Default::default() };
        let mut m = |r: &[f64], z: &mut [f64]| {
            z.copy_from_slice(r);
            solve(z);
        };
        Gmres::with_options(options).solve(&mut a, &mut m, &b[..], x)
    }
}

/// `r = b − A x`, or `r = b` and `x = 0` without a warm start.
fn initial_residual<V: Vector + ?Sized>(a: &mut impl LinearOperator<V>, b: &V, x: &mut V, r: &mut V, scratch: &mut V, warm_start: bool) {
    if warm_start {
        a.apply(x, scratch);
        r.copy_from(b);
        r.axpy(-1.0, scratch);
    } else {
        x.fill_zero();
        r.copy_from(b);
    }
}

fn relative(norm: f64, reference: f64) -> f64 {
    if reference > 0.0 { norm / reference } else { 0.0 }
}

/// Conjugate gradient, with its work vectors kept between solves of the same size.
pub struct Cg<V: Vector + ?Sized> {
    pub options: Options,
    work: Option<[V::Owned; 4]>,
}

impl<V: Vector + ?Sized> Default for Cg<V> {
    fn default() -> Self {
        Self { options: Options::default(), work: None }
    }
}

impl<V: Vector + ?Sized> Cg<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: Options) -> Self {
        Self { options, work: None }
    }

    /// `A x = b` from `x` (or zero), preconditioned by `m`.
    pub fn solve(&mut self, a: &mut impl LinearOperator<V>, m: &mut impl Preconditioner<V>, b: &V, x: &mut V) -> Report {
        let options = self.options;
        let [r, z, p, q] = self.work.get_or_insert_with(|| [b.zeros_like(), b.zeros_like(), b.zeros_like(), b.zeros_like()]);
        let (r, z, p, q): (&mut V, &mut V, &mut V, &mut V) = (r, z, p, q);
        if options.warm_start {
            a.apply(x, q);
            r.copy_from(b);
            r.axpy(-1.0, q);
        } else {
            x.fill_zero();
            r.copy_from(b);
        }
        let rr0 = r.dot(r);
        let reference = match options.relative_to {
            RelativeTo::InitialResidual => rr0,
            RelativeTo::RightHandSide => b.dot(b),
        };
        let report = |iterations, rr: f64, stop| Report { iterations, relative_residual: if reference > 0.0 { (rr / reference).sqrt() } else { 0.0 }, stop };
        if !rr0.is_finite() {
            return report(0, rr0, Stop::NotFinite);
        }
        let tol2 = options.tolerance * options.tolerance * reference;
        if rr0 <= f64::MIN_POSITIVE || rr0 <= tol2 {
            return report(0, rr0, Stop::Converged);
        }
        m.apply(r, z);
        p.copy_from(z);
        let mut rz = r.dot(z);
        let mut rr = rr0;
        let mut iterations = 0;
        while iterations < options.max_iterations {
            a.apply(p, q);
            let pq = p.dot(q);
            if !pq.is_finite() {
                return report(iterations, rr, Stop::NotFinite);
            }
            if pq <= 0.0 {
                return report(iterations, rr, Stop::NegativeCurvature);
            }
            let alpha = rz / pq;
            x.axpy(alpha, p);
            r.axpy(-alpha, q);
            iterations += 1;
            rr = r.dot(r);
            if !rr.is_finite() {
                return report(iterations, rr, Stop::NotFinite);
            }
            if rr <= tol2 {
                return report(iterations, rr, Stop::Converged);
            }
            m.apply(r, z);
            let beta = if options.flexible {
                rz = r.dot(z);
                -z.dot(q) / pq
            } else {
                let rz_new = r.dot(z);
                let beta = rz_new / rz;
                rz = rz_new;
                beta
            };
            p.xpay(z, beta);
        }
        report(iterations, rr, Stop::MaxIterations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1D Laplacian with a shift, as an operator on `[f64]`.
    fn laplacian(shift: f64) -> impl FnMut(&[f64], &mut [f64]) {
        move |x: &[f64], y: &mut [f64]| {
            let n = x.len();
            for i in 0..n {
                let left = if i > 0 { x[i - 1] } else { 0.0 };
                let right = if i + 1 < n { x[i + 1] } else { 0.0 };
                y[i] = (2.0 + shift) * x[i] - left - right;
            }
        }
    }

    fn residual(a: &mut impl FnMut(&[f64], &mut [f64]), b: &[f64], x: &[f64]) -> f64 {
        let mut ax = vec![0.0; b.len()];
        a(x, &mut ax);
        let r: f64 = b.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum();
        (r / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
    }

    #[test]
    fn standard_and_flexible_cg_reach_the_tolerance() {
        let n = 300;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        for flexible in [false, true] {
            let mut a = laplacian(0.01);
            let mut x = vec![0.0; n];
            let options = Options { tolerance: 1e-10, flexible, ..Default::default() };
            let report = Cg::with_options(options).solve(&mut a, &mut Jacobi::new(&vec![2.01; n]), &b[..], &mut x[..]);
            assert!(report.converged(), "flexible {flexible}: {report:?}");
            assert!(residual(&mut a, &b, &x) < 1e-9);
        }
    }

    #[test]
    fn a_warm_start_from_the_answer_needs_no_iterations() {
        let n = 50;
        let b: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut a = laplacian(1.0);
        let mut x = vec![0.0; n];
        let options = Options { tolerance: 1e-12, ..Default::default() };
        Cg::with_options(options).solve(&mut a, &mut Identity, &b[..], &mut x[..]);
        let warm = Options { warm_start: true, tolerance: 1e-3, relative_to: RelativeTo::RightHandSide, ..options };
        let report = Cg::with_options(warm).solve(&mut a, &mut Identity, &b[..], &mut x[..]);
        assert_eq!(report.iterations, 0);
        let report = Cg::with_options(Options { relative_to: RelativeTo::InitialResidual, ..warm }).solve(&mut a, &mut Identity, &b[..], &mut x[..]);
        assert!(report.iterations > 0, "relative to its own residual, a warm start still iterates");
    }

    #[test]
    fn negative_curvature_stops_the_iteration() {
        let mut a = |x: &[f64], y: &mut [f64]| {
            y[0] = x[0];
            y[1] = -x[1];
        };
        let report = Cg::new().solve(&mut a, &mut Identity, &[0.0, 1.0][..], &mut [0.0; 2][..]);
        assert_eq!(report.stop, Stop::NegativeCurvature);
    }

    #[test]
    fn block_jacobi_inverts_node_blocks_whatever_the_layout() {
        let block = [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]];
        let mut m = BlockJacobi::new(&[block, block]);
        let r = [1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
        let mut z = [0.0; 6];
        m.apply(&r[..], &mut z[..]);
        let mut nodes = [[0.0; 3]; 2];
        m.apply(&[[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]][..], &mut nodes[..]);
        for i in 0..2 {
            for row in 0..3 {
                let back: f64 = (0..3).map(|k| block[row][k] * z[3 * i + k]).sum();
                assert!((back - r[3 * i + row]).abs() < 1e-12);
                assert_eq!(nodes[i][row], z[3 * i + row]);
            }
        }
    }

    #[test]
    fn held_lanes_pass_through() {
        let block = [[4.0, 1.0], [1.0, 3.0]];
        let mut m = BlockJacobi::with_held(&[block], &[false, true]);
        let mut z = [0.0; 2];
        m.apply(&[8.0, 5.0][..], &mut z[..]);
        assert_eq!(z, [2.0, 5.0]);
    }

    #[test]
    fn reductions_do_not_depend_on_the_thread_count() {
        let v: Vec<f64> = (0..100_000).map(|i| ((i * 7919) % 1013) as f64 * 1e-3 + 1e-9 * i as f64).collect();
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| v[..].dot(&v[..]));
        let many = rayon::ThreadPoolBuilder::new().num_threads(7).build().unwrap().install(|| v[..].dot(&v[..]));
        assert_eq!(one.to_bits(), many.to_bits());
    }

    #[test]
    fn additive_sums_two_preconditioners() {
        let mut m = Additive::<[f64], _, _>::new(Identity, Jacobi::new(&[2.0, 4.0]));
        let mut z = [0.0; 2];
        m.apply(&[2.0, 4.0][..], &mut z[..]);
        assert_eq!(z, [3.0, 5.0]);
    }
}
