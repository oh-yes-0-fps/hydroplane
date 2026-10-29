use super::{LinearOperator, Preconditioner, Vector};

/// An interval holding the eigenvalues of `M⁻¹A` that matter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spectrum {
    pub min: f64,
    pub max: f64,
}

impl Spectrum {
    /// The upper part of a spectrum whose largest eigenvalue is about `max`, `[0.1, 1.1]·max`:
    /// what a multigrid smoother damps (the lower part is the coarse level's job). The margin above
    /// covers an estimate from a few Lanczos steps, which approaches the largest eigenvalue from
    /// below.
    pub fn smoothing(max: f64) -> Self {
        Self { min: 0.1 * max, max: 1.1 * max }
    }
}

/// Lanczos steps of [`estimate_spectrum`].
const LANCZOS_STEPS: usize = 10;

/// The extreme eigenvalues of `M⁻¹A` (symmetric `A`, symmetric positive definite `M`) estimated by
/// ten steps of Lanczos, run as preconditioned CG from `start` (a vector with some of every
/// eigenvector in it: random, not smooth). The interval is inside the true one and tightens fast
/// at both ends; ten steps place the largest eigenvalue within a few percent.
pub fn estimate_spectrum<V: Vector + ?Sized>(a: &mut impl LinearOperator<V>, m: &mut impl Preconditioner<V>, start: &V) -> Spectrum {
    let steps = LANCZOS_STEPS;
    let mut r = start.zeros_like();
    r.copy_from(start);
    let (mut z, mut p, mut q) = (start.zeros_like(), start.zeros_like(), start.zeros_like());
    m.apply(&r, &mut z);
    p.copy_from(&z);
    let mut rz = r.dot(&z);
    let mut diag = Vec::with_capacity(steps);
    let mut off: Vec<f64> = Vec::with_capacity(steps);
    let (mut alpha_prev, mut beta_prev) = (1.0, 0.0);
    for k in 0..steps {
        a.apply(&p, &mut q);
        let pq = p.dot(&q);
        if !(pq.is_finite() && rz > 0.0) || pq == 0.0 {
            break;
        }
        let alpha = rz / pq;
        diag.push(1.0 / alpha + if k > 0 { beta_prev / alpha_prev } else { 0.0 });
        r.axpy(-alpha, &q);
        m.apply(&r, &mut z);
        let rz_new = r.dot(&z);
        let beta = rz_new / rz;
        if k + 1 < steps {
            if !(beta.is_finite() && beta > 0.0) {
                break;
            }
            off.push(beta.sqrt() / alpha);
        }
        p.xpay(&z, beta);
        (alpha_prev, beta_prev, rz) = (alpha, beta, rz_new);
    }
    off.truncate(diag.len().saturating_sub(1));
    if diag.is_empty() {
        return Spectrum { min: 0.0, max: 0.0 };
    }
    let (min, max) = tridiagonal_extremes(&diag, &off);
    Spectrum { min, max }
}

/// The smallest and largest eigenvalue of the symmetric tridiagonal matrix with diagonal `d` and
/// off-diagonal `e`, by bisection on Sturm counts.
fn tridiagonal_extremes(d: &[f64], e: &[f64]) -> (f64, f64) {
    let n = d.len();
    let radius = |i: usize| if i > 0 { e[i - 1].abs() } else { 0.0 } + if i < n - 1 { e[i].abs() } else { 0.0 };
    let lo = (0..n).map(|i| d[i] - radius(i)).fold(f64::INFINITY, f64::min);
    let hi = (0..n).map(|i| d[i] + radius(i)).fold(f64::NEG_INFINITY, f64::max);
    let below = |x: f64| {
        let mut count = 0;
        let mut q = 1.0;
        for i in 0..n {
            let e2 = if i > 0 { e[i - 1] * e[i - 1] } else { 0.0 };
            q = d[i] - x - if i > 0 { e2 / q } else { 0.0 };
            if q == 0.0 {
                q = -f64::EPSILON * (d[i].abs() + x.abs()).max(f64::MIN_POSITIVE);
            }
            if q < 0.0 {
                count += 1;
            }
        }
        count
    };
    let kth = |k: usize| {
        let (mut a, mut b) = (lo, hi);
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if mid <= a || mid >= b {
                break;
            }
            if below(mid) > k { b = mid } else { a = mid }
        }
        0.5 * (a + b)
    };
    (kth(0), kth(n - 1))
}

/// Chebyshev iteration of a fixed degree from zero: `z = p(M⁻¹A) M⁻¹ r` with `p` the polynomial
/// that is smallest on [`Spectrum`]. A linear, symmetric positive definite operator for a spectrum
/// that covers the top of `M⁻¹A`'s (so a preconditioner for [`super::Cg`]), with no inner products:
/// the usual smoother of polynomial and p-multigrid. `M` is typically [`super::Jacobi`] or
/// [`super::BlockJacobi`].
pub struct Chebyshev<V: Vector + ?Sized, A, M> {
    pub operator: A,
    pub inner: M,
    pub spectrum: Spectrum,
    /// Applications of `M` per application (3 by default); `degree − 1` of `A`.
    pub degree: usize,
    work: Option<[V::Owned; 3]>,
}

impl<V: Vector + ?Sized, A, M> Chebyshev<V, A, M> {
    pub fn new(operator: A, inner: M, spectrum: Spectrum) -> Self {
        Self { operator, inner, spectrum, degree: 3, work: None }
    }
}

impl<V: Vector + ?Sized, A: LinearOperator<V>, M: Preconditioner<V>> Preconditioner<V> for Chebyshev<V, A, M> {
    fn apply(&mut self, r: &V, z: &mut V) {
        let [res, d, t] = self.work.get_or_insert_with(|| core::array::from_fn(|_| r.zeros_like()));
        let (res, d, t): (&mut V, &mut V, &mut V) = (res, d, t);
        let theta = 0.5 * (self.spectrum.max + self.spectrum.min);
        let delta = 0.5 * (self.spectrum.max - self.spectrum.min);
        res.copy_from(r);
        self.inner.apply(res, d);
        d.scale(1.0 / theta);
        z.copy_from(d);
        if delta <= 0.0 {
            return;
        }
        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;
        for _ in 1..self.degree.max(1) {
            self.operator.apply(d, t);
            res.axpy(-1.0, t);
            self.inner.apply(res, t);
            let rho_new = 1.0 / (2.0 * sigma - rho);
            d.scale(rho_new * rho);
            d.axpy(2.0 * rho_new / delta, t);
            z.axpy(1.0, d);
            rho = rho_new;
        }
    }
}
