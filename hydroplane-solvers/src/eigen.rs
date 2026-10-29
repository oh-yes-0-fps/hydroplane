//! The smallest eigenpairs of a symmetric operator, `A x = λ x` or `A x = λ B x` with `B`
//! symmetric positive definite: a structure's lowest natural frequencies (`K x = ω² M x`), its
//! buckling modes, a graph's smoothest vectors.
//!
//! [`Lobpcg`] (Knyazev's locally optimal block preconditioned conjugate gradient) takes the
//! operators as closures and any preconditioner `T ≈ A⁻¹`. With a factorization of `A` (or of a
//! shift `A − σB`) from this crate as `T` it converges in a few dozen iterations; with Jacobi it
//! still converges, more slowly.

use faer::{Mat, Side};
use rayon::prelude::*;

/// When [`Lobpcg`] stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LobpcgOptions {
    /// Each pair converges once `‖A x − λ B x‖ ≤ tolerance · |λ| ‖B x‖`, or once the residual
    /// is down to what rounding allows (a few hundred `ε ‖A‖ ‖x‖`): the smallest eigenvalues of
    /// a badly conditioned `A` (stiff penalties, contact) cannot be had to a relative accuracy
    /// below `ε ‖A‖ / λ`.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for LobpcgOptions {
    fn default() -> Self {
        Self { tolerance: 1e-8, max_iterations: 200 }
    }
}

/// The pairs found, eigenvalues ascending, eigenvectors `B`-orthonormal.
#[derive(Clone, Debug)]
pub struct Eigenpairs {
    pub values: Vec<f64>,
    pub vectors: Vec<Vec<f64>>,
    pub iterations: usize,
    /// Every pair met the tolerance.
    pub converged: bool,
}

/// LOBPCG for the smallest eigenpairs.
#[derive(Clone, Debug, Default)]
pub struct Lobpcg {
    pub options: LobpcgOptions,
}

/// Directions whose Gram eigenvalue is below this fraction of the largest are dropped from a
/// Rayleigh–Ritz basis as linearly dependent.
const DEPENDENT: f64 = 1e-12;
/// Multiples of `ε ‖A‖ ‖x‖` at which a residual is as small as rounding lets it get. On the
/// welded slab's tangent its pairs stall between 20 and 300 of these: iterating on below 300
/// only makes some of them worse.
const ROUNDING: f64 = 300.0;
/// Power-iteration steps estimating `‖A‖` before the iteration starts.
const POWER_STEPS: usize = 8;

impl Lobpcg {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: LobpcgOptions) -> Self {
        Self { options }
    }

    /// The `k` smallest pairs of `A x = λ x` (`n` unknowns), preconditioned by `t` (`z ≈ A⁻¹ r`).
    pub fn smallest(&self, n: usize, k: usize, a: impl FnMut(&[f64], &mut [f64]), t: impl FnMut(&[f64], &mut [f64])) -> Eigenpairs {
        self.solve(n, k, a, |x: &[f64], y: &mut [f64]| y.copy_from_slice(x), t)
    }

    /// The `k` smallest pairs of `A x = λ B x`.
    pub fn smallest_generalized(&self, n: usize, k: usize, a: impl FnMut(&[f64], &mut [f64]), b: impl FnMut(&[f64], &mut [f64]), t: impl FnMut(&[f64], &mut [f64])) -> Eigenpairs {
        self.solve(n, k, a, b, t)
    }

    fn solve(&self, n: usize, wanted: usize, mut a: impl FnMut(&[f64], &mut [f64]), mut b: impl FnMut(&[f64], &mut [f64]), mut t: impl FnMut(&[f64], &mut [f64])) -> Eigenpairs {
        let wanted = wanted.min(n);
        // Guard vectors past the wanted ones, so a cluster of eigenvalues at the edge of the
        // block does not slow its last members down; only the wanted ones are tested.
        let k = (wanted + (wanted / 4).max(2)).min(n);
        let mut a_norm = norm_estimate(n, &mut a);
        let apply = |op: &mut dyn FnMut(&[f64], &mut [f64]), v: &[Vec<f64>]| -> Vec<Vec<f64>> {
            v.iter()
                .map(|x| {
                    let mut y = vec![0.0; n];
                    op(x, &mut y);
                    y
                })
                .collect()
        };
        let mut x: Vec<Vec<f64>> = (0..k).map(|j| (0..n).map(|i| (((i + 1) * (j + 3) * 7919) % 1013) as f64 / 506.5 - 1.0).collect()).collect();
        let (mut ax, mut bx) = (apply(&mut a, &x), apply(&mut b, &x));
        a_norm = a_norm.max(stretch(&x, &ax));
        let mut theta = vec![0.0; k];
        let mut p: Vec<Vec<f64>> = Vec::new();
        let (mut ap, mut bp): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
        let mut w: Vec<Vec<f64>> = Vec::new();
        let (mut aw, mut bw): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
        let mut iterations = 0;
        let mut converged = false;
        loop {
            let basis: Vec<&Vec<f64>> = x.iter().chain(&w).chain(&p).collect();
            let abasis: Vec<&Vec<f64>> = ax.iter().chain(&aw).chain(&ap).collect();
            let bbasis: Vec<&Vec<f64>> = bx.iter().chain(&bw).chain(&bp).collect();
            let Some(c) = rayleigh_ritz(&basis, &abasis, &bbasis, k, &mut theta) else { break };
            let combine = |set: &[&Vec<f64>], col: usize, from: usize| -> Vec<f64> {
                let mut out = vec![0.0; n];
                for (s, v) in set.iter().enumerate().skip(from) {
                    let coef = c[(s, col)];
                    if coef != 0.0 {
                        out.par_iter_mut().zip(v.par_iter()).with_min_len(4096).for_each(|(o, v)| *o += coef * v);
                    }
                }
                out
            };
            let new_x: Vec<Vec<f64>> = (0..k).map(|j| combine(&basis, j, 0)).collect();
            if !w.is_empty() {
                p = (0..k).map(|j| combine(&basis, j, k)).collect();
                ap = (0..k).map(|j| combine(&abasis, j, k)).collect();
                bp = (0..k).map(|j| combine(&bbasis, j, k)).collect();
            }
            // Applied afresh rather than combined like `P`'s: the combination's rounding, scaled
            // up by the basis's normalization, would put a floor under the residuals that the
            // smallest eigenvalues' relative test cannot get past.
            x = new_x;
            ax = apply(&mut a, &x);
            bx = apply(&mut b, &x);
            let residuals: Vec<Vec<f64>> = (0..k).map(|j| ax[j].iter().zip(&bx[j]).map(|(a, b)| a - theta[j] * b).collect()).collect();
            let floor = ROUNDING * f64::EPSILON * a_norm;
            let unconverged = |j: &usize| {
                let r = norm(&residuals[*j]);
                let scale = theta[*j].abs().max(f64::MIN_POSITIVE) * norm(&bx[*j]);
                r > self.options.tolerance * scale && r > floor * norm(&x[*j])
            };
            if !(0..wanted).any(|j| unconverged(&j)) {
                converged = true;
                break;
            }
            let active: Vec<usize> = (0..k).filter(unconverged).collect();
            if iterations >= self.options.max_iterations {
                break;
            }
            iterations += 1;
            w = active
                .iter()
                .map(|&j| {
                    let mut z = vec![0.0; n];
                    t(&residuals[j], &mut z);
                    z
                })
                .collect();
            aw = apply(&mut a, &w);
            bw = apply(&mut b, &w);
            a_norm = a_norm.max(stretch(&w, &aw));
        }
        theta.truncate(wanted);
        x.truncate(wanted);
        Eigenpairs { values: theta, vectors: x, iterations, converged }
    }
}

/// `‖A‖` from a few power steps, from below. The iterates alone see only its low end once they
/// converge, which would put the rounding floor of the stopping rule out of reach, and the
/// iteration would run on, stalled just above it, until the iteration cap.
fn norm_estimate(n: usize, a: &mut impl FnMut(&[f64], &mut [f64])) -> f64 {
    let mut v: Vec<f64> = (0..n).map(|i| ((i * 104_729 + 7) % 2027) as f64 / 1013.5 - 1.0).collect();
    let mut av = vec![0.0; n];
    let mut estimate = 0.0f64;
    for _ in 0..POWER_STEPS {
        let length = norm(&v);
        if length == 0.0 {
            break;
        }
        a(&v, &mut av);
        let stretched = norm(&av);
        estimate = estimate.max(stretched / length);
        if !(stretched > 0.0 && stretched.is_finite()) {
            break;
        }
        v.par_iter_mut().zip(&av).with_min_len(4096).for_each(|(v, a)| *v = a / stretched);
    }
    estimate
}

/// The largest `‖A v‖ / ‖v‖` over a block: a lower bound on `‖A‖` that the iteration's own
/// products keep raising.
fn stretch(v: &[Vec<f64>], av: &[Vec<f64>]) -> f64 {
    v.iter().zip(av).map(|(x, y)| norm(y) / norm(x).max(f64::MIN_POSITIVE)).fold(0.0, f64::max)
}

fn norm(v: &[f64]) -> f64 {
    v.par_iter().with_min_len(4096).map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b).with_min_len(4096).map(|(a, b)| a * b).sum()
}

/// The `k` lowest Ritz pairs on the span of `basis`: their values into `theta`, and the
/// coefficients (columns) of their vectors over the basis, `B`-orthonormal. The basis is first
/// made `B`-orthonormal through an eigendecomposition of its Gram matrix, directions it cannot
/// tell apart dropped. `None` if fewer than `k` directions remain.
fn rayleigh_ritz(basis: &[&Vec<f64>], abasis: &[&Vec<f64>], bbasis: &[&Vec<f64>], k: usize, theta: &mut [f64]) -> Option<Mat<f64>> {
    let m = basis.len();
    let pairs: Vec<(usize, usize)> = (0..m).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let values: Vec<(f64, f64)> = pairs.par_iter().map(|&(i, j)| (dot(basis[i], abasis[j]), dot(basis[i], bbasis[j]))).collect();
    let (mut ga, mut gb) = (Mat::<f64>::zeros(m, m), Mat::<f64>::zeros(m, m));
    for (&(i, j), &(va, vb)) in pairs.iter().zip(&values) {
        ga[(i, j)] = va;
        ga[(j, i)] = va;
        gb[(i, j)] = vb;
        gb[(j, i)] = vb;
    }
    // Every direction scaled to unit `B`-norm first: near convergence the new directions are
    // tiny, and only after scaling does a small Gram eigenvalue mean dependence.
    let unit: Vec<f64> = (0..m).map(|i| if gb[(i, i)] > 0.0 { 1.0 / gb[(i, i)].sqrt() } else { 0.0 }).collect();
    for i in 0..m {
        for j in 0..m {
            gb[(i, j)] *= unit[i] * unit[j];
        }
    }
    let gram = gb.self_adjoint_eigen(Side::Lower).ok()?;
    let (s, v) = (gram.S(), gram.U());
    let largest = (0..m).map(|i| s[i]).fold(0.0f64, f64::max);
    let kept: Vec<usize> = (0..m).filter(|&i| s[i] > DEPENDENT * largest).collect();
    if kept.len() < k {
        return None;
    }
    let r = kept.len();
    let mut tr = Mat::<f64>::zeros(m, r);
    for (c, &i) in kept.iter().enumerate() {
        let scale = 1.0 / s[i].sqrt();
        for row in 0..m {
            tr[(row, c)] = v[(row, i)] * scale * unit[row];
        }
    }
    let h = tr.transpose() * &ga * &tr;
    let reduced = h.self_adjoint_eigen(Side::Lower).ok()?;
    let coefficients = &tr * reduced.U();
    for (j, t) in theta.iter_mut().enumerate().take(k) {
        *t = reduced.S()[j];
    }
    Some(coefficients)
}
