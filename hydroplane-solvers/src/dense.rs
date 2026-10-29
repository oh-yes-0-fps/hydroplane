//! Small dense symmetric positive definite systems: the coarse problems of two-level
//! preconditioners (`WᵀAW` over a handful of rigid-body modes per region).

use rayon::prelude::*;

use crate::iterative::{Element, LinearOperator, Preconditioner, Vector};

/// The Cholesky factor of a dense symmetric matrix, after a relative diagonal shift that keeps a
/// nearly singular coarse operator (modes of fixed regions, near-duplicate modes) factorable.
#[derive(Clone, Debug)]
pub struct DenseCholesky {
    n: usize,
    /// Lower triangle, row-major.
    l: Vec<f64>,
}

impl DenseCholesky {
    /// Factors the `n × n` matrix `a` (row-major; only the lower triangle is read). A pivot that
    /// is not positive is floored at `f64::EPSILON` times the largest diagonal, so the factor
    /// always exists; it is then of a perturbed matrix, which is what a preconditioner can live
    /// with.
    pub fn new(n: usize, a: &[f64]) -> Self {
        Self::shifted(n, a, 0.0)
    }

    /// As [`Self::new`] with `shift · trace / n` added to the diagonal.
    pub fn shifted(n: usize, a: &[f64], shift: f64) -> Self {
        assert_eq!(a.len(), n * n);
        let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
        let delta = shift * trace / n.max(1) as f64;
        let floor = f64::EPSILON * (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
                if i == j {
                    let d = a[i * n + i] + delta - s;
                    l[i * n + i] = if d > floor { d.sqrt() } else { floor.sqrt() };
                } else {
                    l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
                }
            }
        }
        Self { n, l }
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// `x ← A⁻¹ x`.
    pub fn solve_in_place(&self, x: &mut [f64]) {
        let (n, l) = (self.n, &self.l);
        assert_eq!(x.len(), n);
        for i in 0..n {
            let s: f64 = (0..i).map(|k| l[i * n + k] * x[k]).sum();
            x[i] = (x[i] - s) / l[i * n + i];
        }
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
            x[i] = (x[i] - s) / l[i * n + i];
        }
    }
}

const COARSE_SHIFT: f64 = 1e-10;

/// `z = W E⁻¹ Wᵀ r`: the coarse half of a two-level preconditioner, over the columns of `W` (say,
/// six rigid-body modes per region) with `E = WᵀAW` factored after a relative diagonal shift of
/// `1e-10` (modes of fixed regions, near-duplicate modes). Add it to a smoother with
/// [`crate::iterative::Additive`].
pub struct CoarseCorrection<E: Element> {
    columns: Vec<Vec<E>>,
    e: DenseCholesky,
    coefficients: Vec<f64>,
}

impl<E: Element> CoarseCorrection<E> {
    /// With `E` assembled by the caller (row-major, `m × m` for `m` columns).
    pub fn new(columns: Vec<Vec<E>>, e: &[f64]) -> Self {
        let m = columns.len();
        Self { e: DenseCholesky::shifted(m, e, COARSE_SHIFT), coefficients: vec![0.0; m], columns }
    }

    /// With `E = WᵀAW` formed by applying `a` to each column.
    pub fn galerkin(columns: Vec<Vec<E>>, a: &mut impl LinearOperator<[E]>) -> Self {
        let m = columns.len();
        let mut aw = columns.first().map(|c| c[..].zeros_like()).unwrap_or_default();
        let mut e = vec![0.0; m * m];
        for j in 0..m {
            a.apply(&columns[j], &mut aw);
            for i in 0..m {
                e[i * m + j] = columns[i][..].dot(&aw[..]);
            }
        }
        for i in 0..m {
            for j in 0..i {
                let avg = 0.5 * (e[i * m + j] + e[j * m + i]);
                e[i * m + j] = avg;
                e[j * m + i] = avg;
            }
        }
        Self::new(columns, &e)
    }
}

impl<E: Element> Preconditioner<[E]> for CoarseCorrection<E> {
    fn apply(&mut self, r: &[E], z: &mut [E]) {
        for (c, w) in self.coefficients.iter_mut().zip(&self.columns) {
            *c = w[..].dot(r);
        }
        self.e.solve_in_place(&mut self.coefficients);
        let (columns, coefficients) = (&self.columns, &self.coefficients);
        z.par_iter_mut().enumerate().with_min_len(2048).for_each(|(i, zi)| {
            *zi = E::ZERO;
            for (w, &c) in columns.iter().zip(coefficients) {
                E::axpy(zi, c, &w[i]);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iterative::{Additive, Cg, Jacobi, Options};

    #[test]
    fn solves_a_small_spd_system() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let chol = DenseCholesky::new(3, &a);
        let mut x = [1.0, 2.0, 3.0];
        chol.solve_in_place(&mut x);
        for i in 0..3 {
            let ax: f64 = (0..3).map(|k| a[i * 3 + k] * x[k]).sum();
            assert!((ax - [1.0, 2.0, 3.0][i]).abs() < 1e-12);
        }
    }

    #[test]
    fn a_singular_matrix_still_factors() {
        let a = [1.0, 1.0, 1.0, 1.0];
        let mut x = [1.0, 1.0];
        DenseCholesky::shifted(2, &a, 1e-8).solve_in_place(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    /// A long 1D Laplacian with almost no shift is slow for Jacobi alone; a coarse correction over
    /// piecewise-constant columns takes most of the smooth error away.
    #[test]
    fn a_coarse_correction_cuts_the_iterations() {
        let n = 400;
        let mut a = |x: &[f64], y: &mut [f64]| {
            for i in 0..n {
                let left = if i > 0 { x[i - 1] } else { 0.0 };
                let right = if i + 1 < n { x[i + 1] } else { 0.0 };
                y[i] = 2.0001 * x[i] - left - right;
            }
        };
        let b: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64 - 5.0).collect();
        let mut cg = Cg::with_options(Options { tolerance: 1e-8, max_iterations: 2000, ..Default::default() });
        let mut x = vec![0.0; n];
        let plain = cg.solve(&mut a, &mut Jacobi::new(&vec![2.0001; n]), &b[..], &mut x[..]);
        let columns: Vec<Vec<f64>> = (0..20).map(|c| (0..n).map(|i| if i / 20 == c { 1.0 } else { 0.0 }).collect()).collect();
        let coarse = CoarseCorrection::galerkin(columns, &mut a);
        let mut m = Additive::<[f64], _, _>::new(Jacobi::new(&vec![2.0001; n]), coarse);
        let two_level = cg.solve(&mut a, &mut m, &b[..], &mut x[..]);
        assert!(plain.converged() && two_level.converged());
        assert!(two_level.iterations * 2 < plain.iterations, "{} vs {}", two_level.iterations, plain.iterations);
    }
}
