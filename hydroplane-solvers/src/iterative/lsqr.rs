use super::Vector;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsqrOptions {
    /// Least-squares stop: `‖Aᵀr‖ ≤ atol · ‖A‖ ‖r‖`. Also the relative accuracy of `A` in the
    /// consistent-system stop below.
    pub atol: f64,
    /// Consistent-system stop: `‖r‖ ≤ btol · ‖b‖ + atol · ‖A‖ ‖x‖`.
    pub btol: f64,
    /// Tikhonov damping `λ`: minimizes `‖A x − b‖² + λ² ‖x − x₀‖²`.
    pub damp: f64,
    pub max_iterations: usize,
    /// Start from the `x` passed in (the damping then pulls towards it); else from zero.
    pub warm_start: bool,
}

impl Default for LsqrOptions {
    fn default() -> Self {
        Self { atol: 1e-8, btol: 1e-8, damp: 0.0, max_iterations: 1000, warm_start: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LsqrReport {
    pub iterations: usize,
    /// Estimates as the iteration tracked them: `‖b − A x‖` (with the damping term), `‖Aᵀr‖`,
    /// and the Frobenius norm and condition number of `[A; λI]`.
    pub residual: f64,
    pub normal_residual: f64,
    pub norm_a: f64,
    pub cond_a: f64,
    pub stop: LsqrStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LsqrStop {
    /// `A x = b` to `btol`.
    Consistent,
    /// `x` is a least-squares solution to `atol`.
    LeastSquares,
    /// `b` is zero (or in the null space of `Aᵀ`): `x` is the start.
    Trivial,
    MaxIterations,
    NotFinite,
}

impl LsqrReport {
    pub fn converged(&self) -> bool {
        matches!(self.stop, LsqrStop::Consistent | LsqrStop::LeastSquares | LsqrStop::Trivial)
    }
}

/// LSQR (Paige–Saunders): `min ‖A x − b‖` (plus damping) for a rectangular or rank-deficient `A`
/// given by its products, `A x` and `Aᵀ u` (calibration, fitting, a mesh's best-fit
/// displacements). Equivalent to CG on the normal equations, without squaring the conditioning.
/// `x` lives in `V` (columns), `b` in `U` (rows).
pub struct Lsqr<V: Vector + ?Sized, U: Vector + ?Sized> {
    pub options: LsqrOptions,
    columns: Option<[V::Owned; 3]>,
    rows: Option<[U::Owned; 2]>,
}

impl<V: Vector + ?Sized, U: Vector + ?Sized> Default for Lsqr<V, U> {
    fn default() -> Self {
        Self::with_options(LsqrOptions::default())
    }
}

impl<V: Vector + ?Sized, U: Vector + ?Sized> Lsqr<V, U> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: LsqrOptions) -> Self {
        Self { options, columns: None, rows: None }
    }

    /// `a(x, y)` sets `y = A x`; `at(u, v)` sets `v = Aᵀ u`.
    pub fn solve(&mut self, mut a: impl FnMut(&V, &mut U), mut at: impl FnMut(&U, &mut V), b: &U, x: &mut V) -> LsqrReport {
        let options = self.options;
        let [v, w, s] = self.columns.get_or_insert_with(|| core::array::from_fn(|_| x.zeros_like()));
        let [u, t] = self.rows.get_or_insert_with(|| core::array::from_fn(|_| b.zeros_like()));
        let (v, w, s, u, t): (&mut V, &mut V, &mut V, &mut U, &mut U) = (v, w, s, u, t);
        let damp = options.damp;
        if options.warm_start {
            a(x, t);
            u.copy_from(b);
            u.axpy(-1.0, t);
        } else {
            x.fill_zero();
            u.copy_from(b);
        }
        let bnorm = b.dot(b).sqrt();
        let mut beta = u.dot(u).sqrt();
        let mut report = LsqrReport { iterations: 0, residual: beta, normal_residual: 0.0, norm_a: 0.0, cond_a: 0.0, stop: LsqrStop::Trivial };
        if !beta.is_finite() {
            report.stop = LsqrStop::NotFinite;
            return report;
        }
        if beta == 0.0 {
            return report;
        }
        u.scale(1.0 / beta);
        at(u, v);
        let mut alpha = v.dot(v).sqrt();
        if alpha == 0.0 {
            return report;
        }
        v.scale(1.0 / alpha);
        w.copy_from(v);
        let (mut phibar, mut rhobar) = (beta, alpha);
        let (mut anorm2, mut dnorm2, mut res2) = (0.0f64, 0.0f64, 0.0f64);
        report.stop = LsqrStop::MaxIterations;
        while report.iterations < options.max_iterations {
            a(v, t);
            u.xpay(t, -alpha);
            beta = u.dot(u).sqrt();
            anorm2 += alpha * alpha + beta * beta + damp * damp;
            if beta > 0.0 {
                u.scale(1.0 / beta);
                at(u, s);
                v.xpay(s, -beta);
                alpha = v.dot(v).sqrt();
                if alpha > 0.0 {
                    v.scale(1.0 / alpha);
                }
            }
            let rhobar1 = rhobar.hypot(damp);
            let (cs1, sn1) = (rhobar / rhobar1, damp / rhobar1);
            let psi = sn1 * phibar;
            phibar *= cs1;
            let rho = rhobar1.hypot(beta);
            let (cs, sn) = (rhobar1 / rho, beta / rho);
            let theta = sn * alpha;
            rhobar = -cs * alpha;
            let phi = cs * phibar;
            phibar *= sn;
            let tau = sn * phi;
            dnorm2 += w.dot(w) / (rho * rho);
            x.axpy(phi / rho, w);
            w.xpay(v, -theta / rho);
            report.iterations += 1;
            res2 += psi * psi;
            let rnorm = (phibar * phibar + res2).sqrt();
            let anorm = anorm2.sqrt();
            let arnorm = alpha * tau.abs();
            report.residual = rnorm;
            report.normal_residual = arnorm;
            report.norm_a = anorm;
            report.cond_a = anorm * dnorm2.sqrt();
            if !(rnorm.is_finite() && arnorm.is_finite()) {
                report.stop = LsqrStop::NotFinite;
                break;
            }
            let xnorm = x.dot(x).sqrt();
            if rnorm <= options.btol * bnorm + options.atol * anorm * xnorm {
                report.stop = LsqrStop::Consistent;
                break;
            }
            if arnorm <= options.atol * anorm * rnorm || alpha == 0.0 || beta == 0.0 {
                report.stop = LsqrStop::LeastSquares;
                break;
            }
        }
        report
    }
}
