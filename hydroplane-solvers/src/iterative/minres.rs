use super::{LinearOperator, Options, Preconditioner, RelativeTo, Report, Stop, Vector, initial_residual, relative};

/// MINRES (Paige–Saunders): the least-residual Krylov method for a symmetric operator that may be
/// indefinite or singular, such as a saddle-point (KKT) system, a tangent past a limit point, or a
/// shifted operator. The preconditioner must be symmetric positive definite.
///
/// The residual it tracks and stops on is in the preconditioner's norm, `‖r‖_{M⁻¹}`; with
/// [`super::Identity`] that is the Euclidean norm.
pub struct Minres<V: Vector + ?Sized> {
    pub options: Options,
    work: Option<[V::Owned; 7]>,
}

impl<V: Vector + ?Sized> Default for Minres<V> {
    fn default() -> Self {
        Self { options: Options::default(), work: None }
    }
}

impl<V: Vector + ?Sized> Minres<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: Options) -> Self {
        Self { options, work: None }
    }

    /// `A x = b` from `x` (or zero), preconditioned by `m`.
    pub fn solve(&mut self, a: &mut impl LinearOperator<V>, m: &mut impl Preconditioner<V>, b: &V, x: &mut V) -> Report {
        let options = self.options;
        let [r1, r2, y, v, w, w1, w2] = self.work.get_or_insert_with(|| core::array::from_fn(|_| b.zeros_like()));
        initial_residual(a, b, x, r1, y, options.warm_start);
        m.apply(r1, y);
        let beta1 = r1.dot(y);
        if !beta1.is_finite() {
            return Report { iterations: 0, relative_residual: f64::NAN, stop: Stop::NotFinite };
        }
        if beta1 < 0.0 {
            return Report { iterations: 0, relative_residual: f64::NAN, stop: Stop::Breakdown };
        }
        let beta1 = beta1.sqrt();
        let reference = match options.relative_to {
            RelativeTo::InitialResidual => beta1,
            RelativeTo::RightHandSide if options.warm_start => {
                m.apply(b, v);
                b.dot(v).max(0.0).sqrt()
            }
            RelativeTo::RightHandSide => beta1,
        };
        let report = |iterations, norm: f64, stop| Report { iterations, relative_residual: relative(norm, reference), stop };
        let tol = options.tolerance * reference;
        if beta1 <= f64::MIN_POSITIVE || beta1 <= tol {
            return report(0, beta1, Stop::Converged);
        }
        r2.copy_from(r1);
        w.fill_zero();
        w2.fill_zero();
        let (mut oldb, mut beta, mut dbar, mut epsln, mut phibar) = (0.0, beta1, 0.0, 0.0, beta1);
        let (mut cs, mut sn) = (-1.0f64, 0.0f64);
        let mut iterations = 0;
        while iterations < options.max_iterations {
            v.copy_from(y);
            v.scale(1.0 / beta);
            a.apply(v, y);
            if iterations > 0 {
                y.axpy(-beta / oldb, r1);
            }
            let alfa = v.dot(y);
            y.axpy(-alfa / beta, r2);
            core::mem::swap(r1, r2);
            r2.copy_from(y);
            m.apply(r2, y);
            oldb = beta;
            let bb = r2.dot(y);
            if !bb.is_finite() || !alfa.is_finite() {
                return report(iterations, phibar, Stop::NotFinite);
            }
            if bb < 0.0 {
                return report(iterations, phibar, Stop::Breakdown);
            }
            beta = bb.sqrt();
            let oldeps = epsln;
            let delta = cs * dbar + sn * alfa;
            let gbar = sn * dbar - cs * alfa;
            epsln = sn * beta;
            dbar = -cs * beta;
            let gamma = gbar.hypot(beta).max(f64::EPSILON * beta1);
            cs = gbar / gamma;
            sn = beta / gamma;
            let phi = cs * phibar;
            phibar *= sn;
            core::mem::swap(w1, w2);
            core::mem::swap(w2, w);
            w.copy_from(v);
            w.axpy(-oldeps, w1);
            w.axpy(-delta, w2);
            w.scale(1.0 / gamma);
            x.axpy(phi, w);
            iterations += 1;
            if phibar <= tol {
                return report(iterations, phibar, Stop::Converged);
            }
            if beta <= f64::MIN_POSITIVE {
                return report(iterations, phibar, Stop::Breakdown);
            }
        }
        report(iterations, phibar, Stop::MaxIterations)
    }
}
