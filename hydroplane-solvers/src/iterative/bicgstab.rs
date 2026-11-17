use super::{LinearOperator, Options, Preconditioner, RelativeTo, Report, Stop, Vector, initial_residual, relative};

/// BiCGStab (van der Vorst) with right preconditioning: a nonsymmetric solver in constant memory,
/// two operator and two preconditioner applications per iteration. It can break down where GMRES
/// cannot ([`Stop::Breakdown`]); its residual is the true `‖b − A x‖`, as updated.
pub struct BiCgStab<V: Vector + ?Sized> {
    pub options: Options,
    work: Option<[V::Owned; 7]>,
}

impl<V: Vector + ?Sized> Default for BiCgStab<V> {
    fn default() -> Self {
        Self { options: Options::default(), work: None }
    }
}

impl<V: Vector + ?Sized> BiCgStab<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: Options) -> Self {
        Self { options, work: None }
    }

    /// `A x = b` from `x` (or zero), preconditioned on the right by `m`.
    pub fn solve(&mut self, a: &mut impl LinearOperator<V>, m: &mut impl Preconditioner<V>, b: &V, x: &mut V) -> Report {
        let options = self.options;
        let [r, shadow, p, v, s, t, z] = self.work.get_or_insert_with(|| core::array::from_fn(|_| b.zeros_like()));
        let (r, shadow, p, v, s, t, z): (&mut V, &mut V, &mut V, &mut V, &mut V, &mut V, &mut V) = (r, shadow, p, v, s, t, z);
        initial_residual(a, b, x, r, t, options.warm_start);
        let rr0 = r.dot(r);
        let reference = match options.relative_to {
            RelativeTo::InitialResidual => rr0.sqrt(),
            RelativeTo::RightHandSide => b.dot(b).sqrt(),
        };
        let report = |iterations, rr: f64, stop| Report { iterations, relative_residual: relative(rr.sqrt(), reference), stop };
        if !rr0.is_finite() {
            return report(0, rr0, Stop::NotFinite);
        }
        let tol2 = (options.tolerance * reference).powi(2);
        if rr0 <= f64::MIN_POSITIVE || rr0 <= tol2 {
            return report(0, rr0, Stop::Converged);
        }
        shadow.copy_from(r);
        p.copy_from(r);
        let (mut rho, mut rr) = (rr0, rr0);
        let mut iterations = 0;
        while iterations < options.max_iterations {
            m.apply(p, z);
            a.apply(z, v);
            let sv = shadow.dot(v);
            if sv == 0.0 || !sv.is_finite() {
                return report(iterations, rr, if sv.is_finite() { Stop::Breakdown } else { Stop::NotFinite });
            }
            let alpha = rho / sv;
            x.axpy(alpha, z);
            s.copy_from(r);
            s.axpy(-alpha, v);
            iterations += 1;
            let ss = s.dot(s);
            if ss <= tol2 {
                r.copy_from(s);
                return report(iterations, ss, Stop::Converged);
            }
            m.apply(s, z);
            a.apply(z, t);
            let tt = t.dot(t);
            let omega = if tt > 0.0 { t.dot(s) / tt } else { 0.0 };
            x.axpy(omega, z);
            r.copy_from(s);
            r.axpy(-omega, t);
            rr = r.dot(r);
            if !rr.is_finite() {
                return report(iterations, rr, Stop::NotFinite);
            }
            if rr <= tol2 {
                return report(iterations, rr, Stop::Converged);
            }
            let rho_new = shadow.dot(r);
            if omega == 0.0 || rho_new == 0.0 {
                return report(iterations, rr, Stop::Breakdown);
            }
            let beta = (rho_new / rho) * (alpha / omega);
            rho = rho_new;
            p.axpy(-omega, v);
            p.xpay(r, beta);
        }
        report(iterations, rr, Stop::MaxIterations)
    }
}
