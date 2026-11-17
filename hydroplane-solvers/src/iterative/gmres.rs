use super::{LinearOperator, Options, Preconditioner, RelativeTo, Report, Stop, Vector, initial_residual, relative};

/// Restarted GMRES with right preconditioning, for any operator: nonsymmetric (convection,
/// friction, follower loads), or symmetric with a nonsymmetric preconditioner. The residual it
/// tracks and stops on is the true `‖b − A x‖`, recomputed at every restart.
///
/// GMRES([`Options::restart`]) keeps `restart + 1` basis vectors. Flexible (FGMRES,
/// [`Options::flexible`]) keeps every preconditioned direction too, a second basis, and then
/// tolerates a preconditioner that changes between applications (an inner iteration, a
/// multigrid cycle).
pub struct Gmres<V: Vector + ?Sized> {
    pub options: Options,
    basis: Vec<V::Owned>,
    directions: Vec<V::Owned>,
    scratch: Option<[V::Owned; 2]>,
}

impl<V: Vector + ?Sized> Default for Gmres<V> {
    fn default() -> Self {
        Self::with_options(Options::default())
    }
}

impl<V: Vector + ?Sized> Gmres<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_options(options: Options) -> Self {
        Self { options, basis: Vec::new(), directions: Vec::new(), scratch: None }
    }

    /// `A x = b` from `x` (or zero), preconditioned on the right by `m`.
    pub fn solve(&mut self, a: &mut impl LinearOperator<V>, m: &mut impl Preconditioner<V>, b: &V, x: &mut V) -> Report {
        let options = self.options;
        let flexible = options.flexible;
        let k = options.restart.max(1);
        while self.basis.len() < k + 1 {
            self.basis.push(b.zeros_like());
        }
        while flexible && self.directions.len() < k {
            self.directions.push(b.zeros_like());
        }
        let [r, t] = self.scratch.get_or_insert_with(|| [b.zeros_like(), b.zeros_like()]);
        let (r, t): (&mut V, &mut V) = (r, t);
        initial_residual(a, b, x, r, t, options.warm_start);
        let mut beta = r.dot(r).sqrt();
        let reference = match options.relative_to {
            RelativeTo::InitialResidual => beta,
            RelativeTo::RightHandSide => b.dot(b).sqrt(),
        };
        let report = |iterations, norm: f64, stop| Report { iterations, relative_residual: relative(norm, reference), stop };
        if !beta.is_finite() {
            return report(0, beta, Stop::NotFinite);
        }
        let tol = options.tolerance * reference;
        if beta <= f64::MIN_POSITIVE || beta <= tol {
            return report(0, beta, Stop::Converged);
        }
        let mut h = vec![0.0f64; (k + 1) * k];
        let at = |i: usize, j: usize| i * k + j;
        let mut g = vec![0.0f64; k + 1];
        let mut cs = vec![0.0f64; k];
        let mut sn = vec![0.0f64; k];
        let mut coefficients = vec![0.0f64; k];
        let mut iterations = 0;
        loop {
            self.basis[0].copy_from(r);
            self.basis[0].scale(1.0 / beta);
            g.fill(0.0);
            g[0] = beta;
            h.fill(0.0);
            let mut used = 0;
            let mut estimate = beta;
            let mut lucky = false;
            while used < k && iterations < options.max_iterations {
                let j = used;
                let (done, rest) = self.basis.split_at_mut(j + 1);
                let w: &mut V = &mut rest[0];
                if flexible {
                    let z: &mut V = &mut self.directions[j];
                    m.apply(&done[j], z);
                    a.apply(z, w);
                } else {
                    m.apply(&done[j], t);
                    a.apply(t, w);
                }
                // Classical Gram–Schmidt, twice: each pass reads the basis once, and the second
                // restores orthogonality to working precision when the new vector is nearly in
                // the span already.
                let basis: Vec<&V> = done.iter().map(|v| &**v).collect();
                for _ in 0..2 {
                    w.dots(&basis, &mut coefficients[..=j]);
                    for (i, c) in coefficients[..=j].iter_mut().enumerate() {
                        h[at(i, j)] += *c;
                        *c = -*c;
                    }
                    w.axpys(&coefficients[..=j], &basis);
                }
                let hn = w.dot(w).sqrt();
                h[at(j + 1, j)] = hn;
                if !hn.is_finite() {
                    return report(iterations, estimate, Stop::NotFinite);
                }
                for i in 0..j {
                    let (a0, a1) = (h[at(i, j)], h[at(i + 1, j)]);
                    h[at(i, j)] = cs[i] * a0 + sn[i] * a1;
                    h[at(i + 1, j)] = -sn[i] * a0 + cs[i] * a1;
                }
                let (a0, a1) = (h[at(j, j)], h[at(j + 1, j)]);
                let d = a0.hypot(a1);
                (cs[j], sn[j]) = if d > 0.0 { (a0 / d, a1 / d) } else { (1.0, 0.0) };
                h[at(j, j)] = d;
                h[at(j + 1, j)] = 0.0;
                g[j + 1] = -sn[j] * g[j];
                g[j] *= cs[j];
                estimate = g[j + 1].abs();
                used += 1;
                iterations += 1;
                if hn <= f64::EPSILON * beta {
                    lucky = true;
                    break;
                }
                if estimate <= tol {
                    break;
                }
                w.scale(1.0 / hn);
            }
            let mut y = g[..used].to_vec();
            for i in (0..used).rev() {
                let s: f64 = (i + 1..used).map(|c| h[at(i, c)] * y[c]).sum();
                y[i] = if h[at(i, i)] != 0.0 { (y[i] - s) / h[at(i, i)] } else { 0.0 };
            }
            if flexible {
                for (z, &yi) in self.directions.iter().zip(&y) {
                    x.axpy(yi, z);
                }
            } else {
                r.fill_zero();
                for (v, &yi) in self.basis.iter().zip(&y) {
                    r.axpy(yi, v);
                }
                m.apply(r, t);
                x.axpy(1.0, t);
            }
            a.apply(x, t);
            r.copy_from(b);
            r.axpy(-1.0, t);
            beta = r.dot(r).sqrt();
            if !beta.is_finite() {
                return report(iterations, beta, Stop::NotFinite);
            }
            if beta <= tol {
                return report(iterations, beta, Stop::Converged);
            }
            if iterations >= options.max_iterations {
                return report(iterations, beta, Stop::MaxIterations);
            }
            if lucky {
                return report(iterations, beta, Stop::Breakdown);
            }
        }
    }
}
