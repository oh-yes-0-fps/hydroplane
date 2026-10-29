use hydroplane_solvers::dense::DenseCholesky;
use hydroplane_solvers::iterative::{
    BiCgStab, Cg, Chebyshev, Gmres, Identity, Jacobi, Lsqr, LsqrOptions, LsqrStop, Minres, Options, RelativeTo, Spectrum,
    Stop, estimate_spectrum,
};

/// An `n × n` grid's 5-point operator `(4 + shift) u_ij − neighbours`, plus a convection term
/// `c (u_ij − u_{i−1,j})` (upwind, nonsymmetric when `c ≠ 0`).
fn grid(n: usize, shift: f64, c: f64) -> impl FnMut(&[f64], &mut [f64]) + Clone {
    move |x: &[f64], y: &mut [f64]| {
        for j in 0..n {
            for i in 0..n {
                let at = |i: usize, j: usize| x[j * n + i];
                let k = j * n + i;
                let w = if i > 0 { at(i - 1, j) } else { 0.0 };
                let e = if i + 1 < n { at(i + 1, j) } else { 0.0 };
                let s = if j > 0 { at(i, j - 1) } else { 0.0 };
                let nn = if j + 1 < n { at(i, j + 1) } else { 0.0 };
                y[k] = (4.0 + shift + c) * x[k] - w - e - s - nn - c * w;
            }
        }
    }
}

fn rhs(len: usize) -> Vec<f64> {
    (0..len).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect()
}

fn true_residual(a: &mut impl FnMut(&[f64], &mut [f64]), b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a(x, &mut ax);
    let r: f64 = b.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum();
    (r / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

#[test]
fn minres_solves_an_indefinite_shifted_laplacian() {
    let n = 24;
    let mut a = grid(n, -1.3, 0.0);
    let b = rhs(n * n);
    let mut x = vec![0.0; n * n];
    let report = Minres::with_options(Options { tolerance: 1e-10, ..Default::default() }).solve(&mut a, &mut Identity, &b[..], &mut x[..]);
    assert!(report.converged(), "{report:?}");
    assert!(true_residual(&mut a, &b, &x) < 1e-9);
    let mut x_cg = vec![0.0; n * n];
    let cg = Cg::with_options(Options { tolerance: 1e-10, ..Default::default() }).solve(&mut a, &mut Identity, &b[..], &mut x_cg[..]);
    assert_eq!(cg.stop, Stop::NegativeCurvature, "CG cannot take this operator");
}

/// `[K Bᵀ; B −εI]` with `K` a grid Laplacian and `B` averaging pairs of unknowns: a regularized
/// saddle point, preconditioned by the positive block diagonal `diag(diag K, 1)`.
#[test]
fn minres_solves_a_saddle_point_with_a_positive_preconditioner() {
    let n = 16;
    let nk = n * n;
    let nc = nk / 4;
    let mut k = grid(n, 0.0, 0.0);
    let mut kx = vec![0.0; nk];
    let mut a = move |x: &[f64], y: &mut [f64]| {
        k(&x[..nk], &mut kx);
        y[..nk].copy_from_slice(&kx);
        for c in 0..nc {
            let (p, q) = (4 * c, 4 * c + 1);
            let lam = x[nk + c];
            y[p] += 0.5 * lam;
            y[q] += 0.5 * lam;
            y[nk + c] = 0.5 * (x[p] + x[q]) - 1e-8 * lam;
        }
    };
    let mut m = |r: &[f64], z: &mut [f64]| {
        for i in 0..nk {
            z[i] = r[i] / 4.0;
        }
        for i in nk..nk + nc {
            z[i] = r[i] * 8.0;
        }
    };
    let b = rhs(nk + nc);
    let mut x = vec![0.0; nk + nc];
    let report = Minres::with_options(Options { tolerance: 1e-10, ..Default::default() }).solve(&mut a, &mut m, &b[..], &mut x[..]);
    assert!(report.converged(), "{report:?}");
    assert!(true_residual(&mut a, &b, &x) < 1e-8, "{}", true_residual(&mut a, &b, &x));
}

#[test]
fn gmres_and_bicgstab_solve_convection_diffusion() {
    let n = 30;
    let b = rhs(n * n);
    let options = Options { tolerance: 1e-10, max_iterations: 2000, ..Default::default() };
    let mut a = grid(n, 0.0, 3.0);
    let diag = vec![7.0; n * n];
    for restart in [10, 40] {
        let mut x = vec![0.0; n * n];
        let report = Gmres::with_options(Options { restart, ..options }).solve(&mut a, &mut Jacobi::new(&diag), &b[..], &mut x[..]);
        assert!(report.converged(), "GMRES({restart}): {report:?}");
        assert!(true_residual(&mut a, &b, &x) < 1e-9);
    }
    let mut x = vec![0.0; n * n];
    let report = BiCgStab::with_options(options).solve(&mut a, &mut Jacobi::new(&diag), &b[..], &mut x[..]);
    assert!(report.converged(), "BiCGStab: {report:?}");
    assert!(true_residual(&mut a, &b, &x) < 1e-9);
}

/// A preconditioner that changes every application (a growing number of Jacobi sweeps): FGMRES
/// still converges to the true residual.
#[test]
fn flexible_gmres_takes_a_varying_preconditioner() {
    let n = 20;
    let b = rhs(n * n);
    let mut a = grid(n, 0.0, 2.0);
    let mut inner = grid(n, 0.0, 2.0);
    let mut calls = 0usize;
    let mut scratch = vec![0.0; n * n];
    let mut m = move |r: &[f64], z: &mut [f64]| {
        calls += 1;
        z.iter_mut().zip(r).for_each(|(z, r)| *z = r / 6.0);
        for _ in 0..1 + calls % 4 {
            inner(z, &mut scratch);
            for i in 0..z.len() {
                z[i] += 0.8 * (r[i] - scratch[i]) / 6.0;
            }
        }
    };
    let mut x = vec![0.0; n * n];
    let report = Gmres::with_options(Options { restart: 20, flexible: true, tolerance: 1e-10, ..Default::default() }).solve(&mut a, &mut m, &b[..], &mut x[..]);
    assert!(report.converged(), "{report:?}");
    assert!(true_residual(&mut a, &b, &x) < 1e-9);
}

#[test]
fn gmres_tolerance_is_relative_to_the_right_hand_side_on_request() {
    let n = 12;
    let b = rhs(n * n);
    let mut a = grid(n, 0.5, 1.0);
    let mut x = vec![0.0; n * n];
    let options = Options { tolerance: 1e-12, ..Default::default() };
    Gmres::with_options(options).solve(&mut a, &mut Identity, &b[..], &mut x[..]);
    let warm = Options { warm_start: true, tolerance: 1e-6, relative_to: RelativeTo::RightHandSide, ..options };
    assert_eq!(Gmres::with_options(warm).solve(&mut a, &mut Identity, &b[..], &mut x[..]).iterations, 0);
}

#[test]
fn the_spectrum_estimate_brackets_a_known_diagonal() {
    let d: Vec<f64> = (1..=400).map(|i| i as f64 * 0.25).collect();
    let mut a = |x: &[f64], y: &mut [f64]| y.iter_mut().zip(x).zip(&d).for_each(|((y, x), d)| *y = d * x);
    let start = rhs(400);
    let s = estimate_spectrum(&mut a, &mut Identity, &start[..]);
    assert!(s.max <= 100.0 + 1e-9 && s.max > 97.0, "{s:?}");
    assert!(s.min >= 0.25 - 1e-9 && s.min < 2.0, "{s:?}");
}

/// Chebyshev–Jacobi, degree 4, as a CG preconditioner on a Poisson grid: fewer CG iterations than
/// Jacobi alone, and every application the same linear operator (standard CG converges).
#[test]
fn chebyshev_preconditions_cg() {
    let n = 40;
    let b = rhs(n * n);
    let mut a = grid(n, 0.0, 0.0);
    let diag = vec![4.0; n * n];
    let options = Options { tolerance: 1e-10, max_iterations: 2000, ..Default::default() };
    let mut x = vec![0.0; n * n];
    let plain = Cg::with_options(options).solve(&mut a, &mut Jacobi::new(&diag), &b[..], &mut x[..]);
    let lmax = estimate_spectrum(&mut a, &mut Jacobi::new(&diag), &rhs(n * n)[..]).max;
    let full = Spectrum { min: lmax / 400.0, max: 1.1 * lmax };
    let mut m = Chebyshev::new(grid(n, 0.0, 0.0), Jacobi::new(&diag), full);
    m.degree = 4;
    let cheb = Cg::with_options(options).solve(&mut a, &mut m, &b[..], &mut x[..]);
    assert!(plain.converged() && cheb.converged(), "{plain:?} {cheb:?}");
    assert!(cheb.iterations * 2 < plain.iterations, "{} vs {}", cheb.iterations, plain.iterations);
    assert!(true_residual(&mut a, &b, &x) < 1e-9);
    let mut smoother = Chebyshev::new(grid(n, 0.0, 0.0), Jacobi::new(&diag), Spectrum::smoothing(lmax));
    let smoothed = Cg::with_options(options).solve(&mut a, &mut smoother, &b[..], &mut x[..]);
    assert!(smoothed.converged() && smoothed.iterations < plain.iterations);
}

/// An overdetermined dense fit, against the normal equations solved directly.
#[test]
fn lsqr_matches_the_normal_equations() {
    let (m, n) = (120, 15);
    let a: Vec<f64> = (0..m * n).map(|k| (((k * 2654435761usize) % 1000) as f64 / 500.0 - 1.0) + if k % (n + 1) == 0 { 2.0 } else { 0.0 }).collect();
    let b = rhs(m);
    for damp in [0.0, 0.7] {
        let mut ata = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                ata[i * n + j] = (0..m).map(|r| a[r * n + i] * a[r * n + j]).sum::<f64>() + if i == j { damp * damp } else { 0.0 };
            }
        }
        let mut expect: Vec<f64> = (0..n).map(|j| (0..m).map(|r| a[r * n + j] * b[r]).sum()).collect();
        DenseCholesky::new(n, &ata).solve_in_place(&mut expect);
        let mut x = vec![0.0; n];
        let report = Lsqr::with_options(LsqrOptions { atol: 1e-12, btol: 1e-12, damp, ..Default::default() }).solve(
            |x: &[f64], y: &mut [f64]| y.iter_mut().enumerate().for_each(|(r, y)| *y = (0..n).map(|j| a[r * n + j] * x[j]).sum()),
            |u: &[f64], v: &mut [f64]| v.iter_mut().enumerate().for_each(|(j, v)| *v = (0..m).map(|r| a[r * n + j] * u[r]).sum()),
            &b[..],
            &mut x[..],
        );
        assert_eq!(report.stop, LsqrStop::LeastSquares, "{report:?}");
        let err = x.iter().zip(&expect).map(|(x, e)| (x - e).abs()).fold(0.0, f64::max);
        assert!(err < 1e-9, "damp {damp}: {err} {report:?}");
    }
}

#[test]
fn lsqr_solves_a_consistent_square_system() {
    let mut a = grid(5, 0.3, 1.0);
    let mut at = |u: &[f64], v: &mut [f64]| {
        let mut e = vec![0.0; 25];
        v.fill(0.0);
        for k in 0..25 {
            e.fill(0.0);
            e[k] = 1.0;
            let mut col = vec![0.0; 25];
            grid(5, 0.3, 1.0)(&e, &mut col);
            v[k] = col.iter().zip(u).map(|(c, u)| c * u).sum();
        }
    };
    let b = rhs(25);
    let mut x = vec![0.0; 25];
    let report = Lsqr::with_options(LsqrOptions { atol: 1e-12, btol: 1e-12, ..Default::default() }).solve(&mut a, &mut at, &b[..], &mut x[..]);
    assert!(report.converged(), "{report:?}");
    assert!(true_residual(&mut a, &b, &x) < 1e-10);
}

/// The fused reductions GMRES uses match the single ones bit for bit.
#[test]
fn fused_dots_and_axpys_match_the_single_ones() {
    use hydroplane_solvers::iterative::Vector;
    let n = 10_000;
    let v: Vec<Vec<f64>> = (0..5).map(|k| (0..n).map(|i| ((i * (k + 3) * 7919) % 1013) as f64 * 1e-3 - 0.4).collect()).collect();
    let refs: Vec<&[f64]> = v.iter().map(|x| &x[..]).collect();
    let mut fused = [0.0; 5];
    v[0][..].dots(&refs, &mut fused);
    for k in 0..5 {
        assert_eq!(fused[k].to_bits(), v[0][..].dot(&v[k][..]).to_bits());
    }
    let a = [0.5, -1.25, 2.0, 0.0, 3.5];
    let mut one = v[1].clone();
    for k in 0..5 {
        one[..].axpy(a[k], &v[k][..]);
    }
    let mut many = v[1].clone();
    many[..].axpys(&a, &refs);
    assert!(one.iter().zip(&many).all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// Sparse QR's least-squares solution matches LSQR's on a sparse overdetermined fit.
#[test]
fn sparse_qr_matches_lsqr() {
    use hydroplane_solvers::qr::CpuQr;
    let (m, n) = (900, 120);
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    for r in 0..m {
        entries.push((r, r % n, 2.0));
        entries.push((r, (r * 7 + 3) % n, ((r % 5) as f64) - 2.0));
    }
    entries.sort_by_key(|&(r, c, _)| (c, r));
    entries.dedup_by_key(|e| (e.0, e.1));
    let mut col_ptr = vec![0; n + 1];
    for &(_, c, _) in &entries {
        col_ptr[c + 1] += 1;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    let row_idx: Vec<usize> = entries.iter().map(|e| e.0).collect();
    let val: Vec<f64> = entries.iter().map(|e| e.2).collect();
    let b = rhs(m);
    let mut qr = CpuQr::<f64>::new(m, &col_ptr, &row_idx).unwrap();
    qr.factorize(&val).unwrap();
    let mut x = vec![0.0; n];
    qr.solve(&b, &mut x).unwrap();
    let mut y = vec![0.0; n];
    Lsqr::with_options(LsqrOptions { atol: 1e-14, btol: 1e-14, ..Default::default() }).solve(
        |x: &[f64], out: &mut [f64]| {
            out.fill(0.0);
            for c in 0..n {
                for p in col_ptr[c]..col_ptr[c + 1] {
                    out[row_idx[p]] += val[p] * x[c];
                }
            }
        },
        |u: &[f64], out: &mut [f64]| {
            for c in 0..n {
                out[c] = (col_ptr[c]..col_ptr[c + 1]).map(|p| val[p] * u[row_idx[p]]).sum();
            }
        },
        &b[..],
        &mut y[..],
    );
    let err = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(err < 1e-9, "{err}");
}

/// The grid operator assembled: whole matrix and lower triangle, compressed sparse columns.
fn assembled(n: usize, shift: f64, c: f64, lower: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut e = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let k = j * n + i;
            e.push((k, k, 4.0 + shift + c));
            if i > 0 {
                e.push((k, k - 1, -1.0 - c));
            }
            if i + 1 < n {
                e.push((k, k + 1, -1.0));
            }
            if j > 0 {
                e.push((k, k - n, -1.0));
            }
            if j + 1 < n {
                e.push((k, k + n, -1.0));
            }
        }
    }
    e.retain(|&(r, col, _)| !lower || r >= col);
    e.sort_by_key(|&(r, col, _)| (col, r));
    let mut col_ptr = vec![0; n * n + 1];
    for &(_, col, _) in &e {
        col_ptr[col + 1] += 1;
    }
    for col in 0..n * n {
        col_ptr[col + 1] += col_ptr[col];
    }
    (col_ptr, e.iter().map(|x| x.0).collect(), e.iter().map(|x| x.2).collect())
}

#[test]
fn ilu0_and_ic0_cut_the_iterations() {
    use hydroplane_solvers::iterative::{Ic0, Ilu0};
    let n = 40;
    let b = rhs(n * n);
    let options = Options { tolerance: 1e-10, max_iterations: 3000, ..Default::default() };
    let mut a = grid(n, 0.0, 3.0);
    let (cp, ri, v) = assembled(n, 0.0, 3.0, false);
    let mut x = vec![0.0; n * n];
    let jacobi = Gmres::with_options(options).solve(&mut a, &mut Jacobi::new(&vec![7.0; n * n]), &b[..], &mut x[..]);
    let ilu = Gmres::with_options(options).solve(&mut a, &mut Ilu0::new(&cp, &ri, &v).unwrap(), &b[..], &mut x[..]);
    assert!(jacobi.converged() && ilu.converged(), "{jacobi:?} {ilu:?}");
    assert!(ilu.iterations * 2 < jacobi.iterations, "{} vs {}", ilu.iterations, jacobi.iterations);
    assert!(true_residual(&mut a, &b, &x) < 1e-9);

    let mut p = grid(n, 0.0, 0.0);
    let (cp, ri, v) = assembled(n, 0.0, 0.0, true);
    let cg_options = Options { tolerance: 1e-10, max_iterations: 3000, ..Default::default() };
    let plain = Cg::with_options(cg_options).solve(&mut p, &mut Jacobi::new(&vec![4.0; n * n]), &b[..], &mut x[..]);
    let ic = Ic0::new(&cp, &ri, &v).unwrap();
    assert_eq!(ic.shift(), 0.0);
    let ic = Cg::with_options(cg_options).solve(&mut p, &mut { ic }, &b[..], &mut x[..]);
    assert!(plain.converged() && ic.converged(), "{plain:?} {ic:?}");
    assert!(ic.iterations * 2 < plain.iterations, "{} vs {}", ic.iterations, plain.iterations);
    assert!(true_residual(&mut p, &b, &x) < 1e-9);
}

/// 3D Poisson, lower triangle: `g³` unknowns.
fn poisson3(g: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = g * g * g;
    let mut col_ptr = vec![0];
    let (mut row_idx, mut val) = (Vec::new(), Vec::new());
    for k in 0..n {
        let (i, j, l) = (k % g, (k / g) % g, k / (g * g));
        row_idx.push(k);
        val.push(6.0);
        for (ok, o) in [(i + 1 < g, k + 1), (j + 1 < g, k + g), (l + 1 < g, k + g * g)] {
            if ok {
                row_idx.push(o);
                val.push(-1.0);
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx, val)
}

fn multiply_lower(col_ptr: &[usize], row_idx: &[usize], val: &[f64], x: &[f64], y: &mut [f64]) {
    y.fill(0.0);
    for j in 0..col_ptr.len() - 1 {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            y[i] += val[p] * x[j];
            if i != j {
                y[j] += val[p] * x[i];
            }
        }
    }
}

/// AMG-preconditioned CG takes a small number of iterations that barely grows with the grid,
/// where Jacobi's doubles with each refinement.
#[test]
fn amg_iterations_stay_flat_under_refinement() {
    use hydroplane_solvers::iterative::Amg;
    let mut counts = Vec::new();
    for g in [16, 32] {
        let (cp, ri, v) = poisson3(g);
        let n = g * g * g;
        let b = rhs(n);
        let mut amg = Amg::new(&cp, &ri, &v).unwrap();
        assert!(amg.levels() > 1);
        let mut a = |x: &[f64], y: &mut [f64]| multiply_lower(&cp, &ri, &v, x, y);
        let mut x = vec![0.0; n];
        let report = Cg::with_options(Options { tolerance: 1e-10, ..Default::default() }).solve(&mut a, &mut amg, &b[..], &mut x[..]);
        assert!(report.converged(), "{g}: {report:?}");
        assert!(true_residual(&mut a, &b, &x) < 1e-9);
        let jacobi = Cg::with_options(Options { tolerance: 1e-10, ..Default::default() }).solve(&mut a, &mut Jacobi::new(&vec![6.0; n]), &b[..], &mut x[..]);
        counts.push((report.iterations, jacobi.iterations, amg.levels(), amg.operator_complexity()));
    }
    let ((amg16, jac16, ..), (amg32, jac32, ..)) = (counts[0], counts[1]);
    assert!(amg32 <= amg16 + 6 && amg32 * 4 < jac32, "{counts:?}");
    assert!(jac32 > jac16 * 3 / 2, "{counts:?}");
}
