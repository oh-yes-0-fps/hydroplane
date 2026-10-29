#![cfg(target_os = "macos")]

use hydroplane_solvers::accelerate::{AccelerateLdlt, AccelerateLu, AccelerateQr, LdltPivoting};
use hydroplane_solvers::cholesky::{Analysis, Ordering};
use hydroplane_solvers::dense::DenseCholesky;
use hydroplane_solvers::iterative::{Gmres, Options};
use hydroplane_solvers::ldlt::Inertia;

fn csc(n: usize, mut entries: Vec<(usize, usize, f64)>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    entries.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0; n + 1];
    for &(_, c, _) in &entries {
        col_ptr[c + 1] += 1;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    (col_ptr, entries.iter().map(|e| e.0).collect(), entries.iter().map(|e| e.2).collect())
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect()
}

/// `A x` for a whole matrix, or a symmetric one given by its lower triangle.
fn multiply(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64], x: &[f64], symmetric: bool) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for c in 0..col_ptr.len() - 1 {
        for p in col_ptr[c]..col_ptr[c + 1] {
            let r = row_idx[p];
            y[r] += values[p] * x[c];
            if symmetric && r != c {
                y[c] += values[p] * x[r];
            }
        }
    }
    y
}

fn relative_error(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
    (d / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

/// `[H Bᵀ; B 0]`, lower triangle: `H` a grid Laplacian, `B` averaging pairs.
fn kkt(g: usize) -> (usize, usize, Vec<(usize, usize, f64)>) {
    let nh = g * g;
    let mut e = Vec::new();
    for j in 0..g {
        for i in 0..g {
            let k = j * g + i;
            e.push((k, k, 4.01));
            if i > 0 {
                e.push((k, k - 1, -1.0));
            }
            if j > 0 {
                e.push((k, k - g, -1.0));
            }
        }
    }
    let mut m = 0;
    for j in (0..g).step_by(2) {
        for i in (0..g - 1).step_by(2) {
            let row = nh + m;
            let k = j * g + i;
            e.push((row, k, 0.5));
            e.push((row, k + 1, 0.5));
            e.push((row, row, 0.0));
            m += 1;
        }
    }
    (nh, m, e)
}

#[test]
fn threshold_pivoting_ldlt_solves_a_kkt_system_and_reports_its_inertia() {
    let (nh, m, e) = kkt(24);
    let n = nh + m;
    let (col_ptr, row_idx, values) = csc(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let mut ldlt = AccelerateLdlt::<f64>::new(&analysis).unwrap();
    ldlt.factorize(&values).unwrap();
    assert_eq!(ldlt.inertia(), Some(Inertia { positive: nh, negative: m, zero: 0 }));
    let b = rhs(n);
    let mut x = b.clone();
    ldlt.solve_in_place(&mut x).unwrap();
    assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, true), &b) < 1e-12);
    ldlt.factorize(&values.iter().map(|v| 2.0 * v).collect::<Vec<_>>()).unwrap();
    let mut y = b.clone();
    ldlt.solve_in_place(&mut y).unwrap();
    assert!(x.iter().zip(&y).all(|(x, y)| (x - 2.0 * y).abs() < 1e-10), "refactoring reuses the analysis");
}

#[test]
/// Static pivoting perturbs the zero block's pivots; GMRES on the true matrix, preconditioned by
/// the factor, recovers the solution in a few steps.
fn supernode_bunch_kaufman_ldlt_refines_to_a_kkt_solution() {
    let (nh, m, e) = kkt(16);
    let n = nh + m;
    let (col_ptr, row_idx, values) = csc(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let mut ldlt = AccelerateLdlt::<f64>::with_pivoting(&analysis, LdltPivoting::SupernodeBunchKaufman).unwrap();
    ldlt.factorize(&values).unwrap();
    let b = rhs(n);
    let mut direct = b.clone();
    ldlt.solve_in_place(&mut direct).unwrap();
    let mut a = |x: &[f64], y: &mut [f64]| y.copy_from_slice(&multiply(n, &col_ptr, &row_idx, &values, x, true));
    let mut m = |r: &[f64], z: &mut [f64]| {
        z.copy_from_slice(r);
        ldlt.solve_in_place(z).unwrap();
    };
    let mut x = vec![0.0; n];
    let report = Gmres::with_options(Options { tolerance: 1e-12, ..Default::default() }).solve(&mut a, &mut m, &b[..], &mut x[..]);
    let unrefined = relative_error(&multiply(n, &col_ptr, &row_idx, &values, &direct, true), &b);
    assert!(report.converged() && report.iterations <= 10, "{report:?}, unrefined {unrefined:.1e}");
    assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, true), &b) < 1e-11);
}

#[test]
fn lu_solves_convection_diffusion() {
    let g = 30;
    let n = g * g;
    let mut e = Vec::new();
    for j in 0..g {
        for i in 0..g {
            let k = j * g + i;
            e.push((k, k, 9.0));
            if i > 0 {
                e.push((k, k - 1, -6.0));
            }
            if i + 1 < g {
                e.push((k, k + 1, -1.0));
            }
            if j > 0 {
                e.push((k, k - g, -1.0));
            }
            if j + 1 < g {
                e.push((k, k + g, -1.0));
            }
        }
    }
    let (col_ptr, row_idx, values) = csc(n, e);
    let mut lu = AccelerateLu::<f64>::new(&col_ptr, &row_idx).unwrap();
    lu.factorize(&values).unwrap();
    let b = rhs(n);
    let mut x = b.clone();
    lu.solve_in_place(&mut x).unwrap();
    assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12);
}

#[test]
fn qr_gives_the_least_squares_solution() {
    let (m, n) = (300, 40);
    let mut e = Vec::new();
    for r in 0..m {
        for c in 0..n {
            let h = (r * 2654435761usize + c * 40503) % 1000;
            if h < 120 || r % n == c {
                e.push((r, c, h as f64 / 500.0 - 1.0 + if r % n == c { 3.0 } else { 0.0 }));
            }
        }
    }
    let (col_ptr, row_idx, values) = csc(n, e);
    let b = rhs(m);
    let mut qr = AccelerateQr::<f64>::new(m, &col_ptr, &row_idx).unwrap();
    qr.factorize(&values).unwrap();
    let mut x = vec![0.0; n];
    qr.solve(&b, &mut x).unwrap();
    let mut ata = vec![0.0; n * n];
    let mut atb = vec![0.0; n];
    let mut dense = vec![0.0; m * n];
    for c in 0..n {
        for p in col_ptr[c]..col_ptr[c + 1] {
            dense[row_idx[p] * n + c] = values[p];
        }
    }
    for i in 0..n {
        atb[i] = (0..m).map(|r| dense[r * n + i] * b[r]).sum();
        for j in 0..n {
            ata[i * n + j] = (0..m).map(|r| dense[r * n + i] * dense[r * n + j]).sum();
        }
    }
    DenseCholesky::new(n, &ata).solve_in_place(&mut atb);
    assert!(relative_error(&x, &atb) < 1e-10, "{}", relative_error(&x, &atb));
}
