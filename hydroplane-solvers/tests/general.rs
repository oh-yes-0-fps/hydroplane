use hydroplane_solvers::{GeneralSolver, IndefiniteSolver};

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

fn residual(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64], symmetric: bool, b: &[f64], x: &[f64]) -> f64 {
    let mut y = vec![0.0; n];
    for c in 0..n {
        for p in col_ptr[c]..col_ptr[c + 1] {
            let r = row_idx[p];
            y[r] += values[p] * x[c];
            if symmetric && r != c {
                y[c] += values[p] * x[r];
            }
        }
    }
    let rr: f64 = y.iter().zip(b).map(|(y, b)| (y - b) * (y - b)).sum();
    (rr / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect()
}

fn options(portable: bool) -> hydroplane_solvers::general::Options {
    hydroplane_solvers::general::Options { portable, ..Default::default() }
}

#[test]
fn general_solver_solves_convection_and_a_zero_diagonal() {
    let g = 20;
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
    let mut pairs = Vec::new();
    for k in (0..n).step_by(2) {
        pairs.extend([(k, k + 1, 2.0), (k + 1, k, 3.0)]);
        if k + 2 < n {
            pairs.extend([(k + 2, k, 0.5), (k, k + 3, -0.25)]);
        }
    }
    for entries in [e, pairs] {
        let (col_ptr, row_idx, values) = csc(n, entries);
        for portable in [true, false] {
            let mut solver = GeneralSolver::with_options(&col_ptr, &row_idx, options(portable)).unwrap();
            solver.factorize(&values).unwrap();
            let b = rhs(n);
            for _ in 0..2 {
                let mut x = b.clone();
                let report = solver.solve_in_place(&mut x).unwrap();
                assert!(residual(n, &col_ptr, &row_idx, &values, false, &b, &x) < 1e-10, "{report:?}");
            }
        }
    }
}

/// `[H Bᵀ; B 0]`, lower triangle.
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
            let r = nh + m;
            e.extend([(r, j * g + i, 0.5), (r, j * g + i + 1, 0.5), (r, r, 0.0)]);
            m += 1;
        }
    }
    (nh, m, e)
}

#[test]
fn indefinite_solver_solves_a_kkt_system_with_and_without_signs() {
    let (nh, m, e) = kkt(16);
    let n = nh + m;
    let (col_ptr, row_idx, values) = csc(n, e);
    let signs: Vec<i8> = (0..n).map(|i| if i < nh { 1 } else { -1 }).collect();
    for (signs, portable) in [(Some(signs.clone()), true), (None, true), (None, false)] {
        let options = hydroplane_solvers::general::Options { portable, signs: signs.clone(), ..Default::default() };
        let mut solver = IndefiniteSolver::with_options(&col_ptr, &row_idx, options).unwrap();
        solver.factorize(&values).unwrap();
        let b = rhs(n);
        let mut x = b.clone();
        let report = solver.solve_in_place(&mut x).unwrap();
        assert!(residual(n, &col_ptr, &row_idx, &values, true, &b, &x) < 1e-10, "{signs:?} {report:?}");
        if let Some(inertia) = solver.inertia() {
            assert_eq!((inertia.positive, inertia.negative), (nh, m), "{}", report.method);
        }
        solver.factorize(&values.iter().map(|v| 2.0 * v).collect::<Vec<_>>()).unwrap();
        let mut y = b.clone();
        solver.solve_in_place(&mut y).unwrap();
        assert!(x.iter().zip(&y).all(|(x, y)| (x - 2.0 * y).abs() < 1e-8), "refactoring {}", report.method);
    }
}

/// With signs and a Vulkan device, a large saddle point is factored on the GPU, and the solution
/// and inertia match.
#[cfg(feature = "vulkan")]
#[test]
fn indefinite_solver_uses_the_gpu_for_a_large_saddle_point() {
    if hydroplane_solvers::vulkan::Context::shared().is_err() {
        return;
    }
    let (nh, m, e) = kkt(120);
    let n = nh + m;
    let (col_ptr, row_idx, values) = csc(n, e);
    let signs: Vec<i8> = (0..n).map(|i| if i < nh { 1 } else { -1 }).collect();
    let options = hydroplane_solvers::general::Options { signs: Some(signs), portable: true, ..Default::default() };
    let mut solver = IndefiniteSolver::with_options(&col_ptr, &row_idx, options).unwrap();
    solver.factorize(&values).unwrap();
    assert!(solver.method().unwrap().starts_with("Vulkan"), "{:?}", solver.method());
    let b = rhs(n);
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    assert!(residual(n, &col_ptr, &row_idx, &values, true, &b, &x) < 1e-10, "{report:?}");
    assert_eq!(solver.inertia().map(|i| (i.positive, i.negative)), Some((nh, m)));
}
