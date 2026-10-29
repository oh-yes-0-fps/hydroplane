use hydroplane_solvers::cholesky::{Analysis, Ordering};
use hydroplane_solvers::iterative::Refinement;
use hydroplane_solvers::ldlt::{CpuLdlt, Inertia, Pivoting};

/// The lower triangle in compressed sparse columns from `(row, col, value)` with `row ≥ col`.
fn lower(n: usize, mut entries: Vec<(usize, usize, f64)>) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    entries.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0; n + 1];
    let mut row_idx = Vec::new();
    let mut values = Vec::new();
    for &(r, c, v) in &entries {
        assert!(r >= c);
        col_ptr[c + 1] += 1;
        row_idx.push(r);
        values.push(v);
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    (col_ptr, row_idx, values)
}

fn residual(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64], b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; n];
    for c in 0..n {
        for p in col_ptr[c]..col_ptr[c + 1] {
            let r = row_idx[p];
            ax[r] += values[p] * x[c];
            if r != c {
                ax[c] += values[p] * x[r];
            }
        }
    }
    let rr: f64 = b.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum();
    (rr / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect()
}

/// `[H Bᵀ; B −c I]`: `H` a `g × g` grid Laplacian plus a small shift, `B` one row per 2 × 2 cell
/// averaging its first two unknowns.
fn kkt(g: usize, c: f64) -> (usize, usize, Vec<(usize, usize, f64)>) {
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
            e.push((row, row, -c));
            m += 1;
        }
    }
    (nh, m, e)
}

#[test]
fn regularized_ldlt_solves_a_kkt_system_with_a_zero_block() {
    let (nh, m, e) = kkt(20, 0.0);
    let n = nh + m;
    let (col_ptr, row_idx, values) = lower(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let signs: Vec<i8> = (0..n).map(|i| if i < nh { 1 } else { -1 }).collect();
    let mut ldlt = CpuLdlt::<f64>::with_pivoting(&analysis, Pivoting::Regularized { signs, delta: 1e-7, epsilon: 1e-12 });
    ldlt.factorize(&values).unwrap();
    assert_eq!(ldlt.inertia(), Inertia { positive: nh, negative: m, zero: 0 });
    let b = rhs(n);
    let mut x = b.clone();
    ldlt.refinement = Refinement { tolerance: 1e-13, max_steps: 10 };
    let report = ldlt.solve_refined(&mut x);
    assert!(report.converged(), "{report:?}");
    assert!(residual(n, &col_ptr, &row_idx, &values, &b, &x) < 1e-12);
}

/// The 1D Laplacian shifted by `σ` has exactly as many negative eigenvalues as `2 − 2 cos(kπ/(n+1))`
/// below `σ`; the factor's inertia must say so, under every pivoting.
#[test]
fn inertia_counts_the_eigenvalues_below_a_shift() {
    let n = 200;
    let sigma = 1.37;
    let below = (1..=n).filter(|&k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n + 1) as f64).cos() < sigma).count();
    let mut e = Vec::new();
    for i in 0..n {
        e.push((i, i, 2.0 - sigma));
        if i > 0 {
            e.push((i, i - 1, -1.0));
        }
    }
    let (col_ptr, row_idx, values) = lower(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let b = rhs(n);
    for pivoting in [Pivoting::Static, Pivoting::BunchKaufman] {
        let mut ldlt = CpuLdlt::<f64>::with_pivoting(&analysis, pivoting.clone());
        ldlt.factorize(&values).unwrap();
        assert_eq!(ldlt.inertia(), Inertia { positive: n - below, negative: below, zero: 0 }, "{pivoting:?}");
        let mut x = b.clone();
        ldlt.solve_in_place(&mut x);
        assert!(residual(n, &col_ptr, &row_idx, &values, &b, &x) < 1e-10, "{pivoting:?}");
    }
}

#[test]
fn bunch_kaufman_solves_a_stabilized_saddle_point() {
    let (nh, m, e) = kkt(16, 1e-3);
    let n = nh + m;
    let (col_ptr, row_idx, values) = lower(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let mut ldlt = CpuLdlt::<f64>::new(&analysis);
    ldlt.factorize(&values).unwrap();
    assert_eq!(ldlt.inertia(), Inertia { positive: nh, negative: m, zero: 0 });
    let b = rhs(n);
    let mut x = b.clone();
    ldlt.solve_in_place(&mut x);
    assert!(residual(n, &col_ptr, &row_idx, &values, &b, &x) < 1e-10);
}

/// An `f32` factor refined in `f64` reaches `f64` accuracy.
#[test]
fn an_f32_factor_refines_to_f64_accuracy() {
    let (nh, m, e) = kkt(20, 1e-2);
    let n = nh + m;
    let (col_ptr, row_idx, values) = lower(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let signs: Vec<i8> = (0..n).map(|i| if i < nh { 1 } else { -1 }).collect();
    let mut ldlt = CpuLdlt::<f32>::with_pivoting(&analysis, Pivoting::Regularized { signs, delta: 1e-6, epsilon: 1e-10 });
    ldlt.factorize(&values).unwrap();
    let b = rhs(n);
    let mut x = b.clone();
    ldlt.refinement = Refinement { tolerance: 1e-12, max_steps: 20 };
    let report = ldlt.solve_refined(&mut x);
    assert!(report.converged(), "{report:?}");
    assert!(residual(n, &col_ptr, &row_idx, &values, &b, &x) < 1e-11);
}

#[test]
fn pivots_of_the_wrong_sign_are_replaced_and_counted() {
    let n = 50;
    let mut e = Vec::new();
    for i in 0..n {
        e.push((i, i, 3.0));
        if i > 0 {
            e.push((i, i - 1, -1.0));
        }
    }
    let (col_ptr, row_idx, values) = lower(n, e);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let signs: Vec<i8> = (0..n).map(|i| if i < 10 { -1 } else { 1 }).collect();
    let mut ldlt = CpuLdlt::<f64>::with_pivoting(&analysis, Pivoting::Regularized { signs, delta: 1e-3, epsilon: 1e-12 });
    assert_eq!(ldlt.factorize(&values).unwrap().regularized, 10);
}

/// Under the KKT ordering no constraint pivot is zero: the static factorization needs no
/// regularization, and Bunch–Kaufman, which cannot handle a constraint ordered first, succeeds.
#[test]
fn the_kkt_ordering_gives_every_constraint_a_real_pivot() {
    let (nh, m, e) = kkt(20, 0.0);
    let n = nh + m;
    let (col_ptr, row_idx, values) = lower(n, e);
    let signs: Vec<i8> = (0..n).map(|i| if i < nh { 1 } else { -1 }).collect();
    let perm = hydroplane_solvers::ldlt::kkt_ordering(&col_ptr, &row_idx, &signs).unwrap();
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Custom(perm)).unwrap();
    let b = rhs(n);
    for pivoting in [Pivoting::Regularized { signs: signs.clone(), delta: 1e-7, epsilon: 1e-12 }, Pivoting::BunchKaufman] {
        let mut ldlt = CpuLdlt::<f64>::with_pivoting(&analysis, pivoting.clone());
        assert_eq!(ldlt.factorize(&values).unwrap().regularized, 0, "{pivoting:?}");
        assert_eq!(ldlt.inertia(), Inertia { positive: nh, negative: m, zero: 0 });
        let mut x = b.clone();
        ldlt.solve_in_place(&mut x);
        assert!(residual(n, &col_ptr, &row_idx, &values, &b, &x) < 1e-12, "{pivoting:?}");
    }
}
