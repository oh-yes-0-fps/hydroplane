use hydroplane_solvers::iterative::Refinement;
use hydroplane_solvers::lu::CpuLu;

/// Compressed sparse columns from `(row, col, value)`.
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

fn multiply(n: usize, col_ptr: &[usize], row_idx: &[usize], values: &[f64], x: &[f64], transpose: bool) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for c in 0..n {
        for p in col_ptr[c]..col_ptr[c + 1] {
            let r = row_idx[p];
            if transpose {
                y[c] += values[p] * x[r];
            } else {
                y[r] += values[p] * x[c];
            }
        }
    }
    y
}

fn relative_error(a: &[f64], b: &[f64]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
    (d / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

/// Upwinded convection–diffusion on a `g × g` grid.
fn convection(g: usize, c: f64) -> Vec<(usize, usize, f64)> {
    let mut e = Vec::new();
    for j in 0..g {
        for i in 0..g {
            let k = j * g + i;
            e.push((k, k, 4.0 + c));
            if i > 0 {
                e.push((k, k - 1, -1.0 - c));
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
    e
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect()
}

#[test]
fn lu_solves_and_transpose_solves_convection_diffusion() {
    let g = 30;
    let n = g * g;
    let (col_ptr, row_idx, values) = csc(n, convection(g, 5.0));
    let mut lu = CpuLu::<f64>::new(&col_ptr, &row_idx).unwrap();
    lu.factorize(&values).unwrap();
    let b = rhs(n);
    for transpose in [false, true] {
        let mut x = b.clone();
        if transpose {
            lu.solve_transpose_in_place(&mut x);
        } else {
            lu.solve_in_place(&mut x);
        }
        let ax = multiply(n, &col_ptr, &row_idx, &values, &x, transpose);
        assert!(relative_error(&ax, &b) < 1e-12, "transpose {transpose}");
    }
}

/// Zero diagonal everywhere: only row pivoting can factor it.
#[test]
fn lu_pivots_past_a_zero_diagonal() {
    let n = 400;
    let mut e = Vec::new();
    for k in (0..n).step_by(2) {
        e.push((k, k + 1, 2.0));
        e.push((k + 1, k, 3.0));
        if k + 2 < n {
            e.push((k + 2, k, 0.5));
            e.push((k, k + 3, -0.25));
        }
    }
    let (col_ptr, row_idx, values) = csc(n, e);
    let mut lu = CpuLu::<f64>::new(&col_ptr, &row_idx).unwrap();
    lu.factorize(&values).unwrap();
    let b = rhs(n);
    let mut x = b.clone();
    lu.solve_in_place(&mut x);
    assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12);
}

#[test]
fn an_f32_lu_refines_to_f64_accuracy() {
    let g = 24;
    let n = g * g;
    let (col_ptr, row_idx, values) = csc(n, convection(g, 2.0));
    let mut lu = CpuLu::<f32>::new(&col_ptr, &row_idx).unwrap();
    lu.factorize(&values).unwrap();
    let b = rhs(n);
    let mut x = b.clone();
    lu.refinement = Refinement { tolerance: 1e-13, max_steps: 10 };
    let report = lu.solve_refined(&mut x);
    assert!(report.converged(), "{report:?}");
    assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12);
}

#[test]
fn a_singular_matrix_is_reported() {
    let (col_ptr, row_idx, values) = csc(3, vec![(0, 0, 1.0), (1, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
    let mut lu = CpuLu::<f64>::new(&col_ptr, &row_idx).unwrap();
    assert!(lu.factorize(&values).is_err());
}

mod frontal {
    use super::*;
    use hydroplane_solvers::cholesky::Ordering;
    use hydroplane_solvers::lu::{FrontalLu, LuOptions, Matching};

    /// The matching picks the largest product, and its scaling puts every matched entry at 1 and
    /// every other at or below 1.
    #[test]
    fn the_matching_maximizes_the_diagonal_product() {
        let entries = vec![(0, 0, 1.0), (1, 0, 4.0), (0, 1, 3.0), (1, 1, 1.0), (2, 1, 0.5), (1, 2, 2.0), (2, 2, 0.1)];
        let (col_ptr, row_idx, values) = csc(3, entries);
        let m = Matching::new(&col_ptr, &row_idx, &values).unwrap();
        assert_eq!(m.column, vec![1, 0, 2], "rows 0 and 1 swap: 3·4·0.1 = 1.2 beats 1·2·0.5 = 1 and 1·1·0.1");
        for j in 0..3 {
            for p in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[p];
                let scaled = (values[p] * m.row_scale[i] * m.col_scale[j]).abs();
                assert!(scaled <= 1.0 + 1e-12);
                if m.column[i] == j {
                    assert!((scaled - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn frontal_lu_solves_convection_diffusion() {
        let g = 30;
        let n = g * g;
        let (col_ptr, row_idx, values) = csc(n, convection(g, 5.0));
        for ordering in [Ordering::Amd, Ordering::NestedDissection { block: 1 }] {
            let mut lu = FrontalLu::<f64>::with_options(&col_ptr, &row_idx, LuOptions { ordering, matching: Some(false) }).unwrap();
            lu.factorize(&values).unwrap();
            assert_eq!(lu.perturbed(), 0);
            let b = rhs(n);
            let mut x = b.clone();
            lu.solve_in_place(&mut x);
            assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12);
        }
    }

    /// Zero diagonal everywhere: pivoting within supernodes alone breaks down, a matching moves
    /// the large entries onto the diagonal first.
    #[test]
    fn a_matching_makes_a_zero_diagonal_factorable() {
        let n = 400;
        let mut e = Vec::new();
        for k in (0..n).step_by(2) {
            e.push((k, k + 1, 2.0));
            e.push((k + 1, k, 3.0));
            if k + 2 < n {
                e.push((k + 2, k, 0.5));
                e.push((k, k + 3, -0.25));
            }
        }
        let (col_ptr, row_idx, values) = csc(n, e);
        let unmatched = LuOptions { ordering: Ordering::Amd, matching: Some(false) };
        assert!(FrontalLu::<f64>::with_options(&col_ptr, &row_idx, unmatched).unwrap().factorize(&values).is_err(), "static pivots on a zero diagonal");
        let mut lu = FrontalLu::<f64>::with_options(&col_ptr, &row_idx, LuOptions { ordering: Ordering::Amd, ..Default::default() }).unwrap();
        lu.factorize(&values).unwrap();
        assert!(lu.matched(), "a zero diagonal is matched by default");
        let b = rhs(n);
        let mut x = b.clone();
        lu.solve_in_place(&mut x);
        assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12, "{} perturbed", lu.perturbed());
        x.copy_from_slice(&b);
        lu.refinement = Refinement { tolerance: 1e-13, max_steps: 20 };
        let report = lu.solve_refined(&mut x);
        assert!(report.converged(), "{report:?}, {} perturbed", lu.perturbed());
        assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12);
    }

    #[test]
    fn an_f32_frontal_lu_refines_to_f64_accuracy() {
        let g = 24;
        let n = g * g;
        let (col_ptr, row_idx, values) = csc(n, convection(g, 2.0));
        let mut lu = FrontalLu::<f32>::new(&col_ptr, &row_idx).unwrap();
        lu.factorize(&values).unwrap();
        let b = rhs(n);
        let mut x = b.clone();
        lu.refinement = Refinement { tolerance: 1e-13, max_steps: 10 };
        let report = lu.solve_refined(&mut x);
        assert!(report.converged(), "{report:?}");
        assert!(relative_error(&multiply(n, &col_ptr, &row_idx, &values, &x, false), &b) < 1e-12);
    }
}
