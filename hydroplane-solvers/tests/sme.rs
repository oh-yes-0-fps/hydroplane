//! The SME factorization (macOS with SME2) against `faer`'s, and its factor solved on the GPU.
#![cfg(target_os = "macos")]

use hydroplane_solvers::cholesky::{Analysis, CpuCholesky, Ordering};
use hydroplane_solvers::sme::{self, SmeCholesky};

/// Lower triangle of a 3-dof-per-node 27-point stencil on a `side³` grid, diagonally dominant
/// but with strong coupling, numbered node by node.
fn grid(side: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let node = |x: usize, y: usize, z: usize| (z * side + y) * side + x;
    let (mut col_ptr, mut row_idx, mut val) = (vec![0], Vec::new(), Vec::new());
    for n in 0..side * side * side {
        let (x, y, z) = (n % side, n / side % side, n / (side * side));
        let mut near = Vec::new();
        for dz in 0..3usize {
            for dy in 0..3usize {
                for dx in 0..3usize {
                    let (a, b, c) = (x + dx, y + dy, z + dz);
                    if (1..=side).contains(&a) && (1..=side).contains(&b) && (1..=side).contains(&c) && node(a - 1, b - 1, c - 1) >= n {
                        near.push(node(a - 1, b - 1, c - 1));
                    }
                }
            }
        }
        near.sort_unstable();
        for comp in 0..3 {
            for &m in &near {
                for r in 0..3 {
                    let (i, j) = (3 * m + r, 3 * n + comp);
                    if i >= j {
                        row_idx.push(i);
                        val.push(if i == j { 28.7 } else if r == comp { -1.0 } else { -0.05 });
                    }
                }
            }
            col_ptr.push(row_idx.len());
        }
    }
    (col_ptr, row_idx, val)
}

fn residual(col_ptr: &[usize], row_idx: &[usize], val: &[f64], b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    for j in 0..b.len() {
        for p in col_ptr[j]..col_ptr[j + 1] {
            let i = row_idx[p];
            ax[i] += val[p] * x[j];
            if i != j {
                ax[j] += val[p] * x[i];
            }
        }
    }
    let r: f64 = b.iter().zip(&ax).map(|(b, a)| (b - a).powi(2)).sum();
    (r / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

#[test]
fn factors_like_faer_in_its_layout() {
    if !sme::available() {
        return;
    }
    let (col_ptr, row_idx, val) = grid(14);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::NestedDissection { block: 3 }).unwrap();
    let mut reference = CpuCholesky::<f64>::new(&analysis);
    reference.factorize(&val).unwrap();
    let mut chol = SmeCholesky::<f64>::new(&analysis).unwrap();
    chol.factorize(&val).unwrap();
    let biggest = reference.factor().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let sn = analysis.supernodal();
    for s in 0..sn.n_supernodes() {
        let k = sn.supernode_end()[s] - sn.supernode_begin()[s];
        let f = k + sn.supernode(s).pattern().len();
        let off = sn.col_ptr_for_val()[s];
        for j in 0..k {
            for r in j..f {
                let p = off + j * f + r;
                assert!((chol.factor()[p] - reference.factor()[p]).abs() <= 1e-12 * biggest, "supernode {s}, ({r}, {j})");
            }
        }
    }
    let b: Vec<f64> = (0..analysis.dim()).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut x = b.clone();
    chol.solve_in_place(&mut x);
    assert!(residual(&col_ptr, &row_idx, &val, &b, &x) < 1e-13);
}

#[test]
fn a_matrix_that_is_not_positive_definite_is_refused() {
    if !sme::available() {
        return;
    }
    let (col_ptr, row_idx, mut val) = grid(6);
    for j in 0..col_ptr.len() - 1 {
        val[col_ptr[j]] = -1.0;
    }
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::NestedDissection { block: 3 }).unwrap();
    let mut chol = SmeCholesky::<f64>::new(&analysis).unwrap();
    assert!(chol.factorize(&val).is_err());
}

/// The f64 SME factor in the GPU's wide precision, and the f32 one as a GPU iteration's
/// preconditioner.
#[cfg(feature = "vulkan")]
#[test]
fn its_factor_is_solved_on_the_gpu() {
    use hydroplane_solvers::vulkan::{Context, VulkanCholesky, VulkanPcg};
    use hydroplane_solvers::{Ef64, Precision};
    if !sme::available() {
        return;
    }
    let Ok(ctx) = Context::shared() else { return };
    let (col_ptr, row_idx, val) = grid(12);
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::NestedDissection { block: 3 }).unwrap();
    let b: Vec<f64> = (0..analysis.dim()).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut wide = SmeCholesky::<f64>::new(&analysis).unwrap();
    wide.factorize(&val).unwrap();
    let mut x = b.clone();
    match ctx.wide_precision() {
        Precision::F64 => {
            let mut gpu = VulkanCholesky::<f64>::new(&analysis).unwrap();
            gpu.load_factor(wide.factor()).unwrap();
            gpu.solve_in_place(&mut x).unwrap();
        }
        _ => {
            let mut gpu = VulkanCholesky::<Ef64>::new(&analysis).unwrap();
            gpu.load_factor(wide.factor()).unwrap();
            gpu.solve_in_place(&mut x).unwrap();
        }
    }
    assert!(residual(&col_ptr, &row_idx, &val, &b, &x) < 1e-11);
    let mut with_inverse = SmeCholesky::<f64>::new(&analysis).unwrap().with_inverses();
    with_inverse.factorize(&val).unwrap();
    let mut x = b.clone();
    let mut gpu = VulkanCholesky::<Ef64>::new(&analysis).unwrap();
    gpu.load_factor_and_inverse(with_inverse.factor(), with_inverse.inverse().unwrap()).unwrap();
    gpu.solve_in_place(&mut x).unwrap();
    assert!(residual(&col_ptr, &row_idx, &val, &b, &x) < 1e-11, "with the inverses made on the CPU");
    let mut narrow = SmeCholesky::<f32>::new(&analysis).unwrap();
    narrow.factorize(&val).unwrap();
    let mut pcg = VulkanPcg::<Ef64>::new(&analysis).unwrap();
    pcg.options.tolerance = 1e-10;
    pcg.load_preconditioner(&val, narrow.factor()).unwrap();
    let mut x = b.clone();
    let report = pcg.solve_in_place(&mut x).unwrap();
    assert!(report.converged && residual(&col_ptr, &row_idx, &val, &b, &x) < 1e-9, "{report:?}");
}
