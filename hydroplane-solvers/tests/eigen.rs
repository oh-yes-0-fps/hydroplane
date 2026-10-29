use hydroplane_solvers::banded::{Band, BandedLdlt};
use hydroplane_solvers::eigen::{Lobpcg, LobpcgOptions};

fn laplacian(n: usize) -> impl FnMut(&[f64], &mut [f64]) {
    move |x: &[f64], y: &mut [f64]| {
        for i in 0..n {
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i + 1 < n { x[i + 1] } else { 0.0 };
            y[i] = 2.0 * x[i] - left - right;
        }
    }
}

fn exact(n: usize, j: usize) -> f64 {
    2.0 - 2.0 * ((j + 1) as f64 * std::f64::consts::PI / (n + 1) as f64).cos()
}

/// An exact solve of the tridiagonal Laplacian, for a preconditioner.
fn solver(n: usize) -> impl FnMut(&[f64], &mut [f64]) {
    let mut band = Band::zeros(n, 1);
    for i in 0..n {
        *band.at_mut(i, i) = 2.0;
        if i > 0 {
            *band.at_mut(i, i - 1) = -1.0;
        }
    }
    let f = BandedLdlt::positive_definite(band).unwrap();
    move |r: &[f64], z: &mut [f64]| {
        z.copy_from_slice(r);
        f.solve_in_place(z);
    }
}

#[test]
fn the_smallest_eigenvalues_of_a_laplacian() {
    let n = 400;
    let found = Lobpcg::new().smallest(n, 5, laplacian(n), solver(n));
    assert!(found.converged, "{} iterations", found.iterations);
    for (j, &v) in found.values.iter().enumerate() {
        assert!((v - exact(n, j)).abs() < 1e-9 * exact(n, j) + 1e-14, "{j}: {v} vs {}", exact(n, j));
    }
    let mut ax = vec![0.0; n];
    let mut a = laplacian(n);
    for (j, x) in found.vectors.iter().enumerate() {
        a(x, &mut ax);
        let r: f64 = ax.iter().zip(x).map(|(a, x)| (a - found.values[j] * x).powi(2)).sum::<f64>().sqrt();
        assert!(r < 1e-7 * found.values[j], "{j}: residual {r}");
        for (i, y) in found.vectors.iter().enumerate().take(j) {
            let d: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
            assert!(d.abs() < 1e-8, "{i},{j} not orthogonal: {d}");
        }
    }
}

/// `A x = λ B x` with `B = 2 I`: the eigenvalues halve, the vectors are `B`-orthonormal.
#[test]
fn a_mass_matrix_scales_the_eigenvalues() {
    let n = 300;
    let found = Lobpcg::new().smallest_generalized(n, 3, laplacian(n), |x: &[f64], y: &mut [f64]| y.iter_mut().zip(x).for_each(|(y, x)| *y = 2.0 * x), solver(n));
    assert!(found.converged);
    for (j, &v) in found.values.iter().enumerate() {
        assert!((v - exact(n, j) / 2.0).abs() < 1e-9 * exact(n, j));
        let bnorm: f64 = found.vectors[j].iter().map(|x| 2.0 * x * x).sum();
        assert!((bnorm - 1.0).abs() < 1e-8, "{bnorm}");
    }
}

/// Jacobi alone converges too, in more iterations.
#[test]
fn jacobi_preconditioning_converges() {
    let n = 100;
    let options = LobpcgOptions { tolerance: 1e-6, max_iterations: 2000 };
    let found = Lobpcg::with_options(options).smallest(n, 2, laplacian(n), |r: &[f64], z: &mut [f64]| z.iter_mut().zip(r).for_each(|(z, r)| *z = r / 2.0));
    assert!(found.converged, "{} iterations", found.iterations);
    assert!((found.values[0] - exact(n, 0)).abs() < 1e-8);
}
