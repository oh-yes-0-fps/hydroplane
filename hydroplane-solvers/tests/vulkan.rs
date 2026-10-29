//! The GPU solvers in each precision the device runs, on a 3D Laplacian: residuals at the level
//! of the precision, so an emulated `f64` that quietly computes in `f32` fails.
#![cfg(feature = "vulkan")]

use std::sync::Arc;

use hydroplane_solvers::Ef64;
use hydroplane_solvers::cholesky::{Analysis, Ordering};
use hydroplane_solvers::vulkan::{Context, DeviceReal, VulkanCholesky, VulkanPcg};

const SIDE: usize = 16;

/// Lower triangle of the 7-point Laplacian on a `SIDE³` grid with a small diagonal shift.
fn laplacian() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let at = |x: usize, y: usize, z: usize| (z * SIDE + y) * SIDE + x;
    let (mut col_ptr, mut row_idx, mut val) = (vec![0], Vec::new(), Vec::new());
    for z in 0..SIDE {
        for y in 0..SIDE {
            for x in 0..SIDE {
                row_idx.push(at(x, y, z));
                val.push(6.0 + 1e-3);
                for (next, inside) in [(at(x + 1, y, z), x + 1 < SIDE), (at(x, y + 1, z), y + 1 < SIDE), (at(x, y, z + 1), z + 1 < SIDE)] {
                    if inside {
                        row_idx.push(next);
                        val.push(-1.0);
                    }
                }
                col_ptr.push(row_idx.len());
            }
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
    let r: f64 = b.iter().zip(&ax).map(|(b, ax)| (b - ax).powi(2)).sum();
    (r / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

fn context() -> Option<Arc<Context>> {
    match Context::shared() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("no Vulkan device, skipped: {e:?}");
            None
        }
    }
}

fn direct<T: DeviceReal>(ctx: &Arc<Context>, bound: f64) {
    if !ctx.supports(T::PRECISION) {
        return;
    }
    let (col_ptr, row_idx, val) = laplacian();
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let mut chol = VulkanCholesky::<T>::new(&analysis).unwrap();
    assert!(!chol.factorize(&val).unwrap().perturbed);
    let b: Vec<f64> = (0..analysis.dim()).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut x = b.clone();
    chol.solve_in_place(&mut x).unwrap();
    let r = residual(&col_ptr, &row_idx, &val, &b, &x);
    assert!(r < bound, "{:?} direct solve residual {r:e}", T::PRECISION);
}

fn pcg<T: DeviceReal>(ctx: &Arc<Context>, tolerance: f64) {
    if !ctx.supports(T::PRECISION) {
        return;
    }
    let (col_ptr, row_idx, val) = laplacian();
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Amd).unwrap();
    let mut pcg = VulkanPcg::<T>::new(&analysis).unwrap();
    pcg.options.tolerance = tolerance;
    pcg.factorize(&val).unwrap();
    let b: Vec<f64> = (0..analysis.dim()).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut x = b.clone();
    let report = pcg.solve_in_place(&mut x).unwrap();
    let r = residual(&col_ptr, &row_idx, &val, &b, &x);
    assert!(report.converged && r < 2.0 * tolerance, "{:?} pcg: {report:?}, true residual {r:e}", T::PRECISION);
}

#[test]
fn direct_solve_in_each_precision() {
    let Some(ctx) = context() else { return };
    direct::<f32>(&ctx, 1e-4);
    direct::<f64>(&ctx, 1e-13);
    direct::<Ef64>(&ctx, 1e-11);
}

#[test]
fn pcg_in_each_precision() {
    let Some(ctx) = context() else { return };
    pcg::<f32>(&ctx, 1e-5);
    pcg::<f64>(&ctx, 1e-12);
    pcg::<Ef64>(&ctx, 1e-11);
}

/// The solver kept on the GPU, direct and iterating in the device's wide precision: a changed
/// matrix is solved on the old factor to the tolerance.
#[test]
fn updated_values_on_the_gpu() {
    use hydroplane_solvers::Solver;
    use hydroplane_solvers::auto::{Device, Method, Options, Verify};
    let Some(ctx) = context() else { return };
    let (col_ptr, row_idx, val) = laplacian();
    let changed: Vec<f64> = val.iter().enumerate().map(|(p, v)| v * (1.0 + 0.05 * ((p * 7919) % 13) as f64 / 13.0)).collect();
    let b: Vec<f64> = (0..col_ptr.len() - 1).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    for method in [Method::Direct, Method::Pcg] {
        let options = Options { device: Some(Device::Gpu), method: Some(method), precision: Some(ctx.wide_precision()), tolerance: 1e-8, verify: Verify::Always, ..Default::default() };
        let mut solver = Solver::with_options(&col_ptr, &row_idx, options).unwrap();
        solver.factorize(&val).unwrap();
        solver.update_values(&changed).unwrap();
        let mut x = b.clone();
        let report = solver.solve_in_place(&mut x).unwrap();
        assert_eq!(report.meets_tolerance, Some(true), "{method:?}: {report:?}");
    }
}

/// Solvers on several threads share one context: building, factoring, solving and dropping them
/// at once must not race on its queue or command pool.
#[test]
fn solvers_on_many_threads_share_a_context() {
    let Some(ctx) = context() else { return };
    std::thread::scope(|scope| {
        for _ in 0..6 {
            let ctx = ctx.clone();
            scope.spawn(move || {
                for _ in 0..4 {
                    direct::<f32>(&ctx, 1e-4);
                    pcg::<f32>(&ctx, 1e-5);
                }
            });
        }
    });
}

/// `[H Bᵀ; B 0]` on a 3D grid (`H` the Laplacian plus a small shift, `B` averaging pairs), lower
/// triangle, and the constraint signs.
fn saddle_point() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<i8>) {
    let g = 12;
    let nh = g * g * g;
    let mut e = Vec::new();
    for k in 0..nh {
        let (i, j, l) = (k % g, (k / g) % g, k / (g * g));
        e.push((k, k, 6.01));
        if i > 0 {
            e.push((k, k - 1, -1.0));
        }
        if j > 0 {
            e.push((k, k - g, -1.0));
        }
        if l > 0 {
            e.push((k, k - g * g, -1.0));
        }
    }
    let mut m = 0;
    for k in (0..nh).step_by(2) {
        if k % g + 1 < g {
            let r = nh + m;
            e.extend([(r, k, 0.5), (r, k + 1, 0.5), (r, r, 0.0)]);
            m += 1;
        }
    }
    let n = nh + m;
    e.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0; n + 1];
    for &(_, c, _) in &e {
        col_ptr[c + 1] += 1;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    let signs = (0..n).map(|i| if i < nh { 1 } else { -1 }).collect();
    (col_ptr, e.iter().map(|x| x.0).collect(), e.iter().map(|x| x.2).collect(), signs)
}

fn ldlt<T: DeviceReal>(ctx: &Arc<Context>, bound: f64) {
    use hydroplane_solvers::vulkan::VulkanLdlt;
    if !ctx.supports(T::PRECISION) {
        return;
    }
    let (col_ptr, row_idx, val, signs) = saddle_point();
    let perm = hydroplane_solvers::ldlt::kkt_ordering(&col_ptr, &row_idx, &signs).unwrap();
    let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Custom(perm)).unwrap();
    let mut ldlt = VulkanLdlt::<T>::new(&analysis, &signs).unwrap();
    let info = ldlt.factorize(&val).unwrap();
    assert!(!info.perturbed, "{:?}: every pivot has its column's sign", T::PRECISION);
    let b: Vec<f64> = (0..analysis.dim()).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut x = b.clone();
    ldlt.solve_in_place(&mut x).unwrap();
    let r = residual(&col_ptr, &row_idx, &val, &b, &x);
    assert!(r < bound, "{:?} signed solve residual {r:e}", T::PRECISION);
    let mut x = b.clone();
    let report = ldlt.solve_refined(&mut x).unwrap();
    let r = residual(&col_ptr, &row_idx, &val, &b, &x);
    assert!(report.converged() && r < 1e-11, "{:?} refined: {report:?}, residual {r:e}", T::PRECISION);
}

#[test]
fn signed_factorization_solves_a_saddle_point_in_each_precision() {
    let Some(ctx) = context() else { return };
    ldlt::<f32>(&ctx, 1e-3);
    ldlt::<f64>(&ctx, 1e-12);
    ldlt::<Ef64>(&ctx, 1e-9);
}

fn jacobi<T: DeviceReal>(ctx: &Arc<Context>, tolerance: f64) {
    if !ctx.supports(T::PRECISION) {
        return;
    }
    let (col_ptr, row_idx, val) = laplacian();
    let mut pcg = VulkanPcg::<T>::jacobi(&col_ptr, &row_idx).unwrap();
    pcg.options.tolerance = tolerance;
    pcg.options.max_iterations = 2000;
    pcg.factorize(&val).unwrap();
    let b: Vec<f64> = (0..col_ptr.len() - 1).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut x = b.clone();
    let report = pcg.solve_in_place(&mut x).unwrap();
    let r = residual(&col_ptr, &row_idx, &val, &b, &x);
    assert!(report.converged && r < 2.0 * tolerance, "{:?} Jacobi CG: {report:?}, true residual {r:e}", T::PRECISION);
}

#[test]
fn jacobi_cg_in_each_precision() {
    let Some(ctx) = context() else { return };
    jacobi::<f32>(&ctx, 1e-5);
    jacobi::<f64>(&ctx, 1e-11);
    jacobi::<Ef64>(&ctx, 1e-11);
}

fn amg<T: DeviceReal>(ctx: &Arc<Context>, tolerance: f64) {
    if !ctx.supports(T::PRECISION) {
        return;
    }
    let (col_ptr, row_idx, val) = laplacian();
    let mut pcg = VulkanPcg::<T>::amg(&col_ptr, &row_idx).unwrap();
    pcg.options.tolerance = tolerance;
    let b: Vec<f64> = (0..col_ptr.len() - 1).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    for scale in [1.0, 3.0] {
        let val: Vec<f64> = val.iter().map(|v| scale * v).collect();
        pcg.factorize(&val).unwrap();
        let mut x = b.clone();
        let report = pcg.solve_in_place(&mut x).unwrap();
        let r = residual(&col_ptr, &row_idx, &val, &b, &x);
        assert!(report.converged && report.iterations < 30 && r < 2.0 * tolerance, "{:?} AMG CG: {report:?}, true residual {r:e}", T::PRECISION);
    }
}

#[test]
fn amg_cg_in_each_precision() {
    let Some(ctx) = context() else { return };
    amg::<f32>(&ctx, 1e-5);
    amg::<f64>(&ctx, 1e-11);
    amg::<Ef64>(&ctx, 1e-11);
}

/// Upwinded convection–diffusion on a 3D grid, the whole matrix in compressed sparse columns.
fn convection() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let g = SIDE;
    let n = g * g * g;
    let mut e = Vec::new();
    for k in 0..n {
        let (i, j, l) = (k % g, (k / g) % g, k / (g * g));
        e.push((k, k, 9.0));
        for (ok, o, w) in [(i > 0, k.wrapping_sub(1), -4.0), (j > 0, k.wrapping_sub(g), -1.0), (l > 0, k.wrapping_sub(g * g), -1.0), (i + 1 < g, k + 1, -1.0), (j + 1 < g, k + g, -1.0), (l + 1 < g, k + g * g, -1.0)] {
            if ok {
                e.push((k, o, w));
            }
        }
    }
    e.sort_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0; n + 1];
    for &(_, c, _) in &e {
        col_ptr[c + 1] += 1;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    (col_ptr, e.iter().map(|x| x.0).collect(), e.iter().map(|x| x.2).collect())
}

fn bicgstab<T: DeviceReal>(ctx: &Arc<Context>, tolerance: f64) {
    use hydroplane_solvers::vulkan::VulkanBiCgStab;
    if !ctx.supports(T::PRECISION) {
        return;
    }
    let (col_ptr, row_idx, val) = convection();
    let n = col_ptr.len() - 1;
    let mut solver = VulkanBiCgStab::<T>::new(&col_ptr, &row_idx).unwrap();
    solver.options.tolerance = tolerance;
    solver.options.max_iterations = 1000;
    solver.factorize(&val).unwrap();
    let b: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut x = b.clone();
    let report = solver.solve_in_place(&mut x).unwrap();
    let mut ax = vec![0.0; n];
    for j in 0..n {
        for p in col_ptr[j]..col_ptr[j + 1] {
            ax[row_idx[p]] += val[p] * x[j];
        }
    }
    let r = (b.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum::<f64>() / b.iter().map(|b| b * b).sum::<f64>()).sqrt();
    assert!(report.converged && r < 10.0 * tolerance, "{:?} BiCGStab: {report:?}, true residual {r:e}", T::PRECISION);
}

#[test]
fn bicgstab_in_each_precision() {
    let Some(ctx) = context() else { return };
    bicgstab::<f32>(&ctx, 1e-5);
    bicgstab::<f64>(&ctx, 1e-11);
    bicgstab::<Ef64>(&ctx, 1e-11);
}
