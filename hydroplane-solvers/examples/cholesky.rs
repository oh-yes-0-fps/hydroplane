//! Factors a dumped SPD matrix on the CPU and, with `--features vulkan`, on the GPU; checks the
//! GPU factor against the CPU's entry by entry and every factor by the residual of a solve.
//!
//! `cargo run --release --features vulkan --example cholesky -- MATRIX.bin [BLOCK [SHIFT]]`
//!
//! The file holds `n: u64, nnz: u64, col_ptr: [u64; n + 1], row_idx: [u32; nnz], values: [f64; nnz]`,
//! the lower triangle in compressed sparse columns; `BLOCK` (default 3) groups dofs into nodes for
//! the nested dissection. A third argument `SHIFT` scales the diagonal by `1 + SHIFT`, for a
//! better-conditioned variant. `ORDERING` picks the fill-reducing order: a nested-dissection spec
//! (`nd:leaf=120,node_refine=true`, see `order::NestedDissection`'s `FromStr`) or a node
//! permutation file (`perm:PATH`, `u32` per node, `perm[new] = old`). `SCHUR_TILE=32` or `64`
//! overrides the Schur block size `SchurTile::for_precision` picks. `CANDIDATES=4` tries four
//! nested-dissection seeds and keeps the cheapest by `vulkan::ordering_cost`.

use std::time::Instant;

use hydroplane_solvers::cholesky::{Analysis, CpuCholesky, CpuPcg, Ordering, Real};
use hydroplane_solvers::order;

struct Matrix {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    val: Vec<f64>,
}

fn load(path: &str) -> Matrix {
    let bytes = std::fs::read(path).expect("reading the matrix");
    let mut at = 0;
    let mut u64s = |count: usize| -> Vec<u64> {
        let v = bytes[at..at + 8 * count].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        at += 8 * count;
        v
    };
    let head = u64s(2);
    let (n, nnz) = (head[0] as usize, head[1] as usize);
    let col_ptr = u64s(n + 1).into_iter().map(|v| v as usize).collect();
    let row_idx = bytes[at..at + 4 * nnz].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
    at += 4 * nnz;
    let val = bytes[at..at + 8 * nnz].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Matrix { col_ptr, row_idx, val }
}

/// `‖b − A x‖ / ‖b‖` with `A` from its lower triangle.
fn residual(a: &Matrix, b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    for j in 0..b.len() {
        for p in a.col_ptr[j]..a.col_ptr[j + 1] {
            let (i, v) = (a.row_idx[p], a.val[p]);
            ax[i] += v * x[j];
            if i != j {
                ax[j] += v * x[i];
            }
        }
    }
    let r: f64 = b.iter().zip(&ax).map(|(b, ax)| (b - ax) * (b - ax)).sum();
    let bb: f64 = b.iter().map(|b| b * b).sum();
    (r / bb).sqrt()
}

fn solve_residual<T: Real>(analysis: &Analysis, a: &Matrix, factor: &[T], b: &[f64]) -> f64 {
    let mut x: Vec<T> = b.iter().map(|&v| T::from_f64(v)).collect();
    analysis.solve_in_place(factor, &mut x);
    residual(a, b, &x.iter().map(|v| v.to_f64()).collect::<Vec<_>>())
}

/// Largest difference of the factors' lower triangles, relative to the largest entry.
#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
fn factor_difference<T: Real>(analysis: &Analysis, got: &[T], want: &[f64]) -> f64 {
    let sn = analysis.supernodal();
    let (mut diff, mut big) = (0.0f64, 0.0f64);
    for s in 0..sn.n_supernodes() {
        let k = sn.supernode_end()[s] - sn.supernode_begin()[s];
        let f = k + sn.supernode(s).pattern().len();
        let off = sn.col_ptr_for_val()[s];
        for j in 0..k {
            for r in j..f {
                let p = off + j * f + r;
                diff = diff.max((got[p].to_f64() - want[p]).abs());
                big = big.max(want[p].abs());
            }
        }
    }
    diff / big
}

fn best_of(reps: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..reps).map(|_| f()).fold(f64::INFINITY, f64::min)
}

fn cpu<T: Real>(analysis: &Analysis, a: &Matrix, b: &[f64]) -> Vec<T> {
    let mut chol = CpuCholesky::<T>::new(analysis);
    let ms = best_of(3, || {
        let t = Instant::now();
        chol.factorize(&a.val).expect("CPU factorization");
        t.elapsed().as_secs_f64() * 1e3
    });
    let mut x = b.to_vec();
    let solve = best_of(5, || {
        let t = Instant::now();
        chol.solve_in_place(&mut x);
        t.elapsed().as_secs_f64() * 1e3
    });
    let name = if T::F64 { "f64" } else { "f32" };
    println!(
        "cpu {name}: factor {ms:.1} ms, solve {solve:.2} ms, residual {:.2e}",
        solve_residual(analysis, a, chol.factor(), b)
    );
    chol.factor().to_vec()
}

#[cfg(feature = "vulkan")]
fn gpu<T: hydroplane_solvers::vulkan::DeviceReal>(analysis: &Analysis, a: &Matrix, b: &[f64], reference: &[f64]) {
    use hydroplane_solvers::vulkan::VulkanCholesky;
    let name = format!("{:?}", T::PRECISION).to_lowercase();
    let t = Instant::now();
    let mut options = hydroplane_solvers::vulkan::Options::default();
    match std::env::var("SCHUR_TILE").as_deref() {
        Ok("32") => options.schur_tile = Some(hydroplane_solvers::vulkan::SchurTile::T32),
        Ok("64") => options.schur_tile = Some(hydroplane_solvers::vulkan::SchurTile::T64),
        _ => {}
    }
    let mut chol = match VulkanCholesky::<T>::with_options(analysis, options) {
        Ok(c) => c,
        Err(e) => {
            println!("vulkan {name}: {e:?}");
            return;
        }
    };
    let setup = t.elapsed().as_secs_f64() * 1e3;
    let mut status = Ok(Default::default());
    let wall = best_of(5, || {
        let t = Instant::now();
        status = chol.factorize(&a.val);
        t.elapsed().as_secs_f64() * 1e3
    });
    let gpu = best_of(5, || {
        status = chol.factorize(&a.val);
        chol.last_factor_ms()
    });
    let factor: Vec<f64> = chol.factor().expect("downloading the factor").iter().map(|v| v.to_f64()).collect();
    println!(
        "vulkan {name}: setup {setup:.0} ms, factor {gpu:.1} ms on the device ({wall:.1} ms wall), {status:?}, factor vs cpu f64 {:.2e}, cpu solve residual {:.2e}",
        factor_difference(analysis, &factor, reference),
        solve_residual(analysis, a, &factor, b)
    );
    let mut x = b.to_vec();
    let solve_wall = best_of(10, || {
        x.copy_from_slice(b);
        let t = Instant::now();
        chol.solve_in_place(&mut x).expect("GPU solve");
        t.elapsed().as_secs_f64() * 1e3
    });
    let solve_gpu = best_of(10, || {
        x.copy_from_slice(b);
        chol.solve_in_place(&mut x).expect("GPU solve");
        chol.last_solve_ms()
    });
    println!("    gpu solve {solve_gpu:.2} ms on the device ({solve_wall:.2} ms wall), residual {:.2e}", residual(a, b, &x));
    if std::env::var_os("HYDROPLANE_SOLVERS_PROFILE").is_some() {
        chol.factorize(&a.val).expect("GPU factorization");
        let profile = chol.factor_profile().expect("profile");
        let schur_ms: f64 = profile.iter().filter(|t| t.kernel == hydroplane_solvers::vulkan::Kernel::Schur).map(|t| t.ms).sum();
        let stats = chol.stats();
        println!(
            "    schur {:.1} GFLOP in {schur_ms:.2} ms ({:.0} GFLOP/s); panels {:.1} GFLOP in {:.2} ms",
            stats.schur_flops * 1e-9,
            stats.schur_flops / schur_ms * 1e-6,
            (stats.flops - stats.schur_flops) * 1e-9,
            profile.iter().map(|t| t.ms).sum::<f64>() - schur_ms
        );
        print_profile("factor", &profile);
        if let Ok(level) = std::env::var("HYDROPLANE_SOLVERS_TIMELINE") {
            let level: usize = level.parse().expect("a level");
            let line: Vec<String> = chol
                .factor_timeline()
                .expect("timeline")
                .iter()
                .filter(|t| t.1 == level)
                .map(|(k, _, ms)| format!("{}{:.0}", if *k == hydroplane_solvers::vulkan::Kernel::Trsm { "t" } else { "" }, ms * 1e3))
                .collect();
            println!("    level {level} stages (µs, t = TRSM): {}", line.join(" "));
        }
        chol.solve_in_place(&mut x).expect("GPU solve");
        print_profile("solve", &chol.solve_profile().expect("profile"));
    }
    let stats = chol.stats();
    println!("    {stats:?}");
    println!("    {:.1} GFLOP, {:.0} GFLOP/s", stats.flops * 1e-9, stats.flops / gpu * 1e-6);
}

/// `f64` conjugate gradient preconditioned by the `f32` factor, to a few tolerances.
#[cfg(feature = "vulkan")]
fn pcg<T: hydroplane_solvers::vulkan::DeviceReal>(analysis: &Analysis, a: &Matrix, b: &[f64]) {
    use hydroplane_solvers::vulkan::VulkanPcg;
    let name = format!("{:?}", T::PRECISION).to_lowercase();
    let t = Instant::now();
    let mut pcg = match VulkanPcg::<T>::new(analysis) {
        Ok(p) => p,
        Err(e) => {
            println!("pcg {name}: {e:?}");
            return;
        }
    };
    let setup = t.elapsed().as_secs_f64() * 1e3;
    let factor = best_of(5, || {
        let t = Instant::now();
        pcg.factorize(&a.val).expect("factorization");
        t.elapsed().as_secs_f64() * 1e3
    });
    println!(
        "pcg ({name} CG, f32 Cholesky preconditioner): setup {setup:.0} ms, factor {:.1} ms on the device ({factor:.1} ms wall)",
        pcg.cholesky().map_or(0.0, |c| c.last_factor_ms())
    );
    for tol in [1e-6, 1e-8, 1e-10] {
        let mut x = b.to_vec();
        let mut report = None;
        let wall = best_of(5, || {
            x.copy_from_slice(b);
            let t = Instant::now();
            pcg.options.tolerance = tol;
            report = Some(pcg.solve_in_place(&mut x).expect("solve"));
            t.elapsed().as_secs_f64() * 1e3
        });
        let r = report.unwrap();
        println!(
            "    tol {tol:.0e}: {} iterations, {:.2} ms on the device ({wall:.2} ms wall), tracked residual {:.2e}, true residual {:.2e}",
            r.iterations,
            pcg.last_solve_ms(),
            r.relative_residual,
            residual(a, b, &x)
        );
    }
}

/// `f64` conjugate gradient on the CPU preconditioned by the GPU's `f32` factor: the arrangement
/// for GPUs without `f64` (Apple's), the iteration's matrix-vector product and dot products on the
/// CPU and each preconditioner solve on the GPU.
#[cfg(feature = "vulkan")]
fn split_pcg(analysis: &Analysis, a: &Matrix, b: &[f64]) {
    use hydroplane_solvers::vulkan::VulkanCholesky;
    let Ok(mut chol) = VulkanCholesky::<f32>::new(analysis) else { return };
    let mut pcg = CpuPcg::new(analysis);
    let factor = best_of(3, || {
        let t = Instant::now();
        chol.factorize(&a.val).expect("GPU factorization");
        pcg.set_values(&a.val);
        t.elapsed().as_secs_f64() * 1e3
    });
    println!("split pcg (f64 CG on the CPU, f32 factor on the GPU): factor {factor:.1} ms wall");
    for tol in [1e-6, 1e-8, 1e-10] {
        let mut x = b.to_vec();
        let mut report = None;
        let mut in_gpu = f64::INFINITY;
        let ms = best_of(3, || {
            x.copy_from_slice(b);
            let mut gpu = 0.0;
            let t = Instant::now();
            pcg.options.tolerance = tol;
            report = Some(pcg.solve_with(&mut x, |r, z| {
                let t = Instant::now();
                z.copy_from_slice(r);
                chol.solve_in_place(z).expect("GPU solve");
                gpu += t.elapsed().as_secs_f64() * 1e3;
            }));
            in_gpu = in_gpu.min(gpu);
            t.elapsed().as_secs_f64() * 1e3
        });
        let r = report.unwrap();
        println!(
            "    tol {tol:.0e}: {} iterations, {ms:.1} ms wall ({in_gpu:.1} in the GPU preconditioner), tracked residual {:.2e}, true residual {:.2e}",
            r.iterations,
            r.relative_residual,
            residual(a, b, &x)
        );
    }
}

#[cfg(feature = "vulkan")]
fn print_profile(what: &str, times: &[hydroplane_solvers::vulkan::KernelTime]) {
    let total: f64 = times.iter().map(|t| t.ms).sum();
    println!("    {what} profile, {total:.2} ms:");
    let mut by_kernel = std::collections::BTreeMap::<String, (usize, f64)>::new();
    for t in times {
        let e = by_kernel.entry(format!("{:?}", t.kernel)).or_default();
        e.0 += t.dispatches;
        e.1 += t.ms;
    }
    for (k, (d, ms)) in &by_kernel {
        println!("      {k:<16} {d:5} dispatches {ms:8.2} ms");
    }
    let levels = times.iter().map(|t| t.level).max().map_or(0, |l| l + 1);
    for level in 0..levels {
        let row: Vec<String> = times.iter().filter(|t| t.level == level).map(|t| format!("{:?} {}×{:.2}", t.kernel, t.dispatches, t.ms)).collect();
        println!("      level {level:2}: {}", row.join(", "));
    }
}

/// Supernodes per elimination-tree level: count, largest columns `k` and rows `f`, and flops.
fn print_shapes(analysis: &Analysis) {
    let sn = analysis.supernodal();
    let ns = sn.n_supernodes();
    let (begin, end) = (sn.supernode_begin(), sn.supernode_end());
    let mut col_super = vec![0usize; analysis.dim()];
    for s in 0..ns {
        col_super[begin[s]..end[s]].fill(s);
    }
    let mut level = vec![0usize; ns];
    for s in 0..ns {
        if let Some(&first) = sn.supernode(s).pattern().first() {
            let p = col_super[first];
            level[p] = level[p].max(level[s] + 1);
        }
    }
    let levels = level.iter().max().map_or(0, |l| l + 1);
    println!("supernodes by level (count, k median/max, f median/max, GFLOP panel/schur):");
    for lv in 0..levels {
        let mut ks = Vec::new();
        let mut fs = Vec::new();
        let (mut panel, mut schur) = (0.0, 0.0);
        for s in (0..ns).filter(|&s| level[s] == lv) {
            let (k, m) = ((end[s] - begin[s]) as f64, sn.supernode(s).pattern().len() as f64);
            ks.push(k as usize);
            fs.push((k + m) as usize);
            panel += k * k * k / 3.0 + k * k * m;
            schur += k * m * m;
        }
        ks.sort_unstable();
        fs.sort_unstable();
        println!(
            "  {lv:2}: {:5}  k {:4}/{:4}  f {:4}/{:4}  {:.2}/{:.2}",
            ks.len(),
            ks[ks.len() / 2],
            ks[ks.len() - 1],
            fs[fs.len() / 2],
            fs[fs.len() - 1],
            panel * 1e-9,
            schur * 1e-9
        );
    }
}

/// `f64` conjugate gradient on the CPU preconditioned by the `f32` factor, to a few tolerances.
/// This crate's supernodal factorization on the SME units (Accelerate's dense kernels).
#[cfg(target_os = "macos")]
fn sme<T: hydroplane_solvers::sme::SmeReal>(analysis: &Analysis, a: &Matrix, b: &[f64], reference: &[f64]) {
    use hydroplane_solvers::sme::SmeCholesky;
    let t = Instant::now();
    let mut chol = match SmeCholesky::<T>::new(analysis) {
        Ok(c) => c,
        Err(e) => {
            println!("sme: {e:?}");
            return;
        }
    };
    let setup = t.elapsed().as_secs_f64() * 1e3;
    chol.factorize(&a.val).expect("factorization");
    let factor = best_of(5, || {
        let t = Instant::now();
        chol.factorize(&a.val).expect("factorization");
        t.elapsed().as_secs_f64() * 1e3
    });
    let mut x = b.to_vec();
    let solve = best_of(3, || {
        x.copy_from_slice(b);
        let t = Instant::now();
        chol.solve_in_place(&mut x);
        t.elapsed().as_secs_f64() * 1e3
    });
    println!(
        "sme {}: setup {setup:.0} ms, factor {factor:.1} ms ({:.0} GFLOP/s), solve {solve:.2} ms, residual {:.2e}, factor vs cpu f64 {:.2e}",
        std::any::type_name::<T>(),
        analysis.flops() * 1e-6 / factor,
        residual(a, b, &x),
        factor_difference(analysis, chol.factor(), reference)
    );
    #[cfg(feature = "vulkan")]
    {
        use hydroplane_solvers::vulkan::{Context, VulkanCholesky};
        let Ok(ctx) = Context::shared() else { return };
        if T::PRECISION == hydroplane_solvers::Precision::F32 {
            gpu_solve::<f32, T>(analysis, &chol, a, b);
        } else if ctx.supports_f64() {
            gpu_solve::<f64, T>(analysis, &chol, a, b);
        } else {
            gpu_solve::<hydroplane_solvers::Ef64, T>(analysis, &chol, a, b);
        }

        /// The SME factor loaded into the GPU (`P` there) and solved with.
        fn gpu_solve<P: hydroplane_solvers::vulkan::DeviceReal, T: hydroplane_solvers::sme::SmeReal>(
            analysis: &Analysis,
            chol: &hydroplane_solvers::sme::SmeCholesky<T>,
            a: &Matrix,
            b: &[f64],
        ) {
            let mut gpu = VulkanCholesky::<P>::new(analysis).expect("a GPU solver");
            gpu.load_factor(chol.factor()).expect("load");
            let mut device = 0.0;
            let load = best_of(5, || {
                let t = Instant::now();
                device = gpu.load_factor(chol.factor()).expect("load");
                t.elapsed().as_secs_f64() * 1e3
            });
            let mut x = b.to_vec();
            let solve = best_of(5, || {
                x.copy_from_slice(b);
                let t = Instant::now();
                gpu.solve_in_place(&mut x).expect("solve");
                t.elapsed().as_secs_f64() * 1e3
            });
            println!(
                "    loaded into the GPU as {:?}: load {load:.1} ms ({device:.1} ms of it inverting on the device), solve {solve:.2} ms, residual {:.2e}",
                P::PRECISION,
                residual(a, b, &x)
            );
            let mut with_inverses = hydroplane_solvers::sme::SmeCholesky::<T>::new(analysis).expect("SME").with_inverses();
            with_inverses.factorize(&a.val).expect("factorization");
            let factor = best_of(5, || {
                let t = Instant::now();
                with_inverses.factorize(&a.val).expect("factorization");
                t.elapsed().as_secs_f64() * 1e3
            });
            let inverse = with_inverses.inverse().expect("inverses");
            let load = best_of(5, || {
                let t = Instant::now();
                gpu.load_factor_and_inverse(with_inverses.factor(), inverse).expect("load");
                t.elapsed().as_secs_f64() * 1e3
            });
            x.copy_from_slice(b);
            gpu.solve_in_place(&mut x).expect("solve");
            println!("    with the inverses made on the CPU: factor {factor:.1} ms, load {load:.1} ms, residual {:.2e}", residual(a, b, &x));
            if P::PRECISION == hydroplane_solvers::Precision::F32 {
                let mut pcg = hydroplane_solvers::vulkan::VulkanPcg::<hydroplane_solvers::Ef64>::new(analysis).expect("a GPU PCG");
                let factor32: Vec<f32> = chol.factor().iter().map(|v| v.to_f64() as f32).collect();
                pcg.load_preconditioner(&a.val, &factor32).expect("load");
                for tol in [1e-6, 1e-8] {
                    x.copy_from_slice(b);
                    pcg.options.tolerance = tol;
                    let report = pcg.solve_in_place(&mut x).expect("solve");
                    println!("    as the preconditioner of Ef64 PCG to {tol:.0e}: {} iterations, true residual {:.2e}", report.iterations, residual(a, b, &x));
                }
                pcg.factorize(&a.val).expect("factorization");
                for tol in [1e-6, 1e-8] {
                    x.copy_from_slice(b);
                    pcg.options.tolerance = tol;
                    let report = pcg.solve_in_place(&mut x).expect("solve");
                    println!("    against the GPU's own f32 factor, to {tol:.0e}: {} iterations, true residual {:.2e}", report.iterations, residual(a, b, &x));
                }
            }
        }
    }
}

/// Accelerate's sparse Cholesky (its own METIS ordering).
#[cfg(target_os = "macos")]
fn accelerate<T: hydroplane_solvers::accelerate::AccelerateReal>(analysis: &Analysis, a: &Matrix, b: &[f64]) {
    use hydroplane_solvers::accelerate::AccelerateCholesky;
    let t = Instant::now();
    let mut chol = AccelerateCholesky::<T>::new(analysis).expect("symbolic factorization");
    let symbolic = t.elapsed().as_secs_f64() * 1e3;
    chol.factorize(&a.val).expect("factorization");
    let factor = best_of(5, || {
        let t = Instant::now();
        chol.factorize(&a.val).expect("factorization");
        t.elapsed().as_secs_f64() * 1e3
    });
    let mut x = b.to_vec();
    let solve = best_of(5, || {
        x.copy_from_slice(b);
        let t = Instant::now();
        chol.solve_in_place(&mut x).expect("solve");
        t.elapsed().as_secs_f64() * 1e3
    });
    println!("accelerate {:?}: symbolic {symbolic:.0} ms, factor {factor:.1} ms, solve {solve:.2} ms, residual {:.2e}", T::PRECISION, residual(a, b, &x));
}

fn cpu_pcg(analysis: &Analysis, a: &Matrix, b: &[f64]) {
    let mut pcg = CpuPcg::new(analysis);
    let factor = best_of(3, || {
        let t = Instant::now();
        pcg.factorize(&a.val).expect("CPU factorization");
        t.elapsed().as_secs_f64() * 1e3
    });
    println!("cpu pcg (f64 CG, f32 Cholesky preconditioner): factor {factor:.1} ms");
    for tol in [1e-6, 1e-8, 1e-10] {
        let mut x = b.to_vec();
        let mut report = None;
        let ms = best_of(3, || {
            x.copy_from_slice(b);
            let t = Instant::now();
            pcg.options.tolerance = tol;
            report = Some(pcg.solve_in_place(&mut x));
            t.elapsed().as_secs_f64() * 1e3
        });
        let r = report.unwrap();
        println!(
            "    tol {tol:.0e}: {} iterations, {ms:.1} ms, tracked residual {:.2e}, true residual {:.2e}",
            r.iterations,
            r.relative_residual,
            residual(a, b, &x)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut a = load(&args[1]);
    let block = args.get(2).map_or(3, |b| b.parse().unwrap());
    if let Some(shift) = args.get(3).map(|s| s.parse::<f64>().unwrap()) {
        for j in 0..a.col_ptr.len() - 1 {
            for p in a.col_ptr[j]..a.col_ptr[j + 1] {
                if a.row_idx[p] == j {
                    a.val[p] *= 1.0 + shift;
                }
            }
        }
        println!("diagonal scaled by 1 + {shift}");
    }
    let n = a.col_ptr.len() - 1;
    println!("matrix: {n} rows, {} lower nonzeros", a.row_idx.len());
    let t = Instant::now();
    let ordering = match std::env::var("ORDERING").ok().as_deref().and_then(|o| o.split_once(':')) {
        Some(("nd", spec)) => {
            let (row_ptr, col) = order::node_graph(&a.col_ptr, &a.row_idx, block);
            let nodes = order::nested_dissection_with(&row_ptr, &col, &spec.parse().expect("an ND spec"));
            Ordering::Custom(nodes.iter().flat_map(|&v| (0..block).map(move |c| block * v as usize + c)).collect())
        }
        Some(("perm", path)) => {
            let bytes = std::fs::read(path).expect("reading the permutation");
            let nodes = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize);
            Ordering::Custom(nodes.flat_map(|v| (0..block).map(move |c| block * v + c)).collect())
        }
        _ => Ordering::NestedDissection { block },
    };
    let candidates: usize = std::env::var("CANDIDATES").map_or(1, |c| c.parse().expect("a count"));
    let analysis = if candidates > 1 {
        #[cfg(feature = "vulkan")]
        let cost = hydroplane_solvers::vulkan::ordering_cost;
        #[cfg(not(feature = "vulkan"))]
        let cost = Analysis::flops;
        Analysis::best_of(&a.col_ptr, &a.row_idx, block, candidates, cost).expect("analysis")
    } else {
        Analysis::with_ordering(&a.col_ptr, &a.row_idx, ordering).expect("analysis")
    };
    println!(
        "analysis: {:.0} ms, {} supernodes, factor {} nonzeros",
        t.elapsed().as_secs_f64() * 1e3,
        analysis.supernodal().n_supernodes(),
        analysis.factor_len()
    );
    if std::env::var_os("HYDROPLANE_SOLVERS_PROFILE").is_some() {
        print_shapes(&analysis);
    }
    let b: Vec<f64> = (0..n).map(|i| (0.37 * i as f64).sin() + 0.1).collect();
    let reference = cpu::<f64>(&analysis, &a, &b);
    let factor32 = cpu::<f32>(&analysis, &a, &b);
    println!("    cpu f32 factor vs cpu f64 {:.2e}", factor_difference(&analysis, &factor32, &reference));
    cpu_pcg(&analysis, &a, &b);
    #[cfg(target_os = "macos")]
    {
        accelerate::<f32>(&analysis, &a, &b);
        accelerate::<f64>(&analysis, &a, &b);
        sme::<f32>(&analysis, &a, &b, &reference);
        sme::<f64>(&analysis, &a, &b, &reference);
    }
    #[cfg(feature = "vulkan")]
    {
        let ctx = hydroplane_solvers::vulkan::Context::shared().expect("Vulkan device");
        println!(
            "vulkan device: {} (f64: {}, f64 atomic add: {}, unified memory: {}, shared memory {} KiB)",
            ctx.name(),
            ctx.supports_f64(),
            ctx.supports_atomic_f64(),
            ctx.unified_memory(),
            ctx.shared_bytes() / 1024
        );
        gpu::<f64>(&analysis, &a, &b, &reference);
        gpu::<hydroplane_solvers::Ef64>(&analysis, &a, &b, &reference);
        gpu::<f32>(&analysis, &a, &b, &reference);
        pcg::<f64>(&analysis, &a, &b);
        pcg::<hydroplane_solvers::Ef64>(&analysis, &a, &b);
        split_pcg(&analysis, &a, &b);
    }
    let _ = &reference;
}
