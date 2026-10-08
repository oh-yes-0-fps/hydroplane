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
//! permutation file (`perm:PATH`, `u32` per node, `perm[new] = old`).

use std::time::Instant;

use hydroplane_solvers::cholesky::{Analysis, CpuCholesky, Ordering, Real};
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
    let values: Vec<T> = a.val.iter().map(|&v| T::from_f64(v)).collect();
    let mut chol = CpuCholesky::<T>::new(analysis);
    let ms = best_of(3, || {
        let t = Instant::now();
        chol.factorize(analysis, &values).expect("CPU factorization");
        t.elapsed().as_secs_f64() * 1e3
    });
    let mut x: Vec<T> = b.iter().map(|&v| T::from_f64(v)).collect();
    let solve = best_of(5, || {
        let t = Instant::now();
        chol.solve_in_place(analysis, &mut x);
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
fn gpu<T: Real + Default>(ctx: &std::sync::Arc<hydroplane_solvers::vulkan::Context>, analysis: &Analysis, a: &Matrix, b: &[f64], reference: &[f64]) {
    use hydroplane_solvers::vulkan::VulkanCholesky;
    let name = if T::F64 { "f64" } else { "f32" };
    let t = Instant::now();
    let mut chol = match VulkanCholesky::<T>::new(ctx.clone(), analysis) {
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
    let factor = chol.factor().expect("downloading the factor");
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
    let analysis = Analysis::new(&a.col_ptr, &a.row_idx, ordering).expect("analysis");
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
    cpu::<f32>(&analysis, &a, &b);
    #[cfg(feature = "vulkan")]
    {
        let ctx = std::sync::Arc::new(hydroplane_solvers::vulkan::Context::new().expect("Vulkan device"));
        println!(
            "vulkan device: {} (f64: {}, f64 atomic add: {}, shared memory {} KiB)",
            ctx.name(),
            ctx.supports_f64(),
            ctx.supports_atomic_f64(),
            ctx.shared_bytes() / 1024
        );
        gpu::<f64>(&ctx, &analysis, &a, &b, &reference);
        gpu::<f32>(&ctx, &analysis, &a, &b, &reference);
    }
    let _ = &reference;
}
