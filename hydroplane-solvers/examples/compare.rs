//! Every solver in the crate on the same problems, on whatever this machine has: the numbers
//! behind the README's tables. Prints one tab-separated `ROW` line per measurement (table, row,
//! factor ms, solve ms, relative residual, note), progress to stderr.
//!
//! `cargo run --release --features vulkan --example compare -- [SLAB.bin]`
//!
//! `SLAB.bin` is a dumped SPD tangent (the `cholesky` example's format); without it the sparse
//! SPD tables are skipped. `ONLY` runs the tables whose names contain it.

use std::time::Instant;

use hydroplane_solvers::auto::Options;
use hydroplane_solvers::banded::{Band, BandedLdlt};
use hydroplane_solvers::cholesky::{Analysis, CpuCholesky, CpuPcg, Ordering, Real};
use hydroplane_solvers::dense::DenseCholesky;
use hydroplane_solvers::iterative::{
    BiCgStab, Cg, Chebyshev, Gmres, Identity, Jacobi, Lsqr, LsqrOptions, Minres, Options as Krylov, Spectrum, estimate_spectrum,
};
use hydroplane_solvers::ldlt::{CpuLdlt, Pivoting, kkt_ordering};
use hydroplane_solvers::lu::CpuLu;
use hydroplane_solvers::Solver;
use rayon::prelude::*;

struct Csc {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    val: Vec<f64>,
    /// Lower triangle of a symmetric matrix, else the whole matrix.
    symmetric: bool,
}

impl Csc {
    fn from_entries(n: usize, mut entries: Vec<(usize, usize, f64)>, symmetric: bool) -> Self {
        entries.sort_unstable_by_key(|&(r, c, _)| (c, r));
        let mut col_ptr = vec![0; n + 1];
        for &(_, c, _) in &entries {
            col_ptr[c + 1] += 1;
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
        }
        Self { n, col_ptr, row_idx: entries.iter().map(|e| e.0).collect(), val: entries.iter().map(|e| e.2).collect(), symmetric }
    }

    fn load(path: &str) -> Self {
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
        Self { n, col_ptr, row_idx, val, symmetric: true }
    }

    fn residual(&self, b: &[f64], x: &[f64]) -> f64 {
        let mut y = vec![0.0; self.n];
        for c in 0..self.n {
            for p in self.col_ptr[c]..self.col_ptr[c + 1] {
                let r = self.row_idx[p];
                y[r] += self.val[p] * x[c];
                if self.symmetric && r != c {
                    y[c] += self.val[p] * x[r];
                }
            }
        }
        relative(&y, b)
    }
}

fn relative(y: &[f64], b: &[f64]) -> f64 {
    let rr: f64 = y.iter().zip(b).map(|(y, b)| (y - b) * (y - b)).sum();
    (rr / b.iter().map(|b| b * b).sum::<f64>()).sqrt()
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| (0.37 * i as f64).sin() + 0.1).collect()
}

/// The best of `reps` timings of `f`, in ms, and its last result.
fn best<R>(reps: usize, mut f: impl FnMut() -> R) -> (f64, R) {
    let mut out = None;
    let mut ms = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        let r = f();
        ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
        out = Some(r);
    }
    (ms, out.expect("one repetition"))
}

fn row(table: &str, row: &str, factor: Option<f64>, solve: Option<f64>, residual: Option<f64>, note: &str) {
    let f = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.4}"));
    let r = residual.map_or("-".to_string(), |v| format!("{v:.1e}"));
    println!("ROW\t{table}\t{row}\t{}\t{}\t{r}\t{note}", f(factor), f(solve));
    eprintln!("  {row}: factor {} ms, solve {} ms, residual {r} {note}", f(factor), f(solve));
}

/// Times a direct solver: `factor` repeated, then `solve` on a copy of `b`.
fn direct(table: &str, label: &str, a: &Csc, b: &[f64], mut factor: impl FnMut() -> bool, mut solve: impl FnMut(&mut [f64]) -> String) {
    let (factor_ms, ok) = best(3, &mut factor);
    if !ok {
        row(table, label, None, None, None, "fails");
        return;
    }
    let mut x = b.to_vec();
    let (solve_ms, note) = best(5, || {
        x.copy_from_slice(b);
        solve(&mut x)
    });
    row(table, label, Some(factor_ms), Some(solve_ms), Some(a.residual(b, &x)), &note);
}

fn wanted(table: &str) -> bool {
    std::env::var("ONLY").map_or(true, |only| table.contains(&only))
}

const EIGEN: &str = "eigen";
const SPD: &str = "spd-direct";
const PCG: &str = "spd-pcg";
const AUTO: &str = "spd-auto";

fn spd(a: &Csc) {
    let b = rhs(a.n);
    let analysis = Analysis::with_ordering(&a.col_ptr, &a.row_idx, Ordering::NestedDissection { block: 3 }).expect("analysis");
    eprintln!("slab: {} unknowns, factor {:.1}M entries, {:.1} GFLOP", a.n, analysis.factor_len() as f64 / 1e6, analysis.flops() * 1e-9);
    if wanted(SPD) {
        cpu_direct::<f64>(&analysis, a, &b, "faer, CPU, f64");
        cpu_direct::<f32>(&analysis, a, &b, "faer, CPU, f32");
        #[cfg(target_os = "macos")]
        mac_direct(&analysis, a, &b);
        #[cfg(feature = "vulkan")]
        gpu_direct(&analysis, a, &b);
    }
    if wanted(PCG) {
        let mut pcg = CpuPcg::new(&analysis);
        let (factor_ms, _) = best(3, || pcg.factorize(&a.val).expect("factorization"));
        let mut x = b.clone();
        let (solve_ms, report) = best(3, || {
            x.copy_from_slice(&b);
            pcg.solve_in_place(&mut x)
        });
        row(PCG, "faer f32 factor, f64 CG on the CPU", Some(factor_ms), Some(solve_ms), Some(a.residual(&b, &x)), &format!("{} iterations", report.iterations));
        #[cfg(target_os = "macos")]
        mac_pcg(&analysis, a, &b);
        #[cfg(feature = "vulkan")]
        gpu_pcg(&analysis, a, &b);
    }
    if wanted(EIGEN) {
        eigen(a, &analysis);
    }
    if wanted(AUTO) {
        for (label, tolerance, solves) in [("1 solve per factorization, to 1e-8", 1e-8, 1), ("10 solves per factorization, to 1e-8", 1e-8, 10), ("1 solve per factorization, to 1e-10", 1e-10, 1)] {
            let options = Options { tolerance, solves_per_factorization: solves as f64, factorizations: 10, ..Default::default() };
            let mut solver = Solver::with_options(&a.col_ptr, &a.row_idx, options).expect("pattern");
            let cycle = |solver: &mut Solver| {
                let t = Instant::now();
                solver.factorize(&a.val).expect("factorization");
                let factor = t.elapsed().as_secs_f64() * 1e3;
                let mut residual = 0.0f64;
                let t = Instant::now();
                for _ in 0..solves {
                    let mut x = b.clone();
                    solver.solve_in_place(&mut x).expect("solve");
                    residual = residual.max(a.residual(&b, &x));
                }
                let solve = t.elapsed().as_secs_f64() * 1e3 / solves as f64;
                (factor, solve, residual)
            };
            for _ in 0..2 {
                cycle(&mut solver);
            }
            let runs: Vec<(f64, f64, f64)> = (0..3).map(|_| cycle(&mut solver)).collect();
            let best = runs.iter().copied().min_by(|x, y| (x.0 + solves as f64 * x.1).total_cmp(&(y.0 + solves as f64 * y.1))).unwrap();
            row(AUTO, label, Some(best.0), Some(best.1), Some(best.2), &solver.plan().strategy.to_string());
        }
    }
}

/// A preconditioner `z ← T r`.
type Apply<'a> = &'a mut dyn FnMut(&[f64], &mut [f64]);
/// Runs and reports one eigen solve: label, factorization time, preconditioner.
#[cfg(feature = "vulkan")]
type Run<'a> = dyn Fn(&str, f64, Apply) + 'a;

/// The 8 lowest eigenpairs of the slab tangent by LOBPCG, preconditioned by each Cholesky.
fn eigen(a: &Csc, analysis: &Analysis) {
    use hydroplane_solvers::eigen::Lobpcg;
    let k = 8;
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); a.n];
    for c in 0..a.n {
        for p in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[p];
            rows[r].push((c as u32, a.val[p]));
            if r != c {
                rows[c].push((r as u32, a.val[p]));
            }
        }
    }
    let multiply = |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().with_min_len(512).for_each(|(i, y)| *y = rows[i].iter().map(|&(c, v)| v * x[c as usize]).sum());
    };
    let run = |label: &str, factor_ms: f64, precondition: Apply| {
        let t = Instant::now();
        let found = Lobpcg::new().smallest(a.n, k, multiply, |r: &[f64], z: &mut [f64]| precondition(r, z));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let worst = (0..k)
            .map(|j| {
                let mut ax = vec![0.0; a.n];
                multiply(&found.vectors[j], &mut ax);
                let r: f64 = ax.iter().zip(&found.vectors[j]).map(|(a, x)| (a - found.values[j] * x).powi(2)).sum::<f64>().sqrt();
                r / found.values[j]
            })
            .fold(0.0, f64::max);
        let note = format!("{} iterations{}, λ₁ = {:.4e}", found.iterations, if found.converged { "" } else { ", not converged" }, found.values[0]);
        row(EIGEN, label, Some(factor_ms), Some(ms), Some(worst), &note);
    };
    let mut chol = CpuCholesky::<f64>::new(analysis);
    let (ms, _) = best(1, || chol.factorize(&a.val).expect("factorization"));
    run("LOBPCG, faer Cholesky (CPU, f64) preconditioner", ms, &mut |r, z| {
        z.copy_from_slice(r);
        chol.solve_in_place(z);
    });
    #[cfg(feature = "vulkan")]
    {
        use hydroplane_solvers::Ef64;
        use hydroplane_solvers::vulkan::{Context, DeviceReal, VulkanCholesky};
        fn gpu<T: DeviceReal>(analysis: &Analysis, a: &Csc, label: &str, run: &Run) {
            let Ok(mut chol) = VulkanCholesky::<T>::new(analysis) else { return };
            let (ms, _) = best(1, || chol.factorize(&a.val).expect("factorization"));
            run(label, ms, &mut |r, z| {
                z.copy_from_slice(r);
                chol.solve_in_place(z).expect("solve");
            });
        }
        if let Ok(ctx) = Context::shared() {
            if ctx.supports_f64() {
                gpu::<f64>(analysis, a, "LOBPCG, Vulkan Cholesky (GPU, f64) preconditioner", &run);
            }
            gpu::<Ef64>(analysis, a, "LOBPCG, Vulkan Cholesky (GPU, emulated f64) preconditioner", &run);
            gpu::<f32>(analysis, a, "LOBPCG, Vulkan Cholesky (GPU, f32) preconditioner", &run);
        }
    }
    #[cfg(target_os = "macos")]
    {
        use hydroplane_solvers::sme::SmeCholesky;
        if let Ok(mut sme) = SmeCholesky::<f64>::new(analysis) {
            let (ms, _) = best(1, || sme.factorize(&a.val).expect("factorization"));
            run("LOBPCG, SME Cholesky (CPU matrix units, f64) preconditioner", ms, &mut |r, z| {
                z.copy_from_slice(r);
                sme.solve_in_place(z);
            });
        }
    }
}

fn cpu_direct<T: Real>(analysis: &Analysis, a: &Csc, b: &[f64], label: &str) {
    let chol = std::cell::RefCell::new(CpuCholesky::<T>::new(analysis));
    direct(SPD, label, a, b, || chol.borrow_mut().factorize(&a.val).is_ok(), |x| {
        chol.borrow().solve_in_place(x);
        String::new()
    });
}

#[cfg(target_os = "macos")]
fn mac_direct(analysis: &Analysis, a: &Csc, b: &[f64]) {
    use hydroplane_solvers::accelerate::AccelerateCholesky;
    use hydroplane_solvers::sme::SmeCholesky;
    for f64_ in [true, false] {
        let p = if f64_ { "f64" } else { "f32" };
        macro_rules! run {
            ($t:ty) => {{
                let mut acc = AccelerateCholesky::<$t>::new(analysis).expect("Accelerate");
                let acc = std::cell::RefCell::new(&mut acc);
                direct(SPD, &format!("Accelerate, CPU matrix units, {p}"), a, b, || acc.borrow_mut().factorize(&a.val).is_ok(), |x| {
                    acc.borrow_mut().solve_in_place(x).expect("solve");
                    String::new()
                });
                if let Ok(mut sme) = SmeCholesky::<$t>::new(analysis) {
                    let sme = std::cell::RefCell::new(&mut sme);
                    direct(SPD, &format!("SME (this crate), CPU matrix units, {p}"), a, b, || sme.borrow_mut().factorize(&a.val).is_ok(), |x| {
                        sme.borrow().solve_in_place(x);
                        String::new()
                    });
                }
            }};
        }
        if f64_ {
            run!(f64)
        } else {
            run!(f32)
        }
    }
}

#[cfg(feature = "vulkan")]
fn context() -> Option<std::sync::Arc<hydroplane_solvers::vulkan::Context>> {
    hydroplane_solvers::vulkan::Context::shared().ok()
}

#[cfg(feature = "vulkan")]
fn gpu_direct(analysis: &Analysis, a: &Csc, b: &[f64]) {
    use hydroplane_solvers::Ef64;
    use hydroplane_solvers::vulkan::{DeviceReal, VulkanCholesky};
    let Some(ctx) = context() else { return };
    eprintln!("GPU: {}", ctx.name());
    fn one<T: DeviceReal>(analysis: &Analysis, a: &Csc, b: &[f64], label: &str) {
        let Ok(chol) = VulkanCholesky::<T>::new(analysis) else {
            row(SPD, label, None, None, None, "unsupported");
            return;
        };
        let chol = std::cell::RefCell::new(chol);
        direct(SPD, label, a, b, || chol.borrow_mut().factorize(&a.val).is_ok(), |x| {
            chol.borrow_mut().solve_in_place(x).expect("solve");
            String::new()
        });
    }
    if ctx.supports_f64() {
        one::<f64>(analysis, a, b, "Vulkan (this crate), GPU, f64");
    }
    one::<Ef64>(analysis, a, b, "Vulkan (this crate), GPU, emulated f64");
    one::<f32>(analysis, a, b, "Vulkan (this crate), GPU, f32");
    #[cfg(target_os = "macos")]
    {
        use hydroplane_solvers::sme::SmeCholesky;
        let Ok(mut sme) = SmeCholesky::<f64>::new(analysis) else { return };
        let Ok(mut gpu) = VulkanCholesky::<Ef64>::new(analysis) else { return };
        let gpu = std::cell::RefCell::new(&mut gpu);
        direct(SPD, "SME f64 factor → GPU solve (emulated f64)", a, b, || sme.factorize(&a.val).is_ok() && gpu.borrow_mut().load_factor(sme.factor()).is_ok(), |x| {
            gpu.borrow_mut().solve_in_place(x).expect("solve");
            String::new()
        });
    }
}

#[cfg(target_os = "macos")]
fn mac_pcg(analysis: &Analysis, a: &Csc, b: &[f64]) {
    use hydroplane_solvers::accelerate::AccelerateCholesky;
    use hydroplane_solvers::sme::SmeCholesky;
    let mut pcg = CpuPcg::new(analysis);
    pcg.set_values(&a.val);
    if let Ok(mut sme) = SmeCholesky::<f32>::new(analysis) {
        let (factor_ms, _) = best(3, || sme.factorize(&a.val).expect("factorization"));
        let mut x = b.to_vec();
        let (solve_ms, report) = best(3, || {
            x.copy_from_slice(b);
            pcg.solve_with(&mut x, |r, z| {
                z.copy_from_slice(r);
                sme.solve_in_place(z);
            })
        });
        row(PCG, "SME f32 factor, f64 CG on the CPU", Some(factor_ms), Some(solve_ms), Some(a.residual(b, &x)), &format!("{} iterations", report.iterations));
    }
    let mut acc = AccelerateCholesky::<f32>::new(analysis).expect("Accelerate");
    let (factor_ms, _) = best(3, || acc.factorize(&a.val).expect("factorization"));
    let mut x = b.to_vec();
    let (solve_ms, report) = best(3, || {
        x.copy_from_slice(b);
        pcg.solve_with(&mut x, |r, z| {
            z.copy_from_slice(r);
            acc.solve_in_place(z).expect("solve");
        })
    });
    row(PCG, "Accelerate f32 factor, f64 CG on the CPU", Some(factor_ms), Some(solve_ms), Some(a.residual(b, &x)), &format!("{} iterations", report.iterations));
}

#[cfg(feature = "vulkan")]
fn gpu_pcg(analysis: &Analysis, a: &Csc, b: &[f64]) {
    use hydroplane_solvers::Ef64;
    use hydroplane_solvers::vulkan::{DeviceReal, VulkanCholesky, VulkanPcg};
    let Some(ctx) = context() else { return };
    fn one<T: DeviceReal>(analysis: &Analysis, a: &Csc, b: &[f64], label: &str) {
        let Ok(mut pcg) = VulkanPcg::<T>::new(analysis) else { return };
        let (factor_ms, _) = best(3, || pcg.factorize(&a.val).expect("factorization"));
        let mut x = b.to_vec();
        let (solve_ms, report) = best(5, || {
            x.copy_from_slice(b);
            pcg.solve_in_place(&mut x).expect("solve")
        });
        row(PCG, label, Some(factor_ms), Some(solve_ms), Some(a.residual(b, &x)), &format!("{} iterations", report.iterations));
    }
    if ctx.supports_f64() {
        one::<f64>(analysis, a, b, "GPU f32 factor, f64 CG on the GPU");
    }
    one::<Ef64>(analysis, a, b, "GPU f32 factor, emulated-f64 CG on the GPU");
    let Ok(mut chol) = VulkanCholesky::<f32>::new(analysis) else { return };
    let mut pcg = CpuPcg::new(analysis);
    pcg.set_values(&a.val);
    let (factor_ms, _) = best(3, || chol.factorize(&a.val).expect("factorization"));
    let mut x = b.to_vec();
    let (solve_ms, report) = best(3, || {
        x.copy_from_slice(b);
        pcg.solve_with(&mut x, |r, z| {
            z.copy_from_slice(r);
            chol.solve_in_place(z).expect("solve");
        })
    });
    row(PCG, "GPU f32 factor, f64 CG on the CPU", Some(factor_ms), Some(solve_ms), Some(a.residual(b, &x)), &format!("{} iterations", report.iterations));
}

fn neighbours(g: usize, k: usize) -> impl Iterator<Item = (usize, usize)> {
    let (i, j, l) = (k % g, (k / g) % g, k / (g * g));
    [(i > 0, k.wrapping_sub(1), 0), (j > 0, k.wrapping_sub(g), 1), (l > 0, k.wrapping_sub(g * g), 2), (i + 1 < g, k + 1, 3), (j + 1 < g, k + g, 4), (l + 1 < g, k + g * g, 5)]
        .into_iter()
        .filter(|e| e.0)
        .map(|e| (e.1, e.2))
}

const KKT: &str = "ldlt";

fn ldlt() {
    let g = 36;
    let nh = g * g * g;
    let mut e = Vec::new();
    for k in 0..nh {
        e.push((k, k, 6.01));
        e.extend(neighbours(g, k).filter(|&(o, _)| o < k).map(|(o, _)| (k, o, -1.0)));
    }
    let mut m = 0;
    for k in (0..nh).step_by(2) {
        if k % g + 1 < g {
            let r = nh + m;
            e.extend([(r, k, 0.5), (r, k + 1, 0.5), (r, r, 0.0)]);
            m += 1;
        }
    }
    let a = Csc::from_entries(nh + m, e, true);
    let b = rhs(a.n);
    eprintln!("KKT: {nh} + {m} unknowns");
    let signs: Vec<i8> = (0..a.n).map(|i| if i < nh { 1 } else { -1 }).collect();
    let amd = Analysis::with_ordering(&a.col_ptr, &a.row_idx, Ordering::Amd).unwrap();
    let kkt = Analysis::with_ordering(&a.col_ptr, &a.row_idx, Ordering::Custom(kkt_ordering(&a.col_ptr, &a.row_idx, &signs).unwrap())).unwrap();
    let regularized = |delta| Pivoting::Regularized { signs: signs.clone(), delta, epsilon: delta * 1e-4 };
    let run = |label: &str, analysis: &Analysis, pivoting: Pivoting, f32_: bool, refine: bool| {
        macro_rules! go {
            ($t:ty) => {{
                let mut f = CpuLdlt::<$t>::with_pivoting(analysis, pivoting);
                let f = std::cell::RefCell::new(&mut f);
                direct(KKT, label, &a, &b, || f.borrow_mut().factorize(&a.val).is_ok(), |x| {
                    if refine {
                        format!("{} GMRES steps", f.borrow().solve_refined(x).iterations)
                    } else {
                        f.borrow().solve_in_place(x);
                        String::new()
                    }
                });
            }};
        }
        if f32_ { go!(f32) } else { go!(f64) }
    };
    run("faer, f64, AMD (constraint pivots regularized)", &amd, regularized(1e-9), false, false);
    run("faer, f64, AMD, + GMRES refinement to 1e-12", &amd, regularized(1e-9), false, true);
    run("faer, f64, KKT ordering", &kkt, regularized(1e-9), false, false);
    run("faer, f64, KKT ordering, Bunch–Kaufman", &kkt, Pivoting::BunchKaufman, false, false);
    run("faer, f32, KKT ordering, + GMRES refinement to 1e-12", &kkt, regularized(1e-6), true, true);
    #[cfg(feature = "vulkan")]
    {
        use hydroplane_solvers::Ef64;
        use hydroplane_solvers::vulkan::{Context, DeviceReal, VulkanLdlt};
        fn one<T: DeviceReal>(kkt: &Analysis, a: &Csc, b: &[f64], signs: &[i8], label: &str, refine: bool) {
            let Ok(f) = VulkanLdlt::<T>::new(kkt, signs) else { return };
            let f = std::cell::RefCell::new(f);
            direct(KKT, label, a, b, || f.borrow_mut().factorize(&a.val).is_ok(), |x| {
                if refine {
                    format!("{} GMRES steps", f.borrow_mut().solve_refined(x).expect("solve").iterations)
                } else {
                    f.borrow_mut().solve_in_place(x).expect("solve");
                    String::new()
                }
            });
        }
        if let Ok(ctx) = Context::shared() {
            if ctx.supports_f64() {
                one::<f64>(&kkt, &a, &b, &signs, "Vulkan (this crate), GPU, f64, KKT ordering", false);
            }
            one::<Ef64>(&kkt, &a, &b, &signs, "Vulkan (this crate), GPU, emulated f64, KKT ordering", false);
            one::<f32>(&kkt, &a, &b, &signs, "Vulkan (this crate), GPU, f32, KKT ordering, + GMRES refinement to 1e-12", true);
        }
    }
    #[cfg(target_os = "macos")]
    {
        use hydroplane_solvers::accelerate::{AccelerateLdlt, LdltPivoting};
        let mut tpp = AccelerateLdlt::<f64>::new(&amd).unwrap();
        let tpp = std::cell::RefCell::new(&mut tpp);
        direct(KKT, "Accelerate, f64, threshold partial pivoting", &a, &b, || tpp.borrow_mut().factorize(&a.val).is_ok(), |x| {
            tpp.borrow_mut().solve_in_place(x).unwrap();
            String::new()
        });
        let mut sbk = AccelerateLdlt::<f64>::with_pivoting(&amd, LdltPivoting::SupernodeBunchKaufman).unwrap();
        let sbk = std::cell::RefCell::new(&mut sbk);
        direct(KKT, "Accelerate, f64, supernode Bunch–Kaufman", &a, &b, || sbk.borrow_mut().factorize(&a.val).is_ok(), |x| {
            sbk.borrow_mut().solve_in_place(x).unwrap();
            String::new()
        });
    }
}

const LU: &str = "lu";

fn lu() {
    let g = 36;
    let n = g * g * g;
    let mut e = Vec::new();
    for k in 0..n {
        e.push((k, k, 10.0));
        e.extend(neighbours(g, k).map(|(o, dir)| (k, o, if dir == 0 { -5.0 } else { -1.0 })));
    }
    let a = Csc::from_entries(n, e, false);
    let b = rhs(n);
    eprintln!("convection–diffusion: {n} unknowns");
    let mut f = CpuLu::<f64>::new(&a.col_ptr, &a.row_idx).unwrap();
    let f = std::cell::RefCell::new(&mut f);
    direct(LU, "faer, f64", &a, &b, || f.borrow_mut().factorize(&a.val).is_ok(), |x| {
        f.borrow().solve_in_place(x);
        String::new()
    });
    let mut f = CpuLu::<f32>::new(&a.col_ptr, &a.row_idx).unwrap();
    let f = std::cell::RefCell::new(&mut f);
    direct(LU, "faer, f32, + GMRES refinement to 1e-12", &a, &b, || f.borrow_mut().factorize(&a.val).is_ok(), |x| {
        format!("{} GMRES steps", f.borrow().solve_refined(x).iterations)
    });
    use hydroplane_solvers::lu::{FrontalLu, LuOptions};
    for (label, matching) in [("FrontalLu (this crate), f64, nested dissection", false), ("FrontalLu (this crate), f64, nested dissection, matching", true)] {
        let options = LuOptions { matching: Some(matching), ..Default::default() };
        let f = std::cell::RefCell::new(FrontalLu::<f64>::with_options(&a.col_ptr, &a.row_idx, options).unwrap());
        direct(LU, label, &a, &b, || f.borrow_mut().factorize(&a.val).is_ok(), |x| {
            f.borrow().solve_in_place(x);
            format!("{} pivots perturbed", f.borrow().perturbed())
        });
    }
    let f = std::cell::RefCell::new(FrontalLu::<f32>::new(&a.col_ptr, &a.row_idx).unwrap());
    direct(LU, "FrontalLu (this crate), f32, + GMRES refinement to 1e-12", &a, &b, || f.borrow_mut().factorize(&a.val).is_ok(), |x| {
        format!("{} GMRES steps", f.borrow().solve_refined(x).iterations)
    });
    #[cfg(target_os = "macos")]
    {
        use hydroplane_solvers::accelerate::AccelerateLu;
        let mut acc = AccelerateLu::<f64>::new(&a.col_ptr, &a.row_idx).unwrap();
        let acc = std::cell::RefCell::new(&mut acc);
        direct(LU, "Accelerate, f64", &a, &b, || acc.borrow_mut().factorize(&a.val).is_ok(), |x| {
            acc.borrow_mut().solve_in_place(x).unwrap();
            String::new()
        });
    }
}

const BANDED: &str = "banded";

fn banded() {
    let (n, w) = (1_000_000, 4);
    let mut e = Vec::new();
    for i in 0..n {
        e.push((i, i, 2.0 * w as f64 + 1.0));
        for k in 1..=w.min(i) {
            e.push((i, i - k, -1.0 / k as f64));
        }
    }
    let a = Csc::from_entries(n, e, true);
    let b = rhs(n);
    eprintln!("band: {n} unknowns, half-bandwidth {w}");
    let band = Band::from_lower(&a.col_ptr, &a.row_idx, &a.val).unwrap();
    let f = std::cell::RefCell::new(None);
    direct(BANDED, "BandedLdlt, f64", &a, &b, || {
        *f.borrow_mut() = BandedLdlt::positive_definite(band.clone()).ok();
        f.borrow().is_some()
    }, |x| {
        f.borrow().as_ref().unwrap().solve_in_place(x);
        String::new()
    });
    let analysis = Analysis::with_ordering(&a.col_ptr, &a.row_idx, Ordering::Amd).unwrap();
    let mut chol = CpuCholesky::<f64>::new(&analysis);
    let chol = std::cell::RefCell::new(&mut chol);
    direct(BANDED, "faer sparse Cholesky, f64 (AMD)", &a, &b, || chol.borrow_mut().factorize(&a.val).is_ok(), |x| {
        chol.borrow().solve_in_place(x);
        String::new()
    });
}

const KRYLOV: &str = "krylov";

/// The 7-point operator `(6 + shift + c) u − neighbours − c·(west)` on a `g³` grid, threaded.
fn stencil(g: usize, shift: f64, c: f64) -> impl Fn(&[f64], &mut [f64]) + Sync {
    move |x: &[f64], y: &mut [f64]| {
        y.par_iter_mut().enumerate().with_min_len(4096).for_each(|(k, y)| {
            let mut v = (6.0 + shift + c) * x[k];
            for (o, dir) in neighbours(g, k) {
                v -= x[o] * if dir == 0 { 1.0 + c } else { 1.0 };
            }
            *y = v;
        });
    }
}

fn krylov() {
    let g = 64;
    let n = g * g * g;
    let b = rhs(n);
    eprintln!("Krylov: {n} unknowns, matrix-free");
    let report_setup = |label: &str, a: &dyn Fn(&[f64], &mut [f64]), ms: f64, x: &[f64], iterations: usize, converged: bool, extra: &str| {
        let mut ax = vec![0.0; n];
        a(x, &mut ax);
        let extra = if extra.is_empty() { String::new() } else { format!(", {extra}") };
        row(KRYLOV, label, None, Some(ms), Some(relative(&ax, &b)), &format!("{iterations} iterations{}{extra}", if converged { "" } else { ", not converged" }));
    };
    let report = |label: &str, a: &dyn Fn(&[f64], &mut [f64]), ms: f64, x: &[f64], iterations: usize, converged: bool| report_setup(label, a, ms, x, iterations, converged, "");
    let poisson = stencil(g, 0.0, 0.0);
    let diag = vec![6.0; n];
    let cg_options = Krylov { tolerance: 1e-8, max_iterations: 5000, ..Default::default() };
    let mut x = vec![0.0; n];
    let (ms, r) = best(3, || Cg::with_options(cg_options).solve(&mut |x: &[f64], y: &mut [f64]| poisson(x, y), &mut Jacobi::new(&diag), &b[..], &mut x[..]));
    report("CG + Jacobi, Poisson", &poisson, ms, &x, r.iterations, r.converged());
    let start: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
    let lmax = estimate_spectrum(&mut |x: &[f64], y: &mut [f64]| poisson(x, y), &mut Jacobi::new(&diag), &start[..]).max;
    let (ms, r) = best(3, || {
        let mut m = Chebyshev::new(|x: &[f64], y: &mut [f64]| poisson(x, y), Jacobi::new(&diag), Spectrum { min: lmax / 1000.0, max: 1.1 * lmax });
        m.degree = 4;
        Cg::with_options(cg_options).solve(&mut |x: &[f64], y: &mut [f64]| poisson(x, y), &mut m, &b[..], &mut x[..])
    });
    report("CG + Chebyshev(4)–Jacobi, Poisson", &poisson, ms, &x, r.iterations, r.converged());
    let indefinite = stencil(g, -0.05, 0.0);
    let options = Krylov { tolerance: 1e-8, max_iterations: 5000, ..Default::default() };
    let (ms, r) = best(3, || Minres::with_options(options).solve(&mut |x: &[f64], y: &mut [f64]| indefinite(x, y), &mut Identity, &b[..], &mut x[..]));
    report("MINRES, indefinite (Poisson − 0.05 I)", &indefinite, ms, &x, r.iterations, r.converged());
    let convection = stencil(g, 0.0, 2.0);
    let cdiag = vec![8.0; n];
    let (ms, r) = best(3, || Gmres::with_options(options).solve(&mut |x: &[f64], y: &mut [f64]| convection(x, y), &mut Jacobi::new(&cdiag), &b[..], &mut x[..]));
    report("GMRES(30) + Jacobi, convection–diffusion", &convection, ms, &x, r.iterations, r.converged());
    let (ms, r) = best(3, || Gmres::with_options(Krylov { flexible: true, ..options }).solve(&mut |x: &[f64], y: &mut [f64]| convection(x, y), &mut Jacobi::new(&cdiag), &b[..], &mut x[..]));
    report("FGMRES(30) + Jacobi, convection–diffusion", &convection, ms, &x, r.iterations, r.converged());
    let (ms, r) = best(3, || BiCgStab::with_options(options).solve(&mut |x: &[f64], y: &mut [f64]| convection(x, y), &mut Jacobi::new(&cdiag), &b[..], &mut x[..]));
    report("BiCGStab + Jacobi, convection–diffusion", &convection, ms, &x, r.iterations, r.converged());
    let assembled = |shift: f64, c: f64, lower: bool| {
        let mut e = Vec::new();
        for k in 0..n {
            e.push((k, k, 6.0 + shift + c));
            e.extend(neighbours(g, k).filter(|&(o, _)| !lower || o < k).map(|(o, dir)| (k, o, if dir == 0 { -1.0 - c } else { -1.0 })));
        }
        Csc::from_entries(n, e, lower)
    };
    let a = assembled(0.0, 2.0, false);
    #[cfg(feature = "vulkan")]
    {
        use hydroplane_solvers::Ef64;
        use hydroplane_solvers::vulkan::{Context, DeviceReal, VulkanBiCgStab};
        fn gpu<T: DeviceReal>(a: &Csc, b: &[f64], label: &str, apply: &dyn Fn(&[f64], &mut [f64])) {
            let Ok(mut solver) = VulkanBiCgStab::<T>::new(&a.col_ptr, &a.row_idx) else { return };
            solver.options.max_iterations = 5000;
            solver.factorize(&a.val).expect("values");
            let mut x = b.to_vec();
            let (ms, r) = best(3, || {
                x.copy_from_slice(b);
                solver.solve_in_place(&mut x).expect("solve")
            });
            let mut ax = vec![0.0; b.len()];
            apply(&x, &mut ax);
            row(KRYLOV, label, None, Some(ms), Some(relative(&ax, b)), &format!("{} iterations{}", r.iterations, if r.converged { "" } else { ", not converged" }));
        }
        if let Ok(ctx) = Context::shared() {
            if ctx.supports_f64() {
                gpu::<f64>(&a, &b, "BiCGStab + Jacobi on the GPU (Vulkan, f64), convection–diffusion", &convection);
            }
            gpu::<Ef64>(&a, &b, "BiCGStab + Jacobi on the GPU (Vulkan, emulated f64), convection–diffusion", &convection);
        }
    }
    let (setup, mut ilu) = best(1, || hydroplane_solvers::iterative::Ilu0::new(&a.col_ptr, &a.row_idx, &a.val).unwrap());
    let (ms, r) = best(3, || BiCgStab::with_options(options).solve(&mut |x: &[f64], y: &mut [f64]| convection(x, y), &mut ilu, &b[..], &mut x[..]));
    report_setup("BiCGStab + ILU(0), convection–diffusion", &convection, ms, &x, r.iterations, r.converged(), &format!("set up in {setup:.0} ms"));
    let a = assembled(0.0, 0.0, true);
    #[cfg(feature = "vulkan")]
    {
        use hydroplane_solvers::Ef64;
        use hydroplane_solvers::vulkan::{Context, DeviceReal, VulkanPcg};
        fn gpu<T: DeviceReal>(a: &Csc, b: &[f64], label: &str, apply: &dyn Fn(&[f64], &mut [f64]), amg: bool) {
            let made = if amg { VulkanPcg::<T>::amg(&a.col_ptr, &a.row_idx) } else { VulkanPcg::<T>::jacobi(&a.col_ptr, &a.row_idx) };
            let Ok(mut pcg) = made else { return };
            pcg.options.max_iterations = 5000;
            let (setup, _) = best(1, || pcg.factorize(&a.val).expect("values"));
            let mut x = b.to_vec();
            let (ms, r) = best(3, || {
                x.copy_from_slice(b);
                pcg.solve_in_place(&mut x).expect("solve")
            });
            let mut ax = vec![0.0; b.len()];
            apply(&x, &mut ax);
            let set_up = if amg { format!(", set up in {setup:.0} ms") } else { String::new() };
            row(KRYLOV, label, None, Some(ms), Some(relative(&ax, b)), &format!("{} iterations{}{set_up}", r.iterations, if r.converged { "" } else { ", not converged" }));
        }
        if let Ok(ctx) = Context::shared() {
            for (amg, name) in [(false, "Jacobi"), (true, "AMG")] {
                if ctx.supports_f64() {
                    gpu::<f64>(&a, &b, &format!("CG + {name} on the GPU (Vulkan, f64), Poisson"), &poisson, amg);
                }
                gpu::<Ef64>(&a, &b, &format!("CG + {name} on the GPU (Vulkan, emulated f64), Poisson"), &poisson, amg);
            }
        }
    }
    let (setup, mut amg) = best(1, || hydroplane_solvers::iterative::Amg::new(&a.col_ptr, &a.row_idx, &a.val).unwrap());
    let (ms, r) = best(3, || Cg::with_options(cg_options).solve(&mut |x: &[f64], y: &mut [f64]| poisson(x, y), &mut amg, &b[..], &mut x[..]));
    report_setup("CG + AMG, Poisson", &poisson, ms, &x, r.iterations, r.converged(), &format!("{} levels, set up in {setup:.0} ms", amg.levels()));
    let (setup, mut ic) = best(1, || hydroplane_solvers::iterative::Ic0::new(&a.col_ptr, &a.row_idx, &a.val).unwrap());
    let (ms, r) = best(3, || Cg::with_options(cg_options).solve(&mut |x: &[f64], y: &mut [f64]| poisson(x, y), &mut ic, &b[..], &mut x[..]));
    report_setup("CG + IC(0), Poisson", &poisson, ms, &x, r.iterations, r.converged(), &format!("set up in {setup:.0} ms"));
}

const LSQ: &str = "least-squares";

fn least_squares() {
    let g = 300;
    let n = g * g;
    let node = |i: usize, j: usize| j * g + i;
    let mut e = Vec::new();
    let mut m = 0;
    for j in 0..g - 1 {
        for i in 0..g - 1 {
            for s in 0..2usize {
                let h = (m * 2654435761usize) % 1000;
                let (u, v) = ((h % 31) as f64 / 31.0, (h / 31 % 29) as f64 / 29.0 * (0.5 + 0.5 * s as f64));
                e.extend([(m, node(i, j), (1.0 - u) * (1.0 - v)), (m, node(i + 1, j), u * (1.0 - v)), (m, node(i, j + 1), (1.0 - u) * v), (m, node(i + 1, j + 1), u * v)]);
                m += 1;
            }
        }
    }
    let smooth = 0.1;
    for j in 0..g {
        for i in 0..g {
            if i + 1 < g {
                e.extend([(m, node(i, j), -smooth), (m, node(i + 1, j), smooth)]);
                m += 1;
            }
            if j + 1 < g {
                e.extend([(m, node(i, j), -smooth), (m, node(i, j + 1), smooth)]);
                m += 1;
            }
        }
    }
    e.sort_unstable_by_key(|&(r, c, _)| (c, r));
    let mut col_ptr = vec![0; n + 1];
    for &(_, c, _) in &e {
        col_ptr[c + 1] += 1;
    }
    for c in 0..n {
        col_ptr[c + 1] += col_ptr[c];
    }
    let row_idx: Vec<usize> = e.iter().map(|x| x.0).collect();
    let val: Vec<f64> = e.iter().map(|x| x.2).collect();
    let b = rhs(m);
    eprintln!("least squares: {m} × {n}, {} entries", val.len());
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for c in 0..n {
        for p in col_ptr[c]..col_ptr[c + 1] {
            rows[row_idx[p]].push((c, val[p]));
        }
    }
    let normal_residual = |x: &[f64]| {
        let r: Vec<f64> = (0..m).map(|i| b[i] - rows[i].iter().map(|&(c, v)| v * x[c]).sum::<f64>()).collect();
        let atr: Vec<f64> = (0..n).map(|c| (col_ptr[c]..col_ptr[c + 1]).map(|p| val[p] * r[row_idx[p]]).sum()).collect();
        let atb: Vec<f64> = (0..n).map(|c| (col_ptr[c]..col_ptr[c + 1]).map(|p| val[p] * b[row_idx[p]]).sum()).collect();
        (atr.iter().map(|v| v * v).sum::<f64>() / atb.iter().map(|v| v * v).sum::<f64>()).sqrt()
    };
    let mut x = vec![0.0; n];
    let (ms, r) = best(3, || {
        Lsqr::with_options(LsqrOptions { atol: 1e-10, btol: 1e-10, max_iterations: 20_000, ..Default::default() }).solve(
            |x: &[f64], y: &mut [f64]| y.par_iter_mut().enumerate().for_each(|(i, y)| *y = rows[i].iter().map(|&(c, v)| v * x[c]).sum()),
            |u: &[f64], v: &mut [f64]| v.par_iter_mut().enumerate().for_each(|(c, v)| *v = (col_ptr[c]..col_ptr[c + 1]).map(|p| val[p] * u[row_idx[p]]).sum()),
            &b[..],
            &mut x[..],
        )
    });
    row(LSQ, "LSQR, matrix-free, no preconditioner", None, Some(ms), Some(normal_residual(&x)), &format!("{} iterations", r.iterations));
    let mut qr = hydroplane_solvers::qr::CpuQr::<f64>::new(m, &col_ptr, &row_idx).unwrap();
    let (factor_ms, _) = best(3, || qr.factorize(&val).unwrap());
    let (solve_ms, _) = best(5, || qr.solve(&b, &mut x).unwrap());
    row(LSQ, "faer sparse QR, f64", Some(factor_ms), Some(solve_ms), Some(normal_residual(&x)), "");
    #[cfg(target_os = "macos")]
    {
        use hydroplane_solvers::accelerate::AccelerateQr;
        let mut qr = AccelerateQr::<f64>::new(m, &col_ptr, &row_idx).unwrap();
        let (factor_ms, _) = best(3, || qr.factorize(&val).unwrap());
        let (solve_ms, _) = best(5, || qr.solve(&b, &mut x).unwrap());
        row(LSQ, "Accelerate sparse QR, f64", Some(factor_ms), Some(solve_ms), Some(normal_residual(&x)), "");
    }
}

const DENSE: &str = "dense";

fn dense() {
    let n = 400;
    let a: Vec<f64> = (0..n * n).map(|k| if k / n == k % n { n as f64 } else { (((k / n) * (k % n)) % 7) as f64 / 7.0 - 0.5 }).collect();
    let sym: Vec<f64> = (0..n * n).map(|k| 0.5 * (a[k] + a[(k % n) * n + k / n])).collect();
    let b = rhs(n);
    let (factor_ms, chol) = best(5, || DenseCholesky::new(n, &sym));
    let mut x = b.clone();
    let (solve_ms, _) = best(5, || {
        x.copy_from_slice(&b);
        chol.solve_in_place(&mut x)
    });
    let ax: Vec<f64> = (0..n).map(|i| (0..n).map(|j| sym[i * n + j] * x[j]).sum()).collect();
    row(DENSE, "DenseCholesky, f64, n = 400", Some(factor_ms), Some(solve_ms), Some(relative(&ax, &b)), "");
}

fn main() {
    if let Some(path) = std::env::args().nth(1) {
        spd(&Csc::load(&path));
    }
    if wanted(KKT) {
        ldlt();
    }
    if wanted(LU) {
        lu();
    }
    if wanted(BANDED) {
        banded();
    }
    if wanted(KRYLOV) {
        krylov();
    }
    if wanted(LSQ) {
        least_squares();
    }
    if wanted(DENSE) {
        dense();
    }
}
