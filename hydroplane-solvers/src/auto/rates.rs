//! How fast this machine's devices are, measured once per process on two synthetic FEM-like
//! matrices and fitted to the cost model [`super::policy::estimate`] predicts with.

use std::sync::OnceLock;
use std::time::Instant;

use super::caps::Caps;
use crate::Precision;
use crate::cholesky::{Analysis, CpuCholesky, CpuPcg, Ordering, PcgOptions};

/// Measured costs (wall time, as a caller sees it): plain data, so they can be stored and
/// passed back through [`super::Options::rates`] instead of measuring again.
#[derive(Clone, Debug, PartialEq)]
pub struct Rates {
    /// `faer`'s factorizations, and the CPU's CG iteration.
    pub cpu: CpuRates,
    /// Accelerate's factorizations (macOS), with the same CG iteration.
    pub accelerate: Option<CpuRates>,
    /// [`crate::sme`]'s factorizations (SME2), with the same CG iteration.
    pub sme: Option<CpuRates>,
    /// One per precision the GPU runs.
    pub gpu: Vec<GpuRates>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CpuRates {
    /// `f64` factorization: per panel step on the critical path (where the CPU's parallelism
    /// runs out), plus per GFLOP.
    pub factor_ms_per_step: f64,
    pub factor_ms_per_gflop: f64,
    pub factor32_ms_per_step: f64,
    pub factor32_ms_per_gflop: f64,
    /// A substitution, per GB of factor read.
    pub solve_ms_per_gb: f64,
    pub solve32_ms_per_gb: f64,
    /// An `f64` CG iteration apart from its preconditioner: fixed, plus per million stored
    /// entries.
    pub iteration_ms: f64,
    pub iteration_ms_per_mnnz: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRates {
    pub precision: Precision,
    /// Factorization: per panel step on the critical path, plus per GFLOP.
    pub factor_ms_per_step: f64,
    pub factor_ms_per_gflop: f64,
    /// A solve: fixed, plus per GB of factor.
    pub solve_ms: f64,
    pub solve_ms_per_gb: f64,
    /// Of a solve's time, submitting it and waiting for it: what a solve recorded inside a
    /// longer command buffer (each CG iteration's preconditioner) does not pay.
    pub submit_ms: f64,
    /// A CG iteration in this precision on the device beyond its `f32` preconditioner solve:
    /// fixed (its dispatches), plus per million stored entries (the matrix-vector product).
    pub iteration_ms: f64,
    pub iteration_ms_per_mnnz: f64,
    /// Taking a factor made on the CPU (`vulkan::VulkanCholesky::load_factor`):
    /// fixed, plus per GB of factor in this precision.
    pub load_ms: f64,
    pub load_ms_per_gb: f64,
    /// Whether 64-wide Schur blocks measured faster than 32-wide.
    pub wide_schur: bool,
}

/// Grid sides of the calibration matrices (3 dofs per node, 27-point coupling): one where fixed
/// costs dominate, two where work does, so the fits separate the two.
const SIDES: [usize; 3] = [6, 10, 16];
const N: usize = SIDES.len();
/// The largest, where the Schur block size is chosen.
const LARGE: usize = N - 1;

impl Rates {
    /// This machine's, measured on first use (under a second).
    pub fn host() -> &'static Rates {
        static RATES: OnceLock<Rates> = OnceLock::new();
        RATES.get_or_init(|| Rates::measure(Caps::host()))
    }

    pub fn measure(caps: &Caps) -> Rates {
        let problems: [Problem; N] = SIDES.map(Problem::grid);
        let cpu = cpu_rates(&problems);
        #[cfg(accelerate)]
        let accelerate = (!crate::sme::available()).then(|| accelerate_rates(&problems, &cpu));
        #[cfg(not(accelerate))]
        let accelerate = None;
        #[cfg(accelerate)]
        let sme = crate::sme::available().then(|| sme_rates(&problems, &cpu));
        #[cfg(not(accelerate))]
        let sme = None;
        #[cfg(feature = "vulkan")]
        let gpu = match (&caps.gpu, super::caps::context()) {
            (Some(g), Some(ctx)) => gpu::rates(&ctx, &g.precisions(), &problems).unwrap_or_default(),
            _ => Vec::new(),
        };
        #[cfg(not(feature = "vulkan"))]
        let gpu = {
            let _ = caps;
            Vec::new()
        };
        Rates { cpu, accelerate, sme, gpu }
    }
}

/// A synthetic tangent: the 27-point stencil on a `side³` grid with full 3 × 3 node blocks,
/// diagonally dominant.
struct Problem {
    analysis: Analysis,
    values: Vec<f64>,
    rhs: Vec<f64>,
    gflops: f64,
    steps: f64,
    factor_len: f64,
    nnz: f64,
}

impl Problem {
    fn grid(side: usize) -> Problem {
        let node = |x: usize, y: usize, z: usize| (z * side + y) * side + x;
        let nodes = side * side * side;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0], Vec::new(), Vec::<f64>::new());
        for n in 0..nodes {
            let (x, y, z) = (n % side, n / side % side, n / (side * side));
            let mut neighbours = Vec::new();
            for dz in 0..=2usize {
                for dy in 0..=2usize {
                    for dx in 0..=2usize {
                        let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                        if (1..=side).contains(&nx) && (1..=side).contains(&ny) && (1..=side).contains(&nz) {
                            let m = node(nx - 1, ny - 1, nz - 1);
                            if m >= n {
                                neighbours.push(m);
                            }
                        }
                    }
                }
            }
            neighbours.sort_unstable();
            for c in 0..3 {
                for &m in &neighbours {
                    for r in 0..3 {
                        let (i, j) = (3 * m + r, 3 * n + c);
                        if i < j {
                            continue;
                        }
                        row_idx.push(i);
                        values.push(if i == j { 30.0 } else if m == n { 0.1 } else if r == c { -1.0 } else { -0.05 });
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
        let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::NestedDissection { block: 3 }).expect("a valid pattern");
        let st = super::policy::Structure::of(&analysis);
        let rhs = (0..analysis.dim()).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
        Problem { analysis, values, rhs, gflops: st.gflops, steps: st.panel_steps as f64, factor_len: st.factor_len as f64, nnz: st.nnz as f64 }
    }
}

fn best_of(reps: usize, mut f: impl FnMut()) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn cpu_rates(problems: &[Problem; N]) -> CpuRates {
    const ITERATIONS: usize = 5;
    let mut factor = [0.0; N];
    let mut factor32 = [0.0; N];
    let mut iterate = [0.0; N];
    let (mut solve, mut solve32) = (0.0, 0.0);
    for (i, p) in problems.iter().enumerate() {
        let a = &p.analysis;
        let mut chol = CpuCholesky::<f64>::new(a);
        factor[i] = best_of(2, || chol.factorize(&p.values).expect("a positive definite matrix"));
        let mut x = p.rhs.clone();
        solve = best_of(3, || chol.solve_in_place(&mut x));
        let mut pcg = CpuPcg::with_options(a, PcgOptions { tolerance: 0.0, max_iterations: ITERATIONS });
        factor32[i] = best_of(2, || pcg.factorize(&p.values).expect("a positive definite matrix"));
        let mut x32 = p.rhs.clone();
        solve32 = best_of(3, || pcg.cholesky().solve_in_place(&mut x32));
        iterate[i] = best_of(3, || {
            x.copy_from_slice(&p.rhs);
            pcg.solve_with(&mut x, |r, z| z.copy_from_slice(r));
        }) / ITERATIONS as f64;
    }
    let steps = problems.each_ref().map(|p| p.steps);
    let gflops = problems.each_ref().map(|p| p.gflops);
    let (factor_ms_per_step, factor_ms_per_gflop) = fit(&steps, &gflops, &factor);
    let (factor32_ms_per_step, factor32_ms_per_gflop) = fit(&steps, &gflops, &factor32);
    let (iteration_ms, iteration_ms_per_mnnz) = fit(&[1.0; N], &problems.each_ref().map(|p| p.nnz * 1e-6), &iterate);
    CpuRates {
        factor_ms_per_step,
        factor_ms_per_gflop,
        factor32_ms_per_step,
        factor32_ms_per_gflop,
        solve_ms_per_gb: solve / (problems[LARGE].factor_len * 8e-9),
        solve32_ms_per_gb: solve32 / (problems[LARGE].factor_len * 4e-9),
        iteration_ms,
        iteration_ms_per_mnnz,
    }
}

/// Accelerate's factorizations and solves, fitted against this crate's analysis of each problem
/// (Accelerate orders with METIS itself, so its flops are close but not the same); the CG
/// iteration is the CPU's.
#[cfg(accelerate)]
fn accelerate_rates(problems: &[Problem; N], cpu: &CpuRates) -> CpuRates {
    use crate::accelerate::AccelerateCholesky;
    let mut factor = [0.0; N];
    let mut factor32 = [0.0; N];
    let (mut solve, mut solve32) = (0.0, 0.0);
    for (i, p) in problems.iter().enumerate() {
        let mut x = p.rhs.clone();
        let mut chol = AccelerateCholesky::<f64>::new(&p.analysis).expect("a valid pattern");
        chol.factorize(&p.values).expect("a positive definite matrix");
        factor[i] = best_of(3, || chol.factorize(&p.values).expect("a positive definite matrix"));
        solve = best_of(3, || chol.solve_in_place(&mut x).expect("a factorization"));
        let mut chol = AccelerateCholesky::<f32>::new(&p.analysis).expect("a valid pattern");
        chol.factorize(&p.values).expect("a positive definite matrix");
        factor32[i] = best_of(3, || chol.factorize(&p.values).expect("a positive definite matrix"));
        solve32 = best_of(3, || chol.solve_in_place(&mut x).expect("a factorization"));
    }
    let steps = problems.each_ref().map(|p| p.steps);
    let gflops = problems.each_ref().map(|p| p.gflops);
    let (factor_ms_per_step, factor_ms_per_gflop) = fit(&steps, &gflops, &factor);
    let (factor32_ms_per_step, factor32_ms_per_gflop) = fit(&steps, &gflops, &factor32);
    CpuRates {
        factor_ms_per_step,
        factor_ms_per_gflop,
        factor32_ms_per_step,
        factor32_ms_per_gflop,
        solve_ms_per_gb: solve / (problems[LARGE].factor_len * 8e-9),
        solve32_ms_per_gb: solve32 / (problems[LARGE].factor_len * 4e-9),
        ..*cpu
    }
}

/// [`crate::sme`]'s factorizations and solves, fitted as the others; the CG iteration is the
/// CPU's. (Inverting the diagonal blocks for a GPU is counted in [`GpuRates::load_ms`].)
#[cfg(accelerate)]
fn sme_rates(problems: &[Problem; N], cpu: &CpuRates) -> CpuRates {
    use crate::sme::SmeCholesky;
    let mut factor = [0.0; N];
    let mut factor32 = [0.0; N];
    let (mut solve, mut solve32) = (0.0, 0.0);
    for (i, p) in problems.iter().enumerate() {
        let mut x = p.rhs.clone();
        let mut chol = SmeCholesky::<f64>::new(&p.analysis).expect("SME2");
        chol.factorize(&p.values).expect("a positive definite matrix");
        factor[i] = best_of(3, || chol.factorize(&p.values).expect("a positive definite matrix"));
        solve = best_of(3, || chol.solve_in_place(&mut x));
        let mut chol = SmeCholesky::<f32>::new(&p.analysis).expect("SME2");
        chol.factorize(&p.values).expect("a positive definite matrix");
        factor32[i] = best_of(3, || chol.factorize(&p.values).expect("a positive definite matrix"));
        solve32 = best_of(3, || chol.solve_in_place(&mut x));
    }
    let steps = problems.each_ref().map(|p| p.steps);
    let gflops = problems.each_ref().map(|p| p.gflops);
    let (factor_ms_per_step, factor_ms_per_gflop) = fit(&steps, &gflops, &factor);
    let (factor32_ms_per_step, factor32_ms_per_gflop) = fit(&steps, &gflops, &factor32);
    let large = &problems[LARGE];
    CpuRates {
        factor_ms_per_step,
        factor_ms_per_gflop,
        factor32_ms_per_step,
        factor32_ms_per_gflop,
        solve_ms_per_gb: solve / (large.factor_len * 8e-9),
        solve32_ms_per_gb: solve32 / (large.factor_len * 4e-9),
        ..*cpu
    }
}

/// `t ≈ a x + b y` by least squares in relative error (so the small problems' fixed costs
/// weigh as much as the large ones' work), `a, b ≥ 0`.
fn fit(x: &[f64], y: &[f64], t: &[f64]) -> (f64, f64) {
    let w: Vec<f64> = t.iter().map(|t| 1.0 / (t * t).max(1e-12)).collect();
    let sum = |f: &dyn Fn(usize) -> f64| (0..t.len()).map(|i| w[i] * f(i)).sum::<f64>();
    let (xx, xy, yy) = (sum(&|i| x[i] * x[i]), sum(&|i| x[i] * y[i]), sum(&|i| y[i] * y[i]));
    let (xt, yt) = (sum(&|i| x[i] * t[i]), sum(&|i| y[i] * t[i]));
    let det = xx * yy - xy * xy;
    if det > 1e-12 * xx * yy {
        let (a, b) = ((xt * yy - yt * xy) / det, (yt * xx - xt * xy) / det);
        if a >= 0.0 && b >= 0.0 {
            return (a, b);
        }
    }
    let (only_x, only_y) = (xt / xx, yt / yy);
    let err = |a: f64, b: f64| sum(&|i| (a * x[i] + b * y[i] - t[i]).powi(2));
    if err(0.0, only_y) <= err(only_x, 0.0) { (0.0, only_y) } else { (only_x, 0.0) }
}

#[cfg(feature = "vulkan")]
mod gpu {
    use std::sync::Arc;

    use super::{GpuRates, LARGE, N, Problem, best_of, fit};
    use crate::cholesky::Error;
    use crate::vulkan::{Context, DeviceReal, Options, SchurTile, VulkanCholesky, VulkanPcg};
    use crate::{Ef64, Precision};

    pub(super) fn rates(ctx: &Arc<Context>, precisions: &[Precision], problems: &[Problem; N]) -> Result<Vec<GpuRates>, Error> {
        let (mut f32_rates, solve32) = precision::<f32>(ctx, problems)?;
        (f32_rates.iteration_ms, f32_rates.iteration_ms_per_mnnz) = iteration::<f32>(ctx, problems, solve32)?;
        let mut out = vec![f32_rates];
        for &p in precisions {
            let (mut rates, iteration) = match p {
                Precision::F32 => continue,
                Precision::F64 => (precision::<f64>(ctx, problems)?.0, iteration::<f64>(ctx, problems, solve32)?),
                Precision::Ef64 => (precision::<Ef64>(ctx, problems)?.0, iteration::<Ef64>(ctx, problems, solve32)?),
            };
            (rates.iteration_ms, rates.iteration_ms_per_mnnz) = iteration;
            out.push(rates);
        }
        Ok(out)
    }

    /// `T`'s rates apart from the iteration's, and the device time of a solve of each problem
    /// (the `f32` one is every iteration's preconditioner). The Schur block size is the faster
    /// on the largest problem, and the rest is measured with it.
    fn precision<T: DeviceReal>(ctx: &Arc<Context>, problems: &[Problem; N]) -> Result<(GpuRates, [f64; N]), Error> {
        let large = &problems[LARGE];
        let mut chosen: Option<(f64, bool)> = None;
        for (tile, wide) in [(SchurTile::T32, false), (SchurTile::T64, true)] {
            let mut chol = VulkanCholesky::<T>::with_options(&large.analysis, Options { context: Some(ctx.clone()), schur_tile: Some(tile), ..Default::default() })?;
            chol.factorize(&large.values)?;
            let t = best_of(3, || {
                chol.factorize(&large.values).ok();
            });
            if chosen.is_none_or(|(best, _)| t < best) {
                chosen = Some((t, wide));
            }
        }
        let wide_schur = chosen.expect("a Schur tile").1;
        let options = Options { context: Some(ctx.clone()), schur_tile: Some(if wide_schur { SchurTile::T64 } else { SchurTile::T32 }), ..Default::default() };
        let mut factor = [0.0; N];
        let mut solve = [0.0; N];
        let mut device = [f64::INFINITY; N];
        for (i, p) in problems.iter().enumerate() {
            let mut chol = VulkanCholesky::<T>::with_options(&p.analysis, options.clone())?;
            chol.factorize(&p.values)?;
            factor[i] = best_of(3, || {
                chol.factorize(&p.values).ok();
            });
            let mut x = p.rhs.clone();
            chol.solve_in_place(&mut x)?;
            solve[i] = best_of(5, || {
                x.copy_from_slice(&p.rhs);
                chol.solve_in_place(&mut x).ok();
                device[i] = device[i].min(chol.last_solve_ms());
            });
        }
        let gb = problems.each_ref().map(|p| p.factor_len * T::PRECISION.bytes() as f64 * 1e-9);
        let (factor_ms_per_step, factor_ms_per_gflop) = fit(&problems.each_ref().map(|p| p.steps), &problems.each_ref().map(|p| p.gflops), &factor);
        let (solve_ms, solve_ms_per_gb) = fit(&[1.0; N], &gb, &solve);
        let submit_ms = (0..N).map(|i| solve[i] - device[i]).fold(f64::INFINITY, f64::min).max(0.0);
        #[cfg(accelerate)]
        let (load_ms, load_ms_per_gb) = if crate::sme::available() { load::<T>(problems, options)? } else { (0.0, 0.0) };
        #[cfg(not(accelerate))]
        let (load_ms, load_ms_per_gb) = (0.0, 0.0);
        let rates = GpuRates {
            precision: T::PRECISION,
            factor_ms_per_step,
            factor_ms_per_gflop,
            solve_ms,
            solve_ms_per_gb,
            submit_ms,
            iteration_ms: 0.0,
            iteration_ms_per_mnnz: 0.0,
            load_ms,
            load_ms_per_gb,
            wide_schur,
        };
        Ok((rates, device))
    }

    /// Handing an SME factor of each problem to a `T` solver: inverting its diagonal blocks
    /// during the factorization (the time that adds) and loading factor and inverses, fitted as
    /// fixed plus per GB of factor.
    #[cfg(accelerate)]
    fn load<T: DeviceReal>(problems: &[Problem; N], options: Options) -> Result<(f64, f64), Error> {
        let mut load = [0.0; N];
        for (i, p) in problems.iter().enumerate() {
            let mut chol = VulkanCholesky::<T>::with_options(&p.analysis, options.clone())?;
            let mut plain = crate::sme::SmeCholesky::<f64>::new(&p.analysis)?;
            let mut sme = crate::sme::SmeCholesky::<f64>::new(&p.analysis)?.with_inverses();
            sme.factorize(&p.values)?;
            plain.factorize(&p.values)?;
            let without = best_of(3, || {
                plain.factorize(&p.values).ok();
            });
            let mut with = best_of(3, || {
                sme.factorize(&p.values).ok();
            });
            with = with.max(without);
            let inverse = sme.inverse().expect("inverses");
            chol.load_factor_and_inverse(sme.factor(), inverse)?;
            load[i] = with - without
                + best_of(3, || {
                    chol.load_factor_and_inverse(sme.factor(), inverse).ok();
                });
        }
        let gb = problems.each_ref().map(|p| p.factor_len * T::PRECISION.bytes() as f64 * 1e-9);
        Ok(fit(&[1.0; N], &gb, &load))
    }

    /// A `T` CG iteration beyond its preconditioner solve, fixed and per million stored
    /// entries: the device time of a recorded batch less the `f32` solve's device time
    /// `solve32`, fitted over the problems.
    fn iteration<T: DeviceReal>(ctx: &Arc<Context>, problems: &[Problem; N], solve32: [f64; N]) -> Result<(f64, f64), Error> {
        let mut extra = [0.0; N];
        for (i, p) in problems.iter().enumerate() {
            let mut pcg = VulkanPcg::<T>::with_options(&p.analysis, Options { context: Some(ctx.clone()), ..Default::default() })?;
            pcg.factorize(&p.values)?;
            let mut x = p.rhs.clone();
            pcg.solve_to(&mut x, 0.0, 4)?;
            let per_iteration = (0..3).map(|_| pcg.iteration_ms()).collect::<Result<Vec<_>, _>>()?.into_iter().fold(f64::INFINITY, f64::min);
            extra[i] = (per_iteration - solve32[i]).max(0.0);
        }
        Ok(fit(&[1.0; N], &problems.each_ref().map(|p| p.nnz * 1e-6), &extra))
    }
}

#[cfg(test)]
mod tests {
    use super::fit;

    #[test]
    fn fit_recovers_a_line_and_stays_non_negative() {
        let (a, b) = fit(&[4.0, 10.0, 40.0], &[0.02, 1.0, 9.0], &[0.28 + 0.04, 0.7 + 2.0, 2.8 + 18.0]);
        assert!((a - 0.07).abs() < 1e-9 && (b - 2.0).abs() < 1e-9, "{a} {b}");
        let (a, b) = fit(&[4.0, 10.0, 40.0], &[0.02, 1.0, 9.0], &[5.0, 5.0, 5.5]);
        assert!(a >= 0.0 && b >= 0.0);
    }
}
