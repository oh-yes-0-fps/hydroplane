//! [`Solver`], the default entry point: give it a pattern, then values and right-hand sides; it
//! decides where and how to solve.
//!
//! The decision has four inputs, and any of its outputs can be fixed by the caller instead:
//! - what the machine offers ([`Caps`]: a GPU, whether it has `f64`), detected once per process;
//! - how fast its devices are ([`Rates`]), measured once per process on two synthetic tangents,
//!   and only when there is a real choice (not for a tiny problem, a CPU-only machine, or a
//!   pinned strategy);
//! - the pattern's numbers ([`Structure`]: flops, the GPU factorization's critical path, factor
//!   size), from its symbolic analysis, with the dofs per node read from the pattern itself;
//! - how this matrix turns rounding into residual ([`Accuracy`]), starting from a prior and
//!   corrected by the true residual of the first solve: a strategy that misses the tolerance is
//!   replaced at once, and one predicted to be clearly cheaper is switched to when the
//!   expected savings repay the rebuild.
//!
//! [`Solver::plan`] says what was chosen and why. The concrete solvers ([`crate::cholesky`],
//! `vulkan`) remain available for callers that want no decisions made at all.

mod caps;
mod policy;
mod rates;

use std::time::Instant;

pub use caps::{Caps, GpuCaps};
pub use policy::{Accuracy, Device, Estimate, Library, Method, Plan, Reason, Strategy, Structure, Verdict};
pub use rates::{CpuRates, GpuRates, Rates};

use crate::Precision;
#[cfg(accelerate)]
use crate::accelerate::AccelerateCholesky;
use crate::cholesky::{Analysis, CpuCholesky, CpuPcg, Error, Ordering, PcgOptions};
use crate::sparse::SellMatrix;
use crate::iterative::Cg;
use policy::{Needs, Restrict};

/// When a solve's result is checked against the true residual `‖b − A x‖ / ‖b‖` (one `f64`
/// matrix-vector product on the CPU). A check that misses the tolerance moves the solver to a
/// strategy predicted to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verify {
    /// The first solve after each (re)build of the strategy.
    First,
    /// Those, and every `n`th solve: how close a direct solve comes to its floor varies with the
    /// right-hand side (30-fold on a welded slab's tangent), so one check is not the last word.
    Every(usize),
    Always,
    Never,
}

impl Default for Verify {
    fn default() -> Self {
        Verify::Every(16)
    }
}

/// How much work goes into choosing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Planning {
    /// The measured rates and the cost model.
    Estimate,
    /// Those, then the candidates predicted within 30 % of the cheapest timed on this pattern
    /// (with stand-in values): a factorization and a solve or an iteration's worth each, worth
    /// it when many factorizations will follow.
    Measure,
}

/// What the caller needs, hints at, and insists on. The defaults decide everything.
#[derive(Clone, Debug)]
pub struct Options {
    /// The relative residual `‖b − A x‖ / ‖b‖` a solve must reach.
    pub tolerance: f64,
    /// How many solves follow each factorization, on average: many favour a direct factor,
    /// one favours iterating on a cheap one.
    pub solves_per_factorization: f64,
    /// How many factorizations of this pattern to expect: how much a better ordering or a
    /// cheaper strategy found later is worth.
    pub factorizations: usize,
    pub verify: Verify,
    /// `None` measures when at least 8 factorizations are expected, else estimates.
    pub planning: Option<Planning>,
    /// Dofs per mesh node, which nested dissection keeps together. `None` reads it from the
    /// pattern.
    pub block: Option<usize>,
    pub device: Option<Device>,
    pub method: Option<Method>,
    pub precision: Option<Precision>,
    pub library: Option<Library>,
    pub ordering: Option<Ordering>,
    /// Nested-dissection seeds to try. `None` tries more the more factorizations there are to
    /// amortize them over.
    pub ordering_candidates: Option<usize>,
    #[cfg(feature = "vulkan")]
    pub schur_tile: Option<crate::vulkan::SchurTile>,
    /// Rates to plan with instead of measuring this machine's.
    pub rates: Option<Rates>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            solves_per_factorization: 1.0,
            factorizations: 1,
            verify: Verify::default(),
            planning: None,
            block: None,
            device: None,
            method: None,
            precision: None,
            library: None,
            ordering: None,
            ordering_candidates: None,
            #[cfg(feature = "vulkan")]
            schur_tile: None,
            rates: None,
        }
    }
}

impl Options {
    fn restrict(&self) -> Restrict {
        Restrict { device: self.device, method: self.method, precision: self.precision, library: self.library }
    }

    fn needs(&self) -> Needs {
        Needs { tolerance: self.tolerance, solves_per_factorization: self.solves_per_factorization }
    }
}

/// How a solve went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport {
    pub strategy: Strategy,
    pub iterations: usize,
    /// The true relative residual when verified, else the iteration's own estimate (`None`
    /// for an unverified direct solve).
    pub residual: Option<f64>,
    pub verified: bool,
    /// Whether the residual is known to meet the tolerance.
    pub meets_tolerance: Option<bool>,
}

/// A sparse symmetric positive definite solver that picks its own device, method, precision,
/// ordering and kernel tuning.
pub struct Solver {
    analysis: Analysis,
    csr: SellMatrix,
    structure: Structure,
    options: Options,
    rates: Option<Rates>,
    accuracy: Accuracy,
    plan: Plan,
    backend: Backend,
    build_ms: f64,
    failed: Vec<Strategy>,
    values: Vec<f64>,
    factorizations: usize,
    solves: usize,
    verify_next: bool,
    /// The values changed since the last factorization ([`Solver::update_values`]).
    stale: bool,
    /// Iterations a solve on a stale factor has been taking (moving average).
    stale_iterations: f64,
}

impl Solver {
    /// For the matrix whose lower triangle has this pattern (compressed sparse columns), with
    /// every decision left to the solver.
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::with_options(col_ptr, row_idx, Options::default())
    }

    pub fn with_options(col_ptr: &[usize], row_idx: &[usize], options: Options) -> Result<Self, Error> {
        let caps = Caps::host();
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let block = options.block.unwrap_or_else(|| detect_block(n, col_ptr, row_idx));
        let ordering = options.ordering.clone().unwrap_or(Ordering::NestedDissection { block });
        let started = Instant::now();
        let mut analysis = Analysis::with_ordering(col_ptr, row_idx, ordering)?;
        let analysis_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut structure = Structure::of(&analysis);
        let restrict = options.restrict();
        let accuracy = Accuracy::default();
        let rates = match &options.rates {
            Some(r) => Some(r.clone()),
            None if caps.gpu.is_none() || restrict.device == Some(Device::Cpu) || structure.tiny() || restrict.pinned() => None,
            None => Some(Rates::host().clone()),
        };
        let mut plan = policy::choose(caps, rates.as_ref(), &structure, &options.needs(), &accuracy, &restrict, &[]);
        if options.ordering.is_none() {
            let candidates = options.ordering_candidates.unwrap_or_else(|| match plan.estimate {
                // About 4 % of the factorization per extra seed, against an analysis each.
                Some(e) => (1.0 + 0.04 * e.factor_ms * options.factorizations as f64 / analysis_ms).clamp(1.0, 8.0) as usize,
                None => 1,
            });
            let strategy = plan.strategy;
            let (rates_ref, acc, tol) = (rates.as_ref(), accuracy, options.tolerance);
            let cost = |st: &Structure| rates_ref.and_then(|r| policy::estimate(r, st, &strategy, &acc, tol)).map_or(st.gflops, |e| e.factor_ms);
            if candidates > 1 && n.is_multiple_of(block) {
                analysis = Analysis::best_of(col_ptr, row_idx, block, candidates, |a| cost(&Structure::of(a)))?;
                structure = Structure::of(&analysis);
            }
            // Nested dissection wins on solids, minimum degree on shells: keep whichever the
            // model prefers.
            let amd = Analysis::with_ordering(col_ptr, row_idx, Ordering::Amd)?;
            let amd_structure = Structure::of(&amd);
            if cost(&amd_structure) < cost(&structure) {
                analysis = amd;
                structure = amd_structure;
            }
            plan = policy::choose(caps, rates.as_ref(), &structure, &options.needs(), &accuracy, &restrict, &[]);
        }
        let csr = SellMatrix::from_lower_csc_symmetric(n, analysis.col_ptr(), analysis.row_idx());
        let measure = options.planning.unwrap_or(if options.factorizations >= 8 { Planning::Measure } else { Planning::Estimate }) == Planning::Measure;
        let measured = if measure && !restrict.pinned() { measure_close(&mut plan, &analysis, &options, rates.as_ref())? } else { None };
        let started = Instant::now();
        let backend = match measured {
            Some(b) => b,
            None => Backend::build(&plan.strategy, &analysis, &options, rates.as_ref())?,
        };
        let build_ms = started.elapsed().as_secs_f64() * 1e3;
        let verify_next = options.verify != Verify::Never;
        Ok(Self { analysis, csr, structure, options, rates, accuracy, plan, backend, build_ms, failed: Vec::new(), values: Vec::new(), factorizations: 0, solves: 0, verify_next, stale: false, stale_iterations: STALE_ITERATIONS })
    }

    /// What is being used and why.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    /// Factors the matrix whose lower-triangle values, in the pattern's order, are `values`.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        assert_eq!(values.len(), self.analysis.row_idx().len());
        self.values.clear();
        self.values.extend_from_slice(values);
        self.csr.set_values(values);
        self.switch_if_cheaper()?;
        loop {
            match self.backend.factorize(&self.values) {
                Ok(()) => break,
                Err(Error::NotPositiveDefinite) if self.plan.strategy.precision != Precision::F64 && !self.options.restrict().pinned() => {
                    self.failed.push(self.plan.strategy);
                    if !self.replan()? {
                        return Err(Error::NotPositiveDefinite);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        self.factorizations += 1;
        self.stale = false;
        Ok(())
    }

    /// Replaces the matrix's values, refactoring only if that is predicted cheaper than
    /// iterating: later solves are of the new matrix either way, to the tolerance. Kept, the last
    /// factor preconditions conjugate gradient on the new values, as in modified Newton but with
    /// the exact Newton step for a result; a tangent a few iterations on typically takes 5 to 10
    /// iterations (5 at a 5 % change of each element's stiffness, 8 at 20 %). Where the factor is
    /// too poor (the iteration stalls or meets negative curvature), the solve factors the new
    /// values and starts over.
    pub fn update_values(&mut self, values: &[f64]) -> Result<(), Error> {
        assert_eq!(values.len(), self.analysis.row_idx().len());
        if self.factorizations == 0 || self.refactor_is_cheaper() {
            return self.factorize(values);
        }
        self.values.clear();
        self.values.extend_from_slice(values);
        self.csr.set_values(values);
        self.backend.set_values(&self.values)?;
        self.stale = true;
        Ok(())
    }

    /// `rhs ← A⁻¹ rhs` with the last factorization. When this solve is verified and misses the
    /// tolerance, the solver moves to a strategy predicted to reach it, refactors and solves
    /// again, until one does or none is left.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<SolveReport, Error> {
        assert_eq!(rhs.len(), self.structure.n);
        if self.factorizations == 0 {
            return Err(Error::NotFactorized);
        }
        let tolerance = self.options.tolerance;
        self.solves += 1;
        let verify = self.verify_next
            || match self.options.verify {
                Verify::Always => true,
                Verify::Every(n) => n > 0 && self.solves.is_multiple_of(n),
                Verify::First | Verify::Never => false,
            };
        let b = verify.then(|| rhs.to_vec());
        loop {
            let max_iterations = self.accuracy.iterations(tolerance).map_or(200, |i| (3 * i + 10).min(500));
            let (iterations, tracked) = if self.stale && self.plan.strategy.method == Method::Direct {
                let (backend, csr) = (&mut self.backend, &self.csr);
                let mut failure = None;
                let mut a = |x: &[f64], y: &mut [f64]| csr.spmv(x, y);
                let mut m = |r: &[f64], z: &mut [f64]| {
                    if let Err(e) = backend.precondition(r, z) {
                        failure.get_or_insert(e);
                    }
                };
                let rhs_copy = rhs.to_vec();
                let options = crate::iterative::Options { tolerance, max_iterations: 50, ..Default::default() };
                let report = Cg::with_options(options).solve(&mut a, &mut m, &rhs_copy[..], rhs);
                if let Some(e) = failure {
                    return Err(e);
                }
                if report.converged() {
                    self.stale_iterations = 0.7 * self.stale_iterations + 0.3 * report.iterations as f64;
                    (report.iterations, Some(report.relative_residual))
                } else {
                    // The old factor no longer preconditions the matrix well enough: factor it.
                    self.stale_iterations = 0.7 * self.stale_iterations + 0.3 * options.max_iterations as f64;
                    self.backend.factorize(&self.values)?;
                    self.stale = false;
                    rhs.copy_from_slice(&rhs_copy);
                    let (iterations, tracked) = self.backend.solve(rhs, tolerance, max_iterations)?;
                    (report.iterations + iterations, tracked)
                }
            } else {
                self.backend.solve(rhs, tolerance, max_iterations)?
            };
            let strategy = self.plan.strategy;
            let Some(b) = &b else {
                return Ok(SolveReport { strategy, iterations, residual: tracked, verified: false, meets_tolerance: tracked.map(|r| r <= tolerance) });
            };
            let residual = self.csr.relative_residual(b, rhs);
            self.learn(&strategy, residual, iterations, tracked);
            if residual <= tolerance || self.options.restrict().pinned() {
                self.verify_next = false;
                return Ok(SolveReport { strategy, iterations, residual: Some(residual), verified: true, meets_tolerance: Some(residual <= tolerance) });
            }
            self.failed.push(strategy);
            if !self.replan()? {
                return Ok(SolveReport { strategy, iterations, residual: Some(residual), verified: true, meets_tolerance: Some(false) });
            }
            self.backend.factorize(&self.values)?;
            self.stale = false;
            rhs.copy_from_slice(b);
        }
    }

    /// Whether factoring and solving once costs less than the iterations a stale factor has been
    /// taking (by the measured rates; without them, iterating is assumed cheaper).
    fn refactor_is_cheaper(&self) -> bool {
        let s = self.plan.strategy;
        let Some(rates) = &self.rates else { return false };
        if s.method != Method::Direct {
            return false;
        }
        let Some(e) = policy::estimate(rates, &self.structure, &s, &self.accuracy, self.options.tolerance) else { return false };
        let product = rates.cpu.iteration_ms + rates.cpu.iteration_ms_per_mnnz * self.structure.nnz as f64 * 1e-6;
        e.factor_ms + e.solve_ms < self.stale_iterations * (e.solve_ms + product)
    }

    /// Folds a verified solve into [`Accuracy`]. A higher amplification is taken at once; a
    /// lower one by at most [`TRUST`] per observation, since a single right-hand side can land
    /// well below the floor others reach.
    fn learn(&mut self, s: &Strategy, residual: f64, iterations: usize, tracked: Option<f64>) {
        let eps = s.precision.epsilon();
        let mut observe = |amplification: f64| {
            let current = self.accuracy.amplification;
            self.accuracy.amplification = if amplification >= current { amplification } else { amplification.max(current / TRUST) }.max(1.0);
        };
        match s.method {
            Method::Direct if !self.stale => observe(residual / eps),
            Method::Direct => {}
            Method::Pcg | Method::SplitPcg => {
                if residual > self.options.tolerance {
                    observe(residual / (0.5 * eps));
                }
                if let (Some(t), true) = (tracked, iterations > 0) {
                    self.accuracy.pcg_rate = Some(t.max(1e-300).powf(1.0 / iterations as f64));
                }
            }
        }
        self.accuracy.learned = true;
    }

    /// Plans again with what has been learned; rebuilds if the strategy changed. `false` if
    /// nothing new is left to try, or nothing left is predicted to reach the tolerance (trying
    /// each in turn would only repeat the miss at the price of a factorization each).
    fn replan(&mut self) -> Result<bool, Error> {
        let plan = policy::choose(Caps::host(), self.rates.as_ref(), &self.structure, &self.options.needs(), &self.accuracy, &self.options.restrict(), &self.failed);
        if self.failed.contains(&plan.strategy) || plan.reason == Reason::MostAccurate {
            self.plan = plan;
            return Ok(false);
        }
        let changed = plan.strategy != self.plan.strategy;
        self.plan = plan;
        if changed {
            self.rebuild()?;
        }
        Ok(changed)
    }

    fn rebuild(&mut self) -> Result<(), Error> {
        let started = Instant::now();
        self.backend = Backend::build(&self.plan.strategy, &self.analysis, &self.options, self.rates.as_ref())?;
        self.build_ms = started.elapsed().as_secs_f64() * 1e3;
        self.verify_next = self.options.verify != Verify::Never;
        Ok(())
    }

    /// Once the matrix's accuracy is known, a strategy predicted cheaper per factorization
    /// replaces the current one if the factorizations still expected repay the rebuild. Not
    /// when the current one was chosen by timing: the measurement outranks the model.
    fn switch_if_cheaper(&mut self) -> Result<(), Error> {
        if !self.accuracy.learned || self.plan.accuracy == self.accuracy || self.options.restrict().pinned() || self.plan.reason == Reason::Measured {
            return Ok(());
        }
        let plan = policy::choose(Caps::host(), self.rates.as_ref(), &self.structure, &self.options.needs(), &self.accuracy, &self.options.restrict(), &self.failed);
        let solves = self.options.solves_per_factorization;
        let current = self.rates.as_ref().and_then(|r| policy::estimate(r, &self.structure, &self.plan.strategy, &self.accuracy, self.options.tolerance));
        let remaining = self.options.factorizations.saturating_sub(self.factorizations) as f64;
        let worth = match (plan.estimate, current) {
            (Some(new), Some(old)) => plan.strategy != self.plan.strategy && (old.cycle_ms(solves) - new.cycle_ms(solves)) * remaining > 2.0 * self.build_ms,
            _ => false,
        };
        if worth {
            self.plan = plan;
            self.rebuild()
        } else {
            self.plan.accuracy = self.accuracy;
            Ok(())
        }
    }
}

/// Times the feasible candidates predicted within 30 % of the cheapest (at most three) on this
/// pattern, with stand-in values (diagonally dominant, so positive definite): a factorization
/// and a direct solve, or as many iterations as predicted. Keeps the fastest in `plan` and
/// returns its backend, ready to factor; `None` if there was nothing to decide.
fn measure_close(plan: &mut Plan, analysis: &Analysis, options: &Options, rates: Option<&Rates>) -> Result<Option<Backend>, Error> {
    let solves = options.solves_per_factorization;
    let mut close: Vec<(Strategy, Estimate)> = plan.candidates.iter().filter_map(|(s, v)| if let Verdict::Feasible(e) = v { Some((*s, *e)) } else { None }).filter(|(_, e)| !e.factor_ms.is_nan()).collect();
    close.sort_by(|a, b| a.1.cycle_ms(solves).total_cmp(&b.1.cycle_ms(solves)));
    let Some(best) = close.first().map(|c| c.1.cycle_ms(solves)) else { return Ok(None) };
    close.retain(|c| c.1.cycle_ms(solves) <= 1.3 * best);
    close.truncate(3);
    if close.len() < 2 {
        return Ok(None);
    }
    let (col_ptr, row_idx) = (analysis.col_ptr(), analysis.row_idx());
    let n = analysis.dim();
    let mut off = vec![0.0f64; n];
    for j in 0..n {
        for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            if i != j {
                off[i] += 1.0;
                off[j] += 1.0;
            }
        }
    }
    let values: Vec<f64> = (0..n).flat_map(|j| (col_ptr[j]..col_ptr[j + 1]).map(move |p| (j, p))).map(|(j, p)| if row_idx[p] == j { off[j] + 1.0 } else { -1.0 }).collect();
    let rhs: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let mut fastest: Option<(f64, Strategy, Backend)> = None;
    for (s, e) in close {
        let Ok(mut backend) = Backend::build(&s, analysis, options, rates) else { continue };
        if backend.factorize(&values).is_err() {
            continue;
        }
        let t = Instant::now();
        backend.factorize(&values)?;
        let factor = t.elapsed().as_secs_f64() * 1e3;
        let mut x = rhs.clone();
        let t = Instant::now();
        backend.solve(&mut x, 0.0, e.iterations.max(1))?;
        let solve = t.elapsed().as_secs_f64() * 1e3;
        let cycle = factor + solves * solve;
        if fastest.as_ref().is_none_or(|f| cycle < f.0) {
            fastest = Some((cycle, s, backend));
        }
    }
    let Some((_, strategy, backend)) = fastest else { return Ok(None) };
    plan.strategy = strategy;
    plan.estimate = plan.candidates.iter().find(|(s, _)| *s == strategy).and_then(|(_, v)| if let Verdict::Feasible(e) = v { Some(*e) } else { None });
    plan.reason = Reason::Measured;
    Ok(Some(backend))
}

/// How far one verified solve may lower the learned error amplification.
const TRUST: f64 = 30.0;

/// Iterations a solve on a stale factor is expected to take before any has been seen.
const STALE_ITERATIONS: f64 = 6.0;

/// The dofs per node of a pattern built from node blocks: the largest of 6, 3, 2 whose blocks
/// of columns share their rows outside the diagonal block, else 1.
fn detect_block(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> usize {
    let nodes_below = |j: usize, b: usize| {
        let own = j / b;
        let mut v: Vec<usize> = row_idx[col_ptr[j]..col_ptr[j + 1]].iter().map(|&i| i / b).filter(|&m| m != own).collect();
        v.dedup();
        v
    };
    [6, 3, 2]
        .into_iter()
        .find(|&b| n.is_multiple_of(b) && n >= b && (0..n / b).all(|node| (1..b).all(|c| nodes_below(node * b + c, b) == nodes_below(node * b, b))))
        .unwrap_or(1)
}

enum Backend {
    CpuDirect(CpuCholesky<f64>),
    CpuDirect32(CpuCholesky<f32>),
    CpuPcg(CpuPcg),
    #[cfg(accelerate)]
    AccelerateDirect(AccelerateCholesky<f64>),
    #[cfg(accelerate)]
    AccelerateDirect32(AccelerateCholesky<f32>),
    /// An `f32` Accelerate factor preconditioning the CPU's `f64` iteration.
    #[cfg(accelerate)]
    AcceleratePcg(AccelerateCholesky<f32>, CpuPcg),
    #[cfg(accelerate)]
    SmeDirect(crate::sme::SmeCholesky<f64>),
    #[cfg(accelerate)]
    SmeDirect32(crate::sme::SmeCholesky<f32>),
    /// An `f32` SME factor preconditioning the CPU's `f64` iteration.
    #[cfg(accelerate)]
    SmePcg(Box<crate::sme::SmeCholesky<f32>>, CpuPcg),
    #[cfg(feature = "vulkan")]
    Gpu(Box<gpu::Backend>),
}

impl Backend {
    fn build(s: &Strategy, analysis: &Analysis, options: &Options, rates: Option<&Rates>) -> Result<Self, Error> {
        let single = s.precision == Precision::F32;
        Ok(match (s.method, s.device) {
            #[cfg(accelerate)]
            (Method::Direct, Device::Cpu) if s.library == Library::Sme && single => Backend::SmeDirect32(crate::sme::SmeCholesky::new(analysis)?),
            #[cfg(accelerate)]
            (Method::Direct, Device::Cpu) if s.library == Library::Accelerate && single => Backend::AccelerateDirect32(AccelerateCholesky::new(analysis)?),
            (Method::Direct, Device::Cpu) if s.library == Library::Faer && single => Backend::CpuDirect32(CpuCholesky::new(analysis)),
            #[cfg(accelerate)]
            (Method::Direct, Device::Cpu) if s.library == Library::Sme => Backend::SmeDirect(crate::sme::SmeCholesky::new(analysis)?),
            #[cfg(accelerate)]
            (Method::Pcg, Device::Cpu) if s.library == Library::Sme => Backend::SmePcg(Box::new(crate::sme::SmeCholesky::new(analysis)?), CpuPcg::new(analysis)),
            #[cfg(accelerate)]
            (Method::Direct, Device::Cpu) if s.library == Library::Accelerate => Backend::AccelerateDirect(AccelerateCholesky::new(analysis)?),
            #[cfg(accelerate)]
            (Method::Pcg, Device::Cpu) if s.library == Library::Accelerate => Backend::AcceleratePcg(AccelerateCholesky::new(analysis)?, CpuPcg::new(analysis)),
            (Method::Direct, Device::Cpu) => Backend::CpuDirect(CpuCholesky::new(analysis)),
            (Method::Pcg, Device::Cpu) => Backend::CpuPcg(CpuPcg::new(analysis)),
            #[cfg(feature = "vulkan")]
            (_, Device::Gpu) => Backend::Gpu(Box::new(gpu::Backend::build(s, analysis, options, rates)?)),
            #[cfg(not(feature = "vulkan"))]
            (_, Device::Gpu) => {
                let _ = (options, rates);
                return Err(Error::Device("built without the `vulkan` feature".into()));
            }
            (Method::SplitPcg, Device::Cpu) => unreachable!("not a strategy"),
        })
    }

    fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        match self {
            Backend::CpuDirect(c) => c.factorize(values),
            Backend::CpuDirect32(c) => c.factorize(values),
            Backend::CpuPcg(p) => p.factorize(values),
            #[cfg(accelerate)]
            Backend::SmeDirect32(c) => c.factorize(values),
            #[cfg(accelerate)]
            Backend::AccelerateDirect32(c) => c.factorize(values),
            #[cfg(accelerate)]
            Backend::SmeDirect(c) => c.factorize(values),
            #[cfg(accelerate)]
            Backend::SmePcg(c, p) => {
                p.set_values(values);
                c.factorize(values)
            }
            #[cfg(accelerate)]
            Backend::AccelerateDirect(c) => c.factorize(values),
            #[cfg(accelerate)]
            Backend::AcceleratePcg(c, p) => {
                p.set_values(values);
                c.factorize(values)
            }
            #[cfg(feature = "vulkan")]
            Backend::Gpu(g) => g.factorize(values),
        }
    }

    /// The matrix the iteration solves with, without refactoring.
    fn set_values(&mut self, values: &[f64]) -> Result<(), Error> {
        match self {
            Backend::CpuPcg(p) => p.set_values(values),
            #[cfg(accelerate)]
            Backend::AcceleratePcg(_, p) => p.set_values(values),
            #[cfg(accelerate)]
            Backend::SmePcg(_, p) => p.set_values(values),
            #[cfg(feature = "vulkan")]
            Backend::Gpu(g) => return g.set_values(values),
            _ => {}
        }
        Ok(())
    }

    /// `z = M⁻¹ r` with the last factor (of a direct strategy).
    fn precondition(&mut self, r: &[f64], z: &mut [f64]) -> Result<(), Error> {
        z.copy_from_slice(r);
        match self {
            Backend::CpuDirect(c) => {
                c.solve_in_place(z);
                Ok(())
            }
            Backend::CpuDirect32(c) => {
                c.solve_in_place(z);
                Ok(())
            }
            #[cfg(accelerate)]
            Backend::AccelerateDirect32(c) => c.solve_in_place(z),
            #[cfg(accelerate)]
            Backend::SmeDirect32(c) => {
                c.solve_in_place(z);
                Ok(())
            }
            #[cfg(accelerate)]
            Backend::AccelerateDirect(c) => c.solve_in_place(z),
            #[cfg(accelerate)]
            Backend::SmeDirect(c) => {
                c.solve_in_place(z);
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            Backend::Gpu(g) => g.precondition(z),
            _ => unreachable!("only direct strategies precondition with their factor"),
        }
    }

    /// Iterations and the iteration's own residual estimate.
    fn solve(&mut self, rhs: &mut [f64], tolerance: f64, max_iterations: usize) -> Result<(usize, Option<f64>), Error> {
        match self {
            Backend::CpuDirect(c) => {
                c.solve_in_place(rhs);
                Ok((0, None))
            }
            Backend::CpuDirect32(c) => {
                c.solve_in_place(rhs);
                Ok((0, None))
            }
            #[cfg(accelerate)]
            Backend::AccelerateDirect32(c) => c.solve_in_place(rhs).map(|_| (0, None)),
            #[cfg(accelerate)]
            Backend::SmeDirect32(c) => {
                c.solve_in_place(rhs);
                Ok((0, None))
            }
            Backend::CpuPcg(p) => {
                let r = p.solve_own(rhs, PcgOptions { tolerance, max_iterations });
                Ok((r.iterations, Some(r.relative_residual)))
            }
            #[cfg(accelerate)]
            Backend::SmeDirect(c) => {
                c.solve_in_place(rhs);
                Ok((0, None))
            }
            #[cfg(accelerate)]
            Backend::SmePcg(c, p) => {
                let r = p.solve_to(rhs, PcgOptions { tolerance, max_iterations }, |r, z| {
                    z.copy_from_slice(r);
                    c.solve_in_place(z);
                });
                Ok((r.iterations, Some(r.relative_residual)))
            }
            #[cfg(accelerate)]
            Backend::AccelerateDirect(c) => c.solve_in_place(rhs).map(|_| (0, None)),
            #[cfg(accelerate)]
            Backend::AcceleratePcg(c, p) => {
                let mut failure = None;
                let r = p.solve_to(rhs, PcgOptions { tolerance, max_iterations }, |r, z| {
                    z.copy_from_slice(r);
                    if let Err(e) = c.solve_in_place(z) {
                        failure.get_or_insert(e);
                    }
                });
                failure.map_or(Ok((r.iterations, Some(r.relative_residual))), Err)
            }
            #[cfg(feature = "vulkan")]
            Backend::Gpu(g) => g.solve(rhs, tolerance, max_iterations),
        }
    }
}

#[cfg(feature = "vulkan")]
mod gpu {
    use super::{Options, Rates, Strategy};
    use crate::cholesky::{Analysis, CpuPcg, Error, PcgOptions};
    #[cfg(accelerate)]
    use crate::sme::SmeCholesky;
    use crate::vulkan::{self, DeviceReal, SchurTile, VulkanCholesky, VulkanPcg};
    use crate::{Ef64, Precision};

    pub(super) enum Backend {
        Direct32(VulkanCholesky<f32>),
        Direct64(VulkanCholesky<f64>),
        DirectEf64(VulkanCholesky<Ef64>),
        Pcg32(VulkanPcg<f32>),
        Pcg64(VulkanPcg<f64>),
        PcgEf64(VulkanPcg<Ef64>),
        Split(VulkanCholesky<f32>, CpuPcg),
        /// A factor made on the CPU's SME units, loaded into the GPU: for a direct solve, an
        /// iteration's preconditioner, or the GPU half of a split iteration.
        #[cfg(accelerate)]
        SmeDirect64(SmeCholesky<f64>, VulkanCholesky<f64>),
        #[cfg(accelerate)]
        SmeDirectEf64(SmeCholesky<f64>, VulkanCholesky<Ef64>),
        #[cfg(accelerate)]
        SmePcg64(SmeCholesky<f32>, VulkanPcg<f64>),
        #[cfg(accelerate)]
        SmePcgEf64(SmeCholesky<f32>, VulkanPcg<Ef64>),
        #[cfg(accelerate)]
        SmeSplit(SmeCholesky<f32>, VulkanCholesky<f32>, CpuPcg),
    }

    fn direct<T: DeviceReal>(analysis: &Analysis, options: &Options, rates: Option<&Rates>) -> Result<VulkanCholesky<T>, Error> {
        let measured = rates.and_then(|r| r.gpu.iter().find(|g| g.precision == T::PRECISION)).map(|g| if g.wide_schur { SchurTile::T64 } else { SchurTile::T32 });
        VulkanCholesky::with_options(analysis, vulkan::Options { schur_tile: options.schur_tile.or(measured), ..Default::default() })
    }

    fn pcg<T: DeviceReal>(analysis: &Analysis) -> Result<VulkanPcg<T>, Error> {
        VulkanPcg::new(analysis)
    }

    impl Backend {
        pub(super) fn build(s: &Strategy, analysis: &Analysis, options: &Options, rates: Option<&Rates>) -> Result<Self, Error> {
            use super::Method::*;
            #[cfg(accelerate)]
            if s.library == super::Library::Sme {
                let sme64 = || SmeCholesky::<f64>::new(analysis).map(SmeCholesky::with_inverses);
                let sme32 = || SmeCholesky::<f32>::new(analysis).map(SmeCholesky::with_inverses);
                return Ok(match (s.method, s.precision) {
                    (Direct, Precision::F64) => Backend::SmeDirect64(sme64()?, direct(analysis, options, rates)?),
                    (Direct, _) => Backend::SmeDirectEf64(sme64()?, direct(analysis, options, rates)?),
                    (Pcg, Precision::F64) => Backend::SmePcg64(sme32()?, pcg(analysis)?),
                    (Pcg, _) => Backend::SmePcgEf64(sme32()?, pcg(analysis)?),
                    (SplitPcg, _) => Backend::SmeSplit(sme32()?, direct(analysis, options, rates)?, CpuPcg::new(analysis)),
                });
            }
            Ok(match (s.method, s.precision) {
                (Direct, Precision::F32) => Backend::Direct32(direct(analysis, options, rates)?),
                (Direct, Precision::F64) => Backend::Direct64(direct(analysis, options, rates)?),
                (Direct, Precision::Ef64) => Backend::DirectEf64(direct(analysis, options, rates)?),
                (Pcg, Precision::F32) => Backend::Pcg32(pcg(analysis)?),
                (Pcg, Precision::F64) => Backend::Pcg64(pcg(analysis)?),
                (Pcg, Precision::Ef64) => Backend::PcgEf64(pcg(analysis)?),
                (SplitPcg, _) => Backend::Split(direct(analysis, options, rates)?, CpuPcg::new(analysis)),
            })
        }

        pub(super) fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
            match self {
                Backend::Direct32(c) => c.factorize(values).map(drop),
                Backend::Direct64(c) => c.factorize(values).map(drop),
                Backend::DirectEf64(c) => c.factorize(values).map(drop),
                Backend::Pcg32(p) => p.factorize(values).map(drop),
                Backend::Pcg64(p) => p.factorize(values).map(drop),
                Backend::PcgEf64(p) => p.factorize(values).map(drop),
                Backend::Split(c, p) => {
                    p.set_values(values);
                    c.factorize(values).map(drop)
                }
                #[cfg(accelerate)]
                Backend::SmeDirect64(sme, c) => {
                    sme.factorize(values)?;
                    c.load_factor_and_inverse(sme.factor(), sme.inverse().expect("inverses"))
                }
                #[cfg(accelerate)]
                Backend::SmeDirectEf64(sme, c) => {
                    sme.factorize(values)?;
                    c.load_factor_and_inverse(sme.factor(), sme.inverse().expect("inverses"))
                }
                #[cfg(accelerate)]
                Backend::SmePcg64(sme, p) => {
                    sme.factorize(values)?;
                    p.load_preconditioner_and_inverse(values, sme.factor(), sme.inverse().expect("inverses"))
                }
                #[cfg(accelerate)]
                Backend::SmePcgEf64(sme, p) => {
                    sme.factorize(values)?;
                    p.load_preconditioner_and_inverse(values, sme.factor(), sme.inverse().expect("inverses"))
                }
                #[cfg(accelerate)]
                Backend::SmeSplit(sme, c, p) => {
                    p.set_values(values);
                    sme.factorize(values)?;
                    c.load_factor_and_inverse(sme.factor(), sme.inverse().expect("inverses"))
                }
            }
        }

        pub(super) fn set_values(&mut self, values: &[f64]) -> Result<(), Error> {
            match self {
                Backend::Pcg32(p) => p.set_values(values),
                Backend::Pcg64(p) => p.set_values(values),
                Backend::PcgEf64(p) => p.set_values(values),
                Backend::Split(_, p) => {
                    p.set_values(values);
                    Ok(())
                }
                #[cfg(accelerate)]
                Backend::SmePcg64(_, p) => p.set_values(values),
                #[cfg(accelerate)]
                Backend::SmePcgEf64(_, p) => p.set_values(values),
                #[cfg(accelerate)]
                Backend::SmeSplit(_, _, p) => {
                    p.set_values(values);
                    Ok(())
                }
                _ => Ok(()),
            }
        }

        /// `z ← M⁻¹ z` with a direct strategy's factor.
        pub(super) fn precondition(&mut self, z: &mut [f64]) -> Result<(), Error> {
            match self {
                Backend::Direct32(c) => c.solve_in_place(z),
                Backend::Direct64(c) => c.solve_in_place(z),
                Backend::DirectEf64(c) => c.solve_in_place(z),
                #[cfg(accelerate)]
                Backend::SmeDirect64(_, c) => c.solve_in_place(z),
                #[cfg(accelerate)]
                Backend::SmeDirectEf64(_, c) => c.solve_in_place(z),
                _ => unreachable!("only direct strategies precondition with their factor"),
            }
        }

        pub(super) fn solve(&mut self, rhs: &mut [f64], tolerance: f64, max_iterations: usize) -> Result<(usize, Option<f64>), Error> {
            let report = match self {
                Backend::Direct32(c) => return c.solve_in_place(rhs).map(|_| (0, None)),
                Backend::Direct64(c) => return c.solve_in_place(rhs).map(|_| (0, None)),
                Backend::DirectEf64(c) => return c.solve_in_place(rhs).map(|_| (0, None)),
                #[cfg(accelerate)]
                Backend::SmeDirect64(_, c) => return c.solve_in_place(rhs).map(|_| (0, None)),
                #[cfg(accelerate)]
                Backend::SmeDirectEf64(_, c) => return c.solve_in_place(rhs).map(|_| (0, None)),
                #[cfg(accelerate)]
                Backend::SmePcg64(_, p) => p.solve_to(rhs, tolerance, max_iterations)?,
                #[cfg(accelerate)]
                Backend::SmePcgEf64(_, p) => p.solve_to(rhs, tolerance, max_iterations)?,
                #[cfg(accelerate)]
                Backend::SmeSplit(_, c, p) => split(c, p, rhs, tolerance, max_iterations)?,
                Backend::Pcg32(p) => p.solve_to(rhs, tolerance, max_iterations)?,
                Backend::Pcg64(p) => p.solve_to(rhs, tolerance, max_iterations)?,
                Backend::PcgEf64(p) => p.solve_to(rhs, tolerance, max_iterations)?,
                Backend::Split(c, p) => split(c, p, rhs, tolerance, max_iterations)?,
            };
            Ok((report.iterations, Some(report.relative_residual)))
        }
    }

    /// `f64` CG on the CPU, each preconditioner application an `f32` solve on the GPU.
    fn split(c: &mut VulkanCholesky<f32>, p: &CpuPcg, rhs: &mut [f64], tolerance: f64, max_iterations: usize) -> Result<crate::cholesky::PcgReport, Error> {
        let mut failure = None;
        let report = p.solve_to(rhs, PcgOptions { tolerance, max_iterations }, |r, z| {
            z.copy_from_slice(r);
            if let Err(e) = c.solve_in_place(z) {
                failure.get_or_insert(e);
            }
        });
        failure.map_or(Ok(report), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::detect_block;

    /// Lower triangle of a chain of `nodes` nodes with `b` dofs each, every node coupled fully
    /// to itself and its successor.
    fn chain(nodes: usize, b: usize) -> (Vec<usize>, Vec<usize>) {
        let (mut col_ptr, mut row_idx) = (vec![0], Vec::new());
        for node in 0..nodes {
            for c in 0..b {
                let j = node * b + c;
                row_idx.extend(j..(node + 1) * b);
                if node + 1 < nodes {
                    row_idx.extend((node + 1) * b..(node + 2) * b);
                }
                col_ptr.push(row_idx.len());
            }
        }
        (col_ptr, row_idx)
    }

    #[test]
    fn reads_the_dofs_per_node_from_the_pattern() {
        for b in [1, 2, 3] {
            let (c, r) = chain(10, b);
            assert_eq!(detect_block(10 * b, &c, &r), b);
        }
    }
}
