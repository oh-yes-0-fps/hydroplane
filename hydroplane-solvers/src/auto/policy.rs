//! The choice itself: a pure function of what the machine offers ([`Caps`]), how fast it is
//! ([`Rates`]), the pattern's numbers ([`Structure`]), what the caller needs, and what has been
//! learned about the matrix's accuracy ([`Accuracy`]).

use std::fmt;

use super::caps::Caps;
use super::rates::Rates;
use crate::Precision;
use crate::cholesky::Analysis;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// Factor in the strategy's precision, then substitute.
    Direct,
    /// Conjugate gradient in the strategy's precision on its device, preconditioned by an `f32`
    /// factor on the same device.
    Pcg,
    /// Conjugate gradient in `f64` on the CPU, preconditioned by an `f32` factor on the GPU.
    SplitPcg,
}

/// The implementation a strategy's factorization runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Library {
    /// `faer`, on every CPU.
    Faer,
    /// Apple's Accelerate (macOS), on the CPU's matrix units. Only where [`Library::Sme`] is
    /// not, unless the caller names it.
    Accelerate,
    /// This crate's rust-gpu kernels.
    Vulkan,
    /// This crate's supernodal factorization on the CPU's SME units ([`crate::sme`]), handed
    /// to the GPU for the solves.
    Sme,
}

/// One way to solve: what [`super::Solver`] chooses among.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Strategy {
    pub method: Method,
    pub device: Device,
    /// The factorization's precision for [`Method::Direct`], the iteration's otherwise.
    pub precision: Precision,
    pub library: Library,
}

impl Strategy {
    pub const fn new(method: Method, device: Device, precision: Precision, library: Library) -> Self {
        Self { method, device, precision, library }
    }

    /// Every strategy there is an implementation of.
    pub const ALL: [Strategy; 21] = {
        use Device::*;
        use Library::*;
        use Method::*;
        use Precision::*;
        [
            Strategy::new(Direct, Cpu, F64, Faer),
            Strategy::new(Direct, Cpu, F32, Faer),
            Strategy::new(Direct, Cpu, F32, Accelerate),
            Strategy::new(Direct, Cpu, F32, Sme),
            Strategy::new(Direct, Cpu, F64, Accelerate),
            Strategy::new(Pcg, Cpu, F64, Faer),
            Strategy::new(Pcg, Cpu, F64, Accelerate),
            Strategy::new(Direct, Cpu, F64, Sme),
            Strategy::new(Pcg, Cpu, F64, Sme),
            Strategy::new(Direct, Gpu, F32, Vulkan),
            Strategy::new(Direct, Gpu, F64, Vulkan),
            Strategy::new(Direct, Gpu, Ef64, Vulkan),
            Strategy::new(Pcg, Gpu, F32, Vulkan),
            Strategy::new(Pcg, Gpu, F64, Vulkan),
            Strategy::new(Pcg, Gpu, Ef64, Vulkan),
            Strategy::new(SplitPcg, Gpu, F64, Vulkan),
            Strategy::new(Direct, Gpu, F64, Sme),
            Strategy::new(Direct, Gpu, Ef64, Sme),
            Strategy::new(Pcg, Gpu, F64, Sme),
            Strategy::new(Pcg, Gpu, Ef64, Sme),
            Strategy::new(SplitPcg, Gpu, F64, Sme),
        ]
    };

    fn supported(&self, caps: &Caps) -> bool {
        let gpu = || caps.gpu.as_ref().is_some_and(|g| self.method == Method::SplitPcg || g.precisions().contains(&self.precision));
        match (self.device, self.library) {
            (Device::Cpu, Library::Faer) => true,
            (Device::Cpu, Library::Accelerate) => caps.accelerate,
            (Device::Gpu, Library::Vulkan) => gpu(),
            (Device::Gpu, Library::Sme) => caps.sme && gpu(),
            (Device::Cpu, Library::Sme) => caps.sme,
            _ => false,
        }
    }
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let on = if self.device == Device::Cpu { "the Cpu" } else { "the Gpu" };
        let by = match self.library {
            Library::Accelerate => " by Accelerate",
            Library::Sme => " by SME",
            _ => "",
        };
        match (self.method, self.library) {
            (Method::Direct, Library::Sme) if self.device == Device::Gpu => write!(f, "direct {:?} on {on} (f64 factor by SME)", self.precision),
            (Method::Direct, _) => write!(f, "direct {:?} on {on}{by}", self.precision),
            (Method::Pcg, _) => write!(f, "{:?} PCG on {on} (f32 factor{by})", self.precision),
            (Method::SplitPcg, Library::Sme) => write!(f, "{:?} PCG on the CPU (f32 factor by SME, solves on the GPU)", self.precision),
            (Method::SplitPcg, _) => write!(f, "{:?} PCG on the CPU (f32 factor on the GPU)", self.precision),
        }
    }
}

/// The caller's restrictions: `None` leaves the choice open.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Restrict {
    pub device: Option<Device>,
    pub method: Option<Method>,
    pub precision: Option<Precision>,
    pub library: Option<Library>,
}

impl Restrict {
    fn allows(&self, s: &Strategy) -> bool {
        self.device.is_none_or(|d| d == s.device)
            && self.method.is_none_or(|m| m == s.method)
            && self.precision.is_none_or(|p| p == s.precision)
            && self.library.is_none_or(|l| l == s.library)
    }

    /// Every choice that decides accuracy made by the caller, so nothing is left to switch to
    /// (the library may still be chosen by speed).
    pub fn pinned(&self) -> bool {
        self.device.is_some() && self.method.is_some() && self.precision.is_some()
    }
}

/// The numbers of a symbolic analysis that set every strategy's cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Structure {
    pub n: usize,
    /// Stored entries of both triangles.
    pub nnz: usize,
    /// Of one factorization.
    pub gflops: f64,
    /// The GPU factorization's critical path in 32-column panel steps.
    pub panel_steps: usize,
    /// Entries of the factor.
    pub factor_len: usize,
}

impl Structure {
    pub fn of(analysis: &Analysis) -> Self {
        let (col_ptr, row_idx) = (analysis.col_ptr(), analysis.row_idx());
        let diagonal = (0..analysis.dim()).filter(|&j| row_idx[col_ptr[j]..col_ptr[j + 1]].contains(&j)).count();
        Self {
            n: analysis.dim(),
            nnz: 2 * row_idx.len() - diagonal,
            gflops: analysis.flops() * 1e-9,
            panel_steps: analysis.levels().1,
            factor_len: analysis.factor_len(),
        }
    }

    /// So small that a GPU's fixed costs outweigh its speed: decided without measuring.
    pub fn tiny(&self) -> bool {
        self.gflops < 0.05
    }
}

/// How this matrix turns rounding into residual. A solve whose accuracy is limited by
/// precision `p` reaches about `amplification · ε(p)`; an `f32` factor as a preconditioner cuts
/// the residual by about half of that per iteration. Starts from a prior typical of FEM
/// tangents and is replaced by what verified solves show.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Accuracy {
    pub amplification: f64,
    /// Residual reduction per iteration of an `f32`-preconditioned CG, once observed.
    pub pcg_rate: Option<f64>,
    /// Whether the numbers come from this matrix rather than the prior.
    pub learned: bool,
}

impl Default for Accuracy {
    fn default() -> Self {
        Self { amplification: 1e6, pcg_rate: None, learned: false }
    }
}

/// Above this per-iteration reduction an `f32` factor is not worth iterating with.
const USEFUL_RATE: f64 = 0.7;

impl Accuracy {
    /// The smallest residual `s` reaches. CG's lands at about half of a direct solve's in the
    /// same precision (measured: 6.5e-9 against 1.2e-8 in `Ef64`, 4.4e-10 against 5.9e-10 in
    /// `f64`).
    pub fn floor(&self, s: &Strategy) -> f64 {
        let direct = self.amplification * s.precision.epsilon();
        if s.method == Method::Direct { direct } else { 0.5 * direct }
    }

    pub fn pcg_rate(&self) -> f64 {
        self.pcg_rate.unwrap_or(0.5 * self.amplification * Precision::F32.epsilon()).clamp(1e-6, 1.0)
    }

    /// Iterations to `tolerance`, if the preconditioner is good enough to iterate with.
    pub fn iterations(&self, tolerance: f64) -> Option<usize> {
        let rate = self.pcg_rate();
        (rate < USEFUL_RATE).then(|| (tolerance.ln() / rate.ln()).ceil().max(1.0) as usize)
    }
}

/// What a strategy is predicted to cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub factor_ms: f64,
    /// Per solve, all iterations included.
    pub solve_ms: f64,
    pub iterations: usize,
    pub floor: f64,
}

impl Estimate {
    pub fn cycle_ms(&self, solves_per_factorization: f64) -> f64 {
        self.factor_ms + solves_per_factorization * self.solve_ms
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    Feasible(Estimate),
    /// Predicted to stop above the tolerance.
    Inaccurate { floor: f64 },
    /// The `f32` factor is too poor to iterate with.
    WeakPreconditioner { rate: f64 },
    /// Tried on this matrix and missed the tolerance.
    Failed,
    Unsupported,
    /// Ruled out by the caller.
    Excluded,
    /// Accelerate on a CPU with SME, where this crate's SME factorization is used instead.
    Superseded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Every choice was the caller's.
    Pinned,
    /// The only strategy left.
    Only,
    /// Too small for a GPU to pay off, so not measured.
    Tiny,
    /// The cheapest feasible by the measured rates.
    Cheapest,
    /// No measured rates: the first feasible in a fixed preference.
    Preference,
    /// Nothing is predicted to meet the tolerance: the most accurate.
    MostAccurate,
    /// The fastest of the closest candidates, timed on this pattern.
    Measured,
}

/// A choice and its grounds: the strategy, its predicted cost, and every candidate's verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub strategy: Strategy,
    pub estimate: Option<Estimate>,
    pub reason: Reason,
    pub tolerance: f64,
    pub accuracy: Accuracy,
    pub candidates: Vec<(Strategy, Verdict)>,
}

/// What the caller needs and expects.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Needs {
    pub tolerance: f64,
    pub solves_per_factorization: f64,
}

/// Fixed order for when there are no rates: the CPU for small problems, then the cheapest
/// accurate GPU paths. On the CPU the matrix units come first where present (SME, or
/// Accelerate where there is no SME: 66 and 62 ms for the welded slab's factorization against
/// `faer`'s 757 ms).
const PREFERENCE: [Strategy; 21] = {
    use Device::*;
    use Library::*;
    use Method::*;
    use Precision::*;
    [
        Strategy::new(Direct, Cpu, F64, Sme),
        Strategy::new(Direct, Cpu, F64, Accelerate),
        Strategy::new(Direct, Cpu, F64, Faer),
        Strategy::new(Pcg, Gpu, F64, Vulkan),
        Strategy::new(Pcg, Gpu, Ef64, Vulkan),
        Strategy::new(Direct, Gpu, F64, Vulkan),
        Strategy::new(Direct, Gpu, Ef64, Vulkan),
        Strategy::new(SplitPcg, Gpu, F64, Vulkan),
        Strategy::new(Pcg, Cpu, F64, Accelerate),
        Strategy::new(Pcg, Cpu, F64, Faer),
        Strategy::new(Pcg, Gpu, F32, Vulkan),
        Strategy::new(Direct, Gpu, F32, Vulkan),
        Strategy::new(Pcg, Gpu, F64, Sme),
        Strategy::new(Pcg, Gpu, Ef64, Sme),
        Strategy::new(Direct, Gpu, F64, Sme),
        Strategy::new(Direct, Gpu, Ef64, Sme),
        Strategy::new(SplitPcg, Gpu, F64, Sme),
        Strategy::new(Pcg, Cpu, F64, Sme),
        Strategy::new(Direct, Cpu, F32, Sme),
        Strategy::new(Direct, Cpu, F32, Accelerate),
        Strategy::new(Direct, Cpu, F32, Faer),
    ]
};

pub fn estimate(rates: &Rates, st: &Structure, s: &Strategy, acc: &Accuracy, tolerance: f64) -> Option<Estimate> {
    let floor = acc.floor(s);
    let g = st.gflops;
    let steps = st.panel_steps as f64;
    let mnnz = st.nnz as f64 * 1e-6;
    let factor_gb = |p: Precision| st.factor_len as f64 * p.bytes() as f64 * 1e-9;
    let gpu = |p: Precision| rates.gpu.iter().find(|r| r.precision == p);
    // On the GPU's own kernels, or by SME (in `f64` for a wide precision, else `f32`) and loaded.
    let gpu_factor = |p: Precision| match s.library {
        Library::Sme => {
            let c = rates.sme.as_ref()?;
            let load = gpu(p).map(|r| r.load_ms + r.load_ms_per_gb * factor_gb(p))?;
            let sme = if p == Precision::F32 { c.factor32_ms_per_step * steps + c.factor32_ms_per_gflop * g } else { c.factor_ms_per_step * steps + c.factor_ms_per_gflop * g };
            Some(sme + load)
        }
        _ => gpu(p).map(|r| r.factor_ms_per_step * st.panel_steps as f64 + r.factor_ms_per_gflop * g),
    };
    let gpu_solve = |p: Precision| gpu(p).map(|r| r.solve_ms + r.solve_ms_per_gb * factor_gb(p));
    let iterations = acc.iterations(tolerance).unwrap_or(0);
    let cpu = match s.library {
        Library::Accelerate => rates.accelerate.as_ref(),
        Library::Sme => rates.sme.as_ref(),
        _ => Some(&rates.cpu),
    };
    let (factor_ms, solve_ms, iterations) = match (s.method, s.device) {
        (Method::Direct, Device::Cpu) if s.precision == Precision::F32 => {
            let c = cpu?;
            (c.factor32_ms_per_step * steps + c.factor32_ms_per_gflop * g, c.solve32_ms_per_gb * factor_gb(Precision::F32), 0)
        }
        (Method::Direct, Device::Cpu) => {
            let c = cpu?;
            (c.factor_ms_per_step * steps + c.factor_ms_per_gflop * g, c.solve_ms_per_gb * factor_gb(Precision::F64), 0)
        }
        (Method::Pcg, Device::Cpu) => {
            let c = cpu?;
            let it = c.solve32_ms_per_gb * factor_gb(Precision::F32) + rates.cpu.iteration_ms + rates.cpu.iteration_ms_per_mnnz * mnnz;
            (c.factor32_ms_per_step * steps + c.factor32_ms_per_gflop * g, (iterations + 1) as f64 * it, iterations)
        }
        (Method::Direct, Device::Gpu) => (gpu_factor(s.precision)?, gpu_solve(s.precision)?, 0),
        (Method::Pcg, Device::Gpu) => {
            // The start applies the preconditioner once more; every batch of four is a submission.
            let submit = gpu(Precision::F32)?.submit_ms;
            let inline_solve = (gpu_solve(Precision::F32)? - submit).max(0.0);
            let q = gpu(s.precision)?;
            let it = inline_solve + q.iteration_ms + q.iteration_ms_per_mnnz * mnnz;
            let submissions = 2 + iterations.div_ceil(4);
            (gpu_factor(Precision::F32)?, iterations as f64 * it + inline_solve + submissions as f64 * submit, iterations)
        }
        (Method::SplitPcg, _) => {
            let it = gpu_solve(Precision::F32)? + rates.cpu.iteration_ms + rates.cpu.iteration_ms_per_mnnz * mnnz;
            (gpu_factor(Precision::F32)?, (iterations + 1) as f64 * it, iterations)
        }
    };
    Some(Estimate { factor_ms, solve_ms, iterations, floor })
}

/// The plan for these inputs. A strategy is feasible when it is supported, allowed, not known
/// to fail, predicted to reach half the tolerance, and (iterating) has a useful preconditioner;
/// of the feasible ones the cheapest per factorization and its solves wins, by `rates` when
/// there are some and by [`PREFERENCE`] when not.
pub fn choose(caps: &Caps, rates: Option<&Rates>, st: &Structure, needs: &Needs, acc: &Accuracy, restrict: &Restrict, failed: &[Strategy]) -> Plan {
    let tolerance = needs.tolerance;
    let candidates: Vec<(Strategy, Verdict)> = Strategy::ALL
        .iter()
        .map(|s| {
            let verdict = if !s.supported(caps) {
                Verdict::Unsupported
            } else if !restrict.allows(s) {
                Verdict::Excluded
            } else if s.library == Library::Accelerate && caps.sme && restrict.library != Some(Library::Accelerate) {
                Verdict::Superseded
            } else if failed.contains(s) {
                Verdict::Failed
            } else if acc.floor(s) > 0.5 * tolerance {
                Verdict::Inaccurate { floor: acc.floor(s) }
            } else if s.method != Method::Direct && acc.iterations(tolerance).is_none() {
                Verdict::WeakPreconditioner { rate: acc.pcg_rate() }
            } else {
                match rates.and_then(|r| estimate(r, st, s, acc, tolerance)) {
                    Some(e) => Verdict::Feasible(e),
                    None => Verdict::Feasible(Estimate { factor_ms: f64::NAN, solve_ms: f64::NAN, iterations: acc.iterations(tolerance).unwrap_or(0), floor: acc.floor(s) }),
                }
            };
            (*s, verdict)
        })
        .collect();
    let feasible: Vec<(Strategy, Estimate)> = candidates.iter().filter_map(|(s, v)| if let Verdict::Feasible(e) = v { Some((*s, *e)) } else { None }).collect();
    let measured = |e: &Estimate| !e.factor_ms.is_nan();
    let (strategy, estimate, reason) = if restrict.pinned() && feasible.len() <= 1 {
        let s = feasible.first().map(|f| f.0).or_else(|| Strategy::ALL.into_iter().find(|s| restrict.allows(s) && s.supported(caps))).unwrap_or(Strategy::ALL[0]);
        (s, feasible.first().map(|f| f.1).filter(measured), Reason::Pinned)
    } else if feasible.len() == 1 {
        (feasible[0].0, Some(feasible[0].1).filter(measured), Reason::Only)
    } else if let Some((s, e)) = feasible
        .iter()
        .filter(|(_, e)| measured(e))
        .min_by(|a, b| a.1.cycle_ms(needs.solves_per_factorization).total_cmp(&b.1.cycle_ms(needs.solves_per_factorization)))
    {
        (*s, Some(*e), Reason::Cheapest)
    } else if let Some(s) = PREFERENCE.iter().find(|p| feasible.iter().any(|(s, _)| s == *p)) {
        // A single-precision factor that meets the tolerance, alone or in a couple of
        // iterations, beats the double-precision one on the same library: its factorization is
        // 1.5–2× cheaper on every CPU measured (faer 443 against 673 ms on a welded slab,
        // Accelerate 31 against 62 ms) and its solves move half the bytes.
        let direct = Strategy::new(Method::Direct, Device::Cpu, Precision::F32, s.library);
        let iterated = Strategy::new(Method::Pcg, Device::Cpu, Precision::F64, s.library);
        let has = |x: &Strategy| feasible.iter().any(|(f, _)| f == x);
        let s = if s.device != Device::Cpu || s.method != Method::Direct {
            *s
        } else if has(&direct) {
            direct
        } else if acc.iterations(tolerance).is_some_and(|i| i <= 2) && has(&iterated) {
            iterated
        } else {
            *s
        };
        (s, None, if st.tiny() { Reason::Tiny } else { Reason::Preference })
    } else {
        // The most accurate, and of those within a factor of two of it the cheapest.
        let usable: Vec<Strategy> = candidates
            .iter()
            .filter(|(s, v)| !matches!(v, Verdict::Unsupported | Verdict::Excluded | Verdict::Superseded | Verdict::Failed) || (restrict.pinned() && restrict.allows(s)))
            .map(|(s, _)| *s)
            .collect();
        let best_floor = usable.iter().map(|s| acc.floor(s)).fold(f64::INFINITY, f64::min);
        let cost = |s: &Strategy| rates.and_then(|r| estimate(r, st, s, acc, tolerance)).map_or(f64::INFINITY, |e| e.cycle_ms(needs.solves_per_factorization));
        let s = usable
            .iter()
            .filter(|s| acc.floor(s) <= 2.0 * best_floor)
            .min_by(|a, b| cost(a).total_cmp(&cost(b)))
            .copied()
            .or_else(|| candidates.iter().find(|(s, v)| *v == Verdict::Failed && restrict.allows(s)).map(|(s, _)| *s))
            .unwrap_or(Strategy::ALL[0]);
        (s, rates.and_then(|r| estimate(r, st, &s, acc, tolerance)), Reason::MostAccurate)
    };
    Plan { strategy, estimate, reason, tolerance, accuracy: *acc, candidates }
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let why = match self.reason {
            Reason::Pinned => "chosen by the caller",
            Reason::Only => "the only strategy left",
            Reason::Tiny => "too small for the GPU to pay off",
            Reason::Cheapest => "the cheapest by the measured rates",
            Reason::Preference => "the first feasible (no rates measured)",
            Reason::MostAccurate => "nothing is predicted to reach the tolerance; the most accurate",
            Reason::Measured => "the fastest of the closest candidates, timed on this pattern",
        };
        writeln!(f, "{}: {why}", self.strategy)?;
        let source = if self.accuracy.learned { "learned" } else { "prior" };
        write!(f, "  tolerance {:.0e}, error amplification {:.1e}", self.tolerance, self.accuracy.amplification)?;
        match self.accuracy.pcg_rate {
            Some(rate) => writeln!(f, ", PCG residual × {rate:.2} per iteration ({source})")?,
            None => writeln!(f, " ({source})")?,
        }
        for (s, v) in &self.candidates {
            let mark = if *s == self.strategy { '*' } else { ' ' };
            let verdict = match v {
                Verdict::Feasible(e) if e.factor_ms.is_nan() => format!("feasible, floor {:.0e}", e.floor),
                Verdict::Feasible(e) if e.iterations > 0 => {
                    format!("{:.1} ms factorization + {:.1} ms per solve ({} iterations), floor {:.0e}", e.factor_ms, e.solve_ms, e.iterations, e.floor)
                }
                Verdict::Feasible(e) => format!("{:.1} ms factorization + {:.1} ms per solve, floor {:.0e}", e.factor_ms, e.solve_ms, e.floor),
                Verdict::Inaccurate { floor } => format!("too inaccurate (floor {floor:.0e})"),
                Verdict::WeakPreconditioner { rate } => format!("preconditioner too weak ({rate:.2} per iteration)"),
                Verdict::Failed => "missed the tolerance on this matrix".to_string(),
                Verdict::Unsupported => "unsupported here".to_string(),
                Verdict::Excluded => "excluded by the caller".to_string(),
                Verdict::Superseded => "SME is used instead".to_string(),
            };
            writeln!(f, "  {mark} {:<44} {verdict}", s.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auto::caps::GpuCaps;
    use crate::auto::rates::{CpuRates, GpuRates};

    fn mac() -> Caps {
        Caps { threads: 14, accelerate: false, sme: false, gpu: Some(GpuCaps { name: "Apple M5 Pro".into(), f64: false, discrete: false, unified_memory: true }) }
    }

    fn rtx() -> Caps {
        Caps { threads: 16, accelerate: false, sme: false, gpu: Some(GpuCaps { name: "RTX 5090".into(), f64: true, discrete: true, unified_memory: false }) }
    }

    fn rates(gpu: &[(Precision, f64)]) -> Rates {
        Rates {
            cpu: CpuRates {
                factor_ms_per_step: 0.0,
                factor_ms_per_gflop: 18.0,
                factor32_ms_per_step: 0.0,
                factor32_ms_per_gflop: 9.0,
                solve_ms_per_gb: 100.0,
                solve32_ms_per_gb: 100.0,
                iteration_ms: 0.1,
                iteration_ms_per_mnnz: 1.0,
            },
            accelerate: None,
            sme: None,
            gpu: gpu
                .iter()
                .map(|&(precision, ms_per_gflop)| GpuRates {
                    precision,
                    factor_ms_per_step: 0.07,
                    factor_ms_per_gflop: ms_per_gflop,
                    solve_ms: 0.3,
                    solve_ms_per_gb: 4.0,
                    submit_ms: 0.2,
                    iteration_ms: 0.05,
                    iteration_ms_per_mnnz: 0.1,
                    load_ms: 1.0,
                    load_ms_per_gb: 40.0,
                    wide_schur: precision == Precision::F32,
                })
                .collect(),
        }
    }

    /// The welded slab's tangent.
    const SLAB: Structure = Structure { n: 87_954, nnz: 6_594_732, gflops: 41.5, panel_steps: 230, factor_len: 51_228_459 };

    fn needs(tolerance: f64) -> Needs {
        Needs { tolerance, solves_per_factorization: 1.0 }
    }

    /// With no GPU there are no rates; a tolerance a single-precision factor meets, alone or in a
    /// couple of iterations, takes it over the double-precision one, a tight tolerance does not.
    #[test]
    fn without_rates_a_loose_tolerance_takes_the_single_precision_factor() {
        let cpu = Caps { threads: 16, accelerate: false, sme: false, gpu: None };
        let pick = |tolerance| choose(&cpu, None, &SLAB, &needs(tolerance), &Accuracy::default(), &Restrict::default(), &[]).strategy;
        assert_eq!(pick(0.5), Strategy::new(Method::Direct, Device::Cpu, Precision::F32, Library::Faer));
        assert_eq!(pick(0.05), Strategy::new(Method::Pcg, Device::Cpu, Precision::F64, Library::Faer));
        assert_eq!(pick(1e-8), Strategy::new(Method::Direct, Device::Cpu, Precision::F64, Library::Faer));
        let mac = Caps { threads: 14, accelerate: true, sme: true, gpu: None };
        let plan = choose(&mac, None, &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &[]);
        assert_eq!(plan.strategy, Strategy::new(Method::Direct, Device::Cpu, Precision::F64, Library::Sme));
    }

    #[test]
    fn without_f64_the_iteration_is_emulated() {
        let r = rates(&[(Precision::F32, 1.6), (Precision::Ef64, 3.1)]);
        let plan = choose(&mac(), Some(&r), &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &[]);
        assert_eq!(plan.strategy, Strategy::new(Method::Pcg, Device::Gpu, Precision::Ef64, Library::Vulkan), "{plan}");
    }

    #[test]
    fn below_the_emulated_floor_the_iteration_moves_to_the_cpu() {
        let r = rates(&[(Precision::F32, 1.6), (Precision::Ef64, 3.1)]);
        let acc = Accuracy { amplification: 4e6, pcg_rate: Some(0.1), learned: true };
        let plan = choose(&mac(), Some(&r), &SLAB, &needs(1e-9), &acc, &Restrict::default(), &[]);
        assert_eq!(plan.strategy, Strategy::new(Method::SplitPcg, Device::Gpu, Precision::F64, Library::Vulkan), "{plan}");
    }

    #[test]
    fn many_solves_per_factorization_favour_a_direct_factor() {
        let r = rates(&[(Precision::F32, 0.26), (Precision::F64, 0.94), (Precision::Ef64, 0.5)]);
        let n = Needs { tolerance: 1e-6, solves_per_factorization: 50.0 };
        let plan = choose(&rtx(), Some(&r), &SLAB, &n, &Accuracy::default(), &Restrict::default(), &[]);
        assert_eq!(plan.strategy.method, Method::Direct, "{plan}");
    }

    #[test]
    fn a_failed_strategy_is_not_chosen_again() {
        let r = rates(&[(Precision::F32, 1.6), (Precision::Ef64, 3.1)]);
        let failed = [Strategy::new(Method::Pcg, Device::Gpu, Precision::Ef64, Library::Vulkan)];
        let plan = choose(&mac(), Some(&r), &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &failed);
        assert_ne!(plan.strategy, failed[0], "{plan}");
    }

    #[test]
    fn the_caller_decides_when_pinned() {
        let pin = Restrict { device: Some(Device::Cpu), method: Some(Method::Direct), precision: Some(Precision::F64), library: None };
        let plan = choose(&rtx(), None, &SLAB, &needs(1e-30), &Accuracy::default(), &pin, &[]);
        assert_eq!((plan.strategy, plan.reason), (Strategy::ALL[0], Reason::Pinned));
    }

    #[test]
    fn accelerate_wins_on_a_mac_where_it_is_fastest() {
        let mut r = rates(&[(Precision::F32, 1.6), (Precision::Ef64, 3.1)]);
        r.accelerate = Some(CpuRates { factor_ms_per_gflop: 1.4, factor32_ms_per_gflop: 0.7, solve_ms_per_gb: 37.0, solve32_ms_per_gb: 43.0, ..r.cpu });
        let caps = Caps { accelerate: true, ..mac() };
        let plan = choose(&caps, Some(&r), &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &[]);
        assert_eq!(plan.strategy, Strategy::new(Method::Direct, Device::Cpu, Precision::F64, Library::Accelerate), "{plan}");
    }

    #[test]
    fn sme_is_chosen_over_accelerate_even_where_accelerate_measures_faster() {
        let mut r = rates(&[(Precision::F32, 1.6), (Precision::Ef64, 3.1)]);
        r.accelerate = Some(CpuRates { factor_ms_per_gflop: 0.5, factor32_ms_per_gflop: 0.25, solve_ms_per_gb: 10.0, solve32_ms_per_gb: 10.0, ..r.cpu });
        r.sme = Some(CpuRates { factor_ms_per_gflop: 1.4, factor32_ms_per_gflop: 0.7, solve_ms_per_gb: 37.0, solve32_ms_per_gb: 43.0, ..r.cpu });
        let caps = Caps { accelerate: true, sme: true, gpu: None, ..mac() };
        for rates in [Some(&r), None] {
            let plan = choose(&caps, rates, &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &[]);
            assert_eq!(plan.strategy.library, Library::Sme, "{plan}");
        }
        let named = Restrict { library: Some(Library::Accelerate), ..Default::default() };
        let plan = choose(&caps, Some(&r), &SLAB, &needs(1e-8), &Accuracy::default(), &named, &[]);
        assert_eq!(plan.strategy.library, Library::Accelerate, "{plan}");
    }

    #[test]
    fn an_sme_factor_with_gpu_iterations_wins_where_sme_factors_fastest() {
        let mut r = rates(&[(Precision::F32, 1.6), (Precision::Ef64, 3.1)]);
        r.accelerate = Some(CpuRates { factor_ms_per_gflop: 1.5, factor32_ms_per_gflop: 0.75, solve_ms_per_gb: 37.0, solve32_ms_per_gb: 43.0, ..r.cpu });
        r.sme = Some(CpuRates { factor_ms_per_gflop: 1.8, factor32_ms_per_gflop: 0.6, ..r.cpu });
        let caps = Caps { accelerate: true, sme: true, ..mac() };
        let plan = choose(&caps, Some(&r), &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &[]);
        assert_eq!(plan.strategy, Strategy::new(Method::Pcg, Device::Gpu, Precision::Ef64, Library::Sme), "{plan}");
    }

    #[test]
    fn a_restriction_narrows_the_choice() {
        let r = rates(&[(Precision::F32, 0.26), (Precision::F64, 0.94), (Precision::Ef64, 0.5)]);
        let only_cpu = Restrict { device: Some(Device::Cpu), ..Default::default() };
        let plan = choose(&rtx(), Some(&r), &SLAB, &needs(1e-8), &Accuracy::default(), &only_cpu, &[]);
        assert_eq!(plan.strategy.device, Device::Cpu, "{plan}");
    }

    #[test]
    fn without_a_gpu_the_cpu_is_all_there_is() {
        let caps = Caps { threads: 8, accelerate: false, sme: false, gpu: None };
        let plan = choose(&caps, None, &SLAB, &needs(1e-8), &Accuracy::default(), &Restrict::default(), &[]);
        assert_eq!(plan.strategy.device, Device::Cpu);
        assert!(plan.candidates.iter().filter(|(s, _)| s.device == Device::Gpu).all(|(_, v)| *v == Verdict::Unsupported));
    }
}
