//! Front doors for matrices that are not positive definite, as [`crate::Solver`] is for those
//! that are: each picks a factorization for the platform and the matrix, falls back when it
//! fails, checks the true residual of the first solve after each factorization, and refines with
//! GMRES when that misses the tolerance.
//!
//! - [`GeneralSolver`]: any nonsingular square matrix. Accelerate's LU on macOS; elsewhere our
//!   [`FrontalLu`] (with a [`crate::lu::Matching`] when the diagonal is weak), then `faer`'s
//!   fully pivoting [`CpuLu`] if static pivoting breaks down.
//! - [`IndefiniteSolver`]: symmetric, any inertia. Accelerate's threshold-pivoting `L D Lᵀ` on
//!   macOS; elsewhere `faer`'s `L D Lᵀ`, in the [`kkt_ordering`] with regularized pivots when the
//!   expected pivot signs are given (saddle points), else Bunch–Kaufman, and a general LU of the
//!   whole matrix if neither factors.

use crate::cholesky::{Analysis, Error, Ordering};
use crate::sparse::SellMatrix;
use crate::iterative::{Gmres, RelativeTo};
use crate::ldlt::{CpuLdlt, Inertia, Pivoting, kkt_ordering};
use crate::lu::{CpuLu, FrontalLu};

/// What a solve did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Report {
    /// The factorization that served it.
    pub method: &'static str,
    /// GMRES steps taken on top of the direct solve (0 when it met the tolerance alone).
    pub refinement_steps: usize,
    /// `‖b − A x‖ / ‖b‖` when checked (the first solve after a factorization, and any refined
    /// one).
    pub residual: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The relative residual a solve must reach.
    pub tolerance: f64,
    /// GMRES steps allowed to reach it.
    pub max_refinement: usize,
    /// Use only portable (`faer` and this crate's) factorizations, never Accelerate.
    pub portable: bool,
    /// [`IndefiniteSolver`] only: the expected sign of each pivot, `+1` for a primal row and `−1`
    /// for a constraint (the inertia a saddle-point matrix has), which allows the faster
    /// regularized factorization in the KKT ordering.
    pub signs: Option<Vec<i8>>,
}

impl Default for Options {
    fn default() -> Self {
        Self { tolerance: 1e-10, max_refinement: 30, portable: false, signs: None }
    }
}

/// The checks and refinement both front doors share.
struct Checked {
    tolerance: f64,
    max_refinement: usize,
    check_next: bool,
    refine_always: bool,
}

impl Checked {
    fn new(options: &Options) -> Self {
        Self { tolerance: options.tolerance, max_refinement: options.max_refinement, check_next: true, refine_always: false }
    }

    /// Solves with `direct`, checking with `multiply` when due and refining when short.
    fn solve(&mut self, method: &'static str, rhs: &mut [f64], mut multiply: impl FnMut(&[f64], &mut [f64]), mut direct: impl FnMut(&mut [f64]) -> Result<(), Error>) -> Result<Report, Error> {
        let b = rhs.to_vec();
        direct(rhs)?;
        if !self.check_next && !self.refine_always {
            return Ok(Report { method, refinement_steps: 0, residual: None });
        }
        self.check_next = false;
        let residual = relative_residual(&mut multiply, &b, rhs);
        if residual <= self.tolerance {
            return Ok(Report { method, refinement_steps: 0, residual: Some(residual) });
        }
        self.refine_always = true;
        let mut precondition = |r: &[f64], z: &mut [f64]| {
            z.copy_from_slice(r);
            if direct(z).is_err() {
                z.copy_from_slice(r);
            }
        };
        let options = crate::iterative::Options { tolerance: self.tolerance, relative_to: RelativeTo::RightHandSide, max_iterations: self.max_refinement, warm_start: true, restart: self.max_refinement.clamp(1, 50), ..Default::default() };
        let report = Gmres::with_options(options).solve(&mut multiply, &mut precondition, &b[..], rhs);
        Ok(Report { method, refinement_steps: report.iterations, residual: Some(relative_residual(&mut multiply, &b, rhs)) })
    }
}

fn relative_residual(multiply: &mut impl FnMut(&[f64], &mut [f64]), b: &[f64], x: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    multiply(x, &mut ax);
    let rr: f64 = b.iter().zip(&ax).map(|(b, a)| (b - a) * (b - a)).sum();
    let bb: f64 = b.iter().map(|b| b * b).sum();
    if bb > 0.0 { (rr / bb).sqrt() } else { rr.sqrt() }
}

enum LuBackend {
    #[cfg(accelerate)]
    Accelerate(crate::accelerate::AccelerateLu<f64>),
    Frontal(Box<FrontalLu<f64>>),
    Faer(Box<CpuLu<f64>>),
}

/// A sparse solver for any nonsingular square matrix, given whole in compressed sparse columns.
pub struct GeneralSolver {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    csr: SellMatrix,
    values: Vec<f64>,
    #[cfg_attr(not(accelerate), allow(dead_code))]
    options: Options,
    backend: Option<LuBackend>,
    checked: Checked,
}

impl GeneralSolver {
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::with_options(col_ptr, row_idx, Options::default())
    }

    pub fn with_options(col_ptr: &[usize], row_idx: &[usize], options: Options) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        if row_idx[..col_ptr[n]].iter().any(|&i| i >= n) {
            return Err(Error::Pattern);
        }
        Ok(Self { csr: SellMatrix::from_csc(n, col_ptr, row_idx), col_ptr: col_ptr.to_vec(), row_idx: row_idx.to_vec(), values: Vec::new(), checked: Checked::new(&options), options, backend: None })
    }

    /// The factorization in use.
    pub fn method(&self) -> Option<&'static str> {
        self.backend.as_ref().map(lu_name)
    }

    /// Factors the matrix whose values (in the pattern's order) are `values`. The first call
    /// chooses the factorization; later ones keep it unless it fails.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.values.clear();
        self.values.extend_from_slice(values);
        self.csr.set_values(values);
        self.checked.check_next = true;
        self.checked.refine_always = false;
        if let Some(backend) = &mut self.backend {
            let ok = match backend {
                #[cfg(accelerate)]
                LuBackend::Accelerate(lu) => lu.factorize(values).is_ok(),
                LuBackend::Frontal(lu) => lu.factorize(values).is_ok(),
                LuBackend::Faer(lu) => lu.factorize(values).is_ok(),
            };
            if ok {
                return Ok(());
            }
        }
        let mut last = Error::Singular;
        for candidate in self.candidates() {
            match candidate.and_then(|mut b| {
                match &mut b {
                    #[cfg(accelerate)]
                    LuBackend::Accelerate(lu) => lu.factorize(values)?,
                    LuBackend::Frontal(lu) => lu.factorize(values)?,
                    LuBackend::Faer(lu) => lu.factorize(values)?,
                }
                Ok(b)
            }) {
                Ok(b) => {
                    self.backend = Some(b);
                    return Ok(());
                }
                Err(e) => last = e,
            }
        }
        self.backend = None;
        Err(last)
    }

    /// The factorizations to try, best first.
    fn candidates(&self) -> Vec<Result<LuBackend, Error>> {
        let (col_ptr, row_idx) = (&self.col_ptr[..], &self.row_idx[..]);
        #[cfg_attr(not(accelerate), allow(unused_mut))]
        let mut out = vec![FrontalLu::new(col_ptr, row_idx).map(|lu| LuBackend::Frontal(Box::new(lu))), CpuLu::new(col_ptr, row_idx).map(|lu| LuBackend::Faer(Box::new(lu)))];
        #[cfg(accelerate)]
        if !self.options.portable {
            out.insert(0, crate::accelerate::AccelerateLu::new(col_ptr, row_idx).map(LuBackend::Accelerate));
        }
        out
    }

    /// `rhs ← A⁻¹ rhs`.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<Report, Error> {
        let backend = self.backend.as_mut().ok_or(Error::NotFactorized)?;
        let method = lu_name(backend);
        let csr = &self.csr;
        self.checked.solve(method, rhs, |x, y| csr.spmv(x, y), |x| match backend {
            #[cfg(accelerate)]
            LuBackend::Accelerate(lu) => lu.solve_in_place(x),
            LuBackend::Frontal(lu) => {
                lu.solve_in_place(x);
                Ok(())
            }
            LuBackend::Faer(lu) => {
                lu.solve_in_place(x);
                Ok(())
            }
        })
    }
}

fn lu_name(backend: &LuBackend) -> &'static str {
    match backend {
        #[cfg(accelerate)]
        LuBackend::Accelerate(_) => "Accelerate LU",
        LuBackend::Frontal(_) => "FrontalLu",
        LuBackend::Faer(_) => "faer LU",
    }
}

/// Dofs from which a signed GPU factorization is tried: below, dispatch overheads outweigh it.
#[cfg(feature = "vulkan")]
const GPU_DOFS: usize = 10_000;

/// A signed GPU factorization and what the last one reported.
#[cfg(feature = "vulkan")]
struct Gpu<T: crate::vulkan::DeviceReal> {
    f: crate::vulkan::VulkanLdlt<T>,
    negative: usize,
    perturbed: bool,
}

enum LdltBackend {
    #[cfg(accelerate)]
    Accelerate(crate::accelerate::AccelerateLdlt<f64>),
    #[cfg(feature = "vulkan")]
    GpuF64(Box<Gpu<f64>>),
    #[cfg(feature = "vulkan")]
    GpuEf64(Box<Gpu<crate::Ef64>>),
    Faer(Box<CpuLdlt<f64>>),
    /// The whole matrix's LU, and for each of its entries the lower-triangle entry it copies.
    Lu(Box<GeneralSolver>, Vec<usize>),
}

/// A sparse solver for symmetric matrices of any inertia, given by their lower triangle in
/// compressed sparse columns.
pub struct IndefiniteSolver {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    csr: SellMatrix,
    values: Vec<f64>,
    options: Options,
    backend: Option<LdltBackend>,
    checked: Checked,
}

impl IndefiniteSolver {
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::with_options(col_ptr, row_idx, Options::default())
    }

    pub fn with_options(col_ptr: &[usize], row_idx: &[usize], options: Options) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        if options.signs.as_ref().is_some_and(|s| s.len() != n) {
            return Err(Error::Pattern);
        }
        Ok(Self {
            csr: SellMatrix::from_lower_csc_symmetric(n, col_ptr, row_idx),
            col_ptr: col_ptr.to_vec(),
            row_idx: row_idx.to_vec(),
            values: Vec::new(),
            checked: Checked::new(&options),
            options,
            backend: None,
        })
    }

    pub fn method(&self) -> Option<&'static str> {
        self.backend.as_ref().map(ldlt_name)
    }

    /// The signs of the last factorization's pivots, where the factorization reports them
    /// (not after falling back to LU).
    pub fn inertia(&self) -> Option<Inertia> {
        match self.backend.as_ref()? {
            #[cfg(accelerate)]
            LdltBackend::Accelerate(f) => f.inertia(),
            #[cfg(feature = "vulkan")]
            LdltBackend::GpuF64(g) => g.inertia(self.col_ptr.len() - 1),
            #[cfg(feature = "vulkan")]
            LdltBackend::GpuEf64(g) => g.inertia(self.col_ptr.len() - 1),
            LdltBackend::Faer(f) => Some(f.inertia()),
            LdltBackend::Lu(..) => None,
        }
    }

    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.values.clear();
        self.values.extend_from_slice(values);
        self.csr.set_values(values);
        self.checked.check_next = true;
        self.checked.refine_always = false;
        if let Some(backend) = &mut self.backend
            && refactor(backend, values).is_ok()
        {
            return Ok(());
        }
        let mut last = Error::Singular;
        for candidate in self.candidates() {
            match candidate.and_then(|mut b| refactor(&mut b, values).map(|_| b)) {
                Ok(b) => {
                    self.backend = Some(b);
                    return Ok(());
                }
                Err(e) => last = e,
            }
        }
        self.backend = None;
        Err(last)
    }

    fn candidates(&self) -> Vec<Result<LdltBackend, Error>> {
        let (col_ptr, row_idx) = (&self.col_ptr[..], &self.row_idx[..]);
        let mut out = Vec::new();
        #[cfg(feature = "vulkan")]
        let gpu = self.gpu_candidate();
        #[cfg(feature = "vulkan")]
        let gpu = match gpu {
            Some((true, c)) => {
                out.push(c);
                None
            }
            other => other,
        };
        #[cfg(accelerate)]
        if !self.options.portable {
            out.push(Analysis::with_ordering(col_ptr, row_idx, Ordering::Amd).and_then(|a| {
                Ok(LdltBackend::Accelerate(crate::accelerate::AccelerateLdlt::new(&a)?))
            }));
        }
        #[cfg(feature = "vulkan")]
        out.extend(gpu.map(|(_, c)| c));
        if let Some(signs) = &self.options.signs {
            out.push(kkt_ordering(col_ptr, row_idx, signs).and_then(|perm| {
                let a = Analysis::with_ordering(col_ptr, row_idx, Ordering::Custom(perm))?;
                Ok(LdltBackend::Faer(Box::new(CpuLdlt::with_pivoting(&a, Pivoting::regularized(signs.clone())))))
            }));
        }
        out.push(Analysis::with_ordering(col_ptr, row_idx, Ordering::Amd).map(|a| LdltBackend::Faer(Box::new(CpuLdlt::new(&a)))));
        let (cp, ri, source) = whole_matrix(col_ptr, row_idx);
        out.push(GeneralSolver::with_options(&cp, &ri, Options { portable: true, signs: None, ..self.options.clone() }).map(|g| LdltBackend::Lu(Box::new(g), source)));
        out
    }

    /// The signed GPU factorization in the KKT ordering, when the signs are given, the matrix is
    /// large enough and a Vulkan device is present; and whether to try it first (a discrete GPU).
    #[cfg(feature = "vulkan")]
    fn gpu_candidate(&self) -> Option<(bool, Result<LdltBackend, Error>)> {
        let signs = self.options.signs.as_ref()?;
        let (col_ptr, row_idx) = (&self.col_ptr[..], &self.row_idx[..]);
        if col_ptr.len() - 1 < GPU_DOFS {
            return None;
        }
        let ctx = crate::vulkan::Context::shared().ok()?;
        let negative = signs.iter().filter(|&&s| s < 0).count();
        let build = || -> Result<LdltBackend, Error> {
            let a = Analysis::with_ordering(col_ptr, row_idx, Ordering::Custom(kkt_ordering(col_ptr, row_idx, signs)?))?;
            Ok(if ctx.supports_f64() {
                LdltBackend::GpuF64(Box::new(Gpu { f: crate::vulkan::VulkanLdlt::new(&a, signs)?, negative, perturbed: false }))
            } else {
                LdltBackend::GpuEf64(Box::new(Gpu { f: crate::vulkan::VulkanLdlt::new(&a, signs)?, negative, perturbed: false }))
            })
        };
        Some((ctx.discrete(), build()))
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<Report, Error> {
        let backend = self.backend.as_mut().ok_or(Error::NotFactorized)?;
        let method = ldlt_name(backend);
        let csr = &self.csr;
        self.checked.solve(method, rhs, |x, y| csr.spmv(x, y), |x| match backend {
            #[cfg(accelerate)]
            LdltBackend::Accelerate(f) => f.solve_in_place(x),
            #[cfg(feature = "vulkan")]
            LdltBackend::GpuF64(g) => g.f.solve_in_place(x),
            #[cfg(feature = "vulkan")]
            LdltBackend::GpuEf64(g) => g.f.solve_in_place(x),
            LdltBackend::Faer(f) => {
                f.solve_in_place(x);
                Ok(())
            }
            LdltBackend::Lu(g, _) => g.solve_in_place(x).map(|_| ()),
        })
    }
}

fn refactor(backend: &mut LdltBackend, values: &[f64]) -> Result<(), Error> {
    match backend {
        #[cfg(accelerate)]
        LdltBackend::Accelerate(f) => f.factorize(values),
        #[cfg(feature = "vulkan")]
        LdltBackend::GpuF64(g) => g.factorize(values),
        #[cfg(feature = "vulkan")]
        LdltBackend::GpuEf64(g) => g.factorize(values),
        LdltBackend::Faer(f) => f.factorize(values).map(|_| ()),
        LdltBackend::Lu(g, source) => g.factorize(&source.iter().map(|&p| values[p]).collect::<Vec<_>>()),
    }
}

fn ldlt_name(backend: &LdltBackend) -> &'static str {
    match backend {
        #[cfg(accelerate)]
        LdltBackend::Accelerate(_) => "Accelerate LDLᵀ (threshold pivoting)",
        #[cfg(feature = "vulkan")]
        LdltBackend::GpuF64(_) => "Vulkan LDLᵀ (signed, f64)",
        #[cfg(feature = "vulkan")]
        LdltBackend::GpuEf64(_) => "Vulkan LDLᵀ (signed, emulated f64)",
        LdltBackend::Faer(f) if f.info().regularized > 0 => "faer LDLᵀ (regularized)",
        LdltBackend::Faer(_) => "faer LDLᵀ",
        LdltBackend::Lu(..) => "LU of the whole matrix",
    }
}

#[cfg(feature = "vulkan")]
impl<T: crate::vulkan::DeviceReal> Gpu<T> {
    fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.perturbed = self.f.factorize(values)?.perturbed;
        Ok(())
    }

    /// The signs given, which every pivot has unless one was floored.
    fn inertia(&self, n: usize) -> Option<Inertia> {
        (!self.perturbed).then_some(Inertia { positive: n - self.negative, negative: self.negative, zero: 0 })
    }
}

/// The whole matrix's pattern (compressed sparse columns) from its lower triangle, and for each
/// entry the lower-triangle entry it copies.
fn whole_matrix(col_ptr: &[usize], row_idx: &[usize]) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = col_ptr.len() - 1;
    let mut columns: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for j in 0..n {
        for (p, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
            columns[j].push((i, p));
            if i != j {
                columns[i].push((j, p));
            }
        }
    }
    let mut cp = vec![0];
    let mut ri = Vec::new();
    let mut source = Vec::new();
    for column in &mut columns {
        column.sort_unstable();
        ri.extend(column.iter().map(|e| e.0));
        source.extend(column.iter().map(|e| e.1));
        cp.push(ri.len());
    }
    (cp, ri, source)
}
