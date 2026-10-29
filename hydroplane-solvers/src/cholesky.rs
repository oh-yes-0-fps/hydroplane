//! Sparse Cholesky `A = L Lᵀ` of a symmetric positive definite matrix given by its lower triangle
//! in compressed sparse columns.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::{
    CholeskySymbolicParams, LltRef, SymbolicCholesky, SymbolicCholeskyRaw, SymmetricOrdering, factorize_symbolic_cholesky,
    supernodal::SymbolicSupernodalCholesky,
};
use faer::sparse::linalg::{SupernodalThreshold, SymbolicSupernodalParams};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};

use std::sync::Arc;

use crate::sparse::SellMatrix;
use crate::iterative::Cg;

/// Supernode relaxation `(columns, zero fraction)` thresholds: amalgamated more freely than
/// `faer`'s default, for larger dense blocks (a few percent more fill, less time on FEM tangents).
const SUPERNODE_RELAX: &[(usize, f64)] = &[(8, 1.0), (32, 0.9), (96, 0.3), (usize::MAX, 0.1)];

/// Dofs from which factorizations and solves run threaded, on [`FACTOR_POOL`]: below, the
/// supernodes are too small for threads to pay.
const POOL_DOFS: usize = 20_000;
/// Timed factorizations on each pool before one is chosen.
const POOL_TRIALS: usize = 2;

static FACTOR_POOL: FactorPool = FactorPool { state: std::sync::Mutex::new(None) };

/// The thread pool big factorizations run on. A supernodal factorization is bound by the cores'
/// arithmetic, and on a machine with two hardware threads per core the global pool (one thread
/// per hardware thread) measured 20–25 % slower than a pool of one thread per core, while other
/// work wants every hardware thread. Rather than guess the topology, the first factorizations are
/// timed on the global pool and on a half-size one, and the faster keeps the job.
struct FactorPool {
    state: std::sync::Mutex<Option<PoolChoice>>,
}

struct PoolChoice {
    half: Option<rayon::ThreadPool>,
    global: Vec<f64>,
    halved: Vec<f64>,
    /// `Some(true)`: the half pool; `Some(false)`: the global one; `None`: still trying.
    chosen: Option<bool>,
}

impl FactorPool {
    /// Runs `job` on the pool the factorizations chose, or the global one before they have;
    /// never counted as a trial.
    fn run_chosen<R: Send>(&self, job: impl FnOnce() -> R + Send) -> R {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match state.as_ref().and_then(|c| (c.chosen == Some(true)).then_some(c.half.as_ref()).flatten()) {
            Some(pool) => pool.install(job),
            None => {
                drop(state);
                job()
            }
        }
    }

    fn run<R: Send>(&self, job: impl FnOnce() -> R + Send) -> R {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let choice = state.get_or_insert_with(|| {
            let threads = rayon::current_num_threads();
            let half = (threads >= 4).then(|| rayon::ThreadPoolBuilder::new().num_threads(threads / 2).build().ok()).flatten();
            PoolChoice { chosen: half.is_none().then_some(false), half, global: Vec::new(), halved: Vec::new() }
        });
        let use_half = match choice.chosen {
            Some(h) => h,
            None => choice.global.len() > choice.halved.len(),
        };
        let started = std::time::Instant::now();
        let out = match (&choice.half, use_half) {
            (Some(pool), true) => pool.install(job),
            _ => job(),
        };
        if choice.chosen.is_none() {
            let took = started.elapsed().as_secs_f64();
            if use_half { &mut choice.halved } else { &mut choice.global }.push(took);
            if choice.global.len() >= POOL_TRIALS && choice.halved.len() >= POOL_TRIALS {
                let best = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
                choice.chosen = Some(best(&choice.halved) < best(&choice.global));
            }
        }
        out
    }
}

/// A factorization's working precision.
pub trait Real: faer::traits::RealField + Copy + Send + Sync + 'static {
    const F64: bool;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    const F64: bool = false;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    const F64: bool = true;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Fill-reducing ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// Nested dissection ([`crate::order`]) of the graph of `block × block` blocks: 3 for a
    /// 3-dof-per-node FEM matrix numbered `3·node + component`.
    NestedDissection { block: usize },
    /// `faer`'s approximate minimum degree.
    Amd,
    /// `perm[new] = old`.
    Custom(Vec<usize>),
}

/// Nested dissection of the scalar graph.
impl Default for Ordering {
    fn default() -> Self {
        Self::NestedDissection { block: 1 }
    }
}

#[derive(Debug)]
pub enum Error {
    /// The matrix is not the lower triangle of a square pattern the ordering accepts.
    Pattern,
    /// A pivot was not positive: not positive definite in the working precision.
    NotPositiveDefinite,
    /// A pivot was zero: singular in the working precision, or needing a pivot order the
    /// factorization could not choose (see [`crate::ldlt::Pivoting`]).
    Singular,
    /// A device or memory failure.
    Device(String),
    /// A solve before any factorization.
    NotFactorized,
}

/// The symbolic factorization of one sparsity pattern: ordering, supernodes, and the factor's
/// layout. Every numeric factorization of matrices with this pattern reuses it; cloning it shares
/// it.
#[derive(Clone)]
pub struct Analysis {
    n: usize,
    col_ptr: Arc<[usize]>,
    row_idx: Arc<[usize]>,
    symbolic: Arc<SymbolicCholesky<usize>>,
}

impl Analysis {
    /// The lower triangle's pattern in compressed sparse columns, in the default [`Ordering`].
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::with_ordering(col_ptr, row_idx, Ordering::default())
    }

    pub fn with_ordering(col_ptr: &[usize], row_idx: &[usize], ordering: Ordering) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_idx);
        let fwd = match ordering {
            Ordering::NestedDissection { block } => Some(nested_dissection(n, col_ptr, row_idx, block)?),
            Ordering::Amd => None,
            Ordering::Custom(p) => Some(p),
        };
        let inv = fwd.as_ref().map(|f| {
            let mut inv = vec![0; n];
            for (new, &old) in f.iter().enumerate() {
                inv[old] = new;
            }
            inv
        });
        let ordering = match (&fwd, &inv) {
            (Some(f), Some(i)) => SymmetricOrdering::Custom(PermRef::new_checked(f, i, n)),
            _ => SymmetricOrdering::Amd,
        };
        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SUPERNODAL,
            supernodal_params: SymbolicSupernodalParams { relax: Some(SUPERNODE_RELAX) },
            ..Default::default()
        };
        let symbolic = factorize_symbolic_cholesky(pattern, Side::Lower, ordering, params).map_err(|_| Error::Pattern)?;
        Ok(Self { n, col_ptr: col_ptr.into(), row_idx: row_idx.into(), symbolic: Arc::new(symbolic) })
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// The structure `faer` factors with; its supernodal layout is the factor's layout on every
    /// backend.
    pub fn symbolic(&self) -> &SymbolicCholesky<usize> {
        &self.symbolic
    }

    pub fn supernodal(&self) -> &SymbolicSupernodalCholesky<usize> {
        match self.symbolic.raw() {
            SymbolicCholeskyRaw::Supernodal(s) => s,
            SymbolicCholeskyRaw::Simplicial(_) => unreachable!("the analysis forces a supernodal structure"),
        }
    }

    /// `perm[new] = old`.
    pub fn perm(&self) -> Vec<usize> {
        match self.symbolic.perm() {
            Some(p) => p.arrays().0.to_vec(),
            None => (0..self.n).collect(),
        }
    }

    /// Entries of `L`.
    pub fn factor_len(&self) -> usize {
        self.symbolic.len_val()
    }

    /// Each supernode's parent in the elimination tree (`usize::MAX` for a root).
    pub fn parents(&self) -> Vec<usize> {
        let sn = self.supernodal();
        let ns = sn.n_supernodes();
        let mut col_super = vec![0usize; self.n];
        for s in 0..ns {
            col_super[sn.supernode_begin()[s]..sn.supernode_end()[s]].fill(s);
        }
        (0..ns).map(|s| sn.supernode(s).pattern().first().map_or(usize::MAX, |&r| col_super[r])).collect()
    }

    /// Supernodes grouped into levels for a level-by-level factorization (children before
    /// parents), and the critical path that grouping costs in 32-column panel steps (each level
    /// its widest supernode). Of placing each supernode as early as possible (one level above its
    /// highest child) and as late as possible (one below its parent), the shorter.
    pub fn levels(&self) -> (Vec<usize>, usize) {
        let sn = self.supernodal();
        let ns = sn.n_supernodes();
        let parent = self.parents();
        let panels: Vec<usize> = (0..ns).map(|s| (sn.supernode_end()[s] - sn.supernode_begin()[s]).div_ceil(32)).collect();
        let mut early = vec![0usize; ns];
        for s in 0..ns {
            if parent[s] != usize::MAX {
                early[parent[s]] = early[parent[s]].max(early[s] + 1);
            }
        }
        let levels = early.iter().max().map_or(0, |l| l + 1);
        let mut late = vec![0usize; ns];
        for s in (0..ns).rev() {
            late[s] = if parent[s] == usize::MAX { levels - 1 } else { late[parent[s]] - 1 };
        }
        let steps = |assign: &[usize]| {
            let mut widest = vec![0usize; levels];
            for s in 0..ns {
                widest[assign[s]] = widest[assign[s]].max(panels[s]);
            }
            widest.iter().sum::<usize>()
        };
        let (a, b) = (steps(&early), steps(&late));
        if b < a { (late, b) } else { (early, a) }
    }

    /// Floating-point operations of the numeric factorization (multiply and add counted apart).
    pub fn flops(&self) -> f64 {
        let sn = self.supernodal();
        (0..sn.n_supernodes())
            .map(|s| {
                let k = (sn.supernode_end()[s] - sn.supernode_begin()[s]) as f64;
                let m = sn.supernode(s).pattern().len() as f64;
                k * k * k / 3.0 + k * k * m + k * m * m
            })
            .sum()
    }

    /// Nested dissection tried with `candidates` seeds ([`order::NestedDissection::seed`]), the
    /// one with the least `cost` kept: the separators' sizes barely move with the seed, but how
    /// they line up along the elimination tree does, and that is the critical path of a GPU
    /// factorization (see `vulkan::ordering_cost`). For the CPU, [`Self::flops`] is the cost.
    ///
    /// [`order::NestedDissection::seed`]: crate::order::NestedDissection::seed
    pub fn best_of(col_ptr: &[usize], row_idx: &[usize], block: usize, candidates: usize, cost: impl Fn(&Analysis) -> f64) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        if block == 0 || !n.is_multiple_of(block) {
            return Err(Error::Pattern);
        }
        let (row_ptr, col) = crate::order::node_graph(col_ptr, row_idx, block);
        let mut best: Option<(f64, Analysis)> = None;
        for seed in 0..candidates.max(1) {
            let p = crate::order::NestedDissection { seed, ..Default::default() };
            let perm = crate::order::nested_dissection_with(&row_ptr, &col, &p)
                .into_iter()
                .flat_map(|node| (0..block).map(move |c| block * node as usize + c))
                .collect();
            let analysis = Self::with_ordering(col_ptr, row_idx, Ordering::Custom(perm))?;
            let c = cost(&analysis);
            if best.as_ref().is_none_or(|(b, _)| c < *b) {
                best = Some((c, analysis));
            }
        }
        Ok(best.expect("at least one candidate").1)
    }

    /// `x = A⁻¹ b` in place from a factor in this analysis's layout.
    pub fn solve_in_place<T: Real>(&self, factor: &[T], rhs: &mut [T]) {
        let n = self.n;
        let mut run = |par: Par| {
            let mut scratch = MemBuffer::new(self.symbolic.solve_in_place_scratch::<T>(1, par));
            LltRef::new(&self.symbolic, factor).solve_in_place_with_conj(
                Conj::No,
                MatMut::from_column_major_slice_mut(rhs, n, 1),
                par,
                MemStack::new(&mut scratch),
            );
        };
        // Threaded on the factorizations' pool measured a few percent faster than sequential on
        // a whole tangent; on the global pool it was slower.
        if n >= POOL_DOFS { FACTOR_POOL.run_chosen(|| run(Par::rayon(0))) } else { run(Par::Seq) }
    }
}

fn nested_dissection(n: usize, col_ptr: &[usize], row_idx: &[usize], block: usize) -> Result<Vec<usize>, Error> {
    if block == 0 || !n.is_multiple_of(block) {
        return Err(Error::Pattern);
    }
    let (row_ptr, col) = crate::order::node_graph(col_ptr, row_idx, block);
    Ok(crate::order::nested_dissection(&row_ptr, &col)
        .into_iter()
        .flat_map(|node| (0..block).map(move |c| block * node as usize + c))
        .collect())
}

/// Supernodal factorization on the CPU (`faer`; big matrices on a thread pool chosen by timing),
/// in precision `T`.
pub struct CpuCholesky<T: Real> {
    analysis: Analysis,
    factor: Vec<T>,
    values: Vec<T>,
    buffer: std::sync::Mutex<Vec<T>>,
}

impl<T: Real> CpuCholesky<T> {
    pub fn new(analysis: &Analysis) -> Self {
        Self { factor: vec![T::from_f64(0.0); analysis.factor_len()], analysis: analysis.clone(), values: Vec::new(), buffer: Default::default() }
    }

    /// Factors the matrix whose lower-triangle values (in the analysis's pattern order) are
    /// `values`.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.values.clear();
        self.values.extend(values.iter().map(|&v| T::from_f64(v)));
        let analysis = &self.analysis;
        let n = analysis.n;
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, &analysis.col_ptr, None, &analysis.row_idx);
        let mat = SparseColMatRef::new(pattern, &self.values[..]);
        let par = Par::rayon(0);
        let mut scratch = MemBuffer::new(analysis.symbolic.factorize_numeric_llt_scratch::<T>(par, Default::default()));
        let factor = &mut self.factor;
        let mut run = || {
            analysis
                .symbolic
                .factorize_numeric_llt(factor, mat, Side::Lower, Default::default(), par, MemStack::new(&mut scratch), Default::default())
                .map(|_| ())
                .map_err(|_| Error::NotPositiveDefinite)
        };
        if n >= POOL_DOFS { FACTOR_POOL.run(run) } else { run() }
    }

    pub fn factor(&self) -> &[T] {
        &self.factor
    }

    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    /// `rhs ← A⁻¹ rhs`, solved in precision `T`.
    pub fn solve_in_place(&self, rhs: &mut [f64]) {
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        buffer.clear();
        buffer.extend(rhs.iter().map(|&r| T::from_f64(r)));
        self.analysis.solve_in_place(&self.factor, &mut buffer);
        for (r, &b) in rhs.iter_mut().zip(buffer.iter()) {
            *r = b.to_f64();
        }
    }
}

/// How a preconditioned conjugate-gradient solve ended.
#[derive(Clone, Copy, Debug)]
pub struct PcgReport {
    pub iterations: usize,
    /// `‖b − A x‖ / ‖b‖` as the iteration tracked it.
    pub relative_residual: f64,
    pub converged: bool,
}

/// When a conjugate-gradient solve on a cheap factor stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PcgOptions {
    /// `‖b − A x‖ / ‖b‖` to reach.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for PcgOptions {
    fn default() -> Self {
        Self { tolerance: 1e-8, max_iterations: 200 }
    }
}

/// `f64` conjugate gradient on the CPU preconditioned by an `f32` [`CpuCholesky`]: the CPU
/// counterpart of `vulkan::VulkanPcg`. The matrix-vector product runs over both triangles by rows
/// (rayon), the dot products in `f64`.
pub struct CpuPcg {
    pub options: PcgOptions,
    chol: CpuCholesky<f32>,
    matrix: SellMatrix,
}

impl CpuPcg {
    pub fn new(analysis: &Analysis) -> Self {
        Self::with_options(analysis, PcgOptions::default())
    }

    pub fn with_options(analysis: &Analysis, options: PcgOptions) -> Self {
        Self {
            options,
            chol: CpuCholesky::new(analysis),
            matrix: SellMatrix::from_lower_csc_symmetric(analysis.n, &analysis.col_ptr, &analysis.row_idx),
        }
    }

    /// Factors the preconditioner in `f32` and keeps the matrix in `f64` for the iteration.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.set_values(values);
        self.chol.factorize(values)
    }

    /// Keeps the matrix (lower triangle, the analysis's pattern order) for [`Self::solve_with`]
    /// without factoring it here: for a preconditioner factored elsewhere (a GPU).
    pub fn set_values(&mut self, values: &[f64]) {
        self.matrix.set_values(values);
    }

    pub fn cholesky(&self) -> &CpuCholesky<f32> {
        &self.chol
    }

    /// `rhs ← A⁻¹ rhs` to [`PcgOptions::tolerance`], preconditioned by this solver's `f32`
    /// factor.
    pub fn solve_in_place(&self, rhs: &mut [f64]) -> PcgReport {
        self.solve_with(rhs, |r, z| {
            z.copy_from_slice(r);
            self.chol.solve_in_place(z);
        })
    }

    /// As [`Self::solve_in_place`] with any preconditioner `z ← M⁻¹ r`, e.g. a GPU factor's solve.
    pub fn solve_with(&self, rhs: &mut [f64], precondition: impl FnMut(&[f64], &mut [f64])) -> PcgReport {
        self.solve_to(rhs, self.options, precondition)
    }

    /// With this solver's own factor, to `options` instead of [`Self::options`].
    pub(crate) fn solve_own(&self, rhs: &mut [f64], options: PcgOptions) -> PcgReport {
        self.solve_to(rhs, options, |r, z| {
            z.copy_from_slice(r);
            self.chol.solve_in_place(z);
        })
    }

    pub(crate) fn solve_to(&self, rhs: &mut [f64], options: PcgOptions, precondition: impl FnMut(&[f64], &mut [f64])) -> PcgReport {
        let b = rhs.to_vec();
        let mut a = |x: &[f64], y: &mut [f64]| self.matrix.spmv(x, y);
        let mut cg = Cg::with_options(crate::iterative::Options { tolerance: options.tolerance, max_iterations: options.max_iterations, ..Default::default() });
        let report = cg.solve(&mut a, &mut { precondition }, &b[..], rhs);
        PcgReport { iterations: report.iterations, relative_residual: report.relative_residual, converged: report.converged() }
    }
}
