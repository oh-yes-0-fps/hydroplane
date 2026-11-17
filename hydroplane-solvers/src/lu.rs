//! Sparse `L U` of a general square matrix (nonsymmetric: convection, friction, follower loads,
//! upwinded transport), given whole in compressed sparse columns.
//!
//! [`FrontalLu`] is the fast one where the pattern is nearly symmetric (FEM): it keeps the
//! ordering and supernodes of a Cholesky of `A + Aᵀ` and pivots within supernodes, with a
//! [`Matching`] first for small or zero diagonals. [`CpuLu`] (`faer`) pivots freely and so
//! handles any nonsingular matrix, at several times the cost on such patterns.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu::{LuRef, LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu};
use faer::sparse::linalg::{SupernodalThreshold, colamd};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Conj, MatMut, Par};

use rayon::prelude::*;

use crate::cholesky::{Analysis, Error, Real};
use crate::sparse::SellMatrix;
use crate::cholesky::Ordering;
use crate::iterative::{Refinement, Report};

/// Supernodal LU with partial (row) pivoting on the CPU, `faer`, after a COLAMD column ordering.
/// The ordering and supernodes are computed once per pattern in [`Self::new`].
pub struct CpuLu<T: Real> {
    pub refinement: Refinement,
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    symbolic: SymbolicLu<usize>,
    numeric: NumericLu<usize, T>,
    factored: bool,
    matrix: Option<SellMatrix>,
}

impl<T: Real> CpuLu<T> {
    /// The pattern of the whole matrix: column `j`'s rows are `row_idx[col_ptr[j]..col_ptr[j + 1]]`.
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, col_ptr, None, row_idx);
        let params = LuSymbolicParams {
            colamd_params: colamd::Control::default(),
            supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
            supernodal_params: Default::default(),
        };
        let symbolic = factorize_symbolic_lu(pattern, params).map_err(|_| Error::Pattern)?;
        Ok(Self { refinement: Refinement::default(), n, col_ptr: col_ptr.to_vec(), row_idx: row_idx.to_vec(), symbolic, numeric: NumericLu::new(), factored: false, matrix: None })
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// Factors the matrix whose values (in the pattern's order) are `values`, keeping them for
    /// [`Self::solve_refined`].
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        let n = self.n;
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, &self.col_ptr, None, &self.row_idx);
        let converted: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
        let mat = SparseColMatRef::new(pattern, &converted[..]);
        let par = Par::rayon(0);
        let mut scratch = MemBuffer::new(self.symbolic.factorize_numeric_lu_scratch::<T>(par, Default::default()));
        self.factored = false;
        self.symbolic
            .factorize_numeric_lu(&mut self.numeric, mat, par, MemStack::new(&mut scratch), Default::default())
            .map_err(|_| Error::Singular)?;
        self.factored = true;
        // `faer` reports a structurally singular matrix, but factors a numerically singular one
        // with a zero pivot; one solve exposes it as a non-finite entry.
        let mut probe = vec![1.0; n];
        self.solve_in_place(&mut probe);
        if probe.iter().any(|v| !v.is_finite()) {
            self.factored = false;
            return Err(Error::Singular);
        }
        let (col_ptr, row_idx) = (&self.col_ptr, &self.row_idx);
        self.matrix.get_or_insert_with(|| SellMatrix::from_csc(n, col_ptr, row_idx)).set_values(values);
        Ok(())
    }

    fn factor(&self) -> LuRef<'_, usize, T> {
        assert!(self.factored, "factorize first");
        LuRef::new_unchecked(&self.symbolic, &self.numeric)
    }

    /// `rhs ← A⁻¹ rhs`, in precision `T`.
    pub fn solve_in_place(&self, rhs: &mut [f64]) {
        self.solve(rhs, false);
    }

    /// `rhs ← A⁻ᵀ rhs`: the adjoint solve of sensitivities and of BiCG-type methods.
    pub fn solve_transpose_in_place(&self, rhs: &mut [f64]) {
        self.solve(rhs, true);
    }

    fn solve(&self, rhs: &mut [f64], transpose: bool) {
        let par = Par::Seq;
        let mut x: Vec<T> = rhs.iter().map(|&v| T::from_f64(v)).collect();
        let view = MatMut::from_column_major_slice_mut(&mut x, self.n, 1);
        if transpose {
            let mut scratch = MemBuffer::new(self.symbolic.solve_transpose_in_place_scratch::<T>(1, par));
            self.factor().solve_transpose_in_place_with_conj(Conj::No, view, par, MemStack::new(&mut scratch));
        } else {
            let mut scratch = MemBuffer::new(self.symbolic.solve_in_place_scratch::<T>(1, par));
            self.factor().solve_in_place_with_conj(Conj::No, view, par, MemStack::new(&mut scratch));
        }
        for (r, v) in rhs.iter_mut().zip(&x) {
            *r = v.to_f64();
        }
    }

    /// `rhs ← A⁻¹ rhs` to [`Self::refinement`]'s tolerance by GMRES on the true matrix in `f64`,
    /// preconditioned by the factor: `f64` accuracy from an `f32` factor, or recovery of the
    /// digits partial pivoting lost on a badly scaled matrix.
    pub fn solve_refined(&self, rhs: &mut [f64]) -> Report {
        let matrix = self.matrix.as_ref().expect("factorize first");
        self.refinement.run(|x, y| matrix.spmv(x, y), |z| self.solve_in_place(z), rhs)
    }
}

/// Settings of a [`FrontalLu`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LuOptions {
    /// The ordering of `A + Aᵀ` (with the rows matched first, if they are).
    pub ordering: Ordering,
    /// Permute and scale rows by a [`Matching`] before factoring: `None` (the default) decides at
    /// the first factorization, matching when some diagonal entry is missing or under a tenth
    /// of its column's largest.
    pub matching: Option<bool>,
}

/// Multifrontal `P A = L U` over the symmetric structure of `A + Aᵀ`: the ordering, supernodes
/// and elimination tree of the Cholesky [`Analysis`] (nested dissection by default), with rows
/// pivoted only within each supernode and a pivot smaller than `√ε · max |aᵢⱼ|` perturbed to that
/// size rather than fail. On a nonsymmetric matrix whose structure is nearly symmetric (FEM with
/// convection, friction or follower loads) that keeps the fill and parallelism of a Cholesky,
/// where `faer`'s LU ([`CpuLu`]) sizes its factor for any row pivoting. The PARDISO approach:
/// [`Self::solve_refined`] recovers the digits a restricted or perturbed pivot costs.
pub struct FrontalLu<T: Real> {
    pub refinement: Refinement,
    options: LuOptions,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    inner: Option<Box<Frontal<T>>>,
}

#[allow(private_bounds)]
impl<T: Field> FrontalLu<T> {
    /// The pattern of the whole matrix in compressed sparse columns.
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::with_options(col_ptr, row_idx, LuOptions::default())
    }

    pub fn with_options(col_ptr: &[usize], row_idx: &[usize], options: LuOptions) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        if row_idx.len() < col_ptr[n] || row_idx[..col_ptr[n]].iter().any(|&i| i >= n) {
            return Err(Error::Pattern);
        }
        Ok(Self { refinement: Refinement::default(), options, col_ptr: col_ptr.to_vec(), row_idx: row_idx.to_vec(), inner: None })
    }

    pub fn dim(&self) -> usize {
        self.col_ptr.len() - 1
    }

    /// The symbolic analysis, once the first factorization has made it.
    pub fn analysis(&self) -> Option<&Analysis> {
        self.inner.as_ref().map(|f| &f.analysis)
    }

    /// Pivots perturbed in the last factorization.
    pub fn perturbed(&self) -> usize {
        self.inner.as_ref().map_or(0, |f| f.perturbed)
    }

    /// Whether the rows are matched (decided at the first factorization).
    pub fn matched(&self) -> bool {
        self.inner.as_ref().is_some_and(|f| f.matching.is_some())
    }

    /// Factors the matrix whose values (in the pattern's order) are `values`; the first call
    /// also orders and analyses the pattern.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        if self.inner.is_none() {
            let (col_ptr, row_idx) = (&self.col_ptr[..], &self.row_idx[..]);
            let matching = match self.options.matching.unwrap_or_else(|| weak_diagonal(col_ptr, row_idx, values)) {
                true => Some(Matching::new(col_ptr, row_idx, values)?),
                false => None,
            };
            self.inner = Some(Box::new(Frontal::build(col_ptr, row_idx, matching, self.options.ordering.clone())?));
        }
        self.inner.as_mut().expect("built").factorize(values)
    }

    /// `rhs ← A⁻¹ rhs` with the factor alone (exact up to rounding unless pivots were perturbed).
    pub fn solve_in_place(&self, rhs: &mut [f64]) {
        self.inner.as_ref().expect("factorize first").solve_in_place(rhs);
    }

    /// `rhs ← A⁻¹ rhs` to [`Self::refinement`]'s tolerance by GMRES on the true matrix in `f64`,
    /// preconditioned by the factor.
    pub fn solve_refined(&self, rhs: &mut [f64]) -> Report {
        let inner = self.inner.as_ref().expect("factorize first");
        let matrix = inner.matrix.as_ref().expect("factorize first");
        self.refinement.run(|x, y| matrix.spmv(x, y), |z| inner.solve_in_place(z), rhs)
    }
}

/// Whether some column's diagonal entry is missing or under a tenth of the column's largest:
/// static pivoting then wants a matching first.
pub(crate) fn weak_diagonal(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> bool {
    (0..col_ptr.len() - 1).any(|j| {
        let range = col_ptr[j]..col_ptr[j + 1];
        let largest = values[range.clone()].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let diagonal = range.clone().find(|&p| row_idx[p] == j).map_or(0.0, |p| values[p].abs());
        diagonal < 0.1 * largest
    })
}

/// [`FrontalLu`]'s factorization once its pattern is ordered and analysed.
struct Frontal<T: Real> {
    n: usize,
    analysis: Analysis,
    perm: Vec<usize>,
    /// The permuted matrix by columns and by rows: `(permuted index, value index)`.
    by_col: Vec<Vec<(u32, u32)>>,
    by_row: Vec<Vec<(u32, u32)>>,
    /// Supernodes grouped by level of the elimination tree, children before parents.
    levels: Vec<Vec<usize>>,
    children: Vec<Vec<usize>>,
    /// For each supernode, where its update block's rows land in its parent's front.
    relative: Vec<Vec<u32>>,
    /// Each supernode's front columns (`L` below the diagonal, `U₁₁` on and above) in the
    /// [`Analysis`]'s factor layout, its `U₁₂` block (`k × m`, at `u_offset`), and its row
    /// order (`ipiv[begin + i]`: the supernode row that pivot `i` came from).
    lu: Vec<T>,
    u12: Vec<T>,
    u_offset: Vec<usize>,
    ipiv: Vec<u32>,
    /// Front and update buffers kept between supernodes and factorizations.
    pool: std::sync::Mutex<Vec<Vec<T>>>,
    perturbed: usize,
    factored: bool,
    matrix: Option<SellMatrix>,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    matching: Option<Matching>,
}

/// The arithmetic the dense front kernels need of a [`Real`].
trait Field: Real + core::ops::Add<Output = Self> + core::ops::Sub<Output = Self> + core::ops::Mul<Output = Self> + core::ops::Div<Output = Self> + core::ops::AddAssign + core::ops::SubAssign + core::ops::DivAssign + PartialOrd {}

impl<T: Real + core::ops::Add<Output = T> + core::ops::Sub<Output = T> + core::ops::Mul<Output = T> + core::ops::Div<Output = T> + core::ops::AddAssign + core::ops::SubAssign + core::ops::DivAssign + PartialOrd> Field for T {}

/// Panel width of the dense front factorization.
const PANEL: usize = 32;
/// Fronts from this size use threaded dense kernels.
const PARALLEL_FRONT: usize = 384;

/// A pointer for tasks that write disjoint ranges of one buffer.
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);

// Each task writes only its own supernode's ranges.
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    /// # Safety
    /// No other live reference may overlap `start..start + len`.
    unsafe fn slice<'a>(self, start: usize, len: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

impl<T: Field> Frontal<T> {
    fn build(col_ptr: &[usize], row_idx: &[usize], matching: Option<Matching>, ordering: Ordering) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let row_of = |i: usize| matching.as_ref().map_or(i, |m| m.column[i]);
        let mut lower: Vec<Vec<usize>> = (0..n).map(|j| vec![j]).collect();
        for j in 0..n {
            for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if i >= n {
                    return Err(Error::Pattern);
                }
                let i = row_of(i);
                let (r, c) = if i >= j { (i, j) } else { (j, i) };
                lower[c].push(r);
            }
        }
        let mut sym_ptr = vec![0];
        let mut sym_idx = Vec::new();
        for col in &mut lower {
            col.sort_unstable();
            col.dedup();
            sym_idx.extend_from_slice(col);
            sym_ptr.push(sym_idx.len());
        }
        let analysis = Analysis::with_ordering(&sym_ptr, &sym_idx, ordering)?;
        let perm = analysis.perm();
        let mut inv = vec![0; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let mut by_col = vec![Vec::new(); n];
        let mut by_row = vec![Vec::new(); n];
        for j in 0..n {
            for p in col_ptr[j]..col_ptr[j + 1] {
                let (r, c) = (inv[row_of(row_idx[p])], inv[j]);
                by_col[c].push((r as u32, p as u32));
                by_row[r].push((c as u32, p as u32));
            }
        }
        let parents = analysis.parents();
        let sn = analysis.supernodal();
        let ns = parents.len();
        let mut children = vec![Vec::new(); ns];
        let mut relative = vec![Vec::new(); ns];
        let mut position = vec![u32::MAX; n];
        for (s, &p) in parents.iter().enumerate() {
            if p == usize::MAX {
                continue;
            }
            children[p].push(s);
            let (begin, end) = (sn.supernode_begin()[p], sn.supernode_end()[p]);
            for (i, g) in (begin..end).chain(sn.supernode(p).pattern().iter().copied()).enumerate() {
                position[g] = i as u32;
            }
            relative[s] = sn.supernode(s).pattern().iter().map(|&g| position[g]).collect();
            if relative[s].contains(&u32::MAX) {
                return Err(Error::Pattern);
            }
        }
        let mut depth = vec![0usize; ns];
        for s in 0..ns {
            if parents[s] != usize::MAX {
                depth[parents[s]] = depth[parents[s]].max(depth[s] + 1);
            }
        }
        let mut levels = vec![Vec::new(); depth.iter().max().map_or(0, |d| d + 1)];
        for (s, &d) in depth.iter().enumerate() {
            levels[d].push(s);
        }
        let mut u_offset = vec![0usize; ns + 1];
        for s in 0..ns {
            u_offset[s + 1] = u_offset[s] + (sn.supernode_end()[s] - sn.supernode_begin()[s]) * sn.supernode(s).pattern().len();
        }
        let zero = T::from_f64(0.0);
        Ok(Self {
            n,
            lu: vec![zero; analysis.factor_len()],
            u12: vec![zero; u_offset[ns]],
            u_offset,
            ipiv: vec![0; n],
            analysis,
            perm,
            by_col,
            by_row,
            levels,
            children,
            relative,
            pool: std::sync::Mutex::new(Vec::new()),
            perturbed: 0,
            factored: false,
            matrix: None,
            col_ptr: col_ptr.to_vec(),
            row_idx: row_idx.to_vec(),
            matching,
        })
    }

    fn buffer(&self, len: usize) -> Vec<T> {
        let mut b = self.pool.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_default();
        b.clear();
        b.resize(len, T::from_f64(0.0));
        b
    }

    fn give_back(&self, b: Vec<T>) {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(b);
    }

    fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        let n = self.n;
        let scaled: Vec<f64>;
        let original = values;
        let values = match &self.matching {
            Some(m) => {
                scaled = (0..n)
                    .flat_map(|j| (self.col_ptr[j]..self.col_ptr[j + 1]).map(move |p| (p, j)))
                    .map(|(p, j)| values[p] * m.row_scale[self.row_idx[p]] * m.col_scale[j])
                    .collect();
                &scaled[..]
            }
            None => values,
        };
        let largest = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let eps = if T::F64 { f64::EPSILON } else { f32::EPSILON as f64 };
        let tau = T::from_f64(eps.sqrt() * largest.max(f64::MIN_POSITIVE));
        let ns = self.analysis.supernodal().n_supernodes();
        let mut updates: Vec<Option<Vec<T>>> = vec![None; ns];
        let mut lu = std::mem::take(&mut self.lu);
        let mut u12 = std::mem::take(&mut self.u12);
        let mut ipiv = std::mem::take(&mut self.ipiv);
        let out = (Shared(lu.as_mut_ptr()), Shared(u12.as_mut_ptr()), Shared(ipiv.as_mut_ptr()));
        let mut perturbed = 0;
        for level in &self.levels {
            let threaded = level.len() < rayon::current_num_threads();
            let done: Vec<(usize, Vec<T>, usize)> = level
                .par_iter()
                .map(|&s| {
                    let (update, count) = self.factor_front(s, values, tau, &updates, threaded, out);
                    (s, update, count)
                })
                .collect();
            for (s, update, count) in done {
                for &c in &self.children[s] {
                    if let Some(b) = updates[c].take() {
                        self.give_back(b);
                    }
                }
                updates[s] = Some(update);
                perturbed += count;
            }
        }
        for b in updates.into_iter().flatten() {
            self.give_back(b);
        }
        self.lu = lu;
        self.u12 = u12;
        self.ipiv = ipiv;
        self.perturbed = perturbed;
        self.factored = true;
        let (col_ptr, row_idx) = (&self.col_ptr, &self.row_idx);
        self.matrix.get_or_insert_with(|| SellMatrix::from_csc(n, col_ptr, row_idx)).set_values(original);
        let mut probe = vec![1.0; n];
        self.solve_in_place(&mut probe);
        if probe.iter().any(|v| !v.is_finite()) {
            self.factored = false;
            return Err(Error::Singular);
        }
        Ok(())
    }

    /// Adds supernode `s`'s entries of `A` and its children's updates into its front, held as
    /// its first `k` columns (`left`, the factor's storage) and the rest (`right`), both with
    /// `f` rows and zeroed.
    #[allow(clippy::too_many_arguments)]
    fn assemble(&self, s: usize, values: &[f64], updates: &[Option<Vec<T>>], left: &mut [T], right: &mut [T], threaded: bool) {
        let sn = self.analysis.supernodal();
        let (begin, end) = (sn.supernode_begin()[s], sn.supernode_end()[s]);
        let pattern = sn.supernode(s).pattern();
        let k = end - begin;
        let f = k + pattern.len();
        let local = |g: usize| if g < end { g - begin } else { k + pattern.partition_point(|&p| p < g) };
        for c in 0..k {
            for &(r, p) in &self.by_col[begin + c] {
                if r as usize >= begin {
                    let i = local(r as usize);
                    left[i + c * f] += T::from_f64(values[p as usize]);
                }
            }
            for &(col, p) in &self.by_row[begin + c] {
                if col as usize >= end {
                    let j = local(col as usize) - k;
                    right[c + j * f] += T::from_f64(values[p as usize]);
                }
            }
        }
        let columns = (Shared(left.as_mut_ptr()), Shared(right.as_mut_ptr()));
        for &child in &self.children[s] {
            let rel = &self.relative[child];
            let update = updates[child].as_ref().expect("children first");
            let kc = sn.supernode_end()[child] - sn.supernode_begin()[child];
            let fc = kc + rel.len();
            let add = |(b, &j): (usize, &u32)| {
                let j = j as usize;
                // SAFETY: each of the child's columns lands in a different front column.
                let column = unsafe { if j < k { columns.0.slice(j * f, f) } else { columns.1.slice((j - k) * f, f) } };
                for (&i, &u) in rel.iter().zip(&update[b * fc + kc..(b + 1) * fc]) {
                    column[i as usize] += u;
                }
            };
            if threaded && f >= PARALLEL_FRONT {
                rel.par_iter().enumerate().for_each(add);
            } else {
                rel.iter().enumerate().for_each(add);
            }
        }
    }

    /// Assembles supernode `s`'s front, eliminates its columns into the factor, and returns its
    /// right part (rows `k..` of which are the update for the parent) and the pivots perturbed.
    fn factor_front(&self, s: usize, values: &[f64], tau: T, updates: &[Option<Vec<T>>], threaded: bool, out: (Shared<T>, Shared<T>, Shared<u32>)) -> (Vec<T>, usize) {
        let sn = self.analysis.supernodal();
        let (begin, end) = (sn.supernode_begin()[s], sn.supernode_end()[s]);
        let k = end - begin;
        let m = sn.supernode(s).pattern().len();
        let f = k + m;
        let zero = T::from_f64(0.0);
        let big = threaded && f >= PARALLEL_FRONT;
        let par = if big { Par::rayon(0) } else { Par::Seq };
        // SAFETY: supernode `s` alone owns its factor columns, its `U₁₂` block and its pivots.
        let (left, u12, ipiv) = unsafe { (out.0.slice(sn.col_ptr_for_val()[s], f * k), out.1.slice(self.u_offset[s], k * m), out.2.slice(begin, k)) };
        let mut right = self.buffer(0);
        if big {
            right.reserve(f * m);
            right.spare_capacity_mut()[..f * m].par_chunks_mut(f).for_each(|c| c.iter_mut().for_each(|v| _ = v.write(zero)));
            // SAFETY: the first `f · m` entries were just written.
            unsafe { right.set_len(f * m) };
            left.par_chunks_mut(f).for_each(|c| c.fill(zero));
        } else {
            right.resize(f * m, zero);
            left.fill(zero);
        }
        self.assemble(s, values, updates, left, &mut right, threaded);
        let perturbed = match eliminate_blocked(left, &mut right, f, k, tau, ipiv, par) {
            Some(count) => count,
            None => {
                let mut front = vec![zero; f * f];
                left.fill(zero);
                right.fill(zero);
                self.assemble(s, values, updates, left, &mut right, false);
                front[..f * k].copy_from_slice(left);
                front[f * k..].copy_from_slice(&right);
                let mut swaps = vec![0u32; k];
                let count = eliminate(&mut front, f, k, tau, &mut swaps, Par::Seq);
                left.copy_from_slice(&front[..f * k]);
                right.copy_from_slice(&front[f * k..]);
                for (i, p) in ipiv.iter_mut().enumerate() {
                    *p = i as u32;
                }
                for (c, &r) in swaps.iter().enumerate() {
                    ipiv.swap(c, r as usize);
                }
                count
            }
        };
        for j in 0..m {
            u12[j * k..(j + 1) * k].copy_from_slice(&right[j * f..j * f + k]);
        }
        (right, perturbed)
    }

    /// `rhs ← A⁻¹ rhs`. The elimination tree's levels run in parallel: going up, each supernode hands its
    /// ancestors' share of the forward substitution to its parent as a vector, as the
    /// factorization hands up update blocks; going down, each reads only finished ancestors.
    fn solve_in_place(&self, rhs: &mut [f64]) {
        assert!(self.factored, "factorize first");
        let sn = self.analysis.supernodal();
        let ns = sn.n_supernodes();
        let zero = T::from_f64(0.0);
        if let Some(m) = &self.matching {
            let mut moved = vec![0.0; self.n];
            for (i, &j) in m.column.iter().enumerate() {
                moved[j] = rhs[i] * m.row_scale[i];
            }
            rhs.copy_from_slice(&moved);
        }
        let mut x: Vec<T> = self.perm.iter().map(|&old| T::from_f64(rhs[old])).collect();
        let xs = Shared(x.as_mut_ptr());
        let shape = |s: usize| {
            let (begin, end) = (sn.supernode_begin()[s], sn.supernode_end()[s]);
            let pattern = sn.supernode(s).pattern();
            let k = end - begin;
            (begin, k, pattern, k + pattern.len(), &self.lu[sn.col_ptr_for_val()[s]..][..(k + pattern.len()) * k])
        };
        let mut carried: Vec<Option<Vec<T>>> = vec![None; ns];
        for level in &self.levels {
            let done: Vec<(usize, Vec<T>)> = level
                .par_iter()
                .map(|&s| {
                    let (begin, k, _, f, lu) = shape(s);
                    // SAFETY: supernode `s` alone writes its own unknowns.
                    let own = unsafe { xs.slice(begin, k) };
                    let mut w = vec![zero; f];
                    w[..k].copy_from_slice(own);
                    for &child in &self.children[s] {
                        let rel = &self.relative[child];
                        let up = carried[child].as_ref().expect("children first");
                        for (&i, &u) in rel.iter().zip(&up[up.len() - rel.len()..]) {
                            w[i as usize] += u;
                        }
                    }
                    let mut v = w.clone();
                    for (i, &p) in self.ipiv[begin..begin + k].iter().enumerate() {
                        v[i] = w[p as usize];
                    }
                    for c in 0..k {
                        let y = v[c];
                        let col = &lu[c * f..(c + 1) * f];
                        for i in c + 1..f {
                            v[i] -= col[i] * y;
                        }
                    }
                    own.copy_from_slice(&v[..k]);
                    (s, v)
                })
                .collect();
            for (s, v) in done {
                for &c in &self.children[s] {
                    carried[c] = None;
                }
                carried[s] = Some(v);
            }
        }
        for level in self.levels.iter().rev() {
            level.par_iter().for_each(|&s| {
                let (begin, k, pattern, f, lu) = shape(s);
                let u12 = &self.u12[self.u_offset[s]..self.u_offset[s + 1]];
                let mut seg = vec![zero; k];
                // SAFETY: only read: this supernode's unknowns (the forward result) and its
                // ancestors' (final, from earlier levels of this pass); nothing writes them now.
                for (c, v) in seg.iter_mut().enumerate() {
                    *v = unsafe { xs.0.add(begin + c).read() };
                }
                for (j, &g) in pattern.iter().enumerate() {
                    let v = unsafe { xs.0.add(g).read() };
                    for (c, x) in seg.iter_mut().enumerate() {
                        *x -= u12[c + j * k] * v;
                    }
                }
                for c in (0..k).rev() {
                    let v = seg[c] / lu[c + c * f];
                    seg[c] = v;
                    for i in 0..c {
                        seg[i] -= lu[i + c * f] * v;
                    }
                }
                // SAFETY: supernode `s` alone writes its own unknowns.
                unsafe { xs.slice(begin, k) }.copy_from_slice(&seg);
            });
        }
        for (new, &old) in self.perm.iter().enumerate() {
            rhs[old] = x[new].to_f64();
        }
        if let Some(m) = &self.matching {
            for (r, s) in rhs.iter_mut().zip(&m.col_scale) {
                *r *= s;
            }
        }
    }
}

/// As [`eliminate`] for a front whose pivots need no perturbation, on the front split into its
/// first `k` columns (`left`) and the rest (`right`): `faer`'s blocked partial pivoting LU of
/// `F₁₁`, then two triangular solves and one update of `F₂₂` with inner dimension `k`. `None`
/// when a pivot is below `tau`, the front then spoilt. `order[i]`: the row of `F₁₁` that pivot
/// `i` came from.
fn eliminate_blocked<T: Field>(left: &mut [T], right: &mut [T], f: usize, k: usize, tau: T, order: &mut [u32], par: Par) -> Option<usize> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    let zero = T::from_f64(0.0);
    let abs = |v: T| if v < zero { zero - v } else { v };
    let m = f - k;
    let mut fwd = vec![0usize; k];
    let mut inv = vec![0usize; k];
    let mut l = MatMut::from_column_major_slice_mut(left, f, k);
    {
        let mut scratch = MemBuffer::new(faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T>(k, k, par, Default::default()));
        faer::linalg::lu::partial_pivoting::factor::lu_in_place(l.rb_mut().submatrix_mut(0, 0, k, k), &mut fwd, &mut inv, par, MemStack::new(&mut scratch), Default::default());
    }
    if (0..k).any(|c| !matches!(abs(l[(c, c)]).partial_cmp(&tau), Some(core::cmp::Ordering::Greater | core::cmp::Ordering::Equal))) {
        return None;
    }
    for (o, &r) in order.iter_mut().zip(&fwd) {
        *o = r as u32;
    }
    if m == 0 {
        return Some(0);
    }
    let permute = |column: &mut [T]| {
        let rows: Vec<T> = fwd.iter().map(|&r| column[r]).collect();
        column[..k].copy_from_slice(&rows);
    };
    if par == Par::Seq {
        right.chunks_mut(f).for_each(permute);
    } else {
        right.par_chunks_mut(f).for_each(permute);
    }
    let r = MatMut::from_column_major_slice_mut(right, f, m);
    let (l11, l21) = l.split_at_row_mut(k);
    let (mut r1, r2) = r.split_at_row_mut(k);
    let mut l21 = l21;
    faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(l11.rb(), r1.rb_mut(), par);
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(l11.rb().transpose(), l21.rb_mut().transpose_mut(), par);
    faer::linalg::matmul::matmul(r2, faer::Accum::Add, l21.rb(), r1.rb(), T::from_f64(-1.0), par);
    Some(0)
}

/// Eliminates the first `k` columns of the `f × f` column-major `front` in place: `P F₁₁ = L U`
/// with rows swapped only among the first `k`, `U₁₂ = L₁₁⁻¹ P F₁₂`, `L₂₁ = F₂₁ U₁₁⁻¹`, and
/// `F₂₂ − L₂₁ U₁₂` left in the trailing block. Returns the number of perturbed pivots.
fn eliminate<T: Field>(front: &mut [T], f: usize, k: usize, tau: T, ipiv: &mut [u32], par: Par) -> usize {
    let zero = T::from_f64(0.0);
    let abs = |v: T| if v < zero { zero - v } else { v };
    let mut perturbed = 0;
    let mut p0 = 0;
    while p0 < k {
        let p1 = (p0 + PANEL).min(k);
        for c in p0..p1 {
            let mut best = c;
            for r in c + 1..k {
                if abs(front[r + c * f]) > abs(front[best + c * f]) {
                    best = r;
                }
            }
            ipiv[c] = best as u32;
            if best != c {
                for j in 0..f {
                    front.swap(c + j * f, best + j * f);
                }
            }
            let mut pivot = front[c + c * f];
            if abs(pivot) < tau {
                pivot = if pivot < zero { zero - tau } else { tau };
                front[c + c * f] = pivot;
                perturbed += 1;
            }
            for i in c + 1..f {
                front[i + c * f] /= pivot;
            }
            for j in c + 1..p1 {
                let u = front[c + j * f];
                if u != zero {
                    for i in c + 1..f {
                        front[i + j * f] -= front[i + c * f] * u;
                    }
                }
            }
        }
        if p1 < f {
            let mat = MatMut::from_column_major_slice_mut(front, f, f);
            let (tl, tr, bl, br) = mat.split_at_mut(p1, p1);
            let mut tr = tr.subrows_mut(p0, p1 - p0);
            faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(tl.rb().submatrix(p0, p0, p1 - p0, p1 - p0), tr.rb_mut(), par);
            faer::linalg::matmul::matmul(br, faer::Accum::Add, bl.rb().subcols(p0, p1 - p0), tr.rb(), T::from_f64(-1.0), par);
        }
        p0 = p1;
    }
    perturbed
}

/// A maximum-product matching of a square sparse matrix (MC64's job 5): a row for every column
/// such that the product of the matched entries' magnitudes is largest, and row and column
/// scalings under which every matched entry has magnitude 1 and no other entry exceeds 1.
/// Permuting rows onto their columns puts large entries on the diagonal, what pivoting without
/// row interchanges ([`FrontalLu`]) needs.
#[derive(Clone, Debug)]
pub struct Matching {
    /// `column[i]`: the column row `i` is matched to.
    pub column: Vec<usize>,
    pub row_scale: Vec<f64>,
    pub col_scale: Vec<f64>,
}

impl Matching {
    /// From the whole matrix in compressed sparse columns. [`Error::Singular`] when no perfect
    /// matching exists (structurally singular).
    pub fn new(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let col_max: Vec<f64> = (0..n).map(|j| values[col_ptr[j]..col_ptr[j + 1]].iter().fold(0.0f64, |m, v| m.max(v.abs()))).collect();
        let cost = |p: usize, j: usize| {
            let a = values[p].abs();
            if a > 0.0 { col_max[j].ln() - a.ln() } else { f64::INFINITY }
        };
        let mut u = vec![f64::INFINITY; n];
        for j in 0..n {
            for p in col_ptr[j]..col_ptr[j + 1] {
                u[row_idx[p]] = u[row_idx[p]].min(cost(p, j));
            }
        }
        if u.iter().any(|v| !v.is_finite()) {
            return Err(Error::Singular);
        }
        let mut v = vec![0.0f64; n];
        let mut row_of = vec![usize::MAX; n];
        let mut col_of = vec![usize::MAX; n];
        for j in 0..n {
            v[j] = (col_ptr[j]..col_ptr[j + 1]).map(|p| cost(p, j) - u[row_idx[p]]).fold(f64::INFINITY, f64::min);
            if !v[j].is_finite() {
                return Err(Error::Singular);
            }
            if let Some(p) = (col_ptr[j]..col_ptr[j + 1]).find(|&p| col_of[row_idx[p]] == usize::MAX && cost(p, j) - u[row_idx[p]] - v[j] <= 0.0) {
                col_of[row_idx[p]] = j;
                row_of[j] = row_idx[p];
            }
        }
        let mut dist = vec![f64::INFINITY; n];
        let mut from = vec![usize::MAX; n];
        let mut settled = vec![false; n];
        let mut touched: Vec<usize> = Vec::new();
        let mut done: Vec<usize> = Vec::new();
        let mut heap = std::collections::BinaryHeap::new();
        for j0 in 0..n {
            if row_of[j0] != usize::MAX {
                continue;
            }
            let relax = |j: usize, base: f64, dist: &mut [f64], from: &mut [usize], touched: &mut Vec<usize>, heap: &mut std::collections::BinaryHeap<(core::cmp::Reverse<Key>, usize)>, settled: &[bool], u: &[f64], v: &[f64]| {
                for (p, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
                    if settled[i] {
                        continue;
                    }
                    let d = base + (cost(p, j) - u[i] - v[j]).max(0.0);
                    if d < dist[i] {
                        if dist[i].is_infinite() {
                            touched.push(i);
                        }
                        dist[i] = d;
                        from[i] = j;
                        heap.push((core::cmp::Reverse(Key(d)), i));
                    }
                }
            };
            relax(j0, 0.0, &mut dist, &mut from, &mut touched, &mut heap, &settled, &u, &v);
            let mut free = None;
            while let Some((core::cmp::Reverse(Key(d)), i)) = heap.pop() {
                if settled[i] || d > dist[i] {
                    continue;
                }
                settled[i] = true;
                done.push(i);
                if col_of[i] == usize::MAX {
                    free = Some((i, d));
                    break;
                }
                relax(col_of[i], d, &mut dist, &mut from, &mut touched, &mut heap, &settled, &u, &v);
            }
            let Some((free_row, length)) = free else { return Err(Error::Singular) };
            for &i in &done {
                u[i] += dist[i] - length;
                if col_of[i] != usize::MAX {
                    v[col_of[i]] -= dist[i] - length;
                }
            }
            v[j0] += length;
            let mut i = free_row;
            loop {
                let j = from[i];
                let previous = row_of[j];
                row_of[j] = i;
                col_of[i] = j;
                if j == j0 {
                    break;
                }
                i = previous;
            }
            for &i in &touched {
                dist[i] = f64::INFINITY;
                from[i] = usize::MAX;
                settled[i] = false;
            }
            touched.clear();
            done.clear();
            heap.clear();
        }
        Ok(Self { column: col_of, row_scale: u.iter().map(|u| u.exp()).collect(), col_scale: (0..n).map(|j| v[j].exp() / col_max[j]).collect() })
    }
}

/// A distance in the matching's Dijkstra, ordered for the heap.
#[derive(Clone, Copy, PartialEq)]
struct Key(f64);

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
