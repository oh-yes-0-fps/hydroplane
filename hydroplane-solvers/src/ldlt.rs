//! Sparse `A = L D Lᵀ` of a symmetric matrix that need not be positive definite: saddle-point
//! (KKT) systems of constrained mechanics and interior-point methods, tangents past a limit point,
//! shifted operators. Same input and [`Analysis`] as the Cholesky solvers.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::perm::PermRef;
use faer::sparse::linalg::cholesky::{IntranodeLbltRef, LdltRef};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par, Side};

use crate::cholesky::{Analysis, Error, Real};
use crate::sparse::SellMatrix;
use crate::iterative::{Refinement, Report};

/// How pivots are chosen. The elimination order is always the [`Analysis`]'s.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Pivoting {
    /// Every pivot taken as it comes, with an expected sign each: a quasi-definite matrix
    /// `[H Bᵀ; B −C]` (`H`, `C` positive definite or semidefinite) factors stably in any order with
    /// `+1` for `H`'s rows and `−1` for `C`'s (`0`: no expectation, never replaced). A pivot of the wrong sign, or smaller in magnitude
    /// than `epsilon`, is replaced by `±delta` (counted in [`LdltInfo::regularized`]): the factor
    /// is then of a nearby matrix, and [`CpuLdlt::solve_refined`] recovers the exact solution.
    /// The interior-point default (QDLDL, Clarabel).
    Regularized { signs: Vec<i8>, delta: f64, epsilon: f64 },
    /// Every pivot taken as it comes, none replaced: for a matrix known to have no small pivots in
    /// the analysis's order (a negative definite or shifted operator). A zero pivot fails with
    /// [`Error::Singular`].
    Static,
    /// Bunch–Kaufman (the default): `1 × 1` or `2 × 2` pivots chosen for stability, but only among the columns
    /// of the same supernode. Handles indefiniteness without known signs, but not a zero diagonal
    /// block the size of a whole supernode (a constraint row the ordering eliminated first).
    #[default]
    BunchKaufman,
}

impl Pivoting {
    /// [`Self::Regularized`] with `delta = 1e-9` and `epsilon = 1e-13`, for a matrix scaled to
    /// entries around 1.
    pub fn regularized(signs: Vec<i8>) -> Self {
        Self::Regularized { signs, delta: 1e-9, epsilon: 1e-13 }
    }
}

/// A fill-reducing elimination order for a saddle-point matrix `[H Bᵀ; B −C]` given by its lower
/// triangle, with `signs` marking the constraint rows (negative). Plain fill-reducing orderings
/// eliminate a constraint first, its zero diagonal untouched, so every such pivot is zero and has
/// to be regularized. Here the primal unknowns are ordered (AMD) over the pattern of
/// `H + BᵀB`, which is the fill eliminating the constraints will cause, and each constraint is
/// placed right after the first of its primal neighbours: one eliminated neighbour already gives
/// it a negative pivot (`−b²/d` and so on), with the elimination's fill no worse than `H + BᵀB`'s.
/// After its last neighbour instead it would enter the row structure of every column in between,
/// several times the fill. Pass the result as [`crate::cholesky::Ordering::Custom`].
pub fn kkt_ordering(col_ptr: &[usize], row_idx: &[usize], signs: &[i8]) -> Result<Vec<usize>, Error> {
    let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
    assert_eq!(signs.len(), n, "one expected sign per row");
    let primal: Vec<usize> = (0..n).filter(|&i| signs[i] >= 0).collect();
    let mut local = vec![usize::MAX; n];
    for (k, &i) in primal.iter().enumerate() {
        local[i] = k;
    }
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edges: Vec<(usize, usize)> = (0..primal.len()).map(|k| (k, k)).collect();
    for j in 0..n {
        for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            match (local[i], local[j]) {
                (usize::MAX, usize::MAX) => {}
                (usize::MAX, p) => neighbours[i].push(p),
                (p, usize::MAX) => neighbours[j].push(p),
                (a, b) => edges.push((a.max(b), a.min(b))),
            }
        }
    }
    for list in &neighbours {
        for (x, &a) in list.iter().enumerate() {
            for &b in &list[..x] {
                edges.push((a.max(b), a.min(b)));
            }
        }
    }
    edges.sort_unstable_by_key(|&(r, c)| (c, r));
    edges.dedup();
    let m = primal.len();
    let mut cp = vec![0usize; m + 1];
    for &(_, c) in &edges {
        cp[c + 1] += 1;
    }
    for c in 0..m {
        cp[c + 1] += cp[c];
    }
    let ri: Vec<usize> = edges.iter().map(|e| e.0).collect();
    let order = Analysis::with_ordering(&cp, &ri, crate::cholesky::Ordering::Amd)?.perm();
    let mut position = vec![0usize; m];
    for (new, &k) in order.iter().enumerate() {
        position[k] = new;
    }
    let mut keyed: Vec<(usize, usize, usize)> = order.iter().map(|&k| (position[k], 0, primal[k])).collect();
    for (i, list) in neighbours.iter().enumerate() {
        if signs[i] < 0 {
            keyed.push((list.iter().map(|&p| position[p]).min().unwrap_or(0), 1, i));
        }
    }
    keyed.sort_unstable();
    Ok(keyed.into_iter().map(|(_, _, old)| old).collect())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LdltInfo {
    /// Pivots replaced under [`Pivoting::Regularized`].
    pub regularized: usize,
}

/// The signs of the pivots: by Sylvester's law, the eigenvalue signs of `A` (of the regularized
/// matrix, if pivots were replaced). An interior-point method checks the KKT matrix has as many
/// positive eigenvalues as primal variables; a Newton solver detects a non-convex step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Inertia {
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

/// Supernodal `L D Lᵀ` (or `L B Lᵀ` with Bunch–Kaufman's `2 × 2` blocks) on the CPU, `faer`.
pub struct CpuLdlt<T: Real> {
    pub refinement: Refinement,
    analysis: Analysis,
    pivoting: Pivoting,
    factor: Vec<T>,
    /// Bunch–Kaufman's `2 × 2` blocks' off-diagonals (zero for a `1 × 1` pivot), and the order
    /// the pivoting chose within each supernode.
    subdiag: Vec<T>,
    fwd: Vec<usize>,
    inv: Vec<usize>,
    signs: Vec<i8>,
    info: LdltInfo,
    matrix: Option<SellMatrix>,
}

impl<T: Real> CpuLdlt<T> {
    /// Bunch–Kaufman pivoting.
    pub fn new(analysis: &Analysis) -> Self {
        Self::with_pivoting(analysis, Pivoting::default())
    }

    pub fn with_pivoting(analysis: &Analysis, pivoting: Pivoting) -> Self {
        let n = analysis.dim();
        if let Pivoting::Regularized { signs, .. } = &pivoting {
            assert_eq!(signs.len(), n, "one expected sign per row");
        }
        let signs = match &pivoting {
            Pivoting::Regularized { signs, .. } => signs.clone(),
            _ => Vec::new(),
        };
        Self {
            refinement: Refinement::default(),
            analysis: analysis.clone(),
            pivoting,
            factor: vec![T::from_f64(0.0); analysis.factor_len()],
            subdiag: vec![T::from_f64(0.0); n],
            fwd: (0..n).collect(),
            inv: (0..n).collect(),
            signs,
            info: LdltInfo::default(),
            matrix: None,
        }
    }

    /// Factors the matrix whose lower-triangle values (in the analysis's pattern order) are
    /// `values`, keeping them for [`Self::solve_refined`].
    pub fn factorize(&mut self, values: &[f64]) -> Result<LdltInfo, Error> {
        let analysis = self.analysis.clone();
        let n = analysis.dim();
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, analysis.col_ptr(), None, analysis.row_idx());
        let converted: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
        let mat = SparseColMatRef::new(pattern, &converted[..]);
        let symbolic = analysis.symbolic();
        let par = Par::rayon(0);
        self.info = match &self.pivoting {
            Pivoting::Regularized { delta, epsilon, .. } => {
                let regularization = LdltRegularization {
                    dynamic_regularization_signs: Some(&self.signs),
                    dynamic_regularization_delta: T::from_f64(*delta),
                    dynamic_regularization_epsilon: T::from_f64(*epsilon),
                };
                let mut scratch = MemBuffer::new(symbolic.factorize_numeric_ldlt_scratch::<T>(par, Default::default()));
                symbolic
                    .factorize_numeric_ldlt(&mut self.factor, mat, Side::Lower, regularization, par, MemStack::new(&mut scratch), Default::default())
                    .map_err(|_| Error::Singular)?;
                LdltInfo { regularized: self.count_regularized(*delta) }
            }
            Pivoting::Static => {
                let mut scratch = MemBuffer::new(symbolic.factorize_numeric_ldlt_scratch::<T>(par, Default::default()));
                symbolic
                    .factorize_numeric_ldlt(&mut self.factor, mat, Side::Lower, Default::default(), par, MemStack::new(&mut scratch), Default::default())
                    .map_err(|_| Error::Singular)?;
                LdltInfo::default()
            }
            Pivoting::BunchKaufman => {
                let mut scratch = MemBuffer::new(symbolic.factorize_numeric_intranode_lblt_scratch::<T>(par, Default::default()));
                symbolic.factorize_numeric_intranode_lblt(
                    &mut self.factor,
                    &mut self.subdiag,
                    &mut self.fwd,
                    &mut self.inv,
                    mat,
                    Side::Lower,
                    par,
                    MemStack::new(&mut scratch),
                    Default::default(),
                );
                LdltInfo::default()
            }
        };
        if self.diagonal().any(|(d, _)| !d.is_finite()) || self.inertia().zero > 0 {
            return Err(Error::Singular);
        }
        self.matrix
            .get_or_insert_with(|| SellMatrix::from_lower_csc_symmetric(n, analysis.col_ptr(), analysis.row_idx()))
            .set_values(values);
        Ok(self.info)
    }

    /// `faer`'s sparse factorization returns no count of replaced pivots; they are exactly
    /// `±delta`, so they are counted from the factor.
    fn count_regularized(&self, delta: f64) -> usize {
        let delta = T::from_f64(delta).to_f64();
        self.diagonal().filter(|&(d, _)| d.abs() == delta).count()
    }

    pub fn info(&self) -> LdltInfo {
        self.info
    }

    /// Each pivot block's diagonal entry in the factor's order, with the off-diagonal of a
    /// `2 × 2` block on its first column (zero otherwise).
    fn diagonal(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        let sn = self.analysis.supernodal();
        (0..sn.n_supernodes()).flat_map(move |s| {
            let (begin, end) = (sn.supernode_begin()[s], sn.supernode_end()[s]);
            let rows = (end - begin) + sn.supernode(s).pattern().len();
            let base = sn.col_ptr_for_val()[s];
            (0..end - begin).map(move |k| (self.factor[base + k * rows + k].to_f64(), self.subdiag[begin + k].to_f64()))
        })
    }

    pub fn inertia(&self) -> Inertia {
        let mut inertia = Inertia::default();
        let pivots: Vec<(f64, f64)> = self.diagonal().collect();
        let mut k = 0;
        while k < pivots.len() {
            let (d, e) = pivots[k];
            if e != 0.0 && k + 1 < pivots.len() {
                let d2 = pivots[k + 1].0;
                let det = d * d2 - e * e;
                if det < 0.0 {
                    inertia.positive += 1;
                    inertia.negative += 1;
                } else if det > 0.0 {
                    if d + d2 > 0.0 { inertia.positive += 2 } else { inertia.negative += 2 }
                } else {
                    inertia.zero += 1;
                    if d + d2 > 0.0 { inertia.positive += 1 } else { inertia.negative += 1 }
                }
                k += 2;
            } else {
                match d.partial_cmp(&0.0) {
                    Some(core::cmp::Ordering::Greater) => inertia.positive += 1,
                    Some(core::cmp::Ordering::Less) => inertia.negative += 1,
                    _ => inertia.zero += 1,
                }
                k += 1;
            }
        }
        inertia
    }

    /// `rhs ← A⁻¹ rhs` with the factor alone, in precision `T`: exact up to rounding, unless
    /// pivots were replaced.
    pub fn solve_in_place(&self, rhs: &mut [f64]) {
        let n = self.analysis.dim();
        let symbolic = self.analysis.symbolic();
        let par = Par::Seq;
        let mut scratch = MemBuffer::new(symbolic.solve_in_place_scratch::<T>(1, par));
        let mut x: Vec<T> = rhs.iter().map(|&v| T::from_f64(v)).collect();
        let view = MatMut::from_column_major_slice_mut(&mut x, n, 1);
        match self.pivoting {
            Pivoting::BunchKaufman => IntranodeLbltRef::new(symbolic, &self.factor, &self.subdiag, PermRef::new_checked(&self.fwd, &self.inv, n))
                .solve_in_place_with_conj(Conj::No, view, par, MemStack::new(&mut scratch)),
            _ => LdltRef::new(symbolic, &self.factor).solve_in_place_with_conj(Conj::No, view, par, MemStack::new(&mut scratch)),
        }
        for (r, v) in rhs.iter_mut().zip(&x) {
            *r = v.to_f64();
        }
    }

    /// `rhs ← A⁻¹ rhs` to [`Self::refinement`]'s tolerance by GMRES on the true matrix in `f64`,
    /// preconditioned by the factor: solutions to the true matrix from a regularized or `f32`
    /// factor, in a few steps where fixed-point refinement can stall.
    pub fn solve_refined(&self, rhs: &mut [f64]) -> Report {
        let matrix = self.matrix.as_ref().expect("factorize first");
        self.refinement.run(|x, y| matrix.spmv(x, y), |z| self.solve_in_place(z), rhs)
    }
}
