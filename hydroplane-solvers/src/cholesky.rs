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

/// Supernode relaxation `(columns, zero fraction)` thresholds: amalgamated more freely than
/// `faer`'s default, for larger dense blocks (a few percent more fill, less time on FEM tangents).
const SUPERNODE_RELAX: &[(usize, f64)] = &[(8, 1.0), (32, 0.9), (96, 0.3), (usize::MAX, 0.1)];

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
#[derive(Clone, Debug)]
pub enum Ordering {
    /// Nested dissection ([`crate::order`]) of the graph of `block × block` blocks: 3 for a
    /// 3-dof-per-node FEM matrix numbered `3·node + component`.
    NestedDissection { block: usize },
    /// `faer`'s approximate minimum degree.
    Amd,
    /// `perm[new] = old`.
    Custom(Vec<usize>),
}

#[derive(Debug)]
pub enum Error {
    /// The matrix is not the lower triangle of a square pattern the ordering accepts.
    Pattern,
    /// A pivot was not positive: not positive definite in the working precision.
    NotPositiveDefinite,
    /// A device or memory failure.
    Device(String),
}

/// The symbolic factorization of one sparsity pattern: ordering, supernodes, and the factor's
/// layout. Every numeric factorization of matrices with this pattern reuses it.
pub struct Analysis {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    symbolic: SymbolicCholesky<usize>,
}

impl Analysis {
    pub fn new(col_ptr: &[usize], row_idx: &[usize], ordering: Ordering) -> Result<Self, Error> {
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
        Ok(Self { n, col_ptr: col_ptr.to_vec(), row_idx: row_idx.to_vec(), symbolic })
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

    /// `x = A⁻¹ b` in place from a factor in this analysis's layout.
    pub fn solve_in_place<T: Real>(&self, factor: &[T], rhs: &mut [T]) {
        let n = self.n;
        let mut scratch = MemBuffer::new(self.symbolic.solve_in_place_scratch::<T>(1, Par::rayon(0)));
        LltRef::new(&self.symbolic, factor).solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(rhs, n, 1),
            Par::rayon(0),
            MemStack::new(&mut scratch),
        );
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

/// Supernodal factorization on the CPU (`faer`, all rayon threads).
pub struct CpuCholesky<T: Real> {
    factor: Vec<T>,
}

impl<T: Real> CpuCholesky<T> {
    pub fn new(analysis: &Analysis) -> Self {
        Self { factor: vec![T::from_f64(0.0); analysis.factor_len()] }
    }

    /// Factors the matrix whose lower-triangle values (in the analysis's pattern order) are
    /// `values`.
    pub fn factorize(&mut self, analysis: &Analysis, values: &[T]) -> Result<(), Error> {
        let n = analysis.n;
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, &analysis.col_ptr, None, &analysis.row_idx);
        let mat = SparseColMatRef::new(pattern, values);
        let par = Par::rayon(0);
        let mut scratch = MemBuffer::new(analysis.symbolic.factorize_numeric_llt_scratch::<T>(par, Default::default()));
        analysis
            .symbolic
            .factorize_numeric_llt(&mut self.factor, mat, Side::Lower, Default::default(), par, MemStack::new(&mut scratch), Default::default())
            .map(|_| ())
            .map_err(|_| Error::NotPositiveDefinite)
    }

    pub fn factor(&self) -> &[T] {
        &self.factor
    }

    pub fn solve_in_place(&self, analysis: &Analysis, rhs: &mut [T]) {
        analysis.solve_in_place(&self.factor, rhs);
    }
}
