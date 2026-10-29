//! Sparse `A P = Q R` of an `m × n` matrix (`m ≥ n`): least-squares solutions without forming
//! the normal equations, on any CPU (`faer`, after a COLAMD column ordering). On macOS,
//! `accelerate::AccelerateQr` does the same on the matrix units.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::qr::{QrRef, QrSymbolicParams, SymbolicQr, factorize_symbolic_qr};
use faer::sparse::linalg::{SupernodalThreshold, colamd};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Conj, MatMut, Par};

use crate::cholesky::{Error, Real};

/// Supernodal multifrontal QR on the CPU, `faer`; `Q` is kept as Householder reflections.
pub struct CpuQr<T: Real> {
    rows: usize,
    cols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    symbolic: SymbolicQr<usize>,
    indices: Vec<usize>,
    val: Vec<T>,
    factored: bool,
}

impl<T: Real> CpuQr<T> {
    /// `rows` and the whole matrix's pattern in compressed sparse columns (its column count from
    /// `col_ptr`), with `rows ≥` columns.
    pub fn new(rows: usize, col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        let cols = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        if rows < cols || row_idx[..col_ptr[cols]].iter().any(|&i| i >= rows) {
            return Err(Error::Pattern);
        }
        let pattern = SymbolicSparseColMatRef::new_checked(rows, cols, col_ptr, None, row_idx);
        let params = QrSymbolicParams { colamd_params: colamd::Control::default(), supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO, supernodal_params: Default::default() };
        let symbolic = factorize_symbolic_qr(pattern, params).map_err(|_| Error::Pattern)?;
        Ok(Self {
            rows,
            cols,
            col_ptr: col_ptr.to_vec(),
            row_idx: row_idx.to_vec(),
            indices: vec![0; symbolic.len_idx()],
            val: vec![T::from_f64(0.0); symbolic.len_val()],
            symbolic,
            factored: false,
        })
    }

    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        let pattern = SymbolicSparseColMatRef::new_checked(self.rows, self.cols, &self.col_ptr, None, &self.row_idx);
        let converted: Vec<T> = values.iter().map(|&v| T::from_f64(v)).collect();
        let par = Par::rayon(0);
        let mut scratch = MemBuffer::new(self.symbolic.factorize_numeric_qr_scratch::<T>(par, Default::default()));
        self.symbolic.factorize_numeric_qr(&mut self.indices, &mut self.val, SparseColMatRef::new(pattern, &converted[..]), par, MemStack::new(&mut scratch), Default::default());
        self.factored = true;
        Ok(())
    }

    /// `x = argmin ‖A x − b‖` (`b` of length `m`, `x` of length `n`).
    pub fn solve(&self, b: &[f64], x: &mut [f64]) -> Result<(), Error> {
        assert_eq!(b.len(), self.rows);
        assert_eq!(x.len(), self.cols);
        if !self.factored {
            return Err(Error::NotFactorized);
        }
        let mut work: Vec<T> = b.iter().map(|&v| T::from_f64(v)).collect();
        let par = Par::Seq;
        let mut scratch = MemBuffer::new(self.symbolic.solve_in_place_scratch::<T>(1, par));
        // SAFETY: `indices` and `val` were filled by `factorize_numeric_qr` on this symbolic.
        let qr = unsafe { QrRef::new_unchecked(&self.symbolic, &self.indices, &self.val) };
        qr.solve_in_place_with_conj(Conj::No, MatMut::from_column_major_slice_mut(&mut work, self.rows, 1), par, MemStack::new(&mut scratch));
        for (x, w) in x.iter_mut().zip(&work) {
            *x = w.to_f64();
        }
        if x.iter().all(|v| v.is_finite()) { Ok(()) } else { Err(Error::Singular) }
    }
}
