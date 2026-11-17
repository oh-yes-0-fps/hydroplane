//! Sparse factorizations through Apple's Accelerate (macOS), whose supernodal kernels run on the
//! CPU's matrix units: on an M5 Pro faster than the GPU in `f32` (C layer in `accelerate.c`).
//! It orders and analyses each pattern itself (METIS), so its factors are not in the
//! [`Analysis`]'s layout; only the pattern is taken from it.
//!
//! [`AccelerateCholesky`] for symmetric positive definite matrices, [`AccelerateLdlt`] for
//! symmetric indefinite ones (with delayed pivots, which `faer`'s does not have), [`AccelerateLu`]
//! for general square ones, and [`AccelerateQr`] for least squares.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use crate::Precision;
use crate::cholesky::{Analysis, Error};
use crate::ldlt::Inertia;

unsafe extern "C" {
    fn hs_accelerate_new(kind: i32, rows: i64, cols: i64, symmetric: i32, column_starts: *const i64, row_indices: *const i32, status: *mut i32) -> *mut c_void;
    fn hs_accelerate_factor_f32(h: *mut c_void, values: *mut f32) -> i32;
    fn hs_accelerate_factor_f64(h: *mut c_void, values: *mut f64) -> i32;
    fn hs_accelerate_solve_f32(h: *mut c_void, xb: *mut f32);
    fn hs_accelerate_solve_f64(h: *mut c_void, xb: *mut f64);
    fn hs_accelerate_solve_rect_f32(h: *mut c_void, b: *mut f32, x: *mut f32);
    fn hs_accelerate_solve_rect_f64(h: *mut c_void, b: *mut f64, x: *mut f64);
    fn hs_accelerate_inertia_f32(h: *mut c_void, positive: *mut i32, zero: *mut i32, negative: *mut i32) -> i32;
    fn hs_accelerate_inertia_f64(h: *mut c_void, positive: *mut i32, zero: *mut i32, negative: *mut i32) -> i32;
    fn hs_accelerate_free(h: *mut c_void);
}

/// `SparseStatusOK`.
const OK: i32 = 0;
/// `SparseFactorizationFailed`.
const FAILED: i32 = -1;
/// `SparseMatrixIsSingular`.
const SINGULAR: i32 = -2;

/// `SparseFactorization_t` values.
const CHOLESKY: i32 = 0;
const LDLT_UNPIVOTED: i32 = 2;
const LDLT_SBK: i32 = 3;
const LDLT_TPP: i32 = 4;
const QR: i32 = 40;
const LU_TPP: i32 = 83;

/// `f32` or `f64`, the precisions Accelerate factors in.
pub trait AccelerateReal: Copy + Default + Send + Sync + 'static + private::Sealed {
    const PRECISION: Precision;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    #[doc(hidden)]
    unsafe fn factor(h: *mut c_void, values: &mut [Self]) -> i32;
    #[doc(hidden)]
    unsafe fn solve(h: *mut c_void, xb: &mut [Self]);
    #[doc(hidden)]
    unsafe fn solve_rect(h: *mut c_void, b: &mut [Self], x: &mut [Self]);
    #[doc(hidden)]
    unsafe fn inertia(h: *mut c_void, counts: &mut [i32; 3]) -> i32;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! real {
    ($t:ty, $p:expr, $factor:ident, $solve:ident, $rect:ident, $inertia:ident) => {
        impl AccelerateReal for $t {
            const PRECISION: Precision = $p;
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            unsafe fn factor(h: *mut c_void, values: &mut [Self]) -> i32 {
                unsafe { $factor(h, values.as_mut_ptr()) }
            }
            unsafe fn solve(h: *mut c_void, xb: &mut [Self]) {
                unsafe { $solve(h, xb.as_mut_ptr()) }
            }
            unsafe fn solve_rect(h: *mut c_void, b: &mut [Self], x: &mut [Self]) {
                unsafe { $rect(h, b.as_mut_ptr(), x.as_mut_ptr()) }
            }
            unsafe fn inertia(h: *mut c_void, counts: &mut [i32; 3]) -> i32 {
                let [p, z, n] = counts;
                unsafe { $inertia(h, p, z, n) }
            }
        }
    };
}

real!(f32, Precision::F32, hs_accelerate_factor_f32, hs_accelerate_solve_f32, hs_accelerate_solve_rect_f32, hs_accelerate_inertia_f32);
real!(f64, Precision::F64, hs_accelerate_factor_f64, hs_accelerate_solve_f64, hs_accelerate_solve_rect_f64, hs_accelerate_inertia_f64);

/// One symbolic factorization and its numeric factor in precision `T`, refactored in place.
struct Handle<T: AccelerateReal> {
    ptr: NonNull<c_void>,
    rows: usize,
    cols: usize,
    a_len: usize,
    factored: bool,
    values: Vec<T>,
    buffer: Vec<T>,
    out: Vec<T>,
    _real: PhantomData<T>,
}

// Accelerate's factorization objects are plain heap data, used through `&mut self` only, so a
// shared reference can do nothing with the handle.
unsafe impl<T: AccelerateReal> Send for Handle<T> {}
unsafe impl<T: AccelerateReal> Sync for Handle<T> {}

impl<T: AccelerateReal> Handle<T> {
    fn new(kind: i32, rows: usize, symmetric: bool, col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        let cols = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        if row_idx[..col_ptr[cols]].iter().any(|&i| i >= rows) || (symmetric && rows != cols) {
            return Err(Error::Pattern);
        }
        let starts: Vec<i64> = col_ptr.iter().map(|&p| p as i64).collect();
        let idx: Vec<i32> = row_idx[..col_ptr[cols]].iter().map(|&i| i32::try_from(i).map_err(|_| Error::Pattern)).collect::<Result<_, _>>()?;
        let mut status = 0;
        let ptr = unsafe { hs_accelerate_new(kind, rows as i64, cols as i64, symmetric as i32, starts.as_ptr(), idx.as_ptr(), &mut status) };
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::Device(format!("Accelerate symbolic factorization failed ({status})")))?;
        Ok(Self { ptr, rows, cols, a_len: idx.len(), factored: false, values: Vec::new(), buffer: Vec::new(), out: Vec::new(), _real: PhantomData })
    }

    fn factorize(&mut self, values: &[f64], failed: Error) -> Result<(), Error> {
        assert_eq!(values.len(), self.a_len);
        self.values.clear();
        self.values.extend(values.iter().map(|&v| T::from_f64(v)));
        self.factored = false;
        match unsafe { T::factor(self.ptr.as_ptr(), &mut self.values) } {
            OK => {
                self.factored = true;
                Ok(())
            }
            FAILED => Err(failed),
            SINGULAR => Err(Error::Singular),
            s => Err(Error::Device(format!("Accelerate numeric factorization failed ({s})"))),
        }
    }

    fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), Error> {
        assert_eq!(rhs.len(), self.rows);
        if !self.factored {
            return Err(Error::NotFactorized);
        }
        self.buffer.clear();
        self.buffer.extend(rhs.iter().map(|&v| T::from_f64(v)));
        unsafe { T::solve(self.ptr.as_ptr(), &mut self.buffer) };
        for (r, s) in rhs.iter_mut().zip(&self.buffer) {
            *r = s.to_f64();
        }
        Ok(())
    }

    fn solve_rect(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), Error> {
        assert_eq!(b.len(), self.rows);
        assert_eq!(x.len(), self.cols);
        if !self.factored {
            return Err(Error::NotFactorized);
        }
        self.buffer.clear();
        self.buffer.extend(b.iter().map(|&v| T::from_f64(v)));
        self.out.clear();
        self.out.resize(self.cols, T::default());
        unsafe { T::solve_rect(self.ptr.as_ptr(), &mut self.buffer, &mut self.out) };
        for (x, s) in x.iter_mut().zip(&self.out) {
            *x = s.to_f64();
        }
        Ok(())
    }
}

impl<T: AccelerateReal> Drop for Handle<T> {
    fn drop(&mut self) {
        unsafe { hs_accelerate_free(self.ptr.as_ptr()) }
    }
}

/// A sparse Cholesky factorization by Accelerate in precision `T`.
pub struct AccelerateCholesky<T: AccelerateReal> {
    handle: Handle<T>,
}

impl<T: AccelerateReal> AccelerateCholesky<T> {
    pub fn new(analysis: &Analysis) -> Result<Self, Error> {
        Ok(Self { handle: Handle::new(CHOLESKY, analysis.dim(), true, analysis.col_ptr(), analysis.row_idx())? })
    }

    /// Factors the matrix whose lower-triangle values, in the pattern's order, are `values`.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.handle.factorize(values, Error::NotPositiveDefinite).map_err(|e| match e {
            Error::Singular => Error::NotPositiveDefinite,
            e => e,
        })
    }

    /// `rhs ← A⁻¹ rhs` with the last factorization.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), Error> {
        self.handle.solve_in_place(rhs)
    }
}

/// Accelerate's `L D Lᵀ` pivoting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LdltPivoting {
    /// Threshold partial pivoting, `1 × 1` and `2 × 2` pivots, delaying a column to an ancestor
    /// supernode when no stable pivot is in its own: any nonsingular symmetric matrix, including
    /// a KKT matrix with a zero block. Reports [`AccelerateLdlt::inertia`].
    #[default]
    ThresholdPartial,
    /// Supernode Bunch–Kaufman with static pivoting: faster, never delays; a pivot too small is
    /// perturbed, so refine the solution.
    SupernodeBunchKaufman,
    /// `1 × 1` pivots taken as they come (quasi-definite matrices).
    Unpivoted,
}

/// A sparse symmetric indefinite `L D Lᵀ` by Accelerate in precision `T`, over the lower triangle.
pub struct AccelerateLdlt<T: AccelerateReal> {
    handle: Handle<T>,
    pivoting: LdltPivoting,
}

impl<T: AccelerateReal> AccelerateLdlt<T> {
    /// Threshold partial pivoting.
    pub fn new(analysis: &Analysis) -> Result<Self, Error> {
        Self::with_pivoting(analysis, LdltPivoting::default())
    }

    pub fn with_pivoting(analysis: &Analysis, pivoting: LdltPivoting) -> Result<Self, Error> {
        let kind = match pivoting {
            LdltPivoting::ThresholdPartial => LDLT_TPP,
            LdltPivoting::SupernodeBunchKaufman => LDLT_SBK,
            LdltPivoting::Unpivoted => LDLT_UNPIVOTED,
        };
        Ok(Self { handle: Handle::new(kind, analysis.dim(), true, analysis.col_ptr(), analysis.row_idx())?, pivoting })
    }

    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.handle.factorize(values, Error::Singular)
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), Error> {
        self.handle.solve_in_place(rhs)
    }

    /// The pivots' signs, under [`LdltPivoting::ThresholdPartial`] only.
    pub fn inertia(&self) -> Option<Inertia> {
        if self.pivoting != LdltPivoting::ThresholdPartial || !self.handle.factored {
            return None;
        }
        let mut counts = [0i32; 3];
        let status = unsafe { T::inertia(self.handle.ptr.as_ptr(), &mut counts) };
        (status == 0).then(|| Inertia { positive: counts[0] as usize, zero: counts[1] as usize, negative: counts[2] as usize })
    }
}

/// A sparse `L U` of a general square matrix by Accelerate (threshold partial pivoting), in
/// precision `T`. Needs macOS 15.5.
pub struct AccelerateLu<T: AccelerateReal> {
    handle: Handle<T>,
}

impl<T: AccelerateReal> AccelerateLu<T> {
    /// The whole matrix's pattern in compressed sparse columns.
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        Ok(Self { handle: Handle::new(LU_TPP, n, false, col_ptr, row_idx)? })
    }

    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.handle.factorize(values, Error::Singular)
    }

    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), Error> {
        self.handle.solve_in_place(rhs)
    }
}

/// A sparse `Q R` of an `m × n` matrix by Accelerate, in precision `T`: the least-squares solution
/// of an overdetermined system (`m ≥ n`, full column rank), or the least-norm solution of an
/// underdetermined one.
pub struct AccelerateQr<T: AccelerateReal> {
    handle: Handle<T>,
}

impl<T: AccelerateReal> AccelerateQr<T> {
    /// `rows` and the whole matrix's pattern in compressed sparse columns (its column count from
    /// `col_ptr`).
    pub fn new(rows: usize, col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Ok(Self { handle: Handle::new(QR, rows, false, col_ptr, row_idx)? })
    }

    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        self.handle.factorize(values, Error::Singular)
    }

    /// `x = argmin ‖A x − b‖` (`b` of length `m`, `x` of length `n`).
    pub fn solve(&mut self, b: &[f64], x: &mut [f64]) -> Result<(), Error> {
        self.handle.solve_rect(b, x)
    }
}
