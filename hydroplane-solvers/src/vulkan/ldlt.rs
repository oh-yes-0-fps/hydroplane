use crate::cholesky::{Analysis, Error};
use crate::sparse::SellMatrix;
use crate::iterative::{Refinement, Report};

use super::{DeviceReal, Info, Options, VulkanCholesky};

/// The signed factorization `A = L S Lᵀ` (`S = diag(±1)`) of a symmetric quasi-definite matrix on
/// a Vulkan GPU: the supernodal Cholesky's kernels, every product over a block's columns taken
/// with its signs. Each pivot must have the sign given for its column, which holds for a
/// saddle point `[H Bᵀ; B −C]` (`+1` for `H`'s rows, `−1` for the constraints') in an order that
/// eliminates each constraint after one of its neighbours: [`crate::ldlt::kkt_ordering`]. A pivot
/// of the wrong sign or too small is floored ([`Info::perturbed`]), and
/// [`Self::solve_refined`] recovers the exact solution.
pub struct VulkanLdlt<T: DeviceReal> {
    pub refinement: Refinement,
    chol: VulkanCholesky<T>,
    csr: SellMatrix,
    values: Vec<f64>,
}

impl<T: DeviceReal> VulkanLdlt<T> {
    /// `signs`: one per column in the matrix's own order, negative for `−1`.
    pub fn new(analysis: &Analysis, signs: &[i8]) -> Result<Self, Error> {
        Self::with_options(analysis, signs, Options::default())
    }

    pub fn with_options(analysis: &Analysis, signs: &[i8], options: Options) -> Result<Self, Error> {
        if signs.len() != analysis.dim() {
            return Err(Error::Pattern);
        }
        Ok(Self {
            refinement: Refinement::default(),
            chol: VulkanCholesky::build(analysis, options, Some(signs))?,
            csr: SellMatrix::from_lower_csc_symmetric(analysis.dim(), analysis.col_ptr(), analysis.row_idx()),
            values: Vec::new(),
        })
    }

    /// Factors the matrix whose lower-triangle values (in the analysis's pattern order) are
    /// `values`, keeping them for [`Self::solve_refined`].
    pub fn factorize(&mut self, values: &[f64]) -> Result<Info, Error> {
        self.values.clear();
        self.values.extend_from_slice(values);
        self.csr.set_values(values);
        self.chol.factorize(values)
    }

    /// `rhs ← A⁻¹ rhs` with the factor alone.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), Error> {
        self.chol.solve_in_place(rhs)
    }

    /// `rhs ← A⁻¹ rhs` to [`Self::refinement`]'s tolerance by GMRES on the true matrix in `f64`
    /// on the CPU, each preconditioner application a solve on the GPU.
    pub fn solve_refined(&mut self, rhs: &mut [f64]) -> Result<Report, Error> {
        let (csr, chol) = (&self.csr, &mut self.chol);
        let mut failure = None;
        let report = self.refinement.run(
            |x, y| csr.spmv(x, y),
            |z| {
                if let Err(e) = chol.solve_in_place(z) {
                    failure.get_or_insert(e);
                }
            },
            rhs,
        );
        failure.map_or(Ok(report), Err)
    }

    /// Device time of the last factorization (ms).
    pub fn last_factor_ms(&self) -> f64 {
        self.chol.last_factor_ms()
    }
}
