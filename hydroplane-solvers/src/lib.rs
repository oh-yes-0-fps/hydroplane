//! Sparse symmetric positive definite solvers tuned per device. [`Solver`] is the entry point:
//! it reads the pattern, detects the machine's devices, measures them once, and picks the
//! device, method (direct, or conjugate gradient on a cheap factor), precision (`f32`, `f64`, or
//! `f64` emulated in `f32` pairs, [`Ef64`], on GPUs without it), ordering and kernel tuning, then
//! checks its first answer against the true residual and adapts. [`auto::Options`] fixes any of
//! those choices instead.
//!
//! Underneath, usable directly: [`cholesky`] (the symbolic [`cholesky::Analysis`] every
//! factorization shares, and the CPU solvers on `faer`), [`order`] (nested dissection), on macOS
//! `accelerate` (Apple's sparse Cholesky, `L D Lᵀ`, LU and QR on the CPU's matrix units) and
//! `sme` (our supernodal Cholesky on the same units), and, with the `vulkan` feature, `vulkan`
//! (supernodal Cholesky and preconditioned CG in rust-gpu kernels on any Vulkan GPU).
//!
//! Beyond positive definite matrices, [`GeneralSolver`] (any square matrix) and
//! [`IndefiniteSolver`] (symmetric, any inertia) choose for themselves in the same way. Under
//! them, [`ldlt`] factors symmetric indefinite ones (saddle-point
//! and KKT systems, with an ordering that gives every constraint a real pivot, and the inertia),
//! [`lu`] general square ones (`faer`'s LU, and our multifrontal LU over the symmetric structure
//! with an MC64-style matching for small diagonals), and [`banded`] symmetric band matrices
//! (chains over time). [`qr`] solves sparse least-squares problems directly.
//!
//! [`eigen`] finds the smallest eigenpairs of `A x = λ x` or `A x = λ B x` (natural frequencies,
//! buckling modes) by LOBPCG, preconditioned by any of the factorizations here.
//!
//! For operators that are not an assembled matrix (matrix-free FEM tangents), [`iterative`] has
//! Krylov methods over any vector layout: conjugate gradient (standard or flexible), MINRES for
//! symmetric indefinite operators, GMRES / FGMRES and BiCGStab for nonsymmetric ones, LSQR for
//! least squares, and Jacobi, block-Jacobi, Chebyshev and additive preconditioners. [`dense`] has
//! the shifted dense Cholesky and coarse correction of two-level preconditioners.
//!
//! [`sparse::SellMatrix`] is the fast `y = A x` for an assembled matrix (SELL-C-σ layout), the
//! operator to hand the [`iterative`] solvers.
//!
//! [`batched`] solves thousands of independent small problems at once, one per SIMD lane: 3×3
//! symmetric eigen-decomposition, signed SVD and polar decomposition, and `N×N` Cholesky and
//! `L D Lᵀ` solves.
//!
//! [`bvh`] is a bounding volume hierarchy with nodes as wide as the SIMD registers, for the
//! geometric queries around a solve (contact candidates, point location, nearest neighbours).

#[cfg(accelerate)]
pub mod accelerate;
pub mod auto;
pub mod banded;
pub mod batched;
pub mod block;
pub mod bvh;
pub mod cholesky;
pub mod dense;
pub mod eigen;
pub mod general;
mod ef64;
pub mod iterative;
pub mod ldlt;
pub mod lu;
pub mod order;
mod precision;
pub mod qr;
pub mod sparse;
#[cfg(accelerate)]
pub mod sme;
#[cfg(feature = "vulkan")]
pub mod vulkan;

pub use auto::Solver;
pub use general::{GeneralSolver, IndefiniteSolver};
pub use ef64::Ef64;
pub use precision::Precision;
