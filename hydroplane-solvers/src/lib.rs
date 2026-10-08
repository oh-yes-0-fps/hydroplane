//! Solvers tuned per device. [`cholesky`] factors sparse symmetric positive definite matrices:
//! one symbolic [`cholesky::Analysis`] (fill-reducing [`order`]ing and supernodal structure) serves
//! a supernodal factorization on the CPU and, with the `vulkan` feature, a multifrontal one on
//! any Vulkan GPU, in `f32` or `f64`, with the factor in the same layout either way.

pub mod cholesky;
pub mod order;
#[cfg(feature = "vulkan")]
pub mod vulkan;
