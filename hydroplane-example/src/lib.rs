// Under the analysis driver's `--cfg hp_analyze` pass, register the tool namespace so the
// `#[kernel]`-emitted `#[hp_analyze::metrics(..)]` attributes are recognized.
#![cfg_attr(hp_analyze, feature(register_tool))]
#![cfg_attr(hp_analyze, register_tool(hp_analyze))]

//! Paired implementations per workload over identical inputs: `*_hp` (hydroplane `#[kernel]`),
//! `*_wide` (hand-rolled `wide` SIMD), and `*_scalar` (the correctness oracle), plus a
//! deterministic input generator. Shared by `benches/workloads.rs` and `tests/correctness.rs`.

pub mod asum;
pub mod cosine;
pub mod cmul;
pub mod dot;
pub mod double_polysum;
pub mod horner;
pub mod l1dist;
pub mod l2norm;
pub mod mandelbrot;
pub mod mat3_inverse;
pub mod matmul;
pub mod normalize;
pub mod pipeline;
pub mod polysum;
pub mod saxpy;
pub mod transform;

/// Max relative error between two slices. SIMD reorders FP arithmetic, so bit-exact equality is
/// too strict; the suite uses `1e-3` relative.
pub fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |e, (&x, &y)| e.max((x - y).abs() / y.abs().max(1.0)))
}

/// A column of `n` deterministic, well-conditioned values in roughly `[-1, 1]·scale`.
pub fn ramp(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.137).sin() * scale)
        .collect()
}
