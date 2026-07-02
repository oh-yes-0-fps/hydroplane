//! A deliberately complex multi-stage pipeline that exercises every optimizer decision at once.
//!
//! An orchestrator kernel runs three stages through their `_on` companions (one dispatch, shared
//! backend) and finishes with a scalar helper:
//!
//! 1. [`scale_bias_hp`] — elementwise, one FMA/element (memory-bound) → should route to the scalar
//!    backend (the auto-vectorizer wins).
//! 2. [`activate_hp`] — elementwise, an eight-FMA polynomial/element (compute-bound) → stays SIMD but
//!    at a low unroll factor.
//! 3. [`energy_hp`] — a sum-of-squares reduction → stays SIMD at a high unroll factor (the compiler
//!    can't auto-vectorize an FP reduction, so hydroplane's ILP is the only path).
//!
//! Along the way the kernels call several **non-kernel** functions (`gains`, `activation_coeffs`,
//! `activate_poly`, `finalize`) — these must be ignored as cross-kernel edges, and the orchestrator's
//! unroll cap must compose down to satisfy the tightest stage it runs.

use crate::ramp;
use hydroplane::{Backend, Gang, Varying, kernel};

pub fn inputs(n: usize) -> Vec<f32> {
    ramp(n, 3.0, 1.5)
}

/// Non-kernel scalar setup: the affine gain and bias.
fn gains() -> (f32, f32) {
    (0.7, -0.15)
}

/// Non-kernel scalar setup: coefficients of a smooth, well-conditioned degree-8 activation.
fn activation_coeffs() -> [f32; 9] {
    [0.0, 1.0, -0.16, 8.0e-3, -1.9e-4, 2.5e-6, -1.7e-8, 5.5e-11, -6.6e-14]
}

/// Non-kernel generic **Varying** helper: the activation evaluated per lane. Called from inside
/// `activate_hp`'s map closure — a non-kernel call, so not a cross-kernel edge.
#[inline]
fn activate_poly<S: Backend<f32>>(ctx: Gang<S>, v: Varying<f32, S>) -> Varying<f32, S> {
    let c = activation_coeffs();
    let mut acc = ctx.splat(c[8]);
    for k in (0..8).rev() {
        acc = acc.fma(v, ctx.splat(c[k]));
    }
    acc
}

/// Non-kernel scalar finalizer: RMS of the accumulated energy.
fn finalize(energy: f32, n: usize) -> f32 {
    (energy / n.max(1) as f32).sqrt()
}

/// Stage 1 — elementwise, one FMA/element (memory-bound), so it uses the streaming (auto-vectorized)
/// combinator rather than SIMD `map`. Calls the non-kernel `gains`.
#[kernel]
pub fn scale_bias_hp<'a>(ctx: Gang, x: &'a [f32], out: &'a mut [f32]) {
    let (g, b) = gains();
    ctx.stream_map(x, out, |xi| g * xi + b);
}

/// Stage 2 — elementwise, eight FMAs/element (compute-bound). Calls the non-kernel `activate_poly`.
#[kernel]
pub fn activate_hp<'a>(ctx: Gang, x: &'a [f32], out: &'a mut [f32]) {
    ctx.map(x, out, 0.0, |v| activate_poly(ctx, v));
}

/// Stage 3 — a reduction (sum of squares).
#[kernel]
pub fn energy_hp<'a>(ctx: Gang, x: &'a [f32]) -> f32 {
    ctx.sum(x, |acc, v| v.fma(v, acc))
}

/// The orchestrator: pre-scale → activate → energy → RMS, each stage run through its `_on` companion
/// and a scalar helper at the end. Calls two elementwise stages and one reduction stage plus two
/// non-kernel functions; its unroll cap must compose to satisfy every stage.
#[kernel]
pub fn feature_score_hp<'a>(
    ctx: Gang,
    x: &'a [f32],
    scratch_a: &'a mut [f32],
    scratch_b: &'a mut [f32],
) -> f32 {
    scale_bias_hp_on(ctx, x, scratch_a);
    activate_hp_on(ctx, scratch_a, scratch_b);
    let energy = energy_hp_on(ctx, scratch_b);
    finalize(energy, x.len())
}

/// Scalar reference for the whole pipeline (correctness oracle).
pub fn feature_score_scalar(x: &[f32]) -> f32 {
    let (g, b) = gains();
    let c = activation_coeffs();
    let mut energy = 0.0f32;
    for &xi in x {
        let s = g * xi + b;
        let mut a = c[8];
        for k in (0..8).rev() {
            a = a * s + c[k];
        }
        energy += a * a;
    }
    finalize(energy, x.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pipeline_matches_scalar() {
        for n in [15usize, 64, 257, 1024] {
            let x = inputs(n);
            let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
            let hp = feature_score_hp(&x, &mut a, &mut b);
            let sc = feature_score_scalar(&x);
            assert!((hp - sc).abs() <= 1e-3 * sc.abs().max(1.0), "n={n}: hp={hp} sc={sc}");
        }
    }
}
