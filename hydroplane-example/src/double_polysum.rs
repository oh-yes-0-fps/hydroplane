//! `Σ P(xᵢ) + Σ P(yᵢ)` for degree-8 `P`, an outer kernel calling a heavy inner reduction twice:
//! stresses cross-kernel unroll-cost composition.

use hydroplane::{Backend, Gang, Varying, kernel};
use wide::f32x8;

use crate::ramp;

pub const COEFFS: [f32; 9] = [0.5, -1.2, 0.8, -0.3, 0.15, -0.07, 0.04, -0.01, 0.002];

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    (ramp(n, 2.0, 1.0), ramp(n, 5.0, 1.0))
}

/// Degree-8 Horner evaluation on a whole register: eight serial FMAs, the register-heavy step that
/// makes `polystep` want a low unroll factor.
#[inline(always)]
fn horner8<S: Backend<f32>>(ctx: Gang<S>, x: Varying<f32, S>) -> Varying<f32, S> {
    let mut acc = ctx.splat(COEFFS[8]);
    for k in (0..8).rev() {
        acc = acc.fma(x, ctx.splat(COEFFS[k]));
    }
    acc
}

/// The heavy inner reduction, reused by the outer kernel through its `_on` companion.
#[kernel]
pub fn polystep<'a>(ctx: Gang, x: &'a [f32]) -> f32 {
    // `sum` zero-fills tail lanes and `horner8(0) == COEFFS[0]`, so padding lanes would each add
    // `COEFFS[0]`. Subtract it per lane and add back `COEFFS[0]·n` for the real lanes.
    ctx.sum(x, |acc, v| acc + (horner8(ctx, v) - COEFFS[0])) + COEFFS[0] * x.len() as f32
}

/// Outer body: two runs of the heavy reduction on the already-dispatched backend, then an add.
/// The inner reduction's register cost governs the unroll factor here, not the outer surface.
#[kernel]
pub fn double_polysum_hp<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> f32 {
    polystep_on(ctx, x) + polystep_on(ctx, y)
}

fn horner8_scalar(xi: f32) -> f32 {
    let mut acc = COEFFS[8];
    for k in (0..8).rev() {
        acc = acc * xi + COEFFS[k];
    }
    acc
}

/// Both heavy passes over a single `f32x8` accumulator chain: full register width plus a scalar
/// tail, no independent accumulators.
pub fn double_polysum_wide(x: &[f32], y: &[f32]) -> f32 {
    let cv: [f32x8; 9] = std::array::from_fn(|k| f32x8::splat(COEFFS[k]));
    let mut acc = f32x8::splat(0.0);
    let mut tail = 0.0f32;
    for col in [x, y] {
        let n = col.len();
        let mut off = 0;
        while off + 8 <= n {
            let xv = f32x8::from(<[f32; 8]>::try_from(&col[off..off + 8]).unwrap());
            let mut p = cv[8];
            for k in (0..8).rev() {
                p = p.mul_add(xv, cv[k]);
            }
            acc += p;
            off += 8;
        }
        while off < n {
            tail += horner8_scalar(col[off]);
            off += 1;
        }
    }
    acc.reduce_add() + tail
}

pub fn double_polysum_scalar(x: &[f32], y: &[f32]) -> f32 {
    x.iter().chain(y).map(|&v| horner8_scalar(v)).sum()
}
