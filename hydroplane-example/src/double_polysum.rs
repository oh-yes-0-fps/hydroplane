//! `Σ P(xᵢ) + Σ P(yᵢ)` for a degree-8 polynomial `P` — a *cross-kernel* composition. The heavy
//! per-element work (an eight-FMA Horner chain) lives in one reduction kernel, `polystep`; the outer
//! `double_polysum_hp` just calls it twice and adds. That outer body reads trivially light, but each
//! call runs the register-heavy inner reduction, so the unroll factor the optimizer picks for the
//! *outer* kernel is what the *inner* reduction pays for. A cost model that only looks at the outer
//! body hands it a high `K_CAP` and the heavy Horner chain spills; one that composes the inner
//! kernel's cost keeps `K_CAP` down at `polystep`'s own. The hand-written baselines inline both heavy
//! passes into a single-chain `wide` accumulator, deliberately omitting the ILP the optimizer adds.

use hydroplane::{Backend, Gang, Varying, kernel};
use wide::f32x8;

use crate::ramp;

pub const COEFFS: [f32; 9] = [0.5, -1.2, 0.8, -0.3, 0.15, -0.07, 0.04, -0.01, 0.002];

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    (ramp(n, 2.0, 1.0), ramp(n, 5.0, 1.0))
}

/// Degree-8 Horner evaluation on a whole register — eight serial FMAs, the register-heavy step that
/// makes `polystep` (and therefore the composed `double_polysum`) want a low unroll factor.
#[inline(always)]
fn horner8<S: Backend<f32>>(ctx: Gang<S>, x: Varying<f32, S>) -> Varying<f32, S> {
    let mut acc = ctx.splat(COEFFS[8]);
    for k in (0..8).rev() {
        acc = acc.fma(x, ctx.splat(COEFFS[k]));
    }
    acc
}

/// The heavy inner reduction, reused by the outer kernel through its `_on` companion: sum the
/// degree-8 polynomial over one column with the auto-tuned ILP `sum` supplies.
#[kernel]
pub fn polystep<'a>(ctx: Gang, x: &'a [f32]) -> f32 {
    // `sum` zero-fills the inactive tail lanes, and `horner8(0) == COEFFS[0]`, so those padding lanes
    // would each add `COEFFS[0]` to the reduction. Subtract it per lane (padding then contributes 0)
    // and add back the `COEFFS[0]·n` that removes from the real lanes.
    ctx.sum(x, |acc, v| acc + (horner8(ctx, v) - COEFFS[0])) + COEFFS[0] * x.len() as f32
}

/// The light-looking outer body: two runs of the heavy reduction on the *already-dispatched*
/// backend, then an add. One dispatch, and the inner reduction's register cost is what governs the
/// unroll factor here — not the two-call-plus-add surface.
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

/// Both heavy passes inlined into one function over a *single* `f32x8` accumulator chain: full
/// register width plus a scalar tail, no independent accumulators — the ILP is left for hydroplane's
/// optimizer to add on the `_hp` side.
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
