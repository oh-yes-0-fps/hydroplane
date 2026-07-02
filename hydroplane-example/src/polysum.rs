//! `Σ P(xᵢ)` for a degree-8 polynomial `P` — a reduction with a heavy per-element body: eight serial
//! FMAs (a full Horner evaluation) feed a single running sum. Unlike [`dot`](crate::dot), where the
//! per-element work is one FMA, here the accumulator step drags a long chain of varying temporaries,
//! so the reduction is register-heavier and the optimizer must cap its independent-chain count lower
//! than it would for a lean norm/dot to stay off the spill cliff.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub const COEFFS: [f32; 9] = [1.0, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005];

pub fn inputs(n: usize) -> Vec<f32> {
    ramp(n, 3.0, 1.0)
}

#[kernel]
pub fn polysum_hp<'a>(ctx: Gang, c: [f32; 9], x: &'a [f32]) -> f32 {
    // `sum` zero-fills the inactive tail lanes, and `P(0) = c[0] ≠ 0`, so accumulating `P` directly
    // would let each padding lane leak a `c[0]` into the total. Summing `P(x) − c[0]` (which vanishes
    // at the fill value) keeps the tail clean; the dropped constant returns as `c[0]·n` at the end.
    let bias = c[0];
    ctx.sum(x, |acc, v| {
        let mut p = ctx.splat(c[8]);
        for k in (0..8).rev() {
            p = p.fma(v, ctx.splat(c[k]));
        }
        acc + (p - bias)
    }) + bias * x.len() as f32
}

/// One f32x8 accumulator, a full Horner chain evaluated per lane-vector before the add — a single
/// dependency chain, so hydroplane's runtime unroll is what has to supply the ILP this omits.
pub fn polysum_wide(c: &[f32; 9], x: &[f32]) -> f32 {
    let n = x.len();
    let cv: [f32x8; 9] = std::array::from_fn(|k| f32x8::splat(c[k]));
    let mut acc = f32x8::splat(0.0);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        let mut p = cv[8];
        for k in (0..8).rev() {
            p = p.mul_add(xv, cv[k]);
        }
        acc += p;
        off += 8;
    }
    let mut s = acc.reduce_add();
    while off < n {
        let xi = x[off];
        let mut p = c[8];
        for k in (0..8).rev() {
            p = p * xi + c[k];
        }
        s += p;
        off += 1;
    }
    s
}

pub fn polysum_scalar(c: &[f32; 9], x: &[f32]) -> f32 {
    let mut s = 0.0;
    for &xi in x {
        let mut p = c[8];
        for k in (0..8).rev() {
            p = p * xi + c[k];
        }
        s += p;
    }
    s
}
