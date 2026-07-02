//! `Σ xᵢ²` — the squared L2 norm, a single-input reduction. The per-element body is one FMA and the
//! whole cost is the accumulation dependency, so instruction-level parallelism (several independent
//! accumulator chains feeding one final combine) is the only thing that matters. The `wide` baseline
//! deliberately keeps a *single* chain; hydroplane's runtime unroll is what supplies the ILP the
//! hand-written code omits, so this is the cleanest demonstration that the optimizer earns its keep.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> Vec<f32> {
    ramp(n, 3.0, 2.0)
}

#[kernel]
pub fn l2norm_hp<'a>(ctx: Gang<f32>, x: &'a [f32]) -> f32 {
    ctx.sum(x, |acc, v| v.fma(v, acc))
}

/// One f32x8 accumulator (8 lanes/iter) plus a scalar tail — a single chain, no manual ILP.
pub fn l2norm_wide(x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = f32x8::splat(0.0);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        acc = xv.mul_add(xv, acc);
        off += 8;
    }
    let mut s = acc.reduce_add();
    while off < n {
        s += x[off] * x[off];
        off += 1;
    }
    s
}

pub fn l2norm_scalar(x: &[f32]) -> f32 {
    x.iter().map(|&v| v * v).sum()
}
