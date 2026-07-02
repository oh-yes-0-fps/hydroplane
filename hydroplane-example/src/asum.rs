//! `Σ |xᵢ|` — the L1 norm (BLAS `sasum`). A single-input reduction with one absolute value and one
//! add per element, so it is memory-bound like [`dot`](crate::dot) but even lighter on arithmetic;
//! the accumulation dependency again makes instruction-level parallelism the deciding factor. The
//! hand-rolled `wide` baseline deliberately keeps a single accumulator chain, leaving the ILP for
//! hydroplane's runtime unroll to supply.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> Vec<f32> {
    ramp(n, 5.0, 3.0)
}

#[kernel]
pub fn asum_hp<'a>(ctx: Gang, x: &'a [f32]) -> f32 {
    ctx.sum(x, |acc, v| acc + v.abs())
}

/// One f32x8 accumulator (8 lanes/iter) — a single chain, no manual ILP, plus a scalar tail.
pub fn asum_wide(x: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = f32x8::splat(0.0);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        acc += xv.abs();
        off += 8;
    }
    let mut s = acc.reduce_add();
    while off < n {
        s += x[off].abs();
        off += 1;
    }
    s
}

pub fn asum_scalar(x: &[f32]) -> f32 {
    x.iter().map(|&a| a.abs()).sum()
}
