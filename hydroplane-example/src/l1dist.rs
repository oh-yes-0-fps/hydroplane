//! `Σ |xᵢ − yᵢ|` two-input reduction: memory-bound, ILP-dominated.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    (ramp(n, 2.0, 2.0), ramp(n, 7.0, 2.0))
}

#[kernel]
pub fn l1dist_hp<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> f32 {
    ctx.zip_reduce(
        x,
        y,
        0.0,
        0.0,
        ctx.splat(0.0),
        |acc, a, b| acc + (a - b).abs(),
        |p, q| p + q,
    )
    .reduce_sum()
}

/// One f32x8 accumulator, single chain, no manual ILP; a scalar tail finishes the leftover lanes.
pub fn l1dist_wide(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = f32x8::splat(0.0);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        let yv = f32x8::from(<[f32; 8]>::try_from(&y[off..off + 8]).unwrap());
        acc += (xv - yv).abs();
        off += 8;
    }
    let mut s = acc.reduce_add();
    while off < n {
        s += (x[off] - y[off]).abs();
        off += 1;
    }
    s
}

pub fn l1dist_scalar(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(&a, &b)| (a - b).abs()).sum()
}
