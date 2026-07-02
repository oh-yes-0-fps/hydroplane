//! `Σ xᵢ·yᵢ` — a reduction. Memory-bound like saxpy, but the accumulation dependency makes
//! instruction-level parallelism (multiple independent accumulator chains) the deciding factor,
//! which is exactly what hydroplane's runtime unroll and the hand-rolled 4×-ILP baseline both target.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    (ramp(n, 2.0, 2.0), ramp(n, 7.0, 2.0))
}

#[kernel]
pub fn dot_hp<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> f32 {
    ctx.dot(x, y)
}

/// Four independent f32x8 FMA chains (32 lanes/iter) — the textbook superscalar reduction.
pub fn dot_wide(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let mut acc = [f32x8::splat(0.0); 4];
    let mut off = 0;
    while off + 32 <= n {
        for (j, accj) in acc.iter_mut().enumerate() {
            let o = off + j * 8;
            let xv = f32x8::from(<[f32; 8]>::try_from(&x[o..o + 8]).unwrap());
            let yv = f32x8::from(<[f32; 8]>::try_from(&y[o..o + 8]).unwrap());
            *accj = xv.mul_add(yv, *accj);
        }
        off += 32;
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add();
    while off < n {
        s += x[off] * y[off];
        off += 1;
    }
    s
}

pub fn dot_scalar(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(&a, &b)| a * b).sum()
}
