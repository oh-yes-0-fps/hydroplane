//! Cosine similarity `Σxᵢyᵢ / √(Σxᵢ²·Σyᵢ²)`: three dot products on one dispatch, stressing
//! kernel reuse via `_on` companions.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    (ramp(n, 3.0, 2.0), ramp(n, 8.0, 2.0))
}

#[kernel]
fn dotp<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.dot(a, b)
}

/// Three `dotp` passes on the same dispatched backend via the `dotp_on` companion, then a scalar
/// combine.
#[kernel]
pub fn cosine_hp<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> f32 {
    let d = dotp_on(ctx, x, y);
    let nx = dotp_on(ctx, x, x);
    let ny = dotp_on(ctx, y, y);
    d / (nx * ny).sqrt()
}

/// All three sums in a single fused pass, one chain each (full register + scalar tail).
pub fn cosine_wide(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    let mut sxy = f32x8::splat(0.0);
    let mut sxx = f32x8::splat(0.0);
    let mut syy = f32x8::splat(0.0);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        let yv = f32x8::from(<[f32; 8]>::try_from(&y[off..off + 8]).unwrap());
        sxy = xv.mul_add(yv, sxy);
        sxx = xv.mul_add(xv, sxx);
        syy = yv.mul_add(yv, syy);
        off += 8;
    }
    let mut dxy = sxy.reduce_add();
    let mut dxx = sxx.reduce_add();
    let mut dyy = syy.reduce_add();
    while off < n {
        let xi = x[off];
        let yi = y[off];
        dxy += xi * yi;
        dxx += xi * xi;
        dyy += yi * yi;
        off += 1;
    }
    dxy / (dxx * dyy).sqrt()
}

pub fn cosine_scalar(x: &[f32], y: &[f32]) -> f32 {
    let (mut dxy, mut dxx, mut dyy) = (0.0f32, 0.0f32, 0.0f32);
    for (&xi, &yi) in x.iter().zip(y) {
        dxy += xi * yi;
        dxx += xi * xi;
        dyy += yi * yi;
    }
    dxy / (dxx * dyy).sqrt()
}
