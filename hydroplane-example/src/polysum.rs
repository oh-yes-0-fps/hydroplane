//! `Σ P(xᵢ)` for degree-8 `P`: a register-heavy reduction, stressing the unroll/spill trade-off.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub const COEFFS: [f32; 9] = [1.0, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005];

pub fn inputs(n: usize) -> Vec<f32> {
    ramp(n, 3.0, 1.0)
}

#[kernel]
pub fn polysum_hp<'a>(ctx: Gang, c: [f32; 9], x: &'a [f32]) -> f32 {
    // `sum` zero-fills tail lanes and `P(0) = c[0] ≠ 0`, so each padding lane would leak a `c[0]`.
    // Sum `P(x) − c[0]` instead and add back `c[0]·n` at the end.
    let bias = c[0];
    ctx.sum(x, |acc, v| {
        let mut p = ctx.splat(c[8]);
        for k in (0..8).rev() {
            p = p.fma(v, ctx.splat(c[k]));
        }
        acc + (p - bias)
    }) + bias * x.len() as f32
}

/// One f32x8 accumulator, a full Horner chain per lane-vector before the add: a single dependency
/// chain, omitting the ILP hydroplane's runtime unroll supplies.
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
