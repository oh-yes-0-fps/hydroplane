//! Degree-8 Horner polynomial per element: compute-bound, high arithmetic intensity.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub const COEFFS: [f32; 9] = [0.5, -1.2, 0.8, -0.3, 0.15, -0.07, 0.04, -0.01, 0.002];

pub fn inputs(n: usize) -> Vec<f32> {
    ramp(n, 3.0, 1.5)
}

#[kernel]
pub fn horner_hp<'a>(ctx: Gang, c: [f32; 9], x: &'a [f32], out: &'a mut [f32]) {
    ctx.map(x, out, 0.0, |xv| {
        let mut acc = ctx.splat(c[8]);
        for k in (0..8).rev() {
            acc = acc.fma(xv, ctx.splat(c[k]));
        }
        acc
    });
}

pub fn horner_wide(c: &[f32; 9], x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let cv: [f32x8; 9] = std::array::from_fn(|k| f32x8::splat(c[k]));
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        let mut acc = cv[8];
        for k in (0..8).rev() {
            acc = acc.mul_add(xv, cv[k]);
        }
        out[off..off + 8].copy_from_slice(&acc.to_array());
        off += 8;
    }
    while off < n {
        let xi = x[off];
        let mut acc = c[8];
        for k in (0..8).rev() {
            acc = acc * xi + c[k];
        }
        out[off] = acc;
        off += 1;
    }
}

pub fn horner_scalar(c: &[f32; 9], x: &[f32], out: &mut [f32]) {
    for (o, &xi) in out.iter_mut().zip(x) {
        let mut acc = c[8];
        for k in (0..8).rev() {
            acc = acc * xi + c[k];
        }
        *o = acc;
    }
}
