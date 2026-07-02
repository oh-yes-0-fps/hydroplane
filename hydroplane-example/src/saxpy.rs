//! `y = a·x + y` — the simplest workload: one FMA per element, two loads and a store. Pure
//! memory bandwidth; SIMD width buys almost nothing past what the load/store ports allow.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> (f32, Vec<f32>, Vec<f32>) {
    (2.5, ramp(n, 1.0, 4.0), ramp(n, 9.0, 4.0))
}

// Memory-bandwidth-bound (~0.16 flop/byte): explicit SIMD buys nothing over what LLVM's
// auto-vectorizer does with a plain loop, so this reaches for the streaming combinator — a scalar
// closure over an auto-vectorized loop — rather than the SIMD `map` family. `#[hint_cnt]` records the
// expected length (thrown out for now).
#[kernel]
pub fn saxpy_hp<'a>(ctx: Gang, a: f32, #[hint_cnt(4096)] x: &'a [f32], y: &'a mut [f32]) {
    ctx.stream_zip_inplace(x, y, |xi, yi| a * xi + yi);
}

pub fn saxpy_wide(a: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len();
    let av = f32x8::splat(a);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        let yv = f32x8::from(<[f32; 8]>::try_from(&y[off..off + 8]).unwrap());
        y[off..off + 8].copy_from_slice(&xv.mul_add(av, yv).to_array());
        off += 8;
    }
    while off < n {
        y[off] += a * x[off];
        off += 1;
    }
}

pub fn saxpy_scalar(a: f32, x: &[f32], y: &mut [f32]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}
