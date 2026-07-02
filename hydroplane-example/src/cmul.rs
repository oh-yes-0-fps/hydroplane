//! Elementwise complex multiply `(a+bi)·(c+di) = (ac−bd) + (ad+bc)i` over split real/imag columns.
//! Four loads feed four products and two adds into two stores — memory-bound with a little arithmetic,
//! and every lane is independent, so the deciding factor is whether the `&`/`&mut` columns carry
//! `noalias` (the `#[kernel]` boundary default) so LLVM may cluster the four loads and two stores.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (ramp(n, 1.0, 2.0), ramp(n, 5.0, 2.0), ramp(n, 9.0, 2.0), ramp(n, 13.0, 2.0))
}

// Four input columns (a's real/imag, b's real/imag) into two outputs — the asymmetric `map_cols`
// drives the full-register pass, masked tail, and ILP; the closure is just the complex product.
#[kernel]
pub fn cmul_hp<'a>(
    ctx: Gang<f32>,
    ar: &'a [f32],
    ai: &'a [f32],
    br: &'a [f32],
    bi: &'a [f32],
    outr: &'a mut [f32],
    outi: &'a mut [f32],
) {
    ctx.map_cols::<4, 2>(
        [ar, ai, br, bi],
        [outr, outi],
        0.0,
        |[a, b, c, d]| [a * c - b * d, a * d + b * c],
    );
}

pub fn cmul_wide(ar: &[f32], ai: &[f32], br: &[f32], bi: &[f32], outr: &mut [f32], outi: &mut [f32]) {
    let n = ar.len();
    let mut off = 0;
    while off + 8 <= n {
        let a = f32x8::from(<[f32; 8]>::try_from(&ar[off..off + 8]).unwrap());
        let b = f32x8::from(<[f32; 8]>::try_from(&ai[off..off + 8]).unwrap());
        let c = f32x8::from(<[f32; 8]>::try_from(&br[off..off + 8]).unwrap());
        let d = f32x8::from(<[f32; 8]>::try_from(&bi[off..off + 8]).unwrap());
        outr[off..off + 8].copy_from_slice(&(a * c - b * d).to_array());
        outi[off..off + 8].copy_from_slice(&(a * d + b * c).to_array());
        off += 8;
    }
    while off < n {
        outr[off] = ar[off] * br[off] - ai[off] * bi[off];
        outi[off] = ar[off] * bi[off] + ai[off] * br[off];
        off += 1;
    }
}

pub fn cmul_scalar(ar: &[f32], ai: &[f32], br: &[f32], bi: &[f32], outr: &mut [f32], outi: &mut [f32]) {
    for i in 0..ar.len() {
        outr[i] = ar[i] * br[i] - ai[i] * bi[i];
        outi[i] = ar[i] * bi[i] + ai[i] * br[i];
    }
}
