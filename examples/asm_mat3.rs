//! Disassembly harness for the batched 3x3 inverse: `cargo build --release --features glam
//! --example asm_mat3` then `otool -tV target/release/examples/asm_mat3` and read `_probe_hydro_inv`
//! (the `Mat3Wide::inverse` SoA-wide path) vs `_probe_scalar_inv` (per-matrix `glam::Mat3::inverse`).

use glam::Mat3;
use hydroplane::{Gang, GangGlamExt, kernel};
use std::hint::black_box;

#[kernel]
fn invert<'a>(ctx: Gang<f32>, m: [&'a [f32]; 9], out: [&'a mut [f32]; 9]) {
    let n = m[0].len();
    let lanes = ctx.lanes();
    let mut out = out;
    let mut off = 0;
    while off + lanes <= n {
        let cols: [&[f32]; 9] = std::array::from_fn(|c| &m[c][off..off + lanes]);
        let inv = ctx.load_mat3(cols).inverse();
        inv.store(out.each_mut().map(|o| &mut o[off..off + lanes]));
        off += lanes;
    }
    if off < n {
        let cols: [&[f32]; 9] = std::array::from_fn(|c| &m[c][off..n]);
        let inv = ctx.load_partial_mat3(cols, 1.0).inverse();
        inv.store_partial(out.each_mut().map(|o| &mut o[off..n]));
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_hydro_inv(m: [&[f32]; 9], out: [&mut [f32]; 9]) {
    invert(m, out);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_scalar_inv(m: [&[f32]; 9], out: [&mut [f32]; 9]) {
    for i in 0..m[0].len() {
        let cols: [f32; 9] = std::array::from_fn(|c| m[c][i]);
        let inv = Mat3::from_cols_array(&cols).inverse().to_cols_array();
        for c in 0..9 {
            out[c][i] = inv[c];
        }
    }
}

fn main() {
    let n = 64;
    let m: Vec<Vec<f32>> = (0..9)
        .map(|c| (0..n).map(|i| ((i + c) as f32 * 0.1).sin() + 4.0).collect())
        .collect();
    let mref: [&[f32]; 9] = std::array::from_fn(|c| m[c].as_slice());

    let mut a: Vec<Vec<f32>> = (0..9).map(|_| vec![0.0f32; n]).collect();
    let mut b: Vec<Vec<f32>> = (0..9).map(|_| vec![0.0f32; n]).collect();
    {
        let mut it = a.iter_mut();
        let oref: [&mut [f32]; 9] = std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
        probe_hydro_inv(black_box(mref), oref);
    }
    {
        let mut it = b.iter_mut();
        let oref: [&mut [f32]; 9] = std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
        probe_scalar_inv(black_box(mref), oref);
    }
    println!("{} {}", black_box(a[0][0]), black_box(b[0][0]));
}
