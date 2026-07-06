//! Disassembly harness for the `map_cols` kernels: `cargo build --release --example asm_cols`,
//! then `otool -tV` the binary and read `_probe_cmul` / `_probe_transform`.

use hydroplane_example::{cmul, transform};
use std::hint::black_box;

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_cmul(ar: &[f32], ai: &[f32], br: &[f32], bi: &[f32], outr: &mut [f32], outi: &mut [f32]) {
    cmul::cmul_hp(ar, ai, br, bi, outr, outi);
}

#[unsafe(no_mangle)]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub fn probe_transform(
    m: [&[f32]; 9],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    ox: &mut [f32],
    oy: &mut [f32],
    oz: &mut [f32],
) {
    transform::transform_hp(m, vx, vy, vz, ox, oy, oz);
}

fn main() {
    let n = 1024;
    let col: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 + 1.0).collect();
    let mut o1 = vec![0.0f32; n];
    let mut o2 = vec![0.0f32; n];
    let mut o3 = vec![0.0f32; n];
    probe_cmul(
        black_box(&col),
        black_box(&col),
        black_box(&col),
        black_box(&col),
        black_box(&mut o1),
        black_box(&mut o2),
    );
    let m: [&[f32]; 9] = [&col; 9];
    probe_transform(
        black_box(m),
        black_box(&col),
        black_box(&col),
        black_box(&col),
        black_box(&mut o1),
        black_box(&mut o2),
        black_box(&mut o3),
    );
    println!("{} {} {}", o1[7], o2[7], o3[7]);
}
