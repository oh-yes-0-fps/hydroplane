//! Disassembly harness for saxpy: `cargo build --release --example asm_saxpy` then
//! `otool -tV target/release/examples/asm_saxpy` and read `_probe_hp` vs `_probe_scalar`.

use hydroplane_example::saxpy;
use std::hint::black_box;

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_hp(a: f32, x: &[f32], y: &mut [f32]) {
    saxpy::saxpy_hp(a, x, y);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_scalar(a: f32, x: &[f32], y: &mut [f32]) {
    saxpy::saxpy_scalar(a, x, y);
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_wide(a: f32, x: &[f32], y: &mut [f32]) {
    saxpy::saxpy_wide(a, x, y);
}

fn main() {
    let x: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let mut y = vec![1.0f32; 1024];
    probe_hp(black_box(2.0), black_box(&x), black_box(&mut y));
    probe_scalar(black_box(2.0), black_box(&x), black_box(&mut y));
    probe_wide(black_box(2.0), black_box(&x), black_box(&mut y));
    println!("{}", black_box(y[0]));
}
