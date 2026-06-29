//! Disassembly harness: `cargo build --release --example asm_probe` then
//! `otool -tV target/release/examples/asm_probe` and look at `_probe_hydro` / `_probe_wide4`.

use hydroplane::{Gang, kernel};
use std::hint::black_box;
use wide::f32x4;

#[kernel]
fn dot<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    let mut acc = ctx.splat(0.0);
    for (off, cnt) in ctx.chunks(a.len()) {
        let x = ctx.load_partial(&a[off..off + cnt], 0.0);
        let y = ctx.load_partial(&b[off..off + cnt], 0.0);
        acc = acc + x * y;
    }
    acc.reduce_sum()
}

#[kernel]
fn dot_fold<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_fold(a, b, 0.0, 0.0, ctx.splat(0.0), |acc, x, y| acc + x * y)
        .reduce_sum()
}

#[kernel]
fn dot_ilp<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_reduce(
        a,
        b,
        0.0,
        0.0,
        ctx.splat(0.0),
        |acc, x, y| x.fma(y, acc),
        |p, q| p + q,
    )
    .reduce_sum()
}

#[kernel]
fn dot_ilp8<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_reduce_k::<8, _, _, _>(
        a,
        b,
        0.0,
        0.0,
        ctx.splat(0.0),
        |acc, x, y| x.fma(y, acc),
        |p, q| p + q,
    )
    .reduce_sum()
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_hydro(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_fold(a: &[f32], b: &[f32]) -> f32 {
    dot_fold(a, b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_ilp(a: &[f32], b: &[f32]) -> f32 {
    dot_ilp(a, b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_ilp8(a: &[f32], b: &[f32]) -> f32 {
    dot_ilp8(a, b)
}

#[unsafe(no_mangle)]
#[inline(never)]
pub fn probe_wide4(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 4;
    let mut acc = f32x4::splat(0.0);
    for i in 0..chunks {
        let off = i * 4;
        let va = f32x4::from(<[f32; 4]>::try_from(&a[off..off + 4]).unwrap());
        let vb = f32x4::from(<[f32; 4]>::try_from(&b[off..off + 4]).unwrap());
        acc += va * vb;
    }
    let mut s = acc.reduce_add();
    for i in chunks * 4..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn main() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32).collect();
    println!(
        "{} {} {} {} {}",
        black_box(probe_hydro(black_box(&a), black_box(&b))),
        black_box(probe_fold(black_box(&a), black_box(&b))),
        black_box(probe_ilp(black_box(&a), black_box(&b))),
        black_box(probe_ilp8(black_box(&a), black_box(&b))),
        black_box(probe_wide4(black_box(&a), black_box(&b)))
    );
}
