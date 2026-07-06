//! `K`-chain unroll in `Gang::map`/`map_n`. Build twice and compare:
//!   cargo bench --bench map_ilp                       # unrolled (K = backend UNROLL)
//!   RUSTFLAGS="--cfg hp_no_ilp" cargo bench --bench map_ilp   # forced single chain (K = 1)

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hydroplane::{Gang, kernel};
use wide::f32x8;

fn input(n: usize) -> Vec<f32> {
    (0..n).map(|i| 1.0 + (i as f32 % 97.0) * 0.13).collect()
}

#[inline(always)]
fn heavy_scalar(x: f32) -> f32 {
    // Each sqrt waits on the previous: latency-bound per element.
    let a = (x + 1.0).sqrt();
    let b = (a + 1.0).sqrt();
    let c = (b + 1.0).sqrt();
    (c + 1.0).sqrt()
}

#[kernel]
fn map_heavy_k<'a>(ctx: Gang, a: &'a [f32], out: &'a mut [f32]) {
    ctx.map(a, out, 1.0, |x| {
        let a = (x + 1.0).sqrt();
        let b = (a + 1.0).sqrt();
        let c = (b + 1.0).sqrt();
        (c + 1.0).sqrt()
    });
}

#[kernel]
fn map_cheap_k<'a>(ctx: Gang, a: &'a [f32], out: &'a mut [f32]) {
    ctx.map(a, out, 0.0, |x| x + ctx.splat(1.5));
}

fn wide_heavy(a: &[f32], out: &mut [f32]) {
    let one = f32x8::splat(1.0);
    let mut i = 0;
    while i + 8 <= a.len() {
        let x = f32x8::new(a[i..i + 8].try_into().unwrap());
        let p = ((((x + one).sqrt() + one).sqrt() + one).sqrt() + one).sqrt();
        out[i..i + 8].copy_from_slice(&p.to_array());
        i += 8;
    }
    while i < a.len() {
        out[i] = heavy_scalar(a[i]);
        i += 1;
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("map");
    for n in [64usize, 256, 1024, 4096] {
        let a = input(n);
        let mut out = vec![0.0f32; n];
        g.bench_with_input(BenchmarkId::new("hydro_heavy", n), &n, |b, _| {
            b.iter(|| map_heavy_k(std::hint::black_box(&a), std::hint::black_box(&mut out)))
        });
        g.bench_with_input(BenchmarkId::new("wide_heavy", n), &n, |b, _| {
            b.iter(|| wide_heavy(std::hint::black_box(&a), std::hint::black_box(&mut out)))
        });
        g.bench_with_input(BenchmarkId::new("scalar_heavy", n), &n, |b, _| {
            b.iter(|| {
                for (o, &x) in out.iter_mut().zip(a.iter()) {
                    *o = heavy_scalar(std::hint::black_box(x));
                }
            })
        });
        g.bench_with_input(BenchmarkId::new("hydro_cheap", n), &n, |b, _| {
            b.iter(|| map_cheap_k(std::hint::black_box(&a), std::hint::black_box(&mut out)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
