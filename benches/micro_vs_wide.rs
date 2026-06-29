//! Micro-kernel (`#[kernel]`, cached dispatch) vs hand-written `wide` SIMD vs a scalar loop, for a
//! small `f32` dot product across sizes. The interesting regime is small `N`: there the per-call
//! dispatch + `#[target_feature]` call boundary competes with `wide`'s inlined portable SIMD.
//!
//!   cargo bench --bench micro_vs_wide
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench micro_vs_wide   # lets the backend inline
//!
//! On aarch64 the dispatched backend is NEON, which is baseline, so hydroplane's ops inline even
//! without `target-cpu=native`; on x86 a generic build keeps them behind the call boundary (where
//! `wide`, compiled at the SSE2 baseline, runs fully inlined) until you pass `target-cpu=native`.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hydroplane::{Gang, kernel};
use std::hint::black_box;
use wide::{f32x4, f32x8};

// Idiomatic `chunks` + `load_partial`: ergonomic, but the variable chunk count forces a
// partial-vs-full branch and per-iteration bounds checks in the hot loop.
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

// Naive-looking, but the loop shape is the library's: `zip_fold` does the fixed-stride full-register
// pass + masked tail internally, so this one expression should compile like `dot_opt`.
#[kernel]
fn dot_fold<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_fold(a, b, 0.0, 0.0, ctx.splat(0.0), |acc, x, y| acc + x * y)
        .reduce_sum()
}

// Optimal hand-pattern: a fixed-stride full-register loop bounded by `min(len)` (so both slices are
// provably in bounds — no per-iteration checks) plus a single masked tail.
#[kernel]
fn dot_opt<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    let n = ctx.lanes();
    let len = a.len().min(b.len());
    let mut acc = ctx.splat(0.0);
    let mut i = 0;
    while i + n <= len {
        acc = acc + ctx.load(&a[i..i + n]) * ctx.load(&b[i..i + n]);
        i += n;
    }
    if i < len {
        let x = ctx.load_partial(&a[i..len], 0.0);
        let y = ctx.load_partial(&b[i..len], 0.0);
        acc = acc + x * y;
    }
    acc.reduce_sum()
}

// `wide`, with a buffer-staged masked SIMD tail (`[0.0; 8]` padded, full-width multiply) — the same
// partial-load strategy as hydroplane's `load_partial`, so both sides are on even ground.
fn dot_wide(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;
    let mut acc = f32x8::splat(0.0);
    for i in 0..chunks {
        let off = i * 8;
        let va = f32x8::from(<[f32; 8]>::try_from(&a[off..off + 8]).unwrap());
        let vb = f32x8::from(<[f32; 8]>::try_from(&b[off..off + 8]).unwrap());
        acc += va * vb;
    }
    let off = chunks * 8;
    let rem = len - off;
    if rem > 0 {
        let mut ba = [0.0f32; 8];
        let mut bb = [0.0f32; 8];
        ba[..rem].copy_from_slice(&a[off..len]);
        bb[..rem].copy_from_slice(&b[off..len]);
        acc += f32x8::from(ba) * f32x8::from(bb);
    }
    acc.reduce_add()
}

// Width-matched to hydroplane's NEON backend (4 lanes), to isolate abstraction overhead from the
// 8-vs-4 lane-count difference. Also a buffer-staged masked SIMD tail.
fn dot_wide4(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 4;
    let mut acc = f32x4::splat(0.0);
    for i in 0..chunks {
        let off = i * 4;
        let va = f32x4::from(<[f32; 4]>::try_from(&a[off..off + 4]).unwrap());
        let vb = f32x4::from(<[f32; 4]>::try_from(&b[off..off + 4]).unwrap());
        acc += va * vb;
    }
    let off = chunks * 4;
    let rem = len - off;
    if rem > 0 {
        let mut ba = [0.0f32; 4];
        let mut bb = [0.0f32; 4];
        ba[..rem].copy_from_slice(&a[off..len]);
        bb[..rem].copy_from_slice(&b[off..len]);
        acc += f32x4::from(ba) * f32x4::from(bb);
    }
    acc.reduce_add()
}

// "f32x16": 16 lanes per step as four *independent* f32x4 accumulators, so four FMA chains run in
// parallel — the point being to feed a CPU with several NEON pipelines (ILP a single accumulator
// can't expose). Masked SIMD tail like the others.
fn dot_wide16(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut acc = [f32x4::splat(0.0); 4];
    let mut off = 0;
    while off + 16 <= len {
        for (j, accj) in acc.iter_mut().enumerate() {
            let o = off + j * 4;
            let va = f32x4::from(<[f32; 4]>::try_from(&a[o..o + 4]).unwrap());
            let vb = f32x4::from(<[f32; 4]>::try_from(&b[o..o + 4]).unwrap());
            *accj += va * vb;
        }
        off += 16;
    }
    let mut acc = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while off + 4 <= len {
        let va = f32x4::from(<[f32; 4]>::try_from(&a[off..off + 4]).unwrap());
        let vb = f32x4::from(<[f32; 4]>::try_from(&b[off..off + 4]).unwrap());
        acc += va * vb;
        off += 4;
    }
    let rem = len - off;
    if rem > 0 {
        let mut ba = [0.0f32; 4];
        let mut bb = [0.0f32; 4];
        ba[..rem].copy_from_slice(&a[off..len]);
        bb[..rem].copy_from_slice(&b[off..len]);
        acc += f32x4::from(ba) * f32x4::from(bb);
    }
    acc.reduce_add()
}

// Multi-accumulator reduction with `K` chosen from the cached runtime saturation point — the warm
// path is one atomic load + a match, then the K-unrolled FMA loop. The library equivalent of
// `dot_wide16`, but with `K` measured per core instead of pinned at four.
#[kernel]
fn dot_zip_reduce<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
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

// The transparent form: a plain sum kernel. No K, no combine, no fills, no init — `zip_sum` implies
// all of them and dispatches the cached per-core unroll factor. Should match `dot_zip_reduce`.
#[kernel]
fn dot_sum<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_sum(a, b, |acc, x, y| x.fma(y, acc))
}

macro_rules! dot_zip_reduce_k {
    ($name:ident, $k:literal) => {
        #[kernel]
        fn $name<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> f32 {
            ctx.zip_reduce_k::<$k, _, _, _>(
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
    };
}
dot_zip_reduce_k!(dot_zip_reduce_k4, 4);
dot_zip_reduce_k!(dot_zip_reduce_k8, 8);
dot_zip_reduce_k!(dot_zip_reduce_k12, 12);
dot_zip_reduce_k!(dot_zip_reduce_k16, 16);

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot");
    for &n in &[6usize, 8, 16, 64, 256, 1024] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();

        let want = dot_scalar(&a, &b);
        let tol = 1e-3 * (1.0 + want.abs());
        assert!((dot(&a, &b) - want).abs() <= tol, "hydroplane disagrees at n={n}");
        assert!((dot_fold(&a, &b) - want).abs() <= tol, "hydroplane_fold disagrees at n={n}");
        assert!((dot_opt(&a, &b) - want).abs() <= tol, "hydroplane_opt disagrees at n={n}");
        assert!((dot_wide(&a, &b) - want).abs() <= tol, "wide8 disagrees at n={n}");
        assert!((dot_wide4(&a, &b) - want).abs() <= tol, "wide4 disagrees at n={n}");
        assert!((dot_wide16(&a, &b) - want).abs() <= tol, "wide16 disagrees at n={n}");
        assert!((dot_zip_reduce(&a, &b) - want).abs() <= tol, "zip_reduce disagrees at n={n}");
        assert!((dot_sum(&a, &b) - want).abs() <= tol, "zip_sum disagrees at n={n}");
        assert!((dot_zip_reduce_k4(&a, &b) - want).abs() <= tol, "zip_reduce_k4 disagrees at n={n}");
        assert!((dot_zip_reduce_k8(&a, &b) - want).abs() <= tol, "zip_reduce_k8 disagrees at n={n}");
        assert!((dot_zip_reduce_k12(&a, &b) - want).abs() <= tol, "zip_reduce_k12 disagrees at n={n}");
        assert!((dot_zip_reduce_k16(&a, &b) - want).abs() <= tol, "zip_reduce_k16 disagrees at n={n}");

        g.bench_with_input(BenchmarkId::new("hydroplane", n), &n, |bch, _| {
            bch.iter(|| dot(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("hydroplane_fold", n), &n, |bch, _| {
            bch.iter(|| dot_fold(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("hydroplane_opt", n), &n, |bch, _| {
            bch.iter(|| dot_opt(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("wide_f32x4", n), &n, |bch, _| {
            bch.iter(|| dot_wide4(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("wide_f32x8", n), &n, |bch, _| {
            bch.iter(|| dot_wide(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("wide_f32x16", n), &n, |bch, _| {
            bch.iter(|| dot_wide16(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("auto", n), &n, |bch, _| {
            bch.iter(|| dot_zip_reduce(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("zip_sum", n), &n, |bch, _| {
            bch.iter(|| dot_sum(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("zip_reduce_k4", n), &n, |bch, _| {
            bch.iter(|| dot_zip_reduce_k4(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("zip_reduce_k8", n), &n, |bch, _| {
            bch.iter(|| dot_zip_reduce_k8(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("zip_reduce_k12", n), &n, |bch, _| {
            bch.iter(|| dot_zip_reduce_k12(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("zip_reduce_k16", n), &n, |bch, _| {
            bch.iter(|| dot_zip_reduce_k16(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| dot_scalar(black_box(&a), black_box(&b)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
