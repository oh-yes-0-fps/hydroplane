//! `#[kernel]` dispatch vs hand-written `wide` SIMD vs scalar, for a small-`N` f32 dot product.
//! On x86 a generic build keeps hydroplane's ops behind the `#[target_feature]` call boundary;
//! pass RUSTFLAGS="-C target-cpu=native" to inline them.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hydroplane::{Gang, kernel};
use std::hint::black_box;
use wide::{f32x4, f32x8};

// `for_each_chunk` with one uniform `(off, cnt)` body, inlined separately for full chunks and tail.
#[kernel]
fn dot<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    let mut acc = ctx.splat(0.0);
    ctx.for_each_chunk::<f32>(a.len(), |off, cnt| {
        let x = ctx.load_partial(&a[off..off + cnt], 0.0);
        let y = ctx.load_partial(&b[off..off + cnt], 0.0);
        acc = acc + x * y;
    });
    acc.reduce_sum()
}

// `chunks_exact` + `remainder`: iterator form of `dot_opt`'s loop.
#[kernel]
fn dot_exact<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    let n = ctx.lanes::<f32>();
    let mut acc = ctx.splat(0.0);
    for off in ctx.chunks_exact::<f32>(a.len()) {
        acc = acc + ctx.load(&a[off..off + n]) * ctx.load(&b[off..off + n]);
    }
    if let Some((off, cnt)) = ctx.remainder::<f32>(a.len()) {
        let x = ctx.load_partial(&a[off..off + cnt], 0.0);
        let y = ctx.load_partial(&b[off..off + cnt], 0.0);
        acc = acc + x * y;
    }
    acc.reduce_sum()
}

// `zip_fold` does the full-register pass + masked tail internally.
#[kernel]
fn dot_fold<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_fold(a, b, 0.0, 0.0, ctx.splat(0.0), |acc, x, y| acc + x * y)
        .reduce_sum()
}

// Hand-written loop: full-register pass bounded by `min(len)` so both slices are provably in
// bounds, plus a single masked tail.
#[kernel]
fn dot_opt<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    let n = ctx.lanes::<f32>();
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

// Same body as `dot_opt`; `tiny` opts out of the `noalias` `_on` call boundary.
#[kernel(tiny)]
fn dot_opt_tiny<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    let n = ctx.lanes::<f32>();
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

// `wide` with a buffer-staged masked tail, the same partial-load strategy as `load_partial`.
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

// Width-matched to the NEON backend (4 lanes) to isolate abstraction overhead from lane count.
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

// 16 lanes per step as four independent f32x4 accumulators, so four FMA chains run in parallel.
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

// Multi-accumulator reduction; the unroll factor K is measured per core and cached, rather than
// pinned at four like `dot_wide16`.
#[kernel]
fn dot_zip_reduce<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
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

// `zip_sum` implies the init, fills, combine, and unroll factor of `dot_zip_reduce`.
#[kernel]
fn dot_sum<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_sum(a, b, |acc, x, y| x.fma(y, acc))
}

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
        assert!((dot_exact(&a, &b) - want).abs() <= tol, "hydroplane_exact disagrees at n={n}");
        assert!((dot_fold(&a, &b) - want).abs() <= tol, "hydroplane_fold disagrees at n={n}");
        assert!((dot_opt(&a, &b) - want).abs() <= tol, "hydroplane_opt disagrees at n={n}");
        assert!((dot_wide(&a, &b) - want).abs() <= tol, "wide8 disagrees at n={n}");
        assert!((dot_wide4(&a, &b) - want).abs() <= tol, "wide4 disagrees at n={n}");
        assert!((dot_wide16(&a, &b) - want).abs() <= tol, "wide16 disagrees at n={n}");
        assert!((dot_zip_reduce(&a, &b) - want).abs() <= tol, "zip_reduce disagrees at n={n}");
        assert!((dot_sum(&a, &b) - want).abs() <= tol, "zip_sum disagrees at n={n}");

        g.bench_with_input(BenchmarkId::new("hydroplane_each", n), &n, |bch, _| {
            bch.iter(|| dot(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("hydroplane_exact", n), &n, |bch, _| {
            bch.iter(|| dot_exact(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("hydroplane_fold", n), &n, |bch, _| {
            bch.iter(|| dot_fold(black_box(&a), black_box(&b)))
        });
        g.bench_with_input(BenchmarkId::new("hydroplane_opt_tiny", n), &n, |bch, _| {
            bch.iter(|| dot_opt_tiny(black_box(&a), black_box(&b)))
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
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |bch, _| {
            bch.iter(|| dot_scalar(black_box(&a), black_box(&b)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
