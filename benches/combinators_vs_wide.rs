//! Zero-cost check for the new fixed-`N` column combinators: each is pitted against a hand-written
//! `wide` SIMD kernel (f32x8 and width-matched f32x4) using the same buffer-staged masked tail, plus
//! a scalar baseline. The abstraction should land on top of the lane-matched `wide` code, not behind.
//!
//!   cargo bench --bench combinators_vs_wide
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --bench combinators_vs_wide
//!
//! Workload is bounding-sphere broadphase over an SoA: `count_n` (tally overlaps),
//! `for_each_hit_n` (mark every overlap), and `masked_chunks` + `gather_n` (two-phase any-overlap
//! from an AoS slice).

#![allow(deprecated)]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hydroplane::{Gang, kernel};
use std::hint::black_box;
use wide::{f32x4, f32x8};

fn data(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut xs, mut ys, mut zs, mut rs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let f = i as f32;
        xs.push((f * 0.37).sin() * 10.0);
        ys.push((f * 0.51).cos() * 10.0);
        zs.push((f * 0.13).sin() * 10.0);
        rs.push(0.5 + (f * 0.07).cos().abs());
    }
    (xs, ys, zs, rs)
}

const Q: [f32; 4] = [1.0, -2.0, 0.5, 4.0];

#[kernel]
fn count_hp<'a>(ctx: Gang<f32>, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32], rs: &'a [f32], q: [f32; 4]) -> usize {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    ctx.count_n([xs, ys, zs, rs], |[x, y, z, r]| {
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        (dx * dx + dy * dy + dz * dz).le(rsum * rsum)
    })
}

macro_rules! count_wide {
    ($name:ident, $vec:ident, $lanes:literal) => {
        fn $name(xs: &[f32], ys: &[f32], zs: &[f32], rs: &[f32], q: [f32; 4]) -> usize {
            let len = xs.len();
            let (cx, cy, cz, sr) = ($vec::splat(q[0]), $vec::splat(q[1]), $vec::splat(q[2]), $vec::splat(q[3]));
            let (one, zero) = ($vec::splat(1.0), $vec::splat(0.0));
            let mut acc = $vec::splat(0.0);
            let chunks = len / $lanes;
            for i in 0..chunks {
                let o = i * $lanes;
                let x = cx - $vec::from(<[f32; $lanes]>::try_from(&xs[o..o + $lanes]).unwrap());
                let y = cy - $vec::from(<[f32; $lanes]>::try_from(&ys[o..o + $lanes]).unwrap());
                let z = cz - $vec::from(<[f32; $lanes]>::try_from(&zs[o..o + $lanes]).unwrap());
                let rsum = sr + $vec::from(<[f32; $lanes]>::try_from(&rs[o..o + $lanes]).unwrap());
                let m = (x * x + y * y + z * z).simd_le(rsum * rsum);
                acc += m.blend(one, zero);
            }
            let off = chunks * $lanes;
            let rem = len - off;
            if rem > 0 {
                let (mut bx, mut by, mut bz) = ([0.0f32; $lanes], [0.0f32; $lanes], [0.0f32; $lanes]);
                let mut br = [f32::NAN; $lanes];
                bx[..rem].copy_from_slice(&xs[off..len]);
                by[..rem].copy_from_slice(&ys[off..len]);
                bz[..rem].copy_from_slice(&zs[off..len]);
                br[..rem].copy_from_slice(&rs[off..len]);
                let x = cx - $vec::from(bx);
                let y = cy - $vec::from(by);
                let z = cz - $vec::from(bz);
                let rsum = sr + $vec::from(br);
                let m = (x * x + y * y + z * z).simd_le(rsum * rsum);
                acc += m.blend(one, zero);
            }
            acc.reduce_add() as usize
        }
    };
}
count_wide!(count_wide8, f32x8, 8);
count_wide!(count_wide4, f32x4, 4);

// Hand-written ILP reference: four independent f32x4 count chains (16 lanes/iter), the same
// superscalar shape `count_n` engages — so a fair "did the abstraction get the ILP win" baseline.
fn count_wide_ilp(xs: &[f32], ys: &[f32], zs: &[f32], rs: &[f32], q: [f32; 4]) -> usize {
    let len = xs.len();
    let (cx, cy, cz, sr) = (f32x4::splat(q[0]), f32x4::splat(q[1]), f32x4::splat(q[2]), f32x4::splat(q[3]));
    let (one, zero) = (f32x4::splat(1.0), f32x4::splat(0.0));
    let mut acc = [f32x4::splat(0.0); 4];
    let mut off = 0;
    while off + 16 <= len {
        for (j, accj) in acc.iter_mut().enumerate() {
            let o = off + j * 4;
            let x = cx - f32x4::from(<[f32; 4]>::try_from(&xs[o..o + 4]).unwrap());
            let y = cy - f32x4::from(<[f32; 4]>::try_from(&ys[o..o + 4]).unwrap());
            let z = cz - f32x4::from(<[f32; 4]>::try_from(&zs[o..o + 4]).unwrap());
            let rsum = sr + f32x4::from(<[f32; 4]>::try_from(&rs[o..o + 4]).unwrap());
            *accj += (x * x + y * y + z * z).simd_le(rsum * rsum).blend(one, zero);
        }
        off += 16;
    }
    let mut acc = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    while off < len {
        let cnt = 4.min(len - off);
        let (mut bx, mut by, mut bz) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let mut br = [f32::NAN; 4];
        bx[..cnt].copy_from_slice(&xs[off..off + cnt]);
        by[..cnt].copy_from_slice(&ys[off..off + cnt]);
        bz[..cnt].copy_from_slice(&zs[off..off + cnt]);
        br[..cnt].copy_from_slice(&rs[off..off + cnt]);
        let x = cx - f32x4::from(bx);
        let y = cy - f32x4::from(by);
        let z = cz - f32x4::from(bz);
        let rsum = sr + f32x4::from(br);
        acc += (x * x + y * y + z * z).simd_le(rsum * rsum).blend(one, zero);
        off += 4;
    }
    acc.reduce_add() as usize
}

fn count_scalar(xs: &[f32], ys: &[f32], zs: &[f32], rs: &[f32], q: [f32; 4]) -> usize {
    (0..xs.len())
        .filter(|&i| {
            let (dx, dy, dz) = (q[0] - xs[i], q[1] - ys[i], q[2] - zs[i]);
            let rsum = q[3] + rs[i];
            dx * dx + dy * dy + dz * dz <= rsum * rsum
        })
        .count()
}

// Full-scan count over an AoS `&[[f32; 4]]` via `gather_n` — deterministic throughput (no early
// exit). Masked variant: `masked_chunks` active mask `&`-ed into the predicate (always correct).
#[kernel]
fn gather_count_hp<'a>(ctx: Gang<f32>, pts: &'a [[f32; 4]], q: [f32; 4]) -> usize {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    let (one, zero) = (ctx.splat(1.0), ctx.splat(0.0));
    let mut acc = zero;
    for (off, cnt, active) in ctx.masked_chunks(pts.len()) {
        let [x, y, z, r] = ctx.gather_n(&pts[off..off + cnt], [0.0; 4], |p| *p);
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        acc = acc + one.select((dx * dx + dy * dy + dz * dz).le(rsum * rsum) & active, zero);
    }
    acc.reduce_sum() as usize
}

// Sentinel variant: NaN-fill the radius lane so inactive tail lanes self-reject — no active mask,
// matching the hand-written `wide` strategy exactly.
#[kernel]
fn gather_count_sentinel_hp<'a>(ctx: Gang<f32>, pts: &'a [[f32; 4]], q: [f32; 4]) -> usize {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    let (one, zero) = (ctx.splat(1.0), ctx.splat(0.0));
    let mut acc = zero;
    for (off, cnt) in ctx.chunks(pts.len()) {
        let [x, y, z, r] = ctx.gather_n(&pts[off..off + cnt], [0.0, 0.0, 0.0, f32::NAN], |p| *p);
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        acc = acc + one.select((dx * dx + dy * dy + dz * dz).le(rsum * rsum), zero);
    }
    acc.reduce_sum() as usize
}

macro_rules! gather_count_wide {
    ($name:ident, $vec:ident, $lanes:literal) => {
        fn $name(pts: &[[f32; 4]], q: [f32; 4]) -> usize {
            let len = pts.len();
            let (cx, cy, cz, sr) = ($vec::splat(q[0]), $vec::splat(q[1]), $vec::splat(q[2]), $vec::splat(q[3]));
            let (one, zero) = ($vec::splat(1.0), $vec::splat(0.0));
            let mut acc = zero;
            let mut off = 0;
            while off < len {
                let cnt = $lanes.min(len - off);
                let (mut bx, mut by, mut bz) = ([0.0f32; $lanes], [0.0f32; $lanes], [0.0f32; $lanes]);
                let mut br = [f32::NAN; $lanes];
                for j in 0..cnt {
                    let p = pts[off + j];
                    bx[j] = p[0];
                    by[j] = p[1];
                    bz[j] = p[2];
                    br[j] = p[3];
                }
                let x = cx - $vec::from(bx);
                let y = cy - $vec::from(by);
                let z = cz - $vec::from(bz);
                let rsum = sr + $vec::from(br);
                let m = (x * x + y * y + z * z).simd_le(rsum * rsum);
                acc += m.blend(one, zero);
                off += $lanes;
            }
            acc.reduce_add() as usize
        }
    };
}
gather_count_wide!(gather_count_wide8, f32x8, 8);
gather_count_wide!(gather_count_wide4, f32x4, 4);

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("broadphase");
    for &n in &[15usize, 64, 256, 1024, 4096] {
        let (xs, ys, zs, rs) = data(n);
        let pts: Vec<[f32; 4]> = (0..n).map(|i| [xs[i], ys[i], zs[i], rs[i]]).collect();

        let want = count_scalar(&xs, &ys, &zs, &rs, Q);
        assert_eq!(count_hp(&xs, &ys, &zs, &rs, Q), want, "count_n n={n}");
        assert_eq!(count_wide8(&xs, &ys, &zs, &rs, Q), want, "wide8 n={n}");
        assert_eq!(count_wide4(&xs, &ys, &zs, &rs, Q), want, "wide4 n={n}");
        assert_eq!(count_wide_ilp(&xs, &ys, &zs, &rs, Q), want, "wide_ilp n={n}");
        assert_eq!(gather_count_hp(&pts, Q), want, "gather_n n={n}");
        assert_eq!(gather_count_sentinel_hp(&pts, Q), want, "gather_n sentinel n={n}");
        assert_eq!(gather_count_wide8(&pts, Q), want, "gather wide8 n={n}");
        assert_eq!(gather_count_wide4(&pts, Q), want, "gather wide4 n={n}");

        g.bench_with_input(BenchmarkId::new("count_n", n), &n, |b, _| {
            b.iter(|| count_hp(black_box(&xs), black_box(&ys), black_box(&zs), black_box(&rs), Q))
        });
        g.bench_with_input(BenchmarkId::new("count_wide_f32x4", n), &n, |b, _| {
            b.iter(|| count_wide4(black_box(&xs), black_box(&ys), black_box(&zs), black_box(&rs), Q))
        });
        g.bench_with_input(BenchmarkId::new("count_wide_f32x8", n), &n, |b, _| {
            b.iter(|| count_wide8(black_box(&xs), black_box(&ys), black_box(&zs), black_box(&rs), Q))
        });
        g.bench_with_input(BenchmarkId::new("count_wide_ilp_4x", n), &n, |b, _| {
            b.iter(|| count_wide_ilp(black_box(&xs), black_box(&ys), black_box(&zs), black_box(&rs), Q))
        });
        g.bench_with_input(BenchmarkId::new("count_scalar", n), &n, |b, _| {
            b.iter(|| count_scalar(black_box(&xs), black_box(&ys), black_box(&zs), black_box(&rs), Q))
        });
        g.bench_with_input(BenchmarkId::new("gather_n", n), &n, |b, _| {
            b.iter(|| gather_count_hp(black_box(&pts), Q))
        });
        g.bench_with_input(BenchmarkId::new("gather_n_sentinel", n), &n, |b, _| {
            b.iter(|| gather_count_sentinel_hp(black_box(&pts), Q))
        });
        g.bench_with_input(BenchmarkId::new("gather_wide_f32x4", n), &n, |b, _| {
            b.iter(|| gather_count_wide4(black_box(&pts), Q))
        });
        g.bench_with_input(BenchmarkId::new("gather_wide_f32x8", n), &n, |b, _| {
            b.iter(|| gather_count_wide8(black_box(&pts), Q))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
