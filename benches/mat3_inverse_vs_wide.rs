//! Batched 3x3 inverse — the FEM "invert the Jacobian at every quadrature point" kernel. The
//! `Mat3Wide::inverse` combinator is pitted against a hand-written `wide` SIMD kernel (f32x8 and
//! width-matched f32x4) computing the same cofactor/adjugate form, plus a scalar `glam` baseline.
//! The abstraction should land on top of the lane-matched `wide` code, not behind.
//!
//!   cargo bench --features glam --bench mat3_inverse_vs_wide
//!   RUSTFLAGS="-C target-cpu=native" cargo bench --features glam --bench mat3_inverse_vs_wide
//!
//! Storage is a struct-of-arrays of the nine column-major components across `n` matrices; the matrices
//! are diagonally dominant, so every one is non-singular.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use glam::Mat3;
use hydroplane::{Gang, GangGlamExt, kernel};
use std::hint::black_box;
use wide::{f32x4, f32x8};

type Soa9 = [Vec<f32>; 9];

fn data(n: usize) -> Soa9 {
    let mut cols: Soa9 = Default::default();
    for i in 0..n {
        let f = i as f32;
        let diag = 4.0 + (f * 0.09).sin();
        let m = [
            diag,
            (f * 0.013).sin() * 0.5,
            (f * 0.027).cos() * 0.5,
            (f * 0.041).sin() * 0.5,
            diag + (f * 0.05).cos(),
            (f * 0.067).sin() * 0.5,
            (f * 0.083).cos() * 0.5,
            (f * 0.011).sin() * 0.5,
            diag + (f * 0.073).sin(),
        ];
        for c in 0..9 {
            cols[c].push(m[c]);
        }
    }
    cols
}

fn cols_ref(s: &Soa9) -> [&[f32]; 9] {
    std::array::from_fn(|c| s[c].as_slice())
}

#[kernel]
fn invert_hp<'a>(ctx: Gang, m: [&'a [f32]; 9], out: [&'a mut [f32]; 9]) {
    let n = m[0].len();
    let lanes = ctx.lanes::<f32>();
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

macro_rules! invert_wide {
    ($name:ident, $vec:ident, $lanes:literal) => {
        fn $name(m: &Soa9, out: &mut Soa9) {
            let n = m[0].len();
            let mut off = 0;
            while off < n {
                let cnt = $lanes.min(n - off);
                let mut r = [$vec::splat(1.0); 9];
                if cnt == $lanes {
                    for c in 0..9 {
                        r[c] = $vec::from(<[f32; $lanes]>::try_from(&m[c][off..off + $lanes]).unwrap());
                    }
                } else {
                    for c in 0..9 {
                        let mut b = [1.0f32; $lanes];
                        b[..cnt].copy_from_slice(&m[c][off..off + cnt]);
                        r[c] = $vec::from(b);
                    }
                }
                let t0x = r[4] * r[8] - r[5] * r[7];
                let t0y = r[5] * r[6] - r[3] * r[8];
                let t0z = r[3] * r[7] - r[4] * r[6];
                let t1x = r[7] * r[2] - r[8] * r[1];
                let t1y = r[8] * r[0] - r[6] * r[2];
                let t1z = r[6] * r[1] - r[7] * r[0];
                let t2x = r[1] * r[5] - r[2] * r[4];
                let t2y = r[2] * r[3] - r[0] * r[5];
                let t2z = r[0] * r[4] - r[1] * r[3];
                let id = (r[6] * t2x + r[7] * t2y + r[8] * t2z).recip();
                let o = [
                    t0x * id, t1x * id, t2x * id,
                    t0y * id, t1y * id, t2y * id,
                    t0z * id, t1z * id, t2z * id,
                ];
                for c in 0..9 {
                    let a = o[c].to_array();
                    out[c][off..off + cnt].copy_from_slice(&a[..cnt]);
                }
                off += $lanes;
            }
        }
    };
}
invert_wide!(invert_wide8, f32x8, 8);
invert_wide!(invert_wide4, f32x4, 4);

fn invert_scalar(m: &Soa9, out: &mut Soa9) {
    for i in 0..m[0].len() {
        let cols: [f32; 9] = std::array::from_fn(|c| m[c][i]);
        let inv = Mat3::from_cols_array(&cols).inverse().to_cols_array();
        for c in 0..9 {
            out[c][i] = inv[c];
        }
    }
}

fn max_rel_err(a: &Soa9, b: &Soa9) -> f32 {
    let mut e = 0.0f32;
    for c in 0..9 {
        for i in 0..a[c].len() {
            let d = (a[c][i] - b[c][i]).abs() / b[c][i].abs().max(1.0);
            e = e.max(d);
        }
    }
    e
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_inverse");
    for &n in &[15usize, 64, 256, 1024, 4096] {
        let m = data(n);
        let mr = cols_ref(&m);
        let mut want: Soa9 = std::array::from_fn(|_| vec![0.0; n]);
        invert_scalar(&m, &mut want);

        let mut got: Soa9 = std::array::from_fn(|_| vec![0.0; n]);
        {
            let mut it = got.iter_mut();
            let out: [&mut [f32]; 9] = std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
            invert_hp(mr, out);
        }
        assert!(max_rel_err(&got, &want) < 1e-3, "invert_hp n={n}");

        let mut got8: Soa9 = std::array::from_fn(|_| vec![0.0; n]);
        invert_wide8(&m, &mut got8);
        assert!(max_rel_err(&got8, &want) < 1e-3, "wide8 n={n}");
        let mut got4: Soa9 = std::array::from_fn(|_| vec![0.0; n]);
        invert_wide4(&m, &mut got4);
        assert!(max_rel_err(&got4, &want) < 1e-3, "wide4 n={n}");

        g.bench_with_input(BenchmarkId::new("inverse_hp", n), &n, |b, _| {
            b.iter(|| {
                let mut it = got.iter_mut();
                let out: [&mut [f32]; 9] = std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
                invert_hp(black_box(mr), out)
            })
        });
        g.bench_with_input(BenchmarkId::new("inverse_wide_f32x4", n), &n, |b, _| {
            b.iter(|| invert_wide4(black_box(&m), black_box(&mut got4)))
        });
        g.bench_with_input(BenchmarkId::new("inverse_wide_f32x8", n), &n, |b, _| {
            b.iter(|| invert_wide8(black_box(&m), black_box(&mut got8)))
        });
        g.bench_with_input(BenchmarkId::new("inverse_scalar", n), &n, |b, _| {
            b.iter(|| invert_scalar(black_box(&m), black_box(&mut want)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
