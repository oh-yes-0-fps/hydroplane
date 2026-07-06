//! Every workload benched three ways (hp / wide / scalar), asserted against the scalar oracle
//! before timing. Run: `cargo bench --bench workloads` (optionally with `-C target-cpu=native`).

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hydroplane_example::{
    asum, cmul, cosine, dot, double_polysum, horner, l1dist, l2norm, mandelbrot, mat3_inverse,
    max_rel_err, normalize, polysum, saxpy, transform,
};
use std::hint::black_box;

const SIZES: [usize; 5] = [15, 64, 256, 1024, 4096];
const TOL: f32 = 1e-3;

fn out9(n: usize) -> [Vec<f32>; 9] {
    std::array::from_fn(|_| vec![0.0f32; n])
}
fn mut9(o: &mut [Vec<f32>; 9]) -> [&mut [f32]; 9] {
    let mut it = o.iter_mut();
    std::array::from_fn(|_| it.next().unwrap().as_mut_slice())
}

fn bench_saxpy(c: &mut Criterion) {
    let mut g = c.benchmark_group("saxpy");
    for &n in &SIZES {
        let (a, x, y0) = saxpy::inputs(n);
        let mut want = y0.clone();
        saxpy::saxpy_scalar(a, &x, &mut want);
        let mut yh = y0.clone();
        saxpy::saxpy_hp(a, &x, &mut yh);
        let mut yw = y0.clone();
        saxpy::saxpy_wide(a, &x, &mut yw);
        assert!(max_rel_err(&yh, &want) < TOL, "saxpy hp n={n}");
        assert!(max_rel_err(&yw, &want) < TOL, "saxpy wide n={n}");

        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| saxpy::saxpy_hp(black_box(a), black_box(&x), black_box(&mut y)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| saxpy::saxpy_wide(black_box(a), black_box(&x), black_box(&mut y)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let mut y = y0.clone();
            b.iter(|| saxpy::saxpy_scalar(black_box(a), black_box(&x), black_box(&mut y)))
        });
    }
    g.finish();
}

fn bench_dot(c: &mut Criterion) {
    let mut g = c.benchmark_group("dot");
    for &n in &SIZES {
        let (x, y) = dot::inputs(n);
        let want = dot::dot_scalar(&x, &y);
        assert!((dot::dot_hp(&x, &y) - want).abs() <= TOL * want.abs().max(1.0), "dot hp n={n}");
        assert!((dot::dot_wide(&x, &y) - want).abs() <= TOL * want.abs().max(1.0), "dot wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| dot::dot_hp(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| dot::dot_wide(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| dot::dot_scalar(black_box(&x), black_box(&y)))
        });
    }
    g.finish();
}

fn bench_horner(c: &mut Criterion) {
    let mut g = c.benchmark_group("horner");
    for &n in &SIZES {
        let x = horner::inputs(n);
        let (mut want, mut oh, mut ow) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        horner::horner_scalar(&horner::COEFFS, &x, &mut want);
        horner::horner_hp(horner::COEFFS, &x, &mut oh);
        horner::horner_wide(&horner::COEFFS, &x, &mut ow);
        assert!(max_rel_err(&oh, &want) < TOL, "horner hp n={n}");
        assert!(max_rel_err(&ow, &want) < TOL, "horner wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let mut o = vec![0.0; n];
            b.iter(|| horner::horner_hp(black_box(horner::COEFFS), black_box(&x), black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let mut o = vec![0.0; n];
            b.iter(|| horner::horner_wide(black_box(&horner::COEFFS), black_box(&x), black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let mut o = vec![0.0; n];
            b.iter(|| horner::horner_scalar(black_box(&horner::COEFFS), black_box(&x), black_box(&mut o)))
        });
    }
    g.finish();
}

fn bench_normalize(c: &mut Criterion) {
    let mut g = c.benchmark_group("normalize");
    for &n in &SIZES {
        let [x, y, z] = normalize::inputs(n);
        let (mut wx, mut wy, mut wz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        normalize::normalize_scalar(&x, &y, &z, &mut wx, &mut wy, &mut wz);
        let (mut hx, mut hy, mut hz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        normalize::normalize_hp(&x, &y, &z, &mut hx, &mut hy, &mut hz);
        assert!(max_rel_err(&hx, &wx).max(max_rel_err(&hy, &wy)).max(max_rel_err(&hz, &wz)) < TOL, "normalize hp n={n}");

        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            b.iter(|| normalize::normalize_hp(black_box(&x), black_box(&y), black_box(&z), &mut ox, &mut oy, &mut oz))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            b.iter(|| normalize::normalize_wide(black_box(&x), black_box(&y), black_box(&z), &mut ox, &mut oy, &mut oz))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            b.iter(|| normalize::normalize_scalar(black_box(&x), black_box(&y), black_box(&z), &mut ox, &mut oy, &mut oz))
        });
    }
    g.finish();
}

fn bench_transform(c: &mut Criterion) {
    let mut g = c.benchmark_group("transform");
    for &n in &SIZES {
        let (m, v) = transform::inputs(n);
        let (mut wx, mut wy, mut wz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        transform::transform_scalar(&m, &v[0], &v[1], &v[2], &mut wx, &mut wy, &mut wz);
        let (mut hx, mut hy, mut hz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mref: [&[f32]; 9] = std::array::from_fn(|c| m[c].as_slice());
        transform::transform_hp(mref, &v[0], &v[1], &v[2], &mut hx, &mut hy, &mut hz);
        assert!(max_rel_err(&hx, &wx).max(max_rel_err(&hy, &wy)).max(max_rel_err(&hz, &wz)) < TOL, "transform hp n={n}");

        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            b.iter(|| transform::transform_hp(black_box(mref), black_box(&v[0]), &v[1], &v[2], &mut ox, &mut oy, &mut oz))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            b.iter(|| transform::transform_wide(black_box(&m), black_box(&v[0]), &v[1], &v[2], &mut ox, &mut oy, &mut oz))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let (mut ox, mut oy, mut oz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            b.iter(|| transform::transform_scalar(black_box(&m), black_box(&v[0]), &v[1], &v[2], &mut ox, &mut oy, &mut oz))
        });
    }
    g.finish();
}

fn bench_mat3_inverse(c: &mut Criterion) {
    let mut g = c.benchmark_group("mat3_inverse");
    for &n in &SIZES {
        let m = mat3_inverse::inputs(n);
        let mref: [&[f32]; 9] = std::array::from_fn(|c| m[c].as_slice());
        let mut want = out9(n);
        mat3_inverse::invert_scalar(&m, &mut want);
        let mut gh = out9(n);
        mat3_inverse::invert_hp(mref, mut9(&mut gh));
        let mut gw = out9(n);
        mat3_inverse::invert_wide(&m, &mut gw);
        for c in 0..9 {
            assert!(max_rel_err(&gh[c], &want[c]) < TOL, "inverse hp n={n} c={c}");
            assert!(max_rel_err(&gw[c], &want[c]) < TOL, "inverse wide n={n} c={c}");
        }
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let mut o = out9(n);
            b.iter(|| mat3_inverse::invert_hp(black_box(mref), mut9(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let mut o = out9(n);
            b.iter(|| mat3_inverse::invert_wide(black_box(&m), black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let mut o = out9(n);
            b.iter(|| mat3_inverse::invert_scalar(black_box(&m), black_box(&mut o)))
        });
    }
    g.finish();
}

fn bench_mandelbrot(c: &mut Criterion) {
    let mut g = c.benchmark_group("mandelbrot");
    let mi = mandelbrot::MAX_ITER;
    for &n in &SIZES {
        let (cx, cy) = mandelbrot::inputs(n);
        let (mut want, mut oh, mut ow) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        mandelbrot::mandelbrot_scalar(&cx, &cy, mi, &mut want);
        mandelbrot::mandelbrot_hp(&cx, &cy, mi, &mut oh);
        mandelbrot::mandelbrot_wide(&cx, &cy, mi, &mut ow);
        assert!(max_rel_err(&oh, &want) < TOL, "mandelbrot hp n={n}");
        assert!(max_rel_err(&ow, &want) < TOL, "mandelbrot wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let mut o = vec![0.0; n];
            b.iter(|| mandelbrot::mandelbrot_hp(black_box(&cx), black_box(&cy), mi, black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let mut o = vec![0.0; n];
            b.iter(|| mandelbrot::mandelbrot_wide(black_box(&cx), black_box(&cy), mi, black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let mut o = vec![0.0; n];
            b.iter(|| mandelbrot::mandelbrot_scalar(black_box(&cx), black_box(&cy), mi, black_box(&mut o)))
        });
    }
    g.finish();
}

fn bench_l2norm(c: &mut Criterion) {
    let mut g = c.benchmark_group("l2norm");
    for &n in &SIZES {
        let x = l2norm::inputs(n);
        let want = l2norm::l2norm_scalar(&x);
        assert!((l2norm::l2norm_hp(&x) - want).abs() <= TOL * want.abs().max(1.0), "l2norm hp n={n}");
        assert!((l2norm::l2norm_wide(&x) - want).abs() <= TOL * want.abs().max(1.0), "l2norm wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| l2norm::l2norm_hp(black_box(&x)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| l2norm::l2norm_wide(black_box(&x)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| l2norm::l2norm_scalar(black_box(&x)))
        });
    }
    g.finish();
}

fn bench_asum(c: &mut Criterion) {
    let mut g = c.benchmark_group("asum");
    for &n in &SIZES {
        let x = asum::inputs(n);
        let want = asum::asum_scalar(&x);
        assert!((asum::asum_hp(&x) - want).abs() <= TOL * want.abs().max(1.0), "asum hp n={n}");
        assert!((asum::asum_wide(&x) - want).abs() <= TOL * want.abs().max(1.0), "asum wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| asum::asum_hp(black_box(&x)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| asum::asum_wide(black_box(&x)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| asum::asum_scalar(black_box(&x)))
        });
    }
    g.finish();
}

fn bench_l1dist(c: &mut Criterion) {
    let mut g = c.benchmark_group("l1dist");
    for &n in &SIZES {
        let (x, y) = l1dist::inputs(n);
        let want = l1dist::l1dist_scalar(&x, &y);
        assert!((l1dist::l1dist_hp(&x, &y) - want).abs() <= TOL * want.abs().max(1.0), "l1dist hp n={n}");
        assert!((l1dist::l1dist_wide(&x, &y) - want).abs() <= TOL * want.abs().max(1.0), "l1dist wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| l1dist::l1dist_hp(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| l1dist::l1dist_wide(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| l1dist::l1dist_scalar(black_box(&x), black_box(&y)))
        });
    }
    g.finish();
}

fn bench_polysum(c: &mut Criterion) {
    let mut g = c.benchmark_group("polysum");
    for &n in &SIZES {
        let x = polysum::inputs(n);
        let want = polysum::polysum_scalar(&polysum::COEFFS, &x);
        assert!((polysum::polysum_hp(polysum::COEFFS, &x) - want).abs() <= TOL * want.abs().max(1.0), "polysum hp n={n}");
        assert!((polysum::polysum_wide(&polysum::COEFFS, &x) - want).abs() <= TOL * want.abs().max(1.0), "polysum wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| polysum::polysum_hp(black_box(polysum::COEFFS), black_box(&x)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| polysum::polysum_wide(black_box(&polysum::COEFFS), black_box(&x)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| polysum::polysum_scalar(black_box(&polysum::COEFFS), black_box(&x)))
        });
    }
    g.finish();
}

fn bench_cmul(c: &mut Criterion) {
    let mut g = c.benchmark_group("cmul");
    for &n in &SIZES {
        let (ar, ai, br, bi) = cmul::inputs(n);
        let (mut wr, mut wi) = (vec![0.0; n], vec![0.0; n]);
        cmul::cmul_scalar(&ar, &ai, &br, &bi, &mut wr, &mut wi);
        let (mut hr, mut hi) = (vec![0.0; n], vec![0.0; n]);
        cmul::cmul_hp(&ar, &ai, &br, &bi, &mut hr, &mut hi);
        let (mut vr, mut vi) = (vec![0.0; n], vec![0.0; n]);
        cmul::cmul_wide(&ar, &ai, &br, &bi, &mut vr, &mut vi);
        assert!(max_rel_err(&hr, &wr).max(max_rel_err(&hi, &wi)) < TOL, "cmul hp n={n}");
        assert!(max_rel_err(&vr, &wr).max(max_rel_err(&vi, &wi)) < TOL, "cmul wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            let (mut or, mut oi) = (vec![0.0; n], vec![0.0; n]);
            b.iter(|| cmul::cmul_hp(black_box(&ar), black_box(&ai), black_box(&br), black_box(&bi), &mut or, &mut oi))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            let (mut or, mut oi) = (vec![0.0; n], vec![0.0; n]);
            b.iter(|| cmul::cmul_wide(black_box(&ar), black_box(&ai), black_box(&br), black_box(&bi), &mut or, &mut oi))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            let (mut or, mut oi) = (vec![0.0; n], vec![0.0; n]);
            b.iter(|| cmul::cmul_scalar(black_box(&ar), black_box(&ai), black_box(&br), black_box(&bi), &mut or, &mut oi))
        });
    }
    g.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut g = c.benchmark_group("cosine");
    for &n in &SIZES {
        let (x, y) = cosine::inputs(n);
        let want = cosine::cosine_scalar(&x, &y);
        assert!((cosine::cosine_hp(&x, &y) - want).abs() <= TOL * want.abs().max(1.0), "cosine hp n={n}");
        assert!((cosine::cosine_wide(&x, &y) - want).abs() <= TOL * want.abs().max(1.0), "cosine wide n={n}");
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| cosine::cosine_hp(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| cosine::cosine_wide(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| cosine::cosine_scalar(black_box(&x), black_box(&y)))
        });
    }
    g.finish();
}

fn bench_double_polysum(c: &mut Criterion) {
    let mut g = c.benchmark_group("double_polysum");
    for &n in &SIZES {
        let (x, y) = double_polysum::inputs(n);
        let want = double_polysum::double_polysum_scalar(&x, &y);
        assert!(
            (double_polysum::double_polysum_hp(&x, &y) - want).abs() <= TOL * want.abs().max(1.0),
            "double_polysum hp n={n}"
        );
        assert!(
            (double_polysum::double_polysum_wide(&x, &y) - want).abs() <= TOL * want.abs().max(1.0),
            "double_polysum wide n={n}"
        );
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| double_polysum::double_polysum_hp(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("wide", n), &n, |b, _| {
            b.iter(|| double_polysum::double_polysum_wide(black_box(&x), black_box(&y)))
        });
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| double_polysum::double_polysum_scalar(black_box(&x), black_box(&y)))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_saxpy,
    bench_dot,
    bench_horner,
    bench_normalize,
    bench_transform,
    bench_mat3_inverse,
    bench_mandelbrot,
    bench_l2norm,
    bench_asum,
    bench_l1dist,
    bench_polysum,
    bench_cmul,
    bench_cosine,
    bench_double_polysum
);
criterion_main!(benches);
