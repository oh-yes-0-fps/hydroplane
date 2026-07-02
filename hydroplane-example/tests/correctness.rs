//! Every `*_hp` and `*_wide` workload agrees with its scalar oracle, across sizes that cross the
//! register boundary (including short final tails). Shares the exact implementations the benches time.

use hydroplane_example::{dot, horner, mandelbrot, mat3_inverse, max_rel_err, normalize, saxpy, transform};

const SIZES: [usize; 6] = [0, 1, 7, 8, 9, 1000];
const TOL: f32 = 1e-3;

fn out9(n: usize) -> [Vec<f32>; 9] {
    std::array::from_fn(|_| vec![0.0f32; n])
}
fn mut9(o: &mut [Vec<f32>; 9]) -> [&mut [f32]; 9] {
    let mut it = o.iter_mut();
    std::array::from_fn(|_| it.next().unwrap().as_mut_slice())
}

#[test]
fn saxpy_matches() {
    for n in SIZES {
        let (a, x, y0) = saxpy::inputs(n);
        let mut want = y0.clone();
        saxpy::saxpy_scalar(a, &x, &mut want);
        let mut h = y0.clone();
        saxpy::saxpy_hp(a, &x, &mut h);
        let mut w = y0.clone();
        saxpy::saxpy_wide(a, &x, &mut w);
        assert!(max_rel_err(&h, &want) < TOL, "hp n={n}");
        assert!(max_rel_err(&w, &want) < TOL, "wide n={n}");
    }
}

#[test]
fn dot_matches() {
    for n in SIZES {
        let (x, y) = dot::inputs(n);
        let want = dot::dot_scalar(&x, &y);
        let tol = TOL * want.abs().max(1.0);
        assert!((dot::dot_hp(&x, &y) - want).abs() <= tol, "hp n={n}");
        assert!((dot::dot_wide(&x, &y) - want).abs() <= tol, "wide n={n}");
    }
}

#[test]
fn horner_matches() {
    for n in SIZES {
        let x = horner::inputs(n);
        let (mut want, mut h, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        horner::horner_scalar(&horner::COEFFS, &x, &mut want);
        horner::horner_hp(horner::COEFFS, &x, &mut h);
        horner::horner_wide(&horner::COEFFS, &x, &mut w);
        assert!(max_rel_err(&h, &want) < TOL, "hp n={n}");
        assert!(max_rel_err(&w, &want) < TOL, "wide n={n}");
    }
}

#[test]
fn normalize_matches() {
    for n in SIZES {
        let [x, y, z] = normalize::inputs(n);
        let (mut wx, mut wy, mut wz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        normalize::normalize_scalar(&x, &y, &z, &mut wx, &mut wy, &mut wz);
        let (mut hx, mut hy, mut hz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        normalize::normalize_hp(&x, &y, &z, &mut hx, &mut hy, &mut hz);
        let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        normalize::normalize_wide(&x, &y, &z, &mut ax, &mut ay, &mut az);
        for (got, want) in [(&hx, &wx), (&hy, &wy), (&hz, &wz), (&ax, &wx), (&ay, &wy), (&az, &wz)] {
            assert!(max_rel_err(got, want) < TOL, "n={n}");
        }
    }
}

#[test]
fn transform_matches() {
    for n in SIZES {
        let (m, v) = transform::inputs(n);
        let (mut wx, mut wy, mut wz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        transform::transform_scalar(&m, &v[0], &v[1], &v[2], &mut wx, &mut wy, &mut wz);
        let mref: [&[f32]; 9] = std::array::from_fn(|c| m[c].as_slice());
        let (mut hx, mut hy, mut hz) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        transform::transform_hp(mref, &v[0], &v[1], &v[2], &mut hx, &mut hy, &mut hz);
        let (mut ax, mut ay, mut az) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        transform::transform_wide(&m, &v[0], &v[1], &v[2], &mut ax, &mut ay, &mut az);
        for (got, want) in [(&hx, &wx), (&hy, &wy), (&hz, &wz), (&ax, &wx), (&ay, &wy), (&az, &wz)] {
            assert!(max_rel_err(got, want) < TOL, "n={n}");
        }
    }
}

#[test]
fn mat3_inverse_matches() {
    for n in SIZES {
        let m = mat3_inverse::inputs(n);
        let mut want = out9(n);
        mat3_inverse::invert_scalar(&m, &mut want);
        let mref: [&[f32]; 9] = std::array::from_fn(|c| m[c].as_slice());
        let mut h = out9(n);
        mat3_inverse::invert_hp(mref, mut9(&mut h));
        let mut w = out9(n);
        mat3_inverse::invert_wide(&m, &mut w);
        for c in 0..9 {
            assert!(max_rel_err(&h[c], &want[c]) < TOL, "hp n={n} c={c}");
            assert!(max_rel_err(&w[c], &want[c]) < TOL, "wide n={n} c={c}");
        }
    }
}

#[test]
fn mandelbrot_matches() {
    for n in SIZES {
        let (cx, cy) = mandelbrot::inputs(n);
        let mi = mandelbrot::MAX_ITER;
        let (mut want, mut h, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        mandelbrot::mandelbrot_scalar(&cx, &cy, mi, &mut want);
        mandelbrot::mandelbrot_hp(&cx, &cy, mi, &mut h);
        mandelbrot::mandelbrot_wide(&cx, &cy, mi, &mut w);
        assert_eq!(h, want, "hp n={n}");
        assert_eq!(w, want, "wide n={n}");
    }
}
