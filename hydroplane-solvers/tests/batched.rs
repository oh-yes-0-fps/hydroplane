use hydroplane::{FloatScalar, SimdDispatch};
use hydroplane_solvers::batched::{cholesky_solve, ldlt_solve, polar3, svd3, sym3_eigen};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

type M = [[f64; 3]; 3];

fn mul(a: &M, b: &M) -> M {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

fn t(a: &M) -> M {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[j][i];
        }
    }
    c
}

fn det(a: &M) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn norm(a: &M) -> f64 {
    a.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}

fn diff(a: &M, b: &M) -> f64 {
    let mut d = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            d += (a[i][j] - b[i][j]).powi(2);
        }
    }
    d.sqrt()
}

fn rotation(r: &mut Lcg) -> M {
    let (mut w, mut x, mut y, mut z) = (r.next(), r.next(), r.next(), r.next());
    let n = (w * w + x * x + y * y + z * z).sqrt();
    (w, x, y, z) = (w / n, x / n, y / n, z / n);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Symmetric test matrices, including the cases closed-form solvers get wrong.
fn symmetric_cases(n: usize) -> Vec<M> {
    let mut r = Lcg(7);
    let mut out = vec![
        [[0.0; 3]; 3],
        [[3.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 2.0]],
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
    ];
    while out.len() < n {
        let q = rotation(&mut r);
        let scale = 10f64.powf(r.next() * 3.0);
        let d = match out.len() % 4 {
            0 => [r.next(), r.next(), r.next()],
            1 => [1.0, 1.0, 1.0 + r.next() * 1e-9],
            2 => [0.0, 0.0, r.next()],
            _ => [r.next(), r.next(), r.next() * 1e-6],
        };
        let dm = [[d[0] * scale, 0.0, 0.0], [0.0, d[1] * scale, 0.0], [0.0, 0.0, d[2] * scale]];
        out.push(mul(&mul(&q, &dm), &t(&q)));
    }
    out.truncate(n);
    out
}

fn general_cases(n: usize) -> Vec<M> {
    let mut r = Lcg(11);
    let mut out = vec![[[0.0; 3]; 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]];
    while out.len() < n {
        let mut a = [[0.0; 3]; 3];
        for row in &mut a {
            for x in row {
                *x = r.next() * 5.0;
            }
        }
        if out.len() % 5 == 0 {
            a[2] = [a[0][0] + a[1][0], a[0][1] + a[1][1], a[0][2] + a[1][2]];
        }
        out.push(a);
    }
    out.truncate(n);
    out
}

fn cols<T: FloatScalar>(ms: &[M], entries: &[(usize, usize)]) -> Vec<Vec<T>> {
    entries.iter().map(|&(i, j)| ms.iter().map(|m| T::from_f64(m[i][j])).collect()).collect()
}

const SYM: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
const FULL: [(usize, usize); 9] = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)];

fn mat<T: FloatScalar>(c: &[Vec<T>], k: usize) -> M {
    let mut m = [[0.0; 3]; 3];
    for (e, &(i, j)) in FULL.iter().enumerate() {
        m[i][j] = c[e][k].into_f64();
    }
    m
}

fn check_orthonormal(v: &M, tol: f64, what: &str) {
    let i3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(diff(&mul(&t(v), v), &i3) < tol, "{what} not orthonormal: {v:?}");
    assert!((det(v) - 1.0).abs() < tol, "{what} det {} != 1", det(v));
}

fn eigen<T: FloatScalar + SimdDispatch>(tol: f64) {
    for n in [1, 7, 33, 500] {
        let ms = symmetric_cases(n);
        let a = cols::<T>(&ms, &SYM);
        let (mut vals, mut vecs) = (vec![vec![T::ZERO; n]; 3], vec![vec![T::ZERO; n]; 9]);
        {
            let [v0, v1, v2] = &mut vals[..] else { unreachable!() };
            let mut vv = vecs.iter_mut();
            let vref: [&mut [T]; 9] = core::array::from_fn(|_| vv.next().unwrap().as_mut_slice());
            sym3_eigen(core::array::from_fn(|k| a[k].as_slice()), [v0, v1, v2], vref);
        }
        for (k, m) in ms.iter().enumerate() {
            let lam: [f64; 3] = core::array::from_fn(|i| vals[i][k].into_f64());
            assert!(lam[0] <= lam[1] && lam[1] <= lam[2], "not ascending: {lam:?}");
            let v = mat(&vecs, k);
            check_orthonormal(&v, tol * 10.0, "eigenvectors");
            let d = [[lam[0], 0.0, 0.0], [0.0, lam[1], 0.0], [0.0, 0.0, lam[2]]];
            let back = mul(&mul(&v, &d), &t(&v));
            assert!(diff(&back, m) <= tol * norm(m).max(1e-30), "case {k}: {m:?} vs {back:?}");
        }
    }
}

#[test]
fn sym3_eigen_f64() {
    eigen::<f64>(1e-12);
}

#[test]
fn sym3_eigen_f32() {
    eigen::<f32>(2e-6);
}

fn svd_polar<T: FloatScalar + SimdDispatch>(tol: f64) {
    for n in [1, 9, 40, 500] {
        let ms = general_cases(n);
        let a = cols::<T>(&ms, &FULL);
        let ar: [&[T]; 9] = core::array::from_fn(|k| a[k].as_slice());
        let mut u = vec![vec![T::ZERO; n]; 9];
        let mut s = vec![vec![T::ZERO; n]; 3];
        let mut v = vec![vec![T::ZERO; n]; 9];
        {
            let mut ui = u.iter_mut();
            let mut si = s.iter_mut();
            let mut vi = v.iter_mut();
            svd3(
                ar,
                core::array::from_fn(|_| ui.next().unwrap().as_mut_slice()),
                core::array::from_fn(|_| si.next().unwrap().as_mut_slice()),
                core::array::from_fn(|_| vi.next().unwrap().as_mut_slice()),
            );
        }
        let mut rot = vec![vec![T::ZERO; n]; 9];
        let mut st = vec![vec![T::ZERO; n]; 6];
        {
            let mut ri = rot.iter_mut();
            let mut sti = st.iter_mut();
            polar3(
                ar,
                core::array::from_fn(|_| ri.next().unwrap().as_mut_slice()),
                core::array::from_fn(|_| sti.next().unwrap().as_mut_slice()),
            );
        }
        for (k, m) in ms.iter().enumerate() {
            let (uk, vk) = (mat(&u, k), mat(&v, k));
            let sg: [f64; 3] = core::array::from_fn(|i| s[i][k].into_f64());
            check_orthonormal(&uk, tol * 10.0, "U");
            check_orthonormal(&vk, tol * 10.0, "V");
            assert!(sg[0] >= sg[1] - tol * sg[0] && sg[1] >= sg[2].abs() - tol * sg[0], "order {sg:?}");
            let d = det(m);
            if d.abs() > 1e-6 * norm(m).powi(3) {
                assert_eq!(sg[2] < 0.0, d < 0.0, "sign of sigma2 {sg:?} vs det {d}");
            }
            let sm = [[sg[0], 0.0, 0.0], [0.0, sg[1], 0.0], [0.0, 0.0, sg[2]]];
            let back = mul(&mul(&uk, &sm), &t(&vk));
            assert!(diff(&back, m) <= tol * norm(m).max(1e-30), "svd case {k}: {m:?} vs {back:?}");

            let r = mat(&rot, k);
            check_orthonormal(&r, tol * 10.0, "R");
            let sv: [f64; 6] = core::array::from_fn(|e| st[e][k].into_f64());
            let sym = [[sv[0], sv[1], sv[2]], [sv[1], sv[3], sv[4]], [sv[2], sv[4], sv[5]]];
            let back = mul(&r, &sym);
            assert!(diff(&back, m) <= tol * norm(m).max(1e-30), "polar case {k}: {m:?} vs {back:?}");
        }
    }
}

#[test]
fn svd3_and_polar3_f64() {
    svd_polar::<f64>(1e-11);
}

#[test]
fn svd3_and_polar3_f32() {
    svd_polar::<f32>(5e-6);
}

/// Random `N×N` systems: SPD for Cholesky, symmetric indefinite (but strongly diagonal) for LDLᵀ.
fn solve<T: FloatScalar + SimdDispatch, const N: usize>(tol: f64, spd: bool) {
    let mut r = Lcg(N as u64 * 31 + spd as u64);
    for m in [1, 13, 300] {
        let mats: Vec<Vec<Vec<f64>>> = (0..m)
            .map(|_| {
                let g: Vec<Vec<f64>> = (0..N).map(|_| (0..N).map(|_| r.next()).collect()).collect();
                let mut a = vec![vec![0.0; N]; N];
                for i in 0..N {
                    for j in 0..N {
                        a[i][j] = if spd {
                            (0..N).map(|k| g[k][i] * g[k][j]).sum::<f64>() + if i == j { 0.5 } else { 0.0 }
                        } else {
                            0.5 * (g[i][j] + g[j][i]) + if i == j { (N as f64 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 } } else { 0.0 }
                        };
                    }
                }
                a
            })
            .collect();
        let rhs: Vec<Vec<f64>> = (0..N).map(|_| (0..m).map(|_| r.next()).collect()).collect();
        let mut packed: Vec<Vec<T>> = Vec::new();
        for i in 0..N {
            for j in 0..=i {
                packed.push(mats.iter().map(|a| T::from_f64(a[i][j])).collect());
            }
        }
        let b: Vec<Vec<T>> = rhs.iter().map(|c| c.iter().map(|&x| T::from_f64(x)).collect()).collect();
        let mut x = vec![vec![T::ZERO; m]; N];
        {
            let ar: Vec<&[T]> = packed.iter().map(|c| c.as_slice()).collect();
            let br: Vec<&[T]> = b.iter().map(|c| c.as_slice()).collect();
            let mut xr: Vec<&mut [T]> = x.iter_mut().map(|c| c.as_mut_slice()).collect();
            if spd {
                cholesky_solve::<T, N>(&ar, &br, &mut xr);
            } else {
                ldlt_solve::<T, N>(&ar, &br, &mut xr);
            }
        }
        for (k, a) in mats.iter().enumerate() {
            for i in 0..N {
                let ax: f64 = (0..N).map(|j| a[i][j] * x[j][k].into_f64()).sum();
                let scale = 1.0 + rhs[i][k].abs();
                assert!((ax - rhs[i][k]).abs() <= tol * scale, "N={N} m={m} k={k} row {i}: {ax} vs {}", rhs[i][k]);
            }
        }
    }
}

#[test]
fn cholesky_solve_matches() {
    solve::<f64, 3>(1e-10, true);
    solve::<f64, 6>(1e-9, true);
    solve::<f32, 6>(2e-3, true);
    solve::<f64, 8>(1e-9, true);
}

#[test]
fn ldlt_solve_matches() {
    solve::<f64, 3>(1e-10, false);
    solve::<f64, 6>(1e-10, false);
    solve::<f32, 6>(1e-4, false);
}

#[test]
fn not_positive_definite_gives_nan() {
    let a: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![1.0]];
    let b = [vec![1.0], vec![1.0]];
    let mut x = [vec![0.0], vec![0.0]];
    let ar: Vec<&[f64]> = a.iter().map(|c| c.as_slice()).collect();
    let br: Vec<&[f64]> = b.iter().map(|c| c.as_slice()).collect();
    let [x0, x1] = &mut x;
    cholesky_solve::<f64, 2>(&ar, &br, &mut [x0.as_mut_slice(), x1.as_mut_slice()]);
    assert!(x[0][0].is_nan() && x[1][0].is_nan());
}
