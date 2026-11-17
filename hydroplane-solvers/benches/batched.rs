//! Batched 3×3 symmetric eigen, signed SVD and 6×6 Cholesky solves through `hydroplane_solvers::batched`,
//! against the same algorithms run one matrix at a time in scalar code.
//!
//! `cargo bench -p hydroplane-solvers --bench batched -- [matrices]`

use std::hint::black_box;
use std::time::Instant;

use hydroplane_solvers::batched::{cholesky_solve, svd3, sym3_eigen};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

/// One matrix's cyclic Jacobi, the scalar reference for the batched kernel.
fn scalar_eigen(a: [f32; 6]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut m = [[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0f32]];
    for _ in 0..4 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let r = 3 - p - q;
            let apq = m[p][q];
            if apq == 0.0 {
                continue;
            }
            let theta = (m[q][q] - m[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            m[p][p] -= t * apq;
            m[q][q] += t * apq;
            m[p][q] = 0.0;
            m[q][p] = 0.0;
            let (arp, arq) = (m[r][p], m[r][q]);
            m[r][p] = c * arp - s * arq;
            m[p][r] = m[r][p];
            m[r][q] = s * arp + c * arq;
            m[q][r] = m[r][q];
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    ([m[0][0], m[1][1], m[2][2]], v)
}

fn per_matrix(label: &str, n: usize, ns: u128) {
    println!("{label:<34} {:>7.2} ns/matrix", ns as f64 / n as f64);
}

fn main() {
    let n: usize = std::env::args().skip(1).find_map(|s| s.parse().ok()).unwrap_or(100_000);
    let mut r = Lcg(3);
    let sym: Vec<Vec<f32>> = (0..6).map(|_| (0..n).map(|_| r.next()).collect()).collect();
    let full: Vec<Vec<f32>> = (0..9).map(|_| (0..n).map(|_| r.next()).collect()).collect();
    let reps = 20;

    let mut vals = vec![vec![0.0f32; n]; 3];
    let mut vecs = vec![vec![0.0f32; n]; 9];
    let t = Instant::now();
    for _ in 0..reps {
        let mut vi = vals.iter_mut();
        let mut ve = vecs.iter_mut();
        sym3_eigen(
            core::array::from_fn(|k| sym[k].as_slice()),
            core::array::from_fn(|_| vi.next().unwrap().as_mut_slice()),
            core::array::from_fn(|_| ve.next().unwrap().as_mut_slice()),
        );
    }
    per_matrix("sym3_eigen f32 (batched)", n * reps, t.elapsed().as_nanos());
    let rows: Vec<[f32; 6]> = (0..n).map(|k| core::array::from_fn(|e| sym[e][k])).collect();
    let t = Instant::now();
    for _ in 0..reps {
        for &m in &rows {
            black_box(scalar_eigen(m));
        }
    }
    per_matrix("sym3_eigen f32 (scalar loop)", n * reps, t.elapsed().as_nanos());

    let mut u = vec![vec![0.0f32; n]; 9];
    let mut s = vec![vec![0.0f32; n]; 3];
    let mut v = vec![vec![0.0f32; n]; 9];
    let t = Instant::now();
    for _ in 0..reps {
        let (mut ui, mut si, mut vi) = (u.iter_mut(), s.iter_mut(), v.iter_mut());
        svd3(
            core::array::from_fn(|k| full[k].as_slice()),
            core::array::from_fn(|_| ui.next().unwrap().as_mut_slice()),
            core::array::from_fn(|_| si.next().unwrap().as_mut_slice()),
            core::array::from_fn(|_| vi.next().unwrap().as_mut_slice()),
        );
    }
    per_matrix("svd3 f32 (batched)", n * reps, t.elapsed().as_nanos());

    let packed: Vec<Vec<f64>> = (0..21)
        .map(|e| (0..n).map(|_| if [0, 2, 5, 9, 14, 20].contains(&e) { 8.0 } else { r.next() as f64 * 0.5 }).collect())
        .collect();
    let rhs: Vec<Vec<f64>> = (0..6).map(|_| (0..n).map(|_| r.next() as f64).collect()).collect();
    let mut x = vec![vec![0.0f64; n]; 6];
    let ar: Vec<&[f64]> = packed.iter().map(|c| c.as_slice()).collect();
    let br: Vec<&[f64]> = rhs.iter().map(|c| c.as_slice()).collect();
    let t = Instant::now();
    for _ in 0..reps {
        let mut xr: Vec<&mut [f64]> = x.iter_mut().map(|c| c.as_mut_slice()).collect();
        cholesky_solve::<f64, 6>(&ar, &br, &mut xr);
    }
    per_matrix("cholesky_solve 6x6 f64 (batched)", n * reps, t.elapsed().as_nanos());
    black_box((&vals, &vecs, &u, &s, &v, &x));
}
