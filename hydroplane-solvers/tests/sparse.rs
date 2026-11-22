use hydroplane_solvers::sparse::SellMatrix;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 31) as f64 - 1.0
    }
}

/// A random pattern with rows from empty to dense-ish, as dense `n×n` plus its CSC.
fn random(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut r = Lcg(seed);
    let mut dense = vec![vec![0.0; n]; n];
    for (i, row) in dense.iter_mut().enumerate() {
        let len = match i % 5 {
            0 => 0,
            1 => 1,
            2 => (r.next() % 40) as usize,
            _ => (r.next() % 8) as usize,
        };
        for _ in 0..len {
            row[(r.next() as usize) % n] = r.unit();
        }
    }
    let (mut col_ptr, mut row_idx, mut vals) = (vec![0], Vec::new(), Vec::new());
    for j in 0..n {
        for (i, row) in dense.iter().enumerate() {
            if row[j] != 0.0 {
                row_idx.push(i);
                vals.push(row[j]);
            }
        }
        col_ptr.push(row_idx.len());
    }
    (dense, col_ptr, row_idx, vals)
}

fn dense_mv(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| row.iter().zip(x).map(|(a, x)| a * x).sum()).collect()
}

fn close(got: &[f64], want: &[f64]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-12 * (1.0 + w.abs()), "row {i}: {g} vs {w}");
    }
}

#[test]
fn general_matches_dense() {
    for n in [1, 7, 8, 9, 255, 256, 257, 1000] {
        let (dense, col_ptr, row_idx, vals) = random(n, n as u64);
        let x: Vec<f64> = (0..n).map(|i| (i % 13) as f64 - 6.0).collect();
        let mut a = SellMatrix::from_csc(n, &col_ptr, &row_idx);
        a.set_values(&vals);
        let mut y = vec![f64::NAN; n];
        a.spmv(&x, &mut y);
        close(&y, &dense_mv(&dense, &x));

        // New values on the same pattern repack without rebuilding.
        let doubled: Vec<f64> = vals.iter().map(|v| 2.0 * v).collect();
        a.set_values(&doubled);
        a.spmv(&x, &mut y);
        let want: Vec<f64> = dense_mv(&dense, &x).iter().map(|v| 2.0 * v).collect();
        close(&y, &want);
    }
}

#[test]
fn csr_matches_dense() {
    for n in [3, 64, 300] {
        let (dense, _, _, _) = random(n, 99 + n as u64);
        let (mut row_ptr, mut cols, mut vals) = (vec![0], Vec::new(), Vec::new());
        for row in &dense {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    cols.push(j);
                    vals.push(v);
                }
            }
            row_ptr.push(cols.len());
        }
        let x: Vec<f64> = (0..n).map(|i| 1.0 / (1.0 + i as f64)).collect();
        let mut a = SellMatrix::from_csr(n, &row_ptr, &cols);
        a.set_values(&vals);
        let mut y = vec![0.0; n];
        a.spmv(&x, &mut y);
        close(&y, &dense_mv(&dense, &x));
    }
}

#[test]
fn symmetric_lower_matches_dense() {
    for n in [1, 9, 300, 777] {
        let (mut dense, _, _, _) = random(n, 7 * n as u64);
        let lower = dense.clone();
        for (i, row) in lower.iter().enumerate() {
            for (j, &v) in row.iter().enumerate().take(i) {
                dense[j][i] = v;
            }
        }
        let (mut col_ptr, mut row_idx, mut vals) = (vec![0], Vec::new(), Vec::new());
        for j in 0..n {
            for (i, row) in dense.iter().enumerate().skip(j) {
                if row[j] != 0.0 {
                    row_idx.push(i);
                    vals.push(row[j]);
                }
            }
            col_ptr.push(row_idx.len());
        }
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let mut a = SellMatrix::from_lower_csc_symmetric(n, &col_ptr, &row_idx);
        a.set_values(&vals);
        let mut y = vec![0.0; n];
        a.spmv(&x, &mut y);
        let want = dense_mv(&dense, &x);
        close(&y, &want);
        let b = want.clone();
        assert!(a.relative_residual(&b, &x) < 1e-14);
    }
}
