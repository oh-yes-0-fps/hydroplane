//! `y = A x` on a 27-point 3D stencil (FEM-like rows of up to 27 entries): `SellMatrix` against
//! compressed rows read through a value index, the layout the solvers used before.
//!
//! `cargo bench -p hydroplane-solvers --bench spmv -- [grid edge]`

use std::time::Instant;

use hydroplane_solvers::sparse::SellMatrix;
use rayon::prelude::*;

fn stencil(m: usize) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = m * m * m;
    let id = |x: usize, y: usize, z: usize| (z * m + y) * m + x;
    let (mut row_ptr, mut cols, mut vals) = (vec![0], Vec::new(), Vec::new());
    for z in 0..m {
        for y in 0..m {
            for x in 0..m {
                for dz in -1i64..=1 {
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let (xx, yy, zz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                            if (0..m as i64).contains(&xx) && (0..m as i64).contains(&yy) && (0..m as i64).contains(&zz) {
                                cols.push(id(xx as usize, yy as usize, zz as usize));
                                vals.push(if (dx, dy, dz) == (0, 0, 0) { 26.0 } else { -1.0 });
                            }
                        }
                    }
                }
                row_ptr.push(cols.len());
            }
        }
    }
    (n, row_ptr, cols, vals)
}

fn main() {
    let m: usize = std::env::args().skip(1).find_map(|s| s.parse().ok()).unwrap_or(64);
    let (n, row_ptr, cols, vals) = stencil(m);
    let x: Vec<f64> = (0..n).map(|i| (i % 17) as f64 * 0.25 - 2.0).collect();
    let reps = 50;

    // The previous scheme: rows, a column index and a value index per entry.
    let col32: Vec<u32> = cols.iter().map(|&c| c as u32).collect();
    let vidx: Vec<u32> = (0..cols.len() as u32).collect();
    let mut y0 = vec![0.0; n];
    // The first dispatch resolves hydroplane's unroll factor; keep that out of the timings.
    let mut warm = SellMatrix::from_csr(n, &row_ptr, &cols);
    warm.set_values(&vals);
    warm.spmv(&x, &mut y0);
    let t = Instant::now();
    for _ in 0..reps {
        y0.par_iter_mut().enumerate().with_min_len(512).for_each(|(i, yi)| {
            *yi = (row_ptr[i]..row_ptr[i + 1]).map(|k| vals[vidx[k] as usize] * x[col32[k] as usize]).sum();
        });
    }
    let csr = t.elapsed().as_secs_f64() / reps as f64;

    let mut sell = SellMatrix::from_csr(n, &row_ptr, &cols);
    sell.set_values(&vals);
    let mut y1 = vec![0.0; n];
    let t = Instant::now();
    for _ in 0..reps {
        sell.spmv(&x, &mut y1);
    }
    let fast = t.elapsed().as_secs_f64() / reps as f64;

    let err = y0.iter().zip(&y1).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let gbs = |s: f64| (cols.len() * 12 + n * 16) as f64 / s / 1e9;
    println!("{m}^3 grid: {n} rows, {} nonzeros, max |diff| {err:e}", cols.len());
    println!("general, compressed rows     {:>8.3} ms  ({:.1} GB/s)", csr * 1e3, gbs(csr));
    println!("general, SELL-C-sigma        {:>8.3} ms  ({:.1} GB/s)  {:.2}x", fast * 1e3, gbs(fast), csr / fast);

    // Symmetric, as the solvers hold it: the lower triangle by columns, expanded to both
    // triangles by rows with each upper entry reading its mirrored lower value.
    let mut col_ptr = vec![0usize];
    let (mut row_idx, mut lower) = (Vec::new(), Vec::new());
    let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for k in row_ptr[i]..row_ptr[i + 1] {
            if cols[k] <= i {
                by_col[cols[k]].push((i, vals[k]));
            }
        }
    }
    for c in &by_col {
        for &(i, v) in c {
            row_idx.push(i);
            lower.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    let (mut srow_ptr, mut scol, mut svidx) = (vec![0usize; n + 1], Vec::new(), Vec::new());
    let mut rows: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    for j in 0..n {
        for (k, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
            rows[i].push((j as u32, k as u32));
            if i != j {
                rows[j].push((i as u32, k as u32));
            }
        }
    }
    for (i, r) in rows.iter().enumerate() {
        for &(c, v) in r {
            scol.push(c);
            svidx.push(v);
        }
        srow_ptr[i + 1] = scol.len();
    }
    let t = Instant::now();
    for _ in 0..reps {
        y0.par_iter_mut().enumerate().with_min_len(512).for_each(|(i, yi)| {
            *yi = (srow_ptr[i]..srow_ptr[i + 1]).map(|k| lower[svidx[k] as usize] * x[scol[k] as usize]).sum();
        });
    }
    let scsr = t.elapsed().as_secs_f64() / reps as f64;
    let mut ssell = SellMatrix::from_lower_csc_symmetric(n, &col_ptr, &row_idx);
    ssell.set_values(&lower);
    let t = Instant::now();
    for _ in 0..reps {
        ssell.spmv(&x, &mut y1);
    }
    let sfast = t.elapsed().as_secs_f64() / reps as f64;
    let err = y0.iter().zip(&y1).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("symmetric, compressed rows   {:>8.3} ms  ({:.1} GB/s)  max |diff| {err:e}", scsr * 1e3, gbs(scsr));
    println!("symmetric, SELL-C-sigma      {:>8.3} ms  ({:.1} GB/s)  {:.2}x", sfast * 1e3, gbs(sfast), scsr / sfast);
}
