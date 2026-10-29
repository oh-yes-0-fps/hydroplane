//! Sparse matrix-vector products in SELL-C-σ layout (Kreutzer et al. 2014): rows sorted by length
//! within windows of σ rows, grouped into chunks of `C` rows, and each chunk stored column by
//! column and padded to its longest row. A chunk's `j`-th entries are then `C` contiguous values
//! and column indices, so the product is full-register value loads and one gather of `x` per
//! register, where compressed rows need a scalar loop per row.

use hydroplane::{Gang, kernel};
use rayon::prelude::*;

/// Rows per chunk: a whole number of registers on every backend (2, 4 or 8 `f64` lanes).
const C: usize = 8;
/// Rows per sorting window, and per parallel task.
const SIGMA: usize = 256;
/// Marks a padding slot in [`SellMatrix::src`].
const PAD: u32 = u32::MAX;

/// A square sparse matrix stored for fast `y = A x`. Built once from a sparsity pattern; values
/// are packed in by [`set_values`](Self::set_values) in the pattern's own storage order, so
/// refactorizing with new values doesn't rebuild the layout.
pub struct SellMatrix {
    n: usize,
    /// Row of each sorted position, window by window.
    perm: Vec<u32>,
    /// First slot of each chunk; one extra entry closes the last.
    chunk_ptr: Vec<usize>,
    cols: Vec<u32>,
    /// Index into the caller's values for each slot, [`PAD`] for padding.
    src: Vec<u32>,
    vals: Vec<f64>,
}

impl SellMatrix {
    /// From rows given as `(column, value index)` lists: the shared core of the constructors.
    fn from_rows(n: usize, rows: &[Vec<(u32, u32)>]) -> Self {
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for w in perm.chunks_mut(SIGMA) {
            w.sort_by_key(|&r| std::cmp::Reverse(rows[r as usize].len()));
        }
        let chunks = n.div_ceil(C);
        let mut chunk_ptr = Vec::with_capacity(chunks + 1);
        let (mut cols, mut src) = (Vec::new(), Vec::new());
        chunk_ptr.push(0);
        for ch in perm.chunks(C) {
            let width = ch.iter().map(|&r| rows[r as usize].len()).max().unwrap_or(0);
            for j in 0..width {
                for lane in 0..C {
                    let row = ch.get(lane).map(|&r| &rows[r as usize]);
                    match row.and_then(|row| row.get(j)) {
                        Some(&(c, s)) => {
                            cols.push(c);
                            src.push(s);
                        }
                        None => {
                            // Padding multiplies 0 by an entry of `x` the row already reads.
                            let c = row.and_then(|row| row.last()).map_or(0, |&(c, _)| c);
                            cols.push(c);
                            src.push(PAD);
                        }
                    }
                }
            }
            chunk_ptr.push(cols.len());
        }
        let vals = vec![0.0; cols.len()];
        Self { n, perm, chunk_ptr, cols, src, vals }
    }

    /// From compressed sparse rows: row `i`'s entries are `col_idx[row_ptr[i]..row_ptr[i + 1]]`,
    /// and [`set_values`](Self::set_values) takes values in that same order.
    pub fn from_csr(n: usize, row_ptr: &[usize], col_idx: &[usize]) -> Self {
        let rows: Vec<Vec<(u32, u32)>> =
            (0..n).map(|i| (row_ptr[i]..row_ptr[i + 1]).map(|k| (col_idx[k] as u32, k as u32)).collect()).collect();
        Self::from_rows(n, &rows)
    }

    /// From compressed sparse columns (`row_idx[col_ptr[j]..col_ptr[j + 1]]` are column `j`'s
    /// rows); [`set_values`](Self::set_values) takes values in CSC order.
    pub fn from_csc(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Self {
        let mut rows = vec![Vec::new(); n];
        for j in 0..n {
            for (k, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
                rows[i].push((j as u32, k as u32));
            }
        }
        Self::from_rows(n, &rows)
    }

    /// A symmetric matrix from its lower triangle in compressed sparse columns (entries above the
    /// diagonal are ignored); [`set_values`](Self::set_values) takes the lower triangle's values
    /// in CSC order, and each off-diagonal one is used for both `A[i][j]` and `A[j][i]`.
    pub fn from_lower_csc_symmetric(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Self {
        let mut rows = vec![Vec::new(); n];
        for j in 0..n {
            for (k, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
                if i < j {
                    continue;
                }
                rows[i].push((j as u32, k as u32));
                if i != j {
                    rows[j].push((i as u32, k as u32));
                }
            }
        }
        Self::from_rows(n, &rows)
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Packs `values` (in the constructor's storage order) into the layout.
    pub fn set_values(&mut self, values: &[f64]) {
        let src = &self.src;
        self.vals.par_iter_mut().zip(src.par_iter()).with_min_len(4096).for_each(|(v, &s)| {
            *v = if s == PAD { 0.0 } else { values[s as usize] };
        });
    }

    /// `y = A x`.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n, "SellMatrix::spmv: x has the wrong length");
        assert_eq!(y.len(), self.n, "SellMatrix::spmv: y has the wrong length");
        let chunks_per_window = SIGMA / C;
        y.par_chunks_mut(SIGMA).enumerate().for_each(|(w, yw)| {
            let first = w * chunks_per_window;
            let last = (first + chunks_per_window).min(self.chunk_ptr.len() - 1);
            let rows = &self.perm[w * SIGMA..w * SIGMA + yw.len()];
            sell_window(&self.chunk_ptr[first..=last], &self.cols, &self.vals, x, rows, w * SIGMA, yw);
        });
    }

    /// `‖b − A x‖ / ‖b‖` (`‖b − A x‖` when `b = 0`).
    #[must_use]
    pub fn relative_residual(&self, b: &[f64], x: &[f64]) -> f64 {
        let mut ax = vec![0.0; b.len()];
        self.spmv(x, &mut ax);
        let rr: f64 = b.par_iter().zip(&ax).map(|(b, ax)| (b - ax) * (b - ax)).sum();
        let bb: f64 = b.par_iter().map(|b| b * b).sum();
        if bb > 0.0 { (rr / bb).sqrt() } else { rr.sqrt() }
    }
}

/// One window's chunks: each `C`-row chunk keeps a register of sums per `lanes` rows, two sets
/// alternating over its columns so every chunk runs several independent FMA chains, then scatters
/// the sums back to the rows' unsorted positions within `y`.
#[kernel]
fn sell_window<'a, 'b>(
    ctx: Gang,
    chunk_ptr: &'a [usize],
    cols: &'a [u32],
    vals: &'a [f64],
    x: &'a [f64],
    rows: &'a [u32],
    row_base: usize,
    y: &'b mut [f64],
) {
    let lanes = ctx.lanes::<f64>();
    let groups = C / lanes;
    let zero = ctx.splat(0.0f64);
    let mut sums = [0.0f64; C];
    for (ci, ptr) in chunk_ptr.windows(2).enumerate() {
        let (start, end) = (ptr[0], ptr[1]);
        let mut even = [zero; C];
        let mut odd = [zero; C];
        let mut k = start;
        while k + 2 * C <= end {
            for g in 0..groups {
                let (a, b) = (k + g * lanes, k + C + g * lanes);
                let xa = ctx.gather(x, ctx.load_u32::<f64>(&cols[a..a + lanes]));
                let xb = ctx.gather(x, ctx.load_u32::<f64>(&cols[b..b + lanes]));
                even[g] = ctx.load(&vals[a..a + lanes]).fma(xa, even[g]);
                odd[g] = ctx.load(&vals[b..b + lanes]).fma(xb, odd[g]);
            }
            k += 2 * C;
        }
        if k < end {
            for (g, acc) in even.iter_mut().take(groups).enumerate() {
                let a = k + g * lanes;
                let xa = ctx.gather(x, ctx.load_u32::<f64>(&cols[a..a + lanes]));
                *acc = ctx.load(&vals[a..a + lanes]).fma(xa, *acc);
            }
        }
        for (g, (e, o)) in even.iter().zip(&odd).take(groups).enumerate() {
            (*e + *o).store(&mut sums[g * lanes..(g + 1) * lanes]);
        }
        for (lane, s) in sums.iter().enumerate() {
            if let Some(&r) = rows.get(ci * C + lane) {
                y[r as usize - row_base] = *s;
            }
        }
    }
}
