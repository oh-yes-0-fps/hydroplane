use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

use rayon::prelude::*;

use super::{Preconditioner, estimate_spectrum};
use crate::cholesky::{Analysis, CpuCholesky, Error, Ordering};
use crate::dense::DenseCholesky;

/// Settings of an [`Amg`].
#[derive(Clone, Debug, PartialEq)]
pub struct AmgOptions {
    /// Couplings `|aᵢⱼ| ≥ strength · √(|aᵢᵢ aⱼⱼ|)` (block norms for `block > 1`) are strong; only
    /// those join aggregates. 0 (the default) aggregates over every coupling, as PyAMG's smoothed
    /// aggregation does; a positive value follows strong directions of an anisotropic problem,
    /// but on a smoothed coarse operator, whose couplings are many and weak, it stops coarsening.
    pub strength: f64,
    /// Rows from which a level is not coarsened further but factored.
    pub max_coarse: usize,
    pub max_levels: usize,
    /// Degree of the Chebyshev smoother before and after each coarse correction.
    pub smoothing_degree: usize,
    /// Dofs per node, aggregated together (3 for a 3D solid numbered `3·node + component`).
    pub block: usize,
    /// Vectors the operator nearly annihilates, which the coarse levels must represent: the six
    /// rigid-body modes of an elastic solid. Empty (the default): each component held constant.
    pub near_null_space: Vec<Vec<f64>>,
}

impl Default for AmgOptions {
    fn default() -> Self {
        Self { strength: 0.0, max_coarse: 1000, max_levels: 12, smoothing_degree: 2, block: 1, near_null_space: Vec::new() }
    }
}

/// A sparse matrix by rows.
#[derive(Clone, Debug)]
pub(crate) struct Mat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ptr: Vec<usize>,
    pub(crate) col: Vec<u32>,
    pub(crate) val: Vec<f64>,
}

impl Mat {
    /// Both triangles of the symmetric matrix whose lower triangle is given by columns: read by
    /// rows, those columns are the upper triangle, and their transpose the lower.
    fn symmetric(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Self {
        let n = col_ptr.len() - 1;
        let upper = Self { rows: n, cols: n, ptr: col_ptr.to_vec(), col: row_idx[..col_ptr[n]].par_iter().map(|&i| i as u32).collect(), val: values[..col_ptr[n]].to_vec() };
        let lower = upper.transpose();
        Self::build(n, n, || (), |i, _, row| {
            row.extend((lower.ptr[i]..lower.ptr[i + 1]).map(|k| (lower.col[k], lower.val[k])));
            row.extend((upper.ptr[i]..upper.ptr[i + 1]).filter(|&k| upper.col[k] as usize != i).map(|k| (upper.col[k], upper.val[k])));
            if !row.is_sorted_by_key(|e| e.0) {
                row.sort_unstable_by_key(|e| e.0);
            }
        })
    }

    /// A matrix built row by row in parallel: `fill(i, scratch, row)` puts row `i`'s entries,
    /// sorted, into the empty `row`. Rows go to a few tasks per thread in contiguous ranges, each
    /// task with its own `scratch()` (a dense accumulator) and output, and the outputs are copied
    /// into place in parallel.
    fn build<S>(rows: usize, cols: usize, scratch: impl Fn() -> S + Sync, fill: impl Fn(usize, &mut S, &mut Vec<(u32, f64)>) + Sync) -> Self {
        let tasks = (rayon::current_num_threads() * 4).clamp(1, rows.max(1));
        let per = rows.div_ceil(tasks).max(1);
        let parts: Vec<(Vec<usize>, Vec<u32>, Vec<f64>)> = (0..tasks)
            .into_par_iter()
            .map(|t| {
                let mut state = scratch();
                let (mut row, mut lens, mut cs, mut vs) = (Vec::new(), Vec::with_capacity(per), Vec::new(), Vec::new());
                for i in (t * per).min(rows)..((t + 1) * per).min(rows) {
                    row.clear();
                    fill(i, &mut state, &mut row);
                    lens.push(row.len());
                    cs.extend(row.iter().map(|e| e.0));
                    vs.extend(row.iter().map(|e| e.1));
                }
                (lens, cs, vs)
            })
            .collect();
        let mut ptr = Vec::with_capacity(rows + 1);
        ptr.push(0);
        for (lens, _, _) in &parts {
            for &l in lens {
                ptr.push(ptr[ptr.len() - 1] + l);
            }
        }
        let nnz = ptr[rows];
        let (mut col, mut val) = (vec![0u32; nnz], vec![0.0; nnz]);
        let mut slots = Vec::with_capacity(parts.len());
        let (mut c, mut v) = (&mut col[..], &mut val[..]);
        for (_, cs, vs) in &parts {
            let (ch, ct) = c.split_at_mut(cs.len());
            let (vh, vt) = v.split_at_mut(vs.len());
            slots.push((ch, vh, cs, vs));
            (c, v) = (ct, vt);
        }
        slots.into_par_iter().for_each(|(ch, vh, cs, vs)| {
            ch.copy_from_slice(cs);
            vh.copy_from_slice(vs);
        });
        Self { rows, cols, ptr, col, val }
    }

    fn nnz(&self) -> usize {
        self.ptr[self.rows]
    }

    /// `y = A x`.
    fn multiply(&self, x: &[f64], y: &mut [f64]) {
        let (ptr, col, val) = (&self.ptr, &self.col, &self.val);
        y.par_iter_mut().enumerate().with_min_len(1024).for_each(|(i, y)| {
            *y = (ptr[i]..ptr[i + 1]).map(|k| val[k] * x[col[k] as usize]).sum();
        });
    }

    fn diagonal(&self) -> Vec<f64> {
        (0..self.rows).map(|i| (self.ptr[i]..self.ptr[i + 1]).find(|&k| self.col[k] as usize == i).map_or(0.0, |k| self.val[k])).collect()
    }

    /// The transpose, in parallel: entries counted per column and placed through atomic
    /// cursors, then each row of the result put in column order (its entries arrive in
    /// whatever order the threads reached them).
    fn transpose(&self) -> Self {
        let counts: Vec<AtomicUsize> = (0..=self.cols).into_par_iter().map(|_| AtomicUsize::new(0)).collect();
        self.col.par_iter().with_min_len(1 << 14).for_each(|&c| {
            counts[c as usize + 1].fetch_add(1, Relaxed);
        });
        let mut ptr: Vec<usize> = counts.into_iter().map(AtomicUsize::into_inner).collect();
        for c in 0..self.cols {
            ptr[c + 1] += ptr[c];
        }
        let next: Vec<AtomicUsize> = ptr[..self.cols].par_iter().map(|&p| AtomicUsize::new(p)).collect();
        let mut col = vec![0u32; self.nnz()];
        let mut val = vec![0.0; self.nnz()];
        let (cp, vp) = (Scatter(col.as_mut_ptr()), Scatter(val.as_mut_ptr()));
        (0..self.rows).into_par_iter().with_min_len(1024).for_each(|i| {
            for k in self.ptr[i]..self.ptr[i + 1] {
                let at = next[self.col[k] as usize].fetch_add(1, Relaxed);
                // Each cursor hands out every slot once.
                unsafe {
                    cp.write(at, i as u32);
                    vp.write(at, self.val[k]);
                }
            }
        });
        (0..self.cols).into_par_iter().with_min_len(1024).for_each(|r| {
            let (start, len) = (ptr[r], ptr[r + 1] - ptr[r]);
            // Rows are disjoint ranges.
            let (c, v) = unsafe { (cp.slice(start, len), vp.slice(start, len)) };
            if c.is_sorted() {
                return;
            }
            let mut pairs: Vec<(u32, f64)> = c.iter().copied().zip(v.iter().copied()).collect();
            pairs.sort_unstable_by_key(|e| e.0);
            for (k, (cc, vv)) in pairs.into_iter().enumerate() {
                c[k] = cc;
                v[k] = vv;
            }
        });
        Self { rows: self.cols, cols: self.rows, ptr, col, val }
    }

    /// `self · other`, rows in parallel (Gustavson, a dense accumulator per worker).
    fn product(&self, other: &Self) -> Self {
        self.product_plus(other, None)
    }

    /// `diag(scale) · self · other + plus`, `plus` of the product's shape: with `scale = −ω D⁻¹`
    /// and `self = A`, `other = plus = P`, the smoothed prolongator `(I − ω D⁻¹ A) P`.
    fn product_plus(&self, other: &Self, plus: Option<(&[f64], &Self)>) -> Self {
        Self::build(self.rows, other.cols, || vec![u32::MAX; other.cols], |i, marker, row| {
            let mut add = |col: u32, v: f64, row: &mut Vec<(u32, f64)>| {
                let m = &mut marker[col as usize];
                if *m == u32::MAX {
                    *m = row.len() as u32;
                    row.push((col, v));
                } else {
                    row[*m as usize].1 += v;
                }
            };
            let scale = plus.map_or(1.0, |(s, _)| s[i]);
            for k in self.ptr[i]..self.ptr[i + 1] {
                let (j, a) = (self.col[k] as usize, scale * self.val[k]);
                for q in other.ptr[j]..other.ptr[j + 1] {
                    add(other.col[q], a * other.val[q], row);
                }
            }
            if let Some((_, p)) = plus {
                for q in p.ptr[i]..p.ptr[i + 1] {
                    add(p.col[q], p.val[q], row);
                }
            }
            for &(col, _) in row.iter() {
                marker[col as usize] = u32::MAX;
            }
            row.sort_unstable_by_key(|e| e.0);
        })
    }

    /// The lower triangle by columns, for the coarsest level's Cholesky.
    fn lower_columns(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let t = self.transpose();
        let mut col_ptr = vec![0];
        let (mut row_idx, mut values) = (Vec::new(), Vec::new());
        for j in 0..t.rows {
            for k in t.ptr[j]..t.ptr[j + 1] {
                if t.col[k] as usize >= j {
                    row_idx.push(t.col[k] as usize);
                    values.push(t.val[k]);
                }
            }
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx, values)
    }
}

/// Aggregates of the nodes (`block` consecutive rows each) of `a` over its strong couplings:
/// `aggregate[node]` and the count.
fn aggregate(a: &Mat, block: usize, strength: f64) -> (Vec<usize>, usize) {
    let nodes = a.rows / block;
    // Squared block norms of node `i`'s couplings, by node, as a row of a `nodes × nodes` matrix.
    let couplings = Mat::build(nodes, nodes, || (), |i, _, row| {
        for r in i * block..(i + 1) * block {
            row.extend((a.ptr[r]..a.ptr[r + 1]).map(|k| ((a.col[k] as usize / block) as u32, a.val[k] * a.val[k])));
        }
        row.sort_unstable_by_key(|e| e.0);
        row.dedup_by(|e, kept| {
            let same = e.0 == kept.0;
            if same {
                kept.1 += e.1;
            }
            same
        });
    });
    let diag: Vec<f64> = (0..nodes).into_par_iter().map(|i| (couplings.ptr[i]..couplings.ptr[i + 1]).find(|&k| couplings.col[k] as usize == i).map_or(0.0, |k| couplings.val[k])).collect();
    let strong = Mat::build(nodes, nodes, || (), |i, _, row| {
        row.extend((couplings.ptr[i]..couplings.ptr[i + 1]).map(|k| (couplings.col[k], couplings.val[k])).filter(|&(j, v)| j as usize != i && v >= strength * strength * (diag[i] * diag[j as usize]).sqrt()));
    });
    let strong = |i: usize| &strong.col[strong.ptr[i]..strong.ptr[i + 1]];
    let mut agg = vec![usize::MAX; nodes];
    let mut count = 0;
    for i in 0..nodes {
        if agg[i] == usize::MAX && strong(i).iter().all(|&j| agg[j as usize] == usize::MAX) {
            agg[i] = count;
            for &j in strong(i) {
                agg[j as usize] = count;
            }
            count += 1;
        }
    }
    let first = agg.clone();
    for (i, a) in agg.iter_mut().enumerate() {
        if *a == usize::MAX
            && let Some(&j) = strong(i).iter().find(|&&j| first[j as usize] != usize::MAX)
        {
            *a = first[j as usize];
        }
    }
    for i in 0..nodes {
        if agg[i] == usize::MAX {
            agg[i] = count;
            for &j in strong(i) {
                if agg[j as usize] == usize::MAX {
                    agg[j as usize] = count;
                }
            }
            count += 1;
        }
    }
    (agg, count)
}

/// One aggregate's rows and the `Q` (by columns, over those rows) and `R` of its thin QR.
struct LocalQr {
    rows: Vec<usize>,
    q: Vec<Vec<f64>>,
    r: Vec<Vec<f64>>,
}

/// The tentative prolongator of the aggregates: for each, an orthonormal basis (thin QR) of the
/// near-null space `b` (`n × k`, by columns) restricted to its rows, and the coarse near-null
/// space (`k` per aggregate, by columns) from the `R` factors.
fn tentative(n: usize, block: usize, agg: &[usize], count: usize, b: &[Vec<f64>]) -> (Mat, Vec<Vec<f64>>) {
    let k = b.len();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (node, &a) in agg.iter().enumerate() {
        members[a].push(node);
    }
    let local: Vec<LocalQr> = members
        .par_iter()
        .map(|nodes| {
            let rows: Vec<usize> = nodes.iter().flat_map(|&nd| nd * block..(nd + 1) * block).collect();
            let mut q: Vec<Vec<f64>> = b.iter().map(|v| rows.iter().map(|&r| v[r]).collect()).collect();
            let mut r = vec![vec![0.0; k]; k];
            for c in 0..k {
                for p in 0..c {
                    let d: f64 = q[c].iter().zip(&q[p]).map(|(x, y)| x * y).sum();
                    r[p][c] = d;
                    let qp = q[p].clone();
                    q[c].iter_mut().zip(&qp).for_each(|(x, y)| *x -= d * y);
                }
                let norm = q[c].iter().map(|x| x * x).sum::<f64>().sqrt();
                r[c][c] = norm;
                if norm > 1e-12 {
                    q[c].iter_mut().for_each(|x| *x /= norm);
                } else {
                    q[c].iter_mut().for_each(|x| *x = 0.0);
                }
            }
            LocalQr { rows, q, r }
        })
        .collect();
    let mut dense = vec![0.0; n * k];
    let mut coarse = vec![vec![0.0; count * k]; k];
    for (a, LocalQr { rows, q, r }) in local.into_iter().enumerate() {
        for (ri, &row) in rows.iter().enumerate() {
            for (c, qc) in q.iter().enumerate() {
                dense[row * k + c] = qc[ri];
            }
        }
        for c in 0..k {
            for p in 0..k {
                coarse[c][a * k + p] = r[p][c];
            }
        }
    }
    let mut ptr = Vec::with_capacity(n + 1);
    ptr.push(0);
    let (mut col, mut val) = (Vec::with_capacity(n * k), Vec::with_capacity(n * k));
    for row in 0..n {
        for c in 0..k {
            let v = dense[row * k + c];
            if v != 0.0 {
                col.push((agg[row / block] * k + c) as u32);
                val.push(v);
            }
        }
        ptr.push(col.len());
    }
    (Mat { rows: n, cols: count * k, ptr, col, val }, coarse)
}

pub(crate) struct Level {
    pub(crate) a: Mat,
    pub(crate) dinv: Vec<f64>,
    /// Chebyshev interval on `D⁻¹A`.
    pub(crate) low: f64,
    pub(crate) high: f64,
    pub(crate) p: Mat,
    pub(crate) r: Mat,
    x: Vec<f64>,
    b: Vec<f64>,
    res: Vec<f64>,
    d: Vec<f64>,
    t: Vec<f64>,
}

enum Coarsest {
    Sparse(Box<CpuCholesky<f64>>),
    Dense(DenseCholesky),
}

/// Smoothed-aggregation algebraic multigrid, as a preconditioner for [`super::Cg`]: one
/// symmetric V-cycle per application (Chebyshev smoothing before and after each coarse
/// correction, a sparse Cholesky on the coarsest level). For the assembled matrices of elliptic
/// problems (heat conduction, elasticity with its rigid-body modes as the near-null space) its
/// iteration counts stay nearly flat as the mesh is refined, where Jacobi's grow with it.
pub struct Amg {
    levels: Vec<Level>,
    /// The coarsest level's operator, for the GPU cycle's dense inverse.
    #[cfg(feature = "vulkan")]
    coarse: Mat,
    coarsest: Coarsest,
    coarse_rhs: Vec<f64>,
    degree: usize,
    complexity: f64,
}

impl Amg {
    /// From the lower triangle of a symmetric positive definite matrix in compressed sparse
    /// columns.
    pub fn new(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Result<Self, Error> {
        Self::with_options(col_ptr, row_idx, values, AmgOptions::default())
    }

    pub fn with_options(col_ptr: &[usize], row_idx: &[usize], values: &[f64], options: AmgOptions) -> Result<Self, Error> {
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let block = options.block.max(1);
        if !n.is_multiple_of(block) || options.near_null_space.iter().any(|v| v.len() != n) {
            return Err(Error::Pattern);
        }
        let mut a = Mat::symmetric(col_ptr, row_idx, values);
        let fine_nnz = a.nnz() as f64;
        let mut total_nnz = fine_nnz;
        let mut b: Vec<Vec<f64>> = if options.near_null_space.is_empty() {
            (0..block).map(|c| (0..n).map(|i| if i % block == c { 1.0 } else { 0.0 }).collect()).collect()
        } else {
            options.near_null_space.clone()
        };
        let mut block = block;
        let mut levels = Vec::new();
        while a.rows > options.max_coarse && levels.len() + 1 < options.max_levels {
            let n = a.rows;
            let diag = a.diagonal();
            let dinv: Vec<f64> = diag.iter().map(|&d| if d > 0.0 { 1.0 / d } else { 1.0 }).collect();
            let start: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
            let mut op = |x: &[f64], y: &mut [f64]| a.multiply(x, y);
            let mut jacobi = |r: &[f64], z: &mut [f64]| z.iter_mut().zip(r).zip(&dinv).for_each(|((z, r), d)| *z = r * d);
            let lmax = estimate_spectrum(&mut op, &mut jacobi, &start[..]).max.max(f64::MIN_POSITIVE);
            let (agg, count) = aggregate(&a, block, options.strength);
            if count * b.len() >= n * 4 / 5 {
                break;
            }
            let (p_tent, coarse_b) = tentative(n, block, &agg, count, &b);
            let omega = 4.0 / (3.0 * lmax);
            let scale: Vec<f64> = dinv.iter().map(|d| -omega * d).collect();
            let p = a.product_plus(&p_tent, Some((&scale, &p_tent)));
            let r = p.transpose();
            let coarse = r.product(&a.product(&p));
            total_nnz += coarse.nnz() as f64;
            levels.push(Level {
                x: vec![0.0; n],
                b: vec![0.0; n],
                res: vec![0.0; n],
                d: vec![0.0; n],
                t: vec![0.0; n],
                dinv,
                low: 1.1 * lmax / 30.0,
                high: 1.1 * lmax,
                p,
                r,
                a: std::mem::replace(&mut a, coarse),
            });
            block = b.len();
            b = coarse_b;
        }
        let (cp, ri, v) = a.lower_columns();
        let sparse = Analysis::with_ordering(&cp, &ri, Ordering::Amd).ok().and_then(|an| {
            let mut c = CpuCholesky::<f64>::new(&an);
            c.factorize(&v).ok().map(|_| c)
        });
        let coarsest = match sparse {
            Some(c) => Coarsest::Sparse(Box::new(c)),
            None => {
                let m = a.rows;
                let mut dense = vec![0.0; m * m];
                for i in 0..m {
                    for k in a.ptr[i]..a.ptr[i + 1] {
                        dense[i * m + a.col[k] as usize] = a.val[k];
                    }
                }
                Coarsest::Dense(DenseCholesky::shifted(m, &dense, 1e-10))
            }
        };
        Ok(Self {
            coarse_rhs: vec![0.0; a.rows],
            levels,
            #[cfg(feature = "vulkan")]
            coarse: a,
            coarsest,
            degree: options.smoothing_degree.max(1),
            complexity: total_nnz / fine_nnz,
        })
    }

    /// Levels, the coarsest included.
    pub fn levels(&self) -> usize {
        self.levels.len() + 1
    }

    /// Rows of each level, finest first.
    pub fn level_sizes(&self) -> Vec<usize> {
        self.levels.iter().map(|l| l.a.rows).chain([self.coarse_rhs.len()]).collect()
    }

    /// Stored entries of every level over the fine matrix's: the cost of a cycle in fine
    /// matrix-vector products, about.
    pub fn operator_complexity(&self) -> f64 {
        self.complexity
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn hierarchy(&self) -> (&[Level], &Mat) {
        (&self.levels, &self.coarse)
    }

    #[cfg(feature = "vulkan")]
    pub(crate) fn smoothing_degree(&self) -> usize {
        self.degree
    }

    fn cycle(&mut self, l: usize) {
        if l == self.levels.len() {
            match &self.coarsest {
                Coarsest::Sparse(c) => c.solve_in_place(&mut self.coarse_rhs),
                Coarsest::Dense(c) => c.solve_in_place(&mut self.coarse_rhs),
            }
            return;
        }
        let degree = self.degree;
        {
            let lv = &mut self.levels[l];
            lv.x.fill(0.0);
            chebyshev(lv, degree);
            lv.a.multiply(&lv.x, &mut lv.t);
            lv.res.iter_mut().zip(&lv.b).zip(&lv.t).for_each(|((r, b), t)| *r = b - t);
        }
        let (fine, coarse) = self.levels.split_at_mut(l + 1);
        let lv = &mut fine[l];
        match coarse.first_mut() {
            Some(next) => lv.r.multiply(&lv.res, &mut next.b),
            None => lv.r.multiply(&lv.res, &mut self.coarse_rhs),
        }
        self.cycle(l + 1);
        let (fine, coarse) = self.levels.split_at_mut(l + 1);
        let lv = &mut fine[l];
        let correction = coarse.first().map_or(&self.coarse_rhs, |next| &next.x);
        lv.p.multiply(correction, &mut lv.t);
        lv.x.iter_mut().zip(&lv.t).for_each(|(x, t)| *x += t);
        chebyshev(lv, degree);
    }
}

/// `x += p(D⁻¹A) D⁻¹ (b − A x)`, the Chebyshev polynomial of the level's interval.
fn chebyshev(lv: &mut Level, degree: usize) {
    let theta = 0.5 * (lv.high + lv.low);
    let delta = 0.5 * (lv.high - lv.low);
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    lv.a.multiply(&lv.x, &mut lv.t);
    lv.res.par_iter_mut().zip(&lv.b).zip(&lv.t).with_min_len(4096).for_each(|((r, b), t)| *r = b - t);
    lv.d.par_iter_mut().zip(&lv.res).zip(&lv.dinv).with_min_len(4096).for_each(|((d, r), di)| *d = r * di / theta);
    for step in 0..degree {
        lv.x.par_iter_mut().zip(&lv.d).with_min_len(4096).for_each(|(x, d)| *x += d);
        if step + 1 == degree {
            break;
        }
        lv.a.multiply(&lv.d, &mut lv.t);
        lv.res.par_iter_mut().zip(&lv.t).with_min_len(4096).for_each(|(r, t)| *r -= t);
        let rho_new = 1.0 / (2.0 * sigma - rho);
        let (c1, c2) = (rho_new * rho, 2.0 * rho_new / delta);
        lv.d.par_iter_mut().zip(&lv.res).zip(&lv.dinv).with_min_len(4096).for_each(|((d, r), di)| *d = c1 * *d + c2 * r * di);
        rho = rho_new;
    }
}

/// A pointer for tasks that write disjoint slots or ranges of one buffer.
#[derive(Clone, Copy)]
struct Scatter<T>(*mut T);

unsafe impl<T> Send for Scatter<T> {}
unsafe impl<T> Sync for Scatter<T> {}

impl<T> Scatter<T> {
    /// # Safety
    /// `i` is in bounds, and no other task reads or writes slot `i`.
    unsafe fn write(self, i: usize, v: T) {
        unsafe { self.0.add(i).write(v) }
    }

    /// # Safety
    /// `start..start + len` is in bounds, and no other live reference overlaps it.
    unsafe fn slice<'a>(self, start: usize, len: usize) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

impl Preconditioner<[f64]> for Amg {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        match self.levels.first_mut() {
            Some(top) => {
                top.b.copy_from_slice(r);
                self.cycle(0);
                z.copy_from_slice(&self.levels[0].x);
            }
            None => {
                self.coarse_rhs.copy_from_slice(r);
                self.cycle(0);
                z.copy_from_slice(&self.coarse_rhs);
            }
        }
    }
}
