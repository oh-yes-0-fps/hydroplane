use rayon::prelude::*;

use super::Preconditioner;

/// Rows from which the triangular solves run level by level in parallel: below, the levels
/// are too narrow to pay for the synchronization.
const PARALLEL_ROWS: usize = 20_000;

/// A sparse matrix by rows, sorted, with each row's diagonal position.
#[derive(Clone, Debug)]
struct Rows {
    ptr: Vec<usize>,
    col: Vec<u32>,
    val: Vec<f64>,
    diag: Vec<usize>,
}

impl Rows {
    /// The rows of the compressed-sparse-column matrix, keeping the entries `keep(row, col)`.
    fn from_csc(col_ptr: &[usize], row_idx: &[usize], values: &[f64], keep: impl Fn(usize, usize) -> bool) -> Option<Self> {
        let n = col_ptr.len() - 1;
        let mut ptr = vec![0usize; n + 1];
        for j in 0..n {
            for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if keep(i, j) {
                    ptr[i + 1] += 1;
                }
            }
        }
        for i in 0..n {
            ptr[i + 1] += ptr[i];
        }
        let mut fill = ptr.clone();
        let mut col = vec![0u32; ptr[n]];
        let mut val = vec![0.0; ptr[n]];
        for j in 0..n {
            for (p, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
                if keep(i, j) {
                    col[fill[i]] = j as u32;
                    val[fill[i]] = values[p];
                    fill[i] += 1;
                }
            }
        }
        let mut diag = vec![0; n];
        for i in 0..n {
            diag[i] = (ptr[i]..ptr[i + 1]).find(|&p| col[p] as usize == i)?;
        }
        Some(Self { ptr, col, val, diag })
    }

    fn n(&self) -> usize {
        self.diag.len()
    }

    /// The transpose's rows, keeping the diagonal positions.
    fn transpose(&self) -> Self {
        let n = self.n();
        let mut ptr = vec![0usize; n + 1];
        for &c in &self.col {
            ptr[c as usize + 1] += 1;
        }
        for i in 0..n {
            ptr[i + 1] += ptr[i];
        }
        let mut fill = ptr.clone();
        let mut col = vec![0u32; self.col.len()];
        let mut val = vec![0.0; self.val.len()];
        for i in 0..n {
            for p in self.ptr[i]..self.ptr[i + 1] {
                let c = self.col[p] as usize;
                col[fill[c]] = i as u32;
                val[fill[c]] = self.val[p];
                fill[c] += 1;
            }
        }
        let diag = (0..n).map(|i| (ptr[i]..ptr[i + 1]).find(|&p| col[p] as usize == i).expect("a diagonal")).collect();
        Self { ptr, col, val, diag }
    }
}

/// A triangular solve's rows grouped into levels: every row depends only on rows of earlier
/// levels.
#[derive(Clone, Debug, Default)]
struct Levels(Vec<Vec<u32>>);

impl Levels {
    /// For the rows' entries left of the diagonal (`lower`) or right of it, in that order of
    /// solving.
    fn new(rows: &Rows, lower: bool) -> Self {
        let n = rows.n();
        if n < PARALLEL_ROWS {
            return Self::default();
        }
        let mut level = vec![0u32; n];
        let order: Box<dyn Iterator<Item = usize>> = if lower { Box::new(0..n) } else { Box::new((0..n).rev()) };
        let mut levels: Vec<Vec<u32>> = Vec::new();
        for i in order {
            let deps = if lower { rows.ptr[i]..rows.diag[i] } else { rows.diag[i] + 1..rows.ptr[i + 1] };
            let l = deps.map(|p| level[rows.col[p] as usize] + 1).max().unwrap_or(0);
            level[i] = l;
            if levels.len() <= l as usize {
                levels.push(Vec::new());
            }
            levels[l as usize].push(i as u32);
        }
        Self(levels)
    }
}

/// `z_i = (r_i − Σ_{j in deps(i)} a_ij z_j) / d_i` over `rows` in the levels' order (or plain
/// order when there are none), `d_i` the diagonal when `divide`.
fn substitute(rows: &Rows, levels: &Levels, lower: bool, divide: bool, r: &[f64], z: &mut [f64]) {
    let row = |i: usize, z: &[f64]| {
        let deps = if lower { rows.ptr[i]..rows.diag[i] } else { rows.diag[i] + 1..rows.ptr[i + 1] };
        let mut v = r[i];
        for p in deps {
            v -= rows.val[p] * z[rows.col[p] as usize];
        }
        if divide { v / rows.val[rows.diag[i]] } else { v }
    };
    if levels.0.is_empty() {
        if lower {
            for i in 0..rows.n() {
                z[i] = row(i, z);
            }
        } else {
            for i in (0..rows.n()).rev() {
                z[i] = row(i, z);
            }
        }
        return;
    }
    let out = SharedSlice(z.as_mut_ptr(), z.len());
    for level in &levels.0 {
        level.par_iter().with_min_len(64).for_each(|&i| {
            // SAFETY: rows of one level read only rows of earlier levels and write themselves.
            let v = row(i as usize, unsafe { out.read() });
            unsafe { out.write(i as usize, v) };
        });
    }
}

#[derive(Clone, Copy)]
struct SharedSlice(*mut f64, usize);

// Each level's rows write disjoint entries and read only finished ones.
unsafe impl Send for SharedSlice {}
unsafe impl Sync for SharedSlice {}

impl SharedSlice {
    /// # Safety
    /// No entry read through it may be written concurrently.
    unsafe fn read(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.0, self.1) }
    }

    /// # Safety
    /// Entry `i` must not be read or written concurrently.
    unsafe fn write(&self, i: usize, v: f64) {
        unsafe { self.0.add(i).write(v) }
    }
}

/// Incomplete LU with no fill, ILU(0): `L U ≈ A` on `A`'s own pattern. The usual first
/// preconditioner for [`super::Gmres`] or [`super::BiCgStab`] on an assembled nonsymmetric matrix:
/// one factorization pass, then two sparse triangular solves per application (sequential).
#[derive(Clone, Debug)]
pub struct Ilu0 {
    lu: Rows,
    lower: Levels,
    upper: Levels,
}

impl Ilu0 {
    /// From the whole matrix in compressed sparse columns; `None` if a diagonal entry is missing
    /// from the pattern or a pivot vanishes.
    pub fn new(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Option<Self> {
        let mut lu = Rows::from_csc(col_ptr, row_idx, values, |_, _| true)?;
        let n = lu.n();
        let mut position = vec![usize::MAX; n];
        for i in 0..n {
            let (start, end) = (lu.ptr[i], lu.ptr[i + 1]);
            for p in start..end {
                position[lu.col[p] as usize] = p;
            }
            for p in start..lu.diag[i] {
                let k = lu.col[p] as usize;
                let pivot = lu.val[lu.diag[k]];
                if pivot == 0.0 || !pivot.is_finite() {
                    return None;
                }
                let lik = lu.val[p] / pivot;
                lu.val[p] = lik;
                for q in lu.diag[k] + 1..lu.ptr[k + 1] {
                    let at = position[lu.col[q] as usize];
                    if at != usize::MAX {
                        lu.val[at] -= lik * lu.val[q];
                    }
                }
            }
            for p in start..end {
                position[lu.col[p] as usize] = usize::MAX;
            }
        }
        let last = lu.val[*lu.diag.last()?];
        (last != 0.0 && last.is_finite()).then(|| Self { lower: Levels::new(&lu, true), upper: Levels::new(&lu, false), lu })
    }
}

impl Preconditioner<[f64]> for Ilu0 {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        let mut y = vec![0.0; r.len()];
        substitute(&self.lu, &self.lower, true, false, r, &mut y);
        substitute(&self.lu, &self.upper, false, true, &y, z);
    }
}

/// Incomplete Cholesky with no fill, IC(0): `L Lᵀ ≈ A` on the pattern of `A`'s lower triangle,
/// for [`super::Cg`]. A pivot that is not positive restarts the factorization with the diagonal
/// raised by a growing multiple of itself (Manteuffel's shift), so it always exists for a
/// symmetric positive definite `A`.
#[derive(Clone, Debug)]
pub struct Ic0 {
    l: Rows,
    lt: Rows,
    lower: Levels,
    upper: Levels,
    shift: f64,
}

impl Ic0 {
    /// From the lower triangle in compressed sparse columns; `None` if a diagonal entry is
    /// missing or no shift up to 10× the diagonal helps.
    pub fn new(col_ptr: &[usize], row_idx: &[usize], values: &[f64]) -> Option<Self> {
        let base = Rows::from_csc(col_ptr, row_idx, values, |i, j| i >= j)?;
        let mut shift = 0.0;
        loop {
            if let Some(l) = Self::factor(&base, shift) {
                let lt = l.transpose();
                return Some(Self { lower: Levels::new(&l, true), upper: Levels::new(&lt, false), l, lt, shift });
            }
            shift = if shift == 0.0 { 1e-3 } else { shift * 2.0 };
            if shift > 10.0 {
                return None;
            }
        }
    }

    /// The diagonal shift, relative to the diagonal, the factorization needed.
    pub fn shift(&self) -> f64 {
        self.shift
    }

    fn factor(base: &Rows, shift: f64) -> Option<Rows> {
        let mut l = base.clone();
        let n = l.n();
        for i in 0..n {
            l.val[l.diag[i]] *= 1.0 + shift;
        }
        let mut position = vec![usize::MAX; n];
        for i in 0..n {
            let (start, d) = (l.ptr[i], l.diag[i]);
            for p in start..=d {
                position[l.col[p] as usize] = p;
            }
            for p in start..d {
                let k = l.col[p] as usize;
                let mut v = l.val[p];
                for q in l.ptr[k]..l.diag[k] {
                    let at = position[l.col[q] as usize];
                    if at != usize::MAX && at < p {
                        v -= l.val[at] * l.val[q];
                    }
                }
                l.val[p] = v / l.val[l.diag[k]];
            }
            let mut dv = l.val[d];
            for p in start..d {
                dv -= l.val[p] * l.val[p];
            }
            for p in start..=d {
                position[l.col[p] as usize] = usize::MAX;
            }
            if dv.is_nan() || dv <= 0.0 {
                return None;
            }
            l.val[d] = dv.sqrt();
        }
        Some(l)
    }
}

impl Preconditioner<[f64]> for Ic0 {
    fn apply(&mut self, r: &[f64], z: &mut [f64]) {
        let mut y = vec![0.0; r.len()];
        substitute(&self.l, &self.lower, true, true, r, &mut y);
        substitute(&self.lt, &self.upper, false, true, &y, z);
    }
}
