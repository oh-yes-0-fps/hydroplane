//! Batched small dense kernels: many independent 3×3 (or up to 8×8) problems at once, one per SIMD
//! lane. The FEM and robotics staples (principal stresses, deformation gradients, inertia tensors,
//! 6×6 spatial blocks) come in their thousands per step, and branch-free per-lane math is where
//! wide registers pay off.
//!
//! Inputs and outputs are structure-of-arrays: one slice per matrix entry, all the same length
//! (the batch size). A 3×3 matrix is 9 slices in row-major order (`a[3 * i + j]` holds every
//! matrix's `A[i][j]`); a symmetric one is 6 slices `[a00, a01, a02, a11, a12, a22]`. Each kernel
//! runs in `f32` or `f64`.

use hydroplane::{Backend, FloatScalar, Gang, SimdDispatch, Varying, kernel};

/// Cyclic Jacobi sweeps over the three off-diagonal pairs. Each sweep roughly squares the
/// off-diagonal mass once it is small, so a fixed count reaches working precision for every
/// input without a data-dependent loop: 3 sweeps suffice in `f32` and 4 in `f64` on the test
/// matrices (repeated, near-repeated and zero eigenvalues, six decades of scale); one more of each
/// is margin.
const fn sweeps<T>() -> usize {
    if core::mem::size_of::<T>() <= 4 { 4 } else { 5 }
}

type M3<T, S> = [[Varying<T, S>; 3]; 3];

/// A signed SVD's `(U, sigma, V)`.
type Svd3<T, S> = (M3<T, S>, [Varying<T, S>; 3], M3<T, S>);

/// One Jacobi rotation zeroing `a[p][q]`, accumulated into the columns of `v` (Numerical Recipes'
/// form, stable for every `θ`). Lanes where `a[p][q]` is already zero rotate by the identity.
#[inline(always)]
fn rotate<T: FloatScalar, S: Backend<T>>(g: Gang<S>, a: &mut M3<T, S>, v: &mut M3<T, S>, p: usize, q: usize) {
    let r = 3 - p - q;
    let (zero, one) = (g.splat(T::ZERO), g.splat(T::ONE));
    let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
    let theta = (aqq - app) / (apq + apq);
    let t = one / (theta.abs() + (theta * theta + one).sqrt());
    let t = t.select(theta.ge(zero), -t);
    let t = zero.select(apq.abs().le(zero), t);
    let c = one / (t * t + one).sqrt();
    let s = t * c;

    a[p][p] = app - t * apq;
    a[q][q] = aqq + t * apq;
    a[p][q] = zero;
    a[q][p] = zero;
    let (arp, arq) = (a[r][p], a[r][q]);
    a[r][p] = c * arp - s * arq;
    a[p][r] = a[r][p];
    a[r][q] = s * arp + c * arq;
    a[q][r] = a[r][q];
    for row in v.iter_mut() {
        let (vp, vq) = (row[p], row[q]);
        row[p] = c * vp - s * vq;
        row[q] = s * vp + c * vq;
    }
}

/// Puts lanes `i`/`j` in order (ascending unless `descending`), carrying the eigenvector columns
/// along. The moved column is negated so `v` stays a proper rotation (`det = +1`).
#[inline(always)]
fn order<T: FloatScalar, S: Backend<T>>(lam: &mut [Varying<T, S>; 3], v: &mut M3<T, S>, i: usize, j: usize, descending: bool) {
    let swap = if descending { lam[i].lt(lam[j]) } else { lam[i].gt(lam[j]) };
    let (li, lj) = (lam[i], lam[j]);
    lam[i] = lj.select(swap, li);
    lam[j] = li.select(swap, lj);
    for row in v.iter_mut() {
        let (vi, vj) = (row[i], row[j]);
        row[i] = vj.select(swap, vi);
        row[j] = (-vi).select(swap, vj);
    }
}

/// Eigen-decomposition of the symmetric `a` in place: on return `a` is (numerically) diagonal and
/// `v` holds the eigenvectors as columns, as a proper rotation.
#[inline(always)]
fn jacobi<T: FloatScalar, S: Backend<T>>(g: Gang<S>, a: &mut M3<T, S>) -> M3<T, S> {
    let (zero, one) = (g.splat(T::ZERO), g.splat(T::ONE));
    let mut v = [[one, zero, zero], [zero, one, zero], [zero, zero, one]];
    for _ in 0..sweeps::<T>() {
        rotate(g, a, &mut v, 0, 1);
        rotate(g, a, &mut v, 0, 2);
        rotate(g, a, &mut v, 1, 2);
    }
    v
}

#[inline(always)]
fn sort3<T: FloatScalar, S: Backend<T>>(lam: &mut [Varying<T, S>; 3], v: &mut M3<T, S>, descending: bool) {
    order(lam, v, 0, 1, descending);
    order(lam, v, 1, 2, descending);
    order(lam, v, 0, 1, descending);
}

/// Signed SVD of each lane's `a`: `a = u · diag(sigma) · vᵀ` with `u`, `v` proper rotations,
/// `sigma[0] ≥ sigma[1] ≥ |sigma[2]|`, and the sign of `det a` carried by `sigma[2]`.
#[inline(always)]
fn svd<T: FloatScalar, S: Backend<T>>(g: Gang<S>, a: &M3<T, S>) -> Svd3<T, S> {
    let (zero, one) = (g.splat(T::ZERO), g.splat(T::ONE));
    // Normal matrix AᵀA: its eigenvectors are the right singular vectors.
    let mut ata = [[zero; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            let s = a[0][i] * a[0][j] + a[1][i] * a[1][j] + a[2][i] * a[2][j];
            ata[i][j] = s;
            ata[j][i] = s;
        }
    }
    let mut v = jacobi(g, &mut ata);
    let mut lam = [ata[0][0], ata[1][1], ata[2][2]];
    sort3(&mut lam, &mut v, true);

    // B = A·V has orthogonal columns; Givens QR splits it into U and the (diagonal) R.
    let mut b = [[zero; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = a[i][0] * v[0][j] + a[i][1] * v[1][j] + a[i][2] * v[2][j];
        }
    }
    let mut u = [[one, zero, zero], [zero, one, zero], [zero, zero, one]];
    for (p, r) in [(0, 1), (0, 2), (1, 2)] {
        givens(g, &mut b, &mut u, p, r);
    }
    (u, [b[0][0], b[1][1], b[2][2]], v)
}

/// Zeroes `b[r][p]` with a rotation of rows `p`, `r`, accumulating its transpose into `u`'s columns.
/// `b[p][p]` comes out non-negative; an all-zero pair rotates by the identity.
#[inline(always)]
fn givens<T: FloatScalar, S: Backend<T>>(g: Gang<S>, b: &mut M3<T, S>, u: &mut M3<T, S>, p: usize, r: usize) {
    let (zero, one) = (g.splat(T::ZERO), g.splat(T::ONE));
    let (x, y) = (b[p][p], b[r][p]);
    let h = (x * x + y * y).sqrt();
    let nonzero = h.gt(zero);
    let c = (x / h).select(nonzero, one);
    let s = (y / h).select(nonzero, zero);
    for k in 0..3 {
        let (bp, br) = (b[p][k], b[r][k]);
        b[p][k] = c * bp + s * br;
        b[r][k] = c * br - s * bp;
        let (up, ur) = (u[k][p], u[k][r]);
        u[k][p] = c * up + s * ur;
        u[k][r] = c * ur - s * up;
    }
}

#[inline(always)]
fn load3<T: FloatScalar, S: Backend<T>>(g: Gang<S>, a: &[&[T]; 9], off: usize, cnt: usize) -> M3<T, S> {
    let at = |i: usize, j: usize| g.load_partial(&a[3 * i + j][off..off + cnt], T::ZERO);
    [[at(0, 0), at(0, 1), at(0, 2)], [at(1, 0), at(1, 1), at(1, 2)], [at(2, 0), at(2, 1), at(2, 2)]]
}

#[inline(always)]
fn store3<T: FloatScalar, S: Backend<T>>(m: &M3<T, S>, out: &mut [&mut [T]; 9], off: usize, cnt: usize) {
    for i in 0..3 {
        for j in 0..3 {
            m[i][j].store_partial(&mut out[3 * i + j][off..off + cnt]);
        }
    }
}

#[kernel]
fn sym3_eigen_k<'a, 'b, T: FloatScalar>(
    ctx: Gang,
    a: [&'a [T]; 6],
    values: [&'b mut [T]; 3],
    vectors: [&'b mut [T]; 9],
) {
    let mut values = values;
    let mut vectors = vectors;
    ctx.for_each_chunk::<T>(a[0].len(), |off, cnt| {
        let ld = |k: usize| ctx.load_partial(&a[k][off..off + cnt], T::ZERO);
        let (a00, a01, a02, a11, a12, a22) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5));
        let mut m = [[a00, a01, a02], [a01, a11, a12], [a02, a12, a22]];
        let mut v = jacobi(ctx, &mut m);
        let mut lam = [m[0][0], m[1][1], m[2][2]];
        sort3(&mut lam, &mut v, false);
        for (k, l) in lam.iter().enumerate() {
            l.store_partial(&mut values[k][off..off + cnt]);
        }
        store3(&v, &mut vectors, off, cnt);
    });
}

#[kernel]
fn svd3_k<'a, 'b, T: FloatScalar>(
    ctx: Gang,
    a: [&'a [T]; 9],
    u: [&'b mut [T]; 9],
    sigma: [&'b mut [T]; 3],
    v: [&'b mut [T]; 9],
) {
    let (mut u, mut sigma, mut v) = (u, sigma, v);
    ctx.for_each_chunk::<T>(a[0].len(), |off, cnt| {
        let m = load3(ctx, &a, off, cnt);
        let (mu, ms, mv) = svd(ctx, &m);
        store3(&mu, &mut u, off, cnt);
        for (k, s) in ms.iter().enumerate() {
            s.store_partial(&mut sigma[k][off..off + cnt]);
        }
        store3(&mv, &mut v, off, cnt);
    });
}

#[kernel]
fn polar3_k<'a, 'b, T: FloatScalar>(ctx: Gang, a: [&'a [T]; 9], rot: [&'b mut [T]; 9], stretch: [&'b mut [T]; 6]) {
    let (mut rot, mut stretch) = (rot, stretch);
    ctx.for_each_chunk::<T>(a[0].len(), |off, cnt| {
        let m = load3(ctx, &a, off, cnt);
        let (u, s, v) = svd(ctx, &m);
        let mut r = m;
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = u[i][0] * v[j][0] + u[i][1] * v[j][1] + u[i][2] * v[j][2];
            }
        }
        store3(&r, &mut rot, off, cnt);
        let sym = |i: usize, j: usize| v[i][0] * s[0] * v[j][0] + v[i][1] * s[1] * v[j][1] + v[i][2] * s[2] * v[j][2];
        for (k, (i, j)) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)].into_iter().enumerate() {
            sym(i, j).store_partial(&mut stretch[k][off..off + cnt]);
        }
    });
}

/// Packed index of `L[i][j]`, `j ≤ i`, in a row-major lower triangle.
const fn tri(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// Largest `N` the `N×N` solvers take: their factor lives in registers.
pub const MAX_N: usize = 8;

#[kernel]
fn cholesky_solve_k<'a, 'b, 'c, T: FloatScalar, const N: usize>(
    ctx: Gang,
    a: &'a [&'a [T]],
    b: &'a [&'a [T]],
    x: &'c mut [&'b mut [T]],
) {
    ctx.for_each_chunk::<T>(b[0].len(), |off, cnt| {
        let (zero, one) = (ctx.splat(T::ZERO), ctx.splat(T::ONE));
        let mut l = [zero; MAX_N * (MAX_N + 1) / 2];
        for i in 0..N {
            for j in 0..=i {
                let fill = if i == j { T::ONE } else { T::ZERO };
                l[tri(i, j)] = ctx.load_partial(&a[tri(i, j)][off..off + cnt], fill);
            }
        }
        for j in 0..N {
            let mut d = l[tri(j, j)];
            for k in 0..j {
                d = d - l[tri(j, k)] * l[tri(j, k)];
            }
            let d = d.sqrt();
            l[tri(j, j)] = d;
            let inv = one / d;
            for i in j + 1..N {
                let mut s = l[tri(i, j)];
                for k in 0..j {
                    s = s - l[tri(i, k)] * l[tri(j, k)];
                }
                l[tri(i, j)] = s * inv;
            }
        }
        let mut y = [zero; MAX_N];
        for i in 0..N {
            let mut s = ctx.load_partial(&b[i][off..off + cnt], T::ZERO);
            for k in 0..i {
                s = s - l[tri(i, k)] * y[k];
            }
            y[i] = s / l[tri(i, i)];
        }
        for i in (0..N).rev() {
            let mut s = y[i];
            for k in i + 1..N {
                s = s - l[tri(k, i)] * y[k];
            }
            y[i] = s / l[tri(i, i)];
        }
        for (i, yi) in y.iter().enumerate().take(N) {
            yi.store_partial(&mut x[i][off..off + cnt]);
        }
    });
}

#[kernel]
fn ldlt_solve_k<'a, 'b, 'c, T: FloatScalar, const N: usize>(
    ctx: Gang,
    a: &'a [&'a [T]],
    b: &'a [&'a [T]],
    x: &'c mut [&'b mut [T]],
) {
    ctx.for_each_chunk::<T>(b[0].len(), |off, cnt| {
        let zero = ctx.splat(T::ZERO);
        // Unit-lower L below the diagonal, D on it.
        let mut l = [zero; MAX_N * (MAX_N + 1) / 2];
        for i in 0..N {
            for j in 0..=i {
                let fill = if i == j { T::ONE } else { T::ZERO };
                l[tri(i, j)] = ctx.load_partial(&a[tri(i, j)][off..off + cnt], fill);
            }
        }
        for j in 0..N {
            let mut d = l[tri(j, j)];
            for k in 0..j {
                d = d - l[tri(j, k)] * l[tri(j, k)] * l[tri(k, k)];
            }
            l[tri(j, j)] = d;
            for i in j + 1..N {
                let mut s = l[tri(i, j)];
                for k in 0..j {
                    s = s - l[tri(i, k)] * l[tri(j, k)] * l[tri(k, k)];
                }
                l[tri(i, j)] = s / d;
            }
        }
        let mut y = [zero; MAX_N];
        for i in 0..N {
            let mut s = ctx.load_partial(&b[i][off..off + cnt], T::ZERO);
            for k in 0..i {
                s = s - l[tri(i, k)] * y[k];
            }
            y[i] = s;
        }
        for (i, yi) in y.iter_mut().enumerate().take(N) {
            *yi = *yi / l[tri(i, i)];
        }
        for i in (0..N).rev() {
            let mut s = y[i];
            for k in i + 1..N {
                s = s - l[tri(k, i)] * y[k];
            }
            y[i] = s;
        }
        for (i, yi) in y.iter().enumerate().take(N) {
            yi.store_partial(&mut x[i][off..off + cnt]);
        }
    });
}

fn same_len<T>(what: &str, n: usize, cols: &[&[T]]) {
    assert!(cols.iter().all(|c| c.len() == n), "batched::{what}: every column must hold the same number of matrices");
}

fn same_len_mut<T>(what: &str, n: usize, cols: &[&mut [T]]) {
    assert!(cols.iter().all(|c| c.len() == n), "batched::{what}: every column must hold the same number of matrices");
}

/// Eigen-decomposition of each symmetric 3×3 `a` (`[a00, a01, a02, a11, a12, a22]`): eigenvalues
/// ascending in `values`, unit eigenvectors as the columns of the row-major `vectors` (a proper
/// rotation, so `a = V · diag(values) · Vᵀ`). Cyclic Jacobi with a fixed sweep count: accurate for
/// repeated and near-repeated eigenvalues, where closed-form cubic solutions lose precision.
pub fn sym3_eigen<T: FloatScalar + SimdDispatch>(a: [&[T]; 6], values: [&mut [T]; 3], vectors: [&mut [T]; 9]) {
    let n = a[0].len();
    same_len("sym3_eigen", n, &a);
    same_len_mut("sym3_eigen", n, &values);
    same_len_mut("sym3_eigen", n, &vectors);
    sym3_eigen_k(a, values, vectors);
}

/// Signed SVD of each 3×3 `a` (row-major): `a = U · diag(sigma) · Vᵀ` with `U`, `V` proper
/// rotations and `sigma[0] ≥ sigma[1] ≥ |sigma[2]|`. An inverted `a` (`det < 0`) shows as a
/// negative `sigma[2]` rather than a reflection in `U` or `V`, the convention FEM needs to
/// recover from inverted elements. Built on the eigenvectors of `AᵀA`, so singular values far
/// below `sigma[0]` carry an absolute error near `ε · sigma[0]`.
pub fn svd3<T: FloatScalar + SimdDispatch>(a: [&[T]; 9], u: [&mut [T]; 9], sigma: [&mut [T]; 3], v: [&mut [T]; 9]) {
    let n = a[0].len();
    same_len("svd3", n, &a);
    same_len_mut("svd3", n, &u);
    same_len_mut("svd3", n, &sigma);
    same_len_mut("svd3", n, &v);
    svd3_k(a, u, sigma, v);
}

/// Polar decomposition of each 3×3 `a` (row-major): `a = R · S` with `R` a proper rotation
/// (row-major) and `S` symmetric (`[s00, s01, s02, s11, s12, s22]`). From the signed SVD, so an
/// inverted `a` gives an indefinite `S` instead of a reflection in `R`.
pub fn polar3<T: FloatScalar + SimdDispatch>(a: [&[T]; 9], rotation: [&mut [T]; 9], stretch: [&mut [T]; 6]) {
    let n = a[0].len();
    same_len("polar3", n, &a);
    same_len_mut("polar3", n, &rotation);
    same_len_mut("polar3", n, &stretch);
    polar3_k(a, rotation, stretch);
}

fn check_system<T>(what: &str, n_dim: usize, a: &[&[T]], b: &[&[T]], x: &[&mut [T]]) {
    assert!(n_dim <= MAX_N, "batched::{what}: N = {n_dim} exceeds MAX_N = {MAX_N}");
    assert_eq!(a.len(), tri(n_dim, 0), "batched::{what}: `a` needs N(N+1)/2 columns (packed lower triangle)");
    assert_eq!(b.len(), n_dim, "batched::{what}: `b` needs N columns");
    assert_eq!(x.len(), n_dim, "batched::{what}: `x` needs N columns");
    let n = b.first().map_or(0, |c| c.len());
    same_len(what, n, a);
    same_len(what, n, b);
    same_len_mut(what, n, x);
}

/// Solves `A x = b` for each symmetric positive definite `N×N` `A` (`N ≤ MAX_N`), given as its
/// row-major packed lower triangle (`a[i(i+1)/2 + j]` holds `A[i][j]`, `j ≤ i`). A matrix that is
/// not positive definite yields NaN in its `x`.
pub fn cholesky_solve<T: FloatScalar + SimdDispatch, const N: usize>(a: &[&[T]], b: &[&[T]], x: &mut [&mut [T]]) {
    check_system("cholesky_solve", N, a, b, x);
    if N == 0 || b[0].is_empty() {
        return;
    }
    cholesky_solve_k::<T, N>(a, b, x);
}

/// Solves `A x = b` for each symmetric `N×N` `A` (`N ≤ MAX_N`, same packing as
/// [`cholesky_solve`]) by `L D Lᵀ` without pivoting: no square roots, and indefinite matrices work
/// as long as no leading principal minor is singular (a zero pivot yields inf/NaN in that `x`).
pub fn ldlt_solve<T: FloatScalar + SimdDispatch, const N: usize>(a: &[&[T]], b: &[&[T]], x: &mut [&mut [T]]) {
    check_system("ldlt_solve", N, a, b, x);
    if N == 0 || b[0].is_empty() {
        return;
    }
    ldlt_solve_k::<T, N>(a, b, x);
}
