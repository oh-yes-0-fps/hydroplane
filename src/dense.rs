//! Runtime-dimensioned dense linear algebra (the compile-time tile counterpart is
//! [`matrix`](crate::matrix)). Every routine goes to Apple Accelerate where present, otherwise
//! to a fallback built on the SIMD backend ([`Gang`]/[`Varying`](crate::Varying)).

use crate::dispatch::{Kernel, SimdDispatch, dispatch};
use crate::matrix::Layout;
use crate::scalar::{FloatScalar, Scalar};
#[cfg(feature = "alloc")]
use crate::scratch::Scratch;
use crate::varying::Gang;
use crate::{Backend, BackendAll};

/// Operand transposition flag, mirroring BLAS `TRANS`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trans {
    /// Use the operand as stored.
    N,
    /// Use its transpose without materializing it.
    T,
}

/// Which triangle of a symmetric/triangular matrix is referenced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Uplo {
    Lower,
    Upper,
}

/// Whether the triangular operand multiplies from the left or the right (TRSM/TRMM).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
}

/// Whether a triangular matrix has an implicit unit diagonal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Diag {
    Unit,
    NonUnit,
}

/// A borrowing runtime-dimensioned matrix view: a slice plus its shape, leading stride, and
/// [`Layout`]. Loading copies nothing. The dense row-major case ([`Mat::new`]) is the common one;
/// [`Mat::strided`] covers sub-matrices and column-major data.
#[derive(Clone, Copy)]
pub struct Mat<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    stride: usize,
    layout: Layout,
}

/// The mutable counterpart of [`Mat`]: the output matrix of `gemm`/`syrk`/`trsm`/`potrf`
/// (available with the `alloc` feature; the vector-output ops write plain `&mut [T]`).
#[cfg(feature = "alloc")]
pub struct MatMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
    stride: usize,
    layout: Layout,
}

/// Whether a `len`-element slice holds a `rows × cols` view with this stride and layout. The
/// BLAS routines (Accelerate's included) are handed raw pointers, so the views alone guarantee
/// every access is in bounds.
#[inline]
fn covers(len: usize, rows: usize, cols: usize, stride: usize, layout: Layout) -> bool {
    let (inner, outer) = match layout {
        Layout::RowMajor => (cols, rows),
        Layout::ColMajor => (rows, cols),
    };
    stride >= inner && (inner == 0 || outer == 0 || (outer - 1).checked_mul(stride).and_then(|o| o.checked_add(inner)).is_some_and(|end| end <= len))
}

impl<'a, T> Mat<'a, T> {
    /// A dense row-major `rows × cols` view (`stride = cols`).
    #[inline]
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        assert!(rows.checked_mul(cols).is_some_and(|n| n <= data.len()), "Mat::new: slice too short");
        Mat { data, rows, cols, stride: cols, layout: Layout::RowMajor }
    }

    /// A view with an explicit leading stride and layout: a sub-matrix of a larger buffer, or
    /// column-major data. `data` must reach the last element, `(outer − 1)·stride + inner`.
    #[inline]
    pub fn strided(data: &'a [T], rows: usize, cols: usize, stride: usize, layout: Layout) -> Self {
        assert!(covers(data.len(), rows, cols, stride, layout), "Mat::strided: stride below inner dimension, or slice too short");
        Mat { data, rows, cols, stride, layout }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
}

#[cfg(feature = "alloc")]
impl<'a, T: Copy> Mat<'a, T> {
    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> T {
        self.data[self.off(r, c)]
    }

    #[inline(always)]
    fn off(&self, r: usize, c: usize) -> usize {
        match self.layout {
            Layout::RowMajor => r * self.stride + c,
            Layout::ColMajor => c * self.stride + r,
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a, T> MatMut<'a, T> {
    /// A dense row-major mutable view (`stride = cols`).
    #[inline]
    pub fn new(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        assert!(rows.checked_mul(cols).is_some_and(|n| n <= data.len()), "MatMut::new: slice too short");
        MatMut { data, rows, cols, stride: cols, layout: Layout::RowMajor }
    }

    /// A mutable view with an explicit leading stride and layout; `data` must reach the last
    /// element, as for [`Mat::strided`].
    #[inline]
    pub fn strided(
        data: &'a mut [T],
        rows: usize,
        cols: usize,
        stride: usize,
        layout: Layout,
    ) -> Self {
        assert!(covers(data.len(), rows, cols, stride, layout), "MatMut::strided: stride below inner dimension, or slice too short");
        MatMut { data, rows, cols, stride, layout }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    fn off(&self, r: usize, c: usize) -> usize {
        match self.layout {
            Layout::RowMajor => r * self.stride + c,
            Layout::ColMajor => c * self.stride + r,
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a, T: Copy> MatMut<'a, T> {
    #[inline(always)]
    fn get(&self, r: usize, c: usize) -> T {
        self.data[self.off(r, c)]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: T) {
        let o = self.off(r, c);
        self.data[o] = v;
    }
}

/// General matrix–vector product `y := α·op(A)·x + β·y`, where `op(A)` is `A` (`Trans::N`) or
/// `Aᵀ` (`Trans::T`). Routes to `cblas_*gemv` on Apple, otherwise to a gang kernel that never
/// gathers: whichever of the operated rows or columns is contiguous drives the reduction.
pub fn gemv<T: FloatScalar + SimdDispatch>(
    trans: Trans,
    alpha: T,
    a: Mat<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    let (out_len, in_len) = match trans {
        Trans::N => (a.rows, a.cols),
        Trans::T => (a.cols, a.rows),
    };
    assert_eq!(x.len(), in_len, "gemv: x length mismatch");
    assert_eq!(y.len(), out_len, "gemv: y length mismatch");
    if out_len == 0 {
        return;
    }

    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if accel_gemv(trans, alpha, a, x, beta, y) {
        return;
    }

    dispatch::<T, _>(GemvK { trans, alpha, a, x, beta, y });
}

#[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
fn accel_gemv<T: FloatScalar>(
    trans: Trans,
    alpha: T,
    a: Mat<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) -> bool {
    use crate::matrix::accel;
    use core::any::TypeId;
    use core::mem::transmute_copy;

    let order = match a.layout {
        Layout::RowMajor => accel::ROW_MAJOR,
        Layout::ColMajor => accel::COL_MAJOR,
    };
    let tr = match trans {
        Trans::N => accel::NO_TRANS,
        Trans::T => accel::TRANS,
    };
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        unsafe {
            accel::cblas_sgemv(
                order, tr, a.rows as _, a.cols as _,
                transmute_copy::<T, f32>(&alpha), a.data.as_ptr() as *const f32, a.stride as _,
                x.as_ptr() as *const f32, 1,
                transmute_copy::<T, f32>(&beta), y.as_mut_ptr() as *mut f32, 1,
            );
        }
        return true;
    }
    if t == TypeId::of::<f64>() {
        unsafe {
            accel::cblas_dgemv(
                order, tr, a.rows as _, a.cols as _,
                transmute_copy::<T, f64>(&alpha), a.data.as_ptr() as *const f64, a.stride as _,
                x.as_ptr() as *const f64, 1,
                transmute_copy::<T, f64>(&beta), y.as_mut_ptr() as *mut f64, 1,
            );
        }
        return true;
    }
    false
}

struct GemvK<'a, T: FloatScalar> {
    trans: Trans,
    alpha: T,
    a: Mat<'a, T>,
    x: &'a [T],
    beta: T,
    y: &'a mut [T],
}

impl<'a, T: FloatScalar> Kernel<T> for GemvK<'a, T> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let GemvK { trans, alpha, a, x, beta, y } = self;
        let op_t = trans == Trans::T;
        let col_major = a.layout == Layout::ColMajor;
        // Operated rows are contiguous exactly when un-transposed row-major or transposed
        // col-major: reduce per output element. Otherwise accumulate the contiguous columns
        // axpy-style.
        let dot_form = op_t == col_major;
        let (op_rows, op_cols) = if op_t { (a.cols, a.rows) } else { (a.rows, a.cols) };

        if dot_form {
            let x = &x[..op_cols];
            let mut i = 0;
            while i + GEMV_ROWS <= op_rows {
                let d = dot_rows::<T, S, GEMV_ROWS>(simd, core::array::from_fn(|r| &a.data[(i + r) * a.stride..][..op_cols]), x);
                for (yi, d) in y[i..i + GEMV_ROWS].iter_mut().zip(d) {
                    *yi = fma_scalar(alpha, d, beta, *yi);
                }
                i += GEMV_ROWS;
            }
            for (r, yi) in y[i..op_rows].iter_mut().enumerate() {
                let base = (i + r) * a.stride;
                let d = simd.dot(&a.data[base..base + op_cols], x);
                *yi = fma_scalar(alpha, d, beta, *yi);
            }
        } else {
            scale(y, beta);
            let y = &mut y[..op_rows];
            let mut k = 0;
            while k + GEMV_ROWS <= op_cols {
                let lines = core::array::from_fn(|r| &a.data[(k + r) * a.stride..][..op_rows]);
                let mut cs = [simd.splat(T::ZERO); GEMV_ROWS];
                for (c, &xk) in cs.iter_mut().zip(&x[k..k + GEMV_ROWS]) {
                    *c = simd.splat(alpha.wmul(xk));
                }
                axpy_rows::<T, S, GEMV_ROWS>(simd, lines, cs, y);
                k += GEMV_ROWS;
            }
            for (k, &xk) in x.iter().enumerate().take(op_cols).skip(k) {
                let base = k * a.stride;
                let line = &a.data[base..base + op_rows];
                let cs = simd.splat(alpha.wmul(xk));
                simd.zip_map_inplace(line, y, T::ZERO, T::ZERO, |lv, yv| lv.fma(cs, yv));
            }
        }
    }
}

/// Rows of `A` a matrix-vector product reduces together: independent chains for short rows, and
/// one load of `x` per group.
const GEMV_ROWS: usize = 4;

/// `R` dot products against the same `x`, all rows as long as `x`, in one pass.
#[inline(always)]
fn dot_rows<T: FloatScalar, S: Backend<T>, const R: usize>(simd: Gang<S>, rows: [&[T]; R], x: &[T]) -> [T; R] {
    let n = simd.lanes::<T>();
    let len = x.len();
    let zero = simd.splat(T::ZERO);
    let mut acc = [zero; R];
    let mut off = 0;
    while off + n <= len {
        let xv = simd.load(&x[off..off + n]);
        for (acc_r, row) in acc.iter_mut().zip(&rows) {
            *acc_r = simd.load(&row[off..off + n]).fma(xv, *acc_r);
        }
        off += n;
    }
    if off < len {
        let xv = simd.load_partial(&x[off..], T::ZERO);
        for (acc_r, row) in acc.iter_mut().zip(&rows) {
            *acc_r = simd.load_partial(&row[off..len], T::ZERO).fma(xv, *acc_r);
        }
    }
    let mut out = [T::ZERO; R];
    for (o, acc_r) in out.iter_mut().zip(acc) {
        *o = acc_r.reduce_sum();
    }
    out
}

/// `y += Σ_r c[r]·lines[r]`, all lines as long as `y`, reading and writing `y` once.
#[inline(always)]
fn axpy_rows<T: FloatScalar, S: Backend<T>, const R: usize>(
    simd: Gang<S>,
    lines: [&[T]; R],
    c: [crate::Varying<T, S>; R],
    y: &mut [T],
) {
    let n = simd.lanes::<T>();
    let len = y.len();
    let mut off = 0;
    while off + n <= len {
        let mut yv = simd.load(&y[off..off + n]);
        for (line, &cr) in lines.iter().zip(&c) {
            yv = simd.load(&line[off..off + n]).fma(cr, yv);
        }
        yv.store(&mut y[off..off + n]);
        off += n;
    }
    if off < len {
        let mut yv = simd.load_partial(&y[off..], T::ZERO);
        for (line, &cr) in lines.iter().zip(&c) {
            yv = simd.load_partial(&line[off..len], T::ZERO).fma(cr, yv);
        }
        yv.store_partial(&mut y[off..]);
    }
}

/// `α·d + β·y`, honoring the BLAS rule that `β = 0` writes `y` without reading it (a `NaN` in an
/// uninitialized `y` must not poison the result).
#[inline]
fn fma_scalar<T: FloatScalar>(alpha: T, d: T, beta: T, y: T) -> T {
    if beta.into_f64() == 0.0 {
        alpha.wmul(d)
    } else {
        alpha.wmul(d).wadd(beta.wmul(y))
    }
}

/// In-place `y := β·y`, with the `β ∈ {0, 1}` fast exits.
#[inline]
fn scale<T: Scalar>(y: &mut [T], beta: T) {
    let b = beta.into_f64();
    if b == 1.0 {
        return;
    }
    if b == 0.0 {
        for yi in y.iter_mut() {
            *yi = T::ZERO;
        }
    } else {
        for yi in y.iter_mut() {
            *yi = beta.wmul(*yi);
        }
    }
}

/// Frobenius norm `√Σ Aᵢⱼ²`. Sums squares along whichever axis is contiguous, so the walk is a
/// plain gang reduction regardless of layout.
pub fn fro_norm<T: FloatScalar + SimdDispatch>(a: Mat<T>) -> T {
    T::sqrt(dispatch::<T, _>(FroK { a }))
}

struct FroK<'a, T> {
    a: Mat<'a, T>,
}

impl<'a, T: FloatScalar> Kernel<T> for FroK<'a, T> {
    type Output = T;

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) -> T {
        let a = self.a;
        let (lines, len) = match a.layout {
            Layout::RowMajor => (a.rows, a.cols),
            Layout::ColMajor => (a.cols, a.rows),
        };
        let mut ss = T::ZERO;
        for i in 0..lines {
            let base = i * a.stride;
            let line = &a.data[base..base + len];
            ss = ss.wadd(simd.sum(line, |acc, v| v.fma(v, acc)));
        }
        ss
    }
}

/// Row sums `outᵣ = Σ_c Aᵣ_c`.
pub fn row_sums<T: FloatScalar + SimdDispatch>(a: Mat<T>, out: &mut [T]) {
    assert_eq!(out.len(), a.rows, "row_sums: out length mismatch");
    dispatch::<T, _>(AxisSumK { a, out, along_rows: true });
}

/// Column sums `out_c = Σ_r Aᵣ_c`.
pub fn col_sums<T: FloatScalar + SimdDispatch>(a: Mat<T>, out: &mut [T]) {
    assert_eq!(out.len(), a.cols, "col_sums: out length mismatch");
    dispatch::<T, _>(AxisSumK { a, out, along_rows: false });
}

struct AxisSumK<'a, T> {
    a: Mat<'a, T>,
    out: &'a mut [T],
    along_rows: bool,
}

impl<'a, T: FloatScalar> Kernel<T> for AxisSumK<'a, T> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let AxisSumK { a, out, along_rows } = self;
        // When the reduced axis is contiguous in memory each output is one gang reduction;
        // otherwise accumulate the strided lines vector-wise.
        let reduce_contiguous = along_rows == (a.layout == Layout::RowMajor);
        let (out_len, line_len) = if along_rows { (a.rows, a.cols) } else { (a.cols, a.rows) };

        if reduce_contiguous {
            for (i, oi) in out.iter_mut().enumerate() {
                let base = i * a.stride;
                *oi = simd.total(&a.data[base..base + line_len]);
            }
        } else {
            for o in out.iter_mut() {
                *o = T::ZERO;
            }
            for k in 0..line_len {
                let base = k * a.stride;
                let line = &a.data[base..base + out_len];
                simd.zip_map_inplace(line, out, T::ZERO, T::ZERO, |lv, ov| lv + ov);
            }
        }
    }
}

#[cfg(feature = "alloc")]
#[inline]
fn op_dims<T>(t: Trans, m: &Mat<T>) -> (usize, usize) {
    match t {
        Trans::N => (m.rows, m.cols),
        Trans::T => (m.cols, m.rows),
    }
}

#[cfg(feature = "alloc")]
#[inline]
fn flip(t: Trans) -> Trans {
    match t {
        Trans::N => Trans::T,
        Trans::T => Trans::N,
    }
}

#[cfg(feature = "alloc")]
#[inline]
fn flip_uplo(u: Uplo) -> Uplo {
    match u {
        Uplo::Lower => Uplo::Upper,
        Uplo::Upper => Uplo::Lower,
    }
}

/// General matrix–matrix product `C := α·op(A)·op(B) + β·C`, each `op` identity (`Trans::N`) or
/// transpose (`Trans::T`) applied without materializing it. Row-major `C` on Apple routes to
/// `cblas_*gemm`; otherwise a register-blocked gang microkernel streams the contraction over a
/// packed `B` panel. A column-major `C` computes `Cᵀ = op(B)ᵀ·op(A)ᵀ` into the same buffer viewed
/// row-major, so the kernel only ever writes contiguous rows.
#[cfg(feature = "alloc")]
pub fn gemm<T: FloatScalar + SimdDispatch>(
    ta: Trans,
    tb: Trans,
    alpha: T,
    a: Mat<T>,
    b: Mat<T>,
    beta: T,
    c: MatMut<T>,
) {
    let (m, ka) = op_dims(ta, &a);
    let (kb, n) = op_dims(tb, &b);
    assert_eq!(ka, kb, "gemm: inner dimension mismatch");
    assert_eq!(c.rows, m, "gemm: C rows mismatch");
    assert_eq!(c.cols, n, "gemm: C cols mismatch");
    let k = ka;
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        for i in 0..m {
            for j in 0..n {
                let o = c.off(i, j);
                c.data[o] = if beta == T::ZERO { T::ZERO } else { beta.wmul(c.data[o]) };
            }
        }
        return;
    }

    if c.layout == Layout::ColMajor {
        // Cᵀ = op(B)ᵀ·op(A)ᵀ; the same buffer read row-major is Cᵀ (n×m) with the same stride.
        let ct = MatMut::strided(c.data, c.cols, c.rows, c.stride, Layout::RowMajor);
        gemm(flip(tb), flip(ta), alpha, b, a, beta, ct);
        return;
    }

    // A one-column or one-row product is a matrix-vector product, which the GEMM's register
    // tiles would mostly fill with padding.
    if k > 0
        && n == 1
        && (c.stride == 1 || m == 1)
        && let Some(x) = b.as_vector()
    {
        gemv(ta, alpha, a, x, beta, &mut c.data[..m]);
        return;
    }
    if k > 0
        && m == 1
        && let Some(x) = a.as_vector()
    {
        gemv(flip(tb), alpha, b, x, beta, &mut c.data[..n]);
        return;
    }

    // `f16`/`bf16` compute in `f32` anyway: run the `f32` GEMM instead of widening per lane
    // inside every multiply-add, which made `gemm::<f16>` ~100× slower than `gemm::<f32>`.
    if size_of::<T>() < size_of::<T::Compute>() {
        gemm_widened(ta, tb, alpha, a, b, beta, c, m, n, k);
        return;
    }

    let b_rows = matches!((tb, b.layout), (Trans::N, Layout::RowMajor) | (Trans::T, Layout::ColMajor));
    if b_rows && k > 0 && k <= GEMM_KC && gemm_direct_fits(m, n, k, size_of::<T>()) {
        dispatch::<T, _>(GemmDirectK { ta, a, b, alpha, beta, c, m, n, k });
        return;
    }

    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if accel_gemm(ta, tb, alpha, a, b, beta, &c, m, n, k) {
        return;
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "std",
        not(hp_no_sme),
        any(not(target_vendor = "apple"), hp_no_apple_accelerate)
    ))]
    let c = {
        let bp = pack_b(tb, &b, k, n);
        let mut c = c;
        if gemm_sme(alpha, beta, ta, a, &bp, &mut c, m, n, k) {
            return;
        }
        c
    };
    dispatch::<T, _>(GemmK::<T, T> { ta, tb, a, b, alpha, beta, c, m, n, k, tri: None });
}

#[cfg(feature = "alloc")]
impl<'a, T: Copy> Mat<'a, T> {
    /// The `rows×cols` sub-matrix at `(r0, c0)`, sharing this one's stride and layout.
    fn sub(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> Mat<'a, T> {
        let o = self.off(r0, c0).min(self.data.len());
        Mat { data: &self.data[o..], rows, cols, stride: self.stride, layout: self.layout }
    }

    /// The elements of a one-row or one-column matrix as a slice, when they are adjacent.
    fn as_vector(&self) -> Option<&'a [T]> {
        let len = self.rows * self.cols;
        let adjacent = match (self.rows, self.cols, self.layout) {
            (1, 1, _) => true,
            (1, _, Layout::RowMajor) | (_, 1, Layout::ColMajor) => true,
            (1, _, Layout::ColMajor) | (_, 1, Layout::RowMajor) => self.stride == 1,
            _ => false,
        };
        (adjacent && self.data.len() >= len).then(|| &self.data[..len])
    }
}

/// [`gemm`] for a 16-bit float `T` (whose `Compute` is `f32`) into a row-major `C`, computed in
/// `f32` with `C` narrowed back once. The microkernel widens `A` and `B` as it packs them; the
/// BLAS and SME paths take whole widened copies.
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
fn gemm_widened<T: FloatScalar>(
    ta: Trans,
    tb: Trans,
    alpha: T,
    a: Mat<T>,
    b: Mat<T>,
    beta: T,
    c: MatMut<T>,
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(core::any::TypeId::of::<T::Compute>() == core::any::TypeId::of::<f32>());
    // SAFETY: `T::Compute` is `f32` (asserted above).
    let widen = |x: &T| unsafe { core::mem::transmute_copy::<T::Compute, f32>(&x.widen()) };
    // Only the `m×n` view is widened and narrowed back: the rest of `c.data` may be other
    // matrices or padding.
    // Unread when `β = 0`: the product overwrites it without loading it.
    let mut c32 = Scratch::<f32>::new(m * n);
    if beta != T::ZERO {
        for i in 0..m {
            for j in 0..n {
                c32[i * n + j] = widen(&c.data[c.off(i, j)]);
            }
        }
    }
    let (alpha32, beta32) = (widen(&alpha), widen(&beta));
    if cfg!(any(
        all(target_vendor = "apple", not(hp_no_apple_accelerate)),
        all(target_arch = "aarch64", feature = "std", not(hp_no_sme))
    )) {
        let compact = |x: &Mat<T>| {
            let mut out = Scratch::<f32>::new(x.rows * x.cols);
            for r in 0..x.rows {
                for col in 0..x.cols {
                    out[r * x.cols + col] = widen(&x.get(r, col));
                }
            }
            out
        };
        let (a32, b32) = (compact(&a), compact(&b));
        gemm::<f32>(
            ta,
            tb,
            alpha32,
            Mat::new(&a32, a.rows, a.cols),
            Mat::new(&b32, b.rows, b.cols),
            beta32,
            MatMut::new(&mut c32, m, n),
        );
    } else {
        let c32m = MatMut::new(&mut c32, m, n);
        dispatch::<f32, _>(GemmK::<f32, T> { ta, tb, a, b, alpha: alpha32, beta: beta32, c: c32m, m, n, k, tri: None });
    }
    for i in 0..m {
        for j in 0..n {
            let o = c.off(i, j);
            // SAFETY: `T::Compute` is `f32`.
            c.data[o] = T::narrow(unsafe { core::mem::transmute_copy::<f32, T::Compute>(&c32[i * n + j]) });
        }
    }
}

#[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate), feature = "alloc"))]
#[allow(clippy::too_many_arguments)]
fn accel_gemm<T: FloatScalar>(
    ta: Trans,
    tb: Trans,
    alpha: T,
    a: Mat<T>,
    b: Mat<T>,
    beta: T,
    c: &MatMut<T>,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    use crate::matrix::accel;
    use core::any::TypeId;
    use core::mem::transmute_copy;

    // A column-major operand is a row-major transpose, so fold layout into the trans flag.
    let eff = |t: Trans, layout: Layout| {
        let logical_t = t == Trans::T;
        let stored_t = layout == Layout::ColMajor;
        if logical_t ^ stored_t { accel::TRANS } else { accel::NO_TRANS }
    };
    let (tra, trb) = (eff(ta, a.layout), eff(tb, b.layout));
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        unsafe {
            accel::cblas_sgemm(
                accel::ROW_MAJOR, tra, trb, m as _, n as _, k as _,
                transmute_copy::<T, f32>(&alpha), a.data.as_ptr() as *const f32, a.stride as _,
                b.data.as_ptr() as *const f32, b.stride as _,
                transmute_copy::<T, f32>(&beta), c.data.as_ptr() as *mut f32, c.stride as _,
            );
        }
        return true;
    }
    if t == TypeId::of::<f64>() {
        unsafe {
            accel::cblas_dgemm(
                accel::ROW_MAJOR, tra, trb, m as _, n as _, k as _,
                transmute_copy::<T, f64>(&alpha), a.data.as_ptr() as *const f64, a.stride as _,
                b.data.as_ptr() as *const f64, b.stride as _,
                transmute_copy::<T, f64>(&beta), c.data.as_ptr() as *mut f64, c.stride as _,
            );
        }
        return true;
    }
    false
}

/// Pack `op(B)` into a contiguous `k×n` row-major panel for the SME path.
#[cfg(all(feature = "alloc", target_arch = "aarch64", feature = "std", not(hp_no_sme), any(not(target_vendor = "apple"), hp_no_apple_accelerate)))]
fn pack_b<T: FloatScalar>(tb: Trans, b: &Mat<T>, k: usize, n: usize) -> alloc::vec::Vec<T> {
    let mut bp = alloc::vec![T::ZERO; k * n];
    for p in 0..k {
        let dst = &mut bp[p * n..p * n + n];
        for (j, d) in dst.iter_mut().enumerate() {
            *d = match tb {
                Trans::N => b.get(p, j),
                Trans::T => b.get(j, p),
            };
        }
    }
    bp
}

/// The microkernel's register block: `GEMM_MR` rows × `GEMM_NR` vectors of columns, 12 independent
/// accumulator chains (enough to cover FMA latency on two pipes). With the `B` vectors and the
/// broadcast `A` element it fits AVX2's 16 vector registers.
#[cfg(feature = "alloc")]
const GEMM_MR: usize = 6;
#[cfg(feature = "alloc")]
const _: () = assert!(GEMM_MR == 6, "solve_tile unrolls GEMM_MR by hand");
#[cfg(feature = "alloc")]
const GEMM_NR: usize = 2;
/// Depth of the contraction slice a packed `B` panel covers per pass, so the panel stays in L1
/// while the rows of `A` stream past it.
#[cfg(feature = "alloc")]
const GEMM_KC: usize = 256;
/// `A` panels (of `GEMM_MR` rows) swept against each `B` panel before moving on: 72 rows × one
/// `GEMM_KC` slice is 147 KB of `f64`, which leaves half of a 512 KB L2 for the `B` panels
/// streaming past. 120 rows was 1–3% slower on Zen 3 for both `f32` and `f64`.
#[cfg(feature = "alloc")]
const GEMM_MC_BLOCKS: usize = 12;

#[cfg(feature = "alloc")]
/// `C := α·op(A)·op(B) + β·C` computed in `T`, from inputs of `U` that widen to `T` as they are
/// packed: `U` is `T`, or a narrower float whose `Compute` is `T`.
struct GemmK<'a, T: FloatScalar, U> {
    ta: Trans,
    tb: Trans,
    a: Mat<'a, U>,
    b: Mat<'a, U>,
    alpha: T,
    beta: T,
    c: MatMut<'a, T>,
    m: usize,
    n: usize,
    k: usize,
    /// Only this triangle of `C` (square) is computed and written.
    tri: Option<Uplo>,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar, U: FloatScalar> Kernel<T> for GemmK<'a, T, U> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let GemmK { ta, tb, a, b, alpha, beta, mut c, m, n, k, tri } = self;
        let l = simd.lanes::<T>();
        let width = GEMM_NR * l;
        let panels = n.div_ceil(width);
        assert!(core::any::TypeId::of::<U::Compute>() == core::any::TypeId::of::<T>());
        // SAFETY: `U::Compute` is `T` (asserted above).
        let widen = |x: U| unsafe { core::mem::transmute_copy::<U::Compute, T>(&x.widen()) };
        // `op(A)[i][p]` sits at `i * sr + p * sp` in `a.data`.
        let (sr, sp) = match (ta, a.layout) {
            (Trans::N, Layout::RowMajor) | (Trans::T, Layout::ColMajor) => (a.stride, 1),
            _ => (1, a.stride),
        };
        assert!((m - 1) * sr + (k - 1) * sp < a.data.len(), "gemm: A is shorter than its shape");
        // `op(B)` rows `k0..k1` in panels of `width` columns, each `k1 - k0` contiguous rows of
        // `width` values (zero-padded past `n`), so a microkernel step loads its `GEMM_NR` vectors
        // from one line. One contraction slice at a time keeps the buffer at `GEMM_KC` rows
        // however deep `k` is.
        let kc = GEMM_KC.min(k);
        let mut bp = Scratch::<T>::new(panels * kc * width);
        // Row `p` of `op(B)` is contiguous in `b.data` from `p * stride`.
        let b_rows = matches!((tb, b.layout), (Trans::N, Layout::RowMajor) | (Trans::T, Layout::ColMajor));
        let scale = GemmScale { alpha: simd.splat(alpha), beta: simd.splat(beta), beta_nz: beta != T::ZERO, tri };
        // `op(A)` for one contraction slice in panels of `GEMM_MR` rows, each step's rows
        // contiguous (zero rows past `m`), so the microkernel broadcasts from one cache line.
        let blocks = m.div_ceil(GEMM_MR);
        let mut ap = Scratch::<T>::new(blocks * kc * GEMM_MR);
        let mut k0 = 0;
        while k0 < k {
            let k1 = (k0 + GEMM_KC).min(k);
            // Along whichever lines of `B` are contiguous: rows of `op(B)`, or its columns, which
            // then sit at `j * stride` and scatter into one panel that stays in L1.
            for q in 0..panels {
                let (j0, w) = (q * width, (n - q * width).min(width));
                let panel = &mut bp[q * kc * width..(q * kc + k1 - k0) * width];
                if b_rows {
                    for (p, row) in (k0..k1).zip(panel.chunks_exact_mut(width)) {
                        for (d, &x) in row.iter_mut().zip(&b.data[p * b.stride + j0..][..w]) {
                            *d = widen(x);
                        }
                    }
                } else {
                    for x in 0..w {
                        let col = &b.data[(j0 + x) * b.stride + k0..(j0 + x) * b.stride + k1];
                        for (row, &v) in panel.chunks_exact_mut(width).zip(col) {
                            row[x] = widen(v);
                        }
                    }
                }
                // Padding lanes are never stored, but a stale denormal there would slow every
                // multiply-add that reads it on cores that assist denormals in microcode.
                if w < width {
                    for row in panel.chunks_exact_mut(width) {
                        row[w..].fill(T::ZERO);
                    }
                }
            }
            // Each panel walks `A` along its contiguous direction: its `GEMM_MR` rows of `op(A)`
            // side by side when rows run along the contraction, else one step's `GEMM_MR`
            // adjacent values.
            for ib in 0..blocks {
                let panel = &mut ap[ib * kc * GEMM_MR..(ib * kc + k1 - k0) * GEMM_MR];
                let rows = (m - ib * GEMM_MR).min(GEMM_MR);
                if sp == 1 && rows == GEMM_MR {
                    let lines: [&[U]; GEMM_MR] = core::array::from_fn(|r| {
                        let at = (ib * GEMM_MR + r) * sr;
                        &a.data[at + k0..at + k1]
                    });
                    for (s, step) in panel.as_chunks_mut::<GEMM_MR>().0.iter_mut().enumerate() {
                        for (d, line) in step.iter_mut().zip(&lines) {
                            *d = widen(line[s]);
                        }
                    }
                } else if sp == 1 {
                    for r in 0..rows {
                        let line = &a.data[(ib * GEMM_MR + r) * sr + k0..(ib * GEMM_MR + r) * sr + k1];
                        for (step, &x) in panel.as_chunks_mut::<GEMM_MR>().0.iter_mut().zip(line) {
                            step[r] = widen(x);
                        }
                    }
                } else {
                    for (step, p) in panel.as_chunks_mut::<GEMM_MR>().0.iter_mut().zip(k0..k1) {
                        let at = ib * GEMM_MR + p * sp;
                        for (d, &x) in step.iter_mut().zip(&a.data[at..at + rows]) {
                            *d = widen(x);
                        }
                    }
                }
                if rows < GEMM_MR {
                    for step in panel.as_chunks_mut::<GEMM_MR>().0.iter_mut() {
                        step[rows..].fill(T::ZERO);
                    }
                }
            }
            for chunk in (0..blocks).step_by(GEMM_MC_BLOCKS) {
                for q in 0..panels {
                    let b_panel = &bp[q * kc * width..(q + 1) * kc * width];
                    for ib in chunk..(chunk + GEMM_MC_BLOCKS).min(blocks) {
                        let a_panel = &ap[ib * kc * GEMM_MR..(ib + 1) * kc * GEMM_MR];
                        let rows = (m - ib * GEMM_MR).min(GEMM_MR);
                        let outside = match tri {
                            Some(Uplo::Lower) => q * width >= ib * GEMM_MR + rows,
                            Some(Uplo::Upper) => (q + 1) * width <= ib * GEMM_MR,
                            None => false,
                        };
                        if outside {
                            continue;
                        }
                        gemm_block(simd, a_panel, b_panel, &mut c, ib * GEMM_MR, rows, q * width, k0, k1, n, &scale);
                    }
                }
            }
            k0 = k1;
        }
    }
}

#[cfg(feature = "alloc")]
struct GemmScale<V> {
    alpha: V,
    beta: V,
    beta_nz: bool,
    tri: Option<Uplo>,
}

/// One `GEMM_MR`-row panel of packed `A` against one packed `B` panel, both holding contraction
/// steps `k0..k1`, writing the first `rows` rows of `C` from row `i`: the first slice applies `α` and
/// `β`, later ones accumulate onto `C`.
#[cfg(feature = "alloc")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gemm_block<T: FloatScalar, S: Backend<T>>(
    simd: Gang<S>,
    a_panel: &[T],
    b_panel: &[T],
    c: &mut MatMut<T>,
    i: usize,
    rows: usize,
    j0: usize,
    k0: usize,
    k1: usize,
    n: usize,
    scale: &GemmScale<crate::Varying<T, S>>,
) {
    let l = simd.lanes::<T>();
    let width = GEMM_NR * l;
    let zero = simd.splat(T::ZERO);
    let mut acc = [[zero; GEMM_NR]; GEMM_MR];
    let b_steps = b_panel[..(k1 - k0) * width].chunks_exact(width);
    let a_steps = a_panel[..(k1 - k0) * GEMM_MR].as_chunks::<GEMM_MR>().0;
    for (row, a_step) in b_steps.zip(a_steps) {
        let bv = [simd.load(&row[..l]), simd.load(&row[l..])];
        for (acc_r, &a) in acc.iter_mut().zip(a_step) {
            let av = simd.splat(a);
            for (acc_rj, &bj) in acc_r.iter_mut().zip(&bv) {
                *acc_rj = av.fma(bj, *acc_rj);
            }
        }
    }
    gemm_store(simd, &acc, c, i, rows, j0, n, k0 > 0, scale);
}

/// Write a register block to `C` rows `i..i + rows`, columns `j0..n_end`: `α·acc` added to `C`
/// when `accumulate`, else to `β·C`. With `scale.tri` set, only that triangle's entries change.
#[cfg(feature = "alloc")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gemm_store<T: FloatScalar, S: Backend<T>, const NV: usize>(
    simd: Gang<S>,
    acc: &[[crate::Varying<T, S>; NV]; GEMM_MR],
    c: &mut MatMut<T>,
    i: usize,
    rows: usize,
    j0: usize,
    n_end: usize,
    accumulate: bool,
    scale: &GemmScale<crate::Varying<T, S>>,
) {
    let l = simd.lanes::<T>();
    let zero = simd.splat(T::ZERO);
    let one = simd.splat(T::ONE);
    for (r, acc_r) in acc.iter().enumerate().take(rows) {
        let row = i + r;
        for (jj, accv) in acc_r.iter().enumerate() {
            let c0 = j0 + jj * l;
            if c0 >= n_end {
                break;
            }
            let w = (n_end - c0).min(l);
            let (keep_lo, keep_hi) = match scale.tri {
                Some(Uplo::Lower) => (0, (row + 1).saturating_sub(c0).min(w)),
                Some(Uplo::Upper) => (row.saturating_sub(c0).min(w), w),
                None => (0, w),
            };
            if keep_lo >= keep_hi {
                continue;
            }
            let base = row * c.stride + c0;
            let crow = &mut c.data[base..base + w];
            let load = |crow: &[T]| if w == l { simd.load(crow) } else { simd.load_partial(crow, T::ZERO) };
            let mut out = accv.fma(scale.alpha, zero);
            if accumulate {
                out = load(crow).fma(one, out);
            } else if scale.beta_nz {
                out = load(crow).fma(scale.beta, out);
            }
            if keep_hi - keep_lo < w {
                let mut tmp = [T::ZERO; crate::MAX_LANES];
                out.store(&mut tmp[..l]);
                crow[keep_lo..keep_hi].copy_from_slice(&tmp[keep_lo..keep_hi]);
            } else if w == l {
                out.store(crow);
            } else {
                out.store_partial(crow);
            }
        }
    }
}

/// Whether a product skips packing. Reading the operands in place beats packing them while each
/// is a fraction of L2: on Zen 3 the crossover sits where `op(B)` passes about 200 KB (136² `f64`,
/// 240² `f32`), with the unpacked path ahead by 10–20% just below it and 4× at 16³. `op(A)` is
/// bounded too, since without the packed path's row blocking a tall `A` is streamed again for
/// every column strip.
#[cfg(all(feature = "alloc", not(all(target_vendor = "apple", not(hp_no_apple_accelerate)))))]
#[inline]
fn gemm_direct_fits(m: usize, n: usize, k: usize, size: usize) -> bool {
    const OPERAND_BYTES: usize = 192 << 10;
    k * n * size <= OPERAND_BYTES && m * k * size <= OPERAND_BYTES
}

/// Accelerate's AMX path overtakes the unpacked one from about 32³.
#[cfg(all(feature = "alloc", target_vendor = "apple", not(hp_no_apple_accelerate)))]
#[inline]
fn gemm_direct_fits(m: usize, n: usize, k: usize, size: usize) -> bool {
    m * n * k * size <= 1 << 14
}

/// [`GemmK`] without the packing, for small products whose `op(B)` rows are contiguous: each
/// register block broadcasts `A` straight from the operand and loads `B` rows in place.
#[cfg(feature = "alloc")]
struct GemmDirectK<'a, T> {
    ta: Trans,
    a: Mat<'a, T>,
    b: Mat<'a, T>,
    alpha: T,
    beta: T,
    c: MatMut<'a, T>,
    m: usize,
    n: usize,
    k: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for GemmDirectK<'a, T> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let GemmDirectK { ta, a, b, alpha, beta, mut c, m, n, k } = self;
        let l = simd.lanes::<T>();
        let scale = GemmScale { alpha: simd.splat(alpha), beta: simd.splat(beta), beta_nz: beta != T::ZERO, tri: None };
        // `op(A)[i][p]` sits at `i * sr + p * sp` in `a.data`.
        let (sr, sp) = match (ta, a.layout) {
            (Trans::N, Layout::RowMajor) | (Trans::T, Layout::ColMajor) => (a.stride, 1),
            _ => (1, a.stride),
        };
        assert!((m - 1) * sr + (k - 1) * sp < a.data.len(), "gemm: A is shorter than its shape");
        let width = GEMM_NR * l;
        let mut strip = (m > GEMM_MR && strip_aliases(b.stride * size_of::<T>(), k)).then(|| Scratch::<T>::new(k * width));
        let mut j0 = 0;
        while j0 < n {
            let w = (n - j0).min(width);
            let (bs, b_stride, b_col) = match strip.as_deref_mut() {
                Some(strip) => {
                    for (p, row) in strip.chunks_exact_mut(width).enumerate() {
                        row[..w].copy_from_slice(&b.data[p * b.stride + j0..p * b.stride + j0 + w]);
                    }
                    (&*strip, width, 0)
                }
                None => (b.data, b.stride, j0),
            };
            let mut i = 0;
            while i < m {
                let rows: [usize; GEMM_MR] = core::array::from_fn(|r| (i + r).min(m - 1) * sr);
                let at = DirectAt { a: a.data, rows, sp, b: bs, b_stride, b_col };
                if w > l {
                    gemm_direct_block::<T, S, GEMM_NR>(simd, &at, &mut c, i, m, j0, w, k, &scale);
                } else {
                    gemm_direct_block::<T, S, 1>(simd, &at, &mut c, i, m, j0, w, k, &scale);
                }
                i += GEMM_MR;
            }
            j0 += w;
        }
    }
}

/// Where a [`GemmDirectK`] block reads its operands: `op(A)` row `r` of the block at
/// `a[rows[r] + p * sp]`, and the block's columns of `op(B)` row `p` from `b[p * b_stride + b_col]`.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy)]
struct DirectAt<'a, T> {
    a: &'a [T],
    rows: [usize; GEMM_MR],
    sp: usize,
    b: &'a [T],
    b_stride: usize,
    b_col: usize,
}

/// Whether `k` rows `stride_bytes` apart, one cache line each, overflow the L1 sets they land in
/// (64 sets of 8 ways). A power-of-two stride reaches only a few sets: at 1 KB the 128 lines of a
/// 128-step `f64` strip share four, and reading `B` in place ran 30% slower than packing it.
#[cfg(feature = "alloc")]
#[inline]
fn strip_aliases(stride_bytes: usize, k: usize) -> bool {
    const LINE: usize = 64;
    const SETS: usize = 64;
    const WAYS: usize = 8;
    if !stride_bytes.is_multiple_of(LINE) {
        return false;
    }
    let step = (stride_bytes / LINE) % SETS;
    let reached = if step == 0 { 1 } else { SETS >> step.trailing_zeros() };
    k > reached * WAYS
}

/// One `GEMM_MR × NV`-register block of [`GemmDirectK`] at row `i`, columns `j0..j0 + w` of `C`.
/// Rows past `m` repeat the last row so the block stays fully unrolled; their results are not
/// stored.
#[cfg(feature = "alloc")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gemm_direct_block<T: FloatScalar, S: Backend<T>, const NV: usize>(
    simd: Gang<S>,
    at: &DirectAt<T>,
    c: &mut MatMut<T>,
    i: usize,
    m: usize,
    j0: usize,
    w: usize,
    k: usize,
    scale: &GemmScale<crate::Varying<T, S>>,
) {
    let l = simd.lanes::<T>();
    let zero = simd.splat(T::ZERO);
    let DirectAt { a, rows, sp, b, b_stride, b_col } = *at;
    let mut acc = [[zero; NV]; GEMM_MR];
    assert!((k - 1) * b_stride + b_col + w <= b.len(), "gemm: B is shorter than its shape");
    // Two copies of the loop, so the full-width one carries no tail test: LLVM keeps a
    // loop-invariant `w` check inside the loop otherwise.
    macro_rules! steps {
        ($row:ident => $load:expr) => {{
            let mut $row = b_col;
            for p in 0..k {
                let bv: [crate::Varying<T, S>; NV] = $load;
                for (acc_r, &ro) in acc.iter_mut().zip(&rows) {
                    // SAFETY: the caller checked the last row and step are inside `a`.
                    let av = simd.splat(unsafe { *a.get_unchecked(ro + p * sp) });
                    for (x, &bj) in acc_r.iter_mut().zip(&bv) {
                        *x = av.fma(bj, *x);
                    }
                }
                $row += b_stride;
            }
        }};
    }
    if w == NV * l {
        // SAFETY: rows `< k` and columns `j0..j0 + w` are inside `b`, checked above.
        steps!(at => core::array::from_fn(|jj| simd.load(unsafe { b.get_unchecked(at + jj * l..at + jj * l + l) })));
    } else {
        steps!(at => {
            let row = &b[at..at + w];
            core::array::from_fn(|jj| {
                let lo = jj * l;
                if lo + l <= w { simd.load(&row[lo..lo + l]) } else { simd.load_partial(&row[lo..], T::ZERO) }
            })
        });
    }
    gemm_store(simd, &acc, c, i, (m - i).min(GEMM_MR), j0, j0 + w, false, scale);
}

/// The SME dispatch predicate: aarch64, `std` (thread-local pack buffers), and Accelerate out of
/// the way. Same gate as `matrix`'s tile GEMM.
macro_rules! sme_item {
    ($item:item) => {
        #[cfg(all(
            target_arch = "aarch64",
            feature = "std",
            not(hp_no_sme),
            any(not(target_vendor = "apple"), hp_no_apple_accelerate)
        ))]
        $item
    };
}

sme_item! {
    /// The ZA-grid tile width (`svl/2` for f32, `svl/4` for f64) and the set it must land in.
    fn sme_blk<T: FloatScalar>(svl: usize) -> Option<usize> {
        use core::any::TypeId;
        let t = TypeId::of::<T>();
        if t == TypeId::of::<f32>() {
            let blk = svl / 2;
            (blk == 16 || blk == 32 || blk == 64).then_some(blk)
        } else if t == TypeId::of::<f64>() {
            let blk = svl / 4;
            (blk == 8 || blk == 16 || blk == 32).then_some(blk)
        } else {
            None
        }
    }
}

sme_item! {
    /// `C += pack(α·op(A)) · pack(op(B))` on the ZA grid, both operands row-major (`ac` is `m×k`,
    /// `bc` is `k×n`), `c` row-major with element stride `ldc`. The grid preloads and stores `C`, so
    /// this accumulates.
    ///
    /// # Safety
    /// `blk` divides `m` and `n` and equals the runtime ZA-grid width; `ac.len() >= m·k`,
    /// `bc.len() >= k·n`; `c` addresses an `m×n` region with row stride `ldc`; `T` is `f32`/`f64`.
    unsafe fn sme_mma<T: FloatScalar>(ac: &[T], bc: &[T], c: *mut T, ldc: usize, m: usize, n: usize, k: usize, blk: usize) {
        use crate::matrix::sme_pack;
        use core::any::TypeId;
        let (pm, pn) = (m / blk, n / blk);
        let ldc_b = ldc * core::mem::size_of::<T>();
        let need = m * k + k * n;
        let t = TypeId::of::<T>();
        if t == TypeId::of::<f32>() {
            sme_pack::F32.with_borrow_mut(|buf| {
                if buf.len() < need {
                    buf.resize(need, 0.0);
                }
                let (ap, bpk) = buf.split_at_mut(m * k);
                unsafe {
                    sme_pack::pack_a_f32(ac.as_ptr() as *const f32, ap, pm, k, blk);
                    sme_pack::pack_b(bc.as_ptr() as *const f32, &mut bpk[..k * n], pn, k, n, blk);
                    crate::arch::sme2::mma_f32_grid_packed(ap.as_ptr(), bpk.as_ptr(), c as *mut f32, ldc_b, pm, pn, k);
                }
            });
        } else if t == TypeId::of::<f64>() {
            sme_pack::F64.with_borrow_mut(|buf| {
                if buf.len() < need {
                    buf.resize(need, 0.0);
                }
                let (ap, bpk) = buf.split_at_mut(m * k);
                unsafe {
                    sme_pack::pack_a_f64(ac.as_ptr() as *const f64, ap, pm, k, blk);
                    sme_pack::pack_b(bc.as_ptr() as *const f64, &mut bpk[..k * n], pn, k, n, blk);
                    crate::arch::sme2::mma_f64_grid_packed(ap.as_ptr(), bpk.as_ptr(), c as *mut f64, ldc_b, pm, pn, k);
                }
            });
        }
    }
}

sme_item! {
    /// Materialize `α·op(A)` as a contiguous row-major `m×k` panel, the scalar folded in so the
    /// grid's `C += A·B` yields `C += α·op(A)·B`.
    fn pack_op_a_scaled<T: FloatScalar>(ta: Trans, a: &Mat<T>, alpha: T, m: usize, k: usize) -> alloc::vec::Vec<T> {
        let mut ac = alloc::vec![T::ZERO; m * k];
        for i in 0..m {
            let dst = &mut ac[i * k..i * k + k];
            for (p, d) in dst.iter_mut().enumerate() {
                let v = match ta {
                    Trans::N => a.get(i, p),
                    Trans::T => a.get(p, i),
                };
                *d = alpha.wmul(v);
            }
        }
        ac
    }
}

sme_item! {
    /// Route a large, `blk`-aligned GEMM to the ZA engine: β-scale `C` in place, then accumulate
    /// `α·op(A)·op(B)`. Returns `false` (leaving `C` untouched) when the shape or host isn't a fit,
    /// so the caller falls through to the gang kernel. `c` is row-major (col-major is normalized away
    /// upstream) and `bp` is `op(B)` packed row-major `k×n`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_sme<T: FloatScalar>(alpha: T, beta: T, ta: Trans, a: Mat<T>, bp: &[T], c: &mut MatMut<T>, m: usize, n: usize, k: usize) -> bool {
        if m < SME_MIN || n < SME_MIN || k < SME_MIN || !crate::arch::sme1::is_supported() || !crate::arch::sme2::is_supported() {
            return false;
        }
        let svl = crate::arch::sme1::streaming_vl_bytes();
        let Some(blk) = sme_blk::<T>(svl) else { return false };
        if !m.is_multiple_of(blk) || !n.is_multiple_of(blk) {
            return false;
        }
        let ac = pack_op_a_scaled(ta, &a, alpha, m, k);
        beta_scale_block(c, beta, m, n);
        unsafe {
            sme_mma(&ac, bp, c.data.as_mut_ptr(), c.stride, m, n, k, blk);
        }
        true
    }
}

sme_item! {
    /// In-place `C[..m, ..n] := β·C`, row-major, with the `β ∈ {0, 1}` fast exits.
    fn beta_scale_block<T: FloatScalar>(c: &mut MatMut<T>, beta: T, m: usize, n: usize) {
        let b = beta.into_f64();
        if b == 1.0 {
            return;
        }
        for i in 0..m {
            let base = i * c.stride;
            let row = &mut c.data[base..base + n];
            if b == 0.0 {
                for v in row.iter_mut() {
                    *v = T::ZERO;
                }
            } else {
                for v in row.iter_mut() {
                    *v = beta.wmul(*v);
                }
            }
        }
    }
}

sme_item! {
    /// Route a large, `blk`-aligned SYRK to the ZA engine: form the full `α·op(A)·op(A)ᵀ` product
    /// in a scratch panel, then fold only the `uplo` triangle back with `β`. Trades the
    /// dot-triangle's half-flop saving for the matrix engine's throughput.
    #[allow(clippy::too_many_arguments)]
    fn syrk_sme<T: FloatScalar>(uplo: Uplo, trans: Trans, alpha: T, a: Mat<T>, beta: T, c: &mut MatMut<T>, n: usize, k: usize) -> bool {
        if n < SME_MIN || k < SME_MIN || !crate::arch::sme1::is_supported() || !crate::arch::sme2::is_supported() {
            return false;
        }
        let svl = crate::arch::sme1::streaming_vl_bytes();
        let Some(blk) = sme_blk::<T>(svl) else { return false };
        if !n.is_multiple_of(blk) {
            return false;
        }
        let ac = pack_op_a_scaled(trans, &a, alpha, n, k);
        // op(A)ᵀ row-major k×n: entry (p, j) is op(A)[j][p].
        let mut bt = alloc::vec![T::ZERO; k * n];
        for p in 0..k {
            let dst = &mut bt[p * n..p * n + n];
            for (j, d) in dst.iter_mut().enumerate() {
                *d = match trans {
                    Trans::N => a.get(j, p),
                    Trans::T => a.get(p, j),
                };
            }
        }
        let mut prod = alloc::vec![T::ZERO; n * n];
        unsafe {
            sme_mma(&ac, &bt, prod.as_mut_ptr(), n, n, n, k, blk);
        }
        let zero_beta = beta.into_f64() == 0.0;
        for i in 0..n {
            let (lo, hi) = match uplo {
                Uplo::Lower => (0, i + 1),
                Uplo::Upper => (i, n),
            };
            for j in lo..hi {
                let p = prod[i * n + j];
                let o = c.off(i, j);
                c.data[o] = if zero_beta { p } else { beta.wmul(c.data[o]).wadd(p) };
            }
        }
        true
    }
}

/// Minimum dimension for the ZA matrix engine to amortize its streaming-mode entry; mirrors
/// `matrix`'s `SME_MIN_DIM`.
#[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(hp_no_sme),
    any(not(target_vendor = "apple"), hp_no_apple_accelerate)
))]
const SME_MIN: usize = 16;

/// Symmetric rank-`k` update `C := α·op(A)·op(A)ᵀ + β·C`, writing only the `uplo` triangle of
/// the symmetric `n×n` result: half the flops of the equivalent `gemm`. `op(A)` is `A`
/// (`Trans::N`, `A` is `n×k`) or `Aᵀ` (`Trans::T`, `A` is `k×n`). Apple routes to `cblas_*syrk`;
/// elsewhere it is the GEMM microkernel skipping the register blocks outside the triangle.
#[cfg(feature = "alloc")]
pub fn syrk<T: FloatScalar + SimdDispatch>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: Mat<T>,
    beta: T,
    c: MatMut<T>,
) {
    let (n, k) = match trans {
        Trans::N => (a.rows, a.cols),
        Trans::T => (a.cols, a.rows),
    };
    assert_eq!(c.rows, n, "syrk: C rows mismatch");
    assert_eq!(c.cols, n, "syrk: C must be square (n×n)");
    if n == 0 {
        return;
    }

    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if accel_syrk(uplo, trans, alpha, a, beta, &c, n, k) {
        return;
    }

    #[cfg(all(
        target_arch = "aarch64",
        feature = "std",
        not(hp_no_sme),
        any(not(target_vendor = "apple"), hp_no_apple_accelerate)
    ))]
    let mut c = c;
    #[cfg(all(
        target_arch = "aarch64",
        feature = "std",
        not(hp_no_sme),
        any(not(target_vendor = "apple"), hp_no_apple_accelerate)
    ))]
    if syrk_sme(uplo, trans, alpha, a, beta, &mut c, n, k) {
        return;
    }

    if k > 0 && size_of::<T>() == size_of::<T::Compute>() {
        let (c, tri) = match c.layout {
            Layout::RowMajor => (c, uplo),
            // The same buffer read row-major is `Cᵀ`, whose `uplo` triangle is `C`'s other one.
            Layout::ColMajor => (MatMut::strided(c.data, n, n, c.stride, Layout::RowMajor), flip_uplo(uplo)),
        };
        dispatch::<T, _>(GemmK::<T, T> { ta: trans, tb: flip(trans), a, b: a, alpha, beta, c, m: n, n, k, tri: Some(tri) });
        return;
    }
    let ap = pack_syrk(trans, &a, n, k);
    dispatch::<T, _>(SyrkK { uplo, alpha, beta, ap: &ap, c, n, k });
}

#[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate), feature = "alloc"))]
#[allow(clippy::too_many_arguments)]
fn accel_syrk<T: FloatScalar>(
    uplo: Uplo,
    trans: Trans,
    alpha: T,
    a: Mat<T>,
    beta: T,
    c: &MatMut<T>,
    n: usize,
    k: usize,
) -> bool {
    use crate::matrix::accel;
    use core::any::TypeId;
    use core::mem::transmute_copy;

    let order = match c.layout {
        Layout::RowMajor => accel::ROW_MAJOR,
        Layout::ColMajor => accel::COL_MAJOR,
    };
    let up = match uplo {
        Uplo::Upper => accel::UPPER,
        Uplo::Lower => accel::LOWER,
    };
    let logical_t = trans == Trans::T;
    let stored_t = a.layout != c.layout;
    let tr = if logical_t ^ stored_t { accel::TRANS } else { accel::NO_TRANS };
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        unsafe {
            accel::cblas_ssyrk(
                order, up, tr, n as _, k as _,
                transmute_copy::<T, f32>(&alpha), a.data.as_ptr() as *const f32, a.stride as _,
                transmute_copy::<T, f32>(&beta), c.data.as_ptr() as *mut f32, c.stride as _,
            );
        }
        return true;
    }
    if t == TypeId::of::<f64>() {
        unsafe {
            accel::cblas_dsyrk(
                order, up, tr, n as _, k as _,
                transmute_copy::<T, f64>(&alpha), a.data.as_ptr() as *const f64, a.stride as _,
                transmute_copy::<T, f64>(&beta), c.data.as_ptr() as *mut f64, c.stride as _,
            );
        }
        return true;
    }
    false
}

#[cfg(feature = "alloc")]
fn pack_syrk<T: FloatScalar>(trans: Trans, a: &Mat<T>, n: usize, k: usize) -> alloc::vec::Vec<T> {
    let mut ap = alloc::vec![T::ZERO; n * k];
    for i in 0..n {
        let dst = &mut ap[i * k..i * k + k];
        for (p, d) in dst.iter_mut().enumerate() {
            *d = match trans {
                Trans::N => a.get(i, p),
                Trans::T => a.get(p, i),
            };
        }
    }
    ap
}

#[cfg(feature = "alloc")]
struct SyrkK<'a, T: FloatScalar> {
    uplo: Uplo,
    alpha: T,
    beta: T,
    ap: &'a [T],
    c: MatMut<'a, T>,
    n: usize,
    k: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for SyrkK<'a, T> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let SyrkK { uplo, alpha, beta, ap, c, n, k } = self;
        let off = |i: usize, j: usize| match c.layout {
            Layout::RowMajor => i * c.stride + j,
            Layout::ColMajor => j * c.stride + i,
        };
        for i in 0..n {
            let (lo, hi) = match uplo {
                Uplo::Lower => (0, i + 1),
                Uplo::Upper => (i, n),
            };
            let ai = &ap[i * k..i * k + k];
            for j in lo..hi {
                let d = simd.dot(ai, &ap[j * k..j * k + k]);
                let o = off(i, j);
                c.data[o] = fma_scalar(alpha, d, beta, c.data[o]);
            }
        }
    }
}

#[cfg(feature = "alloc")]
#[inline]
fn flip_layout(l: Layout) -> Layout {
    match l {
        Layout::RowMajor => Layout::ColMajor,
        Layout::ColMajor => Layout::RowMajor,
    }
}

/// Triangular solve `op(A)·X = α·B` (`Side::Left`) or `X·op(A) = α·B` (`Side::Right`), `X`
/// overwriting `B`. `uplo` picks `A`'s stored triangle; `Diag::Unit` takes the diagonal as an
/// implicit 1. Apple with matching operand layouts routes to `cblas_*trsm`; the fallback reduces
/// a right-side solve to a left-side one on `Bᵀ`, packs the RHS row-major, and runs substitution
/// whose per-step scale/axpy vectorize across the free dimension.
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: FloatScalar + SimdDispatch>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: T,
    a: Mat<T>,
    b: MatMut<T>,
) {
    let q = match side {
        Side::Left => b.rows,
        Side::Right => b.cols,
    };
    assert_eq!(a.rows, q, "trsm: A must be square, matching B");
    assert_eq!(a.cols, q, "trsm: A must be square, matching B");
    if b.rows == 0 || b.cols == 0 {
        return;
    }

    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    let mut b = b;
    // Accelerate's solve against a single right-hand side is slower than the substitution
    // kernel from about 64 rows (3.4x at 512).
    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    let one_rhs = match side {
        Side::Left => b.cols == 1,
        Side::Right => b.rows == 1,
    };
    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if !(one_rhs && q >= 64) && accel_trsm(side, uplo, trans, diag, alpha, a, &mut b) {
        return;
    }

    match side {
        Side::Left => solve_left(uplo, trans, diag, alpha, a, b),
        Side::Right => {
            // X·op(A) = α·B  ⟺  op(A)ᵀ·Xᵀ = α·Bᵀ; the same buffer read with swapped dims and
            // flipped layout is Bᵀ, and A's triangle (uplo) is intrinsic.
            let bt = MatMut::strided(b.data, b.cols, b.rows, b.stride, flip_layout(b.layout));
            solve_left(uplo, flip(trans), diag, alpha, a, bt);
        }
    }
}

#[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate), feature = "alloc"))]
#[allow(clippy::too_many_arguments)]
fn accel_trsm<T: FloatScalar>(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: T,
    a: Mat<T>,
    b: &mut MatMut<T>,
) -> bool {
    use crate::matrix::accel;
    use core::any::TypeId;
    use core::mem::transmute_copy;

    // Folding a mismatched A layout into cblas would also flip its triangle, so take the native
    // path only when A and B share an order.
    if a.layout != b.layout {
        return false;
    }
    let order = match b.layout {
        Layout::RowMajor => accel::ROW_MAJOR,
        Layout::ColMajor => accel::COL_MAJOR,
    };
    let sd = match side {
        Side::Left => accel::LEFT,
        Side::Right => accel::RIGHT,
    };
    let up = match uplo {
        Uplo::Upper => accel::UPPER,
        Uplo::Lower => accel::LOWER,
    };
    let tr = match trans {
        Trans::N => accel::NO_TRANS,
        Trans::T => accel::TRANS,
    };
    let dg = match diag {
        Diag::Unit => accel::UNIT,
        Diag::NonUnit => accel::NON_UNIT,
    };
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        unsafe {
            accel::cblas_strsm(
                order, sd, up, tr, dg, b.rows as _, b.cols as _,
                transmute_copy::<T, f32>(&alpha), a.data.as_ptr() as *const f32, a.stride as _,
                b.data.as_mut_ptr() as *mut f32, b.stride as _,
            );
        }
        return true;
    }
    if t == TypeId::of::<f64>() {
        unsafe {
            accel::cblas_dtrsm(
                order, sd, up, tr, dg, b.rows as _, b.cols as _,
                transmute_copy::<T, f64>(&alpha), a.data.as_ptr() as *const f64, a.stride as _,
                b.data.as_mut_ptr() as *mut f64, b.stride as _,
            );
        }
        return true;
    }
    false
}

#[cfg(feature = "alloc")]
fn solve_left<T: FloatScalar + SimdDispatch>(
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: T,
    a: Mat<T>,
    b: MatMut<T>,
) {
    let (m, n) = (b.rows, b.cols);
    // op(A) is lower-triangular for (Lower, N) and (Upper, T), solved forward; upper otherwise.
    let op_lower = (uplo == Uplo::Lower) == (trans == Trans::N);
    if b.layout == Layout::RowMajor && n > 1 {
        assert!((m - 1) * b.stride + n <= b.data.len(), "trsm: B is shorter than its shape");
        solve_rows(op_lower, trans, diag, alpha, a, b.data, m, n, b.stride);
        return;
    }
    let mut rp = Scratch::<T>::new(m * n);
    match b.layout {
        Layout::ColMajor => transpose(b.data, n, m, b.stride, &mut rp, n),
        Layout::RowMajor => {
            for i in 0..m {
                rp[i * n..i * n + n].copy_from_slice(&b.data[i * b.stride..i * b.stride + n]);
            }
        }
    }
    if n == 1 {
        dispatch::<T, _>(SolveVecK { a, alpha, diag, trans, op_lower, x: &mut rp });
    } else {
        solve_rows(op_lower, trans, diag, alpha, a, &mut rp, m, n, n);
    }
    match b.layout {
        Layout::ColMajor => transpose(&rp, m, n, n, b.data, b.stride),
        Layout::RowMajor => {
            for i in 0..m {
                b.data[i * b.stride..i * b.stride + n].copy_from_slice(&rp[i * n..i * n + n]);
            }
        }
    }
}

/// `dst[j * dst_ld + i] = src[i * src_ld + j]` over `rows × cols`, in tiles of a cache line or so
/// a side, so neither matrix is strided across whole lines at a time.
#[cfg(feature = "alloc")]
fn transpose<T: Copy>(src: &[T], rows: usize, cols: usize, src_ld: usize, dst: &mut [T], dst_ld: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    assert!((rows - 1) * src_ld + cols <= src.len() && (cols - 1) * dst_ld + rows <= dst.len());
    const TILE: usize = 8;
    for i0 in (0..rows).step_by(TILE) {
        let i1 = (i0 + TILE).min(rows);
        for j0 in (0..cols).step_by(TILE) {
            for j in j0..(j0 + TILE).min(cols) {
                for i in i0..i1 {
                    // SAFETY: `i < rows` and `j < cols`, checked against both slices above.
                    unsafe { *dst.get_unchecked_mut(j * dst_ld + i) = *src.get_unchecked(i * src_ld + j) };
                }
            }
        }
    }
}

/// `op(A)·X = α·B` in place over the `m×n` row-major `B` whose row `i` starts at `x[i * ld]`.
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
fn solve_rows<T: FloatScalar + SimdDispatch>(
    op_lower: bool,
    trans: Trans,
    diag: Diag,
    alpha: T,
    a: Mat<T>,
    x: &mut [T],
    m: usize,
    n: usize,
    ld: usize,
) {
    if alpha != T::ONE {
        for i in 0..m {
            for v in &mut x[i * ld..i * ld + n] {
                *v = alpha.wmul(*v);
            }
        }
    }
    solve_split(op_lower, trans, diag, a, x, m, n, ld);
}

/// Halves the rows until a block fits [`SolveK`], so the updates between halves are GEMMs with
/// a deep contraction that each stream `B` once.
#[cfg(feature = "alloc")]
#[allow(clippy::too_many_arguments)]
fn solve_split<T: FloatScalar + SimdDispatch>(op_lower: bool, trans: Trans, diag: Diag, a: Mat<T>, x: &mut [T], m: usize, n: usize, ld: usize) {
    if m <= TRSM_NB {
        dispatch::<T, _>(SolveK { a, diag, trans, op_lower, rp: x, m, n, ld });
        return;
    }
    let h = m.div_ceil(2 * TRSM_NB) * TRSM_NB;
    let op_sub = |r0: usize, c0: usize, rows: usize, cols: usize| match trans {
        Trans::N => a.sub(r0, c0, rows, cols),
        Trans::T => a.sub(c0, r0, cols, rows),
    };
    let (top, bottom) = x.split_at_mut(h * ld);
    let (top_a, bottom_a) = (a.sub(0, 0, h, h), a.sub(h, h, m - h, m - h));
    if op_lower {
        solve_split(op_lower, trans, diag, top_a, top, h, n, ld);
        let solved = Mat::strided(top, h, n, ld, Layout::RowMajor);
        gemm(trans, Trans::N, Scalar::neg(T::ONE), op_sub(h, 0, m - h, h), solved, T::ONE, MatMut::strided(bottom, m - h, n, ld, Layout::RowMajor));
        solve_split(op_lower, trans, diag, bottom_a, bottom, m - h, n, ld);
    } else {
        solve_split(op_lower, trans, diag, bottom_a, bottom, m - h, n, ld);
        let solved = Mat::strided(bottom, m - h, n, ld, Layout::RowMajor);
        gemm(trans, Trans::N, Scalar::neg(T::ONE), op_sub(0, h, h, m - h), solved, T::ONE, MatMut::strided(top, h, n, ld, Layout::RowMajor));
        solve_split(op_lower, trans, diag, top_a, top, h, n, ld);
    }
}

#[cfg(feature = "alloc")]
const TRSM_NB: usize = 64;

/// A triangular solve against one right-hand side, `op(A)·x = α·b`, by whichever of the two
/// substitution orders walks `A` along its contiguous lines: row dot products when the operated
/// rows are contiguous, column updates when its columns are.
#[cfg(feature = "alloc")]
struct SolveVecK<'a, T: FloatScalar> {
    a: Mat<'a, T>,
    alpha: T,
    diag: Diag,
    trans: Trans,
    op_lower: bool,
    x: &'a mut [T],
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for SolveVecK<'a, T> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let SolveVecK { a, alpha, diag, trans, op_lower, x } = self;
        let m = x.len();
        // `op(A)[i][p]` sits at `i * sr + p * sp` in `a.data`.
        let (sr, sp) = match (trans, a.layout) {
            (Trans::N, Layout::RowMajor) | (Trans::T, Layout::ColMajor) => (a.stride, 1),
            _ => (1, a.stride),
        };
        let inv = |i: usize| if diag == Diag::Unit { T::ONE } else { T::from_f64(1.0 / a.data[i * (sr + sp)].into_f64()) };
        if alpha.into_f64() != 1.0 {
            for v in x.iter_mut() {
                *v = alpha.wmul(*v);
            }
        }
        match (op_lower, sp == 1) {
            (true, true) => {
                for i in 0..m {
                    let (done, rest) = x.split_at_mut(i);
                    let dot = simd.dot(&a.data[i * sr..i * sr + i], done);
                    rest[0] = rest[0].wsub(dot).wmul(inv(i));
                }
            }
            (false, true) => {
                for i in (0..m).rev() {
                    let (head, done) = x.split_at_mut(i + 1);
                    let dot = simd.dot(&a.data[i * sr + i + 1..i * sr + m], done);
                    head[i] = head[i].wsub(dot).wmul(inv(i));
                }
            }
            (true, false) => {
                for j in 0..m {
                    let (head, rest) = x.split_at_mut(j + 1);
                    head[j] = head[j].wmul(inv(j));
                    let c = simd.splat(Scalar::neg(head[j]));
                    simd.zip_map_inplace(&a.data[j * sp + j + 1..j * sp + m], rest, T::ZERO, T::ZERO, |col, r| col.fma(c, r));
                }
            }
            (false, false) => {
                for j in (0..m).rev() {
                    let (head, rest) = x.split_at_mut(j);
                    rest[0] = rest[0].wmul(inv(j));
                    let c = simd.splat(Scalar::neg(rest[0]));
                    simd.zip_map_inplace(&a.data[j * sp..j * sp + j], head, T::ZERO, T::ZERO, |col, r| col.fma(c, r));
                }
            }
        }
    }
}

#[cfg(feature = "alloc")]
/// `op(A)·X = B` for the `m×n` block of `B` whose row `i` starts at `rp[i * ld]`, in place.
struct SolveK<'a, T: FloatScalar> {
    a: Mat<'a, T>,
    diag: Diag,
    trans: Trans,
    op_lower: bool,
    rp: &'a mut [T],
    m: usize,
    n: usize,
    ld: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for SolveK<'a, T> {
    type Output = ();

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let SolveK { a, diag, trans, op_lower, rp, m, n, ld } = self;
        assert!(m <= TRSM_NB);
        // `op(A)[i][p]` sits at `i * sr + p * sp` in `a.data`.
        let (sr, sp) = match (trans, a.layout) {
            (Trans::N, Layout::RowMajor) | (Trans::T, Layout::ColMajor) => (a.stride, 1),
            _ => (1, a.stride),
        };
        assert!((m - 1) * (sr + sp) < a.data.len(), "trsm: A is shorter than its shape");
        assert!((m - 1) * ld + n <= rp.len());
        let at = |i: usize, p: usize| a.data[i * sr + p * sp];
        let l = simd.lanes::<T>();
        let groups = m.div_ceil(GEMM_MR);
        let mut packed = [T::ZERO; GEMM_MR * TRSM_NB];
        for gi in 0..groups {
            let g0 = if op_lower { gi } else { groups - 1 - gi } * GEMM_MR;
            let rows = (m - g0).min(GEMM_MR);
            // Rows past `m` repeat the last one so the tile stays fully unrolled; they are
            // never stored.
            let line = |r: usize| (g0 + r).min(m - 1);
            let solved = if op_lower { 0..g0 } else { g0 + rows..m };
            for (s, p) in solved.clone().enumerate() {
                for r in 0..GEMM_MR {
                    packed[s * GEMM_MR + r] = Scalar::neg(at(line(r), p));
                }
            }
            let mut tri = [[T::ZERO; GEMM_MR]; GEMM_MR];
            let mut inv = [T::ONE; GEMM_MR];
            for (r, (tri_r, inv_r)) in tri.iter_mut().zip(&mut inv).enumerate().take(rows) {
                for (q, t) in tri_r.iter_mut().enumerate().take(rows) {
                    *t = Scalar::neg(at(g0 + r, g0 + q));
                }
                if diag == Diag::NonUnit {
                    *inv_r = T::from_f64(1.0 / at(g0 + r, g0 + r).into_f64());
                }
            }
            let tile = SolveTile { packed: &packed[..solved.len() * GEMM_MR], first: solved.start, tri, inv, line: core::array::from_fn(line), g0, rows };
            let mut j0 = 0;
            while j0 < n {
                let w = (n - j0).min(GEMM_NR * l);
                match (w > l, op_lower) {
                    (true, true) => solve_tile::<T, S, GEMM_NR, true>(simd, &tile, rp, ld, j0, w),
                    (true, false) => solve_tile::<T, S, GEMM_NR, false>(simd, &tile, rp, ld, j0, w),
                    (false, true) => solve_tile::<T, S, 1, true>(simd, &tile, rp, ld, j0, w),
                    (false, false) => solve_tile::<T, S, 1, false>(simd, &tile, rp, ld, j0, w),
                }
                j0 += w;
            }
        }
    }
}

/// The `GEMM_MR` rows `g0..` of a [`SolveK`] block (the first `rows` real, the rest repeating
/// row `line[GEMM_MR - 1]`): `−op(A)` against the solved rows `first..` packed one step of
/// `GEMM_MR` values at a time, `−op(A)` on the diagonal block, and the reciprocal pivots.
#[cfg(feature = "alloc")]
struct SolveTile<'a, T> {
    packed: &'a [T],
    first: usize,
    tri: [[T; GEMM_MR]; GEMM_MR],
    inv: [T; GEMM_MR],
    line: [usize; GEMM_MR],
    g0: usize,
    rows: usize,
}

/// Solves the tile's rows over columns `j0..j0 + w` of `x` in registers: the solved rows are
/// subtracted as in a GEMM microkernel, then the triangle on the diagonal is substituted.
#[cfg(feature = "alloc")]
#[inline(always)]
fn solve_tile<T: FloatScalar, S: Backend<T>, const NV: usize, const LOWER: bool>(
    simd: Gang<S>,
    t: &SolveTile<T>,
    x: &mut [T],
    ld: usize,
    j0: usize,
    w: usize,
) {
    let l = simd.lanes::<T>();
    let load = |x: &[T], at: usize, lo: usize| {
        if lo + l <= w { simd.load(&x[at + lo..at + lo + l]) } else { simd.load_partial(&x[at + lo..at + w], T::ZERO) }
    };
    let mut acc: [[crate::Varying<T, S>; NV]; GEMM_MR] = core::array::from_fn(|r| core::array::from_fn(|v| load(x, t.line[r] * ld + j0, v * l)));
    for (s, step) in t.packed.as_chunks::<GEMM_MR>().0.iter().enumerate() {
        let at = (t.first + s) * ld + j0;
        let xv: [crate::Varying<T, S>; NV] = core::array::from_fn(|v| load(x, at, v * l));
        for (acc_r, &c) in acc.iter_mut().zip(step) {
            let c = simd.splat(c);
            for (acc_rv, &xvv) in acc_r.iter_mut().zip(&xv) {
                *acc_rv = c.fma(xvv, *acc_rv);
            }
        }
    }
    // Unrolled by hand: a loop over the rows here leaves `acc` indexed at run time, which keeps
    // it in memory through the loop above as well.
    crate::unroll!(6, k, {
        let r = if LOWER { k } else { GEMM_MR - 1 - k };
        if r < t.rows {
            crate::unroll!(6, q, {
                if q < t.rows && if LOWER { q < r } else { q > r } {
                    let c = simd.splat(t.tri[r][q]);
                    let done = acc[q];
                    for (acc_rv, d) in acc[r].iter_mut().zip(done) {
                        *acc_rv = c.fma(d, *acc_rv);
                    }
                }
            });
            for v in acc[r].iter_mut() {
                *v = *v * t.inv[r];
            }
        }
    });
    crate::unroll!(6, r, {
        if r < t.rows {
            let at = (t.g0 + r) * ld + j0;
            for (v, accv) in acc[r].iter().enumerate() {
                let lo = v * l;
                if lo + l <= w {
                    accv.store(&mut x[at + lo..at + lo + l]);
                } else if lo < w {
                    accv.store_partial(&mut x[at + lo..at + w]);
                }
            }
        }
    });
}

/// Cholesky factorization of a symmetric positive-definite `A`, in place: `A = L·Lᵀ` (`Uplo::Lower`)
/// or `A = Uᵀ·U` (`Uplo::Upper`), overwriting the `uplo` triangle with the factor. Returns
/// `Err(j)` if the leading minor of order `j+1` is not positive definite (a non-positive pivot),
/// with the factor's first `j` columns written. A triangle whose factor rows are not contiguous in
/// `A` is factored in a packed copy so the Crout dot-products stay vectorized.
#[cfg(feature = "alloc")]
pub fn potrf<T: FloatScalar + SimdDispatch>(uplo: Uplo, mut a: MatMut<T>) -> Result<(), usize> {
    let n = a.rows;
    assert_eq!(a.cols, n, "potrf: A must be square");
    if n == 0 {
        return Ok(());
    }
    // Row `i` of the lower factor is contiguous in `a` when the stored triangle is the lower
    // one of a row-major or the upper one of a column-major matrix.
    if matches!((uplo, a.layout), (Uplo::Lower, Layout::RowMajor) | (Uplo::Upper, Layout::ColMajor)) {
        assert!((n - 1) * a.stride + n <= a.data.len(), "potrf: A is shorter than its shape");
        return potrf_lower(a.data, n, a.stride);
    }
    let mut l = Scratch::<T>::new(n * n);
    for i in 0..n {
        for p in 0..=i {
            // Symmetry lets either stored triangle fill the lower work panel.
            l[i * n + p] = match uplo {
                Uplo::Lower => a.get(i, p),
                Uplo::Upper => a.get(p, i),
            };
        }
    }
    let result = potrf_lower(&mut l, n, n);
    for i in 0..n {
        for p in 0..=i {
            match uplo {
                Uplo::Lower => a.set(i, p, l[i * n + p]),
                Uplo::Upper => a.set(p, i, l[i * n + p]),
            }
        }
    }
    result
}

/// The lower Cholesky factor of the `n×n` matrix whose row `i` starts at `l[i * ld]`, in place;
/// only entries on or below the diagonal are read or written.
#[cfg(feature = "alloc")]
fn potrf_lower<T: FloatScalar + SimdDispatch>(l: &mut [T], n: usize, ld: usize) -> Result<(), usize> {
    if n <= POTRF_NB {
        return dispatch::<T, _>(FactorK { l, n, ld });
    }
    // Halves, so the trailing update is one rank-`h` `syrk` with a deep contraction rather than
    // a rank-`POTRF_NB` one per block column.
    let h = (n / 2).next_multiple_of(GEMM_MR * 4).min(n - 1);
    let r = n - h;
    potrf_lower(l, h, ld)?;
    let (top, bottom) = l.split_at_mut(h * ld);
    let l11 = Mat::strided(top, h, h, ld, Layout::RowMajor);
    // `X·L11ᵀ = B` is solved as `L11·Xᵀ = Bᵀ` on a transposed copy, whose rows are the
    // right-hand sides a left solve sweeps in place.
    let mut panel = Scratch::<T>::new(h * r);
    transpose(bottom, r, h, ld, &mut panel, r);
    trsm(Side::Left, Uplo::Lower, Trans::N, Diag::NonUnit, T::ONE, l11, MatMut::new(&mut panel, h, r));
    transpose(&panel, h, r, r, bottom, ld);
    syrk(Uplo::Lower, Trans::T, Scalar::neg(T::ONE), Mat::new(&panel, h, r), T::ONE, MatMut::strided(&mut bottom[h..], r, r, ld, Layout::RowMajor));
    drop(panel);
    potrf_lower(&mut bottom[h..], r, ld).map_err(|e| e + h)
}

#[cfg(feature = "alloc")]
const POTRF_NB: usize = 96;

#[cfg(feature = "alloc")]
struct FactorK<'a, T> {
    l: &'a mut [T],
    n: usize,
    ld: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for FactorK<'a, T> {
    type Output = Result<(), usize>;

    #[inline(always)]
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) -> Result<(), usize> {
        let FactorK { l, n, ld } = self;
        for j in 0..n {
            let rj = &l[j * ld..j * ld + j];
            let d = l[j * ld + j].wsub(simd.dot(rj, rj));
            if d.into_f64() <= 0.0 {
                return Err(j);
            }
            let ljj = d.sqrt();
            l[j * ld + j] = ljj;
            let inv = T::from_f64(1.0 / ljj.into_f64());
            let mut i = j + 1;
            while i + GEMV_ROWS <= n {
                let dots = {
                    let (rows_j, rows_i) = l.split_at(i * ld);
                    let rows = core::array::from_fn(|r| &rows_i[r * ld..r * ld + j]);
                    dot_rows::<T, S, GEMV_ROWS>(simd, rows, &rows_j[j * ld..j * ld + j])
                };
                for (r, dot) in dots.into_iter().enumerate() {
                    let o = (i + r) * ld + j;
                    l[o] = l[o].wsub(dot).wmul(inv);
                }
                i += GEMV_ROWS;
            }
            for i in i..n {
                let dot = {
                    let (rows_j, rows_i) = l.split_at(i * ld);
                    simd.dot(&rows_i[..j], &rows_j[j * ld..j * ld + j])
                };
                let s = l[i * ld + j].wsub(dot);
                l[i * ld + j] = s.wmul(inv);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "alloc")]
    use alloc::{vec, vec::Vec};

    fn ramp<T: FloatScalar>(n: usize, start: f64, step: f64) -> Vec<T> {
        (0..n).map(|i| T::from_f64(start + step * i as f64)).collect()
    }

    fn close<T: FloatScalar>(a: T, b: T, tol: f64) -> bool {
        (a.into_f64() - b.into_f64()).abs() <= tol * (1.0 + b.into_f64().abs())
    }

    fn gemv_oracle<T: FloatScalar>(trans: Trans, alpha: T, a: &Mat<T>, x: &[T], beta: T, y: &[T]) -> Vec<T> {
        let out_len = if trans == Trans::T { a.cols } else { a.rows };
        let in_len = if trans == Trans::T { a.rows } else { a.cols };
        (0..out_len)
            .map(|i| {
                let mut acc = 0.0;
                for (j, xj) in x.iter().enumerate().take(in_len) {
                    let aij = if trans == Trans::T { a.get(j, i) } else { a.get(i, j) };
                    acc += aij.into_f64() * xj.into_f64();
                }
                T::from_f64(alpha.into_f64() * acc + beta.into_f64() * y[i].into_f64())
            })
            .collect()
    }

    fn check_gemv<T: FloatScalar + SimdDispatch>(rows: usize, cols: usize, layout: Layout, trans: Trans, alpha: T, beta: T) {
        let pad = 3;
        let (inner, outer) = match layout {
            Layout::RowMajor => (cols, rows),
            Layout::ColMajor => (rows, cols),
        };
        let stride = inner + pad;
        let mut data = ramp::<T>(outer * stride, 1.0, 0.5);
        for i in 0..outer {
            for p in inner..stride {
                data[i * stride + p] = T::from_f64(9999.0);
            }
        }
        let a = Mat::strided(&data, rows, cols, stride, layout);
        let (out_len, in_len) = if trans == Trans::T { (cols, rows) } else { (rows, cols) };
        let x = ramp::<T>(in_len, 0.3, 0.7);
        let mut y = ramp::<T>(out_len, 2.0, -0.4);
        let want = gemv_oracle(trans, alpha, &a, &x, beta, &y);
        gemv(trans, alpha, a, &x, beta, &mut y);
        for (g, w) in y.iter().zip(&want) {
            assert!(close(*g, *w, 1e-5), "gemv {layout:?} {trans:?}: {} vs {}", g.into_f64(), w.into_f64());
        }
    }

    #[test]
    fn gemv_all_combos() {
        for &layout in &[Layout::RowMajor, Layout::ColMajor] {
            for &trans in &[Trans::N, Trans::T] {
                check_gemv::<f32>(11, 7, layout, trans, 1.0, 0.0);
                check_gemv::<f32>(11, 7, layout, trans, 2.5, -1.5);
                check_gemv::<f64>(6, 13, layout, trans, 0.75, 1.0);
                check_gemv::<f32>(1, 20, layout, trans, 1.0, 0.0);
                check_gemv::<f32>(20, 1, layout, trans, 1.0, 0.0);
                check_gemv::<f32>(70, 37, layout, trans, -0.5, 2.0);
                check_gemv::<f64>(37, 70, layout, trans, 1.25, 0.0);
            }
        }
    }

    #[cfg(feature = "alloc")]
    fn build<T: FloatScalar>(rows: usize, cols: usize, layout: Layout, seed: f64) -> (Vec<T>, usize) {
        let (inner, outer) = match layout {
            Layout::RowMajor => (cols, rows),
            Layout::ColMajor => (rows, cols),
        };
        let stride = inner + 2;
        let mut data = ramp::<T>(outer * stride, seed, 0.13);
        for i in 0..outer {
            for p in inner..stride {
                data[i * stride + p] = T::from_f64(-7777.0);
            }
        }
        (data, stride)
    }

    #[cfg(feature = "alloc")]
    #[allow(clippy::too_many_arguments)]
    fn check_gemm<T: FloatScalar + SimdDispatch>(
        m: usize, n: usize, k: usize,
        ta: Trans, tb: Trans, la: Layout, lb: Layout, lc: Layout,
        alpha: T, beta: T,
    ) {
        let (ar, ac) = if ta == Trans::T { (k, m) } else { (m, k) };
        let (br, bc) = if tb == Trans::T { (n, k) } else { (k, n) };
        let (adata, astride) = build::<T>(ar, ac, la, 1.0);
        let (bdata, bstride) = build::<T>(br, bc, lb, 2.0);
        let (mut cdata, cstride) = build::<T>(m, n, lc, 0.5);
        let c0 = cdata.clone();
        let a = Mat::strided(&adata, ar, ac, astride, la);
        let b = Mat::strided(&bdata, br, bc, bstride, lb);

        let a_op = |i: usize, p: usize| if ta == Trans::T { a.get(p, i) } else { a.get(i, p) };
        let b_op = |p: usize, j: usize| if tb == Trans::T { b.get(j, p) } else { b.get(p, j) };
        let c_off = |i: usize, j: usize| match lc {
            Layout::RowMajor => i * cstride + j,
            Layout::ColMajor => j * cstride + i,
        };
        let mut want = c0.clone();
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a_op(i, p).into_f64() * b_op(p, j).into_f64();
                }
                let o = c_off(i, j);
                want[o] = T::from_f64(alpha.into_f64() * acc + beta.into_f64() * c0[o].into_f64());
            }
        }

        let c = MatMut::strided(&mut cdata, m, n, cstride, lc);
        gemm(ta, tb, alpha, a, b, beta, c);
        let tol = if size_of::<T>() < 4 { 1e-2 } else { 1e-4 };
        for i in 0..m {
            for j in 0..n {
                let o = c_off(i, j);
                assert!(
                    close(cdata[o], want[o], tol),
                    "gemm m{m} n{n} k{k} {ta:?}{tb:?} a{la:?} b{lb:?} c{lc:?}: {} vs {}",
                    cdata[o].into_f64(), want[o].into_f64(),
                );
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn gemm_all_combos() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        use Trans::{N, T};
        for &ta in &[N, T] {
            for &tb in &[N, T] {
                for &la in &[RM, CM] {
                    for &lb in &[RM, CM] {
                        for &lc in &[RM, CM] {
                            check_gemm::<f32>(5, 9, 7, ta, tb, la, lb, lc, 1.0, 0.0);
                            check_gemm::<f32>(5, 9, 7, ta, tb, la, lb, lc, 2.0, -0.5);
                            check_gemm::<f64>(10, 6, 11, ta, tb, la, lb, lc, 0.7, 1.0);
                            check_gemm::<half::f16>(5, 9, 7, ta, tb, la, lb, lc, half::f16::from_f32(1.5), half::f16::from_f32(0.25));
                            check_gemm::<f64>(23, 41, 19, ta, tb, la, lb, lc, -1.25, 0.5);
                            check_gemm::<f32>(4, 3, 0, ta, tb, la, lb, lc, 1.0, 0.5);
                            check_gemm::<f32>(4, 3, 0, ta, tb, la, lb, lc, 1.0, 0.0);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn gemm_shapes() {
        use Layout::RowMajor as RM;
        use Trans::N;
        for &(m, n, k) in &[(1, 1, 1), (1, 17, 3), (17, 1, 3), (16, 16, 16), (3, 8, 1), (9, 33, 5), (13, 37, 600), (6, 64, 257)] {
            check_gemm::<f32>(m, n, k, N, N, RM, RM, RM, 1.5, 0.25);
        }
        check_gemm::<f64>(13, 37, 600, Trans::T, Trans::T, Layout::ColMajor, Layout::ColMajor, RM, 0.5, -1.0);
        let h = half::f16::from_f32;
        check_gemm::<half::f16>(13, 37, 40, N, N, RM, RM, RM, h(0.01), h(0.0));
        check_gemm::<half::bf16>(6, 64, 257, Trans::T, N, Layout::ColMajor, RM, RM, half::bf16::from_f32(0.01), half::bf16::ONE);
    }

    #[cfg(feature = "alloc")]
    #[allow(clippy::too_many_arguments)]
    fn check_syrk<T: FloatScalar + SimdDispatch>(
        n: usize, k: usize, uplo: Uplo, trans: Trans, la: Layout, lc: Layout, alpha: T, beta: T,
    ) {
        let (ar, ac) = if trans == Trans::T { (k, n) } else { (n, k) };
        let (adata, astride) = build::<T>(ar, ac, la, 1.0);
        let (mut cdata, cstride) = build::<T>(n, n, lc, 0.5);
        let c0 = cdata.clone();
        let a = Mat::strided(&adata, ar, ac, astride, la);
        let a_op = |i: usize, p: usize| if trans == Trans::T { a.get(p, i) } else { a.get(i, p) };
        let c_off = |i: usize, j: usize| match lc {
            Layout::RowMajor => i * cstride + j,
            Layout::ColMajor => j * cstride + i,
        };
        let mut want = c0.clone();
        for i in 0..n {
            let (lo, hi) = match uplo {
                Uplo::Lower => (0, i + 1),
                Uplo::Upper => (i, n),
            };
            for j in lo..hi {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a_op(i, p).into_f64() * a_op(j, p).into_f64();
                }
                let o = c_off(i, j);
                want[o] = T::from_f64(alpha.into_f64() * acc + beta.into_f64() * c0[o].into_f64());
            }
        }
        let c = MatMut::strided(&mut cdata, n, n, cstride, lc);
        syrk(uplo, trans, alpha, a, beta, c);
        for (o, (g, w)) in cdata.iter().zip(&want).enumerate() {
            assert!(close(*g, *w, 1e-4), "syrk n{n} k{k} {uplo:?} {trans:?} off{o}: {} vs {}", g.into_f64(), w.into_f64());
        }
    }

    #[test]
    fn views_must_cover_their_shape() {
        let data = [0.0f32; 11];
        assert!(std::panic::catch_unwind(|| Mat::strided(&data, 3, 4, 4, Layout::RowMajor)).is_err());
        assert!(std::panic::catch_unwind(|| Mat::strided(&data, 4, 3, 4, Layout::ColMajor)).is_err());
        assert!(std::panic::catch_unwind(|| Mat::new(&data, usize::MAX, 2)).is_err());
        let _ = Mat::strided(&data, 3, 3, 4, Layout::RowMajor);
        let _ = Mat::strided(&data, 0, 9, 9, Layout::RowMajor);
        let _ = Mat::strided(&data, 3, 3, 4, Layout::ColMajor);
    }

    /// Packing buffers are reused within a thread: what a product leaves in them (here NaN) must
    /// not reach the padding of a later, smaller one.
    #[cfg(feature = "alloc")]
    #[test]
    fn reused_buffers_do_not_leak_between_products() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        use Trans::{N, T};
        let poison = vec![f64::NAN; 300 * 300];
        let mut sink = vec![0.0f64; 300 * 300];
        for &tb in &[N, T] {
            gemm(N, tb, 1.0, Mat::new(&poison, 300, 300), Mat::new(&poison, 300, 300), 0.0, MatMut::new(&mut sink, 300, 300));
            check_gemm::<f64>(13, 7, 29, N, tb, RM, RM, RM, 1.0, 0.0);
            check_gemm::<f64>(13, 7, 29, T, tb, CM, RM, RM, 1.0, 0.5);
            let mut l = poison.clone();
            let _ = potrf(Uplo::Lower, MatMut::new(&mut l, 300, 300));
            check_potrf::<f64>(230, Uplo::Lower, RM);
            check_trsm::<f64>(150, 9, Side::Right, Uplo::Lower, Trans::T, Diag::NonUnit, RM, RM, 1.5);
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn syrk_all_combos() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        use Trans::{N, T};
        for &uplo in &[Uplo::Lower, Uplo::Upper] {
            for &trans in &[N, T] {
                for &la in &[RM, CM] {
                    for &lc in &[RM, CM] {
                        check_syrk::<f32>(9, 5, uplo, trans, la, lc, 1.0, 0.0);
                        check_syrk::<f32>(9, 5, uplo, trans, la, lc, 1.5, -0.5);
                        check_syrk::<f64>(7, 12, uplo, trans, la, lc, 0.6, 1.0);
                        check_syrk::<f64>(53, 300, uplo, trans, la, lc, -1.0, 1.0);
                        check_syrk::<f32>(37, 0, uplo, trans, la, lc, 1.0, 0.5);
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    fn tri_entry<T: FloatScalar>(a: &Mat<T>, uplo: Uplo, diag: Diag, i: usize, p: usize) -> f64 {
        let in_tri = match uplo {
            Uplo::Lower => i >= p,
            Uplo::Upper => i <= p,
        };
        if i == p {
            if diag == Diag::Unit { 1.0 } else { a.get(i, i).into_f64() }
        } else if in_tri {
            a.get(i, p).into_f64()
        } else {
            0.0
        }
    }

    #[cfg(feature = "alloc")]
    #[allow(clippy::too_many_arguments)]
    fn check_trsm<T: FloatScalar + SimdDispatch>(
        n: usize, r: usize, side: Side, uplo: Uplo, trans: Trans, diag: Diag,
        la: Layout, lb: Layout, alpha: T,
    ) {
        let (adata, astride) = build::<T>(n, n, la, 1.0);
        let mut a = adata;
        // Off-diagonal entries scaled to at most `1/n`, so the solve stays well-conditioned at
        // any size, unit diagonal included. Only the `n×n` view is touched; the padding keeps its
        // sentinel.
        let max = (0..n).flat_map(|i| (0..n).map(move |p| i * astride + p)).map(|o| a[o].into_f64().abs()).fold(1.0, f64::max);
        for i in 0..n {
            for p in 0..n {
                let o = i * astride + p;
                a[o] = if i == p { T::from_f64(2.0) } else { T::from_f64(a[o].into_f64() / (max * n as f64)) };
            }
        }
        let a = Mat::strided(&a, n, n, astride, la);
        let (br, bc) = match side {
            Side::Left => (n, r),
            Side::Right => (r, n),
        };
        let (mut bdata, bstride) = build::<T>(br, bc, lb, 2.0);
        let b_off = |i: usize, j: usize| match lb {
            Layout::RowMajor => i * bstride + j,
            Layout::ColMajor => j * bstride + i,
        };
        let b0: Vec<f64> = (0..br).flat_map(|i| (0..bc).map(move |j| (i, j))).map(|(i, j)| bdata[b_off(i, j)].into_f64()).collect();
        let b0_at = |i: usize, j: usize| b0[i * bc + j];

        let op_a = |i: usize, p: usize| if trans == Trans::T { tri_entry(&a, uplo, diag, p, i) } else { tri_entry(&a, uplo, diag, i, p) };

        let b = MatMut::strided(&mut bdata, br, bc, bstride, lb);
        trsm(side, uplo, trans, diag, alpha, a, b);
        let x = |i: usize, j: usize| bdata[b_off(i, j)].into_f64();

        match side {
            Side::Left => {
                for i in 0..n {
                    for j in 0..r {
                        let lhs: f64 = (0..n).map(|p| op_a(i, p) * x(p, j)).sum();
                        assert!((lhs - alpha.into_f64() * b0_at(i, j)).abs() <= 1e-3 * (1.0 + lhs.abs()), "trsm L residual");
                    }
                }
            }
            Side::Right => {
                for i in 0..r {
                    for j in 0..n {
                        let lhs: f64 = (0..n).map(|p| x(i, p) * op_a(p, j)).sum();
                        assert!((lhs - alpha.into_f64() * b0_at(i, j)).abs() <= 1e-3 * (1.0 + lhs.abs()), "trsm R residual");
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn trsm_all_combos() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        use Trans::{N, T};
        for &side in &[Side::Left, Side::Right] {
            for &uplo in &[Uplo::Lower, Uplo::Upper] {
                for &trans in &[N, T] {
                    for &diag in &[Diag::NonUnit, Diag::Unit] {
                        for &(la, lb) in &[(RM, RM), (CM, RM), (RM, CM), (CM, CM)] {
                            check_trsm::<f64>(6, 4, side, uplo, trans, diag, la, lb, 1.0);
                            check_trsm::<f64>(6, 4, side, uplo, trans, diag, la, lb, 2.5);
                            check_trsm::<f32>(5, 3, side, uplo, trans, diag, la, lb, 1.0);
                            check_trsm::<f64>(13, 11, side, uplo, trans, diag, la, lb, -0.5);
                            check_trsm::<f32>(20, 19, side, uplo, trans, diag, la, lb, 1.0);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn trsm_blocked() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        for &side in &[Side::Left, Side::Right] {
            for &uplo in &[Uplo::Lower, Uplo::Upper] {
                for &trans in &[Trans::N, Trans::T] {
                    check_trsm::<f64>(150, 9, side, uplo, trans, Diag::NonUnit, RM, RM, 1.5);
                    check_trsm::<f64>(9, 150, side, uplo, trans, Diag::Unit, CM, RM, 1.0);
                    for &(la, lb) in &[(RM, RM), (CM, RM), (RM, CM), (CM, CM)] {
                        check_trsm::<f64>(70, 1, side, uplo, trans, Diag::NonUnit, la, lb, 0.5);
                        check_trsm::<f32>(1, 33, side, uplo, trans, Diag::Unit, la, lb, 1.0);
                    }
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    fn spd<T: FloatScalar>(n: usize, layout: Layout) -> (Vec<T>, usize, Vec<f64>) {
        let mut full = vec![0.0f64; n * n];
        let src = ramp::<f64>(n * n, 0.4, 0.11);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..n {
                    s += src[i * n + p] * src[j * n + p];
                }
                full[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }
        let (mut data, stride) = build::<T>(n, n, layout, 0.0);
        let off = |i: usize, j: usize| match layout {
            Layout::RowMajor => i * stride + j,
            Layout::ColMajor => j * stride + i,
        };
        for i in 0..n {
            for j in 0..n {
                data[off(i, j)] = T::from_f64(full[i * n + j]);
            }
        }
        (data, stride, full)
    }

    #[cfg(feature = "alloc")]
    fn check_potrf<T: FloatScalar + SimdDispatch>(n: usize, uplo: Uplo, layout: Layout) {
        let (mut data, stride, orig) = spd::<T>(n, layout);
        let a = MatMut::strided(&mut data, n, n, stride, layout);
        potrf(uplo, a).expect("SPD should factor");
        let off = |i: usize, j: usize| match layout {
            Layout::RowMajor => i * stride + j,
            Layout::ColMajor => j * stride + i,
        };
        let fac = |i: usize, j: usize| data[off(i, j)].into_f64();
        for i in 0..n {
            for j in 0..n {
                let recon: f64 = (0..n)
                    .filter(|&p| p <= i && p <= j)
                    .map(|p| match uplo {
                        Uplo::Lower => fac(i, p) * fac(j, p),
                        Uplo::Upper => fac(p, i) * fac(p, j),
                    })
                    .sum();
                assert!((recon - orig[i * n + j]).abs() <= 1e-3 * (1.0 + orig[i * n + j].abs()), "potrf recon ({i},{j})");
                let outside = match uplo {
                    Uplo::Lower => j > i,
                    Uplo::Upper => j < i,
                };
                if outside {
                    assert_eq!(fac(i, j), T::from_f64(orig[i * n + j]).into_f64(), "potrf wrote ({i},{j})");
                }
            }
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn potrf_recon() {
        for &uplo in &[Uplo::Lower, Uplo::Upper] {
            for &layout in &[Layout::RowMajor, Layout::ColMajor] {
                check_potrf::<f64>(8, uplo, layout);
                check_potrf::<f32>(5, uplo, layout);
                check_potrf::<f64>(230, uplo, layout);
            }
        }
    }

    /// The failing pivot is reported by its index in the whole matrix, wherever the
    /// factorization splits it.
    #[cfg(feature = "alloc")]
    #[test]
    fn potrf_reports_the_first_failing_pivot() {
        for (n, bad) in [(300, 250), (300, 7), (300, 160), (97, 96)] {
            let (mut data, stride, _) = spd::<f64>(n, Layout::RowMajor);
            data[bad * stride + bad] = -1e12;
            assert_eq!(potrf(Uplo::Lower, MatMut::strided(&mut data, n, n, stride, Layout::RowMajor)), Err(bad), "n {n}");
        }
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn potrf_non_spd() {
        let mut data = vec![1.0f64, 2.0, 2.0, 1.0];
        let a = MatMut::strided(&mut data, 2, 2, 2, Layout::RowMajor);
        assert!(potrf(Uplo::Lower, a).is_err());
    }

    // M,N multiples of 64 clear any valid ZA blk (16/32/64) and SME_MIN, so with Accelerate off
    // on an SME host these drive the ZA-grid kernel; elsewhere the gang kernel at scale.
    #[cfg(feature = "alloc")]
    #[test]
    fn gemm_large_aligned() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        use Trans::{N, T};
        for &ta in &[N, T] {
            for &tb in &[N, T] {
                check_gemm::<f32>(64, 64, 32, ta, tb, RM, RM, RM, 1.0, 0.0);
                check_gemm::<f32>(64, 64, 48, ta, tb, RM, RM, RM, 2.0, -0.5);
                check_gemm::<f32>(64, 64, 32, ta, tb, CM, RM, RM, 1.5, 1.0);
                check_gemm::<f64>(64, 64, 32, ta, tb, RM, RM, RM, 0.75, 1.0);
            }
        }
        check_gemm::<f32>(64, 64, 32, N, N, RM, RM, CM, 1.0, 0.0);
        check_gemm::<f32>(64, 64, 32, N, N, RM, RM, CM, 1.5, -2.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn syrk_large_aligned() {
        use Layout::{ColMajor as CM, RowMajor as RM};
        use Trans::{N, T};
        for &uplo in &[Uplo::Lower, Uplo::Upper] {
            for &trans in &[N, T] {
                check_syrk::<f32>(64, 32, uplo, trans, RM, RM, 1.0, 0.0);
                check_syrk::<f32>(64, 48, uplo, trans, RM, RM, 1.5, -0.5);
                check_syrk::<f32>(64, 32, uplo, trans, CM, CM, 0.7, 1.0);
                check_syrk::<f64>(64, 32, uplo, trans, RM, RM, 0.9, 1.0);
            }
        }
    }

    #[test]
    fn reductions() {
        for &layout in &[Layout::RowMajor, Layout::ColMajor] {
            let (rows, cols) = (9usize, 5usize);
            let (inner, outer) = match layout {
                Layout::RowMajor => (cols, rows),
                Layout::ColMajor => (rows, cols),
            };
            let stride = inner + 2;
            let data = ramp::<f64>(outer * stride, 1.0, 0.25);
            let a = Mat::strided(&data, rows, cols, stride, layout);

            let mut rs = vec![0.0; rows];
            row_sums(a, &mut rs);
            for (r, &got) in rs.iter().enumerate() {
                let want: f64 = (0..cols).map(|c| a.get(r, c)).sum();
                assert!(close(got, want, 1e-9));
            }

            let mut cs = vec![0.0; cols];
            col_sums(a, &mut cs);
            for (c, &got) in cs.iter().enumerate() {
                let want: f64 = (0..rows).map(|r| a.get(r, c)).sum();
                assert!(close(got, want, 1e-9));
            }

            let mut ss = 0.0;
            for r in 0..rows {
                for c in 0..cols {
                    ss += a.get(r, c) * a.get(r, c);
                }
            }
            assert!(close(fro_norm(a), ss.sqrt(), 1e-9));
        }
    }

    #[test]
    fn half_gemm_leaves_memory_outside_the_view_alone() {
        let (rows, cols, stride) = (4usize, 3usize, 8usize);
        let a = vec![half::f16::ONE; 8 * stride];
        let b = vec![half::f16::ONE; 8 * stride];
        for layout in [Layout::RowMajor, Layout::ColMajor] {
            for beta in [half::f16::ZERO, half::f16::ONE] {
                let mut c = vec![half::f16::from_f32(7.0); 8 * stride];
                gemm(
                    Trans::N,
                    Trans::N,
                    half::f16::ONE,
                    Mat::strided(&a, rows, 5, stride, Layout::RowMajor),
                    Mat::strided(&b, 5, cols, stride, Layout::RowMajor),
                    beta,
                    MatMut::strided(&mut c, rows, cols, stride, layout),
                );
                let inside = if beta == half::f16::ZERO { 5.0 } else { 12.0 };
                for (o, v) in c.iter().enumerate() {
                    let (r, col) = match layout {
                        Layout::RowMajor => (o / stride, o % stride),
                        Layout::ColMajor => (o % stride, o / stride),
                    };
                    let want = if r < rows && col < cols { inside } else { 7.0 };
                    assert_eq!(v.to_f32(), want, "{layout:?}, beta {beta}, ({r}, {col})");
                }
            }
        }
    }
}
