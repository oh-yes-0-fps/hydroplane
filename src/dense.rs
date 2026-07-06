//! Runtime-dimensioned dense linear algebra (the compile-time tile counterpart is
//! [`matrix`](crate::matrix)). Every routine goes to Apple Accelerate where present, otherwise
//! to a fallback built on the SIMD backend ([`Gang`]/[`Varying`](crate::Varying)).

use crate::dispatch::{Kernel, SimdDispatch, dispatch};
use crate::matrix::Layout;
use crate::scalar::{FloatScalar, Scalar};
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

impl<'a, T> Mat<'a, T> {
    /// A dense row-major `rows × cols` view (`stride = cols`).
    #[inline]
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        assert!(data.len() >= rows * cols, "Mat::new: slice too short");
        Mat { data, rows, cols, stride: cols, layout: Layout::RowMajor }
    }

    /// A view with an explicit leading stride and layout: a sub-matrix of a larger buffer, or
    /// column-major data.
    #[inline]
    pub fn strided(data: &'a [T], rows: usize, cols: usize, stride: usize, layout: Layout) -> Self {
        let inner = match layout {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        };
        assert!(stride >= inner, "Mat::strided: stride below inner dimension");
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
    #[inline]
    fn get(&self, r: usize, c: usize) -> T {
        self.data[self.off(r, c)]
    }

    #[inline]
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
        assert!(data.len() >= rows * cols, "MatMut::new: slice too short");
        MatMut { data, rows, cols, stride: cols, layout: Layout::RowMajor }
    }

    /// A mutable view with an explicit leading stride and layout.
    #[inline]
    pub fn strided(
        data: &'a mut [T],
        rows: usize,
        cols: usize,
        stride: usize,
        layout: Layout,
    ) -> Self {
        let inner = match layout {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        };
        assert!(stride >= inner, "MatMut::strided: stride below inner dimension");
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

    #[inline]
    fn off(&self, r: usize, c: usize) -> usize {
        match self.layout {
            Layout::RowMajor => r * self.stride + c,
            Layout::ColMajor => c * self.stride + r,
        }
    }
}

#[cfg(feature = "alloc")]
impl<'a, T: Copy> MatMut<'a, T> {
    #[inline]
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
            for (i, yi) in y.iter_mut().enumerate() {
                let base = i * a.stride;
                let d = simd.dot(&a.data[base..base + op_cols], x);
                *yi = fma_scalar(alpha, d, beta, *yi);
            }
        } else {
            scale(y, beta);
            for (k, &xk) in x.iter().enumerate() {
                let base = k * a.stride;
                let line = &a.data[base..base + op_rows];
                let cs = simd.splat(alpha.wmul(xk));
                simd.zip_map_inplace(line, y, T::ZERO, T::ZERO, |lv, yv| lv.fma(cs, yv));
            }
        }
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

    if c.layout == Layout::ColMajor {
        // Cᵀ = op(B)ᵀ·op(A)ᵀ; the same buffer read row-major is Cᵀ (n×m) with the same stride.
        let ct = MatMut::strided(c.data, c.cols, c.rows, c.stride, Layout::RowMajor);
        gemm(flip(tb), flip(ta), alpha, b, a, beta, ct);
        return;
    }

    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if accel_gemm(ta, tb, alpha, a, b, beta, &c, m, n, k) {
        return;
    }

    let bp = pack_b(tb, &b, k, n);
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
    if gemm_sme(alpha, beta, ta, a, &bp, &mut c, m, n, k) {
        return;
    }
    dispatch::<T, _>(GemmK { ta, a, alpha, beta, bp: &bp, c, m, n, k });
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

/// Pack `op(B)` into a contiguous `k×n` row-major panel so the microkernel loads each
/// contraction row as one gang vector regardless of `B`'s storage.
#[cfg(feature = "alloc")]
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

/// Register-blocking factor: independent accumulator chains held across the contraction.
#[cfg(feature = "alloc")]
const MR: usize = 4;

#[cfg(feature = "alloc")]
struct GemmK<'a, T: FloatScalar> {
    ta: Trans,
    a: Mat<'a, T>,
    alpha: T,
    beta: T,
    bp: &'a [T],
    c: MatMut<'a, T>,
    m: usize,
    n: usize,
    k: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for GemmK<'a, T> {
    type Output = ();

    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let GemmK { ta, a, alpha, beta, bp, c, m, n, k } = self;
        let l = simd.lanes::<T>();
        let zero = simd.splat(T::ZERO);
        let a_alpha = simd.splat(alpha);
        let a_beta = simd.splat(beta);
        let beta_nz = beta.into_f64() != 0.0;
        let a_op = |i: usize, p: usize| match ta {
            Trans::N => a.get(i, p),
            Trans::T => a.get(p, i),
        };

        let mut j = 0;
        while j < n {
            let w = (n - j).min(l);
            let mut i = 0;
            while i < m {
                let mr = (m - i).min(MR);
                let mut acc = [zero; MR];
                for p in 0..k {
                    let brow = &bp[p * n + j..p * n + j + w];
                    let bv = if w == l { simd.load(brow) } else { simd.load_partial(brow, T::ZERO) };
                    for (ii, accm) in acc.iter_mut().enumerate().take(mr) {
                        *accm = simd.splat(a_op(i + ii, p)).fma(bv, *accm);
                    }
                }
                for (ii, accm) in acc.iter().enumerate().take(mr) {
                    let base = (i + ii) * c.stride + j;
                    let crow = &mut c.data[base..base + w];
                    let mut out = accm.fma(a_alpha, zero);
                    if beta_nz {
                        let cv = if w == l { simd.load(crow) } else { simd.load_partial(crow, T::ZERO) };
                        out = cv.fma(a_beta, out);
                    }
                    if w == l {
                        out.store(crow);
                    } else {
                        out.store_partial(crow);
                    }
                }
                i += MR;
            }
            j += l;
        }
    }
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
/// the fallback packs `op(A)`'s rows contiguous and dots each referenced `(i, j)` pair.
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
    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if accel_trsm(side, uplo, trans, diag, alpha, a, &mut b) {
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
    mut b: MatMut<T>,
) {
    let (m, n) = (b.rows, b.cols);
    // op(A) is lower-triangular for (Lower, N) and (Upper, T), solved forward; upper otherwise.
    let op_lower = (uplo == Uplo::Lower) == (trans == Trans::N);
    let mut rp = alloc::vec![T::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            rp[i * n + j] = b.get(i, j);
        }
    }
    dispatch::<T, _>(SolveK { a, alpha, diag, trans, op_lower, rp: &mut rp, m, n });
    for i in 0..m {
        for j in 0..n {
            b.set(i, j, rp[i * n + j]);
        }
    }
}

#[cfg(feature = "alloc")]
struct SolveK<'a, T: FloatScalar> {
    a: Mat<'a, T>,
    alpha: T,
    diag: Diag,
    trans: Trans,
    op_lower: bool,
    rp: &'a mut [T],
    m: usize,
    n: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for SolveK<'a, T> {
    type Output = ();

    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) {
        let SolveK { a, alpha, diag, trans, op_lower, rp, m, n } = self;
        let op_a = |i: usize, p: usize| match trans {
            Trans::N => a.get(i, p),
            Trans::T => a.get(p, i),
        };
        if alpha.into_f64() != 1.0 {
            for v in rp.iter_mut() {
                *v = alpha.wmul(*v);
            }
        }
        let recip = |d: T| T::from_f64(1.0 / d.into_f64());
        let solve_row = |row_i: &mut [T], prior: &[T], p: usize, i: usize| {
            let c = op_a(i, p);
            if c.into_f64() != 0.0 {
                let cs = simd.splat(Scalar::neg(c));
                simd.zip_map_inplace(prior, row_i, T::ZERO, T::ZERO, |pv, rv| pv.fma(cs, rv));
            }
        };
        if op_lower {
            for i in 0..m {
                let (done, rest) = rp.split_at_mut(i * n);
                let row_i = &mut rest[..n];
                for p in 0..i {
                    solve_row(row_i, &done[p * n..p * n + n], p, i);
                }
                if diag == Diag::NonUnit {
                    let inv = recip(op_a(i, i));
                    for v in row_i.iter_mut() {
                        *v = v.wmul(inv);
                    }
                }
            }
        } else {
            for i in (0..m).rev() {
                let (left, rest) = rp.split_at_mut(i * n);
                let _ = left;
                let (row_i, after) = rest.split_at_mut(n);
                for p in (i + 1)..m {
                    solve_row(row_i, &after[(p - i - 1) * n..(p - i - 1) * n + n], p, i);
                }
                if diag == Diag::NonUnit {
                    let inv = recip(op_a(i, i));
                    for v in row_i.iter_mut() {
                        *v = v.wmul(inv);
                    }
                }
            }
        }
    }
}

/// Cholesky factorization of a symmetric positive-definite `A`, in place: `A = L·Lᵀ` (`Uplo::Lower`)
/// or `A = Uᵀ·U` (`Uplo::Upper`), overwriting the `uplo` triangle with the factor. Returns
/// `Err(j)` if the leading minor of order `j+1` is not positive definite (a non-positive pivot).
/// The referenced triangle is packed into a contiguous lower panel so the Crout dot-products stay
/// vectorized regardless of `A`'s storage.
#[cfg(feature = "alloc")]
pub fn potrf<T: FloatScalar + SimdDispatch>(uplo: Uplo, mut a: MatMut<T>) -> Result<(), usize> {
    let n = a.rows;
    assert_eq!(a.cols, n, "potrf: A must be square");
    if n == 0 {
        return Ok(());
    }
    let mut l = alloc::vec![T::ZERO; n * n];
    for i in 0..n {
        for p in 0..=i {
            // Symmetry lets either stored triangle fill the lower work panel.
            l[i * n + p] = match uplo {
                Uplo::Lower => a.get(i, p),
                Uplo::Upper => a.get(p, i),
            };
        }
    }
    dispatch::<T, _>(FactorK { l: &mut l, n })?;
    for i in 0..n {
        for p in 0..=i {
            match uplo {
                Uplo::Lower => a.set(i, p, l[i * n + p]),
                Uplo::Upper => a.set(p, i, l[i * n + p]),
            }
        }
    }
    Ok(())
}

#[cfg(feature = "alloc")]
struct FactorK<'a, T> {
    l: &'a mut [T],
    n: usize,
}

#[cfg(feature = "alloc")]
impl<'a, T: FloatScalar> Kernel<T> for FactorK<'a, T> {
    type Output = Result<(), usize>;

    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) -> Result<(), usize> {
        let FactorK { l, n } = self;
        for j in 0..n {
            let rj = &l[j * n..j * n + j];
            let d = l[j * n + j].wsub(simd.dot(rj, rj));
            if d.into_f64() <= 0.0 {
                return Err(j);
            }
            let ljj = d.sqrt();
            l[j * n + j] = ljj;
            let inv = T::from_f64(1.0 / ljj.into_f64());
            for i in (j + 1)..n {
                let dot = {
                    let (rows_j, rows_i) = l.split_at(i * n);
                    simd.dot(&rows_i[..j], &rows_j[j * n..j * n + j])
                };
                let s = l[i * n + j].wsub(dot);
                l[i * n + j] = s.wmul(inv);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        for i in 0..m {
            for j in 0..n {
                let o = c_off(i, j);
                assert!(
                    close(cdata[o], want[o], 1e-4),
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
        for &(m, n, k) in &[(1, 1, 1), (1, 17, 3), (17, 1, 3), (16, 16, 16), (3, 8, 1), (9, 33, 5)] {
            check_gemm::<f32>(m, n, k, N, N, RM, RM, RM, 1.5, 0.25);
        }
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
        // Diagonal dominance keeps the solve well-conditioned.
        for i in 0..n {
            let o = match la {
                Layout::RowMajor => i * astride + i,
                Layout::ColMajor => i * astride + i,
            };
            a[o] = T::from_f64(n as f64 + 5.0);
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
                        }
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
            }
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
}
