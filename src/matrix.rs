//! Portable tile matrix-multiply: [`MatrixBackend`] extends the element-wise [`Backend`] with a
//! 2-D tile and the fused multiply-add `D = A·B + C`, lowered per backend. Mixed precision
//! follows [`FloatScalar::Compute`]: an `f16` matmul accumulates in `f32`.

use crate::backend::Backend;
use crate::scalar::{FloatScalar, Scalar};

mod sealed {
    pub trait Sealed {}
}

/// Which operand of `D = A·B + C` a [tile](MatrixBackend::Tile) is. A compile-time marker mapping
/// to SPIR-V's cooperative-matrix `Use`; phantom on the CPU. Sealed: the three roles are exhaustive.
pub trait Role: sealed::Sealed + Copy + 'static {
    /// The SPIR-V `CooperativeMatrixUse` value (`MatrixA`/`MatrixB`/`MatrixAccumulator`).
    const USE: u32;
    /// How the CPU/array backends store a tile of this role: a borrowing [`View`] for the
    /// read-only inputs (`A`/`B`), an owned `[[E; C]; R]` for the accumulator. The GPU backend
    /// ignores this and uses its own opaque tile.
    type Repr<'a, E: Scalar, const R: usize, const C: usize>: CpuTile<'a, E, R, C>;
}

/// The left operand `A` (`M×K`).
#[derive(Clone, Copy)]
pub struct MatrixA;
/// The right operand `B` (`K×N`).
#[derive(Clone, Copy)]
pub struct MatrixB;
/// The accumulator `C`/`D` (`M×N`).
#[derive(Clone, Copy)]
pub struct Accumulator;

impl sealed::Sealed for MatrixA {}
impl sealed::Sealed for MatrixB {}
impl sealed::Sealed for Accumulator {}
impl Role for MatrixA {
    const USE: u32 = 0;
    type Repr<'a, E: Scalar, const R: usize, const C: usize> = View<'a, E, R, C>;
}
impl Role for MatrixB {
    const USE: u32 = 1;
    type Repr<'a, E: Scalar, const R: usize, const C: usize> = View<'a, E, R, C>;
}
impl Role for Accumulator {
    const USE: u32 = 2;
    type Repr<'a, E: Scalar, const R: usize, const C: usize> = [[E; C]; R];
}

/// In-memory layout of a tile when loading from / storing to a slice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layout {
    /// Element `(r, c)` lives at `r * row_stride + c`.
    RowMajor,
    /// Element `(r, c)` lives at `c * row_stride + r`.
    ColMajor,
}

/// A borrowing view of a read-only input tile (`A`/`B`): a pointer + stride + layout into the
/// caller's slice, so loading an input copies nothing. `'a` ties it to the borrowed slice.
/// [`get`](View::get) honors the layout; [`dense_ptr`](View::dense_ptr) exposes the raw pointer in the dense
/// row-major case so the GEMM packs straight from the caller's memory.
pub struct View<'a, E, const R: usize, const C: usize> {
    ptr: *const E,
    row_stride: usize,
    layout: Layout,
    _p: core::marker::PhantomData<&'a [E]>,
}
impl<E, const R: usize, const C: usize> Clone for View<'_, E, R, C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<E, const R: usize, const C: usize> Copy for View<'_, E, R, C> {}
// SAFETY: a `View` is an immutable borrow of `[E]`, so Send/Sync exactly when `&[E]` is.
unsafe impl<E: Sync, const R: usize, const C: usize> Send for View<'_, E, R, C> {}
unsafe impl<E: Sync, const R: usize, const C: usize> Sync for View<'_, E, R, C> {}
impl<'a, E: Scalar, const R: usize, const C: usize> View<'a, E, R, C> {
    /// Element `(r, c)`, honoring the view's stride and layout.
    #[inline]
    pub fn get(self, r: usize, c: usize) -> E {
        // SAFETY: `(r, c)` is in `R×C` and the view borrows a slice large enough for the tile.
        unsafe { *self.ptr.add(tile_index(r, c, self.row_stride, self.layout)) }
    }
    /// The backing pointer iff the tile is dense row-major (`row_stride == C`), the zero-copy
    /// fast path; `None` when a gather/copy is needed.
    #[inline]
    pub fn dense_ptr(self) -> Option<*const E> {
        if matches!(self.layout, Layout::RowMajor) && self.row_stride == C {
            Some(self.ptr)
        } else {
            None
        }
    }
}

/// How a tile is stored by the CPU/array backends, per [`Role`]: a zero-copy [`View`] for the
/// read-only inputs, an owned `[[E; C]; R]` for the accumulator. The backend's tile methods
/// delegate here.
pub trait CpuTile<'a, E: Scalar, const R: usize, const C: usize>: Copy {
    /// Reference the tile in `mem` (a view) or copy it in (an owned tile).
    fn ct_load(mem: &'a [E], row_stride: usize, layout: Layout) -> Self;
    /// Write the tile to `out`. Only the (owned) accumulator is ever stored; inputs panic.
    fn ct_store(self, out: &mut [E], row_stride: usize, layout: Layout);
    /// An owned tile with every element `v` (accumulator only; inputs panic).
    fn ct_splat(v: E) -> Self;
    /// Apply `f` elementwise (accumulator only; inputs panic).
    fn ct_map(self, f: impl Fn(E) -> E) -> Self;
    /// Element `(r, c)`.
    fn ct_get(self, r: usize, c: usize) -> E;
    /// Dense row-major backing pointer for the zero-copy pack, or `None` (copy needed / owned).
    fn ct_dense_ptr(self) -> Option<*const E>;
}

impl<'a, E: Scalar, const R: usize, const C: usize> CpuTile<'a, E, R, C> for View<'a, E, R, C> {
    #[inline]
    fn ct_load(mem: &'a [E], row_stride: usize, layout: Layout) -> Self {
        View { ptr: mem.as_ptr(), row_stride, layout, _p: core::marker::PhantomData }
    }
    #[inline]
    fn ct_store(self, _out: &mut [E], _row_stride: usize, _layout: Layout) {
        unreachable!("a read-only input tile is never stored")
    }
    #[inline]
    fn ct_splat(_v: E) -> Self {
        unreachable!("a read-only input tile is never splatted")
    }
    #[inline]
    fn ct_map(self, _f: impl Fn(E) -> E) -> Self {
        unreachable!("a read-only input tile is never mapped")
    }
    #[inline]
    fn ct_get(self, r: usize, c: usize) -> E {
        View::get(self, r, c)
    }
    #[inline]
    fn ct_dense_ptr(self) -> Option<*const E> {
        View::dense_ptr(self)
    }
}

impl<'a, E: Scalar, const R: usize, const C: usize> CpuTile<'a, E, R, C> for [[E; C]; R] {
    #[inline]
    fn ct_load(mem: &'a [E], row_stride: usize, layout: Layout) -> Self {
        let mut t = [[E::ZERO; C]; R];
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                t[r][c] = mem[tile_index(r, c, row_stride, layout)];
                c += 1;
            }
            r += 1;
        }
        t
    }
    #[inline]
    fn ct_store(self, out: &mut [E], row_stride: usize, layout: Layout) {
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                out[tile_index(r, c, row_stride, layout)] = self[r][c];
                c += 1;
            }
            r += 1;
        }
    }
    #[inline]
    fn ct_splat(v: E) -> Self {
        [[v; C]; R]
    }
    #[inline]
    fn ct_map(mut self, f: impl Fn(E) -> E) -> Self {
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                self[r][c] = f(self[r][c]);
                c += 1;
            }
            r += 1;
        }
        self
    }
    #[inline]
    fn ct_get(self, r: usize, c: usize) -> E {
        self[r][c]
    }
    #[inline]
    fn ct_dense_ptr(self) -> Option<*const E> {
        None
    }
}

/// A backend that can hold matrix tiles and fuse-multiply-add them. Every backend that runs a
/// kernel also implements this, so a matmul kernel tightens its bound from `Backend<T>` to
/// `MatrixBackend<T>` and the existing dispatch delivers it.
pub trait MatrixBackend<T: FloatScalar>: Backend<T> {
    /// An `R×C` tile of element `E` in role `Ro`. Concrete `[[E; C]; R]` on the CPU/scalar/GPU
    /// floor; an opaque distributed handle once a cooperative-matrix lowering is available.
    type Tile<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>: Copy;

    /// Load an `R×C` tile from `mem` with the given row stride and layout. For the input roles
    /// this is zero-copy: the tile borrows `mem` for `'a` (a [`View`]). The accumulator copies in.
    fn tile_load<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        mem: &'a [E],
        row_stride: usize,
        layout: Layout,
    ) -> Self::Tile<'a, E, R, C, Ro>;

    /// Store an `R×C` tile back to `out` with the given row stride and layout.
    fn tile_store<E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        t: Self::Tile<'_, E, R, C, Ro>,
        out: &mut [E],
        row_stride: usize,
        layout: Layout,
    );

    /// A tile with every element set to `v` (owned, so valid for any `'a`).
    fn tile_splat<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        v: E,
    ) -> Self::Tile<'a, E, R, C, Ro>;

    /// Apply `f` to every element. Position-independent only (activation / bias / scale): on the
    /// GPU the element→`(row, col)` mapping is opaque, so an index-dependent `f` is not portable.
    fn tile_map<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        t: Self::Tile<'a, E, R, C, Ro>,
        f: impl Fn(E) -> E,
    ) -> Self::Tile<'a, E, R, C, Ro>;

    /// `D = A·B + C`. `A`/`B` hold element `T`; the accumulator holds `T::Compute`. All three
    /// tiles and the result share one lifetime `'i`: the inputs borrow their source slices for it,
    /// the owned accumulator just carries it.
    fn mma<'i, const M: usize, const N: usize, const K: usize>(
        self,
        a: Self::Tile<'i, T, M, K, MatrixA>,
        b: Self::Tile<'i, T, K, N, MatrixB>,
        c: Self::Tile<'i, T::Compute, M, N, Accumulator>,
    ) -> Self::Tile<'i, T::Compute, M, N, Accumulator>;
}

#[inline]
fn tile_index(r: usize, c: usize, row_stride: usize, layout: Layout) -> usize {
    match layout {
        Layout::RowMajor => r * row_stride + c,
        Layout::ColMajor => c * row_stride + r,
    }
}

/// Register-blocked GEMM, all operands already in compute precision `C`. Blocks `N` into
/// `lanes()`-wide accumulators kept in registers across the `K` reduction, folding with the
/// backend's `fma`; a scalar tail handles `N % lanes`.
#[inline]
fn simd_gemm<C: FloatScalar, B: Backend<C>, const M: usize, const N: usize, const K: usize>(
    backend: B,
    a: [[C; K]; M],
    b: [[C; N]; K],
    mut c: [[C; N]; M],
) -> [[C; N]; M] {
    let lanes = backend.lanes();
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j + lanes <= N {
            let mut acc = backend.load(&c[i][j..j + lanes]);
            let mut k = 0;
            while k < K {
                let aik = backend.splat(a[i][k]);
                let br = backend.load(&b[k][j..j + lanes]);
                acc = backend.fma(aik, br, acc);
                k += 1;
            }
            backend.store(acc, &mut c[i][j..j + lanes]);
            j += lanes;
        }
        while j < N {
            let mut s = c[i][j];
            let mut k = 0;
            while k < K {
                s = a[i][k].fma(b[k][j], s);
                k += 1;
            }
            c[i][j] = s;
            j += 1;
        }
        i += 1;
    }
    c
}

/// Register-blocked + B-packed GEMM: the BLIS micro-kernel generalized to every element-wise
/// [`Backend`]. Two fixes over [`simd_gemm`]: pack B once into contiguous `lanes`-wide column
/// panels so the `K`-loop's B loads are unit-stride and L1-resident, and keep an `MR×NR`
/// micro-tile of `C` in independent accumulators so the `fma` latency chains overlap. The caller
/// sizes `NR` from the cached per-core unroll factor (`Gang::unroll`), capped to fit the SIMD
/// register file (`MR·NR + NR + 1` registers).
///
/// Numerically identical to [`simd_gemm`] (same per-element `fma` order). Worth it only for large
/// tiles; the caller gates on size.
#[cfg(feature = "std")]
#[inline]
#[allow(clippy::needless_range_loop)]
fn packed_gemm<
    C: FloatScalar,
    B: Backend<C>,
    const M: usize,
    const N: usize,
    const K: usize,
    const MR: usize,
    const NR: usize,
>(
    backend: B,
    a: [[C; K]; M],
    b: [[C; N]; K],
    mut c: [[C; N]; M],
) -> [[C; N]; M] {
    let lanes = backend.lanes();
    let nb = N / lanes;
    let (full, panel) = (nb * lanes, K * lanes);
    let mut bp = vec![C::ZERO; nb * panel];
    for jb in 0..nb {
        let j0 = jb * lanes;
        for k in 0..K {
            // Slot and source run are both `lanes`-wide contiguous, so this is one vector copy.
            let dst = jb * panel + k * lanes;
            backend.store(backend.load(&b[k][j0..j0 + lanes]), &mut bp[dst..dst + lanes]);
        }
    }

    // One packed B vector (panel `jb`, row `k`).
    let bvec = |jb: usize, k: usize| backend.load(&bp[jb * panel + k * lanes..jb * panel + k * lanes + lanes]);
    // Reduce one `lanes`-wide C block over K (the N-block and M-row tails).
    let edge = |c: &mut [[C; N]; M], i: usize, jb: usize| {
        let cj = jb * lanes;
        let mut acc = backend.load(&c[i][cj..cj + lanes]);
        for k in 0..K {
            acc = backend.fma(backend.splat(a[i][k]), bvec(jb, k), acc);
        }
        backend.store(acc, &mut c[i][cj..cj + lanes]);
    };

    let mut i = 0;
    while i + MR <= M {
        let mut jb = 0;
        while jb + NR <= nb {
            let mut acc = [[backend.splat(C::ZERO); NR]; MR];
            for (mr, row) in acc.iter_mut().enumerate() {
                for (nv, cell) in row.iter_mut().enumerate() {
                    let cj = (jb + nv) * lanes;
                    *cell = backend.load(&c[i + mr][cj..cj + lanes]);
                }
            }
            for k in 0..K {
                let bv: [B::Vector; NR] = core::array::from_fn(|nv| bvec(jb + nv, k));
                for (mr, row) in acc.iter_mut().enumerate() {
                    let av = backend.splat(a[i + mr][k]);
                    for (nv, cell) in row.iter_mut().enumerate() {
                        *cell = backend.fma(av, bv[nv], *cell);
                    }
                }
            }
            for (mr, row) in acc.iter().enumerate() {
                for (nv, cell) in row.iter().enumerate() {
                    let cj = (jb + nv) * lanes;
                    backend.store(*cell, &mut c[i + mr][cj..cj + lanes]);
                }
            }
            jb += NR;
        }
        while jb < nb {
            for mr in 0..MR {
                edge(&mut c, i + mr, jb);
            }
            jb += 1;
        }
        i += MR;
    }
    while i < M {
        for jb in 0..nb {
            edge(&mut c, i, jb);
        }
        i += 1;
    }

    for i in 0..M {
        for j in full..N {
            let mut s = c[i][j];
            for k in 0..K {
                s = a[i][k].fma(b[k][j], s);
            }
            c[i][j] = s;
        }
    }
    c
}

// Apple Accelerate cblas bindings; Accelerate routes the GEMM to the AMX/SME coprocessor.
// The `#[link]` lives here rather than in a dependency so `--cfg hp_no_apple_accelerate` can drop
// the framework link entirely (Cargo can't gate a dependency on a custom `--cfg`).
#[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
pub(crate) mod accel {
    use core::ffi::{c_int, c_uint};
    pub const ROW_MAJOR: c_uint = 101;
    pub const COL_MAJOR: c_uint = 102;
    pub const NO_TRANS: c_uint = 111;
    pub const TRANS: c_uint = 112;
    pub const UPPER: c_uint = 121;
    pub const LOWER: c_uint = 122;
    pub const NON_UNIT: c_uint = 131;
    pub const UNIT: c_uint = 132;
    pub const LEFT: c_uint = 141;
    pub const RIGHT: c_uint = 142;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: c_uint, transa: c_uint, transb: c_uint, m: c_int, n: c_int, k: c_int,
            alpha: f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int,
            beta: f32, c: *mut f32, ldc: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_dgemm(
            order: c_uint, transa: c_uint, transb: c_uint, m: c_int, n: c_int, k: c_int,
            alpha: f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int,
            beta: f64, c: *mut f64, ldc: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemv(
            order: c_uint, trans: c_uint, m: c_int, n: c_int,
            alpha: f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int,
            beta: f32, y: *mut f32, incy: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_dgemv(
            order: c_uint, trans: c_uint, m: c_int, n: c_int,
            alpha: f64, a: *const f64, lda: c_int, x: *const f64, incx: c_int,
            beta: f64, y: *mut f64, incy: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_ssyrk(
            order: c_uint, uplo: c_uint, trans: c_uint, n: c_int, k: c_int,
            alpha: f32, a: *const f32, lda: c_int, beta: f32, c: *mut f32, ldc: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_dsyrk(
            order: c_uint, uplo: c_uint, trans: c_uint, n: c_int, k: c_int,
            alpha: f64, a: *const f64, lda: c_int, beta: f64, c: *mut f64, ldc: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_strsm(
            order: c_uint, side: c_uint, uplo: c_uint, transa: c_uint, diag: c_uint,
            m: c_int, n: c_int, alpha: f32, a: *const f32, lda: c_int, b: *mut f32, ldb: c_int,
        );
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_dtrsm(
            order: c_uint, side: c_uint, uplo: c_uint, transa: c_uint, diag: c_uint,
            m: c_int, n: c_int, alpha: f64, a: *const f64, lda: c_int, b: *mut f64, ldb: c_int,
        );
    }
}

/// Tiles at or above this size on each dimension go to Accelerate; below it the library-call
/// overhead dominates and the register-blocked GEMM stays.
#[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
const ACCEL_MIN_DIM: usize = 48;

/// Tiles at or above this size route to the SME ZA engine; below it the `SMSTART`/`SMSTOP`
/// round-trip isn't worth it.
#[cfg(all(
    target_arch = "aarch64",
    not(hp_no_sme),
    any(not(target_vendor = "apple"), hp_no_apple_accelerate)
))]
const SME_MIN_DIM: usize = 16;

/// Scratch + packers for the SME blocked GEMM, internal to dispatch. A thread-local buffer is
/// reused across calls, so packing allocates nothing in steady state.
#[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(hp_no_sme),
    any(not(target_vendor = "apple"), hp_no_apple_accelerate)
))]
pub(crate) mod sme_pack {
    #![allow(clippy::needless_range_loop)]
    use crate::scalar::{FloatScalar, Scalar};
    use core::cell::RefCell;
    use half::{bf16, f16};
    thread_local! {
        pub static F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        pub static F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
        pub static BF16: RefCell<Vec<bf16>> = const { RefCell::new(Vec::new()) };
        pub static F16: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    }
    /// Pack A (`M×K` row-major at `ac`) into `pm` column-major `K×blk` panels via a NEON 4×4
    /// `vtrn` register transpose, blocked in 32-wide `K` tiles so the strided panel stores stay
    /// L1-resident. Rows go 8 at a time (two 4×4 blocks per `k`-step) for store ILP. `K % 4`
    /// columns fall to a scalar tail; `blk` (`svl/2 ∈ {16, 32, 64}`) is a multiple of 8, so the
    /// 4-row path is only a guard.
    /// # Safety
    /// `ac` is a valid `M×K` (`M = pm·blk`); `ap.len() >= pm·k·blk`.
    #[inline]
    pub unsafe fn pack_a_f32(ac: *const f32, ap: &mut [f32], pm: usize, k: usize, blk: usize) {
        use core::arch::aarch64::*;
        #[inline(always)]
        unsafe fn t4x4(col: *const f32, k: usize, j: usize, o: *mut f32, blk: usize) {
            unsafe {
                let (r0, r1) = (col.add(j), col.add(k + j));
                let (r2, r3) = (col.add(2 * k + j), col.add(3 * k + j));
                let (a0, a1, a2, a3) = (vld1q_f32(r0), vld1q_f32(r1), vld1q_f32(r2), vld1q_f32(r3));
                let (t0, t1) = (vtrn1q_f32(a0, a1), vtrn2q_f32(a0, a1));
                let (t2, t3) = (vtrn1q_f32(a2, a3), vtrn2q_f32(a2, a3));
                let (d0, d2) = (vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2));
                let (d1, d3) = (vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3));
                vst1q_f32(o, vreinterpretq_f32_f64(vtrn1q_f64(d0, d2)));
                vst1q_f32(o.add(blk), vreinterpretq_f32_f64(vtrn1q_f64(d1, d3)));
                vst1q_f32(o.add(2 * blk), vreinterpretq_f32_f64(vtrn2q_f64(d0, d2)));
                vst1q_f32(o.add(3 * blk), vreinterpretq_f32_f64(vtrn2q_f64(d1, d3)));
            }
        }
        let k4 = k & !3;
        let ap = ap.as_mut_ptr();
        for p in 0..pm {
            let base = p * k * blk;
            let mut kb = 0;
            while kb < k4 {
                let kend = (kb + 32).min(k4);
                let mut m = 0;
                while m + 8 <= blk {
                    let c0 = unsafe { ac.add((p * blk + m) * k) };
                    let c4 = unsafe { ac.add((p * blk + m + 4) * k) };
                    let mut j = kb;
                    while j < kend {
                        unsafe {
                            t4x4(c0, k, j, ap.add(base + j * blk + m), blk);
                            t4x4(c4, k, j, ap.add(base + j * blk + m + 4), blk);
                        }
                        j += 4;
                    }
                    m += 8;
                }
                while m < blk {
                    let col = unsafe { ac.add((p * blk + m) * k) };
                    let mut j = kb;
                    while j < kend {
                        unsafe { t4x4(col, k, j, ap.add(base + j * blk + m), blk) };
                        j += 4;
                    }
                    m += 4;
                }
                kb += 32;
            }
            for kk in k4..k {
                for m in 0..blk {
                    unsafe { *ap.add(base + kk * blk + m) = *ac.add((p * blk + m) * k + kk) };
                }
            }
        }
    }

    /// f64 counterpart of [`pack_a_f32`]: NEON 2×2 `vtrn` transpose, 32-tiled in `K`, rows 4 at a
    /// time for store ILP. `K % 2` falls to a scalar tail; `blk` (`svl/4 ∈ {8, 16, 32}`) is a
    /// multiple of 4, so the 2-row path is only a guard.
    /// # Safety
    /// `ac` is a valid `M×K` (`M = pm·blk`); `ap.len() >= pm·k·blk`.
    #[inline]
    pub unsafe fn pack_a_f64(ac: *const f64, ap: &mut [f64], pm: usize, k: usize, blk: usize) {
        use core::arch::aarch64::*;
        #[inline(always)]
        unsafe fn t2x2(col: *const f64, k: usize, j: usize, o: *mut f64, blk: usize) {
            unsafe {
                let (a0, a1) = (vld1q_f64(col.add(j)), vld1q_f64(col.add(k + j)));
                vst1q_f64(o, vtrn1q_f64(a0, a1));
                vst1q_f64(o.add(blk), vtrn2q_f64(a0, a1));
            }
        }
        let k2 = k & !1;
        let ap = ap.as_mut_ptr();
        for p in 0..pm {
            let base = p * k * blk;
            let mut kb = 0;
            while kb < k2 {
                let kend = (kb + 32).min(k2);
                let mut m = 0;
                while m + 4 <= blk {
                    let c0 = unsafe { ac.add((p * blk + m) * k) };
                    let c2 = unsafe { ac.add((p * blk + m + 2) * k) };
                    let mut j = kb;
                    while j < kend {
                        unsafe {
                            t2x2(c0, k, j, ap.add(base + j * blk + m), blk);
                            t2x2(c2, k, j, ap.add(base + j * blk + m + 2), blk);
                        }
                        j += 2;
                    }
                    m += 4;
                }
                while m < blk {
                    let col = unsafe { ac.add((p * blk + m) * k) };
                    let mut j = kb;
                    while j < kend {
                        unsafe { t2x2(col, k, j, ap.add(base + j * blk + m), blk) };
                        j += 2;
                    }
                    m += 2;
                }
                kb += 32;
            }
            for kk in k2..k {
                for m in 0..blk {
                    unsafe { *ap.add(base + kk * blk + m) = *ac.add((p * blk + m) * k + kk) };
                }
            }
        }
    }
    /// Pack B (`K×N` row-major at `bc`) into `pn` row-major `K×blk` panels. Each panel row is a
    /// contiguous `blk` run, so each copy is a `memcpy`.
    /// # Safety
    /// `bc` is a valid `K×N` (`N = pn·blk`); `bp.len() >= pn·k·blk`.
    #[inline]
    pub unsafe fn pack_b<T: Copy>(bc: *const T, bp: &mut [T], pn: usize, k: usize, n: usize, blk: usize) {
        for p in 0..pn {
            let base = p * k * blk;
            for kk in 0..k {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        bc.add(kk * n + p * blk),
                        bp.as_mut_ptr().add(base + kk * blk),
                        blk,
                    );
                }
            }
        }
    }

    /// Pair-pack A (`M×K` row-major 16-bit at `ac`) for the widening 2×2 grid: `pm` column panels,
    /// each `⌈K/2⌉` pair-rows of `blk×2`. The two `k`-neighbours `A[r][2pp]`,`A[r][2pp+1]` sit
    /// adjacent so one 2-vector `LD1H` per pair-step feeds both halves of the widening `FMOPA`.
    /// For even `K` a 16-bit pair is one 32-bit unit and the pair-transpose is exactly
    /// [`pack_a_f32`] over `M×(K/2)`, so it reuses that path. Odd `K` falls to a scalar gather,
    /// zero-padding the last neighbour. `T` is 16-bit.
    /// # Safety
    /// `ac` is a valid `M×K` (`M = pm·blk`); `ap.len() >= pm·⌈k/2⌉·blk·2`.
    #[inline]
    pub unsafe fn pack_a_pairs<T: FloatScalar>(ac: *const T, ap: &mut [T], pm: usize, k: usize, blk: usize) {
        debug_assert_eq!(core::mem::size_of::<T>(), 2);
        if k.is_multiple_of(2) {
            let apf = unsafe { core::slice::from_raw_parts_mut(ap.as_mut_ptr() as *mut f32, ap.len() / 2) };
            unsafe { pack_a_f32(ac as *const f32, apf, pm, k / 2, blk) };
            return;
        }
        let pairs = k.div_ceil(2);
        for pi in 0..pm {
            let base = pi * pairs * blk * 2;
            for pp in 0..pairs {
                let pbase = base + pp * blk * 2;
                for r in 0..blk {
                    let row = unsafe { ac.add((pi * blk + r) * k) };
                    let (k0, k1) = (2 * pp, 2 * pp + 1);
                    ap[pbase + r * 2] = unsafe { *row.add(k0) };
                    ap[pbase + r * 2 + 1] = if k1 < k { unsafe { *row.add(k1) } } else { T::ZERO };
                }
            }
        }
    }

    /// Pair-pack B (`K×N` row-major 16-bit at `bc`), companion of [`pack_a_pairs`]: `pn` row
    /// panels, each `⌈K/2⌉` pair-rows of `blk×2` with `B[2pp][c]`,`B[2pp+1][c]` adjacent. The two
    /// neighbours come from different B rows (`N` apart), so this is a NEON `zip` of two
    /// contiguous row runs; the odd final pair zero-pads its high half. `blk` (`svl/2`) is a
    /// multiple of 8. `T` is 16-bit.
    /// # Safety
    /// `bc` is a valid `K×N` (`N = pn·blk`); `bp.len() >= pn·⌈k/2⌉·blk·2`.
    #[inline]
    pub unsafe fn pack_b_pairs<T: FloatScalar>(bc: *const T, bp: &mut [T], pn: usize, k: usize, n: usize, blk: usize) {
        use core::arch::aarch64::*;
        debug_assert_eq!(core::mem::size_of::<T>(), 2);
        let pairs = k.div_ceil(2);
        let (bc, bp) = (bc as *const u16, bp.as_mut_ptr() as *mut u16);
        for pj in 0..pn {
            let base = pj * pairs * blk * 2;
            for pp in 0..pairs {
                let pbase = base + pp * blk * 2;
                let (k0, k1) = (2 * pp, 2 * pp + 1);
                let r0 = unsafe { bc.add(k0 * n + pj * blk) };
                if k1 < k {
                    let r1 = unsafe { bc.add(k1 * n + pj * blk) };
                    let mut c = 0;
                    while c + 8 <= blk {
                        unsafe {
                            let (v0, v1) = (vld1q_u16(r0.add(c)), vld1q_u16(r1.add(c)));
                            vst1q_u16(bp.add(pbase + c * 2), vzip1q_u16(v0, v1));
                            vst1q_u16(bp.add(pbase + c * 2 + 8), vzip2q_u16(v0, v1));
                        }
                        c += 8;
                    }
                } else {
                    for c in 0..blk {
                        unsafe {
                            *bp.add(pbase + c * 2) = *r0.add(c);
                            *bp.add(pbase + c * 2 + 1) = 0;
                        }
                    }
                }
            }
        }
    }
}

/// Blocked SME2 GEMM: tile `M×N` into `BLK×BLK` ZA-grid sub-tiles, each reducing the full `K`.
/// `ac`/`bc`/`c` are row-major in compute precision (strides `K`/`N`/`N`); the caller guarantees
/// `BLK` divides `M` and `N` and equals the runtime ZA-grid width.
#[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(hp_no_sme),
    any(not(target_vendor = "apple"), hp_no_apple_accelerate)
))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sme2_blocked_f32<const M: usize, const N: usize, const K: usize, const BLK: usize>(
    ac: *const f32,
    bc: *const f32,
    c: *mut f32,
) {
    // BLIS-style pack-once into a reused thread-local buffer; one streaming session per grid.
    let (pm, pn) = (M / BLK, N / BLK);
    sme_pack::F32.with_borrow_mut(|buf| {
        let need = M * K + K * N;
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        let (ap, bp) = buf.split_at_mut(M * K);
        unsafe {
            sme_pack::pack_a_f32(ac, ap, pm, K, BLK);
            sme_pack::pack_b(bc, &mut bp[..K * N], pn, K, N, BLK);
            crate::arch::sme2::mma_f32_grid_packed(ap.as_ptr(), bp.as_ptr(), c, N * 4, pm, pn, K);
        }
    });
}

#[cfg(all(
    target_arch = "aarch64",
    feature = "std",
    not(hp_no_sme),
    any(not(target_vendor = "apple"), hp_no_apple_accelerate)
))]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sme2_blocked_f64<const M: usize, const N: usize, const K: usize, const BLK: usize>(
    ac: *const f64,
    bc: *const f64,
    c: *mut f64,
) {
    // Pack-once, BLIS-style; see `sme2_blocked_f32`.
    let (pm, pn) = (M / BLK, N / BLK);
    sme_pack::F64.with_borrow_mut(|buf| {
        let need = M * K + K * N;
        if buf.len() < need {
            buf.resize(need, 0.0);
        }
        let (ap, bp) = buf.split_at_mut(M * K);
        unsafe {
            sme_pack::pack_a_f64(ac, ap, pm, K, BLK);
            sme_pack::pack_b(bc, &mut bp[..K * N], pn, K, N, BLK);
            crate::arch::sme2::mma_f64_grid_packed(ap.as_ptr(), bp.as_ptr(), c, N * 8, pm, pn, K);
        }
    });
}

/// Blocked SME2 widening GEMM for a 16-bit input type (`bf16`/`f16`) with an `f32` accumulator:
/// pair-pack `A`/`B` straight from the 16-bit operands (no `f32` widen pass, half the pack/load
/// bytes) and run the `BFMOPA` / FP16-widening-`FMOPA` grid. `c` is f32.
macro_rules! sme2_blocked_widen {
    ($name:ident, $t:ty, $tl:ident, $kern:path) => {
        #[cfg(all(
            target_arch = "aarch64",
            feature = "std",
            not(hp_no_sme),
            any(not(target_vendor = "apple"), hp_no_apple_accelerate)
        ))]
        #[inline]
        #[allow(unsafe_op_in_unsafe_fn)]
        unsafe fn $name<const M: usize, const N: usize, const K: usize, const BLK: usize>(
            a: *const $t,
            b: *const $t,
            c: *mut f32,
        ) {
            let (pm, pn) = (M / BLK, N / BLK);
            let pairs = K.div_ceil(2);
            let (asz, bsz) = (pm * pairs * BLK * 2, pn * pairs * BLK * 2);
            sme_pack::$tl.with_borrow_mut(|buf| {
                if buf.len() < asz + bsz {
                    buf.resize(asz + bsz, <$t>::ZERO);
                }
                let (ap, bp) = buf.split_at_mut(asz);
                unsafe {
                    sme_pack::pack_a_pairs(a, ap, pm, K, BLK);
                    sme_pack::pack_b_pairs(b, &mut bp[..bsz], pn, K, N, BLK);
                    $kern(ap.as_ptr(), bp.as_ptr(), c, N * 4, pm, pn, pairs);
                }
            });
        }
    };
}
sme2_blocked_widen!(sme2_blocked_bf16, half::bf16, BF16, crate::arch::sme2::mma_bf16_grid_packed);
sme2_blocked_widen!(sme2_blocked_f16, half::f16, F16, crate::arch::sme2::mma_f16_grid_packed);

/// `bf16` tiles at or above this size on each dimension, and within one AMX tile block
/// (`M, N ≤ 16`, `K ≤ 32`), route to the `tdpbf16ps` engine. Below it the tile-config/load/release
/// overhead isn't worth it.
#[cfg(all(target_arch = "x86_64", feature = "std", not(hp_no_amx)))]
const AMX_MIN_DIM: usize = 8;

/// `D = A·B + C` for `bf16` tiles via AVX-512-BF16 `vdpbf16ps`: `A`/`B` stay in `bf16`, `k`-pairs
/// accumulate into `f32` with the hardware packed dot-product. `N` is processed in whole 16-wide
/// column blocks; the caller gates on `N % 16 == 0`. Each `k`-pair broadcasts the `A` pair and
/// VNNI-packs the two `B` rows ([`crate::arch::avx512bf16`]); an odd final `k` folds in one FMA.
#[cfg(all(
    target_arch = "x86_64",
    not(any(hp_no_avx, hp_no_avx512)),
    any(feature = "std", hp_static_dispatch)
))]
#[target_feature(enable = "avx512bf16,avx512f,avx512bw")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn bf16_dpbf16_gemm<const M: usize, const N: usize, const K: usize>(
    a: &[[half::bf16; K]; M],
    b: &[[half::bf16; N]; K],
    mut c: [[f32; N]; M],
) -> [[f32; N]; M] {
    use crate::arch::avx512bf16 as p;
    use core::arch::x86_64::*;
    let kpairs = K / 2;
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j + 16 <= N {
            let mut acc = _mm512_loadu_ps(c[i].as_ptr().add(j));
            let mut t = 0;
            while t < kpairs {
                let (k0, k1) = (2 * t, 2 * t + 1);
                let av = p::bcast_pair(a[i][k0], a[i][k1]);
                let bv = p::pack_pair(b[k0].as_ptr().add(j), b[k1].as_ptr().add(j));
                acc = p::dp(acc, av, bv);
                t += 1;
            }
            if K & 1 == 1 {
                let k = K - 1;
                let av = _mm512_set1_ps(a[i][k].to_f32());
                let bv = p::widen(b[k].as_ptr().add(j));
                acc = _mm512_fmadd_ps(av, bv, acc);
            }
            _mm512_storeu_ps(c[i].as_mut_ptr().add(j), acc);
            j += 16;
        }
        i += 1;
    }
    c
}

/// `mma` for a SIMD backend: widen `A`/`B` from `T` to `T::Compute`, then run [`simd_gemm`] in
/// compute precision; `f32`/`f64` are pass-through.
///
/// The matrix-coprocessor path is chosen by platform, mutually exclusive by `target_vendor`:
/// Apple takes Accelerate `cblas_*gemm` for large tiles (`--cfg hp_no_apple_accelerate` drops the
/// framework and routes to the SME2 ZA engine instead); non-Apple aarch64 takes the SME ZA engine
/// when the CPU has it (`--cfg hp_no_sme` opts out). Below the per-path threshold, and everywhere
/// else, it's the inlinable register-blocked GEMM.
#[inline]
fn array_mma_simd<'i, T, S, const M: usize, const N: usize, const K: usize>(
    backend: S,
    a: View<'i, T, M, K>,
    b: View<'i, T, K, N>,
    c: [[T::Compute; N]; M],
) -> [[T::Compute; N]; M]
where
    T: FloatScalar,
    S: Backend<T::Compute>,
{
    // Hardware paths pack straight from the view's slice when dense row-major; otherwise, or when
    // `T` must widen to `T::Compute`, materialize `ac`/`bc` once and pack from those.
    let needs_widen = core::any::TypeId::of::<T>() != core::any::TypeId::of::<T::Compute>();
    let (a_dense, b_dense) = (a.dense_ptr(), b.dense_ptr());
    let needs_materialize = needs_widen || a_dense.is_none() || b_dense.is_none();
    let mut ac = [[<T::Compute as Scalar>::ZERO; K]; M];
    let mut bc = [[<T::Compute as Scalar>::ZERO; N]; K];
    let materialize = |ac: &mut [[T::Compute; K]; M], bc: &mut [[T::Compute; N]; K]| {
        let mut i = 0;
        while i < M {
            let mut k = 0;
            while k < K {
                ac[i][k] = a.get(i, k).widen();
                k += 1;
            }
            i += 1;
        }
        let mut k = 0;
        while k < K {
            let mut j = 0;
            while j < N {
                bc[k][j] = b.get(k, j).widen();
                j += 1;
            }
            k += 1;
        }
    };
    // bf16/f16 → f32 via the SME widening grid: pair-pack the 16-bit operands directly, no f32
    // widen pass. Dense, SME2, full BLK-multiple tiles only; anything else falls through.
    #[cfg(all(
        target_arch = "aarch64",
        feature = "std",
        not(hp_no_sme),
        any(not(target_vendor = "apple"), hp_no_apple_accelerate)
    ))]
    if M >= SME_MIN_DIM && N >= SME_MIN_DIM && K >= SME_MIN_DIM {
        use core::any::TypeId;
        let is_bf16 = TypeId::of::<T>() == TypeId::of::<half::bf16>();
        let is_f16 = TypeId::of::<T>() == TypeId::of::<half::f16>();
        if (is_bf16 || is_f16)
            && let (Some(adp), Some(bdp)) = (a_dense, b_dense)
            && crate::arch::sme1::is_supported()
            && crate::arch::sme2::is_supported()
        {
            let blk = crate::arch::sme1::streaming_vl_bytes() / 2;
            if (blk == 16 || blk == 32 || blk == 64) && M.is_multiple_of(blk) && N.is_multiple_of(blk) {
                let mut c = c;
                let cp = c.as_mut_ptr() as *mut f32;
                // SAFETY: `is_bf16`/`is_f16` imply `T::Compute == f32`, so the `c`/`adp`/`bdp`
                // reinterprets match the real types; inputs are dense row-major; SME2 present.
                unsafe {
                    if is_bf16 {
                        let (a, b) = (adp as *const half::bf16, bdp as *const half::bf16);
                        match blk {
                            16 => sme2_blocked_bf16::<M, N, K, 16>(a, b, cp),
                            32 => sme2_blocked_bf16::<M, N, K, 32>(a, b, cp),
                            _ => sme2_blocked_bf16::<M, N, K, 64>(a, b, cp),
                        }
                    } else {
                        let (a, b) = (adp as *const half::f16, bdp as *const half::f16);
                        match blk {
                            16 => sme2_blocked_f16::<M, N, K, 16>(a, b, cp),
                            32 => sme2_blocked_f16::<M, N, K, 32>(a, b, cp),
                            _ => sme2_blocked_f16::<M, N, K, 64>(a, b, cp),
                        }
                    }
                }
                return c;
            }
        }
    }

    if needs_materialize {
        materialize(&mut ac, &mut bc);
    }
    // Compute-precision source pointers: the view's own pointer when dense, else the materialized
    // array. When `!needs_materialize`, `T == T::Compute`, so the cast is identity.
    let cap: *const T::Compute = if needs_materialize {
        ac.as_ptr() as *const T::Compute
    } else {
        a_dense.unwrap() as _
    };
    let cbp: *const T::Compute = if needs_materialize {
        bc.as_ptr() as *const T::Compute
    } else {
        b_dense.unwrap() as _
    };
    let _ = (cap, cbp); // used by the cfg-gated hardware paths below

    // Apple: hand large tiles to Accelerate. `M`/`N`/`K` are const, so the threshold is a
    // compile-time choice. Accelerate computes `C := 1·A·B + 1·C = D`.
    #[cfg(all(target_vendor = "apple", not(hp_no_apple_accelerate)))]
    if M >= ACCEL_MIN_DIM && N >= ACCEL_MIN_DIM && K >= ACCEL_MIN_DIM {
        use core::any::TypeId;
        let compute = TypeId::of::<T::Compute>();
        if compute == TypeId::of::<f32>() {
            let (acp, bcp) = (cap as *const f32, cbp as *const f32);
            let mut c = c;
            unsafe {
                accel::cblas_sgemm(
                    accel::ROW_MAJOR, accel::NO_TRANS, accel::NO_TRANS,
                    M as _, N as _, K as _,
                    1.0, acp, K as _, bcp, N as _,
                    1.0, c.as_mut_ptr() as *mut f32, N as _,
                );
            }
            return c;
        }
        if compute == TypeId::of::<f64>() {
            let (acp, bcp) = (cap as *const f64, cbp as *const f64);
            let mut c = c;
            unsafe {
                accel::cblas_dgemm(
                    accel::ROW_MAJOR, accel::NO_TRANS, accel::NO_TRANS,
                    M as _, N as _, K as _,
                    1.0, acp, K as _, bcp, N as _,
                    1.0, c.as_mut_ptr() as *mut f64, N as _,
                );
            }
            return c;
        }
    }

    // Non-Apple aarch64: the SME ZA engine. Operands already carry `T::Compute`, so the f32/f64
    // cores apply directly. SME2 hosts take the wide 2×2 multi-vector kernel; SME-only hosts the
    // single tile. Tiles that overflow the ZA grid fall through to SIMD GEMM.
    #[cfg(all(
        target_arch = "aarch64",
        feature = "std",
        not(hp_no_sme),
        any(not(target_vendor = "apple"), hp_no_apple_accelerate)
    ))]
    if M >= SME_MIN_DIM && N >= SME_MIN_DIM && K >= SME_MIN_DIM && crate::arch::sme1::is_supported() {
        use core::any::TypeId;
        let svl = crate::arch::sme1::streaming_vl_bytes();
        let sme2 = crate::arch::sme2::is_supported();
        let compute = TypeId::of::<T::Compute>();
        if compute == TypeId::of::<f32>() {
            let (acp, bcp) = (cap as *const f32, cbp as *const f32);
            // Larger than one ZA tile: block M×N into svl/2-wide grid tiles, full K per tile.
            let blk = svl / 2;
            if sme2 && (blk == 16 || blk == 32 || blk == 64) && M.is_multiple_of(blk) && N.is_multiple_of(blk) {
                let mut c = c;
                let cp = c.as_mut_ptr() as *mut f32;
                unsafe {
                    match blk {
                        16 => sme2_blocked_f32::<M, N, K, 16>(acp, bcp, cp),
                        32 => sme2_blocked_f32::<M, N, K, 32>(acp, bcp, cp),
                        _ => sme2_blocked_f32::<M, N, K, 64>(acp, bcp, cp),
                    }
                }
                return c;
            }
            // f32 ZA tile is svl/4 wide; the 2×2 wide grid reaches svl/2.
            if sme2 && M <= svl / 2 && N <= svl / 2 {
                let mut c = c;
                unsafe {
                    crate::arch::sme2::mma_f32_wide::<M, N, K>(acp, K, bcp, N, c.as_mut_ptr() as *mut f32, N);
                }
                return c;
            }
            if M <= svl / 4 && N <= svl / 4 {
                let mut c = c;
                unsafe {
                    crate::arch::sme1::mma_f32::<M, N, K>(acp, K, bcp, N, c.as_mut_ptr() as *mut f32, N);
                }
                return c;
            }
        }
        if compute == TypeId::of::<f64>() {
            let (acp, bcp) = (cap as *const f64, cbp as *const f64);
            // Larger than one ZA tile: block M×N into svl/4-wide grid tiles, full K per tile.
            let blk = svl / 4;
            if sme2 && (blk == 8 || blk == 16 || blk == 32) && M.is_multiple_of(blk) && N.is_multiple_of(blk) {
                let mut c = c;
                let cp = c.as_mut_ptr() as *mut f64;
                unsafe {
                    match blk {
                        8 => sme2_blocked_f64::<M, N, K, 8>(acp, bcp, cp),
                        16 => sme2_blocked_f64::<M, N, K, 16>(acp, bcp, cp),
                        _ => sme2_blocked_f64::<M, N, K, 32>(acp, bcp, cp),
                    }
                }
                return c;
            }
            // f64 ZA tile is svl/8 wide; the wide grid reaches svl/4.
            if sme2 && M <= svl / 4 && N <= svl / 4 {
                let mut c = c;
                unsafe {
                    crate::arch::sme2::mma_f64_wide::<M, N, K>(acp, K, bcp, N, c.as_mut_ptr() as *mut f64, N);
                }
                return c;
            }
            if M <= svl / 8 && N <= svl / 8 {
                let mut c = c;
                unsafe {
                    crate::arch::sme1::mma_f64::<M, N, K>(acp, K, bcp, N, c.as_mut_ptr() as *mut f64, N);
                }
                return c;
            }
        }
    }

    // x86_64: a bf16 tile that fits one AMX block (M,N ≤ 16, K ≤ 32) takes the `tdpbf16ps` engine,
    // operands in bf16 with an f32 accumulator. Larger tiles fall through to the `vdpbf16ps` path
    // below. `--cfg hp_no_amx` opts out.
    #[cfg(all(target_arch = "x86_64", feature = "std", not(hp_no_amx)))]
    if M >= AMX_MIN_DIM && N >= AMX_MIN_DIM && K >= AMX_MIN_DIM && M <= 16 && N <= 16 && K <= 32 {
        use core::any::TypeId;
        if let (Some(adp), Some(bdp)) = (a_dense, b_dense)
            && TypeId::of::<T>() == TypeId::of::<half::bf16>()
            && TypeId::of::<T::Compute>() == TypeId::of::<f32>()
            && crate::arch::amx::is_supported()
        {
            // SAFETY: `T == bf16` and `T::Compute == f32` (TypeId-checked), inputs dense
            // row-major, so each reinterpret is an identity layout cast; AMX-BF16 is present with
            // tile-data permission. Strides are `K`/`N`/`N`.
            unsafe {
                let ab = &*(adp as *const [[half::bf16; K]; M]);
                let bb = &*(bdp as *const [[half::bf16; N]; K]);
                let mut cf = *(&c as *const [[T::Compute; N]; M] as *const [[f32; N]; M]);
                crate::arch::amx::mma_bf16::<M, N, K>(
                    ab.as_ptr().cast(), K, bb.as_ptr().cast(), N, cf.as_mut_ptr().cast(), N,
                );
                return *(&cf as *const [[f32; N]; M] as *const [[T::Compute; N]; M]);
            }
        }
    }

    // x86_64: the same AMX tile block for f16 via `tdpfp16ps`. AMX-FP16 is a distinct CPUID bit
    // from AMX-BF16, so `is_supported_f16` gates it separately; bf16-only AMX hosts fall through
    // to the widen paths below.
    #[cfg(all(target_arch = "x86_64", feature = "std", not(hp_no_amx)))]
    if M >= AMX_MIN_DIM && N >= AMX_MIN_DIM && K >= AMX_MIN_DIM && M <= 16 && N <= 16 && K <= 32 {
        use core::any::TypeId;
        if let (Some(adp), Some(bdp)) = (a_dense, b_dense)
            && TypeId::of::<T>() == TypeId::of::<half::f16>()
            && TypeId::of::<T::Compute>() == TypeId::of::<f32>()
            && crate::arch::amx::is_supported_f16()
        {
            // SAFETY: `T == f16` and `T::Compute == f32` (TypeId-checked), inputs dense row-major, so
            // each reinterpret is an identity layout cast; AMX-FP16 is present with tile-data permission.
            unsafe {
                let ab = &*(adp as *const [[half::f16; K]; M]);
                let bb = &*(bdp as *const [[half::f16; N]; K]);
                let mut cf = *(&c as *const [[T::Compute; N]; M] as *const [[f32; N]; M]);
                crate::arch::amx::mma_f16::<M, N, K>(
                    ab.as_ptr().cast(), K, bb.as_ptr().cast(), N, cf.as_mut_ptr().cast(), N,
                );
                return *(&cf as *const [[f32; N]; M] as *const [[T::Compute; N]; M]);
            }
        }
    }

    // x86_64: bf16 tiles on an AVX-512-BF16 host take the `vdpbf16ps` packed dot-product, gated
    // to whole 16-wide column blocks; other `N` fall through to the f32 SIMD GEMM below.
    #[cfg(all(
        target_arch = "x86_64",
        not(any(hp_no_avx, hp_no_avx512)),
        any(feature = "std", hp_static_dispatch)
    ))]
    if N >= 16 && N.is_multiple_of(16) && K >= 2 {
        use core::any::TypeId;
        // `hp_static_dispatch` reads the guaranteed `target_feature`s at compile time (no std needed).
        #[cfg(not(hp_static_dispatch))]
        let avx512bf16 = is_x86_feature_detected!("avx512bf16")
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw");
        #[cfg(hp_static_dispatch)]
        let avx512bf16 = cfg!(all(
            target_feature = "avx512bf16",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ));
        if let (Some(adp), Some(bdp)) = (a_dense, b_dense)
            && TypeId::of::<T>() == TypeId::of::<half::bf16>()
            && TypeId::of::<T::Compute>() == TypeId::of::<f32>()
            && avx512bf16
        {
            // SAFETY: `T == bf16` and `T::Compute == f32` (TypeId-checked), inputs dense row-major, so
            // each reinterpret is an identity layout cast; the CPU has avx512bf16+f+bw (detected above).
            unsafe {
                let ab = &*(adp as *const [[half::bf16; K]; M]);
                let bb = &*(bdp as *const [[half::bf16; N]; K]);
                let cf = *(&c as *const [[T::Compute; N]; M] as *const [[f32; N]; M]);
                let out = bf16_dpbf16_gemm::<M, N, K>(ab, bb, cf);
                return *(&out as *const [[f32; N]; M] as *const [[T::Compute; N]; M]);
            }
        }
    }

    // No hardware path took the tile; the array GEMMs need the materialized arrays.
    if !needs_materialize {
        materialize(&mut ac, &mut bc);
    }
    // Large tiles (B overflows L1) take the B-packed GEMM; below this the pack doesn't pay.
    #[cfg(feature = "std")]
    if M >= 32 && N >= 32 && K >= 32 {
        // Size the MR×NR block from the cached per-core unroll factor, capped to fit the SIMD
        // register file with scratch to spare (`5*NR + 1 <= REGS - 2`).
        const REGS: usize = if cfg!(target_arch = "aarch64") || cfg!(target_feature = "avx512f") {
            32
        } else {
            16
        };
        let k = crate::varying::Gang::new(backend).unroll_for::<T::Compute>();
        let by_regs = (REGS - 3) / 5;
        let nr = k.div_ceil(4).clamp(2, by_regs).min(4);
        return match nr {
            2 => packed_gemm::<_, _, M, N, K, 4, 2>(backend, ac, bc, c),
            3 => packed_gemm::<_, _, M, N, K, 4, 3>(backend, ac, bc, c),
            _ => packed_gemm::<_, _, M, N, K, 4, 4>(backend, ac, bc, c),
        };
    }
    simd_gemm(backend, ac, bc, c)
}

/// `mma` for the GPU `Subgroup` floor: each invocation computes the whole tile scalar-wise. The
/// subgroup `load` is per-invocation (one element), not a SIMD register, so [`simd_gemm`]'s
/// lane-blocking would be wrong here.
#[cfg(target_arch = "spirv")]
#[inline]
fn array_mma_scalar<'i, T, const M: usize, const N: usize, const K: usize>(
    a: View<'i, T, M, K>,
    b: View<'i, T, K, N>,
    mut c: [[T::Compute; N]; M],
) -> [[T::Compute; N]; M]
where
    T: FloatScalar,
{
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < N {
            let mut s = c[i][j];
            let mut k = 0;
            while k < K {
                s = a.get(i, k).widen().fma(b.get(k, j).widen(), s);
                k += 1;
            }
            c[i][j] = s;
            j += 1;
        }
        i += 1;
    }
    c
}

/// The tile load/store/splat/map methods (and `Tile = [[E; C]; R]`), shared by both `mma` variants.
macro_rules! array_tile_methods {
    () => {
        type Tile<'a, E: Scalar, const R: usize, const C: usize, Ro: Role> =
            <Ro as $crate::matrix::Role>::Repr<'a, E, R, C>;

        #[inline]
        fn tile_load<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            mem: &'a [E],
            row_stride: usize,
            layout: Layout,
        ) -> <Ro as $crate::matrix::Role>::Repr<'a, E, R, C> {
            $crate::matrix::CpuTile::ct_load(mem, row_stride, layout)
        }

        #[inline]
        fn tile_store<E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            t: <Ro as $crate::matrix::Role>::Repr<'_, E, R, C>,
            out: &mut [E],
            row_stride: usize,
            layout: Layout,
        ) {
            $crate::matrix::CpuTile::ct_store(t, out, row_stride, layout)
        }

        #[inline]
        fn tile_splat<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            v: E,
        ) -> <Ro as $crate::matrix::Role>::Repr<'a, E, R, C> {
            $crate::matrix::CpuTile::ct_splat(v)
        }

        #[inline]
        fn tile_map<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            t: <Ro as $crate::matrix::Role>::Repr<'a, E, R, C>,
            f: impl Fn(E) -> E,
        ) -> <Ro as $crate::matrix::Role>::Repr<'a, E, R, C> {
            $crate::matrix::CpuTile::ct_map(t, f)
        }
    };
}

/// Implements [`MatrixBackend`] with the `[[E; C]; R]` tile. `simd` uses the vectorized GEMM
/// (CPU tokens + scalar oracle); `scalar` uses the per-invocation triple loop (GPU `Subgroup`).
macro_rules! impl_array_matrix_backend {
    ($backend:ty, simd) => {
        impl<T: FloatScalar> $crate::matrix::MatrixBackend<T> for $backend
        where
            $backend: $crate::backend::Backend<T> + $crate::backend::Backend<<T as FloatScalar>::Compute>,
        {
            array_tile_methods!();

            #[inline]
            fn mma<'i, const M: usize, const N: usize, const K: usize>(
                self,
                a: Self::Tile<'i, T, M, K, $crate::matrix::MatrixA>,
                b: Self::Tile<'i, T, K, N, $crate::matrix::MatrixB>,
                c: Self::Tile<'i, <T as FloatScalar>::Compute, M, N, $crate::matrix::Accumulator>,
            ) -> Self::Tile<'i, <T as FloatScalar>::Compute, M, N, $crate::matrix::Accumulator> {
                array_mma_simd::<T, $backend, M, N, K>(self, a, b, c)
            }
        }
    };
    ($backend:ty, scalar) => {
        impl<T: FloatScalar> $crate::matrix::MatrixBackend<T> for $backend
        where
            $backend: $crate::backend::Backend<T>,
        {
            array_tile_methods!();

            #[inline]
            fn mma<'i, const M: usize, const N: usize, const K: usize>(
                self,
                a: Self::Tile<'i, T, M, K, $crate::matrix::MatrixA>,
                b: Self::Tile<'i, T, K, N, $crate::matrix::MatrixB>,
                c: Self::Tile<'i, <T as FloatScalar>::Compute, M, N, $crate::matrix::Accumulator>,
            ) -> Self::Tile<'i, <T as FloatScalar>::Compute, M, N, $crate::matrix::Accumulator> {
                array_mma_scalar::<T, M, N, K>(a, b, c)
            }
        }
    };
}

impl_array_matrix_backend!(crate::backend::ScalarBackend, simd);

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl_array_matrix_backend!(crate::backend::avx1::Avx1, simd);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl_array_matrix_backend!(crate::backend::avx2::Avx2, simd);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl_array_matrix_backend!(crate::backend::sse4::Sse4, simd);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl_array_matrix_backend!(crate::backend::avx512::Avx512, simd);
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
impl_array_matrix_backend!(crate::backend::avx512fp16::Avx512Fp16, simd);
#[cfg(target_arch = "aarch64")]
impl_array_matrix_backend!(crate::backend::neon::Neon, simd);
// armv7 NEON is f32-only: the macro's `Backend<T> + Backend<T::Compute>` bound resolves this for
// `MatrixBackend<f32>` only; f64 matmul takes the scalar path.
#[cfg(target_arch = "arm")]
impl_array_matrix_backend!(crate::backend::neon_a32::Neon, simd);
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
impl_array_matrix_backend!(crate::backend::wasm::Simd128, simd);
#[cfg(all(target_arch = "wasm32", target_feature = "relaxed-simd"))]
impl_array_matrix_backend!(crate::backend::wasm::RelaxedSimd, simd);

// The SVE tokens carry a const generic, so they can't go through the macro. `mma` routes through
// `array_mma_simd`, so on SME hosts it lands on the ZA engine and on plain SVE it's the
// register-blocked GEMM over the SVE element-wise ops.
#[cfg(target_arch = "aarch64")]
impl<const W: usize, T: FloatScalar> MatrixBackend<T> for crate::backend::sve::Sve<W>
where
    crate::backend::sve::Sve<W>:
        crate::backend::Backend<T> + crate::backend::Backend<<T as FloatScalar>::Compute>,
{
    array_tile_methods!();

    #[inline]
    fn mma<'i, const M: usize, const N: usize, const K: usize>(
        self,
        a: Self::Tile<'i, T, M, K, MatrixA>,
        b: Self::Tile<'i, T, K, N, MatrixB>,
        c: Self::Tile<'i, <T as FloatScalar>::Compute, M, N, Accumulator>,
    ) -> Self::Tile<'i, <T as FloatScalar>::Compute, M, N, Accumulator> {
        array_mma_simd::<T, crate::backend::sve::Sve<W>, M, N, K>(self, a, b, c)
    }
}

// RVV tokens also carry a const generic, so they go around the macro. No matrix engine (the
// RISC-V matrix extension isn't ratified), so `mma` is the register-blocked GEMM over RVV ops.
#[cfg(target_arch = "riscv64")]
impl<const W: usize, T: FloatScalar> MatrixBackend<T> for crate::backend::rvv::Rvv<W>
where
    crate::backend::rvv::Rvv<W>:
        crate::backend::Backend<T> + crate::backend::Backend<<T as FloatScalar>::Compute>,
{
    array_tile_methods!();

    #[inline]
    fn mma<'i, const M: usize, const N: usize, const K: usize>(
        self,
        a: Self::Tile<'i, T, M, K, MatrixA>,
        b: Self::Tile<'i, T, K, N, MatrixB>,
        c: Self::Tile<'i, <T as FloatScalar>::Compute, M, N, Accumulator>,
    ) -> Self::Tile<'i, <T as FloatScalar>::Compute, M, N, Accumulator> {
        array_mma_simd::<T, crate::backend::rvv::Rvv<W>, M, N, K>(self, a, b, c)
    }
}

// GPU non-coop fallback; the cooperative-matrix lowering layers on top once rust-gpu emits
// OpCooperativeMatrixMulAddKHR.
#[cfg(target_arch = "spirv")]
impl_array_matrix_backend!(crate::backend::subgroup::Subgroup, scalar);

use core::marker::PhantomData;

use crate::varying::Gang;

/// The tile/MMA surface reached from a [`Gang`] context via [`Gang::tiles`]. The matrix analogue
/// of building [`Varying`](crate::Varying)s through `Gang`.
#[derive(Clone, Copy)]
pub struct Tiles<T: FloatScalar, S: MatrixBackend<T>> {
    backend: S,
    _t: PhantomData<T>,
}

/// An ergonomic tile value: carries the backend token (like [`Varying`](crate::Varying)) so a kernel
/// chains `load`/`mma`/`map`/`store` without naming the backend.
#[derive(Clone, Copy)]
pub struct Tile<
    'a,
    T: FloatScalar,
    S: MatrixBackend<T>,
    E: Scalar,
    const R: usize,
    const C: usize,
    Ro: Role,
> {
    backend: S,
    inner: S::Tile<'a, E, R, C, Ro>,
    _p: PhantomData<(T, Ro)>,
}

impl<S: Copy> Gang<S> {
    /// Gateway to the tile / matrix-multiply surface for element `T` (usually inferred from the
    /// tile loads that follow).
    #[inline(always)]
    pub fn tiles<T: FloatScalar>(self) -> Tiles<T, S>
    where
        S: MatrixBackend<T>,
    {
        Tiles {
            backend: self.backend(),
            _t: PhantomData,
        }
    }
}

impl<T: FloatScalar, S: MatrixBackend<T>> Tiles<T, S> {
    /// Load the `M×K` left operand `A` from memory. The returned tile borrows `mem` for `'a`; on
    /// a dense CPU tile this is a zero-copy view and any copy or widen is deferred to `mma`.
    #[inline]
    pub fn load_a<'a, const M: usize, const K: usize>(
        self,
        mem: &'a [T],
        row_stride: usize,
        layout: Layout,
    ) -> Tile<'a, T, S, T, M, K, MatrixA> {
        Tile {
            backend: self.backend,
            inner: self.backend.tile_load(mem, row_stride, layout),
            _p: PhantomData,
        }
    }

    /// Load the `K×N` right operand `B` from memory (zero-copy view; see [`load_a`](Tiles::load_a)).
    #[inline]
    pub fn load_b<'a, const K: usize, const N: usize>(
        self,
        mem: &'a [T],
        row_stride: usize,
        layout: Layout,
    ) -> Tile<'a, T, S, T, K, N, MatrixB> {
        Tile {
            backend: self.backend,
            inner: self.backend.tile_load(mem, row_stride, layout),
            _p: PhantomData,
        }
    }

    /// Load the `M×N` accumulator `C` from memory (owned; the accumulator is read-modify-written).
    #[inline]
    pub fn load_acc<'a, const M: usize, const N: usize>(
        self,
        mem: &'a [T::Compute],
        row_stride: usize,
        layout: Layout,
    ) -> Tile<'a, T, S, T::Compute, M, N, Accumulator> {
        Tile {
            backend: self.backend,
            inner: self.backend.tile_load(mem, row_stride, layout),
            _p: PhantomData,
        }
    }

    /// Load a contiguous row-major `M×K` `A` tile (`row_stride = K`, the common case). For a
    /// sub-tile of a larger matrix (stride > `K`) or column-major data, use [`load_a`](Tiles::load_a).
    #[inline]
    pub fn load_a_rm<'a, const M: usize, const K: usize>(
        self,
        mem: &'a [T],
    ) -> Tile<'a, T, S, T, M, K, MatrixA> {
        self.load_a::<M, K>(mem, K, Layout::RowMajor)
    }

    /// Load a contiguous row-major `K×N` `B` tile (`row_stride = N`).
    #[inline]
    pub fn load_b_rm<'a, const K: usize, const N: usize>(
        self,
        mem: &'a [T],
    ) -> Tile<'a, T, S, T, K, N, MatrixB> {
        self.load_b::<K, N>(mem, N, Layout::RowMajor)
    }

    /// Load a contiguous row-major `M×N` accumulator (`row_stride = N`).
    #[inline]
    pub fn load_acc_rm<'a, const M: usize, const N: usize>(
        self,
        mem: &'a [T::Compute],
    ) -> Tile<'a, T, S, T::Compute, M, N, Accumulator> {
        self.load_acc::<M, N>(mem, N, Layout::RowMajor)
    }

    /// A zeroed `M×N` accumulator. Owned, so it carries whatever lifetime `mma` unifies it to.
    #[inline]
    pub fn zero_acc<'a, const M: usize, const N: usize>(
        self,
    ) -> Tile<'a, T, S, T::Compute, M, N, Accumulator> {
        self.splat_acc(<T::Compute as Scalar>::ZERO)
    }

    /// An `M×N` accumulator with every element set to `v`.
    #[inline]
    pub fn splat_acc<'a, const M: usize, const N: usize>(
        self,
        v: T::Compute,
    ) -> Tile<'a, T, S, T::Compute, M, N, Accumulator> {
        Tile {
            backend: self.backend,
            inner: self.backend.tile_splat(v),
            _p: PhantomData,
        }
    }

    /// `D = A·B + C`. The operands and accumulator share one lifetime `'i`; loaded in the same
    /// kernel scope, the borrowed inputs and owned accumulator unify naturally.
    #[inline]
    pub fn mma<'i, const M: usize, const N: usize, const K: usize>(
        self,
        a: Tile<'i, T, S, T, M, K, MatrixA>,
        b: Tile<'i, T, S, T, K, N, MatrixB>,
        c: Tile<'i, T, S, T::Compute, M, N, Accumulator>,
    ) -> Tile<'i, T, S, T::Compute, M, N, Accumulator> {
        Tile {
            backend: self.backend,
            inner: self.backend.mma(a.inner, b.inner, c.inner),
            _p: PhantomData,
        }
    }
}

impl<'a, T: FloatScalar, S: MatrixBackend<T>, E: Scalar, const R: usize, const C: usize, Ro: Role>
    Tile<'a, T, S, E, R, C, Ro>
{
    /// The raw backend tile.
    #[inline(always)]
    pub fn raw(self) -> S::Tile<'a, E, R, C, Ro> {
        self.inner
    }

    /// Store the tile to memory.
    #[inline]
    pub fn store(self, out: &mut [E], row_stride: usize, layout: Layout) {
        self.backend.tile_store(self.inner, out, row_stride, layout);
    }

    /// Store the tile contiguously row-major (`row_stride = C`, the tile's column count). The
    /// companion to the `*_rm` loaders for the common dense case.
    #[inline]
    pub fn store_rm(self, out: &mut [E]) {
        self.backend.tile_store(self.inner, out, C, Layout::RowMajor);
    }

    /// Store with a fused element-wise epilogue `f` (scale, bias, clamp, activation) folded into
    /// the writeback, avoiding a second pass. Position-independent like [`map`](Tile::map): `f`
    /// must not depend on `(row, col)`.
    #[inline]
    pub fn store_ex(self, out: &mut [E], row_stride: usize, layout: Layout, f: impl Fn(E) -> E) {
        self.map(f).store(out, row_stride, layout);
    }

    /// Row-major [`store_ex`](Tile::store_ex) (`row_stride = C`).
    #[inline]
    pub fn store_rm_ex(self, out: &mut [E], f: impl Fn(E) -> E) {
        self.map(f).store_rm(out);
    }

    /// Apply a position-independent function to every element (activation / bias / scale). On the
    /// GPU the element→`(row, col)` mapping is opaque, so `f` must not depend on position.
    #[inline]
    pub fn map(self, f: impl Fn(E) -> E) -> Self {
        Tile {
            backend: self.backend,
            inner: self.backend.tile_map(self.inner, f),
            _p: PhantomData,
        }
    }
}

/// A matmul kernel — like [`Kernel`](crate::Kernel) but its `run` receives a context whose backend
/// also supports tiles. Run with [`run_matrix_scalar`] (oracle) or `dispatch`-style selection.
pub trait MatrixKernel<T: FloatScalar> {
    type Output;
    fn run<S: MatrixBackend<T>>(self, ctx: Gang<S>) -> Self::Output;
}

/// Run a matmul kernel on the always-available scalar backend (correctness oracle / baseline).
#[inline]
pub fn run_matrix_scalar<T: FloatScalar, K: MatrixKernel<T>>(kernel: K) -> K::Output {
    kernel.run(Gang::new(crate::backend::ScalarBackend))
}

/// Per-scalar dispatch policy for matmul kernels — the [`SimdDispatch`](crate::SimdDispatch)
/// analogue for [`MatrixKernel`].
pub trait MatrixDispatch: FloatScalar {
    fn dispatch_matrix<K: MatrixKernel<Self>>(kernel: K) -> K::Output;
}

/// Run `kernel` on the best available backend for `T`, chosen by runtime CPU detection.
#[inline]
pub fn dispatch_matrix<T: MatrixDispatch, K: MatrixKernel<T>>(kernel: K) -> K::Output {
    T::dispatch_matrix(kernel)
}

macro_rules! impl_matrix_dispatch_simd {
    ($ty:ty $(, $arm_tail:ident)?) => {
        impl MatrixDispatch for $ty {
            #[inline]
            #[allow(unreachable_code)]
            fn dispatch_matrix<K: MatrixKernel<Self>>(kernel: K) -> K::Output {
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    feature = "std",
                    not(hp_static_dispatch)
                ))]
                {
                    #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                    if let Some(b) = crate::backend::avx512::Avx512::detect() {
                        return kernel.run(Gang::new(b));
                    }
                    #[cfg(not(hp_no_avx))]
                    if let Some(b) = crate::backend::avx2::Avx2::detect() {
                        return kernel.run(Gang::new(b));
                    }
                    #[cfg(not(hp_no_avx))]
                    if let Some(b) = crate::backend::avx1::Avx1::detect() {
                        return kernel.run(Gang::new(b));
                    }
                    if let Some(b) = crate::backend::sse4::Sse4::detect() {
                        return kernel.run(Gang::new(b));
                    }
                }
                // Compile-time selection (no-std, or `hp_static_dispatch` on std): the widest
                // `target_feature`-guaranteed tier surviving the `hp_no_avx*` cfgs, no branch.
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    target_feature = "avx512f",
                    not(any(hp_no_avx, hp_no_avx512))
                ))]
                {
                    // SAFETY: target compiled with avx512f.
                    let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    target_feature = "avx2",
                    target_feature = "fma",
                    not(hp_no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx2+fma.
                    let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    not(all(target_feature = "avx2", target_feature = "fma", not(hp_no_avx))),
                    target_feature = "avx",
                    not(hp_no_avx)
                ))]
                {
                    // SAFETY: target compiled with avx.
                    let b = unsafe { crate::backend::avx1::Avx1::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                #[cfg(all(
                    any(target_arch = "x86_64", target_arch = "x86"),
                    any(not(feature = "std"), hp_static_dispatch),
                    not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                    not(all(target_feature = "avx2", target_feature = "fma", not(hp_no_avx))),
                    not(all(target_feature = "avx", not(hp_no_avx))),
                    target_feature = "sse4.1"
                ))]
                {
                    // SAFETY: target compiled with sse4.1.
                    let b = unsafe { crate::backend::sse4::Sse4::new_unchecked() };
                    return kernel.run(Gang::new(b));
                }
                // aarch64: non-Apple SVE token (its `mma` lands on the SME ZA engine where
                // present), else NEON; Apple NEON delegates large tiles to Accelerate.
                crate::dispatch::aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
                crate::dispatch::riscv_dispatch_tail!(kernel);
                $( crate::dispatch::$arm_tail!(kernel); )?
                crate::dispatch::wasm_dispatch_tail!(kernel);
                kernel.run(Gang::new(crate::backend::ScalarBackend))
            }
        }
    };
}

impl_matrix_dispatch_simd!(f32, arm_dispatch_tail);
impl_matrix_dispatch_simd!(f64);

mod half_matrix_dispatch {
    use super::{MatrixDispatch, MatrixKernel, Gang};
    use half::{bf16, f16};

    // f16 matmul widens to f32, so the AVX2 F16C backend applies (`Backend<f16>` + `Backend<f32>`
    // give it `MatrixBackend<f16>`). The native AVX-512-FP16 backend is `Backend<f16>`-only (no
    // f32 accumulate), so it serves element-wise dispatch but not `mma`.
    impl MatrixDispatch for f16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch_matrix<K: MatrixKernel<Self>>(kernel: K) -> K::Output {
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(hp_static_dispatch),
                not(hp_no_avx)
            ))]
            {
                if let Some(b) = crate::backend::avx2::Avx2::detect() {
                    return kernel.run(Gang::new(b));
                }
            }
            // Compile-time AVX2 F16C widen tile (no-std, or `hp_static_dispatch` on std).
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                target_feature = "avx2",
                target_feature = "fma",
                target_feature = "f16c",
                not(hp_no_avx)
            ))]
            {
                // SAFETY: target compiled with avx2+fma+f16c.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            crate::dispatch::aarch64_dispatch_tail!(kernel, crate::backend::ScalarBackend);
            kernel.run(Gang::new(crate::backend::ScalarBackend))
        }
    }

    // bf16 uses the widen-path tile backend; the AMX / AVX-512-BF16 fast paths layer into `mma`.
    impl MatrixDispatch for bf16 {
        #[inline]
        #[allow(unreachable_code)]
        fn dispatch_matrix<K: MatrixKernel<Self>>(kernel: K) -> K::Output {
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                feature = "std",
                not(hp_static_dispatch)
            ))]
            {
                #[cfg(not(any(hp_no_avx, hp_no_avx512)))]
                if let Some(b) = crate::backend::avx512::Avx512::detect() {
                    return kernel.run(Gang::new(b));
                }
                #[cfg(not(hp_no_avx))]
                if let Some(b) = crate::backend::avx2::Avx2::detect() {
                    return kernel.run(Gang::new(b));
                }
            }
            // Compile-time bf16 widen tile: AVX-512 if the build guarantees `avx512f`, else AVX2.
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                target_feature = "avx512f",
                not(any(hp_no_avx, hp_no_avx512))
            ))]
            {
                // SAFETY: target compiled with avx512f.
                let b = unsafe { crate::backend::avx512::Avx512::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            #[cfg(all(
                any(target_arch = "x86_64", target_arch = "x86"),
                any(not(feature = "std"), hp_static_dispatch),
                not(all(target_feature = "avx512f", not(any(hp_no_avx, hp_no_avx512)))),
                target_feature = "avx2",
                target_feature = "fma",
                not(hp_no_avx)
            ))]
            {
                // SAFETY: target compiled with avx2+fma.
                let b = unsafe { crate::backend::avx2::Avx2::new_unchecked() };
                return kernel.run(Gang::new(b));
            }
            crate::dispatch::aarch64_dispatch_tail!(kernel, crate::backend::neon::Neon::new());
            crate::dispatch::wasm_dispatch_tail!(kernel);
            kernel.run(Gang::new(crate::backend::ScalarBackend))
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod packed_parity {
    use crate::{Backend, Gang, Kernel, dispatch};

    const M: usize = 9;
    const N: usize = 22;
    const K: usize = 7;

    #[allow(clippy::needless_range_loop, clippy::type_complexity)]
    fn data() -> ([[f32; K]; M], [[f32; N]; K], [[f32; N]; M]) {
        let mut a = [[0.0f32; K]; M];
        let mut b = [[0.0f32; N]; K];
        let mut c = [[0.0f32; N]; M];
        for i in 0..M {
            for k in 0..K {
                a[i][k] = ((i * K + k) as f32 * 0.13).sin();
            }
        }
        for k in 0..K {
            for j in 0..N {
                b[k][j] = ((k * N + j) as f32 * 0.07).cos();
            }
        }
        for i in 0..M {
            for j in 0..N {
                c[i][j] = ((i + j) as f32 * 0.01) - 0.5;
            }
        }
        (a, b, c)
    }

    struct Probe;
    impl Kernel<f32> for Probe {
        type Output = ();
        fn run<S: crate::backend::BackendAll + Backend<f32>>(self, ctx: Gang<S>) {
            let be = ctx.backend();
            let (a, b, c) = data();
            // Every register-block width the dispatcher can pick shares `simd_gemm`'s per-element
            // `fma` order, so the results must be bit-equal, not merely close.
            let want = super::simd_gemm::<f32, S, M, N, K>(be, a, b, c);
            assert_eq!(super::packed_gemm::<_, _, M, N, K, 4, 2>(be, a, b, c), want);
            assert_eq!(super::packed_gemm::<_, _, M, N, K, 4, 3>(be, a, b, c), want);
            assert_eq!(super::packed_gemm::<_, _, M, N, K, 4, 4>(be, a, b, c), want);
        }
    }

    #[test]
    fn packed_matches_simd_every_block_width() {
        dispatch(Probe);
    }
}
