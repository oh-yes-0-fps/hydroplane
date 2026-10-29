//! Sparse Cholesky on the CPU's SME matrix units (Apple M4 and later), in this crate's factor
//! layout: the [`Analysis`]'s supernodes, factored by Accelerate's dense kernels.
//!
//! Accelerate's own sparse Cholesky ([`crate::accelerate`]) is the same dense kernels (SME2
//! outer products into ZA tiles, one matrix unit per CPU cluster) under its own ordering and an
//! opaque factor. Factoring in the [`Analysis`]'s layout instead gives a factor that the faer
//! solve and the GPU solve read, at Accelerate's speed.
//!
//! The factorization is multifrontal and runs as one task graph over the elimination tree: a
//! supernode starts when its last child finishes, adds the children's update blocks into its own
//! columns and its own update block (one writer per column, so no locks), factors its columns,
//! and forms its update block for its parent. A small supernode is one task. A large one is
//! many: its columns as a graph of 128-wide tiles (`potrf` on each diagonal tile, `trsm` below
//! it, `syrk`/`gemm` updates of the tiles to its right, each task started the moment its inputs
//! are final), then its update block in 256-wide blocks over the full width. That is how
//! Accelerate's own sparse Cholesky keeps all three matrix units busy: an M5 Pro has one unit per
//! CPU cluster, two of them a third as fast as the third, and only small tasks let the slow ones
//! take half the work without a straggler holding the critical path (a blocked `potrf` and
//! row-tiled `trsm` of the largest supernodes ran 1.3–1.8 times slower). The worker threads are
//! user-interactive and call the dense kernels single-threaded.

use std::ffi::{c_char, c_int};
use std::sync::OnceLock;

use rayon::prelude::*;

use crate::cholesky::{Analysis, Error, Real};

/// Whether the CPU has SME2 (`hw.optional.arm.FEAT_SME2`).
pub fn available() -> bool {
    static SME2: OnceLock<bool> = OnceLock::new();
    *SME2.get_or_init(|| {
        let mut value: c_int = 0;
        let mut len = size_of::<c_int>();
        let name = c"hw.optional.arm.FEAT_SME2";
        let rc = unsafe { sysctlbyname(name.as_ptr(), (&raw mut value).cast(), &mut len, std::ptr::null_mut(), 0) };
        rc == 0 && value != 0
    })
}

const COL_MAJOR: c_int = 102;
const NO_TRANS: c_int = 111;
const TRANS: c_int = 112;
const LOWER: c_int = 122;
const NON_UNIT: c_int = 131;
const RIGHT: c_int = 142;
/// `BLAS_THREADING_SINGLE_THREADED`.
const SINGLE_THREADED: c_int = 1;
/// `QOS_CLASS_USER_INTERACTIVE`: lower classes are kept off the matrix units' fast clusters.
const USER_INTERACTIVE: u32 = 0x21;

unsafe extern "C" {
    fn sysctlbyname(name: *const c_char, old: *mut std::ffi::c_void, len: *mut usize, new: *mut std::ffi::c_void, new_len: usize) -> c_int;
    fn pthread_set_qos_class_self_np(qos: u32, priority: c_int) -> c_int;
    fn BLASSetThreading(threading: c_int) -> c_int;
    fn dtrtri_(uplo: *const c_char, diag: *const c_char, n: *const c_int, a: *mut f64, lda: *const c_int, info: *mut c_int) -> c_int;
    fn strtri_(uplo: *const c_char, diag: *const c_char, n: *const c_int, a: *mut f32, lda: *const c_int, info: *mut c_int) -> c_int;
    fn cblas_dtrsv(order: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: *const f64, lda: c_int, x: *mut f64, incx: c_int);
    fn cblas_strsv(order: c_int, uplo: c_int, trans: c_int, diag: c_int, n: c_int, a: *const f32, lda: c_int, x: *mut f32, incx: c_int);
    fn cblas_dgemv(order: c_int, trans: c_int, m: c_int, n: c_int, alpha: f64, a: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int);
    fn cblas_sgemv(order: c_int, trans: c_int, m: c_int, n: c_int, alpha: f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int);
    fn dpotrf_(uplo: *const c_char, n: *const c_int, a: *mut f64, lda: *const c_int, info: *mut c_int) -> c_int;
    fn spotrf_(uplo: *const c_char, n: *const c_int, a: *mut f32, lda: *const c_int, info: *mut c_int) -> c_int;
    fn cblas_dtrsm(order: c_int, side: c_int, uplo: c_int, trans: c_int, diag: c_int, m: c_int, n: c_int, alpha: f64, a: *const f64, lda: c_int, b: *mut f64, ldb: c_int);
    fn cblas_strsm(order: c_int, side: c_int, uplo: c_int, trans: c_int, diag: c_int, m: c_int, n: c_int, alpha: f32, a: *const f32, lda: c_int, b: *mut f32, ldb: c_int);
    fn cblas_dsyrk(order: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f64, a: *const f64, lda: c_int, beta: f64, c: *mut f64, ldc: c_int);
    fn cblas_ssyrk(order: c_int, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: f32, a: *const f32, lda: c_int, beta: f32, c: *mut f32, ldc: c_int);
    fn cblas_dgemm(order: c_int, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int, beta: f64, c: *mut f64, ldc: c_int);
    fn cblas_sgemm(order: c_int, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int, beta: f32, c: *mut f32, ldc: c_int);
}

/// `f32` or `f64`: the dense kernels in that precision (BLAS's argument lists, hence their
/// length).
#[allow(clippy::too_many_arguments)]
pub trait SmeReal: Real + Default + private::Sealed {
    const PRECISION: crate::Precision;
    /// `Some` for `f64`, whose kernels write the `f64` update blocks directly.
    #[doc(hidden)]
    fn wide(u: *mut f64) -> Option<*mut Self>;
    #[doc(hidden)]
    unsafe fn potrf(n: usize, a: *mut Self, lda: usize) -> c_int;
    /// `A ← A⁻¹`, `A` lower `n × n`.
    #[doc(hidden)]
    unsafe fn trtri(n: usize, a: *mut Self, lda: usize) -> c_int;
    /// `x ← A⁻¹ x` (or `A⁻ᵀ x`), `A` lower `n × n`.
    #[doc(hidden)]
    unsafe fn trsv(transpose: bool, n: usize, a: *const Self, lda: usize, x: *mut Self);
    /// `y ← y − A x` (or `y − Aᵀ x`), `A` `m × n`.
    #[doc(hidden)]
    unsafe fn gemv_sub(transpose: bool, m: usize, n: usize, a: *const Self, lda: usize, x: *const Self, y: *mut Self);
    /// `B ← B A⁻ᵀ`, `B` `m × n`, `A` lower `n × n`.
    #[doc(hidden)]
    unsafe fn trsm(m: usize, n: usize, a: *const Self, lda: usize, b: *mut Self, ldb: usize);
    /// `C ← C − A Aᵀ`, lower, `C` `n × n`, `A` `n × k`.
    #[doc(hidden)]
    unsafe fn syrk(n: usize, k: usize, a: *const Self, lda: usize, c: *mut Self, ldc: usize);
    /// `C ← alpha A B`, `C` `m × n`, `A` `m × k`, `B` `k × n`.
    #[doc(hidden)]
    unsafe fn gemm_nn(m: usize, n: usize, k: usize, alpha: f64, a: *const Self, lda: usize, b: *const Self, ldb: usize, c: *mut Self, ldc: usize);
    /// `C ← C − A Bᵀ`, `C` `m × n`, `A` `m × k`, `B` `n × k`.
    #[doc(hidden)]
    unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, lda: usize, b: *const Self, ldb: usize, c: *mut Self, ldc: usize);
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

macro_rules! sme_real {
    ($t:ty, $p:ident, $trsv:ident, $gemv:ident, $trtri:ident, $potrf:ident, $trsm:ident, $syrk:ident, $gemm:ident) => {
        impl SmeReal for $t {
            const PRECISION: crate::Precision = crate::Precision::$p;
            fn wide(u: *mut f64) -> Option<*mut Self> {
                (crate::Precision::$p == crate::Precision::F64).then_some(u.cast())
            }
            unsafe fn potrf(n: usize, a: *mut Self, lda: usize) -> c_int {
                let (n, lda, mut info) = (n as c_int, lda as c_int, 0);
                unsafe { $potrf(c"L".as_ptr(), &n, a, &lda, &mut info) };
                info
            }
            unsafe fn trsv(transpose: bool, n: usize, a: *const Self, lda: usize, x: *mut Self) {
                unsafe { $trsv(COL_MAJOR, LOWER, if transpose { TRANS } else { NO_TRANS }, NON_UNIT, n as c_int, a, lda as c_int, x, 1) }
            }
            unsafe fn gemv_sub(transpose: bool, m: usize, n: usize, a: *const Self, lda: usize, x: *const Self, y: *mut Self) {
                unsafe { $gemv(COL_MAJOR, if transpose { TRANS } else { NO_TRANS }, m as c_int, n as c_int, -1.0, a, lda as c_int, x, 1, 1.0, y, 1) }
            }
            unsafe fn trtri(n: usize, a: *mut Self, lda: usize) -> c_int {
                let (n, lda, mut info) = (n as c_int, lda as c_int, 0);
                unsafe { $trtri(c"L".as_ptr(), c"N".as_ptr(), &n, a, &lda, &mut info) };
                info
            }
            unsafe fn trsm(m: usize, n: usize, a: *const Self, lda: usize, b: *mut Self, ldb: usize) {
                unsafe { $trsm(COL_MAJOR, RIGHT, LOWER, TRANS, NON_UNIT, m as c_int, n as c_int, 1.0, a, lda as c_int, b, ldb as c_int) }
            }
            unsafe fn syrk(n: usize, k: usize, a: *const Self, lda: usize, c: *mut Self, ldc: usize) {
                unsafe { $syrk(COL_MAJOR, LOWER, NO_TRANS, n as c_int, k as c_int, -1.0, a, lda as c_int, 1.0, c, ldc as c_int) }
            }
            unsafe fn gemm_nn(m: usize, n: usize, k: usize, alpha: f64, a: *const Self, lda: usize, b: *const Self, ldb: usize, c: *mut Self, ldc: usize) {
                unsafe { $gemm(COL_MAJOR, NO_TRANS, NO_TRANS, m as c_int, n as c_int, k as c_int, alpha as $t, a, lda as c_int, b, ldb as c_int, 0.0, c, ldc as c_int) }
            }
            unsafe fn gemm(m: usize, n: usize, k: usize, a: *const Self, lda: usize, b: *const Self, ldb: usize, c: *mut Self, ldc: usize) {
                unsafe { $gemm(COL_MAJOR, NO_TRANS, TRANS, m as c_int, n as c_int, k as c_int, -1.0, a, lda as c_int, b, ldb as c_int, 1.0, c, ldc as c_int) }
            }
        }
    };
}

sme_real!(f64, F64, cblas_dtrsv, cblas_dgemv, dtrtri_, dpotrf_, cblas_dtrsm, cblas_dsyrk, cblas_dgemm);
sme_real!(f32, F32, cblas_strsv, cblas_sgemv, strtri_, spotrf_, cblas_strsm, cblas_ssyrk, cblas_sgemm);

/// The worker threads: user-interactive, each with Accelerate's kernels single-threaded.
fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .thread_name(|i| format!("hydroplane-sme-{i}"))
            .stack_size(64 << 20)
            .start_handler(|_| unsafe {
                pthread_set_qos_class_self_np(USER_INTERACTIVE, 0);
                BLASSetThreading(SINGLE_THREADED);
            })
            .build()
            .expect("a thread pool")
    })
}

/// Tile width of a large supernode's column task graph. Accelerate uses 96; on the slab tangent
/// 96 to 256 measured the same (65.5–66.5 ms in `f64`).
const NB: usize = 128;
/// Rows and columns per block of a large supernode's update block (192–384 the same; 512 2 ms
/// slower).
const OB: usize = 256;
/// Supernodes with more work than this (flops) are split into tasks; the slab tangent measured
/// the same from 1e7 to 3e8.
const SPLIT_FLOPS: f64 = 5e7;

/// The symbolic part: the supernodes' shapes, children, where each child's update rows land in
/// its parent, and where each entry of the matrix lands in the factor.
struct Plan {
    /// Columns `k` and rows `f` of each supernode's block.
    k: Vec<usize>,
    f: Vec<usize>,
    offset: Vec<usize>,
    parent: Vec<usize>,
    children: Vec<Vec<usize>>,
    leaves: Vec<usize>,
    /// For each row of a supernode's pattern, its row in the parent's block (`0 .. f_parent`).
    rel: Vec<Vec<u32>>,
    /// The matrix's stored values by supernode: `source[a_ptr[s] .. a_ptr[s + 1]]` are indices
    /// into the values, `target[..]` their positions in the factor.
    a_ptr: Vec<usize>,
    source: Vec<u32>,
    target: Vec<usize>,
    factor_len: usize,
    /// Offset of each supernode's diagonal-block inverse (`k × k`) in the inverses.
    inverse_offset: Vec<usize>,
    /// Each supernode's first column, and the rows below its diagonal block.
    begin: Vec<usize>,
    pattern: Vec<Vec<u32>>,
    /// `perm[new] = old`.
    perm: Vec<usize>,
    roots: Vec<usize>,
}

impl Plan {
    fn new(analysis: &Analysis) -> Result<Self, Error> {
        let sn = analysis.supernodal();
        let ns = sn.n_supernodes();
        let n = analysis.dim();
        let begin: Vec<usize> = sn.supernode_begin()[..ns].to_vec();
        let end: Vec<usize> = sn.supernode_end()[..ns].to_vec();
        let k: Vec<usize> = (0..ns).map(|s| end[s] - begin[s]).collect();
        let f: Vec<usize> = (0..ns).map(|s| k[s] + sn.supernode(s).pattern().len()).collect();
        let offset: Vec<usize> = sn.col_ptr_for_val()[..ns].to_vec();
        let parent = analysis.parents();
        let mut children = vec![Vec::new(); ns];
        for s in 0..ns {
            if parent[s] != usize::MAX {
                children[parent[s]].push(s);
            }
        }
        let leaves = (0..ns).filter(|&s| children[s].is_empty()).collect();
        let rel = (0..ns)
            .map(|s| {
                let pattern = sn.supernode(s).pattern();
                if pattern.is_empty() {
                    return Ok(Vec::new());
                }
                let p = parent[s];
                let below = sn.supernode(p).pattern();
                let mut j = 0;
                pattern
                    .iter()
                    .map(|&r| {
                        if r < end[p] {
                            Ok((r - begin[p]) as u32)
                        } else {
                            while j < below.len() && below[j] < r {
                                j += 1;
                            }
                            if j < below.len() && below[j] == r { Ok((k[p] + j) as u32) } else { Err(Error::Pattern) }
                        }
                    })
                    .collect()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut col_super = vec![0usize; n];
        for s in 0..ns {
            col_super[begin[s]..end[s]].fill(s);
        }
        let mut inverse = vec![0usize; n];
        for (new, &old) in analysis.perm().iter().enumerate() {
            inverse[old] = new;
        }
        let (col_ptr, row_idx) = (analysis.col_ptr(), analysis.row_idx());
        let mut entries: Vec<(usize, u32, usize)> = Vec::with_capacity(row_idx.len());
        for j in 0..n {
            for p in col_ptr[j]..col_ptr[j + 1] {
                let (a, b) = (inverse[row_idx[p]], inverse[j]);
                let (r, c) = if a >= b { (a, b) } else { (b, a) };
                let s = col_super[c];
                let row = if r < end[s] { r - begin[s] } else { k[s] + sn.supernode(s).pattern().binary_search(&r).map_err(|_| Error::Pattern)? };
                entries.push((s, p as u32, offset[s] + (c - begin[s]) * f[s] + row));
            }
        }
        entries.sort_unstable_by_key(|e| e.0);
        let mut a_ptr = vec![0usize; ns + 1];
        for e in &entries {
            a_ptr[e.0 + 1] += 1;
        }
        for s in 0..ns {
            a_ptr[s + 1] += a_ptr[s];
        }
        let source = entries.iter().map(|e| e.1).collect();
        let target = entries.iter().map(|e| e.2).collect();
        let inverse_offset = k.iter().scan(0, |at, &k| { let o = *at; *at += k * k; Some(o) }).chain(std::iter::once(k.iter().map(|k| k * k).sum())).collect();
        let pattern = (0..ns).map(|s| sn.supernode(s).pattern().iter().map(|&r| r as u32).collect()).collect();
        let roots = (0..ns).filter(|&s| parent[s] == usize::MAX).collect();
        Ok(Self { k, f, offset, parent, children, leaves, rel, a_ptr, source, target, factor_len: analysis.factor_len(), inverse_offset, begin, pattern, perm: analysis.perm(), roots })
    }

    /// Flops of supernode `s`'s own factorization and update block.
    fn work(&self, s: usize) -> f64 {
        let (k, m) = (self.k[s] as f64, (self.f[s] - self.k[s]) as f64);
        k * k * k / 3.0 + k * k * m + k * m * m
    }
}

/// A raw pointer that may cross threads: the factorization writes through it to places no
/// other task touches at the same time (a supernode's own block, disjoint columns or tiles).
#[derive(Clone, Copy)]
struct Shared<T>(*mut T);

unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

impl<T> Shared<T> {
    fn at(self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

/// Sparse Cholesky in precision `T` on the SME units, in the [`Analysis`]'s layout.
pub struct SmeCholesky<T: SmeReal> {
    plan: Plan,
    factor: Vec<T>,
    /// Each supernode's diagonal-block inverse `L₁₁⁻¹` (lower, `k × k` column-major, one after
    /// another), when asked for.
    inverse: Option<Vec<T>>,
    /// Update blocks kept between factorizations, so the large ones are not mapped and zeroed by
    /// the kernel anew each time.
    buffers: std::sync::Mutex<Vec<Vec<f64>>>,
}

impl<T: SmeReal> SmeCholesky<T> {
    pub fn new(analysis: &Analysis) -> Result<Self, Error> {
        if !available() {
            return Err(Error::Device("no SME2 on this CPU".into()));
        }
        let plan = Plan::new(analysis)?;
        let factor = vec![T::default(); plan.factor_len];
        Ok(Self { plan, factor, inverse: None, buffers: std::sync::Mutex::new(Vec::new()) })
    }

    /// Also invert each supernode's diagonal block during factorizations: what a GPU solve
    /// needs besides the factor (`vulkan::VulkanCholesky::load_factor_and_inverse`),
    /// done here as tasks beside the factorization's own, instead of on the device afterwards.
    pub fn with_inverses(mut self) -> Self {
        self.inverse = Some(vec![T::default(); *self.plan.inverse_offset.last().unwrap_or(&0)]);
        self
    }

    /// The diagonal-block inverses of the last factorization ([`Self::with_inverses`]).
    pub fn inverse(&self) -> Option<&[T]> {
        self.inverse.as_deref()
    }

    /// The factor in the analysis's (`faer`'s supernodal) layout.
    pub fn factor(&self) -> &[T] {
        &self.factor
    }

    /// Factors the matrix whose lower-triangle values, in the pattern's order, are `values`.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
        assert_eq!(values.len(), self.plan.source.len());
        let ns = self.plan.k.len();
        let run = Run {
            plan: &self.plan,
            values,
            l: Shared(self.factor.as_mut_ptr()),
            inverse: self.inverse.as_mut().map(|v| Shared(v.as_mut_ptr())),
            updates: (0..ns).map(|_| Slot(std::cell::UnsafeCell::new(Vec::new()))).collect(),
            pending: self.plan.children.iter().map(|c| std::sync::atomic::AtomicUsize::new(c.len())).collect(),
            buffers: &self.buffers,
            panels: (0..ns).map(|_| OnceLock::new()).collect(),
            failed: std::sync::atomic::AtomicBool::new(false),
        };
        pool().install(|| {
            rayon::scope(|scope| {
                for &s in &run.plan.leaves {
                    let run = &run;
                    scope.spawn(move |scope| run.start(scope, s));
                }
            })
        });
        for slot in &run.updates {
            let block = std::mem::take(unsafe { &mut *slot.0.get() });
            run.give(block);
        }
        if run.failed.load(std::sync::atomic::Ordering::Relaxed) { Err(Error::NotPositiveDefinite) } else { Ok(()) }
    }

    /// `rhs ← A⁻¹ rhs` with the last factorization, in `T`: forward up the tree, each supernode
    /// solving its diagonal block (`trsv`) and handing its parent a vector of contributions to
    /// the rows below it (`gemv`; the parent adds them, so siblings never write the same entry),
    /// then backward down it, each supernode gathering its ancestors' finished unknowns.
    pub fn solve_in_place(&self, rhs: &mut [f64]) {
        let plan = &self.plan;
        let n = plan.perm.len();
        assert_eq!(rhs.len(), n);
        let mut x: Vec<T> = plan.perm.iter().map(|&old| T::from_f64(rhs[old])).collect();
        let xs = Shared(x.as_mut_ptr());
        let l = Shared(self.factor.as_ptr().cast_mut());
        let ns = plan.k.len();
        let pending: Vec<std::sync::atomic::AtomicUsize> = plan.children.iter().map(|c| std::sync::atomic::AtomicUsize::new(c.len())).collect();
        let outgoing: Vec<VecSlot<T>> = (0..ns).map(|_| VecSlot(std::cell::UnsafeCell::new(Vec::new()))).collect();
        let (pending, outgoing) = (&pending, &outgoing);
        pool().install(|| {
            rayon::scope(|scope| {
                fn forward<'s, T: SmeReal>(scope: &rayon::Scope<'s>, plan: &'s Plan, l: Shared<T>, xs: Shared<T>, pending: &'s [std::sync::atomic::AtomicUsize], outgoing: &'s [VecSlot<T>], s: usize) {
                    let (k, f, b) = (plan.k[s], plan.f[s], plan.begin[s]);
                    let m = f - k;
                    let block = l.at(plan.offset[s]);
                    let mut w = vec![T::default(); m];
                    unsafe {
                        for &c in &plan.children[s] {
                            let child = std::mem::take(&mut *outgoing[c].0.get());
                            for (i, &t) in plan.rel[c].iter().enumerate() {
                                let t = t as usize;
                                if t < k {
                                    *xs.at(b + t) = T::from_f64((*xs.at(b + t)).to_f64() + child[i].to_f64());
                                } else {
                                    w[t - k] = T::from_f64(w[t - k].to_f64() + child[i].to_f64());
                                }
                            }
                        }
                        T::trsv(false, k, block, f, xs.at(b));
                        if m > 0 {
                            T::gemv_sub(false, m, k, block.add(k), f, xs.at(b), w.as_mut_ptr());
                        }
                        *outgoing[s].0.get() = w;
                    }
                    let p = plan.parent[s];
                    if p != usize::MAX && pending[p].fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                        scope.spawn(move |scope| forward(scope, plan, l, xs, pending, outgoing, p));
                    }
                }
                for &s in &plan.leaves {
                    scope.spawn(move |scope| forward(scope, plan, l, xs, pending, outgoing, s));
                }
            });
            rayon::scope(|scope| {
                fn backward<'s, T: SmeReal>(scope: &rayon::Scope<'s>, plan: &'s Plan, l: Shared<T>, xs: Shared<T>, s: usize) {
                    let (k, f, b) = (plan.k[s], plan.f[s], plan.begin[s]);
                    let m = f - k;
                    let block = l.at(plan.offset[s]);
                    unsafe {
                        if m > 0 {
                            let below: Vec<T> = plan.pattern[s].iter().map(|&r| *xs.at(r as usize)).collect();
                            T::gemv_sub(true, m, k, block.add(k), f, below.as_ptr(), xs.at(b));
                        }
                        T::trsv(true, k, block, f, xs.at(b));
                    }
                    for &c in &plan.children[s] {
                        scope.spawn(move |scope| backward(scope, plan, l, xs, c));
                    }
                }
                for &s in &plan.roots {
                    scope.spawn(move |scope| backward(scope, plan, l, xs, s));
                }
            });
        });
        for (new, &old) in plan.perm.iter().enumerate() {
            rhs[old] = x[new].to_f64();
        }
    }
}

/// A supernode's forward contributions, written by its task and taken by its parent's.
struct VecSlot<T>(std::cell::UnsafeCell<Vec<T>>);

unsafe impl<T> Sync for VecSlot<T> {}

/// A supernode's update block, written by its own tasks and read by its parent's after the
/// child-count handshake.
struct Slot(std::cell::UnsafeCell<Vec<f64>>);

unsafe impl Sync for Slot {}

/// One factorization in flight.
struct Run<'a, T: SmeReal> {
    plan: &'a Plan,
    values: &'a [f64],
    l: Shared<T>,
    inverse: Option<Shared<T>>,
    updates: Vec<Slot>,
    /// Children of each supernode not yet finished.
    pending: Vec<std::sync::atomic::AtomicUsize>,
    buffers: &'a std::sync::Mutex<Vec<Vec<f64>>>,
    /// The task graphs of the large supernodes, made when each starts.
    panels: Vec<OnceLock<Box<Panel<'a, T>>>>,
    failed: std::sync::atomic::AtomicBool,
}

type Scope<'s> = rayon::Scope<'s>;

impl<'a, T: SmeReal> Run<'a, T> {
    fn start(&'a self, scope: &Scope<'a>, s: usize) {
        if self.failed.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        if self.plan.work(s) > SPLIT_FLOPS {
            self.assemble(s, true);
            Panel::start(self, scope, s);
        } else {
            self.assemble(s, false);
            let (k, f) = (self.plan.k[s], self.plan.f[s]);
            let m = f - k;
            let block = Shared(self.l.at(self.plan.offset[s]));
            if unsafe { T::potrf(k, block.0, f) } != 0 {
                return self.fail();
            }
            if m > 0 {
                unsafe { T::trsm(m, k, block.0, f, block.at(k), f) };
                let u = Shared(unsafe { (*self.updates[s].0.get()).as_mut_ptr() });
                update_tile(Shared(block.at(k)), f, k, u, m, 0, 0, m, m);
            }
            self.invert(s);
            self.finish(scope, s);
        }
    }

    /// `L₁₁⁻¹` of supernode `s`, once its columns are final, if inverses are wanted.
    fn invert(&self, s: usize) {
        let Some(inverse) = self.inverse else { return };
        let (k, f) = (self.plan.k[s], self.plan.f[s]);
        let out = Shared(inverse.at(self.plan.inverse_offset[s]));
        if !invert_lower(k, Shared(self.l.at(self.plan.offset[s])), f, out, k) {
            self.fail();
        }
    }

    fn fail(&self) {
        self.failed.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Supernode `s`'s update block is complete: its parent starts if it was the last child.
    fn finish(&'a self, scope: &Scope<'a>, s: usize) {
        let p = self.plan.parent[s];
        if p != usize::MAX && self.pending[p].fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            scope.spawn(move |scope| self.start(scope, p));
        }
    }

    /// Supernode `s`'s block zeroed and given its values of the matrix and its children's update
    /// blocks (which go back to the pool), and its own update block zeroed.
    fn assemble(&self, s: usize, parallel: bool) {
        let (k, f) = (self.plan.k[s], self.plan.f[s]);
        let m = f - k;
        let off = self.plan.offset[s];
        let block = Shared(self.l.at(off));
        let len = f * k;
        let zero = |v: &mut [T]| v.fill(T::default());
        let mine = unsafe { std::slice::from_raw_parts_mut(block.0, len) };
        if parallel { mine.par_chunks_mut(1 << 15).for_each(zero) } else { zero(mine) }
        let (from, to) = (self.plan.a_ptr[s], self.plan.a_ptr[s + 1]);
        for (&src, &dst) in self.plan.source[from..to].iter().zip(&self.plan.target[from..to]) {
            unsafe { *self.l.at(dst) = T::from_f64(self.values[src as usize]) };
        }
        let mut update = self.take(m * m, parallel);
        let u = Shared(update.as_mut_ptr());
        for &c in &self.plan.children[s] {
            let child = std::mem::take(unsafe { &mut *self.updates[c].0.get() });
            extend_add(&self.plan.rel[c], &child, block, f, k, u, m, parallel);
            self.give(child);
        }
        unsafe { *self.updates[s].0.get() = update };
    }

    /// A zeroed buffer of `len` from the pool (the smallest that fits), else a new one.
    fn take(&self, len: usize, parallel: bool) -> Vec<f64> {
        let mut v = {
            let mut pool = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
            match pool.iter().enumerate().filter(|(_, b)| b.capacity() >= len).min_by_key(|(_, b)| b.capacity()).map(|(i, _)| i) {
                Some(i) => pool.swap_remove(i),
                None => Vec::new(),
            }
        };
        v.clear();
        v.reserve_exact(len);
        let spare = &mut v.spare_capacity_mut()[..len];
        let zero = |c: &mut [std::mem::MaybeUninit<f64>]| c.fill(std::mem::MaybeUninit::new(0.0));
        if parallel { spare.par_chunks_mut(1 << 15).for_each(zero) } else { zero(spare) }
        unsafe { v.set_len(len) };
        v
    }

    fn give(&self, v: Vec<f64>) {
        if v.capacity() >= 1 << 12 {
            self.buffers.lock().unwrap_or_else(|e| e.into_inner()).push(v);
        }
    }
}

/// A large supernode's columns as a task graph over 128-wide tiles, then its update block in
/// 256-wide blocks. Tile rows are the column tiles' (`0, NB, …, k`) and then the rows below
/// (`k, k + NB, …, f`); tasks are `potrf` of diagonal tile `j`, `trsm` of tile `(i, j)`, and the
/// update of tile `(i, l)` by column tile `j < l`. Each tile's updates are applied in order of
/// `j`, the last one releasing its `potrf` or `trsm`.
struct Panel<'a, T: SmeReal> {
    run: &'a Run<'a, T>,
    s: usize,
    block: Shared<T>,
    f: usize,
    k: usize,
    /// Tile row boundaries; the first `cols + 1` are also the column tiles'.
    rows: Vec<usize>,
    cols: usize,
    /// Unfinished inputs of `potrf(j)`, `trsm(i, j)` and `update(i, l, j)`.
    potrf_deps: Vec<std::sync::atomic::AtomicU32>,
    trsm_deps: Vec<std::sync::atomic::AtomicU32>,
    update_deps: Vec<std::sync::atomic::AtomicU32>,
    /// Column tasks not yet done, then update blocks not yet done.
    remaining: std::sync::atomic::AtomicUsize,
}

impl<'a, T: SmeReal> Panel<'a, T> {
    fn start(run: &'a Run<'a, T>, scope: &Scope<'a>, s: usize) {
        use std::sync::atomic::AtomicU32;
        let (k, f) = (run.plan.k[s], run.plan.f[s]);
        let mut rows: Vec<usize> = (0..k).step_by(NB).collect();
        let cols = rows.len();
        rows.extend((k..f).step_by(NB));
        rows.push(f);
        let r = rows.len() - 1;
        let potrf_deps = (0..cols).map(|j| AtomicU32::new(u32::from(j > 0))).collect();
        let trsm_deps = (0..r * cols).map(|x| AtomicU32::new(1 + u32::from(x % cols > 0))).collect();
        let update_deps = (0..r * cols * cols)
            .map(|x| {
                let (i, l, j) = (x / (cols * cols), x / cols % cols, x % cols);
                AtomicU32::new(if i == l { 1 } else { 2 } + u32::from(j > 0) * u32::from(j < l && l <= i))
            })
            .collect();
        let tasks = cols + (0..cols).map(|j| r - 1 - j).sum::<usize>() + (0..cols).map(|j| (j + 1..cols).map(|l| r - l).sum::<usize>()).sum::<usize>();
        let panel: &'a Panel<'a, T> = run.panels[s].get_or_init(|| Box::new(Panel {
            run,
            s,
            block: Shared(run.l.at(run.plan.offset[s])),
            f,
            k,
            rows,
            cols,
            potrf_deps,
            trsm_deps,
            update_deps,
            remaining: std::sync::atomic::AtomicUsize::new(tasks),
        }));
        scope.spawn(move |scope| panel.potrf(scope, 0));
    }

    fn rows(&self, i: usize) -> (usize, usize) {
        (self.rows[i], self.rows[i + 1] - self.rows[i])
    }

    fn tile(&self, i: usize, j: usize) -> *mut T {
        self.block.at(self.rows[j] * self.f + self.rows[i])
    }

    fn release(&'a self, scope: &Scope<'a>, deps: &std::sync::atomic::AtomicU32, task: impl FnOnce(&Scope<'a>) + Send + 'a) {
        if deps.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            scope.spawn(task);
        }
    }

    fn done(&'a self, scope: &Scope<'a>) {
        if self.remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
            self.update(scope);
        }
    }

    fn potrf(&'a self, scope: &Scope<'a>, j: usize) {
        let (_, w) = self.rows(j);
        if self.run.failed.load(std::sync::atomic::Ordering::Relaxed) || unsafe { T::potrf(w, self.tile(j, j), self.f) } != 0 {
            return self.run.fail();
        }
        let r = self.rows.len() - 1;
        for i in j + 1..r {
            self.release(scope, &self.trsm_deps[i * self.cols + j], move |scope| self.trsm(scope, i, j));
        }
        self.done(scope);
    }

    fn trsm(&'a self, scope: &Scope<'a>, i: usize, j: usize) {
        if self.run.failed.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let ((_, h), (_, w)) = (self.rows(i), self.rows(j));
        unsafe { T::trsm(h, w, self.tile(j, j), self.f, self.tile(i, j), self.f) };
        let c = self.cols;
        for l in j + 1..=i.min(c - 1) {
            self.release(scope, &self.update_deps[(i * c + l) * c + j], move |scope| self.update_tile(scope, i, l, j));
        }
        if i < c {
            for i2 in i + 1..self.rows.len() - 1 {
                self.release(scope, &self.update_deps[(i2 * c + i) * c + j], move |scope| self.update_tile(scope, i2, i, j));
            }
        }
        self.done(scope);
    }

    /// Tile `(i, l)` less tile `(i, j)` times tile `(l, j)`ᵀ.
    fn update_tile(&'a self, scope: &Scope<'a>, i: usize, l: usize, j: usize) {
        if self.run.failed.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let ((_, h), (_, w), (_, d)) = (self.rows(i), self.rows(l), self.rows(j));
        unsafe {
            if i == l {
                T::syrk(w, d, self.tile(l, j), self.f, self.tile(l, l), self.f);
            } else {
                T::gemm(h, w, d, self.tile(i, j), self.f, self.tile(l, j), self.f, self.tile(i, l), self.f);
            }
        }
        let c = self.cols;
        if j + 1 < l {
            self.release(scope, &self.update_deps[(i * c + l) * c + j + 1], move |scope| self.update_tile(scope, i, l, j + 1));
        } else if i == l {
            self.release(scope, &self.potrf_deps[l], move |scope| self.potrf(scope, l));
        } else {
            self.release(scope, &self.trsm_deps[i * c + l], move |scope| self.trsm(scope, i, l));
        }
        self.done(scope);
    }

    /// The columns are final: the update block in 256-wide blocks over the full width, the last
    /// one finishing the supernode.
    fn update(&'a self, scope: &Scope<'a>) {
        if self.run.inverse.is_some() {
            scope.spawn(move |_| self.run.invert(self.s));
        }
        let m = self.f - self.k;
        if m == 0 {
            return self.run.finish(scope, self.s);
        }
        let tiles = m.div_ceil(OB);
        let blocks: Vec<(usize, usize)> = (0..tiles).flat_map(|c| (c..tiles).map(move |r| (c, r))).collect();
        self.remaining.store(blocks.len(), std::sync::atomic::Ordering::Release);
        let u = Shared(unsafe { (*self.run.updates[self.s].0.get()).as_mut_ptr() });
        let below = Shared(self.block.at(self.k));
        for (c, r) in blocks {
            scope.spawn(move |scope| {
                let (c0, r0) = (c * OB, r * OB);
                update_tile(below, self.f, self.k, u, m, r0, c0, (m - r0).min(OB), (m - c0).min(OB));
                if self.remaining.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                    self.run.finish(scope, self.s);
                }
            });
        }
    }
}

/// Blocks at most this wide are inverted by one `trtri`; wider ones by halves in parallel.
const INVERT_LEAF: usize = 256;

/// `X ← L⁻¹` for the `k × k` lower triangle `L` at `a` (leading dimension `lda`), into `out`
/// (leading dimension `ldo`, its strict upper triangle zeroed). Wide blocks by halves: with
/// `L = [A 0; B C]`, `L⁻¹ = [A⁻¹ 0; −C⁻¹ B A⁻¹ C⁻¹]`, the two inverses in parallel and the two
/// products in row tiles, so one wide block keeps all the matrix units busy instead of one.
/// `false` if a pivot is zero.
fn invert_lower<T: SmeReal>(k: usize, a: Shared<T>, lda: usize, out: Shared<T>, ldo: usize) -> bool {
    unsafe {
        if k <= INVERT_LEAF {
            for j in 0..k {
                for i in 0..k {
                    *out.at(j * ldo + i) = if i >= j { *a.at(j * lda + i) } else { T::default() };
                }
            }
            return T::trtri(k, out.0, ldo) == 0;
        }
        let h = k / 2;
        let (top, bottom) = rayon::join(
            || invert_lower(h, a, lda, out, ldo),
            || invert_lower(k - h, Shared(a.at(h * lda + h)), lda, Shared(out.at(h * ldo + h)), ldo),
        );
        if !(top && bottom) {
            return false;
        }
        for j in h..k {
            for i in 0..h {
                *out.at(j * ldo + i) = T::default();
            }
        }
        let rest = k - h;
        let mut product = vec![T::default(); rest * h];
        let p = Shared(product.as_mut_ptr());
        let tiles = rest.div_ceil(OB);
        (0..tiles).into_par_iter().for_each(|t| {
            let r0 = t * OB;
            T::gemm_nn((rest - r0).min(OB), h, h, 1.0, a.at(r0 + h), lda, out.at(0), ldo, p.at(r0), rest);
        });
        (0..tiles).into_par_iter().for_each(|t| {
            let r0 = t * OB;
            T::gemm_nn((rest - r0).min(OB), h, rest, -1.0, out.at(h * ldo + h + r0), ldo, p.at(0), rest, out.at(h + r0), ldo);
        });
        true
    }
}

/// Adds a child's update block (`rows × rows`, lower) into the parent: entries whose column is
/// one of the parent's columns into its block (`f` rows, column-major), each rounded once to
/// `T`, the rest into its update block (`m × m`). `rel` maps the child's rows to the parent's
/// block rows; distinct columns of the child land in distinct columns of the parent, so columns
/// may go in parallel.
#[allow(clippy::too_many_arguments)]
fn extend_add<T: SmeReal>(rel: &[u32], child: &[f64], block: Shared<T>, f: usize, k: usize, u: Shared<f64>, m: usize, parallel: bool) {
    let rows = rel.len();
    let column = |j: usize| {
        let col = &child[j * rows..(j + 1) * rows];
        let t = rel[j] as usize;
        unsafe {
            if t < k {
                let dst = block.at(t * f);
                for i in j..rows {
                    let v = dst.add(rel[i] as usize);
                    *v = T::from_f64((*v).to_f64() + col[i]);
                }
            } else {
                let dst = u.at((t - k) * m);
                for i in j..rows {
                    *dst.add(rel[i] as usize - k) += col[i];
                }
            }
        }
    };
    if parallel {
        (0..rows).into_par_iter().with_min_len(16).for_each(column);
    } else {
        (0..rows).for_each(column);
    }
}

/// Block `(r0 .. r0 + nr, c0 .. c0 + nc)` of the `f64` update block `U` (`m × m`) less the
/// matching rows of `L₂₁` (leading dimension `ld`, `k` columns) times their transpose: in place
/// for `f64`, through an `f32` tile for `f32`; the strict upper part of a diagonal block is left
/// alone.
#[allow(clippy::too_many_arguments)]
fn update_tile<T: SmeReal>(below: Shared<T>, ld: usize, k: usize, u: Shared<f64>, m: usize, r0: usize, c0: usize, nr: usize, nc: usize) {
    let diagonal = r0 == c0;
    unsafe {
        if let Some(wide) = T::wide(u.at(c0 * m + r0)) {
            if diagonal {
                T::syrk(nc, k, below.at(c0), ld, wide, m);
            } else {
                T::gemm(nr, nc, k, below.at(r0), ld, below.at(c0), ld, wide, m);
            }
            return;
        }
        let mut tile = vec![T::default(); nr * nc];
        if diagonal {
            T::syrk(nc, k, below.at(c0), ld, tile.as_mut_ptr(), nr);
        } else {
            T::gemm(nr, nc, k, below.at(r0), ld, below.at(c0), ld, tile.as_mut_ptr(), nr);
        }
        for j in 0..nc {
            let dst = u.at((c0 + j) * m + r0);
            for i in if diagonal { j } else { 0 }..nr {
                *dst.add(i) += tile[j * nr + i].to_f64();
            }
        }
    }
}
