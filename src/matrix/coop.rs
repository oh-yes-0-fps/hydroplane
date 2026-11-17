//! [`MatrixBackend`] over `VK_KHR_cooperative_matrix` for `Subgroup<Coop<S>>`: an `f16` tile is
//! one subgroup-wide cooperative matrix and `mma` is `OpCooperativeMatrixMulAddKHR`. Elements
//! with no cooperative-matrix form worth using keep the per-invocation array tiles.
//!
//! The per-element ops (`tile_map`, `tile_zip`) walk this invocation's share of the matrix with
//! `OpCooperativeMatrixLengthKHR`, which needs a rust-gpu whose SPIR-T accepts a type operand
//! inside a function. The row/column ops need each element's `(row, col)`, which the extension
//! never exposes, so each stages the tile through a workgroup array it declares itself.

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use spirv_std::spirv;

use super::{Accumulator, Layout, MatrixA, MatrixB, MatrixBackend, Role};
use crate::backend::subgroup::device::{num_subgroups, subgroup_id, subgroup_local_invocation_id, subgroup_size};
use crate::backend::subgroup::{Coop, Subgroup};
use crate::scalar::{FloatScalar, Scalar};

/// An `R×C` cooperative matrix of SPIR-V component `E` with `CooperativeMatrixUse` `USE`,
/// distributed across the subgroup.
#[spirv(cooperative_matrix)]
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CoopMatrix<E, const USE: u32, const R: usize, const C: usize> {
    // Keeps the Rust layout non-zero-sized so the attribute sees the type before it is elided.
    _anti_zst: MaybeUninit<u32>,
    _e: PhantomData<E>,
}

#[inline]
fn declare_coop() {
    unsafe {
        core::arch::asm!(
            "OpCapability CooperativeMatrixKHR",
            "OpExtension \"SPV_KHR_cooperative_matrix\"",
        )
    }
}

/// The capabilities a component or storage element of `E`'s width needs.
#[inline]
fn declare_width<E>() {
    match size_of::<E>() {
        2 => unsafe {
            core::arch::asm!(
                "OpCapability Float16",
                "OpCapability Int16",
                "OpCapability StorageBuffer16BitAccess",
                "OpExtension \"SPV_KHR_16bit_storage\"",
            )
        },
        8 => unsafe { core::arch::asm!("OpCapability Float64") },
        _ => {}
    }
}

/// Load a matrix from `ptr`, which must point into a storage-buffer or workgroup array. The
/// array's element type may differ from the component type; `stride` counts array elements.
#[inline]
unsafe fn load<S, M>(ptr: *const S, layout: Layout, stride: u32) -> M {
    declare_coop();
    declare_width::<S>();
    let mut out = MaybeUninit::<M>::uninit();
    unsafe {
        match layout {
            Layout::RowMajor => core::arch::asm!(
                "%u32 = OpTypeInt 32 0",
                "%layout = OpConstant %u32 0",
                "%stride = OpLoad %u32 {stride}",
                "%m = OpCooperativeMatrixLoadKHR typeof* {out} {ptr} %layout %stride",
                "OpStore {out} %m",
                ptr = in(reg) ptr,
                stride = in(reg) &stride,
                out = in(reg) out.as_mut_ptr(),
            ),
            Layout::ColMajor => core::arch::asm!(
                "%u32 = OpTypeInt 32 0",
                "%layout = OpConstant %u32 1",
                "%stride = OpLoad %u32 {stride}",
                "%m = OpCooperativeMatrixLoadKHR typeof* {out} {ptr} %layout %stride",
                "OpStore {out} %m",
                ptr = in(reg) ptr,
                stride = in(reg) &stride,
                out = in(reg) out.as_mut_ptr(),
            ),
        }
        out.assume_init()
    }
}

/// Store `m` through `ptr` (see [`load`]).
#[inline]
unsafe fn store<S, M>(ptr: *mut S, m: M, layout: Layout, stride: u32) {
    declare_coop();
    declare_width::<S>();
    unsafe {
        match layout {
            Layout::RowMajor => core::arch::asm!(
                "%u32 = OpTypeInt 32 0",
                "%layout = OpConstant %u32 0",
                "%stride = OpLoad %u32 {stride}",
                "%m = OpLoad _ {m}",
                "OpCooperativeMatrixStoreKHR {ptr} %m %layout %stride",
                ptr = in(reg) ptr,
                m = in(reg) &m,
                stride = in(reg) &stride,
            ),
            Layout::ColMajor => core::arch::asm!(
                "%u32 = OpTypeInt 32 0",
                "%layout = OpConstant %u32 1",
                "%stride = OpLoad %u32 {stride}",
                "%m = OpLoad _ {m}",
                "OpCooperativeMatrixStoreKHR {ptr} %m %layout %stride",
                ptr = in(reg) ptr,
                m = in(reg) &m,
                stride = in(reg) &stride,
            ),
        }
    }
}

/// A matrix with every component `v`.
#[inline]
fn splat<N, M>(v: N) -> M {
    declare_coop();
    declare_width::<N>();
    let mut out = MaybeUninit::<M>::uninit();
    unsafe {
        core::arch::asm!(
            "%v = OpLoad _ {v}",
            "%m = OpCompositeConstruct typeof* {out} %v",
            "OpStore {out} %m",
            v = in(reg) &v,
            out = in(reg) out.as_mut_ptr(),
        );
        out.assume_init()
    }
}

/// `a·b + c`.
#[inline]
fn mul_add<A, B, C>(a: A, b: B, c: C) -> C {
    declare_coop();
    let mut out = MaybeUninit::<C>::uninit();
    unsafe {
        core::arch::asm!(
            "%a = OpLoad _ {a}",
            "%b = OpLoad _ {b}",
            "%c = OpLoad _ {c}",
            "%d = OpCooperativeMatrixMulAddKHR typeof* {out} %a %b %c",
            "OpStore {out} %d",
            a = in(reg) &a,
            b = in(reg) &b,
            c = in(reg) &c,
            out = in(reg) out.as_mut_ptr(),
        );
        out.assume_init()
    }
}

/// How many of the matrix's components this invocation holds.
#[inline]
fn length<M>(m: &M) -> u32 {
    let mut n = 0u32;
    unsafe {
        core::arch::asm!(
            "%u32 = OpTypeInt 32 0",
            "%n = OpCooperativeMatrixLengthKHR %u32 typeof* {m}",
            "OpStore {n} %n",
            m = in(reg) m,
            n = in(reg) &mut n,
        );
    }
    n
}

/// This invocation's component `i` (`i < length`).
#[inline]
fn get<M, N>(m: &mut M, i: u32) -> N {
    let mut v = MaybeUninit::<N>::uninit();
    unsafe {
        core::arch::asm!(
            "%p = OpAccessChain typeof{v} {m} {i}",
            "%x = OpLoad _ %p",
            "OpStore {v} %x",
            m = in(reg) m,
            i = in(reg) i,
            v = in(reg) v.as_mut_ptr(),
        );
        v.assume_init()
    }
}

/// Set this invocation's component `i` (`i < length`).
#[inline]
fn set<M, N>(m: &mut M, i: u32, v: N) {
    unsafe {
        core::arch::asm!(
            "%p = OpAccessChain typeof{v} {m} {i}",
            "%x = OpLoad _ {v}",
            "OpStore %p %x",
            m = in(reg) m,
            i = in(reg) i,
            v = in(reg) &v,
        );
    }
}

/// Subgroup execution barrier ordering the subgroup's workgroup and buffer accesses
/// (`AcquireRelease | UniformMemory | WorkgroupMemory`).
#[inline]
fn subgroup_barrier() {
    unsafe {
        core::arch::asm!(
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 3",
            "%semantics = OpConstant %u32 328",
            "OpControlBarrier %scope %scope %semantics",
        );
    }
}

/// Element `i` of a row-major tile of `E` packed little-endian into `u32` words at `words`.
#[inline]
fn unpack<E: Scalar, const S: usize>(words: &[u32; S], base: usize, i: usize) -> E {
    const { assert!(size_of::<E>() == 2 || size_of::<E>() == 4, "row/column tile ops take 16- or 32-bit elements") };
    if size_of::<E>() == 4 {
        E::from_bits32(words[base + i])
    } else {
        E::from_bits32((words[base + i / 2] >> (16 * (i as u32 & 1))) & 0xffff)
    }
}

/// `f` over this invocation's components of `a`, paired with the same components of `b`.
#[inline]
fn zip_components<M: Copy, E: Scalar>(mut a: M, mut b: M, f: impl Fn(E, E) -> E) -> M {
    let n = length(&a);
    let mut i = 0;
    while i < n {
        let x = E::from_spirv(get(&mut a, i));
        let y = E::from_spirv(get(&mut b, i));
        set(&mut a, i, f(x, y).to_spirv());
        i += 1;
    }
    a
}

/// A fresh `S`-word workgroup array. Every call is its own `OpVariable`, so an op takes one and
/// does all its staging through it.
#[inline]
fn workgroup_scratch<const S: usize>() -> *mut [u32; S] {
    let mut p = MaybeUninit::<*mut [u32; S]>::uninit();
    unsafe {
        core::arch::asm!(
            "%scratch = OpVariable typeof*{p} Workgroup",
            "OpStore {p} %scratch",
            p = in(reg) p.as_mut_ptr(),
        );
        p.assume_init()
    }
}

impl<const S: usize> Coop<S> {
    /// First word of this subgroup's even share of a scratch array.
    #[inline]
    fn base(self) -> usize {
        (subgroup_id() * (S as u32 / num_subgroups())) as usize
    }

    /// Words a row-major `R×C` tile of `E` takes in the scratch; the coop store's stride is
    /// counted in those words.
    #[inline]
    const fn row_words<E, const C: usize>() -> usize {
        C * size_of::<E>() / 4
    }

    /// Store `t` row-major into this subgroup's share of `scratch` and make it visible to the
    /// subgroup.
    #[inline]
    fn stage<E: Scalar, const R: usize, const C: usize>(self, scratch: *mut [u32; S], t: CoopMatrix<E::Spirv, 2, R, C>) {
        let base = self.base();
        unsafe {
            let ptr: *mut u32 = &mut (*scratch)[base];
            store(ptr, t, Layout::RowMajor, Self::row_words::<E, C>() as u32);
        }
        subgroup_barrier();
    }

    /// Fold every staged element into `acc[key(r, c)]`; each invocation reads the whole tile.
    #[inline]
    fn fold<E: Scalar, const R: usize, const C: usize, const K: usize>(
        self,
        t: CoopMatrix<E::Spirv, 2, R, C>,
        mut acc: [E; K],
        key: impl Fn(usize, usize) -> usize,
        f: impl Fn(E, E) -> E,
    ) -> [E; K] {
        let scratch = workgroup_scratch::<S>();
        self.stage::<E, R, C>(scratch, t);
        let words = unsafe { &*scratch };
        let base = self.base();
        let mut r = 0;
        while r < R {
            let mut c = 0;
            while c < C {
                let k = key(r, c);
                acc[k] = f(acc[k], unpack::<E, S>(words, base, r * C + c));
                c += 1;
            }
            r += 1;
        }
        subgroup_barrier();
        acc
    }

    /// Rewrite every staged element as `f(elem, v[key(r, c)])` and reload the tile. Each
    /// invocation rewrites whole words, so two 16-bit elements sharing a word stay with one
    /// writer.
    #[inline]
    fn zip<E: Scalar, const R: usize, const C: usize, const K: usize>(
        self,
        t: CoopMatrix<E::Spirv, 2, R, C>,
        v: [E; K],
        key: impl Fn(usize, usize) -> usize,
        f: impl Fn(E, E) -> E,
    ) -> CoopMatrix<E::Spirv, 2, R, C> {
        let scratch = workgroup_scratch::<S>();
        self.stage::<E, R, C>(scratch, t);
        let words = unsafe { &mut *scratch };
        let base = self.base();
        let per_word = 4 / size_of::<E>();
        let total = R * C / per_word;
        let lanes = subgroup_size() as usize;
        let mut w = subgroup_local_invocation_id() as usize;
        while w < total {
            let mut packed = 0u32;
            let mut j = 0;
            while j < per_word {
                let i = w * per_word + j;
                let x = f(unpack::<E, S>(words, base, i), v[key(i / C, i % C)]);
                packed |= x.to_bits32() << (32 / per_word as u32 * j as u32);
                j += 1;
            }
            words[base + w] = packed;
            w += lanes;
        }
        subgroup_barrier();
        let out = unsafe { load(&words[base] as *const u32, Layout::RowMajor, Self::row_words::<E, C>() as u32) };
        subgroup_barrier();
        out
    }
}

macro_rules! coop_array_backend {
    ($t:ty) => {
        impl<const S: usize, const W: usize> MatrixBackend<$t> for Subgroup<Coop<S>, W> {
            super::lanes::lane_tile_methods!($t, W);
        }
    };
}

// No cooperative-matrix shape takes these as `A`/`B` on mainstream GPUs (and rust-gpu has no
// bf16 float type), so their tiles are lane-distributed arrays.
coop_array_backend!(f32);
coop_array_backend!(f64);
coop_array_backend!(half::bf16);

impl<const S: usize, const W: usize> MatrixBackend<half::f16> for Subgroup<Coop<S>, W> {
    type Tile<'a, E: Scalar, const R: usize, const C: usize, Ro: Role> = <Ro as Role>::Coop<E::Spirv, R, C>;

    #[inline]
    fn tile_load<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        mem: &'a [E],
        row_stride: usize,
        layout: Layout,
    ) -> Self::Tile<'a, E, R, C, Ro> {
        unsafe { load(mem.as_ptr(), layout, row_stride as u32) }
    }

    #[inline]
    fn tile_store<E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        t: Self::Tile<'_, E, R, C, Ro>,
        out: &mut [E],
        row_stride: usize,
        layout: Layout,
    ) {
        unsafe { store(out.as_mut_ptr(), t, layout, row_stride as u32) }
    }

    #[inline]
    fn tile_splat<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        v: E,
    ) -> Self::Tile<'a, E, R, C, Ro> {
        splat(v.to_spirv())
    }

    #[inline]
    fn tile_map<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
        self,
        mut t: Self::Tile<'a, E, R, C, Ro>,
        f: impl Fn(E) -> E,
    ) -> Self::Tile<'a, E, R, C, Ro> {
        let n = length(&t);
        let mut i = 0;
        while i < n {
            let x = E::from_spirv(get(&mut t, i));
            set(&mut t, i, f(x).to_spirv());
            i += 1;
        }
        t
    }

    #[inline]
    fn tile_zip<'a, E: Scalar, const R: usize, const C: usize>(
        self,
        a: Self::Tile<'a, E, R, C, Accumulator>,
        b: Self::Tile<'a, E, R, C, Accumulator>,
        f: impl Fn(E, E) -> E,
    ) -> Self::Tile<'a, E, R, C, Accumulator> {
        zip_components(a, b, f)
    }

    #[inline]
    fn tile_fold_rows<E: Scalar, const R: usize, const C: usize>(
        self,
        t: Self::Tile<'_, E, R, C, Accumulator>,
        acc: [E; R],
        f: impl Fn(E, E) -> E,
    ) -> [E; R] {
        self.mode.fold::<E, R, C, R>(t, acc, |r, _| r, f)
    }

    #[inline]
    fn tile_fold_cols<E: Scalar, const R: usize, const C: usize>(
        self,
        t: Self::Tile<'_, E, R, C, Accumulator>,
        acc: [E; C],
        f: impl Fn(E, E) -> E,
    ) -> [E; C] {
        self.mode.fold::<E, R, C, C>(t, acc, |_, c| c, f)
    }

    #[inline]
    fn tile_zip_rows<'a, E: Scalar, const R: usize, const C: usize>(
        self,
        t: Self::Tile<'a, E, R, C, Accumulator>,
        v: [E; R],
        f: impl Fn(E, E) -> E,
    ) -> Self::Tile<'a, E, R, C, Accumulator> {
        self.mode.zip::<E, R, C, R>(t, v, |r, _| r, f)
    }

    #[inline]
    fn tile_zip_cols<'a, E: Scalar, const R: usize, const C: usize>(
        self,
        t: Self::Tile<'a, E, R, C, Accumulator>,
        v: [E; C],
        f: impl Fn(E, E) -> E,
    ) -> Self::Tile<'a, E, R, C, Accumulator> {
        self.mode.zip::<E, R, C, C>(t, v, |_, c| c, f)
    }

    #[inline]
    fn mma<'i, 'c, const M: usize, const N: usize, const K: usize>(
        self,
        a: Self::Tile<'i, half::f16, M, K, MatrixA>,
        b: Self::Tile<'i, half::f16, K, N, MatrixB>,
        c: Self::Tile<'c, f32, M, N, Accumulator>,
    ) -> Self::Tile<'c, f32, M, N, Accumulator> {
        mul_add(a, b, c)
    }
}
