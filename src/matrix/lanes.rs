//! Array tiles spread across the subgroup, for [`MatrixBackend`] on `Subgroup` wherever the
//! cooperative-matrix lowering does not apply. A tile still means one value per subgroup, but
//! each lane works only on the accumulator elements it owns, so `mma` costs each lane
//! `R·C·K / lanes` multiply-adds instead of the whole tile, and loads and stores touch only owned
//! elements. Where a tile splits evenly into at most 16 rows and 16 columns per lane, within a
//! register budget, the lanes form an `lr × lc` [`grid`] and each owns a 2D block: the rows
//! `≡ lane / lc (mod lr)` and the columns `≡ lane % lc (mod lc)`, so an `mma` step reads `R/lr`
//! elements of `A` and `C/lc` of `B`. A single `64×64` tile thus runs as the register-blocked
//! kernel one would write by hand. Other tiles split row-major: element `e = r * C + c` belongs to
//! lane `e % lanes`, slot `e / lanes`.
//! Input tiles stay borrowed views. The row/column folds broadcast every element in order,
//! keeping the sequential fold order the CPU backends use.

use super::{Layout, View, tile_index};
use crate::backend::subgroup::device::{subgroup_local_invocation_id, width};
use crate::scalar::{FloatScalar, Scalar};


#[inline]
fn lane() -> usize {
    subgroup_local_invocation_id() as usize
}

/// Bytes of accumulator one lane may own in a tile: 128 `f32`s, which leaves room in a 255-register
/// budget for the operands of an `mma` step.
const LANE_BYTES: usize = 512;

/// The `(lr, lc)` lane grid a `w`-wide subgroup lays over an `r×c` tile of `bytes`-wide elements,
/// or `(0, 0)` for the row-major split. Of the grids that tile it exactly with at most 16 rows and
/// 16 columns per lane and at most [`LANE_BYTES`] per lane, the one whose lanes read the fewest
/// elements per `mma` step, `r/lr + c/lc`.
const fn grid(w: usize, r: usize, c: usize, bytes: usize) -> (usize, usize) {
    let bytes = if bytes < 4 { 4 } else { bytes };
    if w == 0 || r * c < w || r * c % w != 0 || r * c / w * bytes > LANE_BYTES {
        return (0, 0);
    }
    let (mut best, mut cost) = ((0, 0), usize::MAX);
    let mut lc = 1;
    while lc <= w {
        let lr = w / lc;
        if w % lc == 0 && c % lc == 0 && r % lr == 0 && r / lr <= 16 && c / lc <= 16 && r / lr + c / lc < cost {
            cost = r / lr + c / lc;
            best = (lr, lc);
        }
        lc += 1;
    }
    best
}

/// How many slots `n` elements take per lane of a `w`-wide subgroup, `0` for `w = 0` (a width
/// only known at run time, whose fixed-width code is compiled but never taken).
const fn slots(n: usize, w: usize) -> usize {
    if w == 0 { 0 } else { n.div_ceil(w) }
}

/// The rows and columns of an `r×c` tile each lane owns on the [`grid`], `(0, 0)` without one.
const fn per_lane(w: usize, r: usize, c: usize, bytes: usize) -> (usize, usize) {
    match grid(w, r, c, bytes) {
        (0, _) => (0, 0),
        (lr, lc) => (r / lr, c / lc),
    }
}

/// `$body` with `$i` bound to each literal below 8 that is under the constant `$n`.
/// `upto8!(16 n, …)` goes up to 16.
macro_rules! upto8 {
    (16 $n:expr, $i:ident, $body:block) => {
        upto8!(@ $n, $i, $body, 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    };
    ($n:expr, $i:ident, $body:block) => {
        upto8!(@ $n, $i, $body, 0 1 2 3 4 5 6 7)
    };
    (@ $n:expr, $i:ident, $body:block, $($k:literal)*) => {
        $(if const { $k < $n } {
            let $i: usize = $k;
            $body
        })*
    };
}

/// `$body` for each element `($r, $c)` of an `$R×$C` tile of `$E` this lane owns, with `$j` its
/// slot, on a `$W`-wide subgroup. A known width unrolls the slots under literal indices on the
/// [`grid`] or wherever there are at most 8 of them, so a tile scalar-replaces down to the lane's
/// own elements instead of being carried, and copied, whole. `$W = 0` is an unusual width read at
/// run time and takes a loop: `dispatch_matrix_subgroup` gives each usual width its own `$W`.
macro_rules! each_slot {
    ($W:ident; $E:ty, $R:ident, $C:ident, $j:ident, $r:ident, $c:ident, $body:block) => {{
        let l = lane();
        if const { $W != 0 } {
            each_slot!(@width $W, $E, $R, $C, l, $j, $r, $c, $body)
        } else {
            each_slot!(@loop width::<0>(), $R, $C, l, $j, $r, $c, $body)
        }
    }};
    (@width $w:expr, $E:ty, $R:ident, $C:ident, $l:ident, $j:ident, $r:ident, $c:ident, $body:block) => {
        if const { grid($w, $R, $C, size_of::<$E>()).0 > 0 } {
            let (lr, lc) = const { grid($w, $R, $C, size_of::<$E>()) };
            upto8!(16 per_lane($w, $R, $C, size_of::<$E>()).0, p, {
                upto8!(16 per_lane($w, $R, $C, size_of::<$E>()).1, q, {
                    let $j = p * const { per_lane($w, $R, $C, size_of::<$E>()).1 } + q;
                    let $r = $l / lc + p * lr;
                    let $c = $l % lc + q * lc;
                    $body
                });
            });
        } else if const { slots($R * $C, $w) <= 8 } {
            upto8!(slots($R * $C, $w), q, {
                let $j = q;
                let e = $l + q * $w;
                if (q + 1) * $w <= $R * $C || e < $R * $C {
                    let ($r, $c) = (e / $C, e % $C);
                    $body
                }
            });
        } else {
            each_slot!(@loop $w, $R, $C, $l, $j, $r, $c, $body)
        }
    };
    (@loop $w:expr, $R:ident, $C:ident, $l:ident, $j:ident, $r:ident, $c:ident, $body:block) => {{
        let (mut e, mut $j) = ($l, 0usize);
        while e < $R * $C {
            let ($r, $c) = (e / $C, e % $C);
            $body
            e += $w;
            $j += 1;
        }
    }};
}

/// The lane owning element `(r, c)` of an `R×C` tile on an `n`-wide subgroup, and its slot there:
/// the inverse of [`each_slot!`].
#[inline]
fn owner<const W: usize, E, const R: usize, const C: usize>(n: usize, r: usize, c: usize) -> (usize, usize) {
    let (lr, lc) = if const { W != 0 } {
        const { grid(W, R, C, size_of::<E>()) }
    } else {
        match n {
            32 => const { grid(32, R, C, size_of::<E>()) },
            64 => const { grid(64, R, C, size_of::<E>()) },
            16 => const { grid(16, R, C, size_of::<E>()) },
            8 => const { grid(8, R, C, size_of::<E>()) },
            _ => (0, 0),
        }
    };
    if lr == 0 {
        let e = r * C + c;
        (e % n, e / n)
    } else {
        ((r % lr) * lc + c % lc, (r / lr) * (C / lc) + c / lc)
    }
}

#[inline]
fn slot<E: Copy, const R: usize, const C: usize>(t: &[[E; C]; R], j: usize) -> E {
    t[j / C][j % C]
}

#[inline]
fn set_slot<E: Copy, const R: usize, const C: usize>(t: &mut [[E; C]; R], j: usize, v: E) {
    t[j / C][j % C] = v;
}

/// Lane `id`'s `v`, for a dynamically uniform `id`.
#[inline]
fn broadcast_raw<N: Copy>(v: N, id: u32) -> N {
    let mut out = core::mem::MaybeUninit::<N>::uninit();
    unsafe {
        core::arch::asm!(
            "OpCapability GroupNonUniformBallot",
            "%u32 = OpTypeInt 32 0",
            "%scope = OpConstant %u32 3",
            "%v = OpLoad _ {v}",
            "%r = OpGroupNonUniformBroadcast typeof*{out} %scope %v {id}",
            "OpStore {out} %r",
            v = in(reg) &v,
            id = in(reg) id,
            out = in(reg) out.as_mut_ptr(),
        );
        out.assume_init()
    }
}

/// Lane `id`'s `v`. 16- and 32-bit elements travel as `u32`, which every device can broadcast;
/// wider ones as themselves.
#[inline]
fn broadcast<E: Scalar>(v: E, id: u32) -> E {
    if size_of::<E>() <= 4 {
        E::from_bits32(broadcast_raw(v.to_bits32(), id))
    } else {
        E::from_spirv(broadcast_raw(v.to_spirv(), id))
    }
}

/// How a tile of each [`Role`](super::Role) is spread: input views are left whole, the
/// accumulator is split into lane-owned slots.
pub trait LaneTile<'a, E: Scalar, const R: usize, const C: usize>: Copy {
    fn lt_load<const W: usize>(mem: &'a [E], row_stride: usize, layout: Layout) -> Self;
    fn lt_store<const W: usize>(self, out: &mut [E], row_stride: usize, layout: Layout);
    fn lt_splat(v: E) -> Self;
    fn lt_map<const W: usize>(self, f: impl Fn(E) -> E) -> Self;
}

impl<'a, E: Scalar, const R: usize, const C: usize> LaneTile<'a, E, R, C> for View<'a, E, R, C> {
    #[inline]
    fn lt_load<const W: usize>(mem: &'a [E], row_stride: usize, layout: Layout) -> Self {
        super::CpuTile::ct_load(mem, row_stride, layout)
    }
    #[inline]
    fn lt_store<const W: usize>(self, _out: &mut [E], _row_stride: usize, _layout: Layout) {
        unreachable!("a read-only input tile is never stored")
    }
    #[inline]
    fn lt_splat(_v: E) -> Self {
        unreachable!("a read-only input tile is never splatted")
    }
    #[inline(always)]
    fn lt_map<const W: usize>(self, _f: impl Fn(E) -> E) -> Self {
        unreachable!("a read-only input tile is never mapped")
    }
}

impl<'a, E: Scalar, const R: usize, const C: usize> LaneTile<'a, E, R, C> for [[E; C]; R] {
    #[inline]
    fn lt_load<const W: usize>(mem: &'a [E], row_stride: usize, layout: Layout) -> Self {
        let mut t = [[E::ZERO; C]; R];
        each_slot!(W; E, R, C, j, r, c, {
            set_slot(&mut t, j, mem[tile_index(r, c, row_stride, layout)]);
        });
        t
    }
    #[inline]
    fn lt_store<const W: usize>(self, out: &mut [E], row_stride: usize, layout: Layout) {
        each_slot!(W; E, R, C, j, r, c, {
            out[tile_index(r, c, row_stride, layout)] = slot(&self, j);
        });
    }
    #[inline]
    fn lt_splat(v: E) -> Self {
        [[v; C]; R]
    }
    #[inline(always)]
    fn lt_map<const W: usize>(mut self, f: impl Fn(E) -> E) -> Self {
        each_slot!(W; E, R, C, j, _r, _c, {
            let x = f(slot(&self, j));
            set_slot(&mut self, j, x);
        });
        self
    }
}

#[inline(always)]
pub(crate) fn zip<const W: usize, E: Scalar, const R: usize, const C: usize>(
    mut a: [[E; C]; R],
    b: [[E; C]; R],
    f: impl Fn(E, E) -> E,
) -> [[E; C]; R] {
    each_slot!(W; E, R, C, j, _r, _c, {
        let x = f(slot(&a, j), slot(&b, j));
        set_slot(&mut a, j, x);
    });
    a
}

/// `f(elem, v[key(r, c)])` on every owned element.
#[inline(always)]
pub(crate) fn zip_keyed<const W: usize, E: Scalar, const R: usize, const C: usize, const K: usize>(
    mut t: [[E; C]; R],
    v: [E; K],
    key: impl Fn(usize, usize) -> usize,
    f: impl Fn(E, E) -> E,
) -> [[E; C]; R] {
    each_slot!(W; E, R, C, j, r, c, {
        let x = f(slot(&t, j), v[key(r, c)]);
        set_slot(&mut t, j, x);
    });
    t
}

/// Fold every element, in row-major order, into `acc[key(r, c)]`; every lane gets the result.
#[inline(always)]
pub(crate) fn fold_keyed<const W: usize, E: Scalar, const R: usize, const C: usize, const K: usize>(
    t: [[E; C]; R],
    mut acc: [E; K],
    key: impl Fn(usize, usize) -> usize,
    f: impl Fn(E, E) -> E,
) -> [E; K] {
    let n = width::<W>();
    let mut e = 0;
    while e < R * C {
        let (r, c) = (e / C, e % C);
        let (from, j) = owner::<W, E, R, C>(n, r, c);
        let v = broadcast(slot(&t, j), from as u32);
        let k = key(r, c);
        acc[k] = f(acc[k], v);
        e += 1;
    }
    acc
}

/// `D = A·B + C` over the lane's owned elements of `C`.
#[inline]
pub(crate) fn mma<const W: usize, T: FloatScalar, const M: usize, const N: usize, const K: usize>(
    a: View<'_, T, M, K>,
    b: View<'_, T, K, N>,
    c: [[T::Compute; N]; M],
) -> [[T::Compute; N]; M] {
    mma_block::<W, T, 1, 1, M, N, K>([a], [b], [[c]])[0][0]
}

/// [`mma`] over a `BM×BN` block in one pass over `k`, so each loaded `A` element feeds `BN`
/// accumulators and each `B` element `BM`. Without a known width `W` the block takes the
/// width-agnostic loop; the usual widths reach here with their own `W`.
#[inline]
pub(crate) fn mma_block<const W: usize, T: FloatScalar, const BM: usize, const BN: usize, const M: usize, const N: usize, const K: usize>(
    a: [View<'_, T, M, K>; BM],
    b: [View<'_, T, K, N>; BN],
    c: [[[[T::Compute; N]; M]; BN]; BM],
) -> [[[[T::Compute; N]; M]; BN]; BM] {
    if const { W != 0 } {
        block_w::<T, W, BM, BN, M, N, K>(a, b, c)
    } else {
        pairs::<0, T, BM, BN, M, N, K>(a, b, c)
    }
}

#[inline]
fn block_w<T: FloatScalar, const W: usize, const BM: usize, const BN: usize, const M: usize, const N: usize, const K: usize>(
    a: [View<'_, T, M, K>; BM],
    b: [View<'_, T, K, N>; BN],
    c: [[[[T::Compute; N]; M]; BN]; BM],
) -> [[[[T::Compute; N]; M]; BN]; BM] {
    if const { grid(W, M, N, size_of::<T::Compute>()).0 > 0 } {
        grid_block::<T, W, BM, BN, M, N, K>(a, b, c)
    } else if const { W % N == 0 && slots(M * N, W) <= 8 } {
        columnar_block::<T, W, BM, BN, M, N, K>(a, b, c)
    } else {
        pairs::<W, T, BM, BN, M, N, K>(a, b, c)
    }
}

/// [`mma_block`] on the lane [`grid`]: per step, a lane reads its `M/lr` rows of each `A` and its
/// `N/lc` columns of each `B` once, and multiplies every pair.
#[inline]
fn grid_block<T: FloatScalar, const W: usize, const BM: usize, const BN: usize, const M: usize, const N: usize, const K: usize>(
    a: [View<'_, T, M, K>; BM],
    b: [View<'_, T, K, N>; BN],
    mut c: [[[[T::Compute; N]; M]; BN]; BM],
) -> [[[[T::Compute; N]; M]; BN]; BM] {
    let (lr, lc) = const { grid(W, M, N, size_of::<T::Compute>()) };
    let l = lane();
    let (row, col) = (l / lc, l % lc);
    let zero = <T::Compute as Scalar>::ZERO;
    let mut k = 0;
    while k < K {
        let mut y = [[zero; 16]; BN];
        upto8!(BN, j, {
            upto8!(16 per_lane(W, M, N, size_of::<T::Compute>()).1, q, {
                y[j][q] = b[j].get(k, col + q * lc).widen();
            });
        });
        upto8!(BM, i, {
            upto8!(16 per_lane(W, M, N, size_of::<T::Compute>()).0, p, {
                let x = a[i].get(row + p * lr, k).widen();
                upto8!(BN, j, {
                    upto8!(16 per_lane(W, M, N, size_of::<T::Compute>()).1, q, {
                        let s = p * const { per_lane(W, M, N, size_of::<T::Compute>()).1 } + q;
                        let v = x.fma(y[j][q], slot(&c[i][j], s));
                        set_slot(&mut c[i][j], s, v);
                    });
                });
            });
        });
        k += 1;
    }
    c
}

/// [`mma_block`] on the row-major split when every slot of a lane lies in column `lane % N`, at
/// row `lane / N + q · W / N` for slot `q`: one `B` element per tile per step.
#[inline]
fn columnar_block<T: FloatScalar, const W: usize, const BM: usize, const BN: usize, const M: usize, const N: usize, const K: usize>(
    a: [View<'_, T, M, K>; BM],
    b: [View<'_, T, K, N>; BN],
    mut c: [[[[T::Compute; N]; M]; BN]; BM],
) -> [[[[T::Compute; N]; M]; BN]; BM] {
    let l = lane();
    let (row, col) = (l / N, l % N);
    let mut k = 0;
    while k < K {
        let mut y = [<T::Compute as Scalar>::ZERO; BN];
        upto8!(BN, j, {
            y[j] = b[j].get(k, col).widen();
        });
        upto8!(slots(M * N, W), q, {
            if (q + 1) * W <= M * N || l + q * W < M * N {
                upto8!(BM, i, {
                    let x = a[i].get(row + q * (W / N), k).widen();
                    upto8!(BN, j, {
                        let v = x.fma(y[j], slot(&c[i][j], q));
                        set_slot(&mut c[i][j], q, v);
                    });
                });
            }
        });
        k += 1;
    }
    c
}

#[inline]
fn pairs<const W: usize, T: FloatScalar, const BM: usize, const BN: usize, const M: usize, const N: usize, const K: usize>(
    a: [View<'_, T, M, K>; BM],
    b: [View<'_, T, K, N>; BN],
    mut c: [[[[T::Compute; N]; M]; BN]; BM],
) -> [[[[T::Compute; N]; M]; BN]; BM] {
    upto8!(BM, i, {
        upto8!(BN, j, {
            c[i][j] = mma_any::<W, T, M, N, K>(a[i], b[j], c[i][j]);
        });
    });
    c
}

/// `D = A·B + C` for any tile shape and subgroup width, `K` outermost so the slots' running sums
/// stay in registers across the reduction.
#[inline]
fn mma_any<const W: usize, T: FloatScalar, const M: usize, const N: usize, const K: usize>(
    a: View<'_, T, M, K>,
    b: View<'_, T, K, N>,
    mut c: [[T::Compute; N]; M],
) -> [[T::Compute; N]; M] {
    let mut k = 0;
    while k < K {
        each_slot!(W; T::Compute, M, N, j, r, col, {
            let s = a.get(r, k).widen().fma(b.get(k, col).widen(), slot(&c, j));
            set_slot(&mut c, j, s);
        });
        k += 1;
    }
    c
}

/// The [`MatrixBackend`](super::MatrixBackend) methods over lane-distributed array tiles; the
/// impl supplies `T` and the subgroup width `W` (`0`: read at run time).
macro_rules! lane_tile_methods {
    ($t:ty, $w:ident) => {
        type Tile<'a, E: Scalar, const R: usize, const C: usize, Ro: Role> = <Ro as Role>::Repr<'a, E, R, C>;

        #[inline]
        fn tile_load<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            mem: &'a [E],
            row_stride: usize,
            layout: Layout,
        ) -> Self::Tile<'a, E, R, C, Ro> {
            $crate::matrix::lanes::LaneTile::lt_load::<$w>(mem, row_stride, layout)
        }

        #[inline]
        fn tile_store<E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            t: Self::Tile<'_, E, R, C, Ro>,
            out: &mut [E],
            row_stride: usize,
            layout: Layout,
        ) {
            $crate::matrix::lanes::LaneTile::lt_store::<$w>(t, out, row_stride, layout)
        }

        #[inline]
        fn tile_splat<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(self, v: E) -> Self::Tile<'a, E, R, C, Ro> {
            $crate::matrix::lanes::LaneTile::lt_splat(v)
        }

        #[inline(always)]
        fn tile_map<'a, E: Scalar, const R: usize, const C: usize, Ro: Role>(
            self,
            t: Self::Tile<'a, E, R, C, Ro>,
            f: impl Fn(E) -> E,
        ) -> Self::Tile<'a, E, R, C, Ro> {
            $crate::matrix::lanes::LaneTile::lt_map::<$w>(t, f)
        }

        #[inline(always)]
        fn tile_zip<'a, E: Scalar, const R: usize, const C: usize>(
            self,
            a: Self::Tile<'a, E, R, C, Accumulator>,
            b: Self::Tile<'a, E, R, C, Accumulator>,
            f: impl Fn(E, E) -> E,
        ) -> Self::Tile<'a, E, R, C, Accumulator> {
            $crate::matrix::lanes::zip::<$w, E, R, C>(a, b, f)
        }

        #[inline(always)]
        fn tile_fold_rows<E: Scalar, const R: usize, const C: usize>(
            self,
            t: Self::Tile<'_, E, R, C, Accumulator>,
            acc: [E; R],
            f: impl Fn(E, E) -> E,
        ) -> [E; R] {
            $crate::matrix::lanes::fold_keyed::<$w, E, R, C, R>(t, acc, |r, _| r, f)
        }

        #[inline(always)]
        fn tile_fold_cols<E: Scalar, const R: usize, const C: usize>(
            self,
            t: Self::Tile<'_, E, R, C, Accumulator>,
            acc: [E; C],
            f: impl Fn(E, E) -> E,
        ) -> [E; C] {
            $crate::matrix::lanes::fold_keyed::<$w, E, R, C, C>(t, acc, |_, c| c, f)
        }

        #[inline(always)]
        fn tile_zip_rows<'a, E: Scalar, const R: usize, const C: usize>(
            self,
            t: Self::Tile<'a, E, R, C, Accumulator>,
            v: [E; R],
            f: impl Fn(E, E) -> E,
        ) -> Self::Tile<'a, E, R, C, Accumulator> {
            $crate::matrix::lanes::zip_keyed::<$w, E, R, C, R>(t, v, |r, _| r, f)
        }

        #[inline(always)]
        fn tile_zip_cols<'a, E: Scalar, const R: usize, const C: usize>(
            self,
            t: Self::Tile<'a, E, R, C, Accumulator>,
            v: [E; C],
            f: impl Fn(E, E) -> E,
        ) -> Self::Tile<'a, E, R, C, Accumulator> {
            $crate::matrix::lanes::zip_keyed::<$w, E, R, C, C>(t, v, |_, c| c, f)
        }

        #[inline]
        fn mma<'i, 'c, const M: usize, const N: usize, const K: usize>(
            self,
            a: Self::Tile<'i, $t, M, K, MatrixA>,
            b: Self::Tile<'i, $t, K, N, MatrixB>,
            c: Self::Tile<'c, <$t as FloatScalar>::Compute, M, N, Accumulator>,
        ) -> Self::Tile<'c, <$t as FloatScalar>::Compute, M, N, Accumulator> {
            $crate::matrix::lanes::mma::<$w, $t, M, N, K>(a, b, c)
        }

        #[inline]
        fn mma_block<'i, 'c, const BM: usize, const BN: usize, const M: usize, const N: usize, const K: usize>(
            self,
            a: [Self::Tile<'i, $t, M, K, MatrixA>; BM],
            b: [Self::Tile<'i, $t, K, N, MatrixB>; BN],
            c: [[Self::Tile<'c, <$t as FloatScalar>::Compute, M, N, Accumulator>; BN]; BM],
        ) -> [[Self::Tile<'c, <$t as FloatScalar>::Compute, M, N, Accumulator>; BN]; BM] {
            $crate::matrix::lanes::mma_block::<$w, $t, BM, BN, M, N, K>(a, b, c)
        }
    };
}
pub(crate) use lane_tile_methods;
