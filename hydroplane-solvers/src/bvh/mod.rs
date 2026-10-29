//! Bounding volume hierarchies whose branching factor follows the SIMD register width: 4 children
//! per node on NEON, SSE and 128-bit SVE, 8 on AVX2, 16 on AVX-512. A node visit tests every
//! child box in one pass over a register, so a wider machine gets a shallower tree for the same
//! per-node cost instead of idle lanes.
//!
//! [`Bvh`] holds only boxes. Queries hand each candidate leaf to the caller as a range of
//! *slots* (positions in [`Bvh::order`]); a leaf's slots are contiguous, so per-primitive data
//! laid out in slot order ([`Bvh::in_slot_order`]) can be tested a register at a time with the
//! same [`Gang`] the traversal runs on. Mark those methods `#[inline(always)]`: they must inline into the
//! traversal to share its SIMD target features, or every op becomes a call on x86. Closures work as
//! leaf visitors too, for scalar tests.
//!
//! ```
//! use hydroplane_solvers::bvh::{Aabb, Bvh};
//!
//! let centers = [[0.0f32, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
//! let boxes: Vec<Aabb> = centers.iter().map(|&c| Aabb::cube(c, 0.5)).collect();
//! let bvh = Bvh::new(&boxes);
//!
//! let mut hit = None;
//! bvh.ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], f32::INFINITY, &mut |slots: core::ops::Range<usize>, t_max: f32| {
//!     let mut best = None;
//!     for s in slots {
//!         let pi = bvh.order()[s] as usize;
//!         let t = centers[pi][0] - 0.5 + 5.0;
//!         if t < t_max && best.is_none_or(|(_, b)| t < b) {
//!             best = Some((pi, t));
//!         }
//!     }
//!     best.map(|(pi, t)| {
//!         hit = Some(pi);
//!         t
//!     })
//! });
//! assert_eq!(hit, Some(0));
//! ```

mod binary;
mod wide;

use core::mem::MaybeUninit;
use core::ops::{ControlFlow, Range};

pub use hydroplane::{Backend, BackendAll, Gang};
use hydroplane::{Kernel, dispatch};

use binary::Binary;
use wide::{NearestQuery, Node, OverlapQuery, RayQuery};

/// An axis-aligned box. [`EMPTY`](Aabb::EMPTY) is inverted, so a union with it is a no-op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: [f32::INFINITY; 3],
        max: [f32::NEG_INFINITY; 3],
    };

    #[inline]
    #[must_use]
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    #[inline]
    #[must_use]
    pub const fn point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    /// The cube of half-width `r` around `c`.
    #[inline]
    #[must_use]
    pub fn cube(c: [f32; 3], r: f32) -> Self {
        Self {
            min: c.map(|v| v - r),
            max: c.map(|v| v + r),
        }
    }

    /// The box around `points`, [`EMPTY`](Aabb::EMPTY) if there are none.
    #[must_use]
    pub fn from_points(points: &[[f32; 3]]) -> Self {
        points.iter().fold(Self::EMPTY, |b, &p| b.union(&Self::point(p)))
    }

    /// The box around `points` given in `f64`, rounded outward so it still encloses them.
    #[must_use]
    pub fn from_points_f64(points: &[[f64; 3]]) -> Self {
        let mut b = Self::EMPTY;
        for p in points {
            for (a, &v) in p.iter().enumerate() {
                b.min[a] = b.min[a].min(round_down(v));
                b.max[a] = b.max[a].max(round_up(v));
            }
        }
        b
    }

    #[inline]
    #[must_use]
    pub fn union(&self, o: &Self) -> Self {
        Self {
            min: [0, 1, 2].map(|a| self.min[a].min(o.min[a])),
            max: [0, 1, 2].map(|a| self.max[a].max(o.max[a])),
        }
    }

    #[inline]
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [0, 1, 2].map(|a| (self.min[a] + self.max[a]) * 0.5)
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        (0..3).any(|a| self.min[a] > self.max[a])
    }

    #[inline]
    #[must_use]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|a| self.min[a] <= p[a] && p[a] <= self.max[a])
    }

    #[inline]
    #[must_use]
    pub fn overlaps(&self, o: &Self) -> bool {
        (0..3).all(|a| self.min[a] <= o.max[a] && o.min[a] <= self.max[a])
    }
}

fn round_down(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) > v { f.next_down() } else { f }
}

fn round_up(v: f64) -> f32 {
    let f = v as f32;
    if f64::from(f) < v { f.next_up() } else { f }
}

/// Build settings; the defaults suit the machine the tree is built on.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Children per node, rounded up to 4, 8 or 16. `None` follows the `f32` register width
    /// of the backend [`dispatch()`] runs on ([`native_width`]), which is fastest for rays. A tree
    /// queried mostly with [`Bvh::nearest`] gains a few percent from half that width on 8-lane
    /// AVX2, since best-first ordering of wide nodes costs more than the extra depth.
    pub width: Option<usize>,
    /// Most primitives per leaf, `None` for [`DEFAULT_MAX_LEAF`].
    pub max_leaf: Option<usize>,
}

/// Leaves of 16 run four NEON registers or one AVX-512 register deep: on 4-wide hardware, a
/// node visit costs more than testing four more primitives a register at a time.
pub const DEFAULT_MAX_LEAF: usize = 16;

/// The node width a default-built [`Bvh`] uses here: the `f32` lanes of the widest backend
/// runtime detection picks, rounded up to 4, 8 or 16.
#[must_use]
pub fn native_width() -> usize {
    struct Lanes;
    impl Kernel<f32> for Lanes {
        type Output = usize;
        fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> usize {
            g.lanes::<f32>()
        }
    }
    supported_width(dispatch(Lanes))
}

fn supported_width(w: usize) -> usize {
    match w {
        0..=4 => 4,
        5..=8 => 8,
        _ => 16,
    }
}

/// Leaf visitor for [`Bvh::ray`]. A closure `FnMut(Range<usize>, f32) -> Option<f32>` is one.
pub trait RayLeaves {
    /// The nearest hit nearer than `t_max` among the primitives at `slots`, or `None`. Every
    /// hit returned is nearer than all before it in the query, so the last one is the answer.
    fn hit<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
        t_max: f32,
    ) -> Option<f32>;
}

impl<F: FnMut(Range<usize>, f32) -> Option<f32>> RayLeaves for F {
    #[inline]
    fn hit<S: BackendAll + Backend<f32>>(
        &mut self,
        _: Gang<S>,
        slots: Range<usize>,
        t_max: f32,
    ) -> Option<f32> {
        self(slots, t_max)
    }
}

/// Leaf visitor for [`Bvh::nearest`]. A closure `FnMut(Range<usize>, f32) -> Option<f32>` is
/// one.
pub trait NearestLeaves {
    /// The smallest squared distance below `best_d2` among the primitives at `slots`, or
    /// `None`. As with [`RayLeaves::hit`], the last one returned is the answer.
    fn nearest<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
        best_d2: f32,
    ) -> Option<f32>;
}

impl<F: FnMut(Range<usize>, f32) -> Option<f32>> NearestLeaves for F {
    #[inline]
    fn nearest<S: BackendAll + Backend<f32>>(
        &mut self,
        _: Gang<S>,
        slots: Range<usize>,
        best_d2: f32,
    ) -> Option<f32> {
        self(slots, best_d2)
    }
}

/// Leaf visitor for [`Bvh::overlapping`] and [`Bvh::containing`]. A closure
/// `FnMut(Range<usize>) -> ControlFlow<()>` is one.
pub trait OverlapLeaves {
    /// Visits a leaf whose box meets the query; `Break` ends the query.
    fn visit<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
    ) -> ControlFlow<()>;
}

impl<F: FnMut(Range<usize>) -> ControlFlow<()>> OverlapLeaves for F {
    #[inline]
    fn visit<S: BackendAll + Backend<f32>>(
        &mut self,
        _: Gang<S>,
        slots: Range<usize>,
    ) -> ControlFlow<()> {
        self(slots)
    }
}

enum Tree {
    W4(Vec<Node<4>>),
    W8(Vec<Node<8>>),
    W16(Vec<Node<16>>),
}

macro_rules! with_nodes {
    ($tree:expr, $nodes:ident => $body:expr) => {
        match $tree {
            Tree::W4($nodes) => $body,
            Tree::W8($nodes) => $body,
            Tree::W16($nodes) => $body,
        }
    };
}

/// A bounding volume hierarchy over primitive boxes, built with binned SAH splits and
/// collapsed to nodes as wide as the SIMD registers. See the [module docs](self).
pub struct Bvh {
    tree: Tree,
    order: Vec<u32>,
    bounds: Aabb,
}

impl Bvh {
    #[must_use]
    pub fn new(boxes: &[Aabb]) -> Self {
        Self::with_options(boxes, Options::default())
    }

    #[must_use]
    pub fn with_options(boxes: &[Aabb], options: Options) -> Self {
        let width = supported_width(options.width.unwrap_or_else(native_width));
        let max_leaf = options.max_leaf.unwrap_or(DEFAULT_MAX_LEAF).max(1);
        let bin = Binary::build(boxes, max_leaf);
        let tree = match width {
            4 => Tree::W4(wide::collapse(&bin)),
            8 => Tree::W8(wide::collapse(&bin)),
            _ => Tree::W16(wide::collapse(&bin)),
        };
        let bounds = with_nodes!(&tree, n => wide::root_bounds(n));
        Self {
            tree,
            order: bin.order,
            bounds,
        }
    }

    /// Children per node: 4, 8 or 16.
    #[must_use]
    pub fn width(&self) -> usize {
        match self.tree {
            Tree::W4(_) => 4,
            Tree::W8(_) => 8,
            Tree::W16(_) => 16,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The box around every primitive.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.bounds
    }

    /// Slot to primitive index: the slots a query hands to a leaf visitor index this.
    #[must_use]
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// `data` (one entry per primitive) rearranged into slot order, so a leaf's entries are
    /// contiguous.
    #[must_use]
    pub fn in_slot_order<T: Copy>(&self, data: &[T]) -> Vec<T> {
        self.order.iter().map(|&pi| data[pi as usize]).collect()
    }

    /// Updates every node box for primitives that moved, keeping the tree's topology. Cheap,
    /// but the tree degrades as the motion departs from the one it was built for.
    pub fn refit(&mut self, boxes: &[Aabb]) {
        assert_eq!(boxes.len(), self.order.len(), "Bvh::refit: one box per primitive");
        let order = &self.order;
        with_nodes!(&mut self.tree, n => wide::refit(n, order, boxes));
        self.bounds = with_nodes!(&self.tree, n => wide::root_bounds(n));
    }

    /// Nearest hit along `origin + t * dir` for `0 <= t < t_max`, as found by `leaves`, which
    /// sees the leaves the ray enters roughly nearest first.
    pub fn ray<L: RayLeaves>(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_max: f32,
        leaves: &mut L,
    ) -> Option<f32> {
        if self.is_empty() {
            return None;
        }
        with_nodes!(&self.tree, nodes => dispatch(RayQuery { nodes, origin, dir, t_max, leaves }))
    }

    /// Smallest squared distance from `p` below `max_d2`, as found by `leaves`, which sees the
    /// leaves within the current best distance roughly nearest first.
    pub fn nearest<L: NearestLeaves>(&self, p: [f32; 3], max_d2: f32, leaves: &mut L) -> Option<f32> {
        if self.is_empty() {
            return None;
        }
        with_nodes!(&self.tree, nodes => dispatch(NearestQuery { nodes, p, max_d2, leaves }))
    }

    /// Visits every leaf whose box meets `query`, until `leaves` breaks.
    pub fn overlapping<L: OverlapLeaves>(&self, query: &Aabb, leaves: &mut L) -> ControlFlow<()> {
        if self.is_empty() || !self.bounds.overlaps(query) {
            return ControlFlow::Continue(());
        }
        let query = *query;
        with_nodes!(&self.tree, nodes => dispatch(OverlapQuery { nodes, query, leaves }))
    }

    /// Visits every leaf whose box contains `p`, until `leaves` breaks.
    pub fn containing<L: OverlapLeaves>(&self, p: [f32; 3], leaves: &mut L) -> ControlFlow<()> {
        self.overlapping(&Aabb::point(p), leaves)
    }
}

/// LIFO traversal stack: an inline array covers any realistic depth, spilling to the heap only
/// for pathological builds, so queries stay allocation-free. The array is left uninitialized:
/// zeroing it would cost a 1 KiB `memset` on every query.
struct Stack<T> {
    inline: [MaybeUninit<T>; 128],
    len: usize,
    spill: Vec<T>,
}

impl<T: Copy> Stack<T> {
    #[inline(always)]
    fn new() -> Self {
        Self {
            inline: [const { MaybeUninit::uninit() }; 128],
            len: 0,
            spill: Vec::new(),
        }
    }

    #[inline(always)]
    fn push(&mut self, v: T) {
        if self.len < self.inline.len() {
            self.inline[self.len].write(v);
            self.len += 1;
        } else {
            self.spill.push(v);
        }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        if let Some(v) = self.spill.pop() {
            return Some(v);
        }
        self.len.checked_sub(1).map(|l| {
            self.len = l;
            // SAFETY: every slot below `len` was written by `push`.
            unsafe { self.inline[l].assume_init() }
        })
    }
}
