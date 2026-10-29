//! `W`-wide nodes collapsed from the binary tree, and the single-query traversals over them:
//! one node visit tests all `W` child boxes in `W / lanes` SoA passes, so with `W` equal to the
//! register width it is one pass with no horizontal reduction, at a tree `log2 W` times shallower.

use core::ops::{ControlFlow, Range};

use hydroplane::{Backend, BackendAll, Gang, Kernel, Varying};

use super::binary::{Binary, half_area};
use super::{Aabb, NearestLeaves, OverlapLeaves, RayLeaves, Stack};

/// Widening of the slab exit that absorbs the rounding of `(bound - origin) * inv_dir`
/// (Ize, *Robust BVH Ray Traversal*: `1 + 2γ₃`), so a ray grazing a box corner or face is
/// never culled from a primitive it touches.
const SLAB_EXIT_SCALE: f32 =
    1.0 + 2.0 * (3.0 * f32::EPSILON * 0.5) / (1.0 - 3.0 * f32::EPSILON * 0.5);

/// `W` child boxes in SoA form (`min[axis]` holds that axis for every child). Children fill
/// lanes `0..len`; the rest have both bounds at `+∞`, which every ray and point misses.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(super) struct Node<const W: usize> {
    pub(super) min: [[f32; W]; 3],
    pub(super) max: [[f32; W]; 3],
    /// Leaf lane: first slot. Internal lane: index of the child node.
    pub(super) child: [u32; W],
    /// Primitives in a leaf lane; `0` marks an internal lane.
    pub(super) count: [u32; W],
    pub(super) len: u32,
}

impl<const W: usize> Node<W> {
    const EMPTY: Self = Self {
        min: [[f32::INFINITY; W]; 3],
        max: [[f32::INFINITY; W]; 3],
        child: [0; W],
        count: [0; W],
        len: 0,
    };

    #[inline(always)]
    fn slots(&self, lane: usize) -> Range<usize> {
        let first = self.child[lane] as usize;
        first..first + self.count[lane] as usize
    }

    fn set(&mut self, lane: usize, b: &Aabb) {
        for axis in 0..3 {
            self.min[axis][lane] = b.min[axis];
            self.max[axis][lane] = b.max[axis];
        }
    }

    fn bounds(&self) -> Aabb {
        (0..self.len as usize).fold(Aabb::EMPTY, |acc, lane| {
            acc.union(&Aabb {
                min: [0, 1, 2].map(|a| self.min[a][lane]),
                max: [0, 1, 2].map(|a| self.max[a][lane]),
            })
        })
    }
}

/// Collapses the binary tree into `W`-wide nodes, repeatedly opening the internal child with
/// the largest surface area (the one a ray is most likely to enter) until `W` children remain.
/// Leaves keep their slot ranges. A parent always precedes its children.
pub(super) fn collapse<const W: usize>(bin: &Binary) -> Vec<Node<W>> {
    let mut out = Vec::new();
    if !bin.nodes.is_empty() {
        collapse_node(bin, 0, &mut out);
    }
    out
}

fn collapse_node<const W: usize>(bin: &Binary, ni: u32, out: &mut Vec<Node<W>>) -> u32 {
    let idx = out.len();
    out.push(Node::EMPTY);

    let node = &bin.nodes[ni as usize];
    let mut kids = [0u32; W];
    let mut len = if node.is_leaf() {
        kids[0] = ni;
        1
    } else {
        kids[0] = node.first;
        kids[1] = node.first + 1;
        2
    };
    while len < W {
        let area = |j: usize| half_area(&bin.nodes[kids[j] as usize].bounds);
        let open = (0..len)
            .filter(|&j| !bin.nodes[kids[j] as usize].is_leaf())
            .max_by(|&a, &b| area(a).total_cmp(&area(b)));
        let Some(j) = open else { break };
        let first = bin.nodes[kids[j] as usize].first;
        kids[j] = first;
        kids[len] = first + 1;
        len += 1;
    }

    out[idx].len = len as u32;
    for (lane, &k) in kids[..len].iter().enumerate() {
        let c = &bin.nodes[k as usize];
        out[idx].set(lane, &c.bounds);
        if c.is_leaf() {
            out[idx].child[lane] = c.first;
            out[idx].count[lane] = c.count;
        } else {
            out[idx].child[lane] = collapse_node(bin, k, out);
        }
    }
    idx as u32
}

/// Recomputes every box from new primitive bounds, keeping the topology. Children follow
/// their parent, so one reverse sweep sees every child before its parent.
pub(super) fn refit<const W: usize>(nodes: &mut [Node<W>], order: &[u32], boxes: &[Aabb]) {
    for ni in (0..nodes.len()).rev() {
        for lane in 0..nodes[ni].len as usize {
            let b = if nodes[ni].count[lane] > 0 {
                order[nodes[ni].slots(lane)]
                    .iter()
                    .fold(Aabb::EMPTY, |acc, &pi| acc.union(&boxes[pi as usize]))
            } else {
                nodes[nodes[ni].child[lane] as usize].bounds()
            };
            nodes[ni].set(lane, &b);
        }
    }
}

pub(super) fn root_bounds<const W: usize>(nodes: &[Node<W>]) -> Aabb {
    nodes.first().map_or(Aabb::EMPTY, Node::bounds)
}

/// Lanes `c..c + lanes` of a node row, past its end reading as `+∞`.
#[inline(always)]
fn row<S: Backend<f32>, const W: usize>(
    g: Gang<S>,
    r: &[f32; W],
    c: usize,
    lanes: usize,
) -> Varying<f32, S> {
    g.load_partial(&r[c..(c + lanes).min(W)], f32::INFINITY)
}

/// Each component broadcast. Hot kernel code avoids `[T; N]::map`: its closure goes through
/// `core::array`'s `try_map` machinery, which is not reliably inlined, and an out-of-line closure
/// is compiled without the backend's target features, so every op in it becomes a call.
#[inline(always)]
fn splat3<S: Backend<f32>>(g: Gang<S>, v: [f32; 3]) -> [Varying<f32, S>; 3] {
    [g.splat(v[0]), g.splat(v[1]), g.splat(v[2])]
}

/// Bits `0..n` set.
#[inline(always)]
fn low_bits(n: usize) -> u32 {
    if n >= 32 { u32::MAX } else { (1 << n) - 1 }
}

/// The best-first order of a node's children: one register of child lanes, starting at lane
/// `c`, to a key per lane (entry `t`, squared distance), `+∞` for a child to skip. A trait with an
/// `#[inline(always)]` method rather than a closure, so it is sure to inline into the
/// target-feature function the query runs in.
trait ChildKey<S: Backend<f32>, const W: usize> {
    fn key(&self, n: &Node<W>, c: usize, bound: f32) -> Varying<f32, S>;
}

/// The node's children whose key comes out below `bound`, nearest first.
#[inline(always)]
fn sorted_below<S: Backend<f32>, const W: usize, K: ChildKey<S, W>>(
    g: Gang<S>,
    node: &Node<W>,
    bound: f32,
    key: &K,
) -> Sorted<W> {
    let lanes = g.lanes::<f32>();
    let len = node.len as usize;
    let b = g.splat(bound);
    let mut keys = [0.0f32; W];
    let mut hits = Sorted::new();
    let mut c = 0;
    while c < len {
        let n = (len - c).min(lanes);
        let k = key.key(node, c, bound);
        let mut bits = k.lt(b).to_bitmask() & low_bits(n);
        if bits != 0 {
            k.store_partial(&mut keys[c..(c + lanes).min(W)]);
            while bits != 0 {
                let lane = c + bits.trailing_zeros() as usize;
                hits.insert(lane, keys[lane]);
                bits &= bits - 1;
            }
        }
        c += lanes;
    }
    hits
}

/// Lanes ordered by key, nearest first: `(lane, key)`.
struct Sorted<const W: usize> {
    items: [(u32, f32); W],
    len: usize,
}

impl<const W: usize> Sorted<W> {
    #[inline(always)]
    fn new() -> Self {
        Self { items: [(0, 0.0); W], len: 0 }
    }

    #[inline(always)]
    fn insert(&mut self, lane: usize, key: f32) {
        let mut p = self.len;
        while p > 0 && self.items[p - 1].1 > key {
            self.items[p] = self.items[p - 1];
            p -= 1;
        }
        self.items[p] = (lane as u32, key);
        self.len += 1;
    }

    #[inline(always)]
    fn as_slice(&self) -> &[(u32, f32)] {
        &self.items[..self.len]
    }
}

/// Best-first descent shared by the ray and nearest-point queries: visit the node's leaves
/// nearest first (each improvement tightens `best` for every box after it), then descend into
/// the nearest internal child, deferring the rest farthest-first. Returns whether any leaf
/// improved on the initial `best`.
#[inline(always)]
fn best_first<S: Backend<f32>, const W: usize, K: ChildKey<S, W>>(
    g: Gang<S>,
    nodes: &[Node<W>],
    best: &mut f32,
    key: &K,
    mut leaf: impl FnMut(Range<usize>, f32) -> Option<f32>,
) -> bool {
    let mut improved = false;
    let mut stack: Stack<(u32, f32)> = Stack::new();
    let mut ni = 0u32;
    loop {
        let node = &nodes[ni as usize];
        let hits = sorted_below(g, node, *best, key);

        for &(lane, k) in hits.as_slice() {
            let lane = lane as usize;
            if node.count[lane] > 0
                && k < *best
                && let Some(v) = leaf(node.slots(lane), *best)
                && v < *best
            {
                *best = v;
                improved = true;
            }
        }

        let mut next = None;
        for &(lane, k) in hits.as_slice().iter().rev() {
            let lane = lane as usize;
            if node.count[lane] == 0
                && k < *best
                && let Some(deferred) = next.replace((node.child[lane], k))
            {
                stack.push(deferred);
            }
        }
        if let Some((child, _)) = next {
            ni = child;
            continue;
        }
        loop {
            match stack.pop() {
                Some((child, k)) if k < *best => {
                    ni = child;
                    break;
                }
                Some(_) => {}
                None => return improved,
            }
        }
    }
}

/// `1 / dir` with zero components mapped to a signed huge finite value, so a ray lying in a
/// box's face plane yields `0 · big = 0` instead of a NaN the vector `min`/`max` would turn
/// into a spurious miss.
#[inline]
fn safe_inv_dir(dir: [f32; 3]) -> [f32; 3] {
    dir.map(|d| if d.abs() < 1e-30 { 1e30f32.copysign(d) } else { 1.0 / d })
}

/// Ray entry `t` into each child box, by slab test.
struct Entry<S: Backend<f32>> {
    g: Gang<S>,
    lanes: usize,
    o: [Varying<f32, S>; 3],
    inv: [Varying<f32, S>; 3],
}

impl<S: Backend<f32>> Entry<S> {
    #[inline(always)]
    fn slab<const W: usize>(
        &self,
        n: &Node<W>,
        a: usize,
        c: usize,
    ) -> (Varying<f32, S>, Varying<f32, S>) {
        let t0 = (row(self.g, &n.min[a], c, self.lanes) - self.o[a]) * self.inv[a];
        let t1 = (row(self.g, &n.max[a], c, self.lanes) - self.o[a]) * self.inv[a];
        (t0.min(t1), t0.max(t1))
    }
}

impl<S: Backend<f32>, const W: usize> ChildKey<S, W> for Entry<S> {
    #[inline(always)]
    fn key(&self, n: &Node<W>, c: usize, bound: f32) -> Varying<f32, S> {
        let (nx, fx) = self.slab(n, 0, c);
        let (ny, fy) = self.slab(n, 1, c);
        let (nz, fz) = self.slab(n, 2, c);
        let near = nx.max(ny).max(nz.max(self.g.splat(0.0)));
        let far = (fx.min(fy).min(fz) * SLAB_EXIT_SCALE).min(self.g.splat(bound));
        near.select(near.le(far), self.g.splat(f32::INFINITY))
    }
}

/// Squared distance from a point to each child box.
struct Dist2<S: Backend<f32>> {
    g: Gang<S>,
    lanes: usize,
    p: [Varying<f32, S>; 3],
}

impl<S: Backend<f32>> Dist2<S> {
    #[inline(always)]
    fn gap<const W: usize>(&self, n: &Node<W>, a: usize, c: usize) -> Varying<f32, S> {
        let lo = row(self.g, &n.min[a], c, self.lanes) - self.p[a];
        let hi = self.p[a] - row(self.g, &n.max[a], c, self.lanes);
        lo.max(hi).max(self.g.splat(0.0))
    }
}

impl<S: Backend<f32>, const W: usize> ChildKey<S, W> for Dist2<S> {
    #[inline(always)]
    fn key(&self, n: &Node<W>, c: usize, _: f32) -> Varying<f32, S> {
        let (dx, dy, dz) = (self.gap(n, 0, c), self.gap(n, 1, c), self.gap(n, 2, c));
        dx * dx + dy * dy + dz * dz
    }
}

pub(super) struct RayQuery<'a, const W: usize, L> {
    pub(super) nodes: &'a [Node<W>],
    pub(super) origin: [f32; 3],
    pub(super) dir: [f32; 3],
    pub(super) t_max: f32,
    pub(super) leaves: &'a mut L,
}

impl<const W: usize, L: RayLeaves> Kernel<f32> for RayQuery<'_, W, L> {
    type Output = Option<f32>;

    #[inline]
    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> Option<f32> {
        let entry = Entry {
            g,
            lanes: g.lanes::<f32>(),
            o: splat3(g, self.origin),
            inv: splat3(g, safe_inv_dir(self.dir)),
        };
        let leaves = self.leaves;
        let mut best = self.t_max;
        let improved =
            best_first(g, self.nodes, &mut best, &entry, |slots, t| leaves.hit(g, slots, t));
        improved.then_some(best)
    }
}

pub(super) struct NearestQuery<'a, const W: usize, L> {
    pub(super) nodes: &'a [Node<W>],
    pub(super) p: [f32; 3],
    pub(super) max_d2: f32,
    pub(super) leaves: &'a mut L,
}

impl<const W: usize, L: NearestLeaves> Kernel<f32> for NearestQuery<'_, W, L> {
    type Output = Option<f32>;

    #[inline]
    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> Option<f32> {
        let dist2 = Dist2 {
            g,
            lanes: g.lanes::<f32>(),
            p: splat3(g, self.p),
        };
        let leaves = self.leaves;
        let mut best = self.max_d2;
        let improved =
            best_first(g, self.nodes, &mut best, &dist2, |slots, d| leaves.nearest(g, slots, d));
        improved.then_some(best)
    }
}

pub(super) struct OverlapQuery<'a, const W: usize, L> {
    pub(super) nodes: &'a [Node<W>],
    pub(super) query: Aabb,
    pub(super) leaves: &'a mut L,
}

impl<const W: usize, L: OverlapLeaves> Kernel<f32> for OverlapQuery<'_, W, L> {
    type Output = ControlFlow<()>;

    #[inline]
    fn run<S: BackendAll + Backend<f32>>(self, g: Gang<S>) -> ControlFlow<()> {
        let lanes = g.lanes::<f32>();
        let qmin = splat3(g, self.query.min);
        let qmax = splat3(g, self.query.max);
        let mut stack: Stack<u32> = Stack::new();
        let mut ni = 0u32;
        loop {
            let node = &self.nodes[ni as usize];
            let len = node.len as usize;
            let mut c = 0;
            while c < len {
                let mut m = row(g, &node.min[0], c, lanes).le(qmax[0])
                    & row(g, &node.max[0], c, lanes).ge(qmin[0]);
                for a in 1..3 {
                    m = m
                        & row(g, &node.min[a], c, lanes).le(qmax[a])
                        & row(g, &node.max[a], c, lanes).ge(qmin[a]);
                }
                let mut bits = m.to_bitmask() & low_bits(len - c);
                while bits != 0 {
                    let lane = c + bits.trailing_zeros() as usize;
                    if node.count[lane] > 0 {
                        self.leaves.visit(g, node.slots(lane))?;
                    } else {
                        stack.push(node.child[lane]);
                    }
                    bits &= bits - 1;
                }
                c += lanes;
            }
            match stack.pop() {
                Some(n) => ni = n,
                None => return ControlFlow::Continue(()),
            }
        }
    }
}
