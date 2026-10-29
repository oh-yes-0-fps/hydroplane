//! The binary tree every wide tree is collapsed from, built top-down with binned
//! surface-area-heuristic splits (Wald 2007).

use super::Aabb;

const BINS: usize = 16;

#[derive(Clone, Copy)]
pub(super) struct Node {
    pub(super) bounds: Aabb,
    /// Leaf: first slot. Internal: index of the left child (right is `+ 1`).
    pub(super) first: u32,
    /// Primitives in the leaf; `0` marks an internal node.
    pub(super) count: u32,
}

impl Node {
    #[inline]
    pub(super) fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

pub(super) struct Binary {
    pub(super) nodes: Vec<Node>,
    /// Slot to primitive: every leaf owns a contiguous run of slots.
    pub(super) order: Vec<u32>,
}

impl Binary {
    /// Splits until leaves hold at most `max_leaf` primitives. Empty input yields an empty tree.
    pub(super) fn build(boxes: &[Aabb], max_leaf: usize) -> Self {
        let n = boxes.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        if n == 0 {
            return Self { nodes: Vec::new(), order };
        }
        let centroids: Vec<[f32; 3]> = boxes.iter().map(Aabb::center).collect();

        let pending = |first: usize, count: usize| Node {
            bounds: Aabb::EMPTY,
            first: first as u32,
            count: count as u32,
        };
        let mut nodes = Vec::with_capacity(2 * n.div_ceil(max_leaf.max(1)));
        nodes.push(pending(0, n));

        // Nodes awaiting processing still hold their slot range in (first, count).
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            let start = nodes[ni].first as usize;
            let count = nodes[ni].count as usize;
            let range = &mut order[start..start + count];

            let mut bounds = Aabb::EMPTY;
            let mut cb = Aabb::EMPTY;
            for &pi in range.iter() {
                bounds = bounds.union(&boxes[pi as usize]);
                cb = cb.union(&Aabb::point(centroids[pi as usize]));
            }
            nodes[ni].bounds = bounds;
            if count <= max_leaf {
                continue;
            }

            let mid = split(range, boxes, &centroids, &cb).unwrap_or(count / 2);
            let left = nodes.len();
            nodes.push(pending(start, mid));
            nodes.push(pending(start + mid, count - mid));
            nodes[ni].first = left as u32;
            nodes[ni].count = 0;
            stack.push(left);
            stack.push(left + 1);
        }
        Self { nodes, order }
    }
}

/// Half the surface area; relative SAH costs only need proportionality.
#[inline]
pub(super) fn half_area(b: &Aabb) -> f32 {
    let e = [0, 1, 2].map(|a| (b.max[a] - b.min[a]).max(0.0));
    e[0] * e[1] + e[1] * e[2] + e[2] * e[0]
}

/// Partitions `range` around the best binned-SAH split of the centroid bounds `cb`, returning
/// the split offset. `None` when no split separates the primitives (coincident centroids); the
/// caller falls back to the median.
fn split(range: &mut [u32], boxes: &[Aabb], centroids: &[[f32; 3]], cb: &Aabb) -> Option<usize> {
    let ext = [0, 1, 2].map(|a| cb.max[a] - cb.min[a]);
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0
    } else if ext[1] >= ext[2] {
        1
    } else {
        2
    };
    let extent = ext[axis];
    // `<=` alone would let a NaN extent through to the binning math.
    if extent.is_nan() || extent <= 0.0 {
        return None;
    }
    let scale = BINS as f32 / extent;
    let origin = cb.min[axis];
    let bin_of =
        |pi: u32| (((centroids[pi as usize][axis] - origin) * scale) as usize).min(BINS - 1);

    let mut bin_boxes = [Aabb::EMPTY; BINS];
    let mut bin_counts = [0u32; BINS];
    for &pi in range.iter() {
        let b = bin_of(pi);
        bin_counts[b] += 1;
        bin_boxes[b] = bin_boxes[b].union(&boxes[pi as usize]);
    }

    let mut right_cost = [f32::INFINITY; BINS];
    let mut acc = Aabb::EMPTY;
    let mut cnt = 0u32;
    for k in (1..BINS).rev() {
        acc = acc.union(&bin_boxes[k]);
        cnt += bin_counts[k];
        if cnt > 0 {
            right_cost[k] = half_area(&acc) * cnt as f32;
        }
    }

    let mut best_k = 0;
    let mut best_cost = f32::INFINITY;
    let mut acc = Aabb::EMPTY;
    let mut cnt = 0u32;
    for k in 1..BINS {
        acc = acc.union(&bin_boxes[k - 1]);
        cnt += bin_counts[k - 1];
        if cnt == 0 {
            continue;
        }
        let cost = half_area(&acc) * cnt as f32 + right_cost[k];
        if cost < best_cost {
            best_cost = cost;
            best_k = k;
        }
    }
    if best_k == 0 {
        return None;
    }

    let mut i = 0;
    let mut j = range.len();
    while i < j {
        if bin_of(range[i]) < best_k {
            i += 1;
        } else {
            j -= 1;
            range.swap(i, j);
        }
    }
    (i > 0 && i < range.len()).then_some(i)
}
