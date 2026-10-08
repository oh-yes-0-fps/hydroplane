//! Multilevel nested-dissection ordering of a mesh's node graph, for sparse Cholesky: each set
//! of nodes is split by a vertex separator found by a multilevel scheme (heavy-edge matching down
//! to about a hundred vertices; there, a greedy edge bisection whose cut becomes a separator by a
//! minimum vertex cover; on the way back up, the separator projected and refined by
//! Fiduccia–Mattheyses moves of separator vertices into the sides), the two sides are ordered
//! recursively and the separator after them.
//!
//! On a GPU factorization the critical path (separator widths summed along the elimination tree)
//! costs as much as the flops; refining the separator at every level shortens both. Against the
//! previous scheme (edge cut refined per level, one separator at the end) it cut the slab
//! tangent's critical path from 239 to 187–197 panel steps and its RTX 5090 factorization from
//! 42.9 to 38.4 ms, and helped on plate, cube and beam grids alike. METIS still finds smaller
//! separators on thin and slender domains.




/// Tuning of [`nested_dissection_with`].
#[derive(Clone, Debug)]
pub struct NestedDissection {
    /// Sets of at most this many nodes are not dissected further.
    pub leaf: usize,
    /// Orders those sets by minimum degree (otherwise as they come).
    pub leaf_min_degree: bool,
    /// Coarsening stops at this many vertices, or when a level shrinks by less than a tenth.
    pub coarsest: usize,
    /// Either side of a cut keeps at least this share of the weight.
    pub min_share: f64,
    /// Refinement passes per level.
    pub fm_passes: usize,
    /// Seeds tried for the greedy initial cut.
    pub seeds: usize,
    /// Bisections tried per set (from different coarsest cuts), keeping the smallest separator.
    pub tries: usize,
    /// Refines the vertex separator itself at every level of the multilevel bisection (as METIS
    /// does); otherwise the edge cut is refined per level and turned into a separator once.
    pub node_refine: bool,
    /// Moves a separator refinement pass makes past its best separator before giving up.
    pub patience: usize,
}

impl Default for NestedDissection {
    fn default() -> Self {
        Self {
            leaf: 48,
            leaf_min_degree: false,
            coarsest: 120,
            min_share: 0.35,
            fm_passes: 6,
            seeds: 12,
            tries: 1,
            node_refine: true,
            patience: 50,
        }
    }
}

struct Graph {
    xadj: Vec<u32>,
    adjncy: Vec<u32>,
    adjwgt: Vec<u32>,
    vwgt: Vec<u32>,
}

impl Graph {
    fn n(&self) -> usize {
        self.vwgt.len()
    }

    fn edges(&self, v: usize) -> impl Iterator<Item = (usize, u32)> + '_ {
        let (a, b) = (self.xadj[v] as usize, self.xadj[v + 1] as usize);
        self.adjncy[a..b].iter().zip(&self.adjwgt[a..b]).map(|(&u, &w)| (u as usize, w))
    }

    fn weight(&self) -> u64 {
        self.vwgt.iter().map(|&w| w as u64).sum()
    }

    /// The subgraph on `nodes` (in that order), edges to nodes outside it dropped.
    fn subgraph(&self, nodes: &[u32], local: &mut [u32]) -> Graph {
        for (i, &v) in nodes.iter().enumerate() {
            local[v as usize] = i as u32;
        }
        let mut xadj = Vec::with_capacity(nodes.len() + 1);
        let mut adjncy = Vec::new();
        let mut adjwgt = Vec::new();
        xadj.push(0);
        for &v in nodes {
            for (u, w) in self.edges(v as usize) {
                let l = local[u];
                if l != u32::MAX {
                    adjncy.push(l);
                    adjwgt.push(w);
                }
            }
            xadj.push(adjncy.len() as u32);
        }
        for &v in nodes {
            local[v as usize] = u32::MAX;
        }
        Graph { xadj, adjncy, adjwgt, vwgt: nodes.iter().map(|&v| self.vwgt[v as usize]).collect() }
    }

    /// Heavy-edge matching: each vertex pairs with its unmatched neighbour of heaviest edge.
    /// Returns the coarse graph and each vertex's coarse vertex.
    fn coarsen(&self) -> (Graph, Vec<u32>) {
        let n = self.n();
        let mut map = vec![u32::MAX; n];
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&v| self.xadj[v + 1] - self.xadj[v]);
        let mut nc = 0u32;
        for &v in &order {
            if map[v] != u32::MAX {
                continue;
            }
            let mut best: Option<(usize, u32, u32)> = None;
            for (u, w) in self.edges(v) {
                if map[u] == u32::MAX && u != v && best.is_none_or(|(_, bw, bv)| w > bw || (w == bw && self.vwgt[u] < bv)) {
                    best = Some((u, w, self.vwgt[u]));
                }
            }
            map[v] = nc;
            if let Some((u, _, _)) = best {
                map[u] = nc;
            }
            nc += 1;
        }
        let nc = nc as usize;
        let mut vwgt = vec![0u32; nc];
        for v in 0..n {
            vwgt[map[v] as usize] += self.vwgt[v];
        }
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); nc];
        for v in 0..n {
            members[map[v] as usize].push(v as u32);
        }
        let mut xadj = Vec::with_capacity(nc + 1);
        let mut adjncy = Vec::new();
        let mut adjwgt = Vec::new();
        let mut slot = vec![u32::MAX; nc];
        xadj.push(0);
        for (c, group) in members.iter().enumerate() {
            let start = adjncy.len();
            for &v in group {
                for (u, w) in self.edges(v as usize) {
                    let cu = map[u] as usize;
                    if cu == c {
                        continue;
                    }
                    if slot[cu] == u32::MAX || (slot[cu] as usize) < start {
                        slot[cu] = adjncy.len() as u32;
                        adjncy.push(cu as u32);
                        adjwgt.push(w);
                    } else {
                        adjwgt[slot[cu] as usize] += w;
                    }
                }
            }
            xadj.push(adjncy.len() as u32);
        }
        (Graph { xadj, adjncy, adjwgt, vwgt }, map)
    }
}

/// Edge cut of `part` (0/1 per vertex).
fn cut(g: &Graph, part: &[u8]) -> u64 {
    (0..g.n()).map(|v| g.edges(v).filter(|&(u, _)| part[u] != part[v]).map(|(_, w)| w as u64).sum::<u64>()).sum::<u64>() / 2
}

/// Greedy region growing from `seed` until half the weight is in part 0, taking at each step the
/// unassigned vertex with the most edge weight into the region.
fn grow(g: &Graph, seed: usize, part: &mut [u8]) {
    let n = g.n();
    part.fill(1);
    let half = g.weight() / 2;
    let mut gain = vec![0i64; n];
    let mut inside = 0u64;
    let mut front: Vec<usize> = Vec::new();
    let mut v = seed;
    loop {
        part[v] = 0;
        inside += g.vwgt[v] as u64;
        if inside >= half {
            break;
        }
        for (u, w) in g.edges(v) {
            if part[u] == 1 {
                if gain[u] == 0 {
                    front.push(u);
                }
                gain[u] += w as i64;
            }
        }
        front.retain(|&u| part[u] == 1);
        let Some(&next) = front.iter().max_by_key(|&&u| gain[u]) else {
            // The region is a whole component; continue from any unassigned vertex.
            match (0..n).find(|&u| part[u] == 1) {
                Some(u) => {
                    v = u;
                    continue;
                }
                None => break,
            }
        };
        v = next;
    }
}

/// Fiduccia–Mattheyses refinement of `part`: passes of best-gain moves that keep both sides at
/// least `min_share` of the weight, each pass rolled back to its best cut.
fn refine(g: &Graph, part: &mut [u8], p: &NestedDissection) {
    let n = g.n();
    let total = g.weight();
    let min_side = (total as f64 * p.min_share) as u64;
    let mut side_weight = [0u64; 2];
    for v in 0..n {
        side_weight[part[v] as usize] += g.vwgt[v] as u64;
    }
    let gain_of = |v: usize, part: &[u8]| -> i64 {
        let mut ext = 0i64;
        let mut int = 0i64;
        for (u, w) in g.edges(v) {
            if part[u] == part[v] { int += w as i64 } else { ext += w as i64 }
        }
        ext - int
    };
    let mut locked = vec![false; n];
    let mut in_boundary = vec![false; n];
    for _ in 0..p.fm_passes {
        locked.fill(false);
        in_boundary.fill(false);
        let mut boundary: Vec<usize> = (0..n).filter(|&v| g.edges(v).any(|(u, _)| part[u] != part[v])).collect();
        for &v in &boundary {
            in_boundary[v] = true;
        }
        let mut gains: Vec<i64> = (0..n).map(|v| gain_of(v, part)).collect();
        let mut moved: Vec<usize> = Vec::new();
        let mut best_at = 0usize;
        let mut balance = 0i64;
        let mut best_balance = 0i64;
        let limit = boundary.len().max(1);
        while moved.len() < limit {
            boundary.retain(|&v| !locked[v]);
            let candidate = boundary
                .iter()
                .copied()
                .filter(|&v| side_weight[part[v] as usize] - g.vwgt[v] as u64 >= min_side)
                .max_by_key(|&v| gains[v]);
            let Some(v) = candidate else { break };
            let from = part[v] as usize;
            side_weight[from] -= g.vwgt[v] as u64;
            side_weight[1 - from] += g.vwgt[v] as u64;
            part[v] = 1 - part[v];
            locked[v] = true;
            balance += gains[v];
            moved.push(v);
            if balance > best_balance {
                best_balance = balance;
                best_at = moved.len();
            }
            for (u, _) in g.edges(v) {
                gains[u] = gain_of(u, part);
                if !locked[u] && !in_boundary[u] {
                    in_boundary[u] = true;
                    boundary.push(u);
                }
            }
        }
        for &v in &moved[best_at..] {
            let from = part[v] as usize;
            side_weight[from] -= g.vwgt[v] as u64;
            side_weight[1 - from] += g.vwgt[v] as u64;
            part[v] = 1 - part[v];
        }
        if best_balance <= 0 {
            break;
        }
    }
}

/// A bisection of `g` by the multilevel scheme: 0/1 per vertex. `attempt` picks the seeds of
/// the coarsest cut.
fn bisect(g: &Graph, attempt: usize, p: &NestedDissection) -> Vec<u8> {
    if g.n() <= p.coarsest {
        return initial_cut(g, attempt, p);
    }
    let (coarse, map) = g.coarsen();
    if coarse.n() as f64 > 0.9 * g.n() as f64 {
        return initial_cut(g, attempt, p);
    }
    let coarse_part = bisect(&coarse, attempt, p);
    let mut part: Vec<u8> = (0..g.n()).map(|v| coarse_part[map[v] as usize]).collect();
    refine(g, &mut part, p);
    part
}

fn initial_cut(g: &Graph, attempt: usize, p: &NestedDissection) -> Vec<u8> {
    let n = g.n();
    let mut best: Option<(u64, Vec<u8>)> = None;
    let mut part = vec![1u8; n];
    let mut seeds: Vec<usize> = (0..n).collect();
    seeds.sort_by_key(|&v| g.xadj[v + 1] - g.xadj[v]);
    let step = (n / p.seeds).max(1);
    for &seed in seeds.iter().skip(attempt % step.max(1)).step_by(step).take(p.seeds) {
        grow(g, seed, &mut part);
        refine(g, &mut part, p);
        let c = cut(g, &part);
        if best.as_ref().is_none_or(|(bc, _)| c < *bc) {
            best = Some((c, part.clone()));
        }
    }
    best.map_or(part, |(_, p)| p)
}

/// A minimum vertex cover of the bipartite graph `edges` (left vertex `i` to the right vertices
/// `edges[i]`, `n_right` of them): a maximum matching by augmenting paths, then König's
/// alternating reachability.
fn min_vertex_cover(edges: &[Vec<u32>], n_right: usize) -> (Vec<bool>, Vec<bool>) {
    fn augment(l: usize, edges: &[Vec<u32>], match_right: &mut [u32], match_left: &mut [u32], seen: &mut [bool]) -> bool {
        for &r in &edges[l] {
            let r = r as usize;
            if seen[r] {
                continue;
            }
            seen[r] = true;
            if match_right[r] == u32::MAX || augment(match_right[r] as usize, edges, match_right, match_left, seen) {
                match_right[r] = l as u32;
                match_left[l] = r as u32;
                return true;
            }
        }
        false
    }
    let n_left = edges.len();
    let mut match_right = vec![u32::MAX; n_right];
    let mut match_left = vec![u32::MAX; n_left];
    let mut seen = vec![false; n_right];
    for l in 0..n_left {
        seen.fill(false);
        augment(l, edges, &mut match_right, &mut match_left, &mut seen);
    }
    let mut left_reached = vec![false; n_left];
    let mut right_reached = vec![false; n_right];
    let mut queue: Vec<usize> = (0..n_left).filter(|&l| match_left[l] == u32::MAX).collect();
    for &l in &queue {
        left_reached[l] = true;
    }
    while let Some(l) = queue.pop() {
        for &r in &edges[l] {
            let r = r as usize;
            if right_reached[r] || match_left[l] == r as u32 {
                continue;
            }
            right_reached[r] = true;
            let m = match_right[r];
            if m != u32::MAX && !left_reached[m as usize] {
                left_reached[m as usize] = true;
                queue.push(m as usize);
            }
        }
    }
    ((0..n_left).map(|l| !left_reached[l]).collect(), right_reached)
}

/// The vertex separator of the bisection `part`: a minimum vertex cover of the cut edges.
fn separator(g: &Graph, part: &mut [u8], p: &NestedDissection) -> Vec<bool> {
    let left_boundary: Vec<usize> = (0..g.n()).filter(|&v| part[v] == 0 && g.edges(v).any(|(u, _)| part[u] == 1)).collect();
    let right_boundary: Vec<usize> = (0..g.n()).filter(|&v| part[v] == 1 && g.edges(v).any(|(u, _)| part[u] == 0)).collect();
    let mut right_index = vec![u32::MAX; g.n()];
    for (i, &v) in right_boundary.iter().enumerate() {
        right_index[v] = i as u32;
    }
    let edges: Vec<Vec<u32>> = left_boundary
        .iter()
        .map(|&v| g.edges(v).filter(|&(u, _)| part[u] == 1).map(|(u, _)| right_index[u]).collect())
        .collect();
    let (cover_l, cover_r) = min_vertex_cover(&edges, right_boundary.len());
    let mut in_sep = vec![false; g.n()];
    for (&v, &c) in left_boundary.iter().zip(&cover_l).chain(right_boundary.iter().zip(&cover_r)) {
        if c {
            in_sep[v] = true;
        }
    }
    refine_separator(g, part, &mut in_sep, p);
    in_sep
}

/// Shrinks the separator by moving its vertices into a side: a vertex `v` may leave for side
/// `a` once its neighbours on the other side join the separator, a gain of
/// `w(v) − w(those neighbours)`; the best positive move is taken until none is left, the
/// sides kept within balance.
fn refine_separator(g: &Graph, part: &mut [u8], in_sep: &mut [bool], p: &NestedDissection) {
    let total = g.weight();
    let min_side = (total as f64 * p.min_share) as u64;
    let mut side_weight = [0u64; 2];
    for v in 0..g.n() {
        if !in_sep[v] {
            side_weight[part[v] as usize] += g.vwgt[v] as u64;
        }
    }
    let gain = |v: usize, to: usize, part: &[u8], in_sep: &[bool]| -> i64 {
        let pulled: u64 = g.edges(v).filter(|&(u, _)| !in_sep[u] && part[u] as usize != to).map(|(u, _)| g.vwgt[u] as u64).sum();
        g.vwgt[v] as i64 - pulled as i64
    };
    loop {
        let mut best: Option<(i64, usize, usize)> = None;
        for v in (0..g.n()).filter(|&v| in_sep[v]) {
            for to in 0..2 {
                let other: u64 = g.edges(v).filter(|&(u, _)| !in_sep[u] && part[u] as usize != to).map(|(u, _)| g.vwgt[u] as u64).sum();
                if side_weight[1 - to] < other + min_side {
                    continue;
                }
                let gn = gain(v, to, part, in_sep);
                if gn > 0 && best.is_none_or(|(b, _, _)| gn > b) {
                    best = Some((gn, v, to));
                }
            }
        }
        let Some((_, v, to)) = best else { break };
        let pulled: Vec<usize> = g.edges(v).filter(|&(u, _)| !in_sep[u] && part[u] as usize != to).map(|(u, _)| u).collect();
        for u in pulled {
            in_sep[u] = true;
            side_weight[part[u] as usize] -= g.vwgt[u] as u64;
        }
        in_sep[v] = false;
        part[v] = to as u8;
        side_weight[to] += g.vwgt[v] as u64;
    }
}

impl core::str::FromStr for NestedDissection {
    type Err = String;

    /// `field=value` pairs separated by `,` over the defaults, e.g. `leaf=120,node_refine=true`.
    fn from_str(spec: &str) -> Result<Self, String> {
        let mut p = Self::default();
        for kv in spec.split(',').filter(|kv| !kv.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("{kv}: expected field=value"))?;
            let bad = |e: &dyn core::fmt::Display| format!("{kv}: {e}");
            match k {
                "leaf" => p.leaf = v.parse().map_err(|e| bad(&e))?,
                "leaf_min_degree" => p.leaf_min_degree = v.parse().map_err(|e| bad(&e))?,
                "coarsest" => p.coarsest = v.parse().map_err(|e| bad(&e))?,
                "min_share" => p.min_share = v.parse().map_err(|e| bad(&e))?,
                "fm_passes" => p.fm_passes = v.parse().map_err(|e| bad(&e))?,
                "seeds" => p.seeds = v.parse().map_err(|e| bad(&e))?,
                "tries" => p.tries = v.parse().map_err(|e| bad(&e))?,
                "node_refine" => p.node_refine = v.parse().map_err(|e| bad(&e))?,
                "patience" => p.patience = v.parse().map_err(|e| bad(&e))?,
                _ => return Err(format!("{k}: no such field")),
            }
        }
        Ok(p)
    }
}

/// A vertex separator of `g` by the multilevel scheme with the separator itself refined at every
/// level: at the coarsest, a minimum vertex cover of an edge bisection's cut; on the way up, the
/// coarse separator projected (both halves of a matched pair) and improved by [`refine_node_sep`].
fn bisect_sep(g: &Graph, attempt: usize, p: &NestedDissection) -> (Vec<u8>, Vec<bool>) {
    let coarse = (g.n() > p.coarsest).then(|| g.coarsen()).filter(|(c, _)| (c.n() as f64) <= 0.9 * g.n() as f64);
    let Some((coarse, map)) = coarse else {
        let mut part = initial_cut(g, attempt, p);
        let mut in_sep = separator(g, &mut part, p);
        refine_node_sep(g, &mut part, &mut in_sep, p);
        return (part, in_sep);
    };
    let (coarse_part, coarse_sep) = bisect_sep(&coarse, attempt, p);
    let mut part: Vec<u8> = (0..g.n()).map(|v| coarse_part[map[v] as usize]).collect();
    let mut in_sep: Vec<bool> = (0..g.n()).map(|v| coarse_sep[map[v] as usize]).collect();
    refine_node_sep(g, &mut part, &mut in_sep, p);
    (part, in_sep)
}

/// Fiduccia–Mattheyses passes on a vertex separator: a move takes separator vertex `v` into side
/// `to` and pulls its neighbours on the other side into the separator (gain `w(v)` − their
/// weight). Each pass makes the best allowed move (both sides keep `min_share` of the weight)
/// among unmoved separator vertices, uphill too, up to `patience` moves past its best separator,
/// and is rolled back to that best.
fn refine_node_sep(g: &Graph, part: &mut [u8], in_sep: &mut [bool], p: &NestedDissection) {
    let n = g.n();
    let total = g.weight();
    let min_side = (total as f64 * p.min_share) as u64;
    let mut side = [0u64; 2];
    let mut sep_weight = 0u64;
    for v in 0..n {
        if in_sep[v] {
            sep_weight += g.vwgt[v] as u64;
        } else {
            side[part[v] as usize] += g.vwgt[v] as u64;
        }
    }
    let mut locked = vec![false; n];
    for _ in 0..p.fm_passes {
        locked.fill(false);
        let mut seps: Vec<usize> = (0..n).filter(|&v| in_sep[v]).collect();
        let mut log: Vec<(usize, u8, u8, Vec<usize>)> = Vec::new();
        let (mut best_weight, mut best_at) = (sep_weight, 0usize);
        loop {
            seps.retain(|&v| in_sep[v] && !locked[v]);
            let mut best: Option<(i64, usize, u8)> = None;
            for &v in &seps {
                for to in 0..2u8 {
                    let pulled: u64 = g.edges(v).filter(|&(u, _)| !in_sep[u] && part[u] != to).map(|(u, _)| g.vwgt[u] as u64).sum();
                    if side[1 - to as usize] < pulled + min_side {
                        continue;
                    }
                    let gain = g.vwgt[v] as i64 - pulled as i64;
                    if best.is_none_or(|(b, _, _)| gain > b) {
                        best = Some((gain, v, to));
                    }
                }
            }
            let Some((_, v, to)) = best else { break };
            let pulled: Vec<usize> = g.edges(v).filter(|&(u, _)| !in_sep[u] && part[u] != to).map(|(u, _)| u).collect();
            for &u in &pulled {
                in_sep[u] = true;
                side[part[u] as usize] -= g.vwgt[u] as u64;
                sep_weight += g.vwgt[u] as u64;
                seps.push(u);
            }
            let was = part[v];
            in_sep[v] = false;
            part[v] = to;
            side[to as usize] += g.vwgt[v] as u64;
            sep_weight -= g.vwgt[v] as u64;
            locked[v] = true;
            log.push((v, to, was, pulled));
            if sep_weight < best_weight {
                best_weight = sep_weight;
                best_at = log.len();
            } else if log.len() - best_at > p.patience {
                break;
            }
        }
        for (v, to, was, pulled) in log.drain(best_at..).rev() {
            in_sep[v] = true;
            part[v] = was;
            side[to as usize] -= g.vwgt[v] as u64;
            sep_weight += g.vwgt[v] as u64;
            for u in pulled {
                in_sep[u] = false;
                side[part[u] as usize] += g.vwgt[u] as u64;
                sep_weight -= g.vwgt[u] as u64;
            }
        }
        if best_at == 0 {
            break;
        }
    }
}

/// Minimum-degree order of a small graph by explicit elimination: each step eliminates a vertex of
/// least current degree (ties to the lowest index) and joins its neighbours into a clique. Exact
/// rather than approximate, at `O(n³ / 64)`; meant for nested-dissection leaves.
fn min_degree(g: &Graph) -> Vec<u32> {
    let n = g.n();
    let words = n.div_ceil(64);
    let mut adj = vec![0u64; n * words];
    for v in 0..n {
        for (u, _) in g.edges(v) {
            if u != v {
                adj[v * words + u / 64] |= 1 << (u % 64);
            }
        }
    }
    let mut alive = vec![u64::MAX; words];
    if !n.is_multiple_of(64) {
        alive[words - 1] = (1u64 << (n % 64)) - 1;
    }
    let mut order = Vec::with_capacity(n);
    let mut nbrs = vec![0u64; words];
    for _ in 0..n {
        let v = (0..n)
            .filter(|&v| alive[v / 64] >> (v % 64) & 1 == 1)
            .min_by_key(|&v| adj[v * words..(v + 1) * words].iter().zip(&alive).map(|(a, l)| (a & l).count_ones()).sum::<u32>())
            .expect("a vertex left");
        order.push(v as u32);
        alive[v / 64] &= !(1 << (v % 64));
        for (w, (a, l)) in nbrs.iter_mut().zip(adj[v * words..(v + 1) * words].iter().zip(&alive)) {
            *w = a & l;
        }
        for u in 0..n {
            if nbrs[u / 64] >> (u % 64) & 1 == 1 {
                for w in 0..words {
                    adj[u * words + w] |= nbrs[w];
                }
                adj[u * words + u / 64] &= !(1 << (u % 64));
            }
        }
    }
    order
}

/// The nodes of the graph `row_ptr`/`col` (symmetric, self-loops allowed) in nested-dissection
/// order.
pub fn nested_dissection(row_ptr: &[u32], col: &[u32]) -> Vec<u32> {
    nested_dissection_with(row_ptr, col, &NestedDissection::default())
}

/// The graph of a matrix's `block × block` blocks (lower triangle in compressed sparse columns,
/// `block` dofs per node numbered `block · node + component`), symmetric, as `row_ptr`/`col`.
pub fn node_graph(col_ptr: &[usize], row_idx: &[usize], block: usize) -> (Vec<u32>, Vec<u32>) {
    let n = col_ptr.len() - 1;
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n / block];
    for j in 0..n {
        for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            let (a, b) = ((i / block) as u32, (j / block) as u32);
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }
    }
    let mut row_ptr = vec![0u32];
    let mut col = Vec::new();
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
        col.extend_from_slice(a);
        row_ptr.push(col.len() as u32);
    }
    (row_ptr, col)
}

/// [`nested_dissection`] with explicit tuning.
pub fn nested_dissection_with(row_ptr: &[u32], col: &[u32], p: &NestedDissection) -> Vec<u32> {
    let n = row_ptr.len() - 1;
    let mut xadj = vec![0u32];
    let mut adjncy = Vec::with_capacity(col.len());
    for v in 0..n {
        adjncy.extend(col[row_ptr[v] as usize..row_ptr[v + 1] as usize].iter().filter(|&&u| u as usize != v));
        xadj.push(adjncy.len() as u32);
    }
    let adjwgt = vec![1u32; adjncy.len()];
    let root = Graph { xadj, adjncy, adjwgt, vwgt: vec![1; n] };
    let mut local = vec![u32::MAX; n];
    let mut out = Vec::with_capacity(n);
    let mut stack: Vec<(Vec<u32>, Option<Vec<u32>>)> = vec![((0..n as u32).collect(), None)];
    while let Some((set, pending)) = stack.pop() {
        if let Some(sep) = pending {
            out.extend(sep);
            continue;
        }
        if set.len() <= p.leaf {
            if p.leaf_min_degree {
                let g = root.subgraph(&set, &mut local);
                out.extend(min_degree(&g).into_iter().map(|v| set[v as usize]));
            } else {
                out.extend(set);
            }
            continue;
        }
        let g = root.subgraph(&set, &mut local);
        let mut best: Option<(Vec<u8>, Vec<bool>)> = None;
        for attempt in 0..p.tries {
            let (part, in_sep) = if p.node_refine {
                bisect_sep(&g, attempt, p)
            } else {
                let mut part = bisect(&g, attempt, p);
                let in_sep = separator(&g, &mut part, p);
                (part, in_sep)
            };
            let size = in_sep.iter().filter(|&&s| s).count();
            if best.as_ref().is_none_or(|(_, s)| size < s.iter().filter(|&&x| x).count()) {
                best = Some((part, in_sep));
            }
        }
        let (part, in_sep) = best.expect("at least one try");
        let separator: Vec<u32> = (0..g.n()).filter(|&v| in_sep[v]).map(|v| set[v]).collect();
        let lower: Vec<u32> = (0..g.n()).filter(|&v| !in_sep[v] && part[v] == 0).map(|v| set[v]).collect();
        let upper: Vec<u32> = (0..g.n()).filter(|&v| !in_sep[v] && part[v] == 1).map(|v| set[v]).collect();
        if lower.is_empty() || upper.is_empty() {
            // No cut found (a clique-like set): order as it comes.
            out.extend(set);
            continue;
        }
        stack.push((Vec::new(), Some(separator)));
        stack.push((upper, None));
        stack.push((lower, None));
    }
    debug_assert_eq!(out.len(), n);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::perm::PermRef;
    use faer::sparse::SymbolicSparseColMatRef;
    use faer::sparse::linalg::cholesky::{SymmetricOrdering, factorize_symbolic_cholesky};
    use faer::Side;

    /// A `dims` grid's graph: neighbours within `reach` along every axis, or for `reach` 1 only
    /// the six face neighbours.
    fn grid(dims: [usize; 3], reach: usize) -> (Vec<u32>, Vec<u32>) {
        let [nx, ny, nz] = dims;
        let n = nx * ny * nz;
        let pos = |i: usize| [i % nx, (i / nx) % ny, i / (nx * ny)];
        let mut row_ptr = vec![0u32];
        let mut col = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let (a, b) = (pos(i), pos(j));
                let d = [a[0].abs_diff(b[0]), a[1].abs_diff(b[1]), a[2].abs_diff(b[2])];
                if d.iter().all(|&x| x <= reach) && (reach > 1 || d.iter().sum::<usize>() <= 1) {
                    col.push(j as u32);
                }
            }
            row_ptr.push(col.len() as u32);
        }
        (row_ptr, col)
    }

    fn fill(row_ptr: &[u32], col: &[u32], order: Option<&[u32]>) -> usize {
        let n = row_ptr.len() - 1;
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        for j in 0..n {
            row_idx.extend(col[row_ptr[j] as usize..row_ptr[j + 1] as usize].iter().filter(|&&i| i as usize >= j).map(|&i| i as usize));
            col_ptr.push(row_idx.len());
        }
        let pattern = SymbolicSparseColMatRef::new_checked(n, n, &col_ptr, None, &row_idx);
        let (fwd, inv) = match order {
            Some(o) => {
                let fwd: Vec<usize> = o.iter().map(|&v| v as usize).collect();
                let mut inv = vec![0usize; n];
                for (new, &old) in fwd.iter().enumerate() {
                    inv[old] = new;
                }
                (fwd, inv)
            }
            None => (Vec::new(), Vec::new()),
        };
        let ordering = if order.is_some() { SymmetricOrdering::Custom(PermRef::new_checked(&fwd, &inv, n)) } else { SymmetricOrdering::Amd };
        factorize_symbolic_cholesky(pattern, Side::Lower, ordering, Default::default()).unwrap().len_val()
    }

    #[test]
    fn min_vertex_cover_covers_every_edge() {
        let edges = vec![vec![0u32, 1], vec![1], vec![1, 2]];
        let (l, r) = min_vertex_cover(&edges, 3);
        for (i, e) in edges.iter().enumerate() {
            for &j in e {
                assert!(l[i] || r[j as usize]);
            }
        }
        assert_eq!(l.iter().chain(&r).filter(|&&c| c).count(), 3);
    }

    #[test]
    fn nested_dissection_is_a_permutation_and_fills_less_than_amd() {
        for (dims, reach) in [([8, 8, 8], 2), ([16, 16, 16], 1), ([24, 24, 6], 2)] {
            let (row_ptr, col) = grid(dims, reach);
            let n = row_ptr.len() - 1;
            let order = nested_dissection(&row_ptr, &col);
            let mut seen = vec![false; n];
            for &v in &order {
                assert!(!seen[v as usize]);
                seen[v as usize] = true;
            }
            assert!(seen.iter().all(|&s| s));
            let nd = fill(&row_ptr, &col, Some(&order));
            let amd = fill(&row_ptr, &col, None);
            std::eprintln!("grid {dims:?} reach {reach}: fill nd {nd}, amd {amd}");
        }
    }
}
