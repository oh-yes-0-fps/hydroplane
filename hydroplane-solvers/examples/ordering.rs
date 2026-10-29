//! Compares fill-reducing orderings of a dumped SPD matrix by what they cost the factorization:
//! factor entries, flops, supernodes, elimination-tree levels, the critical path in 32-column panel
//! steps (the sum over levels of each level's widest supernode), and the largest supernode.
//!
//! `cargo run --release --example ordering -- MATRIX.bin [BLOCK [PERM.bin …]]`
//!
//! The matrix file is as for the `cholesky` example. Each `PERM.bin` is a node ordering
//! (`u32` per node, `perm[new] = old`) from another tool, expanded to `BLOCK` dofs per node.
//! `ND="leaf=16;min_share=0.45,tries=6"` adds nested-dissection variants, `;`-separated, each a
//! `,`-separated list of [`order::NestedDissection`] fields.

use std::time::Instant;

use hydroplane_solvers::cholesky::{Analysis, Ordering};
use hydroplane_solvers::order::{self, NestedDissection};

fn load(path: &str) -> (Vec<usize>, Vec<usize>) {
    let bytes = std::fs::read(path).expect("reading the matrix");
    let word = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().unwrap()) as usize;
    let (n, nnz) = (word(0), word(1));
    let col_ptr: Vec<usize> = (0..=n).map(|i| word(2 + i)).collect();
    let at = 8 * (n + 3);
    let row_idx = bytes[at..at + 4 * nnz].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
    (col_ptr, row_idx)
}

fn report(name: &str, analysis: &Analysis, seconds: f64) {
    let sn = analysis.supernodal();
    let ns = sn.n_supernodes();
    let (begin, end) = (sn.supernode_begin(), sn.supernode_end());
    let mut col_super = vec![0usize; analysis.dim()];
    for s in 0..ns {
        col_super[begin[s]..end[s]].fill(s);
    }
    let mut level = vec![0usize; ns];
    // Longest chain of panel steps from a leaf through supernode s (its own included): what a
    // schedule that starts each supernode as soon as its children finish would take.
    let mut chain = vec![0usize; ns];
    let mut flops = 0.0;
    let mut largest = 0;
    for s in 0..ns {
        let (k, m) = (end[s] - begin[s], sn.supernode(s).pattern().len());
        largest = largest.max(k);
        let (kf, mf) = (k as f64, m as f64);
        flops += kf * kf * kf / 3.0 + kf * kf * mf + kf * mf * mf;
        chain[s] += (end[s] - begin[s]).div_ceil(32);
        if let Some(&first) = sn.supernode(s).pattern().first() {
            let p = col_super[first];
            level[p] = level[p].max(level[s] + 1);
            chain[p] = chain[p].max(chain[s]);
        }
    }
    let levels = level.iter().max().map_or(0, |l| l + 1);
    // As late as possible instead: one level below the parent (the root at the top level).
    let mut late = vec![0usize; ns];
    for s in (0..ns).rev() {
        late[s] = match sn.supernode(s).pattern().first() {
            Some(&first) => late[col_super[first]] - 1,
            None => levels - 1,
        };
    }
    let mut widest_late = vec![0usize; levels];
    for s in 0..ns {
        widest_late[late[s]] = widest_late[late[s]].max((end[s] - begin[s]).div_ceil(32));
    }
    let steps_late: usize = widest_late.iter().sum();
    if std::env::var_os("DETAIL").is_some() {
        for lv in 0..levels {
            let (mut count, mut cols, mut widest_k, mut gf) = (0, 0, 0, 0.0);
            for s in (0..ns).filter(|&s| level[s] == lv) {
                let (k, m) = ((end[s] - begin[s]) as f64, sn.supernode(s).pattern().len() as f64);
                count += 1;
                cols += end[s] - begin[s];
                widest_k = widest_k.max(end[s] - begin[s]);
                gf += (k * k * k / 3.0 + k * k * m + k * m * m) * 1e-9;
            }
            println!("    level {lv:2}: {count:5} supernodes, {cols:6} columns, widest {widest_k:5}, {gf:6.2} GFLOP");
        }
    }
    let mut widest = vec![0usize; levels];
    for s in 0..ns {
        widest[level[s]] = widest[level[s]].max((end[s] - begin[s]).div_ceil(32));
    }
    let steps = analysis.levels().1;
    // Fitted on the RTX 5090 (f64) over four orderings of the slab tangent: a 32-column panel
    // step on the critical path costs about as much as 0.15 GFLOP.
    let estimate = 0.073 * steps as f64 + 0.48 * flops * 1e-9;
    println!(
        "{name:<28} factor {:>6.2}M  {:>5.1} GFLOP  supernodes {:>5}  levels {:>2}  panel steps {:>4} (early {:>4}, late {:>4}, chain {:>4})  largest {:>5}  est {:>5.1} ms  ({:.2} s)",
        analysis.factor_len() as f64 * 1e-6,
        flops * 1e-9,
        ns,
        levels,
        steps,
        widest.iter().sum::<usize>(),
        steps_late,
        chain.iter().max().copied().unwrap_or(0),
        largest,
        estimate,
        seconds
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (col_ptr, row_idx) = load(&args[1]);
    let block: usize = args.get(2).map_or(3, |b| b.parse().unwrap());
    let n = col_ptr.len() - 1;
    println!("matrix: {n} rows, {} lower nonzeros", row_idx.len());
    let run = |name: &str, ordering: Ordering| {
        let t = Instant::now();
        let analysis = Analysis::with_ordering(&col_ptr, &row_idx, ordering).expect("analysis");
        report(name, &analysis, t.elapsed().as_secs_f64());
    };
    run("nested dissection", Ordering::NestedDissection { block });
    let (row_ptr, col) = order::node_graph(&col_ptr, &row_idx, block);
    for spec in std::env::var("ND").unwrap_or_default().split(';').filter(|s| !s.is_empty()) {
        let p: NestedDissection = spec.parse().expect("an ND spec");
        let t = Instant::now();
        let nodes = order::nested_dissection_with(&row_ptr, &col, &p);
        let perm: Vec<usize> = nodes.iter().flat_map(|&v| (0..block).map(move |c| block * v as usize + c)).collect();
        let analysis = Analysis::with_ordering(&col_ptr, &row_idx, Ordering::Custom(perm)).expect("analysis");
        report(&format!("ND {spec}"), &analysis, t.elapsed().as_secs_f64());
    }
    run("AMD (faer)", Ordering::Amd);
    if let Some(c) = std::env::var("CANDIDATES").ok().and_then(|c| c.parse::<usize>().ok()) {
        let t = Instant::now();
        let gpu_cost = |a: &Analysis| 0.073 * a.levels().1 as f64 + 0.48 * a.flops() * 1e-9;
        let analysis = Analysis::best_of(&col_ptr, &row_idx, block, c, gpu_cost).expect("analysis");
        report(&format!("ND best of {c}"), &analysis, t.elapsed().as_secs_f64());
    }
    for path in args.iter().skip(3) {
        let bytes = std::fs::read(path).expect("reading a permutation");
        let nodes: Vec<usize> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
        let perm: Vec<usize> = nodes.iter().flat_map(|&v| (0..block).map(move |c| block * v + c)).collect();
        run(path, Ordering::Custom(perm));
    }
}
