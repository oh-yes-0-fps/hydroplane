//! Matrices held as `B × B` blocks per node pair (a 3D solid's 3 × 3 blocks, a shell's 6 × 6),
//! as the scalar lower triangle the solvers take.

/// The scalar lower-triangle pattern (compressed sparse columns) of a symmetric block-sparse
/// matrix, and where each of its entries comes from in the blocks. Made once per pattern;
/// [`Self::values`] then turns each new set of blocks into the solvers' values.
#[derive(Clone, Debug)]
pub struct BlockPattern<const B: usize> {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    /// For each scalar entry, its block and the entry within it (`row * B + col`).
    source: Vec<(u32, u8)>,
}

impl<const B: usize> BlockPattern<B> {
    /// From the block pattern by rows over nodes (`col[row_ptr[i] .. row_ptr[i + 1]]` the
    /// columns of node row `i`; block `p` is the one at `col[p]`): either triangle or both, and
    /// every diagonal block present. A pair stored both ways is taken from the block below the
    /// diagonal.
    pub fn new(row_ptr: &[usize], col: &[u32]) -> Self {
        let nodes = row_ptr.len() - 1;
        let mut below: Vec<Vec<(u32, usize, bool)>> = vec![Vec::new(); nodes];
        for i in 0..nodes {
            for (p, &j) in col.iter().enumerate().take(row_ptr[i + 1]).skip(row_ptr[i]) {
                let j = j as usize;
                if i >= j {
                    below[j].push((i as u32, p, false));
                } else {
                    below[i].push((j as u32, p, true));
                }
            }
        }
        for column in &mut below {
            column.sort_by_key(|&(r, _, transposed)| (r, transposed));
            column.dedup_by_key(|e| e.0);
        }
        let mut col_ptr = vec![0];
        let mut row_idx = Vec::new();
        let mut source = Vec::new();
        for (j, column) in below.iter().enumerate() {
            for c in 0..B {
                for &(i, p, transposed) in column {
                    let first = if i as usize == j { c } else { 0 };
                    for r in first..B {
                        row_idx.push(i as usize * B + r);
                        let (a, b) = if transposed { (c, r) } else { (r, c) };
                        source.push((p as u32, (a * B + b) as u8));
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
        Self { col_ptr, row_idx, source }
    }

    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    pub fn row_idx(&self) -> &[usize] {
        &self.row_idx
    }

    /// The scalar values for these blocks, `block(p, a, b)` being entry `(a, b)` of block `p`.
    pub fn values(&self, block: impl Fn(usize, usize, usize) -> f64 + Sync, out: &mut [f64]) {
        use rayon::prelude::*;
        assert_eq!(out.len(), self.source.len());
        out.par_iter_mut().zip(&self.source).with_min_len(4096).for_each(|(v, &(p, ab))| {
            let ab = ab as usize;
            *v = block(p as usize, ab / B, ab % B);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::BlockPattern;

    /// Two nodes with 2-dof blocks, stored by full rows.
    #[test]
    fn reads_the_lower_triangle_from_either_block_of_a_pair() {
        let row_ptr = [0, 2, 4];
        let col = [0u32, 1, 0, 1];
        let blocks = [[[4.0, 1.0], [1.0, 5.0]], [[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.3], [0.2, 0.4]], [[6.0, 0.5], [0.5, 7.0]]];
        let pattern = BlockPattern::<2>::new(&row_ptr, &col);
        let mut values = vec![0.0; pattern.row_idx().len()];
        pattern.values(|p, a, b| blocks[p][a][b], &mut values);
        let dense = |i: usize, j: usize| blocks[(i / 2) * 2 + j / 2][i % 2][j % 2];
        for j in 0..4 {
            let column = pattern.col_ptr()[j]..pattern.col_ptr()[j + 1];
            for (&i, &v) in pattern.row_idx()[column.clone()].iter().zip(&values[column]) {
                assert!(i >= j);
                assert_eq!(v, dense(i, j), "({i}, {j})");
            }
        }
        assert_eq!(pattern.row_idx().len(), 10);
    }

    #[test]
    fn takes_an_upper_triangle_alone() {
        let row_ptr = [0, 2, 3];
        let col = [0u32, 1, 1];
        let blocks = [[[4.0, 1.0], [1.0, 5.0]], [[0.1, 0.2], [0.3, 0.4]], [[6.0, 0.5], [0.5, 7.0]]];
        let pattern = BlockPattern::<2>::new(&row_ptr, &col);
        let mut values = vec![0.0; pattern.row_idx().len()];
        pattern.values(|p, a, b| blocks[p][a][b], &mut values);
        let at = |i: usize, j: usize| {
            let p = (pattern.col_ptr()[j]..pattern.col_ptr()[j + 1]).find(|&p| pattern.row_idx()[p] == i).unwrap();
            values[p]
        };
        assert_eq!(at(2, 0), 0.1);
        assert_eq!(at(3, 0), 0.2);
        assert_eq!(at(2, 1), 0.3);
        assert_eq!(at(3, 1), 0.4);
    }
}
