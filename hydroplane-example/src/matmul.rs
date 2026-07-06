//! `#[kernel(matrix)]` GEMM kernels: the tiled `ctx.tiles()` surface
//! (`load_a`/`load_b`/`mma`/`store`) with const-generic shapes.

use hydroplane::{Gang, Layout, kernel};

/// Plain `C = A·B` for an `M×K` times `K×N` product, all contiguous row-major.
#[kernel(matrix)]
pub fn gemm<'a, const M: usize, const N: usize, const K: usize>(
    ctx: Gang,
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
) {
    let tl = ctx.tiles();
    let a = tl.load_a_rm::<M, K>(a);
    let b = tl.load_b_rm::<K, N>(b);
    tl.mma::<M, N, K>(a, b, tl.zero_acc::<M, N>()).store_rm(out);
}

/// Linear layer: `Y = X·W + bias`. The bias becomes the accumulator (`load_acc_rm` instead of
/// `zero_acc`), so the bias-add is fused into `mma`.
#[kernel(matrix)]
pub fn linear<'a, const M: usize, const N: usize, const K: usize>(
    ctx: Gang,
    x: &'a [f32],
    w: &'a [f32],
    bias: &'a [f32],
    out: &'a mut [f32],
) {
    let tl = ctx.tiles();
    let x = tl.load_a_rm::<M, K>(x);
    let w = tl.load_b_rm::<K, N>(w);
    tl.mma::<M, N, K>(x, w, tl.load_acc_rm::<M, N>(bias)).store_rm(out);
}

/// Tiled GEMM with a K-accumulation loop. The product is summed over `k_total / KT` tiles of the
/// contraction dimension, threading one accumulator across the `mma` calls. Each `A` sub-tile is a
/// strided column slice (`row_stride = k_total`); each `B` sub-tile is a contiguous row block.
#[kernel(matrix)]
pub fn gemm_ktiled<'a, const M: usize, const N: usize, const KT: usize>(
    ctx: Gang,
    a: &'a [f32],
    b: &'a [f32],
    k_total: usize,
    out: &'a mut [f32],
) {
    let tl = ctx.tiles();
    let mut acc = tl.zero_acc::<M, N>();
    let mut k0 = 0;
    while k0 + KT <= k_total {
        let at = tl.load_a::<M, KT>(&a[k0..], k_total, Layout::RowMajor);
        let bt = tl.load_b::<KT, N>(&b[k0 * N..], N, Layout::RowMajor);
        acc = tl.mma::<M, N, KT>(at, bt, acc);
        k0 += KT;
    }
    acc.store_rm(out);
}

/// Matrix and vector surface on one context: compute `C = A·B`, store it, then reduce the result
/// with an ordinary `sum`. `MatrixBackend<f32>: Backend<f32>`, so the matrix context does vector
/// ops too.
#[kernel(matrix)]
pub fn gemm_sum<'a, const M: usize, const N: usize, const K: usize>(
    ctx: Gang,
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
) -> f32 {
    let tl = ctx.tiles();
    let a = tl.load_a_rm::<M, K>(a);
    let b = tl.load_b_rm::<K, N>(b);
    tl.mma::<M, N, K>(a, b, tl.zero_acc::<M, N>()).store_rm(out);
    ctx.sum(out, |acc, v| acc + v)
}

/// Scalar reference for `C = A·B` (`M×K · K×N`, row-major).
pub fn gemm_scalar<const M: usize, const N: usize, const K: usize>(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..M {
        for j in 0..N {
            let mut s = 0.0f32;
            for k in 0..K {
                s += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ramp;

    #[test]
    fn matmul_kernels_match_scalar() {
        const M: usize = 4;
        const N: usize = 6;
        const K: usize = 5;
        let a = ramp(M * K, 1.0, 1.0);
        let b = ramp(K * N, 7.0, 1.0);

        let mut want = vec![0.0f32; M * N];
        gemm_scalar::<M, N, K>(&a, &b, &mut want);

        let mut got = vec![0.0f32; M * N];
        gemm::<M, N, K>(&a, &b, &mut got);
        assert!(crate::max_rel_err(&got, &want) < 1e-3, "gemm");

        // K-tiled with KT == K, a single tile.
        let mut gk = vec![0.0f32; M * N];
        gemm_ktiled::<M, N, K>(&a, &b, K, &mut gk);
        assert!(crate::max_rel_err(&gk, &want) < 1e-3, "gemm_ktiled");

        // linear with a zero bias == plain gemm.
        let bias = vec![0.0f32; M * N];
        let mut gl = vec![0.0f32; M * N];
        linear::<M, N, K>(&a, &b, &bias, &mut gl);
        assert!(crate::max_rel_err(&gl, &want) < 1e-3, "linear(0 bias)");

        // gemm_sum returns Σ C and writes the same C.
        let mut gs = vec![0.0f32; M * N];
        let s = gemm_sum::<M, N, K>(&a, &b, &mut gs);
        let want_sum: f32 = want.iter().sum();
        assert!((s - want_sum).abs() <= 1e-3 * want_sum.abs().max(1.0), "gemm_sum");
        assert!(crate::max_rel_err(&gs, &want) < 1e-3, "gemm_sum output");
    }
}
