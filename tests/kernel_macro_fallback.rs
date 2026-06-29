//! The `macro_rules!` `kernel!` fallback (compiled only with `--no-default-features`, i.e. the
//! `macros` feature off): the generated kernels must behave exactly like the hand-written
//! struct+impl form, for both the element-wise and matrix flavours, including lifetimes and const
//! generics. The proc-macro `#[kernel]` form is covered in `kernel_macro.rs`.
#![cfg(not(feature = "macros"))]

use hydroplane::{Scalar, kernel};

kernel! {
    /// Sum of `xs` scaled by `k`, reduced across lanes — exercises load/operators/reduce.
    pub fn scaled_sum['a, T: Scalar](ctx, xs: &'a [T], k: T) -> f64 {
        let mut acc = ctx.splat(T::ZERO);
        for (off, cnt) in ctx.chunks(xs.len()) {
            let v = ctx.load_partial(&xs[off..off + cnt], T::ZERO);
            acc = acc + v * k;
        }
        acc.reduce_sum().to_f64()
    }
}

kernel! {
    matrix fn gemm['a, T: Scalar, const M: usize, const N: usize, const K: usize](
        ctx, a: &'a [T], b: &'a [T], out: &'a mut [T::Compute]
    ) -> () {
        let tl = ctx.tiles();
        let at = tl.load_a_rm::<M, K>(a);
        let bt = tl.load_b_rm::<K, N>(b);
        tl.mma::<M, N, K>(at, bt, tl.zero_acc::<M, N>()).store_rm(out);
    }
}

#[test]
fn element_wise_macro_kernel() {
    let xs: Vec<f32> = (0..50).map(|i| i as f32 * 0.5 - 3.0).collect();
    let got = scaled_sum(&xs, 2.0);
    let want: f64 = xs.iter().map(|&x| (x * 2.0) as f64).sum();
    assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "got {got}, want {want}");
}

#[test]
fn matrix_macro_kernel() {
    const M: usize = 3;
    const N: usize = 4;
    const K: usize = 5;
    let a: Vec<f32> = (0..M * K).map(|x| x as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..K * N).map(|x| x as f32 * -0.25 + 1.0).collect();
    let mut out = vec![0.0f32; M * N];
    gemm::<f32, M, N, K>(&a, &b, &mut out);

    for i in 0..M {
        for j in 0..N {
            let mut want = 0.0f32;
            for kk in 0..K {
                want += a[i * K + kk] * b[kk * N + j];
            }
            assert!((out[i * N + j] - want).abs() <= 1e-4, "[{i}][{j}]");
        }
    }
}
