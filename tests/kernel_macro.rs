//! `#[kernel]` attribute: the generated kernels must behave exactly like the hand-written
//! struct+impl form, for both the element-wise and matrix flavours, including lifetimes, const
//! generics, ordinary `<…>` generics, multiple bounds, where-clauses, a renamed/overridden
//! scalar, and several type parameters.

use hydroplane::{FloatScalar, Gang, kernel};

#[kernel]
/// Sum of `xs` scaled by `k`, reduced across lanes — exercises load/operators/reduce.
pub fn scaled_sum<'a, T: FloatScalar>(ctx: Gang<T>, xs: &'a [T], k: T) -> f64 {
    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        let v = ctx.load_partial(&xs[off..off + cnt], T::ZERO);
        acc = acc + v * k;
    });
    acc.reduce_sum().into_f64()
}

// Multiple bounds on the scalar + an extra non-scalar type parameter used only in the return type:
// `T` is still picked as the scalar (it carries the `Scalar` bound), `R` rides along via PhantomData.
#[kernel]
pub fn scaled_sum_into<'a, T: FloatScalar + Copy, R: From<f64>>(ctx: Gang<T>, xs: &'a [T], k: T) -> R {
    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        acc = acc + ctx.load_partial(&xs[off..off + cnt], T::ZERO) * k;
    });
    R::from(acc.reduce_sum().into_f64())
}

// Scalar bound expressed in a where-clause (the fallback can't parse this at all).
#[kernel]
pub fn dot<'a, T>(ctx: Gang<T>, xs: &'a [T], ys: &'a [T]) -> f64
where
    T: FloatScalar,
{
    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        let x = ctx.load_partial(&xs[off..off + cnt], T::ZERO);
        let y = ctx.load_partial(&ys[off..off + cnt], T::ZERO);
        acc = acc + x * y;
    });
    acc.reduce_sum().into_f64()
}

// The scalar parameter need not be named `T` — it's found by its `Scalar` bound (the `macro_rules!`
// fallback hard-requires the name `T`).
#[kernel]
pub fn renamed_scalar<'a, Elem: FloatScalar>(ctx: Gang<Elem>, xs: &'a [Elem]) -> f64 {
    let mut acc = ctx.splat(Elem::ZERO);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        acc = acc + ctx.load_partial(&xs[off..off + cnt], Elem::ZERO);
    });
    acc.reduce_sum().into_f64()
}

// A plain vector reduction over a generic scalar (bare `#[kernel]`).
#[kernel]
pub fn vector_sum<'a, T: FloatScalar>(ctx: Gang<T>, xs: &'a [T]) -> f64 {
    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        acc = acc + ctx.load_partial(&xs[off..off + cnt], T::ZERO);
    });
    acc.reduce_sum().into_f64()
}

// A concrete-`f32` micro-kernel (scalar inferred from the context type) — like a joint-limit check.
// The full-register pass short-circuits with plain loads; only the remainder stages sentinels so
// inactive lanes never produce a false hit.
#[kernel]
pub fn any_gt<'a>(ctx: Gang<f32>, a: &'a [f32], b: &'a [f32]) -> bool {
    let n = ctx.lanes();
    for off in ctx.chunks_exact(a.len()) {
        if ctx.load(&a[off..off + n]).gt(ctx.load(&b[off..off + n])).any() {
            return true;
        }
    }
    if let Some((off, cnt)) = ctx.remainder(a.len()) {
        let x = ctx.load_partial(&a[off..off + cnt], f32::NEG_INFINITY);
        let y = ctx.load_partial(&b[off..off + cnt], f32::INFINITY);
        return x.gt(y).any();
    }
    false
}

#[kernel(matrix)]
pub fn gemm<'a, T: FloatScalar, const M: usize, const N: usize, const K: usize>(
    ctx: Gang<T>,
    a: &'a [T],
    b: &'a [T],
    out: &'a mut [T::Compute],
) {
    let tl = ctx.tiles();
    let at = tl.load_a_rm::<M, K>(a);
    let bt = tl.load_b_rm::<K, N>(b);
    tl.mma::<M, N, K>(at, bt, tl.zero_acc::<M, N>()).store_rm(out);
}

// A matrix context is also a `Backend`, so one `ctx` drives both the tile matmul (`out = A·B`) and a
// vector reduction (the lane-sum of `a`) in the same kernel.
#[kernel(matrix)]
pub fn gemm_and_sum_a<'a, T: FloatScalar, const M: usize, const N: usize, const K: usize>(
    ctx: Gang<T>,
    a: &'a [T],
    b: &'a [T],
    out: &'a mut [T::Compute],
) -> f64 {
    let tl = ctx.tiles();
    let at = tl.load_a_rm::<M, K>(a);
    let bt = tl.load_b_rm::<K, N>(b);
    tl.mma::<M, N, K>(at, bt, tl.zero_acc::<M, N>()).store_rm(out);

    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk(a.len(), |off, cnt| {
        acc = acc + ctx.load_partial(&a[off..off + cnt], T::ZERO);
    });
    acc.reduce_sum().into_f64()
}

#[test]
fn element_wise_macro_kernel() {
    let xs: Vec<f32> = (0..50).map(|i| i as f32 * 0.5 - 3.0).collect();
    let got = scaled_sum(&xs, 2.0);
    let want: f64 = xs.iter().map(|&x| (x * 2.0) as f64).sum();
    assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "got {got}, want {want}");
}

#[test]
fn extra_type_param_and_multiple_bounds() {
    let xs: Vec<f32> = (0..50).map(|i| i as f32 * 0.5 - 3.0).collect();
    let got: f64 = scaled_sum_into::<f32, f64>(&xs, 2.0);
    let want: f64 = xs.iter().map(|&x| (x * 2.0) as f64).sum();
    assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "got {got}, want {want}");
}

#[test]
fn where_clause_kernel() {
    let xs: Vec<f64> = (0..40).map(|i| i as f64 * 0.25 - 2.0).collect();
    let ys: Vec<f64> = (0..40).map(|i| i as f64 * -0.5 + 1.0).collect();
    let got = dot(&xs, &ys);
    let want: f64 = xs.iter().zip(&ys).map(|(&x, &y)| x * y).sum();
    assert!((got - want).abs() <= 1e-6 * (1.0 + want.abs()), "got {got}, want {want}");
}

#[test]
fn scalar_param_need_not_be_named_t() {
    let xs: Vec<f32> = (0..33).map(|i| i as f32 - 5.0).collect();
    let got = renamed_scalar(&xs);
    let want: f64 = xs.iter().map(|&x| x as f64).sum();
    assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "got {got}, want {want}");
}

// Kernel-calls-kernel: `scaled` is a normal kernel, but `scaled_then_sum` calls its `scaled_on`
// companion with its *own* dispatched context, so dispatch runs once (at `scaled_then_sum`'s entry)
// rather than again per inner call.
#[kernel]
pub fn scaled<'a, T: FloatScalar>(ctx: Gang<T>, xs: &'a [T], k: T) -> f64 {
    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        acc = acc + ctx.load_partial(&xs[off..off + cnt], T::ZERO) * k;
    });
    acc.reduce_sum().into_f64()
}

#[kernel]
pub fn scaled_then_sum<'a, T: FloatScalar>(ctx: Gang<T>, xs: &'a [T], ys: &'a [T], k: T) -> f64 {
    scaled_on(ctx, xs, k) + scaled_on(ctx, ys, k)
}

#[test]
fn explicit_vector_mode() {
    let xs: Vec<f32> = (0..27).map(|i| i as f32 - 4.0).collect();
    let got = vector_sum(&xs);
    let want: f64 = xs.iter().map(|&x| x as f64).sum();
    assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "got {got}, want {want}");
}

#[test]
fn kernel_calls_kernel_via_on() {
    let xs: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 - 2.0).collect();
    let ys: Vec<f32> = (0..20).map(|i| i as f32 * -0.25 + 1.0).collect();
    let got = scaled_then_sum(&xs, &ys, 2.0);
    let want = scaled(&xs, 2.0) + scaled(&ys, 2.0);
    assert!((got - want).abs() <= 1e-3 * (1.0 + want.abs()), "got {got}, want {want}");
}

#[test]
fn concrete_scalar_kernel_matches_oracle() {
    // Match the brute-force `any(a > b)` across odd lengths (so the partial tail is exercised) and
    // both outcomes; calling repeatedly also exercises the warm cached-backend path.
    for n in [1usize, 6, 7, 8, 9, 31, 64] {
        let a: Vec<f32> = (0..n).map(|i| (i as f32 % 5.0) - 2.0).collect();
        for shift in [-1.0f32, 0.0, 1.0] {
            let b: Vec<f32> = a.iter().map(|&x| x + shift).collect();
            let want = a.iter().zip(&b).any(|(&x, &y)| x > y);
            assert_eq!(any_gt(&a, &b), want, "n={n} shift={shift}");
        }
    }
}

// Splat-constant scalars arrive as separate fields, so kernels routinely exceed clippy's argument
// limit; the generated wrapper and `_on` fn carry their own opt-out (the author can't annotate them),
// so `cargo clippy` over this file stays clean.
#[kernel]
pub fn scale7<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    a0: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,
    a5: f32,
    a6: f32,
) -> f32 {
    let mut acc = ctx.splat(0.0);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        let v = ctx.load_partial(&xs[off..off + cnt], 0.0);
        acc = acc + v * a0 * a1 * a2 * a3 * a4 * a5 * a6;
    });
    acc.reduce_sum()
}

#[test]
fn many_arg_kernel_matches_oracle() {
    let xs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let got = scale7(&xs, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    let want: f32 = xs.iter().sum::<f32>() * 2.0;
    assert!((got - want).abs() < 1e-4, "got {got} want {want}");
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

#[test]
fn vector_and_matrix_modes_together() {
    const M: usize = 3;
    const N: usize = 4;
    const K: usize = 5;
    let a: Vec<f32> = (0..M * K).map(|x| x as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..K * N).map(|x| x as f32 * -0.25 + 1.0).collect();
    let mut out = vec![0.0f32; M * N];

    let sum_a = gemm_and_sum_a::<f32, M, N, K>(&a, &b, &mut out);

    let want_sum: f64 = a.iter().map(|&x| x as f64).sum();
    assert!((sum_a - want_sum).abs() <= 1e-3 * (1.0 + want_sum.abs()));
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
