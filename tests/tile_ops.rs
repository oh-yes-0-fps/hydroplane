use hydroplane::{Gang, MatrixBackend, MatrixKernel, dispatch_matrix, run_matrix_scalar};

/// `D = A·Bᵀ` with B loaded straight from its transposed (`N×K` row-major) storage.
struct GemmBt<'a, const M: usize, const N: usize, const K: usize> {
    a: &'a [f32],
    bt: &'a [f32],
    out: &'a mut [f32],
}

impl<const M: usize, const N: usize, const K: usize> MatrixKernel<f32> for GemmBt<'_, M, N, K> {
    type Output = ();
    fn run<S: MatrixBackend<f32>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles();
        let a = tl.load_a_rm::<M, K>(self.a);
        let b = tl.load_b_t::<K, N>(self.bt);
        tl.mma::<M, N, K>(a, b, tl.zero_acc()).store_rm(self.out);
    }
}

/// `D = Aᵀ·B` with A loaded from its transposed (`K×M` row-major) storage.
struct GemmAt<'a, const M: usize, const N: usize, const K: usize> {
    at: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
}

impl<const M: usize, const N: usize, const K: usize> MatrixKernel<f32> for GemmAt<'_, M, N, K> {
    type Output = ();
    fn run<S: MatrixBackend<f32>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles();
        let a = tl.load_a_t::<M, K>(self.at);
        let b = tl.load_b_rm::<K, N>(self.b);
        tl.mma::<M, N, K>(a, b, tl.zero_acc()).store_rm(self.out);
    }
}

fn reference_bt(a: &[f32], bt: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                s += (a[i * k + kk] as f64) * (bt[j * k + kk] as f64);
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn reference_at(at: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                s += (at[kk * m + i] as f64) * (b[kk * n + j] as f64);
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        assert!((g - w).abs() <= tol + tol * w.abs(), "got {g}, want {w}");
    }
}

#[test]
fn gemm_bt_small() {
    const M: usize = 3;
    const N: usize = 4;
    const K: usize = 5;
    let a: Vec<f32> = (0..M * K).map(|x| (x as f32) * 0.5 - 3.0).collect();
    let bt: Vec<f32> = (0..N * K).map(|x| (x as f32) * -0.25 + 1.0).collect();
    let want = reference_bt(&a, &bt, M, N, K);

    let mut out = vec![0.0f32; M * N];
    run_matrix_scalar(GemmBt::<M, N, K> { a: &a, bt: &bt, out: &mut out });
    assert_close(&out, &want, 1e-5);

    let mut out_d = vec![0.0f32; M * N];
    dispatch_matrix(GemmBt::<M, N, K> { a: &a, bt: &bt, out: &mut out_d });
    assert_close(&out_d, &want, 1e-5);
}

#[test]
fn gemm_at_small() {
    const M: usize = 4;
    const N: usize = 3;
    const K: usize = 6;
    let at: Vec<f32> = (0..K * M).map(|x| (x as f32) * 0.3 - 2.0).collect();
    let b: Vec<f32> = (0..K * N).map(|x| (x as f32) * 0.7 - 1.0).collect();
    let want = reference_at(&at, &b, M, N, K);

    let mut out = vec![0.0f32; M * N];
    run_matrix_scalar(GemmAt::<M, N, K> { at: &at, b: &b, out: &mut out });
    assert_close(&out, &want, 1e-5);

    let mut out_d = vec![0.0f32; M * N];
    dispatch_matrix(GemmAt::<M, N, K> { at: &at, b: &b, out: &mut out_d });
    assert_close(&out_d, &want, 1e-5);
}

// Large transposed tiles take the BLAS `Trans` fast path on Apple (no gather); elsewhere the
// gather + register-blocked GEMM. Both must match the f64 reference.
#[test]
fn gemm_bt_large() {
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;
    let a: Vec<f32> = (0..M * K).map(|x| ((x % 17) as f32) * 0.1 - 0.8).collect();
    let bt: Vec<f32> = (0..N * K).map(|x| ((x % 13) as f32) * 0.07 - 0.4).collect();
    let want = reference_bt(&a, &bt, M, N, K);
    let mut out = vec![0.0f32; M * N];
    dispatch_matrix(GemmBt::<M, N, K> { a: &a, bt: &bt, out: &mut out });
    assert_close(&out, &want, 1e-3);
}

#[test]
fn gemm_at_large() {
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;
    let at: Vec<f32> = (0..K * M).map(|x| ((x % 19) as f32) * 0.09 - 0.7).collect();
    let b: Vec<f32> = (0..K * N).map(|x| ((x % 11) as f32) * 0.05 - 0.3).collect();
    let want = reference_at(&at, &b, M, N, K);
    let mut out = vec![0.0f32; M * N];
    dispatch_matrix(GemmAt::<M, N, K> { at: &at, b: &b, out: &mut out });
    assert_close(&out, &want, 1e-3);
}

/// `J·Jᵀ` from one buffer: both operands view the same row-major `J`.
#[test]
fn gram_from_single_buffer() {
    const M: usize = 4;
    const K: usize = 7;
    let j: Vec<f32> = (0..M * K).map(|x| (x as f32) * 0.11 - 1.5).collect();
    let want = reference_bt(&j, &j, M, M, K);

    struct Gram<'a, const M: usize, const K: usize> {
        j: &'a [f32],
        out: &'a mut [f32],
    }
    impl<const M: usize, const K: usize> MatrixKernel<f32> for Gram<'_, M, K> {
        type Output = ();
        fn run<S: MatrixBackend<f32>>(self, ctx: Gang<S>) {
            let tl = ctx.tiles();
            let a = tl.load_a_rm::<M, K>(self.j);
            let b = tl.load_b_t::<K, M>(self.j);
            tl.mma::<M, M, K>(a, b, tl.zero_acc()).store_rm(self.out);
        }
    }

    let mut out = vec![0.0f32; M * M];
    dispatch_matrix(Gram::<M, K> { j: &j, out: &mut out });
    assert_close(&out, &want, 1e-4);
    // Gram matrices are symmetric.
    for i in 0..M {
        for jj in 0..M {
            assert_eq!(out[i * M + jj], out[jj * M + i]);
        }
    }
}

struct AccOps;

impl MatrixKernel<f32> for AccOps {
    type Output = ();
    fn run<S: MatrixBackend<f32>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f32>();
        const R: usize = 2;
        const C: usize = 3;
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = tl.load_acc_rm::<R, C>(&data);

        assert_eq!(t.row_sums(), [6.0, 15.0]);
        assert_eq!(t.col_sums(), [5.0, 7.0, 9.0]);
        assert_eq!(t.reduce_rows(f32::NEG_INFINITY, f32::max), [3.0, 6.0]);
        assert_eq!(t.reduce_cols(f32::INFINITY, f32::min), [1.0, 2.0, 3.0]);
        assert_eq!(t.fold_rows([10.0, 20.0], |a, v| a + v), [16.0, 35.0]);
        assert_eq!(t.fold_cols([1.0, 1.0, 1.0], |a, v| a * v), [4.0, 10.0, 18.0]);

        let mut got = [0.0f32; R * C];
        t.zip(t, |a, b| a * b).store_rm(&mut got);
        assert_eq!(got, [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

        t.zip_rows([10.0, 100.0], |x, s| x + s).store_rm(&mut got);
        assert_eq!(got, [11.0, 12.0, 13.0, 104.0, 105.0, 106.0]);

        t.zip_cols([1.0, 2.0, 3.0], |x, s| x * s).store_rm(&mut got);
        assert_eq!(got, [1.0, 4.0, 9.0, 4.0, 10.0, 18.0]);

        let doubled = t + t;
        doubled.store_rm(&mut got);
        assert_eq!(got, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        (doubled - t).store_rm(&mut got);
        assert_eq!(got, data);
    }
}

#[test]
fn accumulator_ops() {
    run_matrix_scalar(AccOps);
    dispatch_matrix(AccOps);
}

/// Row softmax composed from the reduction/broadcast surface; each row must sum to 1.
struct Softmax;

impl MatrixKernel<f32> for Softmax {
    type Output = ();
    fn run<S: MatrixBackend<f32>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f32>();
        const R: usize = 3;
        const C: usize = 4;
        let data: [f32; R * C] = core::array::from_fn(|x| (x as f32) * 0.7 - 4.0);
        let t = tl.load_acc_rm::<R, C>(&data);

        let m = t.reduce_rows(f32::NEG_INFINITY, f32::max);
        let e = t.zip_rows(m, |x, mx| (x - mx).exp());
        let sm = e.zip_rows(e.row_sums(), |x, d| x / d);

        for s in sm.row_sums() {
            assert!((s - 1.0).abs() < 1e-6, "row sum {s}");
        }
    }
}

#[test]
fn softmax_rows() {
    run_matrix_scalar(Softmax);
    dispatch_matrix(Softmax);
}
