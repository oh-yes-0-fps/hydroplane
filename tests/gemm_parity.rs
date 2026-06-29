use hydroplane::{Layout, MatrixBackend, MatrixKernel, Scalar, Simd, dispatch_matrix, run_matrix_scalar};

struct Gemm<'a, T: Scalar, const M: usize, const N: usize, const K: usize> {
    a: &'a [T],
    b: &'a [T],
    out: &'a mut [T::Compute],
}

impl<T: Scalar, const M: usize, const N: usize, const K: usize> MatrixKernel<T>
    for Gemm<'_, T, M, N, K>
{
    type Output = ();
    fn run<S: MatrixBackend<T>>(self, ctx: Simd<T, S>) {
        let tl = ctx.tiles();
        let a = tl.load_a::<M, K>(self.a, K, Layout::RowMajor);
        let b = tl.load_b::<K, N>(self.b, N, Layout::RowMajor);
        let acc = tl.mma::<M, N, K>(a, b, tl.zero_acc::<M, N>());
        acc.store(self.out, N, Layout::RowMajor);
    }
}

/// Naive reference `D = A·B`, accumulating in the compute precision exactly as the backend does.
fn reference<T: Scalar>(a: &[T], b: &[T], m: usize, n: usize, k: usize) -> Vec<T::Compute> {
    let mut out = vec![<T::Compute as Scalar>::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = <T::Compute as Scalar>::ZERO;
            for kk in 0..k {
                s = a[i * k + kk].widen().fma(b[kk * n + j].widen(), s);
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= tol + tol * w.abs(), "got {g}, want {w}");
    }
}

#[test]
fn gemm_f32() {
    const M: usize = 3;
    const N: usize = 4;
    const K: usize = 5;
    let a: Vec<f32> = (0..M * K).map(|x| (x as f32) * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..K * N).map(|x| (x as f32) * -0.25 + 1.0).collect();
    let mut out = vec![0.0f32; M * N];
    run_matrix_scalar(Gemm::<f32, M, N, K> { a: &a, b: &b, out: &mut out });

    let want = reference::<f32>(&a, &b, M, N, K);
    let got64: Vec<f64> = out.iter().map(|&x| x as f64).collect();
    let want64: Vec<f64> = want.iter().map(|&x| x as f64).collect();
    assert_close(&got64, &want64, 1e-4);

    // Same kernel through the runtime-dispatched backend.
    let mut out_d = vec![0.0f32; M * N];
    dispatch_matrix(Gemm::<f32, M, N, K> { a: &a, b: &b, out: &mut out_d });
    assert_eq!(out, out_d);
}

#[test]
fn gemm_f64() {
    const M: usize = 4;
    const N: usize = 2;
    const K: usize = 6;
    let a: Vec<f64> = (0..M * K).map(|x| (x as f64) * 0.3 - 2.0).collect();
    let b: Vec<f64> = (0..K * N).map(|x| (x as f64) * 0.7 - 1.0).collect();
    let mut out = vec![0.0f64; M * N];
    run_matrix_scalar(Gemm::<f64, M, N, K> { a: &a, b: &b, out: &mut out });

    let want = reference::<f64>(&a, &b, M, N, K);
    assert_close(&out, &want, 1e-12);
}

#[test]
fn gemm_bf16() {
    use half::bf16;
    const M: usize = 4;
    const N: usize = 10; // > NEON lanes (4): exercises the lane-blocked body and the scalar tail
    const K: usize = 4;
    let a: Vec<bf16> = (0..M * K).map(|x| bf16::from_f32((x as f32) * 0.1 - 0.5)).collect();
    let b: Vec<bf16> = (0..K * N).map(|x| bf16::from_f32((x as f32) * 0.2 - 0.3)).collect();
    let want = reference::<bf16>(&a, &b, M, N, K);
    let want64: Vec<f64> = want.iter().map(|&x| x as f64).collect();

    let mut out_s = vec![0.0f32; M * N];
    run_matrix_scalar(Gemm::<bf16, M, N, K> { a: &a, b: &b, out: &mut out_s });
    assert_close(&out_s.iter().map(|&x| x as f64).collect::<Vec<_>>(), &want64, 1e-2);

    // Dispatched path = NEON bf16 widen-SIMD matmul on this aarch64 host.
    let mut out_d = vec![0.0f32; M * N];
    dispatch_matrix(Gemm::<bf16, M, N, K> { a: &a, b: &b, out: &mut out_d });
    assert_close(&out_d.iter().map(|&x| x as f64).collect::<Vec<_>>(), &want64, 1e-2);
}

#[test]
fn gemm_f16() {
    use half::f16;
    const M: usize = 4;
    const N: usize = 4;
    const K: usize = 4;
    let a: Vec<f16> = (0..M * K).map(|x| f16::from_f32((x as f32) * 0.1 - 0.5)).collect();
    let b: Vec<f16> = (0..K * N).map(|x| f16::from_f32((x as f32) * 0.2 - 0.3)).collect();
    let mut out = vec![0.0f32; M * N]; // accumulator is f32 (f16::Compute)
    run_matrix_scalar(Gemm::<f16, M, N, K> { a: &a, b: &b, out: &mut out });

    let want = reference::<f16>(&a, &b, M, N, K);
    let got64: Vec<f64> = out.iter().map(|&x| x as f64).collect();
    let want64: Vec<f64> = want.iter().map(|&x| x as f64).collect();
    assert_close(&got64, &want64, 1e-3);
}

// Large tiles (≥ ACCEL_MIN_DIM) take the Accelerate (AMX/SME) path on Apple by default, and the
// register-blocked `simd_gemm` under `--cfg no_apple_accelerate`. Either way must match the
// reference (tolerance: Accelerate's accumulation order/fusion differs).
#[test]
fn gemm_f32_large() {
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;
    let a: Vec<f32> = (0..M * K).map(|x| ((x % 17) as f32) * 0.1 - 0.8).collect();
    let b: Vec<f32> = (0..K * N).map(|x| ((x % 13) as f32) * 0.07 - 0.4).collect();
    let want: Vec<f64> = reference::<f32>(&a, &b, M, N, K).iter().map(|&x| x as f64).collect();
    let mut out = vec![0.0f32; M * N];
    dispatch_matrix(Gemm::<f32, M, N, K> { a: &a, b: &b, out: &mut out });
    let got: Vec<f64> = out.iter().map(|&x| x as f64).collect();
    assert_close(&got, &want, 1e-3);
}

// Large 16-bit tiles (multiple of the SME grid width) take the native SME widening grid (BFMOPA /
// FP16-widening FMOPA) under `no_apple_accelerate`, and Accelerate (via f32 widen) by default. Both
// accumulate in f32, so both must match the f32-accumulate reference.
#[test]
fn gemm_bf16_large() {
    use half::bf16;
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;
    let a: Vec<bf16> = (0..M * K).map(|x| bf16::from_f32(((x % 17) as f32) * 0.05 - 0.4)).collect();
    let b: Vec<bf16> = (0..K * N).map(|x| bf16::from_f32(((x % 13) as f32) * 0.04 - 0.25)).collect();
    let want: Vec<f64> = reference::<bf16>(&a, &b, M, N, K).iter().map(|&x| x as f64).collect();
    let mut out = vec![0.0f32; M * N];
    dispatch_matrix(Gemm::<bf16, M, N, K> { a: &a, b: &b, out: &mut out });
    let got: Vec<f64> = out.iter().map(|&x| x as f64).collect();
    assert_close(&got, &want, 2e-2);
}

#[test]
fn gemm_f16_large() {
    use half::f16;
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;
    let a: Vec<f16> = (0..M * K).map(|x| f16::from_f32(((x % 17) as f32) * 0.05 - 0.4)).collect();
    let b: Vec<f16> = (0..K * N).map(|x| f16::from_f32(((x % 13) as f32) * 0.04 - 0.25)).collect();
    let want: Vec<f64> = reference::<f16>(&a, &b, M, N, K).iter().map(|&x| x as f64).collect();
    let mut out = vec![0.0f32; M * N];
    dispatch_matrix(Gemm::<f16, M, N, K> { a: &a, b: &b, out: &mut out });
    let got: Vec<f64> = out.iter().map(|&x| x as f64).collect();
    assert_close(&got, &want, 1e-2);
}

#[test]
fn gemm_f64_large() {
    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;
    let a: Vec<f64> = (0..M * K).map(|x| ((x % 17) as f64) * 0.1 - 0.8).collect();
    let b: Vec<f64> = (0..K * N).map(|x| ((x % 13) as f64) * 0.07 - 0.4).collect();
    let want = reference::<f64>(&a, &b, M, N, K);
    let mut out = vec![0.0f64; M * N];
    dispatch_matrix(Gemm::<f64, M, N, K> { a: &a, b: &b, out: &mut out });
    assert_close(&out, &want, 1e-10);
}
