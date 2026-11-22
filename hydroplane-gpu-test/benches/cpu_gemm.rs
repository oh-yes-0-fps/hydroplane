//! The GPU GEMM kernels run unchanged on the CPU through `hydroplane::dispatch_matrix`, against
//! `hydroplane::gemm`: one 64-row strip of the `GEMM³` f16 problem, single-threaded.

use half::f16;
use hydroplane::{Mat, MatMut, Trans};
use hydroplane_gpu_test_shader::{GEMM, Gemm, GemmPipelined};
use std::time::Instant;

const STRIP: usize = 64;

fn time(label: &str, mut f: impl FnMut()) {
    f();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let flops = 2.0 * (STRIP * GEMM * GEMM) as f64;
    println!("  {label:<44} {:>8.2} ms  {:>7.1} GFLOP/s", best * 1e3, flops / best / 1e9);
}

fn main() {
    let n = GEMM;
    let a: Vec<f16> = (0..n * n).map(|i| f16::from_f32(((i * 7919 % 17) as f32 - 8.0) * 0.125)).collect();
    let b: Vec<f16> = (0..n * n).map(|i| f16::from_f32(((i * 104729 % 13) as f32 - 6.0) * 0.125)).collect();
    let mut c = vec![0f32; n * n];
    println!("CPU, one thread — {STRIP}×{n}×{n} f16 GEMM strip");
    time("tile kernel: 64×64 pipelined (GPU source)", || {
        for col in 0..n / 64 {
            hydroplane::dispatch_matrix(GemmPipelined::<4, 4> { a: &a, b: &b, c: &mut c, tile_row: 0, tile_col: col });
        }
    });
    time("tile kernel: 64×64, k unrolled 2 (GPU source)", || {
        for col in 0..n / 64 {
            hydroplane::dispatch_matrix(Gemm::<4, 4, 2> { a: &a, b: &b, c: &mut c, tile_row: 0, tile_col: col });
        }
    });
    let mut c16 = vec![f16::ZERO; STRIP * n];
    time("hydroplane::gemm (f16)", || {
        hydroplane::gemm(Trans::N, Trans::N, f16::ONE, Mat::new(&a[..STRIP * n], STRIP, n), Mat::new(&b, n, n), f16::ZERO, MatMut::new(&mut c16, STRIP, n));
    });
    let a32: Vec<f32> = a[..STRIP * n].iter().map(|x| x.to_f32()).collect();
    let b32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
    let mut c32 = vec![0f32; STRIP * n];
    time("hydroplane::gemm (f32)", || {
        hydroplane::gemm(Trans::N, Trans::N, 1.0, Mat::new(&a32, STRIP, n), Mat::new(&b32, n, n), 0.0, MatMut::new(&mut c32, STRIP, n));
    });
    let worst = (0..STRIP * n).map(|i| (c[i] - c32[i]).abs() / c32[i].abs().max(1.0)).fold(0f32, f32::max);
    println!("  tile kernel vs gemm (f32): max rel diff {worst:.1e}");

    for size in [512usize, 1024, 2048] {
        let a: Vec<f32> = (0..size * size).map(|i| ((i * 7919 % 17) as f32 - 8.0) * 0.125).collect();
        let b: Vec<f32> = (0..size * size).map(|i| ((i * 104729 % 13) as f32 - 6.0) * 0.125).collect();
        let mut c = vec![0f32; size * size];
        let mut best = f64::MAX;
        for _ in 0..4 {
            let t = Instant::now();
            hydroplane::gemm(Trans::N, Trans::N, 1.0, Mat::new(&a, size, size), Mat::new(&b, size, size), 0.0, MatMut::new(&mut c, size, size));
            best = best.min(t.elapsed().as_secs_f64());
        }
        let flops = 2.0 * (size as f64).powi(3);
        println!("  hydroplane::gemm (f32) {size}³{:>27} {:>8.2} ms  {:>7.1} GFLOP/s", "", best * 1e3, flops / best / 1e9);
    }
}
