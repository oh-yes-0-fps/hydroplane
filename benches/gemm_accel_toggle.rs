//! f32 GEMM through `dispatch_matrix`, comparing the compile-time backend choice: Apple
//! Accelerate (default) vs the SME2 grid kernel (`--cfg hp_no_apple_accelerate`). Compare two builds:
//!   cargo bench --bench gemm_accel_toggle -- --save-baseline accelerate
//!   RUSTFLAGS="--cfg hp_no_apple_accelerate" cargo bench --bench gemm_accel_toggle -- --baseline accelerate

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use hydroplane::{FloatScalar, Layout, MatrixBackend, MatrixKernel, Gang, dispatch_matrix};
use std::hint::black_box;

/// `out = A·B` for a single `S×S×S` tile, large enough to exceed both engines' minimum dims.
struct Gemm<'a, T: FloatScalar, const M: usize, const N: usize, const K: usize> {
    a: &'a [T],
    b: &'a [T],
    out: &'a mut [T::Compute],
}
impl<T: FloatScalar, const M: usize, const N: usize, const K: usize> MatrixKernel<T>
    for Gemm<'_, T, M, N, K>
{
    type Output = ();
    fn run<S: MatrixBackend<T>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles();
        let a = tl.load_a::<M, K>(self.a, K, Layout::RowMajor);
        let b = tl.load_b::<K, N>(self.b, N, Layout::RowMajor);
        let acc = tl.mma::<M, N, K>(a, b, tl.zero_acc::<M, N>());
        acc.store(self.out, N, Layout::RowMajor);
    }
}

fn bench_size<const S: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let a: Vec<f32> = (0..S * S).map(|i| ((i % 7) as f32) - 3.0).collect();
    let b: Vec<f32> = (0..S * S).map(|i| ((i % 5) as f32) - 2.0).collect();
    let mut out = vec![0.0f32; S * S];
    group.throughput(Throughput::Elements(2 * (S as u64).pow(3))); // FLOPs, reported as elem/s
    group.bench_with_input(BenchmarkId::from_parameter(S), &S, |bch, _| {
        bch.iter(|| {
            dispatch_matrix(Gemm::<f32, S, S, S> {
                a: black_box(&a),
                b: black_box(&b),
                out: black_box(&mut out),
            })
        });
    });
}

fn benches(c: &mut Criterion) {
    let mode = if cfg!(hp_no_apple_accelerate) {
        "hp_no_apple_accelerate → hand-rolled SME2 grid kernel"
    } else {
        "default → Apple Accelerate (cblas)"
    };
    eprintln!("== GEMM dispatch backend: {mode} ==");
    let mut group = c.benchmark_group("gemm_f32_dispatch");
    bench_size::<128>(&mut group);
    bench_size::<256>(&mut group);
    bench_size::<512>(&mut group);
    bench_size::<1024>(&mut group);
    group.finish();
}

fn main() {
    // The single-tile kernel holds an owned `M×N` accumulator (4 MiB at N=1024) on the stack, so
    // run criterion on a thread with a generous stack.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(|| {
            let mut c = Criterion::default().configure_from_args();
            benches(&mut c);
            c.final_summary();
        })
        .unwrap()
        .join()
        .unwrap();
}
