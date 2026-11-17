//! Cost of one CPU `mma` on L1-resident 16×16×16 tiles, f16 and f32, through `dispatch_matrix`.

use half::f16;
use hydroplane::{FloatScalar, Gang, Layout, MatrixBackend, MatrixDispatch, MatrixKernel, Scalar};
use std::time::Instant;

const REPS: usize = 100_000;

struct Repeat<'a, T: FloatScalar> {
    a: &'a [T],
    b: &'a [T],
    c: &'a mut [T::Compute],
}

impl<T: FloatScalar> MatrixKernel<T> for Repeat<'_, T> {
    type Output = ();
    #[inline]
    fn run<S: MatrixBackend<T>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<T>();
        let mut acc = tl.zero_acc::<16, 16>();
        for _ in 0..REPS {
            acc = tl.mma(tl.load_a::<16, 16>(self.a, 16, Layout::RowMajor), tl.load_b::<16, 16>(self.b, 16, Layout::RowMajor), acc);
        }
        acc.store_rm(self.c);
    }
}

fn time<T: FloatScalar + MatrixDispatch>(label: &str, from: impl Fn(f32) -> T) {
    let a: Vec<T> = (0..256).map(|i| from((i % 7) as f32 * 0.001)).collect();
    let b: Vec<T> = (0..256).map(|i| from((i % 5) as f32 * 0.001)).collect();
    let mut c = vec![<T::Compute as Scalar>::ZERO; 256];
    hydroplane::dispatch_matrix(Repeat { a: &a, b: &b, c: &mut c });
    let t = Instant::now();
    hydroplane::dispatch_matrix(Repeat { a: &a, b: &b, c: &mut c });
    let per = t.elapsed().as_secs_f64() / REPS as f64;
    println!("  {label:<6} {:>8.1} ns per 16×16×16 mma  {:>6.1} GFLOP/s", per * 1e9, 2.0 * 4096.0 / per / 1e9);
}

fn main() {
    println!("CPU tile mma, L1-resident operands");
    time::<f32>("f32", |x| x);
    time::<f16>("f16", f16::from_f32);
}
