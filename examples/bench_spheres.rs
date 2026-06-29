//! Lightweight throughput check for the ported sphere kernel: scalar backend vs the
//! dispatched SIMD backend, for f32 and f64. Run with:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench_spheres
//!
//! (Not a statistical benchmark — for that, wire criterion as in `wreck/benches`. This is
//! a sanity check that the SIMD path is actually faster and that dispatch picks it up.)

use std::time::Instant;

use hydroplane::{Backend, Scalar, Gang, Soa};

fn spheres_soa<T: Scalar>(rows: &[[T; 4]]) -> Soa<T> {
    let mut soa = Soa::with_pad_fills(&[T::ZERO, T::ZERO, T::ZERO, T::from_f64(f64::NAN)]);
    for r in rows {
        soa.push_row(r);
    }
    soa
}

fn any_overlap<T: Scalar, S: Backend<T>>(ctx: Gang<T, S>, soa: &Soa<T>, q: [T; 4]) -> bool {
    let lanes = ctx.lanes();
    let (cx, cy, cz, sr) = (
        ctx.splat(q[0]),
        ctx.splat(q[1]),
        ctx.splat(q[2]),
        ctx.splat(q[3]),
    );
    let (xs, ys, zs, rs) = (soa.column(0), soa.column(1), soa.column(2), soa.column(3));
    let mut k = 0;
    while k < soa.padded() {
        let x = ctx.load(&xs[k..k + lanes]);
        let y = ctx.load(&ys[k..k + lanes]);
        let z = ctx.load(&zs[k..k + lanes]);
        let r = ctx.load(&rs[k..k + lanes]);
        let dx = cx - x;
        let dy = cy - y;
        let dz = cz - z;
        let d2 = dx * dx + dy * dy + dz * dz;
        let rsum = sr + r;
        if d2.le(rsum * rsum).any() {
            return true;
        }
        k += lanes;
    }
    false
}

fn bench<T: hydroplane::SimdDispatch>(label: &str, to: impl Fn(f64) -> T) {
    // A batch the query (mostly) misses, so the kernel scans the whole SoA — worst case.
    let n = 4096;
    let rows: Vec<[T; 4]> = (0..n)
        .map(|i| {
            let f = i as f64;
            [to(f * 0.001 + 100.0), to(0.0), to(0.0), to(0.25)]
        })
        .collect();
    let soa = spheres_soa(&rows);
    let queries: Vec<[T; 4]> = (0..1000)
        .map(|i| [to((i as f64) * 0.01 - 5.0), to(0.0), to(0.0), to(0.1)])
        .collect();

    let iters = 2000usize;

    // scalar
    let t0 = Instant::now();
    let mut acc = 0u64;
    for _ in 0..iters {
        for q in &queries {
            acc += hydroplane::run_scalar(Query { soa: &soa, q: *q }) as u64;
        }
    }
    let scalar = t0.elapsed();

    // dispatched (SIMD where available)
    let t1 = Instant::now();
    let mut acc2 = 0u64;
    for _ in 0..iters {
        for q in &queries {
            acc2 += hydroplane::dispatch(Query { soa: &soa, q: *q }) as u64;
        }
    }
    let simd = t1.elapsed();

    assert_eq!(acc, acc2, "scalar and dispatched disagree!");
    let total = (iters * queries.len() * n) as f64;
    println!(
        "{label:>4}: scalar {:>8.2} M sphere-tests/s | dispatched {:>8.2} M | speedup {:.2}x",
        total / scalar.as_secs_f64() / 1e6,
        total / simd.as_secs_f64() / 1e6,
        scalar.as_secs_f64() / simd.as_secs_f64(),
    );
}

struct Query<'a, T: Scalar> {
    soa: &'a Soa<T>,
    q: [T; 4],
}
impl<T: Scalar> hydroplane::Kernel<T> for Query<'_, T> {
    type Output = bool;
    fn run<S: Backend<T>>(self, simd: Gang<T, S>) -> bool {
        any_overlap(simd, self.soa, self.q)
    }
}

fn main() {
    bench::<f32>("f32", |v| v as f32);
    bench::<f64>("f64", |v| v);
}
