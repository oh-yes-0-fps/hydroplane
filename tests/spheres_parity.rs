//! The ported sphere kernel must give identical hit/miss results to a brute-force reference
//! — for f32 and f64, on whatever backend `dispatch` selects. This is the `hydroplane` analogue of
//! `wreck`'s "0 mismatches vs CPU" methodology.

use rand::Rng;
use hydroplane::{BackendAll, FloatScalar, Backend, Kernel, Gang, SimdDispatch, Soa, dispatch};

fn spheres_soa<T: FloatScalar>(rows: &[[T; 4]]) -> Soa<T> {
    let mut soa = Soa::with_pad_fills(&[T::ZERO, T::ZERO, T::ZERO, T::from_f64(f64::NAN)]);
    for row in rows {
        soa.push_row(row);
    }
    soa
}

fn any_overlap<T: FloatScalar, S: Backend<T>>(ctx: Gang<S>, soa: &Soa<T>, q: [T; 4]) -> bool {
    let lanes = ctx.lanes::<T>();
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

struct AnyOverlap<'a, T: FloatScalar> {
    soa: &'a Soa<T>,
    q: [T; 4],
}
impl<T: FloatScalar> Kernel<T> for AnyOverlap<'_, T> {
    type Output = bool;
    fn run<S: BackendAll + Backend<T>>(self, simd: Gang<S>) -> bool {
        any_overlap(simd, self.soa, self.q)
    }
}

/// Brute-force reference using the scalar ops directly — exact same arithmetic, so the
/// SIMD path must agree bit-for-bit (the kernel uses no FMA).
fn naive<T: FloatScalar>(rows: &[[T; 4]], q: [T; 4]) -> bool {
    rows.iter().any(|s| {
        let dx = q[0] - s[0];
        let dy = q[1] - s[1];
        let dz = q[2] - s[2];
        let d2 = dx * dx + (dy * dy + dz * dz);
        let rsum = q[3] + s[3];
        d2 <= rsum * rsum
    })
}

fn run_for<T: FloatScalar + SimdDispatch>(to: impl Fn(f64) -> T) {
    let mut rng = rand::rng();
    for _ in 0..400 {
        let n = rng.random_range(0..40);
        let rows: Vec<[T; 4]> = (0..n)
            .map(|_| {
                [
                    to(rng.random_range(-10.0..10.0)),
                    to(rng.random_range(-10.0..10.0)),
                    to(rng.random_range(-10.0..10.0)),
                    to(rng.random_range(0.1..1.5)),
                ]
            })
            .collect();
        let soa = spheres_soa(&rows);

        for _ in 0..20 {
            let q = [
                to(rng.random_range(-11.0..11.0)),
                to(rng.random_range(-11.0..11.0)),
                to(rng.random_range(-11.0..11.0)),
                to(rng.random_range(0.1..1.5)),
            ];
            let want = naive(&rows, q);
            let scalar = hydroplane::run_scalar(AnyOverlap { soa: &soa, q });
            let dispatched = dispatch(AnyOverlap { soa: &soa, q });
            assert_eq!(scalar, want, "scalar vs naive (n={n})");
            assert_eq!(dispatched, want, "dispatched vs naive (n={n})");
        }
    }
}

#[test]
fn f32_parity() {
    run_for(|v| v as f32);
}

#[test]
fn f64_parity() {
    run_for(|v| v);
}
