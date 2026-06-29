//! End-to-end proof: `wreck`'s sphere–sphere "does the query overlap any sphere in the
//! batch?" kernel, written **once** against the `hydroplane` varying layer and run for `f32`
//! *and* `f64` on whatever backend `dispatch` selects (AVX2 when present, else scalar).
//!
//! Run with `cargo run --example spheres --release` (add `-C target-cpu=native` via
//! RUSTFLAGS to force the compile-time AVX2 path).

use hydroplane::{Backend, Kernel, Scalar, Simd, Soa, SimdDispatch, dispatch};

/// Columns of the sphere SoA.
const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;
const R: usize = 3;

/// Build a sphere SoA whose radius column pads inactive lanes with `NaN`, so the
/// `d² ≤ (r+R)²` test is always false on padding (no false positives).
pub fn spheres_soa<T: Scalar>(rows: &[[T; 4]]) -> Soa<T> {
    let mut soa = Soa::with_pad_fills(&[T::ZERO, T::ZERO, T::ZERO, T::from_f64(f64::NAN)]);
    for row in rows {
        soa.push_row(row);
    }
    soa
}

/// The kernel — reads like scalar Rust, runs as SIMD. `q = [cx, cy, cz, radius]`.
pub fn any_overlap<T: Scalar, S: Backend<T>>(ctx: Simd<T, S>, soa: &Soa<T>, q: [T; 4]) -> bool {
    let lanes = ctx.lanes();

    let cx = ctx.splat(q[X]);
    let cy = ctx.splat(q[Y]);
    let cz = ctx.splat(q[Z]);
    let sr = ctx.splat(q[R]);

    let (xs, ys, zs, rs) = (soa.column(X), soa.column(Y), soa.column(Z), soa.column(R));

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

/// `Kernel` wrapper so the same code runs through `hydroplane::dispatch`.
pub struct AnyOverlap<'a, T: Scalar> {
    pub soa: &'a Soa<T>,
    pub query: [T; 4],
}
impl<T: Scalar> Kernel<T> for AnyOverlap<'_, T> {
    type Output = bool;
    fn run<S: Backend<T>>(self, simd: Simd<T, S>) -> bool {
        any_overlap(simd, self.soa, self.query)
    }
}

/// Convenience: build + dispatch in one call.
pub fn query<T: SimdDispatch>(soa: &Soa<T>, query: [T; 4]) -> bool {
    dispatch(AnyOverlap { soa, query })
}

fn main() {
    // a handful of f32 spheres on a line, plus an f64 run of the same data
    let rows_f32 = [
        [0.0f32, 0.0, 0.0, 0.5],
        [5.0, 0.0, 0.0, 0.5],
        [10.0, 0.0, 0.0, 0.5],
    ];
    let soa = spheres_soa(&rows_f32);
    println!("f32 hit  near (5,0,0): {}", query(&soa, [5.2f32, 0.0, 0.0, 0.1])); // true
    println!("f32 miss near (2,0,0): {}", query(&soa, [2.0f32, 0.0, 0.0, 0.1])); // false

    let rows_f64 = rows_f32.map(|[a, b, c, d]| [a as f64, b as f64, c as f64, d as f64]);
    let soa = spheres_soa(&rows_f64);
    println!("f64 hit  near (10,0,0): {}", query(&soa, [10.0f64, 0.4, 0.0, 0.2])); // true
}
