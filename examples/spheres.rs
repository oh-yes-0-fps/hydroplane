//! An SoA sphere-overlap query kernel ("does the query overlap any sphere in the batch?"),
//! written once with `#[kernel]` and run for `f32` and `f64` on whatever backend `dispatch`
//! selects.

use hydroplane::{Gang, Scalar, Soa, kernel};

/// Columns of the sphere SoA.
const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;
const R: usize = 3;

/// Sphere SoA whose radius column pads inactive lanes with `NaN`, so the `d² ≤ (r+R)²` test is
/// always false on padding.
pub fn spheres_soa<T: Scalar>(rows: &[[T; 4]]) -> Soa<T> {
    let mut soa = Soa::with_pad_fills(&[T::ZERO, T::ZERO, T::ZERO, T::from_f64(f64::NAN)]);
    for row in rows {
        soa.push_row(row);
    }
    soa
}

/// `q = [cx, cy, cz, radius]`. `#[kernel]` generates the dispatching `any_overlap(soa, q)`
/// callable; no struct, impl, or `dispatch` by hand.
#[kernel]
pub fn any_overlap<'a, T: Scalar>(ctx: Gang, soa: &'a Soa<T>, q: [T; 4]) -> bool {
    let lanes = ctx.lanes::<T>();

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

fn main() {
    let rows_f32 = [
        [0.0f32, 0.0, 0.0, 0.5],
        [5.0, 0.0, 0.0, 0.5],
        [10.0, 0.0, 0.0, 0.5],
    ];
    let soa = spheres_soa(&rows_f32);
    println!("f32 hit  near (5,0,0): {}", any_overlap(&soa, [5.2f32, 0.0, 0.0, 0.1])); // true
    println!("f32 miss near (2,0,0): {}", any_overlap(&soa, [2.0f32, 0.0, 0.0, 0.1])); // false

    let rows_f64 = rows_f32.map(|[a, b, c, d]| [a as f64, b as f64, c as f64, d as f64]);
    let soa = spheres_soa(&rows_f64);
    println!("f64 hit  near (10,0,0): {}", any_overlap(&soa, [10.0f64, 0.4, 0.0, 0.2])); // true
}
