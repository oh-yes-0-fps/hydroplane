//! Interop with the [`soa-rs`](https://docs.rs/soa-rs) crate: write your data as an ordinary
//! `#[derive(Soars)]` struct, then run a hydroplane kernel over its field slices, either zero-copy
//! in place or via the `Soa::from_columns` bridge.

use soa_rs::{Soars, soa};
use hydroplane::{BackendAll, Kernel, Gang, dispatch};

#[derive(Soars, Debug, Clone, Copy)]
struct Sphere {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
}

/// Zero-copy: the kernel reads the borrowed `soa-rs` field slices directly. `load_partial` stages
/// the tail, filling the radius column's inactive lanes with `NaN` so they never produce a false
/// overlap.
struct AnyOverlap<'a> {
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
}
impl Kernel<f32> for AnyOverlap<'_> {
    type Output = bool;
    fn run<S: BackendAll>(self, ctx: Gang<S>) -> bool {
        let (cx, cy, cz, sr) = (
            ctx.splat(self.q[0]),
            ctx.splat(self.q[1]),
            ctx.splat(self.q[2]),
            ctx.splat(self.q[3]),
        );
        let n = ctx.lanes::<f32>();
        for k in ctx.chunks_exact::<f32>(self.xs.len()) {
            let dx = cx - ctx.load(&self.xs[k..k + n]);
            let dy = cy - ctx.load(&self.ys[k..k + n]);
            let dz = cz - ctx.load(&self.zs[k..k + n]);
            let r = ctx.load(&self.rs[k..k + n]);
            let d2 = dx * dx + dy * dy + dz * dz;
            let rsum = sr + r;
            if d2.le(rsum * rsum).any() {
                return true;
            }
        }
        if let Some((k, cnt)) = ctx.remainder::<f32>(self.xs.len()) {
            let dx = cx - ctx.load_partial(&self.xs[k..k + cnt], 0.0);
            let dy = cy - ctx.load_partial(&self.ys[k..k + cnt], 0.0);
            let dz = cz - ctx.load_partial(&self.zs[k..k + cnt], 0.0);
            // NaN radius fill: `d2 <= (sr + NaN)^2` is false, so padding lanes can never hit.
            let r = ctx.load_partial(&self.rs[k..k + cnt], f32::NAN);
            let d2 = dx * dx + dy * dy + dz * dz;
            let rsum = sr + r;
            return d2.le(rsum * rsum).any();
        }
        false
    }
}

fn main() {
    let s: soa_rs::Soa<Sphere> = soa![
        Sphere { x: 0.0, y: 0.0, z: 0.0, r: 0.5 },
        Sphere { x: 5.0, y: 0.0, z: 0.0, r: 0.5 },
        Sphere { x: 10.0, y: 0.0, z: 0.0, r: 0.5 },
    ];

    // Zero-copy: operate on the borrowed field slices, nothing allocated.
    let query = |q: [f32; 4]| {
        dispatch(AnyOverlap {
            xs: s.x(),
            ys: s.y(),
            zs: s.z(),
            rs: s.r(),
            q,
        })
    };
    println!("borrow  hit  near (5,0,0): {}", query([5.2, 0.0, 0.0, 0.1])); // true
    println!("borrow  miss near (2,0,0): {}", query([2.0, 0.0, 0.0, 0.1])); // false

    // Copy bridge: build a padded `hydroplane::Soa` from the same field slices when you'd rather
    // reuse a padded-column kernel.
    let cols = hydroplane::Soa::from_columns(&[s.x(), s.y(), s.z(), s.r()], &[0.0, 0.0, 0.0, f32::NAN]);
    println!("bridged columns: {} rows, padded to {}", cols.len(), cols.padded());
}
