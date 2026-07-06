//! `soa-rs` interop: a `#[derive(Soars)]` struct's field slices feed hydroplane two ways, the
//! zero-copy `chunks` + `load_partial` path and the `Soa::from_columns` copy bridge. Both must
//! agree with a brute-force reference.

use rand::Rng;
use soa_rs::{Soars, soa};
use hydroplane::{BackendAll, Kernel, Gang, dispatch};

#[derive(Soars, Debug, Clone, Copy)]
struct Sphere {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
}

/// Zero-copy: run directly over the borrowed `soa-rs` field slices. The tail is staged by
/// `load_partial` with `NaN` in the radius column so inactive lanes never produce a false hit.
struct AnyOverlapBorrowed<'a> {
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
}
impl Kernel<f32> for AnyOverlapBorrowed<'_> {
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
            let r = ctx.load_partial(&self.rs[k..k + cnt], f32::NAN);
            let d2 = dx * dx + dy * dy + dz * dz;
            let rsum = sr + r;
            return d2.le(rsum * rsum).any();
        }
        false
    }
}

/// Copy bridge: a kernel written against `hydroplane`'s padded columns, fed by `Soa::from_columns`.
struct AnyOverlapPadded<'a> {
    soa: &'a hydroplane::Soa<f32>,
    q: [f32; 4],
}
impl Kernel<f32> for AnyOverlapPadded<'_> {
    type Output = bool;
    fn run<S: BackendAll>(self, ctx: Gang<S>) -> bool {
        let n = ctx.lanes::<f32>();
        let (cx, cy, cz, sr) = (
            ctx.splat(self.q[0]),
            ctx.splat(self.q[1]),
            ctx.splat(self.q[2]),
            ctx.splat(self.q[3]),
        );
        let (xs, ys, zs, rs) = (
            self.soa.column(0),
            self.soa.column(1),
            self.soa.column(2),
            self.soa.column(3),
        );
        let mut k = 0;
        while k < self.soa.padded() {
            let dx = cx - ctx.load(&xs[k..k + n]);
            let dy = cy - ctx.load(&ys[k..k + n]);
            let dz = cz - ctx.load(&zs[k..k + n]);
            let d2 = dx * dx + dy * dy + dz * dz;
            let rsum = sr + ctx.load(&rs[k..k + n]);
            if d2.le(rsum * rsum).any() {
                return true;
            }
            k += n;
        }
        false
    }
}

fn naive(spheres: &[Sphere], q: [f32; 4]) -> bool {
    spheres.iter().any(|s| {
        let (dx, dy, dz) = (q[0] - s.x, q[1] - s.y, q[2] - s.z);
        let rsum = q[3] + s.r;
        dx * dx + dy * dy + dz * dz <= rsum * rsum
    })
}

#[test]
fn both_paths_match_reference() {
    let mut rng = rand::rng();
    for _ in 0..300 {
        let n = rng.random_range(0..40);
        let spheres: Vec<Sphere> = (0..n)
            .map(|_| Sphere {
                x: rng.random_range(-10.0..10.0),
                y: rng.random_range(-10.0..10.0),
                z: rng.random_range(-10.0..10.0),
                r: rng.random_range(0.1..1.5),
            })
            .collect();

        let s: soa_rs::Soa<Sphere> = spheres.iter().copied().collect();
        let padded = hydroplane::Soa::from_columns(&[s.x(), s.y(), s.z(), s.r()], &[0.0, 0.0, 0.0, f32::NAN]);

        for _ in 0..20 {
            let q = [
                rng.random_range(-11.0..11.0),
                rng.random_range(-11.0..11.0),
                rng.random_range(-11.0..11.0),
                rng.random_range(0.1..1.5),
            ];
            let want = naive(&spheres, q);

            let borrowed = dispatch(AnyOverlapBorrowed {
                xs: s.x(),
                ys: s.y(),
                zs: s.z(),
                rs: s.r(),
                q,
            });
            let bridged = dispatch(AnyOverlapPadded { soa: &padded, q });

            assert_eq!(borrowed, want, "zero-copy borrow vs naive (n={n})");
            assert_eq!(bridged, want, "copy bridge vs naive (n={n})");
        }
    }
}

#[test]
fn store_partial_writes_back_into_soa_rs() {
    let _ = soa![Sphere { x: 0.0, y: 0.0, z: 0.0, r: 1.0 }];
    // Scale every sphere's radius by 2 in place, writing through a `soa-rs` mutable slice.
    struct Scale<'a> {
        rs: &'a mut [f32],
        k: f32,
    }
    impl Kernel<f32> for Scale<'_> {
        type Output = ();
        fn run<S: BackendAll>(self, ctx: Gang<S>) {
            let kv = ctx.splat(self.k);
            ctx.for_each_chunk::<f32>(self.rs.len(), |i, cnt| {
                let v = ctx.load_partial(&self.rs[i..i + cnt], 0.0) * kv;
                v.store_partial(&mut self.rs[i..i + cnt]);
            });
        }
    }

    let mut s: soa_rs::Soa<Sphere> = (0..50)
        .map(|i| Sphere { x: 0.0, y: 0.0, z: 0.0, r: i as f32 })
        .collect();
    dispatch(Scale { rs: s.r_mut(), k: 2.0 });
    for (i, r) in s.r().iter().enumerate() {
        assert_eq!(*r, (i as f32) * 2.0, "lane {i}");
    }
}
