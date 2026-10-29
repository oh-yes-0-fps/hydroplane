//! Times ray and nearest-point queries on each node width, against the width this machine picks.
//!
//! `cargo bench -p hydroplane-solvers --bench bvh -- [primitives] [max_leaf]`

use std::hint::black_box;
use std::ops::Range;
use std::time::Instant;

use hydroplane_solvers::bvh::{Aabb, Backend, BackendAll, Bvh, Gang, NearestLeaves, Options, native_width};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn point(&mut self, scale: f32) -> [f32; 3] {
        [0; 3].map(|_| (self.next() * 2.0 - 1.0) * scale)
    }
}

/// Points in slot order, tested a register at a time.
struct Cloud<'a> {
    cols: &'a [Vec<f32>; 3],
    p: [f32; 3],
}

impl NearestLeaves for Cloud<'_> {
    #[inline(always)]
    fn nearest<S: BackendAll + Backend<f32>>(&mut self, g: Gang<S>, slots: Range<usize>, best: f32) -> Option<f32> {
        let lanes = g.lanes::<f32>();
        let mut m = g.splat(f32::INFINITY);
        let mut s = slots.start;
        while s < slots.end {
            let e = (s + lanes).min(slots.end);
            let dx = g.load_partial(&self.cols[0][s..e], f32::INFINITY) - self.p[0];
            let dy = g.load_partial(&self.cols[1][s..e], f32::INFINITY) - self.p[1];
            let dz = g.load_partial(&self.cols[2][s..e], f32::INFINITY) - self.p[2];
            m = m.min(dx * dx + dy * dy + dz * dz);
            s = e;
        }
        let m = m.reduce_min();
        (m < best).then_some(m)
    }
}

fn slab(b: &Aabb, o: [f32; 3], inv: [f32; 3]) -> f32 {
    let (mut near, mut far) = (0.0f32, f32::INFINITY);
    for a in 0..3 {
        let (t0, t1) = ((b.min[a] - o[a]) * inv[a], (b.max[a] - o[a]) * inv[a]);
        near = near.max(t0.min(t1));
        far = far.min(t0.max(t1));
    }
    if near <= far { near } else { f32::INFINITY }
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let n = args.first().copied().unwrap_or(200_000);
    let max_leaf = args.get(1).copied();
    let mut r = Lcg(1);
    let pts: Vec<[f32; 3]> = (0..n).map(|_| r.point(100.0)).collect();
    let boxes: Vec<Aabb> = pts.iter().map(|&p| Aabb::cube(p, 0.4)).collect();
    let rays: Vec<([f32; 3], [f32; 3])> = (0..100_000).map(|_| (r.point(120.0), r.point(1.0))).collect();
    let probes: Vec<[f32; 3]> = (0..100_000).map(|_| r.point(110.0)).collect();

    println!("{n} primitives, native width {}", native_width());
    for width in [4, 8, 16] {
        let t = Instant::now();
        let bvh = Bvh::with_options(&boxes, Options { width: Some(width), max_leaf });
        let build = t.elapsed();
        let sorted_boxes = bvh.in_slot_order(&boxes);

        let t = Instant::now();
        let mut hits = 0;
        for &(o, d) in &rays {
            let inv = d.map(|v| 1.0 / v);
            let hit = bvh.ray(o, d, f32::INFINITY, &mut |slots: Range<usize>, t_max: f32| {
                let t = sorted_boxes[slots].iter().map(|b| slab(b, o, inv)).fold(f32::INFINITY, f32::min);
                (t < t_max).then_some(t)
            });
            hits += usize::from(hit.is_some());
        }
        let ray = t.elapsed();

        let sorted = bvh.in_slot_order(&pts);
        let cols = [0, 1, 2].map(|a| sorted.iter().map(|p| p[a]).collect());
        let t = Instant::now();
        for &p in &probes {
            black_box(bvh.nearest(p, f32::INFINITY, &mut Cloud { cols: &cols, p }));
        }
        let nearest = t.elapsed();

        println!(
            "width {width:2}: build {build:>9.2?}  rays {:>6.0} ns/query ({hits} hits)  nearest {:>6.0} ns/query",
            ray.as_nanos() as f64 / rays.len() as f64,
            nearest.as_nanos() as f64 / probes.len() as f64,
        );
    }
}
