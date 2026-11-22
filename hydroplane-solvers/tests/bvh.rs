use core::ops::{ControlFlow, Range};

use hydroplane_solvers::bvh::{
    Aabb, Backend, BackendAll, Bvh, Gang, NearestLeaves, Options, native_width,
};

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

fn random_boxes(n: usize, seed: u64) -> Vec<Aabb> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let c = r.point(10.0);
            let h = [0; 3].map(|_| r.next() * 0.6);
            Aabb::new([0, 1, 2].map(|a| c[a] - h[a]), [0, 1, 2].map(|a| c[a] + h[a]))
        })
        .collect()
}

fn ray_box(b: &Aabb, o: [f32; 3], d: [f32; 3]) -> Option<f32> {
    let (mut near, mut far) = (0.0f32, f32::INFINITY);
    for a in 0..3 {
        let inv = 1.0 / d[a];
        let (t0, t1) = ((b.min[a] - o[a]) * inv, (b.max[a] - o[a]) * inv);
        near = near.max(t0.min(t1));
        far = far.min(t0.max(t1));
    }
    (near <= far).then_some(near)
}

fn dist2_box(b: &Aabb, p: [f32; 3]) -> f32 {
    (0..3)
        .map(|a| {
            let d = (b.min[a] - p[a]).max(p[a] - b.max[a]).max(0.0);
            d * d
        })
        .sum()
}

fn configs() -> Vec<Options> {
    let mut out = vec![Options::default()];
    for width in [4, 8, 16] {
        for max_leaf in [None, Some(1), Some(3)] {
            out.push(Options { width: Some(width), max_leaf });
        }
    }
    out
}

fn closest_hit(bvh: &Bvh, boxes: &[Aabb], o: [f32; 3], d: [f32; 3]) -> Option<(f32, u32)> {
    let mut prim = None;
    let t = bvh.ray(o, d, f32::INFINITY, &mut |slots: Range<usize>, t_max: f32| {
        let mut best = None::<f32>;
        for s in slots {
            let pi = bvh.order()[s];
            if let Some(t) = ray_box(&boxes[pi as usize], o, d)
                && t < best.unwrap_or(t_max)
            {
                best = Some(t);
                prim = Some(pi);
            }
        }
        best
    });
    t.map(|t| (t, prim.unwrap()))
}

#[test]
fn native_width_is_supported() {
    assert!([4, 8, 16].contains(&native_width()));
    assert_eq!(Bvh::new(&random_boxes(10, 1)).width(), native_width());
}

#[test]
fn ray_matches_brute_force() {
    let boxes = random_boxes(2000, 7);
    let mut r = Lcg(99);
    for opt in configs() {
        let bvh = Bvh::with_options(&boxes, opt);
        for _ in 0..300 {
            let o = r.point(15.0);
            let mut d = r.point(1.0);
            if r.next() < 0.2 {
                d[(r.next() * 3.0) as usize % 3] = 0.0;
            }
            let brute = boxes
                .iter()
                .filter_map(|b| ray_box(b, o, d))
                .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.min(t))));
            let got = closest_hit(&bvh, &boxes, o, d);
            assert_eq!(got.map(|g| g.0), brute, "{opt:?} o={o:?} d={d:?}");
            if let Some((t, pi)) = got {
                assert_eq!(ray_box(&boxes[pi as usize], o, d), Some(t));
            }
        }
    }
}

#[test]
fn ray_respects_t_max() {
    let boxes = vec![Aabb::cube([0.0, 0.0, 0.0], 1.0), Aabb::cube([5.0, 0.0, 0.0], 1.0)];
    let bvh = Bvh::new(&boxes);
    let mut leaf = |slots: Range<usize>, t_max: f32| {
        slots
            .filter_map(|s| ray_box(&boxes[bvh.order()[s] as usize], [-5.0, 0.0, 0.0], [1.0, 0.0, 0.0]))
            .filter(|&t| t < t_max)
            .reduce(f32::min)
    };
    assert_eq!(bvh.ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 3.0, &mut leaf), None);
    assert_eq!(bvh.ray([-5.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5.0, &mut leaf), Some(4.0));
}

#[test]
fn nearest_matches_brute_force() {
    let boxes = random_boxes(1500, 3);
    let mut r = Lcg(5);
    for opt in configs() {
        let bvh = Bvh::with_options(&boxes, opt);
        for _ in 0..300 {
            let p = r.point(14.0);
            let brute = boxes.iter().map(|b| dist2_box(b, p)).fold(f32::INFINITY, f32::min);
            let got = bvh.nearest(p, f32::INFINITY, &mut |slots: Range<usize>, best: f32| {
                slots
                    .map(|s| dist2_box(&boxes[bvh.order()[s] as usize], p))
                    .filter(|&d| d < best)
                    .reduce(f32::min)
            });
            assert_eq!(got, Some(brute), "{opt:?} p={p:?}");
        }
    }
}

#[test]
fn overlap_and_containing_match_brute_force() {
    let boxes = random_boxes(1500, 11);
    let mut r = Lcg(17);
    for opt in configs() {
        let bvh = Bvh::with_options(&boxes, opt);
        for _ in 0..200 {
            let q = Aabb::cube(r.point(12.0), r.next() * 3.0);
            let mut got = Vec::new();
            let flow = bvh.overlapping(&q, &mut |slots: Range<usize>| {
                got.extend(slots.map(|s| bvh.order()[s]).filter(|&pi| boxes[pi as usize].overlaps(&q)));
                ControlFlow::Continue(())
            });
            assert!(flow.is_continue());
            got.sort_unstable();
            let brute: Vec<u32> = (0..boxes.len() as u32).filter(|&i| boxes[i as usize].overlaps(&q)).collect();
            assert_eq!(got, brute, "{opt:?}");

            let p = r.point(10.0);
            let mut got = Vec::new();
            let _ = bvh.containing(p, &mut |slots: Range<usize>| {
                got.extend(slots.map(|s| bvh.order()[s]).filter(|&pi| boxes[pi as usize].contains(p)));
                ControlFlow::Continue(())
            });
            got.sort_unstable();
            let brute: Vec<u32> = (0..boxes.len() as u32).filter(|&i| boxes[i as usize].contains(p)).collect();
            assert_eq!(got, brute, "{opt:?}");
        }
    }
}

#[test]
fn overlap_stops_on_break() {
    let boxes = random_boxes(500, 2);
    let bvh = Bvh::with_options(&boxes, Options { max_leaf: Some(1), ..Default::default() });
    let mut visits = 0;
    let flow = bvh.overlapping(&bvh.bounds(), &mut |_: Range<usize>| {
        visits += 1;
        if visits == 3 { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    });
    assert!(flow.is_break());
    assert_eq!(visits, 3);
}

#[test]
fn refit_tracks_moved_boxes() {
    let mut boxes = random_boxes(800, 23);
    for opt in configs() {
        let mut bvh = Bvh::with_options(&boxes, opt);
        for (i, b) in boxes.iter_mut().enumerate() {
            let s = (i % 7) as f32 * 0.3;
            *b = Aabb::new(b.min.map(|v| v * 1.5 + s), b.max.map(|v| v * 1.5 + s));
        }
        bvh.refit(&boxes);
        assert_eq!(bvh.bounds(), boxes.iter().fold(Aabb::EMPTY, |a, b| a.union(b)));
        let mut r = Lcg(31);
        for _ in 0..100 {
            let p = r.point(20.0);
            let brute = boxes.iter().map(|b| dist2_box(b, p)).fold(f32::INFINITY, f32::min);
            let got = bvh.nearest(p, f32::INFINITY, &mut |slots: Range<usize>, best: f32| {
                slots
                    .map(|s| dist2_box(&boxes[bvh.order()[s] as usize], p))
                    .filter(|&d| d < best)
                    .reduce(f32::min)
            });
            assert_eq!(got, Some(brute), "{opt:?}");
        }
    }
}

#[test]
fn empty_and_degenerate() {
    let bvh = Bvh::new(&[]);
    assert!(bvh.is_empty());
    assert_eq!(bvh.ray([0.0; 3], [1.0, 0.0, 0.0], f32::INFINITY, &mut |_: Range<usize>, _: f32| Some(0.0)), None);
    assert!(bvh.overlapping(&Aabb::cube([0.0; 3], 1e9), &mut |_: Range<usize>| ControlFlow::Break(())).is_continue());

    let same = vec![Aabb::cube([1.0, 2.0, 3.0], 0.5); 100];
    for opt in configs() {
        let bvh = Bvh::with_options(&same, opt);
        let mut seen = vec![false; same.len()];
        let _ = bvh.containing([1.0, 2.0, 3.0], &mut |slots: Range<usize>| {
            for s in slots {
                seen[bvh.order()[s] as usize] = true;
            }
            ControlFlow::Continue(())
        });
        assert!(seen.iter().all(|&s| s), "{opt:?}");
    }
}

/// A point cloud in slot order, searched a register at a time on the traversal's own gang.
struct Cloud {
    cols: [Vec<f32>; 3],
    p: [f32; 3],
    slot: usize,
}

impl NearestLeaves for Cloud {
    #[inline(always)]
    fn nearest<S: BackendAll + Backend<f32>>(
        &mut self,
        g: Gang<S>,
        slots: Range<usize>,
        best: f32,
    ) -> Option<f32> {
        let lanes = g.lanes::<f32>();
        let mut found = None;
        let mut best = best;
        let mut s = slots.start;
        while s < slots.end {
            let e = (s + lanes).min(slots.end);
            let dx = g.load_partial(&self.cols[0][s..e], f32::INFINITY) - self.p[0];
            let dy = g.load_partial(&self.cols[1][s..e], f32::INFINITY) - self.p[1];
            let dz = g.load_partial(&self.cols[2][s..e], f32::INFINITY) - self.p[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            let m = d2.reduce_min();
            if m < best {
                best = m;
                found = Some(m);
                self.slot = s + d2.le(g.splat(m)).to_bitmask().trailing_zeros() as usize;
            }
            s = e;
        }
        found
    }
}

#[test]
fn gang_leaves_find_nearest_point() {
    let mut r = Lcg(41);
    let pts: Vec<[f32; 3]> = (0..5000).map(|_| r.point(10.0)).collect();
    let boxes: Vec<Aabb> = pts.iter().map(|&p| Aabb::point(p)).collect();
    for opt in configs() {
        let bvh = Bvh::with_options(&boxes, opt);
        let sorted = bvh.in_slot_order(&pts);
        let cols = [0, 1, 2].map(|a| sorted.iter().map(|p| p[a]).collect());
        let mut cloud = Cloud { cols, p: [0.0; 3], slot: 0 };
        for _ in 0..200 {
            let p = r.point(12.0);
            cloud.p = p;
            let d2 = bvh.nearest(p, f32::INFINITY, &mut cloud).unwrap();
            let got = bvh.order()[cloud.slot] as usize;
            let brute = (0..pts.len())
                .min_by(|&a, &b| {
                    let da: f32 = (0..3).map(|k| (pts[a][k] - p[k]).powi(2)).sum();
                    let db: f32 = (0..3).map(|k| (pts[b][k] - p[k]).powi(2)).sum();
                    da.total_cmp(&db)
                })
                .unwrap();
            let bd: f32 = (0..3).map(|k| (pts[brute][k] - p[k]).powi(2)).sum();
            assert_eq!(d2, bd, "{opt:?}");
            assert_eq!(pts[got], pts[brute], "{opt:?}");
        }
    }
}

#[test]
fn f64_points_round_outward() {
    let pts = [[0.1f64, 1e-9, -0.3], [1.0 / 3.0, 2.0, 7.7]];
    let b = Aabb::from_points_f64(&pts);
    for p in pts {
        for (a, &v) in p.iter().enumerate() {
            assert!(f64::from(b.min[a]) <= v && v <= f64::from(b.max[a]));
        }
    }
}
