//! The fixed-`N` column combinators (`count_n`, `for_each_hit_n`), the masked-chunk driver, and the
//! array splat/load/gather conveniences each agree with a scalar reference across lengths that cross
//! the register boundary (including a short final tail).

use hydroplane::{Gang, MAX_LANES, Scalar, kernel};

fn spheres(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut xs = Vec::new();
    let (mut ys, mut zs, mut rs) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let f = i as f32;
        xs.push((f * 0.37).sin() * 10.0);
        ys.push((f * 0.51).cos() * 10.0);
        zs.push((f * 0.13).sin() * 10.0);
        rs.push(0.5 + (f * 0.07).cos().abs());
    }
    (xs, ys, zs, rs)
}

fn overlaps(
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    rs: &[f32],
    q: [f32; 4],
    i: usize,
) -> bool {
    let (dx, dy, dz) = (q[0] - xs[i], q[1] - ys[i], q[2] - zs[i]);
    let rsum = q[3] + rs[i];
    dx * dx + dy * dy + dz * dz <= rsum * rsum
}

#[kernel]
fn count_k<'a>(ctx: Gang<f32>, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32], rs: &'a [f32], q: [f32; 4]) -> usize {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    ctx.count_n([xs, ys, zs, rs], |[x, y, z, r]| {
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        (dx * dx + dy * dy + dz * dz).le(rsum * rsum)
    })
}

#[kernel]
fn collect_k<'a>(
    ctx: Gang<f32>,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    rs: &'a [f32],
    q: [f32; 4],
    out: &'a mut [bool],
) -> bool {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    ctx.for_each_hit_n(
        [xs, ys, zs, rs],
        |[x, y, z, r]| {
            let (dx, dy, dz) = (cx - x, cy - y, cz - z);
            let rsum = sr + r;
            (dx * dx + dy * dy + dz * dz).le(rsum * rsum)
        },
        |i| out[i] = true,
    )
}

/// Two-phase loop over `masked_chunks` + `gather_n` from an AoS `&[(x,y,z,r)]`, mirroring the
/// broadphase-reject-then-narrowphase shape — must match the scalar any-overlap.
#[kernel]
fn gather_any_k<'a>(ctx: Gang<f32>, pts: &'a [(f32, f32, f32, f32)], q: [f32; 4]) -> bool {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    for (off, cnt, active) in ctx.masked_chunks(pts.len()) {
        let [x, y, z, r] = ctx.gather_n(&pts[off..off + cnt], [0.0; 4], |p| [p.0, p.1, p.2, p.3]);
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        if ((dx * dx + dy * dy + dz * dz).le(rsum * rsum) & active).any() {
            return true;
        }
    }
    false
}

fn lengths() -> impl Iterator<Item = usize> {
    [0usize, 1, 7, MAX_LANES - 1, MAX_LANES, MAX_LANES + 1, 100].into_iter()
}

const QUERIES: [[f32; 4]; 3] = [
    [0.0, 0.0, 0.0, 3.0],
    [5.0, -5.0, 2.0, 1.0],
    [50.0, 50.0, 50.0, 0.1],
];

#[test]
fn count_matches_scalar() {
    for n in lengths() {
        let (xs, ys, zs, rs) = spheres(n);
        for q in QUERIES {
            let want = (0..n).filter(|&i| overlaps(&xs, &ys, &zs, &rs, q, i)).count();
            assert_eq!(count_k(&xs, &ys, &zs, &rs, q), want, "count n={n} q={q:?}");
        }
    }
}

#[test]
fn for_each_hit_matches_scalar() {
    for n in lengths() {
        let (xs, ys, zs, rs) = spheres(n);
        for q in QUERIES {
            let want: Vec<bool> = (0..n).map(|i| overlaps(&xs, &ys, &zs, &rs, q, i)).collect();
            let mut got = vec![false; n];
            let any = collect_k(&xs, &ys, &zs, &rs, q, &mut got);
            assert_eq!(got, want, "hits n={n} q={q:?}");
            assert_eq!(any, want.iter().any(|&b| b), "any n={n} q={q:?}");
        }
    }
}

#[test]
fn gather_n_matches_scalar() {
    for n in lengths() {
        let (xs, ys, zs, rs) = spheres(n);
        let pts: Vec<(f32, f32, f32, f32)> =
            (0..n).map(|i| (xs[i], ys[i], zs[i], rs[i])).collect();
        for q in QUERIES {
            let want = (0..n).any(|i| overlaps(&xs, &ys, &zs, &rs, q, i));
            assert_eq!(gather_any_k(&pts, q), want, "gather n={n} q={q:?}");
        }
    }
}

#[kernel]
fn load_roundtrip_k<'a>(ctx: Gang<f32>, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32]) -> f32 {
    let n = ctx.lanes();
    debug_assert!(xs.len() == n);
    let [vx, vy, vz] = ctx.load_n([xs, ys, zs]);
    let [px, py, pz] = ctx.load_partial_n([xs, ys, zs], 0.0);
    ((vx + px) * (vy + py) * (vz + pz)).reduce_sum()
}

#[test]
fn array_loaders_match_manual() {
    let n = f32::ZERO; // touch Scalar to keep the import meaningful when lanes vary
    let _ = n;
    let lanes = hydroplane::dispatch(LanesProbe);
    let xs: Vec<f32> = (0..lanes).map(|i| i as f32 + 1.0).collect();
    let ys: Vec<f32> = (0..lanes).map(|i| (i as f32) * 2.0).collect();
    let zs: Vec<f32> = (0..lanes).map(|i| (i as f32) - 1.0).collect();
    let want: f32 = (0..lanes)
        .map(|i| (xs[i] + xs[i]) * (ys[i] + ys[i]) * (zs[i] + zs[i]))
        .sum();
    let got = load_roundtrip_k(&xs, &ys, &zs);
    assert!((got - want).abs() <= want.abs() * 1e-4 + 1e-3, "got {got}, want {want}");
}

struct LanesProbe;
impl hydroplane::Kernel<f32> for LanesProbe {
    type Output = usize;
    fn run<S: hydroplane::Backend<f32>>(self, ctx: Gang<f32, S>) -> usize {
        ctx.lanes()
    }
}

#[kernel]
fn bitmask_k<'a>(ctx: Gang<f32>, xs: &'a [f32], thresh: f32) -> u32 {
    debug_assert!(xs.len() == ctx.lanes());
    ctx.load(xs).gt(ctx.splat(thresh)).to_bitmask()
}

#[test]
fn to_bitmask_matches_lane_predicate() {
    let lanes = hydroplane::dispatch(LanesProbe);
    // A few distinct set patterns across the active lanes; bit `i` must track lane `i`.
    for pat in [0b0u64, 0b1, 0b10, 0b1010, 0xFFFF_FFFF] {
        let xs: Vec<f32> = (0..lanes).map(|i| if (pat >> i) & 1 == 1 { 1.0 } else { -1.0 }).collect();
        let got = bitmask_k(&xs, 0.0);
        let want = (pat as u32) & ((1u64 << lanes) - 1) as u32;
        assert_eq!(got, want, "lanes={lanes} pat={pat:#b}");
    }
}

#[kernel]
fn rotate3_k<'a>(ctx: Gang<f32>, xs: &'a mut [f32], ys: &'a mut [f32], zs: &'a mut [f32], m: [f32; 9]) {
    ctx.map_n([xs, ys, zs], 0.0, |[x, y, z]| {
        [
            x * m[0] + y * m[1] + z * m[2],
            x * m[3] + y * m[4] + z * m[5],
            x * m[6] + y * m[7] + z * m[8],
        ]
    });
}

#[test]
fn map_n_matches_scalar_oracle() {
    let m = [0.36f32, -0.48, 0.80, 0.80, 0.60, 0.0, -0.48, 0.64, 0.60];
    for len in [0usize, 1, 3, 7, 8, 9, 31, 33, 64, 100, 257] {
        let base: Vec<[f32; 3]> = (0..len)
            .map(|i| {
                let f = i as f32;
                [(f * 0.3).sin() * 4.0, (f * 0.7).cos() * 4.0, (f * 0.13).sin() * 4.0]
            })
            .collect();
        let (mut xs, mut ys, mut zs): (Vec<f32>, Vec<f32>, Vec<f32>) =
            (base.iter().map(|p| p[0]).collect(), base.iter().map(|p| p[1]).collect(), base.iter().map(|p| p[2]).collect());
        rotate3_k(&mut xs, &mut ys, &mut zs, m);
        for (i, p) in base.iter().enumerate() {
            let want = [
                p[0] * m[0] + p[1] * m[1] + p[2] * m[2],
                p[0] * m[3] + p[1] * m[4] + p[2] * m[5],
                p[0] * m[6] + p[1] * m[7] + p[2] * m[8],
            ];
            for c in 0..3 {
                let got = [xs[i], ys[i], zs[i]][c];
                assert!((got - want[c]).abs() <= 1e-4, "len={len} i={i} c={c}: got {got}, want {}", want[c]);
            }
        }
    }
}
