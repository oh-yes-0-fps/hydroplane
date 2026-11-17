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
fn count_k<'a>(ctx: Gang, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32], rs: &'a [f32], q: [f32; 4]) -> usize {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    ctx.count_n([xs, ys, zs, rs], |[x, y, z, r]| {
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        (dx * dx + dy * dy + dz * dz).le(rsum * rsum)
    })
}

#[kernel]
fn collect_k<'a>(
    ctx: Gang,
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
/// broadphase-reject-then-narrowphase shape.
#[kernel]
fn gather_any_k<'a>(ctx: Gang, pts: &'a [(f32, f32, f32, f32)], q: [f32; 4]) -> bool {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    for (off, cnt, active) in ctx.masked_chunks::<f32>(pts.len()) {
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

/// Overlap count from `[x, y, z, r]` rows via `gather_rows`, NaN-filling the radius so a short
/// tail self-rejects.
#[kernel]
fn gather_rows_count_k<'a>(ctx: Gang, pts: &'a [[f32; 4]], q: [f32; 4]) -> usize {
    let [cx, cy, cz, sr] = ctx.splat_n(q);
    let (one, zero) = (ctx.splat(1.0), ctx.splat(0.0));
    let mut acc = zero;
    ctx.for_each_chunk::<f32>(pts.len(), |off, cnt| {
        let [x, y, z, r] = ctx.gather_rows(&pts[off..off + cnt], [0.0, 0.0, 0.0, f32::NAN]);
        let (dx, dy, dz) = (cx - x, cy - y, cz - z);
        let rsum = sr + r;
        acc = acc + one.select((dx * dx + dy * dy + dz * dz).le(rsum * rsum), zero);
    });
    acc.reduce_sum() as usize
}

#[test]
fn gather_rows_matches_scalar() {
    for n in lengths() {
        let (xs, ys, zs, rs) = spheres(n);
        let pts: Vec<[f32; 4]> = (0..n).map(|i| [xs[i], ys[i], zs[i], rs[i]]).collect();
        for q in QUERIES {
            let want = (0..n).filter(|&i| overlaps(&xs, &ys, &zs, &rs, q, i)).count();
            assert_eq!(gather_rows_count_k(&pts, q), want, "gather_rows n={n} q={q:?}");
        }
    }
}

#[kernel]
fn load_roundtrip_k<'a>(ctx: Gang, xs: &'a [f32], ys: &'a [f32], zs: &'a [f32]) -> f32 {
    let n = ctx.lanes::<f32>();
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
    fn run<S: hydroplane::Backend<f32>>(self, ctx: Gang<S>) -> usize {
        ctx.lanes::<f32>()
    }
}

/// Lane `i` is `1.0` where bit `i` of `pat` is set; returns the lane count with the bitmask, since
/// a kernel touching integers may run on a different backend than a plain `f32` dispatch.
#[kernel]
fn bitmask_k(ctx: Gang, pat: u64) -> (usize, u32) {
    let n = ctx.lanes::<f32>();
    let mut xs = [0.0f32; MAX_LANES];
    for (i, x) in xs[..n].iter_mut().enumerate() {
        *x = if (pat >> i) & 1 == 1 { 1.0 } else { -1.0 };
    }
    (n, ctx.load(&xs[..n]).gt(ctx.splat(0.0)).to_bitmask())
}

#[test]
fn to_bitmask_matches_lane_predicate() {
    // A few distinct set patterns across the active lanes; bit `i` must track lane `i`.
    for pat in [0b0u64, 0b1, 0b10, 0b1010, 0xFFFF_FFFF] {
        let (lanes, got) = bitmask_k(pat);
        let want = (pat as u32) & ((1u64 << lanes) - 1) as u32;
        assert_eq!(got, want, "lanes={lanes} pat={pat:#b}");
    }
}

#[kernel]
fn rotate3_k<'a>(ctx: Gang, xs: &'a mut [f32], ys: &'a mut [f32], zs: &'a mut [f32], m: [f32; 9]) {
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

#[kernel]
fn shifted_sum<'a>(ctx: Gang, x: &'a [f32]) -> f32 {
    ctx.sum(x, |acc, v| acc + (v + ctx.splat(1.0)))
}

#[kernel]
fn shifted_zip_sum<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> f32 {
    ctx.zip_sum(x, y, |acc, a, b| acc + (a * b + ctx.splat(1.0)))
}

#[test]
fn sums_count_only_real_elements_when_step_of_zero_is_not_zero() {
    for n in [0, 1, 3, 7, 13, 31, MAX_LANES + 5, 8 * MAX_LANES + 3, 1000] {
        let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let want: f32 = x.iter().sum::<f32>() + n as f32;
        assert_eq!(shifted_sum(&x), want, "sum, n = {n}");
        let want: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f32>() + n as f32;
        assert_eq!(shifted_zip_sum(&x, &y), want, "zip_sum, n = {n}");
    }
}

#[kernel]
fn moments<'a>(ctx: Gang, x: &'a [f32]) -> [f32; 3] {
    let one = ctx.splat(1.0);
    ctx.sum_n(x, |[c, s, q], v| [c + one, s + v, q + v * v])
}

#[kernel]
fn cosine_terms<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> [f32; 3] {
    ctx.zip_sum_n(x, y, |[xy, xx, yy], a, b| [xy + a * b, xx + a * a, yy + b * b])
}

#[test]
fn multi_sums_match_separate_sums() {
    for n in [0, 1, 3, 7, 13, 31, MAX_LANES + 5, 8 * MAX_LANES + 3, 1000] {
        let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let s: f32 = x.iter().sum();
        let q: f32 = x.iter().map(|v| v * v).sum();
        assert_eq!(moments(&x), [n as f32, s, q], "sum_n, n = {n}");
        let xy: f32 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let yy: f32 = y.iter().map(|v| v * v).sum();
        assert_eq!(cosine_terms(&x, &y), [xy, q, yy], "zip_sum_n, n = {n}");
    }
}

#[kernel]
fn offset_reductions<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> [f32; 6] {
    let one = ctx.splat(1.0);
    let [c, s] = ctx.sum_n(x, |[c, s], v| [c + one, s + v]);
    let [xy, yy] = ctx.zip_sum_n(x, y, |[xy, yy], a, b| [xy + a * b, yy + b * b]);
    let shifted = ctx.zip_sum(x, y, |acc, a, b| acc + (a + b + one));
    let hits = ctx.count_n([x, y], |[a, b]| (a + b).ge(ctx.splat(3.0))) as f32;
    [c, s, xy, yy, shifted, hits]
}

#[kernel]
fn offset_maps<'a>(ctx: Gang, x: &'a [f32], y: &'a mut [f32], out: &'a mut [f32]) {
    ctx.zip_map(x, y, out, 0.0, 0.0, |a, b| a * b + a);
    ctx.map(x, y, 0.0, |a| a + a);
    ctx.zip_map_inplace(x, y, 0.0, 0.0, |a, b| a + b);
}

#[test]
fn reductions_and_maps_agree_at_every_start_offset() {
    let total = 1000 + 2 * MAX_LANES;
    let xs: Vec<f32> = (0..total).map(|i| (i % 5) as f32).collect();
    let ys: Vec<f32> = (0..total).map(|i| (i % 3) as f32).collect();
    for n in [8 * MAX_LANES + 3, 1000] {
        for ox in 0..MAX_LANES {
            for oy in [ox, (ox + 1) % MAX_LANES] {
                let (x, y) = (&xs[ox..ox + n], &ys[oy..oy + n]);
                let s: f32 = x.iter().sum();
                let xy: f32 = x.iter().zip(y).map(|(a, b)| a * b).sum();
                let yy: f32 = y.iter().map(|b| b * b).sum();
                let shifted: f32 = x.iter().zip(y).map(|(a, b)| a + b + 1.0).sum();
                let hits = x.iter().zip(y).filter(|(a, b)| *a + *b >= 3.0).count() as f32;
                let what = format!("n = {n}, x at {ox}, y at {oy}");
                assert_eq!(offset_reductions(x, y), [n as f32, s, xy, yy, shifted, hits], "{what}");

                let mut ybuf = ys.clone();
                let mut obuf = vec![0.0f32; total];
                offset_maps(x, &mut ybuf[oy..oy + n], &mut obuf[ox..ox + n]);
                for i in 0..n {
                    assert_eq!(obuf[ox + i], x[i] * y[i] + x[i], "zip_map {what}, i = {i}");
                    assert_eq!(ybuf[oy + i], 3.0 * x[i], "map then zip_map_inplace {what}, i = {i}");
                }
                assert!(obuf[..ox].iter().chain(&obuf[ox + n..]).all(|&v| v == 0.0), "zip_map wrote outside, {what}");
            }
        }
    }
}

#[kernel]
fn any_all_with_bad_fills<'a>(ctx: Gang, x: &'a [f32], y: &'a [f32]) -> [bool; 4] {
    let lim = ctx.splat(100.0f32);
    [
        ctx.any(x, 1.0e9, |v| v.gt(lim)),
        ctx.all(x, 1.0e9, |v| v.lt(lim)),
        ctx.zip_any(x, y, 1.0e9, 1.0e9, |a, b| (a + b).gt(lim)),
        ctx.zip_all(x, y, 1.0e9, 1.0e9, |a, b| (a + b).lt(lim)),
    ]
}

#[test]
fn any_and_all_ignore_their_padding() {
    for n in [0, 1, 3, 7, 13, 31, MAX_LANES + 5, 1000] {
        let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        assert_eq!(any_all_with_bad_fills(&x, &y), [false, true, false, true], "n = {n}");
    }
}

#[kernel]
fn double_and_dot<'a, T: hydroplane::FloatScalar>(ctx: Gang, x: &'a [T], out: &'a mut [T]) -> T {
    let two = ctx.splat(T::from_f64(2.0));
    ctx.map(x, out, T::ZERO, |v| v * two);
    ctx.dot(x, x)
}

#[test]
fn every_tail_length_loads_and_stores_exactly_its_elements() {
    for n in 0..=3 * MAX_LANES + 1 {
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + 1.0).collect();
        let mut out = vec![-1.0f32; n + 4];
        let d = double_and_dot(&x, &mut out[..n]);
        assert_eq!(d, x.iter().map(|v| v * v).sum::<f32>(), "f32 dot, n = {n}");
        assert!(out[..n].iter().zip(&x).all(|(o, v)| *o == 2.0 * v), "f32 map, n = {n}");
        assert!(out[n..].iter().all(|&o| o == -1.0), "f32 map wrote past the end, n = {n}");

        let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let mut out = vec![-1.0f64; n + 4];
        let d = double_and_dot(&x, &mut out[..n]);
        assert_eq!(d, x.iter().map(|v| v * v).sum::<f64>(), "f64 dot, n = {n}");
        assert!(out[..n].iter().zip(&x).all(|(o, v)| *o == 2.0 * v), "f64 map, n = {n}");
        assert!(out[n..].iter().all(|&o| o == -1.0), "f64 map wrote past the end, n = {n}");
    }
}

type Extremes = (Option<f32>, Option<f32>, Option<(usize, f32)>, Option<(usize, f32)>);

#[kernel]
fn extremes<'a>(ctx: Gang, x: &'a [f32]) -> Extremes {
    (ctx.max_of(x), ctx.min_of(x), ctx.argmin(x), ctx.argmax(x))
}

#[test]
fn extremes_match_iterators_and_take_the_first_index_on_ties() {
    for n in [0, 1, 2, 3, 7, 13, 31, MAX_LANES, MAX_LANES + 5, 8 * MAX_LANES + 3, 1000] {
        for seed in 0..4u32 {
            let x: Vec<f32> = (0..n as u32).map(|i| ((i.wrapping_mul(2654435761) ^ seed) % 97) as f32 - 40.0).collect();
            let max = x.iter().copied().reduce(f32::max);
            let min = x.iter().copied().reduce(f32::min);
            let argmin = min.map(|m| (x.iter().position(|&v| v == m).unwrap(), m));
            let argmax = max.map(|m| (x.iter().position(|&v| v == m).unwrap(), m));
            assert_eq!(extremes(&x), (max, min, argmin, argmax), "n = {n}, seed = {seed}");
        }
    }
}

#[kernel]
fn eq_ne_counts<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> (u32, u32) {
    let lanes = ctx.lanes::<f32>();
    let (mut eq, mut ne) = (0, 0);
    for off in (0..a.len()).step_by(lanes) {
        let (x, y) = (ctx.load(&a[off..off + lanes]), ctx.load(&b[off..off + lanes]));
        eq += x.eq(y).to_bitmask().count_ones();
        ne += x.ne(y).to_bitmask().count_ones();
    }
    (eq, ne)
}

#[kernel]
fn eq_ne_counts_f64<'a>(ctx: Gang, a: &'a [f64], b: &'a [f64]) -> (u32, u32) {
    let lanes = ctx.lanes::<f64>();
    let (mut eq, mut ne) = (0, 0);
    for off in (0..a.len()).step_by(lanes) {
        let (x, y) = (ctx.load(&a[off..off + lanes]), ctx.load(&b[off..off + lanes]));
        eq += x.eq(y).to_bitmask().count_ones();
        ne += x.ne(y).to_bitmask().count_ones();
    }
    (eq, ne)
}

#[test]
fn float_equality_follows_ieee() {
    let pairs = [(1.0, 1.0), (1.0, 2.0), (f64::NAN, f64::NAN), (f64::NAN, 1.0), (0.0, -0.0), (f64::INFINITY, f64::INFINITY), (-3.5, -3.5), (5.0, f64::NAN)];
    let n = 4 * MAX_LANES;
    let a: Vec<f64> = (0..n).map(|i| pairs[i % pairs.len()].0).collect();
    let b: Vec<f64> = (0..n).map(|i| pairs[i % pairs.len()].1).collect();
    let want_eq = a.iter().zip(&b).filter(|(x, y)| x == y).count() as u32;
    let want_ne = a.iter().zip(&b).filter(|(x, y)| x != y).count() as u32;
    assert_eq!(eq_ne_counts_f64(&a, &b), (want_eq, want_ne));
    let (a, b): (Vec<f32>, Vec<f32>) = (a.iter().map(|&v| v as f32).collect(), b.iter().map(|&v| v as f32).collect());
    assert_eq!(eq_ne_counts(&a, &b), (want_eq, want_ne));
}
