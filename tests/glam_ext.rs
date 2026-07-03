//! The `glam`-aware wide-vector helpers agree with the equivalent scalar glam expression, across
//! lengths that cross the register boundary (including a short final tail), on whatever backend the
//! host dispatches to.
#![cfg(feature = "glam")]

use glam::{Mat3, Vec3};
use hydroplane::{Gang, GangGlamExt, kernel};

fn cols(n: usize, seed: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let f = |i: usize, k: f32| ((i as f32 + seed) * k).sin() * 4.0;
    (
        (0..n).map(|i| f(i, 0.31)).collect(),
        (0..n).map(|i| f(i, 0.53)).collect(),
        (0..n).map(|i| f(i, 0.71)).collect(),
    )
}

const LENS: [usize; 7] = [0, 1, 7, 8, 9, 31, 100];

#[kernel]
fn dist_sq_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    q: [f32; 3],
    out: &'a mut [f32],
) {
    let qv = ctx.splat_vec3(Vec3::from_array(q));
    ctx.for_each_chunk::<f32>(out.len(), |off, cnt| {
        let r = off..off + cnt;
        let p = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r.clone()]], 0.0);
        (qv - p).length_squared().store_partial(&mut out[off..off + cnt]);
    });
}

#[test]
fn vec3wide_dist_sq_matches_glam() {
    let q = Vec3::new(0.5, -1.5, 2.0);
    for len in LENS {
        let (xs, ys, zs) = cols(len, 1.0);
        let mut out = vec![0.0f32; len];
        dist_sq_k(&xs, &ys, &zs, q.to_array(), &mut out);
        for i in 0..len {
            let want = (q - Vec3::new(xs[i], ys[i], zs[i])).length_squared();
            assert!((out[i] - want).abs() <= 1e-3 * want.max(1.0), "len={len} i={i}");
        }
    }
}

#[kernel]
fn transform_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    m: [f32; 9],
    t: [f32; 3],
    ox: &'a mut [f32],
    oy: &'a mut [f32],
    oz: &'a mut [f32],
) {
    let mw = ctx.splat_mat3(Mat3::from_cols_array(&m));
    let tw = ctx.splat_vec3(Vec3::from_array(t));
    ctx.for_each_chunk::<f32>(ox.len(), |off, cnt| {
        let r = off..off + cnt;
        let v = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r.clone()]], 0.0);
        mw.mul_add(v, tw)
            .store_partial([&mut ox[r.clone()], &mut oy[r.clone()], &mut oz[r]]);
    });
}

#[test]
fn mat3wide_transform_matches_glam() {
    let m = Mat3::from_cols(
        Vec3::new(0.36, -0.48, 0.80),
        Vec3::new(0.80, 0.60, 0.0),
        Vec3::new(-0.48, 0.64, 0.60),
    );
    let t = Vec3::new(1.0, -2.0, 0.5);
    for len in LENS {
        let (xs, ys, zs) = cols(len, 2.0);
        let (mut ox, mut oy, mut oz) = (vec![0.0; len], vec![0.0; len], vec![0.0; len]);
        transform_k(&xs, &ys, &zs, m.to_cols_array(), t.to_array(), &mut ox, &mut oy, &mut oz);
        for i in 0..len {
            let want = m.mul_vec3(Vec3::new(xs[i], ys[i], zs[i])) + t;
            let got = Vec3::new(ox[i], oy[i], oz[i]);
            assert!((got - want).length() <= 1e-3 * want.length().max(1.0), "len={len} i={i}");
        }
    }
}

#[kernel]
fn max_proj_k<'a>(ctx: Gang, verts: &'a [Vec3], normal: [f32; 3]) -> f32 {
    let n = ctx.splat_vec3(Vec3::from_array(normal));
    let neg_inf = ctx.splat(f32::NEG_INFINITY);
    let mut acc = neg_inf;
    for (off, cnt, active) in ctx.masked_chunks::<f32>(verts.len()) {
        let v = ctx.gather_vec3(&verts[off..off + cnt], 0.0);
        acc = acc.max(v.dot(n).select(active, neg_inf));
    }
    acc.reduce_max()
}

#[test]
fn gather_vec3_max_projection_matches_glam() {
    let normal = Vec3::new(0.267, 0.535, 0.802);
    for len in [1usize, 5, 8, 13, 40] {
        let verts: Vec<Vec3> = (0..len)
            .map(|i| {
                let f = i as f32;
                Vec3::new((f * 0.3).sin() * 3.0, (f * 0.7).cos() * 3.0, (f * 0.13).sin() * 3.0)
            })
            .collect();
        let got = max_proj_k(&verts, normal.to_array());
        let want = verts.iter().map(|v| v.dot(normal)).fold(f32::NEG_INFINITY, f32::max);
        assert!((got - want).abs() <= 1e-3, "len={len}: got {got}, want {want}");
    }
}

#[kernel]
fn separated_k<'a>(ctx: Gang, planes: &'a [(Vec3, f32)], p: [f32; 3]) -> bool {
    let pw = ctx.splat_vec3(Vec3::from_array(p));
    let zero = ctx.splat(0.0);
    for (off, cnt, active) in ctx.masked_chunks::<f32>(planes.len()) {
        let (n, d) = ctx.gather_plane(&planes[off..off + cnt], 0.0);
        if ((pw.dot(n) - d).gt(zero) & active).any() {
            return true;
        }
    }
    false
}

#[test]
fn gather_plane_separation_matches_glam() {
    let planes: Vec<(Vec3, f32)> = (0..20)
        .map(|i| {
            let a = i as f32 * 0.31;
            (Vec3::new(a.cos(), a.sin(), (a * 0.5).sin()).normalize(), (i as f32 % 4.0) - 1.0)
        })
        .collect();
    for &p in &[Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0), Vec3::new(-1.0, 3.0, 0.5)] {
        let got = separated_k(&planes, p.to_array());
        let want = planes.iter().any(|(n, d)| p.dot(*n) - d > 0.0);
        assert_eq!(got, want, "p={p:?}");
    }
}

#[kernel]
fn add_scaled_len_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    dxs: &'a [f32],
    dys: &'a [f32],
    dzs: &'a [f32],
    t: f32,
    out: &'a mut [f32],
) {
    let tv = ctx.splat(t);
    ctx.for_each_chunk::<f32>(out.len(), |off, cnt| {
        let r = off..off + cnt;
        let p = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r.clone()]], 0.0);
        let d = ctx.load_partial_vec3([&dxs[r.clone()], &dys[r.clone()], &dzs[r.clone()]], 0.0);
        p.add_scaled(d, tv).length().store_partial(&mut out[off..off + cnt]);
    });
}

#[test]
fn vec3wide_add_scaled_length_matches_glam() {
    let t = 0.375f32;
    for len in LENS {
        let (xs, ys, zs) = cols(len, 3.0);
        let (dxs, dys, dzs) = cols(len, 9.0);
        let mut out = vec![0.0f32; len];
        add_scaled_len_k(&xs, &ys, &zs, &dxs, &dys, &dzs, t, &mut out);
        for i in 0..len {
            let p = Vec3::new(xs[i], ys[i], zs[i]);
            let d = Vec3::new(dxs[i], dys[i], dzs[i]);
            let want = (p + d * t).length();
            assert!((out[i] - want).abs() <= 1e-3 * want.max(1.0), "len={len} i={i}");
        }
    }
}

#[kernel]
fn select_scale_k<'a>(
    ctx: Gang,
    xs: &'a [f32],
    ys: &'a [f32],
    zs: &'a [f32],
    thresh: f32,
    out: &'a mut [f32],
) {
    let tv = ctx.splat(thresh);
    ctx.for_each_chunk::<f32>(out.len(), |off, cnt| {
        let r = off..off + cnt;
        let v = ctx.load_partial_vec3([&xs[r.clone()], &ys[r.clone()], &zs[r.clone()]], 0.0);
        let mask = v.0[0].gt(tv);
        // keep `v` where x > thresh, else halve it
        v.select(mask, v * 0.5f32).length_squared().store_partial(&mut out[off..off + cnt]);
    });
}

#[test]
fn vec3wide_select_and_scalar_mul_matches_glam() {
    let thresh = 0.0f32;
    for len in LENS {
        let (xs, ys, zs) = cols(len, 4.0);
        let mut out = vec![0.0f32; len];
        select_scale_k(&xs, &ys, &zs, thresh, &mut out);
        for i in 0..len {
            let v = Vec3::new(xs[i], ys[i], zs[i]);
            let chosen = if xs[i] > thresh { v } else { v * 0.5 };
            let want = chosen.length_squared();
            assert!((out[i] - want).abs() <= 1e-3 * want.max(1.0), "len={len} i={i}");
        }
    }
}

#[kernel]
fn inverse_k<'a>(ctx: Gang, m: [&'a [f32]; 9], out: [&'a mut [f32]; 9]) {
    let n = m[0].len();
    let mut out = out;
    ctx.for_each_chunk::<f32>(n, |off, cnt| {
        let cols: [&[f32]; 9] = std::array::from_fn(|c| &m[c][off..off + cnt]);
        ctx.load_partial_mat3(cols, 1.0)
            .inverse()
            .store_partial(out.each_mut().map(|o| &mut o[off..off + cnt]));
    });
}

#[test]
fn mat3wide_inverse_matches_glam() {
    for len in LENS {
        let m: [Vec<f32>; 9] = std::array::from_fn(|c| {
            (0..len)
                .map(|i| {
                    let base = ((i as f32 + c as f32) * 0.17).sin() * 0.5;
                    if c % 4 == 0 { base + 4.0 } else { base }
                })
                .collect()
        });
        let mref: [&[f32]; 9] = std::array::from_fn(|c| m[c].as_slice());
        let mut out: [Vec<f32>; 9] = std::array::from_fn(|_| vec![0.0; len]);
        {
            let mut it = out.iter_mut();
            let oref: [&mut [f32]; 9] = std::array::from_fn(|_| it.next().unwrap().as_mut_slice());
            inverse_k(mref, oref);
        }
        for i in 0..len {
            let cols: [f32; 9] = std::array::from_fn(|c| m[c][i]);
            let want = Mat3::from_cols_array(&cols).inverse().to_cols_array();
            for c in 0..9 {
                assert!(
                    (out[c][i] - want[c]).abs() <= 1e-3 * want[c].abs().max(1.0),
                    "len={len} i={i} c={c}"
                );
            }
        }
    }
}

use hydroplane::{Backend, BackendAll, Kernel, MatWide, dispatch};

fn plane(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.37).sin() * 3.0).collect()
}

struct BatMul<'a, const R: usize, const K: usize, const N: usize> {
    a: &'a [Vec<f32>],
    b: &'a [Vec<f32>],
    out: &'a mut [Vec<f32>],
}

impl<'a, const R: usize, const K: usize, const N: usize> Kernel<f32> for BatMul<'a, R, K, N> {
    type Output = ();
    fn run<S: BackendAll + Backend<f32>>(self, ctx: Gang<S>) {
        let BatMul { a, b, out } = self;
        let n = out[0].len();
        ctx.for_each_chunk::<f32>(n, |off, cnt| {
            let r = off..off + cnt;
            let am: MatWide<S, R, K> =
                ctx.load_partial_mat(core::array::from_fn(|i| core::array::from_fn(|j| &a[i * K + j][r.clone()])), 0.0);
            let bm: MatWide<S, K, N> =
                ctx.load_partial_mat(core::array::from_fn(|i| core::array::from_fn(|j| &b[i * N + j][r.clone()])), 0.0);
            let rows = am.matmul(bm).rows();
            for i in 0..R {
                for j in 0..N {
                    rows[i][j].store_partial(&mut out[i * N + j][r.clone()]);
                }
            }
        });
    }
}

fn check_batmul<const R: usize, const K: usize, const N: usize>() {
    for &len in &LENS {
        let a: Vec<Vec<f32>> = (0..R * K).map(|p| plane(len, p as f32 + 0.2)).collect();
        let b: Vec<Vec<f32>> = (0..K * N).map(|p| plane(len, p as f32 + 5.0)).collect();
        let mut out: Vec<Vec<f32>> = (0..R * N).map(|_| vec![0.0; len]).collect();
        dispatch::<f32, _>(BatMul::<R, K, N> { a: &a, b: &b, out: &mut out });
        for idx in 0..len {
            for i in 0..R {
                for j in 0..N {
                    let want: f32 = (0..K).map(|k| a[i * K + k][idx] * b[k * N + j][idx]).sum();
                    let got = out[i * N + j][idx];
                    assert!((got - want).abs() <= 1e-4 * (1.0 + want.abs()), "batmul {R}x{K}x{N} len{len} [{i}][{j}]@{idx}: {got} vs {want}");
                }
            }
        }
    }
}

#[test]
fn matwide_batched_gemm_matches_scalar() {
    check_batmul::<2, 3, 4>();
    check_batmul::<4, 4, 4>();
    check_batmul::<6, 6, 1>();
    check_batmul::<3, 3, 3>();
}
