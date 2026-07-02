//! Batched vector normalization `v / ‖v‖` over an SoA of 3-component vectors — a rsqrt plus three
//! scales. Moderate arithmetic intensity across three lanes of storage; the natural fit for
//! hydroplane's `Vec3Wide`, mirrored by a hand-rolled three-register `wide` kernel and `glam`.

use glam::Vec3;
use hydroplane::{Gang, GangGlamExt, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> [Vec<f32>; 3] {
    [ramp(n, 1.0, 3.0), ramp(n, 4.0, 3.0), ramp(n, 8.0, 3.0)]
}

#[kernel]
pub fn normalize_hp<'a>(
    ctx: Gang<f32>,
    x: &'a [f32],
    y: &'a [f32],
    z: &'a [f32],
    ox: &'a mut [f32],
    oy: &'a mut [f32],
    oz: &'a mut [f32],
) {
    let n = x.len();
    let lanes = ctx.lanes();
    let mut off = 0;
    while off + lanes <= n {
        let r = off..off + lanes;
        let v = ctx.load_vec3([&x[r.clone()], &y[r.clone()], &z[r.clone()]]);
        let inv = v.length().recip();
        (v * inv).store([&mut ox[r.clone()], &mut oy[r.clone()], &mut oz[r]]);
        off += lanes;
    }
    if off < n {
        let r = off..n;
        let v = ctx.load_partial_vec3([&x[r.clone()], &y[r.clone()], &z[r.clone()]], 1.0);
        let inv = v.length().recip();
        (v * inv).store_partial([&mut ox[r.clone()], &mut oy[r.clone()], &mut oz[r]]);
    }
}

pub fn normalize_wide(x: &[f32], y: &[f32], z: &[f32], ox: &mut [f32], oy: &mut [f32], oz: &mut [f32]) {
    let n = x.len();
    let one = f32x8::splat(1.0);
    let mut off = 0;
    while off + 8 <= n {
        let xv = f32x8::from(<[f32; 8]>::try_from(&x[off..off + 8]).unwrap());
        let yv = f32x8::from(<[f32; 8]>::try_from(&y[off..off + 8]).unwrap());
        let zv = f32x8::from(<[f32; 8]>::try_from(&z[off..off + 8]).unwrap());
        let inv = one / (xv * xv + yv * yv + zv * zv).sqrt();
        ox[off..off + 8].copy_from_slice(&(xv * inv).to_array());
        oy[off..off + 8].copy_from_slice(&(yv * inv).to_array());
        oz[off..off + 8].copy_from_slice(&(zv * inv).to_array());
        off += 8;
    }
    while off < n {
        let v = Vec3::new(x[off], y[off], z[off]).normalize();
        ox[off] = v.x;
        oy[off] = v.y;
        oz[off] = v.z;
        off += 1;
    }
}

pub fn normalize_scalar(x: &[f32], y: &[f32], z: &[f32], ox: &mut [f32], oy: &mut [f32], oz: &mut [f32]) {
    for i in 0..x.len() {
        let v = Vec3::new(x[i], y[i], z[i]).normalize();
        ox[i] = v.x;
        oy[i] = v.y;
        oz[i] = v.z;
    }
}
