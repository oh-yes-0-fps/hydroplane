//! Batched matrix–vector transform `M·v`, one distinct 3×3 matrix and one vec3 per element. Nine
//! matrix loads + three vector loads feed nine FMAs producing three outputs — denser arithmetic than
//! normalize, the natural fit for hydroplane's `Mat3Wide`, mirrored by hand-rolled `wide` and `glam`.

use glam::{Mat3, Vec3};
use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub fn inputs(n: usize) -> ([Vec<f32>; 9], [Vec<f32>; 3]) {
    let m = std::array::from_fn(|c| ramp(n, c as f32 + 1.0, 1.0));
    let v = std::array::from_fn(|c| ramp(n, c as f32 + 12.0, 2.0));
    (m, v)
}

#[kernel]
pub fn transform_hp<'a>(
    ctx: Gang<f32>,
    m: [&'a [f32]; 9],
    vx: &'a [f32],
    vy: &'a [f32],
    vz: &'a [f32],
    ox: &'a mut [f32],
    oy: &'a mut [f32],
    oz: &'a mut [f32],
) {
    // Twelve input columns (nine column-major matrix components + the vector's x/y/z) into three
    // outputs — `M·v` per lane, matching `Mat3Wide::mul_vec3`. `map_cols` drives the full-register
    // pass, the masked tail, and the ILP; the closure is just the math.
    ctx.map_cols::<12, 3>(
        [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], vx, vy, vz],
        [ox, oy, oz],
        0.0,
        |[m0, m1, m2, m3, m4, m5, m6, m7, m8, x, y, z]| {
            [
                m0 * x + m3 * y + m6 * z,
                m1 * x + m4 * y + m7 * z,
                m2 * x + m5 * y + m8 * z,
            ]
        },
    );
}

pub fn transform_wide(
    m: &[Vec<f32>; 9],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    ox: &mut [f32],
    oy: &mut [f32],
    oz: &mut [f32],
) {
    let n = vx.len();
    let mut off = 0;
    while off + 8 <= n {
        let r: [f32x8; 9] =
            std::array::from_fn(|c| f32x8::from(<[f32; 8]>::try_from(&m[c][off..off + 8]).unwrap()));
        let x = f32x8::from(<[f32; 8]>::try_from(&vx[off..off + 8]).unwrap());
        let y = f32x8::from(<[f32; 8]>::try_from(&vy[off..off + 8]).unwrap());
        let z = f32x8::from(<[f32; 8]>::try_from(&vz[off..off + 8]).unwrap());
        ox[off..off + 8].copy_from_slice(&(r[0] * x + r[3] * y + r[6] * z).to_array());
        oy[off..off + 8].copy_from_slice(&(r[1] * x + r[4] * y + r[7] * z).to_array());
        oz[off..off + 8].copy_from_slice(&(r[2] * x + r[5] * y + r[8] * z).to_array());
        off += 8;
    }
    while off < n {
        transform_one(m, vx, vy, vz, ox, oy, oz, off);
        off += 1;
    }
}

pub fn transform_scalar(
    m: &[Vec<f32>; 9],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    ox: &mut [f32],
    oy: &mut [f32],
    oz: &mut [f32],
) {
    for i in 0..vx.len() {
        transform_one(m, vx, vy, vz, ox, oy, oz, i);
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn transform_one(
    m: &[Vec<f32>; 9],
    vx: &[f32],
    vy: &[f32],
    vz: &[f32],
    ox: &mut [f32],
    oy: &mut [f32],
    oz: &mut [f32],
    i: usize,
) {
    let cols: [f32; 9] = std::array::from_fn(|c| m[c][i]);
    let out = Mat3::from_cols_array(&cols) * Vec3::new(vx[i], vy[i], vz[i]);
    ox[i] = out.x;
    oy[i] = out.y;
    oz[i] = out.z;
}
