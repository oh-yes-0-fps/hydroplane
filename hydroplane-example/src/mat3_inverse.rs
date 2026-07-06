//! Batched 3×3 inverse: register- and bandwidth-bound, stressing register pressure.

use glam::Mat3;
use hydroplane::{Gang, Mat3Wide, kernel};
use wide::f32x8;


pub fn inputs(n: usize) -> [Vec<f32>; 9] {
    std::array::from_fn(|c| {
        (0..n)
            .map(|i| {
                let base = ((i as f32 + c as f32) * 0.17).sin() * 0.5;
                if c % 4 == 0 { base + 4.0 } else { base }
            })
            .collect()
    })
}

#[kernel]
pub fn invert_hp<'a>(ctx: Gang, m: [&'a [f32]; 9], out: [&'a mut [f32]; 9]) {
    ctx.map_cols::<f32, 9, 9>(m, out, 1.0, |m| Mat3Wide::from(m).inverse().cols());
}

pub fn invert_wide(m: &[Vec<f32>; 9], out: &mut [Vec<f32>; 9]) {
    let n = m[0].len();
    let mut off = 0;
    while off < n {
        let cnt = 8.min(n - off);
        let mut r = [f32x8::splat(1.0); 9];
        if cnt == 8 {
            for c in 0..9 {
                r[c] = f32x8::from(<[f32; 8]>::try_from(&m[c][off..off + 8]).unwrap());
            }
        } else {
            for c in 0..9 {
                let mut b = [1.0f32; 8];
                b[..cnt].copy_from_slice(&m[c][off..off + cnt]);
                r[c] = f32x8::from(b);
            }
        }
        let t0x = r[4] * r[8] - r[5] * r[7];
        let t0y = r[5] * r[6] - r[3] * r[8];
        let t0z = r[3] * r[7] - r[4] * r[6];
        let t1x = r[7] * r[2] - r[8] * r[1];
        let t1y = r[8] * r[0] - r[6] * r[2];
        let t1z = r[6] * r[1] - r[7] * r[0];
        let t2x = r[1] * r[5] - r[2] * r[4];
        let t2y = r[2] * r[3] - r[0] * r[5];
        let t2z = r[0] * r[4] - r[1] * r[3];
        let id = f32x8::splat(1.0) / (r[6] * t2x + r[7] * t2y + r[8] * t2z);
        let o = [
            t0x * id, t1x * id, t2x * id,
            t0y * id, t1y * id, t2y * id,
            t0z * id, t1z * id, t2z * id,
        ];
        for c in 0..9 {
            out[c][off..off + cnt].copy_from_slice(&o[c].to_array()[..cnt]);
        }
        off += 8;
    }
}

pub fn invert_scalar(m: &[Vec<f32>; 9], out: &mut [Vec<f32>; 9]) {
    for i in 0..m[0].len() {
        let cols: [f32; 9] = std::array::from_fn(|c| m[c][i]);
        let inv = Mat3::from_cols_array(&cols).inverse().to_cols_array();
        for c in 0..9 {
            out[c][i] = inv[c];
        }
    }
}
