//! Escape-time Mandelbrot — the most complex workload: a data-dependent iteration count per element,
//! driven by a per-lane active mask with early exit when a whole register has escaped. The classic
//! SPMD showcase (divergent control flow expressed as masks), and the sharpest test that hydroplane's
//! mask algebra matches a hand-rolled `wide` blend loop.

use hydroplane::{Gang, kernel};
use wide::f32x8;

use crate::ramp;

pub const MAX_ITER: u32 = 100;

pub fn inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    // A diagonal sweep across the interesting region of the complex plane.
    let cx = (0..n).map(|i| -2.0 + 3.0 * (i as f32 / n as f32)).collect();
    let cy = ramp(n, 5.0, 1.2);
    (cx, cy)
}

#[kernel]
pub fn mandelbrot_hp<'a>(ctx: Gang, cx: &'a [f32], cy: &'a [f32], max_iter: u32, out: &'a mut [f32]) {
    let (zero, one, four) = (ctx.splat(0.0), ctx.splat(1.0), ctx.splat(4.0));
    // Elementwise `(cx, cy) -> escape count`; `zip_map` handles the chunking and masked tail, and the
    // closure is the escape-time iteration itself — the active mask and early-exit stay per lane.
    ctx.zip_map(cx, cy, out, 0.0, 0.0, |cxv, cyv| {
        let (mut zx, mut zy, mut count) = (zero, zero, zero);
        for _ in 0..max_iter {
            let zx2 = zx * zx;
            let zy2 = zy * zy;
            let active = (zx2 + zy2).le(four);
            if !active.any() {
                break;
            }
            count = count + one.select(active, zero);
            let nzx = zx2 - zy2 + cxv;
            zy = (zx * zy) * 2.0 + cyv;
            zx = nzx;
        }
        count
    });
}

pub fn mandelbrot_wide(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    let n = cx.len();
    let four = f32x8::splat(4.0);
    let one = f32x8::splat(1.0);
    let zero = f32x8::splat(0.0);
    let mut off = 0;
    while off < n {
        let cnt = 8.min(n - off);
        let (mut bx, mut by) = ([0.0f32; 8], [0.0f32; 8]);
        bx[..cnt].copy_from_slice(&cx[off..off + cnt]);
        by[..cnt].copy_from_slice(&cy[off..off + cnt]);
        let cxv = f32x8::from(bx);
        let cyv = f32x8::from(by);
        let mut zx = zero;
        let mut zy = zero;
        let mut count = zero;
        for _ in 0..max_iter {
            let zx2 = zx * zx;
            let zy2 = zy * zy;
            let active = (zx2 + zy2).simd_le(four);
            if active.none() {
                break;
            }
            count += active.blend(one, zero);
            let nzx = zx2 - zy2 + cxv;
            zy = (zx * zy) * 2.0 + cyv;
            zx = nzx;
        }
        out[off..off + cnt].copy_from_slice(&count.to_array()[..cnt]);
        off += 8;
    }
}

pub fn mandelbrot_scalar(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    for i in 0..cx.len() {
        let (cxi, cyi) = (cx[i], cy[i]);
        let (mut zx, mut zy) = (0.0f32, 0.0f32);
        let mut count = 0.0f32;
        for _ in 0..max_iter {
            let (zx2, zy2) = (zx * zx, zy * zy);
            if zx2 + zy2 > 4.0 {
                break;
            }
            count += 1.0;
            let nzx = zx2 - zy2 + cxi;
            zy = (zx * zy) * 2.0 + cyi;
            zx = nzx;
        }
        out[i] = count;
    }
}
