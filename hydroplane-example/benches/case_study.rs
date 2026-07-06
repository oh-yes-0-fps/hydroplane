//! The mandelbrot implementations from CASE_STUDY.md benched side by side.
//! `neon_ilp` (the raw-intrinsics honourable mention) only exists on aarch64.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use hydroplane_example::mandelbrot::{MAX_ITER, inputs, mandelbrot_hp, mandelbrot_scalar};
use hydroplane_example::max_rel_err;
use std::hint::black_box;
use wide::f32x4;

/// Stage 1: straightforward port to portable SIMD, one 4-lane block at a time.
pub fn mandelbrot_wide4(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    let n = cx.len();
    let four = f32x4::splat(4.0);
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    let mut off = 0;
    while off < n {
        let cnt = 4.min(n - off);
        let (mut bx, mut by) = ([0.0f32; 4], [0.0f32; 4]);
        bx[..cnt].copy_from_slice(&cx[off..off + cnt]);
        by[..cnt].copy_from_slice(&cy[off..off + cnt]);
        let cxv = f32x4::from(bx);
        let cyv = f32x4::from(by);
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
        off += 4;
    }
}

/// Stage 2: same `wide` ops restructured for ILP — four independent blocks in flight
/// (16 points/pass) so the iteration dependency chains overlap. K = 4 suits this machine;
/// the best K differs per core even within one architecture.
pub fn mandelbrot_wide_ilp(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    let n = cx.len();
    let four = f32x4::splat(4.0);
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    let load = |s: &[f32], o: usize| f32x4::from(<[f32; 4]>::try_from(&s[o..o + 4]).unwrap());
    let mut i = 0;
    while i + 16 <= n {
        macro_rules! block {
            ($cxv:ident, $cyv:ident, $zx:ident, $zy:ident, $count:ident, $j:literal) => {
                let $cxv = load(cx, i + 4 * $j);
                let $cyv = load(cy, i + 4 * $j);
                let (mut $zx, mut $zy, mut $count) = (zero, zero, zero);
            };
        }
        block!(cx0, cy0, zx0, zy0, n0, 0);
        block!(cx1, cy1, zx1, zy1, n1, 1);
        block!(cx2, cy2, zx2, zy2, n2, 2);
        block!(cx3, cy3, zx3, zy3, n3, 3);
        for _ in 0..max_iter {
            macro_rules! step {
                ($cxv:ident, $cyv:ident, $zx:ident, $zy:ident, $count:ident) => {{
                    let x2 = $zx * $zx;
                    let y2 = $zy * $zy;
                    let active = (x2 + y2).simd_le(four);
                    $count += active & one;
                    let nzx = x2 - y2 + $cxv;
                    $zy = ($zx * $zy).mul_add(f32x4::splat(2.0), $cyv);
                    $zx = nzx;
                    active
                }};
            }
            let a0 = step!(cx0, cy0, zx0, zy0, n0);
            let a1 = step!(cx1, cy1, zx1, zy1, n1);
            let a2 = step!(cx2, cy2, zx2, zy2, n2);
            let a3 = step!(cx3, cy3, zx3, zy3, n3);
            if ((a0 | a1) | (a2 | a3)).none() {
                break;
            }
        }
        out[i..i + 4].copy_from_slice(&n0.to_array());
        out[i + 4..i + 8].copy_from_slice(&n1.to_array());
        out[i + 8..i + 12].copy_from_slice(&n2.to_array());
        out[i + 12..i + 16].copy_from_slice(&n3.to_array());
        i += 16;
    }
    mandelbrot_wide4(&cx[i..], &cy[i..], max_iter, &mut out[i..]);
}

/// Honourable mention: raw NEON with the same 4-block ILP structure. Not practical — unsafe,
/// one ISA, and the unroll factor is still per-machine — but the ceiling to compare against.
#[cfg(target_arch = "aarch64")]
pub fn mandelbrot_neon_ilp(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    use core::arch::aarch64::*;
    let n = cx.len();
    let mut i = 0;
    unsafe {
        let four = vdupq_n_f32(4.0);
        let one_bits = vreinterpretq_u32_f32(vdupq_n_f32(1.0));
        let two = vdupq_n_f32(2.0);
        while i + 16 <= n {
            let cxv: [float32x4_t; 4] = core::array::from_fn(|j| vld1q_f32(cx.as_ptr().add(i + 4 * j)));
            let cyv: [float32x4_t; 4] = core::array::from_fn(|j| vld1q_f32(cy.as_ptr().add(i + 4 * j)));
            let mut zx = [vdupq_n_f32(0.0); 4];
            let mut zy = [vdupq_n_f32(0.0); 4];
            let mut count = [vdupq_n_f32(0.0); 4];
            for _ in 0..max_iter {
                let mut alive = 0u32;
                for j in 0..4 {
                    let zx2 = vmulq_f32(zx[j], zx[j]);
                    let zy2 = vmulq_f32(zy[j], zy[j]);
                    let active = vcleq_f32(vaddq_f32(zx2, zy2), four);
                    count[j] = vaddq_f32(
                        count[j],
                        vreinterpretq_f32_u32(vandq_u32(active, one_bits)),
                    );
                    let nzx = vaddq_f32(vsubq_f32(zx2, zy2), cxv[j]);
                    zy[j] = vfmaq_f32(cyv[j], vmulq_f32(zx[j], zy[j]), two);
                    zx[j] = nzx;
                    alive |= vmaxvq_u32(active);
                }
                if alive == 0 {
                    break;
                }
            }
            for (j, c) in count.iter().enumerate() {
                vst1q_f32(out.as_mut_ptr().add(i + 4 * j), *c);
            }
            i += 16;
        }
    }
    mandelbrot_scalar(&cx[i..], &cy[i..], max_iter, &mut out[i..]);
}

fn bench_case_study(c: &mut Criterion) {
    let mut g = c.benchmark_group("mandelbrot_case_study");
    for &n in &[256usize, 1024, 4096] {
        let (cx, cy) = inputs(n);
        let mut want = vec![0.0f32; n];
        mandelbrot_scalar(&cx, &cy, MAX_ITER, &mut want);
        let mut got = vec![0.0f32; n];
        mandelbrot_wide4(&cx, &cy, MAX_ITER, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-3, "wide4 n={n}");
        mandelbrot_wide_ilp(&cx, &cy, MAX_ITER, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-3, "wide_ilp n={n}");
        mandelbrot_hp(&cx, &cy, MAX_ITER, &mut got);
        assert!(max_rel_err(&got, &want) < 1e-3, "hp n={n}");
        #[cfg(target_arch = "aarch64")]
        {
            mandelbrot_neon_ilp(&cx, &cy, MAX_ITER, &mut got);
            assert!(max_rel_err(&got, &want) < 1e-3, "neon_ilp n={n}");
        }

        let mut o = vec![0.0f32; n];
        g.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| mandelbrot_scalar(black_box(&cx), black_box(&cy), MAX_ITER, black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("wide4", n), &n, |b, _| {
            b.iter(|| mandelbrot_wide4(black_box(&cx), black_box(&cy), MAX_ITER, black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("wide_ilp", n), &n, |b, _| {
            b.iter(|| mandelbrot_wide_ilp(black_box(&cx), black_box(&cy), MAX_ITER, black_box(&mut o)))
        });
        #[cfg(target_arch = "aarch64")]
        g.bench_with_input(BenchmarkId::new("neon_ilp", n), &n, |b, _| {
            b.iter(|| mandelbrot_neon_ilp(black_box(&cx), black_box(&cy), MAX_ITER, black_box(&mut o)))
        });
        g.bench_with_input(BenchmarkId::new("hp", n), &n, |b, _| {
            b.iter(|| mandelbrot_hp(black_box(&cx), black_box(&cy), MAX_ITER, black_box(&mut o)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench_case_study);
criterion_main!(benches);
