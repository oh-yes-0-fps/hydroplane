//! The 32-bit integer companion through `dispatch`: argmin-style index tracking in lockstep
//! with float compares, float bit manipulation via `to_bits`/`from_bits`, and the signed view.

use hydroplane::{Gang, kernel};

/// Index of the first minimum — the canonical companion workload: a `u32` lane ramp `select`ed
/// by the same mask the float compare produces.
#[kernel]
fn argmin<'a>(ctx: Gang<f32>, xs: &'a [f32]) -> u32 {
    let n = ctx.lanes();
    let mut best = ctx.splat(f32::INFINITY);
    let mut best_i = ctx.splat_u32(u32::MAX);
    let mut idx = ctx.ramp_u32();
    let step = ctx.splat_u32(n as u32);
    for off in ctx.chunks_exact(xs.len()) {
        let v = ctx.load(&xs[off..off + n]);
        let m = v.lt(best);
        best = v.select(m, best);
        best_i = idx.select(m, best_i);
        idx = idx + step;
    }
    if let Some((off, cnt)) = ctx.remainder(xs.len()) {
        let v = ctx.load_partial(&xs[off..off + cnt], f32::INFINITY);
        let m = v.lt(best);
        best = v.select(m, best);
        best_i = idx.select(m, best_i);
    }
    let overall = ctx.splat(best.reduce_min());
    let hit = (best.le(overall) & overall.le(best)).to_bitmask();
    let mut lanes_i = [0u32; hydroplane::MAX_LANES];
    best_i.store(&mut lanes_i[..n]);
    let mut ans = u32::MAX;
    let mut bits = hit;
    while bits != 0 {
        ans = ans.min(lanes_i[bits.trailing_zeros() as usize]);
        bits &= bits - 1;
    }
    ans
}

/// `floor(log2 x)` for positive normal floats, straight off the exponent field — the bit-trick
/// shape `to_bits` exists for.
#[kernel]
fn exponents<'a>(ctx: Gang<f32>, xs: &'a [f32], out: &'a mut [i32]) {
    let n = ctx.lanes();
    let bias = ctx.splat_i32(127);
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        let v = ctx.load_partial(&xs[off..off + cnt], 1.0);
        let e = ((v.to_bits() >> 23) & ctx.splat_u32(0xff)).as_i32() - bias;
        let mut buf = [0i32; hydroplane::MAX_LANES];
        e.store(&mut buf[..n]);
        out[off..off + cnt].copy_from_slice(&buf[..cnt]);
    });
}

/// `2^k` built by placing the biased exponent with integer ops and reinterpreting.
#[kernel]
fn exp2_int<'a>(ctx: Gang<f32>, ks: &'a [i32], out: &'a mut [f32]) {
    let n = ctx.lanes();
    let bias = ctx.splat_i32(127);
    ctx.for_each_chunk(ks.len(), |off, cnt| {
        let mut buf = [0i32; hydroplane::MAX_LANES];
        buf[..cnt].copy_from_slice(&ks[off..off + cnt]);
        let k = ctx.load_i32(&buf[..n]);
        let v = ctx.from_bits(((k + bias).as_u32()) << 23);
        v.store_partial(&mut out[off..off + cnt]);
    });
}

#[test]
fn argmin_matches_scalar() {
    for len in [1usize, 3, 4, 5, 7, 8, 15, 16, 17, 100, 1003] {
        let xs: Vec<f32> = (0..len).map(|i| ((i * 37 + 11) % 101) as f32 - 50.0).collect();
        let want = xs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert_eq!(argmin(&xs), want, "len={len}");
    }
    // duplicate minima: first index wins, matching the scalar scan
    let xs = [3.0f32, 1.0, 4.0, 1.0, 5.0, 1.0, 9.0];
    assert_eq!(argmin(&xs), 1);
}

#[test]
fn float_bit_tricks() {
    let xs: Vec<f32> = (0..37).map(|i| 0.03f32 * (i as f32 + 1.0) * 1.7f32.powi(i % 9)).collect();
    let mut got = vec![0i32; xs.len()];
    exponents(&xs, &mut got);
    for (i, (&x, &e)) in xs.iter().zip(&got).enumerate() {
        assert_eq!(e, x.log2().floor() as i32, "exponent of {x} (lane {i})");
    }

    let ks: Vec<i32> = (-20..=20).collect();
    let mut got = vec![0.0f32; ks.len()];
    exp2_int(&ks, &mut got);
    for (&k, &v) in ks.iter().zip(&got) {
        assert_eq!(v, 2.0f32.powi(k), "2^{k}");
    }
}

/// Signed view: arithmetic shift keeps the sign, comparisons order negatives correctly, and
/// wrapping arithmetic matches the scalar ops.
#[kernel]
fn signed_halve_count_neg<'a>(ctx: Gang<f32>, xs: &'a [i32], out: &'a mut [i32]) -> u32 {
    let n = ctx.lanes();
    let zero = ctx.splat_i32(0);
    let mut neg = 0u32;
    ctx.for_each_chunk(xs.len(), |off, cnt| {
        let mut buf = [0i32; hydroplane::MAX_LANES];
        buf[..cnt].copy_from_slice(&xs[off..off + cnt]);
        let v = ctx.load_i32(&buf[..n]);
        let halved = v >> 1;
        let mut o = [0i32; hydroplane::MAX_LANES];
        halved.store(&mut o[..n]);
        out[off..off + cnt].copy_from_slice(&o[..cnt]);
        neg += (v.lt(zero).to_bitmask() & ((1u32 << cnt) - 1)).count_ones();
    });
    neg
}

#[test]
fn signed_view() {
    let xs: Vec<i32> = (0..29).map(|i| (i - 14) * 3).collect();
    let mut out = vec![0i32; xs.len()];
    let neg = signed_halve_count_neg(&xs, &mut out);
    assert_eq!(neg, xs.iter().filter(|&&x| x < 0).count() as u32);
    for (&x, &h) in xs.iter().zip(&out) {
        assert_eq!(h, x >> 1, "arithmetic shift of {x}");
    }
}
