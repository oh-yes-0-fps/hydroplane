//! Integer elements as first-class kernel scalars: `Gang<u32>`/`Gang<i32>` through `#[kernel]`
//! and `dispatch`, with the full combinator machinery.

use hydroplane::{Gang, IntScalar, kernel};

/// SWAR popcount per lane, summed: exercises shifts, bitwise ops, wrapping mul, and reduce_sum
/// on a `u32` gang.
#[kernel]
fn popcount_sum<'a>(ctx: Gang, xs: &'a [u32]) -> u32 {
    let n = ctx.lanes::<u32>();
    let m1 = ctx.splat(0x5555_5555);
    let m2 = ctx.splat(0x3333_3333);
    let m4 = ctx.splat(0x0f0f_0f0f);
    let h01 = ctx.splat(0x0101_0101);
    let mut acc = ctx.splat(0);
    ctx.for_each_chunk::<u32>(xs.len(), |off, cnt| {
        let x = ctx.load_partial(&xs[off..off + cnt], 0);
        let x = x - ((x >> 1) & m1);
        let x = (x & m2) + ((x >> 2) & m2);
        let x = (x + (x >> 4)) & m4;
        acc = acc + ((x * h01) >> 24);
    });
    let mut lanes = [0u32; hydroplane::MAX_LANES];
    acc.store(&mut lanes[..n]);
    lanes[..n].iter().sum()
}

/// Min/max/abs over a signed gang with a masked count of negatives: `Ord` semantics and the same
/// `Mask` machinery the float kernels use.
#[kernel]
fn signed_stats<'a>(ctx: Gang, xs: &'a [i32]) -> (i32, i32, u32) {
    let n = ctx.lanes::<i32>();
    let zero = ctx.splat(0);
    let mut lo = ctx.splat(i32::MAX);
    let mut hi = ctx.splat(i32::MIN);
    let mut neg = 0u32;
    for off in ctx.chunks_exact::<i32>(xs.len()) {
        let v = ctx.load(&xs[off..off + n]);
        lo = lo.min(v);
        hi = hi.max(v);
        neg += v.lt(zero).to_bitmask().count_ones();
    }
    if let Some((off, cnt)) = ctx.remainder::<i32>(xs.len()) {
        let v = ctx.load_partial(&xs[off..off + cnt], i32::MAX);
        lo = lo.min(v);
        let v = ctx.load_partial(&xs[off..off + cnt], i32::MIN);
        hi = hi.max(v);
        let v = ctx.load_partial(&xs[off..off + cnt], 0);
        neg += v.lt(zero).to_bitmask().count_ones();
    }
    (lo.reduce_min(), hi.reduce_max(), neg)
}

/// Generic over the integer family; the macro finds the scalar through the `IntScalar` bound.
#[kernel]
fn wrapping_sum<'a, T: IntScalar>(ctx: Gang, xs: &'a [T]) -> T {
    let mut acc = ctx.splat(T::ZERO);
    ctx.for_each_chunk::<T>(xs.len(), |off, cnt| {
        acc = acc + ctx.load_partial(&xs[off..off + cnt], T::ZERO);
    });
    acc.reduce_sum()
}

#[test]
fn popcount_matches_scalar() {
    for len in [0usize, 1, 3, 4, 7, 8, 15, 16, 33, 1000] {
        let xs: Vec<u32> = (0..len).map(|i| (i as u32).wrapping_mul(0x9e37_79b9) ^ 0xdead_beef).collect();
        let want: u32 = xs.iter().map(|x| x.count_ones()).sum();
        assert_eq!(popcount_sum(&xs), want, "len={len}");
    }
}

#[test]
fn signed_stats_match_scalar() {
    for len in [1usize, 5, 8, 13, 100, 1003] {
        let xs: Vec<i32> = (0..len).map(|i| ((i as i32).wrapping_mul(2_654_435_761u32 as i32)) >> 8).collect();
        let (lo, hi, neg) = signed_stats(&xs);
        assert_eq!(lo, *xs.iter().min().unwrap(), "min len={len}");
        assert_eq!(hi, *xs.iter().max().unwrap(), "max len={len}");
        assert_eq!(neg, xs.iter().filter(|&&x| x < 0).count() as u32, "neg len={len}");
    }
}

#[test]
fn generic_int_kernel() {
    let xu: Vec<u32> = (0..37).map(|i| u32::MAX - i).collect();
    let want = xu.iter().fold(0u32, |a, &x| a.wrapping_add(x));
    assert_eq!(wrapping_sum(&xu), want);

    let xi: Vec<i32> = (0..37).map(|i| i - 18).collect();
    let want: i32 = xi.iter().sum();
    assert_eq!(wrapping_sum(&xi), want);
}
