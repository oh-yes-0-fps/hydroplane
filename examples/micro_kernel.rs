//! Micro-kernels: tiny SIMD functions over a handful of elements, the kind you call in a hot loop.
//!
//! A robot joint-limit check — "is any joint below its lower bound or above its upper bound?" — over
//! a 6-DOF configuration. Each check is only six `f32`s, so the *dispatch* (picking a backend) must
//! be near-free or it dwarfs the work. `dispatch` caches the resolved backend in a process-global
//! atomic, so a planner calling this millions of times pays CPU detection only once.
//!
//! `out_of_limits` composes the two sub-kernels with their `_on` companions, so the whole check is a
//! single dispatch even though it runs three kernel bodies. `dist_sq` is a plain sum reduction
//! written with no awareness of instruction-level parallelism — `zip_sum` runs as many independent
//! accumulator chains as this core's FP pipes want (a factor detected and cached like the backend),
//! so the obvious kernel saturates the machine for free.
//!
//! Run with `cargo run --example micro_kernel --release` (add `-C target-cpu=native` via RUSTFLAGS
//! to fold the chosen backend in and inline across the kernels entirely).

use hydroplane::{Gang, kernel};

/// Any `a[i] > b[i]`? The full-register pass short-circuits on plain loads; only the remainder
/// stages sentinels — `-inf` (lhs) / `+inf` (rhs), so `-inf > +inf` is false and padding never
/// trips the reduction.
#[kernel(tiny)]
fn any_gt<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> bool {
    let n = ctx.lanes::<f32>();
    for off in ctx.chunks_exact::<f32>(a.len()) {
        if ctx.load(&a[off..off + n]).gt(ctx.load(&b[off..off + n])).any() {
            return true;
        }
    }
    if let Some((off, cnt)) = ctx.remainder::<f32>(a.len()) {
        let x = ctx.load_partial(&a[off..off + cnt], f32::NEG_INFINITY);
        let y = ctx.load_partial(&b[off..off + cnt], f32::INFINITY);
        return x.gt(y).any();
    }
    false
}

/// Any `a[i] < b[i]`? Remainder sentinels are swapped: `+inf < -inf` is false.
#[kernel(tiny)]
fn any_lt<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> bool {
    let n = ctx.lanes::<f32>();
    for off in ctx.chunks_exact::<f32>(a.len()) {
        if ctx.load(&a[off..off + n]).lt(ctx.load(&b[off..off + n])).any() {
            return true;
        }
    }
    if let Some((off, cnt)) = ctx.remainder::<f32>(a.len()) {
        let x = ctx.load_partial(&a[off..off + cnt], f32::INFINITY);
        let y = ctx.load_partial(&b[off..off + cnt], f32::NEG_INFINITY);
        return x.lt(y).any();
    }
    false
}

/// Below `lo` anywhere, or above `hi` anywhere? One dispatch (at this kernel's entry); the two
/// sub-kernels run on the same already-chosen backend via their `_on` companions.
#[kernel(tiny)]
fn out_of_limits<'a>(ctx: Gang, q: &'a [f32], lo: &'a [f32], hi: &'a [f32]) -> bool {
    any_lt_on(ctx, q, lo) || any_gt_on(ctx, q, hi)
}

/// Squared distance between two joint configs: `Σ (q[i] - p[i])²`. A plain sum reduction — `zip_sum`
/// supplies the `0` identity, the masked tail, the chain combine, and the per-core unroll factor, so
/// nothing here mentions accumulators-per-pipe yet the loop runs them all.
#[kernel(tiny)]
fn dist_sq<'a>(ctx: Gang, q: &'a [f32], p: &'a [f32]) -> f32 {
    ctx.zip_sum(q, p, |acc, a, b| {
        let d = a - b;
        d.fma(d, acc)
    })
}

fn main() {
    let lo = [-3.2_f32, -2.0, -2.8, -3.2, -2.0, -6.4];
    let hi = [3.2_f32, 2.0, 2.8, 3.2, 2.0, 6.4];

    let inside = [0.1_f32, 0.5, -1.0, 1.2, 0.0, 3.0];
    let high = [0.1_f32, 0.5, -1.0, 1.2, 0.0, 7.0]; // joint 5 past its upper bound
    let low = [0.1_f32, -2.5, -1.0, 1.2, 0.0, 3.0]; // joint 1 below its lower bound

    println!("inside limits?  {}", !out_of_limits(&inside, &lo, &hi)); // true
    println!("high  in limits {}", !out_of_limits(&high, &lo, &hi)); // false
    println!("low   in limits {}", !out_of_limits(&low, &lo, &hi)); // false
    println!("dist² inside→high: {}", dist_sq(&inside, &high)); // (3.0-7.0)² = 16
}
