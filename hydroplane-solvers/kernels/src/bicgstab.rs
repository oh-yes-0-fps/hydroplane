//! BiCGStab (van der Vorst) for a general sparse matrix, right-preconditioned by Jacobi, in the
//! module's working precision. Like [`crate::pcg`], the scalars live in a small storage buffer
//! (`sc`) so a batch of iterations needs no host round trip, and once converged, broken down or
//! out of iterations, `DONE` freezes the iterate.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::cholesky::Range;
use crate::pcg::{global, sum_all, tree_sum};
// The `ef64` build's arithmetic is inherent to `Ef64`, so it alone leaves `Scalar` unused.
#[allow(unused_imports)]
use crate::Scalar;
use crate::{Real, Shared, barrier, ld, st};

/// Layout of the scalar buffer.
pub mod sc {
    pub const RHO: usize = 0;
    pub const ALPHA: usize = 1;
    pub const OMEGA: usize = 2;
    pub const RR: usize = 3;
    /// Set by the host to the squared relative tolerance; `bicg_start_reduce` scales it by `‖b‖²`.
    pub const TOL2: usize = 4;
    pub const DONE: usize = 5;
    pub const ITERS: usize = 6;
    pub const MAX_ITERS: usize = 7;
    /// `‖b‖²`.
    pub const RR0: usize = 8;
    pub const BETA: usize = 9;
    pub const LEN: usize = 16;
}

fn done(scalars: &[Real]) -> bool {
    scalars[sc::DONE] != Real::ZERO
}

/// Row `i` of `A x` (values reached through `vidx`, the matrix's own value order).
fn row(row_ptr: &[u32], col: &[u32], vidx: &[u32], val: &[Real], x: &[Real], i: usize) -> Real {
    let mut acc: Real = Real::ZERO;
    let mut k = ld(row_ptr, i) as usize;
    let end = ld(row_ptr, i + 1) as usize;
    while k < end {
        acc = ld(val, ld(vidx, k) as usize).mul_add(ld(x, ld(col, k) as usize), acc);
        k += 1;
    }
    acc
}

/// Row `8 block + t / 32` of `A x` by the 32 lanes of a warp, summed through `part`, for lane 0.
fn warp_row(row_ptr: &[u32], col: &[u32], vidx: &[u32], val: &[Real], x: &[Real], part: &mut Shared<256>, row_index: usize, n: usize, t: usize) -> Real {
    let lane = t % 32;
    let mut acc: Real = Real::ZERO;
    if row_index < n {
        let mut k = ld(row_ptr, row_index) as usize + lane;
        let end = ld(row_ptr, row_index + 1) as usize;
        while k < end {
            acc = ld(val, ld(vidx, k) as usize).mul_add(ld(x, ld(col, k) as usize), acc);
            k += 32;
        }
    }
    part[t] = acc;
    barrier();
    let mut s: Real = Real::ZERO;
    if lane == 0 {
        let mut q = 0;
        while q < 32 {
            s += part[t + q];
            q += 1;
        }
    }
    s
}

/// `y = A x`, an invocation per row; per-workgroup partials of `w·y`.
#[spirv(compute(threads(256)))]
pub fn bicg_spmv_rows(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vidx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] val: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] y: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] w: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] partial: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] scalars: &[Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    if done(scalars) {
        return;
    }
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let yi = row(row_ptr, col, vidx, val, x, i);
        st(y, i, yi);
        v = ld(w, i) * yi;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}

/// [`bicg_spmv_rows`] with a warp per row, eight rows per workgroup: for long rows.
#[spirv(compute(threads(256)))]
pub fn bicg_spmv_warp(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vidx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] val: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] y: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] w: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] partial: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] scalars: &[Real],
    #[spirv(workgroup)] part: &mut Shared<256>,
    #[spirv(workgroup)] rows: &mut Shared<256>,
) {
    if done(scalars) {
        return;
    }
    let t = lid.x as usize;
    let block = (wid.y * prm.stride + wid.x) as usize;
    let i = block * 8 + t / 32;
    let n = prm.count as usize;
    let yi = warp_row(row_ptr, col, vidx, val, x, part, i, n, t);
    rows[t] = Real::ZERO;
    barrier();
    if t % 32 == 0 && i < n {
        st(y, i, yi);
        rows[t / 32] = ld(w, i) * yi;
    }
    let s = tree_sum(rows, t);
    if t == 0 {
        st(partial, block, s);
    }
}

/// `y = A x`, an invocation per row; per-workgroup partials of `w·y` and `y·y`.
#[spirv(compute(threads(256)))]
pub fn bicg_spmv2_rows(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vidx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] val: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] y: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] w: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] partial_wy: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] partial_yy: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] scalars: &[Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    if done(scalars) {
        return;
    }
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let (mut wy, mut yy): (Real, Real) = (Real::ZERO, Real::ZERO);
    if i < prm.count as usize {
        let yi = row(row_ptr, col, vidx, val, x, i);
        st(y, i, yi);
        wy = ld(w, i) * yi;
        yy = yi * yi;
    }
    let g = (wid.y * prm.stride + wid.x) as usize;
    shared[l] = wy;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial_wy, g, s);
    }
    barrier();
    shared[l] = yy;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial_yy, g, s);
    }
}

/// [`bicg_spmv2_rows`] with a warp per row, eight rows per workgroup.
#[spirv(compute(threads(256)))]
pub fn bicg_spmv2_warp(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vidx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] val: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] y: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] w: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] partial_wy: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] partial_yy: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 9)] scalars: &[Real],
    #[spirv(workgroup)] part: &mut Shared<256>,
    #[spirv(workgroup)] rows: &mut Shared<256>,
) {
    if done(scalars) {
        return;
    }
    let t = lid.x as usize;
    let block = (wid.y * prm.stride + wid.x) as usize;
    let i = block * 8 + t / 32;
    let n = prm.count as usize;
    let yi = warp_row(row_ptr, col, vidx, val, x, part, i, n, t);
    barrier();
    rows[t] = Real::ZERO;
    part[t] = Real::ZERO;
    barrier();
    if t % 32 == 0 && i < n {
        st(y, i, yi);
        rows[t / 32] = ld(w, i) * yi;
        part[t / 32] = yi * yi;
    }
    let s = tree_sum(rows, t);
    if t == 0 {
        st(partial_wy, block, s);
    }
    let s = tree_sum(part, t);
    if t == 0 {
        st(partial_yy, block, s);
    }
}

/// `y = D⁻¹ x`.
#[spirv(compute(threads(256)))]
pub fn bicg_jacobi(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] y: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] scalars: &[Real],
) {
    let i = global(wid, lid, prm);
    if i < prm.count as usize && !done(scalars) {
        st(y, i, ld(dinv, i) * ld(x, i));
    }
}

/// From `x = 0`: `r = r̂ = p = b`; per-workgroup partials of `r·r`.
#[spirv(compute(threads(256)))]
pub fn bicg_start(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] b: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] x: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] r: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] rhat: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] p: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let bi = ld(b, i);
        st(x, i, Real::ZERO);
        st(r, i, bi);
        st(rhat, i, bi);
        st(p, i, bi);
        v = bi * bi;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}

/// `ρ = ‖b‖²`, the absolute tolerance, done if already within it.
#[spirv(compute(threads(256)))]
pub fn bicg_start_reduce(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let rr = sum_all(partial, shared, l);
    if l == 0 {
        scalars[sc::RR0] = rr;
        scalars[sc::RR] = rr;
        scalars[sc::RHO] = rr;
        scalars[sc::TOL2] = scalars[sc::TOL2] * rr;
        if rr <= scalars[sc::TOL2] {
            scalars[sc::DONE] = Real::ONE;
        }
    }
}

/// `α = ρ / (r̂·v)`, done on breakdown.
#[spirv(compute(threads(256)))]
pub fn bicg_alpha(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let d = sum_all(partial, shared, l);
    if l == 0 && !done(scalars) {
        if d == Real::ZERO {
            scalars[sc::DONE] = Real::ONE;
        } else {
            scalars[sc::ALPHA] = scalars[sc::RHO] / d;
        }
    }
}

/// `s = r − α v`, `x += α z`; per-workgroup partials of `s·s`.
#[spirv(compute(threads(256)))]
pub fn bicg_s(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scalars: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] r: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] v: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] z: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] s: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] x: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    if done(scalars) {
        return;
    }
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let alpha = scalars[sc::ALPHA];
    let mut q: Real = Real::ZERO;
    if i < prm.count as usize {
        let si = (-alpha).mul_add(ld(v, i), ld(r, i));
        st(s, i, si);
        st(x, i, alpha.mul_add(ld(z, i), ld(x, i)));
        q = si * si;
    }
    shared[l] = q;
    let t = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, t);
    }
}

/// Done if `‖s‖` already meets the tolerance (then `x` is final).
#[spirv(compute(threads(256)))]
pub fn bicg_s_reduce(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let ss = sum_all(partial, shared, l);
    if l == 0 && !done(scalars) && ss <= scalars[sc::TOL2] {
        scalars[sc::RR] = ss;
        scalars[sc::ITERS] += Real::ONE;
        scalars[sc::DONE] = Real::ONE;
    }
}

/// `ω = (t·s) / (t·t)`.
#[spirv(compute(threads(256)))]
pub fn bicg_omega(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial_ts: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] partial_tt: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let ts = sum_all(partial_ts, shared, l);
    barrier();
    let tt = sum_all(partial_tt, shared, l);
    if l == 0 && !done(scalars) {
        scalars[sc::OMEGA] = if tt > Real::ZERO { ts / tt } else { Real::ZERO };
    }
}

/// `x += ω z`, `r = s − ω t`; per-workgroup partials of `r·r` and `r̂·r`.
#[spirv(compute(threads(256)))]
pub fn bicg_r(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scalars: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] s: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] t: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] z: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rhat: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] x: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] r: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] partial_rr: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 8)] partial_rho: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    if done(scalars) {
        return;
    }
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let omega = scalars[sc::OMEGA];
    let (mut rr, mut rho): (Real, Real) = (Real::ZERO, Real::ZERO);
    if i < prm.count as usize {
        st(x, i, omega.mul_add(ld(z, i), ld(x, i)));
        let ri = (-omega).mul_add(ld(t, i), ld(s, i));
        st(r, i, ri);
        rr = ri * ri;
        rho = ld(rhat, i) * ri;
    }
    let g = (wid.y * prm.stride + wid.x) as usize;
    shared[l] = rr;
    let q = tree_sum(shared, l);
    if l == 0 {
        st(partial_rr, g, q);
    }
    barrier();
    shared[l] = rho;
    let q = tree_sum(shared, l);
    if l == 0 {
        st(partial_rho, g, q);
    }
}

/// Counts the iteration, stops at the tolerance, the cap or a breakdown, and sets
/// `β = (ρ_new / ρ)(α / ω)`.
#[spirv(compute(threads(256)))]
pub fn bicg_beta(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial_rr: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] partial_rho: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let rr = sum_all(partial_rr, shared, l);
    barrier();
    let rho = sum_all(partial_rho, shared, l);
    if l == 0 && !done(scalars) {
        scalars[sc::ITERS] += Real::ONE;
        scalars[sc::RR] = rr;
        let omega = scalars[sc::OMEGA];
        if rr <= scalars[sc::TOL2] || scalars[sc::ITERS] >= scalars[sc::MAX_ITERS] || omega == Real::ZERO || rho == Real::ZERO {
            scalars[sc::DONE] = Real::ONE;
        } else {
            scalars[sc::BETA] = (rho / scalars[sc::RHO]) * (scalars[sc::ALPHA] / omega);
            scalars[sc::RHO] = rho;
        }
    }
}

/// `p = r + β (p − ω v)`.
#[spirv(compute(threads(256)))]
pub fn bicg_p(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scalars: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] r: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] v: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] p: &mut [Real],
) {
    let i = global(wid, lid, prm);
    if i >= prm.count as usize || done(scalars) {
        return;
    }
    let (beta, omega) = (scalars[sc::BETA], scalars[sc::OMEGA]);
    st(p, i, beta.mul_add((-omega).mul_add(ld(v, i), ld(p, i)), ld(r, i)));
}
