//! Preconditioned conjugate gradient in the module's working precision, with the preconditioner
//! applied by kernels of another module between `pcg_update` (which hands it `r` in `f32`) and
//! `pcg_zr` (which takes its `f32` result back): an `f32` Cholesky factor preconditioning an `f64`
//! iteration.
//!
//! The CG scalars live in a small storage buffer (`sc`), so a batch of iterations runs without a
//! host round trip; once converged, broken down or out of iterations, `DONE` freezes the iterate
//! and the remaining iterations of the batch do nothing.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::cholesky::Range;
use crate::{Real, Scalar, Shared, barrier, ld, st};

/// Layout of the scalar buffer.
pub mod sc {
    pub const RZ: usize = 0;
    pub const ALPHA: usize = 1;
    pub const BETA: usize = 2;
    pub const RR: usize = 3;
    /// Set by the host to the squared relative tolerance; `pcg_start_reduce` scales it by `‖b‖²`.
    pub const TOL2: usize = 4;
    pub const DONE: usize = 5;
    pub const ITERS: usize = 6;
    pub const MAX_ITERS: usize = 7;
    /// `‖b‖²`.
    pub const RR0: usize = 8;
    pub const LEN: usize = 16;
}

pub(crate) fn tree_sum(shared: &mut Shared<256>, l: usize) -> Real {
    barrier();
    let mut s = 128usize;
    while s > 0 {
        if l < s {
            let v = shared[l + s];
            shared[l] += v;
        }
        barrier();
        s >>= 1;
    }
    shared[0]
}

/// Sum of all of `partial`, by one workgroup.
pub(crate) fn sum_all(partial: &[Real], shared: &mut Shared<256>, l: usize) -> Real {
    let mut acc: Real = Real::ZERO;
    let mut i = l;
    while i < partial.len() {
        acc += ld(partial, i);
        i += 256;
    }
    shared[l] = acc;
    tree_sum(shared, l)
}

pub(crate) fn global(wid: UVec3, lid: UVec3, prm: &Range) -> usize {
    ((wid.y * prm.stride + wid.x) * 256 + lid.x) as usize
}

/// `q = A p` over the full symmetric matrix (`row_ptr`/`col` over both triangles, `vidx` into
/// the lower triangle's values): eight rows per workgroup, a warp-sized group per row reading it
/// contiguously; per-workgroup partials of `p·q`.
#[spirv(compute(threads(256)))]
pub fn pcg_spmv(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vidx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] val: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] p: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] q: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] partial: &mut [Real],
    #[spirv(workgroup)] part: &mut Shared<256>,
    #[spirv(workgroup)] rows: &mut Shared<256>,
) {
    let t = lid.x as usize;
    let (g, lane) = (t / 32, t % 32);
    let block = (wid.y * prm.stride + wid.x) as usize;
    let row = block * 8 + g;
    let n = prm.count as usize;
    let mut acc: Real = Real::ZERO;
    if row < n {
        let mut k = ld(row_ptr, row) as usize + lane;
        let end = ld(row_ptr, row + 1) as usize;
        while k < end {
            acc = ld(val, ld(vidx, k) as usize).mul_add(ld(p, ld(col, k) as usize), acc);
            k += 32;
        }
    }
    part[t] = acc;
    rows[t] = Real::ZERO;
    barrier();
    if lane == 0 && row < n {
        let mut s: Real = Real::ZERO;
        let mut i = 0;
        while i < 32 {
            s += part[g * 32 + i];
            i += 1;
        }
        st(q, row, s);
        rows[g] = ld(p, row) * s;
    }
    let s = tree_sum(rows, t);
    if t == 0 {
        st(partial, block, s);
    }
}

/// [`pcg_spmv`] for short rows (a stencil's handful of entries): one invocation per row, 256
/// rows per workgroup, so no lanes idle on rows shorter than a warp.
#[spirv(compute(threads(256)))]
pub fn pcg_spmv_rows(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] vidx: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] val: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] p: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] q: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let mut acc: Real = Real::ZERO;
        let mut k = ld(row_ptr, i) as usize;
        let end = ld(row_ptr, i + 1) as usize;
        while k < end {
            acc = ld(val, ld(vidx, k) as usize).mul_add(ld(p, ld(col, k) as usize), acc);
            k += 1;
        }
        st(q, i, acc);
        v = ld(p, i) * acc;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}

/// `α = rz / pᵀAp`, or zero (and done) once converged or broken down.
#[spirv(compute(threads(256)))]
pub fn pcg_alpha(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let pap = sum_all(partial, shared, l);
    if l == 0 {
        let mut alpha = Real::ZERO;
        if scalars[sc::DONE] == Real::ZERO {
            if pap > Real::ZERO {
                alpha = scalars[sc::RZ] / pap;
            } else {
                scalars[sc::DONE] = Real::ONE;
            }
        }
        scalars[sc::ALPHA] = alpha;
    }
}

/// `x += α p`, `r −= α q`; `r` in `f32` for the preconditioner; per-workgroup partials of `r·r`.
#[spirv(compute(threads(256)))]
pub fn pcg_update(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scalars: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] p: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] q: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] x: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] r: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] r32: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let alpha = scalars[sc::ALPHA];
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let mut ri = ld(r, i);
        if alpha != Real::ZERO {
            st(x, i, alpha.mul_add(ld(p, i), ld(x, i)));
            ri = (-alpha).mul_add(ld(q, i), ri);
            st(r, i, ri);
        }
        st(r32, i, ri.to_f32());
        v = ri * ri;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}

/// `z` from the preconditioner's `f32` result; per-workgroup partials of `r·z`.
#[spirv(compute(threads(256)))]
pub fn pcg_zr(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] z32: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] r: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] z: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let zi = Real::from_f32(ld(z32, i));
        st(z, i, zi);
        v = ld(r, i) * zi;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}

/// Reduces `r·z` and `r·r`; counts the iteration, stops at the tolerance or the cap, and sets
/// `β = rz_new / rz`.
#[spirv(compute(threads(256)))]
pub fn pcg_beta(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial_rz: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] partial_rr: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let rz_new = sum_all(partial_rz, shared, l);
    let rr = sum_all(partial_rr, shared, l);
    if l == 0 && scalars[sc::DONE] == Real::ZERO {
        scalars[sc::ITERS] += Real::ONE;
        scalars[sc::RR] = rr;
        if rr <= scalars[sc::TOL2] || scalars[sc::ITERS] >= scalars[sc::MAX_ITERS] || rz_new <= Real::ZERO {
            scalars[sc::DONE] = Real::ONE;
            scalars[sc::BETA] = Real::ZERO;
        } else {
            scalars[sc::BETA] = rz_new / scalars[sc::RZ];
            scalars[sc::RZ] = rz_new;
        }
    }
}

/// `p = z + β p`.
#[spirv(compute(threads(256)))]
pub fn pcg_direction(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scalars: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] z: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] p: &mut [Real],
) {
    let i = global(wid, lid, prm);
    if i >= prm.count as usize || scalars[sc::DONE] != Real::ZERO {
        return;
    }
    st(p, i, scalars[sc::BETA].mul_add(ld(p, i), ld(z, i)));
}

/// From `x = 0`: `r = b`, in `f32` for the preconditioner; per-workgroup partials of `r·r`.
#[spirv(compute(threads(256)))]
pub fn pcg_start(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] b: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] x: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] r: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] r32: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let bi = ld(b, i);
        st(x, i, Real::ZERO);
        st(r, i, bi);
        st(r32, i, bi.to_f32());
        v = bi * bi;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}

/// Starting scalars: `rz`, `‖b‖²`, the absolute tolerance, iteration count zero, `β = 0` (so the
/// next `pcg_direction` sets `p = z`), done if already within tolerance.
#[spirv(compute(threads(256)))]
pub fn pcg_start_reduce(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] partial_rz: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] partial_rr: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scalars: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let l = lid.x as usize;
    let rz = sum_all(partial_rz, shared, l);
    let rr = sum_all(partial_rr, shared, l);
    if l == 0 {
        let tol2 = scalars[sc::TOL2] * rr;
        scalars[sc::TOL2] = tol2;
        scalars[sc::RZ] = rz;
        scalars[sc::RR] = rr;
        scalars[sc::RR0] = rr;
        scalars[sc::ITERS] = Real::ZERO;
        scalars[sc::ALPHA] = Real::ZERO;
        scalars[sc::BETA] = Real::ZERO;
        scalars[sc::DONE] = if rr <= tol2 || rz <= Real::ZERO { Real::ONE } else { Real::ZERO };
    }
}

/// Jacobi's `z = D⁻¹ r`; per-workgroup partials of `r·z`: the preconditioner step of conjugate
/// gradient without a factor.
#[spirv(compute(threads(256)))]
pub fn pcg_jacobi(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] r: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] z: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] partial: &mut [Real],
    #[spirv(workgroup)] shared: &mut Shared<256>,
) {
    let i = global(wid, lid, prm);
    let l = lid.x as usize;
    let mut v: Real = Real::ZERO;
    if i < prm.count as usize {
        let ri = ld(r, i);
        let zi = ld(dinv, i) * ri;
        st(z, i, zi);
        v = ri * zi;
    }
    shared[l] = v;
    let s = tree_sum(shared, l);
    if l == 0 {
        st(partial, (wid.y * prm.stride + wid.x) as usize, s);
    }
}
