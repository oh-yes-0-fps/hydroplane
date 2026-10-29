//! The V-cycle of a smoothed-aggregation multigrid preconditioner, in `f32` whatever the module's
//! working precision: it stands where an `f32` Cholesky solve stands in [`crate::pcg`], between
//! `pcg_update`'s `r` in `f32` and `pcg_zr`.
//!
//! Every matrix is by rows (`row_ptr`, `col`, `val`). A product takes an invocation per row, or,
//! with `prm.tau` non-zero, a warp per row and eight rows per workgroup, for long rows. The
//! Chebyshev smoother's coefficients are in `coef` from `prm.first`: `1/θ` for a first step, the
//! pair `c₁, c₂` for each later one.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::cholesky::Range;
use crate::pcg::global;
use crate::{Shared, barrier, ld, st};

/// This invocation's row of `M x` and whether it is the one to store it.
fn product(wid: UVec3, lid: UVec3, prm: &Range, row_ptr: &[u32], col: &[u32], val: &[f32], x: &[f32], part: &mut Shared<256, f32>) -> (usize, f32, bool) {
    let n = prm.count as usize;
    if prm.tau == 0 {
        let i = global(wid, lid, prm);
        let mut acc = 0.0f32;
        if i < n {
            let mut k = ld(row_ptr, i) as usize;
            let end = ld(row_ptr, i + 1) as usize;
            while k < end {
                acc += ld(val, k) * ld(x, ld(col, k) as usize);
                k += 1;
            }
        }
        return (i, acc, i < n);
    }
    let t = lid.x as usize;
    let lane = t % 32;
    let i = (wid.y * prm.stride + wid.x) as usize * 8 + t / 32;
    let mut acc = 0.0f32;
    if i < n {
        let mut k = ld(row_ptr, i) as usize + lane;
        let end = ld(row_ptr, i + 1) as usize;
        while k < end {
            acc += ld(val, k) * ld(x, ld(col, k) as usize);
            k += 32;
        }
    }
    part[t] = acc;
    barrier();
    let mut s = 0.0f32;
    if lane == 0 {
        let mut q = 0;
        while q < 32 {
            s += part[t + q];
            q += 1;
        }
    }
    (i, s, lane == 0 && i < n)
}

/// `y = M x`: a restriction, or the coarsest level's inverse applied.
#[spirv(compute(threads(256)))]
pub fn amg_spmv(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] val: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] y: &mut [f32],
    #[spirv(workgroup)] part: &mut Shared<256, f32>,
) {
    let (i, v, ok) = product(wid, lid, prm, row_ptr, col, val, x, part);
    if ok {
        st(y, i, v);
    }
}

/// `x += P xc`: the coarse correction interpolated.
#[spirv(compute(threads(256)))]
pub fn amg_prolong(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] val: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] xc: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &mut [f32],
    #[spirv(workgroup)] part: &mut Shared<256, f32>,
) {
    let (i, v, ok) = product(wid, lid, prm, row_ptr, col, val, xc, part);
    if ok {
        st(x, i, ld(x, i) + v);
    }
}

/// `res = b − A x`.
#[spirv(compute(threads(256)))]
pub fn amg_residual(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] val: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] res: &mut [f32],
    #[spirv(workgroup)] part: &mut Shared<256, f32>,
) {
    let (i, v, ok) = product(wid, lid, prm, row_ptr, col, val, x, part);
    if ok {
        st(res, i, ld(b, i) - v);
    }
}

/// `res −= A d`: the residual after a smoothing step.
#[spirv(compute(threads(256)))]
pub fn amg_smooth_residual(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] row_ptr: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] col: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] val: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] d: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] res: &mut [f32],
    #[spirv(workgroup)] part: &mut Shared<256, f32>,
) {
    let (i, v, ok) = product(wid, lid, prm, row_ptr, col, val, d, part);
    if ok {
        st(res, i, ld(res, i) - v);
    }
}

/// The first smoothing step from `x = 0`: `res = b`, `d = D⁻¹ b / θ`, `x = d`.
#[spirv(compute(threads(256)))]
pub fn amg_smooth_zero(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] coef: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dinv: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] res: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] d: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] x: &mut [f32],
) {
    let i = global(wid, lid, prm);
    if i >= prm.count as usize {
        return;
    }
    let bi = ld(b, i);
    let di = ld(dinv, i) * bi * ld(coef, prm.first as usize);
    st(res, i, bi);
    st(d, i, di);
    st(x, i, di);
}

/// The first smoothing step from the current `x`, `res` its residual: `d = D⁻¹ res / θ`, `x += d`.
#[spirv(compute(threads(256)))]
pub fn amg_smooth_start(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] coef: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dinv: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] res: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] d: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &mut [f32],
) {
    let i = global(wid, lid, prm);
    if i >= prm.count as usize {
        return;
    }
    let di = ld(dinv, i) * ld(res, i) * ld(coef, prm.first as usize);
    st(d, i, di);
    st(x, i, ld(x, i) + di);
}

/// A later smoothing step: `d = c₁ d + c₂ D⁻¹ res`, `x += d`.
#[spirv(compute(threads(256)))]
pub fn amg_smooth_step(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] coef: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] dinv: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] res: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] d: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &mut [f32],
) {
    let i = global(wid, lid, prm);
    if i >= prm.count as usize {
        return;
    }
    let at = prm.first as usize;
    let di = ld(coef, at) * ld(d, i) + ld(coef, at + 1) * ld(dinv, i) * ld(res, i);
    st(d, i, di);
    st(x, i, ld(x, i) + di);
}
