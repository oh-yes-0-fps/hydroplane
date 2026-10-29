//! Supernodal right-looking sparse Cholesky, one elimination-tree level at a time, shaped after
//! NVIDIA cuDSS.
//!
//! Supernode `s` owns columns `BEGIN .. BEGIN + k` of the permuted matrix and stores them as a dense
//! `f × k` column-major block (`f = k + m`, `m` the rows of its off-diagonal pattern) at `L_OFF` in
//! `lval`: `faer`'s supernodal layout, so `lval` is the factor. A level's supernodes first factor
//! their own blocks by 32-wide tiles (diagonal-tile Cholesky, triangular solve of the tiles below,
//! update of the trailing tiles of the block), then each computes its update `L₂₁ L₂₁ᵀ` tile by tile
//! and subtracts it straight from its ancestors' blocks with atomic adds (`chol_schur`): column `j`
//! of the update lands in supernode `smap[j].0`, column `smap[j].1` of it, row `rmap[smap[j].2 + i]`.
//!
//! Each supernode's diagonal block is also inverted (`chol_inv_*`, by recursive doubling over its
//! tiles: two parallel products per doubling, no column-by-column chain), so the solve has no
//! triangular substitution at all. Forward, level by level from the leaves, a supernode forms
//! `y = L₁₁⁻¹ x₁` and subtracts `L₂₁ y` from its pattern rows with atomic adds; backward, from the
//! root, it gathers its ancestors' finished unknowns through its pattern and forms
//! `x₁ = L₁₁⁻ᵀ (y − L₂₁ᵀ x₂)`. Every step is a matrix-vector product spread over many workgroups.
//!
//! Work items are four words (`supernode, i, j, kb`) and one workgroup handles one item; a dispatch
//! covers `count` items from `first`, laid out as a `stride × ⌈count / stride⌉` grid.

use spirv_std::glam::UVec3;
use spirv_std::spirv;

use crate::{Real, Scalar, Shared, barrier, ld, st};

pub const TILE: usize = 32;

/// Words per supernode in the supernode table.
pub const FRONT_WORDS: usize = 7;
pub const L_OFF: usize = 0;
/// Block rows `f`.
pub const ORDER: usize = 1;
/// Columns `k`.
pub const COLS: usize = 2;
/// Offset of the block's tile boundaries in `bounds` (one of them at `k`).
pub const BOUNDS: usize = 3;
/// The supernode's first column in the permuted matrix.
pub const BEGIN: usize = 4;
/// Offset of the supernode's `m` pattern rows in `pat` (and of their entries in `smap`).
pub const PAT: usize = 5;
/// Offset in `xinv` of the inverse of the supernode's diagonal block, `k × k` column-major (lower
/// triangle; the upper holds intermediate products while it is formed).
pub const X_OFF: usize = 6;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Range {
    pub first: u32,
    pub count: u32,
    pub stride: u32,
    /// `chol_potrf`: the static-pivoting threshold `τ` as `f32` bits.
    pub tau: u32,
}

/// `−v` if column `c` is signed negative under a signed factorization, else `v`.
///
/// The `_signed` entry points factor `A = L S Lᵀ`, `S = diag(±1)` given per column in `sign`
/// (non-zero: −1): a quasi-definite matrix in an order whose pivot signs are known (a saddle point
/// in the KKT ordering). Every product `L Lᵀ` over a block's columns becomes `L S Lᵀ`, and the
/// pivots' roots are of their magnitudes. The plain ones are the same code with `SIGNED` false,
/// which never reads `sign`.
#[inline(always)]
fn flip<const SIGNED: bool>(v: Real, sign: &[u32], c: usize) -> Real {
    if SIGNED && ld(sign, c) != 0 { -v } else { v }
}

fn item(wid: UVec3, r: &Range) -> (bool, usize) {
    let g = wid.y * r.stride + wid.x;
    (g < r.count, (r.first + g) as usize * 4)
}

fn word(fronts: &[u32], s: usize, w: usize) -> usize {
    ld(fronts, s * FRONT_WORDS + w) as usize
}

/// Tile `t` of supernode `s`'s rows: first row and size.
fn tile(fronts: &[u32], bounds: &[u32], s: usize, t: usize) -> (usize, usize) {
    let b = word(fronts, s, BOUNDS);
    let lo = ld(bounds, b + t) as usize;
    (lo, ld(bounds, b + t + 1) as usize - lo)
}

/// `lval[dst[g]] = aval[g]`: the matrix's lower triangle into the supernode blocks.
#[spirv(compute(threads(256)))]
pub fn chol_scatter(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] dst: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] aval: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] lval: &mut [Real],
) {
    let g = ((wid.y * prm.stride + wid.x) * 256 + lid.x) as usize;
    if g < prm.count as usize {
        st(lval, ld(dst, g) as usize, ld(aval, g));
    }
}

/// Marks a `chol_step` item that factors a diagonal tile instead of updating one.
pub const POTRF: u32 = u32::MAX;

/// One panel step with one panel of lookahead. An item `(s, i, j, kb)` updates tile `(i, j)` of
/// supernode `s`'s own block with panel `kb`: `−= L(i, kb) L(j, kb)ᵀ`, the lower half only on the
/// diagonal. An item `(s, p, POTRF, -)` applies panel `p − 1`'s update to diagonal tile `p` itself
/// and factors it, so the Cholesky of the next panel runs alongside the bulk of this panel's
/// update instead of after it.
#[spirv(compute(threads(256)))]
pub fn chol_step(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] status: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] diag: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dinv: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] sign: &[u32],
    #[spirv(workgroup)] ta: &mut Shared<1024>,
    #[spirv(workgroup)] tb: &mut Shared<1024>,
    #[spirv(workgroup)] root: &mut Shared<32>,
    #[spirv(workgroup)] rd: &mut Shared<32>,
    #[spirv(workgroup)] floor: &mut Shared<32>,
    #[spirv(workgroup)] negative: &mut Shared<32, u32>,
) {
    step::<false>(wid, lid, prm, items, fronts, bounds, lval, status, diag, dinv, sign, ta, tb, root, rd, floor, negative);
}

#[spirv(compute(threads(256)))]
pub fn chol_step_signed(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] status: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] diag: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] dinv: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)] sign: &[u32],
    #[spirv(workgroup)] ta: &mut Shared<1024>,
    #[spirv(workgroup)] tb: &mut Shared<1024>,
    #[spirv(workgroup)] root: &mut Shared<32>,
    #[spirv(workgroup)] rd: &mut Shared<32>,
    #[spirv(workgroup)] floor: &mut Shared<32>,
    #[spirv(workgroup)] negative: &mut Shared<32, u32>,
) {
    step::<true>(wid, lid, prm, items, fronts, bounds, lval, status, diag, dinv, sign, ta, tb, root, rd, floor, negative);
}

fn step<const SIGNED: bool>(
    wid: UVec3,
    lid: UVec3,
    prm: &Range,
    items: &[u32],
    fronts: &[u32],
    bounds: &[u32],
    lval: &mut [Real],
    status: &mut [u32],
    diag: &[Real],
    dinv: &mut [Real],
    sign: &[u32],
    ta: &mut Shared<1024>,
    tb: &mut Shared<1024>,
    root: &mut Shared<32>,
    rd: &mut Shared<32>,
    floor: &mut Shared<32>,
    negative: &mut Shared<32, u32>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let j = ld(items, base + 2);
    let t = lid.x as usize;
    if j == POTRF {
        potrf::<SIGNED>(fronts, bounds, lval, status, diag, dinv, sign, ta, tb, root, rd, floor, negative, prm.tau, s, i, t);
    } else {
        update::<SIGNED>(fronts, bounds, lval, sign, ta, tb, s, i, j as usize, ld(items, base + 3) as usize, t);
    }
}

fn update<const SIGNED: bool>(fronts: &[u32], bounds: &[u32], lval: &mut [Real], sign: &[u32], ta: &mut Shared<1024>, tb: &mut Shared<1024>, s: usize, i: usize, j: usize, kb: usize, t: usize) {
    let f = word(fronts, s, ORDER);
    let l = word(fronts, s, L_OFF);
    let col = word(fronts, s, BEGIN);
    let (c0, b) = tile(fronts, bounds, s, kb);
    let (r0, rb) = tile(fronts, bounds, s, i);
    let (q0, qb) = tile(fronts, bounds, s, j);
    each_entry!(t, rb, b, |r, c| {
            ta[c * TILE + r] = flip::<SIGNED>(ld(lval, l + (c0 + c) * f + r0 + r), sign, col + c0 + c);
    });
    each_entry!(t, qb, b, |r, c| {
            tb[c * TILE + r] = ld(lval, l + (c0 + c) * f + q0 + r);
    });
    barrier();
    let r = t % TILE;
    let mut c = t / TILE;
    while c < TILE {
        if r < rb && c < qb && (i != j || r >= c) {
            let mut acc: Real = Real::ZERO;
            let mut p = 0;
            while p < b {
                acc = ta[p * TILE + r].mul_add(tb[p * TILE + c], acc);
                p += 1;
            }
            let at = l + (q0 + c) * f + r0 + r;
            st(lval, at, ld(lval, at) - acc);
        }
        c += 8;
    }
}

/// Diagonal tile `p` of supernode `s`: panel `p − 1`'s update applied, then its Cholesky in
/// shared memory and the inverse of the factor: the inverted pivots go to `dinv`, the rest of
/// `L_pp⁻¹`, transposed, into the tile's unused upper triangle, so `chol_trsm` is a
/// multiplication. A pivot below `τ |A_jj|` (`diag` is the permuted matrix's diagonal), or not a
/// number, is replaced by `τ |A_jj|` and sets `status[0]`: static pivoting, so a matrix at the
/// edge of the working precision still yields a usable (perturbed) factor. When `SIGNED`, the
/// pivot's sign is the column's, its root that of its magnitude, and a pivot of the wrong sign is
/// floored the same way.
fn potrf<const SIGNED: bool>(
    fronts: &[u32],
    bounds: &[u32],
    lval: &mut [Real],
    status: &mut [u32],
    diag: &[Real],
    dinv: &mut [Real],
    sign: &[u32],
    a: &mut Shared<1024>,
    inv: &mut Shared<1024>,
    root: &mut Shared<32>,
    rd: &mut Shared<32>,
    floor: &mut Shared<32>,
    negative: &mut Shared<32, u32>,
    tau: u32,
    s: usize,
    p: usize,
    t: usize,
) {
    let f = word(fronts, s, ORDER);
    let l = word(fronts, s, L_OFF);
    let col = word(fronts, s, BEGIN);
    let (c0, b) = tile(fronts, bounds, s, p);
    each_entry!(t, b, b, |r, c| {
            a[c * TILE + r] = ld(lval, l + (c0 + c) * f + c0 + r);
    });
    if t < b {
        floor[t] = Real::from_f32(f32::from_bits(tau)) * ld(diag, col + c0 + t).abs();
        if SIGNED {
            negative[t] = ld(sign, col + c0 + t);
        }
    }
    if p > 0 {
        let (k0, kw) = tile(fronts, bounds, s, p - 1);
        each_entry!(t, b, kw, |r, c| {
                inv[c * TILE + r] = ld(lval, l + (k0 + c) * f + c0 + r);
        });
        barrier();
        each_entry!(t, b, b, |r, c| {
                if r >= c {
                    let mut acc: Real = Real::ZERO;
                    let mut e = 0;
                    while e < kw {
                        acc = flip::<SIGNED>(inv[e * TILE + r], sign, col + k0 + e).mul_add(inv[e * TILE + c], acc);
                        e += 1;
                    }
                    a[c * TILE + r] -= acc;
                }
        });
        barrier();
    }
    let mut q = t;
    while q < b * b {
        inv[q] = Real::ZERO;
        q += 256;
    }
    barrier();
    // The invocation that applies the last update to pivot `c + 1` takes its root straight away,
    // so each column costs one barrier; the columns themselves are scaled at the end.
    if t == 0 {
        pivot::<SIGNED>(a, root, rd, floor, negative, status, 0);
    }
    barrier();
    let mut c = 0;
    while c < b {
        let rc = rd[c];
        // Invocation (i, g) updates row i of columns c + 1 + g, c + 9 + g, … up to the diagonal;
        // invocation c + 1 (row c + 1, group 0) holds the next pivot and takes its root at once.
        let (i, g) = (t % TILE, t / TILE);
        if i > c && i < b {
            let lic = a[c * TILE + i] * rc * rc;
            let lic = if SIGNED && negative[c] != 0 { -lic } else { lic };
            let mut j = c + 1 + g;
            while j <= i {
                a[j * TILE + i] = (-lic).mul_add(a[c * TILE + j], a[j * TILE + i]);
                j += 8;
            }
            if i == c + 1 && g == 0 {
                pivot::<SIGNED>(a, root, rd, floor, negative, status, c + 1);
            }
        }
        barrier();
        c += 1;
    }
    each_entry!(t, b, b, |r, c| {
            if r > c {
                let v = a[c * TILE + r] * rd[c];
                a[c * TILE + r] = if SIGNED && negative[c] != 0 { -v } else { v };
            } else if r == c {
                a[c * TILE + c] = root[c];
            }
    });
    barrier();
    // L⁻¹ by columns, one per invocation: inv[i][t] = −(Σ_{t≤p<i} L[i][p] inv[p][t]) / L[i][i],
    // stored row-major (`inv[i * TILE + t]`) so a warp's reads are consecutive.
    // Four partial sums, so the dependent chain is a quarter as long.
    if t < b {
        inv[t * TILE + t] = rd[t];
        let mut i = t + 1;
        while i < b {
            let (mut s0, mut s1, mut s2, mut s3): (Real, Real, Real, Real) = (Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO);
            let mut e = t;
            while e + 4 <= i {
                s0 = a[e * TILE + i].mul_add(inv[e * TILE + t], s0);
                s1 = a[(e + 1) * TILE + i].mul_add(inv[(e + 1) * TILE + t], s1);
                s2 = a[(e + 2) * TILE + i].mul_add(inv[(e + 2) * TILE + t], s2);
                s3 = a[(e + 3) * TILE + i].mul_add(inv[(e + 3) * TILE + t], s3);
                e += 4;
            }
            while e < i {
                s0 = a[e * TILE + i].mul_add(inv[e * TILE + t], s0);
                e += 1;
            }
            inv[i * TILE + t] = -((s0 + s1) + (s2 + s3)) * rd[i];
            i += 1;
        }
    }
    barrier();
    each_entry!(t, b, b, |r, c| {
            // Lower triangle: the factor; strict upper: (L⁻¹)ᵀ, entry (r, c) = L⁻¹[c][r].
            let v = if r >= c { a[c * TILE + r] } else { inv[c * TILE + r] };
            st(lval, l + (c0 + c) * f + c0 + r, v);
    });
    if t < b {
        st(dinv, col + c0 + t, rd[t]);
    }
}

/// Root and inverted root of pivot `c` (of its magnitude, its sign the column's when `SIGNED`),
/// floored as `potrf` says.
fn pivot<const SIGNED: bool>(a: &Shared<1024>, root: &mut Shared<32>, rd: &mut Shared<32>, floor: &Shared<32>, negative: &Shared<32, u32>, status: &mut [u32], c: usize) {
    let d = a[c * TILE + c];
    let d = if SIGNED && negative[c] != 0 { -d } else { d };
    let fl = floor[c];
    let v = if d > fl {
        d
    } else {
        st(status, 0, 1);
        if fl > Real::ZERO { fl } else { Real::ONE }
    };
    let y = v.rsqrt();
    root[c] = v * y;
    rd[c] = y;
}

/// Tile `(i, kb)` of the panel: `X = B L_kk⁻ᵀ`, from the inverse `chol_step` left in the diagonal
/// tile's upper triangle and `dinv`.
#[spirv(compute(threads(256)))]
pub fn chol_trsm(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sign: &[u32],
    #[spirv(workgroup)] ui: &mut Shared<1024>,
    #[spirv(workgroup)] x: &mut Shared<1024>,
) {
    trsm::<false>(wid, lid, prm, items, fronts, bounds, lval, dinv, sign, ui, x);
}

#[spirv(compute(threads(256)))]
pub fn chol_trsm_signed(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] sign: &[u32],
    #[spirv(workgroup)] ui: &mut Shared<1024>,
    #[spirv(workgroup)] x: &mut Shared<1024>,
) {
    trsm::<true>(wid, lid, prm, items, fronts, bounds, lval, dinv, sign, ui, x);
}

fn trsm<const SIGNED: bool>(
    wid: UVec3,
    lid: UVec3,
    prm: &Range,
    items: &[u32],
    fronts: &[u32],
    bounds: &[u32],
    lval: &mut [Real],
    dinv: &[Real],
    sign: &[u32],
    ui: &mut Shared<1024>,
    x: &mut Shared<1024>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let kb = ld(items, base + 3) as usize;
    let f = word(fronts, s, ORDER);
    let l = word(fronts, s, L_OFF);
    let col = word(fronts, s, BEGIN);
    let (c0, b) = tile(fronts, bounds, s, kb);
    let (r0, rb) = tile(fronts, bounds, s, i);
    let t = lid.x as usize;
    // ui[q][c] = (L⁻¹)[c][q] for q ≤ c.
    each_entry!(t, b, b, |r, c| {
            ui[c * TILE + r] = if r < c {
                ld(lval, l + (c0 + c) * f + c0 + r)
            } else if r == c {
                ld(dinv, col + c0 + c)
            } else {
                Real::ZERO
            };
    });
    each_entry!(t, rb, b, |r, c| {
            x[c * TILE + r] = ld(lval, l + (c0 + c) * f + r0 + r);
    });
    barrier();
    let r = t % TILE;
    let c = t / TILE;
    let (mut a0, mut a1, mut a2, mut a3): (Real, Real, Real, Real) = (Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO);
    if r < rb {
        let mut p = 0;
        while p < b {
            let xv = x[p * TILE + r];
            a0 = xv.mul_add(ui[c * TILE + p], a0);
            a1 = xv.mul_add(ui[(c + 8) * TILE + p], a1);
            a2 = xv.mul_add(ui[(c + 16) * TILE + p], a2);
            a3 = xv.mul_add(ui[(c + 24) * TILE + p], a3);
            p += 1;
        }
    }
    barrier();
    if r < rb {
        let base = l + c0 * f + r0 + r;
        let at = col + c0 + c;
        if c < b {
            st(lval, base + c * f, flip::<SIGNED>(a0, sign, at));
        }
        if c + 8 < b {
            st(lval, base + (c + 8) * f, flip::<SIGNED>(a1, sign, at + 8));
        }
        if c + 16 < b {
            st(lval, base + (c + 16) * f, flip::<SIGNED>(a2, sign, at + 16));
        }
        if c + 24 < b {
            st(lval, base + (c + 24) * f, flip::<SIGNED>(a3, sign, at + 24));
        }
    }
}

/// Tile `(i, j)` of supernode `s`'s update `L₂₁ L₂₁ᵀ` (rows and columns in its pattern), over all
/// `k` of its columns, subtracted from the ancestors' blocks with atomic adds.
#[spirv(compute(threads(256)))]
pub fn chol_schur(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] smap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rmap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sign: &[u32],
    #[spirv(workgroup)] ta: &mut Shared<1024>,
    #[spirv(workgroup)] tb: &mut Shared<1024>,
) {
    schur32::<false>(wid, lid, prm, items, fronts, bounds, smap, rmap, lval, sign, ta, tb);
}

#[spirv(compute(threads(256)))]
pub fn chol_schur_signed(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] smap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rmap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sign: &[u32],
    #[spirv(workgroup)] ta: &mut Shared<1024>,
    #[spirv(workgroup)] tb: &mut Shared<1024>,
) {
    schur32::<true>(wid, lid, prm, items, fronts, bounds, smap, rmap, lval, sign, ta, tb);
}

fn schur32<const SIGNED: bool>(
    wid: UVec3,
    lid: UVec3,
    prm: &Range,
    items: &[u32],
    fronts: &[u32],
    bounds: &[u32],
    smap: &[u32],
    rmap: &[u32],
    lval: &mut [Real],
    sign: &[u32],
    ta: &mut Shared<1024>,
    tb: &mut Shared<1024>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let j = ld(items, base + 2) as usize;
    let f = word(fronts, s, ORDER);
    let k = word(fronts, s, COLS);
    let l = word(fronts, s, L_OFF);
    let po = word(fronts, s, PAT);
    let col = word(fronts, s, BEGIN);
    let (r0, rb) = tile(fronts, bounds, s, i);
    let (q0, qb) = tile(fronts, bounds, s, j);
    let t = lid.x as usize;
    let r = t % TILE;
    let c = t / TILE;
    let (mut a0, mut a1, mut a2, mut a3): (Real, Real, Real, Real) = (Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO);
    let mut c0 = 0;
    while c0 < k {
        let b = if k - c0 < TILE { k - c0 } else { TILE };
        each_entry!(t, rb, b, |rr, cc| {
                ta[cc * TILE + rr] = flip::<SIGNED>(ld(lval, l + (c0 + cc) * f + r0 + rr), sign, col + c0 + cc);
        });
        each_entry!(t, qb, b, |rr, cc| {
                tb[cc * TILE + rr] = ld(lval, l + (c0 + cc) * f + q0 + rr);
        });
        barrier();
        let mut p = 0;
        while p < b {
            let av = ta[p * TILE + r];
            a0 = av.mul_add(tb[p * TILE + c], a0);
            a1 = av.mul_add(tb[p * TILE + c + 8], a1);
            a2 = av.mul_add(tb[p * TILE + c + 16], a2);
            a3 = av.mul_add(tb[p * TILE + c + 24], a3);
            p += 1;
        }
        barrier();
        c0 += TILE;
    }
    if r >= rb {
        return;
    }
    let pi = r0 + r - k;
    let mut u = 0;
    while u < 4 {
        let cc = c + 8 * u;
        if cc < qb && (i != j || r >= cc) {
            let acc = if u == 0 {
                a0
            } else if u == 1 {
                a1
            } else if u == 2 {
                a2
            } else {
                a3
            };
            let sm = 3 * (po + q0 + cc - k);
            let target = ld(smap, sm) as usize;
            let row = ld(rmap, (ld(smap, sm + 2).wrapping_add(pi as u32)) as usize) as usize;
            let at = word(fronts, target, L_OFF) + ld(smap, sm + 1) as usize * word(fronts, target, ORDER) + row;
            Real::atomic_sub(lval, at, acc);
        }
        u += 1;
    }
}

/// `chol_schur` over a 64 × 64 block of the pattern (rows from tile `i` up to tile `ie`, columns
/// from `j` up to `je`; item `(s, i, j, ie << 16 | je)`). Each invocation accumulates a 4 × 4 block
/// of outputs in registers, so a 16-wide slice of the panel costs 8 shared loads per 16
/// multiply-adds instead of 5 per 4: worth it where the update is throughput-bound (smaller GPUs,
/// `f32`), not where a quarter as many workgroups balances worse (a large GPU in `f64`).
#[spirv(compute(threads(256)))]
pub fn chol_schur64(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] smap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rmap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sign: &[u32],
    #[spirv(workgroup)] sa: &mut Shared<1024>,
    #[spirv(workgroup)] sb: &mut Shared<1024>,
) {
    schur64::<false>(wid, lid, prm, items, fronts, bounds, smap, rmap, lval, sign, sa, sb);
}

#[spirv(compute(threads(256)))]
pub fn chol_schur64_signed(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] smap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] rmap: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] lval: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sign: &[u32],
    #[spirv(workgroup)] sa: &mut Shared<1024>,
    #[spirv(workgroup)] sb: &mut Shared<1024>,
) {
    schur64::<true>(wid, lid, prm, items, fronts, bounds, smap, rmap, lval, sign, sa, sb);
}

fn schur64<const SIGNED: bool>(
    wid: UVec3,
    lid: UVec3,
    prm: &Range,
    items: &[u32],
    fronts: &[u32],
    bounds: &[u32],
    smap: &[u32],
    rmap: &[u32],
    lval: &mut [Real],
    sign: &[u32],
    sa: &mut Shared<1024>,
    sb: &mut Shared<1024>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let j = ld(items, base + 2) as usize;
    let ends = ld(items, base + 3);
    let (ie, je) = ((ends >> 16) as usize, (ends & 0xffff) as usize);
    let f = word(fronts, s, ORDER);
    let k = word(fronts, s, COLS);
    let l = word(fronts, s, L_OFF);
    let po = word(fronts, s, PAT);
    let bo = word(fronts, s, BOUNDS);
    let col = word(fronts, s, BEGIN);
    let r0 = ld(bounds, bo + i) as usize;
    let rn = ld(bounds, bo + ie) as usize - r0;
    let q0 = ld(bounds, bo + j) as usize;
    let qn = ld(bounds, bo + je) as usize - q0;
    let t = lid.x as usize;
    let (tr, tc) = (t % 16, t / 16);
    let mut acc: [Real; 16] = [Real::ZERO; 16];
    let mut c0 = 0;
    while c0 < k {
        let b = if k - c0 < 16 { k - c0 } else { 16 };
        unroll4!(w, {
            let q = t + 256 * w;
            let (rr, e) = (q % 64, q / 64);
            sa[q] = if rr < rn && e < b { flip::<SIGNED>(ld(lval, l + (c0 + e) * f + r0 + rr), sign, col + c0 + e) } else { Real::ZERO };
            sb[q] = if rr < qn && e < b { ld(lval, l + (c0 + e) * f + q0 + rr) } else { Real::ZERO };
        });
        barrier();
        let mut e = 0;
        while e < b {
            let av: [Real; 4] = [sa[e * 64 + tr], sa[e * 64 + tr + 16], sa[e * 64 + tr + 32], sa[e * 64 + tr + 48]];
            let bv: [Real; 4] = [sb[e * 64 + tc], sb[e * 64 + tc + 16], sb[e * 64 + tc + 32], sb[e * 64 + tc + 48]];
            unroll4!(u, {
                unroll4!(v, {
                    acc[u * 4 + v] = av[u].mul_add(bv[v], acc[u * 4 + v]);
                });
            });
            e += 1;
        }
        barrier();
        c0 += 16;
    }
    unroll4!(u, {
        unroll4!(v, {
            let (rr, cc) = (tr + 16 * u, tc + 16 * v);
            if rr < rn && cc < qn && (i != j || r0 + rr >= q0 + cc) {
                let sm = 3 * (po + q0 + cc - k);
                let target = ld(smap, sm) as usize;
                let row = ld(rmap, (ld(smap, sm + 2).wrapping_add((r0 + rr - k) as u32)) as usize) as usize;
                let at = word(fronts, target, L_OFF) + ld(smap, sm + 1) as usize * word(fronts, target, ORDER) + row;
                Real::atomic_sub(lval, at, acc[u * 4 + v]);
            }
        });
    });
}

/// Diagonal tile `i` of supernode `s`'s block inverse: the tile inverse `chol_step` left in the
/// tile's upper triangle and `dinv`.
#[spirv(compute(threads(256)))]
pub fn chol_inv_diag(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] dinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] xinv: &mut [Real],
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let f = word(fronts, s, ORDER);
    let k = word(fronts, s, COLS);
    let l = word(fronts, s, L_OFF);
    let xo = word(fronts, s, X_OFF);
    let col = word(fronts, s, BEGIN);
    let (c0, b) = tile(fronts, bounds, s, i);
    let t = lid.x as usize;
    each_entry!(t, b, b, |r, c| {
        let v = if r > c {
            ld(lval, l + (c0 + r) * f + c0 + c)
        } else if r == c {
            ld(dinv, col + c0 + r)
        } else {
            Real::ZERO
        };
        st(xinv, xo + (c0 + c) * k + c0 + r, v);
    });
}

/// The first product of a doubling step over tiles `A = [a, a + h)`, `C = [a + h, a + 2h)` of
/// supernode `s`'s block inverse: `T_ij = Σ_p L_ip X_pj` for tile `i` of `C`, `j` of `A`, written
/// transposed into the unused upper triangle (entry `(r, c)` of `T_ij` at row `c₀ⱼ + c`, column
/// `c₀ᵢ + r`).
#[spirv(compute(threads(256)))]
pub fn chol_inv_left(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] xinv: &mut [Real],
    #[spirv(workgroup)] ta: &mut Shared<1024>,
    #[spirv(workgroup)] tb: &mut Shared<1024>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let j = ld(items, base + 2) as usize;
    let h = ld(items, base + 3) as usize;
    let f = word(fronts, s, ORDER);
    let k = word(fronts, s, COLS);
    let l = word(fronts, s, L_OFF);
    let xo = word(fronts, s, X_OFF);
    let a_end = j - j % (2 * h) + h;
    let (ci, ib) = tile(fronts, bounds, s, i);
    let (cj, jb) = tile(fronts, bounds, s, j);
    let t = lid.x as usize;
    let r = t % TILE;
    let c = t / TILE;
    let (mut a0, mut a1, mut a2, mut a3): (Real, Real, Real, Real) = (Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO);
    let mut p = j;
    while p < a_end {
        let (cp, pb) = tile(fronts, bounds, s, p);
        each_entry!(t, ib, pb, |rr, e| {
            ta[e * TILE + rr] = ld(lval, l + (cp + e) * f + ci + rr);
        });
        each_entry!(t, pb, jb, |e, cc| {
            tb[e * TILE + cc] = ld(xinv, xo + (cj + cc) * k + cp + e);
        });
        barrier();
        let mut e = 0;
        while e < pb {
            let av = ta[e * TILE + r];
            a0 = av.mul_add(tb[e * TILE + c], a0);
            a1 = av.mul_add(tb[e * TILE + c + 8], a1);
            a2 = av.mul_add(tb[e * TILE + c + 16], a2);
            a3 = av.mul_add(tb[e * TILE + c + 24], a3);
            e += 1;
        }
        barrier();
        p += 1;
    }
    if r < ib {
        let base = xo + (ci + r) * k + cj;
        if c < jb {
            st(xinv, base + c, a0);
        }
        if c + 8 < jb {
            st(xinv, base + c + 8, a1);
        }
        if c + 16 < jb {
            st(xinv, base + c + 16, a2);
        }
        if c + 24 < jb {
            st(xinv, base + c + 24, a3);
        }
    }
}

/// The second product of a doubling step: `X_ij = −Σ_{p ≤ i} X_ip T_pj` for tile `i` of `C`, `j` of
/// `A`, from `chol_inv_left`'s transposed `T`.
#[spirv(compute(threads(256)))]
pub fn chol_inv_right(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] bounds: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] xinv: &mut [Real],
    #[spirv(workgroup)] ta: &mut Shared<1024>,
    #[spirv(workgroup)] tb: &mut Shared<1024>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let i = ld(items, base + 1) as usize;
    let j = ld(items, base + 2) as usize;
    let h = ld(items, base + 3) as usize;
    let k = word(fronts, s, COLS);
    let xo = word(fronts, s, X_OFF);
    let c_begin = j - j % (2 * h) + h;
    let (ci, ib) = tile(fronts, bounds, s, i);
    let (cj, jb) = tile(fronts, bounds, s, j);
    let t = lid.x as usize;
    let r = t % TILE;
    let c = t / TILE;
    let (mut a0, mut a1, mut a2, mut a3): (Real, Real, Real, Real) = (Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO);
    let mut p = c_begin;
    while p <= i {
        let (cp, pb) = tile(fronts, bounds, s, p);
        each_entry!(t, ib, pb, |rr, e| {
            ta[e * TILE + rr] = ld(xinv, xo + (cp + e) * k + ci + rr);
        });
        each_entry!(t, jb, pb, |cc, e| {
            tb[e * TILE + cc] = ld(xinv, xo + (cp + e) * k + cj + cc);
        });
        barrier();
        let mut e = 0;
        while e < pb {
            let av = ta[e * TILE + r];
            a0 = av.mul_add(tb[e * TILE + c], a0);
            a1 = av.mul_add(tb[e * TILE + c + 8], a1);
            a2 = av.mul_add(tb[e * TILE + c + 16], a2);
            a3 = av.mul_add(tb[e * TILE + c + 24], a3);
            e += 1;
        }
        barrier();
        p += 1;
    }
    if r < ib {
        let base = xo + cj * k + ci + r;
        if c < jb {
            st(xinv, base + c * k, -a0);
        }
        if c + 8 < jb {
            st(xinv, base + (c + 8) * k, -a1);
        }
        if c + 16 < jb {
            st(xinv, base + (c + 16) * k, -a2);
        }
        if c + 24 < jb {
            st(xinv, base + (c + 24) * k, -a3);
        }
    }
}

/// `x[c] = b[perm[c]]`.
#[spirv(compute(threads(256)))]
pub fn solve_gather(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] perm: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] x: &mut [Real],
) {
    let c = ((wid.y * prm.stride + wid.x) * 256 + lid.x) as usize;
    if c < prm.count as usize {
        st(x, c, ld(b, ld(perm, c) as usize));
    }
}

/// `m[at + q · stride]` for `q` in 0 ‥ 32, all 32 loads issued before any is used.
fn load32(m: &[Real], at: usize, stride: usize) -> [Real; 32] {
    let mut r: [Real; 32] = [Real::ZERO; 32];
    unroll32!(q, {
        r[q] = ld(m, at + q * stride);
    });
    r
}

/// `Σ_{q < 32} r[q] y[o + q]` in four independent chains.
fn dot_reg<const N: usize>(r: &[Real; 32], y: &Shared<N>, o: usize) -> Real {
    let mut s: [Real; 4] = [Real::ZERO; 4];
    unroll32!(q, {
        s[q % 4] = r[q].mul_add(y[o + q], s[q % 4]);
    });
    (s[0] + s[1]) + (s[2] + s[3])
}

/// `Σ_{q < n} m[at + q · stride] y[o + q]`, 32 independent loads at a time.
fn dot32<const N: usize>(m: &[Real], at: usize, stride: usize, y: &Shared<N>, o: usize, n: usize) -> Real {
    let mut acc: Real = Real::ZERO;
    let mut q = 0;
    while q + 32 <= n {
        acc += dot_reg(&load32(m, at + q * stride, stride), y, o + q);
        q += 32;
    }
    acc + dot8(m, at + q * stride, stride, y, o + q, n - q)
}

/// `Σ_{q < n} m[at + q · stride] y[o + q]`, eight independent loads at a time so their latencies
/// overlap.
fn dot8<const N: usize>(m: &[Real], at: usize, stride: usize, y: &Shared<N>, o: usize, n: usize) -> Real {
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7): (Real, Real, Real, Real, Real, Real, Real, Real) =
        (Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO, Real::ZERO);
    let mut q = 0;
    while q + 8 <= n {
        let p = at + q * stride;
        let (m0, m1, m2, m3) = (ld(m, p), ld(m, p + stride), ld(m, p + 2 * stride), ld(m, p + 3 * stride));
        let (m4, m5, m6, m7) = (ld(m, p + 4 * stride), ld(m, p + 5 * stride), ld(m, p + 6 * stride), ld(m, p + 7 * stride));
        s0 = m0.mul_add(y[o + q], s0);
        s1 = m1.mul_add(y[o + q + 1], s1);
        s2 = m2.mul_add(y[o + q + 2], s2);
        s3 = m3.mul_add(y[o + q + 3], s3);
        s4 = m4.mul_add(y[o + q + 4], s4);
        s5 = m5.mul_add(y[o + q + 5], s5);
        s6 = m6.mul_add(y[o + q + 6], s6);
        s7 = m7.mul_add(y[o + q + 7], s7);
        q += 8;
    }
    while q < n {
        s0 = ld(m, at + q * stride).mul_add(y[o + q], s0);
        q += 1;
    }
    ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

/// Forward, supernode `s`, rows `256 chunk ..` of its block: `y = L₁₁⁻¹ x₁` into `ytmp`
/// (written apart from `x`, which other workgroups of the dispatch are still reading).
#[spirv(compute(threads(256)))]
pub fn solve_fgemv(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] xinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] ytmp: &mut [Real],
    #[spirv(workgroup)] b: &mut Shared<256>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let chunk = ld(items, base + 1) as usize;
    let k = word(fronts, s, COLS);
    let xo = word(fronts, s, X_OFF);
    let col = word(fronts, s, BEGIN);
    let t = lid.x as usize;
    let r = chunk * 256 + t;
    let last = if k < chunk * 256 + 256 { k } else { chunk * 256 + 256 };
    let mut acc: Real = Real::ZERO;
    let mut qc = 0;
    while qc < last {
        if qc + t < k {
            b[t] = ld(x, col + qc + t);
        }
        barrier();
        if r < k && qc <= r {
            let n = if r + 1 - qc < 256 { r + 1 - qc } else { 256 };
            acc += dot8(xinv, xo + qc * k + r, k, b, 0, n);
        }
        barrier();
        qc += 256;
    }
    if r < k {
        st(ytmp, col + r, acc);
    }
}

/// Forward, supernode `s`, pattern rows `256 chunk ..`: `x[pattern_i] −= L₂₁ y`, atomically.
#[spirv(compute(threads(256)))]
pub fn solve_fpattern(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pat: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] ytmp: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] x: &mut [Real],
    #[spirv(workgroup)] y: &mut Shared<256>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let chunk = ld(items, base + 1) as usize;
    let f = word(fronts, s, ORDER);
    let k = word(fronts, s, COLS);
    let l = word(fronts, s, L_OFF);
    let col = word(fronts, s, BEGIN);
    let t = lid.x as usize;
    let i = chunk * 256 + t;
    let mut acc: Real = Real::ZERO;
    let mut qc = 0;
    while qc < k {
        if qc + t < k {
            y[t] = ld(ytmp, col + qc + t);
        }
        barrier();
        if k + i < f {
            let n = if k - qc < 256 { k - qc } else { 256 };
            acc += dot32(lval, l + qc * f + k + i, f, y, 0, n);
        }
        barrier();
        qc += 256;
    }
    if k + i < f {
        Real::atomic_sub(x, ld(pat, word(fronts, s, PAT) + i) as usize, acc);
    }
}

/// Backward, supernode `s`, columns `8 chunk ..`: `v_j = y_j − Σ_i L_(k+i) j x[pattern_i]` in
/// `ytmp` (`s_j y_j` when signed, the `S` between the two substitutions), from the
/// ancestors' finished unknowns. A warp-sized group per column, its lanes reading
/// the column's pattern part contiguously.
#[spirv(compute(threads(256)))]
pub fn solve_bpattern(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pat: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] ytmp: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sign: &[u32],
    #[spirv(workgroup)] part: &mut Shared<256>,
) {
    bpattern::<false>(wid, lid, prm, items, fronts, pat, lval, x, ytmp, sign, part);
}

#[spirv(compute(threads(256)))]
pub fn solve_bpattern_signed(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] pat: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] lval: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] ytmp: &mut [Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] sign: &[u32],
    #[spirv(workgroup)] part: &mut Shared<256>,
) {
    bpattern::<true>(wid, lid, prm, items, fronts, pat, lval, x, ytmp, sign, part);
}

fn bpattern<const SIGNED: bool>(
    wid: UVec3,
    lid: UVec3,
    prm: &Range,
    items: &[u32],
    fronts: &[u32],
    pat: &[u32],
    lval: &[Real],
    x: &[Real],
    ytmp: &mut [Real],
    sign: &[u32],
    part: &mut Shared<256>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let chunk = ld(items, base + 1) as usize;
    let f = word(fronts, s, ORDER);
    let k = word(fronts, s, COLS);
    let m = f - k;
    let l = word(fronts, s, L_OFF);
    let col = word(fronts, s, BEGIN);
    let po = word(fronts, s, PAT);
    let t = lid.x as usize;
    let (g, lane) = (t / 32, t % 32);
    let j = chunk * 8 + g;
    let mut acc: Real = Real::ZERO;
    if j < k {
        let mut i = lane;
        while i < m {
            acc = ld(lval, l + j * f + k + i).mul_add(ld(x, ld(pat, po + i) as usize), acc);
            i += 32;
        }
    }
    part[t] = acc;
    barrier();
    if lane == 0 && j < k {
        let mut sum: Real = Real::ZERO;
        let mut q = 0;
        while q < 32 {
            sum += part[g * 32 + q];
            q += 1;
        }
        st(ytmp, col + j, flip::<SIGNED>(ld(ytmp, col + j), sign, col + j) - sum);
    }
}

/// Backward, supernode `s`, columns `8 chunk ..`: `x₁ = L₁₁⁻ᵀ v`, column `r` of the inverse read
/// contiguously by a warp-sized group.
#[spirv(compute(threads(256)))]
pub fn solve_bgemv(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] items: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] fronts: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] xinv: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] ytmp: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] x: &mut [Real],
    #[spirv(workgroup)] part: &mut Shared<256>,
) {
    let (ok, base) = item(wid, prm);
    if !ok {
        return;
    }
    let s = ld(items, base) as usize;
    let chunk = ld(items, base + 1) as usize;
    let k = word(fronts, s, COLS);
    let xo = word(fronts, s, X_OFF);
    let col = word(fronts, s, BEGIN);
    let t = lid.x as usize;
    let (g, lane) = (t / 32, t % 32);
    let r = chunk * 8 + g;
    let mut acc: Real = Real::ZERO;
    if r < k {
        let mut q = r + lane;
        while q < k {
            acc = ld(xinv, xo + r * k + q).mul_add(ld(ytmp, col + q), acc);
            q += 32;
        }
    }
    part[t] = acc;
    barrier();
    if lane == 0 && r < k {
        let mut sum: Real = Real::ZERO;
        let mut q = 0;
        while q < 32 {
            sum += part[g * 32 + q];
            q += 1;
        }
        st(x, col + r, sum);
    }
}

/// `out[perm[c]] = x[c]`.
#[spirv(compute(threads(256)))]
pub fn solve_scatter(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(push_constant)] prm: &Range,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] perm: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] x: &[Real],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [Real],
) {
    let c = ((wid.y * prm.stride + wid.x) * 256 + lid.x) as usize;
    if c < prm.count as usize {
        st(out, ld(perm, c) as usize, ld(x, c));
    }
}
