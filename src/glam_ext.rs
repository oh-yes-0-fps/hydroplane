//! `glam`-aware wide-vector helpers (opt-in `glam` feature).
//!
//! A glam [`Vec3`] spread across the lanes is three [`Varying`]s — one register per component. This
//! module wraps that triple in [`Vec3Wide`] (with `dot`/`length_squared`/operators), adds a
//! [`Mat3Wide`] for the rotate/transform kernels, and a [`GangGlamExt`] bridge that builds them from
//! glam values (`splat_vec3`, `gather_vec3`, …). Every method is a thin `#[inline]` wrapper over the
//! existing [`Gang`]/[`Varying`] primitives, so after monomorphization it lowers to exactly the
//! hand-rolled per-component code — the geometry layer is a pure ergonomic veneer, and it stays out
//! of the core (this whole module is behind the `glam` feature, off by default).

use core::ops::{Add, Mul, Sub};

use glam::{Mat3, Vec3};

use crate::backend::Backend;
use crate::varying::{Gang, Mask, Varying};

/// A `Vec3` whose three components are each a full register of lanes — the SIMD-wide form of a glam
/// [`Vec3`]. The public `.0` is the `[Varying; 3]` the [`Gang`] combinators (`gather_n`, `any_n`, …)
/// speak in, so it destructures freely.
#[derive(Clone, Copy)]
pub struct Vec3Wide<S: Backend<f32>>(pub [Varying<f32, S>; 3]);

impl<S: Backend<f32>> From<[Varying<f32, S>; 3]> for Vec3Wide<S> {
    #[inline(always)]
    fn from(v: [Varying<f32, S>; 3]) -> Self {
        Self(v)
    }
}

impl<S: Backend<f32>> Vec3Wide<S> {
    /// Per-lane dot product with another lane-vector.
    #[inline(always)]
    pub fn dot(self, o: Self) -> Varying<f32, S> {
        let [a, b, c] = self.0;
        let [x, y, z] = o.0;
        a * x + b * y + c * z
    }

    /// Per-lane squared length (`self · self`) — the form distance tests want (no `sqrt`).
    #[inline(always)]
    pub fn length_squared(self) -> Varying<f32, S> {
        self.dot(self)
    }

    /// Per-lane length.
    #[inline(always)]
    pub fn length(self) -> Varying<f32, S> {
        self.length_squared().sqrt()
    }

    /// `self + dir * t` — point-plus-scaled-direction. Kept as a separate multiply then add (not a
    /// fused [`Varying::fma`]) so results match the hand-rolled kernels bit-for-bit.
    #[inline(always)]
    pub fn add_scaled(self, dir: Self, t: Varying<f32, S>) -> Self {
        self + dir * t
    }

    /// Per-component lane select: `mask ? self : other`.
    #[inline(always)]
    pub fn select(self, mask: Mask<f32, S>, other: Self) -> Self {
        let [a, b, c] = self.0;
        let [x, y, z] = other.0;
        Self([a.select(mask, x), b.select(mask, y), c.select(mask, z)])
    }

    /// Write each component back to its column — one full register per column (`out[c].len()`
    /// must be exactly `lanes()`). Use [`store_partial`](Self::store_partial) for a short tail.
    #[inline(always)]
    pub fn store(self, out: [&mut [f32]; 3]) {
        let [a, b, c] = self.0;
        let [ox, oy, oz] = out;
        a.store(ox);
        b.store(oy);
        c.store(oz);
    }

    /// Write each component back to its column (`out[c]` gets component `c`'s active lanes).
    #[inline(always)]
    pub fn store_partial(self, out: [&mut [f32]; 3]) {
        let [a, b, c] = self.0;
        let [ox, oy, oz] = out;
        a.store_partial(ox);
        b.store_partial(oy);
        c.store_partial(oz);
    }
}

impl<S: Backend<f32>> Add for Vec3Wide<S> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        let [a, b, c] = self.0;
        let [x, y, z] = o.0;
        Self([a + x, b + y, c + z])
    }
}

impl<S: Backend<f32>> Sub for Vec3Wide<S> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        let [a, b, c] = self.0;
        let [x, y, z] = o.0;
        Self([a - x, b - y, c - z])
    }
}

/// Scale by a per-lane scalar (`dir * t`).
impl<S: Backend<f32>> Mul<Varying<f32, S>> for Vec3Wide<S> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, t: Varying<f32, S>) -> Self {
        let [a, b, c] = self.0;
        Self([a * t, b * t, c * t])
    }
}

/// Scale by a uniform scalar (`v * 0.5`).
impl<S: Backend<f32>> Mul<f32> for Vec3Wide<S> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, t: f32) -> Self {
        let [a, b, c] = self.0;
        Self([a * t, b * t, c * t])
    }
}

/// A `Mat3` splatted across the lanes (column-major, matching glam's [`Mat3`] layout) — for the
/// rotate / affine-transform kernels.
#[derive(Clone, Copy)]
pub struct Mat3Wide<S: Backend<f32>>([Varying<f32, S>; 9]);

impl<S: Backend<f32>> Mat3Wide<S> {
    /// `self * v` per lane, matching [`glam::Mat3::mul_vec3`] (columns `x_axis`/`y_axis`/`z_axis`).
    #[inline(always)]
    pub fn mul_vec3(self, v: Vec3Wide<S>) -> Vec3Wide<S> {
        let m = self.0;
        let [x, y, z] = v.0;
        Vec3Wide([
            m[0] * x + m[3] * y + m[6] * z,
            m[1] * x + m[4] * y + m[7] * z,
            m[2] * x + m[5] * y + m[8] * z,
        ])
    }

    /// `self * v + t` per lane — an affine transform (rotate/scale then translate).
    #[inline(always)]
    pub fn mul_add(self, v: Vec3Wide<S>, t: Vec3Wide<S>) -> Vec3Wide<S> {
        self.mul_vec3(v) + t
    }

    /// The nine column-major components as `[Varying; 9]` (the form the [`Gang`] column
    /// primitives speak in) — destructures freely.
    #[inline(always)]
    pub fn cols(self) -> [Varying<f32, S>; 9] {
        self.0
    }

    /// Per-lane determinant, matching [`glam::Mat3::determinant`]'s
    /// `z_axis · (x_axis × y_axis)` evaluation order.
    #[inline(always)]
    pub fn determinant(self) -> Varying<f32, S> {
        let m = self.0;
        let c0 = m[1] * m[5] - m[2] * m[4];
        let c1 = m[2] * m[3] - m[0] * m[5];
        let c2 = m[0] * m[4] - m[1] * m[3];
        m[6] * c0 + m[7] * c1 + m[8] * c2
    }

    /// Per-lane inverse via the cofactor/adjugate form, matching [`glam::Mat3::inverse`]
    /// (a single reciprocal of the determinant, then scaled cofactors). Lanes whose matrix is
    /// singular get non-finite components — guard the determinant if that can happen.
    #[inline(always)]
    pub fn inverse(self) -> Self {
        let m = self.0;
        let t0x = m[4] * m[8] - m[5] * m[7];
        let t0y = m[5] * m[6] - m[3] * m[8];
        let t0z = m[3] * m[7] - m[4] * m[6];
        let t1x = m[7] * m[2] - m[8] * m[1];
        let t1y = m[8] * m[0] - m[6] * m[2];
        let t1z = m[6] * m[1] - m[7] * m[0];
        let t2x = m[1] * m[5] - m[2] * m[4];
        let t2y = m[2] * m[3] - m[0] * m[5];
        let t2z = m[0] * m[4] - m[1] * m[3];
        let id = (m[6] * t2x + m[7] * t2y + m[8] * t2z).recip();
        Self([
            t0x * id, t1x * id, t2x * id,
            t0y * id, t1y * id, t2y * id,
            t0z * id, t1z * id, t2z * id,
        ])
    }

    /// Write each component back to its column — one full register per column (`out[c].len()`
    /// must be exactly `lanes()`). Use [`store_partial`](Self::store_partial) for a short tail.
    #[inline(always)]
    pub fn store(self, out: [&mut [f32]; 9]) {
        for (v, o) in self.0.into_iter().zip(out) {
            v.store(o);
        }
    }

    /// Write each component back to its column (`out[c]` gets component `c`'s active lanes).
    #[inline(always)]
    pub fn store_partial(self, out: [&mut [f32]; 9]) {
        for (v, o) in self.0.into_iter().zip(out) {
            v.store_partial(o);
        }
    }
}

impl<S: Backend<f32>> From<[Varying<f32, S>; 9]> for Mat3Wide<S> {
    #[inline(always)]
    fn from(v: [Varying<f32, S>; 9]) -> Self {
        Self(v)
    }
}

/// Builds [`Vec3Wide`]/[`Mat3Wide`] from glam values — the conversion bridge over the [`Gang`]
/// primitives (`splat_n`/`gather_n`/`load_n`/`load_partial_n`).
pub trait GangGlamExt<S: Backend<f32>> {
    /// Broadcast a uniform [`Vec3`] to a lane-vector (every lane the same).
    fn splat_vec3(self, v: Vec3) -> Vec3Wide<S>;
    /// Gather one register's worth of [`Vec3`]s (≤ `lanes()`), inactive tail lanes filled with `fill`.
    fn gather_vec3(self, s: &[Vec3], fill: f32) -> Vec3Wide<S>;
    /// Gather a chunk of `(normal, offset)` planes into a lane-vector of normals and a lane of offsets.
    fn gather_plane(self, s: &[(Vec3, f32)], fill: f32) -> (Vec3Wide<S>, Varying<f32, S>);
    /// Load one full register from each of three columns into a lane-vector.
    fn load_vec3(self, cols: [&[f32]; 3]) -> Vec3Wide<S>;
    /// Load up to one register from each of three columns, inactive tail lanes filled with `fill`.
    fn load_partial_vec3(self, cols: [&[f32]; 3], fill: f32) -> Vec3Wide<S>;
    /// Broadcast a uniform [`Mat3`] across the lanes.
    fn splat_mat3(self, m: Mat3) -> Mat3Wide<S>;
    /// Load one full register from each of nine column-major component columns into a lane-matrix.
    fn load_mat3(self, cols: [&[f32]; 9]) -> Mat3Wide<S>;
    /// Load up to one register from each of nine columns, inactive tail lanes filled with `fill`.
    fn load_partial_mat3(self, cols: [&[f32]; 9], fill: f32) -> Mat3Wide<S>;
}

impl<S: Backend<f32>> GangGlamExt<S> for Gang<S> {
    #[inline(always)]
    fn splat_vec3(self, v: Vec3) -> Vec3Wide<S> {
        Vec3Wide(self.splat_n([v.x, v.y, v.z]))
    }
    #[inline(always)]
    fn gather_vec3(self, s: &[Vec3], fill: f32) -> Vec3Wide<S> {
        Vec3Wide(self.gather_n(s, [fill; 3], |v| [v.x, v.y, v.z]))
    }
    #[inline(always)]
    fn gather_plane(self, s: &[(Vec3, f32)], fill: f32) -> (Vec3Wide<S>, Varying<f32, S>) {
        let [nx, ny, nz, d] = self.gather_n(s, [fill; 4], |&(n, dd)| [n.x, n.y, n.z, dd]);
        (Vec3Wide([nx, ny, nz]), d)
    }
    #[inline(always)]
    fn load_vec3(self, cols: [&[f32]; 3]) -> Vec3Wide<S> {
        Vec3Wide(self.load_n(cols))
    }
    #[inline(always)]
    fn load_partial_vec3(self, cols: [&[f32]; 3], fill: f32) -> Vec3Wide<S> {
        Vec3Wide(self.load_partial_n(cols, fill))
    }
    #[inline(always)]
    fn splat_mat3(self, m: Mat3) -> Mat3Wide<S> {
        Mat3Wide(self.splat_n(m.to_cols_array()))
    }
    #[inline(always)]
    fn load_mat3(self, cols: [&[f32]; 9]) -> Mat3Wide<S> {
        Mat3Wide(self.load_n(cols))
    }
    #[inline(always)]
    fn load_partial_mat3(self, cols: [&[f32]; 9], fill: f32) -> Mat3Wide<S> {
        Mat3Wide(self.load_partial_n(cols, fill))
    }
}
