//! rust-gpu shader crate that exercises hydroplane's SPIR-V `Subgroup` backend end-to-end.
//!
//! Each entry point is one compute shader where every invocation is a lane: it builds a varying
//! value through the hydroplane `Simd` context, does element-wise math + a subgroup reduction, and
//! writes the result back. Building this with `cargo gpu` forces the backend code into real
//! SPIR-V (it is otherwise dead-code-eliminated), so the emitted module can be disassembled to
//! confirm the hardware instructions — `OpExtInst Sqrt`/`Fma`, `PackHalf2x16`/`UnpackHalf2x16`,
//! `OpGroupNonUniformFAdd` — and run on a device.
//!
//! Indexing is unchecked throughout: rust-gpu has no panic infrastructure, so a bounds-checked
//! `slice[i]` (which can panic) fails to compile. A real host harness sizes the buffers to the
//! dispatch, so the accesses are in-bounds by construction.
//!
//! Build:  `cargo gpu build --shader-crate ./hydroplane-test --spirv-builder-version 0.10.0-alpha.1`

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(dead_code)]

use core::marker::PhantomData;

use hydroplane::{Backend, Kernel, Scalar, Simd};

/// Splat this invocation's value, run `mul`/`add`/`fma`/`sqrt`, then sum across the subgroup.
/// The math is native `OpFMul`/`OpFAdd` + GLSL.std.450 `Fma`/`Sqrt`, and the reduce is
/// `OpGroupNonUniformFAdd`. For `f16`/`bf16` the `splat` widens (`UnpackHalf2x16` for `f16`) and
/// `reduce_sum` narrows. Bounded to `Compute = f32` (f32/f16/bf16) so the value is narrowed from
/// `f32` — avoiding the `f64` (and its u64 bit-twiddling) that `from_f64` would pull in.
struct OpKernel<T: Scalar<Compute = f32>> {
    val: f32,
    _t: PhantomData<T>,
}

impl<T: Scalar<Compute = f32>> Kernel<T> for OpKernel<T> {
    type Output = T;
    #[inline]
    fn run<S: Backend<T>>(self, ctx: Simd<T, S>) -> T {
        let x = ctx.splat(T::narrow(self.val));
        let z = (x * x + x).fma(x, x).sqrt();
        z.reduce_sum()
    }
}

/// The `f64` counterpart (its `Compute` is `f64`, so it can't use [`OpKernel`]).
struct WideKernel {
    val: f64,
}

impl Kernel<f64> for WideKernel {
    type Output = f64;
    #[inline]
    fn run<S: Backend<f64>>(self, ctx: Simd<f64, S>) -> f64 {
        let x = ctx.splat(self.val);
        let z = (x * x + x).fma(x, x).sqrt();
        z.reduce_sum()
    }
}

// Large enough that `choose` always selects the subgroup (vs. sequential) path.
#[cfg(target_arch = "spirv")]
const SUBGROUP_WORK: usize = 1 << 20;

#[cfg(target_arch = "spirv")]
use half::{bf16, f16};
#[cfg(target_arch = "spirv")]
use spirv_std::{glam::UVec3, spirv};

#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(32)))]
pub fn op_f32(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let i = gid.x as usize;
    let r: f32 =
        hydroplane::dispatch_subgroup(OpKernel::<f32> { val: input[i], _t: PhantomData }, SUBGROUP_WORK, 1);
    output[i] = r;
}

#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(32)))]
pub fn op_f64(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f64],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f64],
) {
    let i = gid.x as usize;
    let r: f64 = hydroplane::dispatch_subgroup(WideKernel { val: input[i] }, SUBGROUP_WORK, 1);
    output[i] = r;
}

#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(32)))]
pub fn op_f16(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let i = gid.x as usize;
    let r: f16 =
        hydroplane::dispatch_subgroup(OpKernel::<f16> { val: input[i], _t: PhantomData }, SUBGROUP_WORK, 1);
    output[i] = r.to_f32();
}

#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(32)))]
pub fn op_bf16(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let i = gid.x as usize;
    let r: bf16 = hydroplane::dispatch_subgroup(
        OpKernel::<bf16> { val: input[i], _t: PhantomData },
        SUBGROUP_WORK,
        1,
    );
    output[i] = r.to_f32();
}

// NOTE: a cooperative-matrix entry point (per COOPERATIVE_MATRIX.md) was attempted here and
// removed — rust-gpu's codegen has no `SpirvType` for cooperative matrices and stubs the ops as
// `reserved!`/`unreachable!`, so `OpTypeCooperativeMatrixKHR` in `asm!` ICEs the compiler on both
// spirv-std 0.10.0-alpha.1 and git rev 36e3348. See COOPERATIVE_MATRIX.md § Status.

