//! GPU subgroup support for the rust-gpu / SPIR-V target — the ISPC→SIMT mapping.
//!
//! When `hydroplane` is compiled to SPIR-V (`target_arch = "spirv"`), the SPMD "gang" is a GPU
//! **subgroup** (warp). This module has two halves:
//!
//! * **Portable scheduling policy** (always compiled, unit-tested below): the [`choose`]
//!   policy that decides, from the item count and the subgroup size, whether to run a single
//!   invocation **sequentially** or fan the work across the **subgroup**. It runs (and is
//!   tested) on the CPU like any other code.
//! * **The SPIR-V [`Subgroup`] backend** (`#[cfg(target_arch = "spirv")]`): `Vector = T`,
//!   `Mask = bool` (one lane *per invocation*), with the cross-lane ops lowering to subgroup
//!   collectives (`OpGroupNonUniformAny`/`All`/`FAdd`/…). The warp width is read straight
//!   from the hardware `SubgroupSize` builtin — no host plumbing, no atomics, no entry-point
//!   parameter to wire. It compiles only under the rust-gpu toolchain (see `GPU.md`); it is
//!   not built in a normal host `cargo build`.

/// How a batch should be executed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Execution {
    /// Too little work to fill a warp — one invocation loops over the items.
    Sequential,
    /// Enough work — distribute the items across the subgroup's lanes.
    Subgroup,
}

/// Decide [`Execution`] from the item count and the subgroup size.
///
/// Runs the subgroup path only when there are at least `subgroup_size * fill_factor` items,
/// so the cost of subgroup collectives is paid only when there is enough work to amortise it;
/// otherwise a single invocation's serial loop wins. A `subgroup_size` of 0 or 1 (unknown /
/// scalar) always selects [`Execution::Sequential`].
#[inline]
pub fn choose(item_count: usize, subgroup_size: u32, fill_factor: u32) -> Execution {
    if subgroup_size <= 1 {
        return Execution::Sequential;
    }
    // Plain `usize` (32-bit on SPIR-V) multiply: `subgroup_size * fill_factor` is always small
    // (warp width ≤ ~128), so it can't overflow — and a plain `*` avoids the `saturating_mul`
    // intrinsic, which rust-gpu can't lower.
    let threshold = subgroup_size as usize * fill_factor.max(1) as usize;
    if item_count < threshold {
        Execution::Sequential
    } else {
        Execution::Subgroup
    }
}

// ─────────────────────────── SPIR-V backend (device-only) ───────────────────────────
//
// Compiled only under rust-gpu. `Vector = T` / `Mask = bool` (one lane per invocation);
// cross-lane ops are subgroup collectives. The intrinsic names below target spirv-std's
// `arch` subgroup ops; pin them to the rust-gpu rev used by the shader crate (cf. GPU.md).
#[cfg(target_arch = "spirv")]
pub use device::{Subgroup, dispatch_subgroup};

#[cfg(target_arch = "spirv")]
mod device {
    use super::{Execution, choose};
    use crate::backend::{Backend, ScalarBackend};
    use crate::dispatch::Kernel;
    use crate::scalar::Scalar;
    use crate::varying::Simd;
    use half::{bf16, f16};

    /// Device entry point — the GPU analogue of the host [`dispatch`](crate::dispatch).
    ///
    /// The host picks a backend by *which ISA the CPU has*; here the runtime axis is *how much
    /// work there is*. With at least `subgroup_size * fill_factor` items the batch is spread
    /// across the warp and the kernel's cross-lane ops become subgroup collectives
    /// ([`Subgroup`]); below that threshold the collectives don't pay for themselves, so a
    /// single invocation loops over the items on the scalar backend. Either way the kernel body
    /// is the one written against [`Simd`].
    ///
    /// `item_count` is the batch size the calling subgroup is responsible for (from the buffer
    /// length / push constant the entry point already knows); `fill_factor` is the occupancy
    /// threshold (`4` is a sane default). The warp width is read from the `SubgroupSize`
    /// builtin, so there is nothing else to pass in.
    #[inline]
    pub fn dispatch_subgroup<T, K>(kernel: K, item_count: usize, fill_factor: u32) -> K::Output
    where
        T: Scalar,
        K: Kernel<T>,
        Subgroup: Backend<T>,
    {
        match choose(item_count, subgroup_size(), fill_factor) {
            Execution::Subgroup => kernel.run(Simd::new(Subgroup::new())),
            Execution::Sequential => kernel.run(Simd::new(ScalarBackend)),
        }
    }

    /// Read the hardware `SubgroupSize` builtin — the warp width the running invocation
    /// belongs to. This is the per-invocation ground truth (correct even on hardware whose
    /// subgroup size is variable) and lowers to a single builtin load, so it is effectively
    /// free to call. The `Input` variable and its decoration are hoisted to module scope by
    /// rust-gpu's inline-asm lowering; pin to the rust-gpu rev (cf. GPU.md).
    #[inline]
    fn subgroup_size() -> u32 {
        let mut size: u32;
        unsafe {
            core::arch::asm!(
                "%u32 = OpTypeInt 32 0",
                // rust-gpu requires `Generic` for `OpTypePointer` in asm! and infers the real
                // storage class (here `Input`) from the `OpVariable` below.
                "%ptr = OpTypePointer Generic %u32",
                "%var = OpVariable %ptr Input",
                "OpDecorate %var BuiltIn SubgroupSize",
                "{size} = OpLoad %u32 %var",
                size = out(reg) size,
            );
        }
        size
    }

    /// Hardware square root — GLSL.std.450 `Sqrt` (opcode 31), generic over the float width
    /// (`f32`/`f64`). Replaces the portable Babylonian fallback in [`Scalar::sqrt`] on-device.
    #[inline]
    fn gpu_sqrt<T: Copy + Default>(x: T) -> T {
        let mut result = T::default();
        unsafe {
            core::arch::asm!(
                "%glsl = OpExtInstImport \"GLSL.std.450\"",
                "%x = OpLoad _ {x}",
                "%r = OpExtInst typeof*{result} %glsl 31 %x",
                "OpStore {result} %r",
                x = in(reg) &x,
                result = in(reg) &mut result,
            );
        }
        result
    }

    /// Hardware fused multiply-add `a * b + c` — GLSL.std.450 `Fma` (opcode 50), generic over
    /// the float width. Replaces the unfused `mul`+`add` of the default [`Scalar::fma`].
    #[inline]
    fn gpu_fma<T: Copy + Default>(a: T, b: T, c: T) -> T {
        let mut result = T::default();
        unsafe {
            core::arch::asm!(
                "%glsl = OpExtInstImport \"GLSL.std.450\"",
                "%a = OpLoad _ {a}",
                "%b = OpLoad _ {b}",
                "%c = OpLoad _ {c}",
                "%r = OpExtInst typeof*{result} %glsl 50 %a %b %c",
                "OpStore {result} %r",
                a = in(reg) &a,
                b = in(reg) &b,
                c = in(reg) &c,
                result = in(reg) &mut result,
            );
        }
        result
    }

    // Each backend op declares the SPIR-V capabilities it needs via `OpCapability` (which
    // rust-gpu hoists to the module header). Because the call sits inside the op, a capability
    // is emitted *only* when that op for that scalar is monomorphized into the module — so a
    // compiled shader requires exactly the capabilities of the backends it actually uses, with
    // no host-side `--capabilities` flag. Capability dependencies are implicit: declaring
    // `GroupNonUniformArithmetic` also pulls in the base `GroupNonUniform` (which the
    // `SubgroupSize` builtin read needs).
    #[inline]
    fn cap_none() {}
    #[inline]
    fn cap_float64() {
        unsafe { core::arch::asm!("OpCapability Float64") }
    }
    #[inline]
    fn cap_int16() {
        unsafe { core::arch::asm!("OpCapability Int16") }
    }
    #[inline]
    fn cap_group_arithmetic() {
        unsafe { core::arch::asm!("OpCapability GroupNonUniformArithmetic") }
    }
    #[inline]
    fn cap_group_vote() {
        unsafe { core::arch::asm!("OpCapability GroupNonUniformVote") }
    }

    /// GPU subgroup execution token. Zero-sized: the warp width is fetched on demand from the
    /// `SubgroupSize` builtin, so there is nothing for the caller to store or pass in.
    #[derive(Clone, Copy, Default)]
    pub struct Subgroup;

    impl Subgroup {
        #[inline]
        pub fn new() -> Self {
            Self
        }
    }

    // Arithmetic is per-invocation scalar (each lane is one invocation); only the cross-lane
    // reductions use subgroup collectives.
    macro_rules! subgroup_backend {
        ($ty:ty, $cap:path) => {
            impl Backend<$ty> for Subgroup {
                type Vector = $ty;
                type Mask = bool;

                #[inline]
                fn lanes(self) -> usize {
                    subgroup_size() as usize
                }
                #[inline]
                fn splat(self, v: $ty) -> $ty {
                    $cap();
                    v
                }
                #[inline]
                fn load(self, s: &[$ty]) -> $ty {
                    // Each invocation owns one element of the distributed register.
                    s[0]
                }
                #[inline]
                fn store(self, v: $ty, s: &mut [$ty]) {
                    s[0] = v;
                }
                #[inline]
                fn add(self, a: $ty, b: $ty) -> $ty {
                    a.add(b)
                }
                #[inline]
                fn sub(self, a: $ty, b: $ty) -> $ty {
                    a.sub(b)
                }
                #[inline]
                fn mul(self, a: $ty, b: $ty) -> $ty {
                    a.mul(b)
                }
                #[inline]
                fn div(self, a: $ty, b: $ty) -> $ty {
                    a.div(b)
                }
                #[inline]
                fn neg(self, a: $ty) -> $ty {
                    a.neg()
                }
                #[inline]
                fn fma(self, a: $ty, b: $ty, c: $ty) -> $ty {
                    gpu_fma(a, b, c)
                }
                #[inline]
                fn sqrt(self, a: $ty) -> $ty {
                    gpu_sqrt(a)
                }
                #[inline]
                fn min(self, a: $ty, b: $ty) -> $ty {
                    a.min(b)
                }
                #[inline]
                fn max(self, a: $ty, b: $ty) -> $ty {
                    a.max(b)
                }
                #[inline]
                fn le(self, a: $ty, b: $ty) -> bool {
                    a.le(b)
                }
                #[inline]
                fn lt(self, a: $ty, b: $ty) -> bool {
                    a.lt(b)
                }
                #[inline]
                fn ge(self, a: $ty, b: $ty) -> bool {
                    a.ge(b)
                }
                #[inline]
                fn gt(self, a: $ty, b: $ty) -> bool {
                    a.gt(b)
                }
                #[inline]
                fn mask_and(self, a: bool, b: bool) -> bool {
                    a & b
                }
                #[inline]
                fn mask_or(self, a: bool, b: bool) -> bool {
                    a | b
                }
                #[inline]
                fn mask_not(self, a: bool) -> bool {
                    !a
                }
                #[inline]
                fn select(self, m: bool, a: $ty, b: $ty) -> $ty {
                    if m { a } else { b }
                }
                // Cross-lane: subgroup collectives over the whole warp.
                #[inline]
                fn any(self, m: bool) -> bool {
                    cap_group_vote();
                    spirv_std::arch::subgroup_any(m)
                }
                #[inline]
                fn all(self, m: bool) -> bool {
                    cap_group_vote();
                    spirv_std::arch::subgroup_all(m)
                }
                #[inline]
                fn reduce_sum(self, v: $ty) -> $ty {
                    cap_group_arithmetic();
                    spirv_std::arch::subgroup_f_add(v)
                }
                #[inline]
                fn reduce_min(self, v: $ty) -> $ty {
                    cap_group_arithmetic();
                    spirv_std::arch::subgroup_f_min(v)
                }
                #[inline]
                fn reduce_max(self, v: $ty) -> $ty {
                    cap_group_arithmetic();
                    spirv_std::arch::subgroup_f_max(v)
                }
            }
        };
    }

    subgroup_backend!(f32, cap_none);
    subgroup_backend!(f64, cap_float64);

    // f16 conversions via the hardware GLSL.std.450 PackHalf2x16 / UnpackHalf2x16 ops
    // (`spirv_std::float`), not `half`'s software bit-twiddling.
    #[inline]
    fn widen_f16(v: f16) -> f32 {
        spirv_std::float::f16_to_f32(v.to_bits() as u32)
    }
    #[inline]
    fn narrow_f16(c: f32) -> f16 {
        f16::from_bits(spirv_std::float::f32_to_f16(c) as u16)
    }
    // bf16 has no native SPIR-V form (no `SPV_KHR_bfloat16` in spirv-std 0.10); `half`'s
    // conversion is a cheap shift (bf16 is just the high 16 bits of an f32), so use it.
    #[inline]
    fn widen_bf16(v: bf16) -> f32 {
        v.to_f32()
    }
    #[inline]
    fn narrow_bf16(c: f32) -> bf16 {
        bf16::from_f32(c)
    }

    // Half precision on the GPU: storage is 16-bit, compute is `f32`. The lane widens to `f32`
    // on `load`/`splat` and narrows on `store`, so all arithmetic and the subgroup collectives
    // run in native `f32` — conversions hit only the memory boundary, never per op. `Vector` is
    // therefore `f32` here (as on the AVX2 F16C path), not the 16-bit storage type.
    macro_rules! subgroup_widen_backend {
        ($ty:ty, $widen:path, $narrow:path) => {
            impl Backend<$ty> for Subgroup {
                type Vector = f32;
                type Mask = bool;

                #[inline]
                fn lanes(self) -> usize {
                    subgroup_size() as usize
                }
                #[inline]
                fn splat(self, v: $ty) -> f32 {
                    cap_int16();
                    $widen(v)
                }
                #[inline]
                fn load(self, s: &[$ty]) -> f32 {
                    cap_int16();
                    $widen(s[0])
                }
                #[inline]
                fn store(self, v: f32, s: &mut [$ty]) {
                    s[0] = $narrow(v);
                }
                #[inline]
                fn add(self, a: f32, b: f32) -> f32 {
                    a + b
                }
                #[inline]
                fn sub(self, a: f32, b: f32) -> f32 {
                    a - b
                }
                #[inline]
                fn mul(self, a: f32, b: f32) -> f32 {
                    a * b
                }
                #[inline]
                fn div(self, a: f32, b: f32) -> f32 {
                    a / b
                }
                #[inline]
                fn neg(self, a: f32) -> f32 {
                    -a
                }
                #[inline]
                fn fma(self, a: f32, b: f32, c: f32) -> f32 {
                    gpu_fma(a, b, c)
                }
                #[inline]
                fn sqrt(self, a: f32) -> f32 {
                    gpu_sqrt(a)
                }
                #[inline]
                fn min(self, a: f32, b: f32) -> f32 {
                    if b < a { b } else { a }
                }
                #[inline]
                fn max(self, a: f32, b: f32) -> f32 {
                    if b > a { b } else { a }
                }
                #[inline]
                fn le(self, a: f32, b: f32) -> bool {
                    a <= b
                }
                #[inline]
                fn lt(self, a: f32, b: f32) -> bool {
                    a < b
                }
                #[inline]
                fn ge(self, a: f32, b: f32) -> bool {
                    a >= b
                }
                #[inline]
                fn gt(self, a: f32, b: f32) -> bool {
                    a > b
                }
                #[inline]
                fn mask_and(self, a: bool, b: bool) -> bool {
                    a & b
                }
                #[inline]
                fn mask_or(self, a: bool, b: bool) -> bool {
                    a | b
                }
                #[inline]
                fn mask_not(self, a: bool) -> bool {
                    !a
                }
                #[inline]
                fn select(self, m: bool, a: f32, b: f32) -> f32 {
                    if m { a } else { b }
                }
                #[inline]
                fn any(self, m: bool) -> bool {
                    cap_group_vote();
                    spirv_std::arch::subgroup_any(m)
                }
                #[inline]
                fn all(self, m: bool) -> bool {
                    cap_group_vote();
                    spirv_std::arch::subgroup_all(m)
                }
                #[inline]
                fn reduce_sum(self, v: f32) -> $ty {
                    cap_group_arithmetic();
                    $narrow(spirv_std::arch::subgroup_f_add(v))
                }
                #[inline]
                fn reduce_min(self, v: f32) -> $ty {
                    cap_group_arithmetic();
                    $narrow(spirv_std::arch::subgroup_f_min(v))
                }
                #[inline]
                fn reduce_max(self, v: f32) -> $ty {
                    cap_group_arithmetic();
                    $narrow(spirv_std::arch::subgroup_f_max(v))
                }
            }
        };
    }

    subgroup_widen_backend!(f16, widen_f16, narrow_f16);
    subgroup_widen_backend!(bf16, widen_bf16, narrow_bf16);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_policy() {
        // unknown / scalar size -> always sequential
        assert_eq!(choose(1000, 0, 4), Execution::Sequential);
        assert_eq!(choose(1000, 1, 4), Execution::Sequential);
        // below threshold (size*factor = 32*4 = 128) -> sequential
        assert_eq!(choose(100, 32, 4), Execution::Sequential);
        // at/above threshold -> subgroup
        assert_eq!(choose(128, 32, 4), Execution::Subgroup);
        assert_eq!(choose(10_000, 32, 4), Execution::Subgroup);
        // factor 0 is treated as 1
        assert_eq!(choose(32, 32, 0), Execution::Subgroup);
    }
}
