//! GPU subgroup backend for the rust-gpu / SPIR-V target: the SPMD gang is a subgroup, one
//! lane per invocation. The `choose` scheduling policy is always compiled; the `Subgroup`
//! backend itself compiles only under the rust-gpu toolchain, not a normal host `cargo build`.

/// How a batch should be executed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Execution {
    /// Too little work to fill a warp: one invocation loops over the items.
    Sequential,
    /// Enough work to distribute the items across the subgroup's lanes.
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
    // Plain multiply: `subgroup_size * fill_factor` is always small (warp width ≤ ~128), so it
    // can't overflow, and a plain `*` avoids the `saturating_mul` intrinsic rust-gpu can't lower.
    let threshold = subgroup_size as usize * fill_factor.max(1) as usize;
    if item_count < threshold {
        Execution::Sequential
    } else {
        Execution::Subgroup
    }
}

// The intrinsic names below target spirv-std's `arch` subgroup ops; pin them to the rust-gpu
// rev used by the shader crate.
#[cfg(target_arch = "spirv")]
pub use device::{
    Coop, MATRIX_SCRATCH_WORDS, MatrixSubgroup, Mode, Plain, Subgroup, dispatch_matrix_subgroup,
    dispatch_subgroup,
};

#[cfg(target_arch = "spirv")]
pub(crate) mod device {
    use super::{Execution, choose};
    use crate::backend::{Backend, ScalarBackend};
    use crate::dispatch::Kernel;
    use crate::matrix::{MatrixBackend, MatrixKernel};
    use crate::scalar::FloatScalar;
    use crate::scalar::Scalar;
    use crate::varying::Gang;
    use half::{bf16, f16};

    /// Device entry point, the GPU analogue of the host [`dispatch`](crate::dispatch).
    ///
    /// With at least `subgroup_size * fill_factor` items the batch is spread across the warp
    /// and the kernel's cross-lane ops become subgroup collectives (`Subgroup`); below that
    /// threshold the collectives don't pay for themselves, so a single invocation loops over
    /// the items on the scalar backend. Either way the kernel body is the one written against
    /// [`Gang`].
    ///
    /// `item_count` is the batch size the calling subgroup is responsible for; `fill_factor` is
    /// the occupancy threshold (`4` is a sane default). The warp width is read from the
    /// `SubgroupSize` builtin, so there is nothing else to pass in.
    #[inline]
    pub fn dispatch_subgroup<T, K>(kernel: K, item_count: usize, fill_factor: u32) -> K::Output
    where
        T: Scalar,
        K: Kernel<T>,
        Subgroup: Backend<T>,
    {
        match choose(item_count, subgroup_size(), fill_factor) {
            Execution::Subgroup => kernel.run(Gang::new(Subgroup::new())),
            Execution::Sequential => kernel.run(Gang::new(ScalarBackend)),
        }
    }

    /// Workgroup memory, in words, each row/column tile op (`fold_rows`, `zip_cols`, …) stages
    /// through on cooperative-matrix builds, split evenly between the workgroup's subgroups:
    /// enough for one `16×16` `f32` accumulator per subgroup with up to four subgroups.
    pub const MATRIX_SCRATCH_WORDS: usize = 1024;

    /// The tile lowering of this build: cooperative matrices when it enables the
    /// `CooperativeMatrixKHR` capability (`f16` tiles; other elements stay lane-distributed
    /// arrays), lane-distributed array tiles otherwise.
    #[cfg(hp_coop_matrix)]
    pub type MatrixSubgroup<const W: usize = 0> = Subgroup<Coop<MATRIX_SCRATCH_WORDS>, W>;
    /// The tile lowering of this build: cooperative matrices when it enables the
    /// `CooperativeMatrixKHR` capability (`f16` tiles; other elements stay lane-distributed
    /// arrays), lane-distributed array tiles otherwise.
    #[cfg(not(hp_coop_matrix))]
    pub type MatrixSubgroup<const W: usize = 0> = Subgroup<Plain, W>;

    /// Device entry point for a [`MatrixKernel`], the GPU twin of
    /// [`dispatch_matrix`](crate::dispatch_matrix): the calling subgroup is the gang and holds
    /// each tile together.
    ///
    /// The same kernel source builds for every device; the shader build picks the lowering.
    /// Enabling the `CooperativeMatrixKHR` capability (and `SPV_KHR_cooperative_matrix`) gives
    /// hardware matrix tiles. Such a module only loads on devices with
    /// `VK_KHR_cooperative_matrix`, so build it once with and once without, and use tile shapes
    /// the device lists in `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`.
    ///
    /// The warp width is matched here, once, and the kernel runs on a backend fixed to it: a
    /// tile op deciding the width for itself would leave the code for every other width beside
    /// it, and drivers keep a tile that some unused width indexes at run time in memory.
    #[inline]
    pub fn dispatch_matrix_subgroup<T, K>(kernel: K) -> K::Output
    where
        T: FloatScalar,
        K: MatrixKernel<T>,
        MatrixSubgroup<32>: MatrixBackend<T>,
        MatrixSubgroup<64>: MatrixBackend<T>,
        MatrixSubgroup<16>: MatrixBackend<T>,
        MatrixSubgroup<8>: MatrixBackend<T>,
        MatrixSubgroup<0>: MatrixBackend<T>,
    {
        match subgroup_size() {
            32 => kernel.run(Gang::new(MatrixSubgroup::<32>::default())),
            64 => kernel.run(Gang::new(MatrixSubgroup::<64>::default())),
            16 => kernel.run(Gang::new(MatrixSubgroup::<16>::default())),
            8 => kernel.run(Gang::new(MatrixSubgroup::<8>::default())),
            _ => kernel.run(Gang::new(MatrixSubgroup::<0>::default())),
        }
    }

    /// Read a `u32` input builtin with a single load. rust-gpu hoists the `Input` variable and
    /// its decoration to module scope, and requires `Generic` for `OpTypePointer` in `asm!`,
    /// inferring the real storage class from the `OpVariable`.
    macro_rules! builtin_u32 {
        ($name:ident, $builtin:literal) => {
            #[inline]
            pub(crate) fn $name() -> u32 {
                let v: u32;
                unsafe {
                    core::arch::asm!(
                        "OpCapability GroupNonUniform",
                        "%u32 = OpTypeInt 32 0",
                        "%ptr = OpTypePointer Generic %u32",
                        "%var = OpVariable %ptr Input",
                        concat!("OpDecorate %var BuiltIn ", $builtin),
                        "{v} = OpLoad %u32 %var",
                        v = out(reg) v,
                    );
                }
                v
            }
        };
    }
    builtin_u32!(subgroup_id, "SubgroupId");
    builtin_u32!(num_subgroups, "NumSubgroups");
    builtin_u32!(subgroup_local_invocation_id, "SubgroupLocalInvocationId");

    // The `SubgroupSize` builtin is per-invocation ground truth, correct even where the subgroup
    // size is variable.
    builtin_u32!(subgroup_size, "SubgroupSize");

    /// Hardware square root, GLSL.std.450 `Sqrt` (opcode 31), generic over the float width.
    /// Replaces the portable Babylonian fallback in [`Scalar::sqrt`] on-device.
    #[inline]
    pub(crate) fn gpu_sqrt<T: Copy + Default>(x: T) -> T {
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

    /// Hardware fused multiply-add `a * b + c`, GLSL.std.450 `Fma` (opcode 50), generic over
    /// the float width. Replaces the unfused `mul`+`add` of the default [`Scalar::fma`].
    #[inline]
    pub(crate) fn gpu_fma<T: Copy + Default>(a: T, b: T, c: T) -> T {
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

    // Each backend op declares the SPIR-V capabilities it needs via `OpCapability` (hoisted to
    // the module header by rust-gpu). The call sits inside the op, so a capability is emitted
    // only when that op is monomorphized into the module: a compiled shader requires exactly
    // what it uses. Declaring `GroupNonUniformArithmetic` also pulls in the base
    // `GroupNonUniform` that the `SubgroupSize` read needs.
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

    /// GPU subgroup execution token. `M` picks the tile lowering: [`Plain`] array tiles, or
    /// [`Coop`] cooperative matrices. `W` is the warp width when the entry point already matched
    /// it, so every tile op is laid out for that width alone; with `0` the width is read from the
    /// `SubgroupSize` builtin on demand.
    #[derive(Clone, Copy, Default)]
    pub struct Subgroup<M = Plain, const W: usize = 0> {
        pub(crate) mode: M,
    }

    /// The warp width: `W` when known at compile time, else the `SubgroupSize` builtin.
    #[inline]
    pub(crate) fn width<const W: usize>() -> usize {
        if const { W != 0 } {
            W
        } else {
            subgroup_size() as usize
        }
    }

    impl Subgroup {
        #[inline]
        pub fn new() -> Self {
            Self { mode: Plain }
        }
    }

    mod sealed {
        pub trait Sealed {}
    }

    /// How a [`Subgroup`] lowers matrix tiles. Sealed: [`Plain`] or [`Coop`].
    pub trait Mode: sealed::Sealed + Copy + 'static {}

    /// Array tiles, held whole by every invocation.
    #[derive(Clone, Copy, Default)]
    pub struct Plain;

    /// Cooperative-matrix tiles; the row/column ops stage through `N` words of workgroup memory.
    /// Zero-sized, so tiles can sit in arrays and locals freely: SPIR-V allows no pointer
    /// inside a composite.
    #[derive(Clone, Copy, Default)]
    pub struct Coop<const N: usize>;

    impl sealed::Sealed for Plain {}
    impl<const N: usize> sealed::Sealed for Coop<N> {}
    impl Mode for Plain {}
    impl<const N: usize> Mode for Coop<N> {}

    // Arithmetic is per-invocation scalar (each lane is one invocation); only the cross-lane
    // reductions use subgroup collectives.
    macro_rules! subgroup_backend {
        ($ty:ty, $cap:path) => {
            impl<M: Mode, const W: usize> Backend<$ty> for Subgroup<M, W> {
                type Vector = $ty;
                type Mask = bool;

                type IVector = u32;
                #[inline]
                fn iload(self, s: &[u32]) -> u32 {
                    s[0]
                }
                #[inline]
                fn istore(self, v: u32, out: &mut [u32]) {
                    out[0] = v;
                }


                #[inline]
                fn lanes(self) -> usize {
                    width::<W>()
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
                    a.wadd(b)
                }
                #[inline]
                fn sub(self, a: $ty, b: $ty) -> $ty {
                    a.wsub(b)
                }
                #[inline]
                fn mul(self, a: $ty, b: $ty) -> $ty {
                    a.wmul(b)
                }
                #[inline]
                fn div(self, a: $ty, b: $ty) -> $ty {
                    a / b
                }
                #[inline]
                fn neg(self, a: $ty) -> $ty {
                    crate::scalar::Scalar::neg(a)
                }
                #[inline]
                fn fma(self, a: $ty, b: $ty, c: $ty) -> $ty {
                    gpu_fma(a, b, c)
                }
                            #[inline(always)]
                fn madd(self, a: $ty, b: $ty, acc: $ty) -> $ty {
                    <Self as Backend<$ty>>::fma(self, a, b, acc)
                }
    #[inline]
                fn sqrt(self, a: $ty) -> $ty {
                    gpu_sqrt(a)
                }
                #[inline]
                fn min(self, a: $ty, b: $ty) -> $ty {
                    crate::scalar::Scalar::min(a, b)
                }
                #[inline]
                fn max(self, a: $ty, b: $ty) -> $ty {
                    crate::scalar::Scalar::max(a, b)
                }
                #[inline]
                fn le(self, a: $ty, b: $ty) -> bool {
                    a <= b
                }
                #[inline]
                fn lt(self, a: $ty, b: $ty) -> bool {
                    a < b
                }
                #[inline]
                fn ge(self, a: $ty, b: $ty) -> bool {
                    a >= b
                }
                #[inline]
                fn gt(self, a: $ty, b: $ty) -> bool {
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

                // One invocation per lane, so a varying-index gather is this invocation reading
                // its own `base[idx]` — a single buffer access, no subgroup collective. Overriding
                // the trait default keeps it a clean `OpAccessChain`+`OpLoad` instead of the
                // default's dynamically-indexed stack array, which rust-gpu spills to memory.
                // Cross-invocation duplicate scatter indices race (as any GPU scatter does); a
                // histogram-style accumulation needs atomics, not this.
                #[inline]
                fn gather(self, base: &[$ty], idx: u32) -> $ty {
                    base[idx as usize]
                }
                #[inline]
                fn gather_masked(self, base: &[$ty], idx: u32, m: bool, fill: $ty) -> $ty {
                    if m { base[idx as usize] } else { fill }
                }
                #[inline]
                fn scatter(self, out: &mut [$ty], idx: u32, v: $ty) {
                    out[idx as usize] = v;
                }
                #[inline]
                fn scatter_masked(self, out: &mut [$ty], idx: u32, v: $ty, m: bool) {
                    if m {
                        out[idx as usize] = v;
                    }
                }
                #[inline]
                fn igather(self, base: &[u32], idx: u32) -> u32 {
                    base[idx as usize]
                }
                #[inline]
                fn igather_masked(self, base: &[u32], idx: u32, m: bool, fill: u32) -> u32 {
                    if m { base[idx as usize] } else { fill }
                }
                #[inline]
                fn iscatter(self, out: &mut [u32], idx: u32, v: u32) {
                    out[idx as usize] = v;
                }
                #[inline]
                fn iscatter_masked(self, out: &mut [u32], idx: u32, v: u32, m: bool) {
                    if m {
                        out[idx as usize] = v;
                    }
                }
            }
        };
    }

/// Integer elements on the subgroup backend: per-invocation scalar ops with the same warp
    /// collectives for the cross-lane surface (`OpGroupNonUniform` integer arithmetic).
    macro_rules! subgroup_int_backend {
        ($ty:ty, $umin:ident, $umax:ident) => {
            impl<M: Mode, const W: usize> Backend<$ty> for Subgroup<M, W> {
                type Vector = $ty;
                type Mask = bool;

                type IVector = u32;
                #[inline]
                fn iload(self, s: &[u32]) -> u32 {
                    s[0]
                }
                #[inline]
                fn istore(self, v: u32, out: &mut [u32]) {
                    out[0] = v;
                }

                #[inline]
                fn lanes(self) -> usize {
                    width::<W>()
                }
                #[inline]
                fn splat(self, v: $ty) -> $ty {
                    v
                }
                #[inline]
                fn load(self, s: &[$ty]) -> $ty {
                    s[0]
                }
                #[inline]
                fn store(self, v: $ty, s: &mut [$ty]) {
                    s[0] = v;
                }
                #[inline]
                fn add(self, a: $ty, b: $ty) -> $ty {
                    a.wadd(b)
                }
                #[inline]
                fn sub(self, a: $ty, b: $ty) -> $ty {
                    a.wsub(b)
                }
                #[inline]
                fn mul(self, a: $ty, b: $ty) -> $ty {
                    a.wmul(b)
                }
                #[inline]
                fn neg(self, a: $ty) -> $ty {
                    crate::scalar::Scalar::neg(a)
                }
                #[inline]
                fn abs(self, a: $ty) -> $ty {
                    crate::scalar::Scalar::abs(a)
                }
                #[inline]
                fn min(self, a: $ty, b: $ty) -> $ty {
                    crate::scalar::Scalar::min(a, b)
                }
                #[inline]
                fn max(self, a: $ty, b: $ty) -> $ty {
                    crate::scalar::Scalar::max(a, b)
                }
                #[inline]
                fn le(self, a: $ty, b: $ty) -> bool {
                    a <= b
                }
                #[inline]
                fn lt(self, a: $ty, b: $ty) -> bool {
                    a < b
                }
                #[inline]
                fn ge(self, a: $ty, b: $ty) -> bool {
                    a >= b
                }
                #[inline]
                fn gt(self, a: $ty, b: $ty) -> bool {
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
                fn select(self, m: bool, a: $ty, b: $ty) -> $ty {
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
                fn reduce_sum(self, v: $ty) -> $ty {
                    cap_group_arithmetic();
                    spirv_std::arch::subgroup_i_add(v)
                }
                #[inline]
                fn reduce_min(self, v: $ty) -> $ty {
                    cap_group_arithmetic();
                    spirv_std::arch::$umin(v)
                }
                #[inline]
                fn reduce_max(self, v: $ty) -> $ty {
                    cap_group_arithmetic();
                    spirv_std::arch::$umax(v)
                }
                #[inline]
                fn shl(self, a: $ty, k: u32) -> $ty {
                    num_traits::PrimInt::unsigned_shl(a, k)
                }
                #[inline]
                fn shr(self, a: $ty, k: u32) -> $ty {
                    a >> (k as usize)
                }
                #[inline]
                fn bit_and(self, a: $ty, b: $ty) -> $ty {
                    a & b
                }
                #[inline]
                fn bit_or(self, a: $ty, b: $ty) -> $ty {
                    a | b
                }
                #[inline]
                fn bit_xor(self, a: $ty, b: $ty) -> $ty {
                    a ^ b
                }
                #[inline]
                fn bit_not(self, a: $ty) -> $ty {
                    !a
                }

                // Per-invocation indexed access; see the float macro for the rationale.
                #[inline]
                fn gather(self, base: &[$ty], idx: u32) -> $ty {
                    base[idx as usize]
                }
                #[inline]
                fn gather_masked(self, base: &[$ty], idx: u32, m: bool, fill: $ty) -> $ty {
                    if m { base[idx as usize] } else { fill }
                }
                #[inline]
                fn scatter(self, out: &mut [$ty], idx: u32, v: $ty) {
                    out[idx as usize] = v;
                }
                #[inline]
                fn scatter_masked(self, out: &mut [$ty], idx: u32, v: $ty, m: bool) {
                    if m {
                        out[idx as usize] = v;
                    }
                }
                #[inline]
                fn igather(self, base: &[u32], idx: u32) -> u32 {
                    base[idx as usize]
                }
                #[inline]
                fn igather_masked(self, base: &[u32], idx: u32, m: bool, fill: u32) -> u32 {
                    if m { base[idx as usize] } else { fill }
                }
                #[inline]
                fn iscatter(self, out: &mut [u32], idx: u32, v: u32) {
                    out[idx as usize] = v;
                }
                #[inline]
                fn iscatter_masked(self, out: &mut [u32], idx: u32, v: u32, m: bool) {
                    if m {
                        out[idx as usize] = v;
                    }
                }
            }
        };
    }

    subgroup_int_backend!(u32, subgroup_u_min, subgroup_u_max);
    subgroup_int_backend!(i32, subgroup_s_min, subgroup_s_max);

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

    // Half precision on the GPU: 16-bit storage, f32 compute. Widen on `load`/`splat`, narrow
    // on `store`; conversions hit only the memory boundary, so `Vector` is `f32`, not the
    // 16-bit storage type.
    macro_rules! subgroup_widen_backend {
        ($ty:ty, $widen:path, $narrow:path) => {
            impl<M: Mode, const W: usize> Backend<$ty> for Subgroup<M, W> {
                type Vector = f32;
                type Mask = bool;

                type IVector = u32;
                #[inline]
                fn iload(self, s: &[u32]) -> u32 {
                    s[0]
                }
                #[inline]
                fn istore(self, v: u32, out: &mut [u32]) {
                    out[0] = v;
                }


                #[inline]
                fn lanes(self) -> usize {
                    width::<W>()
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
                            #[inline(always)]
                fn madd(self, a: f32, b: f32, acc: f32) -> f32 {
                    <Self as Backend<$ty>>::fma(self, a, b, acc)
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

                // 16-bit storage, f32 register: gather widens at the memory boundary, scatter
                // narrows, matching this backend's `load`/`store`. See the float macro for the
                // per-invocation rationale.
                #[inline]
                fn gather(self, base: &[$ty], idx: u32) -> f32 {
                    cap_int16();
                    $widen(base[idx as usize])
                }
                #[inline]
                fn gather_masked(self, base: &[$ty], idx: u32, m: bool, fill: f32) -> f32 {
                    if m {
                        cap_int16();
                        $widen(base[idx as usize])
                    } else {
                        fill
                    }
                }
                #[inline]
                fn scatter(self, out: &mut [$ty], idx: u32, v: f32) {
                    out[idx as usize] = $narrow(v);
                }
                #[inline]
                fn scatter_masked(self, out: &mut [$ty], idx: u32, v: f32, m: bool) {
                    if m {
                        out[idx as usize] = $narrow(v);
                    }
                }
                #[inline]
                fn igather(self, base: &[u32], idx: u32) -> u32 {
                    base[idx as usize]
                }
                #[inline]
                fn igather_masked(self, base: &[u32], idx: u32, m: bool, fill: u32) -> u32 {
                    if m { base[idx as usize] } else { fill }
                }
                #[inline]
                fn iscatter(self, out: &mut [u32], idx: u32, v: u32) {
                    out[idx as usize] = v;
                }
                #[inline]
                fn iscatter_masked(self, out: &mut [u32], idx: u32, v: u32, m: bool) {
                    if m {
                        out[idx as usize] = v;
                    }
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
