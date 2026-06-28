# Cooperative matrices from rust-gpu

How to emit Vulkan **cooperative matrix** instructions — the `SPV_KHR_cooperative_matrix`
extension and the older `SPV_NV_cooperative_matrix` — from Rust compiled with rust-gpu, as
used by `spmd-test` / the rev pinned in `Cargo.toml` (`spirv-std =0.10.0-alpha.1`, also
verified against git rev `36e3348`).

A cooperative matrix is a matrix whose storage and arithmetic are spread across a whole
**subgroup** (warp): the lanes cooperate to hold one M×N tile and to compute `A·B + C` on it
in a single hardware instruction (NVIDIA tensor cores, AMD WMMA, Intel XMX, Apple
simdgroup_matrix). It is the SPIR-V equivalent of CUDA `wmma` / HLSL SM6.8
`WaveMatrix` / Metal `simdgroup_matrix`.

## Status: does not compile on rust-gpu yet (empirical, 2026-06-28)

The asm blocks below are **grammar-correct but do not compile** — verified by building
`spmd-test` with the exact KHR example against **both** `spirv-std 0.10.0-alpha.1` (the pinned
release) and the **git rev `36e3348`** named above, targeting `spirv-unknown-vulkan1.3`. Both ICE
identically:

```
internal error: entered unreachable code: OpTypeCooperativeMatrixKHR is reserved for SPV_KHR_cooperative_matrix
```

Root cause, traced in `rustc_codegen_spirv`: the codegen's internal `SpirvType` enum **has no
cooperative-matrix variant** (`spirv_type.rs` — no `Cooperative*`), and `instruction_signatures`
stubs every `*CooperativeMatrix*` opcode as `reserved!(…)`, which is an `unreachable!`. So while
the inline-`asm!` *parser* understands the operand kinds (`CooperativeMatrixLayout`/`Use`/…), the
codegen cannot lower the type: declaring `OpTypeCooperativeMatrixKHR` in `asm!` falls through the
`OpType*` match unregistered, and the first reference to it hits the `reserved!` `unreachable!`.
Giving explicit result types (this doc's point 2) does **not** avoid it — the type *declaration*
itself triggers the path. `options(nostack)` is also rejected (`asm flags not supported: NOSTACK`).

**Conclusion:** cooperative matrices need an upstream rust-gpu codegen feature (a
`SpirvType::CooperativeMatrix` + real `instruction_signatures`), not just hand-written `asm!`.
Until a rust-gpu version ships that, this path is blocked. The rest of this file remains a correct
operand-grammar reference for when it lands. (The `spmd` non-coop GPU fallback — per-invocation
scalar tiles in `MatrixBackend for Subgroup` — is unaffected and remains the working path.)

## What rust-gpu actually gives you

**Inline SPIR-V assembly only.** There is no typed wrapper, no intrinsic, no `spirv-std`
helper — grep the crate for "cooperative" and you get nothing. What exists in the compiler
(`rustc_codegen_spirv`) is:

* the opcodes are recognised (`OpTypeCooperativeMatrixKHR`, `OpCooperativeMatrixLoadKHR`,
  `…StoreKHR`, `…MulAddKHR`, `…LengthKHR`, and the `…NV` set), and
* the inline-`asm!` parser understands their special operand kinds
  (`CooperativeMatrixOperands` bit-flags, `CooperativeMatrixLayout`, `CooperativeMatrixUse`).

So you reach the feature exactly the way `subgroup.rs` reaches `OpGroupNonUniformFAdd` and
the GLSL.std.450 `Fma`/`Sqrt`: a hand-written `core::arch::asm!` block, with the capability
declared inline and hoisted into the module header by rust-gpu.

Two consequences fall out of the "no API" state, and both shape the code below:

1. **The matrix type is opaque and has no Rust value.** `OpTypeCooperativeMatrixKHR` is not
   expressible as a Rust type, so a loaded matrix cannot live in a Rust local that survives
   between `asm!` blocks — only as an `%id` *inside* one block. The working unit is therefore
   one **fused** `asm!` that loads, multiplies, and stores in a single sequence. You pass it
   raw pointers in and out; the matrix `%id`s never escape.
2. **Result-type inference is deliberately disabled for these ops** (they are tagged
   `reserved!` in `spirv_type_constraints.rs`, which is an `unreachable!`). That path only
   fires for asm operands written with the `_`/inference placeholder. **Always give an
   explicit result type** (`%tA`, `%tC`, …) on every cooperative-matrix instruction — which
   you must do anyway — and you never touch it.

## It does *not* fit spmd's `Backend` / `Simd` model

spmd's lane abstraction is **element-wise per invocation** (`Vector = T`, one lane = one
invocation). A cooperative matrix is the opposite shape: the *whole subgroup* cooperatively
owns *one tile*, and `MulAdd` is a single op over that tile, not a per-lane operation. There
is no sensible `Backend<T>` method for it, and you cannot build one out of `splat`/`add`/…

Treat it as a **separate tile-level primitive**, parallel to the `Subgroup` backend rather
than part of it — e.g. a `coop` module exposing `matmul_tile(...)` entry helpers. The
subgroup execution scope is the one thing it shares with the `Subgroup` backend (both are
warp-wide), which is why it belongs in this crate next to it, not why it belongs *inside* it.

## The KHR path (`SPV_KHR_cooperative_matrix`) — preferred

Operand layout, straight from the SPIR-V grammar rust-gpu uses (`Scope`, `Rows`, `Columns`,
`Use`, `MemoryLayout`, `Stride` are all **`<id>`s of `OpConstant`s**, not inline literals —
the only inline literal is the optional trailing `CooperativeMatrixOperands` bit-mask on
`MulAdd`):

```
OpTypeCooperativeMatrixKHR   Result   ComponentType  Scope  Rows  Columns  Use
OpCooperativeMatrixLoadKHR   ResultTy Result  Pointer  MemoryLayout  [Stride]  [MemoryOperand]
OpCooperativeMatrixStoreKHR           Pointer Object   MemoryLayout  [Stride]  [MemoryOperand]
OpCooperativeMatrixMulAddKHR ResultTy Result  A  B  C   [CooperativeMatrixOperands]
OpCooperativeMatrixLengthKHR ResultTy Result  Type
```

Enumerant values you need: `Scope::Subgroup = 3`; `CooperativeMatrixUse` `MatrixAKHR=0`,
`MatrixBKHR=1`, `MatrixAccumulatorKHR=2`; `CooperativeMatrixLayout` `RowMajorKHR=0`,
`ColumnMajorKHR=1`.

A fused 16×16×16 tile, `f16` inputs accumulated in `f32` (the most widely supported combo —
NVIDIA/AMD/Intel all expose it):

```rust
#[cfg(target_arch = "spirv")]
#[inline]
unsafe fn coop_mma_16x16x16_f16_f32(
    a: *const f16, // 16×16, row-major, row stride = 16 elems
    b: *const f16, // 16×16, row-major, row stride = 16 elems
    c: *const f32, // 16×16 accumulator in
    d: *mut f32,   // 16×16 result out
) {
    core::arch::asm!(
        // Module-scope decls — rust-gpu hoists these into the header and de-dups them.
        "OpCapability Float16",
        "OpCapability CooperativeMatrixKHR",
        "OpExtension \"SPV_KHR_cooperative_matrix\"",

        // Component + index scalar types.
        "%f16 = OpTypeFloat 16",
        "%f32 = OpTypeFloat 32",
        "%u32 = OpTypeInt 32 0",

        // Constant ids feeding the matrix-type and load/store operands.
        "%scope = OpConstant %u32 3",   // Scope::Subgroup
        "%dim   = OpConstant %u32 16",  // M = N = K = 16, reused as stride
        "%row   = OpConstant %u32 0",   // CooperativeMatrixLayout::RowMajorKHR
        "%useA  = OpConstant %u32 0",   // MatrixAKHR
        "%useB  = OpConstant %u32 1",   // MatrixBKHR
        "%useC  = OpConstant %u32 2",   // MatrixAccumulatorKHR

        // Three matrix types: A is M×K, B is K×N, accumulator/result is M×N.
        "%tA = OpTypeCooperativeMatrixKHR %f16 %scope %dim %dim %useA",
        "%tB = OpTypeCooperativeMatrixKHR %f16 %scope %dim %dim %useB",
        "%tC = OpTypeCooperativeMatrixKHR %f32 %scope %dim %dim %useC",

        // Load from the raw pointers (row-major, stride = 16 elements).
        "%mA = OpCooperativeMatrixLoadKHR %tA {a} %row %dim",
        "%mB = OpCooperativeMatrixLoadKHR %tB {b} %row %dim",
        "%mC = OpCooperativeMatrixLoadKHR %tC {c} %row %dim",

        // D = A·B + C. Float operands → omit the trailing CooperativeMatrixOperands.
        "%mD = OpCooperativeMatrixMulAddKHR %tC %mA %mB %mC",

        // Store the result tile.
        "OpCooperativeMatrixStoreKHR {d} %mD %row %dim",

        a = in(reg) a,
        b = in(reg) b,
        c = in(reg) c,
        d = in(reg) d,
        options(nostack),
    );
}
```

Reading it operand-by-operand: each `OpConstant %u32 …` materialises the dimension / scope /
layout / use code the type and memory ops need as `<id>`s; the three `OpTypeCooperativeMatrixKHR`
build the A/B/accumulator types; each `OpCooperativeMatrixLoadKHR` takes `(ResultType,
Pointer, MemoryLayout, Stride)`; `OpCooperativeMatrixMulAddKHR` takes `(ResultType, A, B, C)`
and returns the accumulator-typed result; `OpCooperativeMatrixStoreKHR` takes `(Pointer,
Object, MemoryLayout, Stride)` and produces no result.

**Integer matmul** (e.g. `i8·i8 → i32`, the other broadly-supported combo) is the same
shape, but the signedness of each operand lives in the trailing bit-mask, so the `MulAdd`
becomes:

```
"%mD = OpCooperativeMatrixMulAddKHR %tC %mA %mB %mC \
       MatrixASignedComponentsKHR|MatrixBSignedComponentsKHR|MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR",
```

Add `SaturatingAccumulationKHR` to that mask for saturating integer accumulation. The token
names are exactly those in the `COOPERATIVE_MATRIX_OPERANDS` table in
`rustc_codegen_spirv/src/builder/spirv_asm.rs`.

## The NV path (`SPV_NV_cooperative_matrix`) — legacy

Use this only for hardware/drivers predating the KHR extension (very old Turing-era Vulkan
stacks). The differences from KHR:

* the matrix type has **no `Use` operand** — `(ComponentType, Scope, Rows, Columns)` only;
* load/store carry **`Stride` then a `Column Major` boolean `<id>`** (an `OpConstantFalse`
  for row-major), in place of the KHR `MemoryLayout`+`Stride`;
* `MulAdd` has **no operands bit-mask** — signedness is implied by the component types.

```
OpTypeCooperativeMatrixNV   Result   ComponentType  Scope  Rows  Columns
OpCooperativeMatrixLoadNV   ResultTy Result  Pointer  Stride  ColumnMajor  [MemoryOperand]
OpCooperativeMatrixStoreNV           Pointer Object   Stride  ColumnMajor  [MemoryOperand]
OpCooperativeMatrixMulAddNV ResultTy Result  A  B  C
```

```rust
#[cfg(target_arch = "spirv")]
#[inline]
unsafe fn coop_mma_16x16x16_f16_f32_nv(
    a: *const f16, b: *const f16, c: *const f32, d: *mut f32,
) {
    core::arch::asm!(
        "OpCapability Float16",
        "OpCapability CooperativeMatrixNV",
        "OpExtension \"SPV_NV_cooperative_matrix\"",

        "%f16  = OpTypeFloat 16",
        "%f32  = OpTypeFloat 32",
        "%u32  = OpTypeInt 32 0",
        "%bool = OpTypeBool",

        "%scope = OpConstant %u32 3",      // Subgroup
        "%dim   = OpConstant %u32 16",
        "%colm  = OpConstantFalse %bool",  // row-major

        "%tA = OpTypeCooperativeMatrixNV %f16 %scope %dim %dim",
        "%tB = OpTypeCooperativeMatrixNV %f16 %scope %dim %dim",
        "%tC = OpTypeCooperativeMatrixNV %f32 %scope %dim %dim",

        // Load: (ResultType, Pointer, Stride, ColumnMajor).
        "%mA = OpCooperativeMatrixLoadNV %tA {a} %dim %colm",
        "%mB = OpCooperativeMatrixLoadNV %tB {b} %dim %colm",
        "%mC = OpCooperativeMatrixLoadNV %tC {c} %dim %colm",

        "%mD = OpCooperativeMatrixMulAddNV %tC %mA %mB %mC",

        // Store: (Pointer, Object, Stride, ColumnMajor).
        "OpCooperativeMatrixStoreNV {d} %mD %dim %colm",

        a = in(reg) a, b = in(reg) b, c = in(reg) c, d = in(reg) d,
        options(nostack),
    );
}
```

## Wiring it into a compute entry point

Same structure as the `op_*` entry points in `spmd-test/src/lib.rs`: a `#[spirv(compute(...))]`
function with storage buffers, indexing unchecked (rust-gpu has no panic machinery). The one
real constraint is **execution scope** — `Scope::Subgroup` means an *entire subgroup* must
reach the `MulAdd` in uniform control flow, so launch with `threads()` a multiple of the
hardware subgroup size (32 on NVIDIA, 32/64 on AMD) and don't branch around it per lane.

```rust
#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(32)))]
pub fn mma_tile(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] d: &mut [f32],
) {
    unsafe { coop_mma_16x16x16_f16_f32(a.as_ptr(), b.as_ptr(), c.as_ptr(), d.as_mut_ptr()) };
}
```

`a.as_ptr()` hands the load a pointer into the storage buffer; rust-gpu types it and the load
gathers the tile per `MemoryLayout`+`Stride`. The buffers must be at least tile-sized
(16×16); the host harness sizes them to the dispatch, as the existing entry points assume.

## Capabilities & host-side requirements

Because the matrix elements are 16-bit and live in a storage buffer, you need **more than the
cooperative-matrix capability** — the same 16-bit-storage plumbing any `f16` buffer shader
needs. Declare inline (alongside the two `OpCapability` lines already in the examples), or via
`spirv-builder`'s `.capability(...)`:

* `Float16` — `OpTypeFloat 16` (already in the examples).
* `CooperativeMatrixKHR` (or `CooperativeMatrixNV`).
* `StorageBuffer16BitAccess` + `OpExtension "SPV_KHR_16bit_storage"` — to read `f16` out of a
  storage buffer at all. (Drop this if you stage the tile through `Workgroup` shared memory in
  `f32` instead.)
* For integer matmul: `Int8` and/or `Int16` for the component types.

Build it the way `spmd-test` documents, but raise the target environment so the validator
accepts the extension (cooperative matrix needs a Vulkan 1.3 / SPIR-V 1.6-class target):

```
cargo gpu build --shader-crate ./spmd-test \
  --spirv-builder-version 0.10.0-alpha.1 \
  --target spirv-unknown-vulkan1.3
```

On the **host** (wgpu / ash), before any of this runs on device:

* Enable the device extension `VK_KHR_cooperative_matrix` (it pulls in
  `VK_KHR_vulkan_memory_model`), and enable the `cooperativeMatrix` feature in
  `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`.
* **Query, don't assume, the supported shapes.** Call
  `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` and pick an `(MSize, NSize, KSize,
  AType, BType, CType, ResultType, scope)` tuple it reports. Hardware only implements a small
  set (commonly 16×16×16 and 8×8×… variants); a tile shape the device doesn't list will fail
  to dispatch. Hard-coding 16×16×16 `f16→f32` works on most current discrete GPUs but is not
  guaranteed — gate it on the query.
* Enable `VK_KHR_16bit_storage` if you keep `f16` in the buffers (matching
  `StorageBuffer16BitAccess` above).

## Validate the output — these examples are unverified by construction

There is no compiler type-checking behind raw `asm!`, and this crate can't compile a SPIR-V
shader in a normal `cargo build`, so treat the blocks above as a **correct-by-the-grammar
starting point, not tested code**. Verify the same way `spmd-test` verifies its subgroup
shaders:

1. `cargo gpu build … --target spirv-unknown-vulkan1.3` — forces the backend code into real
   SPIR-V (it is otherwise dead-code-eliminated).
2. Run `spirv-val --target-env vulkan1.3` on the emitted module — the validator is the real
   arbiter of operand correctness (constant types, storage classes, scope uniformity, that
   the chosen shape is internally consistent). Most mistakes in hand-written cooperative-matrix
   asm surface here.
3. `spirv-dis` and confirm the `OpCooperativeMatrix*` instructions and the hoisted
   `OpCapability`/`OpExtension` look as intended.
4. Only then dispatch on a device against a CPU reference GEMM (cf. `tests/gemm_parity.rs`).

The two spots most likely to need a tweak for your target: the **pointer storage class**
the load accepts (storage-buffer vs. workgroup), and the exact **tile shape** — both are
driver-reported, so let `spirv-val` and the properties query drive the final values rather
than the constants picked here.
