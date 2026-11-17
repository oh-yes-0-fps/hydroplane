# Vendor-specific Vulkan compute features

What NVIDIA, AMD, Intel and Apple (MoltenVK) expose beyond the baseline the GPU backend uses
today, and what each would take under the forked rust-gpu 0.11
(`oh-yes-0-fps/rust-gpu@coop-matrix-0.11`). Surveyed 2026-11-13. Items marked unverified had no
primary source.

## Fork limits that decide feasibility

- New SPIR-V reaches Rust only through inline `asm!`. The asm parser uses rspirv 0.13
  (`spirv 0.4.0+sdk-1.4.341`); SPIR-T 0.4 vendors SPIRV-Headers `2acb319` (June 2024), and its
  reader rejects unknown opcodes (`read.rs:317`). A newer op needs both grammars bumped.
- No instruction may take a function id. That rules out `OpCooperativeMatrixReduce*` and
  `OpCooperativeMatrixPerElementOp*`, in both the EXT and NV forms.
- `OpTypeFloat` with an `FPEncoding` operand has no `spirv_asm.rs` arm, and `SpirvType::Float`
  carries no encoding, so bf16 and fp8 component types are unreachable until the fork models them.
- Cooperative-matrix scope is hard-coded to Subgroup (`abi.rs`).
- Fixed in the fork:
  - Arrays unsize to slices and slices sub-slice (`&arr[..]`, `&s[a..]`) in Function, Workgroup
    and StorageBuffer memory, so tiles load from workgroup arrays with the ordinary `&[T]` API.
  - Optimized builds treat panics as unreachable, so bounds checks fold away; unoptimized builds
    keep them. With the checks, a 64×64-per-subgroup coop GEMM on the 5090 ran at 35 TFLOP/s
    instead of 158.

## The RTX 5090 (svc01, driver 595.99.02)

`hydroplane-whatcompute` reports these cooperative-matrix shapes (all at Subgroup scope):
- f16·f16 accumulating to f16 or f32: 16×16×16, 16×8×16, 16×8×8;
- bf16·bf16→f32: 16×16×16;
- i8/u8: 16×16×32 and 16×8×32;
- fp8 E4M3/E5M2 accumulating to f16 or f32: 16×16×32.

There is no f32·f32 shape, so f32 and f64 tiles always take the lane-distributed array path there.
The subgroup size is fixed at 32. `cooperative_matrix2` and `cooperative_vector` are present.

## Ranked opportunities

1. **Stage A/B through workgroup memory** (tiled GEMM). NVIDIA shows a naive coopmat1 loop at
   ~8 TFLOPS against ~98 TFLOPS blocked on an RTX 4070
   ([Bolz, Vulkanised 2025, slides 5–6](https://www.vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf)).
   Tiles now load from workgroup arrays. On the 5090, register blocking alone reaches
   158 TFLOP/s (unstaged, 64×64 per subgroup). A staged kernel with scalar f16 copies reaches only
   ~10 TFLOP/s, because the copy loop is latency-bound. Staging pays off only with vectorized
   copies and double buffering, and probably matters more on GPUs with weaker caches than NVIDIA's.
2. **Shape-portable tiles chosen on the host.** Intel never reports 16×16. ANV reports 8×8×16 on
   Xe-HPG (Arc A-series, ARL-H) and 8×16×16 on Xe2 (Battlemage, Lunar Lake); int8 uses K=32
   ([anv_physical_device.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/main/src/intel/vulkan/anv_physical_device.c)).
   A kernel hard-coded to 16×16×16 gets no XMX path on Intel. Build one module per shape and pick
   from whatcompute's `shapes`.
3. **Pin `requiredSubgroupSize` per pipeline.**
   - RADV compiles a cooperative-matrix shader with full subgroups and a local size above 32 as
     wave64 ([radv_shader.c](https://gitlab.freedesktop.org/mesa/mesa/-/raw/main/src/amd/vulkan/radv_shader.c)).
     WMMA works in both wave32 and wave64, which use different accumulator layouts
     ([GPUOpen](https://gpuopen.com/learn/wmma_on_rdna3/)).
   - Intel lowers at SIMD8, 16 or 32.
   - The lane-distributed array path specialises on widths 8/16/32/64, so a pinned width always
     hits a fast copy.
   - The speedup is unverified; measure it.
4. **f16 accumulators and i8/u8→i32 tiles, plus `OpSDot`/`OpUDot`.** Reachable today: the parser
   accepts the `CooperativeMatrixOperands` signedness mask. f16 accumulation exists on RADV and on
   Xe2, but not on Xe-HPG. Needs element types the `Scalar` set does not have yet.
5. **Maximal reconvergence** (`MaximallyReconvergesKHR`). It gives a formal guarantee for the
   "gang = subgroup" model after divergent code. Supported on NVIDIA 538.31+, RADV, ANV and
   MoltenVK. The fork needs a one-line entry-point attribute in `symbols.rs`.
6. **`VK_EXT_cooperative_matrix_maintenance1`.**
   - `GetCoordinate` would replace the workgroup staging the row/column tile ops use.
     `ConvertUse` would feed an accumulator into the next A operand without a store and reload.
   - Supported on NVIDIA 595.44.14+ and RADV GFX11+; ANV does not expose it.
   - Needs both grammar bumps. Neither op takes a function id.
7. **bf16 tiles.** Supported on the 5090, ANV Xe-HPG/Xe2 and RADV RDNA4 (RDNA3 only behind an
   experimental flag). Needs FPEncoding support in three fork layers.
8. **fp8 tiles.** Supported on the 5090 and RADV RDNA4/GFX11.7. Same fork work as bf16; low
   relevance to the solvers.
9. **Not worth it under these limits:** NV cooperative vectors, the coopmat2 tensor and
   per-element ops, and buffer device address from asm.

## What ThunderKittens and HipKittens suggest

Reviewed 2026-11-15 ([ThunderKittens](https://github.com/HazyResearch/ThunderKittens),
[HipKittens](https://github.com/HazyResearch/HipKittens)).

- **Done:** register blocks. `Tiles::mma_block` runs one k loop over a block of tiles, and the
  lane-distributed path gives each lane a 2D block of each accumulator. On the 5090, array-tile
  GEMM at 64×64 per subgroup went from 6.8 to ~30 TFLOP/s (f16→f32, 4096³).
- **Applicable, not done:**
  - Staging through workgroup memory with conflict-free (swizzled or padded) layouts, 128-bit copies
    and double buffering; it needs a cooperative-matrix load from a `uvec4` workgroup array.
  - Grouped workgroup order for L2 reuse (TK `SUPER_M`, HK `WGM`), for operands larger than L2.
  - Persistent grids.
- **Not reachable from Vulkan:** TMA / `cp.async` / buffer-to-LDS loads, WGMMA and tcgen05,
  producer/consumer warp specialisation with register reallocation. Nor HipKittens' `s_setprio`
  ping-pong, register pinning and XCD swizzle: those target CDNA, which has no Vulkan driver.

## Per vendor

| | NVIDIA Turing→Blackwell | AMD RDNA2/3/4 | Intel Xe-HPG / Xe2 | Apple (MoltenVK) |
|---|---|---|---|---|
| Coop-matrix shapes | KHR since 532.28; 5090 listed above | RDNA2: none. RDNA3/4: 16×16×16 for fp16, bf16, i8 and fp8 | 8×8×16 (Xe-HPG), 8×16×16 (Xe2); Meteor Lake none | None. A WIP MoltenVK PR (#2753) would add 8×8×8 |
| bf16 / fp8 | 572.91 / 573.38 | bf16 RDNA4, fp8 GFX11.7+ | bf16 on both; no fp8 | No |
| Subgroup size | 32 | 32–64 | 8–32 (Xe-HPG), 16–32 (Xe2) | 32 |
| Other | coopmat2, coop vector, maint1 | maint1; `INTEL_shader_integer_functions2` | `INTEL_shader_integer_functions2` | Fast math defaults on (`MVK_CONFIG_FAST_MATH_ENABLED`); set it to 0 for oracle checks |

## Host memory

whatcompute already classifies ReBAR, Window and Unified memory. On ReBAR and unified devices
(iGPUs, Apple), mapping the device-local host-visible heap directly skips the staging copies.
svc01's 5090 has ReBAR off: only a 0.2 GiB window is host-visible.

## whatcompute gaps

- bf16, fp8 and cooperative vectors are detected only by extension name. It does not read
  `shaderBFloat16CooperativeMatrix` or `shaderFloat8CooperativeMatrix`.
- `EXT_cooperative_matrix_maintenance1` is not detected.
- The NV flexible-dimension properties are not queried.
- `cooperativeMatrixSupportedStages`, `requiredSubgroupSizeStages`, maximal reconvergence,
  `float_controls2` and `INTEL_shader_integer_functions2` are not read.
- Integer dot product records only one acceleration bit.
