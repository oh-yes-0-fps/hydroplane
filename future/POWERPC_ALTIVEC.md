# PowerPC / AltiVec backend plan

Goal: add a `PowerPc` AltiVec backend so `#[kernel]` code runs vectorized on NXP e6500 (T2080/T2081, the FANUC R-30iB Plus controller CPU), in both 32-bit (`powerpc`, big-endian) and 64-bit (`powerpc64`, big-endian) mode. Follow the RVV/SVE pattern: raw `asm!` primitives in `src/arch/`, a thin `Backend` impl in `src/backend/`, a dispatch tail, differential tests against `ScalarBackend`.

## What the hardware gives us

e6500 vector facts (E6500RM Rev 0, §2.5 "Registers for vector operations" p.76; §3.4.5 "AltiVec instructions" pp.~205-219):

- AltiVec/VMX only: 32 x 128-bit VRs, VSCR, VRSAVE. 4 x f32, 4 x i32/u32 lanes.
- No SPE (that is e500 only), no VSX, no double-precision vector, no 64-bit vector integer elements, no `vmuluwm`, no `vgbbd`/`vbpermq` (all Power ISA 2.07+).
- Vector FP: `vaddfp vsubfp vmaddfp vnmsubfp vmaxfp vminfp vrefp vrsqrtefp vlogefp vexptefp vrfin/vrfiz/vrfip/vrfim vcmpeqfp/gtfp/gefp/bounds vcfsx/vcfux vctsxs/vctuxs`.
- Vector int: `vaddu{b,h,w}m vsubu{b,h,w}m vmulo/vmule{ub,uh,sb,sh} vmsum* vmladduhm vsl/vsr/vsra{b,h,w} vand vandc vor vxor vnor vsel vcmpequw vcmpgtuw vcmpgtsw vmax/vmin{u,s}w`.
- Permute/splat: `vperm vsldoi vspltw vspltisw vmrghw vmrglw vpkuwum vupk*`, loads `lvx lvxl lvsl lvsr lvewx`, stores `stvx stvxl stvewx`.
- 32-bit vs 64-bit mode (MSR[CM]) changes only GPR width and effective-address width. Vector encodings and semantics are identical, so one backend serves both; gate it on `any(target_arch = "powerpc", target_arch = "powerpc64")` plus `target_feature = "altivec"`.
- Scalar FPU is present (64-bit FPRs): `f64` kernels run on the scalar floor.

## Gotchas that shape the design

1. **Big-endian.** Lane 0 is the lowest address and the most significant word of the VR. `load`/`store` via `lvx`/`stvx` are lane-order-safe, but any `to_bits`/`from_bits`, `mask_bitmask`, `reduce_*` shuffle order, and `iramp` must be verified against the scalar oracle, not assumed from the LE backends.
2. **`lvx`/`stvx` silently force 16-byte alignment** (low 4 EA bits are ignored). Slices are not guaranteed aligned. Unaligned load = `lvx` at `p` and `p+15`, `lvsl` permute control, `vperm`. Unaligned store = `lvsr`/`vperm` + `vsel` + two `stvx`, or fall back to `stvewx` x 4 (cheap, safe, no overwrite of neighbours). Tail handling must never let the 2-load trick read past a page the slice does not own: prefer `stvewx`/`lvewx` or scalar tails at page boundaries, and test with slices placed at the end of a mapped page.
3. **No vector divide / sqrt.** `div` = `vrefp` estimate + 2 Newton-Raphson steps with `vmaddfp`/`vnmsubfp`; `sqrt` = `vrsqrtefp` + NR, then `x * rsqrt(x)` with zero/inf fixup (`sqrt(0)` must be 0, not NaN; `sqrt(inf)` inf). Document the accuracy (a few ULP, not correctly rounded) and have the differential test use a relative tolerance for these two, as it already does for sqrt/fma.
4. **Denormals flush to zero** in AltiVec FP (VSCR[NJ] only chooses the NaN/denorm handling mode; there is no full-precision denormal path). Differential tests must flush expected denormals, and `vminfp/vmaxfp` NaN behaviour does not match the `minimumNumber` contract in `Backend::min/max`: patch with a `vcmpeqfp(a,a)` / `vcmpeqfp(b,b)` NaN select, like the x86 backends do.
5. **Masks** are all-ones/all-zeros lanes (same as SSE). `any`/`all` via the record forms (`vcmpequw.` sets CR6) or `vcmpequw` against zero + store-roundtrip. `mask_bitmask` has no `movemask`: keep the trait's portable default.
6. **Integer companions.** `u32`/`i32` add/sub/logic/shift/compare/min/max are native. Multiply (`imul`) is not: compose from `vmulouh`/`vmuleuh` + shift + add, or take the portable default. Variable shifts are native (`vslw/vsrw/vsraw` take per-lane counts); uniform `k` can `vspltisw`/splat the count.
7. **`abs`/`neg`** are bit ops: `vandc` with sign mask, `vxor` with sign mask. Build constants with `vspltisw -1` + `vslw 31` (no constant pool needed, see 9).
8. **`asm!` on PowerPC is nightly** (`asm_experimental_arch`, and the `vreg` class needs the `altivec` target feature). `core::arch::powerpc{,64}` intrinsics are also unstable. This is acceptable for this target (see Build); keep every use behind the PPC cfgs so x86/aarch64/riscv stay stable-only.
9. **FANUC module constraints** (see FANUC notes, `fanuc-sdk`): modules are pure `.text` plus rodata reached via `r2`; no `.data`/`.bss`, no absolute pointers, no relocations. Therefore: no `static` vector constant tables, no `lazy` detection state in statics, no jump tables from big `match`es in the hot path (use `hp_static_dispatch`). Materialize constants with `vspltis*`/`vsldoi` or load them from a stack array. LLVM-built modules crash the controller; only the GCC-codegen path loads. Anything that reaches hardware goes through the GCC backend.
10. **ABI.** On the 32-bit SVR4 ABI v20-v31 are callee-saved and VRSAVE tracks live VRs. Keep all vector state inside `asm!` blocks that declare their `vreg` clobbers, and do not pass vectors across `extern "C"` boundaries. Do not rely on the compiler to maintain VRSAVE across the FANUC task switcher: a module that is the first to use VRs on a task needs the OS to have enabled MSR[VEC]; FANUC's own code already executes `lvxl`/`vperm`, so the bit is set in user state, but confirm on the emulator first.

## Work items

### 1. Target plumbing
- `src/arch/mod.rs`: `#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))] pub mod altivec;`
- `src/backend.rs` (the module list at ~L1889-1915): add `altivec` under the same cfg.
- `src/lib.rs` `towers`: re-export `PowerPc` (name the token `AltiVec`) under the cfg.
- `Cargo.toml` `unexpected_cfgs` check-cfg: add `cfg(hp_no_altivec)` (opt-out, mirrors `hp_no_avx512`).
- `build.rs`: add `"powerpc" | "powerpc64"` to the pinned-CPU unroll table. Start at 4; e6500 has 32 VRs and a single vector FP pipe, so measure before raising. The fallback `_ => 1` otherwise leaves ILP off.
- `README.md`: add the target row and `hp_no_altivec` to the cfg table.

### 2. `src/arch/altivec.rs` (raw primitives)
Model on `src/arch/rvv.rs`: `#![cfg(...)]`, one `#[inline(always)]` function per op using `core::arch::asm!` with `options(pure, nomem, nostack, preserves_flags)` where valid. Repr: `#[repr(C, align(16))] struct VecF32([f32; 4])` / `VecU32([u32; 4])` memory image, or `core::simd`-free `[u32;4]` with `vreg` operands.
- f32: `splat load store add sub mul neg fma(vmaddfp) div(NR) sqrt(NR) min max le lt ge gt select reduce_sum reduce_min reduce_max`.
- u32/i32 companion: the `Backend` IVector ops listed in item 3, native where the ISA has them.
- `is_supported()`: runtime check. With `std` on Linux read `getauxval(AT_HWCAP)` & `PPC_FEATURE_HAS_ALTIVEC` (0x10000000); under `hp_static_dispatch` or no_std use `cfg!(target_feature = "altivec")`. No detection code may need statics on the FANUC target.

### 3. `src/backend/altivec.rs` (the `Backend` impls)
Model on `src/backend/rvv.rs` (macro-generated impl per element, delegating to arch primitives).
- `pub struct AltiVec;` zero-sized token, `new_unchecked()` + `detect()` like `Rvv`.
- `impl Backend<f32> for AltiVec`: `lanes() = 4`, `Vector = VecF32`, `Mask = VecU32`, `UNROLL = 4`.
- `impl Backend<u32>` / `Backend<i32>`: native add/sub/neg/and/or/xor/not/shl/shr/compare/select/min/max/reduce; `mul` via composition.
- `f64`, `f16`, `bf16`: reuse the existing `emulated_float_element!` macro exactly as `Rvv` does for f16/bf16, and add `f64` (no vector double on e6500), so a `Gang<AltiVec>` kernel generic over the element still compiles; `f64` runs on lanes=2 emulation or the scalar floor.
- Leave gather, `mask_bitmask`, and the I* fallbacks on the trait defaults first; override only after the benchmarks justify it.

### 4. Dispatch
- `src/dispatch.rs`: add `tier::ALTIVEC` (next free code; `tier` currently tops out at 13), a `Caps.altivec: bool`, set it in `host_caps()` under the PPC cfg (`hp_no_altivec` forces false), and add it to `canonical_tier` after SVE/before NEON ordering is irrelevant since the arch cfgs are disjoint. Integer combos are allowed (native u32/i32).
- Add a `powerpc_dispatch_tail!` macro next to `riscv_dispatch_tail!` and call it in the same places (`SimdDispatch`, `MatrixDispatch`, and the `combo` dispatch). Under `hp_static_dispatch` + `target_feature = "altivec"` it is one unconditional call.
- `hydroplane-macros`: add the `AltiVec` arm to the generated tier `match`, pruned the same way as the other single-width tiers.

### 5. Tests
- `src/backend/diff_tests.rs`: instantiate the existing differential suite for `AltiVec` (f32 + u32 + i32). Expect to add: denormal flushing in the oracle comparison, relative tolerance for `div`/`sqrt`, and an alignment sweep (every slice offset 0..4 elements, plus a slice ending exactly at a page boundary).
- Existing integration tests (`tests/kernel_macro.rs`, `int_kernels.rs`, `ilp.rs`, `varying_helpers.rs`, `gang_combinators.rs`) must pass unchanged on the PPC target with no `#[cfg]` edits.
- Big-endian specific tests: `to_bits`/`from_bits` round trip, `iramp`, `reduce_sum` lane order, `mask_bitmask` bit order.

### 6. Running the tests (no PPC hardware locally)
- Primary: QEMU user-mode or system emulation. `qemu-ppc64 -cpu e6500` exists for 64-bit mode; for 32-bit mode check whether `qemu-ppc -cpu e500mc`/`e5500`-family models expose AltiVec (e500mc does NOT have it) and whether `qemu-system-ppc64 -M ppce500 -cpu e6500` can run a 32-bit kernel and userland. Verify what the installed QEMU supports before committing to an approach, and put the working recipe in `tools/qemu-altivec.sh` modeled on `tools/qemu-sve.sh` (static musl binary as PID 1, serial log grep for `ALTIVEC_ALL_OK` / `ALTIVEC_FAILS=N`).
- Targets: `powerpc-unknown-linux-gnu` and `powerpc64-unknown-linux-gnu` (both big-endian, nightly for `asm!`). For the FANUC target use the existing `cargo fanuc build` GCC-backend flow rather than LLVM.
- The in-repo FANUC emulator (`fanuc-emu`) is a PPC603 interpreter with no AltiVec; do not treat it as an AltiVec oracle unless vector support is added to it first. If a second oracle is wanted, unicorn's PPC models are unverified for AltiVec: check before relying on it.
- Never run new vector code on a real controller until it passes under QEMU; an illegal/unavailable-vector exception takes the controller down.

### 7. Benchmarks
Add the PPC row to `hydroplane-example/benches/workloads.rs` runs (QEMU numbers are meaningless; only run on real e6500 hardware or skip). Expected win is the f32 reductions (dot, l2norm, asum) at 4 lanes + ILP; `mandelbrot` benefits from native masks. `f64` workloads will show no gain, and that is correct.

## Out of scope (note, do not build now)
- VSX / Power8+ (`powerpc64le`, `vaddudm`, `vmuluwm`, `xvadddp`): a separate `Vsx` tier later; it is a different CPU family from the e6500.
- Matrix tile ops: no tile engine on e6500; `dense`/`matrix` use the register-blocked fallback over `Backend`.
- GPU/SPIR-V: untouched.

## Definition of done
1. `cargo +nightly test --target powerpc64-unknown-linux-gnu` and the 32-bit equivalent pass under QEMU with the differential suite covering f32/u32/i32.
2. A `#[kernel]` dot product dispatches to `AltiVec` and beats the scalar backend in an emulated instruction-count check (or on hardware).
3. A `no_std` + `hp_static_dispatch` build for the FANUC GCC-backend target links as pure `.text` with no relocations, and runs under the emulator path with AltiVec disabled (scalar floor) unchanged.
4. x86_64, aarch64, riscv64, wasm32 builds are untouched and still stable-Rust.
