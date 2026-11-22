# hydroplane

Float-agnostic, ISPC-style SPMD/SIMD infrastructure for Rust. **Write one kernel, get nearly optimal code everywhere**: the kernel is written once against a lane-agnostic context, and both the ISA width and the ILP unroll factor are raised for you, at compile time when the target is pinned, by runtime detection otherwise. Stable Rust, `no_std`-compatible, no SIMD-crate dependency. Hydroplane also works with GPU backends via rust-gpu and SPIR-V, enabling the same kernel code to run on graphics hardware. GPU functionality is still evolving and the hyper optimized performance promises cannot be made the same there as on the CPU backends.

```rust
use hydroplane::{Gang, kernel};

#[kernel]
fn dot<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_sum(a, b, |acc, x, y| acc + x * y)
}

let s = dot(&a, &b); // dispatched: AVX-512 / AVX2 / SSE4 / NEON / SVE / ... picked per host
```

`#[kernel]` monomorphizes the body per backend and generates a dispatching wrapper (plus an `_on` companion for composing kernels on an already-selected backend). Besides the leading `ctx: Gang` argument, a kernel is no different from any other Rust function: ordinary parameters, borrows, generics, and return type, called like a plain `fn` — `dot(&a, &b)` above. Kernels never name an ISA. Elements are generic too: one body serves `f32`, `f64`, `f16`, `bf16` (half types widen-compute-narrow unless the target has native lanes). Integers are supported as well: `u32` and `i32` are first-class kernel scalars and also run as companion lanes alongside float kernels for argmin-style index tracking and float bit manipulation via `to_bits`/`from_bits`.

## What "raising" means

- **ISA raising.** At run time the dispatcher picks the widest implemented backend the CPU supports; the SIMD ops are `#[target_feature]` bodies, so a generic x86 binary still uses AVX-512 on a host that has it. With `--cfg hp_static_dispatch` + a pinned `target-cpu` the ladder folds into one unconditional call at compile time.
- **ILP raising.** Reductions and maps run `K` independent register chains to saturate the core's FP pipes. `K` is measured once per process by a startup sweep and baked into the monomorphized kernel as a compile-time constant (`Unroll<S, K>`); on pinned builds `build.rs` resolves it at build time instead. A reduction carrying several accumulators (`ctx.zip_sum_n(a, b, |[xy, xx, yy], x, y| [xy + x * y, xx + x * x, yy + y * y])`, one pass for a cosine) runs fewer chains so they all stay in registers. This is routinely worth more than lane width on latency-bound loops — see the [case study](CASE_STUDY.md).
- **Alignment.** With 32- and 64-byte registers, the combinators start with a masked step up to the first register boundary, so the rest of the loads never straddle two cache lines whatever alignment the caller's slices have.

## Backends

| target | backends |
|---|---|
| x86-64 | SSE4, AVX, AVX2, AVX-512 (+ AVX-512-FP16, AVX-512-BF16), Intel AMX (tiles) |
| aarch64 | NEON (+ A32), SVE1/SVE2, SME1/SME2 (tiles) |
| riscv64 | RVV |
| wasm32 | simd128 |
| GPU | rust-gpu/SPIR-V subgroup backend (the scalar backend is the lowering target) |
| anything | `ScalarBackend`, the always-available 1-lane oracle |

Three layers: raw `Backend` ops → the ergonomic `Gang`/`Varying` surface (operator-overloaded registers, masks, chunking/tail combinators, SoA storage) → `MatrixBackend`/`dense` (register-blocked GEMM tiles; runtime-shaped BLAS-lite).

## Benchmarks

Paired implementations of each workload — hydroplane `#[kernel]` (`hp`), hand-rolled [`wide`](https://crates.io/crates/wide) SIMD with manual ILP (`wide`), and a plain scalar loop (`scalar`) — live in [`hydroplane-example`](hydroplane-example/). Apple M5 Pro, rustc 1.99, `RUSTFLAGS="-C target-cpu=native"`, criterion `--quick` (best of three runs), n = 4096 `f32` elements; correctness is asserted against the scalar oracle before timing.

```
RUSTFLAGS="-C target-cpu=native" cargo bench -p hydroplane-example --bench workloads
```

| workload | shape | scalar | hand `wide` | hydroplane | hp vs scalar | hp vs `wide` |
|---|---|---|---|---|---|---|
| l2norm | `√Σx²` reduction | 1960.7 ns | 352.5 ns | **104.5 ns** | **18.76×** | 3.37× |
| asum | `Σ\|x\|` reduction | 1973.6 ns | 290.6 ns | **120.3 ns** | **16.41×** | 2.42× |
| l1dist | `Σ\|x−y\|` reduction | 2022.2 ns | 304.2 ns | **165.7 ns** | **12.21×** | 1.84× |
| dot | `Σ x·y` reduction | 1992.5 ns | 175.6 ns | **166.8 ns** | **11.95×** | 1.05× |
| cosine | three dots + combine | 2320.4 ns | 423.0 ns | **374.6 ns** | **6.19×** | 1.13× |
| polysum | register-heavy poly reduction | 2307.3 ns | 593.7 ns | **586.4 ns** | **3.93×** | 1.01× |
| double_polysum | two fused poly reductions | 4572.3 ns | **1256.6 ns** | 1296.8 ns | 3.53× | 0.97× |
| mandelbrot | divergent escape-time loop | 112.53 µs | 112.39 µs | **43.41 µs** | **2.59×** | 2.59× |
| horner | degree-8 poly, elementwise | 966.1 ns | 549.8 ns | **525.9 ns** | **1.84×** | 1.05× |
| mat3_inverse | batched 3×3 inverse | 7085.7 ns | 28998.0 ns | **6909.7 ns** | **1.03×** | 4.20× |
| normalize | SoA vec3 `v/‖v‖` | 860.8 ns | **825.8 ns** | 850.3 ns | 1.01× | 0.97× |
| saxpy | elementwise FMA (memory-bound) | **174.5 ns** | 197.5 ns | 178.2 ns | 0.98× | 1.11× |
| cmul | streaming complex multiply | **425.0 ns** | 606.0 ns | 467.8 ns | 0.91× | 1.30× |
| transform | batched 3×3 · vec3 | **3131.1 ns** | 3373.1 ns | 3476.3 ns | 0.90× | 0.97× |

For the full progression, scalar → portable `wide` → `wide` restructured for ILP → hydroplane (with raw NEON intrinsics as the ceiling reference) on one problem, with the mechanics of why each step pays (read **[CASE_STUDY.md](CASE_STUDY.md)** ):

| mandelbrot, 4096 pts | scalar | `wide` f32x4 | `wide` + hand ILP | raw NEON + ILP | hydroplane |
|---|---|---|---|---|---|
| time | 108.61 µs | 146.60 µs | 70.20 µs | 41.89 µs | 45.95 µs |
| lines / `unsafe` / ISAs | 17 / none / — | 32 / none / any | ~50 / none / any | ~40 / all / one | 20 / none / all |

## Cargo features

| feature | |
|---|---|
| `std` (default) | runtime CPU detection, `Vec`-based storage |
| `alloc` | `Soa`/`Cols` storage without `std` |
| `libm` | float math on bare-metal `no_std` |
| `glam` | `Vec3Wide`/`Mat3Wide`/`GangGlamExt` wide-geometry helpers |

`f16`/`bf16` support (via `half`) is always on and re-exported from the crate root.

## Build-time configuration

All opt-in via `RUSTFLAGS="--cfg <name>"`:

| cfg | effect |
|---|---|
| `hp_static_dispatch` | no runtime ladder; backend comes from `target_feature` alone. With a pinned `target-cpu`, dispatch folds to a direct call and `build.rs` also resolves the ILP unroll factor at compile time |
| `hp_no_ilp` | compile out the multi-accumulator path (K = 1) |
| `hp_no_avx512` / `hp_no_avx` | floor x86 at AVX2 / SSE4 |
| `hp_no_sve` / `hp_no_sme` / `hp_neon_over_sve` | aarch64 backend trimming |
| `hp_no_amx` | drop Intel AMX tiles |
| `hp_no_apple_accelerate` | `dense` uses its own SIMD GEMM instead of Accelerate |

## Workspace

- [`hydroplane-macros`](hydroplane-macros/) — the `#[kernel]` proc macro.
- [`hydroplane-auto`](hydroplane-auto/) — optional build-time MIR analysis that sizes unroll/inlining decisions per kernel.
- [`hydroplane-example`](hydroplane-example/) — the paired benchmark workloads above.

License: Apache-2.0.
