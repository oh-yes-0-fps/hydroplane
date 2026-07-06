# hydroplane

Float-agnostic, ISPC-style SPMD/SIMD infrastructure for Rust. **Write one kernel, get nearly optimal code everywhere**: the kernel is written once against a lane-agnostic context, and both the ISA width and the ILP unroll factor are raised for you, at compile time when the target is pinned, by runtime detection otherwise. Stable Rust, `no_std`-compatible, no SIMD-crate dependency. Hydroplane also works with GPU backends via rust-gpu and SPIR-V, enabling the same kernel code to run on graphics hardware. GPU functionality is still evolving and the hyper optimized performance promises cannot be made the same there as on the CPU backends.

```rust
use hydroplane::{Gang, kernel};

#[kernel]
fn dot<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_fold(a, b, 0.0, 0.0, ctx.splat(0.0), |acc, x, y| acc + x * y)
        .reduce_sum()
}

let s = dot(&a, &b); // dispatched: AVX-512 / AVX2 / SSE4 / NEON / SVE / ... picked per host
```

`#[kernel]` monomorphizes the body per backend and generates a dispatching wrapper (plus an `_on` companion for composing kernels on an already-selected backend). Kernels never name an ISA. Elements are generic too: one body serves `f32`, `f64`, `f16`, `bf16` (half types widen-compute-narrow unless the target has native lanes). Integers are supported as well: `u32` and `i32` are first-class kernel scalars and also run as companion lanes alongside float kernels for argmin-style index tracking and float bit manipulation via `to_bits`/`from_bits`.

## What "raising" means

- **ISA raising.** At run time the dispatcher picks the widest implemented backend the CPU supports; the SIMD ops are `#[target_feature]` bodies, so a generic x86 binary still uses AVX-512 on a host that has it. With `--cfg hp_static_dispatch` + a pinned `target-cpu` the ladder folds into one unconditional call at compile time.
- **ILP raising.** Reductions and maps run `K` independent register chains to saturate the core's FP pipes. `K` is measured once per process by a startup sweep and baked into the monomorphized kernel as a compile-time constant (`Unroll<S, K>`); on pinned builds `build.rs` resolves it at build time instead. This is routinely worth more than lane width on latency-bound loops — see the [case study](CASE_STUDY.md).

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

Paired implementations of each workload — hydroplane `#[kernel]` (`hp`), hand-rolled [`wide`](https://crates.io/crates/wide) SIMD with manual ILP (`wide`), and a plain scalar loop (`scalar`) — live in [`hydroplane-example`](hydroplane-example/). Apple M5 Pro, rustc 1.95, `RUSTFLAGS="-C target-cpu=native"`, criterion `--quick`, n = 4096 `f32` elements; correctness is asserted against the scalar oracle before timing.

```
RUSTFLAGS="-C target-cpu=native" cargo bench -p hydroplane-example --bench workloads
```

| workload | shape | scalar | hand `wide` | hydroplane | hp vs scalar | hp vs `wide` |
|---|---|---|---|---|---|---|
| l2norm | `√Σx²` reduction | 1976.4 ns | 351.4 ns | **88.8 ns** | **22.26×** | 3.96× |
| asum | `Σ\|x\|` reduction | 1968.9 ns | 294.1 ns | **120.2 ns** | **16.39×** | 2.45× |
| dot | `Σ x·y` reduction | 1977.1 ns | 176.5 ns | **168.1 ns** | **11.76×** | 1.05× |
| l1dist | `Σ\|x−y\|` reduction | 1977.9 ns | 311.1 ns | **169.1 ns** | **11.70×** | 1.84× |
| cosine | three dots + combine | 2317.8 ns | **426.6 ns** | 492.6 ns | 4.71× | 0.87× |
| polysum | register-heavy poly reduction | 2275.5 ns | **584.5 ns** | 636.1 ns | 3.58× | 0.92× |
| double_polysum | two fused poly reductions | 4554.1 ns | **1261.3 ns** | 1371.3 ns | 3.32× | 0.92× |
| mandelbrot | divergent escape-time loop | 113.24 µs | 112.79 µs | **47.12 µs** | **2.40×** | 2.39× |
| horner | degree-8 poly, elementwise | 965.8 ns | **544.4 ns** | 630.8 ns | 1.53× | 0.86× |
| normalize | SoA vec3 `v/‖v‖` | 870.3 ns | 829.7 ns | **823.4 ns** | 1.06× | 1.01× |
| mat3_inverse | batched 3×3 inverse | 7423.3 ns | 30194.8 ns | **7327.3 ns** | **1.01×** | 4.12× |
| saxpy | elementwise FMA (memory-bound) | **172.8 ns** | 207.7 ns | 175.3 ns | 0.99× | 1.18× |
| cmul | streaming complex multiply | **456.5 ns** | 588.3 ns | 481.8 ns | 0.95× | 1.22× |
| transform | batched 3×3 · vec3 | **3191.9 ns** | 3523.9 ns | 3468.9 ns | 0.92× | 1.02× |

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
