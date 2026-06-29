# hydroplane

*Floating but fast.*

Float-agnostic, ISPC-style SPMD/SIMD infrastructure for Rust. Write one kernel generic
over the scalar element (`f32`, `f64`, and `half`'s `f16`/`bf16`) and the execution
backend; it runs scalar, on hand-written `core::arch` SIMD (SSE4 / AVX2 / AVX-512 / NEON),
and — on the rust-gpu SPIR-V target — across a GPU subgroup. **Stable Rust, no SIMD-crate
dependency** (`half` is the only optional dep). Built for SoA collision/geometry libraries
like [`wreck`](../).

## Why

Existing crates each miss something: `pulp` isn't generic over the scalar and has no f16;
`wide` fixes lane width at compile time; `std::simd` is nightly; `simba` has no runtime
dispatch or f16. `hydroplane` is a thin unifier: one generic-over-scalar `Backend` trait, with
the SIMD hand-written via intrinsics so instruction selection (FMA, native f16, AVX-512
`k`-masks) is fully under control and everything stays on stable.

## Writing a kernel

You never name a backend. The `#[kernel]` attribute turns a plain generic function into the
struct + [`Kernel`] impl + dispatching wrapper; its first parameter is the [`Gang`] context —
`dispatch` builds it from the backend it picked by runtime CPU detection — and you build
operator-overloaded `Lane`/`Mask` varying values through it, so the body reads like scalar Rust
but runs as SIMD:

```rust
use hydroplane::{Scalar, Gang, Soa, kernel};

// sphere–sphere "does the query overlap any sphere?" — written ONCE, runs for
// f32/f64/f16 on scalar/SSE4/AVX2/AVX-512/NEON.
#[kernel]
pub fn any_overlap<'a, T: Scalar>(ctx: Gang<T>, soa: &'a Soa<T>, q: [T; 4]) -> bool {
    let n = ctx.lanes();
    let (cx, cy, cz, sr) = (ctx.splat(q[0]), ctx.splat(q[1]), ctx.splat(q[2]), ctx.splat(q[3]));
    let (xs, ys, zs, rs) = (soa.column(0), soa.column(1), soa.column(2), soa.column(3));
    let mut k = 0;
    while k < soa.padded() {
        let dx = cx - ctx.load(&xs[k..k + n]);
        let dy = cy - ctx.load(&ys[k..k + n]);
        let dz = cz - ctx.load(&zs[k..k + n]);
        let d2 = dx * dx + dy * dy + dz * dz;        // looks scalar, is SIMD
        let rsum = sr + ctx.load(&rs[k..k + n]);
        if d2.le(rsum * rsum).any() { return true; } // cross-lane reduce
        k += n;
    }
    false
}

// call site — no struct, no impl, no `dispatch`:
let hit = any_overlap(&soa, q);
```

The leading parameters are contexts, each typed `Gang<T>` (the backend type argument is filled in for
you); the scalar is the type parameter bound by `Scalar` (or named `T`, or set with
`#[kernel(scalar = U)]`). Generics are ordinary `<…>` and may carry multiple bounds, where-clauses,
and several lifetimes/type parameters.

A kernel lists the execution surfaces it needs, in the order their context parameters appear:

- `#[kernel(vector)]` (the default, also bare `#[kernel]`) — one `Backend<T>` context.
- `#[kernel(matrix)]` — one [`MatrixKernel`] context with the `.tiles()` matmul surface;
  dispatched via `dispatch_matrix`.
- `#[kernel(vector, matrix)]` — two leading contexts in that order, both over the *same* dispatched
  backend, so the first is a plain vector handle and the second a matrix handle:

```rust
#[kernel(vector, matrix)]
fn gemm_and_reduce<'a, T: Scalar, const M: usize, const N: usize, const K: usize>(
    v: Gang<T>, m: Gang<T>, a: &'a [T], b: &'a [T], out: &'a mut [T::Compute],
) -> f64 {
    m.tiles()/* …A·B into out… */;
    let _ = v.lanes();      // same backend, vector surface
    0.0
}
```

**Tiny hot kernels** — backend selection inside `dispatch` is cached: each scalar resolves its
runtime tier once into a process-global atomic, so a micro-kernel (a joint-limit check in an inner
loop) is a load + `match` per call, not a fresh CPU probe. The scalar may be concrete — `fn
any_gt(ctx: Gang<f32>, …)` infers `f32` from the context, no generic needed.

**Reductions saturate the pipes for free** — a wide out-of-order core (Apple's ~4 NEON FP pipes, x86's
2–3) runs at a fraction of peak if your reduction is one serial accumulator chain stalling on FMA
latency. `zip_sum`/`sum` run several *independent* accumulator chains and combine them as a tree — and
the chain count `K` is the per-core saturation point, **detected once and cached exactly like the
backend**. You write the obvious sum and never mention ILP, accumulators, or `K`:

```rust
#[kernel]
fn dot(ctx: Gang<f32>, a: &[f32], b: &[f32]) -> f32 {
    ctx.zip_sum(a, b, |acc, x, y| x.fma(y, acc))   // K chains, all implied; one dispatch picks K
}
```

The `0` identity supplies the seed, both masked-tail fills, and the chain combine. The warm-path cost
is one relaxed atomic load + a `match` to the `K`-monomorphized loop. (For non-sum reductions —
max/min/any — `zip_reduce` takes an explicit identity and combine.)

**Kernel calling kernel** — every kernel also gets a `<name>_on(ctx, …)` companion that runs its body
on a context you pass in, skipping dispatch. Call it from inside another kernel to reuse the outer
kernel's already-dispatched backend, so dispatch happens *once* at the outer boundary:

```rust
#[kernel] fn scaled<'a, T: Scalar>(ctx: Gang<T>, xs: &'a [T], k: T) -> f64 { /* … */ }

#[kernel]
fn scaled_then_sum<'a, T: Scalar>(ctx: Gang<T>, xs: &'a [T], ys: &'a [T], k: T) -> f64 {
    scaled_on(ctx, xs, k) + scaled_on(ctx, ys, k)   // one dispatch, two inner runs
}
```

`#[kernel]` is the `macros` feature, on by default. It expands to exactly the hand-written form,
which you can always write yourself when a kernel falls outside the attribute's shape:

```rust
use hydroplane::{Backend, Kernel, Scalar, Gang, Soa, SimdDispatch, dispatch};

struct AnyOverlap<'a, T: Scalar> { soa: &'a Soa<T>, q: [T; 4] }
impl<T: Scalar> Kernel<T> for AnyOverlap<'_, T> {
    type Output = bool;
    fn run<S: Backend<T>>(self, ctx: Gang<T, S>) -> bool { /* same body */ }
}

fn query<T: SimdDispatch>(soa: &Soa<T>, q: [T; 4]) -> bool { dispatch(AnyOverlap { soa, q }) }
```

Building `--no-default-features` (no proc-macro dependency) swaps `#[kernel]` for a `macro_rules!`
`kernel!` fallback of the same name with a slightly stricter syntax — see the `kernel_macro` module.

Underneath sits the **engine**: `Backend<T: Scalar>`, one concrete impl per `(ISA, scalar)`.
You write kernels generic over `S: Backend<T>`, but the concrete tokens (`Avx2`, `Avx512`,
`Sse4`, `Neon`) are crate-internal — they exist only as monomorphization targets `dispatch`
chooses between, never as something you construct. Reach the raw backend, if a kernel needs
the un-wrapped op API, via `ctx.backend()`.

## Dispatch

```rust
// Picks the widest backend the running CPU supports and runs the kernel.
let hit = hydroplane::dispatch(MyKernel { /* … */ });
```

- **Runtime** (default, std): `is_x86_feature_detected!` selects AVX-512 → AVX2 → SSE4 →
  scalar at the dispatch boundary. The SIMD ops are `#[target_feature]` bodies, so a wider
  ISA is used whenever the host has it — independent of how the crate was built.
- **Compile-time fast path**: if the build already guarantees the widest implemented ISA
  (e.g. `RUSTFLAGS="-C target-cpu=native"` on an AVX-512 host), that backend is taken with
  no runtime branch.
- **no-std**: with no runtime detection, the widest ISA the build guarantees via
  `target_feature` is taken.
- `ScalarBackend` is always the fallback, so every scalar type has a path.
  [`run_scalar`](https://docs.rs/hydroplane) forces it directly — handy as a correctness oracle or
  baseline.

## f16 / bf16

- **Stable** (`--features half`): storage is 16-bit (`half`), compute is f32. On AVX2 the
  F16C widen path (`_mm256_cvtph_ps`) gives real SIMD f16 (8-wide) at full f32 precision;
  elsewhere the scalar widen path applies.
- **Native** (`--features f16-native`, **nightly**): true 32-wide hardware f16 via
  AVX-512-FP16 (`__m512h`) — no widen/narrow round-trip. Requires nightly because the `f16`
  primitive type and the AVX-512-FP16 intrinsics are still unstable; the feature enables
  those `#![feature]`s at the crate root. Dispatch prefers it over the AVX2 widen path when
  the CPU supports `avx512fp16`. The stable build is entirely unaffected.

## soa-rs interop

Already storing your data with [`soa-rs`](https://docs.rs/soa-rs)? Its `#[derive(Soars)]`
fields are plain `&[T]` slices, so `hydroplane` runs over them with no glue type — two ways:

```rust
use soa_rs::{Soars, soa};
use hydroplane::{Backend, Kernel, Gang, dispatch};

#[derive(Soars, Clone, Copy)]
struct Sphere { x: f32, y: f32, z: f32, r: f32 }

let s: soa_rs::Soa<Sphere> = soa![/* … */];

// Zero-copy: walk the borrowed field slices in place. `chunks` yields full registers plus a
// final short tail; `load_partial` stages the tail, filling inactive lanes with a sentinel.
impl Kernel<f32> for MyKernel<'_> {
    fn run<S: Backend<f32>>(self, ctx: Gang<f32, S>) -> bool {
        for (k, cnt) in ctx.chunks(self.xs.len()) {
            let x = ctx.load_partial(&self.xs[k..k + cnt], 0.0);
            let r = ctx.load_partial(&self.rs[k..k + cnt], f32::NAN); // NaN tail ⇒ no false hit
            /* … d2.le(rsum * rsum).any() … */
        }
        false
    }
}
let hit = dispatch(MyKernel { xs: s.x(), ys: s.y(), zs: s.z(), rs: s.r(), q });

// Copy bridge: one line into a padded `hydroplane::Soa`, then reuse a padded-column kernel verbatim.
let cols = hydroplane::Soa::from_columns(&[s.x(), s.y(), s.z(), s.r()], &[0.0, 0.0, 0.0, f32::NAN]);
```

`load_partial`/`store_partial`/`chunks` are general (they work on any `&[T]`, no `soa-rs`
dependency); `Soa::from_columns` builds a padded SoA from any equal-length column slices. See
`examples/soa_rs_interop.rs`.

## GPU subgroups (SPIR-V)

Under `target_arch = "spirv"` (rust-gpu), the gang maps to a subgroup: the `Subgroup`
backend's cross-lane ops lower to `OpGroupNonUniform*`, and the warp width is read straight
from the hardware `SubgroupSize` builtin — nothing to configure or pass in. The portable
`backend::subgroup::choose` policy decides **sequential vs. subgroup** execution from the item
count and that width, and is unit-tested on the CPU.

## Layout

```
src/scalar.rs          Scalar element trait + f32/f64/f16/bf16
src/backend.rs         Backend<T> trait + ScalarBackend (oracle / SPIR-V target)
src/backend/{sse4,avx2,avx512}.rs   hand-rolled x86_64 intrinsics (avx2 incl. F16C f16)
src/backend/avx512fp16.rs   native 32-wide f16 (nightly `f16-native`)
src/backend/neon.rs    hand-rolled aarch64 NEON
src/varying.rs         Lane / Mask / Gang surface + chunks / load_partial / store_partial
src/soa.rs             generic padded columnar SoA (NaN-padded tails) + from_columns bridge
src/dispatch.rs        Kernel trait + runtime (cached, default) / compile-time backend selection
src/ilp.rs             cached per-core unroll factor K (multi-accumulator reductions: zip_sum/reduce)
src/backend/subgroup.rs   SPIR-V subgroup backend + portable size/decision policy
src/kernel_macro.rs    `macro_rules!` `kernel!` fallback (only when the `macros` feature is off)
hydroplane-macros/     proc-macro crate: the default `#[kernel]` attribute
```

## Verification

- `src/backend/diff_tests.rs` — every SIMD op of every backend checked against the scalar
  oracle (exact for arithmetic/compare/select, tolerance for fma/sqrt/reduce). In-crate,
  since the backend tokens are crate-internal; runs for whichever ISAs the host CPU supports
  (SSE4/AVX2/AVX-512 on x86_64, NEON on aarch64), plus the AVX2 F16C and native AVX-512-FP16
  f16 paths against the scalar f16 oracle.
- `tests/spheres_parity.rs` — the ported sphere kernel matches a brute-force reference for
  f32 and f64, both `run_scalar` and `dispatch`ed.
- `tests/soa_rs_interop.rs` — a `soa-rs` `#[derive(Soars)]` struct fed through both the
  zero-copy (`chunks` + `load_partial`) and the `from_columns` copy paths matches the
  reference; `store_partial` writes results back into a `soa-rs` mutable field slice.
- `cargo run --release --example bench_spheres` — throughput (≈4.8× f32, ≈2.3× f64 over
  scalar on an AVX-512 host).
```
