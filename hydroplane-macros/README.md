# hydroplane-macros

The `#[kernel]` attribute macro for [hydroplane](https://crates.io/crates/hydroplane). Write a SIMD/SPMD kernel as a plain generic function; the macro generates the dispatching wrapper plus an `_on` companion for composing kernels on an already-selected backend.

```rust
use hydroplane::{Gang, kernel};

#[kernel]
fn dot<'a>(ctx: Gang, a: &'a [f32], b: &'a [f32]) -> f32 {
    ctx.zip_fold(a, b, 0.0, 0.0, ctx.splat(0.0), |acc, x, y| acc + x * y)
        .reduce_sum()
}

let s = dot(&a, &b); // runtime ISA dispatch
```

Attribute options: `scalar = U` (pin the element type), `tiny` / `noalias` (inlining vs aliasing boundary), and `unroll = N`. With [hydroplane-auto](https://crates.io/crates/hydroplane-auto) wired into the consumer's `build.rs`, these are chosen per kernel from build-time MIR metrics instead of defaults.

Use through the `hydroplane` crate, which re-exports the macro as `hydroplane::kernel`; this crate is not intended to be depended on directly.
