# hydroplane-auto

Build-time MIR analysis for [hydroplane](https://crates.io/crates/hydroplane) `#[kernel]`s. It measures each kernel body (stack bytes, varying locals, memory ops, branches, calls) and feeds the metrics back to the `#[kernel]` macro, which uses them to pick `tiny`/`noalias`/unroll decisions per kernel instead of one-size defaults.

Wire it into the crate that defines the kernels:

```toml
[build-dependencies]
hydroplane-auto = "0.0.1"
```

```rust
// build.rs
fn main() {
    hydroplane_auto::build_script();
}
```

The analysis is a StableMIR pass on a pinned nightly `rustc_driver`, run **out of process**: the driver source ships inside this crate, is compiled against the pinned nightly at build time, and analyzes a nested build of the consumer crate. The consumer stays on stable; this crate itself is pure stable Rust with no dependencies.

It no-ops (with a cargo warning, and macro defaults apply) on debug builds, on non-Unix hosts, when the pinned nightly toolchain isn't installed, or when `KANALYZE_DISABLE` is set.
