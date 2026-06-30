//! Build-time resolution of the ILP unroll factor for fully-static builds.
//!
//! When the build pins *both* the ISA (`--cfg static_dispatch`) and the CPU (`-C target-cpu=…`, e.g.
//! `native`), the compile-time backend already matches the machine that will run the binary — so the
//! unroll factor `K` has no reason to be measured at runtime. We resolve it here and hand it to the
//! crate as the `hp_resolved_unroll` cfg plus the `HP_STATIC_UNROLL` env, which lets the dispatch path
//! wrap the backend in a single `Unroll<S, K>` with no startup sweep, no per-dispatch `match`, and
//! one kernel monomorphization instead of one per candidate `K`.
//!
//! Skipped (the runtime sweep stays) unless static_dispatch *and* a pinned target-cpu are both
//! present, and never under `--cfg no_ilp` or the SPIR-V target (which have no FP-pipe ILP at all).

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let flags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    // Encoded rustflags are `\x1f`-separated; a `--cfg foo` is two tokens, `-Cfoo=bar` may be one or
    // two. Match on substrings so both spellings are caught.
    let has = |needle: &str| flags.split('\u{1f}').any(|t| t.contains(needle));

    let static_dispatch = has("static_dispatch");
    let pinned_cpu = has("target-cpu=");
    let no_ilp = has("no_ilp");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if !(static_dispatch && pinned_cpu) || no_ilp || arch == "spirv" {
        return;
    }

    // Per-family saturation defaults — the same `latency × pipes` knees the no-std fallback uses
    // (Apple's wide NEON FP wants more chains than x86's 2–3 vector pipes). `target-cpu=native` has
    // already fixed the ISA, so this lands on the running core's family.
    let k: u8 = match arch.as_str() {
        "aarch64" => 8,
        "x86_64" | "x86" | "riscv64" | "wasm32" => 4,
        _ => 1,
    };

    println!("cargo:rustc-cfg=hp_resolved_unroll");
    println!("cargo:rustc-env=HP_STATIC_UNROLL={k}");
}
