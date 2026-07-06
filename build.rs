//! Resolves the ILP unroll factor at build time when `--cfg hp_static_dispatch` and
//! `-C target-cpu=…` pin the backend, emitting the `hp_resolved_unroll` cfg and `HP_STATIC_UNROLL`
//! env so dispatch uses a single `Unroll<S, K>`. Skipped under `--cfg hp_no_ilp` and on SPIR-V.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    let flags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    // Encoded rustflags are `\x1f`-separated; `-Cfoo=bar` may be one token or two, so match on
    // substrings to catch both spellings.
    let has = |needle: &str| flags.split('\u{1f}').any(|t| t.contains(needle));

    let hp_static_dispatch = has("hp_static_dispatch");
    let pinned_cpu = has("target-cpu=");
    let hp_no_ilp = has("hp_no_ilp");
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if !(hp_static_dispatch && pinned_cpu) || hp_no_ilp || arch == "spirv" {
        return;
    }

    // Per-family latency×pipes defaults, same as the no-std fallback: Apple's wide NEON FP wants
    // more chains than x86's 2-3 vector pipes.
    let k: u8 = match arch.as_str() {
        "aarch64" => 8,
        "x86_64" | "x86" | "riscv64" | "wasm32" => 4,
        _ => 1,
    };

    println!("cargo:rustc-cfg=hp_resolved_unroll");
    println!("cargo:rustc-env=HP_STATIC_UNROLL={k}");
}
