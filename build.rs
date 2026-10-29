//! Resolves the ILP unroll factor at build time when `--cfg hp_static_dispatch` and
//! `-C target-cpu=…` pin the backend, emitting the `hp_resolved_unroll` cfg and `HP_STATIC_UNROLL`
//! env so dispatch uses a single `Unroll<S, K>`. Skipped under `--cfg hp_no_ilp` and on SPIR-V.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    bundle_cfgs();

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

/// `hp_tf_*`: the build's static target features cover a whole dispatch tier bundle
/// (`src/isa_bundles.in`), so static dispatch may build that tier's token without detection.
/// The AVX tier only exists with the `avx1` feature.
fn bundle_cfgs() {
    println!("cargo:rerun-if-changed=src/isa_bundles.in");
    println!(
        "cargo::rustc-check-cfg=cfg(hp_tf_v2, hp_tf_avx, hp_tf_v3, hp_tf_v4, hp_tf_v4bf16, hp_tf_v4fp16)"
    );
    let bundles: &[(&str, &[&str])] = include!("src/isa_bundles.in");
    let enabled = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let enabled: Vec<&str> = enabled.split(',').collect();
    let avx1 = env::var_os("CARGO_FEATURE_AVX1").is_some();
    for (tier, features) in bundles {
        let cfg = match *tier {
            "SSE41" => "hp_tf_v2",
            "AVX" if avx1 => "hp_tf_avx",
            "AVX2" => "hp_tf_v3",
            "AVX512" => "hp_tf_v4",
            "AVX512BF16" => "hp_tf_v4bf16",
            "AVX512FP16" => "hp_tf_v4fp16",
            _ => continue,
        };
        if features.iter().all(|f| enabled.contains(f)) {
            println!("cargo:rustc-cfg={cfg}");
        }
    }
}
