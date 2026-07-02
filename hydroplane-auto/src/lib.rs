//! Build-time MIR analysis for hydroplane `#[kernel]`s.
//!
//! Two halves, both here so the whole pipeline lives in one crate:
//!
//! * [`analysis`] — the decision logic the `#[kernel]` macro calls at expansion: it reads the metrics
//!   file (via the `HYDRO_ANALYSIS` env var the build script sets) and turns a kernel's measured
//!   metrics into codegen choices (`k_cap`, `noalias`). All the thresholds live there.
//! * [`build_script`] — the orchestration a downstream crate's `build.rs` calls. It runs the nightly
//!   StableMIR driver (embedded under `driver/`, written out and compiled on the pinned nightly) as a
//!   `RUSTC_WORKSPACE_WRAPPER` over a marker-gated nested build, writes one line of metrics per kernel,
//!   and points `HYDRO_ANALYSIS` at it.
//!
//! Downstream setup (see `hydroplane-example`): a `build.rs` of `fn main() {
//! hydroplane_auto::build_script(); }`, a `[build-dependencies] hydroplane-auto` entry, the crate-root
//! `#![cfg_attr(hydro_analyze, feature(register_tool), register_tool(hydro_analyze))]`, and a
//! `'cfg(hydro_analyze)'` in the crate's `unexpected_cfgs` check-list.

pub mod analysis;

use std::path::{Path, PathBuf};
use std::process::Command;

/// Pinned nightly for the driver: the one whose `rustc_public` (StableMIR) API the driver is written
/// against. Bump deliberately, in lockstep with `driver/main.rs`.
const TOOLCHAIN: &str = "nightly-2026-06-27";

/// The driver ships as source and is compiled at build time (a crate doing `extern crate rustc_driver`
/// must be the final linked artifact, so it can't be a library dependency).
const DRIVER_MAIN: &str = include_str!("../driver/main.rs");
const DRIVER_CARGO: &str = "\
[package]
name = \"hydro-analyze\"
version = \"0.0.0\"
edition = \"2021\"

[workspace]

[[bin]]
name = \"hydro-analyze\"
path = \"main.rs\"
";
const DRIVER_TOOLCHAIN: &str = "\
[toolchain]
channel = \"nightly-2026-06-27\"
components = [\"rustc-dev\", \"llvm-tools\"]
";

/// Call from a downstream crate's `build.rs` (`fn main() { hydroplane_auto::build_script(); }`).
///
/// Runs the MIR analysis of *that* crate's kernels — release + Unix only — and, on success, emits
/// `cargo:rustc-env=HYDRO_ANALYSIS=<path>#<hash>` so the `#[kernel]` macro can bake per-kernel
/// decisions. On a debug build, a non-Unix host, or with `KANALYZE_DISABLE` set, it no-ops and the
/// macro keeps its defaults.
pub fn build_script() {
    // The nested analysis build re-enters this script; bail before doing anything (no recursion, and
    // crucially no `HYDRO_ANALYSIS` env, so the nested compile uses macro defaults).
    if std::env::var_os("HYDRO_ANALYZE_INNER").is_some() {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap().replace('-', "_");
    let metrics = out_dir.join("hydro_analysis.txt");

    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-env-changed=KANALYZE_DISABLE");

    if let Some(reason) = skip_reason() {
        println!("cargo::warning=hydroplane MIR analysis skipped: {reason}; kernels use macro defaults");
        return;
    }
    if !toolchain_installed() {
        println!(
            "cargo::warning=hydroplane MIR analysis skipped: toolchain {TOOLCHAIN} not installed; \
             kernels use macro defaults. To enable per-kernel tuning, run: \
             rustup toolchain install --profile minimal {TOOLCHAIN} -c rustc-dev,llvm-tools"
        );
        return;
    }

    let driver = build_driver(&out_dir);
    run_nested_analysis(&manifest, &out_dir, &driver, &metrics, &crate_name);

    if !metrics.exists() {
        println!("cargo::warning=hydroplane MIR analysis produced no metrics; kernels use defaults");
        return;
    }
    // The hash makes the env value change when the metrics change, forcing the macro to re-expand.
    let hash = content_hash(&metrics);
    println!("cargo::rustc-env=HYDRO_ANALYSIS={}#{hash}", metrics.display());
    println!("cargo::warning=hydroplane MIR analysis applied ({})", metrics.display());
}

fn skip_reason() -> Option<String> {
    if std::env::var_os("KANALYZE_DISABLE").is_some() {
        return Some("KANALYZE_DISABLE set".into());
    }
    if std::env::var("PROFILE").as_deref() != Ok("release") {
        return Some("debug build (analysis runs only for --release)".into());
    }
    if !cfg!(unix) {
        return Some(format!("only wired up for Unix, not {}", std::env::consts::OS));
    }
    None
}

/// Write the embedded driver source to `out_dir/hydro-driver` and build it (prefer-dynamic + rpath
/// into the nightly runtime, so it links and later loads `librustc_driver`). Returns the binary path.
fn build_driver(out_dir: &Path) -> PathBuf {
    let dir = out_dir.join("hydro-driver");
    std::fs::create_dir_all(&dir).expect("create driver dir");
    std::fs::write(dir.join("main.rs"), DRIVER_MAIN).expect("write driver main.rs");
    std::fs::write(dir.join("Cargo.toml"), DRIVER_CARGO).expect("write driver Cargo.toml");
    std::fs::write(dir.join("rust-toolchain.toml"), DRIVER_TOOLCHAIN).expect("write driver toolchain");

    let sysroot = rustc_out(&["--print", "sysroot"]).expect("nightly sysroot");
    let host = host_triple();
    let rustflags = [
        "-Cprefer-dynamic".to_string(),
        format!("-Clink-arg=-Wl,-rpath,{sysroot}/lib"),
        format!("-Clink-arg=-Wl,-rpath,{sysroot}/lib/rustlib/{host}/lib"),
    ]
    .join("\u{1f}");

    let target_dir = out_dir.join("driver-target");
    let status = clean_cargo()
        .arg(format!("+{TOOLCHAIN}"))
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(&target_dir)
        .env("CARGO_ENCODED_RUSTFLAGS", rustflags)
        .status()
        .expect("failed to build the hydro-analyze driver");
    assert!(status.success(), "building hydro-analyze failed: {status}");
    target_dir.join("debug").join("hydro-analyze")
}

/// Nested `cargo +nightly build` of the downstream crate with the driver as the rustc wrapper. Cargo
/// rebuilds it and its deps under nightly (so the driver's in-process rustc can read their metadata)
/// into a throwaway target dir; the wrapper writes the metrics for the crate named `crate_name`.
fn run_nested_analysis(manifest: &Path, out_dir: &Path, driver: &Path, metrics: &Path, crate_name: &str) {
    let status = clean_cargo()
        .arg(format!("+{TOOLCHAIN}"))
        .args(["build", "--quiet", "--release", "--manifest-path"])
        .arg(manifest.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(out_dir.join("analysis-target"))
        .env("RUSTC_WORKSPACE_WRAPPER", driver)
        .env("HYDRO_ANALYZE_INNER", "1")
        .env("HYDRO_ANALYZE_CRATE", crate_name)
        .env("HYDRO_ANALYSIS_OUT", metrics)
        .env("CARGO_ENCODED_RUSTFLAGS", "--cfg\u{1f}hydro_analyze")
        .status()
        .expect("failed to run the nested analysis build");
    if !status.success() {
        println!("cargo::warning=hydroplane analysis build exited with {status}");
    }
}

/// A cargo command with the outer toolchain's rustc overrides scrubbed, so the nested `+nightly` isn't
/// forced back to the wrong compiler.
fn clean_cargo() -> Command {
    let mut c = Command::new("cargo");
    c.env_remove("RUSTC")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_BUILD_RUSTFLAGS")
        .env_remove("RUSTFLAGS");
    c
}

/// Whether the pinned nightly (with `rustc-dev`) is present. The analysis never installs it —
/// a build script must not touch the network or mutate global rustup state — it only skips with
/// a pointer to the install command.
fn toolchain_installed() -> bool {
    Command::new("rustup")
        .args(["component", "list", "--toolchain", TOOLCHAIN, "--installed"])
        .output()
        .map(|o| {
            o.status.success()
                && String::from_utf8_lossy(&o.stdout).lines().any(|l| l.starts_with("rustc-dev"))
        })
        .unwrap_or(false)
}

fn host_triple() -> String {
    rustc_out(&["-vV"])
        .expect("rustc -vV")
        .lines()
        .find_map(|l| l.strip_prefix("host: ").map(str::to_string))
        .expect("no host line")
}

fn rustc_out(args: &[&str]) -> Option<String> {
    let out = Command::new("rustc").arg(format!("+{TOOLCHAIN}")).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn content_hash(path: &Path) -> u64 {
    let bytes = std::fs::read(path).unwrap_or_default();
    // FNV-1a, enough to change the env value when metrics change.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
