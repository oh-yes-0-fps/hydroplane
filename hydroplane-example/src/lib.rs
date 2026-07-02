// Under the analysis driver's `--cfg hydro_analyze` pass, register the tool namespace so the
// `#[kernel]`-emitted `#[hydro_analyze::metrics(..)]` attributes are recognized. Stripped on the
// ordinary (stable) build, which never sees the cfg.
#![cfg_attr(hydro_analyze, feature(register_tool))]
#![cfg_attr(hydro_analyze, register_tool(hydro_analyze))]

//! Paired hydroplane vs hand-rolled workloads, spanning a range of arithmetic intensity and
//! control-flow complexity. Each module exposes three implementations over identical inputs:
//!
//! * `*_hp`     — a hydroplane `#[kernel]` (float-agnostic, runtime ISA dispatch),
//! * `*_wide`   — a highly optimized hand-rolled `wide` SIMD kernel (full-register + ILP + masked
//!   tail), the "did the abstraction cost anything" baseline,
//! * `*_scalar` — a plain scalar loop, both the correctness oracle and the autovectorization
//!   reference (LLVM SLP / loop-vectorizer gets a clean shot at it).
//!
//! plus a `gen` producing deterministic inputs. Shared by `benches/workloads.rs` and
//! `tests/correctness.rs`, so a kernel is written once and exercised by both.
//!
//! Workloads, roughly by ascending complexity / arithmetic intensity:
//!
//! | module          | shape                          | character                                  |
//! |-----------------|--------------------------------|--------------------------------------------|
//! | [`saxpy`]       | `y = a·x + y` elementwise      | trivial, memory-bound (~0.16 flop/byte)    |
//! | [`dot`]         | `Σ x·y` reduction              | memory-bound reduction (ILP matters)       |
//! | [`horner`]      | degree-8 polynomial eval       | compute-bound elementwise — the SIMD sweet spot |
//! | [`normalize`]   | batched `v / ‖v‖` (SoA vec3)   | moderate, multi-component                  |
//! | [`transform`]   | batched `M·v` (3×3 · vec3)     | moderate, denser arithmetic                |
//! | [`mat3_inverse`]| batched 3×3 inverse            | register-heavy, low-AI (autovec/glam wins) |
//! | [`mandelbrot`]  | escape-time iteration w/ masks | iterative, data-dependent, masked early-exit |

pub mod asum;
pub mod cosine;
pub mod cmul;
pub mod dot;
pub mod double_polysum;
pub mod horner;
pub mod l1dist;
pub mod l2norm;
pub mod mandelbrot;
pub mod mat3_inverse;
pub mod matmul;
pub mod normalize;
pub mod pipeline;
pub mod polysum;
pub mod saxpy;
pub mod transform;

/// Max relative error between two slices — the correctness yardstick. SIMD reorders FP arithmetic,
/// so bit-exact equality is too strict; `1e-3` relative is the tolerance the whole suite uses.
pub fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |e, (&x, &y)| e.max((x - y).abs() / y.abs().max(1.0)))
}

/// A column of `n` deterministic, well-conditioned values in roughly `[-1, 1]·scale`.
pub fn ramp(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.137).sin() * scale)
        .collect()
}
