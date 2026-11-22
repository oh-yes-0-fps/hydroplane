//! Compiles `shader/` to SPIR-V twice, from the same source: with array tiles
//! (`HYDROPLANE_GPU_TEST_PLAIN`) and, by enabling the `CooperativeMatrixKHR` capability, with
//! cooperative-matrix tiles (`HYDROPLANE_GPU_TEST_COOP`). They are separate modules because a
//! module declaring `CooperativeMatrixKHR` cannot be loaded on a device without it.

use cargo_gpu_install::install::Install;
use cargo_gpu_install::spirv_builder::{Capability, ModuleResult};
use std::path::PathBuf;

fn main() {
    let shader = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("shader");
    let hydroplane_src = shader.join("../../src");
    println!("cargo::rerun-if-changed={}", shader.join("src").display());
    println!("cargo::rerun-if-changed={}", shader.join("Cargo.toml").display());
    println!("cargo::rerun-if-changed={}", hydroplane_src.display());
    // `OUT_DIR` is `<target>/<profile>/build/<pkg>-<hash>/out`. The shaders go next to the
    // profiles instead, so `cargo test --release` and `cargo bench` share one build of them.
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = out.ancestors().nth(4).expect("OUT_DIR inside a target dir").join("spirv-shaders");
    let backend = Install::from_shader_crate(shader.clone()).within_build_script().run().expect("installing rust-gpu");
    // Each variant is one single-threaded codegen, so the two build side by side.
    let modules = std::thread::scope(|s| {
        let builds = [("PLAIN", false), ("COOP", true)].map(|(name, coop)| {
            let mut builder = backend.to_spirv_builder(&shader, "spirv-unknown-vulkan1.3");
            builder.release = true;
            builder.target_dir_path = Some(shaders.join(name));
            if coop {
                builder.capabilities.push(Capability::CooperativeMatrixKHR);
                builder.extensions.push("SPV_KHR_cooperative_matrix".into());
            }
            builder.capabilities.extend([Capability::Int16, Capability::StorageBuffer16BitAccess]);
            builder.extensions.push("SPV_KHR_16bit_storage".into());
            (name, s.spawn(move || builder.build()))
        });
        builds.map(|(name, build)| (name, build.join().expect("shader build thread panicked")))
    });
    for (name, result) in modules {
        let result = result.expect("compiling the shader");
        let ModuleResult::SingleModule(path) = result.module else { panic!("expected one SPIR-V module") };
        println!("cargo::rustc-env=HYDROPLANE_GPU_TEST_{name}={}", path.display());
    }
}
