fn main() {
    #[cfg(feature = "vulkan")]
    kernels();
}

/// Compiles `kernels/` to SPIR-V twice, in `f32` and in `f64`, and exposes the modules as
/// `HYDROPLANE_SOLVERS_SPV_F32` / `_F64`.
#[cfg(feature = "vulkan")]
fn kernels() {
    use cargo_gpu_install::install::Install;
    use cargo_gpu_install::spirv_builder::{Capability, ModuleResult};
    use std::path::PathBuf;

    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let backend = Install::from_shader_crate(dir.clone()).within_build_script().run().expect("installing rust-gpu");
    for (name, f64) in [("F32", false), ("F64", true)] {
        let mut builder = backend.to_spirv_builder(&dir, "spirv-unknown-vulkan1.2");
        builder.release = true;
        builder.target_dir_path = Some(out.join(format!("kernels-{name}")));
        builder.extensions.push("SPV_EXT_shader_atomic_float_add".into());
        builder.capabilities.push(Capability::VulkanMemoryModelDeviceScope);
        if f64 {
            builder.shader_crate_features.features = vec!["f64".into()];
            builder.capabilities.extend([Capability::Float64, Capability::AtomicFloat64AddEXT]);
        } else {
            builder.capabilities.push(Capability::AtomicFloat32AddEXT);
        }
        let result = builder.build().expect("compiling the kernels");
        let ModuleResult::SingleModule(path) = result.module else { panic!("expected one SPIR-V module") };
        println!("cargo::rustc-env=HYDROPLANE_SOLVERS_SPV_{name}={}", path.display());
    }
}
