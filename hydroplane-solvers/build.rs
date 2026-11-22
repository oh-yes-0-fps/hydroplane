fn main() {
    #[cfg(feature = "vulkan")]
    kernels();
    accelerate();
}

/// On macOS, the C layer over Accelerate's sparse Cholesky, under `cfg(accelerate)`.
fn accelerate() {
    println!("cargo::rustc-check-cfg=cfg(accelerate)");
    println!("cargo::rerun-if-changed=src/accelerate.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new().file("src/accelerate.c").compile("hydroplane_solvers_accelerate");
        println!("cargo::rustc-link-lib=framework=Accelerate");
        println!("cargo::rustc-cfg=accelerate");
    }
}

/// Compiles `kernels/` to SPIR-V once per working precision, `f32`, `f64` and emulated `f64`, and
/// exposes the modules as `HYDROPLANE_SOLVERS_SPV_F32` / `_F64` / `_EF64`, the last also as
/// `_EF64_PRECISE` with its arithmetic decorated `NoContraction`.
#[cfg(feature = "vulkan")]
fn kernels() {
    use cargo_gpu_install::install::Install;
    use cargo_gpu_install::spirv_builder::{Capability, ModuleResult};
    use std::path::PathBuf;

    let source = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");
    println!("cargo::rerun-if-changed={}", source.display());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dir = staged_kernels(&source, &out);
    let backend = Install::from_shader_crate(dir.clone()).within_build_script().run().expect("installing rust-gpu");
    for (name, feature) in [("F32", None), ("F64", Some("f64")), ("EF64", Some("ef64"))] {
        let mut builder = backend.to_spirv_builder(&dir, "spirv-unknown-vulkan1.2");
        builder.release = true;
        builder.target_dir_path = Some(out.join(format!("kernels-{name}")));
        builder.extensions.push("SPV_EXT_shader_atomic_float_add".into());
        builder.capabilities.push(Capability::VulkanMemoryModelDeviceScope);
        builder.shader_crate_features.features = feature.into_iter().map(String::from).collect();
        if feature == Some("f64") {
            builder.capabilities.extend([Capability::Float64, Capability::AtomicFloat64AddEXT]);
        } else {
            builder.capabilities.push(Capability::AtomicFloat32AddEXT);
        }
        let result = builder.build().expect("compiling the kernels");
        let ModuleResult::SingleModule(path) = result.module else { panic!("expected one SPIR-V module") };
        println!("cargo::rustc-env=HYDROPLANE_SOLVERS_SPV_{name}={}", path.display());
        if feature == Some("ef64") {
            let words: Vec<u32> = std::fs::read(&path).expect("reading the module").as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
            let precise = out.join("kernels-EF64-precise.spv");
            std::fs::write(&precise, no_contraction(&words).iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).expect("writing the module");
            println!("cargo::rustc-env=HYDROPLANE_SOLVERS_SPV_EF64_PRECISE={}", precise.display());
        }
    }
}

/// The shader crate under its real manifest name, in `out`. Its manifest is kept as
/// `Cargo.shader.toml` because cargo leaves any directory holding a `Cargo.toml` out of the
/// published package.
#[cfg(feature = "vulkan")]
fn staged_kernels(source: &std::path::Path, out: &std::path::Path) -> std::path::PathBuf {
    fn copy(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).expect("staging the kernels");
        for entry in std::fs::read_dir(from).expect("reading the kernels") {
            let entry = entry.expect("reading the kernels");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("reading the kernels").is_dir() {
                copy(&entry.path(), &target);
            } else if entry.file_name() == "Cargo.shader.toml" {
                std::fs::copy(entry.path(), to.join("Cargo.toml")).expect("staging the kernels");
            } else {
                std::fs::copy(entry.path(), target).expect("staging the kernels");
            }
        }
    }
    let dir = out.join("kernels-src");
    copy(source, &dir);
    dir
}

/// `module` with every floating-point add, subtract, multiply, divide and negate decorated
/// `NoContraction`, which rust-gpu does not emit and without which a driver may fuse and
/// rearrange the operations `Ef64` arithmetic depends on (NVIDIA's does).
#[cfg(feature = "vulkan")]
fn no_contraction(module: &[u32]) -> Vec<u32> {
    const OP_DECORATE: u32 = 71;
    const NO_CONTRACTION: u32 = 42;
    const FLOAT_ARITHMETIC: [u32; 5] = [127, 129, 131, 133, 136];
    let (header, body) = module.split_at(5);
    let mut ids = Vec::new();
    let mut first_decoration = None;
    let mut at = 0;
    while at < body.len() {
        let (count, op) = ((body[at] >> 16) as usize, body[at] & 0xffff);
        if op == OP_DECORATE && first_decoration.is_none() {
            first_decoration = Some(at);
        }
        if FLOAT_ARITHMETIC.contains(&op) {
            ids.push(body[at + 2]);
        }
        at += count;
    }
    let insert = first_decoration.expect("a module with decorations");
    let mut out = header.to_vec();
    out.extend_from_slice(&body[..insert]);
    for id in ids {
        out.extend([3 << 16 | OP_DECORATE, id, NO_CONTRACTION]);
    }
    out.extend_from_slice(&body[insert..]);
    out
}
