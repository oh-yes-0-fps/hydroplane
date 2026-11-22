//! What the machine offers, detected once per process.

use std::sync::OnceLock;

use crate::Precision;

/// The host's devices as the planner sees them: plain data, so a plan can be computed (and
/// tested) for any machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Caps {
    /// Threads the CPU factorization and iterations run on.
    pub threads: usize,
    /// Apple's Accelerate sparse Cholesky (macOS).
    pub accelerate: bool,
    /// SME2 matrix units for [`crate::sme`] (Apple M4 and later, through Accelerate).
    pub sme: bool,
    pub gpu: Option<GpuCaps>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCaps {
    pub name: String,
    /// Native `f64` kernels (`shaderFloat64`).
    pub f64: bool,
    pub discrete: bool,
    /// Memory shared with the host (Apple silicon).
    pub unified_memory: bool,
}

impl GpuCaps {
    /// Precisions the GPU runs kernels in.
    pub fn precisions(&self) -> Vec<Precision> {
        let mut p = vec![Precision::F32, Precision::Ef64];
        if self.f64 {
            p.insert(1, Precision::F64);
        }
        p
    }
}

impl Caps {
    /// This machine's, detected on first use: the thread count, and the Vulkan device that
    /// [`Self::gpu`] describes (the first discrete GPU, else the first device), opened once and
    /// shared by every solver in the process.
    pub fn host() -> &'static Caps {
        static CAPS: OnceLock<Caps> = OnceLock::new();
        CAPS.get_or_init(|| Caps {
            threads: rayon::current_num_threads(),
            accelerate: cfg!(accelerate),
            sme: sme(),
            gpu: gpu_caps(),
        })
    }
}

#[cfg(feature = "vulkan")]
fn gpu_caps() -> Option<GpuCaps> {
    let ctx = context()?;
    Some(GpuCaps { name: ctx.name().to_string(), f64: ctx.supports_f64(), discrete: ctx.discrete(), unified_memory: ctx.unified_memory() })
}

#[cfg(not(feature = "vulkan"))]
fn gpu_caps() -> Option<GpuCaps> {
    None
}

/// The process's Vulkan context, if there is a device.
#[cfg(feature = "vulkan")]
pub(crate) fn context() -> Option<std::sync::Arc<crate::vulkan::Context>> {
    crate::vulkan::Context::shared().ok()
}

#[cfg(accelerate)]
fn sme() -> bool {
    crate::sme::available()
}

#[cfg(not(accelerate))]
fn sme() -> bool {
    false
}
