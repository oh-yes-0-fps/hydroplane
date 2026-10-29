//! Enumerate the GPUs the Vulkan loader can see and describe each one: design, unit counts,
//! matrix and ray-tracing support, memory topology, shader features and compute limits.
//!
//! Probing creates and destroys a Vulkan instance (tens of milliseconds); no logical device is
//! created. [`current`] probes once per process, [`detect`] every call.

mod arch;
#[cfg(target_vendor = "apple")]
mod apple;
mod vulkan;

use std::fmt;
use std::sync::OnceLock;

pub use arch::{GpuArch, GpuVendor, UnitLayout};
pub use vulkan::load as load_vulkan;
/// The `ash` this crate is built against, so [`load_vulkan`]'s entry has a nameable type.
pub use ash;

/// One Vulkan physical device.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Gpu {
    /// `deviceName` as the driver reports it.
    pub name: String,
    /// Who designed it.
    pub vendor: GpuVendor,
    /// Which generation it is.
    pub arch: GpuArch,
    /// Discrete, integrated, virtual or a CPU implementation.
    pub kind: GpuKind,
    /// PCI (or Khronos) vendor id.
    pub vendor_id: u32,
    /// PCI device id; synthesized by MoltenVK.
    pub device_id: u32,
    /// Highest Vulkan version the device supports.
    pub api_version: Version,
    /// The driver behind it.
    pub driver: Driver,
    /// `deviceUUID`, stable across processes and APIs for the same device.
    pub uuid: [u8; 16],
    /// PCI address (`VK_EXT_pci_bus_info`).
    pub pci: Option<PciAddress>,
    /// Shader, matrix and ray-tracing unit counts.
    pub cores: Cores,
    /// Cooperative matrix (tensor core) support.
    pub matrix: Matrix,
    /// Ray tracing support.
    pub ray_tracing: RayTracing,
    /// Heaps, host visibility of device memory, budget.
    pub memory: Memory,
    /// Subgroup (warp / wavefront / SIMD-group) sizes and operations.
    pub subgroup: Subgroup,
    /// Arithmetic types and atomics shaders may use.
    pub shader: ShaderFeatures,
    /// API features relevant to compute.
    pub features: Features,
    /// Compute dispatch and buffer limits.
    pub limits: Limits,
    /// Queue families.
    pub queues: Vec<QueueFamily>,
    /// Every device extension the driver offers.
    pub extensions: Vec<String>,
}

impl Gpu {
    /// Whether the driver offers device extension `name` (e.g. `"VK_KHR_cooperative_matrix"`).
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|e| e == name)
    }
}

/// `VkPhysicalDeviceType`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum GpuKind {
    /// A separate card with its own memory.
    Discrete,
    /// Shares the package and memory with the CPU.
    Integrated,
    /// A virtualized device (VM passthrough or paravirtual).
    Virtual,
    /// A software implementation on the CPU.
    Cpu,
    /// Anything else.
    Other,
}

/// A Vulkan `major.minor.patch` version.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Version {
    /// Major.
    pub major: u32,
    /// Minor.
    pub minor: u32,
    /// Patch.
    pub patch: u32,
}

impl Version {
    /// Decodes `VK_MAKE_API_VERSION` (the variant bits are dropped).
    pub const fn from_vk(v: u32) -> Self {
        Version { major: (v >> 22) & 0x7F, minor: (v >> 12) & 0x3FF, patch: v & 0xFFF }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The driver (`VkPhysicalDeviceDriverProperties`).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Driver {
    /// Raw `VkDriverId` (1 AMD proprietary, 3 RADV, 4 Nvidia, 5 Intel Windows, 6 ANV,
    /// 13 llvmpipe, 14 MoltenVK, 24 NVK, ...); 0 when unreported.
    pub id: u32,
    /// `driverName` ("NVIDIA", "radv", "MoltenVK").
    pub name: String,
    /// `driverInfo`, usually the version string.
    pub info: String,
    /// `driverVersion` decoded with the vendor's own packing.
    pub version: String,
    /// `driverVersion` as reported.
    pub raw_version: u32,
}

/// PCI domain:bus:device.function.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct PciAddress {
    /// Domain.
    pub domain: u32,
    /// Bus.
    pub bus: u32,
    /// Device.
    pub device: u32,
    /// Function.
    pub function: u32,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}:{:02x}:{:02x}.{}", self.domain, self.bus, self.device, self.function)
    }
}

/// Unit counts. The number of units comes from the driver (`VK_NV_shader_sm_builtins`,
/// `VK_AMD_shader_core_properties2`, `VK_ARM_shader_core_builtins`, or the IORegistry on
/// Apple); the totals multiply it by [`GpuArch::unit_layout`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct Cores {
    /// What a unit is called ("SM", "CU", "Xe-core", "core").
    pub unit: Option<&'static str>,
    /// Shader units on the device.
    pub units: Option<u32>,
    /// FP32 lanes across the device.
    pub fp32_lanes: Option<u32>,
    /// Matrix engines (tensor cores, XMX, AI / neural accelerators) across the device.
    pub matrix_units: Option<u32>,
    /// Ray-tracing units across the device.
    pub rt_units: Option<u32>,
    /// Resident warps / wavefronts per unit.
    pub waves_per_unit: Option<u32>,
}

/// Element type of a cooperative matrix operand.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[allow(missing_docs)]
pub enum Elem {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    /// FP8 E4M3 (`VK_EXT_shader_float8`).
    Fp8E4m3,
    /// FP8 E5M2 (`VK_EXT_shader_float8`).
    Fp8E5m2,
    /// A `VkComponentTypeKHR` this crate does not name.
    Other(i32),
}

/// Which invocations cooperate on one matrix.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[allow(missing_docs)]
pub enum Scope {
    Device,
    Workgroup,
    Subgroup,
    QueueFamily,
    Other(i32),
}

/// One `VkCooperativeMatrixPropertiesKHR`: `Result = A(M×K) · B(K×N) + C(M×N)`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[allow(missing_docs)]
pub struct MatrixShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: Elem,
    pub b: Elem,
    pub c: Elem,
    pub result: Elem,
    pub saturating: bool,
    pub scope: Scope,
}

/// Matrix-multiply hardware as Vulkan exposes it.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct Matrix {
    /// Dedicated matrix engines in the design ([`GpuArch::matrix_hardware`]); `None` when unknown.
    pub hardware: Option<bool>,
    /// `VK_KHR_cooperative_matrix` with its `cooperativeMatrix` feature.
    pub cooperative_matrix: bool,
    /// Shapes and types the driver accelerates.
    pub shapes: Vec<MatrixShape>,
    /// `VK_NV_cooperative_matrix2` (flexible sizes, workgroup scope, reductions).
    pub cooperative_matrix2: bool,
    /// `VK_NV_cooperative_vector` (matrix-vector on tensor cores).
    pub cooperative_vector: bool,
}

impl Matrix {
    /// Shapes whose A and B operands are `elem`.
    pub fn shapes_for(&self, elem: Elem) -> impl Iterator<Item = &MatrixShape> {
        self.shapes.iter().filter(move |s| s.a == elem && s.b == elem)
    }
}

/// Ray tracing support.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct RayTracing {
    /// Fixed-function traversal / intersection hardware ([`GpuArch::ray_tracing_hardware`]);
    /// `None` when unknown.
    pub hardware: Option<bool>,
    /// `VK_KHR_acceleration_structure`.
    pub acceleration_structure: bool,
    /// `VK_KHR_ray_tracing_pipeline`.
    pub pipeline: bool,
    /// `VK_KHR_ray_query` (inline rays from any shader, compute included).
    pub query: bool,
    /// `VK_KHR_ray_tracing_position_fetch`.
    pub position_fetch: bool,
    /// Shader execution reordering (`VK_NV_` or `VK_EXT_ray_tracing_invocation_reorder`).
    pub invocation_reorder: bool,
    /// `VK_EXT_opacity_micromap`.
    pub opacity_micromap: bool,
    /// `VK_NV_ray_tracing_motion_blur`.
    pub motion_blur: bool,
}

/// How much device memory the host can map directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum HostAccess {
    /// Device memory is system memory (integrated GPUs, Apple silicon): no copies needed.
    Unified,
    /// Resizable BAR / Smart Access Memory: all of VRAM is host-mappable.
    ResizableBar,
    /// The classic small BAR window (usually 256 MiB) of host-mappable VRAM.
    Window(u64),
    /// No device-local memory is host-visible.
    None,
}

/// One `VkMemoryHeap` and the properties of the memory types in it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct Heap {
    /// Heap size in bytes.
    pub bytes: u64,
    /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`.
    pub device_local: bool,
    /// Some memory type in the heap is host-visible.
    pub host_visible: bool,
    /// Some memory type in the heap is host-visible and host-coherent.
    pub host_coherent: bool,
    /// Some memory type in the heap is host-cached.
    pub host_cached: bool,
    /// Some memory type in the heap is both device-local and host-visible.
    pub device_local_host_visible: bool,
    /// What this process may allocate now (`VK_EXT_memory_budget`).
    pub budget: Option<u64>,
    /// What this process has allocated (`VK_EXT_memory_budget`).
    pub usage: Option<u64>,
}

/// Memory topology.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Memory {
    /// Unified, resizable BAR, small BAR window, or none.
    pub host_access: HostAccess,
    /// Largest device-local heap: VRAM on a discrete GPU, the shared pool on an integrated one.
    pub device_local_bytes: u64,
    /// Largest heap holding a device-local, host-visible memory type.
    pub mappable_device_local_bytes: u64,
    /// Largest single allocation (`maxMemoryAllocationSize`).
    pub max_allocation_bytes: u64,
    /// Every heap.
    pub heaps: Vec<Heap>,
}

impl Memory {
    /// Device memory is host memory.
    pub fn unified(&self) -> bool {
        self.host_access == HostAccess::Unified
    }

    /// All of a discrete GPU's VRAM is host-mappable.
    pub fn resizable_bar(&self) -> bool {
        self.host_access == HostAccess::ResizableBar
    }
}

/// Subgroup (warp / wavefront / SIMD-group) properties.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct Subgroup {
    /// Default subgroup size.
    pub size: u32,
    /// Smallest size a pipeline may request (`VK_EXT_subgroup_size_control`).
    pub min_size: u32,
    /// Largest size a pipeline may request.
    pub max_size: u32,
    /// Pipelines may pin the subgroup size.
    pub size_control: bool,
    /// Compute shaders can use subgroup operations.
    pub in_compute: bool,
    /// Subgroup ops on 8/16/64-bit types (`shaderSubgroupExtendedTypes`).
    pub extended_types: bool,
    #[allow(missing_docs)]
    pub vote: bool,
    #[allow(missing_docs)]
    pub arithmetic: bool,
    #[allow(missing_docs)]
    pub ballot: bool,
    #[allow(missing_docs)]
    pub shuffle: bool,
    #[allow(missing_docs)]
    pub shuffle_relative: bool,
    #[allow(missing_docs)]
    pub clustered: bool,
    #[allow(missing_docs)]
    pub quad: bool,
    /// `VK_KHR_shader_subgroup_rotate`.
    pub rotate: bool,
}

/// Arithmetic types, atomics and other shader capabilities.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct ShaderFeatures {
    /// `shaderFloat16`.
    pub f16: bool,
    /// `shaderFloat64`.
    pub f64: bool,
    /// `VK_KHR_shader_bfloat16`.
    pub bf16: bool,
    /// `VK_EXT_shader_float8`.
    pub fp8: bool,
    /// `shaderInt8`.
    pub i8: bool,
    /// `shaderInt16`.
    pub i16: bool,
    /// `shaderInt64`.
    pub i64: bool,
    /// 8-bit storage buffer access.
    pub storage_8bit: bool,
    /// 16-bit storage buffer access.
    pub storage_16bit: bool,
    /// `shaderIntegerDotProduct` (packed int8 dot products).
    pub int_dot_product: bool,
    /// The 4x8-bit packed signed dot product runs in hardware.
    pub int_dot_product_accelerated: bool,
    /// 64-bit integer atomics on buffers.
    pub atomic_i64: bool,
    /// `f32` atomic add on buffers.
    pub atomic_f32_add: bool,
    /// `f64` atomic add on buffers.
    pub atomic_f64_add: bool,
    /// `VK_KHR_shader_clock` subgroup / device clocks.
    pub clock: bool,
    /// Workgroup memory may be zero-initialized.
    pub zero_init_workgroup_memory: bool,
}

/// API features relevant to compute.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub struct Features {
    pub timeline_semaphore: bool,
    pub buffer_device_address: bool,
    pub descriptor_indexing: bool,
    pub vulkan_memory_model: bool,
    pub vulkan_memory_model_device_scope: bool,
    pub synchronization2: bool,
    pub dynamic_rendering: bool,
    pub maintenance4: bool,
    pub compute_full_subgroups: bool,
    /// `VK_EXT_mesh_shader`.
    pub mesh_shader: bool,
    /// `VK_KHR_push_descriptor`.
    pub push_descriptor: bool,
    /// `VK_EXT_external_memory_host` (import host pointers as device memory).
    pub external_memory_host: bool,
    pub sparse_binding: bool,
    /// `VK_KHR_portability_subset`: a layered implementation (MoltenVK) with some features cut.
    pub portability_subset: bool,
}

/// Compute dispatch and buffer limits.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[non_exhaustive]
pub struct Limits {
    /// Shared (workgroup / LDS / threadgroup) memory per workgroup, bytes.
    pub workgroup_shared_bytes: u32,
    /// Invocations per workgroup.
    pub workgroup_invocations: u32,
    /// Workgroup size per dimension.
    pub workgroup_size: [u32; 3],
    /// Workgroups per dispatch per dimension.
    pub workgroup_count: [u32; 3],
    /// Subgroups per workgroup.
    pub workgroup_subgroups: Option<u32>,
    /// Largest storage buffer binding, bytes.
    pub storage_buffer_range: u32,
    /// Largest buffer, bytes (`maxBufferSize`).
    pub max_buffer_bytes: Option<u64>,
    /// Push constant space, bytes.
    pub push_constant_bytes: u32,
    /// Storage buffer offset alignment, bytes.
    pub storage_buffer_alignment: u64,
    /// Flush / invalidate granularity of non-coherent memory, bytes.
    pub non_coherent_atom: u64,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period_ns: f32,
}

/// One queue family.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub struct QueueFamily {
    pub count: u32,
    pub graphics: bool,
    pub compute: bool,
    pub transfer: bool,
    pub sparse: bool,
    pub video_decode: bool,
    pub video_encode: bool,
    /// Valid bits in timestamps written on this family; 0 means no timestamps.
    pub timestamp_bits: u32,
}

/// Why GPUs could not be enumerated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No Vulkan loader on the system, or it failed to load.
    Loader(String),
    /// The loader is older than Vulkan 1.1.
    TooOld(Version),
    /// A Vulkan call failed; the `VkResult` code.
    Vulkan(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(e) => write!(f, "Vulkan loader unavailable: {e}"),
            Error::TooOld(v) => write!(f, "Vulkan loader {v} is older than 1.1"),
            Error::Vulkan(r) => write!(f, "Vulkan call failed: VkResult {r}"),
        }
    }
}

impl std::error::Error for Error {}

/// Enumerate the GPUs now.
pub fn detect() -> Result<Vec<Gpu>, Error> {
    vulkan::enumerate()
}

/// The GPUs, enumerated once per process; empty when there is no Vulkan or no device.
pub fn current() -> &'static [Gpu] {
    static GPUS: OnceLock<Vec<Gpu>> = OnceLock::new();
    GPUS.get_or_init(|| detect().unwrap_or_default())
}

/// The GPU a compute workload should default to: the discrete GPU with the most device-local
/// memory, else the first hardware device, else whatever there is.
pub fn primary() -> Option<&'static Gpu> {
    let gpus = current();
    gpus.iter()
        .filter(|g| g.kind == GpuKind::Discrete)
        .max_by_key(|g| g.memory.device_local_bytes)
        .or_else(|| gpus.iter().find(|g| g.kind != GpuKind::Cpu))
        .or_else(|| gpus.first())
}

/// Classify host access from the heaps and the device kind.
fn host_access(kind: GpuKind, heaps: &[Heap]) -> HostAccess {
    const WINDOW_MAX: u64 = 512 << 20;
    let mappable = heaps.iter().filter(|h| h.device_local_host_visible).map(|h| h.bytes).max().unwrap_or(0);
    let all_device_local = !heaps.is_empty() && heaps.iter().all(|h| h.device_local);
    if matches!(kind, GpuKind::Integrated | GpuKind::Cpu) || (all_device_local && mappable > 0) {
        HostAccess::Unified
    } else if mappable > WINDOW_MAX {
        HostAccess::ResizableBar
    } else if mappable > 0 {
        HostAccess::Window(mappable)
    } else {
        HostAccess::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(bytes: u64, device_local: bool, dlhv: bool) -> Heap {
        Heap {
            bytes,
            device_local,
            host_visible: dlhv || !device_local,
            host_coherent: true,
            host_cached: !device_local,
            device_local_host_visible: dlhv,
            budget: None,
            usage: None,
        }
    }

    #[test]
    fn host_access_classes() {
        const G: u64 = 1 << 30;
        let rebar = [heap(32 * G, true, true), heap(64 * G, false, false)];
        assert_eq!(host_access(GpuKind::Discrete, &rebar), HostAccess::ResizableBar);
        let window = [heap(24 * G, true, false), heap(64 * G, false, false), heap(256 << 20, true, true)];
        assert_eq!(host_access(GpuKind::Discrete, &window), HostAccess::Window(256 << 20));
        let none = [heap(8 * G, true, false), heap(16 * G, false, false)];
        assert_eq!(host_access(GpuKind::Discrete, &none), HostAccess::None);
        let apple = [heap(48 * G, true, true)];
        assert_eq!(host_access(GpuKind::Integrated, &apple), HostAccess::Unified);
        assert_eq!(host_access(GpuKind::Virtual, &apple), HostAccess::Unified);
    }

    #[test]
    fn version_decode() {
        let v = Version::from_vk((1 << 22) | (4 << 12) | 357);
        assert_eq!(v, Version { major: 1, minor: 4, patch: 357 });
        assert_eq!(v.to_string(), "1.4.357");
    }
}
