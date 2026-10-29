//! The Vulkan probe: one instance, every physical device, property and feature queries only.
//! Extension structs are chained only when the device offers the extension, and core 1.2 / 1.3
//! structs only when the device's API version has them.

use ash::vk;

use super::arch::{self, GpuVendor};
use super::*;

pub fn enumerate() -> Result<Vec<Gpu>, Error> {
    // SAFETY: the instance is created and destroyed here, and every query passes structs that
    // live until the call returns.
    unsafe {
        let entry = load()?;
        let loader = entry.try_enumerate_instance_version().map_err(vk_error)?.unwrap_or(vk::API_VERSION_1_0);
        if loader < vk::API_VERSION_1_1 {
            return Err(Error::TooOld(Version::from_vk(loader)));
        }
        let available = entry.enumerate_instance_extension_properties(None).map_err(vk_error)?;
        let mut exts = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if available.iter().any(|e| e.extension_name_as_c_str() == Ok(ash::khr::portability_enumeration::NAME)) {
            exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let app = vk::ApplicationInfo::default().application_name(c"hydroplane-whatcompute").api_version(loader);
        let info = vk::InstanceCreateInfo::default().application_info(&app).flags(flags).enabled_extension_names(&exts);
        let instance = entry.create_instance(&info, None).map_err(vk_error)?;
        let gpus = instance
            .enumerate_physical_devices()
            .map_err(vk_error)
            .map(|devices| devices.into_iter().map(|p| describe(&entry, &instance, loader, p)).collect());
        instance.destroy_instance(None);
        gpus
    }
}

/// The system Vulkan loader: dyld's / ld.so's default search first, then on macOS, where
/// Homebrew and the LunarG SDK install outside dyld's default paths, `$VULKAN_SDK/lib`,
/// `$HOMEBREW_PREFIX/lib`, `/opt/homebrew/lib` and `/usr/local/lib`. MoltenVK on its own is the
/// last resort, for machines with the ICD but no loader.
///
/// # Safety
/// Loading runs the library's initializers; see [`ash::Entry::load`].
pub unsafe fn load() -> Result<ash::Entry, Error> {
    let first = match unsafe { ash::Entry::load() } {
        Ok(entry) => return Ok(entry),
        Err(e) => e,
    };
    #[cfg(target_os = "macos")]
    for path in macos_candidates() {
        if let Ok(entry) = unsafe { ash::Entry::load_from(&path) } {
            return Ok(entry);
        }
    }
    Err(Error::Loader(first.to_string()))
}

#[cfg(target_os = "macos")]
fn macos_candidates() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let mut dirs: Vec<PathBuf> = ["VULKAN_SDK", "HOMEBREW_PREFIX"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(|root| PathBuf::from(root).join("lib"))
        .collect();
    for dir in ["/opt/homebrew/lib", "/usr/local/lib"].map(PathBuf::from) {
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    let loaders = dirs.iter().flat_map(|d| ["libvulkan.1.dylib", "libvulkan.dylib"].map(|f| d.join(f)));
    let icds = dirs.iter().map(|d| d.join("libMoltenVK.dylib"));
    loaders.chain(icds).filter(|p| p.exists()).collect()
}

fn vk_error(r: vk::Result) -> Error {
    Error::Vulkan(r.as_raw())
}

fn cstr(chars: &[core::ffi::c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn b(v: vk::Bool32) -> bool {
    v != vk::FALSE
}

unsafe fn describe(entry: &ash::Entry, instance: &ash::Instance, loader: u32, pdev: vk::PhysicalDevice) -> Gpu {
    let core = unsafe { instance.get_physical_device_properties(pdev) };
    let extensions: Vec<String> = unsafe { instance.enumerate_device_extension_properties(pdev) }
        .unwrap_or_default()
        .iter()
        .map(|e| cstr(&e.extension_name))
        .collect();
    let has = |name: &str| extensions.iter().any(|e| e == name);
    let api = core.api_version.min(loader);
    let v12 = api >= vk::API_VERSION_1_2;
    let v13 = api >= vk::API_VERSION_1_3;

    let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
    let mut psub = vk::PhysicalDeviceSubgroupProperties::default();
    let mut psize = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut pdriver = vk::PhysicalDeviceDriverProperties::default();
    let mut ppci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut pnv = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
    let mut pamd = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
    let mut pamd2 = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
    let mut parm = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
    let has_driver = v12 || has("VK_KHR_driver_properties");
    let has_size_control = v13 || has("VK_EXT_subgroup_size_control");
    let has_pci = has("VK_EXT_pci_bus_info");
    let has_nv = has("VK_NV_shader_sm_builtins");
    let has_amd = has("VK_AMD_shader_core_properties");
    let has_amd2 = has("VK_AMD_shader_core_properties2");
    let has_arm = has("VK_ARM_shader_core_builtins");
    {
        let mut p2 = vk::PhysicalDeviceProperties2::default();
        p2 = if v12 { p2.push_next(&mut p11) } else { p2.push_next(&mut psub) };
        if v13 {
            p2 = p2.push_next(&mut p13);
        } else if has_size_control {
            p2 = p2.push_next(&mut psize);
        }
        if has_driver {
            p2 = p2.push_next(&mut pdriver);
        }
        if has_pci {
            p2 = p2.push_next(&mut ppci);
        }
        if has_nv {
            p2 = p2.push_next(&mut pnv);
        }
        if has_amd {
            p2 = p2.push_next(&mut pamd);
        }
        if has_amd2 {
            p2 = p2.push_next(&mut pamd2);
        }
        if has_arm {
            p2 = p2.push_next(&mut parm);
        }
        unsafe { instance.get_physical_device_properties2(pdev, &mut p2) };
    }
    if v12 {
        psub.subgroup_size = p11.subgroup_size;
        psub.supported_stages = p11.subgroup_supported_stages;
        psub.supported_operations = p11.subgroup_supported_operations;
    }
    if v13 {
        psize.min_subgroup_size = p13.min_subgroup_size;
        psize.max_subgroup_size = p13.max_subgroup_size;
        psize.max_compute_workgroup_subgroups = p13.max_compute_workgroup_subgroups;
    }

    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut fatomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
    let mut fcoop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut faccel = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut frtp = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
    let mut frq = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let mut fmesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
    let mut fclock = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
    let has_atomic = has("VK_EXT_shader_atomic_float");
    let has_coop = has("VK_KHR_cooperative_matrix");
    let has_accel = has("VK_KHR_acceleration_structure");
    let has_rtp = has("VK_KHR_ray_tracing_pipeline");
    let has_rq = has("VK_KHR_ray_query");
    let has_mesh = has("VK_EXT_mesh_shader");
    let has_clock = has("VK_KHR_shader_clock");
    let base = {
        let mut f2 = vk::PhysicalDeviceFeatures2::default();
        if v12 {
            f2 = f2.push_next(&mut f11).push_next(&mut f12);
        }
        if v13 {
            f2 = f2.push_next(&mut f13);
        }
        if has_atomic {
            f2 = f2.push_next(&mut fatomic);
        }
        if has_coop {
            f2 = f2.push_next(&mut fcoop);
        }
        if has_accel {
            f2 = f2.push_next(&mut faccel);
        }
        if has_rtp {
            f2 = f2.push_next(&mut frtp);
        }
        if has_rq {
            f2 = f2.push_next(&mut frq);
        }
        if has_mesh {
            f2 = f2.push_next(&mut fmesh);
        }
        if has_clock {
            f2 = f2.push_next(&mut fclock);
        }
        unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
        f2.features
    };

    let vendor = match GpuVendor::from_id(core.vendor_id) {
        GpuVendor::Unknown if cstr(&core.device_name).starts_with("Apple ") => GpuVendor::Apple,
        v => v,
    };
    let name = cstr(&core.device_name);
    let gpu_arch = arch::decode(vendor, core.device_id, &name);
    let kind = match core.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => GpuKind::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => GpuKind::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => GpuKind::Virtual,
        vk::PhysicalDeviceType::CPU => GpuKind::Cpu,
        _ => GpuKind::Other,
    };
    let software = kind == GpuKind::Cpu;

    let driver_id = if has_driver { pdriver.driver_id.as_raw() as u32 } else { 0 };
    let driver = Driver {
        id: driver_id,
        name: if has_driver { cstr(&pdriver.driver_name) } else { String::new() },
        info: if has_driver { cstr(&pdriver.driver_info) } else { String::new() },
        version: driver_version(vendor, driver_id, core.driver_version),
        raw_version: core.driver_version,
    };

    let units = if has_nv && pnv.shader_sm_count > 0 {
        Some(pnv.shader_sm_count)
    } else if has_amd2 && pamd2.active_compute_unit_count > 0 {
        Some(pamd2.active_compute_unit_count)
    } else if has_amd && pamd.compute_units_per_shader_array > 0 {
        Some(pamd.shader_engine_count * pamd.shader_arrays_per_engine_count * pamd.compute_units_per_shader_array)
    } else if has_arm && parm.shader_core_count > 0 {
        Some(parm.shader_core_count)
    } else {
        platform_units(vendor)
    };
    let waves_per_unit = if has_nv && pnv.shader_warps_per_sm > 0 {
        Some(pnv.shader_warps_per_sm)
    } else if has_amd && pamd.simd_per_compute_unit > 0 {
        Some(pamd.simd_per_compute_unit * pamd.wavefronts_per_simd)
    } else if has_arm && parm.shader_warps_per_core > 0 {
        Some(parm.shader_warps_per_core)
    } else {
        None
    };
    let layout = gpu_arch.unit_layout();
    let total = |per: Option<u32>| units.zip(per).map(|(n, p)| n * p);
    let cores = Cores {
        unit: layout.map(|l| l.name).or(if has_nv {
            Some("SM")
        } else if has_amd || has_amd2 {
            Some("CU")
        } else if has_arm {
            Some("core")
        } else {
            None
        }),
        units,
        fp32_lanes: total(layout.map(|l| l.fp32_lanes as u32)),
        matrix_units: total(layout.and_then(|l| l.matrix_units).map(u32::from)),
        rt_units: total(layout.and_then(|l| l.rt_units).map(u32::from)),
        waves_per_unit,
    };

    let cooperative_matrix = has_coop && b(fcoop.cooperative_matrix);
    let shapes = if cooperative_matrix {
        let coop = ash::khr::cooperative_matrix::Instance::new(entry, instance);
        unsafe { coop.get_physical_device_cooperative_matrix_properties(pdev) }
            .unwrap_or_default()
            .iter()
            .map(|p| MatrixShape {
                m: p.m_size,
                n: p.n_size,
                k: p.k_size,
                a: elem(p.a_type),
                b: elem(p.b_type),
                c: elem(p.c_type),
                result: elem(p.result_type),
                saturating: b(p.saturating_accumulation),
                scope: scope(p.scope),
            })
            .collect()
    } else {
        Vec::new()
    };
    let matrix = Matrix {
        hardware: if software { Some(false) } else { gpu_arch.matrix_hardware() },
        cooperative_matrix,
        shapes,
        cooperative_matrix2: has("VK_NV_cooperative_matrix2"),
        cooperative_vector: has("VK_NV_cooperative_vector"),
    };

    let ray_tracing = RayTracing {
        hardware: if software { Some(false) } else { gpu_arch.ray_tracing_hardware() },
        acceleration_structure: has_accel && b(faccel.acceleration_structure),
        pipeline: has_rtp && b(frtp.ray_tracing_pipeline),
        query: has_rq && b(frq.ray_query),
        position_fetch: has("VK_KHR_ray_tracing_position_fetch"),
        invocation_reorder: has("VK_NV_ray_tracing_invocation_reorder") || has("VK_EXT_ray_tracing_invocation_reorder"),
        opacity_micromap: has("VK_EXT_opacity_micromap"),
        motion_blur: has("VK_NV_ray_tracing_motion_blur"),
    };

    let memory = unsafe { memory(instance, pdev, kind, has("VK_EXT_memory_budget"), p11.max_memory_allocation_size, v12) };

    let ops = psub.supported_operations;
    let subgroup = Subgroup {
        size: psub.subgroup_size,
        min_size: if has_size_control { psize.min_subgroup_size } else { psub.subgroup_size },
        max_size: if has_size_control { psize.max_subgroup_size } else { psub.subgroup_size },
        size_control: if v13 { b(f13.subgroup_size_control) } else { has_size_control },
        in_compute: psub.supported_stages.contains(vk::ShaderStageFlags::COMPUTE),
        extended_types: b(f12.shader_subgroup_extended_types),
        vote: ops.contains(vk::SubgroupFeatureFlags::VOTE),
        arithmetic: ops.contains(vk::SubgroupFeatureFlags::ARITHMETIC),
        ballot: ops.contains(vk::SubgroupFeatureFlags::BALLOT),
        shuffle: ops.contains(vk::SubgroupFeatureFlags::SHUFFLE),
        shuffle_relative: ops.contains(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
        clustered: ops.contains(vk::SubgroupFeatureFlags::CLUSTERED),
        quad: ops.contains(vk::SubgroupFeatureFlags::QUAD),
        rotate: has("VK_KHR_shader_subgroup_rotate"),
    };

    let shader = ShaderFeatures {
        f16: b(f12.shader_float16),
        f64: b(base.shader_float64),
        bf16: has("VK_KHR_shader_bfloat16"),
        fp8: has("VK_EXT_shader_float8"),
        i8: b(f12.shader_int8),
        i16: b(base.shader_int16),
        i64: b(base.shader_int64),
        storage_8bit: b(f12.storage_buffer8_bit_access),
        storage_16bit: b(f11.storage_buffer16_bit_access),
        int_dot_product: b(f13.shader_integer_dot_product),
        int_dot_product_accelerated: b(p13.integer_dot_product4x8_bit_packed_signed_accelerated),
        atomic_i64: b(f12.shader_buffer_int64_atomics),
        atomic_f32_add: b(fatomic.shader_buffer_float32_atomic_add),
        atomic_f64_add: b(fatomic.shader_buffer_float64_atomic_add),
        clock: has_clock && (b(fclock.shader_subgroup_clock) || b(fclock.shader_device_clock)),
        zero_init_workgroup_memory: b(f13.shader_zero_initialize_workgroup_memory),
    };

    let features = Features {
        timeline_semaphore: b(f12.timeline_semaphore),
        buffer_device_address: b(f12.buffer_device_address),
        descriptor_indexing: b(f12.descriptor_indexing),
        vulkan_memory_model: b(f12.vulkan_memory_model),
        vulkan_memory_model_device_scope: b(f12.vulkan_memory_model_device_scope),
        synchronization2: b(f13.synchronization2),
        dynamic_rendering: b(f13.dynamic_rendering),
        maintenance4: b(f13.maintenance4),
        compute_full_subgroups: b(f13.compute_full_subgroups),
        mesh_shader: has_mesh && b(fmesh.mesh_shader),
        push_descriptor: has("VK_KHR_push_descriptor"),
        external_memory_host: has("VK_EXT_external_memory_host"),
        sparse_binding: b(base.sparse_binding),
        portability_subset: has("VK_KHR_portability_subset"),
    };

    let l = &core.limits;
    let limits = Limits {
        workgroup_shared_bytes: l.max_compute_shared_memory_size,
        workgroup_invocations: l.max_compute_work_group_invocations,
        workgroup_size: l.max_compute_work_group_size,
        workgroup_count: l.max_compute_work_group_count,
        workgroup_subgroups: has_size_control.then_some(psize.max_compute_workgroup_subgroups),
        storage_buffer_range: l.max_storage_buffer_range,
        max_buffer_bytes: v13.then_some(p13.max_buffer_size),
        push_constant_bytes: l.max_push_constants_size,
        storage_buffer_alignment: l.min_storage_buffer_offset_alignment,
        non_coherent_atom: l.non_coherent_atom_size,
        timestamp_period_ns: l.timestamp_period,
    };

    let queues = unsafe { instance.get_physical_device_queue_family_properties(pdev) }
        .iter()
        .map(|q| QueueFamily {
            count: q.queue_count,
            graphics: q.queue_flags.contains(vk::QueueFlags::GRAPHICS),
            compute: q.queue_flags.contains(vk::QueueFlags::COMPUTE),
            transfer: q.queue_flags.contains(vk::QueueFlags::TRANSFER),
            sparse: q.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING),
            video_decode: q.queue_flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR),
            video_encode: q.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR),
            timestamp_bits: q.timestamp_valid_bits,
        })
        .collect();

    Gpu {
        name,
        vendor,
        arch: gpu_arch,
        kind,
        vendor_id: core.vendor_id,
        device_id: core.device_id,
        api_version: Version::from_vk(core.api_version),
        driver,
        uuid: if v12 { p11.device_uuid } else { [0; 16] },
        pci: has_pci.then_some(PciAddress {
            domain: ppci.pci_domain,
            bus: ppci.pci_bus,
            device: ppci.pci_device,
            function: ppci.pci_function,
        }),
        cores,
        matrix,
        ray_tracing,
        memory,
        subgroup,
        shader,
        features,
        limits,
        queues,
        extensions,
    }
}

unsafe fn memory(
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
    kind: GpuKind,
    has_budget: bool,
    max_allocation: u64,
    v12: bool,
) -> Memory {
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let props = {
        let mut m2 = vk::PhysicalDeviceMemoryProperties2::default();
        if has_budget {
            m2 = m2.push_next(&mut budget);
        }
        unsafe { instance.get_physical_device_memory_properties2(pdev, &mut m2) };
        m2.memory_properties
    };
    let types = &props.memory_types[..props.memory_type_count as usize];
    let heaps: Vec<Heap> = props.memory_heaps[..props.memory_heap_count as usize]
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let any = |f: vk::MemoryPropertyFlags| types.iter().any(|t| t.heap_index as usize == i && t.property_flags.contains(f));
            use vk::MemoryPropertyFlags as M;
            Heap {
                bytes: h.size,
                device_local: h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                host_visible: any(M::HOST_VISIBLE),
                host_coherent: any(M::HOST_VISIBLE | M::HOST_COHERENT),
                host_cached: any(M::HOST_VISIBLE | M::HOST_CACHED),
                device_local_host_visible: any(M::DEVICE_LOCAL | M::HOST_VISIBLE),
                budget: has_budget.then_some(budget.heap_budget[i]),
                usage: has_budget.then_some(budget.heap_usage[i]),
            }
        })
        .collect();
    Memory {
        host_access: host_access(kind, &heaps),
        device_local_bytes: heaps.iter().filter(|h| h.device_local).map(|h| h.bytes).max().unwrap_or(0),
        mappable_device_local_bytes: heaps.iter().filter(|h| h.device_local_host_visible).map(|h| h.bytes).max().unwrap_or(0),
        max_allocation_bytes: if v12 { max_allocation } else { 0 },
        heaps,
    }
}

#[cfg(target_vendor = "apple")]
fn platform_units(vendor: GpuVendor) -> Option<u32> {
    if vendor == GpuVendor::Apple { super::apple::gpu_core_count() } else { None }
}

#[cfg(not(target_vendor = "apple"))]
fn platform_units(_: GpuVendor) -> Option<u32> {
    None
}

fn driver_version(vendor: GpuVendor, driver_id: u32, v: u32) -> String {
    const INTEL_WINDOWS: u32 = 5;
    match vendor {
        GpuVendor::Nvidia => format!("{}.{}.{:02}", v >> 22, (v >> 14) & 0xFF, (v >> 6) & 0xFF),
        GpuVendor::Intel if driver_id == INTEL_WINDOWS => format!("{}.{}", v >> 14, v & 0x3FFF),
        _ => Version::from_vk(v).to_string(),
    }
}

fn elem(t: vk::ComponentTypeKHR) -> Elem {
    const BFLOAT16: i32 = 1_000_141_000;
    const FLOAT8_E4M3: i32 = 1_000_491_002;
    const FLOAT8_E5M2: i32 = 1_000_491_003;
    match t.as_raw() {
        0 => Elem::F16,
        1 => Elem::F32,
        2 => Elem::F64,
        3 => Elem::I8,
        4 => Elem::I16,
        5 => Elem::I32,
        6 => Elem::I64,
        7 => Elem::U8,
        8 => Elem::U16,
        9 => Elem::U32,
        10 => Elem::U64,
        BFLOAT16 => Elem::Bf16,
        FLOAT8_E4M3 => Elem::Fp8E4m3,
        FLOAT8_E5M2 => Elem::Fp8E5m2,
        other => Elem::Other(other),
    }
}

fn scope(s: vk::ScopeKHR) -> Scope {
    match s {
        vk::ScopeKHR::DEVICE => Scope::Device,
        vk::ScopeKHR::WORKGROUP => Scope::Workgroup,
        vk::ScopeKHR::SUBGROUP => Scope::Subgroup,
        vk::ScopeKHR::QUEUE_FAMILY => Scope::QueueFamily,
        other => Scope::Other(other.as_raw()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_versions() {
        let nv = (580 << 22) | (65 << 14) | (6 << 6);
        assert_eq!(driver_version(GpuVendor::Nvidia, 4, nv), "580.65.06");
        assert_eq!(driver_version(GpuVendor::Intel, 5, (101 << 14) | 6130), "101.6130");
        assert_eq!(driver_version(GpuVendor::Amd, 3, vk::make_api_version(0, 25, 2, 1)), "25.2.1");
    }
}
