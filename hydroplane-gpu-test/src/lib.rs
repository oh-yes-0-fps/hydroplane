//! Runs `shader/`'s [`Probe`] on a Vulkan device and on the CPU, for comparison.

use ash::vk;
use half::f16;
use hydroplane_gpu_test_shader::{DIM, Probe};

/// `shader/` built with array tiles: runs on any Vulkan device.
pub const PLAIN: &[u8] = include_bytes!(env!("HYDROPLANE_GPU_TEST_PLAIN"));
/// `shader/` built with cooperative-matrix tiles: needs `VK_KHR_cooperative_matrix`.
pub const COOP: &[u8] = include_bytes!(env!("HYDROPLANE_GPU_TEST_COOP"));

/// What one [`Probe`] writes.
#[derive(Debug, Clone, PartialEq)]
pub struct Outputs {
    pub out: Vec<f32>,
    pub row_max: Vec<f32>,
    pub col_sum: Vec<f32>,
}

impl Outputs {
    fn zeroed() -> Self {
        Self { out: vec![0.0; DIM * DIM], row_max: vec![0.0; DIM], col_sum: vec![0.0; DIM] }
    }

    /// Indices where `self` and `other` differ by more than `tol`, as `(buffer, index)`.
    pub fn mismatches(&self, other: &Self, tol: f32) -> Vec<(&'static str, usize)> {
        let mut bad = Vec::new();
        for (name, a, b) in [("out", &self.out, &other.out), ("row_max", &self.row_max, &other.row_max), ("col_sum", &self.col_sum, &other.col_sum)] {
            bad.extend(a.iter().zip(b).enumerate().filter(|(_, (x, y))| (*x - *y).abs() > tol).map(|(i, _)| (name, i)));
        }
        bad
    }
}

/// Inputs whose products and sums are exact in `f32`, so every lowering must agree bit for bit.
pub struct Inputs {
    pub a: Vec<f16>,
    pub b: Vec<f16>,
    pub c: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Inputs {
    pub fn new() -> Self {
        let n = DIM * DIM;
        Self {
            a: (0..n).map(|i| f16::from_f32(((i * 7 % 13) as f32 - 6.0) * 0.5)).collect(),
            b: (0..n).map(|i| f16::from_f32(((i * 5 % 11) as f32 - 5.0) * 0.25)).collect(),
            c: (0..n).map(|i| (i % 9) as f32 - 4.0).collect(),
            bias: (0..DIM).map(|i| i as f32 - 8.0).chain((0..DIM).map(|i| (i % 4) as f32 * 0.5 + 0.5)).collect(),
        }
    }

    /// [`Probe`] on the CPU's scalar backend.
    pub fn expected(&self) -> Outputs {
        let mut o = Outputs::zeroed();
        hydroplane::run_matrix_scalar(Probe {
            a: &self.a,
            b: &self.b,
            c: &self.c,
            bias: &self.bias,
            out: &mut o.out,
            row_max: &mut o.row_max,
            col_sum: &mut o.col_sum,
        });
        o
    }
}

impl Default for Inputs {
    fn default() -> Self {
        Self::new()
    }
}

/// Why a module could not run.
#[derive(Debug)]
pub enum Skip {
    /// No Vulkan loader or device.
    NoDevice,
    /// No device offers `VK_KHR_cooperative_matrix` with the `16×16×16` `f16 → f32` shape.
    NoCoopMatrix,
}

/// Run `spv`'s `entry::probe` once, one subgroup of 32, on the first suitable device.
pub fn run(spv: &[u8], coop: bool, inputs: &Inputs) -> Result<(String, Outputs), Skip> {
    let gpu = Gpu::new(coop)?;
    let zeros = Outputs::zeroed();
    let buffers = [
        gpu.upload(bytes(&inputs.a)),
        gpu.upload(bytes(&inputs.b)),
        gpu.upload(bytes(&inputs.c)),
        gpu.upload(bytes(&inputs.bias)),
        gpu.upload(bytes(&zeros.out)),
        gpu.upload(bytes(&zeros.row_max)),
        gpu.upload(bytes(&zeros.col_sum)),
    ];
    let kernel = gpu.kernel(spv, c"entry::probe", &buffers.each_ref());
    gpu.dispatch(&kernel, [1, 1, 1]);
    let outputs = Outputs { out: gpu.download(&buffers[4]), row_max: gpu.download(&buffers[5]), col_sum: gpu.download(&buffers[6]) };
    Ok((gpu.name.clone(), outputs))
}

fn has_ext(list: &[vk::ExtensionProperties], name: &std::ffi::CStr) -> bool {
    list.iter().any(|e| e.extension_name_as_c_str() == Ok(name))
}

fn coop_shape(entry: &ash::Entry, instance: &ash::Instance, pdev: vk::PhysicalDevice) -> bool {
    let coop = ash::khr::cooperative_matrix::Instance::new(entry, instance);
    let Ok(shapes) = (unsafe { coop.get_physical_device_cooperative_matrix_properties(pdev) }) else { return false };
    shapes.iter().any(|s| {
        (s.m_size, s.n_size, s.k_size) == (DIM as u32, DIM as u32, DIM as u32)
            && s.a_type == vk::ComponentTypeKHR::FLOAT16
            && s.b_type == vk::ComponentTypeKHR::FLOAT16
            && s.c_type == vk::ComponentTypeKHR::FLOAT32
            && s.result_type == vk::ComponentTypeKHR::FLOAT32
            && s.scope == vk::ScopeKHR::SUBGROUP
    })
}

/// The bytes of a slice of plain numbers.
pub fn bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

/// A device-local storage buffer.
pub struct Buffer {
    buffer: vk::Buffer,
    bytes: usize,
}

/// A compute pipeline with its buffers bound.
pub struct Kernel {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    module: vk::ShaderModule,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

unsafe fn print_stats(ext: &ash::khr::pipeline_executable_properties::Device, pipeline: vk::Pipeline, entry: &std::ffi::CStr) {
    let execs = unsafe { ext.get_pipeline_executable_properties(&vk::PipelineInfoKHR::default().pipeline(pipeline)) }.unwrap_or_default();
    for i in 0..execs.len() as u32 {
        let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i);
        let stats = unsafe { ext.get_pipeline_executable_statistics(&info) }.unwrap_or_default();
        let line: Vec<String> = stats
            .iter()
            .map(|s| {
                let name = s.name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
                let v = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => s.value.b32.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                        _ => s.value.f64.to_string(),
                    }
                };
                format!("{name}={v}")
            })
            .collect();
        println!("    [{}] {}", entry.to_string_lossy(), line.join(", "));
        if let Some(dir) = std::env::var_os("HYDROPLANE_PIPELINE_IR") {
            dump_ir(ext, &info, std::path::Path::new(&dir), entry);
        }
    }
}

/// Writes the driver's internal representations of one pipeline executable (on NVIDIA, its SASS)
/// to `dir/<entry>-<n>-<name>.txt`.
fn dump_ir(ext: &ash::khr::pipeline_executable_properties::Device, info: &vk::PipelineExecutableInfoKHR, dir: &std::path::Path, entry: &std::ffi::CStr) {
    let fp = ext.fp();
    let dev = ext.device();
    let mut count = 0u32;
    unsafe {
        let _ = (fp.get_pipeline_executable_internal_representations_khr)(dev, info, &mut count, std::ptr::null_mut());
        let mut reps = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
        let _ = (fp.get_pipeline_executable_internal_representations_khr)(dev, info, &mut count, reps.as_mut_ptr());
        let mut bufs: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
        for (r, b) in reps.iter_mut().zip(&mut bufs) {
            r.p_data = b.as_mut_ptr().cast();
        }
        let _ = (fp.get_pipeline_executable_internal_representations_khr)(dev, info, &mut count, reps.as_mut_ptr());
        let _ = std::fs::create_dir_all(dir);
        for (n, (r, b)) in reps.iter().zip(&bufs).enumerate() {
            let name = r.name_as_c_str().map(|c| c.to_string_lossy().replace(' ', "_")).unwrap_or_default();
            let file = dir.join(format!("{}-{n}-{name}.txt", entry.to_string_lossy().replace("::", "_")));
            let text: &[u8] = if r.is_text != 0 { b.split(|&c| c == 0).next().unwrap_or(b) } else { b };
            let _ = std::fs::write(file, text);
        }
    }
}

/// One Vulkan device with a compute queue, enabling what the test shaders use.
pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    dev: ash::Device,
    queue: vk::Queue,
    cpool: vk::CommandPool,
    queries: vk::QueryPool,
    mem: vk::PhysicalDeviceMemoryProperties,
    timestamp_ns: f64,
    buffers: std::cell::RefCell<Vec<(vk::Buffer, vk::DeviceMemory)>>,
    stats: Option<ash::khr::pipeline_executable_properties::Device>,
    pub name: String,
}

impl Gpu {
    /// The first device with a compute queue, or with `coop`, the first that also lists the
    /// `16×16×16` `f16 → f32` cooperative-matrix shape.
    pub fn new(coop: bool) -> Result<Self, Skip> {
        unsafe {
            let entry = hydroplane_whatcompute::gpu::load_vulkan().map_err(|_| Skip::NoDevice)?;
            let inst_exts = entry.enumerate_instance_extension_properties(None).expect("listing instance extensions");
            let mut exts = Vec::new();
            let mut flags = vk::InstanceCreateFlags::empty();
            if has_ext(&inst_exts, ash::khr::portability_enumeration::NAME) {
                exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app).flags(flags).enabled_extension_names(&exts), None)
                .expect("creating the Vulkan instance");
            let devices = instance.enumerate_physical_devices().expect("listing devices");
            let suitable = |&p: &vk::PhysicalDevice| {
                let exts = instance.enumerate_device_extension_properties(p).unwrap_or_default();
                !coop || (has_ext(&exts, ash::khr::cooperative_matrix::NAME) && coop_shape(&entry, &instance, p))
            };
            let Some(pdev) = devices.iter().copied().find(suitable) else {
                instance.destroy_instance(None);
                return Err(if devices.is_empty() || !coop { Skip::NoDevice } else { Skip::NoCoopMatrix });
            };
            let props = instance.get_physical_device_properties(pdev);
            let name = props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default();
            let dev_exts = instance.enumerate_device_extension_properties(pdev).unwrap_or_default();

            let mut have11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut have12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut have = vk::PhysicalDeviceFeatures2::default().push_next(&mut have11).push_next(&mut have12);
            instance.get_physical_device_features2(pdev, &mut have);
            let f10 = vk::PhysicalDeviceFeatures::default().shader_int16(have.features.shader_int16 != 0).shader_float64(have.features.shader_float64 != 0);
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(have11.storage_buffer16_bit_access != 0);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_float16(have12.shader_float16 != 0)
                .vulkan_memory_model(have12.vulkan_memory_model != 0);
            let mut fcoop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut fstats = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
            let mut enabled = Vec::new();
            if has_ext(&dev_exts, ash::khr::portability_subset::NAME) {
                enabled.push(ash::khr::portability_subset::NAME.as_ptr());
            }
            let family = instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .expect("a compute queue") as u32;
            let prio = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio)];
            let mut info = vk::DeviceCreateInfo::default().queue_create_infos(&queues).enabled_features(&f10).push_next(&mut f11).push_next(&mut f12);
            if coop {
                enabled.push(ash::khr::cooperative_matrix::NAME.as_ptr());
                info = info.push_next(&mut fcoop);
            }
            let stats = std::env::var_os("HYDROPLANE_PIPELINE_STATS").is_some() && has_ext(&dev_exts, ash::khr::pipeline_executable_properties::NAME);
            if stats {
                enabled.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
                info = info.push_next(&mut fstats);
            }
            let dev = instance.create_device(pdev, &info.enabled_extension_names(&enabled), None).expect("creating the device");
            let stats = stats.then(|| ash::khr::pipeline_executable_properties::Device::new(&instance, &dev));
            let queue = dev.get_device_queue(family, 0);
            let cpool = dev
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .unwrap();
            let queries = dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap();
            Ok(Self {
                _entry: entry,
                mem: instance.get_physical_device_memory_properties(pdev),
                timestamp_ns: props.limits.timestamp_period as f64,
                instance,
                dev,
                queue,
                cpool,
                queries,
                buffers: Default::default(),
                stats,
                name,
            })
        }
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> u32 {
        (0..self.mem.memory_type_count)
            .find(|&t| bits & (1 << t) != 0 && self.mem.memory_types[t as usize].property_flags.contains(want))
            .expect("a suitable memory type")
    }

    fn raw_buffer(&self, bytes: usize, usage: vk::BufferUsageFlags, want: vk::MemoryPropertyFlags) -> (vk::Buffer, vk::DeviceMemory) {
        unsafe {
            let buffer = self.dev.create_buffer(&vk::BufferCreateInfo::default().size(bytes as u64).usage(usage), None).unwrap();
            let req = self.dev.get_buffer_memory_requirements(buffer);
            let info = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(self.memory_type(req.memory_type_bits, want));
            let memory = self.dev.allocate_memory(&info, None).unwrap();
            self.dev.bind_buffer_memory(buffer, memory, 0).unwrap();
            (buffer, memory)
        }
    }

    fn staging(&self, bytes: usize) -> (vk::Buffer, vk::DeviceMemory) {
        self.raw_buffer(
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }

    fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) {
        unsafe {
            let cmd = self.dev.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cpool).command_buffer_count(1)).unwrap()[0];
            self.dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            record(cmd);
            self.dev.end_command_buffer(cmd).unwrap();
            self.dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], vk::Fence::null()).unwrap();
            self.dev.queue_wait_idle(self.queue).unwrap();
            self.dev.free_command_buffers(self.cpool, &[cmd]);
        }
    }

    /// A device-local storage buffer holding `data`.
    pub fn upload(&self, data: &[u8]) -> Buffer {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let (buffer, memory) = self.raw_buffer(data.len(), usage, vk::MemoryPropertyFlags::DEVICE_LOCAL);
        self.buffers.borrow_mut().push((buffer, memory));
        let (stage, stage_mem) = self.staging(data.len());
        unsafe {
            let p = self.dev.map_memory(stage_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast::<u8>();
            std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
            self.dev.unmap_memory(stage_mem);
            let region = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: data.len() as u64 }];
            self.one_shot(|cmd| self.dev.cmd_copy_buffer(cmd, stage, buffer, &region));
            self.dev.destroy_buffer(stage, None);
            self.dev.free_memory(stage_mem, None);
        }
        Buffer { buffer, bytes: data.len() }
    }

    /// `b`'s contents as `f32`s.
    pub fn download(&self, b: &Buffer) -> Vec<f32> {
        let (stage, stage_mem) = self.staging(b.bytes);
        unsafe {
            let region = [vk::BufferCopy { src_offset: 0, dst_offset: 0, size: b.bytes as u64 }];
            self.one_shot(|cmd| self.dev.cmd_copy_buffer(cmd, b.buffer, stage, &region));
            let p = self.dev.map_memory(stage_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap().cast::<f32>();
            let v = std::slice::from_raw_parts(p, b.bytes / 4).to_vec();
            self.dev.unmap_memory(stage_mem);
            self.dev.destroy_buffer(stage, None);
            self.dev.free_memory(stage_mem, None);
            v
        }
    }

    /// `entry` from `spv`, with `buffers` bound to bindings `0..` of set 0.
    pub fn kernel(&self, spv: &[u8], entry: &std::ffi::CStr, buffers: &[&Buffer]) -> Kernel {
        let words: Vec<u32> = spv.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect();
        unsafe {
            let bindings: Vec<_> = (0..buffers.len() as u32)
                .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                .collect();
            let set_layout = self.dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
            let layout = self.dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
            let module = self.dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry)
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
            let flags = if self.stats.is_some() {
                vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR
            } else {
                vk::PipelineCreateFlags::empty()
            };
            let pipeline = self
                .dev
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().flags(flags).stage(stage).layout(layout)], None)
                .unwrap()[0];
            if let Some(stats) = &self.stats {
                print_stats(stats, pipeline, entry);
            }
            let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: buffers.len() as u32 }];
            let pool = self.dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).unwrap();
            let set = self.dev.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
            let infos: Vec<_> = buffers.iter().map(|b| [vk::DescriptorBufferInfo { buffer: b.buffer, offset: 0, range: vk::WHOLE_SIZE }]).collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info))
                .collect();
            self.dev.update_descriptor_sets(&writes, &[]);
            Kernel { pipeline, layout, set_layout, module, pool, set }
        }
    }

    /// Run `k` over `groups` workgroups and return the GPU time it took, in nanoseconds.
    pub fn dispatch(&self, k: &Kernel, groups: [u32; 3]) -> f64 {
        unsafe {
            self.one_shot(|cmd| {
                self.dev.cmd_reset_query_pool(cmd, self.queries, 0, 2);
                self.dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, k.pipeline);
                self.dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, k.layout, 0, &[k.set], &[]);
                self.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.queries, 0);
                self.dev.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
                self.dev.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.queries, 1);
            });
            let mut stamps = [0u64; 2];
            self.dev.get_query_pool_results(self.queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT).unwrap();
            (stamps[1] - stamps[0]) as f64 * self.timestamp_ns
        }
    }

    /// Release `k`'s objects; a kernel not handed back here lives as long as the device.
    pub fn drop_kernel(&self, k: Kernel) {
        unsafe {
            self.dev.destroy_pipeline(k.pipeline, None);
            self.dev.destroy_descriptor_pool(k.pool, None);
            self.dev.destroy_shader_module(k.module, None);
            self.dev.destroy_pipeline_layout(k.layout, None);
            self.dev.destroy_descriptor_set_layout(k.set_layout, None);
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.device_wait_idle();
            for (buffer, memory) in self.buffers.get_mut().drain(..) {
                self.dev.destroy_buffer(buffer, None);
                self.dev.free_memory(memory, None);
            }
            self.dev.destroy_query_pool(self.queries, None);
            self.dev.destroy_command_pool(self.cpool, None);
            self.dev.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
