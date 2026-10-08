//! Vulkan compute: one device and queue, device-local storage buffers, and pipelines from the
//! rust-gpu kernel modules the build script compiled.

mod cholesky;

pub use cholesky::{Info, Kernel, KernelTime, Stats, VulkanCholesky};

use std::ffi::CString;

use ash::vk;

use crate::cholesky::{Error, Real};

static SPV_F32: &[u8] = include_bytes!(env!("HYDROPLANE_SOLVERS_SPV_F32"));
static SPV_F64: &[u8] = include_bytes!(env!("HYDROPLANE_SOLVERS_SPV_F64"));

fn device_error(e: vk::Result) -> Error {
    Error::Device(format!("{e:?}"))
}

/// A Vulkan device chosen for compute: the first discrete GPU, else the first device. On macOS
/// that is MoltenVK over Metal, which has no `f64`.
pub struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    dev: ash::Device,
    queue: vk::Queue,
    mem: vk::PhysicalDeviceMemoryProperties,
    cpool: vk::CommandPool,
    period_ns: f64,
    f64: bool,
    atomic_f64: bool,
    shared_bytes: u32,
    name: String,
}

impl Context {
    pub fn new() -> Result<Self, Error> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| Error::Device(e.to_string()))?;
            let available = entry.enumerate_instance_extension_properties(None).map_err(device_error)?;
            let has = |name: &std::ffi::CStr| available.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            let mut exts = Vec::new();
            let mut flags = vk::InstanceCreateFlags::empty();
            if has(ash::khr::portability_enumeration::NAME) {
                exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app).enabled_extension_names(&exts).flags(flags), None)
                .map_err(device_error)?;
            let devices = instance.enumerate_physical_devices().map_err(device_error)?;
            let pdev = devices
                .iter()
                .copied()
                .find(|&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                .or_else(|| devices.first().copied())
                .ok_or_else(|| Error::Device("no Vulkan device".into()))?;
            let props = instance.get_physical_device_properties(pdev);
            let family = instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| Error::Device("no compute queue".into()))? as u32;

            let available_dev = instance.enumerate_device_extension_properties(pdev).map_err(device_error)?;
            let dev_has = |name: &std::ffi::CStr| available_dev.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            let has_atomic_float = dev_has(ash::ext::shader_atomic_float::NAME);
            let mut have12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut have_atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
            let mut have = vk::PhysicalDeviceFeatures2::default().push_next(&mut have12);
            if has_atomic_float {
                have = have.push_next(&mut have_atomic);
            }
            instance.get_physical_device_features2(pdev, &mut have);
            let have10 = have.features;
            let f10 = vk::PhysicalDeviceFeatures::default()
                .shader_float64(have10.shader_float64 != 0)
                .shader_int64(have10.shader_int64 != 0)
                .shader_int16(have10.shader_int16 != 0);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .vulkan_memory_model(have12.vulkan_memory_model != 0)
                .vulkan_memory_model_device_scope(have12.vulkan_memory_model_device_scope != 0)
                .shader_int8(have12.shader_int8 != 0);
            let atomic_f64 = has_atomic_float && have_atomic.shader_buffer_float64_atomic_add != 0;
            let mut f_atomic = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
                .shader_buffer_float32_atomic_add(have_atomic.shader_buffer_float32_atomic_add != 0)
                .shader_buffer_float64_atomic_add(atomic_f64);
            let mut dev_exts = Vec::new();
            if dev_has(ash::khr::portability_subset::NAME) {
                dev_exts.push(ash::khr::portability_subset::NAME.as_ptr());
            }
            if has_atomic_float {
                dev_exts.push(ash::ext::shader_atomic_float::NAME.as_ptr());
            }
            let prio = [1.0];
            let qi = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio)];
            let dev = instance
                .create_device(
                    pdev,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&qi).enabled_features(&f10).enabled_extension_names(&dev_exts).push_next(&mut f12).push_next(&mut f_atomic),
                    None,
                )
                .map_err(device_error)?;
            let queue = dev.get_device_queue(family, 0);
            let mem = instance.get_physical_device_memory_properties(pdev);
            let cpool = dev
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
                .map_err(device_error)?;
            Ok(Self {
                _entry: entry,
                instance,
                dev,
                queue,
                mem,
                cpool,
                period_ns: props.limits.timestamp_period as f64,
                f64: have10.shader_float64 != 0,
                atomic_f64,
                shared_bytes: props.limits.max_compute_shared_memory_size,
                name: props.device_name_as_c_str().map(|c| c.to_string_lossy().into_owned()).unwrap_or_default(),
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the device runs `f64` kernels.
    pub fn supports_f64(&self) -> bool {
        self.f64
    }

    /// Whether storage-buffer `f64` atomic add (`VK_EXT_shader_atomic_float`) is available.
    pub fn supports_atomic_f64(&self) -> bool {
        self.atomic_f64
    }

    /// Workgroup shared memory per workgroup (bytes).
    pub fn shared_bytes(&self) -> u32 {
        self.shared_bytes
    }

    fn module<T: Real>(&self) -> Result<vk::ShaderModule, Error> {
        if T::F64 && !self.f64 {
            return Err(Error::Device(format!("{} has no shaderFloat64", self.name)));
        }
        let spv = if T::F64 { SPV_F64 } else { SPV_F32 };
        let words = ash::util::read_spv(&mut std::io::Cursor::new(spv)).map_err(|e| Error::Device(e.to_string()))?;
        unsafe { self.dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).map_err(device_error) }
    }

    fn alloc(&self, size: u64, usage: vk::BufferUsageFlags, want: vk::MemoryPropertyFlags) -> Result<Buf, Error> {
        unsafe {
            let size = size.max(16);
            let buf = self.dev.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None).map_err(device_error)?;
            let req = self.dev.get_buffer_memory_requirements(buf);
            let Some(ty) = (0..self.mem.memory_type_count)
                .find(|&i| req.memory_type_bits & (1 << i) != 0 && self.mem.memory_types[i as usize].property_flags.contains(want))
            else {
                self.dev.destroy_buffer(buf, None);
                return Err(Error::Device("no suitable memory type".into()));
            };
            let mem = self
                .dev
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)
                .map_err(device_error)?;
            self.dev.bind_buffer_memory(buf, mem, 0).map_err(device_error)?;
            Ok(Buf { buf, mem })
        }
    }

    fn device_buffer(&self, bytes: u64) -> Result<Buf, Error> {
        self.alloc(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
    }

    /// Host-visible staging, cached where the device offers it so reading results back is not
    /// an uncached crawl.
    fn host_buffer(&self, bytes: u64) -> Result<Buf, Error> {
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        self.alloc(bytes, usage, visible | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|_| self.alloc(bytes, usage, visible))
    }

    fn submit_wait(&self, cmd: vk::CommandBuffer) -> Result<(), Error> {
        unsafe {
            let fence = self.dev.create_fence(&vk::FenceCreateInfo::default(), None).map_err(device_error)?;
            let cmds = [cmd];
            let r = self
                .dev
                .queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence)
                .and_then(|_| self.dev.wait_for_fences(&[fence], true, u64::MAX));
            self.dev.destroy_fence(fence, None);
            r.map_err(device_error)
        }
    }

    fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<(), Error> {
        unsafe {
            let cmd = self
                .dev
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cpool).command_buffer_count(1))
                .map_err(device_error)?[0];
            self.dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(device_error)?;
            record(cmd);
            self.dev.end_command_buffer(cmd).map_err(device_error)?;
            let r = self.submit_wait(cmd);
            self.dev.free_command_buffers(self.cpool, &[cmd]);
            r
        }
    }

    /// A device-local storage buffer holding `data`.
    fn upload<T: Copy>(&self, data: &[T]) -> Result<Buf, Error> {
        let bytes = size_of_val(data) as u64;
        let dst = self.device_buffer(bytes)?;
        if bytes == 0 {
            return Ok(dst);
        }
        let stage = self.host_buffer(bytes)?;
        unsafe {
            let p = self.dev.map_memory(stage.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(device_error)? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, p, bytes as usize);
            self.dev.unmap_memory(stage.mem);
        }
        self.one_shot(|cmd| unsafe { self.dev.cmd_copy_buffer(cmd, stage.buf, dst.buf, &[vk::BufferCopy::default().size(bytes)]) })?;
        self.free(stage);
        Ok(dst)
    }

    fn download<T: Copy + Default>(&self, b: Buf, len: usize) -> Result<Vec<T>, Error> {
        let bytes = (len * size_of::<T>()) as u64;
        let mut out = vec![T::default(); len];
        if bytes == 0 {
            return Ok(out);
        }
        let stage = self.host_buffer(bytes)?;
        self.one_shot(|cmd| unsafe { self.dev.cmd_copy_buffer(cmd, b.buf, stage.buf, &[vk::BufferCopy::default().size(bytes)]) })?;
        unsafe {
            let p = self.dev.map_memory(stage.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(device_error)? as *const u8;
            std::ptr::copy_nonoverlapping(p, out.as_mut_ptr() as *mut u8, bytes as usize);
            self.dev.unmap_memory(stage.mem);
        }
        self.free(stage);
        Ok(out)
    }

    fn free(&self, b: Buf) {
        unsafe {
            self.dev.destroy_buffer(b.buf, None);
            self.dev.free_memory(b.mem, None);
        }
    }

    /// A compute pipeline for `entry` with `bufs` bound to bindings `0..` of set 0 and a 16-byte
    /// push-constant block.
    fn pipeline(&self, module: vk::ShaderModule, entry: &str, bufs: &[Buf]) -> Result<Pipe, Error> {
        unsafe {
            let bindings: Vec<_> = (0..bufs.len() as u32)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dsl = self.dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(device_error)?;
            let pcr = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(16)];
            let dsls = [dsl];
            let layout = self
                .dev
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None)
                .map_err(device_error)?;
            let name = CString::new(entry).unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&name);
            let pipe = self
                .dev
                .create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None)
                .map_err(|(_, e)| device_error(e))?[0];
            let sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(bufs.len() as u32)];
            let pool = self.dev.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None).map_err(device_error)?;
            let set = self
                .dev
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&dsls))
                .map_err(device_error)?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter().map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buf).range(vk::WHOLE_SIZE)]).collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(info))
                .collect();
            self.dev.update_descriptor_sets(&writes, &[]);
            Ok(Pipe { pipe, layout, dsl, pool, set })
        }
    }

    fn destroy_pipeline(&self, p: &Pipe) {
        unsafe {
            self.dev.destroy_pipeline(p.pipe, None);
            self.dev.destroy_pipeline_layout(p.layout, None);
            self.dev.destroy_descriptor_pool(p.pool, None);
            self.dev.destroy_descriptor_set_layout(p.dsl, None);
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.dev.device_wait_idle();
            self.dev.destroy_command_pool(self.cpool, None);
            self.dev.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
struct Buf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
}

struct Pipe {
    pipe: vk::Pipeline,
    layout: vk::PipelineLayout,
    dsl: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}
