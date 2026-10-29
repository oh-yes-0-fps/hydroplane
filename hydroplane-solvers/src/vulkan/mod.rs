//! Vulkan compute: one device and queue, device-local storage buffers, and pipelines from the
//! rust-gpu kernel modules the build script compiled.

mod amg;
mod bicgstab;
mod cholesky;
mod ldlt;
mod pcg;

pub use bicgstab::VulkanBiCgStab;
pub use cholesky::{Info, Kernel, KernelTime, Options, SchurTile, Stats, VulkanCholesky};
pub use ldlt::VulkanLdlt;
pub use pcg::VulkanPcg;

use std::sync::Arc;

use crate::cholesky::Analysis;

/// Estimated device time (ms) of a [`VulkanCholesky`] factorization with `analysis`'s ordering,
/// as a cost for [`Analysis::best_of`]: 0.073 ms per panel step on the critical path plus 0.48 ms
/// per GFLOP, fitted on an RTX 5090 in `f64` over four orderings of a FEM tangent. On smaller
/// GPUs flops weigh more; the ranking is what matters.
pub fn ordering_cost(analysis: &Analysis) -> f64 {
    0.073 * analysis.levels().1 as f64 + 0.48 * analysis.flops() * 1e-9
}

use std::ffi::CString;

use ash::vk;

use crate::Ef64;
use crate::cholesky::Error;

pub use crate::Precision;

/// A working precision of the GPU solvers: a value type laid out as the kernels', converted to
/// and from `f64` at the host boundary.
pub trait DeviceReal: Copy + Default + Send + Sync + 'static {
    const PRECISION: Precision;
    /// Unit roundoff.
    const EPSILON: f64;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl DeviceReal for f32 {
    const PRECISION: Precision = Precision::F32;
    const EPSILON: f64 = Precision::F32.epsilon();
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl DeviceReal for f64 {
    const PRECISION: Precision = Precision::F64;
    const EPSILON: f64 = Precision::F64.epsilon();
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

impl DeviceReal for Ef64 {
    const PRECISION: Precision = Precision::Ef64;
    const EPSILON: f64 = Precision::Ef64.epsilon();
    fn from_f64(v: f64) -> Self {
        Ef64::from_f64(v)
    }
    fn to_f64(self) -> f64 {
        Ef64::to_f64(self)
    }
}

/// How long [`Context::submit_wait`] spins before it sleeps.
const SPIN: std::time::Duration = std::time::Duration::from_millis(5);

static SPV_F32: &[u8] = include_bytes!(env!("HYDROPLANE_SOLVERS_SPV_F32"));
static SPV_F64: &[u8] = include_bytes!(env!("HYDROPLANE_SOLVERS_SPV_F64"));
static SPV_EF64: &[u8] = include_bytes!(env!("HYDROPLANE_SOLVERS_SPV_EF64"));
static SPV_EF64_PRECISE: &[u8] = include_bytes!(env!("HYDROPLANE_SOLVERS_SPV_EF64_PRECISE"));

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
    /// The fence each submission signals, behind the lock that also guards the queue and the
    /// command pool (allocating, recording and freeing command buffers): Vulkan leaves all three
    /// to the caller to synchronize, and every solver on every thread shares one context.
    sync: std::sync::Mutex<vk::Fence>,
    staging: std::sync::Mutex<Option<Staging>>,
    period_ns: f64,
    f64: bool,
    atomic_f64: bool,
    unified: bool,
    discrete: bool,
    moltenvk: bool,
    shared_bytes: u32,
    name: String,
}

impl core::fmt::Debug for Context {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Context").field("name", &self.name).finish_non_exhaustive()
    }
}

impl Context {
    /// One context for the process, made on first use (the first Vulkan device with compute),
    /// shared by every solver that is not given its own.
    pub fn shared() -> Result<Arc<Context>, Error> {
        static SHARED: std::sync::OnceLock<Result<Arc<Context>, String>> = std::sync::OnceLock::new();
        SHARED
            .get_or_init(|| Context::new().map(Arc::new).map_err(|e| format!("{e:?}")))
            .clone()
            .map_err(Error::Device)
    }

    pub fn new() -> Result<Self, Error> {
        unsafe {
            let entry = hydroplane_whatcompute::gpu::load_vulkan().map_err(|e| Error::Device(e.to_string()))?;
            let available = entry.enumerate_instance_extension_properties(None).map_err(device_error)?;
            let has = |name: &std::ffi::CStr| available.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
            let mut exts = Vec::new();
            let mut flags = vk::InstanceCreateFlags::empty();
            if has(ash::khr::portability_enumeration::NAME) {
                exts.push(ash::khr::portability_enumeration::NAME.as_ptr());
                flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
            }
            // MoltenVK compiles with Metal's fast math by default, which reassociates the `Ef64`
            // kernels' error-free sums away; it costs the other kernels nothing.
            let never = 0i32;
            let mut fast_math = vk::LayerSettingEXT::default().layer_name(c"MoltenVK").setting_name(c"MVK_CONFIG_FAST_MATH_ENABLED").ty(vk::LayerSettingTypeEXT::INT32);
            fast_math.value_count = 1;
            fast_math.p_values = (&raw const never).cast();
            let fast_math = [fast_math];
            let mut settings = vk::LayerSettingsCreateInfoEXT::default().settings(&fast_math);
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let mut info = vk::InstanceCreateInfo::default().application_info(&app).flags(flags);
            if has(ash::ext::layer_settings::NAME) {
                exts.push(ash::ext::layer_settings::NAME.as_ptr());
                info = info.push_next(&mut settings);
            }
            let instance = entry.create_instance(&info.enabled_extension_names(&exts), None).map_err(device_error)?;
            let devices = instance.enumerate_physical_devices().map_err(device_error)?;
            let pdev = devices
                .iter()
                .copied()
                .find(|&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                .or_else(|| devices.first().copied())
                .ok_or_else(|| Error::Device("no Vulkan device".into()))?;
            let props = instance.get_physical_device_properties(pdev);
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            instance.get_physical_device_properties2(pdev, &mut vk::PhysicalDeviceProperties2::default().push_next(&mut driver));
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
            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).map_err(device_error)?;
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
                sync: std::sync::Mutex::new(fence),
                staging: std::sync::Mutex::new(None),
                period_ns: props.limits.timestamp_period as f64,
                f64: have10.shader_float64 != 0,
                atomic_f64,
                unified: props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
                    && (0..mem.memory_type_count).any(|i| {
                        mem.memory_types[i as usize].property_flags.contains(
                            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
                    }),
                discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
                moltenvk: driver.driver_id == vk::DriverId::MOLTENVK,
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

    /// Whether the device runs kernels in `precision`.
    pub fn supports(&self, precision: Precision) -> bool {
        precision != Precision::F64 || self.f64
    }

    /// The fastest precision with `f64`'s range of accuracy here: native where the device has
    /// it, emulated where it does not.
    pub fn wide_precision(&self) -> Precision {
        if self.f64 { Precision::F64 } else { Precision::Ef64 }
    }

    /// Whether storage-buffer `f64` atomic add (`VK_EXT_shader_atomic_float`) is available.
    pub fn supports_atomic_f64(&self) -> bool {
        self.atomic_f64
    }

    /// Whether the GPU shares the host's memory (an integrated GPU with device-local memory the
    /// host can map, as Apple silicon's): inputs and results are then written and read in place,
    /// with no staging buffers or copies.
    pub fn unified_memory(&self) -> bool {
        self.unified
    }

    /// Whether the device is a discrete GPU, with its own memory across a bus.
    pub fn discrete(&self) -> bool {
        self.discrete
    }

    /// Workgroup shared memory per workgroup (bytes).
    pub fn shared_bytes(&self) -> u32 {
        self.shared_bytes
    }

    fn module<T: DeviceReal>(&self) -> Result<vk::ShaderModule, Error> {
        if !self.supports(T::PRECISION) {
            return Err(Error::Device(format!("{} has no shaderFloat64", self.name)));
        }
        let spv = match T::PRECISION {
            Precision::F32 => SPV_F32,
            Precision::F64 => SPV_F64,
            Precision::Ef64 if self.moltenvk => SPV_EF64,
            Precision::Ef64 => SPV_EF64_PRECISE,
        };
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

    /// Device-local memory the host maps too (on a [`Self::unified_memory`] device).
    fn shared_buffer(&self, bytes: u64) -> Result<Buf, Error> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let shared = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        self.alloc(bytes, usage, shared | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|_| self.alloc(bytes, usage, shared))
    }

    /// Submits `cmd` and waits for it: spinning on the fence for up to [`SPIN`] (a sleeping
    /// wait wakes late, and a split solve submits every few milliseconds: 0.15 ms of each 2.4 ms
    /// preconditioner call on an M5 Pro), then blocking.
    fn submit_wait(&self, cmd: vk::CommandBuffer) -> Result<(), Error> {
        let fence = self.lock();
        self.submit_locked(*fence, cmd)
    }

    /// The queue, command pool and fence, for as long as the guard lives.
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, vk::Fence> {
        self.sync.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn submit_locked(&self, fence: vk::Fence, cmd: vk::CommandBuffer) -> Result<(), Error> {
        unsafe {
            self.dev.reset_fences(&[fence]).map_err(device_error)?;
            let cmds = [cmd];
            self.dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence).map_err(device_error)?;
            let started = std::time::Instant::now();
            while !self.dev.get_fence_status(fence).map_err(device_error)? {
                if started.elapsed() > SPIN {
                    return self.dev.wait_for_fences(&[fence], true, u64::MAX).map_err(device_error);
                }
                std::hint::spin_loop();
            }
            Ok(())
        }
    }

    fn one_shot(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<(), Error> {
        let fence = self.lock();
        unsafe {
            let cmd = self
                .dev
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(self.cpool).command_buffer_count(1))
                .map_err(device_error)?[0];
            self.dev.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(device_error)?;
            record(cmd);
            self.dev.end_command_buffer(cmd).map_err(device_error)?;
            let r = self.submit_locked(*fence, cmd);
            self.dev.free_command_buffers(self.cpool, &[cmd]);
            r
        }
    }

    /// A device-local storage buffer holding `data`.
    fn upload<T: Copy>(&self, data: &[T]) -> Result<Buf, Error> {
        let mut batch = self.batch()?;
        let buf = batch.upload(data)?;
        batch.send()?;
        Ok(buf)
    }

    /// Uploads to send together through the staging buffer.
    fn batch(&self) -> Result<Batch<'_>, Error> {
        let mut staging = self.staging.lock().unwrap_or_else(|e| e.into_inner());
        if staging.is_none() {
            let buf = self.host_buffer(STAGING)?;
            match unsafe { self.dev.map_memory(buf.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
                Ok(ptr) => *staging = Some(Staging { buf, ptr: ptr as *mut u8 }),
                Err(e) => {
                    self.free(buf);
                    return Err(device_error(e));
                }
            }
        }
        Ok(Batch { ctx: self, staging, used: 0, copies: Vec::new() })
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
            if let Some(stage) = self.staging.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
                self.dev.unmap_memory(stage.buf.mem);
                self.free(stage.buf);
            }
            self.dev.destroy_fence(*self.sync.get_mut().unwrap_or_else(|e| e.into_inner()), None);
            self.dev.destroy_command_pool(self.cpool, None);
            self.dev.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Bytes of the staging buffer uploads stream through.
const STAGING: u64 = 16 << 20;

/// The context's host-visible buffer for uploads, mapped for its lifetime: allocating, first
/// touching and freeing a fresh one per upload cost more than the copies themselves (20 ms of
/// 22 for 45 MB on an RTX 5090).
struct Staging {
    buf: Buf,
    ptr: *mut u8,
}

// Only reached through the context's lock on it.
unsafe impl Send for Staging {}

/// Uploads streamed through the context's staging buffer, a submission each time it fills and
/// one for the rest when sent. Holds the staging buffer until then.
struct Batch<'a> {
    ctx: &'a Context,
    staging: std::sync::MutexGuard<'a, Option<Staging>>,
    used: u64,
    copies: Vec<(vk::Buffer, vk::BufferCopy)>,
}

impl Batch<'_> {
    /// A device-local storage buffer that holds `data` once the batch is sent.
    fn upload<T: Copy>(&mut self, data: &[T]) -> Result<Buf, Error> {
        let len = size_of_val(data) as u64;
        let dst = self.ctx.device_buffer(len)?;
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len as usize) };
        let mut done = 0;
        while done < len {
            if self.used >= STAGING {
                self.flush()?;
            }
            let take = (len - done).min(STAGING - self.used);
            let stage = self.staging.as_ref().expect("a staging buffer");
            unsafe { std::ptr::copy_nonoverlapping(bytes[done as usize..].as_ptr(), stage.ptr.add(self.used as usize), take as usize) };
            self.copies.push((dst.buf, vk::BufferCopy::default().src_offset(self.used).dst_offset(done).size(take)));
            self.used = (self.used + take).next_multiple_of(16);
            done += take;
        }
        Ok(dst)
    }

    fn flush(&mut self) -> Result<(), Error> {
        if !self.copies.is_empty() {
            let src = self.staging.as_ref().expect("a staging buffer").buf.buf;
            self.ctx.one_shot(|cmd| {
                for (dst, region) in &self.copies {
                    unsafe { self.ctx.dev.cmd_copy_buffer(cmd, src, *dst, &[*region]) };
                }
            })?;
        }
        self.copies.clear();
        self.used = 0;
        Ok(())
    }

    fn send(mut self) -> Result<(), Error> {
        self.flush()
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
