# hydroplane-whatcompute

Identify the compute on this machine: the CPU core the current thread runs on, and every GPU the
Vulkan loader can see.

`cargo run --example whatcompute` prints the full report.

## CPU

`cpu` is `no_std`, has no dependencies and does not allocate. With `default-features = false` it
is the whole crate.

```rust
use hydroplane_whatcompute::cpu;

let c = cpu::current();
match c.arch {
    cpu::Microarch::Zen5 => { /* 2x512-bit loads/cycle */ }
    cpu::Microarch::AppleDonan | cpu::Microarch::AppleBrava => {}
    _ => {}
}
```

| Target | Probe | Cost |
|---|---|---|
| x86 / x86_64 | `cpuid` leaves 0, 1, and (Intel hybrid only) 7 and 0x1A | 2 to 4 `cpuid` |
| aarch64 Linux / Android | `mrs MIDR_EL1` (kernel-emulated trap) | one trap |
| aarch64 Apple | `sysctlbyname("hw.cpufamily")` | one syscall |
| anything else | none | `Microarch::Unknown` |

`Microarch::pipeline()` returns the core's sustained limits (loads and stores per cycle, L1D
bytes per cycle, vector and datapath width, FMA bits per cycle, L1D size, line size), or `None`
for cores without trustworthy public figures. Same core, different datapath gets its own variant
(`Zen5` vs `Zen5Mobile`, `GoldenCove` vs `GoldenCoveServer`).

`cpu::detect()` probes every call. `cpu::current()` probes once and caches; on hybrid parts
(Intel P+E, Arm big.LITTLE) the cached answer is whichever core the first call landed on.

## GPU (feature `vulkan`, default)

`gpu` loads the system Vulkan loader at runtime (`ash`, no link-time dependency; on macOS it also
looks in the Homebrew and `/usr/local` locations for the loader or MoltenVK), creates one
instance, and queries properties and features of every physical device. No logical device is
created.

```rust
use hydroplane_whatcompute::gpu;

if let Some(g) = gpu::primary() {
    let tensor = g.matrix.shapes_for(gpu::Elem::F16).next().is_some();
    let zero_copy = g.memory.unified() || g.memory.resizable_bar();
    let sms = g.cores.units;
}
```

| What | Where it comes from |
|---|---|
| vendor, generation (`GpuArch`) | PCI vendor / device id ranges; Apple from the device name |
| shader units (SM / CU / core) | `VK_NV_shader_sm_builtins`, `VK_AMD_shader_core_properties{,2}`, `VK_ARM_shader_core_builtins`, IORegistry `gpu-core-count` on Apple |
| FP32 lanes, tensor / matrix units, RT units | unit count × `GpuArch::unit_layout()` |
| matrix shapes and types | `VK_KHR_cooperative_matrix`; `VK_NV_cooperative_matrix2`, `VK_NV_cooperative_vector` |
| ray tracing | `VK_KHR_acceleration_structure`, `_ray_tracing_pipeline`, `_ray_query`, invocation reorder, micromaps |
| unified memory, resizable BAR | heap and memory-type flags: `HostAccess::{Unified, ResizableBar, Window(bytes), None}` |
| memory budget | `VK_EXT_memory_budget` |
| shader types and atomics | f16 / f64 / bf16 / fp8, int8/16/64, integer dot product, float atomics, shader clock |
| subgroups, workgroup limits, queues, PCI address, driver | core 1.1 to 1.3 properties |

Per-unit figures (`UnitLayout`) are vendors' published numbers; where none exist (Intel unit
counts, Apple RT units) the field is `None` rather than a guess.

`gpu::detect()` enumerates every call (tens of milliseconds). `gpu::current()` enumerates once
per process. `gpu::primary()` picks the discrete GPU with the most VRAM, else the first hardware
device.
