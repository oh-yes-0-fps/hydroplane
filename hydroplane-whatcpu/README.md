# hydroplane-whatcpu

Identify the CPU core the current thread is running on: vendor, microarchitecture, and the raw
hardware id. `no_std`, zero dependencies, no allocation.

```rust
let cpu = hydroplane_whatcpu::current();
match cpu.arch {
    hydroplane_whatcpu::Microarch::Zen5 => { /* 2x512-bit loads/cycle */ }
    hydroplane_whatcpu::Microarch::AppleDonan | hydroplane_whatcpu::Microarch::AppleBrava => {}
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

```rust
if let Some(p) = cpu.arch.pipeline() {
    let f32_fma_lanes = p.fma_lanes_per_cycle(32);
    let tiles = p.tile_loads_per_cycle(64);
}
```

`detect()` probes every call. `current()` probes once and caches; on hybrid parts (Intel P+E,
Arm big.LITTLE) the cached answer is whichever core the first call landed on.
