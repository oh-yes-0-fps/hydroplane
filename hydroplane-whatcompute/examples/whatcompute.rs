use hydroplane_whatcompute::{cpu, gpu};

fn main() {
    let c = cpu::current();
    println!("CPU");
    println!("  vendor: {:?}", c.vendor);
    println!("  arch:   {:?}", c.arch);
    println!("  raw:    {:?}", c.raw);
    if let Some(p) = c.arch.pipeline() {
        println!("  pipe:   {p:?}");
    }

    let gpus = match gpu::detect() {
        Ok(g) => g,
        Err(e) => {
            println!("\nGPU: {e}");
            return;
        }
    };
    if gpus.is_empty() {
        println!("\nGPU: no Vulkan devices");
    }
    for (i, g) in gpus.iter().enumerate() {
        report(i, g);
    }
}

fn report(i: usize, g: &gpu::Gpu) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let opt = |v: Option<u32>| v.map_or("?".to_string(), |v| v.to_string());
    let yn = |v: Option<bool>| match v {
        Some(true) => "yes",
        Some(false) => "no",
        None => "?",
    };

    println!("\nGPU{i}: {}", g.name);
    println!("  vendor:   {:?} ({:#06x}), device {:#x}", g.vendor, g.vendor_id, g.device_id);
    println!("  arch:     {:?}, {:?}", g.arch, g.kind);
    println!("  vulkan:   {}, driver {} {} ({})", g.api_version, g.driver.name, g.driver.info, g.driver.version);
    if let Some(pci) = g.pci {
        println!("  pci:      {pci}");
    }

    let c = &g.cores;
    println!(
        "  units:    {} {}, {} FP32 lanes, {} waves/unit",
        opt(c.units),
        c.unit.unwrap_or("units"),
        opt(c.fp32_lanes),
        opt(c.waves_per_unit)
    );
    println!("  matrix:   hardware {}, {} units", yn(g.matrix.hardware), opt(c.matrix_units));
    println!(
        "            cooperative_matrix {}, matrix2 {}, vector {}",
        g.matrix.cooperative_matrix, g.matrix.cooperative_matrix2, g.matrix.cooperative_vector
    );
    for s in &g.matrix.shapes {
        println!(
            "            {}x{}x{} {:?}*{:?}+{:?}->{:?} {:?}{}",
            s.m,
            s.n,
            s.k,
            s.a,
            s.b,
            s.c,
            s.result,
            s.scope,
            if s.saturating { " sat" } else { "" }
        );
    }
    let rt = &g.ray_tracing;
    println!(
        "  rt:       hardware {}, {} units; accel {}, pipeline {}, query {}, reorder {}",
        yn(rt.hardware),
        opt(c.rt_units),
        rt.acceleration_structure,
        rt.pipeline,
        rt.query,
        rt.invocation_reorder
    );

    let m = &g.memory;
    println!(
        "  memory:   {:?}, {:.1} GiB device-local, {:.1} GiB mappable, max alloc {:.1} GiB",
        m.host_access,
        gib(m.device_local_bytes),
        gib(m.mappable_device_local_bytes),
        gib(m.max_allocation_bytes)
    );
    for (h, heap) in m.heaps.iter().enumerate() {
        println!(
            "    heap{h}: {:.1} GiB{}{}{}{}",
            gib(heap.bytes),
            if heap.device_local { " device-local" } else { "" },
            if heap.host_visible { " host-visible" } else { "" },
            if heap.host_cached { " host-cached" } else { "" },
            heap.budget.map_or(String::new(), |b| format!(", budget {:.1} GiB", gib(b)))
        );
    }

    let s = &g.subgroup;
    println!(
        "  subgroup: {} (min {}, max {}, control {}), arith {}, shuffle {}, clustered {}, rotate {}",
        s.size, s.min_size, s.max_size, s.size_control, s.arithmetic, s.shuffle, s.clustered, s.rotate
    );
    println!("  shader:   {:?}", g.shader);
    println!("  features: {:?}", g.features);
    println!("  limits:   {:?}", g.limits);
    for (q, f) in g.queues.iter().enumerate() {
        println!("  queue{q}:   {f:?}");
    }
    println!("  extensions: {}", g.extensions.len());
}
