use hydroplane_whatcompute::cpu;

#[test]
fn detect_and_current_agree() {
    let a = cpu::detect();
    let b = cpu::current();
    let c = cpu::current();
    assert_eq!(b, c);
    assert_eq!(a.vendor, b.vendor);
    assert_eq!(a.raw, b.raw);
    eprintln!("{a:?}");
}

#[cfg(feature = "vulkan")]
#[test]
fn gpus_are_self_consistent() {
    use hydroplane_whatcompute::gpu;

    let Ok(gpus) = gpu::detect() else { return };
    let ids = |g: &[gpu::Gpu]| g.iter().map(|g| (g.uuid, g.name.clone())).collect::<Vec<_>>();
    assert_eq!(ids(&gpus), ids(gpu::current()));
    for g in &gpus {
        assert!(!g.name.is_empty());
        assert!(g.subgroup.min_size <= g.subgroup.size && g.subgroup.size <= g.subgroup.max_size, "{g:?}");
        assert!(g.matrix.shapes.is_empty() || g.matrix.cooperative_matrix);
        assert!(g.memory.heaps.iter().all(|h| h.budget.is_none_or(|b| b <= h.bytes)), "{:?}", g.memory);
        if let (Some(units), Some(lanes)) = (g.cores.units, g.cores.fp32_lanes) {
            assert!(lanes >= units);
        }
        eprintln!("{} {:?} {:?} {:?}", g.name, g.arch, g.cores, g.memory.host_access);
    }
}
