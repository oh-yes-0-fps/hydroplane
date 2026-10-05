fn main() {
    let cpu = hydroplane_whatcpu::current();
    println!("vendor: {:?}", cpu.vendor);
    println!("arch:   {:?}", cpu.arch);
    println!("raw:    {:?}", cpu.raw);
    if let Some(p) = cpu.arch.pipeline() {
        println!("pipe:   {p:#?}");
    }
}
