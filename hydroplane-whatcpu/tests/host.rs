#[test]
fn detect_and_current_agree() {
    let a = hydroplane_whatcpu::detect();
    let b = hydroplane_whatcpu::current();
    let c = hydroplane_whatcpu::current();
    assert_eq!(b, c);
    assert_eq!(a.vendor, b.vendor);
    assert_eq!(a.raw, b.raw);
    eprintln!("{a:?}");
}
