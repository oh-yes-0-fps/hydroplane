use hydroplane_gpu_test::{COOP, Inputs, PLAIN, Skip, run};

fn check(spv: &[u8], coop: bool) {
    let inputs = Inputs::new();
    let want = inputs.expected();
    match run(spv, coop, &inputs) {
        Ok((device, got)) => {
            let bad = got.mismatches(&want, 1e-3);
            assert!(bad.is_empty(), "{device}: {} mismatches, first {:?}\n got {got:?}\nwant {want:?}", bad.len(), &bad[..bad.len().min(8)]);
        }
        Err(Skip::NoDevice) => eprintln!("skipped: no Vulkan device"),
        Err(Skip::NoCoopMatrix) => eprintln!("skipped: no device with the 16x16x16 f16 cooperative-matrix shape"),
    }
}

#[test]
fn array_tiles_match_cpu() {
    check(PLAIN, false);
}

#[test]
fn coop_tiles_match_cpu() {
    check(COOP, true);
}
