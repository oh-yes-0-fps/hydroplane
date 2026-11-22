//! The tile checks as a plain binary, for running on a remote GPU box through `cargo bench`.

use hydroplane_gpu_test::{COOP, Inputs, PLAIN, run};

fn main() {
    let inputs = Inputs::new();
    let want = inputs.expected();
    let mut failed = false;
    for (name, spv, coop) in [("array tiles", PLAIN, false), ("coop tiles", COOP, true)] {
        match run(spv, coop, &inputs) {
            Ok((device, got)) => {
                let bad = got.mismatches(&want, 1e-3);
                println!("{name} on {device}: {} mismatches", bad.len());
                if !bad.is_empty() {
                    println!("  first {:?}\n  got  {got:?}\n  want {want:?}", &bad[..bad.len().min(8)]);
                    failed = true;
                }
            }
            Err(skip) => println!("{name}: skipped ({skip:?})"),
        }
    }
    if failed {
        std::process::exit(1);
    }
}
