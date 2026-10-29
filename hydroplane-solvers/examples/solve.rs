//! [`Solver`] on a dumped SPD matrix (the format of the `cholesky` example): what it detects,
//! measures and chooses, then a few factorize-and-solve cycles with the true residual of each.
//!
//! `cargo run --release --features vulkan --example solve -- MATRIX.bin`
//!
//! `TOL`, `SOLVES` (per factorization), `FACTORIZATIONS` and `CYCLES` set the needs; `DEVICE`
//! (`cpu`/`gpu`), `METHOD` (`direct`/`pcg`/`split`), `PRECISION` (`f32`/`f64`/`ef64`) and
//! `LIBRARY` (`faer`/`accelerate`/`vulkan`/`sme`) pin choices.

use std::time::Instant;

use hydroplane_solvers::auto::{Caps, Device, Library, Method, Options, Rates};
use hydroplane_solvers::{Precision, Solver};

fn load(path: &str) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let bytes = std::fs::read(path).expect("reading the matrix");
    let mut at = 0;
    let mut u64s = |count: usize| -> Vec<u64> {
        let v = bytes[at..at + 8 * count].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        at += 8 * count;
        v
    };
    let head = u64s(2);
    let (n, nnz) = (head[0] as usize, head[1] as usize);
    let col_ptr = u64s(n + 1).into_iter().map(|v| v as usize).collect();
    let row_idx = bytes[at..at + 4 * nnz].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize).collect();
    at += 4 * nnz;
    let val = bytes[at..at + 8 * nnz].chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    (col_ptr, row_idx, val)
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: solve MATRIX.bin");
    let (col_ptr, row_idx, val) = load(&path);
    let n = col_ptr.len() - 1;
    let mut options = Options {
        tolerance: env("TOL").unwrap_or(1e-8),
        solves_per_factorization: env("SOLVES").unwrap_or(1.0),
        factorizations: env("FACTORIZATIONS").unwrap_or(1),
        ..Default::default()
    };
    options.device = std::env::var("DEVICE").ok().map(|d| if d == "gpu" { Device::Gpu } else { Device::Cpu });
    options.method = std::env::var("METHOD").ok().map(|m| match m.as_str() {
        "direct" => Method::Direct,
        "pcg" => Method::Pcg,
        _ => Method::SplitPcg,
    });
    options.library = std::env::var("LIBRARY").ok().map(|l| match l.as_str() {
        "faer" => Library::Faer,
        "accelerate" => Library::Accelerate,
        "vulkan" => Library::Vulkan,
        _ => Library::Sme,
    });
    options.precision = std::env::var("PRECISION").ok().map(|p| match p.as_str() {
        "f32" => Precision::F32,
        "ef64" => Precision::Ef64,
        _ => Precision::F64,
    });

    let t = Instant::now();
    let caps = Caps::host();
    println!("caps ({:.0} ms): {caps:?}", ms(t));
    if options.device != Some(Device::Cpu) && caps.gpu.is_some() {
        let t = Instant::now();
        let rates = Rates::host();
        println!("rates ({:.0} ms): {rates:#?}", ms(t));
    }
    let t = Instant::now();
    let mut solver = Solver::with_options(&col_ptr, &row_idx, options).expect("a valid pattern");
    println!("setup {:.0} ms; plan:\n{}", ms(t), solver.plan());

    let b: Vec<f64> = (0..n).map(|i| ((i * 7919) % 101) as f64 - 50.0).collect();
    let solves = env::<f64>("SOLVES").unwrap_or(1.0).max(1.0) as usize;
    for cycle in 0..env("CYCLES").unwrap_or(3) {
        let start = Instant::now();
        solver.factorize(&val).expect("factorization");
        let factor = ms(start);
        for _ in 0..solves {
            let mut x = b.clone();
            let t = Instant::now();
            let report = solver.solve_in_place(&mut x).expect("solve");
            println!("cycle {cycle}: factorize {factor:.1} ms, solve {:.1} ms: {report:?}", ms(t));
        }
        println!("cycle {cycle} total {:.1} ms", ms(start));
    }
    println!("final plan:\n{}", solver.plan());
}
