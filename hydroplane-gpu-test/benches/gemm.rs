//! `GEMM³` `f16 → f32` GEMM on the GPU through hydroplane's two tile lowerings, plus a
//! hand-written per-element shader as a reference. GPU time from timestamp queries.

use half::f16;
use hydroplane_gpu_test::{COOP, Gpu, PLAIN, Skip, bytes};
use hydroplane_gpu_test_shader::{BLOCK, DIM, GEMM};

const WARMUP: usize = 2;

/// Label, module, entry point, workgroup grid, and whether it needs cooperative matrices.
type Variant = (&'static str, &'static [u8], &'static std::ffi::CStr, [u32; 3], bool);
const REPS: usize = 7;
/// A variant slower than this is timed once, so the GPU is not held for long.
const SLOW_NS: f64 = 0.5e9;

fn main() {
    let (gpu, coop) = match Gpu::new(true) {
        Ok(gpu) => (gpu, true),
        Err(Skip::NoCoopMatrix) => (Gpu::new(false).expect("a Vulkan device"), false),
        Err(Skip::NoDevice) => {
            println!("skipped: no Vulkan device");
            return;
        }
    };
    let n = GEMM;
    let a: Vec<f16> = (0..n * n).map(|i| f16::from_f32(((i * 7919 % 17) as f32 - 8.0) * 0.125)).collect();
    let b: Vec<f16> = (0..n * n).map(|i| f16::from_f32(((i * 104729 % 13) as f32 - 6.0) * 0.125)).collect();
    let samples: Vec<(usize, usize)> = (0..64).map(|s| ((s * 2654435761usize) % n, (s * 40503 + 17) % n)).collect();
    let want: Vec<f64> = samples
        .iter()
        .map(|&(r, c)| (0..n).map(|k| a[r * n + k].to_f64() * b[k * n + c].to_f64()).sum())
        .collect();

    let da = gpu.upload(bytes(&a));
    let db = gpu.upload(bytes(&b));
    let dc = gpu.upload(bytes(&vec![0f32; n * n]));
    let tiles = (n / DIM) as u32;
    let blocks = (n / BLOCK) as u32;
    let flops = 2.0 * (n as f64).powi(3);
    println!("{} — {n}×{n}×{n} f16 → f32 GEMM, median of {REPS} after {WARMUP} warm-up", gpu.name);
    let variants: [Variant; 13] = [
        ("hand-written, one invocation per element", PLAIN, c"entry::gemm_naive", [tiles, tiles, 1], false),
        ("coop 16×16 per subgroup", COOP, c"entry::gemm_1x1", [tiles, tiles, 1], true),
        ("coop 32×32 per subgroup", COOP, c"entry::gemm_2x2", [tiles / 2, tiles / 2, 1], true),
        ("coop 64×64, k unrolled 2", COOP, c"entry::gemm_4x4_k2", [tiles / 4, tiles / 4, 1], true),
        ("coop 64×64 pipelined", COOP, c"entry::gemm_4x4_pipelined", [tiles / 4, tiles / 4, 1], true),
        ("coop 64×64 pipelined, 2×2 subgroups/workgroup", COOP, c"entry::gemm_4x4_pipelined_wg4", [tiles / 8, tiles / 8, 1], true),
        ("coop staged 128×128 block", COOP, c"entry::gemm_staged", [blocks, blocks, 1], true),
        ("array 16×16 per subgroup", PLAIN, c"entry::gemm_1x1", [tiles, tiles, 1], false),
        ("array 32×32 per subgroup", PLAIN, c"entry::gemm_2x2", [tiles / 2, tiles / 2, 1], false),
        ("array 64×64 per subgroup", PLAIN, c"entry::gemm_4x4", [tiles / 4, tiles / 4, 1], false),
        ("array 64×64 pipelined, 2×2 subgroups/workgroup", PLAIN, c"entry::gemm_4x4_pipelined_wg4", [tiles / 8, tiles / 8, 1], false),
        ("array one 64×64 tile, no blocking by hand", PLAIN, c"entry::gemm_tile64", [tiles / 4, tiles / 4, 1], false),
        ("raw rust-gpu 64×64, no hydroplane", PLAIN, c"entry::gemm_raw", [tiles / 4, tiles / 4, 1], false),
    ];
    for (label, spv, entry, groups, needs_coop) in variants {
        if needs_coop && !coop {
            println!("  {label:<46} skipped (no cooperative matrix)");
            continue;
        }
        let kernel = gpu.kernel(spv, entry, &[&da, &db, &dc]);
        let first = gpu.dispatch(&kernel, groups);
        let mut ns = if first > SLOW_NS {
            vec![first]
        } else {
            for _ in 1..WARMUP {
                gpu.dispatch(&kernel, groups);
            }
            (0..REPS).map(|_| gpu.dispatch(&kernel, groups)).collect()
        };
        ns.sort_by(f64::total_cmp);
        let median = ns[ns.len() / 2];
        let c = gpu.download(&dc);
        let worst = samples
            .iter()
            .zip(&want)
            .map(|(&(r, col), &w)| (c[r * n + col] as f64 - w).abs() / w.abs().max(1.0))
            .fold(0.0, f64::max);
        println!(
            "  {label:<46} {:>9.3} ms  {:>7.2} TFLOP/s  ({} run(s), max rel err {worst:.1e})",
            median / 1e6,
            flops / median / 1e3,
            ns.len()
        );
        gpu.drop_kernel(kernel);
    }
}
