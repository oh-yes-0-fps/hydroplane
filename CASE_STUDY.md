# Case study: escape-time Mandelbrot, five ways

The pitch behind hydroplane is **write once, nearly optimal everywhere**: one kernel, scalar-looking code, and the ISA width and ILP unroll get raised for you (at compile time when the target is pinned, at run time otherwise). To back that up, here's one problem taken through the usual manual progression, with every stage measured.

The problem: for each point `c` in a buffer, iterate `z = z² + c` (up to 100 times) and record how many iterations `|z|² ≤ 4` held. Embarrassingly parallel across points, but *divergent*: neighboring points escape at different iterations, which is exactly what makes it a pain to vectorize by hand. No geometry types involved, just `&[f32]` in, `&mut [f32]` out.

All numbers: Apple M5 Pro, rustc 1.95, `RUSTFLAGS="-C target-cpu=native"`, criterion `--quick`, 4096 points, `MAX_ITER = 100`. Reproduce with:

```
RUSTFLAGS="-C target-cpu=native" cargo bench -p hydroplane-example --bench case_study
```

The full implementations live in [`hydroplane-example/benches/case_study.rs`](hydroplane-example/benches/case_study.rs) (the hydroplane kernel in [`hydroplane-example/src/mandelbrot.rs`](hydroplane-example/src/mandelbrot.rs)).

## Stage 0: scalar

```rust
pub fn mandelbrot_scalar(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    for i in 0..cx.len() {
        let (cxi, cyi) = (cx[i], cy[i]);
        let (mut zx, mut zy) = (0.0f32, 0.0f32);
        let mut count = 0.0f32;
        for _ in 0..max_iter {
            let (zx2, zy2) = (zx * zx, zy * zy);
            if zx2 + zy2 > 4.0 {
                break;
            }
            count += 1.0;
            let nzx = zx2 - zy2 + cxi;
            zy = (zx * zy) * 2.0 + cyi;
            zx = nzx;
        }
        out[i] = count;
    }
}
```

**108.6 µs.** 17 lines, obviously correct. The compiler can't autovectorize it: the inner loop's trip count is data-dependent per element, so there's nothing for it to work with.

## Stage 1: portable SIMD (`wide`, `f32x4`)

The standard first move: grab a portable SIMD crate and port the loop at NEON's native 4-lane width. Control flow turns into mask algebra. The branch becomes a lane mask, the conditional increment a blend, and you write a manual copy-in/copy-out tail for lengths that aren't a multiple of 4:

```rust
pub fn mandelbrot_wide4(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    let n = cx.len();
    let four = f32x4::splat(4.0);
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    let mut off = 0;
    while off < n {
        let cnt = 4.min(n - off);
        let (mut bx, mut by) = ([0.0f32; 4], [0.0f32; 4]);
        bx[..cnt].copy_from_slice(&cx[off..off + cnt]);
        by[..cnt].copy_from_slice(&cy[off..off + cnt]);
        let cxv = f32x4::from(bx);
        let cyv = f32x4::from(by);
        let mut zx = zero;
        let mut zy = zero;
        let mut count = zero;
        for _ in 0..max_iter {
            let zx2 = zx * zx;
            let zy2 = zy * zy;
            let active = (zx2 + zy2).simd_le(four);
            if active.none() {
                break;
            }
            count += active.blend(one, zero);
            let nzx = zx2 - zy2 + cxv;
            zy = (zx * zy) * 2.0 + cyv;
            zx = nzx;
        }
        out[off..off + cnt].copy_from_slice(&count.to_array()[..cnt]);
        off += 4;
    }
}
```

**146.6 µs, slower than scalar.** Twice the code and you actually lose:

- **Divergence.** A 4-lane block keeps iterating until its *slowest* lane escapes, so lanes burn iterations on masked-out work.
- **Latency-bound.** There's still only one dependency chain in flight (`zx`/`zy` feed the next iteration), so the FMA pipes idle waiting on each other, same as the scalar loop. Being 4 lanes wide buys nothing when latency is the limit rather than throughput.
- **Exit-check overhead.** NEON has no movemask instruction, so every iteration's `active.none()` is an emulated compare + mask + horizontal-add dance, amortized over just 4 points.

## Stage 2: same crate, restructured for ILP

The fix for latency-bound loops: put several *independent* blocks in flight so their dependency chains overlap. Four `f32x4` blocks (16 points per pass), FMA via `mul_add`, one combined exit check per iteration. Here it is in full, because the full version is the point:

```rust
pub fn mandelbrot_wide_ilp(cx: &[f32], cy: &[f32], max_iter: u32, out: &mut [f32]) {
    let n = cx.len();
    let four = f32x4::splat(4.0);
    let one = f32x4::splat(1.0);
    let zero = f32x4::splat(0.0);
    let load = |s: &[f32], o: usize| f32x4::from(<[f32; 4]>::try_from(&s[o..o + 4]).unwrap());
    let mut i = 0;
    while i + 16 <= n {
        macro_rules! block {
            ($cxv:ident, $cyv:ident, $zx:ident, $zy:ident, $count:ident, $j:literal) => {
                let $cxv = load(cx, i + 4 * $j);
                let $cyv = load(cy, i + 4 * $j);
                let (mut $zx, mut $zy, mut $count) = (zero, zero, zero);
            };
        }
        block!(cx0, cy0, zx0, zy0, n0, 0);
        block!(cx1, cy1, zx1, zy1, n1, 1);
        block!(cx2, cy2, zx2, zy2, n2, 2);
        block!(cx3, cy3, zx3, zy3, n3, 3);
        for _ in 0..max_iter {
            macro_rules! step {
                ($cxv:ident, $cyv:ident, $zx:ident, $zy:ident, $count:ident) => {{
                    let x2 = $zx * $zx;
                    let y2 = $zy * $zy;
                    let active = (x2 + y2).simd_le(four);
                    $count += active & one;
                    let nzx = x2 - y2 + $cxv;
                    $zy = ($zx * $zy).mul_add(f32x4::splat(2.0), $cyv);
                    $zx = nzx;
                    active
                }};
            }
            let a0 = step!(cx0, cy0, zx0, zy0, n0);
            let a1 = step!(cx1, cy1, zx1, zy1, n1);
            let a2 = step!(cx2, cy2, zx2, zy2, n2);
            let a3 = step!(cx3, cy3, zx3, zy3, n3);
            if ((a0 | a1) | (a2 | a3)).none() {
                break;
            }
        }
        out[i..i + 4].copy_from_slice(&n0.to_array());
        out[i + 4..i + 8].copy_from_slice(&n1.to_array());
        out[i + 8..i + 12].copy_from_slice(&n2.to_array());
        out[i + 12..i + 16].copy_from_slice(&n3.to_array());
        i += 16;
    }
    mandelbrot_wide4(&cx[i..], &cy[i..], max_iter, &mut out[i..]);
}
```

**70.2 µs, 1.5× scalar.** The ILP restructure is where the speedup was hiding, not the lane width. Look at what it took to get there: sixteen state variables tracked by naming convention, two `macro_rules!` definitions inside the loop body so the compiler keeps everything in registers, four hand-numbered stores, and stage 1 still has to hang around as the tail handler. And the math, the part you actually care about, is now buried inside a macro. Getting it to run fast wasn't mechanical either:

- The natural spelling, `[f32x4; 4]` arrays with a `for j in 0..4` inner loop, ran at **139.8 µs**: LLVM kept the union-typed `wide` values on the stack and the loop spilled every iteration. The blocks had to be hand-unrolled into individually named variables (macros, in practice) before they stayed in registers.
- **K = 4 is tuned for this core.** The right number of chains tracks the FP pipe count and FMA latency, and those differ per machine *even on the same architecture*: an M-series performance core, a Cortex-A72 and a Graviton all want a different K. Hardcode one and you're leaving performance on the table everywhere else.

## Honourable mention: raw NEON intrinsics

The same 4-block ILP structure written straight in `core::arch::aarch64` intrinsics (`vfmaq_f32`, `vcleq_f32`, `vmaxvq_u32`, ...):

**41.9 µs, 2.6× scalar.** This is the ceiling, and it's not a practical place to live: ~40 lines of `unsafe` covering exactly one ISA (the AVX2 port is a rewrite with different mask idioms, AVX-512 another with k-registers, SVE another with sizeless registers), the unroll factor is still per-machine, and every element type means another copy. It's here as the reference bar, nothing more.

## Stage 3: hydroplane

```rust
#[kernel]
pub fn mandelbrot_hp<'a>(ctx: Gang, cx: &'a [f32], cy: &'a [f32], max_iter: u32, out: &'a mut [f32]) {
    let (zero, one, four) = (ctx.splat(0.0), ctx.splat(1.0), ctx.splat(4.0));
    ctx.zip_map(cx, cy, out, 0.0, 0.0, |cxv, cyv| {
        let (mut zx, mut zy, mut count) = (zero, zero, zero);
        for _ in 0..max_iter {
            let zx2 = zx * zx;
            let zy2 = zy * zy;
            let active = (zx2 + zy2).le(four);
            if !active.any() {
                break;
            }
            count = count + one.select(active, zero);
            let nzx = zx2 - zy2 + cxv;
            zy = (zx * zy) * 2.0 + cyv;
            zx = nzx;
        }
        count
    });
}
```

**46.0 µs: 2.4× scalar, 1.5× faster than the best portable hand effort, within 10% of raw NEON.** Structurally it's stage 0 with masks. No widths, no tails, no macros to force register allocation, no `unsafe`, no ISA named anywhere. What the `#[kernel]` dispatch brings:

- **ISA raising.** Runtime detection picks the widest implemented backend (NEON here, AVX-512 on an x86 host, same binary). With `--cfg hp_static_dispatch` and a pinned `target-cpu`, the ladder folds away at compile time.
- **ILP raising.** The unroll factor `K` (stage 2's blocks-in-flight) is *measured* once per process by a startup sweep and baked into the monomorphized kernel as a compile-time constant, so `zip_map` runs `K` independent blocks per pass. That's stage 2's per-machine-K problem handled: every machine gets its own K and nobody hardcodes anything. On a pinned build, `build.rs` resolves `K` at compile time instead.
- **Register-resident codegen.** The backends are written per-ISA against `core::arch`, so the unrolled blocks land in registers and the stage-2 spill fight never happens.
- **Tails and masks.** `zip_map` owns the chunking and the masked remainder.
- **Element-agnostic.** The same body runs `f64`, `f16`, `bf16`. Every stage above would be a rewrite per type.

## Scoreboard

| | lines | `unsafe` | ISAs covered | time (4096 pts) | vs scalar |
|---|---|---|---|---|---|
| scalar | 17 | 0 | n/a (no SIMD) | 108.6 µs | 1.0× |
| `wide` f32x4 | 32 | 0 | any | 146.6 µs | 0.74× |
| `wide` + 4× hand ILP | ~50 | 0 | any (K per-machine) | 70.2 µs | 1.55× |
| raw NEON + 4× ILP *(honourable mention)* | ~40 *per ISA* | all of it | 1 | 41.9 µs | 2.59× |
| hydroplane | 20 | 0 | all implemented | 46.0 µs | 2.36× |

Across sizes:

| points | scalar | `wide` f32x4 | `wide` + ILP | raw NEON + ILP | hydroplane |
|---|---|---|---|---|---|
| 256 | 6.57 µs | 9.61 µs | 4.23 µs | 2.55 µs | 2.87 µs |
| 1024 | 25.9 µs | 35.3 µs | 16.5 µs | 9.92 µs | 10.7 µs |
| 4096 | 108.6 µs | 146.6 µs | 70.2 µs | 41.9 µs | 46.0 µs |

The raw-intrinsics build keeps a ~10% edge; a hand-scheduled inner loop with hoisted constants is hard to beat exactly. What it doesn't keep is the other ISAs, the other element types, the per-machine unroll tuning, or the safety. That's the trade hydroplane makes: ceiling-adjacent performance from stage-0-shaped code, on every backend at once.
