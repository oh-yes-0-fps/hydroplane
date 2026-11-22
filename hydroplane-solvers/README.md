# hydroplane-solvers

Sparse and dense linear solvers, tuned per device. The front doors read the pattern, detect
the machine (CPU threads, Apple's matrix units, any Vulkan GPU), and pick the factorization,
device, precision and ordering themselves, check the first solve's true residual, and refine or
fall back when it misses:

```rust
use hydroplane_solvers::{GeneralSolver, IndefiniteSolver, Solver};

let mut s = Solver::new(&col_ptr, &row_idx)?; // symmetric positive definite, lower triangle (CSC)
s.factorize(&values)?;
s.solve_in_place(&mut x)?; // x: b in, A⁻¹ b out

let mut s = IndefiniteSolver::new(&col_ptr, &row_idx)?; // symmetric, any inertia (KKT systems)
let mut s = GeneralSolver::new(&col_ptr, &row_idx)?; // any square matrix, given whole
```

Iterative solvers take the operator as a closure, so the matrix never has to be assembled:

```rust
use hydroplane_solvers::iterative::{Cg, Jacobi};

let report = Cg::new().solve(&mut |x: &[f64], y: &mut [f64]| apply(x, y), &mut Jacobi::new(&diag), &b[..], &mut x[..]);
```

Every solver works with its defaults. To change a setting, each has `with_options` taking a
settings struct with `Default` (`Solver::with_options(&col_ptr, &row_idx, Options { tolerance:
1e-10, ..Default::default() })`), or a public `options` field for the iterative ones. The solvers
underneath are public too, mostly in the same shape: `new(..)`, `factorize(values)`,
`solve_in_place(rhs)`.

| Module | What |
|---|---|
| `auto` | `Solver`: detection, a one-off calibration, a pure policy, override knobs, self-checking |
| `general` | `GeneralSolver` (any square matrix) and `IndefiniteSolver` (symmetric, any inertia): the same kind of front door, with fallbacks and refinement |
| `cholesky` | the symbolic `Analysis` every Cholesky shares; supernodal Cholesky and PCG on the CPU (`faer`) |
| `order` | nested dissection |
| `vulkan` (feature `vulkan`) | rust-gpu kernels on any Vulkan GPU, in `f32`, `f64`, and `f64` emulated in `f32` pairs (`Ef64`) for GPUs without it: supernodal Cholesky, signed `L D Lᵀ` for saddle points, CG on a Cholesky, AMG or Jacobi preconditioner, Jacobi BiCGStab |
| `sme` (macOS) | supernodal Cholesky on Apple's SME matrix units, whose factor the GPU can solve with |
| `accelerate` (macOS) | Apple's sparse Cholesky, `L D Lᵀ`, LU and QR |
| `ldlt` | sparse `L D Lᵀ` for symmetric indefinite (KKT) systems, a KKT ordering, inertia |
| `lu` | sparse LU: `FrontalLu` (multifrontal over the structure of `A + Aᵀ`, an MC64-style matching for weak diagonals) and `faer`'s |
| `qr` | sparse QR for least squares |
| `banded` | `L D Lᵀ` of symmetric band matrices (chains over time) |
| `iterative` | CG, MINRES, GMRES / FGMRES, BiCGStab, LSQR over any vector layout; Jacobi, block-Jacobi, Chebyshev, ILU(0), IC(0), smoothed-aggregation AMG and additive preconditioners |
| `eigen` | LOBPCG for the smallest eigenpairs of `A x = λ x` or `A x = λ B x` (modes, buckling), with any preconditioner |
| `dense` | shifted dense Cholesky, two-level coarse correction |
| `sparse` | `SellMatrix`: `y = A x` for an assembled sparse matrix in SELL-C-σ layout (sorted, register-width row chunks), the product every solver here uses for residuals and refinement, and the operator to hand the `iterative` solvers |
| `batched` | thousands of independent small problems at once, one per SIMD lane: 3×3 symmetric eigen-decomposition, signed SVD and polar decomposition (FEM-ready), `N×N` Cholesky and `L D Lᵀ` solves |
| `bvh` | bounding volume hierarchy with nodes as wide as the `f32` registers (4 on NEON/SSE, 8 on AVX2, 16 on AVX-512): ray, nearest, overlap and point queries, leaves tested on the traversal's own SIMD gang, refit |

## Benchmarks

Every table comes from one program, `examples/compare.rs`, run on two machines:

| | M5 Pro | svc01 |
|---|---|---|
| CPU | Apple M5 Pro, 18 cores (6 + 12), 48 GB, macOS 26.6 | AMD Ryzen 9 5950X, pinned to one CCD: 8 cores, 16 threads |
| CPU matrix units | SME2 (used by Accelerate and `sme`) | — |
| GPU | Apple M5 Pro, 20 cores, through MoltenVK; no `f64` | NVIDIA RTX 5090, Vulkan, driver 595.84 |

- Each cell is `factor / solve` in milliseconds of wall time, the best of 3 factorizations and 5
  solves. GPU times include copying values and right-hand sides in and results out.
- The residual is `‖b − A x‖ / ‖b‖`, computed in `f64` on the CPU. `—` means the machine lacks
  that hardware or library.
- On svc01 the CPU solvers get one of its two CCDs (the CI runner's half of the machine), so its
  CPU numbers are about half of what the whole 5950X could do.
- The Mac is a working machine: each table ran only once its 1-minute load was under 4 and its 5-minute load under 6, and each row is the best of two or three such runs (svc01: best of three, under its benchmark lock). The GPU on svc01 is shared with another service, so its rows are also the best of several runs.

### Sparse Cholesky (symmetric positive definite)

A real tangent stiffness matrix from pliance's weld-distortion slab: 87,954 unknowns, 3.3 M
nonzeros in the lower triangle, 3 dofs per node. Nested dissection gives a factor of 51.2 M
entries. Accelerate orders with its own METIS. cuDSS (NVIDIA's sparse direct solver, CUDA 12.8) is
NVIDIA's reference on the same GPU; its factor time is a refactorization after analysis, plus the
values' upload.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| faer, CPU, f64 | 442 / 37.3 | 341 / 49.4 | 5.0e-10 |  |
| faer, CPU, f32 | 325 / 13.9 | 246 / 25.1 | 1.3e-1 |  |
| Accelerate, CPU matrix units, f64 | 60.6 / 14.7 | — | 5.1e-10 |  |
| SME (this crate), CPU matrix units, f64 | 66.4 / 4.67 | — | 6.9e-10 |  |
| Accelerate, CPU matrix units, f32 | 29.1 / 8.96 | — | 1.3e-1 |  |
| SME (this crate), CPU matrix units, f32 | 29.7 / 2.46 | — | 1.4e-1 |  |
| Vulkan (this crate), GPU, f64 | — | 42.3 / 1.53 | 5.9e-10 |  |
| Vulkan (this crate), GPU, emulated f64 | 133 / 3.97 | 25.0 / 1.35 | 1.3e-8 |  |
| Vulkan (this crate), GPU, f32 | 67.6 / 2.37 | 12.5 / 0.884 | 1.9e-1 |  |
| SME f64 factor → GPU solve (emulated f64) | 85.1 / 3.98 | — | 9.2e-9 |  |
| cuDSS (NVIDIA, CUDA 12.8), GPU, f64 | — | 78.2 / 1.38 | 5.1e-10 | refactorization incl. upload |
| cuDSS (NVIDIA, CUDA 12.8), GPU, f32 | — | 24.2 / 0.918 | 1.5e-1 | refactorization incl. upload |

`f32` factors are only good to about 15 % on this matrix (the residual column); they are the
preconditioners of the next table. `SME f64 factor → GPU solve` factors on the CPU's matrix units
and hands the factor to the GPU for the solves (best when one factorization serves several
solves).

### Conjugate gradient on a cheap factor, to `1e-8`

`f64` conjugate gradient preconditioned by an `f32` Cholesky factor: a cheap factorization, then a
few iterations, each a preconditioner solve and a matrix-vector product. Same matrix.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| faer f32 factor, f64 CG on the CPU | 334 / 270 | 268 / 266 | 5.5e-10 | 9 iterations |
| SME f32 factor, f64 CG on the CPU | 29.7 / 31.4 | — | 9.2e-9 | 8 iterations |
| Accelerate f32 factor, f64 CG on the CPU | 30.0 / 91.8 | — | 1.3e-9 | 9 iterations |
| GPU f32 factor, f64 CG on the GPU | — | 16.9 / 10.2 | 9.4e-10 | 9 iterations |
| GPU f32 factor, emulated-f64 CG on the GPU | 68.3 / 28.7 | 16.7 / 9.43 | 7.2e-9 | M5 Pro: 8 iterations; svc01: 9 iterations |
| GPU f32 factor, f64 CG on the CPU | 67.9 / 31.4 | 12.4 / 32.5 | 8.0e-10 | M5 Pro: 8 iterations; svc01: 9 iterations |

### `Solver`, choosing for itself

`Solver` with only the needs given (tolerance, solves per factorization, 10 factorizations
expected). The times are what a caller sees per cycle: the factorization, then per solve,
including the residual checks it makes on the first solve after each factorization and every
16th. The note is what it chose. On a Mac with SME it uses this crate's SME factorization rather
than Accelerate, which it calibrates only where there is no SME. `1e-10` is below this matrix's
residual floor in `f64` for this right-hand side (`6.9e-10`, whatever the factorization), so for
that row `Solver` tries every strategy and settles on the last; the note is where it ended.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| 1 solve per factorization, to 1e-8 | 37.3 / 42.7 | 19.2 / 17.6 | 7.3e-9 | M5 Pro: emulated f64 PCG on the GPU (f32 factor by SME); svc01: emulated f64 PCG on the GPU (f32 factor) |
| 10 solves per factorization, to 1e-8 | 67.1 / 11.1 | 44.5 / 9.39 | 6.9e-10 | M5 Pro: direct f64 on the CPU by SME; svc01: direct f64 on the GPU |
| 1 solve per factorization, to 1e-10 | 65.7 / 12.8 | 19.2 / 23.1 | 6.9e-10 | M5 Pro: f64 PCG on the CPU (f32 factor); svc01: direct emulated f64 on the GPU |

### Sparse `L D Lᵀ` (symmetric indefinite)

A KKT system `[H Bᵀ; B 0]`: `H` a 36³ 7-point Laplacian (46,656 unknowns), `B` 23,328 constraints
each averaging a pair of neighbours, and a zero block. AMD eliminates every constraint first, so
all 23,328 of its pivots are zero and get regularized (residual `6e-8` without refinement).
`ldlt::kkt_ordering` places each constraint after its first neighbour instead: no pivot is
replaced and the factor is smaller. In that ordering every pivot's sign is known, which the GPU's
signed factorization `A = L S Lᵀ` (`vulkan::VulkanLdlt`) relies on.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| faer, f64, AMD (constraint pivots regularized) | 228 / 6.84 | 194 / 17.8 | 5.9e-8 |  |
| faer, f64, AMD, + GMRES refinement to 1e-12 | 229 / 40.6 | 197 / 92.7 | 6.3e-16 | 3 GMRES steps |
| faer, f64, KKT ordering | 223 / 6.50 | 184 / 17.2 | 5.3e-15 |  |
| faer, f64, KKT ordering, Bunch–Kaufman | 324 / 6.79 | 1,543 / 18.6 | 4.8e-15 |  |
| faer, f32, KKT ordering, + GMRES refinement to 1e-12 | 189 / 24.9 | 141 / 55.5 | 6.2e-13 | 3 GMRES steps |
| Vulkan (this crate), GPU, f64, KKT ordering | — | 24.9 / 1.16 | 3.3e-15 |  |
| Vulkan (this crate), GPU, emulated f64, KKT ordering | 71.5 / 2.98 | 13.1 / 0.992 | 9.6e-14 |  |
| Vulkan (this crate), GPU, f32, KKT ordering, + GMRES refinement to 1e-12 | 37.8 / 14.6 | 7.42 / 4.25 | 3.9e-13 | 3 GMRES steps |
| Accelerate, f64, threshold partial pivoting | 195 / 8.52 | — | 3.8e-15 |  |
| Accelerate, f64, supernode Bunch–Kaufman | 77.2 / 6.30 | — | 8.4e-8 |  |

### Sparse LU (general square)

Upwinded convection–diffusion on a 36³ grid: 46,656 unknowns, nonsymmetric.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| faer, f64 | 1,207 / 18.6 | 1,670 / 35.4 | 4.3e-15 |  |
| faer, f32, + GMRES refinement to 1e-12 | 1,008 / 102 | 953 / 193 | 6.0e-16 | 6 GMRES steps |
| FrontalLu (this crate), f64, nested dissection | 127 / 6.12 | 200 / 11.0 | 3.5e-15 | 0 pivots perturbed |
| FrontalLu (this crate), f64, nested dissection, matching | 121 / 6.75 | 195 / 11.1 | 3.4e-15 | 0 pivots perturbed |
| FrontalLu (this crate), f32, + GMRES refinement to 1e-12 | 86.1 / 58.3 | 94.5 / 60.2 | 6.0e-16 | 6 GMRES steps |
| Accelerate, f64 | 63.8 / 3.11 | — | 4.3e-15 |  |

`faer`'s LU orders columns with COLAMD and sizes its supernodes for any row pivoting, which on
nearly symmetric patterns like this one is expensive. `FrontalLu` keeps the nested-dissection
ordering and supernodes of a Cholesky of `A + Aᵀ` and pivots only within supernodes (perturbing
a pivot that is too small, then refining), as PARDISO does; with a matching first, rows are
permuted so the large entries sit on the diagonal. Its solve walks the elimination tree's levels
in parallel.

### Band matrices

A symmetric positive definite chain: 1,000,000 unknowns, half-bandwidth 4. `BandedLdlt` streams
down the band with no symbolic analysis; the sparse factorization is the general alternative.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| BandedLdlt, f64 | 11.9 / 8.02 | 38.8 / 10.7 | 2.4e-16 |  |
| faer sparse Cholesky, f64 (AMD) | 54.9 / 18.1 | 163 / 76.9 | 2.5e-16 |  |

### Krylov methods

262,144 unknowns (a 64³ 7-point stencil), to `1e-8` relative to `‖b‖`. The CPU solvers apply it
without assembling a matrix; the GPU ones, AMG, ILU(0) and IC(0) work on the assembled matrix.
Solve times only, with set-up times in the notes.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| CG + Jacobi, Poisson | 153 | 88.7 | 1.0e-8 | 152 iterations |
| CG + Chebyshev(4)–Jacobi, Poisson | 270 | 161 | 9.0e-9 | 74 iterations |
| MINRES, indefinite (Poisson − 0.05 I) | 807 | 421 | 1.0e-8 | M5 Pro: 523 iterations; svc01: 508 iterations |
| GMRES(30) + Jacobi, convection–diffusion | 514 | 809 | 9.8e-9 | 272 iterations |
| FGMRES(30) + Jacobi, convection–diffusion | 519 | 857 | 9.8e-9 | 272 iterations |
| BiCGStab + Jacobi, convection–diffusion | 203 | 115 | 5.0e-9 | M5 Pro: 91 iterations; svc01: 90 iterations |
| BiCGStab + Jacobi on the GPU (Vulkan, f64), convection–diffusion | — | 10.4 | 8.5e-9 | 89 iterations |
| BiCGStab + Jacobi on the GPU (Vulkan, emulated f64), convection–diffusion | 46.3 | 8.48 | 2.4e-9 | M5 Pro: 91 iterations; svc01: 94 iterations |
| BiCGStab + ILU(0), convection–diffusion | 1,349 | 303 | 9.9e-9 | M5 Pro: 25 iterations, set up in 10 ms; svc01: 25 iterations, set up in 25 ms |
| CG + Jacobi on the GPU (Vulkan, f64), Poisson | — | 8.08 | 1.0e-8 | 152 iterations |
| CG + Jacobi on the GPU (Vulkan, emulated f64), Poisson | 37.7 | 6.22 | 1.0e-8 | 152 iterations |
| CG + AMG on the GPU (Vulkan, f64), Poisson | — | 2.68 | 8.3e-9 | 14 iterations, set up in 167 ms |
| CG + AMG on the GPU (Vulkan, emulated f64), Poisson | 20.3 | 2.56 | 8.3e-9 | M5 Pro: 14 iterations, set up in 103 ms; svc01: 14 iterations, set up in 136 ms |
| CG + AMG, Poisson | 86.1 | 131 | 8.3e-9 | M5 Pro: 14 iterations, 3 levels, set up in 54 ms; svc01: 14 iterations, 3 levels, set up in 91 ms |
| CG + IC(0), Poisson | 1,319 | 362 | 9.4e-9 | M5 Pro: 58 iterations, set up in 10 ms; svc01: 58 iterations, set up in 27 ms |

Chebyshev(4) halves CG's iterations but costs four operator applications each, so it pays off
when the operator is cheaper than the iteration's other work (a multigrid smoother), not here.
GMRES spends most of its time orthogonalizing against up to 30 basis vectors; BiCGStab is the
cheaper nonsymmetric choice when it converges. ILU(0) and IC(0) cut the iterations 3× and 2.6×,
but their triangular solves crawl through about 190 wavefront levels each way (level-scheduled,
in parallel within a level), so on a stencil this cheap Jacobi still wins; they pay off where
the operator is expensive or convergence is hard.

The right-hand side here is rough, which Jacobi damps quickly; under a smooth load (all ones) on
finer grids its iterations grow with the grid while AMG's do not: 319 against 16 at 2.1 million
unknowns, 398 against 17 at 4.1 million. AMG's set-up is the price.

On the GPU, Jacobi CG and BiCGStab run every iteration on the device, reading the scalars back
every few iterations. GPU AMG (`VulkanPcg::amg`) sets the hierarchy up
on the CPU, uploads it, and runs the V-cycle in `f32` inside a CG of working precision: it takes
the same 14 iterations as the `f64` CPU cycle, in a fiftieth of the time. Its set-up (the CPU
hierarchy, then about 50 ms of upload) costs as much as 22 Jacobi solves here, so it pays off
when a hierarchy serves many right-hand sides or keeps serving as values change
(`VulkanPcg::set_values`), and on fine grids: under the smooth load at 4.1 million unknowns GPU
AMG solves in 48 ms to GPU Jacobi's 216 ms, after 2.9 s of set-up. The set-up is bound by memory
bandwidth rather than cores: on one CCD of the 5950X (about 22 GB/s) it runs as fast on 4
threads as on 16.

None of this reaches the welded slab of the SPD tables: its stiff penalty
terms leave GPU AMG (translations as the near-null space, one or three dofs per node) at a
residual of `2e-4` after 3,000 iterations and Jacobi at `1e-2` after 20,000, as AmgX's stock
aggregation configurations fail on it too. A factorization as the preconditioner (`f32`
Cholesky, nine iterations) is what converges there.

### Least squares

Fitting a 300 × 300 grid of nodes (90,000 unknowns) to 178,802 scattered data points by bilinear
interpolation, plus first-difference smoothing rows: 358,202 rows. The residual is that of the
normal equations, `‖Aᵀ(b − A x)‖ / ‖Aᵀ b‖`.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| LSQR, matrix-free, no preconditioner | — / 130 | — / 195 | 7.6e-10 | 83 iterations |
| faer sparse QR, f64 | 700 / 12.9 | 969 / 38.5 | 2.9e-16 |  |
| Accelerate sparse QR, f64 | 109 / 43.5 | — | 3.5e-16 |  |

LSQR here has no preconditioner; the smoothing rows make the problem well conditioned enough
for that. Sparse QR is exact and costs a factorization.

### Eigenvalues

The 8 lowest eigenpairs of the slab tangent by LOBPCG (with two guard vectors), preconditioned by
a Cholesky factor of the matrix. The residual is the worst pair's `‖A x − λ x‖ / λ`, the worst
of any run. The slab's contact penalties make `‖A‖` so large that rounding stops these residuals
at tens to hundreds of `ε ‖A‖ / λ`, around `1e-7` to `1e-5` here; LOBPCG stops there as well,
measuring `‖A‖` by a few power steps first. The factor time is in the first column, LOBPCG's in
the second.

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| LOBPCG, faer Cholesky (CPU, f64) preconditioner | 770 / 4,124 | 679 / 5,244 | 2.2e-7 | 10 iterations, λ₁ = 7.4128e1 |
| LOBPCG, Vulkan Cholesky (GPU, f64) preconditioner | — | 42.9 / 1,477 | 2.6e-7 | 10 iterations, λ₁ = 7.4128e1 |
| LOBPCG, Vulkan Cholesky (GPU, emulated f64) preconditioner | 144 / 1,500 | 25.8 / 1,468 | 4.6e-7 | 10 iterations, λ₁ = 7.4128e1 |
| LOBPCG, Vulkan Cholesky (GPU, f32) preconditioner | 77.6 / 1,252 | 13.7 / 1,454 | 3.8e-5 | M5 Pro: 9 iterations, λ₁ = 7.4128e1; svc01: 10 iterations, λ₁ = 7.4128e1 |
| LOBPCG, SME Cholesky (CPU matrix units, f64) preconditioner | 79.7 / 1,416 | — | 4.9e-7 | 10 iterations, λ₁ = 7.4128e1 |

### Dense

| Solver | M5 Pro | Ryzen 9 5950X + RTX 5090 | Residual | Notes |
|---|---:|---:|---:|---|
| DenseCholesky, f64, n = 400 | 2.95 / 0.070 | 6.63 / 0.102 | 7.6e-16 |  |

### Reproducing

```sh
cargo run --release --features vulkan --example compare -- SLAB.bin   # every table
ONLY=ldlt cargo run --release --example compare                       # one table
```

`SLAB.bin` is a dumped matrix in the `cholesky` example's format; without it the three
Cholesky tables are skipped. On svc01 the run held the benchmark lock on CCD0
(`flock ~/bench/ccd0.lock taskset -c 0-7,16-23`).
