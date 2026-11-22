#![cfg_attr(target_arch = "spirv", no_std)]

use hydroplane::{Gang, Layout, MatrixBackend, MatrixKernel, f16};

/// Side of the square `f16 → f32` tile [`Probe`] multiplies.
pub const DIM: usize = 16;

/// One `DIM×DIM×DIM` tile through every tile op: `mma`, an elementwise `map` and `zip`, the
/// row/column broadcasts and reductions, and the store.
pub struct Probe<'a> {
    pub a: &'a [f16],
    pub b: &'a [f16],
    pub c: &'a [f32],
    /// Row bias, then column scale, `DIM` each.
    pub bias: &'a [f32],
    pub out: &'a mut [f32],
    pub row_max: &'a mut [f32],
    pub col_sum: &'a mut [f32],
}

impl MatrixKernel<f16> for Probe<'_> {
    type Output = ();

    #[inline]
    fn run<S: MatrixBackend<f16>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f16>();
        let mut row_bias = [0.0f32; DIM];
        let mut col_scale = [0.0f32; DIM];
        let mut i = 0;
        while i < DIM {
            row_bias[i] = self.bias[i];
            col_scale[i] = self.bias[DIM + i];
            i += 1;
        }
        let d = tl
            .mma::<DIM, DIM, DIM>(tl.load_a_rm(self.a), tl.load_b_rm(self.b), tl.load_acc_rm(self.c))
            .map(|x| if x > 0.0 { x } else { 0.0 })
            .zip(tl.splat_acc(0.5), |x, y| x + y)
            .zip_rows(row_bias, |x, s| x + s)
            .zip_cols(col_scale, |x, s| x * s);
        let row_max = d.reduce_rows(-1.0e30, |m, x| if x > m { x } else { m });
        let col_sum = d.col_sums();
        let mut i = 0;
        while i < DIM {
            self.row_max[i] = row_max[i];
            self.col_sum[i] = col_sum[i];
            i += 1;
        }
        d.store_rm(self.out);
    }
}

/// Side of the square matrices [`Gemm`] multiplies.
pub const GEMM: usize = 4096;

/// `c = a·b` for `GEMM×GEMM` row-major matrices: one subgroup computes a `BM×BN` block of
/// `DIM×DIM` output tiles, `(tile_row, tile_col)` in units of blocks.
pub struct Gemm<'a, const BM: usize, const BN: usize, const KU: usize = 1> {
    pub a: &'a [f16],
    pub b: &'a [f16],
    pub c: &'a mut [f32],
    pub tile_row: usize,
    pub tile_col: usize,
}

impl<const BM: usize, const BN: usize, const KU: usize> MatrixKernel<f16> for Gemm<'_, BM, BN, KU> {
    type Output = ();

    #[inline]
    fn run<S: MatrixBackend<f16>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f16>();
        let row0 = self.tile_row * BM * DIM;
        let col0 = self.tile_col * BN * DIM;
        let mut acc = [[tl.zero_acc::<DIM, DIM>(); BN]; BM];
        let mut k = 0;
        while k < GEMM {
            hydroplane::unroll!(KU, u, {
                let kk = k + u * DIM;
                // Each tile is loaded once per step: a coop load is an `asm!` block, which the
                // compiler never merges.
                let first_b = tl.load_b::<DIM, DIM>(&self.b[kk * GEMM + col0..], GEMM, Layout::RowMajor);
                let mut b = [first_b; BN];
                hydroplane::unroll!(BN, j, {
                    if j > 0 {
                        b[j] = tl.load_b::<DIM, DIM>(&self.b[kk * GEMM + col0 + j * DIM..], GEMM, Layout::RowMajor);
                    }
                });
                let first_a = tl.load_a::<DIM, DIM>(&self.a[row0 * GEMM + kk..], GEMM, Layout::RowMajor);
                let mut a = [first_a; BM];
                hydroplane::unroll!(BM, i, {
                    if i > 0 {
                        a[i] = tl.load_a::<DIM, DIM>(&self.a[(row0 + i * DIM) * GEMM + kk..], GEMM, Layout::RowMajor);
                    }
                });
                acc = tl.mma_block(a, b, acc);
            });
            k += KU * DIM;
        }
        hydroplane::unroll!(BM, i, {
            hydroplane::unroll!(BN, j, {
                acc[i][j].store(&mut self.c[(row0 + i * DIM) * GEMM + col0 + j * DIM..], GEMM, Layout::RowMajor);
            });
        });
    }
}

/// [`Gemm`] software-pipelined: the next k step's `A`/`B` tiles are loaded before this step's
/// `mma`s are issued, so their latency hides behind the multiplies. The fastest variant on an
/// RTX 5090 at 4×4, at ~177 TFLOP/s with one subgroup per workgroup and ~188 with four (see
/// `entry::gemm_4x4_pipelined_wg4`).
pub struct GemmPipelined<'a, const BM: usize, const BN: usize> {
    pub a: &'a [f16],
    pub b: &'a [f16],
    pub c: &'a mut [f32],
    pub tile_row: usize,
    pub tile_col: usize,
}

impl<const BM: usize, const BN: usize> MatrixKernel<f16> for GemmPipelined<'_, BM, BN> {
    type Output = ();

    #[inline]
    fn run<S: MatrixBackend<f16>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f16>();
        let row0 = self.tile_row * BM * DIM;
        let col0 = self.tile_col * BN * DIM;
        macro_rules! load_a {
            ($i:expr, $k:expr) => {
                tl.load_a::<DIM, DIM>(&self.a[(row0 + $i * DIM) * GEMM + $k..], GEMM, Layout::RowMajor)
            };
        }
        macro_rules! load_b {
            ($j:expr, $k:expr) => {
                tl.load_b::<DIM, DIM>(&self.b[$k * GEMM + col0 + $j * DIM..], GEMM, Layout::RowMajor)
            };
        }
        let mut acc = [[tl.zero_acc::<DIM, DIM>(); BN]; BM];
        let mut a = [load_a!(0, 0); BM];
        let mut b = [load_b!(0, 0); BN];
        hydroplane::unroll!(BM, i, {
            a[i] = load_a!(i, 0);
        });
        hydroplane::unroll!(BN, j, {
            b[j] = load_b!(j, 0);
        });
        let mut k = DIM;
        while k < GEMM {
            let mut next_a = a;
            let mut next_b = b;
            hydroplane::unroll!(BM, i, {
                next_a[i] = load_a!(i, k);
            });
            hydroplane::unroll!(BN, j, {
                next_b[j] = load_b!(j, k);
            });
            acc = tl.mma_block(a, b, acc);
            a = next_a;
            b = next_b;
            k += DIM;
        }
        acc = tl.mma_block(a, b, acc);
        hydroplane::unroll!(BM, i, {
            hydroplane::unroll!(BN, j, {
                acc[i][j].store(&mut self.c[(row0 + i * DIM) * GEMM + col0 + j * DIM..], GEMM, Layout::RowMajor);
            });
        });
    }
}

/// Side of the output block one [`GemmStaged`] workgroup computes.
pub const BLOCK: usize = 128;
/// [`Gemm`] written the obvious way: one `T×T` output tile per subgroup, one `mma` per `DIM`-deep
/// step of `k`, no register blocking by hand.
pub struct GemmTile<'a, const T: usize> {
    pub a: &'a [f16],
    pub b: &'a [f16],
    pub c: &'a mut [f32],
    pub tile_row: usize,
    pub tile_col: usize,
}

impl<const T: usize> MatrixKernel<f16> for GemmTile<'_, T> {
    type Output = ();

    #[inline]
    fn run<S: MatrixBackend<f16>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f16>();
        let (row0, col0) = (self.tile_row * T, self.tile_col * T);
        let mut acc = tl.zero_acc::<T, T>();
        let mut k = 0;
        while k < GEMM {
            let a = tl.load_a::<T, DIM>(&self.a[row0 * GEMM + k..], GEMM, Layout::RowMajor);
            let b = tl.load_b::<DIM, T>(&self.b[k * GEMM + col0..], GEMM, Layout::RowMajor);
            acc = tl.mma(a, b, acc);
            k += DIM;
        }
        acc.store(&mut self.c[row0 * GEMM + col0..], GEMM, Layout::RowMajor);
    }
}

/// Depth of the `A`/`B` slices staged into workgroup memory per step.
pub const DEPTH: usize = 32;
/// Output tiles per side each of the workgroup's 2×2 subgroups owns.
const SUB_TILES: usize = BLOCK / 2 / DIM;
/// Row strides of the staged slices, padded by 8 halves (16 bytes) so the rows of one tile
/// start on different shared-memory banks.
pub const STRIDE_A: usize = DEPTH + 8;
pub const STRIDE_B: usize = BLOCK + 8;

/// [`Gemm`] with `A` and `B` staged through workgroup memory: the workgroup's threads copy a
/// `BLOCK×DEPTH` slice of `A` and a `DEPTH×BLOCK` slice of `B` once, then its four subgroups each
/// multiply a `64×64` quarter of the block from there.
pub struct GemmStaged<'a> {
    pub a: &'a [f16],
    pub b: &'a [f16],
    pub c: &'a mut [f32],
    pub shared_a: &'a mut [f16; BLOCK * STRIDE_A],
    pub shared_b: &'a mut [f16; DEPTH * STRIDE_B],
    pub block_row: usize,
    pub block_col: usize,
    /// This subgroup's quarter, `0..4` row-major.
    pub quarter: usize,
    pub thread: usize,
    pub threads: usize,
}

#[inline]
fn workgroup_barrier() {
    #[cfg(target_arch = "spirv")]
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
}

impl MatrixKernel<f16> for GemmStaged<'_> {
    type Output = ();

    #[inline]
    fn run<S: MatrixBackend<f16>>(self, ctx: Gang<S>) {
        let tl = ctx.tiles::<f16>();
        let (row0, col0) = (self.block_row * BLOCK, self.block_col * BLOCK);
        let (sub_row, sub_col) = ((self.quarter / 2) * SUB_TILES * DIM, (self.quarter % 2) * SUB_TILES * DIM);
        let mut acc = [[tl.zero_acc::<DIM, DIM>(); SUB_TILES]; SUB_TILES];
        let mut k0 = 0;
        while k0 < GEMM {
            let mut e = self.thread;
            while e < BLOCK * DEPTH {
                let (ar, ac) = (e / DEPTH, e % DEPTH);
                let (br, bc) = (e / BLOCK, e % BLOCK);
                self.shared_a[ar * STRIDE_A + ac] = self.a[(row0 + ar) * GEMM + k0 + ac];
                self.shared_b[br * STRIDE_B + bc] = self.b[(k0 + br) * GEMM + col0 + bc];
                e += self.threads;
            }
            workgroup_barrier();
            let mut kk = 0;
            while kk < DEPTH {
                hydroplane::unroll!(SUB_TILES, i, {
                    let a = tl.load_a::<DIM, DIM>(&self.shared_a[(sub_row + i * DIM) * STRIDE_A + kk..], STRIDE_A, Layout::RowMajor);
                    hydroplane::unroll!(SUB_TILES, j, {
                        let b = tl.load_b::<DIM, DIM>(&self.shared_b[kk * STRIDE_B + sub_col + j * DIM..], STRIDE_B, Layout::RowMajor);
                        acc[i][j] = tl.mma(a, b, acc[i][j]);
                    });
                });
                kk += DIM;
            }
            workgroup_barrier();
            k0 += DEPTH;
        }
        hydroplane::unroll!(SUB_TILES, i, {
            hydroplane::unroll!(SUB_TILES, j, {
                let at = (row0 + sub_row + i * DIM) * GEMM + col0 + sub_col + j * DIM;
                acc[i][j].store(&mut self.c[at..], GEMM, Layout::RowMajor);
            });
        });
    }
}

#[cfg(target_arch = "spirv")]
pub mod entry {
    use super::{BLOCK, DEPTH, Gemm, GemmPipelined, GemmStaged, GemmTile, Probe, STRIDE_A, STRIDE_B};
    use hydroplane::f16;
    use spirv_std::glam::UVec3;
    use spirv_std::spirv;

    /// A [`Gemm`] entry point: one subgroup per `BM×BN` block of `16×16` output tiles, at
    /// workgroup `(x, y) = (tile_col, tile_row)`.
    /// A [`Gemm`] entry point: `SIDE×SIDE` subgroups per workgroup, each a `BM×BN` block of
    /// `16×16` output tiles, the k loop unrolled `KU` times. On an RTX 5090, 4×4 with `KU = 2`
    /// and one subgroup per workgroup does best (~172 TFLOP/s). `KU = 4` spills, and so do larger
    /// workgroups, where the driver lowers the per-thread register budget.
    macro_rules! gemm_entry {
        ($name:ident, $bm:literal, $bn:literal, $ku:literal, $side:literal, $threads:literal) => {
            #[spirv(compute(threads($threads)))]
            pub fn $name(
                #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
                #[spirv(workgroup_id)] group: UVec3,
                #[spirv(subgroup_id)] subgroup: u32,
            ) {
                let (side, subgroup) = ($side, subgroup as usize);
                let kernel = Gemm::<$bm, $bn, $ku> {
                    a,
                    b,
                    c,
                    tile_row: group.y as usize * side + subgroup / side,
                    tile_col: group.x as usize * side + subgroup % side,
                };
                hydroplane::dispatch_matrix_subgroup(kernel);
            }
        };
    }

    gemm_entry!(gemm_1x1, 1, 1, 1, 1, 32);
    gemm_entry!(gemm_2x2, 2, 2, 1, 1, 32);
    gemm_entry!(gemm_4x4, 4, 4, 1, 1, 32);
    gemm_entry!(gemm_4x4_k2, 4, 4, 2, 1, 32);

    macro_rules! gemm_pipelined_entry {
        ($name:ident, $bm:literal, $bn:literal) => {
            #[spirv(compute(threads(32)))]
            pub fn $name(
                #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
                #[spirv(workgroup_id)] group: UVec3,
            ) {
                let kernel = GemmPipelined::<$bm, $bn> { a, b, c, tile_row: group.y as usize, tile_col: group.x as usize };
                hydroplane::dispatch_matrix_subgroup(kernel);
            }
        };
    }

    gemm_pipelined_entry!(gemm_4x4_pipelined, 4, 4);

    macro_rules! gemm_pipelined_multi {
        ($name:ident, $threads:literal, $per_row:literal) => {
            /// [`GemmPipelined`] 4×4 with `$threads / 32` subgroups per workgroup, laid out
            /// `$per_row` wide. On an RTX 5090, 4 subgroups as 2×2 is fastest (~188 TFLOP/s, ~90%
            /// of the f16→f32 tensor peak): neighbouring blocks share A/B through L1. At 8 or
            /// more, a workgroup no longer fits the register file at these kernels' ~200
            /// registers per thread, and the driver spills.
            #[spirv(compute(threads($threads)))]
            pub fn $name(
                #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
                #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
                #[spirv(workgroup_id)] group: UVec3,
                #[spirv(subgroup_id)] subgroup: u32,
            ) {
                let (per_row, rows) = ($per_row, $threads / 32 / $per_row);
                let subgroup = subgroup as usize;
                let kernel = GemmPipelined::<4, 4> {
                    a,
                    b,
                    c,
                    tile_row: group.y as usize * rows + subgroup / per_row,
                    tile_col: group.x as usize * per_row + subgroup % per_row,
                };
                hydroplane::dispatch_matrix_subgroup(kernel);
            }
        };
    }

    gemm_pipelined_multi!(gemm_4x4_pipelined_wg4, 128, 2);


    /// [`GemmStaged`]: one workgroup of four 32-wide subgroups per `128×128` output block.
    #[spirv(compute(threads(128)))]
    pub fn gemm_staged(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
        #[spirv(workgroup_id)] group: UVec3,
        #[spirv(local_invocation_index)] thread: u32,
        #[spirv(subgroup_id)] quarter: u32,
        #[spirv(workgroup)] shared_a: &mut [f16; BLOCK * STRIDE_A],
        #[spirv(workgroup)] shared_b: &mut [f16; DEPTH * STRIDE_B],
    ) {
        let kernel = GemmStaged {
            a,
            b,
            c,
            shared_a,
            shared_b,
            block_row: group.y as usize,
            block_col: group.x as usize,
            quarter: quarter as usize,
            thread: thread as usize,
            threads: 128,
        };
        hydroplane::dispatch_matrix_subgroup(kernel);
    }

    /// Reference without tiles: one invocation per output element, `f16` widened per load.
    #[spirv(compute(threads(16, 16)))]
    pub fn gemm_naive(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
        #[spirv(global_invocation_id)] id: UVec3,
    ) {
        use super::GEMM;
        let (row, col) = (id.y as usize, id.x as usize);
        let mut sum = 0.0f32;
        let mut k = 0;
        while k < GEMM {
            let x = spirv_std::float::f16_to_f32(a[row * GEMM + k].to_bits() as u32);
            let y = spirv_std::float::f16_to_f32(b[k * GEMM + col].to_bits() as u32);
            sum += x * y;
            k += 1;
        }
        c[row * GEMM + col] = sum;
    }

    /// [`GemmTile`] with one `64×64` tile per subgroup.
    #[spirv(compute(threads(32)))]
    pub fn gemm_tile64(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
        #[spirv(workgroup_id)] group: UVec3,
    ) {
        let kernel = GemmTile::<64> { a, b, c, tile_row: group.y as usize, tile_col: group.x as usize };
        hydroplane::dispatch_matrix_subgroup(kernel);
    }

    /// The array-tile 64×64 GEMM written without hydroplane: each lane owns 2×4 elements of each
    /// of the 4×4 output tiles.
    #[spirv(compute(threads(32)))]
    pub fn gemm_raw(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
        #[spirv(workgroup_id)] group: UVec3,
        #[spirv(subgroup_local_invocation_id)] lane: u32,
    ) {
        use super::GEMM;
        let l = lane as usize;
        let (row0, col0) = (group.y as usize * 64, group.x as usize * 64);
        let (lr, lc) = (l / 4, l % 4);
        let mut acc = [[0.0f32; 16]; 8];
        let mut k = 0;
        while k < GEMM {
            let mut y = [0.0f32; 16];
            hydroplane::unroll!(4, t, {
                hydroplane::unroll!(4, q, {
                    y[t * 4 + q] = spirv_std::float::f16_to_f32(b[k * GEMM + col0 + t * 16 + lc + q * 4].to_bits() as u32);
                });
            });
            hydroplane::unroll!(4, t, {
                hydroplane::unroll!(2, p, {
                    let x = spirv_std::float::f16_to_f32(a[(row0 + t * 16 + lr + p * 8) * GEMM + k].to_bits() as u32);
                    hydroplane::unroll!(4, u, {
                        hydroplane::unroll!(4, q, {
                            acc[t * 2 + p][u * 4 + q] = x * y[u * 4 + q] + acc[t * 2 + p][u * 4 + q];
                        });
                    });
                });
            });
            k += 1;
        }
        hydroplane::unroll!(4, t, {
            hydroplane::unroll!(2, p, {
                hydroplane::unroll!(4, u, {
                    hydroplane::unroll!(4, q, {
                        c[(row0 + t * 16 + lr + p * 8) * GEMM + col0 + u * 16 + lc + q * 4] = acc[t * 2 + p][u * 4 + q];
                    });
                });
            });
        });
    }

    /// One subgroup runs one [`Probe`].
    #[spirv(compute(threads(32)))]
    #[allow(clippy::too_many_arguments)]
    pub fn probe(
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f16],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &[f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] bias: &[f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] out: &mut [f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 5)] row_max: &mut [f32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 6)] col_sum: &mut [f32],
    ) {
        let kernel = Probe { a, b, c, bias, out, row_max, col_sum };
        hydroplane::dispatch_matrix_subgroup(kernel);
    }
}
