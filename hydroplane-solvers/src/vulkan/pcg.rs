//! Conjugate gradient on a Vulkan device in working precision `T`, preconditioned by an `f32`
//! Cholesky factor, an `f32` algebraic multigrid cycle or Jacobi (kernels in `kernels/src/pcg.rs`).
//!
//! An `f32` factorization costs a third of an `f64` one on a GPU whose double-precision unit is
//! narrow, but on an ill-conditioned matrix its solve alone leaves a large error (15 % on a
//! welded plate's tangent). Preconditioning `f64` conjugate gradient with it recovers `f64`
//! accuracy in a few iterations, each one `f32` solve and one `f64` matrix-vector product. On a
//! device without `f64` the iteration runs in [`Ef64`](crate::Ef64) instead: on an M5 Pro 18–21 %
//! faster than keeping it on the CPU, to ~14 times `f64`'s residual floor (48 bits against 53).
//! Every dispatch of the start, of a batch of iterations and of the readout is recorded once; a
//! solve submits the start, then batches until the device's scalars say done.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use rayon::prelude::*;

use super::amg::AmgCycle;
use super::cholesky::{Recording, Stamper, VulkanCholesky, barrier, compute_barrier, dispatch, record};
use super::{Buf, Context, DeviceReal, Pipe, device_error};
use crate::cholesky::{Analysis, Error, PcgReport};
use crate::iterative::{Amg, AmgOptions};

/// Iterations recorded per submission; the scalars are read back between submissions. Once
/// the residual's rate says fewer remain, single iterations are submitted instead, since every
/// iteration past convergence still pays for its preconditioner solve.
const BATCH: usize = 4;
/// Matrix rows per workgroup of the matrix-vector product: a warp per row for long rows (FEM
/// tangents), an invocation per row for rows shorter than [`SHORT_ROWS`] entries on average.
const SPMV_ROWS: usize = 8;
const SHORT_ROWS: usize = 16;

/// Layout of the scalar buffer (`kernels/src/pcg.rs`).
mod sc {
    pub const RR: usize = 3;
    pub const TOL2: usize = 4;
    pub const DONE: usize = 5;
    pub const ITERS: usize = 6;
    pub const MAX_ITERS: usize = 7;
    pub const RR0: usize = 8;
    pub const LEN: usize = 16;
}

// One per solver, and `VulkanPcg` sits beside `VulkanCholesky` in `auto`'s backends either way.
#[allow(clippy::large_enum_variant)]
enum Preconditioner {
    Cholesky(VulkanCholesky<f32>),
    Jacobi,
    /// Set up from the values, so each factorization builds the hierarchy, and the iteration's
    /// recordings with it, anew; Jacobi until the first.
    Amg { cycle: Option<AmgCycle>, pattern: Arc<(Vec<usize>, Vec<usize>)>, options: AmgOptions },
}

pub struct VulkanPcg<T: DeviceReal> {
    pub options: crate::cholesky::PcgOptions,
    ctx: Arc<Context>,
    precond: Preconditioner,
    /// Each column's diagonal entry in the values, for Jacobi.
    diag_pos: Vec<Option<usize>>,
    n: usize,
    a_len: usize,
    module: vk::ShaderModule,
    pipes: Vec<Pipe>,
    buffers: Vec<Buf>,
    stage_val: Buf,
    stage_val_ptr: *mut T,
    stage_b: Buf,
    stage_b_ptr: *mut T,
    stage_sc: Buf,
    stage_sc_ptr: *mut T,
    last_solve_ms: f64,
    upload_cmd: Recording,
    start_cmd: Recording,
    batch_cmd: Recording,
    step_cmd: Recording,
    finish_cmd: Recording,
    _real: PhantomData<T>,
}

// The mapped staging pointers are only touched through `&mut self`.
unsafe impl<T: DeviceReal> Send for VulkanPcg<T> {}

impl<T: DeviceReal> VulkanPcg<T> {
    pub fn new(analysis: &Analysis) -> Result<Self, Error> {
        Self::with_options(analysis, super::Options::default())
    }

    pub fn with_options(analysis: &Analysis, options: super::Options) -> Result<Self, Error> {
        let ctx = options.context()?;
        let chol = VulkanCholesky::<f32>::with_options(analysis, super::Options { context: Some(ctx.clone()), ..options.clone() })?;
        Self::build(ctx, analysis.col_ptr(), analysis.row_idx(), Preconditioner::Cholesky(chol), options.pcg)
    }

    /// Jacobi-preconditioned conjugate gradient, with no factorization at all: for a system
    /// whose factor would not fit, or whose diagonal dominates (mass-dominated time steps, heat
    /// conduction). The lower triangle's pattern in compressed sparse columns.
    pub fn jacobi(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::jacobi_with_options(col_ptr, row_idx, super::Options::default())
    }

    pub fn jacobi_with_options(col_ptr: &[usize], row_idx: &[usize], options: super::Options) -> Result<Self, Error> {
        let ctx = options.context()?;
        Self::build(ctx, col_ptr, row_idx, Preconditioner::Jacobi, options.pcg)
    }

    /// Conjugate gradient preconditioned by a V-cycle of smoothed-aggregation multigrid
    /// ([`Amg`]) in `f32`: for the elliptic problems (heat conduction, elasticity) too large to
    /// factor, where Jacobi's iteration count grows with the mesh and AMG's stays nearly flat.
    /// The hierarchy is set up on the CPU at each [`Self::factorize`]; the lower triangle's
    /// pattern in compressed sparse columns.
    pub fn amg(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::amg_with_options(col_ptr, row_idx, AmgOptions::default(), super::Options::default())
    }

    pub fn amg_with_options(col_ptr: &[usize], row_idx: &[usize], amg: AmgOptions, options: super::Options) -> Result<Self, Error> {
        let ctx = options.context()?;
        let pattern = Arc::new((col_ptr.to_vec(), row_idx.to_vec()));
        Self::build(ctx, col_ptr, row_idx, Preconditioner::Amg { cycle: None, pattern, options: amg }, options.pcg)
    }

    fn build(ctx: Arc<Context>, col_ptr: &[usize], row_idx: &[usize], precond: Preconditioner, pcg: crate::cholesky::PcgOptions) -> Result<Self, Error> {
        let module = ctx.module::<T>()?;
        let entry = |name: &str| format!("pcg::pcg_{name}");
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let a_len = col_ptr[n];
        if row_idx.len() < a_len || row_idx[..a_len].iter().any(|&i| i >= n) {
            return Err(Error::Pattern);
        }
        let diag_pos: Vec<Option<usize>> = (0..n).map(|j| (col_ptr[j]..col_ptr[j + 1]).find(|&p| row_idx[p] == j)).collect();

        // Both triangles by rows, each entry pointing at its value in the lower triangle.
        let mut count = vec![0u32; n + 1];
        for j in 0..n {
            for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if i >= j {
                    count[i + 1] += 1;
                    if i != j {
                        count[j + 1] += 1;
                    }
                }
            }
        }
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let row_ptr = count.clone();
        let mut fill = count;
        let mut col = vec![0u32; row_ptr[n] as usize];
        let mut vidx = vec![0u32; row_ptr[n] as usize];
        for j in 0..n {
            for (off, &i) in row_idx[col_ptr[j]..col_ptr[j + 1]].iter().enumerate() {
                let p = col_ptr[j] + off;
                if i < j {
                    continue;
                }
                for (r, c) in [(i, j), (j, i)].into_iter().take(if i == j { 1 } else { 2 }) {
                    let at = fill[r] as usize;
                    col[at] = c as u32;
                    vidx[at] = p as u32;
                    fill[r] += 1;
                }
            }
        }

        let real = size_of::<T>() as u64;
        let reals = |len: usize| ctx.device_buffer(len.max(1) as u64 * real);
        let mut batch = ctx.batch()?;
        let row_ptr_b = batch.upload(&row_ptr)?;
        let col_b = batch.upload(&col)?;
        let vidx_b = batch.upload(&vidx)?;
        batch.send()?;
        let val = reals(a_len)?;
        let (b, x, r, p, q, z) = (reals(n)?, reals(n)?, reals(n)?, reals(n)?, reals(n)?, reals(n)?);
        let vec_groups = n.div_ceil(256);
        let short = (row_ptr[n] as usize) < SHORT_ROWS * n;
        let spmv_groups = if short { vec_groups } else { n.div_ceil(SPMV_ROWS) };
        let partial_pq = reals(spmv_groups)?;
        let partial_rr = reals(vec_groups)?;
        let partial_rz = reals(vec_groups)?;
        let scalars = reals(sc::LEN)?;
        let dinv = reals(n)?;
        let stage_val = ctx.host_buffer((a_len + n) as u64 * real)?;
        let stage_b = ctx.host_buffer(n as u64 * real)?;
        let stage_sc = ctx.host_buffer(sc::LEN as u64 * real)?;
        let (r32, z32, jacobi_r32) = match &precond {
            Preconditioner::Cholesky(c) => (c.solve_buffers().0, c.solve_buffers().1, None),
            Preconditioner::Amg { cycle: Some(c), .. } => (c.buffers().0, c.buffers().1, None),
            _ => {
                let r32 = ctx.device_buffer(n.max(1) as u64 * 4)?;
                (r32, r32, Some(r32))
            }
        };

        let pipes = vec![
            ctx.pipeline(module, &entry(if short { "spmv_rows" } else { "spmv" }), &[row_ptr_b, col_b, vidx_b, val, p, q, partial_pq])?,
            ctx.pipeline(module, &entry("alpha"), &[partial_pq, scalars])?,
            ctx.pipeline(module, &entry("update"), &[scalars, p, q, x, r, r32, partial_rr])?,
            ctx.pipeline(module, &entry("zr"), &[z32, r, z, partial_rz])?,
            ctx.pipeline(module, &entry("beta"), &[partial_rz, partial_rr, scalars])?,
            ctx.pipeline(module, &entry("direction"), &[scalars, z, p])?,
            ctx.pipeline(module, &entry("start"), &[b, x, r, r32, partial_rr])?,
            ctx.pipeline(module, &entry("start_reduce"), &[partial_rz, partial_rr, scalars])?,
            ctx.pipeline(module, &entry("jacobi"), &[dinv, r, z, partial_rz])?,
        ];
        let [spmv, alpha, update, zr, beta, direction, start, start_reduce, jacobi] = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| &pipes[i]);
        let d = &ctx.dev;
        let nn = n as u32;
        let vector = |cmd, pipe: &Pipe| unsafe {
            dispatch(d, cmd, pipe, [0, nn], vec_groups, 0.0);
            compute_barrier(d, cmd);
        };
        let single = |cmd, pipe: &Pipe| unsafe {
            dispatch(d, cmd, pipe, [0, 1], 1, 0.0);
            compute_barrier(d, cmd);
        };
        let precondition = |cmd| match &precond {
            Preconditioner::Cholesky(c) => {
                c.record_solve(cmd);
                vector(cmd, zr);
            }
            Preconditioner::Amg { cycle: Some(c), .. } => {
                c.record(cmd);
                vector(cmd, zr);
            }
            _ => vector(cmd, jacobi),
        };
        let upload_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            d.cmd_copy_buffer(cmd, stage_val.buf, val.buf, &[vk::BufferCopy::default().size(a_len.max(1) as u64 * real)]);
            d.cmd_copy_buffer(cmd, stage_val.buf, dinv.buf, &[vk::BufferCopy::default().src_offset(a_len as u64 * real).size(n.max(1) as u64 * real)]);
        })?;
        let start_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            d.cmd_copy_buffer(cmd, stage_b.buf, b.buf, &[vk::BufferCopy::default().size(n as u64 * real)]);
            d.cmd_copy_buffer(cmd, stage_sc.buf, scalars.buf, &[vk::BufferCopy::default().size(sc::LEN as u64 * real)]);
            d.cmd_fill_buffer(cmd, p.buf, 0, vk::WHOLE_SIZE, 0);
            barrier(d, cmd, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            vector(cmd, start);
            precondition(cmd);
            single(cmd, start_reduce);
            vector(cmd, direction);
        })?;
        let iterations = |count: usize| {
            record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
                for _ in 0..count {
                    dispatch(d, cmd, spmv, [0, nn], spmv_groups, 0.0);
                    compute_barrier(d, cmd);
                    single(cmd, alpha);
                    vector(cmd, update);
                    precondition(cmd);
                    single(cmd, beta);
                    vector(cmd, direction);
                }
                barrier(d, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ);
                d.cmd_copy_buffer(cmd, scalars.buf, stage_sc.buf, &[vk::BufferCopy::default().size(sc::LEN as u64 * real)]);
            })
        };
        let batch_cmd = iterations(BATCH)?;
        let step_cmd = iterations(1)?;
        let finish_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            d.cmd_copy_buffer(cmd, x.buf, stage_b.buf, &[vk::BufferCopy::default().size(n as u64 * real)]);
            d.cmd_copy_buffer(cmd, scalars.buf, stage_sc.buf, &[vk::BufferCopy::default().size(sc::LEN as u64 * real)]);
        })?;
        let map = |b: Buf| unsafe { ctx.dev.map_memory(b.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(device_error) };
        let (stage_val_ptr, stage_b_ptr, stage_sc_ptr) = (map(stage_val)? as *mut T, map(stage_b)? as *mut T, map(stage_sc)? as *mut T);
        Ok(Self {
            options: pcg,
            ctx,
            precond,
            diag_pos,
            n,
            a_len,
            module,
            pipes,
            buffers: [row_ptr_b, col_b, vidx_b, val, b, x, r, p, q, z, partial_pq, partial_rr, partial_rz, scalars, dinv].into_iter().chain(jacobi_r32).collect(),
            stage_val,
            stage_val_ptr,
            stage_b,
            stage_b_ptr,
            stage_sc,
            stage_sc_ptr,
            last_solve_ms: 0.0,
            upload_cmd,
            start_cmd,
            batch_cmd,
            step_cmd,
            finish_cmd,
            _real: PhantomData,
        })
    }

    /// Device time of the last solve (ms), the batches' readbacks included.
    pub fn last_solve_ms(&self) -> f64 {
        self.last_solve_ms
    }

    /// The preconditioner's factor (`None` for Jacobi).
    pub fn cholesky(&self) -> Option<&VulkanCholesky<f32>> {
        match &self.precond {
            Preconditioner::Cholesky(c) => Some(c),
            _ => None,
        }
    }

    /// Factors the preconditioner in `f32` (sets up the multigrid hierarchy, or inverts the
    /// diagonal, for Jacobi) and keeps the matrix in `T` for the iteration; `values` are the
    /// lower triangle in the pattern's order.
    pub fn factorize(&mut self, values: &[f64]) -> Result<super::Info, Error> {
        if let Preconditioner::Amg { pattern, options, .. } = &self.precond {
            let (pattern, options) = (pattern.clone(), options.clone());
            let amg = Amg::with_options(&pattern.0, &pattern.1, values, options.clone())?;
            let cycle = AmgCycle::new(self.ctx.clone(), self.module, &amg)?;
            *self = Self::build(self.ctx.clone(), &pattern.0, &pattern.1, Preconditioner::Amg { cycle: Some(cycle), pattern: pattern.clone(), options }, self.options)?;
        }
        self.set_values(values)?;
        match &mut self.precond {
            Preconditioner::Cholesky(c) => c.factorize(values),
            _ => Ok(super::Info::default()),
        }
    }

    fn factor_mut(&mut self) -> Result<&mut VulkanCholesky<f32>, Error> {
        match &mut self.precond {
            Preconditioner::Cholesky(c) => Ok(c),
            _ => Err(Error::Device("only a Cholesky-preconditioned iteration has a factor to load".into())),
        }
    }

    /// Keeps the matrix (`values`) for the iteration and takes the preconditioner's `f32` factor
    /// from elsewhere (in the analysis's layout, e.g. [`crate::sme::SmeCholesky`]'s) instead of
    /// factoring here.
    pub fn load_preconditioner(&mut self, values: &[f64], factor: &[f32]) -> Result<(), Error> {
        self.set_values(values)?;
        self.factor_mut()?.load_factor(factor).map(drop)
    }

    /// As [`Self::load_preconditioner`], with the factor's diagonal-block inverses made
    /// elsewhere too ([`VulkanCholesky::load_factor_and_inverse`]).
    pub fn load_preconditioner_and_inverse(&mut self, values: &[f64], factor: &[f32], inverse: &[f32]) -> Result<(), Error> {
        self.set_values(values)?;
        self.factor_mut()?.load_factor_and_inverse(factor, inverse)
    }

    /// Replaces the matrix the iteration solves with and keeps the preconditioner: a factor of
    /// an earlier matrix of the same pattern, as in modified Newton, still converges in a few
    /// iterations, to the new matrix's solution.
    pub fn set_values(&mut self, values: &[f64]) -> Result<(), Error> {
        assert_eq!(values.len(), self.a_len);
        let stage = unsafe { std::slice::from_raw_parts_mut(self.stage_val_ptr, self.a_len + self.n) };
        let (vals, dinv) = stage.split_at_mut(self.a_len);
        vals.par_iter_mut().zip(values).with_min_len(1 << 14).for_each(|(s, &v)| *s = T::from_f64(v));
        for (d, pos) in dinv.iter_mut().zip(&self.diag_pos) {
            let a = pos.map_or(0.0, |p| values[p]);
            *d = T::from_f64(if a != 0.0 && a.is_finite() { 1.0 / a } else { 1.0 });
        }
        self.ctx.submit_wait(self.upload_cmd.cmd)
    }

    /// `rhs ← A⁻¹ rhs` to [`Self::options`]' tolerance (`‖b − A x‖ ≤ tol ‖b‖`).
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<PcgReport, Error> {
        self.solve_to(rhs, self.options.tolerance, self.options.max_iterations)
    }

    pub(crate) fn solve_to(&mut self, rhs: &mut [f64], relative_tolerance: f64, max_iterations: usize) -> Result<PcgReport, Error> {
        assert_eq!(rhs.len(), self.n);
        let stage_b = unsafe { std::slice::from_raw_parts_mut(self.stage_b_ptr, self.n) };
        let scalars = unsafe { std::slice::from_raw_parts_mut(self.stage_sc_ptr, sc::LEN) };
        for (s, &v) in stage_b.iter_mut().zip(rhs.iter()) {
            *s = T::from_f64(v);
        }
        scalars.fill(T::default());
        scalars[sc::TOL2] = T::from_f64(relative_tolerance * relative_tolerance);
        scalars[sc::MAX_ITERS] = T::from_f64(max_iterations as f64);
        let get = |k: usize| scalars[k].to_f64();
        let mut ms = self.run(&self.start_cmd)?;
        let mut remaining = f64::INFINITY;
        loop {
            ms += self.run(if remaining < BATCH as f64 { &self.step_cmd } else { &self.batch_cmd })?;
            if get(sc::DONE) != 0.0 {
                break;
            }
            let rr = get(sc::RR);
            let rate = (rr / get(sc::RR0)).ln() / get(sc::ITERS);
            remaining = if rate < 0.0 { (get(sc::TOL2) / rr).ln() / rate } else { f64::INFINITY };
        }
        ms += self.run(&self.finish_cmd)?;
        self.last_solve_ms = ms;
        for (r, s) in rhs.iter_mut().zip(stage_b.iter()) {
            *r = s.to_f64();
        }
        let rr0 = get(sc::RR0);
        let relative_residual = if rr0 > 0.0 { (get(sc::RR) / rr0).sqrt() } else { 0.0 };
        Ok(PcgReport { iterations: get(sc::ITERS) as usize, relative_residual, converged: relative_residual <= relative_tolerance })
    }

    /// Device time of one iteration, from a recorded batch (which runs every dispatch whatever
    /// the iteration's state): what calibration measures.
    pub(crate) fn iteration_ms(&self) -> Result<f64, Error> {
        Ok(self.run(&self.batch_cmd)? / BATCH as f64)
    }

    fn run(&self, r: &Recording) -> Result<f64, Error> {
        self.ctx.submit_wait(r.cmd)?;
        let mut ts = [0u64; 2];
        unsafe {
            self.ctx
                .dev
                .get_query_pool_results(r.queries, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                .map_err(device_error)?;
        }
        Ok(ts[1].wrapping_sub(ts[0]) as f64 * self.ctx.period_ns * 1e-6)
    }
}

impl<T: DeviceReal> Drop for VulkanPcg<T> {
    fn drop(&mut self) {
        let ctx = &self.ctx;
        let _queue = ctx.lock();
        unsafe {
            let _ = ctx.dev.device_wait_idle();
            for b in [self.stage_val, self.stage_b, self.stage_sc] {
                ctx.dev.unmap_memory(b.mem);
            }
            for r in [&self.upload_cmd, &self.start_cmd, &self.batch_cmd, &self.step_cmd, &self.finish_cmd] {
                ctx.dev.free_command_buffers(ctx.cpool, &[r.cmd]);
                ctx.dev.destroy_query_pool(r.queries, None);
            }
            for p in &self.pipes {
                ctx.destroy_pipeline(p);
            }
            ctx.dev.destroy_shader_module(self.module, None);
        }
        for &b in self.buffers.iter().chain([&self.stage_val, &self.stage_b, &self.stage_sc]) {
            ctx.free(b);
        }
    }
}
