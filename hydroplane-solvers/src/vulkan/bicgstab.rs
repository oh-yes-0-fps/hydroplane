//! BiCGStab on a Vulkan device in working precision `T`, right-preconditioned by Jacobi (kernels in
//! `kernels/src/bicgstab.rs`): the nonsymmetric counterpart of [`super::VulkanPcg::jacobi`].
//! Every dispatch of the start, of a batch of iterations and of the readout is recorded once; a
//! solve submits the start, then batches until the device's scalars say done.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use rayon::prelude::*;

use super::cholesky::{Recording, Stamper, barrier, compute_barrier, dispatch, record};
use super::{Buf, Context, DeviceReal, Pipe, device_error};
use crate::cholesky::{Error, PcgOptions, PcgReport};

/// Iterations recorded per submission.
const BATCH: usize = 4;
/// Rows per workgroup of the warp-per-row matrix-vector product, and the average row length
/// below which an invocation per row is used instead.
const WARP_ROWS: usize = 8;
const SHORT_ROWS: usize = 16;

mod sc {
    pub const RR: usize = 3;
    pub const TOL2: usize = 4;
    pub const DONE: usize = 5;
    pub const ITERS: usize = 6;
    pub const MAX_ITERS: usize = 7;
    pub const RR0: usize = 8;
    pub const LEN: usize = 16;
}

/// Jacobi-preconditioned BiCGStab for a general square sparse matrix on the GPU.
pub struct VulkanBiCgStab<T: DeviceReal> {
    pub options: PcgOptions,
    ctx: Arc<Context>,
    n: usize,
    a_len: usize,
    diag_pos: Vec<Option<usize>>,
    module: vk::ShaderModule,
    pipes: Vec<Pipe>,
    buffers: Vec<Buf>,
    stage_val: Buf,
    stage_val_ptr: *mut T,
    stage_b: Buf,
    stage_b_ptr: *mut T,
    stage_sc: Buf,
    stage_sc_ptr: *mut T,
    upload_cmd: Recording,
    start_cmd: Recording,
    batch_cmd: Recording,
    finish_cmd: Recording,
    last_solve_ms: f64,
    _real: PhantomData<T>,
}

// The mapped staging pointers are only touched through `&mut self`.
unsafe impl<T: DeviceReal> Send for VulkanBiCgStab<T> {}

impl<T: DeviceReal> VulkanBiCgStab<T> {
    /// The pattern of the whole matrix in compressed sparse columns.
    pub fn new(col_ptr: &[usize], row_idx: &[usize]) -> Result<Self, Error> {
        Self::with_options(col_ptr, row_idx, super::Options::default())
    }

    pub fn with_options(col_ptr: &[usize], row_idx: &[usize], options: super::Options) -> Result<Self, Error> {
        let ctx = options.context()?;
        let module = ctx.module::<T>()?;
        let entry = |name: &str| format!("bicgstab::bicg_{name}");
        let n = col_ptr.len().checked_sub(1).ok_or(Error::Pattern)?;
        let a_len = col_ptr[n];
        if row_idx.len() < a_len || row_idx[..a_len].iter().any(|&i| i >= n) {
            return Err(Error::Pattern);
        }
        let diag_pos: Vec<Option<usize>> = (0..n).map(|j| (col_ptr[j]..col_ptr[j + 1]).find(|&p| row_idx[p] == j)).collect();
        let mut row_ptr = vec![0u32; n + 1];
        for &i in &row_idx[..a_len] {
            row_ptr[i + 1] += 1;
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }
        let mut fill = row_ptr.clone();
        let mut col = vec![0u32; a_len];
        let mut vidx = vec![0u32; a_len];
        for j in 0..n {
            for (p, &i) in row_idx.iter().enumerate().take(col_ptr[j + 1]).skip(col_ptr[j]) {
                let at = fill[i] as usize;
                col[at] = j as u32;
                vidx[at] = p as u32;
                fill[i] += 1;
            }
        }

        let real = size_of::<T>() as u64;
        let reals = |len: usize| ctx.device_buffer(len.max(1) as u64 * real);
        let row_ptr_b = ctx.upload(&row_ptr)?;
        let col_b = ctx.upload(&col)?;
        let vidx_b = ctx.upload(&vidx)?;
        let val = reals(a_len)?;
        let dinv = reals(n)?;
        let [b, x, r, rhat, p, v, s, t, z] = [(); 9].map(|_| reals(n));
        let (b, x, r, rhat, p, v, s, t, z) = (b?, x?, r?, rhat?, p?, v?, s?, t?, z?);
        let short = a_len < SHORT_ROWS * n;
        let vec_groups = n.div_ceil(256);
        let spmv_groups = if short { vec_groups } else { n.div_ceil(WARP_ROWS) };
        let (pm1, pm2) = (reals(spmv_groups)?, reals(spmv_groups)?);
        let (pv1, pv2) = (reals(vec_groups)?, reals(vec_groups)?);
        let scalars = reals(sc::LEN)?;
        let stage_val = ctx.host_buffer((a_len + n) as u64 * real)?;
        let stage_b = ctx.host_buffer(n as u64 * real)?;
        let stage_sc = ctx.host_buffer(sc::LEN as u64 * real)?;

        let spmv = if short { "spmv_rows" } else { "spmv_warp" };
        let spmv2 = if short { "spmv2_rows" } else { "spmv2_warp" };
        let pipes = vec![
            ctx.pipeline(module, &entry("start"), &[b, x, r, rhat, p, pv1])?,
            ctx.pipeline(module, &entry("start_reduce"), &[pv1, scalars])?,
            ctx.pipeline(module, &entry("jacobi"), &[dinv, p, z, scalars])?,
            ctx.pipeline(module, &entry(spmv), &[row_ptr_b, col_b, vidx_b, val, z, v, rhat, pm1, scalars])?,
            ctx.pipeline(module, &entry("alpha"), &[pm1, scalars])?,
            ctx.pipeline(module, &entry("s"), &[scalars, r, v, z, s, x, pv1])?,
            ctx.pipeline(module, &entry("s_reduce"), &[pv1, scalars])?,
            ctx.pipeline(module, &entry("jacobi"), &[dinv, s, z, scalars])?,
            ctx.pipeline(module, &entry(spmv2), &[row_ptr_b, col_b, vidx_b, val, z, t, s, pm1, pm2, scalars])?,
            ctx.pipeline(module, &entry("omega"), &[pm1, pm2, scalars])?,
            ctx.pipeline(module, &entry("r"), &[scalars, s, t, z, rhat, x, r, pv1, pv2])?,
            ctx.pipeline(module, &entry("beta"), &[pv1, pv2, scalars])?,
            ctx.pipeline(module, &entry("p"), &[scalars, r, v, p])?,
        ];
        let d = &ctx.dev;
        let nn = n as u32;
        let step = |cmd, i: usize| unsafe {
            let groups = match i {
                1 | 4 | 6 | 9 | 11 => 1,
                3 | 8 => spmv_groups,
                _ => vec_groups,
            };
            dispatch(d, cmd, &pipes[i], [0, nn], groups, 0.0);
            compute_barrier(d, cmd);
        };
        let upload_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            d.cmd_copy_buffer(cmd, stage_val.buf, val.buf, &[vk::BufferCopy::default().size(a_len.max(1) as u64 * real)]);
            d.cmd_copy_buffer(cmd, stage_val.buf, dinv.buf, &[vk::BufferCopy::default().src_offset(a_len as u64 * real).size(n.max(1) as u64 * real)]);
        })?;
        let start_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            d.cmd_copy_buffer(cmd, stage_b.buf, b.buf, &[vk::BufferCopy::default().size(n as u64 * real)]);
            d.cmd_copy_buffer(cmd, stage_sc.buf, scalars.buf, &[vk::BufferCopy::default().size(sc::LEN as u64 * real)]);
            barrier(d, cmd, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            step(cmd, 0);
            step(cmd, 1);
        })?;
        let batch_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            for _ in 0..BATCH {
                for i in 2..pipes.len() {
                    step(cmd, i);
                }
            }
            barrier(d, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ);
            d.cmd_copy_buffer(cmd, scalars.buf, stage_sc.buf, &[vk::BufferCopy::default().size(sc::LEN as u64 * real)]);
        })?;
        let finish_cmd = record(&ctx, false, 0, |cmd, _: &mut Stamper| unsafe {
            d.cmd_copy_buffer(cmd, x.buf, stage_b.buf, &[vk::BufferCopy::default().size(n as u64 * real)]);
            d.cmd_copy_buffer(cmd, scalars.buf, stage_sc.buf, &[vk::BufferCopy::default().size(sc::LEN as u64 * real)]);
        })?;
        let map = |b: Buf| unsafe { ctx.dev.map_memory(b.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(device_error) };
        let (stage_val_ptr, stage_b_ptr, stage_sc_ptr) = (map(stage_val)? as *mut T, map(stage_b)? as *mut T, map(stage_sc)? as *mut T);
        Ok(Self {
            options: options.pcg,
            ctx: ctx.clone(),
            n,
            a_len,
            diag_pos,
            module,
            pipes,
            buffers: vec![row_ptr_b, col_b, vidx_b, val, dinv, b, x, r, rhat, p, v, s, t, z, pm1, pm2, pv1, pv2, scalars],
            stage_val,
            stage_val_ptr,
            stage_b,
            stage_b_ptr,
            stage_sc,
            stage_sc_ptr,
            upload_cmd,
            start_cmd,
            batch_cmd,
            finish_cmd,
            last_solve_ms: 0.0,
            _real: PhantomData,
        })
    }

    /// Takes the matrix (values in the pattern's order) and inverts its diagonal.
    pub fn factorize(&mut self, values: &[f64]) -> Result<(), Error> {
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

    /// `rhs ← A⁻¹ rhs` to [`Self::options`]' tolerance (`‖b − A x‖ ≤ tol ‖b‖`, as updated).
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<PcgReport, Error> {
        assert_eq!(rhs.len(), self.n);
        let stage_b = unsafe { std::slice::from_raw_parts_mut(self.stage_b_ptr, self.n) };
        let scalars = unsafe { std::slice::from_raw_parts_mut(self.stage_sc_ptr, sc::LEN) };
        for (s, &v) in stage_b.iter_mut().zip(rhs.iter()) {
            *s = T::from_f64(v);
        }
        scalars.fill(T::default());
        scalars[sc::TOL2] = T::from_f64(self.options.tolerance * self.options.tolerance);
        scalars[sc::MAX_ITERS] = T::from_f64(self.options.max_iterations as f64);
        let mut ms = self.run(&self.start_cmd)?;
        while scalars[sc::DONE].to_f64() == 0.0 {
            ms += self.run(&self.batch_cmd)?;
        }
        ms += self.run(&self.finish_cmd)?;
        self.last_solve_ms = ms;
        for (r, s) in rhs.iter_mut().zip(stage_b.iter()) {
            *r = s.to_f64();
        }
        let get = |k: usize| scalars[k].to_f64();
        let rr0 = get(sc::RR0);
        let relative_residual = if rr0 > 0.0 { (get(sc::RR) / rr0).sqrt() } else { 0.0 };
        Ok(PcgReport { iterations: get(sc::ITERS) as usize, relative_residual, converged: relative_residual <= self.options.tolerance })
    }

    /// Device time of the last solve (ms).
    pub fn last_solve_ms(&self) -> f64 {
        self.last_solve_ms
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

impl<T: DeviceReal> Drop for VulkanBiCgStab<T> {
    fn drop(&mut self) {
        let ctx = &self.ctx;
        let _queue = ctx.lock();
        unsafe {
            let _ = ctx.dev.device_wait_idle();
            for b in [self.stage_val, self.stage_b, self.stage_sc] {
                ctx.dev.unmap_memory(b.mem);
            }
            for r in [&self.upload_cmd, &self.start_cmd, &self.batch_cmd, &self.finish_cmd] {
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
