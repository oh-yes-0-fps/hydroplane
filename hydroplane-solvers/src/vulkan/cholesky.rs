//! Supernodal right-looking Cholesky on a Vulkan device, shaped after NVIDIA cuDSS (kernels in
//! `kernels/src/cholesky.rs`).
//!
//! The plan comes from the [`Analysis`]'s supernodes: each one's tile boundaries, the ancestor
//! column and row every entry of its update lands on, and its elimination-tree level (leaves 0, a
//! parent one above its highest child). Every dispatch of a factorization and of a solve is known
//! once the pattern is, so each is recorded into one command buffer at construction; a
//! factorization or solve only uploads values and resubmits. Updates reach ancestors through
//! atomic adds, so results can differ in the last bits from run to run, as in cuDSS's default mode.

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;
use rayon::prelude::*;

use super::{Buf, Context, DeviceReal, Pipe, Precision, device_error};
use crate::cholesky::{Analysis, Error};

const TILE: usize = 32;
/// `chol_step`'s marker for a diagonal-tile factorization item.
const POTRF: u32 = u32::MAX;
const FRONT_WORDS: usize = 7;
const GRID: usize = 65535;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kernel {
    Scatter,
    Step,
    Trsm,
    Schur,
    InverseDiag,
    InverseLeft,
    InverseRight,
    SolveGather,
    ForwardGemv,
    ForwardPattern,
    BackwardPattern,
    BackwardGemv,
    SolveScatter,
}

#[derive(Clone, Copy)]
struct Segment {
    kernel: Kernel,
    level: usize,
    first: u32,
    count: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub fronts: usize,
    pub levels: usize,
    pub factor_dispatches: usize,
    pub solve_dispatches: usize,
    pub items: usize,
    pub factor_len: usize,
    pub largest_front: usize,
    /// Floating-point operations of one factorization, the diagonal-block inverses included
    /// (multiply and add counted separately).
    pub flops: f64,
    /// Of [`Self::flops`], the supernodes' updates of their ancestors (`chol_schur`).
    pub schur_flops: f64,
}

/// Size of the blocks of a supernode's update of its ancestors (`chol_schur`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchurTile {
    /// 32 × 32 blocks, a 1 × 4 strip of outputs per invocation: the most workgroups, best where
    /// the update is limited by latency or a narrow double-precision unit.
    #[default]
    T32,
    /// 64 × 64 blocks, a 4 × 4 block per invocation: fewer shared-memory loads per multiply-add,
    /// best where the update is limited by throughput.
    T64,
}

/// Settings of the Vulkan solvers.
#[derive(Clone, Debug, Default)]
pub struct Options {
    /// The device: `None` (the default) is [`Context::shared`].
    pub context: Option<Arc<Context>>,
    /// `None` (the default) is the measured best for the precision ([`SchurTile::for_precision`]).
    pub schur_tile: Option<SchurTile>,
    /// [`super::VulkanPcg`]'s stopping rule.
    pub pcg: crate::cholesky::PcgOptions,
}

impl Options {
    pub(crate) fn context(&self) -> Result<Arc<Context>, Error> {
        match &self.context {
            Some(c) => Ok(c.clone()),
            None => Context::shared(),
        }
    }
}

impl SchurTile {
    /// The measured best for precision `T`: 64-wide Schur blocks in `f32` (6–8 % off the whole
    /// factorization on an M5 Pro and on an RTX 5090), 32-wide in `f64` (8 % faster on the 5090,
    /// whose narrow double-precision unit is busy either way and balances better over four times
    /// the workgroups).
    pub fn for_precision<T: DeviceReal>() -> Self {
        match T::PRECISION {
            Precision::F32 => SchurTile::T64,
            Precision::F64 | Precision::Ef64 => SchurTile::T32,
        }
    }
}

/// What a factorization had to do to finish.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Info {
    /// Some pivot fell below `τ |A_jj|` (`τ` = 100 ε of the working precision) and was raised to
    /// it: the factor is of a slightly perturbed matrix, fine as a preconditioner or for
    /// iterative refinement, not as an exact solve.
    pub perturbed: bool,
}

/// A recorded command buffer with timestamps around it and, when profiling, after every
/// dispatch.
pub(super) struct Recording {
    pub(super) cmd: vk::CommandBuffer,
    pub(super) queries: vk::QueryPool,
    stamps: Vec<(Kernel, usize)>,
}

/// Device time of one kernel at one elimination-tree level in the last run.
#[derive(Clone, Copy, Debug)]
pub struct KernelTime {
    pub kernel: Kernel,
    pub level: usize,
    pub dispatches: usize,
    pub ms: f64,
}

pub(super) struct Stamper<'a> {
    d: &'a ash::Device,
    queries: vk::QueryPool,
    on: bool,
    stamps: Vec<(Kernel, usize)>,
}

impl Stamper<'_> {
    unsafe fn stamp(&mut self, cmd: vk::CommandBuffer, kernel: Kernel, level: usize) {
        if self.on {
            unsafe { self.d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.queries, 2 + self.stamps.len() as u32) };
            self.stamps.push((kernel, level));
        }
    }
}

pub struct VulkanCholesky<T: DeviceReal> {
    ctx: Arc<Context>,
    n: usize,
    a_len: usize,
    factor_len: usize,
    diag_src: Vec<(usize, usize)>,
    lval: Buf,
    /// Where a factorization's values (`a_len`) and diagonal (`n`) are written, where a solve's
    /// right-hand side is written and its solution read: staging buffers copied by the recorded
    /// commands, or on a unified-memory device the kernels' own buffers, mapped.
    values_ptr: *mut T,
    diag_ptr: *mut T,
    rhs_ptr: *mut T,
    solution_ptr: *const T,
    stage_status: Buf,
    stage_status_ptr: *const u32,
    mapped: Vec<Buf>,
    buffers: Vec<Buf>,
    pipes: Vec<Pipe>,
    module: vk::ShaderModule,
    /// The right-hand side and the solution of the solve, in the matrix's own order.
    bvec: Buf,
    out: Buf,
    solve_segments: Vec<Segment>,
    tau: f32,
    factor_cmd: Recording,
    solve_cmd: Recording,
    /// The diagonal-block inversions alone, for a factor made elsewhere ([`Self::load_factor`]).
    inverse_cmd: Recording,
    /// Each supernode's block offset, rows `f`, columns `k` and first column.
    shapes: Vec<[usize; 4]>,
    dinv: Buf,
    /// The factor and inverted pivots as the host writes them for [`Self::load_factor`]: the
    /// kernels' own buffers, mapped, on a unified-memory device; else a staging buffer made on
    /// first use.
    load_target: Option<LoadTarget<T>>,
    xinv: Buf,
    xinv_len: usize,
    stats: Stats,
    last_factor_ms: f64,
    last_solve_ms: f64,
    _real: PhantomData<T>,
}

// The mapped staging pointers are only touched through `&mut self`.
unsafe impl<T: DeviceReal> Send for VulkanCholesky<T> {}

impl<T: DeviceReal> VulkanCholesky<T> {
    pub fn new(analysis: &Analysis) -> Result<Self, Error> {
        Self::with_options(analysis, Options::default())
    }

    pub fn with_options(analysis: &Analysis, options: Options) -> Result<Self, Error> {
        Self::build(analysis, options, None)
    }

    /// With `signs` (one per column, in the matrix's own order), the signed factorization
    /// `A = L S Lᵀ` of [`super::VulkanLdlt`].
    pub(super) fn build(analysis: &Analysis, options: Options, signs: Option<&[i8]>) -> Result<Self, Error> {
        let ctx = options.context()?;
        let schur_tile = options.schur_tile.unwrap_or_else(SchurTile::for_precision::<T>);
        let plan = Plan::new(analysis, schur_tile, signs.is_some())?;
        let module = ctx.module::<T>()?;
        let real = size_of::<T>() as u64;
        let n = analysis.dim();
        let a_len = analysis.row_idx().len();
        let factor_len = analysis.factor_len();

        let items = ctx.upload(&plan.items)?;
        let fronts = ctx.upload(&plan.fronts)?;
        let bounds = ctx.upload(&plan.bounds)?;
        let smap = ctx.upload(&plan.smap)?;
        let rmap = ctx.upload(&plan.rmap)?;
        let dst = ctx.upload(&plan.dst)?;
        let pat = ctx.upload(&plan.pat)?;
        let perm = ctx.upload(&plan.perm)?;
        let sign = match signs {
            Some(signs) => ctx.upload(&plan.perm.iter().map(|&old| (signs[old as usize] < 0) as u32).collect::<Vec<_>>())?,
            None => ctx.upload(&[0u32])?,
        };
        let unified = ctx.unified_memory();
        let io = |bytes: u64| if unified { ctx.shared_buffer(bytes) } else { ctx.device_buffer(bytes) };
        let aval = io(a_len as u64 * real)?;
        let lval = io((factor_len as u64 + 1) * real)?;
        let status = ctx.device_buffer(16)?;
        let diag = io(n as u64 * real)?;
        let dinv = io(n as u64 * real)?;
        let x = ctx.device_buffer(n as u64 * real)?;
        let xinv = io(plan.xinv_len as u64 * real)?;
        let ytmp = ctx.device_buffer(n as u64 * real)?;
        let bvec = io(n as u64 * real)?;
        let out = io(n as u64 * real)?;
        let stage_a = if unified { None } else { Some(ctx.host_buffer((a_len + n) as u64 * real)?) };
        let stage_status = ctx.host_buffer(16)?;
        let stage_b = if unified { None } else { Some(ctx.host_buffer(n as u64 * real)?) };

        let variant = |name: &str| if signs.is_some() { format!("cholesky::{name}_signed") } else { format!("cholesky::{name}") };
        let pipes = vec![
            ctx.pipeline(module, "cholesky::chol_scatter", &[dst, aval, lval])?,
            ctx.pipeline(module, &variant("chol_step"), &[items, fronts, bounds, lval, status, diag, dinv, sign])?,
            ctx.pipeline(module, &variant("chol_trsm"), &[items, fronts, bounds, lval, dinv, sign])?,
            ctx.pipeline(
                module,
                &variant(match schur_tile {
                    SchurTile::T32 => "chol_schur",
                    SchurTile::T64 => "chol_schur64",
                }),
                &[items, fronts, bounds, smap, rmap, lval, sign],
            )?,
            ctx.pipeline(module, "cholesky::chol_inv_diag", &[items, fronts, bounds, lval, dinv, xinv])?,
            ctx.pipeline(module, "cholesky::chol_inv_left", &[items, fronts, bounds, lval, xinv])?,
            ctx.pipeline(module, "cholesky::chol_inv_right", &[items, fronts, bounds, xinv])?,
            ctx.pipeline(module, "cholesky::solve_gather", &[perm, bvec, x])?,
            ctx.pipeline(module, "cholesky::solve_fgemv", &[items, fronts, xinv, x, ytmp])?,
            ctx.pipeline(module, "cholesky::solve_fpattern", &[items, fronts, pat, lval, ytmp, x])?,
            ctx.pipeline(module, &variant("solve_bpattern"), &[items, fronts, pat, lval, x, ytmp, sign])?,
            ctx.pipeline(module, "cholesky::solve_bgemv", &[items, fronts, xinv, ytmp, x])?,
            ctx.pipeline(module, "cholesky::solve_scatter", &[perm, x, out])?,
        ];
        let tau = (100.0 * T::EPSILON) as f32;
        let d = &ctx.dev;
        let profile = std::env::var_os("HYDROPLANE_SOLVERS_PROFILE").is_some();
        let factor_cmd = record(&ctx, profile, plan.factor.len() + 1, |cmd, st| unsafe {
            if let Some(stage_a) = stage_a {
                d.cmd_copy_buffer(cmd, stage_a.buf, aval.buf, &[vk::BufferCopy::default().size(a_len as u64 * real)]);
                d.cmd_copy_buffer(cmd, stage_a.buf, diag.buf, &[vk::BufferCopy::default().src_offset(a_len as u64 * real).size(n as u64 * real)]);
            }
            d.cmd_fill_buffer(cmd, lval.buf, 0, vk::WHOLE_SIZE, 0);
            d.cmd_fill_buffer(cmd, status.buf, 0, vk::WHOLE_SIZE, 0);
            barrier(d, cmd, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            dispatch(d, cmd, &pipes[Kernel::Scatter as usize], [0, a_len as u32], a_len.div_ceil(256), tau);
            compute_barrier(d, cmd);
            st.stamp(cmd, Kernel::Scatter, 0);
            for stage in &plan.factor {
                for seg in stage {
                    dispatch(d, cmd, &pipes[seg.kernel as usize], [seg.first, seg.count], seg.count as usize, tau);
                }
                compute_barrier(d, cmd);
                st.stamp(cmd, stage[0].kernel, stage[0].level);
            }
            barrier(d, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ);
            d.cmd_copy_buffer(cmd, status.buf, stage_status.buf, &[vk::BufferCopy::default().size(4)]);
        })?;
        let solve_cmd = record(&ctx, profile, plan.solve.len() + 2, |cmd, st| unsafe {
            if let Some(stage_b) = stage_b {
                d.cmd_copy_buffer(cmd, stage_b.buf, bvec.buf, &[vk::BufferCopy::default().size(n as u64 * real)]);
            }
            barrier(d, cmd, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            solve_body(d, cmd, &pipes, &plan.solve, n, tau, st);
            barrier(d, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ);
            match stage_b {
                Some(stage_b) => d.cmd_copy_buffer(cmd, out.buf, stage_b.buf, &[vk::BufferCopy::default().size(n as u64 * real)]),
                None => barrier(d, cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE, vk::PipelineStageFlags::HOST, vk::AccessFlags::HOST_READ),
            }
        })?;
        let inverse_cmd = record(&ctx, false, 0, |cmd, _| unsafe {
            barrier(d, cmd, vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ);
            for stage in &plan.inverse {
                for seg in stage {
                    dispatch(d, cmd, &pipes[seg.kernel as usize], [seg.first, seg.count], seg.count as usize, tau);
                }
                compute_barrier(d, cmd);
            }
        })?;
        let map = |b: Buf| unsafe { ctx.dev.map_memory(b.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(device_error) };
        let stage_status_ptr = map(stage_status)? as *const u32;
        let mut mapped = vec![stage_status];
        let load_target = if unified {
            mapped.extend([lval, dinv, xinv]);
            Some(LoadTarget { lval: map(lval)? as *mut T, dinv: map(dinv)? as *mut T, xinv: map(xinv)? as *mut T, stage: None })
        } else {
            None
        };
        let (values_ptr, diag_ptr) = match stage_a {
            Some(stage_a) => {
                mapped.push(stage_a);
                let p = map(stage_a)? as *mut T;
                (p, unsafe { p.add(a_len) })
            }
            None => {
                mapped.extend([aval, diag]);
                (map(aval)? as *mut T, map(diag)? as *mut T)
            }
        };
        let (rhs_ptr, solution_ptr) = match stage_b {
            Some(stage_b) => {
                mapped.push(stage_b);
                let p = map(stage_b)? as *mut T;
                (p, p as *const T)
            }
            None => {
                mapped.extend([bvec, out]);
                (map(bvec)? as *mut T, map(out)? as *const T)
            }
        };

        let mut inv = vec![0usize; n];
        for (new, &old) in plan.perm.iter().enumerate() {
            inv[old as usize] = new;
        }
        let mut diag_src = Vec::with_capacity(n);
        for (j, &new) in inv.iter().enumerate() {
            for p in analysis.col_ptr()[j]..analysis.col_ptr()[j + 1] {
                if analysis.row_idx()[p] == j {
                    diag_src.push((p, new));
                }
            }
        }
        let factor_dispatches = plan.factor.iter().map(Vec::len).sum::<usize>() + 1;
        let stats = Stats { factor_dispatches, solve_dispatches: plan.solve.len() + 2, ..plan.stats };
        Ok(Self {
            ctx,
            n,
            a_len,
            factor_len,
            diag_src,
            lval,
            values_ptr,
            diag_ptr,
            rhs_ptr,
            solution_ptr,
            stage_status,
            stage_status_ptr,
            mapped,
            buffers: [items, fronts, bounds, smap, rmap, dst, pat, perm, sign, aval, status, diag, x, ytmp].into_iter().chain(stage_a).chain(stage_b).collect(),
            bvec,
            out,
            solve_segments: plan.solve,
            tau,
            pipes,
            module,
            factor_cmd,
            solve_cmd,
            inverse_cmd,
            shapes: plan.shapes,
            dinv,
            load_target,
            xinv,
            xinv_len: plan.xinv_len,
            stats,
            last_factor_ms: 0.0,
            last_solve_ms: 0.0,
            _real: PhantomData,
        })
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Takes a factor made elsewhere, in the analysis's layout (`faer`'s, or
    /// [`crate::sme::SmeCholesky`]'s), for this device's solves: writes it in `T`, inverts its
    /// 32-wide diagonal tiles on the host (what this device's own factorization leaves behind),
    /// and runs only the diagonal-block inversions here. Returns their device time (ms).
    pub fn load_factor<S: crate::cholesky::Real>(&mut self, factor: &[S]) -> Result<f64, Error> {
        use rayon::prelude::*;
        assert_eq!(factor.len(), self.factor_len);
        let target = self.load_target()?;
        let (lval, dinv) = (Out(target.lval), Out(target.dinv));
        convert(factor, target.lval);
        self.shapes.par_iter().for_each(|&[offset, f, k, begin]| invert_tiles(factor, offset, f, k, begin, lval, dinv));
        self.upload_loaded(&target, false)?;
        self.run(&self.inverse_cmd)
    }

    /// As [`Self::load_factor`], with the diagonal-block inverses made elsewhere too
    /// ([`crate::sme::SmeCholesky::with_inverses`]): `L₁₁⁻¹` of each supernode, lower, `k × k`
    /// column-major, one after another. Nothing then runs on the device: its solves read only
    /// the factor and these.
    pub fn load_factor_and_inverse<S: crate::cholesky::Real>(&mut self, factor: &[S], inverse: &[S]) -> Result<(), Error> {
        assert_eq!(factor.len(), self.factor_len);
        assert_eq!(inverse.len(), self.xinv_len);
        let target = self.load_target()?;
        convert(factor, target.lval);
        convert(inverse, target.xinv);
        self.upload_loaded(&target, true)
    }

    /// Where loaded values are written: the kernels' buffers, mapped, on a unified-memory
    /// device; else a staging buffer, made on first use.
    fn load_target(&mut self) -> Result<LoadTarget<T>, Error> {
        if let Some(t) = self.load_target {
            return Ok(t);
        }
        let real = size_of::<T>();
        let stage = self.ctx.host_buffer(((self.factor_len + self.n + self.xinv_len) * real) as u64)?;
        let p = unsafe { self.ctx.dev.map_memory(stage.mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).map_err(device_error)? } as *mut T;
        let target = unsafe { LoadTarget { lval: p, dinv: p.add(self.factor_len), xinv: p.add(self.factor_len + self.n), stage: Some(stage) } };
        self.load_target = Some(target);
        Ok(target)
    }

    /// Copies a staged load to the kernels' buffers: the factor and pivots, or the factor and
    /// inverses.
    fn upload_loaded(&self, target: &LoadTarget<T>, inverse: bool) -> Result<(), Error> {
        let Some(stage) = target.stage else { return Ok(()) };
        let real = size_of::<T>() as u64;
        let (lval_bytes, dinv_bytes, xinv_bytes) = (self.factor_len as u64 * real, self.n as u64 * real, self.xinv_len as u64 * real);
        self.ctx.one_shot(|cmd| unsafe {
            let d = &self.ctx.dev;
            d.cmd_copy_buffer(cmd, stage.buf, self.lval.buf, &[vk::BufferCopy::default().size(lval_bytes)]);
            if inverse {
                d.cmd_copy_buffer(cmd, stage.buf, self.xinv.buf, &[vk::BufferCopy::default().src_offset(lval_bytes + dinv_bytes).size(xinv_bytes)]);
            } else {
                d.cmd_copy_buffer(cmd, stage.buf, self.dinv.buf, &[vk::BufferCopy::default().src_offset(lval_bytes).size(dinv_bytes)]);
            }
        })
    }

    /// Device time of the last factorization (ms), from timestamps around its command buffer.
    pub fn last_factor_ms(&self) -> f64 {
        self.last_factor_ms
    }

    /// Device time of the last solve (ms), uploads and downloads of the vector included.
    pub fn last_solve_ms(&self) -> f64 {
        self.last_solve_ms
    }

    /// Factors the matrix whose lower-triangle values, in the analysis's pattern order, are
    /// `values`.
    pub fn factorize(&mut self, values: &[f64]) -> Result<Info, Error> {
        assert_eq!(values.len(), self.a_len);
        unsafe {
            let a = std::slice::from_raw_parts_mut(self.values_ptr, self.a_len);
            let diag = std::slice::from_raw_parts_mut(self.diag_ptr, self.n);
            a.par_iter_mut().zip(values).with_min_len(1 << 14).for_each(|(s, &v)| *s = T::from_f64(v));
            diag.fill(T::from_f64(0.0));
            for &(p, j) in &self.diag_src {
                diag[j] = T::from_f64(values[p]);
            }
        }
        self.last_factor_ms = self.run(&self.factor_cmd)?;
        Ok(Info { perturbed: unsafe { *self.stage_status_ptr } != 0 })
    }

    /// `rhs ← A⁻¹ rhs` with the last factorization.
    pub fn solve_in_place(&mut self, rhs: &mut [f64]) -> Result<(), Error> {
        assert_eq!(rhs.len(), self.n);
        let input = unsafe { std::slice::from_raw_parts_mut(self.rhs_ptr, self.n) };
        for (s, &v) in input.iter_mut().zip(rhs.iter()) {
            *s = T::from_f64(v);
        }
        self.last_solve_ms = self.run(&self.solve_cmd)?;
        let solution = unsafe { std::slice::from_raw_parts(self.solution_ptr, self.n) };
        for (r, s) in rhs.iter_mut().zip(solution) {
            *r = s.to_f64();
        }
        Ok(())
    }

    /// The factor in the analysis's (`faer`'s supernodal) layout.
    pub fn factor(&self) -> Result<Vec<T>, Error> {
        self.ctx.download(self.lval, self.factor_len)
    }

    /// Per kernel and level device times of the last factorization, with
    /// `HYDROPLANE_SOLVERS_PROFILE` set when this was built (empty otherwise). A stage's time goes
    /// to its first dispatch: an inversion running alongside panel work is not counted apart.
    pub fn factor_profile(&self) -> Result<Vec<KernelTime>, Error> {
        self.profile(&self.factor_cmd)
    }

    /// As [`Self::factor_profile`], for the last solve.
    pub fn solve_profile(&self) -> Result<Vec<KernelTime>, Error> {
        self.profile(&self.solve_cmd)
    }

    /// Every stage of the last factorization in order: kernel, level, device ms.
    pub fn factor_timeline(&self) -> Result<Vec<(Kernel, usize, f64)>, Error> {
        let ts = self.timestamps(&self.factor_cmd)?;
        let mut prev = ts[0];
        Ok(self
            .factor_cmd
            .stamps
            .iter()
            .enumerate()
            .map(|(i, &(kernel, level))| {
                let t = ts[2 + i];
                let ms = t.wrapping_sub(prev) as f64 * self.ctx.period_ns * 1e-6;
                prev = t;
                (kernel, level, ms)
            })
            .collect())
    }

    fn profile(&self, r: &Recording) -> Result<Vec<KernelTime>, Error> {
        let ts = self.timestamps(r)?;
        let mut acc = std::collections::BTreeMap::<(usize, Kernel), (usize, f64)>::new();
        // Stamp i closes the dispatch it follows; the first one is measured from the start.
        let mut prev = ts[0];
        for (i, &(kernel, level)) in r.stamps.iter().enumerate() {
            let t = ts[2 + i];
            let e = acc.entry((level, kernel)).or_default();
            e.0 += 1;
            e.1 += t.wrapping_sub(prev) as f64 * self.ctx.period_ns * 1e-6;
            prev = t;
        }
        Ok(acc.into_iter().map(|((level, kernel), (dispatches, ms))| KernelTime { kernel, level, dispatches, ms }).collect())
    }

    fn timestamps(&self, r: &Recording) -> Result<Vec<u64>, Error> {
        let mut ts = vec![0u64; 2 + r.stamps.len()];
        unsafe {
            self.ctx
                .dev
                .get_query_pool_results(r.queries, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
                .map_err(device_error)?;
        }
        Ok(ts)
    }

    /// The solve's input and output buffers (`n` values of `T` each, in the matrix's order), for
    /// recording [`Self::record_solve`] into another command buffer.
    pub(super) fn solve_buffers(&self) -> (Buf, Buf) {
        (self.bvec, self.out)
    }

    /// Records `out ← A⁻¹ bvec` with the last factorization into `cmd`, each dispatch followed by
    /// a compute barrier.
    pub(super) fn record_solve(&self, cmd: vk::CommandBuffer) {
        let mut off = Stamper { d: &self.ctx.dev, queries: vk::QueryPool::null(), on: false, stamps: Vec::new() };
        unsafe { solve_body(&self.ctx.dev, cmd, &self.pipes, &self.solve_segments, self.n, self.tau, &mut off) };
    }

    fn run(&self, r: &Recording) -> Result<f64, Error> {
        self.ctx.submit_wait(r.cmd)?;
        let ts = self.timestamps(r)?;
        Ok(ts[1].wrapping_sub(ts[0]) as f64 * self.ctx.period_ns * 1e-6)
    }
}

impl<T: DeviceReal> Drop for VulkanCholesky<T> {
    fn drop(&mut self) {
        let ctx = &self.ctx;
        let _queue = ctx.lock();
        unsafe {
            let _ = ctx.dev.device_wait_idle();
            for b in &self.mapped {
                ctx.dev.unmap_memory(b.mem);
            }
            for r in [&self.factor_cmd, &self.solve_cmd, &self.inverse_cmd] {
                ctx.dev.free_command_buffers(ctx.cpool, &[r.cmd]);
                ctx.dev.destroy_query_pool(r.queries, None);
            }
            for p in &self.pipes {
                ctx.destroy_pipeline(p);
            }
            ctx.dev.destroy_shader_module(self.module, None);
        }
        let load_stage = self.load_target.and_then(|t| t.stage);
        if let Some(stage) = load_stage {
            unsafe { ctx.dev.unmap_memory(stage.mem) };
        }
        for &b in self.buffers.iter().chain([&self.lval, &self.dinv, &self.xinv, &self.stage_status, &self.bvec, &self.out]).chain(load_stage.as_ref()) {
            ctx.free(b);
        }
    }
}

/// Records `body` with timestamps around it, and room for `dispatches` more when profiling.
/// Where [`VulkanCholesky::load_factor`] and its sibling write.
#[derive(Clone, Copy)]
struct LoadTarget<T> {
    lval: *mut T,
    dinv: *mut T,
    xinv: *mut T,
    stage: Option<Buf>,
}

/// `out[..src.len()] = src`, converted to `T`, in parallel.
fn convert<S: crate::cholesky::Real, T: DeviceReal>(src: &[S], out: *mut T) {
    use rayon::prelude::*;
    unsafe { std::slice::from_raw_parts_mut(out, src.len()) }
        .par_iter_mut()
        .zip(src)
        .with_min_len(1 << 16)
        .for_each(|(t, s)| *t = T::from_f64(s.to_f64()));
}

/// A host pointer the tile inversions write through: each supernode's tiles and pivots are its
/// own, so supernodes go in parallel.
#[derive(Clone, Copy)]
struct Out<T>(*mut T);

unsafe impl<T> Send for Out<T> {}
unsafe impl<T> Sync for Out<T> {}

/// The 32-wide diagonal tiles of one supernode's block in `factor` (`f` rows, `k` columns, at
/// `offset`): each tile's inverse, transposed, into the tile's upper triangle of `lval`, and its
/// inverted pivots into `dinv` (at the supernode's first column `begin`), as `chol_step` leaves
/// them for `chol_inv_diag`.
#[allow(clippy::needless_range_loop, reason = "triangular index arithmetic reads clearer as indices")]
fn invert_tiles<S: crate::cholesky::Real, T: DeviceReal>(factor: &[S], offset: usize, f: usize, k: usize, begin: usize, lval: Out<T>, dinv: Out<T>) {
    let mut inv = [[0.0f64; TILE]; TILE];
    for c0 in (0..k).step_by(TILE) {
        let b = (k - c0).min(TILE);
        let l = |i: usize, j: usize| factor[offset + (c0 + j) * f + c0 + i].to_f64();
        for t in 0..b {
            inv[t][t] = 1.0 / l(t, t);
            for i in t + 1..b {
                let s: f64 = (t..i).map(|p| l(i, p) * inv[p][t]).sum();
                inv[i][t] = -s / l(i, i);
            }
        }
        unsafe {
            for c in 0..b {
                *dinv.0.add(begin + c0 + c) = T::from_f64(inv[c][c]);
                for r in 0..c {
                    *lval.0.add(offset + (c0 + c) * f + c0 + r) = T::from_f64(inv[c][r]);
                }
            }
        }
    }
}

pub(super) fn record(ctx: &Context, profile: bool, dispatches: usize, body: impl FnOnce(vk::CommandBuffer, &mut Stamper)) -> Result<Recording, Error> {
    let d = &ctx.dev;
    let count = 2 + if profile { dispatches as u32 } else { 0 };
    let _pool = ctx.lock();
    unsafe {
        let cmd = d
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(ctx.cpool).command_buffer_count(1))
            .map_err(device_error)?[0];
        let queries = d
            .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(count), None)
            .map_err(device_error)?;
        d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default()).map_err(device_error)?;
        d.cmd_reset_query_pool(cmd, queries, 0, count);
        d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
        let mut stamper = Stamper { d, queries, on: profile, stamps: Vec::new() };
        body(cmd, &mut stamper);
        d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
        d.end_command_buffer(cmd).map_err(device_error)?;
        Ok(Recording { cmd, queries, stamps: stamper.stamps })
    }
}

/// The solve's dispatches: gather into the permuted order, the levels' forward and backward
/// steps, scatter back.
unsafe fn solve_body(d: &ash::Device, cmd: vk::CommandBuffer, pipes: &[Pipe], segments: &[Segment], n: usize, tau: f32, st: &mut Stamper) {
    unsafe {
        dispatch(d, cmd, &pipes[Kernel::SolveGather as usize], [0, n as u32], n.div_ceil(256), tau);
        compute_barrier(d, cmd);
        st.stamp(cmd, Kernel::SolveGather, 0);
        for seg in segments {
            dispatch(d, cmd, &pipes[seg.kernel as usize], [seg.first, seg.count], seg.count as usize, tau);
            compute_barrier(d, cmd);
            st.stamp(cmd, seg.kernel, seg.level);
        }
        dispatch(d, cmd, &pipes[Kernel::SolveScatter as usize], [0, n as u32], n.div_ceil(256), tau);
        compute_barrier(d, cmd);
        st.stamp(cmd, Kernel::SolveScatter, 0);
    }
}

pub(super) unsafe fn barrier(
    d: &ash::Device,
    cmd: vk::CommandBuffer,
    src: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let b = [vk::MemoryBarrier::default().src_access_mask(src_access).dst_access_mask(dst_access)];
    unsafe { d.cmd_pipeline_barrier(cmd, src, dst, vk::DependencyFlags::empty(), &b, &[], &[]) };
}

pub(super) unsafe fn compute_barrier(d: &ash::Device, cmd: vk::CommandBuffer) {
    unsafe {
        barrier(
            d,
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        )
    };
}

/// `groups` workgroups as a `stride × ⌈groups / stride⌉` grid, with the item range `[first,
/// count]` and the pivot threshold as push constants.
pub(super) unsafe fn dispatch(d: &ash::Device, cmd: vk::CommandBuffer, pipe: &Pipe, [first, count]: [u32; 2], groups: usize, tau: f32) {
    let stride = groups.clamp(1, GRID);
    let prm = [first, count, stride as u32, tau.to_bits()];
    unsafe {
        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe.pipe);
        d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, pipe.layout, 0, &[pipe.set], &[]);
        d.cmd_push_constants(cmd, pipe.layout, vk::ShaderStageFlags::COMPUTE, 0, std::slice::from_raw_parts(prm.as_ptr() as *const u8, 16));
        d.cmd_dispatch(cmd, stride as u32, groups.div_ceil(stride) as u32, 1);
    }
}

struct Plan {
    fronts: Vec<u32>,
    bounds: Vec<u32>,
    smap: Vec<u32>,
    rmap: Vec<u32>,
    pat: Vec<u32>,
    perm: Vec<u32>,
    items: Vec<u32>,
    dst: Vec<u32>,
    /// Dispatches of a factorization in stages: the dispatches of one stage run concurrently, a
    /// barrier separates stages.
    factor: Vec<Vec<Segment>>,
    solve: Vec<Segment>,
    /// The diagonal-block inversions alone, every level's `i`th dispatch in stage `i` (levels
    /// read and write disjoint blocks).
    inverse: Vec<Vec<Segment>>,
    shapes: Vec<[usize; 4]>,
    xinv_len: usize,
    stats: Stats,
}

/// Appends the items `push` writes as one segment of `kernel`, if it wrote any.
fn segment(items: &mut Vec<u32>, out: &mut Vec<Segment>, kernel: Kernel, level: usize, push: impl FnOnce(&mut Vec<u32>)) {
    let first = items.len() / 4;
    push(items);
    let count = items.len() / 4 - first;
    if count > 0 {
        out.push(Segment { kernel, level, first: first as u32, count: count as u32 });
    }
}

impl Plan {
    fn new(analysis: &Analysis, schur_tile: SchurTile, signed: bool) -> Result<Self, Error> {
        let sn = analysis.supernodal();
        let n = analysis.dim();
        let ns = sn.n_supernodes();
        let (begin, end, cpv) = (sn.supernode_begin(), sn.supernode_end(), sn.col_ptr_for_val());
        let mut col_super = vec![0usize; n];
        for s in 0..ns {
            col_super[begin[s]..end[s]].fill(s);
        }
        let local = |s: usize, row: usize| -> usize {
            if row < end[s] {
                row - begin[s]
            } else {
                end[s] - begin[s] + sn.supernode(s).pattern().binary_search(&row).expect("row in the supernode's pattern")
            }
        };
        let fit = |v: usize| u32::try_from(v).map_err(|_| Error::Device("index beyond 32 bits".into()));

        let mut fronts = vec![0u32; ns * FRONT_WORDS];
        let mut bounds = Vec::new();
        let (mut smap, mut rmap, mut pat) = (Vec::new(), Vec::new(), Vec::new());
        let mut panels = vec![0usize; ns];
        let mut tiles = vec![0usize; ns];
        let mut parent = vec![usize::MAX; ns];
        let mut largest_front = 0;
        let mut xinv_len = 0usize;
        let (mut flops, mut schur_flops) = (0.0, 0.0);
        for s in 0..ns {
            let pattern = sn.supernode(s).pattern();
            let (k, m) = (end[s] - begin[s], pattern.len());
            let f = k + m;
            largest_front = largest_front.max(f);
            let b0 = bounds.len();
            bounds.extend((0..k).step_by(TILE).map(|b| b as u32));
            panels[s] = bounds.len() - b0;
            bounds.extend((k..f).step_by(TILE).map(|b| b as u32));
            bounds.push(f as u32);
            tiles[s] = bounds.len() - b0 - 1;
            if schur_tile == SchurTile::T64 && tiles[s] > u16::MAX as usize {
                return Err(Error::Device("a supernode of more than 65535 tiles for 64-wide Schur blocks".into()));
            }
            let p0 = pat.len();
            pat.extend(pattern.iter().map(|&r| r as u32));
            // The update's columns fall into runs of one ancestor each; a run starting at pattern
            // position `a` maps every pattern row from `a` on to that ancestor's block rows.
            let mut a = 0;
            while a < m {
                let target = col_super[pattern[a]];
                let run = pattern[a..].iter().take_while(|&&r| r < end[target]).count();
                let base = fit(rmap.len())?;
                rmap.extend(pattern[a..].iter().map(|&r| local(target, r) as u32));
                for &r in &pattern[a..a + run] {
                    smap.extend([target as u32, (r - begin[target]) as u32, base.wrapping_sub(a as u32)]);
                }
                a += run;
            }
            if let Some(&first) = pattern.first() {
                parent[s] = col_super[first];
            }
            fronts[s * FRONT_WORDS..][..FRONT_WORDS].copy_from_slice(&[fit(cpv[s])?, f as u32, k as u32, fit(b0)?, begin[s] as u32, fit(p0)?, fit(xinv_len)?]);
            xinv_len += k * k;
            let (kf, mf) = (k as f64, m as f64);
            flops += 2.0 * kf * kf * kf / 3.0 + kf * kf * mf + kf * mf * mf;
            schur_flops += kf * mf * mf;
        }
        let level = analysis.levels().0;
        let levels = level.iter().max().map_or(0, |l| l + 1);
        let mut at_level: Vec<Vec<usize>> = vec![Vec::new(); levels];
        for s in 0..ns {
            at_level[level[s]].push(s);
        }

        let mut items: Vec<u32> = Vec::new();
        let mut panel_work: Vec<Vec<Segment>> = Vec::with_capacity(levels);
        for (level, group) in at_level.iter().enumerate() {
            let mut factor = Vec::new();
            let max_panels = group.iter().map(|&s| panels[s]).max().unwrap_or(0);
            segment(&mut items, &mut factor, Kernel::Step, level, |it| {
                for &s in group {
                    it.extend([s as u32, 0, POTRF, 0]);
                }
            });
            for kb in 0..max_panels {
                let active: Vec<usize> = group.iter().copied().filter(|&s| panels[s] > kb).collect();
                let kb32 = kb as u32;
                segment(&mut items, &mut factor, Kernel::Trsm, level, |it| {
                    for &s in &active {
                        for i in kb + 1..tiles[s] {
                            it.extend([s as u32, i as u32, kb32, kb32]);
                        }
                    }
                });
                // Panel kb's update of the block, with the next panel's diagonal tile updated and
                // factored by one workgroup of the same dispatch.
                segment(&mut items, &mut factor, Kernel::Step, level, |it| {
                    for &s in &active {
                        if kb + 1 < panels[s] {
                            it.extend([s as u32, kb32 + 1, POTRF, 0]);
                        }
                        for j in kb + 1..panels[s] {
                            for i in j..tiles[s] {
                                if (i, j) != (kb + 1, kb + 1) {
                                    it.extend([s as u32, i as u32, j as u32, kb32]);
                                }
                            }
                        }
                    }
                });
            }
            segment(&mut items, &mut factor, Kernel::Schur, level, |it| {
                for &s in group {
                    match schur_tile {
                        SchurTile::T32 => {
                            for j in panels[s]..tiles[s] {
                                for i in j..tiles[s] {
                                    it.extend([s as u32, i as u32, j as u32, 0]);
                                }
                            }
                        }
                        // Pairs of 32-wide tiles, the last tile of each block packed with the first.
                        SchurTile::T64 => {
                            for j in (panels[s]..tiles[s]).step_by(2) {
                                for i in (j..tiles[s]).step_by(2) {
                                    let (ie, je) = ((i + 2).min(tiles[s]), (j + 2).min(tiles[s]));
                                    it.extend([s as u32, i as u32, j as u32, ((ie as u32) << 16) | je as u32]);
                                }
                            }
                        }
                    }
                }
            });
            panel_work.push(factor);
        }

        // Each level's diagonal blocks inverted by doubling over their tiles: the diagonal tiles,
        // then for h = 1, 2, 4, … every pair of h-tile blocks A = [a, a + h), C = [a + h, a + 2h).
        let mut inverse_work: Vec<Vec<Segment>> = Vec::with_capacity(levels);
        for (level, group) in at_level.iter().enumerate() {
            let mut factor = Vec::new();
            segment(&mut items, &mut factor, Kernel::InverseDiag, level, |it| {
                for &s in group {
                    for i in 0..panels[s] {
                        it.extend([s as u32, i as u32, 0, 0]);
                    }
                }
            });
            let mut h = 1;
            while h < group.iter().map(|&s| panels[s]).max().unwrap_or(0) {
                let pairs = |it: &mut Vec<u32>| {
                    for &s in group {
                        for a in (0..panels[s]).step_by(2 * h) {
                            for i in a + h..(a + 2 * h).min(panels[s]) {
                                for j in a..a + h {
                                    it.extend([s as u32, i as u32, j as u32, h as u32]);
                                }
                            }
                        }
                    }
                };
                segment(&mut items, &mut factor, Kernel::InverseLeft, level, pairs);
                segment(&mut items, &mut factor, Kernel::InverseRight, level, pairs);
                h *= 2;
            }
            inverse_work.push(factor);
        }
        // A level's inversion only reads its own finished blocks and the next level's panel work
        // touches neither them nor `xinv`, so the two share stages (no barrier between them): the
        // inversion fills the GPU while the panel steps wait on their critical path.
        let mut inverse: Vec<Vec<Segment>> = Vec::new();
        for level in &inverse_work {
            for (i, &seg) in level.iter().enumerate() {
                if inverse.len() <= i {
                    inverse.push(Vec::new());
                }
                inverse[i].push(seg);
            }
        }
        let mut factor: Vec<Vec<Segment>> = Vec::new();
        let mut pending: std::collections::VecDeque<Segment> = std::collections::VecDeque::new();
        for (panels_l, inverse_l) in panel_work.into_iter().zip(inverse_work) {
            for seg in panels_l {
                let mut stage = vec![seg];
                stage.extend(pending.pop_front());
                factor.push(stage);
            }
            factor.extend(pending.drain(..).map(|seg| vec![seg]));
            pending.extend(inverse_l);
        }
        factor.extend(pending.drain(..).map(|seg| vec![seg]));

        let mut solve = Vec::new();
        let cols = |s: usize| end[s] - begin[s];
        let rows = |s: usize| sn.supernode(s).pattern().len();
        for (level, group) in at_level.iter().enumerate() {
            segment(&mut items, &mut solve, Kernel::ForwardGemv, level, |it| {
                for &s in group {
                    for chunk in 0..cols(s).div_ceil(256) {
                        it.extend([s as u32, chunk as u32, 0, 0]);
                    }
                }
            });
            segment(&mut items, &mut solve, Kernel::ForwardPattern, level, |it| {
                for &s in group {
                    for chunk in 0..rows(s).div_ceil(256) {
                        it.extend([s as u32, chunk as u32, 0, 0]);
                    }
                }
            });
        }
        for (level, group) in at_level.iter().enumerate().rev() {
            segment(&mut items, &mut solve, Kernel::BackwardPattern, level, |it| {
                for &s in group.iter().filter(|&&s| signed || rows(s) > 0) {
                    for chunk in 0..cols(s).div_ceil(8) {
                        it.extend([s as u32, chunk as u32, 0, 0]);
                    }
                }
            });
            segment(&mut items, &mut solve, Kernel::BackwardGemv, level, |it| {
                for &s in group {
                    for chunk in 0..cols(s).div_ceil(8) {
                        it.extend([s as u32, chunk as u32, 0, 0]);
                    }
                }
            });
        }

        let perm: Vec<u32> = analysis.perm().iter().map(|&p| p as u32).collect();
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old as usize] = new;
        }
        let (col_ptr, row_idx) = (analysis.col_ptr(), analysis.row_idx());
        let sink = fit(analysis.factor_len())?;
        let mut dst = vec![sink; row_idx.len()];
        for j in 0..n {
            for p in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[p];
                if i < j {
                    continue;
                }
                let (mut a, mut b) = (inv[i], inv[j]);
                if a < b {
                    std::mem::swap(&mut a, &mut b);
                }
                let s = col_super[b];
                dst[p] = fit(cpv[s] + (b - begin[s]) * (cols(s) + rows(s)) + local(s, a))?;
            }
        }
        let stats = Stats {
            fronts: ns,
            levels,
            items: items.len() / 4,
            factor_len: analysis.factor_len(),
            largest_front,
            flops,
            schur_flops,
            ..Default::default()
        };
        let shapes = (0..ns).map(|s| [cpv[s], cols(s) + rows(s), cols(s), begin[s]]).collect();
        Ok(Self { fronts, bounds, smap, rmap, pat, perm, items, dst, factor, solve, inverse, shapes, xinv_len, stats })
    }
}
