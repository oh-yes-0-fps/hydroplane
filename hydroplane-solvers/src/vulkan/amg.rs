//! An [`Amg`] hierarchy's V-cycle on a Vulkan device, in `f32` (kernels in `kernels/src/amg.rs`):
//! the preconditioner of [`super::VulkanPcg::amg`]. The hierarchy is built on the CPU; its
//! operators, interpolations, restrictions and the coarsest level's dense inverse are uploaded,
//! and the cycle is recorded once, every dispatch followed by a compute barrier.

use std::sync::Arc;

use ash::vk;
use faer::Side;
use faer::linalg::solvers::DenseSolveCore;

use super::cholesky::{compute_barrier, dispatch};
use super::{Batch, Buf, Context, Pipe};
use crate::cholesky::Error;
use crate::iterative::Amg;
use crate::iterative::amg::Mat;

/// Rows the coarsest level may have: its inverse is applied as a dense `f32` matrix.
const MAX_COARSE: usize = 8192;
/// Average entries per row from which a product takes a warp per row.
const LONG_ROWS: usize = 16;
/// Shift of the coarsest level's diagonal, relative to its largest entry, should it not factor
/// as it is (a singular operator: a floating structure, a pure Neumann problem).
const COARSE_SHIFT: f64 = 1e-10;

#[derive(Clone, Copy)]
struct Csr {
    bufs: [Buf; 3],
    rows: usize,
    warp: bool,
}

struct Step {
    pipe: usize,
    rows: usize,
    first: u32,
    warp: bool,
}

pub(super) struct AmgCycle {
    ctx: Arc<Context>,
    pipes: Vec<Pipe>,
    buffers: Vec<Buf>,
    steps: Vec<Step>,
    input: Buf,
    output: Buf,
}

impl AmgCycle {
    pub(super) fn new(ctx: Arc<Context>, module: vk::ShaderModule, amg: &Amg) -> Result<Self, Error> {
        let (levels, coarse) = amg.hierarchy();
        let degree = amg.smoothing_degree();
        if coarse.rows > MAX_COARSE {
            return Err(Error::Device(format!("the coarsest level has {} rows, more than the GPU cycle's dense solve takes ({MAX_COARSE})", coarse.rows)));
        }
        let mut cycle = Self { ctx, pipes: Vec::new(), buffers: Vec::new(), steps: Vec::new(), input: null(), output: null() };
        let vector = |c: &mut Self, len: usize| c.keep(c.ctx.device_buffer(len.max(1) as u64 * 4));

        let mut coef = Vec::new();
        let mut offsets = Vec::new();
        for lv in levels {
            offsets.push(coef.len() as u32);
            let theta = 0.5 * (lv.high + lv.low);
            let delta = 0.5 * (lv.high - lv.low);
            let sigma = theta / delta;
            let mut rho = 1.0 / sigma;
            coef.push((1.0 / theta) as f32);
            for _ in 1..degree {
                let next = 1.0 / (2.0 * sigma - rho);
                coef.extend([(next * rho) as f32, (2.0 * next / delta) as f32]);
                rho = next;
            }
        }
        let ctx = cycle.ctx.clone();
        let mut batch = ctx.batch()?;
        let coef = cycle.upload(&mut batch, &coef)?;

        struct Bufs {
            a: Csr,
            p: Csr,
            r: Csr,
            dinv: Buf,
            b: Buf,
            x: Buf,
            res: Buf,
            d: Buf,
        }
        let mut bufs = Vec::with_capacity(levels.len());
        for lv in levels {
            let n = lv.a.rows;
            let dinv: Vec<f32> = lv.dinv.iter().map(|&d| d as f32).collect();
            bufs.push(Bufs {
                a: cycle.csr(&mut batch, &lv.a)?,
                p: cycle.csr(&mut batch, &lv.p)?,
                r: cycle.csr(&mut batch, &lv.r)?,
                dinv: cycle.upload(&mut batch, &dinv)?,
                b: vector(&mut cycle, n)?,
                x: vector(&mut cycle, n)?,
                res: vector(&mut cycle, n)?,
                d: vector(&mut cycle, n)?,
            });
        }
        let (coarse_b, coarse_x) = (vector(&mut cycle, coarse.rows)?, vector(&mut cycle, coarse.rows)?);
        let inverse = cycle.csr(&mut batch, &dense_inverse(coarse)?)?;
        batch.send()?;
        (cycle.input, cycle.output) = bufs.first().map_or((coarse_b, coarse_x), |top| (top.b, top.x));

        let mut down = Vec::new();
        let mut up = Vec::new();
        for (l, lb) in bufs.iter().enumerate() {
            let (next_b, next_x) = bufs.get(l + 1).map_or((coarse_b, coarse_x), |n| (n.b, n.x));
            let (off, n) = (offsets[l], lb.a.rows);
            let [ap, ac, av] = lb.a.bufs;
            let [rp, rc, rv] = lb.r.bufs;
            let [pp, pc, pv] = lb.p.bufs;
            let zero = cycle.pipe(module, "smooth_zero", &[coef, lb.dinv, lb.b, lb.res, lb.d, lb.x])?;
            let start = cycle.pipe(module, "smooth_start", &[coef, lb.dinv, lb.res, lb.d, lb.x])?;
            let step = cycle.pipe(module, "smooth_step", &[coef, lb.dinv, lb.res, lb.d, lb.x])?;
            let smooth_residual = cycle.pipe(module, "smooth_residual", &[ap, ac, av, lb.d, lb.res])?;
            let residual = cycle.pipe(module, "residual", &[ap, ac, av, lb.x, lb.b, lb.res])?;
            let restrict = cycle.pipe(module, "spmv", &[rp, rc, rv, lb.res, next_b])?;
            let prolong = cycle.pipe(module, "prolong", &[pp, pc, pv, next_x, lb.x])?;
            let on = |pipe, m: &Csr| Step { pipe, rows: m.rows, first: 0, warp: m.warp };
            let smooth = |first_pipe| {
                let mut s = vec![Step { pipe: first_pipe, rows: n, first: off, warp: false }];
                for k in 1..degree as u32 {
                    s.push(on(smooth_residual, &lb.a));
                    s.push(Step { pipe: step, rows: n, first: off + 2 * k - 1, warp: false });
                }
                s
            };
            down.extend(smooth(zero));
            down.extend([on(residual, &lb.a), on(restrict, &lb.r)]);
            let mut back = vec![on(prolong, &lb.p), on(residual, &lb.a)];
            back.extend(smooth(start));
            up.push(back);
        }
        let solve = cycle.pipe(module, "spmv", &[inverse.bufs[0], inverse.bufs[1], inverse.bufs[2], coarse_b, coarse_x])?;
        cycle.steps = down;
        cycle.steps.push(Step { pipe: solve, rows: coarse.rows, first: 0, warp: inverse.warp });
        cycle.steps.extend(up.into_iter().rev().flatten());
        Ok(cycle)
    }

    /// The buffers the cycle reads its right-hand side from and writes its result to.
    pub(super) fn buffers(&self) -> (Buf, Buf) {
        (self.input, self.output)
    }

    /// Records one V-cycle, `output ← M⁻¹ input`, into `cmd`.
    pub(super) fn record(&self, cmd: vk::CommandBuffer) {
        let d = &self.ctx.dev;
        for s in &self.steps {
            let groups = if s.warp { s.rows.div_ceil(8) } else { s.rows.div_ceil(256) };
            unsafe {
                dispatch(d, cmd, &self.pipes[s.pipe], [s.first, s.rows as u32], groups, if s.warp { 1.0 } else { 0.0 });
                compute_barrier(d, cmd);
            }
        }
    }

    fn keep(&mut self, b: Result<Buf, Error>) -> Result<Buf, Error> {
        let b = b?;
        self.buffers.push(b);
        Ok(b)
    }

    fn upload<T: Copy>(&mut self, batch: &mut Batch<'_>, data: &[T]) -> Result<Buf, Error> {
        let b = batch.upload(data);
        self.keep(b)
    }

    fn csr(&mut self, batch: &mut Batch<'_>, m: &Mat) -> Result<Csr, Error> {
        let row_ptr: Vec<u32> = m.ptr.iter().map(|&p| p as u32).collect();
        let val: Vec<f32> = m.val.iter().map(|&v| v as f32).collect();
        let bufs = [self.upload(batch, &row_ptr)?, self.upload(batch, &m.col)?, self.upload(batch, &val)?];
        Ok(Csr { bufs, rows: m.rows, warp: m.ptr[m.rows] >= LONG_ROWS * m.rows })
    }

    fn pipe(&mut self, module: vk::ShaderModule, name: &str, bufs: &[Buf]) -> Result<usize, Error> {
        self.pipes.push(self.ctx.pipeline(module, &format!("amg::amg_{name}"), bufs)?);
        Ok(self.pipes.len() - 1)
    }
}

impl Drop for AmgCycle {
    fn drop(&mut self) {
        let ctx = &self.ctx;
        {
            let _queue = ctx.lock();
            unsafe {
                let _ = ctx.dev.device_wait_idle();
            }
            for p in &self.pipes {
                ctx.destroy_pipeline(p);
            }
        }
        for &b in &self.buffers {
            ctx.free(b);
        }
    }
}

fn null() -> Buf {
    Buf { buf: vk::Buffer::null(), mem: vk::DeviceMemory::null() }
}

/// The inverse of the coarsest level's operator, dense, as a matrix by rows.
fn dense_inverse(a: &Mat) -> Result<Mat, Error> {
    let m = a.rows;
    let mut dense = faer::Mat::<f64>::zeros(m, m);
    for i in 0..m {
        for k in a.ptr[i]..a.ptr[i + 1] {
            dense[(i, a.col[k] as usize)] = a.val[k];
        }
    }
    let llt = match dense.llt(Side::Lower) {
        Ok(llt) => llt,
        Err(_) => {
            let largest = (0..m).map(|i| dense[(i, i)].abs()).fold(0.0, f64::max);
            for i in 0..m {
                dense[(i, i)] += COARSE_SHIFT * largest;
            }
            dense.llt(Side::Lower).map_err(|_| Error::Singular)?
        }
    };
    let inverse = llt.inverse();
    Ok(Mat {
        rows: m,
        cols: m,
        ptr: (0..=m).map(|i| i * m).collect(),
        col: (0..m * m).map(|k| (k % m) as u32).collect(),
        val: (0..m * m).map(|k| inverse[(k / m, k % m)]).collect(),
    })
}
