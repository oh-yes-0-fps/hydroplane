//! Turns per-kernel MIR metrics into codegen choices at macro expansion, looking kernels up by
//! name in the file `HYDRO_ANALYSIS` points at (absent on ordinary builds, so lookups miss and the
//! macro keeps its defaults). All thresholds live here; the driver only measures.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub struct Metrics {
    /// Frame size of the concrete `f32 × ScalarBackend` monomorphization (sum of local sizes).
    pub stack_bytes: u64,
    /// Count of locals of the crate's varying type — a register-pressure proxy.
    pub varying_locals: u64,
    /// Places read/written through a `Deref`/index projection — memory-traffic proxy.
    pub mem_ops: u64,
    /// `SwitchInt` terminators — control-flow divergence.
    pub switches: u64,
    /// `Call` terminators.
    pub calls: u64,
    /// Total MIR statements of the `_on` body — overall (non-closure) body size.
    pub stmts: u64,
    /// Statement count including the kernel's combinator closures (the `|acc,v| …` step bodies), an
    /// arithmetic-intensity proxy: a compute-bound step saturates the FP units on one chain, so
    /// high compute caps K low.
    pub compute: u64,
    /// Element-type bitmask (`Scalar::TYPE_BITS` values) the kernel's MIR actually touches, seen
    /// through generics and helper calls, so tighter than the macro's token scan. `0` when the
    /// driver predates this field or found nothing.
    pub types: u64,
}

/// Below this statement count, and free of calls and heavy memory traffic, a kernel is small enough
/// that the non-inlined `noalias` boundary costs more than it saves — inline it (`tiny`).
const TINY_STMTS: u64 = 24;
/// At or above this many memory ops the kernel is memory-bound and wants the `noalias` boundary so
/// LLVM can cluster and reorder its loads/stores.
const MEM_THRESH: u64 = 4;
/// The one knob for the unroll cap: `K ≈ BUDGET / compute-intensity`, where compute-intensity is
/// the statements reachable from the kernel body (its `_on` body, combinator closures, and the
/// non-kernel helpers they call, folded by the driver). One signal captures both effects that bound
/// `K`: a heavy body is both register-hungrier and more throughput-bound, so both push `K` down
/// together. Calibrated against the benchmark suite on aarch64; where it's off,
/// `#[kernel(unroll = N)]` overrides it.
const COMPUTE_BUDGET: u64 = 200;

const K_CANDIDATES: [u64; 5] = [1, 2, 4, 8, 16];

impl Metrics {
    /// Keep the `noalias` boundary unless the kernel is genuinely a micro-kernel.
    pub fn noalias(&self) -> bool {
        !(self.stmts < TINY_STMTS && self.calls == 0 && self.mem_ops < MEM_THRESH)
    }

    /// The measured element-type bitmask, when the driver recorded one.
    pub fn type_bits(&self) -> Option<u8> {
        (self.types != 0).then_some(self.types as u8)
    }

    /// Largest unroll factor whose per-chain cost still fits the compute budget, snapped to a dispatch
    /// candidate `{1,2,4,8,16}`.
    pub fn k_cap(&self) -> usize {
        let raw = (COMPUTE_BUDGET / self.compute.max(1)).max(1);
        K_CANDIDATES.iter().copied().filter(|&c| c <= raw).max().unwrap_or(1) as usize
    }
}

fn table() -> Option<&'static HashMap<String, Metrics>> {
    static TABLE: OnceLock<Option<HashMap<String, Metrics>>> = OnceLock::new();
    TABLE.get_or_init(load).as_ref()
}

fn load() -> Option<HashMap<String, Metrics>> {
    let spec = std::env::var("HYDRO_ANALYSIS").ok()?;
    let path = spec.split('#').next().unwrap_or(&spec);
    let text = std::fs::read_to_string(path).ok()?;
    let mut map = HashMap::new();
    for line in text.lines() {
        if let Some((name, met)) = parse_line(line) {
            map.insert(name, met);
        }
    }
    Some(map)
}

fn parse_line(line: &str) -> Option<(String, Metrics)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with("//") {
        return None;
    }
    let mut fields = line.split_whitespace();
    let name = fields.next()?.to_string();
    let mut m = Metrics {
        stack_bytes: 0,
        varying_locals: 0,
        mem_ops: 0,
        switches: 0,
        calls: 0,
        stmts: 0,
        compute: 0,
        types: 0,
    };
    for kv in fields {
        let (key, val) = kv.split_once('=')?;
        let val: u64 = val.parse().ok()?;
        match key {
            "stack_bytes" => m.stack_bytes = val,
            "varying_locals" => m.varying_locals = val,
            "mem_ops" => m.mem_ops = val,
            "switches" => m.switches = val,
            "calls" => m.calls = val,
            "stmts" => m.stmts = val,
            "compute" => m.compute = val,
            "types" => m.types = val,
            _ => {}
        }
    }
    Some((name, m))
}

/// The metrics recorded for `name`, or `None` when analysis is unavailable or this kernel was not
/// measured; the caller then keeps its defaults.
pub fn lookup(name: &str) -> Option<Metrics> {
    table()?.get(name).copied()
}
