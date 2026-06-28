//! SVE2 = SVE(v1) + extra integer/DSP ops. The `f32`/`f64` element-wise primitives are identical
//! to [`super::sve1`], so they're re-exported; this module adds SVE2-only ops and the VL reader.
#![allow(dead_code, unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

use core::arch::asm;

pub use super::sve1::*;

/// Read the (non-streaming) SVE vector length in bytes via `RDVL`.
///
/// # Safety
/// The CPU must implement SVE (base, non-streaming) — guard with `is_aarch64_feature_detected!`.
#[target_feature(enable = "sve")]
pub unsafe fn vl_bytes_raw() -> usize {
    let r: usize;
    asm!("rdvl {r}, #1", r = out(reg) r, options(pure, nomem, nostack));
    r
}

/// SVE vector length in bytes. Only valid where base SVE is present (the caller detects it first).
#[inline]
pub fn vl_bytes() -> usize {
    unsafe { vl_bytes_raw() }
}
